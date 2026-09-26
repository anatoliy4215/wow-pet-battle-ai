"""Policy/value network: a small transformer over pet and aura tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..sim import constants as C
from .features import N_AURA_FLOAT, N_PET_FLOAT


class PetNet(nn.Module):
    def __init__(self, n_species, n_abilities, d=128, layers=3, heads=4):
        super().__init__()
        self.species = nn.Embedding(n_species, d)
        self.ability = nn.Embedding(n_abilities, d)  # shared by pet abilities and auras
        self.family = nn.Embedding(11, d)
        self.slot = nn.Embedding(6, d)
        self.holder = nn.Embedding(10, d)
        self.pet_float = nn.Linear(N_PET_FLOAT, d)
        self.aura_float = nn.Linear(N_AURA_FLOAT, d)
        self.cd_proj = nn.Linear(1, d)
        self.glob = nn.Linear(4, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        # action heads: abilities scored from the own active pet's ability embeddings, swaps from pet tokens
        self.ab_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.swap_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.pass_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, obs, mask):
        pet_int, pet_float = obs["pet_int"], obs["pet_float"]
        B = pet_int.shape[0]
        slots = torch.arange(6, device=pet_int.device).expand(B, 6)
        ab_emb = self.ability(pet_int[:, :, 2:5])  # B,6,3,d
        pet_tok = (self.species(pet_int[:, :, 0]) + self.family(pet_int[:, :, 1]) + ab_emb.sum(2)
                   + self.slot(slots) + self.pet_float(pet_float))
        aura_tok = self.ability(obs["aura_int"][:, :, 0]) + self.holder(obs["aura_int"][:, :, 1]) + self.aura_float(obs["aura_float"])
        cls = self.cls.expand(B, 1, -1) + self.glob(obs["glob"]).unsqueeze(1)
        x = torch.cat([cls, pet_tok, aura_tok], 1)
        pad = torch.cat([torch.zeros(B, 7, device=x.device, dtype=torch.bool), obs["aura_mask"] < 0.5], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        g = h[:, 0]
        own = h[:, 1:4]
        active = (pet_float[:, :3, 7:8] * own).sum(1)  # own active pet token
        act_idx = pet_float[:, :3, 7].argmax(1)
        act_ab = ab_emb[torch.arange(B), act_idx]  # B,3,d
        ab_logits = self.ab_head(torch.cat([act_ab, active.unsqueeze(1).expand(-1, 3, -1), g.unsqueeze(1).expand(-1, 3, -1)], -1)).squeeze(-1)
        swap_logits = self.swap_head(torch.cat([own, g.unsqueeze(1).expand(-1, 3, -1)], -1)).squeeze(-1)
        pass_logit = self.pass_head(g)
        logits = torch.cat([ab_logits, swap_logits, pass_logit], 1)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        return logits, self.value(g).squeeze(-1)


assert C.N_ACTIONS == 7
