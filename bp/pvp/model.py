"""PvP policy/value network: a transformer over global, pet, ability-slot and aura tokens.

Heads
  policy    7 logits: ability 0-2 (scored from the active pet's contextualised ability tokens),
            swap/lead 3-5 (from own pet tokens), pass (from the global token)
  value     expected result in [-1, 1] from this player's point of view
  opp       what the opponent will do this round (7 classes) - auxiliary, also shown as a hint
  loadout   for every hidden enemy slot: probability the enemy took the second option - auxiliary
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..sim.data import game_data
from .abilityinfo import descriptor, n_features
from .features import MAX_AURAS, N_ABF, N_ABIL, N_AUF, N_GLOB, N_PET


class PvPNet(nn.Module):
    def __init__(self, d=320, layers=8, heads=8, ff=4, n_pet=N_PET, n_abf=N_ABF):
        super().__init__()
        gd = game_data()
        n_ab = max(gd.abilities) + 1
        desc = np.zeros((n_ab, n_features()), dtype=np.float32)
        for aid in gd.abilities:
            desc[aid] = descriptor(aid)
        self.register_buffer("desc", torch.from_numpy(desc), persistent=False)
        # n_pet / n_abf: how many pet / ability input features this network reads (older networks
        # read fewer: new features are appended at the end, so they just ignore the extra columns)
        self.cfg = {"d": d, "layers": layers, "heads": heads, "ff": ff, "n_pet": n_pet, "n_abf": n_abf}
        self.n_pet, self.n_abf = n_pet, n_abf
        self.ability = nn.Embedding(n_ab, d)
        self.desc_proj = nn.Linear(n_features(), d)
        self.family = nn.Embedding(11, d)
        self.pet_slot = nn.Embedding(6, d)
        self.ab_slot = nn.Embedding(3, d)
        self.holder = nn.Embedding(9, d)
        self.tok_type = nn.Embedding(4, d)
        self.glob = nn.Linear(N_GLOB, d)
        self.pet_float = nn.Linear(n_pet, d)
        self.ab_float = nn.Linear(n_abf, d)
        self.aura_float = nn.Linear(N_AUF, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff * d, dropout=0.0, batch_first=True, norm_first=True,
                                         activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        def mlp(i, o=1):
            return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))

        self.ab_head = mlp(2 * d)
        self.swap_head = mlp(2 * d)
        self.pass_head = mlp(d)
        self.value_head = mlp(d)
        self.opp_head = mlp(2 * d, 7)
        self.loadout_head = mlp(2 * d)

    def _abil(self, ids):
        if self.training or torch.is_grad_enabled():
            return self.ability(ids) + self.desc_proj(self.desc[ids])
        # inference: one table lookup instead of a projection per token (rebuilt after train())
        if getattr(self, "_abil_table", None) is None:
            with torch.autocast(ids.device.type, enabled=False):
                self._abil_table = (self.ability.weight + self.desc_proj(self.desc)).detach()
        return self._abil_table[ids]

    def train(self, mode=True):
        self._abil_table = None
        return super().train(mode)

    def forward(self, o):
        B = o["glob"].shape[0]
        dev = o["glob"].device
        t = self.tok_type.weight
        g = self.glob(o["glob"]).unsqueeze(1) + t[0]
        slots = torch.arange(6, device=dev).expand(B, 6)
        pets = self.family(o["pet_int"]) + self.pet_slot(slots) + self.pet_float(o["pet_float"][..., :self.n_pet]) + t[1]
        ai = o["ab_int"]
        abil = self._abil(ai[..., 0]) + self.pet_slot(ai[..., 1]) + self.ab_slot(ai[..., 2]) + self.ab_float(o["ab_float"][..., :self.n_abf]) + t[2]
        # aura slots are filled from the front: drop the ones no position in the batch uses
        # (masked keys do not change anything; exact)
        if torch.compiler.is_compiling() or not o["aura_mask"].shape[0]:
            n_au = o["aura_mask"].shape[1]  # compiled (infer passes trimmed arrays): no sync inside the graph
        else:
            n_au = int(o["aura_mask"].sum(1).max().item())
        ui = o["aura_int"][:, :n_au]
        aura = self._abil(ui[..., 0]) + self.holder(ui[..., 1]) + self.aura_float(o["aura_float"][:, :n_au]) + t[3]
        x = torch.cat([g, pets, abil, aura], 1)
        pad = torch.cat([torch.zeros(B, 7, dtype=torch.bool, device=dev), o["ab_mask"] < 0.5, o["aura_mask"][:, :n_au] < 0.5], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        hg, hp, ha = h[:, 0], h[:, 1:7], h[:, 7:7 + N_ABIL]
        # own active pet (pet_float[:, :, 6] = active flag); in the lead phase nobody is active
        act = o["pet_float"][:, :3, 6]
        act_idx = act.argmax(1)
        own_ab = ha[:, :9].reshape(B, 3, 3, -1)[torch.arange(B, device=dev), act_idx]  # B,3,d
        gx = hg.unsqueeze(1)
        ab_logits = self.ab_head(torch.cat([own_ab, gx.expand(-1, 3, -1)], -1)).squeeze(-1)
        swap_logits = self.swap_head(torch.cat([hp[:, :3], gx.expand(-1, 3, -1)], -1)).squeeze(-1)
        logits = torch.cat([ab_logits, swap_logits, self.pass_head(hg)], 1)
        value = torch.tanh(self.value_head(hg).squeeze(-1))
        en_act = o["pet_float"][:, 3:, 6].argmax(1)
        en_tok = hp[:, 3:][torch.arange(B, device=dev), en_act]
        opp = self.opp_head(torch.cat([hg, en_tok], -1))
        cand = ha[:, 9:].reshape(B, 9, 2, -1)
        lo = self.loadout_head(torch.cat([cand[:, :, 0], cand[:, :, 1]], -1)).squeeze(-1)
        return logits, value, opp, lo


def build(cfg):
    """Network with the input sizes it was trained with (old checkpoints: 40 pet / 10 ability features)."""
    return PvPNet(**{k: cfg[k] for k in ("d", "layers", "heads", "ff")},
                  n_pet=cfg.get("n_pet", 40), n_abf=cfg.get("n_abf", 10))


def load(ck, device="cpu", upgrade=False):
    """Network from a checkpoint dict. upgrade=True: widen an older network to the current feature
    set; the new input columns start at zero, so it plays exactly as before until it learns them."""
    net = build(ck["cfg"])
    net.load_state_dict(ck["model"])
    if upgrade and (net.n_pet < N_PET or net.n_abf < N_ABF):
        new = PvPNet(**{k: ck["cfg"][k] for k in ("d", "layers", "heads", "ff")})
        sd = dict(ck["model"])
        for name, n_new in (("pet_float", N_PET), ("ab_float", N_ABF)):
            w = sd[f"{name}.weight"]
            wide = torch.zeros(w.shape[0], n_new, dtype=w.dtype)
            wide[:, :w.shape[1]] = w
            sd[f"{name}.weight"] = wide
        new.load_state_dict(sd)
        net = new
    return net.to(device)


def masked(logits, mask):
    return logits.float().masked_fill(mask < 0.5, -1e9)


assert MAX_AURAS > 0


AURA_BUCKETS = (1, 2, 4, 8, 20)


def infer(net, obs, mask, device="cuda", chunk=4096, fixed=False):
    """Batch evaluation for the actors / advisor: numpy in, numpy out (policy, value, loadout).
    Positions are sorted by how many aura slots they use, so each chunk only carries the aura
    tokens it needs (unused slots are masked anyway: exact); fp16 autocast.
    fixed=True (a torch.compile'd net): every chunk is padded to `chunk` rows and the aura slots to
    one of AURA_BUCKETS, so the compiler sees only a handful of shapes."""
    from .features import KEYS
    n = len(mask)
    n_au_all = obs["aura_mask"].sum(1)
    order = np.argsort(n_au_all, kind="stable")
    pols = np.empty((n, 7), np.float32)
    vals = np.empty(n, np.float32)
    los = np.empty((n, 9), np.float32)
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for s in range(0, n, chunk):
            idx = order[s:s + chunk]
            b = len(idx)
            na = int(n_au_all[idx].max()) if b else 0
            rows = idx
            if fixed:
                na = next(x for x in AURA_BUCKETS if x >= na)
                if b < chunk:
                    rows = np.concatenate([idx, np.full(chunk - b, idx[0])])
            else:
                na = max(na, 0)
            x = {}
            for k in KEYS:
                a = obs[k][rows]
                if k.startswith("aura"):
                    a = a[:, :na]
                x[k] = torch.from_numpy(np.ascontiguousarray(a)).to(device)
            m = torch.from_numpy(np.ascontiguousarray(mask[rows])).to(device)
            logits, v, _, lo = net(x)
            pols[idx] = torch.softmax(masked(logits, m), -1).float().cpu().numpy()[:b]
            vals[idx] = v.float().cpu().numpy()[:b]
            los[idx] = torch.sigmoid(lo.float()).cpu().numpy()[:b]
    return pols, vals, los
