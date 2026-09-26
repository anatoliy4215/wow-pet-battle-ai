When the user starts the next PvP training cycle, first do these (agreed 2026-09-25, user said "запиши на будущее"):

1. **Explicit Undead / Mechanical features** in bp/pvp/features.py (pet tokens): "cannot die this round" (Undead immortal aura 242 / S_UNKILLABLE) and "effective health incl. unused Mechanical Failsafe (+20% max hp)" for own and enemy pets. Needs N_PET change → extend PvPNet.pet_float input, init new weight columns to zero so the checkpoint keeps its knowledge; mirror in bp/advisor/state.py (already sets mech_revived / undead_used from auras 244 / 242).
2. **Targeted practice positions**: part of training games start mid-battle in those moments — own/enemy Undead in its immortal round, Mechanical just before and just after Failsafe. Tactics exam "immortal" was only ~+10%.
3. **Value head over-confidence**: ~37% of positions |v|>0.99; the advisor then shows 0%/100% everywhere (a lost PvP battle was rated 95–100% throughout). Fix with value-target smoothing / calibration.
4. **Launch hygiene**: with rented boxes do NOT run the local actor (it starves the learner on the home GPU); learner `--steps N` / `--minutes N` saves, refreshes ratings, removes boxes; fleet now destroys boxes that fail setup.

Advisor ranking was already changed (not part of training): moves ranked by prior·exp(Q/0.15) instead of raw Q, which removed noise-driven swap/pass advice (28 → 4 of 133 positions).

