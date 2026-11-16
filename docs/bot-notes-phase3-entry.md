## 2026-11-17 · bot stream · big2 phase 3: no level change; new public items; what `quality` carries
- **Shipped model file:** none new. `medium` keeps `models/big2-qnet-duty.pknn` (1,159,659 bytes).
  `hard` and `expert` stay ISMCTS (`expert` = ISMCTS-10,000 + exact endgame, Brian 2026-11-11). The
  round-3 policy-value network (about 1.17 MB) stays local: it ties its teacher and costs more.
- **New public items** (nothing `crates/app-api` uses changed shape): `big2_search::RulesMask`
  (`belief="rules"`), `big2_search::NetSampler` with `big2_big2::belief_net::BeliefNet`
  (`belief="net"`, teaching only), `big2_search::infoset_endgame` (settings `infoset_endgame`,
  `infoset_max_cards`; default 10 cards in play), plus the phase-2 `puct` items.
- **`quality` (B20, accepted):** a search move would carry `{"samples": n}` (n = iterations
  actually run); a move from `infoset_endgame` that solved every deal carries `{"exact": true}`.
  When its node budget ran out it carries `{"samples": n}` as before.
- **B2/B3 in three lines:** `expert` vs Charlesworth's network (his rules, 1,000 deals) is a tie,
  +0.06 and +0.04 paired, intervals include 0. Probes against `expert` found no exploit: the
  distilled-start probe ends at +0.01, from scratch −2.57, a rollout attacker −2.21 ± 2.29
  (`1+3`). `expert` vs ISMCTS-1000 at 10,000 deals: +0.92 (1+3), +1.01 (ABAB).
- No action for you. Details: `docs/study-results/build-phase3.md` §9 in big2.

