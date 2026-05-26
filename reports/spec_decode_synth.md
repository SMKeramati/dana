# Phase 1 Day 3 — synthetic spec-decode validation

Smoke run of pillars 5 + 6 on the synthetic tiny MoE. Acceptance numbers are *informational only* — real measurement is Phase 2.

## Pillar 5 — tree speculative decoding

- depth × width: **3 × 2**
- steps run: 8
- tokens accepted total: 24
- tokens per step: **3.0**
- ✓ batched verifier ran without exceptions (single forward over flattened tree)

## Pillar 6 — self-draft

- num_draft_tokens per step: **3**
- total proposed: 24
- total accepted: 24
- acceptance rate (synthetic, deterministic): **1.0**
- ✓ self-draft pipeline ran end-to-end without exceptions

## Gate

- [x] spec-decode-tree pipeline end-to-end on tiny MoE
- [x] moe-self-draft pipeline end-to-end on tiny MoE
- [x] batched tree verify code path exercised
- [ ] real-checkpoint acceptance gates — Phase 2
