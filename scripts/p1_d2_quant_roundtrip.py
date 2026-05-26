"""Phase 1 Day 2 — per-expert quant round-trip on synthetic MoE.

Produces ``reports/quant_roundtrip.csv``.

**Gate (synthetic Gaussian weights, group_size=128):**
  Q8 ≥ 0.9999,  Q4 ≥ 0.995,  Q2 ≥ 0.90

These thresholds are calibrated for **synthetic** weights. Real-model Q4
quant routinely hits 0.999+ because real weights have exploitable low-rank
structure that uniform-Gaussian noise doesn't. Real-weight validation lives
in Phase 2 (rented GPU, actual Qwen3.5 checkpoint).
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.moe_layer import ExpertFFN
from moe_quant.dequantize import dequantize
from moe_quant.quantize import quantize

REPORT_PATH = Path(__file__).parent.parent / "reports" / "quant_roundtrip.csv"
TIER_BITS = {"hot": 8, "warm": 4, "cold": 2}
GROUP_SIZE = 128
NUM_EXPERTS = 8


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def main() -> None:
    torch.manual_seed(0)
    cfg = TinyMoEConfig.micro()
    # Build NUM_EXPERTS independent FFNs as the synthetic experts.
    experts = [ExpertFFN(cfg.hidden_dim, cfg.ffn_dim) for _ in range(NUM_EXPERTS)]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for eid, expert in enumerate(experts):
        # Concatenate every parameter into a single flat fp32 reference tensor.
        ref = torch.cat([p.detach().flatten().float() for p in expert.parameters()])
        for tier, bits in TIER_BITS.items():
            qt = quantize(ref, bits=bits, group_size=GROUP_SIZE)
            recovered = dequantize(qt)
            cos = _cosine(ref, recovered)
            rows.append(
                {
                    "expert_id": eid,
                    "tier": tier,
                    "bits": bits,
                    "num_elements": ref.numel(),
                    "cosine": round(cos, 6),
                }
            )

    with REPORT_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Gate evaluation
    q4 = [r["cosine"] for r in rows if r["bits"] == 4]
    q2 = [r["cosine"] for r in rows if r["bits"] == 2]
    q8 = [r["cosine"] for r in rows if r["bits"] == 8]

    print(f"wrote {REPORT_PATH} ({len(rows)} rows)")
    print(f"  Q8  cosine: min={min(q8):.6f}  mean={sum(q8) / len(q8):.6f}  (gate ≥ 0.9999)")
    print(f"  Q4  cosine: min={min(q4):.6f}  mean={sum(q4) / len(q4):.6f}  (gate ≥ 0.995)")
    print(f"  Q2  cosine: min={min(q2):.6f}  mean={sum(q2) / len(q2):.6f}  (gate ≥ 0.90)")

    failed: list[str] = []
    if min(q8) < 0.9999:
        failed.append(f"Q8 gate FAILED: min cosine {min(q8):.6f} < 0.9999")
    if min(q4) < 0.995:
        failed.append(f"Q4 gate FAILED: min cosine {min(q4):.6f} < 0.995")
    if min(q2) < 0.90:
        failed.append(f"Q2 gate FAILED: min cosine {min(q2):.6f} < 0.90")
    if failed:
        for f in failed:
            print(f"✗ {f}")
        raise SystemExit(1)
    print("✓ all gates passed")


if __name__ == "__main__":
    main()
