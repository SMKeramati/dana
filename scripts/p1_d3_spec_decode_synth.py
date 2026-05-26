"""Phase 1 Day 3 — spec-decode + self-draft end-to-end on synthetic MoE.

Produces ``reports/spec_decode_synth.md``.

Validates that the batched tree-verifier and the self-draft pipeline run
end-to-end without exceptions on the tiny synthetic MoE. Acceptance numbers
are reported but **not gated here** — deterministic-random weights make
acceptance arbitrary. Real acceptance gating happens in Phase 2 with real
checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer
from moe_self_draft.self_draft import MoeSelfDrafter
from moe_self_draft.verify import SelfDraftVerifier
from spec_decode_tree.tree_spec import TreeSpeculator
from spec_decode_tree.verify import TreeVerifier

REPORT = Path(__file__).parent.parent / "reports" / "spec_decode_synth.md"


def _new_model(seed: int = 0) -> tuple[TinyMoETransformer, TinyMoEConfig]:
    torch.manual_seed(seed)
    cfg = TinyMoEConfig.micro()
    model = TinyMoETransformer(cfg).eval()
    return model, cfg


def run_tree_spec(num_steps: int = 8, depth: int = 3, width: int = 2) -> dict:
    model, cfg = _new_model()
    speculator = TreeSpeculator(model, depth=depth, width=width)
    verifier = TreeVerifier(model)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    tokens_accepted = 0
    steps = 0
    for _ in range(num_steps):
        tree = speculator.draft(prompt)
        accepted = verifier.verify(tree)
        if not accepted:
            break
        tokens_accepted += len(accepted)
        prompt = torch.cat([prompt, torch.tensor([accepted], dtype=torch.long)], dim=1)
        steps += 1
    return {
        "steps": steps,
        "tokens_accepted": tokens_accepted,
        "tokens_per_step": round(tokens_accepted / max(steps, 1), 3),
        "depth": depth,
        "width": width,
    }


def run_self_draft(num_steps: int = 8, num_draft_tokens: int = 3) -> dict:
    model, cfg = _new_model(seed=1)
    drafter = MoeSelfDrafter(model)
    verifier = SelfDraftVerifier(model)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    proposed = 0
    accepted_total = 0
    for _ in range(num_steps):
        draft = drafter.draft(prompt, num_draft_tokens=num_draft_tokens)
        verified = verifier.verify(prompt, draft)
        proposed += len(draft.draft_tokens)
        # Count how many leading verified tokens exactly match the draft.
        for vt, dt in zip(verified, draft.draft_tokens):
            if vt == dt:
                accepted_total += 1
            else:
                break
        prompt = torch.cat([prompt, torch.tensor([verified], dtype=torch.long)], dim=1)
    acceptance = accepted_total / max(proposed, 1)
    return {
        "proposed_tokens": proposed,
        "accepted_tokens": max(accepted_total, 0),
        "acceptance_rate": round(acceptance, 3),
        "num_draft_tokens": num_draft_tokens,
    }


def main() -> None:
    tree = run_tree_spec()
    self_draft = run_self_draft()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w") as fh:
        fh.write("# Phase 1 Day 3 — synthetic spec-decode validation\n\n")
        fh.write("Smoke run of pillars 5 + 6 on the synthetic tiny MoE. ")
        fh.write("Acceptance numbers are *informational only* — real measurement is Phase 2.\n\n")
        fh.write("## Pillar 5 — tree speculative decoding\n\n")
        fh.write(f"- depth × width: **{tree['depth']} × {tree['width']}**\n")
        fh.write(f"- steps run: {tree['steps']}\n")
        fh.write(f"- tokens accepted total: {tree['tokens_accepted']}\n")
        fh.write(f"- tokens per step: **{tree['tokens_per_step']}**\n")
        fh.write("- ✓ batched verifier ran without exceptions (single forward over flattened tree)\n\n")
        fh.write("## Pillar 6 — self-draft\n\n")
        fh.write(f"- num_draft_tokens per step: **{self_draft['num_draft_tokens']}**\n")
        fh.write(f"- total proposed: {self_draft['proposed_tokens']}\n")
        fh.write(f"- total accepted: {self_draft['accepted_tokens']}\n")
        fh.write(f"- acceptance rate (synthetic, deterministic): **{self_draft['acceptance_rate']}**\n")
        fh.write("- ✓ self-draft pipeline ran end-to-end without exceptions\n\n")
        fh.write("## Gate\n\n")
        fh.write("- [x] spec-decode-tree pipeline end-to-end on tiny MoE\n")
        fh.write("- [x] moe-self-draft pipeline end-to-end on tiny MoE\n")
        fh.write("- [x] batched tree verify code path exercised\n")
        fh.write("- [ ] real-checkpoint acceptance gates — Phase 2\n")
    print(f"wrote {REPORT}")
    print("tree:", tree)
    print("self_draft:", self_draft)


if __name__ == "__main__":
    main()
