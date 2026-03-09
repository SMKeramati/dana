"""TierBitwidthAssigner — assign optimal bit-widths based on tier + sensitivity.

Default mapping:
  hot  (VRAM)  → Q8 or FP16
  ram  (RAM)   → Q4
  ssd  (SSD)   → Q2

Override: if sensitivity profile shows Q2 quality too low for a cold expert,
bump to Q4 to protect quality.
"""

from __future__ import annotations

from moe_quant.sensitivity import SensitivityProfile


# Default tier → bits mapping
TIER_DEFAULT_BITS: dict[str, int] = {
    "hot": 8,
    "ram": 4,
    "ssd": 2,
}


class TierBitwidthAssigner:
    """Assign quantization bit-widths based on tier and sensitivity profiles.

    Usage:
        assigner = TierBitwidthAssigner(min_quality=0.95)
        assignments = assigner.assign(
            tier_map={"e0": "hot", "e1": "ram", "e2": "ssd"},
            profiles={"e0": profile0, "e1": profile1, "e2": profile2},
        )
        # → {"e0": 8, "e1": 4, "e2": 4}  (e2 bumped from 2→4 due to sensitivity)
    """

    def __init__(
        self,
        min_quality: float = 0.95,
        tier_bits: dict[str, int] = None,
    ) -> None:
        """
        Args:
            min_quality: Minimum acceptable cosine similarity score.
                         If the tier's default bits don't meet this, bits are increased.
            tier_bits: Override default tier → bits mapping.
        """
        self.min_quality = min_quality
        self.tier_bits = tier_bits or dict(TIER_DEFAULT_BITS)

    def assign(
        self,
        tier_map: dict[str, str],
        profiles: dict[str, SensitivityProfile] = None,
    ) -> dict[str, int]:
        """Compute optimal bit-width for each expert.

        Args:
            tier_map: {expert_key: tier_name}
            profiles: Optional {expert_key: SensitivityProfile}. If None,
                      uses default tier mapping without quality check.

        Returns:
            {expert_key: bits}
        """
        result: dict[str, int] = {}
        for key, tier in tier_map.items():
            default_bits = self.tier_bits.get(tier, 8)

            if profiles and key in profiles:
                profile = profiles[key]
                # Start from tier default; bump up if quality too low
                bits = default_bits
                quality = profile.scores.get(bits, 1.0)
                if quality < self.min_quality:
                    bits = profile.recommended_bits(self.min_quality)
            else:
                bits = default_bits

            result[key] = bits

        return result

    def assign_by_expert_id(
        self,
        tier_map: dict[int, str],
        profiles: dict[int, SensitivityProfile] = None,
    ) -> dict[int, int]:
        """Same as assign() but keyed by integer expert ID."""
        str_tier = {str(k): v for k, v in tier_map.items()}
        str_profiles = ({str(k): v for k, v in profiles.items()}
                        if profiles else None)
        str_result = self.assign(str_tier, str_profiles)
        return {int(k): v for k, v in str_result.items()}
