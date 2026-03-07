"""Custom MoE expert offloading manager.

Daneshbonyan: Internal Design & Development

Profiles expert activation frequency across inference requests and manages
GPU/RAM placement of individual experts.  Hot experts (frequently activated)
stay in VRAM; cold experts are offloaded to system RAM and fetched on
demand.  The profiler uses an exponential moving average to track activation
frequency so the placement adapts to shifting workloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExpertProfile:
    """Activation profile for a single expert."""

    layer_idx: int
    expert_idx: int
    total_activations: int = 0
    ema_frequency: float = 0.0
    is_on_gpu: bool = False
    size_bytes: int = 0

    @property
    def expert_key(self) -> tuple[int, int]:
        return (self.layer_idx, self.expert_idx)


@dataclass
class OffloadStats:
    """Aggregate offloading statistics."""

    total_fetches: int = 0
    total_evictions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ExpertOffloadManager:
    """Manages GPU/CPU placement of MoE experts based on activation frequency.

    Daneshbonyan: Internal Design & Development

    Parameters
    ----------
    num_layers : int
        Number of MoE layers in the model.
    num_experts : int
        Number of experts per layer.
    gpu_expert_budget : int
        Maximum number of experts that can reside on GPU at once.
    expert_size_bytes : int
        Size of a single expert's parameters in bytes.
    ema_alpha : float
        Smoothing factor for the exponential moving average of activation
        frequency.  Higher values weight recent activations more.
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_experts: int = 8,
        gpu_expert_budget: int = 64,
        expert_size_bytes: int = 50 * 1024 * 1024,  # 50 MiB default
        ema_alpha: float = 0.15,
    ) -> None:
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._gpu_budget = gpu_expert_budget
        self._ema_alpha = ema_alpha

        # Initialise profiles
        self._profiles: dict[tuple[int, int], ExpertProfile] = {}
        for l in range(num_layers):
            for e in range(num_experts):
                self._profiles[(l, e)] = ExpertProfile(
                    layer_idx=l,
                    expert_idx=e,
                    size_bytes=expert_size_bytes,
                )

        self.stats = OffloadStats()

        # Initial placement: put the first gpu_budget experts on GPU
        self._gpu_residents: set[tuple[int, int]] = set()
        self._initial_placement()

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def _initial_placement(self) -> None:
        """Place the first N experts on GPU uniformly across layers."""
        keys = sorted(self._profiles.keys())
        for key in keys[: self._gpu_budget]:
            self._profiles[key].is_on_gpu = True
            self._gpu_residents.add(key)

    @property
    def gpu_resident_count(self) -> int:
        return len(self._gpu_residents)

    @property
    def total_experts(self) -> int:
        return len(self._profiles)

    # ------------------------------------------------------------------
    # Activation recording
    # ------------------------------------------------------------------

    def record_activations(
        self, layer_idx: int, activated_experts: list[int]
    ) -> None:
        """Record which experts were activated in a forward pass.

        Parameters
        ----------
        layer_idx : int
        activated_experts : list[int]
            Indices of experts that were selected by the gating function.
        """
        for e in activated_experts:
            key = (layer_idx, e)
            profile = self._profiles.get(key)
            if profile is None:
                continue
            profile.total_activations += 1
            profile.ema_frequency = (
                self._ema_alpha * 1.0
                + (1 - self._ema_alpha) * profile.ema_frequency
            )

        # Decay non-activated experts in this layer
        for e in range(self._num_experts):
            if e not in activated_experts:
                key = (layer_idx, e)
                profile = self._profiles.get(key)
                if profile is not None:
                    profile.ema_frequency *= (1 - self._ema_alpha)

    def record_batch_activations(
        self, gate_outputs: dict[int, np.ndarray]
    ) -> None:
        """Record activations from a full batch across all layers.

        Parameters
        ----------
        gate_outputs : dict[int, np.ndarray]
            Mapping from layer index to gating scores of shape
            ``(batch_size, num_experts)``.  The top-k experts per token
            are considered activated.
        """
        for layer_idx, scores in gate_outputs.items():
            if scores.ndim == 1:
                scores = scores.reshape(1, -1)
            # Top-2 experts per token (standard MoE)
            top_k = min(2, scores.shape[1])
            for row in scores:
                activated = np.argsort(row)[-top_k:].tolist()
                self.record_activations(layer_idx, activated)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self) -> list[tuple[tuple[int, int], str]]:
        """Rebalance expert placement based on current activation profiles.

        Returns a list of ``(expert_key, action)`` where action is
        ``"fetch"`` (CPU -> GPU) or ``"evict"`` (GPU -> CPU).
        """
        all_profiles = sorted(
            self._profiles.values(),
            key=lambda p: p.ema_frequency,
            reverse=True,
        )

        desired_gpu = set()
        for p in all_profiles[: self._gpu_budget]:
            desired_gpu.add(p.expert_key)

        to_fetch = desired_gpu - self._gpu_residents
        to_evict = self._gpu_residents - desired_gpu

        actions: list[tuple[tuple[int, int], str]] = []

        for key in to_evict:
            self._profiles[key].is_on_gpu = False
            self._gpu_residents.discard(key)
            self.stats.total_evictions += 1
            actions.append((key, "evict"))

        for key in to_fetch:
            self._profiles[key].is_on_gpu = True
            self._gpu_residents.add(key)
            self.stats.total_fetches += 1
            actions.append((key, "fetch"))

        if actions:
            logger.info(
                "Expert rebalance: %d fetched, %d evicted",
                len(to_fetch),
                len(to_evict),
            )

        return actions

    def get_expert_location(self, layer_idx: int, expert_idx: int) -> str:
        """Return 'gpu' or 'cpu' for the given expert."""
        key = (layer_idx, expert_idx)
        if key in self._gpu_residents:
            self.stats.cache_hits += 1
            return "gpu"
        self.stats.cache_misses += 1
        return "cpu"

    def fetch_on_demand(self, layer_idx: int, expert_idx: int) -> bool:
        """Ensure an expert is on GPU, evicting the coldest if necessary.

        Returns True if a fetch (swap) was performed, False if already on GPU.
        """
        key = (layer_idx, expert_idx)
        if key in self._gpu_residents:
            return False

        if len(self._gpu_residents) >= self._gpu_budget:
            # Evict the coldest GPU-resident expert
            coldest = min(
                self._gpu_residents,
                key=lambda k: self._profiles[k].ema_frequency,
            )
            self._profiles[coldest].is_on_gpu = False
            self._gpu_residents.discard(coldest)
            self.stats.total_evictions += 1

        self._profiles[key].is_on_gpu = True
        self._gpu_residents.add(key)
        self.stats.total_fetches += 1
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_frequency_distribution(self) -> np.ndarray:
        """Return a (num_layers, num_experts) array of EMA frequencies."""
        freq = np.zeros((self._num_layers, self._num_experts), dtype=np.float64)
        for (l, e), p in self._profiles.items():
            freq[l, e] = p.ema_frequency
        return freq

    def get_gpu_utilization(self) -> float:
        """Fraction of GPU expert budget currently used."""
        return len(self._gpu_residents) / max(self._gpu_budget, 1)
