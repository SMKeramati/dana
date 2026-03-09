"""Model loading with KTransformers MoE configuration.

Loads target (large) model and draft (small) model for speculative decoding.
Supports MoE expert partitioning and device placement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class DevicePlacement(Enum):
    """Where a model layer or expert should be placed."""

    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class MoEConfig:
    """KTransformers-style MoE configuration."""

    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_hidden_dim: int = 4096
    expert_intermediate_dim: int = 11008
    gpu_expert_budget: int = 4
    cpu_expert_budget: int = 4


@dataclass
class ModelConfig:
    """Configuration for a loaded model."""

    name: str
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    max_seq_len: int = 4096
    moe_config: MoEConfig | None = None
    device: DevicePlacement = DevicePlacement.GPU


@dataclass
class LoadedModel:
    """Represents a loaded model with its weights and configuration."""

    config: ModelConfig
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    layer_placements: dict[int, DevicePlacement] = field(default_factory=dict)
    is_loaded: bool = False

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights.values())


class ModelLoader:
    """Loads and manages target and draft models for inference."""

    def __init__(
        self,
        gpu_memory_budget_gb: float = 24.0,
        cpu_memory_budget_gb: float = 64.0,
    ) -> None:
        self._gpu_memory_budget = gpu_memory_budget_gb * (1024**3)
        self._cpu_memory_budget = cpu_memory_budget_gb * (1024**3)
        self._target_model: LoadedModel | None = None
        self._draft_model: LoadedModel | None = None

    @property
    def target_model(self) -> LoadedModel | None:
        return self._target_model

    @property
    def draft_model(self) -> LoadedModel | None:
        return self._draft_model

    def _create_mock_weights(self, config: ModelConfig) -> dict[str, np.ndarray]:
        """Create placeholder weight tensors for simulation / testing."""
        weights: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(42)

        # Embedding
        weights["embed_tokens"] = rng.standard_normal(
            (config.vocab_size, config.hidden_dim)
        ).astype(np.float16)

        for layer_idx in range(config.num_layers):
            prefix = f"layers.{layer_idx}"
            # Attention Q/K/V/O projections (small stubs for memory)
            head_dim = config.hidden_dim // config.num_heads
            weights[f"{prefix}.attn.q_proj"] = rng.standard_normal(
                (config.hidden_dim, config.hidden_dim)
            ).astype(np.float16)
            weights[f"{prefix}.attn.k_proj"] = rng.standard_normal(
                (config.hidden_dim, head_dim * config.num_heads)
            ).astype(np.float16)
            weights[f"{prefix}.attn.v_proj"] = rng.standard_normal(
                (config.hidden_dim, head_dim * config.num_heads)
            ).astype(np.float16)

            # MoE or dense FFN
            if config.moe_config is not None:
                moe = config.moe_config
                weights[f"{prefix}.gate"] = rng.standard_normal(
                    (config.hidden_dim, moe.num_experts)
                ).astype(np.float16)
                for e in range(moe.num_experts):
                    weights[f"{prefix}.expert.{e}.up"] = rng.standard_normal(
                        (config.hidden_dim, moe.expert_intermediate_dim // 16)
                    ).astype(np.float16)
                    weights[f"{prefix}.expert.{e}.down"] = rng.standard_normal(
                        (moe.expert_intermediate_dim // 16, config.hidden_dim)
                    ).astype(np.float16)
            else:
                weights[f"{prefix}.ffn.up"] = rng.standard_normal(
                    (config.hidden_dim, config.hidden_dim)
                ).astype(np.float16)
                weights[f"{prefix}.ffn.down"] = rng.standard_normal(
                    (config.hidden_dim, config.hidden_dim)
                ).astype(np.float16)

        # LM head
        weights["lm_head"] = rng.standard_normal(
            (config.vocab_size, config.hidden_dim)
        ).astype(np.float16)

        return weights

    def _compute_layer_placement(
        self, config: ModelConfig
    ) -> dict[int, DevicePlacement]:
        """Decide per-layer device placement based on memory budget."""
        placements: dict[int, DevicePlacement] = {}
        estimated_layer_bytes = (
            config.hidden_dim * config.hidden_dim * 4 * 2  # fp16, 4 matrices
        )
        if config.moe_config:
            estimated_layer_bytes += (
                config.moe_config.num_experts
                * config.moe_config.expert_intermediate_dim
                * config.hidden_dim
                * 2
                * 2  # up + down, fp16
            ) // 16  # mock scaling

        remaining_gpu = self._gpu_memory_budget
        for layer_idx in range(config.num_layers):
            if remaining_gpu >= estimated_layer_bytes:
                placements[layer_idx] = DevicePlacement.GPU
                remaining_gpu -= estimated_layer_bytes
            else:
                placements[layer_idx] = DevicePlacement.CPU

        return placements

    def load_target(self, config: ModelConfig) -> LoadedModel:
        """Load the target (large) model."""
        logger.info("Loading target model: %s", config.name)
        placements = self._compute_layer_placement(config)
        weights = self._create_mock_weights(config)

        model = LoadedModel(
            config=config,
            weights=weights,
            layer_placements=placements,
            is_loaded=True,
        )
        self._target_model = model
        gpu_layers = sum(
            1 for p in placements.values() if p == DevicePlacement.GPU
        )
        logger.info(
            "Target model loaded: %d params, %d/%d layers on GPU",
            model.param_count,
            gpu_layers,
            config.num_layers,
        )
        return model

    def load_draft(self, config: ModelConfig) -> LoadedModel:
        """Load the draft (small) model for speculative decoding."""
        logger.info("Loading draft model: %s", config.name)
        placements = self._compute_layer_placement(config)
        weights = self._create_mock_weights(config)

        model = LoadedModel(
            config=config,
            weights=weights,
            layer_placements=placements,
            is_loaded=True,
        )
        self._draft_model = model
        logger.info("Draft model loaded: %d params", model.param_count)
        return model

    def unload(self, which: str = "all") -> None:
        """Unload models to free memory."""
        if which in ("all", "target"):
            if self._target_model:
                self._target_model.weights.clear()
                self._target_model.is_loaded = False
                self._target_model = None
        if which in ("all", "draft"):
            if self._draft_model:
                self._draft_model.weights.clear()
                self._draft_model.is_loaded = False
                self._draft_model = None
        logger.info("Unloaded models: %s", which)
