"""MoE Router Predict — expert activation predictor for lookahead prefetching."""

from moe_router_predict.predictor import RouterPredictor, ExpertPrediction
from moe_router_predict.residency import ExpertResidencyTracker
from moe_router_predict.async_loader import AsyncExpertLoader
from moe_router_predict.scheduler import PrefetchScheduler

__all__ = [
    "RouterPredictor",
    "ExpertPrediction",
    "ExpertResidencyTracker",
    "AsyncExpertLoader",
    "PrefetchScheduler",
]

__version__ = "0.1.0"
