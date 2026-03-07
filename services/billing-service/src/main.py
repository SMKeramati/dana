"""Billing service -- FastAPI application (port 8003)."""
from __future__ import annotations

from datetime import UTC, datetime

from dana_common.logging import get_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .metering.token_counter import count_tokens
from .metering.usage_aggregator import TimeWindow, UsageAggregator, UsageRecord
from .plans.pricing import PricingEngine
from .plans.subscription import PLAN_CATALOGUE, PlanTier

logger = get_logger(__name__)

app = FastAPI(title="Dana Billing Service", version="0.1.0")

# In-process singletons (replaced by DI in production)
_aggregator = UsageAggregator()
_pricing = PricingEngine()


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class UsageRequest(BaseModel):
    tenant_id: str
    model: str
    prompt: str
    completion: str
    exact_prompt_tokens: int | None = None
    exact_completion_tokens: int | None = None


class UsageResponse(BaseModel):
    tenant_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_cents: float


class PlanResponse(BaseModel):
    tier: str
    display_name: str
    monthly_token_limit: int
    price_cents: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "billing"}


@app.post("/v1/usage", response_model=UsageResponse)
async def record_usage(req: UsageRequest) -> UsageResponse:
    tc = count_tokens(
        prompt=req.prompt,
        completion=req.completion,
        model=req.model,
        exact_prompt_tokens=req.exact_prompt_tokens,
        exact_completion_tokens=req.exact_completion_tokens,
    )

    record = UsageRecord(
        org_id=req.tenant_id,
        model=req.model,
        input_tokens=tc.prompt_tokens,
        output_tokens=tc.completion_tokens,
        timestamp=datetime.now(UTC).timestamp(),
    )
    _aggregator.ingest(record)

    price = _pricing.compute(
        model=req.model,
        prompt_tokens=tc.prompt_tokens,
        completion_tokens=tc.completion_tokens,
        tier=PlanTier.FREE,
    )

    return UsageResponse(
        tenant_id=req.tenant_id,
        model=req.model,
        prompt_tokens=tc.prompt_tokens,
        completion_tokens=tc.completion_tokens,
        total_tokens=tc.total_tokens,
        cost_cents=round(price.total_cents, 6),
    )


@app.get("/v1/usage/{tenant_id}")
async def get_usage(tenant_id: str, window: str = "monthly") -> dict:
    try:
        tw = TimeWindow(window)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid window: {window}")

    # Peek all models for this tenant
    buckets = []
    for (o, m, w, bk), acc in _aggregator._accumulators.items():
        if o == tenant_id and w == tw:
            buckets.append({
                "key": bk,
                "model": m,
                "total_tokens": acc["input_tokens"] + acc["output_tokens"],
                "request_count": acc["request_count"],
            })

    return {
        "tenant_id": tenant_id,
        "window": window,
        "buckets": buckets,
    }


@app.get("/v1/plans", response_model=list[PlanResponse])
async def list_plans() -> list[PlanResponse]:
    return [
        PlanResponse(
            tier=p.tier.value,
            display_name=p.display_name,
            monthly_token_limit=p.monthly_token_limit,
            price_cents=p.price_cents,
        )
        for p in PLAN_CATALOGUE.values()
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
