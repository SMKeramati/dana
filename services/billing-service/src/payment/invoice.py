"""Invoice generation for billing periods."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)


class InvoiceStatus(StrEnum):
    DRAFT = "draft"
    ISSUED = "issued"
    PAID = "paid"
    OVERDUE = "overdue"


@dataclass
class LineItem:
    description: str
    quantity: int
    unit_price_cents: float
    total_cents: float


@dataclass
class Invoice:
    invoice_id: str
    tenant_id: str
    period_start: datetime
    period_end: datetime
    line_items: list[LineItem] = field(default_factory=list)
    status: InvoiceStatus = InvoiceStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_cents: float = 0.0

    def add_line_item(self, description: str, quantity: int, unit_price_cents: float) -> None:
        item_total = quantity * unit_price_cents
        self.line_items.append(
            LineItem(
                description=description,
                quantity=quantity,
                unit_price_cents=unit_price_cents,
                total_cents=item_total,
            )
        )
        self.total_cents += item_total

    def finalise(self) -> None:
        self.status = InvoiceStatus.ISSUED
        logger.info(
            "invoice_issued",
            invoice_id=self.invoice_id,
            tenant_id=self.tenant_id,
            total_cents=round(self.total_cents, 2),
        )


class InvoiceGenerator:
    """Creates invoices from aggregated usage data."""

    def create_invoice(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        usage_summary: dict[str, Any],
    ) -> Invoice:
        invoice = Invoice(
            invoice_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )

        for model, totals in usage_summary.items():
            prompt_tokens = totals.get("prompt_tokens", 0)
            completion_tokens = totals.get("completion_tokens", 0)
            cost_cents = totals.get("cost_cents", 0.0)

            invoice.add_line_item(
                description=f"{model} - prompt tokens",
                quantity=prompt_tokens,
                unit_price_cents=cost_cents / max(prompt_tokens + completion_tokens, 1),
            )
            invoice.add_line_item(
                description=f"{model} - completion tokens",
                quantity=completion_tokens,
                unit_price_cents=cost_cents / max(prompt_tokens + completion_tokens, 1),
            )

        return invoice
