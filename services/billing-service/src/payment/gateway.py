"""Payment gateway abstraction.

TODO: ZarinPal integration -- the current implementation is a stub that
simulates payment processing.  Replace with real ZarinPal API calls once
merchant credentials are provisioned.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum

from dana_common.logging import get_logger

logger = get_logger(__name__)


class PaymentStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class PaymentResult:
    transaction_id: str
    status: PaymentStatus
    amount_cents: int
    gateway: str = "stub"
    reference: str = ""


class PaymentGateway:
    """Stub payment gateway (TODO: replace with ZarinPal)."""

    async def charge(self, tenant_id: str, amount_cents: int, description: str = "") -> PaymentResult:
        # TODO: Call ZarinPal PaymentRequest API
        tx_id = str(uuid.uuid4())
        logger.info("payment_charge_stub", tenant_id=tenant_id, amount_cents=amount_cents, tx_id=tx_id)
        return PaymentResult(
            transaction_id=tx_id,
            status=PaymentStatus.SUCCESS,
            amount_cents=amount_cents,
            gateway="stub",
            reference=f"stub-{tx_id[:8]}",
        )

    async def refund(self, transaction_id: str, amount_cents: int) -> PaymentResult:
        # TODO: Call ZarinPal refund API
        logger.info("payment_refund_stub", transaction_id=transaction_id, amount_cents=amount_cents)
        return PaymentResult(
            transaction_id=transaction_id,
            status=PaymentStatus.REFUNDED,
            amount_cents=amount_cents,
            gateway="stub",
        )
