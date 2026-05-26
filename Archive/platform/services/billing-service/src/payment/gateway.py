"""Payment gateway abstraction with ZarinPal integration.

Daneshbonyan: Internal R&D - Custom Payment Pipeline

Supports ZarinPal and IDPay gateways for Iranian Rial payments.
Implements the full payment lifecycle: initiate -> redirect -> verify.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from dana_common.logging import get_logger

logger = get_logger(__name__)


class PaymentStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    REFUNDED = "refunded"


class GatewayType(StrEnum):
    ZARINPAL = "zarinpal"
    IDPAY = "idpay"
    STUB = "stub"


@dataclass
class PaymentResult:
    transaction_id: str
    status: PaymentStatus
    amount_cents: int
    gateway: str = "stub"
    reference: str = ""
    authority: str = ""  # ZarinPal authority code
    payment_url: str = ""  # redirect URL for user


# ---------------------------------------------------------------------------
# ZarinPal API client
# ---------------------------------------------------------------------------

_ZARINPAL_API_BASE = "https://api.zarinpal.com/pg/v4/payment"
_ZARINPAL_SANDBOX_BASE = "https://sandbox.zarinpal.com/pg/v4/payment"
_ZARINPAL_STARTPAY = "https://www.zarinpal.com/pg/StartPay"
_ZARINPAL_SANDBOX_STARTPAY = "https://sandbox.zarinpal.com/pg/StartPay"


def _http_post(url: str, body: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    """Send a JSON POST request and return parsed response."""
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class ZarinPalClient:
    """ZarinPal payment gateway client.

    Daneshbonyan: Internal R&D - Iranian Payment Gateway Integration

    Implements:
    - Payment request (get authority + redirect URL)
    - Payment verification (after user returns from bank)
    - Sandbox mode for testing
    """

    def __init__(
        self,
        merchant_id: str | None = None,
        sandbox: bool = False,
        callback_url: str = "https://dana.ir/api/v1/payment/callback",
    ) -> None:
        self._merchant_id = merchant_id or os.getenv("ZARINPAL_MERCHANT_ID", "")
        self._sandbox = sandbox or os.getenv("ZARINPAL_SANDBOX", "false").lower() == "true"
        self._callback_url = callback_url
        self._api_base = _ZARINPAL_SANDBOX_BASE if self._sandbox else _ZARINPAL_API_BASE
        self._startpay_base = _ZARINPAL_SANDBOX_STARTPAY if self._sandbox else _ZARINPAL_STARTPAY

    def request_payment(
        self,
        amount_rial: int,
        description: str,
        email: str = "",
        mobile: str = "",
        metadata: dict[str, str] | None = None,
    ) -> PaymentResult:
        """Initiate a payment and get redirect URL.

        Args:
            amount_rial: Amount in Iranian Rial (min 1000)
            description: Payment description shown to user
            email: Optional buyer email
            mobile: Optional buyer mobile (09xxxxxxxxx)

        Returns:
            PaymentResult with payment_url for redirecting user
        """
        if amount_rial < 1000:
            return PaymentResult(
                transaction_id="",
                status=PaymentStatus.FAILED,
                amount_cents=amount_rial,
                gateway="zarinpal",
                reference="amount_too_low",
            )

        body: dict[str, Any] = {
            "merchant_id": self._merchant_id,
            "amount": amount_rial,
            "description": description,
            "callback_url": self._callback_url,
        }
        if email:
            body["metadata"] = {"email": email}
        if mobile:
            body.setdefault("metadata", {})["mobile"] = mobile
        if metadata:
            body.setdefault("metadata", {}).update(metadata)

        tx_id = str(uuid.uuid4())

        try:
            resp = _http_post(f"{self._api_base}/request.json", body)
            data = resp.get("data", {})
            code = data.get("code", -1)

            if code == 100:
                authority = data["authority"]
                payment_url = f"{self._startpay_base}/{authority}"
                logger.info(
                    "zarinpal_payment_requested",
                    authority=authority,
                    amount=amount_rial,
                    tx_id=tx_id,
                )
                return PaymentResult(
                    transaction_id=tx_id,
                    status=PaymentStatus.PENDING,
                    amount_cents=amount_rial,
                    gateway="zarinpal",
                    authority=authority,
                    payment_url=payment_url,
                )
            else:
                errors = resp.get("errors", {})
                logger.error(
                    "zarinpal_request_failed",
                    code=code,
                    errors=errors,
                    tx_id=tx_id,
                )
                return PaymentResult(
                    transaction_id=tx_id,
                    status=PaymentStatus.FAILED,
                    amount_cents=amount_rial,
                    gateway="zarinpal",
                    reference=f"error:{code}",
                )
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.error("zarinpal_network_error", error=str(exc), tx_id=tx_id)
            return PaymentResult(
                transaction_id=tx_id,
                status=PaymentStatus.FAILED,
                amount_cents=amount_rial,
                gateway="zarinpal",
                reference=f"network_error:{type(exc).__name__}",
            )

    def verify_payment(self, authority: str, amount_rial: int) -> PaymentResult:
        """Verify a payment after user returns from bank.

        Call this when user is redirected back to callback_url
        with ?Authority=xxx&Status=OK parameters.
        """
        body = {
            "merchant_id": self._merchant_id,
            "authority": authority,
            "amount": amount_rial,
        }
        tx_id = str(uuid.uuid4())

        try:
            resp = _http_post(f"{self._api_base}/verify.json", body)
            data = resp.get("data", {})
            code = data.get("code", -1)

            if code in (100, 101):
                ref_id = str(data.get("ref_id", ""))
                logger.info(
                    "zarinpal_payment_verified",
                    authority=authority,
                    ref_id=ref_id,
                    amount=amount_rial,
                )
                return PaymentResult(
                    transaction_id=tx_id,
                    status=PaymentStatus.SUCCESS,
                    amount_cents=amount_rial,
                    gateway="zarinpal",
                    authority=authority,
                    reference=ref_id,
                )
            else:
                logger.error("zarinpal_verify_failed", code=code, authority=authority)
                return PaymentResult(
                    transaction_id=tx_id,
                    status=PaymentStatus.FAILED,
                    amount_cents=amount_rial,
                    gateway="zarinpal",
                    authority=authority,
                    reference=f"verify_error:{code}",
                )
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.error("zarinpal_verify_network_error", error=str(exc))
            return PaymentResult(
                transaction_id=tx_id,
                status=PaymentStatus.FAILED,
                amount_cents=amount_rial,
                gateway="zarinpal",
                authority=authority,
                reference=f"network_error:{type(exc).__name__}",
            )


# ---------------------------------------------------------------------------
# Unified payment gateway
# ---------------------------------------------------------------------------


class PaymentGateway:
    """Unified payment gateway with multiple backend support.

    Daneshbonyan: Internal R&D - Custom Payment Abstraction Layer
    """

    def __init__(
        self,
        gateway_type: GatewayType | None = None,
        zarinpal_merchant_id: str | None = None,
        sandbox: bool = False,
    ) -> None:
        self._gateway_type = gateway_type or GatewayType(os.getenv("PAYMENT_GATEWAY", "stub"))
        self._zarinpal: ZarinPalClient | None = None
        if self._gateway_type == GatewayType.ZARINPAL:
            self._zarinpal = ZarinPalClient(
                merchant_id=zarinpal_merchant_id,
                sandbox=sandbox,
            )

    async def charge(
        self,
        tenant_id: str,
        amount_cents: int,
        description: str = "",
    ) -> PaymentResult:
        """Initiate a payment."""
        if self._gateway_type == GatewayType.ZARINPAL and self._zarinpal:
            amount_rial = amount_cents * 10  # 1 Toman = 10 Rial
            result = self._zarinpal.request_payment(
                amount_rial=amount_rial,
                description=description or f"Dana AI - {tenant_id}",
            )
            logger.info("payment_initiated", tenant_id=tenant_id, gateway="zarinpal",
                        status=result.status, amount=amount_cents)
            return result

        # Stub gateway for development
        tx_id = str(uuid.uuid4())
        logger.info("payment_charge_stub", tenant_id=tenant_id, amount_cents=amount_cents)
        return PaymentResult(
            transaction_id=tx_id,
            status=PaymentStatus.SUCCESS,
            amount_cents=amount_cents,
            gateway="stub",
            reference=f"stub-{tx_id[:8]}",
        )

    async def verify(self, authority: str, amount_cents: int) -> PaymentResult:
        """Verify a payment after callback."""
        if self._gateway_type == GatewayType.ZARINPAL and self._zarinpal:
            amount_rial = amount_cents * 10
            return self._zarinpal.verify_payment(authority, amount_rial)

        return PaymentResult(
            transaction_id=str(uuid.uuid4()),
            status=PaymentStatus.SUCCESS,
            amount_cents=amount_cents,
            gateway="stub",
            reference="stub-verified",
        )

    async def refund(self, transaction_id: str, amount_cents: int) -> PaymentResult:
        """Refund a payment (ZarinPal does not support programmatic refunds)."""
        logger.info("payment_refund", transaction_id=transaction_id, amount_cents=amount_cents)
        return PaymentResult(
            transaction_id=transaction_id,
            status=PaymentStatus.REFUNDED,
            amount_cents=amount_cents,
            gateway=self._gateway_type,
        )
