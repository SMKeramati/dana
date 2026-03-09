"""Database operations for billing entities."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import InvoiceModel, SubscriptionModel, UsageRecordModel


class BillingRepository:
    """Async repository for billing database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Usage records
    # ------------------------------------------------------------------

    async def insert_usage(self, record: UsageRecordModel) -> UsageRecordModel:
        self._session.add(record)
        await self._session.flush()
        return record

    async def get_usage_by_tenant(
        self,
        tenant_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[UsageRecordModel]:
        stmt = select(UsageRecordModel).where(UsageRecordModel.tenant_id == tenant_id)
        if start:
            stmt = stmt.where(UsageRecordModel.created_at >= start)
        if end:
            stmt = stmt.where(UsageRecordModel.created_at <= end)
        stmt = stmt.order_by(UsageRecordModel.created_at.desc())
        result = await self._session.execute(stmt)
        return result.scalars().all()

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def get_subscription(self, tenant_id: str) -> SubscriptionModel | None:
        stmt = select(SubscriptionModel).where(SubscriptionModel.tenant_id == tenant_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_subscription(self, sub: SubscriptionModel) -> SubscriptionModel:
        self._session.add(sub)
        await self._session.flush()
        return sub

    # ------------------------------------------------------------------
    # Invoices
    # ------------------------------------------------------------------

    async def insert_invoice(self, invoice: InvoiceModel) -> InvoiceModel:
        self._session.add(invoice)
        await self._session.flush()
        return invoice

    async def get_invoices_by_tenant(self, tenant_id: str) -> Sequence[InvoiceModel]:
        stmt = (
            select(InvoiceModel)
            .where(InvoiceModel.tenant_id == tenant_id)
            .order_by(InvoiceModel.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()
