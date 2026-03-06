"""Model version management with rollback support.

Tracks all deployed model versions, supports promotion, deprecation, and
instant rollback to any previous version.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)


class VersionStatus(StrEnum):
    STAGING = "staging"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """A single version of a model in the registry."""

    model_name: str
    version: str
    endpoint_url: str
    status: VersionStatus = VersionStatus.STAGING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_servable(self) -> bool:
        return self.status in (VersionStatus.STAGING, VersionStatus.ACTIVE)


class ModelStore:
    """In-memory model version store with rollback capability.

    In production this delegates to the database; the in-memory version is
    used for tests and local development.
    """

    def __init__(self) -> None:
        # {model_name: [versions ordered oldest-first]}
        self._versions: dict[str, list[ModelVersion]] = {}

    def register(
        self,
        model_name: str,
        version: str,
        endpoint_url: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Register a new model version in staging."""
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            endpoint_url=endpoint_url,
            metadata=metadata or {},
        )
        self._versions.setdefault(model_name, []).append(mv)
        logger.info("model_registered", model=model_name, version=version)
        return mv

    def promote(self, model_name: str, version: str) -> ModelVersion:
        """Promote a version to active, deprecating the current active version."""
        target = self._find(model_name, version)
        if target is None:
            raise ValueError(f"Version {version} not found for {model_name}")

        # Deprecate current active version(s)
        for mv in self._versions.get(model_name, []):
            if mv.status == VersionStatus.ACTIVE:
                mv.status = VersionStatus.DEPRECATED

        target.status = VersionStatus.ACTIVE
        target.promoted_at = datetime.now(UTC)
        logger.info("model_promoted", model=model_name, version=version)
        return target

    def rollback(self, model_name: str) -> ModelVersion:
        """Roll back to the most recent deprecated version.

        The current active version is marked as ROLLED_BACK, and the most
        recently deprecated version is re-promoted to ACTIVE.
        """
        versions = self._versions.get(model_name, [])
        current_active = next((v for v in reversed(versions) if v.status == VersionStatus.ACTIVE), None)
        previous = next((v for v in reversed(versions) if v.status == VersionStatus.DEPRECATED), None)

        if previous is None:
            raise ValueError(f"No previous version to roll back to for {model_name}")

        if current_active:
            current_active.status = VersionStatus.ROLLED_BACK

        previous.status = VersionStatus.ACTIVE
        previous.promoted_at = datetime.now(UTC)
        logger.info("model_rolled_back", model=model_name, to_version=previous.version)
        return previous

    def get_active(self, model_name: str) -> ModelVersion | None:
        """Return the currently active version, or None."""
        for mv in reversed(self._versions.get(model_name, [])):
            if mv.status == VersionStatus.ACTIVE:
                return mv
        return None

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        return list(self._versions.get(model_name, []))

    def list_models(self) -> list[str]:
        return list(self._versions.keys())

    def _find(self, model_name: str, version: str) -> ModelVersion | None:
        for mv in self._versions.get(model_name, []):
            if mv.version == version:
                return mv
        return None
