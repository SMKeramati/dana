"""Checkpoint and adapter storage manager.

Daneshbonyan: Internal R&D - Custom Model Versioning

Manages fine-tuned model checkpoints:
- Save/load LoRA adapters to/from MinIO
- Version tracking with metadata
- Automatic cleanup of old checkpoints
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "dana_minio")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "")
DEFAULT_BUCKET = os.getenv("CHECKPOINT_BUCKET", "dana-checkpoints")


@dataclass
class AdapterMetadata:
    """Metadata for a saved adapter."""
    adapter_id: str
    base_model: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    lora_rank: int = 16
    lora_alpha: int = 32
    training_samples: int = 0
    training_epochs: int = 0
    final_loss: float = 0.0
    val_loss: float = 0.0
    dataset_name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class CheckpointInfo:
    """Info about a stored checkpoint."""
    adapter_id: str
    path: str
    size_bytes: int = 0
    metadata: AdapterMetadata | None = None


class CheckpointManager:
    """Manage fine-tuned adapter checkpoints in MinIO.

    Daneshbonyan: Internal R&D - Custom Model Storage
    """

    def __init__(
        self,
        bucket: str = DEFAULT_BUCKET,
        max_checkpoints: int = 20,
    ) -> None:
        self._bucket = bucket
        self._max_checkpoints = max_checkpoints
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init MinIO client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    endpoint_url=f"http://{MINIO_ENDPOINT}",
                    aws_access_key_id=MINIO_ACCESS_KEY,
                    aws_secret_access_key=MINIO_SECRET_KEY,
                )
                # Ensure bucket exists
                try:
                    self._client.head_bucket(Bucket=self._bucket)
                except Exception:
                    self._client.create_bucket(Bucket=self._bucket)
                    logger.info("bucket_created", bucket=self._bucket)
            except ImportError:
                raise RuntimeError("boto3 required for checkpoint storage")
        return self._client

    def save_adapter(
        self,
        adapter_dir: str | Path,
        metadata: AdapterMetadata,
    ) -> CheckpointInfo:
        """Upload adapter directory to MinIO.

        Saves all files from adapter_dir under adapters/{adapter_id}/
        """
        adapter_dir = Path(adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

        prefix = f"adapters/{metadata.adapter_id}"
        total_size = 0
        client = self._get_client()

        # Upload all files in adapter dir
        for file_path in adapter_dir.rglob("*"):
            if file_path.is_file():
                key = f"{prefix}/{file_path.relative_to(adapter_dir)}"
                client.upload_file(str(file_path), self._bucket, key)
                total_size += file_path.stat().st_size

        # Upload metadata
        meta_key = f"{prefix}/metadata.json"
        meta_json = json.dumps(asdict(metadata), ensure_ascii=False, indent=2)
        client.put_object(
            Bucket=self._bucket,
            Key=meta_key,
            Body=meta_json.encode("utf-8"),
            ContentType="application/json",
        )

        logger.info(
            "adapter_saved",
            adapter_id=metadata.adapter_id,
            size_bytes=total_size,
            bucket=self._bucket,
        )

        return CheckpointInfo(
            adapter_id=metadata.adapter_id,
            path=f"s3://{self._bucket}/{prefix}",
            size_bytes=total_size,
            metadata=metadata,
        )

    def load_adapter(
        self,
        adapter_id: str,
        target_dir: str | Path,
    ) -> Path:
        """Download adapter from MinIO to local directory."""
        target_dir = Path(target_dir) / adapter_id
        target_dir.mkdir(parents=True, exist_ok=True)

        client = self._get_client()
        prefix = f"adapters/{adapter_id}/"

        response = client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):]
            local_path = target_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(self._bucket, key, str(local_path))

        logger.info("adapter_loaded", adapter_id=adapter_id, target=str(target_dir))
        return target_dir

    def list_adapters(self) -> list[CheckpointInfo]:
        """List all saved adapters."""
        client = self._get_client()
        prefix = "adapters/"
        adapters: list[CheckpointInfo] = []

        response = client.list_objects_v2(Bucket=self._bucket, Prefix=prefix, Delimiter="/")
        for common_prefix in response.get("CommonPrefixes", []):
            adapter_id = common_prefix["Prefix"].strip("/").split("/")[-1]
            meta = self._load_metadata(adapter_id)
            adapters.append(CheckpointInfo(
                adapter_id=adapter_id,
                path=f"s3://{self._bucket}/{common_prefix['Prefix']}",
                metadata=meta,
            ))

        return adapters

    def delete_adapter(self, adapter_id: str) -> bool:
        """Delete an adapter from MinIO."""
        client = self._get_client()
        prefix = f"adapters/{adapter_id}/"

        response = client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        objects = [{"Key": obj["Key"]} for obj in response.get("Contents", [])]

        if objects:
            client.delete_objects(Bucket=self._bucket, Delete={"Objects": objects})
            logger.info("adapter_deleted", adapter_id=adapter_id, files=len(objects))
            return True
        return False

    def cleanup_old(self) -> int:
        """Remove oldest adapters if over max_checkpoints limit."""
        adapters = self.list_adapters()
        if len(adapters) <= self._max_checkpoints:
            return 0

        # Sort by creation date
        adapters.sort(key=lambda a: a.metadata.created_at if a.metadata else "")

        to_remove = adapters[: len(adapters) - self._max_checkpoints]
        for adapter in to_remove:
            self.delete_adapter(adapter.adapter_id)

        logger.info("cleanup_complete", removed=len(to_remove))
        return len(to_remove)

    def _load_metadata(self, adapter_id: str) -> AdapterMetadata | None:
        """Load metadata for an adapter."""
        try:
            client = self._get_client()
            key = f"adapters/{adapter_id}/metadata.json"
            response = client.get_object(Bucket=self._bucket, Key=key)
            data = json.loads(response["Body"].read())
            return AdapterMetadata(**data)
        except Exception:
            return None
