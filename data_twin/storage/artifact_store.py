"""Artifact attachment with checksum indexing."""

from __future__ import annotations

import shutil
from pathlib import Path

from data_twin.core.hashing import file_checksum
from data_twin.core.ids import make_id
from data_twin.core.models import ArtifactRecord, EventRecord, now_iso
from data_twin.storage.jsonl_store import JsonlStore


def attach_artifact(
    campaign_root: Path,
    *,
    campaign_id: str,
    case_id: str,
    run_id: str,
    artifact_path: Path,
    artifact_type: str,
    description: str = "",
    copy: bool = False,
) -> ArtifactRecord:
    source = Path(artifact_path)
    target = source
    if copy:
        target_dir = campaign_root / "artifacts" / case_id / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    checksum = file_checksum(target) if target.exists() and target.is_file() else ""
    relative = str(target.relative_to(campaign_root)) if target.exists() and target.is_relative_to(campaign_root) else str(target)
    artifact = ArtifactRecord(
        artifact_id=make_id("artifact", {"run_id": run_id, "path": str(target), "type": artifact_type}),
        campaign_id=campaign_id,
        case_id=case_id,
        run_id=run_id,
        artifact_type=artifact_type,
        path=str(target),
        relative_path=relative,
        checksum=checksum,
        description=description,
    )
    store = JsonlStore(campaign_root)
    store.append("artifacts.jsonl", artifact.to_dict())
    store.append(
        "events.jsonl",
        EventRecord(
            event_id=make_id("event", artifact.to_dict()),
            timestamp=now_iso(),
            campaign_id=campaign_id,
            object_type="artifact",
            object_id=artifact.artifact_id,
            event_type="artifact_attached",
            message=f"Attached {artifact_type}",
        ).to_dict(),
    )
    return artifact
