"""Campaign initialization and append helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from data_twin.core.models import CampaignRecord, EventRecord, now_iso
from data_twin.storage.index import DEFAULT_ROOT, campaign_dir
from data_twin.storage.jsonl_store import JsonlStore

JSONL_FILES = (
    "cases.jsonl",
    "runs.jsonl",
    "artifacts.jsonl",
    "metrics.jsonl",
    "evaluations.jsonl",
    "decisions.jsonl",
    "events.jsonl",
)


def load_config(path: Path | str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def init_campaign(config_path: Path | str) -> Path:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    campaign_id = cfg["campaign_id"]
    root = Path(cfg.get("storage", {}).get("root", DEFAULT_ROOT))
    root_dir = campaign_dir(campaign_id, root)
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "artifacts").mkdir(exist_ok=True)
    (root_dir / "exports").mkdir(exist_ok=True)
    for filename in JSONL_FILES:
        (root_dir / filename).touch(exist_ok=True)
    campaign = CampaignRecord(
        campaign_id=campaign_id,
        name=cfg.get("name", campaign_id),
        description=cfg.get("description", ""),
        target_type=cfg.get("target_type", ""),
        target_metadata=cfg.get("target_metadata", {}),
        root_dir=str(root_dir),
        config_path=str(config_path),
        notes=cfg.get("notes", ""),
        schema_version=str(cfg.get("schema_version", "0.1")),
    )
    (root_dir / "campaign.yaml").write_text(yaml.safe_dump(campaign.to_dict(), sort_keys=False), encoding="utf-8")
    store = JsonlStore(root_dir)
    if not store.read("events.jsonl"):
        store.append(
            "events.jsonl",
            EventRecord(
                event_id=f"event_{campaign_id}_init",
                timestamp=now_iso(),
                campaign_id=campaign_id,
                object_type="campaign",
                object_id=campaign_id,
                event_type="campaign_initialized",
                message="Campaign initialized.",
            ).to_dict(),
        )
    return root_dir


def campaign_root(campaign_id: str, root: Path | str = DEFAULT_ROOT) -> Path:
    return campaign_dir(campaign_id, root)
