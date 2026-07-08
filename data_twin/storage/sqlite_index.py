"""SQLite materialized index for Data Twin JSONL campaigns."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from data_twin.core.models import MODEL_BY_FILE
from data_twin.storage.index import DEFAULT_ROOT
from data_twin.storage.jsonl_store import JsonlStore


DEFAULT_INDEX_PATH = DEFAULT_ROOT / "data_twin_index.sqlite"

TABLE_BY_FILE = {
    "briefs.jsonl": "briefs",
    "cases.jsonl": "cases",
    "runs.jsonl": "runs",
    "artifacts.jsonl": "artifacts",
    "metrics.jsonl": "metrics",
    "evaluations.jsonl": "evaluations",
    "decisions.jsonl": "decisions",
    "reviews.jsonl": "reviews",
    "events.jsonl": "events",
}

ID_FIELD_BY_TABLE = {
    "briefs": "brief_id",
    "cases": "case_id",
    "runs": "run_id",
    "artifacts": "artifact_id",
    "metrics": "metric_id",
    "evaluations": "evaluation_id",
    "decisions": "decision_id",
    "reviews": "review_id",
    "events": "event_id",
}


def _json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def _stable_row_id(table: str, row: dict[str, Any], ordinal: int) -> str:
    key = row.get(ID_FIELD_BY_TABLE[table])
    if key:
        return str(key)
    digest = hashlib.sha1(f"{table}:{ordinal}:{_json(row)}".encode("utf-8")).hexdigest()[:16]
    return f"{table}_{digest}"


def _campaign_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / "campaign.yaml").exists()
            or (path / "lifecycle.json").exists()
            or any((path / filename).exists() for filename in MODEL_BY_FILE)
        )
    )


def _read_lifecycle(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "lifecycle.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_campaign_yaml(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "campaign.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def connect(index_path: Path | str = DEFAULT_INDEX_PATH) -> sqlite3.Connection:
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            root TEXT NOT NULL,
            name TEXT,
            state TEXT,
            updated_at TEXT,
            json TEXT NOT NULL
        )
        """
    )
    for table in TABLE_BY_FILE.values():
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                created_at TEXT,
                json TEXT NOT NULL
            )
            """
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_campaign ON {table}(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(json)")
    conn.commit()


def rebuild_index(
    root: Path | str = DEFAULT_ROOT,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, int]:
    root = Path(root)
    conn = connect(index_path)
    create_schema(conn)
    with conn:
        for table in ("campaigns", *TABLE_BY_FILE.values()):
            conn.execute(f"DELETE FROM {table}")
        counts = {"campaigns": 0, **{table: 0 for table in TABLE_BY_FILE.values()}}
        for campaign_root in _campaign_dirs(root):
            campaign = _read_campaign_yaml(campaign_root)
            lifecycle = _read_lifecycle(campaign_root)
            campaign_id = str(campaign.get("campaign_id") or campaign_root.name)
            campaign_payload = {
                "campaign": campaign,
                "lifecycle": lifecycle,
            }
            conn.execute(
                """
                INSERT OR REPLACE INTO campaigns
                (campaign_id, root, name, state, updated_at, json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    str(campaign_root),
                    campaign.get("name", campaign_id),
                    lifecycle.get("state", campaign.get("status", "")),
                    lifecycle.get("updated_at", campaign.get("created_at", "")),
                    _json(campaign_payload),
                ),
            )
            counts["campaigns"] += 1
            store = JsonlStore(campaign_root)
            for filename, table in TABLE_BY_FILE.items():
                if filename not in MODEL_BY_FILE:
                    continue
                for ordinal, row in enumerate(store.read(filename), start=1):
                    row_campaign_id = str(row.get("campaign_id") or campaign_id)
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {table}
                        (id, campaign_id, created_at, json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            _stable_row_id(table, row, ordinal),
                            row_campaign_id,
                            row.get("created_at") or row.get("timestamp") or "",
                            _json(row),
                        ),
                    )
                    counts[table] += 1
    conn.close()
    return counts


def campaign_status(
    campaign_id: str,
    root: Path | str = DEFAULT_ROOT,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, Any]:
    index_path = Path(index_path)
    if not index_path.exists():
        rebuild_index(root, index_path)
    conn = connect(index_path)
    create_schema(conn)
    campaign = conn.execute(
        "SELECT campaign_id, root, name, state, updated_at, json FROM campaigns WHERE campaign_id = ?",
        (campaign_id,),
    ).fetchone()
    if campaign is None:
        raise KeyError(f"Campaign not found in index: {campaign_id}")
    counts: dict[str, int] = {}
    for table in TABLE_BY_FILE.values():
        counts[table] = int(
            conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE campaign_id = ?", (campaign_id,)).fetchone()["n"]
        )
    runs_by_status = {
        row["status"]: row["n"]
        for row in conn.execute(
            """
            SELECT json_extract(json, '$.status') AS status, COUNT(*) AS n
            FROM runs
            WHERE campaign_id = ?
            GROUP BY status
            """,
            (campaign_id,),
        )
    }
    latest_review = conn.execute(
        """
        SELECT json FROM reviews
        WHERE campaign_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (campaign_id,),
    ).fetchone()
    latest_decision = conn.execute(
        """
        SELECT json FROM decisions
        WHERE campaign_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (campaign_id,),
    ).fetchone()
    payload = json.loads(campaign["json"])
    conn.close()
    return {
        "campaign_id": campaign["campaign_id"],
        "name": campaign["name"],
        "root": campaign["root"],
        "state": campaign["state"],
        "updated_at": campaign["updated_at"],
        "counts": counts,
        "runs_by_status": runs_by_status,
        "latest_review": json.loads(latest_review["json"]) if latest_review else None,
        "latest_decision": json.loads(latest_decision["json"]) if latest_decision else None,
        "lifecycle": payload.get("lifecycle") or {},
    }


def compare_campaigns(
    campaign_ids: list[str],
    root: Path | str = DEFAULT_ROOT,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, Any]:
    if len(campaign_ids) < 2:
        raise ValueError("compare requires at least two campaigns")
    index_path = Path(index_path)
    if not index_path.exists():
        rebuild_index(root, index_path)
    statuses = [campaign_status(campaign_id, root, index_path) for campaign_id in campaign_ids]
    conn = connect(index_path)
    metrics: dict[str, dict[str, Any]] = {}
    for campaign_id in campaign_ids:
        rows = conn.execute(
            """
            SELECT
              json_extract(json, '$.metric_name') AS metric_name,
              COUNT(*) AS n,
              MIN(CAST(json_extract(json, '$.metric_value') AS REAL)) AS min_value,
              MAX(CAST(json_extract(json, '$.metric_value') AS REAL)) AS max_value
            FROM metrics
            WHERE campaign_id = ? AND json_extract(json, '$.available') = 1
            GROUP BY metric_name
            ORDER BY metric_name
            """,
            (campaign_id,),
        ).fetchall()
        metrics[campaign_id] = {
            row["metric_name"]: {
                "count": row["n"],
                "min": row["min_value"],
                "max": row["max_value"],
            }
            for row in rows
            if row["metric_name"]
        }
    conn.close()
    return {"campaigns": statuses, "metrics": metrics}
