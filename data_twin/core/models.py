"""Validated Data Twin record models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BaseRecord:
    required_fields: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        kwargs = {name: data.get(name) for name in fields if name != "required_fields"}
        return cls(**kwargs)


@dataclass
class CampaignRecord(BaseRecord):
    campaign_id: str
    name: str
    description: str = ""
    target_type: str = ""
    target_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    created_by: str = ""
    status: str = "active"
    root_dir: str = ""
    config_path: str = ""
    notes: str = ""
    schema_version: str = "0.1"

    required_fields: ClassVar[tuple[str, ...]] = ("campaign_id", "name", "created_at", "status", "root_dir")


@dataclass
class CaseRecord(BaseRecord):
    case_id: str
    campaign_id: str
    generation_index: int = 0
    parent_case_ids: list[str] = field(default_factory=list)
    proposal_source: str = ""
    proposal_reason: str = ""
    parameter_hash: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    input_refs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    status: str = "proposed"
    created_at: str = field(default_factory=now_iso)
    notes: str = ""

    required_fields: ClassVar[tuple[str, ...]] = ("case_id", "campaign_id", "status", "created_at")


@dataclass
class RunRecord(BaseRecord):
    run_id: str
    case_id: str
    campaign_id: str
    generation_index: int = 0
    backend: str = ""
    backend_version: str = ""
    command: str = ""
    workdir: str = ""
    status: str = "pending"
    failure_reason: str = ""
    start_time: str = ""
    end_time: str = ""
    runtime_seconds: float | None = None
    stdout_path: str = ""
    stderr_path: str = ""
    config_snapshot_path: str = ""
    environment_snapshot: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    notes: str = ""

    required_fields: ClassVar[tuple[str, ...]] = ("run_id", "case_id", "campaign_id", "status")


@dataclass
class ArtifactRecord(BaseRecord):
    artifact_id: str
    campaign_id: str
    case_id: str
    run_id: str
    generation_index: int = 0
    artifact_type: str = "other"
    path: str = ""
    relative_path: str = ""
    checksum: str = ""
    created_at: str = field(default_factory=now_iso)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    required_fields: ClassVar[tuple[str, ...]] = ("artifact_id", "campaign_id", "case_id", "run_id", "artifact_type", "path")


@dataclass
class MetricRecord(BaseRecord):
    metric_id: str
    campaign_id: str
    case_id: str
    run_id: str
    generation_index: int = 0
    metric_name: str = ""
    metric_value: Any = None
    metric_unit: str = ""
    metric_type: str = ""
    source_artifact_id: str = ""
    extraction_method: str = ""
    created_at: str = field(default_factory=now_iso)
    available: bool = False
    notes: str = ""

    required_fields: ClassVar[tuple[str, ...]] = ("metric_id", "campaign_id", "case_id", "run_id", "metric_name", "available")


@dataclass
class EvaluationRecord(BaseRecord):
    evaluation_id: str
    campaign_id: str
    case_id: str
    run_id: str
    generation_index: int = 0
    evaluator_name: str = "data_twin.simple_evaluator"
    evaluator_version: str = "0.1"
    physics_score: float | None = None
    geometry_score: float | None = None
    numerical_score: float | None = None
    balanced_score: float | None = None
    constraint_status: str = "unknown"
    failure_labels: list[str] = field(default_factory=list)
    summary: str = ""
    created_at: str = field(default_factory=now_iso)
    notes: str = ""

    required_fields: ClassVar[tuple[str, ...]] = ("evaluation_id", "campaign_id", "case_id", "run_id", "evaluator_name")


@dataclass
class DecisionRecord(BaseRecord):
    decision_id: str
    campaign_id: str
    case_id: str
    run_id: str = ""
    generation_index: int = 0
    decision: str = "manual_review"
    reason: str = ""
    next_action: str = ""
    parent_for_future_cases: bool = False
    created_at: str = field(default_factory=now_iso)
    decided_by: str = ""
    notes: str = ""

    required_fields: ClassVar[tuple[str, ...]] = ("decision_id", "campaign_id", "case_id", "decision")


@dataclass
class EventRecord(BaseRecord):
    event_id: str
    timestamp: str
    campaign_id: str
    object_type: str
    object_id: str
    event_type: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    required_fields: ClassVar[tuple[str, ...]] = ("event_id", "timestamp", "campaign_id", "object_type", "object_id", "event_type")


MODEL_BY_FILE = {
    "cases.jsonl": CaseRecord,
    "runs.jsonl": RunRecord,
    "artifacts.jsonl": ArtifactRecord,
    "metrics.jsonl": MetricRecord,
    "evaluations.jsonl": EvaluationRecord,
    "decisions.jsonl": DecisionRecord,
    "events.jsonl": EventRecord,
}
