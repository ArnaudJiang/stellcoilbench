import csv

from data_twin.core.state import init_campaign
from data_twin.core.validation import validate_campaign
from data_twin.ingest.existing_csv import ingest_csv
from data_twin.storage.jsonl_store import JsonlStore


def _config(tmp_path) -> str:
    config = tmp_path / "campaign.yaml"
    config.write_text(
        """
campaign_id: camp
name: Test campaign
target_type: stellarator_coil_optimization
storage:
  root: {root}
""".format(root=tmp_path / "experiments"),
        encoding="utf-8",
    )
    return str(config)


def test_ingest_existing_csv_creates_explicit_missing_metrics(tmp_path) -> None:
    root = init_campaign(_config(tmp_path))
    input_csv = tmp_path / "round.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "run_id", "status", "ncoils", "total_length"])
        writer.writeheader()
        writer.writerow({"case_id": "case_1", "run_id": "run_1", "status": "DONE", "ncoils": "4", "total_length": "12.5"})

    counts = ingest_csv(root, "camp", input_csv)
    errors = validate_campaign(root)
    metrics = JsonlStore(root).read("metrics.jsonl")

    assert counts["cases"] == 1
    assert counts["runs"] == 1
    assert not errors
    assert any(row["metric_name"] == "total_coil_length" and row["available"] for row in metrics)
    assert any(row["metric_name"] == "final_squared_flux" and not row["available"] for row in metrics)
