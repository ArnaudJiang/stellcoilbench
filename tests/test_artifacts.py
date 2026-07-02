from pathlib import Path

from data_twin.storage.artifact_store import attach_artifact


def test_attach_artifact_copies_and_indexes_checksum(tmp_path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    source = tmp_path / "summary.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")

    artifact = attach_artifact(
        campaign_root,
        campaign_id="camp",
        case_id="case_1",
        run_id="run_1",
        artifact_path=source,
        artifact_type="final_summary_json",
        copy=True,
    )

    assert artifact.checksum
    assert artifact.relative_path == "artifacts/case_1/run_1/summary.json"
    assert Path(artifact.path).exists()
