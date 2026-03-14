"""Tests for update-db CLI command."""

from stellcoilbench.cli import update_db_cmd


def test_update_db_cmd_invokes_update_database(tmp_path, monkeypatch):
    calls = {}

    def fake_update_database(
        repo_root, submissions_root=None, docs_dir=None, *, use_local_viz_links=False
    ):
        calls["repo_root"] = repo_root
        calls["submissions_root"] = submissions_root
        calls["docs_dir"] = docs_dir
        calls["use_local_viz_links"] = use_local_viz_links
        return {"surfaces_updated": 0, "submissions_count": 0, "errors": []}

    monkeypatch.setattr(
        "stellcoilbench.update_db.update_database", fake_update_database
    )
    update_db_cmd(submissions_dir=tmp_path / "subs", docs_dir=tmp_path / "docs")
    assert calls["docs_dir"] == tmp_path / "docs"

    calls.clear()
    update_db_cmd(
        submissions_dir=tmp_path / "subs",
        docs_dir=tmp_path / "docs",
        local_viz_links=True,
    )
    assert calls["use_local_viz_links"] is True
