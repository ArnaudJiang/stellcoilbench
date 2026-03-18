"""Tests for update_db metric definition RST generation."""

from __future__ import annotations

from pathlib import Path

from stellcoilbench.update_db._constraints import REACTOR_SCALE_CONSTRAINTS
from stellcoilbench.update_db._writers_metric_defs import (
    _build_composite_score_lines,
    _build_constraint_table_lines,
    _build_hard_constraints_table,
    _build_soft_constraints_table,
    _build_winding_pack_model_lines,
    _load_rst_template,
    _write_metric_definitions_rst,
)


class TestLoadRstTemplate:
    """Tests for _load_rst_template."""

    def test_composite_score_returns_non_empty(self) -> None:
        content = _load_rst_template("composite_score.rst")
        assert len(content) > 100
        assert "Composite Score" in content

    def test_winding_pack_model_returns_non_empty(self) -> None:
        content = _load_rst_template("winding_pack_model.rst")
        assert len(content) > 100
        assert "Winding-Pack" in content


class TestBuildConstraintTableLines:
    """Tests for _build_constraint_table_lines helper."""

    def test_hard_only_returns_hard_constraints(self) -> None:
        """Hard constraints have label, bound_str, desc keys."""
        rows = _build_constraint_table_lines(REACTOR_SCALE_CONSTRAINTS, hard_only=True)
        assert len(rows) >= 3  # coils_linked, linking_number, finite_build
        for row in rows:
            assert "label" in row
            assert "bound_str" in row
            assert "desc" in row

    def test_soft_only_returns_soft_constraints(self) -> None:
        """Soft constraints have label, bound_str, direction, units keys."""
        rows = _build_constraint_table_lines(REACTOR_SCALE_CONSTRAINTS, hard_only=False)
        assert len(rows) >= 8  # avg_BdotN, separations, length, curvature, etc.
        for row in rows:
            assert "label" in row
            assert "bound_str" in row
            assert "direction" in row
            assert "units" in row

    def test_hard_and_soft_partition_constraints(self) -> None:
        """Hard and soft rows together cover all constraints."""
        hard = _build_constraint_table_lines(REACTOR_SCALE_CONSTRAINTS, hard_only=True)
        soft = _build_constraint_table_lines(REACTOR_SCALE_CONSTRAINTS, hard_only=False)
        assert len(hard) + len(soft) == len(REACTOR_SCALE_CONSTRAINTS)


class TestBuildHardConstraintsTable:
    """Tests for _build_hard_constraints_table."""

    def test_returns_rst_lines_with_header(self) -> None:
        """Table includes header and reactor-scale context."""
        lines = _build_hard_constraints_table()
        joined = "\n".join(lines)
        assert "Reactor-Scale Constraints" in joined
        assert "Scaled to ARIES-CS" in joined
        assert "Hard constraints" in joined
        assert "list-table" in joined
        assert "Constraint" in joined
        assert "Bound" in joined
        assert "Description" in joined


class TestBuildSoftConstraintsTable:
    """Tests for _build_soft_constraints_table."""

    def test_returns_rst_lines_with_header(self) -> None:
        """Table includes header and soft constraint context."""
        lines = _build_soft_constraints_table()
        joined = "\n".join(lines)
        assert "Soft constraints" in joined
        assert "list-table" in joined
        assert "Metric" in joined
        assert "Direction" in joined
        assert "Units" in joined


class TestBuildCompositeScoreLines:
    """Tests for _build_composite_score_lines (template-based)."""

    def test_soft_constraint_table_generated_from_constraints(self) -> None:
        """Soft constraint table is generated from REACTOR_SCALE_CONSTRAINTS."""
        lines = _build_composite_score_lines()
        joined = "\n".join(lines)
        assert "{{SOFT_CONSTRAINT_TABLE}}" not in joined
        soft_count = sum(1 for c in REACTOR_SCALE_CONSTRAINTS if not c.get("hard"))
        assert joined.count("   * - ") >= soft_count + 1  # header + rows
        assert "avg_BdotN_over_B" in joined
        assert "Min coil-surface distance" in joined
        assert "total_superconductor_length_km" in joined

    def test_margin_formulas_use_margin_value_rst(self) -> None:
        """Margin formulas use margin_value_rst for special symbols."""
        lines = _build_composite_score_lines()
        joined = "\n".join(lines)
        assert r"\kappa_{\max}" in joined
        assert r"\sqrt{\text{MSC}}" in joined
        assert r"L_{\text{SC}}" in joined


class TestBuildWindingPackModelLines:
    """Tests for _build_winding_pack_model_lines (template-based)."""

    def test_winding_pack_soft_constraint_present(self) -> None:
        """Winding pack template describes N_turns as soft constraint (bound 300)."""
        lines = _build_winding_pack_model_lines()
        joined = "\n".join(lines)
        assert "Winding-Pack Model" in joined
        assert "Soft constraint" in joined
        assert "300" in joined
        assert "rewarded" in joined
        assert "penalized" in joined


class TestWriteMetricDefinitionsRst:
    """Tests for _write_metric_definitions_rst full output."""

    def test_writes_byte_identical_sections(self, tmp_path: Path) -> None:
        """Composite score and winding pack sections match template-based output."""
        _write_metric_definitions_rst(
            [
                "final_squared_flux",
                "avg_BdotN_over_B",
                "reactor_scale_min_cs_separation",
            ],
            tmp_path,
        )
        content = (tmp_path / "metric_definitions.rst").read_text()
        comp_lines = _build_composite_score_lines()
        wind_lines = _build_winding_pack_model_lines()
        comp_section = "\n".join(comp_lines)
        wind_section = "\n".join(wind_lines)
        assert comp_section in content
        assert wind_section in content
