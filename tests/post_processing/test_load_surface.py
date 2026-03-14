"""Tests for load_surface_with_range, load_coils_and_surface, and _apply_post_processing_config."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.conftest import REPO_ROOT

from stellcoilbench.config_scheme import PostProcessingConfig
from stellcoilbench.post_processing import (
    load_coils_and_surface,
    load_surface_with_range,
)


def test_apply_post_processing_config_with_config() -> None:
    """Test _apply_post_processing_config overrides defaults when config is set."""
    from stellcoilbench.post_processing._run_helpers import (
        _apply_post_processing_config,
    )

    cfg = PostProcessingConfig(run_vmec=True, plot_poincare=False, ns=100)
    out = _apply_post_processing_config(
        cfg,
        run_vmec=False,
        plot_poincare=True,
        ns=50,
        helicity_m=1,
        helicity_n=0,
        plot_boozer=False,
        nfieldlines=20,
        run_simple=False,
        simple_executable_path=None,
        run_vmec_original=False,
        plot_finite_build=False,
        finite_build_width=None,
        finite_build_height=None,
        run_structural=False,
        structural_E=None,
        structural_nu=None,
        compute_shape_gradient=False,
    )
    assert out["run_vmec"] is True
    assert out["plot_poincare"] is False
    assert out["ns"] == 100


class TestLoadSurfaceWithRange:
    """Tests for load_surface_with_range function."""

    @pytest.mark.parametrize(
        "ext,loader_attr,range_val,expect_error",
        [
            ("input.test", "from_vmec_input", "half period", False),
            ("test.focus", "from_focus", "full torus", False),
            ("wout.test.nc", "from_wout", "half period", False),
            ("test.unknown", None, "half period", True),
        ],
        ids=["vmec_input", "focus", "wout", "unknown_type"],
    )
    def test_load_surface_with_range_parametrized(
        self,
        tmp_path: Path,
        ext: str,
        loader_attr: str | None,
        range_val: str,
        expect_error: bool,
    ) -> None:
        """Parametrized test for load_surface_with_range across file types."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        surface_file = tmp_path / ext
        surface_file.write_text("dummy content" if ext != "wout.test.nc" else "dummy")
        if expect_error:
            with pytest.raises(ValueError, match="Unknown surface type"):
                load_surface_with_range(surface_file, surface_range=range_val)
        else:
            assert loader_attr is not None
            with patch(
                "stellcoilbench.path_utils._surface_io.SurfaceRZFourier"
            ) as mock_srf:
                mock_surface = Mock()
                mock_surface.minor_radius.return_value = 0.2
                mock_surface.major_radius.return_value = 1.0
                getattr(mock_srf, loader_attr).return_value = mock_surface
                result = load_surface_with_range(surface_file, surface_range=range_val)
                assert result == mock_surface
                getattr(mock_srf, loader_attr).assert_called_once()

    def test_reference_radii_preserved_across_resolution(self) -> None:
        """Reference radii from raw load are stable regardless of surface_resolution."""
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from stellcoilbench.config_scheme import ARIES_CS_MINOR_RADIUS
        from stellcoilbench.path_utils import (
            get_reference_radii,
            load_surface_with_range,
        )

        surf_file = REPO_ROOT / "plasma_surfaces" / "input.LandremanPaul2021_QA"
        if not surf_file.exists():
            pytest.skip("Landreman-Paul QA surface not found")

        for res in [32, 64, 128]:
            s = load_surface_with_range(
                surf_file,
                surface_range="half period",
                nphi=res,
                ntheta=res,
            )
            major, minor = get_reference_radii(s)
            a0 = ARIES_CS_MINOR_RADIUS / minor
            assert minor == pytest.approx(0.1683, rel=0.01), f"res={res}"
            assert major == pytest.approx(1.01, rel=0.01), f"res={res}"
            assert a0 == pytest.approx(10.1, rel=0.01), f"res={res}"


class TestLoadCoilsAndSurface:
    """Tests for load_coils_and_surface function."""

    def test_load_coils_and_surface_finds_case_yaml(self, tmp_path: Path) -> None:
        pytest.importorskip("simsopt.geo", reason="simsopt not available")
        from simsopt.field import BiotSavart
        from simsopt.geo import SurfaceRZFourier

        from tests.conftest import minimal_coils_json

        coils_json = minimal_coils_json(tmp_path / "biot_savart_optimized.json")
        case_yaml = tmp_path / "case.yaml"
        from tests.conftest import write_case_yaml

        write_case_yaml(
            case_yaml,
            surface="input.test",
            surface_params={"surface": "input.test", "range": "half period"},
        )
        plasma_dir = tmp_path / "plasma_surfaces"
        plasma_dir.mkdir()
        surface_file = plasma_dir / "input.test"
        surface_file.write_text("&INDATA\nNFP=2\n/")

        mock_bfield = Mock(spec=BiotSavart)
        mock_bfield.coils = []
        with patch(
            "stellcoilbench.post_processing._coil_io.load",
            return_value=mock_bfield,
        ):
            with patch(
                "stellcoilbench.path_utils._surface_io.SurfaceRZFourier"
            ) as mock_srf:
                mock_surface = Mock(spec=SurfaceRZFourier)
                mock_surface.minor_radius.return_value = 0.2
                mock_surface.major_radius.return_value = 1.0
                mock_srf.from_vmec_input.return_value = mock_surface
                bfield, surface = load_coils_and_surface(
                    coils_json,
                    case_yaml_path=case_yaml,
                    plasma_surfaces_dir=plasma_dir,
                )
                assert bfield is not None
                assert surface is not None

    def test_load_coils_and_surface_root_reached(self, tmp_path: Path) -> None:
        from simsopt.field import BiotSavart

        deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g"
        deep_dir.mkdir(parents=True, exist_ok=True)
        from tests.conftest import minimal_coils_json

        coils_json = minimal_coils_json(deep_dir / "biot_savart_optimized.json")

        mock_bs = Mock(spec=BiotSavart)
        mock_bs.coils = []
        with patch(
            "stellcoilbench.post_processing._coil_io.load",
            return_value=mock_bs,
        ):
            with pytest.raises(FileNotFoundError, match="Could not find case"):
                load_coils_and_surface(coils_json)
