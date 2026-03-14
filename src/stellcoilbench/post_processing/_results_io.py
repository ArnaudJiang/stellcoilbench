"""I/O utilities for post-processing results (JSON serialisation, etc.).

Serialises BdotN, quasisymmetry, SIMPLE, and structural metrics to
post_processing_results.json and integrates timing data from the
global timing store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..utils import get_timing_results


def _save_post_processing_results(
    results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Serialise post-processing results to JSON and annotate *results*.

    Writes ``post_processing_results.json`` into *output_dir* containing
    the key scalar metrics (BdotN, quasisymmetry, SIMPLE, timing).  The
    timing results are also added to *results* under the ``"timing"`` key.

    Parameters
    ----------
    results : Dict[str, Any]
        Accumulated results dictionary from the post-processing pipeline.
    output_dir : Path
        Directory where the JSON file will be written.

    Returns
    -------
    Dict[str, Any]
        The same *results* dict, with ``"timing"`` added / updated.
    """
    results_json: Dict[str, Any] = {
        "BdotN": results.get("BdotN"),
        "BdotN_over_B": results.get("BdotN_over_B"),
        "quasisymmetry_average": results.get("quasisymmetry_average"),
    }

    if "simple_results" in results:
        simple_results = results["simple_results"]
        results_json["simple"] = {
            "loss_fraction": simple_results.get("loss_fraction"),
            "confined_fraction": simple_results.get("confined_fraction"),
            "confined_passing": simple_results.get("confined_passing"),
            "confined_trapped": simple_results.get("confined_trapped"),
            "final_time": simple_results.get("final_time"),
            "loss_fraction_plot": simple_results.get("loss_fraction_plot"),
        }

    if "structural_metrics" in results:
        sm = results["structural_metrics"]
        results_json["structural"] = {
            "skipped": sm.get("skipped", False),
            "reason": sm.get("reason"),
            "max_von_mises_stress_Pa": sm.get("max_von_mises_stress_Pa"),
            "mean_von_mises_stress_Pa": sm.get("mean_von_mises_stress_Pa"),
            "max_displacement_m": sm.get("max_displacement_m"),
            "youngs_modulus_Pa": sm.get("youngs_modulus_Pa"),
            "poisson_ratio": sm.get("poisson_ratio"),
            "bc_type": sm.get("bc_type"),
            "backend": sm.get("backend"),
        }

    results_json["timing"] = get_timing_results()

    with open(output_dir / "post_processing_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    results["timing"] = get_timing_results()
    return results
