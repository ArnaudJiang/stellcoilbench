API Reference
==============

This section provides detailed documentation for the StellCoilBench Python API.
The API is organized into several modules, each handling a specific aspect of the
benchmarking framework.

Module Overview
---------------

- **``stellcoilbench.cli``**: Command-line interface implementation
- **``stellcoilbench.coil_optimization``**: Core coil optimization logic
- **``stellcoilbench.config_scheme``**: Configuration data structures
- **``stellcoilbench.evaluate``**: Data structure for aggregated results (``SubmissionResults``)
- **``stellcoilbench.submission_packaging``**: Builds submission dirs, zips, and ``results.json`` from optimization output
- **``stellcoilbench.update_db``**: Scans submissions, aggregates metrics, generates leaderboards (RST/MD/JSON)
- **``stellcoilbench.validate_config``**: Configuration validation
- **``stellcoilbench.post_processing``**: VMEC, Poincaré, QFM, Boozer plots, SIMPLE, finite-build VTK
- **``stellcoilbench.sensitivity``**: Coil sensitivity analysis via stochastic perturbation
- **``stellcoilbench.path_utils``**: Path resolution, surface/case lookup, YAML load/dump
- **``stellcoilbench.finite_build``**: Finite-build coil geometry (rectangular cross-section swept along centerline)
- **``stellcoilbench.structural_analysis``**: FEM structural analysis (DOLFINx / scikit-fem)

Configuration Module
--------------------

.. automodule:: stellcoilbench.config_scheme
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

The ``config_scheme`` module defines data structures for case configurations and
submission metadata.

**CaseConfig**
   Represents a complete case configuration loaded from a YAML file. Contains:
   
   - ``description``: Case description
   - ``surface_params``: Plasma surface configuration
   - ``coils_params``: Coil geometry parameters
   - ``optimizer_params``: Optimization algorithm settings
   - ``coil_objective_terms``: Objective function terms
   
   Methods:
   
   - ``from_dict(data: Dict[str, Any]) -> CaseConfig``: Create from dictionary

**SubmissionMetadata**
   Metadata for submissions, including method information, contact details,
   hardware information, and timestamps.

Case Loader Module
------------------

.. automodule:: stellcoilbench.case_loader
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Use :func:`stellcoilbench.case_loader.load_case` to load validated case
configurations from a YAML file or directory containing ``case.yaml``.

**load_case(path: Path | str, *, validate: bool = True) -> CaseConfig**
   Load and validate a case configuration.
   
   Parameters:
   
   - ``path``: Path to case.yaml file or directory containing case.yaml
   - ``validate``: If True (default), run validation before constructing CaseConfig
   
   Returns:
   
   - ``CaseConfig``: Validated configuration
   
   Raises:
   
   - ``FileNotFoundError``: If case.yaml not found (includes searched paths and suggested next steps)
   - ``ValueError``: If configuration validation fails

Evaluation Module
-----------------

.. automodule:: stellcoilbench.evaluate
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

The ``evaluate`` module provides ``SubmissionResults`` for aggregated evaluation
metrics. Use :py:func:`stellcoilbench.case_loader.load_case` for case loading.

Coil Optimization Module
-------------------------

.. automodule:: stellcoilbench.coil_optimization
   :members:
   :undoc-members:
   :show-inheritance:

The ``coil_optimization`` module contains the core optimization logic. Key
functions: ``optimize_coils`` (main entry point), ``initialize_coils_loop``,
``optimize_coils_loop``, ``optimize_coils_with_fourier_continuation``.
``LinearPenalty`` provides threshold-based penalty terms. See the automodule
output above for full signatures and docstrings.

**save_coils_config(coils: List, config_path: Path) -> None**
   Save coil configuration to JSON file.
   
   Parameters:
   
   - ``coils``: List of coil objects
   - ``config_path``: Path to save coils.json

**coils_to_vtk(coils: List, filename: Path) -> None**
   Export coils to VTK format for visualization.
   
   Parameters:
   
   - ``coils``: List of coil objects
   - ``filename``: Output VTK file path

**plot_bn_error_3d(surface, bs, coils, out_dir: Path, filename: str = "bn_error_3d_plot.pdf", title: str = "B_N/|B| Error on Plasma Surface with Optimized Coils", plot_upsample: int = 3) -> None**
   Create 3D visualization of B_N error on plasma surface.
   
   Generates a high-resolution PDF plot showing:
   
   - Plasma surface colored by :math:`B_N/|B|` error magnitude
   - Coils colored by current magnitude
   - Colorbars for both
   
   Parameters:
   
   - ``surface``: Plasma surface object
   - ``bs``: BiotSavart field calculator
   - ``coils``: List of coil objects
   - ``out_dir``: Output directory
   - ``filename``: Output PDF filename
   - ``title``: Plot title
   - ``plot_upsample``: Surface upsampling factor for higher resolution

Update Database Module
----------------------

.. automodule:: stellcoilbench.update_db
   :members:
   :undoc-members:
   :show-inheritance:

The ``update_db`` module handles leaderboard generation and management.

**update_database(repo_root: Path, submissions_root: Path | None = None, docs_dir: Path | None = None, cases_root: Path | None = None, plasma_surfaces_dir: Path | None = None, *, use_local_viz_links: bool = False) -> dict**
   Main function to update leaderboards from submissions.
   
   Scans submissions directory, loads results, computes rankings, and generates
   leaderboard files.
   
   Parameters:
   
   - ``repo_root``: Repository root directory
   - ``submissions_root``: Submissions directory (default: repo_root / "submissions")
   - ``docs_dir``: Documentation directory (default: repo_root / "docs")
   - ``cases_root``: Cases directory (default: repo_root / "cases")
   - ``plasma_surfaces_dir``: Plasma surfaces directory (default: repo_root / "plasma_surfaces")
   - ``use_local_viz_links``: If True, use relative paths for PDF links instead of CDN
   
   Returns:
   
   - ``dict``: Summary with ``submissions_count``, ``surfaces_updated``, ``errors``

**build_surface_leaderboards(leaderboard: Dict[str, Any], submissions_root: Path, plasma_surfaces_dir: Path) -> Dict[str, Dict[str, Any]]**
   Group leaderboard entries by plasma surface.
   
   Parameters:
   
   - ``leaderboard``: Overall leaderboard dictionary
   - ``submissions_root``: Submissions directory
   - ``plasma_surfaces_dir``: Plasma surfaces directory
   
   Returns:
   
   - ``Dict[str, Dict[str, Any]]``: Dictionary mapping surface names to leaderboard entries

**write_rst_leaderboard(leaderboard: Dict[str, Any], out_rst: Path, surface_leaderboards: Dict[str, Dict[str, Any]]) -> None**
   Write ReadTheDocs-formatted leaderboard.
   
   Generates a comprehensive RST file with embedded tables for all surfaces.
   
   Parameters:
   
   - ``leaderboard``: Overall leaderboard dictionary
   - ``out_rst``: Output RST file path
   - ``surface_leaderboards``: Per-surface leaderboards

**write_markdown_leaderboard(leaderboard: Dict[str, Any], out_md: Path) -> None**
   Write markdown-formatted leaderboard.
   
   Parameters:
   
   - ``leaderboard``: Leaderboard dictionary
   - ``out_md``: Output markdown file path

**write_surface_leaderboards(surface_leaderboards: Dict[str, Dict[str, Any]], docs_dir: Path, repo_root: Path) -> list[str]**
   Write per-surface markdown leaderboards.
   
   Parameters:
   
   - ``surface_leaderboards``: Per-surface leaderboards
   - ``docs_dir``: Documentation directory
   - ``repo_root``: Repository root
   
   Returns:
   
   - ``list[str]``: List of generated surface names

**load_submissions(submissions_root: Path) -> Iterable[Tuple[str, Path, Dict[str, Any]]]**
   Load all submissions from directory.
   
   Handles both regular directories and zip files. Extracts results.json from
   zips as needed.
   
   Parameters:
   
   - ``submissions_root``: Submissions directory
   
   Yields:
   
   - ``(method_key, path, data)``: Method key, submission path, and results data

**metric_shorthand(metric_name: str) -> str**
   Convert metric names to compact shorthand for display.
   
   Parameters:
   
   - ``metric_name``: Full metric name
   
   Returns:
   
   - Shorthand/acronym (e.g., "f_B" for "final_normalized_squared_flux")

**metric_definition(metric_name: str) -> str**
   Get detailed mathematical definition for a metric.
   
   Parameters:
   
   - ``metric_name``: Metric name
   
   Returns:
   
   - LaTeX-formatted mathematical definition

Post-Processing Module
----------------------

.. automodule:: stellcoilbench.post_processing
   :members:
   :undoc-members:
   :show-inheritance:

The ``post_processing`` module handles post-optimization analysis including VMEC
equilibrium calculations, Poincaré plots, quasisymmetry analysis, and Boozer surface plots.

**run_post_processing(coils_json_path: Path, output_dir: Path, case_yaml_path: Optional[Path] = None, plasma_surfaces_dir: Optional[Path] = None, run_vmec: bool = True, helicity_m: int = 1, helicity_n: int = 0, ns: int = 50, plot_boozer: bool = True, plot_poincare: bool = True, nfieldlines: int = 20, mpi: Optional[Any] = None) -> Dict[str, Any]**
   Run complete post-processing pipeline.
   
   This function:
   
   1. Loads coils and plasma surface
   2. Generates Poincaré plot (if requested)
   3. Computes QFM surface
   4. Optionally runs VMEC equilibrium
   5. Computes quasisymmetry metrics
   6. Generates VMEC-dependent plots (Boozer, iota, quasisymmetry)
   
   Parameters:
   
   - ``coils_json_path``: Path to coils JSON file
   - ``output_dir``: Directory where output files will be saved
   - ``case_yaml_path``: Path to case.yaml file (optional)
   - ``plasma_surfaces_dir``: Directory containing plasma surface files (optional)
   - ``run_vmec``: Whether to run VMEC equilibrium calculation (default: True)
   - ``helicity_m``: Poloidal mode number for quasisymmetry (default: 1)
   - ``helicity_n``: Toroidal mode number for quasisymmetry (default: 0)
   - ``ns``: Number of radial surfaces for quasisymmetry evaluation (default: 50)
   - ``plot_boozer``: Whether to generate Boozer surface plot (default: True)
   - ``plot_poincare``: Whether to generate Poincaré plot (default: True)
   - ``nfieldlines``: Number of fieldlines to trace for Poincaré plot (default: 20)
   - ``mpi``: MPI partition for parallel execution (optional)
   
   Returns:
   
   - ``Dict[str, Any]``: Dictionary containing post-processing results:
     - ``qfm_surface``: QFM surface object
     - ``quasisymmetry_average``: Average quasisymmetry error
     - ``quasisymmetry_profile``: Radial quasisymmetry profile
     - ``vmec``: VMEC equilibrium object (if run_vmec=True)

Sensitivity Module
-----------------

.. automodule:: stellcoilbench.sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

Coil sensitivity analysis via stochastic perturbation (CurvePerturbed / GaussianSampler).
Quantifies robustness to geometric perturbations. Key entry point:
``run_sensitivity_analysis(coils_json_path, case_yaml_path, ...)``.

Path Utils Module
-----------------

.. automodule:: stellcoilbench.path_utils
   :members:
   :undoc-members:
   :show-inheritance:

Path resolution for plasma surfaces, case YAML, and coils. Key functions:
``resolve_case_and_surface``, ``resolve_all``, ``find_plasma_surfaces_dir``,
``load_yaml``, ``dump_yaml``, ``get_surface_filename``.

Finite Build Module
-------------------

.. automodule:: stellcoilbench.finite_build
   :members:
   :undoc-members:
   :show-inheritance:

Finite-build coil geometry: rectangular cross-section swept along centerline.
Key function: ``finite_build_coils_to_vtk``. Requires ``pip install .[structural]`` for Gmsh.

Structural Analysis Module
-------------------------

.. automodule:: stellcoilbench.structural_analysis
   :members:
   :undoc-members:
   :show-inheritance:

FEM structural analysis (Von Mises stress, Lorentz force). Backends: DOLFINx or scikit-fem.
Key function: ``run_structural_analysis``. Optional: ``pip install .[structural]``.

Evaluate and Submission Pipeline (Module Responsibilities)
---------------------------------------------------------

- **``evaluate``**: Defines ``SubmissionResults`` dataclass (metadata + metrics). Lightweight container; used when structuring evaluation output.
- **``submission_packaging``**: Assembles the physical submission: builds dirs, writes ``results.json``, copies case YAML, zips. Used by ``submit-case``.
- **``update_db``**: Consumes submissions (zips/dirs), aggregates metrics, generates leaderboards. Does not create submissions.

Validate Config Module
----------------------

.. automodule:: stellcoilbench.validate_config
   :members:
   :undoc-members:
   :show-inheritance:

The ``validate_config`` module provides configuration validation.

**validate_case_config(data: Dict[str, Any], file_path: Path | None = None, surfaces_dir: Path | None = None) -> List[str]**
   Validate a case configuration dictionary.
   
   Checks for:
   
   - Required fields
   - Valid surface names
   - Valid algorithm names
   - Valid objective term options
   - Type correctness
   
   Parameters:
   
   - ``data``: Configuration dictionary
   - ``file_path``: Optional file path for error messages
   - ``surfaces_dir``: Optional directory for surface file existence checks
   
   Returns:
   
   - ``List[str]``: List of error messages (empty if valid)

**validate_case_yaml_file(file_path: Path, surfaces_dir: Path | None = None) -> List[str]**
   Validate a case YAML file on disk. Loads with load_yaml, then delegates to validate_case_config.

CLI Module
----------

.. automodule:: stellcoilbench.cli
   :members:
   :undoc-members:
   :show-inheritance:

The ``cli`` module implements the command-line interface.

**Public vs internal**: Commands (``submit_case``, ``run_case``, etc.) and ``app`` are
public. Helpers such as ``_detect_github_username``, ``_zip_submission_directory``,
``NumpyJSONEncoder`` are re-exported for tests and advanced use but may change.

**app**
   Typer application instance. Commands: validate-config, list-cases, submit-case,
   run-case, run-ci-case, update-db, generate-submission, post-process, sensitivity.

**NumpyJSONEncoder**
   Custom JSON encoder that handles numpy types and arrays. Used for serializing
   results to JSON.

**validate_config_cmd** — CLI command: Validate a case YAML configuration.

**list_cases** — CLI command: List available benchmark cases from ``cases/*.yaml``.

**submit_case** — CLI command: Run a case and create a submission.

**run_case** — CLI command: Run a case without creating a submission.

**run_ci_case** — CLI command: Run a case in CI mode (optimization only).

**generate_submission** — CLI command: Create a submission from existing results.
   Packages coils.json and metadata.yaml into a results.json submission. Uses a
   placeholder ``chi2_Bn: 0.001`` for metrics (external results are not recomputed).
   For full metric computation, run ``submit-case`` or ``post-process`` first.
   Expected metadata.yaml keys: ``method_version``, ``contact``, ``hardware``.

**post_process** — CLI command: Run post-processing on coils (Poincaré, VMEC, etc.).

**update_db_cmd** — CLI command: Regenerate leaderboards.

**sensitivity_cmd** — CLI command: Run coil sensitivity analysis.

See :doc:`overview` for usage details.

Usage Examples
--------------

**Running Optimization Programmatically**
   
   .. code-block:: python
   
      from pathlib import Path
      from stellcoilbench.case_loader import load_case
      from stellcoilbench.coil_optimization import optimize_coils
      
      # Load case
      case_path = Path("cases/basic_LandremanPaulQA.yaml")
      case_cfg = load_case(case_path)
      
      # Run optimization (returns metrics directly)
      results = optimize_coils(
          case_path=case_path,
          coils_out_path=Path("output/coils.json"),
          case_cfg=case_cfg,
          output_dir=Path("output/")
      )
      
      # Results contain metrics from the optimization pipeline
      print(f"Results: {results}")

**Creating Custom Objective Terms**
   
   .. code-block:: python
   
      from stellcoilbench.coil_optimization import LinearPenalty
      from simsopt.objectives import Weight
      
      # Create a custom penalty term
      penalty = LinearPenalty(
          func=lambda x: compute_something(x),
          threshold=1.0,
          penalty_type="l2_threshold"
      )
      
      # Scale with weight
      weighted_penalty = Weight(1e-3) * penalty
      
      # Add to objective
      objective = flux_term + weighted_penalty

**Updating Leaderboards Programmatically**
   
   .. code-block:: python
   
      from pathlib import Path
      from stellcoilbench.update_db import update_database
      
      # Update leaderboards
      update_database(
          repo_root=Path("."),
          submissions_root=Path("submissions/"),
          docs_dir=Path("docs/")
      )

Next Steps
----------

- **Overview**: See :doc:`overview` for setup and quick start
- **Cases**: Learn about case files in :doc:`cases`
- **Usage**: See :doc:`overview` for commands and workflow
- **Leaderboard**: View results in :doc:`leaderboard`