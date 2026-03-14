# Line Counts by .py File

Generated from `find src tests tools knowledge -name "*.py" | xargs wc -l | sort -n`.

## Summary

- **Total**: 69,166 lines across 161 Python files
- **src/stellcoilbench/**: ~30,400 lines
- **tests/**: ~37,500 lines
- **tools/**: ~1,704 lines
- **knowledge/**: ~1,740 lines

## By Directory

### src/stellcoilbench/ (core package)

| File | Lines |
|------|-------|
| update_db/_constants.py | 4 |
| update_db/_writers.py | 17 |
| coil_optimization/_simsopt_imports.py | 28 |
| __init__.py | 37 |
| _mpl.py | 45 |
| path_utils/__init__.py | 48 |
| sensitivity/__init__.py | 51 |
| cli/update_db_cmd.py | 53 |
| post_processing/_qfm.py | 55 |
| _optional_imports.py | 59 |
| mpi_utils.py | 62 |
| sensitivity/_plotting.py | 75 |
| post_processing/_results_io.py | 76 |
| coil_optimization/_objective_wrappers.py | 77 |
| cli/sensitivity_cmd.py | 80 |
| cli/_shared.py | 81 |
| structural_analysis/__init__.py | 83 |
| cli/__init__.py | 84 |
| finite_build/_vtk.py | 87 |
| path_utils/_yaml.py | 87 |
| post_processing/_bdotn.py | 93 |
| evaluate.py | 94 |
| coil_optimization/__init__.py | 97 |
| cli/post_process.py | 99 |
| version_utils.py | 103 |
| post_processing/_finite_build_runner.py | 104 |
| coil_optimization/_ci_utils.py | 109 |
| case_loader.py | 112 |
| path_utils/_path_search.py | 112 |
| post_processing/_surface_io.py | 112 |
| coil_optimization/_virtual_casing.py | 126 |
| post_processing/_poincare.py | 126 |
| coil_optimization/_post_opt_processing.py | 129 |
| utils.py | 129 |
| update_db/_recompute.py | 142 |
| post_processing/_structural_runner.py | 147 |
| coil_optimization/_optimization_dispatch.py | 149 |
| sensitivity/_sensitivity_io.py | 150 |
| update_db/_backfill.py | 151 |
| update_db/_path_parsing.py | 162 |
| update_db/_plot_composite_score.py | 164 |
| update_db/_writers_reactor.py | 181 |
| update_db/_viz_links.py | 188 |
| finite_build/_parastell.py | 191 |
| coil_optimization/_structural_stress.py | 211 |
| update_db/_metrics_extraction.py | 315 |
| coil_optimization/optimization.py | 318 |
| path_utils/_case_resolution.py | 318 |
| update_db/_load_submissions.py | 330 |
| coil_optimization/_fourier_continuation.py | 333 |
| coil_optimization/_config_parsing.py | 341 |
| update_db/_constraints.py | 350 |
| cli_helpers.py | 368 |
| cli/submit_run.py | 383 |
| submission_packaging.py | 389 |
| structural_analysis/_skfem.py | 396 |
| finite_build/__init__.py | 410 |
| post_processing/__init__.py | 438 |
| coil_optimization/_optimization_setup.py | 434 |
| post_processing/_vmec.py | 441 |
| post_processing/_shape_gradient.py | 449 |
| update_db/_writers_surface.py | 463 |
| coil_optimization/_structural_mesh.py | 464 |
| update_db/submission_io.py | 494 |
| coil_optimization/_optimization_loop.py | 496 |
| structural_analysis/_pipeline.py | 508 |
| reactor_scale.py | 529 |
| sensitivity/_core.py | 536 |
| coil_optimization/_constraint_builders.py | 542 |
| coil_optimization/_structural_objective.py | 550 |
| structural_analysis/_dolfinx.py | 602 |
| coil_optimization/_external_eval.py | 622 |
| validate_config.py | 644 |
| coil_optimization/_results.py | 648 |
| coil_optimization/_plotting.py | 715 |
| structural_analysis/_common.py | 721 |
| coil_optimization/_scipy_optimizer.py | 761 |
| update_db/_writers_common.py | 796 |
| update_db/_writers_metric_defs.py | 801 |
| post_processing/_simple.py | 812 |
| update_db/_formatting.py | 872 |

### tests/

| File | Lines |
|------|-------|
| __init__.py | 3 |
| conftest.py | 95 |
| test_constraint_scaling.py | 101 |
| test_utils.py | 143 |
| test_advanced_landreman_paul_continuation.py | 156 |
| test_constants.py | 160 |
| test_update_db_io.py | 161 |
| test_case_filter.py | 183 |
| test_algorithm_options.py | 205 |
| test_reactor_scale.py | 211 |
| test_evaluate.py | 227 |
| test_update_db_viz_links.py | 227 |
| test_new_plasma_surfaces.py | 241 |
| test_evaluate_comprehensive.py | 274 |
| test_path_utils.py | 276 |
| test_leaderboard_integration.py | 302 |
| test_finite_build.py | 315 |
| test_config_scheme.py | 344 |
| test_shape_gradient.py | 377 |
| test_formatting.py | 419 |
| test_fourier_continuation.py | 427 |
| test_sensitivity.py | 445 |
| test_weight_scaling.py | 493 |
| test_coil_initialization_comprehensive.py | 502 |
| test_update_db_helpers.py | 582 |
| test_coil_objective_options.py | 617 |
| test_linear_penalty.py | 661 |
| test_update_db_coverage.py | 681 |
| test_structural_objective.py | 682 |
| test_mpi_functionality.py | 698 |
| test_coil_optimization.py | 736 |
| test_cli_integration.py | 781 |
| test_knowledge.py | 825 |
| test_coil_optimization_comprehensive.py | 872 |
| test_scipy_algorithms.py | 909 |
| test_post_processing_coverage.py | 940 |
| test_fem_benchmarks.py | 1,001 |
| test_coil_optimization_coverage.py | 1,060 |
| test_validate_config.py | 1,137 |
| test_post_processing_comprehensive.py | 1,581 |
| test_ci_autopilot.py | 1,652 |
| test_cli.py | 1,952 |
| test_structural_analysis.py | 2,275 |
| test_coil_optimization_edge_cases.py | 2,548 |
| test_update_db.py | 2,582 |
| test_post_processing.py | 3,045 |
| test_update_db_comprehensive.py | 3,469 |

### tools/

| File | Lines |
|------|-------|
| generate_metric_definitions.py | 37 |
| build_context.py | 583 |
| propose_batch.py | 1,084 |

### knowledge/

| File | Lines |
|------|-------|
| ingest/__init__.py | 1 |
| llm/__init__.py | 1 |
| services/__init__.py | 1 |
| ingest/make_postmortem.py | 95 |
| ingest/extract_pdf.py | 101 |
| ingest/chunk.py | 116 |
| scripts/kb_updater.py | 116 |
| ingest/make_run_card.py | 145 |
| scripts/ingest_papers.py | 155 |
| services/kb_client.py | 234 |
| llm/llm_client.py | 260 |
| services/kb_server.py | 471 |
| services/llm_endpoints.py | 494 |
| scripts/fetch_papers.py | 685 |
