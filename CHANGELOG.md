# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - Unreleased

### Added

- Benchmark suite for stellarator coil optimization algorithms
- Case YAML schema and validation
- Coil optimization via simsopt (L-BFGS-B, BFGS, SLSQP, augmented Lagrangian)
- Fourier continuation for progressive coil refinement
- Post-processing: VMEC equilibrium, Poincaré plots, QFM surface, quasisymmetry, Boozer plots, SIMPLE particle tracing
- Finite-build coil geometry and VTK export
- Structural analysis (DOLFINx / scikit-fem)
- Sensitivity analysis via stochastic perturbation
- Leaderboard generation from submissions
- CLI: validate-config, list-cases, submit-case, run-case, run-ci-case, generate-submission, post-process, update-db
- Autopilot: GA and LLM proposers for autonomous case exploration
- ReadTheDocs documentation
- CI workflows: lint, test, case validation, self-hosted benchmark runner
