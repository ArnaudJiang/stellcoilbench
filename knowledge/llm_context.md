# StellCoilBench LLM Context

Domain context for the LLM case proposer (single file to limit I/O). For full paper text see `llm_context_full.md`.

## Optimization Guide

### Key parameters

- **`ncoils`** — Unique base coils per half field period; replicated by stellarator symmetry (nfp×2 if stellsym). Typical 3–7 (stell), 6 (tokamak). More coils → better B-field, harder engineering.
- **`order`** — Fourier harmonics on the coil curve; typical 4–16. Higher → more freedom and lower flux error, risk of overfitting. Use `fourier_continuation` to ramp order.
- **`fourier_continuation`** — e.g. `{enabled: true, orders: [4, 8, 16]}`: optimize low order first, then extend coefficients—avoids bad high-order initial minima.
- **Algorithms** — `L-BFGS-B`: fast default. `augmented_lagrangian`: better for tight constraints (sub-solver L-BFGS-B), slower. `SLSQP` / `BFGS`: rare.

### Objective terms (weights + thresholds)

| Term | Weight key | Threshold key | Role |
|------|------------|---------------|------|
| Squared flux (B·n) | `flux_weight` | `flux_threshold` | Primary field accuracy |
| Coil length | `length_weight` | `length_threshold` | Shorter coils |
| Coil–coil distance | `cc_weight` | `cc_threshold` | Minimum separation |
| Coil–surface distance | `cs_weight` | `cs_threshold` | Standoff from plasma |
| Curvature | `curvature_weight` | `curvature_threshold` | Smoothness |
| Mean squared curvature | `msc_weight` | `msc_threshold` | Global smoothness |
| Torsion | `torsion_weight` | `torsion_threshold` | Twist limit |
| Force | `force_weight` | `force_threshold` | EM force per length |
| Torque | `torque_weight` | `torque_threshold` | EM torque |
| Arclength variation | `arclength_weight` | `arclength_variation_threshold` | Uniform spacing along coil |

### Common failure modes

| Mode | Mitigation |
|------|------------|
| `min_sep_violation` | Relax `cc_threshold` / `cs_threshold` or increase weights |
| `line_search_fail` | Smaller trust region, different algorithm, or Fourier continuation |
| `timeout` | Lower `max_iterations`, coarser resolution, fewer coils / lower order |
| `nan_in_objective` | Regularization, check initial coils, new `random_seed` |

### Metrics (results)

- **Score** — `final_squared_flux` / `score_primary`: lower is better (∫(B·n)²).
- **Engineering** — `final_total_length`, `final_min_cc_separation`, `final_min_cs_separation`, `final_mean_squared_curvature`, `final_max_max_coil_force`, `final_linking_number` (0 = unlinked).
- **Leaderboard** — Reactor-scale scaling vs ARIES-CS; `composite_score` = geometric mean of constraint margins (higher better).

## Threshold scaling

Thresholds are specified at **ARIES-CS reactor scale** (minor radius **a = 1.7 m**); the optimizer rescales to the device surface.

**Dimensionless scale:** `a0 = 1.7 / minor_radius` (from `s.minor_radius()`).

| Threshold | Unit | Reactor → device |
|-----------|------|------------------|
| `length_threshold` | m | ÷ a0 |
| `cc_threshold`, `cs_threshold` | m | ÷ a0 |
| `curvature_threshold` | 1/m | × a0 |
| `msc_threshold` | 1/m² | × a0 |
| `torsion_threshold` | 1/m | × a0 |
| `force_threshold` | N/m | ÷ a0, then × I₀² |
| `torque_threshold` | N | × I₀² only |
| `arclength_variation_threshold` | m² | × a0² |

`I₀² = (total_current / total_current_reactor_scale)²` — reactor reference from `initialize_coils_loop` with same surface and coil count; applied to force and torque thresholds.

**Defaults (reactor scale, when omitted):** length 200 m, cc 0.8 m, cs 1.3 m, curvature 1 1/m, msc 1 1/m², force 200 N/m, torque 200 N.

**Example (QA, a = 0.1683 m):** a0 ≈ 10.1 → length 242.4 m → 24.0 m device; cc 1.5 m → 0.15 m; curvature 1.5 → ~15.2 1/m.

## Literature (one-line takeaways)

Curated notes for proposal strategy; full abstracts live in `knowledge/summaries/`.

| Ref | Topic | Key takeaway for proposer |
|-----|--------|---------------------------|
| [1] | Topology / sparse regression coils | Voxel/sparse methods discover topology; can seed filament runs—not the default StellCoilBench filament workflow. |
| [2] | Global optimization (QUASR-style) | Multi-minima landscapes: global search + polish beats single-start; physical-space bounds help. |
| [3] | Constrained optimization (DESC) | Augmented Lagrangian suits hard plasma/coils constraints; curvature limits can trade with MHD/stability. |
| [4] | QUASR database | Large QA/QH+coil set—good benchmark surfaces and engineering ranges (d_cc, d_cs, κ). |
| [5] | Discrete / wireframe coils | Segment/wireframe parametrization for ports and sparse currents—alternative to Fourier coils. |
| [6] | Planar Eos sparse regression | MIQP/LASSO for planar arrays when minimizing B_n at fixed sparsity or current. |
| [7] | Augmented Lagrangian (SIMSOPT) | Prefer aug-lag for many simultaneous engineering constraints; avoids manual weight sweeps. |
| [8] | Surface current / coil cutting | Stage-2: Φ contours → coils; modular vs helical depends on G, I—external TF subtract G_ext. |
| [9] | QUADCOIL / winding proxy | Differentiable winding-surface proxies can bias equilibria before filaments. |
| [10] | Diffusion / generative boundaries | ML-generated boundaries need VMEC/DESC refinement; treat as warm starts. |
| [11] | Deflation | Multiple minima: deflation finds distinct equilibria/coils without many random restarts. |
| [12] | ITG critical gradient | Microstability objectives trade with QS; different minima than flux-only. |
| [13] | PM tokamak→stell | Permanent magnets viable at small scale/low-ι QA; not default superconducting filament cases. |
| [14] | QUADCOIL QCQP | Global quadratic objectives/constraints on winding surface (force, curvature)—fast stage-1. |
| [15] | Vacuum energy | Penalizing magnetic energy correlates with Lorentz loads; alternative to length-only regularization. |
| [16] | Lorentz force (filament) | Pointwise force constraints with regularized self-B; force–d_cs correlations—tighten d_cs inboard. |
| [17] | Dipole arrays reactor | Joint TF+dipole position/current optimization; torque often easier to zero than pointwise force. |
| [18] | Finite-β CSSC | Current-only tweaks can fix ripple before full shape optimization. |
| [19] | Ferromagnetic blanket | Steel/dipole perturbation to B—reoptimize coils if blanket model is on. |
| [20] | Single-stage fixed boundary | Joint plasma+coil with quadratic flux—good when FB equilibrium is fixed. |
| [21] | Simplified coils single-stage | Few coils per period: parametrization (circular, planar, helical) strongly affects success. |
| [22] | Canis HTS planar array | Experimental validation of planar shaping at ~1% field control. |
| [23] | Axisymmetric CWS | Coils on winding surface—fewer DOFs, manufacturing-constrained paths. |
| [24] | Divertor sharp corners | Weighted B_n or manifold objectives for X-point targets—more complex than smooth boundaries. |
| [25] | Microstability in loop | GS2/linear proxies in optimization are expensive; weight vs QS carefully. |
| [26] | ReBCO strain | HTS: add strain/torsion penalties; ε_crit ~0.2% typical for tape. |
| [27] | Manufacturing tolerance | GP(σ,L) on coil deviation—σ dominates field error; CNC vs AM matters at tabletop scale. |
| [28] | Helios planar reactor | Reactor-scale planar-coil reference (d_cs ~1.2 m blanket, many shaping coils). |
| [29] | PM macromagnetic GPMO | Coupled μ and demagnetization change integrated f_B—report integrated metrics. |
| [30] | ConStellaration dataset | QI boundaries + benchmarks for ML and method comparison. |
| [31] | Low-β accessibility | Micro-turbulence can limit “safe” β—case design may need β scans. |

## References

[1] A. A. Kaptanoglu, G. P. Langlois, M. Landreman. Topology optimization for inverse magnetostatics as sparse regression. arXiv:2306.12555 (2023).
[2] A. Giuliani. Direct stellarator coil design using global optimization. arXiv:2310.19097 (2023).
[3] R. Conlin et al. Stellarator Optimization with Constraints. arXiv:2403.11033 (2024).
[4] A. Giuliani et al. Comprehensive exploration of quasisymmetric stellarators and their coil sets. arXiv:2409.04826 (2024).
[5] K. C. Hammond. Framework for discrete optimization of stellarator coils. arXiv:2412.00267 (2024).
[6] R. Wu et al. Planar Coil Optimization for the Eos Stellarator using Sparse Regression. arXiv:2502.07702 (2025).
[7] P. F. Gil et al. Augmented Lagrangian methods produce cutting-edge magnetic coils for stellarator fusion reactors. arXiv:2507.12681 (2025).
[8] D. Panici et al. Surface Current Optimization and Coil-Cutting Algorithms for Stage-Two Stellarator Optimization. arXiv:2508.09321 (2025).
[9] L. Fu et al. A flexible and differentiable coil proxy for stellarator equilibrium optimization. arXiv:2510.16243 (2025).
[10] M. Padidar et al. Diffusion for Fusion: Designing Stellarators with Generative AI. arXiv:2511.20445 (2025).
[11] D. Panici et al. Deflation Techniques for Stellarator Equilibrium and Optimization. arXiv:2602.09957 (2026).
[12] G. T. Roberg-Clark et al. Critical gradient turbulence optimization toward a compact stellarator reactor concept. arXiv:2301.06773 (2023).
[13] M. Madeira, R. Jorge. Tokamak to Stellarator Conversion using Permanent Magnets. arXiv:2403.00901 (2024).
[14] L. Fu et al. Global Stellarator Coil Optimization with Quadratic Constraints and Objectives. arXiv:2408.08267 (2024).
[15] S. Guinchard et al. Including the vacuum energy in stellarator coil design. arXiv:2409.01268 (2024).
[16] S. Hurwitz, M. Landreman, A. Kaptanoglu. Electromagnetic coil optimization for reduced Lorentz forces. arXiv:2410.09337 (2024).
[17] A. A. Kaptanoglu et al. Reactor-scale stellarators with force and torque minimized dipole coils. arXiv:2412.13937 (2024).
[18] H. Qiu et al. Optimization of the Compact Stellarator with Simple Coils at finite-beta. arXiv:2510.26155 (2025).
[19] M. Landreman et al. Efficient calculation of magnetic fields from ferromagnetic materials near strong electromagnets. arXiv:2511.17305 (2025).
[20] R. Jorge et al. Single-Stage Stellarator Optimization: Combining Coils with Fixed Boundary Equilibria. arXiv:2302.10622 (2023).
[21] R. Jorge, A. Giuliani, J. Loizu. Simplified and Flexible Coils for Stellarators using Single-Stage Optimization. arXiv:2406.07830 (2024).
[22] D. Nash et al. Prototyping and Test of the "Canis" HTS Planar Coil Array for Stellarator Field Shaping. arXiv:2503.18960 (2025).
[23] J. Biu, R. Jorge. Axisymmetric Coil Winding Surfaces for Non-Axisymmetric Fusion Devices. arXiv:2505.07703 (2025).
[24] T. Elder et al. Stellarator divertor design by optimizing coils for surfaces with sharp corners. arXiv:2510.27624 (2025).
[25] R. Jorge et al. Direct Microstability Optimization of Stellarator Devices. arXiv:2301.09356 (2023).
[26] P. Huslage et al. Strain Optimization for ReBCO HTS Stellarator Coils in SIMSOPT. arXiv:2409.01925 (2024).
[27] P. F. Gil et al. Manufacturing Tolerances of Non-Planar Coils for an Optimized Tabletop Stellarator. arXiv:2507.22516 (2025).
[28] C. P. S. Swanson et al. Overview of the Helios Design: A Practical Planar Coil Stellarator Fusion Power Plant. arXiv:2512.08027 (2025).
[29] A. Ulrich et al. Permanent magnet optimization of stellarators with coupling from finite permeability and demagnetization effects. arXiv:2512.14997 (2025).
[30] S. A. Cadena et al. ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks. arXiv:2506.19583 (2025).
[31] A. M. Wright, B. J. Faber. On the accessibility of stable reactor operating regimes in quasi-symmetric stellarators. arXiv:2512.22355 (2025).
