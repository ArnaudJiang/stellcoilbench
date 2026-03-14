# StellCoilBench LLM Context

Consolidated domain context for the LLM-based case proposer. Loaded as a single
document to reduce file I/O; sections are preserved for maintainability.

## Optimization Guide

Reference for understanding coil optimization parameters and their effects.

### Key Parameters

#### Coil Count (`ncoils`)
- Number of unique (base) coils per half field period
- More coils → better B-field accuracy, but more complex engineering
- Typical: 3–7 for stellarators, 6 for tokamaks
- Each base coil is replicated by stellarator symmetry (nfp × 2 if stellsym)

#### Fourier Order (`order`)
- Number of Fourier harmonics describing coil shape
- Higher order → more shape freedom → lower flux error, but risk of overfitting
- Typical: 4–16
- Fourier continuation (`fourier_continuation`) progressively increases order

#### Fourier Continuation
- Start at low order (e.g. 4), optimize, then increase to next order (e.g. 8, 16)
- Prevents getting stuck in poor local minima from high-order initialization
- Enabled via `fourier_continuation: {enabled: true, orders: [4, 8, 16]}`

#### Algorithms
- **L-BFGS-B**: Default, fast, gradient-based. Good for most cases.
- **augmented_lagrangian**: Better constraint handling. Uses sub-optimizer (L-BFGS-B).
  Slower but produces more feasible solutions for tight constraints.
- **SLSQP**, **BFGS**: Alternatives, rarely used.

### Objective Terms

The optimizer minimizes a weighted sum of:

| Term | Weight Key | Threshold Key | Effect |
|------|-----------|--------------|--------|
| Squared flux (B·n) | `flux_weight` | `flux_threshold` | Primary objective — field accuracy |
| Coil length | `length_weight` | `length_threshold` | Shorter coils preferred |
| Coil-coil distance | `cc_weight` | `cc_threshold` | Minimum separation between coils |
| Coil-surface distance | `cs_weight` | `cs_threshold` | Minimum distance to plasma |
| Curvature | `curvature_weight` | `curvature_threshold` | Smoother coils |
| Mean squared curvature | `msc_weight` | `msc_threshold` | Global smoothness |
| Torsion | `torsion_weight` | `torsion_threshold` | Coil torsion (twist) constraint |
| Force | `force_weight` | `force_threshold` | Electromagnetic forces on coils |
| Torque | `torque_weight` | `torque_threshold` | Electromagnetic torques |
| Arclength variation | `arclength_weight` | `arclength_variation_threshold` | Uniform coil spacing |

### Common Failure Modes

#### `min_sep_violation`
Coils too close together or to the plasma surface. **Fix**: relax cc/cs thresholds
or increase weights.

#### `line_search_fail`
Optimizer couldn't find a descent direction. **Fix**: reduce trust region,
try different algorithm, or use Fourier continuation.

#### `timeout`
Exceeded time limit. **Fix**: reduce `max_iterations`, use lower resolution,
or simplify the problem (fewer coils, lower order).

#### `nan_in_objective`
Degenerate geometry. **Fix**: add regularization, check initial coils,
try different `random_seed`.

### Metrics Interpretation

#### Primary Score
- `final_squared_flux` (or `score_primary`): Lower is better
- Represents ∫(B·n)² over the plasma surface

#### Engineering Metrics
- `final_total_length`: Total coil length (device scale, meters)
- `final_min_cc_separation`: Minimum coil-to-coil gap
- `final_min_cs_separation`: Minimum coil-to-surface gap
- `final_mean_squared_curvature`: Coil smoothness (lower = smoother)
- `final_max_max_coil_force`: Peak force on any coil segment
- `final_linking_number`: 0 = no linked coils (good)

#### Reactor-Scale Metrics
- Computed via scaling from device to ARIES-CS reference
- Used for feasibility assessment (do coils fit within engineering limits?)
- `composite_score`: Geometric mean of constraint margins (higher = better)

## Threshold Scaling

All constraint thresholds in StellCoilBench are specified at **ARIES-CS reactor scale**
(minor radius a = 1.7 m) and automatically rescaled to device scale in the optimizer.

### Scaling Factors (Dimensionless)

a0 and I₀² are dimensionless; they scale threshold *values* from reactor to device scale. Thresholds retain their dimensions (N/m, m, 1/m, etc.).

```
a0 = 1.7 / minor_radius
```

Where `minor_radius` is computed from the plasma surface via `s.minor_radius()`.

### How Each Threshold Scales

| Threshold | Unit | Reactor → Device | Rationale |
|-----------|------|-----------------|-----------|
| `length_threshold` | m | ÷ a0 | Coil length scales linearly with device size |
| `cc_threshold` | m | ÷ a0 | Coil-coil distance scales linearly |
| `cs_threshold` | m | ÷ a0 | Coil-surface distance scales linearly |
| `curvature_threshold` | 1/m | × a0 | Curvature is inverse length |
| `msc_threshold` | 1/m² | × a0 | Mean squared curvature is inverse length squared (but only a0, not a0²) |
| `torsion_threshold` | 1/m | × a0 | Torsion is inverse length (like curvature) |
| `force_threshold` | N/m | ÷ a0 then × I₀² | Geometric + current scaling |
| `torque_threshold` | N | × I₀² only | Current scaling only (no a0) |
| `arclength_variation_threshold` | m² | × a0² | Scale threshold from reactor to device |

### Force/Torque Additional I₀² Scaling

Force and torque thresholds also scale with the dimensionless current-ratio squared I₀² = (I_device / I_reactor)²:

```
current_scale_factor = I₀² = (total_current / total_current_reactor_scale)²
force_threshold *= current_scale_factor
torque_threshold *= current_scale_factor
```

Where `total_current_reactor_scale` is computed from freshly initialized coils
(via `initialize_coils_loop`) with the same surface and coil count. Both a0 and I₀² are dimensionless scaling factors; thresholds retain their dimensions.

### Defaults (ARIES-CS Scale)

When not specified in the case YAML, these defaults are used (at reactor scale). Source: `_thresholds.py` and `proposer_policy.yaml` ranges.

| Parameter | Default (reactor) |
|-----------|-------------------|
| `length_threshold` | 200.0 m |
| `cc_threshold` | 0.8 m |
| `cs_threshold` | 1.3 m |
| `curvature_threshold` | 1.0 1/m |
| `msc_threshold` | 1.0 1/m² |
| `force_threshold` | 200.0 N/m |
| `torque_threshold` | 200.0 N |

### Example: QA Surface (a = 0.1683 m)

```
a0 = 1.7 / 0.1683 = 10.1

Reactor-scale  →  Device-scale
length: 242.4 m  →  242.4 / 10.1 = 24.0 m
cc:     1.5 m    →  1.5 / 10.1   = 0.15 m
curvature: 1.5   →  1.5 × 10.1   = 15.2 1/m
```

### For the LLM Proposer

When proposing thresholds for new cases:
- **Always specify thresholds at reactor scale** — the code handles rescaling
- **Use `surface_catalog.json`** to look up the a0 for each surface
- **Tighter thresholds** (lower length, higher curvature) make optimization harder
- **Looser thresholds** lead to simpler but less constrained coils
- The proposer policy `proposer_policy.yaml` specifies allowed ranges for exploration

## Literature Context

Curated excerpts from 31 stellarator coil optimization papers.

### Theme: Augmented Lagrangian & Constraint Handling

### [1]

Current voxel method: discretize winding volume into cells with divergence-free current basis; minimize ∫(B_coil·n̂−B_target·n̂)² + λ‖α‖₀. Positions fixed → Biot-Savart linear in J → convex (except L0). Relax-and-split solves L0 sparse regression. Toroidal flux constraint (Ampere loop) prevents trivial J=0. Produces topologically exotic coils (e.g., helical) without pre-specified topology. Coil shape is output, not input. Results: QA, QH, precise QA; voxel solutions interpolated to filaments, then polished. Combines winding-surface convexity with 3D spatial freedom. Divergence-free basis (Cockburn); flux-jump constraints at cell interfaces. Implemented in SIMSOPT. Advantage: no winding surface; topology emerges from optimization. Disadvantage: voxel grid resolution; post-processing to extract coils.

**Optimization Advice:**
- Current voxel: use when coil topology is unknown; sparse L0 yields minimal coil count.
- Voxel solutions can seed filament optimization; interpolate to curves, then polish with standard coil objectives.
- Toroidal flux constraint: fix I_target via ∮(B_coil−B_0)·dl around axis or boundary loop.
- Divergence-free basis + flux-jump constraints at interfaces ensure physical currents.
- Resolution: finer voxels → more DOFs, longer solve; coarser → simpler coils but less accuracy.

- Topology optimization as sparse regression: convex objective + L0 yields principled coil topology discovery.
- Current voxels avoid Biot-Savart nonlinearity in position; linearity enables convex formulation.
- Voxel→filament workflow: topology discovery → interpolation → filament polish.

### [2]

Coil design has many local minima; a single local run is often suboptimal. Global-to-local approach: TuRBO (box-constrained Bayesian optimization) explores design space, then BFGS polishes. Key innovation: box constraints in physical space (coil anchor points in cylindrical R,θ,Z) instead of Fourier space. Phase I: near-axis expansion optimizes for on-axis quasisymmetry; Phase II: BoozerLS surfaces for nested flux + QA on volume; Phase III: BoozerExact for precise QA. Penalties for interlinked coils, coils unlinked with axis, nonzero axis helicity. Design targets: ι=0.1–0.9, L_target=4.5–9 m, n_coils per hp=1–13, n_fp=1–5; d_min=0.1 m, κ_max=5 m⁻¹, κ_msc=5 m⁻². TuRBO finds better minima than naive perturbation; at fixed coil length, more coils per half-period improve QA. QUASR database: ~200,000 devices. Trade-offs: QA vs coil length, aspect ratio, ι; n_fp=2 preferred for volume QA; elongation ~4–7 favored. Full workflow ~1–2 days per device on one core.

**Optimization Advice:**
- Use global exploration (e.g., TuRBO) before local coil optimization when the landscape has many minima; physical-space box constraints are more interpretable than Fourier bounds.
- Penalize linking number and axis helicity to avoid infeasible coil configurations during global search.
- At fixed total coil length, increasing coils per half-period generally improves quasisymmetry; shorter, more numerous coils preferred.
- n_fp=2 tends to achieve better volume QA than n_fp=3,4 in QA stellarators; consider in case design.
- Fourier continuation: start with low N_f (e.g., 2) for global phase, then increase (e.g., to 6) for BFGS polish; prevents degeneracy in initial exploration.

- Global coil optimization (TuRBO + BFGS) finds genuinely different solution branches than naive multi-start.
- Physical anchor-point parametrization makes box constraints meaningful for coil geometry.
- QUASR provides a large benchmark set for testing coil optimization and trade-off studies.

### [3]

This paper surveys constrained optimization methods for stellarators and implements an augmented Lagrangian algorithm in DESC. They compare linear equality constraints, sum-of-squares penalties, projection methods, interior point, SQP, and augmented Lagrangian. The augmented Lagrangian avoids arbitrary penalty weights by iteratively updating Lagrange multiplier estimates. They demonstrate constraining mean curvature \( H < 0 \) (or relaxed \( H < 0.5 \)) on the plasma boundary to avoid strong inboard indentation (“bean” shape) that complicates coil design. With curvature constraint, rotational transform bounds \( 0.43 < \iota(\rho) < 0.5 \), and fixed \( R_0, A, V, \Psi \), they obtain QA configurations with reasonable quasisymmetry (comparable to NCSX). Enforcing \( H < 0 \) strictly yields MHD instability (negative magnetic well); relaxing to \( H < 0.5 \) restores stability. Lagrange multipliers reveal where curvature limits the design.

**Optimization Advice:**
- Use augmented Lagrangian for hard constraints (curvature, coil–plasma distance, \( \iota \) bounds); it auto-tunes penalty weights.
- Constrain mean curvature \( H \leq 0 \) (or slightly positive) to avoid concave “bean” indentation that requires very close coils.
- Bean shapes aid MHD stability; curvature and magnetic well can conflict—check both when tightening curvature.
- Lagrange multipliers identify binding constraints and where relaxation yields the largest benefit.
- Relaxing constraints temporarily can help escape poor local minima; the optimizer can cross infeasible regions to reach better solutions.

- Augmented Lagrangian is well-suited to stellarator problems; existing interior point/SQP solvers often perform poorly due to scaling and feasibility.
- Curvature constraints enable coil-friendly boundaries without sacrificing too much QS.
- Trade-offs between curvature, magnetic well, and quasisymmetry are quantifiable via Lagrange multipliers.

### [4]

QUASR extended to ~370,000 QA+QH stellarators with coil sets. Workflow: Phase I near-axis expansion → Phase II BoozerLS (nested surfaces) → Phase III BoozerExact (precise QS); wrapped in TuRBO globalization. f_QS from B(φ,θ)−B_QA(θ) residual; QA and QH (rotated frame). Scans: ι, A, coil length; constraints: d_cs, d_cc≥0.1 m, κ≤5 m⁻¹, κ_msc≤35 m⁻². Near-axis landscape and PCA for visualization. 1–3 principal components often sufficient. Coil sets: filamentary Fourier; B·n numerically zero at quadrature points (tangent surfaces). Penalty method for constraints (0.1% accuracy). QUASR at quasr.flatironinstitute.org; Zenodo archive.

**Optimization Advice:**
- QUASR provides ~370k coil+equilibrium pairs; use as seeds or benchmarks for case generation.
- Engineering limits: d_cs,d_cc≥0.1 m, κ≤5 m⁻¹, κ_msc≤35 m⁻²; align StellCoilBench constraints.
- Three-phase workflow (near-axis → BoozerLS → BoozerExact) is robust for QS coil design.
- PCA: 1–3 components capture device diversity; useful for clustering and case selection.
- Coil topology (modular, dipole, helical) varies; QUASR focuses on modular; other topologies from Kaptanoglu et al.

- QUASR is the largest public QS+coil database; essential reference for StellCoilBench.
- Globalized workflow (TuRBO + phases) finds diverse QS devices with coils.
- Constraint satisfaction via penalty reweighting (0.1% tolerance) is effective.

### [5]

A wireframe—a mesh of straight current-carrying segments—defines a spatially local coil parametrization. Current continuity, poloidal current I_pol, toroidal current I_tor, and segment blocking (e.g., for ports) are enforced via linear equality constraints Cx=d. Two methods: (1) Regularized Constrained Least Squares (RCLS)—Tikhonov-regularized linear least squares minimizing f_B + f_R, analogous to REGCOIL but with segment currents; (2) Greedy Stellarator Coil Optimization (GSCO)—discrete algorithm adding current loops one-by-one to achieve target field. RCLS: Precise QA example, 8×12 wireframe, 192 segments, ~100 ms solve; ⟨|B·n|/B⟩ ≈ 6.31×10⁻⁴. Segment blocking (x_j=0) trivially reserves space for ports. GSCO yields sparse solutions with arbitrary spatial constraints. Wireframe nodes on a toroidal reference surface; segments connect adjacent nodes in toroidal/poloidal directions. Both modular and sector-confined saddle coils demonstrated. Biot-Savart from straight segments uses compact expression. N_tor×N_pol segments per half-period.

**Optimization Advice:**
- Wireframe parametrization enables segment blocking for ports/diagnostics: set x_j=0 for segments in forbidden regions.
- RCLS provides fast linear solutions (~100 ms) for initial coil current distributions; useful as stage-2 warm start.
- Tikhonov regularization W=(10⁻¹⁰ Tm/A)I controls current magnitude; tune for coil simplicity vs field accuracy.
- GSCO produces sparse, spatially constrained coil patterns; good for designs requiring specific coil placement.
- Wireframe resolution N_tor×N_pol (e.g., 8×12) balances DOFs vs field accuracy; coarser grids yield fewer, simpler coils.

- Spatially local parametrization (wireframe) simplifies constraint handling (blocking, topology) compared to global Fourier basis.
- RCLS is REGCOIL-like in formulation but with segment-based currents; useful when topology or placement matters.
- GSCO extends greedy ideas from PM optimization to coil currents; enables sparse coil solutions with custom placement rules.

### [6]

Eos is a QA planar-coil stellarator (R≈2.7 m, A=6, n_fp=2) with 12 encircling TF coils and an array of small shaping coils offset 50 cm from the plasma. Shaping coil currents are optimized via sparse regression to minimize B_n = ⟨B·n⟩ on the plasma surface while reducing coil count. Four methods: (1) OtD—heuristic, iteratively delete lowest-current coil; (2) Randomized OtD—probabilistic deletion by current magnitude; (3) LASSO—L1-regularized least squares; (4) MIQP—mixed-integer quadratic program (Gurobi) with binary sparsity vector. Induction matrix A: 4225×240 (quadrature points × coils per half-field-period); |i|∞≤1.3 MA. Pareto fronts: MIQP achieves lowest mean B_n at every sparsity level (e.g., 80 coils removed: 20% lower mean B_n than OtD). LASSO best for total-current vs mean B_n trade-off; OtD favors high-current coils. Perturbation study: 1 cm displacement + 1° rotation per coil; sparsity improves robustness; LASSO aligns with robust regression theory (Xu et al.). Masked re-optimization (fix sparsity pattern, re-solve currents) standardizes comparison. DESC for equilibrium; FOCUS for encircling coil optimization.

**Optimization Advice:**
- For planar shaping-coil arrays: MIQP gives Pareto-optimal mean B_n at fixed sparsity; use when coil count is primary constraint.
- LASSO L1 penalty minimizes total current at given B_n; useful when superconductor cost or current limits matter.
- OtD (low-current deletion) is fast but suboptimal; 20% worse mean B_n at same sparsity vs MIQP in Eos study.
- Masked re-optimization: after sparsifying, fix z=𝟙(i≠0) and re-solve currents to fairly compare methods.
- Perturbation robustness: sparser solutions have fewer coils to perturb; LASSO solutions align with robust regression to placement errors.

- Sparse regression (LASSO, relax-and-split, MIQP) substantially outperforms heuristic current-threshold deletion for planar coil arrays.
- Trade-offs: mean B_n vs sparsity, mean B_n vs total current, max B_n—different methods excel on different Pareto fronts.
- Manufacturing perturbations favor sparser configurations; include placement-error studies in case design.

### [7]

Augmented Lagrangian replaces weight-penalty coil optimization with constraint-based formulation: L_A = f(x) − λ^T c(x) + (1/2)‖√μ◦c(x)‖². Lagrange multipliers λ and penalties μ updated automatically; no manual weight tuning. Squared flux f_SF moved to constraint: max(f_SF−10⁻⁶,0)²; once satisfied, algorithm improves engineering terms. Engineering constraints: d_cs≥1.3 m, d_cc≥0.7 m, L≤150–200 m, κ, κ_msc, linking number, pointwise force. Internal minimizer: L-BFGS-B. Applied to five stellarators: Landreman-Paul QA, Landreman-Paul QH, Stellaris (SQUID), W7-X, HSX. Pareto-optimal coils outperform published sets (lower f_SF, better d_cs, d_cc, curvature). Implementation in SIMSOPT. Comparison with penalty method: augmented Lagrangian finds feasible regions without ω→∞; avoids large weight scans. Stellaris: ~15 mm anisotropic deformation from EM loads; force constraints critical for reactor scale.

**Optimization Advice:**
- Use augmented Lagrangian for coil optimization when multiple constraints (d_cs, d_cc, L, κ, force) must be satisfied—avoids weight tuning.
- Move squared flux to constraint with target f_SF≤10⁻⁵ (or 10⁻⁶); frees optimizer to improve engineering once target met.
- Initialize λ from uniform [0,1]; update λ←λ−μ◦c when ‖c‖<η; else μ←τμ.
- Threshold-based constraints: g↦max(g−g_target,0) for inequalities.
- For reactor cases: enforce d_cs~1.3 m (blanket), d_cc~0.7 m; force limits ~400 kN/m for VIPER-class cables.

- Augmented Lagrangian automates constraint satisfaction; no user weight tuning for 10–15 objectives.
- Constraining f_SF (vs minimizing) lets optimizer exploit slack for engineering improvement.
- Five-device validation: method generalizes across QA, QH, QI, W7-X, HSX.

### [8]

Stage-two coil design: minimize χ²_B = ∫(B·n)² dA on the plasma boundary. Surface current K = n×∇Φ with potential Φ = Φ_SV + Gζ'/(2π) + Iθ'/(2π); G = net poloidal current, I = net toroidal current. NESCOIL/REGCOIL solve linear least-squares with Tikhonov regularization. Coil cutting: use contours of constant Φ; current between contours = Φ₁−Φ₀. Modular: I=0, G≠0; helical: I,G≠0 with closure condition (G−G_ext)/(IN_FP) = q̄ integer. External TF coils: subtract G_ext from G. Implementation in DESC with REGCOIL; demo on Precise QA (Landreman–Paul) with λ=10⁻²⁰. Contour finding via marching squares; helical contours need extended θ' domain. Coil currents = equal spacing in Φ. Demonstration: modular (12 per period) and helical (16 coils) cuts from surface current; Poincaré plots show good flux surfaces.

**Optimization Advice:**
- When using external TF coils with a shaping winding surface, subtract G_ext from G in the current potential so Ampere's law is respected.
- Contour topology: I=0,G≠0 → modular; I≠0,G=0 → toroidal/VF-like; I,G≠0 with q̄ integer → helical; I=G=0 → windowpane only.
- For helical coils closing after p toroidal transits and q poloidal transits: I = −p(G−G_ext)/q.
- Marching squares for contours can fail for strongly shaped potentials; consider ODE or optimization-based contour tracing.
- Equally spaced Φ contours with equal current per coil is standard; can refine currents via sub-optimization (Miner/NCSX approach).

- Mathematical link between Φ secular terms (G,I) and physical currents is explicit: I_pol=−G, I_tor=I, I_between=Φ₁−Φ₀.
- Coil-cutting procedure is fully specified for modular and helical topologies; implementation available in DESC.
- Surface current + coil cutting provides stage-2 initial coils; filament refinement typically follows for engineering constraints.

### [9]

Quasi-single-stage (QSS) optimization integrates a winding-surface coil subproblem into stage-1 equilibrium optimization to improve plasma–coil balance without full single-stage complexity. The QUADCOIL code—a QCQP formulation with quadratic constraints and objectives—serves as a differentiable coil proxy via JAX. It supports Lorentz force minimization, curvature proxy, dipole density, and topology control (e.g., K_θ sign constraints). Two studies: (1) MUSE PM stellarator—QSS with QUADCOIL yields 29% fewer magnets than MUSE++, ~10–40 min on RTX 4060 vs 12 h on 16 CPUs; (2) ARIES-CS—force-minimizing QSS yields 27.6% RMS and 31.3% peak force reduction vs ARIES-CS baseline. Augmented Lagrangian solver for QUADCOIL; adjoint differentiation for gradients. Winding surface generator uses self-intersection removal and arc-length reparametrization. Constraints enable target-based tuning instead of weight scans.

**Optimization Advice:**
- Use winding-surface coil proxies (e.g., QUADCOIL) in equilibrium optimization to bias plasma surfaces toward coil-friendly shapes before stage-2 filament optimization.
- Constraint-based formulations (f_B ≤ target, K² ≤ target) avoid multi-solve weight scans; specify targets in physical units.
- Force/lorentz objectives in winding-surface models correlate with filament optimization; useful as stage-1 coil proxies.
- Fourier continuation (48–194 DOFs) improves QSS convergence; start low and increase resolution.
- Differentiable winding surface generation (avoid convex hull for JAX; use self-intersection removal + Tikhonov fit) enables joint plasma–coil optimization.

- QSS with QUADCOIL reaches comparable or better coil performance than REGCOIL-based QSS with far fewer iterations and less tuning.
- Constraints in winding-surface models (topology, force, curvature) make coil proxy more actionable than penalty-only formulations.
- Adjoint differentiation of coil subproblem solutions enables gradient-based equilibrium optimization with coil complexity in the loop.

### [10]

This paper presents a conditional diffusion model trained on the QUASR database to rapidly generate quasisymmetric stellarator boundary designs conditioned on \( (\overline{\iota}, A, n_{\text{fp}}, N) \). The inverse problem maps desired properties \( \mathbf{y} \) to design variables \( \mathbf{x} \) (661 Fourier coefficients) via a learned sampling mechanism. They use a DDPM with 200 timesteps and a 4-layer 2048-width MLP. PCA dimensionality reduction to \( n_r = 50 \) is used before training; samples are projected back for evaluation. In-sample devices achieve \( < 2.5\% \) quasisymmetry deviation and \( c_A, c_\iota < 5\% \); out-of-sample devices reach \( < 6\% \) QS deviation. Evaluation uses VMEC + Boozer coordinates and \( J_{QS} \) from Giuliani et al.; VMEC fails for ~41% of training and ~57% of generated surfaces. The work motivates physics-informed losses, DESC integration for gradients, and guided sampling for improved constraints.

**Optimization Advice:**
- Target \( J_{QS} < 1\% \) for good particle confinement (Wiedman et al.); many generated designs sit near 5%—use as warm starts for refinement.
- Condition on \( (\overline{\iota}, A, n_{\text{fp}}, N) \); QUASR has sparse coverage—interpolation/extrapolation via diffusion can fill gaps.
- Be aware VMEC fails on highly shaped boundaries; consider DESC for differentiable evaluation and constraint refinement.
- Use canonicalization (Giuliani et al.) for surface representation consistency when combining datasets.
- For case proposal, treat diffusion samples as initial guesses; run gradient-based refinement (SIMSOPT/DESC) to reach sub-1% QS.

- Diffusion models can rapidly generate novel stellarator designs conditioned on physical properties; extrapolation to unseen conditions is possible.
- Quality is limited by VMEC robustness and PCA; physics-informed losses and differentiable solvers could improve adherence.
- Dataset curation and unified benchmarks (QUASR, Constellaration, omnigenity DB) are needed for systematic method comparison.

### [11]

This paper introduces deflation methods to stellarator equilibrium and coil optimization, which modify the objective or add constraints to penalize already-found solutions and steer the optimizer toward distinct local minima. For nonlinear systems, deflation uses an operator \(M(\mathbf{x};\mathbf{x}_i^*)=\frac{1}{\|\mathbf{x}-\mathbf{x}_i^*\|_2^p}+\sigma\) multiplied with the residual; for optimization, it is added as a nonlinear inequality constraint \(M\leq r\). Applied to NAE-constrained equilibria, deflation with boundary Fourier coefficients (p=2, σ=500) found 25 distinct QA solutions from one NAE initial guess, sharing axis and on-axis ι but differing in shear and outer shapes. For helical-core equilibria, deflation on the magnetic axis alone (p=2, σ=100) recovered the helical core branch without a hand-tuned perturbation. For stage-one QH optimization (vacuum, R₀=1 m, aspect ratio 8, \(\bar{\iota}\in[0.7,3]\), augmented Lagrangian, ESS scaling α=1.2) with deflation on full equilibrium state (p=2, σ=0), 18 equilibria passed filters (force balance &lt;1%, ι in range). For stage-two coil optimization on a vacuum QA precise stellarator (4 coils per half-field period, Fourier order 5), constraints were d_cc≥0.09 m, d_cp≥0.2 m, curvature κ∈[-8,8] m⁻¹, max single-coil length 6.5 m, arclength variance penalty; sum-reduction deflation (Equation 14) avoided product-reduction pathologies. Six coilsets were found; five were feasible with B_n errors ~1e-3–2e-3, one failed constraints. Larger σ in equilibrium deflation improved force-balance quality; sum-reduction is preferred over product when deflating multiple solutions.

**Optimization Advice:**
- **Deflation hyperparameters:** Use p=2; for equilibrium deflation prefer larger σ (e.g. 100–500) to avoid pathological minima and improve force balance; for optimization constraints use σ=0 and r=1 with sum-reduction deflation when multiple minima are deflated.
- **Stage-two constraint suggestions:** Consider d_cc≥0.09 m, d_cp≥0.2 m, curvature bounds ±8 m⁻¹ per coil, and max length per coil (e.g. 6.5 m) rather than only total length; arclength variance penalty can reduce parametrization degeneracy and improve deflation.
- **Sum vs product deflation:** For multiple deflated solutions, use sum-reduction \(m_{sum}=\sigma+\sum_i\frac{1}{|\mathbf{x}-\mathbf{x}_i^*|^p}\) instead of product-reduction so excluded regions do not shrink as more solutions are added.
- **Degeneracy handling:** Include toroidally shifted copies (e.g. by π/N_FP) of found solutions in the deflation set to avoid rediscovering the same physical configuration; use deflation on the part of state that should be distinct (e.g. axis for helical-core, boundary modes for NAE).
- **Stellarator symmetry:** When relaxing stellarator symmetry, expect more deflation iterations to obtain meaningfully distinct solutions due to toroidal-rotation degeneracy.

- Deflation is a low-overhead way to explore multiple local minima without many initial guesses or weight scans.
- For coil optimization, constrain individual coil lengths and curvature explicitly; an arclength variance penalty helps with Fourier parametrization degeneracy.
- Sum-reduction deflation is preferable to product-reduction when deflating multiple solutions in constrained optimization.
- Coil-coil and coil-plasma separation thresholds from this work (d_cc 0.09 m, d_cp 0.2 m) are practical starting points for case design.


### Theme: Force and Torque Minimization

### [12]

This paper optimizes stellarators for increased ITG critical gradient (CG) via SIMSOPT, producing two QHS configs: HSK (absolute CG) and QSTK (toroidal CG, MHD stable). CG model: \(a/L_{T,\mathrm{crit}} \propto a/R_{\mathrm{eff}} + a/L_{\parallel,\mathrm{Floquet}}\) (absolute) or \(a/L_{T,\mathrm{crit}} = (a/R_{\mathrm{eff}}) F(b)\) with \(b=(\pi a|\nabla\alpha| R_{\mathrm{eff}}/L_{\parallel})^2\) (toroidal). HSK achieves largest known stellarator CG (\(a/L_{T,\mathrm{crit,abs}}=1.75\)) but is Mercier unstable (vacuum magnetic hill). QSTK: \(n_{\mathrm{fp}}=6\), \(A=7.5\), \(\iota\simeq[1.6,1.7]\), \(\varepsilon_{\mathrm{eff}}<1\%\) to half radius, \(\simeq5\%\) alpha loss at \(r/a=0.5\) (ARIES-CS scale), vacuum magnetic well, Mercier stable at \(\beta\sim1.65\%\), modular coils (48 coils, 4 unique). Nonlinear GENE: QSTK heat flux 2–5× lower than HSX. Short \(L_{\parallel}\) (~6a vs. 12a in HSK) increases \(b_{\min}\) and CG; expanded flux surfaces in bad curvature reduce ITG localization.

**Optimization Advice:**
- Use CG model eq. (6) for rapid ITG-threshold estimates: \(R_{\mathrm{eff}}\) from \(K_d\) peak, \(L_{\parallel}\) from sign-reversal distance of \(K_d\).
- Target toroidal ITG CG (not only absolute) to preserve MHD stability; \(f_{\mathrm{well}}\) enforces magnetic well.
- More field periods reduce \(L_{\parallel}\) and can raise CG; QSTK uses \(n_{\mathrm{fp}}=6\).
- Sample multiple \(\alpha\) (e.g. 0, \(\pi/8\), \(\pi/4\)) for CG; 8 poloidal turns for flux tube.
- Optimizing for increased \(|\nabla\alpha|\) on outboard can raise CG but risks magnetic hill; balance with \(f_{\mathrm{well}}\).

- Critical-gradient optimization yields significantly reduced ITG heat flux while preserving MHD stability, neoclassical transport, and fast-ion confinement.
- Short parallel connection length and expanded surfaces in bad curvature are key mechanisms.
- HSK vs. QSTK illustrates trade-off: maximum CG vs. MHD stability; QSTK achieves both with toroidal-CG target.

### [13]

This paper proposes converting a tokamak into a stellarator by replacing the transformer/solenoid with permanent magnets (PMs), providing 3D shaping without modular coils. Applied to ISTTOK (\(R_0=0.46\) m, \(a=8.5\) cm, \(B=0.5\) T). PM optimization uses SIMSOPT GPMO (greedy algorithm) and MAGPIE (trec trapezoidally-enclosed prisms); dipole moments minimize \(\int_{\partial V}[(B_M + B_{\text{coils}})\cdot n]^2\). Equilibria are easier to reproduce with PMs in order: QA (lowest \(\iota\)), QI, QH; low-\(\iota\) QA achieves <0.01% normalized field error. Circular VV cross-section outperforms equilibrium-shaped VV for PM placement; radially oriented magnets dominate. trec design with cubic magnets, face + face-edge + face-corner polarizations; spacing for mounting and ports. ISTELL scenarios: shifted/aligned equilibrium with ISTTOK vessel; plasma-accessing ports (33 total) add complexity. N52 NdFeB used; \(B\leq 0.9\) T inside TF coils is below coercivity. GPMO backtracking (GPMOb) improves accuracy and reduces magnet count.

**Optimization Advice:**
- For PM-based coils: prefer low-\(\iota\) QA equilibria; higher \(\iota\) demands more poloidal field from PMs and degrades achievability.
- Circular VV allows better PM field quality than equilibrium-shaped VV when magnet volume is comparable; PMs farther from plasma on circular grid can still improve.
- Use GPMO backtracking for binary dipole arrays; fewer magnets and better accuracy than baseline GPMO.
- face + face-edge + face-corner polarizations suffice; edge/corner add little benefit but increase cost.
- Include port locations and mounting structure spacing in PM grid; uniform grids overestimate achievable field quality.
- \(\phi_{\text{edge}}\) scan: relative error grows at high flux; QA has a plateau where N52 magnetization is sufficient.

- Tokamak-to-stellarator conversion via PMs is feasible for small machines (ISTTOK scale) with low-\(\iota\) QA.
- PMs reduce coil complexity; GPMO produces buildable binary arrays with fixed orientations.
- Trade-off: PMs cannot be turned off, have discrete strength, and risk demagnetization at high \(B\); reactor-scale may need superconducting tiles or LTS coils.

### [14]

QUADCOIL reformulates the winding-surface coil design as a Quadratically Constrained Quadratic Program (QCQP), enabling global optimization of quadratic objectives and constraints unavailable in NESCOIL/REGCOIL. It can directly minimize or constrain Lorentz force, stored magnetic energy, curvature proxy K'·∇K', max dipole density, and field-current alignment. Topology control: inequality K'_θ·sign(I)≥0 forces purely poloidal Φ' contours (no windowpanes), bypassing REGCOIL's λ₂ scans. Shor relaxation solves QCQPs as conic programs; exactness test via SVD of solution matrix. 40 Φ' harmonics, 1024 quadrature points per field period → core-seconds solve time, ~10²× faster than filament optimization. Validated on NCSX: QUADCOIL achieves 30% lower f_B than best REGCOIL + poloidal topology in 1.77 s vs 43.5 s. Nonconvex curvature penalty min f_B + α·max(K'·∇K') outperforms REGCOIL Tikhonov. Correlation study: QUADCOIL field error predicts filament coil complexity across 4436 equilibrium/curvature/spacing combinations. Winding surface via convex hull of cross-sections avoids self-intersection; arc-length poloidal coordinate.

**Optimization Advice:**
- Use K'_θ·sign(I)≥0 (or K'·Î_helical≥0) to enforce purely poloidal or helical current topology and avoid windowpane coils without λ scans.
- Shor relaxation gives exact global minimum for nearly-convex QCQPs; field error is dominant convex term—test exactness before local refinement.
- Curvature proxy max(K'·∇K') as constraint or nonconvex penalty yields simpler coils than Tikhonov ‖K'‖² alone.
- Winding surface: convex hull of toroidal cross-sections + spline fit → no self-intersection, uniform quadrature—prefer over naive normal offset for shaped plasmas.
- QUADCOIL scales O[(n_Φ'+n_a)²]; keep n_Φ' moderate (e.g., ~40) for fast iteration; use as stage-1 proxy before filament optimization.

- QCQP formulation unlocks constraints and quadratic objectives (force, curvature, energy) impossible in linear NESCOIL/REGCOIL.
- Topology constraints eliminate manual contour selection and REGCOIL weight searches.
- QUADCOIL serves as both initial-state generator and coil-complexity metric for equilibrium-stage optimization.

### [15]

Vacuum magnetic energy E = (1/2μ₀)∫B² dV = Σᵢ(Iᵢ/2)∮ A·dℓ = (μ₀/8π)Σᵢⱼ IᵢIⱼ Lᵢⱼ is the shape-gradient of the Lorentz force: δE = (1/2)∮(j×B)·δx dℓ. Penalizing E (with quadratic flux) regularizes coil optimization and reduces inter-coil forces. Euler-Lagrange: j×[∫RⁿBⁿ dS − (ω_E/2)B]=0. Implementation in SIMSOPT: f_E = mutual + self inductance (Landreman regularization for Lᵢᵢ). Minimizing Φ₂+ω_E E prevents coils from growing to infinity (alternative to length penalty). Results: energy penalty reduces forces; correlation between E and forces. Self-inductance requires finite cross-section (circular/rectangular) for regularization. Virial theorem: structural mass M∝E/σ_Y; lowering E reduces support structure. Comparison with length penalty: both regularize; energy directly targets forces.

**Optimization Advice:**
- Add vacuum energy E to coil objective when force reduction is desired; E gradient = Lorentz force.
- E regularizes coil optimization (prevents infinite-length trivial solutions) and correlates with structural mass.
- Self-inductance Lᵢᵢ needs finite cross-section regularization (δ in Landreman formula); use consistent a,b for circular/rectangular coils.
- Trade-off: ω_E vs ω_L (length); energy more directly targets forces; length targets coil simplicity.
- For reactor structural design: M∝E/σ_Y; include E in multi-objective when structure cost matters.

- Vacuum energy is theoretically linked to Lorentz forces; penalizing E reduces forces.
- Energy regularization is a viable alternative to length regularization for well-posed coil design.
- Virial theorem quantifies structure mass–energy relation for cost scaling.

### [16]

Pointwise Lorentz force dF/dl = I t×B (self + mutual) minimized in coil optimization using reduced self-force model from Hurwitz–Landreman: B_reg from regularized 1D Biot-Savart with δ∝a² for circular cross-section; ~12 quadrature points. Implemented in SIMSOPT with JAX autodiff. Precise QA (Landreman-Paul); 5 coils/half-period, R=1 m scale. Cold-start + hot-start (continuation) Pareto exploration. Trade-offs: force ↔ d_cs (strong positive correlation); force ↔ B·n error (mediated by d_cs); force ↔ fast particle losses. Inboard bean cross-section (φ≈π) has highest mutual forces; coils reduce force by moving away (↑d_cs) or reducing B. Minimum d_cc near ζ≈0 (coils 1,2) mediates force–d_cs correlation more than global d_cc. Force threshold |dF/dl|₀ in penalty. L-BFGS. Constraints: ℓ≤5 m, κ≤12 m⁻¹, κ_MS≤6 m⁻¹, d_cc≥0.083 m, d_cs≥0.166 m (R=1 m).

**Optimization Advice:**
- Use reduced self-force model (regularized B_reg) for fast, differentiable force evaluation in coil loops; avoids full FEM.
- Force–d_cs correlation: inboard coils (bean cross-section) dominate; penalizing small L_∇B in stage-1 may improve d_cs.
- Pareto exploration: cold start + continuation (perturb weights ~5%) expands Pareto front.
- Force threshold: max(|dF/dl|−|dF/dl|₀,0)² avoids over-penalizing; set |dF/dl|₀ from material limits (e.g., 400 kN/m).
- d_cc at inboard vs global d_cc: inboard coil–coil spacing more predictive of force than global minimum.

- Pointwise force optimization significantly reduces coil loads; trade-off with d_cs and field error.
- Reduced self-force model enables gradient-based force optimization without FEM.
- Force reduction mechanism: reduce B magnitude (move coils, redistribute current) more than align t×B.

### [17]

First large-scale optimization of planar dipole coil arrays for reactor stellarators (ARIES-CS: B₀=5.7 T, r₀=1.7 m). New objectives via JAX autodiff: net force, net torque, pointwise force, pointwise torque; plus coil length, curvature, d_cc, d_cs, linking number, vacuum energy. Ablation: force/torque minimization essential—without it, solutions violate ~1 MN/m limits. Net torque can be reduced by orders of magnitude with minimal field-error sacrifice. Planar coils: polar Fourier r(φ), quaternion rotation, center (X,Y,Z), current; 8 DOFs per circular coil. TF coils: nonplanar Fourier (needed for accuracy). Joint optimization of TF + dipole currents, positions, orientations. Results: Landreman-Paul QA (41 unique dipoles, 3 TF), QH (27 dipoles, 2 TF), Schuett-Henneberg QA (16 dipoles, 2 TF); ⟨B·n̂⟩/⟨B⟩ ≲ 0.0019; d_cs≥1.5 m; forces ≲1 MN/m. Fixed locations/orientations severely degrade performance. L-BFGS; curvature terms often zeroed in practice.

**Optimization Advice:**
- For dipole arrays: optimize position and orientation, not just currents—fixed geometry fails reactor-scale.
- Net torque minimization is highly effective; can reach negligible levels without hurting B·n error.
- Pointwise force threshold (e.g., |dF/dl|₀≈10⁴ N/m) avoids coils bending away excessively for force reduction.
- d_cs≥1.5 m (blanket), d_cc≥0.8 m; TF–dipole min distance ~0.4 m.
- Vacuum energy correlates with forces/torques; can simplify objective by dropping one if correlated.

- Reactor-scale dipole arrays feasible with joint TF+dipole optimization and explicit force/torque objectives.
- Planar coils need position/orientation DOFs; current-only optimization insufficient.
- Net torque is easier to drive to zero than pointwise forces; prioritize for engineering.

### [18]

The Compact Stellarator with Simple Coils (CSSC) was originally optimized at vacuum; at finite beta the effective helical ripple degrades significantly, harming neoclassical confinement. This work uses single-stage optimization of coil currents (not shapes) to mitigate finite-beta effects: varying the IL/VF current ratio, IL coil rotation angle, and VF coil vertical displacement. With only 2–3 DOFs, grid search finds optimal parameters at β=1%: normalized IL current reduced from 1.323 to ~1.257 (≈5% reduction), δθ=0, and optional δh≈−0.05 m for VF displacement. VMEC computes free-boundary equilibria; NEO evaluates effective ripple. Bootstrap current is fixed during optimization (then iterated for self-consistency). At β=1%, ripple at finite beta matches vacuum CSSC; at β=2% mitigation is partial. MHD stability via Mercier criterion and TERPSICHORE; critical beta ~3.2–3.5% for global kink stability. Fourier coils: n_f=3 for IL coils, n_f=1 for VF.

**Optimization Advice:**
- For finite-beta cases with simple coil topology, consider current-only optimization (IL/VF ratio, coil rotations) before full coil-shape optimization—can recover vacuum-level effective ripple with minimal DOFs.
- Use effective helical ripple ε_eff^(3/2) as primary confinement target for 1/ν neoclassical transport; correlate with VMEC equilibrium + NEO.
- Fixed bootstrap-current approximation during optimization is valid if iterated for self-consistency afterward; reduces cost substantially.
- Optimal IL current ratio ~1.25–1.26 (≈5% below nominal) for the CSSC family; test similar scaling for other 4-coil configurations.
- Coil current and vertical-field displacement are coupled: larger δh ↔ smaller required current reduction for same ripple.

- Finite-beta degradation of neoclassical metrics can often be mitigated by adjusting coil currents without changing coil geometry.
- Minimal-DOF optimization (current ratio, rotation, displacement) is effective when coil shapes are already optimized at vacuum.
- Bootstrap current can be held fixed during optimization with acceptable error if a post-optimization iteration enforces self-consistency.

### [19]

Ferromagnetic blanket (e.g., EUROFER97 steel) modeled as point dipoles: M saturated, aligned with local B from coils/plasma; small perturbation approximation. No linear/nonlinear solve—only Biot-Savart-type integrals; fast, differentiable. Verified vs COMSOL for QI reactor geometry. Included in free-boundary MHD and coil optimization. Effects: slight ballooning destabilization; edge island shift from ι decrease. Coil reoptimization compensates with minor shape changes. Compatible with SIMSOPT/FAMUS dipole infrastructure. Arbitrary CAD geometry; forces computable.

**Optimization Advice:**
- For reactor cases with steel blanket: add ferromagnetic dipole model to B_coil; then reoptimize coils to compensate.
- Saturation at B≳0.3 T; perturbation valid when B_steel≪B_coil. High-field designs (HTS, 5+ T) well satisfied.
- Differentiable model → gradients for coil params available; gradient-based optimization with steel included.
- Expect small ι decrease and edge island shift; compensate in coil optimization rather than blanket redesign.
- Use when blanket material (e.g., EUROFER) is fixed and coil-plasma compensation is needed.

- Fast, non-iterative steel model enables coil optimization with blanket included.
- Steel effects are modest; minor coil reoptimization suffices for compensation.
- Saturation + small-perturbation approximations are valid for fusion-relevant fields.


### Theme: Curvature and Mean Squared Curvature

### [20]

A single-stage optimization treats plasma boundary (RBC, ZBS) and coil shapes as joint DOFs, penalizing coil–plasma mismatch with quadratic flux f_QF = ∫(B_ext·n/|B_ext|)² dS. No free-boundary equilibrium iterations; one Biot–Savart evaluation per iteration. VMEC fixed-boundary; coils from FOCUS-style Fourier representation, 3(2N_F+1) DOFs per coil. Stage-1 objectives: f_QS or f_QI, aspect ratio, ι target, elongation/mirror ratio. Stage-2 regularization: coil length g_L, curvature g_κ,max, mean-squared curvature g_κ,msc, coil–coil distance g_d, Fourier-mode variance g_ℓ. Finite-difference for VMEC; analytic derivatives for coils → fewer FD steps. Four use cases: QA and QI equilibria with modular coils. Outperforms two-stage: achieves smaller plasma objective with coils than sequential optimization. Applicable to GVEC, vacuum and finite-pressure. Gradient verification via convergence studies.

**Optimization Advice:**
- Single-stage with f_QF avoids free-boundary iteration cost; suitable when fixed-boundary equilibrium is sufficient.
- Combine finite-difference (VMEC) with analytic coil gradients to reduce total derivative evaluations.
- Regularization: length, curvature, d_min, linking number—tune thresholds (L_max, κ_max, κ_msc, d_min) jointly with physics weights.
- One coil current fixed (removed from DOFs) to prevent trivial f_QF minimization by zero currents.
- Use for equilibria without free-boundary capability (e.g., GVEC) or when rapid iteration is needed.

- Joint plasma–coil optimization with fixed-boundary equilibria yields better coil–plasma balance than two-stage.
- Quadratic flux bridges coil and plasma objectives without volumetric field evaluation.
- Hybrid derivatives (analytic for coils, FD for equilibrium) keep single-stage tractable.

### [21]

Single-stage optimization (plasma boundary + coil shapes as joint DOFs) is used to design stellarators with drastically few coils: 1–3 coils per half field-period, trim coils, helical coils, and flexible QA/QH configurations with one coil set. Methods: (1) fixed-boundary single-stage (Jorge et al. 2023)—quadratic flux f_QF penalizes coil–plasma mismatch; (2) direct coil optimization (Giuliani 2024); (3) guided coil optimization. Coil types: modular (Fourier), circular (center, normal, radius), planar non-circular (polar Fourier + quaternion), helical (toroidal or space-curve). Regularization: length, curvature, mean-squared curvature, coil–coil distance, linking number. Examples: circular coils (low N_F) for simplified equilibria; planar non-circular for QA; helical coils on circular torus and arbitrary winding surface; QA with 1–2 external trim coils per half-period; QA–QH flexibility with one coil set. VMEC fixed-boundary; SIMSOPT for coils. Boozer residual targeting enables direct coil optimization without full equilibrium solve. Scripts at github.com/rogeriojorge/simple_coil_paper.

**Optimization Advice:**
- Single-stage with quadratic flux f_QF balances plasma metrics and coil complexity; avoids two-stage dead-ends where stage-2 cannot realize stage-1 target.
- Circular coils (r₀, center, orientation): only ~6 DOFs per coil—use for exploratory designs before increasing Fourier order.
- Planar non-circular: polar Fourier + quaternion rotation; avoids gimbal lock; good for trim-coil and simplified-modular cases.
- Helical coils: toroidal-surface vs space-curve parametrizations; space-curve more flexible for asymmetric designs.
- For 1–2 coils per half-period: combine direct Boozer targeting with guided coil initialization; expect trade-off between aspect ratio and QA.

- Single-stage optimization enables designs (e.g., 1–3 coils/half-period) difficult or impossible with two-stage workflows.
- Coil parametrization choice (circular, planar Fourier, helical) strongly affects convergence and final coil count.
- Guided coil methods can bootstrap optimization when naive initialization fails.

### [22]

Canis is a 3×3 array of nine HTS planar shaping coils built by Thea Energy to demonstrate field-shaping for the Eos planar coil stellarator. Planar coil approach: encircling coils (like tokamak TF) + planar shaping coils tiled on vessel; REBCO at 20 K, >10 T on-coil; shaping coils see >14 T including self-field. Objectives: ≤1 day double-pancake takt time, ≤4 days DP production, HTS from ≥3 suppliers; closed-loop B_z control ≤1% RMS error; Eos-relevant B_z shapes. Array operated at 20 K; closed-loop control within 1% of predicted field. Manufacturing: winding in tension; mutual inductance and thermal coupling considered for control. Proof of concept for HTS planar shaping coils for stellarators. Eos/Eos-scale: 12 encircling + shaping coils; Helios: 324 shaping coils, 12 encircling.

**Optimization Advice:**
- Planar shaping coil arrays require sparse or current-only optimization (e.g., Eos 2502.07702) for B_n; Canis validates that such solutions can be built and controlled.
- Mutual inductance between shaping coils affects control; include in system model when many coils are close.
- Closed-loop field control ≤1% enables tolerance relaxation; design for 1% B_n error as acceptable operation margin.
- REBCO at >10 T, 200 A/mm²: strain/curvature limits from 2409.01925 apply if extending to non-planar HTS.
- Use planar coil cases (Eos, Helios) as benchmarks for StellCoilBench planar-coil workflows.

- HTS planar shaping coils are experimentally validated for stellarator field shaping at Thea Energy scale.
- 1% field control accuracy is achievable and supports relaxed manufacturing tolerances.
- Planar coil architecture enables demountable FSUs and sector maintenance.

### [23]

Coil curves parameterized as curves bound to a fixed coil winding surface (CWS), bypassing current-potential contour cutting. Parametrization: (R,Z) of curve on surface via parameter t; gradients available for both surface and coil shapes. Supports modular and helical coils. Enables single-stage optimization with CWS as DOF. Two CWS choices compared: (1) axisymmetric circular torus; (2) plasma-boundary rescaled outward. Both yield QA coils; axisymmetric torus is viable. Optimal plasma–coil distance studied. SIMSOPT; quadratic flux J=½∫(B_coils·n)²/|B_coils|² d²x; regularization: length, curvature, d_cc. L-BFGS-B. Fewer DOFs than unconstrained FOCUS when coils lie on surface. Applicable to deposition/milling manufacturing where coils are surface-bound.

**Optimization Advice:**
- Surface-bound coils reduce DOFs; use when manufacturing constrains coils to a CWS (e.g., deposition).
- Axisymmetric circular torus CWS can suffice for QA; no need for plasma-shaped CWS in all cases.
- Parametrization (R,Z)(t) on CWS enables analytic gradients for both coil and surface.
- Optimal d_cs depends on CWS choice; scan plasma–coil distance when comparing CWS options.
- Single-stage with CWS+coil DOFs: surface derivatives readily available for joint optimization.

- Axisymmetric winding surfaces are a practical option for QA stellarators.
- Surface-bound formulation is numerically efficient and suitable for certain manufacturing methods.
- CWS choice (circular vs plasma-rescaled) affects achievable accuracy and coil simplicity.

### [24]

First demonstration of LHD-like helical divertor with modular (non-helical) coils. Target surface: "rotating lemon"—two semicircles intersecting at sharp edges, toroidally rotated; sharp corners become X-lines. Standard coil optimization: minimize ∫(B_n/B)² dA plus length, d_cc, d_cs, curvature, mean-squared curvature, linking number. Weighted squared flux (WSF): weight w(x)=(1−|x−x_0|/d_max)^p emphasizes B_n near sharp corners; higher p → more local. Manifold optimization: trace field lines from target surface for one field period; minimize endpoint deviation from target surface—reduces chaos by targeting resonant errors directly. Fixed-point analysis (Greene's residue) quantifies chaos. Results: 4 coils/half-period, 32 total; lemon coil set achieves crisp divertor legs, far less chaos than LHD. WSF yields simpler coils (3/half-period, better d_cc, curvature) but higher separatrix chaos. Manifold optimization (warm start from lemon) reduces primary fixed-point residues by orders of magnitude. SIMSOPT + PyOculus. Randomized weight/threshold search for Pareto exploration.

**Optimization Advice:**
- For divertor targets: use surfaces with sharp corners (rotating lemon) to define X-line topology; standard B_n minimization can achieve LHD-like divertor with modular coils.
- Weighted quadrature: w(x)=(1−dist_to_corner/d_max)^p to emphasize field accuracy at critical corners; p controls locality.
- Manifold optimization (field-line deviation after one period) suppresses resonant chaos better than B_n alone; use as refinement step after initial B_n optimization.
- Manifold optimization needs non-trivial initial coils (cold start fails in pure toroidal field); warm start from B_n-optimized coils.
- Greene's residue for fixed points is sensitive; turnstile area also sensitive—use for analysis, not necessarily as optimization target.

- Modular coils can produce helical-divertor topology; LHD's high chaos is not intrinsic.
- Sharp-corner target surfaces + B_n minimization yield clean X-point divertors; manifold optimization further reduces chaos.
- Trade-off: simpler coils (WSF) ↔ lower chaos (manifold optimization); Pareto exploration via randomized weights.


### Theme: Device-Specific (MUSE, W7-X, etc.)

### [25]

This paper directly optimizes for reduced quasilinear heat flux (ITG proxy) by running GS2 linear gyrokinetics at each optimization step, combined with quasisymmetry and aspect ratio objectives. The objective is \( J = \omega_{f_Q} f_Q + f_{QS} + f_A \) with \( f_Q = \sum_{k_y} \gamma(k_y) / \langle k_\perp^2 \rangle \). They use VMEC + SIMSOPT with Levenberg–Marquardt and finite-difference gradients (MPI-parallel). The initial condition is the precise QH (Landreman–Paul); boundary modes up to \( m, |n| \leq 4 \) are varied. With \( \omega_{f_Q} = 10 \), quasilinear flux drops to W7-X–like levels while QS degrades to \( f_{QS} \sim 0.12 \); \( \omega_{f_Q} = 100 \) further reduces flux but increases alpha losses to 4.5%. Microstability-optimized configs have lower \( \iota \), more circular cross-sections, reduced \( |\nabla\psi|^2 \), and finite shear. \( f_Q \) and \( f_{QS} \) have different local minima; compromise is inevitable.

**Optimization Advice:**
- Balance microstability (\( f_Q \)) and quasisymmetry with weights; \( \omega_{f_Q} \sim 10 \) gives a reasonable compromise (W7-X–level flux, acceptable losses).
- Use large finite-difference steps initially (\( \Delta_r = 0.015/M_{\text{pol}} \)) to escape \( f_Q \) local minima; refine with smaller steps.
- Simulate at \( s = 0.25 \), \( k_y \in [0.3, 3.0] \), \( a/L_T = 3 \), \( a/L_n = 1 \); extend to multiple flux tubes for robustness.
- Expect more circular cross-sections and lower \( \iota \) when favoring microstability over pure QS.
- Linear proxies can conflict with confinement; \( \omega_{f_Q} > 1 \) trades quasisymmetry for turbulence reduction and increases alpha losses.

- First demonstration of gyrokinetics inside the optimization loop; GS2 + VMEC + SIMSOPT is feasible with MPI.
- Quasisymmetry and microstability have different minima; multi-objective weighting is essential.
- Microstability optimization tends toward lower elongation, finite shear, and reduced \( |\nabla\psi|^2 \).

### [26]

ReBCO HTS tape is brittle; strain from binormal curvature and torsion must stay below ~0.2–0.4% (0.2% used for EPOS/CSX). A strain metric is implemented in SIMSOPT: LP-norm penalties on ε_tor and ε_bend above critical thresholds, with winding orientation α as DOF in the N–B plane. Both tape-orientation-only and full (curve+orientation) optimization are supported. EPOS: single-stage optimization for quasisymmetry + strain + ι target; coil+angle DOFs yield ReBCO-compatible coils at R≈0.2 m. CSX: J_twist penalty to avoid net tape rotation; strain below 0.2% on 4 mm tape. Reactor winding pack (W7-X scaled to ARIES-CS, 54×54 cm cross section, 324 stacks): orientation optimization across the pack reduces strain below 0.4% for all stacks. Frenet vs centroid frames: centroid more regular; both require optimization for ReBCO compatibility. REGCOIL λ (regularization) correlates with strain: higher λ → simpler coils, lower strain, higher B·n error.

**Optimization Advice:**
- Add strain penalties (binormal curvature + torsion) when targeting HTS coils; use ε_crit ≈ 0.2% for conservative design.
- Penalize J_twist (net tape rotation) for manufacturability when coils are wound under tension.
- Strain can be optimized by (a) tape orientation only (fixed filament) or (b) curve + orientation; (b) needed when orientation alone cannot meet limits.
- Strain scales with coil curvature and tape geometry; small devices (R~0.2 m) are most sensitive—include strain in objective for tabletop designs.
- For reactor winding packs, optimize winding-pack orientation; curvature/torsion vary ~10%+ across stacks and must be checked for every turn.

- ReBCO compatibility requires explicit strain objectives; standard curvature regularization in REGCOIL is related but not equivalent to HTS strain limits.
- Single-stage optimization combining quasisymmetry and strain yields coils that satisfy both physics and engineering in one pass.
- Winding-pack orientation is a powerful DOF at reactor scale; a small rotation can bring all stacks below strain limits.

### [27]

EPOS tabletop stellarator (R~20 cm, r~4 cm) uses 3D-printed (steel, Ti) and CNC-milled (Al) coil frames for HTS tape. 3D scans with Hexagon Absolute Scanner; deviations along winding trenches modeled with Gaussian processes: k(d)=σ² exp(-d²/2L²)+N. GP fit: σ (amplitude), L (correlation length). CNC Al: σ~0.1 mm, max dev ~0.1 mm; 3D-printed: σ~1.5–2.4 mm, max dev ~1 mm. CNC ~10× better path accuracy than additive manufacturing. Monte Carlo: 10⁴ perturbed coil sets; f_SF (quadratic flux) distribution. Increasing σ degrades field accuracy ~4×; L has little effect. At σ=0.1 mm, CNC yields sufficient field precision for EPOS. Coil design from SIMSOPT; trench splines for path accuracy. Systematic errors (e.g., milling head -0.3 mm) identifiable from scans.

**Optimization Advice:**
- Model manufacturing errors with GP(σ,L): σ drives field degradation; L less critical for f_SF distribution.
- For small devices (R~0.2 m): CNC machining ~10× better than 3D printing for coil path accuracy.
- Include tolerance studies in case design: sample perturbed coils, compute f_SF distribution for proposed σ.
- Splines along trench sides (both walls) capture path accuracy better than full-surface deviation maps.
- Reactor-scale: CNC not viable; assembly, thermal contraction, J×B forces will dominate—GP still useful for local tolerance budgeting.

- Manufacturing process choice (CNC vs AM) has order-of-magnitude impact on coil path accuracy.
- GP parameters (σ,L) enable statistical field-error studies and design margin estimation.
- Tabletop devices are useful testbeds for coil metrology and tolerance methodology.

### [28]

Helios is a preconceptual D-T fusion power plant based on planar coil stellarator architecture. QA equilibrium, n_fp=2, A=4.5, R=8 m, a=1.8 m, B₀=6 T. Coil set: 12 large planar encircling coils + 324 planar shaping coils; all HTS, planar and convex; max 20 T on-coil. d_plasma-coil≥1.2 m (blanket); 40-year coil lifetime. 1.1 GW thermal, 390 MW net electric. Sector maintenance: removable toroidal sectors between encircling coils; ~84 days biennial, 88% capacity factor. Individual shaping coil control relaxes tolerances and enables bootstrap/error-field correction. Design drivers: practicality, conservatism, engineering margin. Compared to ARIES-CS: lower QA error, less strongly shaped boundary, coils further from plasma, planar vs 3D coils, relaxed tolerances via control.

**Optimization Advice:**
- Planar coil power plants: 12 encircling + O(300) shaping coils; benchmark cases against this topology.
- d_cs~1.2 m minimum for blanket; 40-year lifetime drives neutron shielding.
- Individual coil control enables post-assembly correction; design for ~1% B_n margin (Canis).
- QA equilibrium less strongly shaped → coils further out → simpler blanket than ARIES-CS.
- Sector maintenance: gaps between encircling coils determine removable sector size; coil spacing affects maintenance time.

- Planar coil architecture (Thea/Helios) is a viable reactor pathway with relaxed coil complexity.
- Control system compensates for tolerances; optimization can target relaxed manufacturing margins.
- Helios parameters (A=4.5, n_fp=2, 324 shaping coils) define a concrete reactor-scale planar coil target.

### [29]

This paper extends greedy permanent magnet optimization (GPMO) with macromagnetic refinement (GPMOmr) to incorporate finite permeability and demagnetizing interactions at the block level. A device-scale macromagnetics solver solves the linear system A·M=b for equilibrium magnetizations, incorporating anisotropic susceptibility χ_∥=0.05, χ_⊥=0.15 (μ_∥=1.05, μ_⊥=1.15 for Nd-Fe-B) and demagnetization tensors between blocks. On the published MUSE PM grid (~9.7k candidates), finite-μ postprocessing yields degree-scale tilts and ~1–3% magnitude changes per block; B·n differences are ~1% pointwise but the squared-flux objective f_B increases by more than a factor of two. GPMOmr uses a winner-only strategy: score candidates with rigid remanence (ArbVec), commit the winner, then run macromagnetic solve every k_mm iterations (k_mm=50 recommended). With and without backtracking (n_bt=200), GPMOmr achieves f_B within a few percent of classical GPMO; magnetization patterns differ (nonuniform magnitudes vs uniform) but surface B·n and convergence are similar. At higher B₀ (0.5 T, GB50UH grade), GPMOmr stays within ~10% of rigid-remanence GPMO. For AlNiCo (μ=3, B_r≈0.72 T), coupling strongly degrades f_B (≈3 orders of magnitude in postprocessing); GPMOmr saturates earlier with late-iteration spikes. Implementation available in SIMSOPT.

**Optimization Advice:**
- **B·n vs integrated metric:** Percent-level pointwise B·n changes can increase squared-flux f_B by O(1) when integrated; layout quality under idealized models can mask sensitivity to coupling. Consider reporting both pointwise and integrated B·n metrics.
- **Squared-flux objective:** The standard f_B = Σ w_q [(B·n)_q − (B_targ·n)_q]² is used consistently; high surface resolution (e.g. n_φ×n_θ=1024) for evaluation vs coarser (64²) for optimization is a common trade-off.
- **Greedy loop structure:** Winner-only refinement (score with fast model, commit, then refine) keeps cost tractable; embedding a full coupled solve inside the scoring loop would scale as O(N⁴) and is impractical.
- **Material regimes:** For hard magnets (Nd-Fe-B) with μ≈1.05–1.15, rigid-remanence designs remain good; for softer or high-permeability materials, coupled models become essential and achievable f_B degrades substantially.
- **Backtracking:** Backtracking (e.g. depth 200, neighbor count 12) accelerates convergence and can remove anti-parallel configurations; angular threshold (π−5°) accommodates small macromagnetic tilts.

- For Nd-Fe-B PM arrays at MUSE-like fields, macromagnetic corrections are percent-level in B·n but can more than double f_B; GPMOmr yields comparable designs with internally redistributed magnetization.
- Winner-only refinement every k_mm=50 strikes a practical balance between accuracy and cost for device-scale arrays.
- Coil-magnet coupling slightly increases typical deviation from remanence; including H_a (applied field from coils) in the macromagnetic solve improves realism.
- Soft or high-μ materials (e.g. AlNiCo with μ=3) are strongly sensitive to coupling; rigid-remanence designs can become orders of magnitude worse under macromagnetic postprocessing.


### Theme: Other Methods and Coil Design

### [30]

ConStellaration is an open dataset of ~158,000 QI-like stellarator plasma boundaries with ideal-MHD equilibria (VMEC++), computed at vacuum and five \(\beta\) levels, plus figures of merit. Generated by sampling omnigenous poloidal fields (Dudt et al. parametrization), pyQSC near-axis models, and stage-one optimizations; boundaries use Fourier \(R_{mn}, Z_{mn}\) with \(m,n \leq 4\), stellarator symmetry, 80 DoF. Three benchmarks: (1) geometric single-objective; (2) single-objective "simple-to-build" QI; (3) multi-objective ideal-MHD stable QI (compactness vs. coil simplicity). Baselines use classical optimization; learned models can generate feasible configurations without physics oracles. Dataset: HuggingFace proxima-fusion/constellaration; code: github.com/proximafusion/constellaration. Targets: aspect ratio, edge \(\iota\), mirror ratio \(\Delta_{\text{edge}}\), max elongation. Differs from QUASR (QA/QH only) by including QI and public MHD equilibria.

**Optimization Advice:**
- ConStellaration provides QI-like boundaries and MHD equilibria for benchmarking; use for training or validating optimization methods.
- QI targets: poloidally closed \(B\)-contours, straight \(B_{\max}\) lines at \(\varphi=0, 2\pi/n_{\text{fp}}\), invariant bounce distance.
- Three benchmark problems span geometric, single-objective buildability, and multi-objective MHD+coils; adopt similar structure for StellCoilBench cases.
- Fourier \(R_{mn}, Z_{mn}\) with \(m,n \leq 4\) is a practical parameterization; stellarator symmetry reduces DoF.
- VMEC++ (Schilling et al.) used for equilibria; compatible with StellCoilBench VMEC workflows.
- Learned models can generate in-domain feasible configs; useful for warm-starting or exploring QI space.

- First large-scale public QI stellarator dataset with MHD equilibria; lowers barrier for ML/optimization research.
- Standardized benchmarks enable systematic comparison of optimization methods and representations.
- Data-driven approaches can produce feasible QI configurations without expensive oracle queries.

### [31]

Reactor-relevant low-β regime (β~1%) in 3-field-period QA stellarator: MHD-stable, good neoclassical (ϵ_eff^(1/2)), flat core pressure. Macroscopically: ballooning unstable at edge but saturates benignly; no detrimental mode coupling. Microscopically (Gene): abrupt transition to deleterious transport at low local β; ITG stabilizes with β, KBM destabilizes; subdominant modes and nonlinear excitation matter. Implication: high-field low-β may be inaccessible due to micro-turbulence; KBM optimization critical for finite-β design. Equilibrium from Feng et al. (ι, elongation scan); M3D-C1, Gene. Not coil-optimization focused but informs target β and stability requirements for case design.

**Optimization Advice:**
- Low-β (β~1%) reactor scenarios may have micro-turbulence barriers; design cases across β range.
- KBM limits finite-β access; consider KBM in stage-1 optimization when targeting reactor scenarios.
- Macroscopic stability (Mercier, ballooning) can be benign; micro-turbulence can dominate—both matter for case selection.
- Flattened core + steep edge pressure: transport barrier candidate but check micro-stability.

- Low-β "safe" regime may be turbulent-limited; not just MHD-limited.
- KBM optimization should complement neoclassical/QS in reactor-targeted optimization.
- Equilibrium choice (β, ι, elongation) affects both macro and micro stability—case design should span regimes.


## References

[1] Alan A. Kaptanoglu, Gabriel P. Langlois, Matt Landreman. Topology optimization for inverse magnetostatics as sparse regression: application to electromagnetic coils for stellarators. arXiv:2306.12555 (2023).
[2] Andrew Giuliani. Direct stellarator coil design using global optimization: application to a comprehensive exploration of quasi-axisymmetric devices. arXiv:2310.19097 (2023).
[3] Rory Conlin, Patrick Kim, Daniel W. Dudt, Dario Panici, Egemen Kolemen. Stellarator Optimization with Constraints. arXiv:2403.11033 (2024).
[4] Andrew Giuliani, Eduardo Rodríguez, Marina Spivak. A comprehensive exploration of quasisymmetric stellarators and their coil sets. arXiv:2409.04826 (2024).
[5] K. C. Hammond. A framework for discrete optimization of stellarator coils. arXiv:2412.00267 (2024).
[6] Ryan Wu, Thomas Kruger, Charles Swanson. Planar Coil Optimization for the Eos Stellarator using Sparse Regression. arXiv:2502.07702 (2025).
[7] Pedro F. Gil, Weiping Li, Julianne Stratton, Alan A. Kaptanoglu, Eve V. Stenson. Augmented Lagrangian methods produce cutting-edge magnetic coils for stellarator fusion reactors. arXiv:2507.12681 (2025).
[8] Dario Panici, Rory Conlin, Rahul Gaur, Daniel W. Dudt, Yigit Gunsur Elmacioglu, Matt Landreman, Todd Elder, Nadav Snir, Itay Gissis, Yasha Nikulshin, Egemen Kolemen. Surface Current Optimization and Coil-Cutting Algorithms for Stage-Two Stellarator Optimization. arXiv:2508.09321 (2025).
[9] Lanke Fu, Dario Panici, Elizabeth Paul, Alan Kaptanoglu, Amitava Bhattacharjee. A flexible and differentiable coil proxy for stellarator equilibrium optimization. arXiv:2510.16243 (2025).
[10] Misha Padidar, Teresa Huang, Andrew Giuliani, Marina Spivak. Diffusion for Fusion: Designing Stellarators with Generative AI. arXiv:2511.20445 (2025).
[11] Dario Panici, Byoungchan Jang, Rory Conlin, Daniel Dudt, Yigit Gunsur Elmacioglu, Egemen Kolemen. Deflation Techniques for Stellarator Equilibrium and Optimization. arXiv:2602.09957 (2026).
[12] G. T. Roberg-Clark, G. G. Plunk, P. Xanthopoulos, C. Nührenberg, S. A. Henneberg, H. M. Smith. Critical gradient turbulence optimization toward a compact stellarator reactor concept. arXiv:2301.06773 (2023).
[13] M. Madeira, R. Jorge. Tokamak to Stellarator Conversion using Permanent Magnets. arXiv:2403.00901 (2024).
[14] Lanke Fu, Elizabeth J. Paul, Alan A. Kaptanoglu, Amitava Bhattacharjee. Global Stellarator Coil Optimization with Quadratic Constraints and Objectives. arXiv:2408.08267 (2024).
[15] S. Guinchard, S. R. Hudson, E. J. Paul. Including the vacuum energy in stellarator coil design. arXiv:2409.01268 (2024).
[16] Siena Hurwitz, Matt Landreman, Alan Kaptanoglu. Electromagnetic coil optimization for reduced Lorentz forces. arXiv:2410.09337 (2024).
[17] Alan A. Kaptanoglu, Alexander Wiedman, Jacob Halpern, Siena Hurwitz, Elizabeth J. Paul, Matt Landreman. Reactor-scale stellarators with force and torque minimized dipole coils. arXiv:2412.13937 (2024).
[18] Haorong Qiu, Guodong Yu, Peiyou Jiang, Guoyong Fu. Optimization of the Compact Stellarator with Simple Coils at finite-beta. arXiv:2510.26155 (2025).
[19] Matt Landreman, Humberto Torreblanca, Antoine Cerfon. Efficient calculation of magnetic fields from ferromagnetic materials near strong electromagnets, and application to stellarator coil optimization. arXiv:2511.17305 (2025).
[20] R. Jorge, A. Goodman, M. Landreman, J. Rodrigues, F. Wechsung. Single-Stage Stellarator Optimization: Combining Coils with Fixed Boundary Equilibria. arXiv:2302.10622 (2023).
[21] R. Jorge, A. Giuliani, J. Loizu. Simplified and Flexible Coils for Stellarators using Single-Stage Optimization. arXiv:2406.07830 (2024).
[22] D. Nash, D. A. Gates, W. S. Walsh, M. Slepchenkov, et al. (Thea Energy team). Prototyping and Test of the "Canis" HTS Planar Coil Array for Stellarator Field Shaping. arXiv:2503.18960 (2025).
[23] J. Biu, R. Jorge. Axisymmetric Coil Winding Surfaces for Non-Axisymmetric Fusion Devices. arXiv:2505.07703 (2025).
[24] Todd Elder, Matt Landreman, Christopher B. Smiet, Robert Davies. Stellarator divertor design by optimizing coils for surfaces with sharp corners. arXiv:2510.27624 (2025).
[25] R. Jorge, W. Dorland, P. Kim, M. Landreman, N. R. Mandell, G. Merlo, T. Qian. Direct Microstability Optimization of Stellarator Devices. arXiv:2301.09356 (2023).
[26] Paul Huslage, Elisabeth J. Paul, Mohammed Haque, Pedro F. Gil, Nicolo Foppiani, Jason Smoniewski, Eve V. Stenson. Strain Optimization for ReBCO High-Temperature Superconducting Stellarator Coils in SIMSOPT. arXiv:2409.01925 (2024).
[27] Pedro F. Gil, Vitali Brack, Tristan Schuler, Paul Huslage, E. V. Stenson. Manufacturing Tolerances of Non-Planar Coils for an Optimized Tabletop Stellarator. arXiv:2507.22516 (2025).
[28] C. P. S. Swanson, S. T. A. Kumar, D. W. Dudt, et al. (Thea Energy team). Overview of the Helios Design: A Practical Planar Coil Stellarator Fusion Power Plant. arXiv:2512.08027 (2025).
[29] Armin Ulrich, Mason Haberle, Alan A. Kaptanoglu. Permanent magnet optimization of stellarators with coupling from finite permeability and demagnetization effects. arXiv:2512.14997 (2025).
[30] Santiago A. Cadena, Andrea Merlo, Emanuel Laude, Alexander Bauer, Atul Agrawal, Maria Pascu, Marija Savtchouk, Enrico Guiraud, Lukas Bonauer, Stuart Hudson, Markus Kaiser. ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks. arXiv:2506.19583 (2025).
[31] Adelle M. Wright, Benjamin J. Faber. On the accessibility of stable reactor operating regimes in quasi-symmetric stellarators. arXiv:2512.22355 (2025).