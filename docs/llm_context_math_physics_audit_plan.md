# Plan: Mathematical and Physical Correctness Audit for llm_context.md

## Convention: Scaling Factors are Dimensionless

- **a0**: `a0 = 1.7 / minor_radius`. Dimensionless. Converts reactor-scale threshold values to device-scale.
- **I₀²** (current ratio squared): `(I_device / I_reactor)²` — dimensionless. Both a0 and I₀² only scale numerical values; thresholds already have correct dimensions (N/m, m, 1/m, etc.).

---

## Overview

This plan outlines how to audit `knowledge/llm_context.md` for:

1. **Mathematical correctness** — formulas, notation, scaling relations
2. **Physical correctness** — physics claims, units, domain conventions
3. **Relevance** — alignment with StellCoilBench scope (stellarator coil optimization via simsopt)

The audit should cross-check the static sections (Optimization Guide, Threshold Scaling) against the actual implementation in `src/stellcoilbench/`, and validate that the Literature Context summaries accurately represent the cited papers and remain on-topic.

---

## 1. Threshold Scaling vs Implementation

### 1.1 Cross-Reference with `_thresholds.py`

Compare each scaling rule in the **Threshold Scaling** table (llm_context lines 104–114) to the implementation in `src/stellcoilbench/coil_optimization/_thresholds.py`.


| Threshold                       | llm_context claim | Implementation               | Check             |
| ------------------------------- | ----------------- | ---------------------------- | ----------------- |
| `length_threshold`              | ÷ a0              | `length_threshold /= a0`     | ✓                 |
| `cc_threshold`                  | ÷ a0              | `cc_threshold /= a0`         | ✓                 |
| `cs_threshold`                  | ÷ a0              | `cs_threshold /= a0`         | ✓                 |
| `curvature_threshold`           | × a0              | `curvature_threshold *= a0`  | ✓                 |
| `msc_threshold`                 | × a0              | `msc_threshold *= a0`        | ✓                 |
| `torsion_threshold`             | × a0              | `torsion_threshold *= a0`    | ✓                 |
| `force_threshold`               | ÷ a0 then × I²    | **× a0** then × I²           | ⚠ Code discrepancy (user: ÷ a0 correct) |
| `torque_threshold`              | ÷ a0 then × I²    | No a0, × I₀²                 | ⚠ **Fix llm_context**: torque → "× I₀² only" |
| `arclength_variation_threshold` | ÷ a0²             | `*= a0**2`                   | ⚠ **Fix llm_context**: → "× a0²" (user: scaling correct) |


**Force:** Force per unit length (N/m) scales as I²/R. User confirms llm_context "÷ a0 then × I²" is correct (I² = I₀² = current-ratio squared, dimensionless). Code does × a0 — investigate.

**Torque:** Torque per unit length scales as I² only (no length dependence). No a0 factor. Code is correct; llm_context should say "× I₀² only" and clarify I₀² is the dimensionless current ratio squared.

**Arclength variation:** User confirms `*= a0²` is correct. We scale the threshold from reactor-scale to device-scale; it remains a dimensional quantity (m²). No nondimensionalization. Fix llm_context: change "÷ a0²" to "× a0²".

### 1.2 Default Values

Compare the **Defaults** table (llm_context lines 131–141) to `_thresholds.py` kwargs defaults and `config_scheme.py`:

- `length_threshold`: llm says 180 m; code default 200.0
- `cc_threshold`: llm says 1.5 m; code default 0.8
- `cs_threshold`: llm says 2.0 m; code default 1.3
- `force_threshold`: llm says 1e6 N/m; code default 200.0 (in `_thresholds.py`)

Identify the canonical source (e.g. `CaseConfig`, `proposer_policy.yaml`) and reconcile.

---

## 2. Static Section Formulas

### 2.1 Primary Score / Squared Flux

- llm_context: "∫(B·n)² over the plasma surface"
- Verify against `post_processing/_bdotn.py` and simsopt `BiotSavart` usage.
- Check: is it ∫(B·n)² dA or ⟨|B·n|/B⟩² or another convention?

### 2.2 Scaling Factor a0

- llm_context: `a0 = 1.7 / minor_radius`
- Verify `ARIES_CS_MINOR_RADIUS` and `get_reference_radii()` in `path_utils/_surface_resolution.py` or `config_scheme.py`.

### 2.3 Current Scale Factor

- llm_context: `current_scale_factor = (total_current / total_current_reactor_scale)²`
- Matches `_optimization_loop.py` lines 419–422. ✓

---

## 3. Literature Context — Mathematical Correctness

### 3.1 Common Symbols and Conventions

Spot-check that symbols are used consistently:

- **B·n** (flux error): standard notation; confirm no confusion with B_n (Boozer coordinate)
- **ι (iota)** vs i (current)
- **κ (curvature)** vs k (wavenumber)
- **Quasisymmetry (QA, QH, QI)** — definitions consistent with community usage

### 3.2 Key Formulas

Audit paper summaries for:

- **Augmented Lagrangian** [7]: L_A = f(x) − λ^T c(x) + (1/2)‖√μ◦c(x)‖² — verify against Gil et al. paper
- **Deflation** [11]: M(x; x_i^*) = 1/‖x−x_i^*‖^p + σ — check exponent and constraint form
- **Vacuum energy** [15]: E = (1/2μ₀)∫B² dV = (μ₀/8π)Σ IᵢIⱼ Lᵢⱼ — standard expression; δE = (1/2)∮(j×B)·δx dℓ
- **Strain** [26]: ε_tor, ε_bend thresholds ~0.2–0.4% — verify ReBCO literature

### 3.3 Units

- Force: N/m (per unit length) vs N — ensure summaries distinguish pointwise dF/dl from net force
- Curvature: 1/m; msc: 1/m²
- Torque: N (or N·m) — confirm convention in simsopt

---

## 4. Literature Context — Physical Correctness

### 4.1 Plasma Physics Claims

- **Quasisymmetry** (QA, QH, QI): summaries should not conflate types; QA = axisymmetric Boozer |B|, QH = helical, QI = omnigenous
- **Critical gradient, ITG, KBM** [12, 25, 31]: formulas and scaling (e.g. a/L_T,crit) — spot-check against papers
- **Effective ripple** ε_eff [18]: neoclassical confinement; ε_eff^(3/2) for 1/ν regime

### 4.2 Coil Physics

- **Biot-Savart**: linear in current when positions fixed; verify summaries don’t claim nonlinearity where it’s linear
- **Lorentz force**: dF/dl = I t×B — standard; self-force regularization (e.g. Hurwitz–Landreman) in [16]
- **Linking number**: 0 = no linked coils; confirm this is the intended “good” state in StellCoilBench

### 4.3 Reactor-Scale Numbers

- ARIES-CS: a = 1.7 m, B₀ ≈ 5.7 T
- d_cs ~ 1.2–1.3 m (blanket), d_cc ~ 0.7–0.8 m
- Force limits ~ 400 kN/m (VIPER-class cables) — verify units (per-meter vs total)

---

## 5. Relevance to StellCoilBench

### 5.1 Direct vs Tangential

- **Direct**: Coil optimization (filament, winding surface, dipole), simsopt, VMEC, B·n, constraints, Fourier coils
- **Tangential**: Permanent magnets (GPMO) — no coils but relevant for PM stellarators; turbulence/gyrokinetics — informs targets, not coil DOFs
- **Marginal**: [12] ITG critical gradient — under "Force and Torque" but primarily turbulence; keep if useful for reactor target selection

### 5.2 Scope Boundaries

- Papers that are **purely** equilibrium design (no coils) → low relevance
- Papers that are **purely** PM (no electromagnetic coils) → moderate (different topology, but optimization patterns transfer)
- Papers focused on **manufacturing, metrology, control** → relevant for case design and constraints

### 5.3 Advice Usability

For each paper’s **Optimization Advice** bullets, assess:

- Can the advice be applied in StellCoilBench today? (e.g. "use augmented Lagrangian" — yes; "run GS2 in loop" — no)
- Are thresholds/values (d_cc, κ_max, etc.) in reasonable ranges for StellCoilBench scales?
- Are references to external codes (DESC, FOCUS, NESCOIL) accurate and helpful?

---

## 6. Implementation Order

1. **Threshold scaling fixes** — Correct torque to "× I₀² only"; correct arclength_variation to "× a0²"; clarify I₀² dimensionless; reconcile defaults with code
2. **Formula verification** — Spot-check 5–10 key formulas against papers and code
3. **Unit audit** — Grep for N/m, N·m, 1/m, 1/m²; ensure consistent usage
4. **Relevance pass** — For each of 31 papers, tag: direct / tangential / marginal; flag any that should be moved or excluded
5. **Advice usability** — Mark advice that is not actionable in current StellCoilBench (for future enhancement or caveat)

---

## 7. Deliverables

- **Corrected llm_context.md** — Fix static section (Threshold Scaling, Defaults) to match implementation
- **Audit log** — List of formula/claim checks with pass/fail and source
- **Relevance matrix** — Per-paper relevance and actionability notes
- **Optional** — Add short "Caveats" subsection for known limitations (e.g. torque scaling, PM vs coil scope)

---

## 8. Tools and References

- **Code**: `src/stellcoilbench/coil_optimization/_thresholds.py`, `_optimization_loop.py`, `_constraint_builders.py`, `reactor_scale.py`
- **Summaries**: `knowledge/summaries/*.md`
- **Papers**: arxiv.org for full text verification
- **StellCoilBench case schema**: `config_scheme.py`, `proposer_policy.yaml`

