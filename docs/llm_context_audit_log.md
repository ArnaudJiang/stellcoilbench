# llm_context.md Math/Physics Audit Log

## Threshold Scaling (Completed 2025-03)

| Item | Result |
|------|--------|
| Torque scaling | Fixed: "× I₀² only" (no a0) |
| Arclength variation | Fixed: "× a0²" |
| Force scaling | Fixed in `_thresholds.py`: ÷ a0 (was × a0) |
| I₀² clarification | Added: dimensionless current-ratio squared |
| Defaults | Reconciled with _thresholds.py |
| a0 convention | Clarified: dimensionless scaling factor |

## Formula Spot-Check (Literature Context)

| Paper [ref] | Formula | Check |
|-------------|---------|-------|
| [7] Augmented Lagrangian | L_A = f(x) − λ^T c(x) + (1/2)‖√μ◦c(x)‖² | Consistent with Gil et al. |
| [11] Deflation | M = 1/‖x−x_i*‖^p + σ | Correct form |
| [15] Vacuum energy | E = (1/2μ₀)∫B² dV = (μ₀/8π)Σ IᵢIⱼ Lᵢⱼ | Standard expression |
| [26] Strain | ε_crit ≈ 0.2–0.4% | ReBCO literature range |

## Relevance Tags (31 Papers)

All 31 papers are directly or tangentially relevant to stellarator coil optimization:

- **Direct** (coil optimization, simsopt, constraints): [1]–[11], [14]–[17], [20]–[24], [26]–[27], [29]
- **Tangential** (informs targets, different topology): [12] ITG/turbulence, [13] PM tokamak conversion, [18] finite-beta current optimization, [19] ferromagnetic blanket, [25] microstability, [28] Helios reactor, [30] ConStellaration dataset, [31] reactor regimes
