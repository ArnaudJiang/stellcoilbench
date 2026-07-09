"""Metric definition data for leaderboard documentation.

This module holds the canonical definitions and detailed definitions for
all tracked metrics. Used by :mod:`_formatting` for display and by
:mod:`_writers_metric_defs` for RST generation.
"""

from __future__ import annotations

from typing import Any

# Short LaTeX-style definitions (metric key -> string)
METRIC_DEFINITIONS: dict[str, str] = {
    # B-field related
    "final_squared_flux": r"Squared flux objective $f_B = \int_{S} (\mathbf{B} \cdot \mathbf{n} - B_\text{target})^2 dS$ on plasma surface ($\text{T}^2 \text{m}^2$). When virtual casing is used, $B_\text{target} = B_\text{external}^\text{normal}$; otherwise $B_\text{target} = 0$.",
    "final_normalized_squared_flux": r"Squared flux objective $f_B = \int_{S} (\mathbf{B} \cdot \mathbf{n} - B_\text{target})^2 dS$ on plasma surface ($\text{T}^2 \text{m}^2$). Alias for final_squared_flux.",
    "max_BdotN_over_B": r"Maximum normalized normal field component $\max(B_n)$ where $B_n = \frac{|\mathbf{B}_\text{coil} \cdot \mathbf{n} - B_\text{target}|}{|\mathbf{B}_\text{coil}|}$ (dimensionless)",
    "avg_BdotN_over_B": r"Average normalized normal field component $\bar{B}_n = \frac{\langle |\mathbf{B}_\text{coil} \cdot \mathbf{n} - B_\text{target}| \rangle}{\langle |\mathbf{B}_\text{coil}| \rangle}$ (dimensionless). When virtual casing is used, $B_\text{target} = B_\text{external}^\text{normal}$; otherwise $B_\text{target} = 0$.",
    "max_BdotN_over_target_B": r"Maximum normal-field residual normalized by reference field $\max \frac{|\mathbf{B}_\text{coil} \cdot \mathbf{n} - B_\text{target}|}{B_0}$, where $B_0$ is target_B (dimensionless).",
    "avg_BdotN_over_target_B": r"Average normal-field residual normalized by reference field $\frac{\langle |\mathbf{B}_\text{coil} \cdot \mathbf{n} - B_\text{target}| \rangle}{B_0}$, where $B_0$ is target_B (dimensionless).",
    # Curvature
    "final_average_curvature": r"Mean curvature $\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i$ over all coils, where $\kappa_i = |\mathbf{r}''(s)|$ ($\text{m}^{-1}$)",
    "final_max_curvature": r"Maximum curvature $\kappa_\text{max}$ across all coils ($\text{m}^{-1}$)",
    "final_mean_squared_curvature": r"Mean squared curvature $\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} \kappa_i^2$ ($\text{m}^{-2}$)",
    # Separations (d_cc and d_cs are minimum distances)
    "final_min_cs_separation": r"Minimum coil-to-surface distance $d_{cs}$ ($\text{m}$)",
    "final_min_cc_separation": r"Minimum coil-to-coil distance $d_{cc}$ ($\text{m}$)",
    "final_cs_separation": r"Average coil-to-surface separation $d_{cs}$ ($\text{m}$)",
    "final_cc_separation": r"Average coil-to-coil separation $d_{cc}$ ($\text{m}$)",
    # Length
    "final_total_length": r"Total length $L = \sum_{i=1}^{N} \int_{0}^{L_i} ds$ of all coils ($\text{m}$)",
    # Forces/Torques
    "final_max_max_coil_force": r"Maximum force magnitude $F_\text{max}$ across all coils ($\text{N}/\text{m}$)",
    "final_avg_max_coil_force": r"Average of maximum force $\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max(|\mathbf{F}_i|)$ per coil ($\text{N}/\text{m}$)",
    "final_max_max_coil_torque": r"Maximum torque magnitude $\tau_\text{max}$ across all coils ($\text{N}$)",
    "final_avg_max_coil_torque": r"Average of maximum torque $\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max(|\boldsymbol{\tau}_i|)$ per coil ($\text{N}$)",
    # Time
    "optimization_time": r"Total optimization time $t$ ($\text{s}$)",
    # Linking number
    "final_linking_number": r"Linking number $\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot (d\mathbf{r}_i \times d\mathbf{r}_j)}{|\mathbf{r}_i - \mathbf{r}_j|^3}$ between coil pairs (dimensionless)",
    # Total superconductor length
    "total_superconductor_length_km": r"Total superconductor length $L_{\text{SC}} = \frac{1}{1000} \sum_i N_{\text{turns},i} \times L_{\text{reactor},i}$ (km)",
    # Arclength variation
    "final_arclength_variation": r"Variance of incremental arclength $J = \text{Var}(l_i)$ where $l_i$ is the average incremental arclength on interval $I_i$ from a partition $\{I_i\}_{i=1}^L$ of $[0,1]$ ($\text{m}^2$)",
    "final_max_torsion": r"Maximum torsion $\zeta_\text{max}$ across all coils ($\text{m}^{-1}$)",
    # Coil parameters
    "coil_order": r"Fourier order $n$ of coil representation: $\mathbf{r}(\phi) = \mathbf{a}_0 + \sum_{m=1}^{n} \left[\mathbf{a}_m \cos(m\phi) + \mathbf{b}_m \sin(m\phi)\right]$ (dimensionless)",
    "num_coils": r"Number of base coils $N$ (before applying stellarator symmetry) (dimensionless)",
    # Fourier continuation
    "fourier_continuation_orders": r"**Fourier continuation (FC)**: Sequence of Fourier orders used in continuation method. The optimization starts with a low-order representation, converges, then extends the solution to higher orders using the previous solution as initial condition. This helps achieve convergence for complex problems. Format: comma-separated list of orders (e.g., \"4,6,8\" means optimization was performed at orders 4, 6, and 8 sequentially). If not used, the column shows \"—\".",
    # Quasisymmetry
    "quasisymmetry_average": r"Average two-term quasisymmetry error $\text{avg}(QS)$ computed from VMEC equilibrium. The two-term quasisymmetry error measures how well the magnetic field strength $|\mathbf{B}|$ is constant on flux surfaces by evaluating the ratio residual $QS = \frac{|\mathbf{B}|_{m,n}}{|\mathbf{B}|}$ where $(m,n)$ is the target helicity. Lower values indicate better quasisymmetry (dimensionless).",
    # Fast Particle Tracing (SIMPLE)
    "loss_fraction": r"Final particle loss fraction from SIMPLE fast particle tracing. The loss fraction is computed as $1 - f_c$ where $f_c$ is the confined fraction (sum of confined passing and trapped particles). Lower values indicate better particle confinement (dimensionless).",
}

# Detailed structured definitions (metric key -> dict with title, symbol, description, etc.)
METRIC_DETAILED_DEFINITIONS: dict[str, dict[str, Any]] = {
    "final_normalized_squared_flux": {
        "title": "Normalized Squared Flux Error",
        "symbol": r":math:`f_B`",
        "description": "Measures the quality of the magnetic field on the plasma surface by quantifying how well the normal component of the magnetic field vanishes.",
        "math_forms": [
            r"f_B = \frac{1}{|S|} \int_{S} \left(\frac{\mathbf{B} \cdot \mathbf{n}}{|\mathbf{B}|}\right)^2 ds"
        ],
        "where": r"where :math:`|S|` is the total surface area of the plasma surface :math:`S`.",
        "units": "dimensionless",
        "notes": "Lower values indicate better field quality (closer to zero normal field component).",
    },
    "avg_BdotN_over_B": {
        "title": "Average Normalized Normal Field Component",
        "symbol": r":math:`\bar{B}_n`",
        "description": "Average of the absolute value of the normalized normal field component across the plasma surface.",
        "math_forms": [
            r"B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}",
            r"\bar{B}_n = \frac{\int_{S} |\mathbf{B} \cdot \mathbf{n}| ds}{\int_{S} |\mathbf{B}| ds}",
        ],
        "units": "dimensionless",
        "notes": "Lower values indicate better field quality.",
    },
    "max_BdotN_over_B": {
        "title": "Maximum Normalized Normal Field Component",
        "symbol": r":math:`\max(B_n)`",
        "description": "Maximum value of the normalized normal field component across the plasma surface.",
        "math_forms": [
            r"B_n = \frac{|\mathbf{B} \cdot \mathbf{n}|}{|\mathbf{B}|}",
            r"\max(B_n) = \max_{\mathbf{s} \in S} B_n(\mathbf{s})",
        ],
        "units": "dimensionless",
        "notes": "Lower values indicate better field quality.",
    },
    "avg_BdotN_over_target_B": {
        "title": "Average Normal Field Residual Over Target Field",
        "symbol": r":math:`\bar{B}_{n,0}`",
        "description": "Average absolute normal-field residual normalized by the reference target field magnitude.",
        "math_forms": [
            r"\bar{B}_{n,0} = \frac{\langle |\mathbf{B}_\text{coil} \cdot \mathbf{n} - B_\text{target}| \rangle}{B_0}",
        ],
        "where": r"where :math:`B_0` is target_B and :math:`B_\text{target}` is the virtual-casing normal field target when enabled.",
        "units": "dimensionless",
        "notes": "Useful for comparing field residuals at a fixed device reference field.",
    },
    "max_BdotN_over_target_B": {
        "title": "Maximum Normal Field Residual Over Target Field",
        "symbol": r":math:`\max(B_{n,0})`",
        "description": "Maximum absolute normal-field residual normalized by the reference target field magnitude.",
        "math_forms": [
            r"\max(B_{n,0}) = \max_{\mathbf{s} \in S} \frac{|\mathbf{B}_\text{coil} \cdot \mathbf{n} - B_\text{target}|}{B_0}",
        ],
        "where": r"where :math:`B_0` is target_B and :math:`B_\text{target}` is the virtual-casing normal field target when enabled.",
        "units": "dimensionless",
        "notes": "Lower values indicate better field quality at the device reference field.",
    },
    "coil_order": {
        "title": "Fourier Order",
        "symbol": r":math:`n`",
        "description": "Order of the Fourier series representation used for coil curves.",
        "math_forms": [
            r"\mathbf{r}(\phi) = \mathbf{a}_0 + \sum_{m=1}^{n} \left[\mathbf{a}_m \cos(m\phi) + \mathbf{b}_m \sin(m\phi)\right]"
        ],
        "where": r"where :math:`\mathbf{a}_0`, :math:`\mathbf{a}_m`, and :math:`\mathbf{b}_m` are Fourier coefficients and :math:`\phi` is the parameterization angle.",
        "units": "dimensionless",
        "notes": "Higher orders allow more complex coil shapes but increase the number of optimization variables.",
    },
    "num_coils": {
        "title": "Number of Base Coils",
        "symbol": r":math:`N`",
        "description": "Number of base coils before applying stellarator symmetry.",
        "units": "dimensionless",
        "notes": "Typical values: 4, 6, 8, 12. More coils allow more complex field shaping but increase computational cost.",
    },
    "final_total_length": {
        "title": "Total Length",
        "symbol": r":math:`L`",
        "description": "Total length of all coils.",
        "math_forms": [r"L = \sum_{i=1}^{N} \int_{C_i} d\ell_i"],
        "units": r":math:`\text{m}`",
        "notes": "Shorter coils are generally preferred for reduced material costs and improved manufacturability.",
    },
    "final_average_curvature": {
        "title": "Mean Curvature",
        "symbol": r":math:`\bar{\kappa}`",
        "description": "Average curvature across all coils.",
        "math_forms": [
            r"\kappa_i(\ell_i) = \left|\mathbf{r}_i''(\ell_i)\right|",
            r"\bar{\kappa} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{L_i} \int_{C_i} \kappa_i(\ell_i) ~d\ell_i",
        ],
        "where": r"where :math:`\mathbf{r}_i(\ell_i)` is the parameterization of coil curve :math:`C_i` by arclength.",
        "units": r":math:`\text{m}^{-1}`",
        "notes": "Lower curvature values indicate smoother coils that are easier to manufacture.",
    },
    "final_max_curvature": {
        "title": "Maximum Curvature",
        "symbol": r":math:`\kappa_\text{max}`",
        "description": "Maximum curvature value across all coils.",
        "math_forms": [
            r"\kappa_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \kappa_i(\ell_i)"
        ],
        "units": r":math:`\text{m}^{-1}`",
        "notes": "Lower values indicate coils without extreme curvature regions.",
    },
    "final_mean_squared_curvature": {
        "title": "Mean Squared Curvature",
        "symbol": r":math:`\text{MSC}`",
        "description": "Mean squared curvature per coil, averaged across all coils.",
        "math_forms": [
            r"J = \frac{1}{L_i} \int_{C_i} \kappa_i^2(\ell_i) ~d\ell_i",
            r"\text{MSC} = \frac{1}{N} \sum_{i=1}^{N} J_i",
        ],
        "where": r"where :math:`L_i` is the total length of coil curve :math:`C_i`, :math:`\ell_i` is the arclength along the curve, and :math:`\kappa_i` is the curvature.",
        "units": r":math:`\text{m}^{-2}`",
        "notes": "This provides a smoother penalty than maximum curvature, encouraging overall smoothness rather than just avoiding extreme values.",
    },
    "final_arclength_variation": {
        "title": "Arclength Variation",
        "symbol": r":math:`J`",
        "description": "Variance of incremental arclength between coil segments.",
        "math_forms": [r"J = \text{Var}(l_i)"],
        "where": r"where :math:`l_i` is the average incremental arclength on interval :math:`I_i` from a partition :math:`\{I_i\}_{i=1}^L` of :math:`[0,1]`.",
        "units": r":math:`\text{m}^2`",
        "notes": "Lower values indicate more uniform spacing along coils, which is important for manufacturing and field quality.",
    },
    "final_min_cc_separation": {
        "title": "Minimum Coil-to-Coil Distance",
        "symbol": r":math:`d_{cc}`",
        "description": "Minimum distance between any two coils.",
        "math_forms": [
            r"d_{cc} = \min_{i \neq j} \min_{\mathbf{r}_i \in C_i, \mathbf{r}_j \in C_j} \left\| \mathbf{r}_i - \mathbf{r}_j \right\|_2"
        ],
        "units": r":math:`\text{m}`",
        "notes": "Ensures coils maintain a safe separation distance to prevent collisions.",
    },
    "final_min_cs_separation": {
        "title": "Minimum Coil-to-Surface Distance",
        "symbol": r":math:`d_{cs}`",
        "description": "Minimum distance between any coil and the plasma surface.",
        "math_forms": [
            r"d_{cs} = \min_{i} \min_{\mathbf{r}_i \in C_i, \mathbf{s} \in S} \left\| \mathbf{r}_i - \mathbf{s} \right\|_2"
        ],
        "units": r":math:`\text{m}`",
        "notes": "Ensures coils maintain a safe distance from the plasma surface.",
    },
    "final_avg_max_coil_force": {
        "title": "Average of Maximum Force",
        "symbol": r":math:`\bar{F}`",
        "description": "Average across coils of the maximum force magnitude per coil.",
        "math_forms": [
            r"\bar{F} = \frac{1}{N} \sum_{i=1}^{N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{F}_i}{d\ell_i}\right|"
        ],
        "where": r"where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length on coil curve :math:`C_i`.",
        "units": r":math:`\text{N}/\text{m}`",
        "notes": "Lower values indicate coils that are easier to support mechanically.",
    },
    "final_max_max_coil_force": {
        "title": "Maximum Force Magnitude",
        "symbol": r":math:`F_\text{max}`",
        "description": "Maximum force magnitude across all coils.",
        "math_forms": [
            r"F_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{F}_i}{d\ell_i}\right|"
        ],
        "units": r":math:`\text{N}/\text{m}`",
        "notes": "High forces indicate coils that may be difficult to support mechanically.",
    },
    "final_avg_max_coil_torque": {
        "title": "Average of Maximum Torque",
        "symbol": r":math:`\bar{\tau}`",
        "description": "Average across coils of the maximum torque magnitude per coil.",
        "math_forms": [
            r"\bar{\tau} = \frac{1}{N} \sum_{i=1}^{N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{T}_i}{d\ell_i}\right|"
        ],
        "where": r"where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length on coil curve :math:`C_i`.",
        "units": r":math:`\text{N}`",
        "notes": "Lower values indicate coils with reduced rotational forces that must be resisted by supports.",
    },
    "final_max_max_coil_torque": {
        "title": "Maximum Torque Magnitude",
        "symbol": r":math:`\tau_\text{max}`",
        "description": "Maximum torque magnitude across all coils.",
        "math_forms": [
            r"\tau_\text{max} = \max_{i=1,\ldots,N} \max_{\ell_i \in [0,L_i]} \left|\frac{d\vec{T}_i}{d\ell_i}\right|"
        ],
        "units": r":math:`\text{N}`",
        "notes": "High torques can lead to mechanical instability.",
    },
    "final_linking_number": {
        "title": "Linking Number",
        "symbol": r":math:`\text{LN}`",
        "description": "Topological measure of how coils are linked together.",
        "math_forms": [
            r"\text{LN} = \frac{1}{4\pi} \sum_{i \neq j} \oint_{C_i} \oint_{C_j} \frac{\left(\mathbf{r}_i - \mathbf{r}_j\right) \cdot \left(d\mathbf{r}_i \times d\mathbf{r}_j\right)}{\left|\mathbf{r}_i - \mathbf{r}_j\right|^3}"
        ],
        "units": "dimensionless",
        "notes": "This metric ensures coils maintain their topological structure during optimization.",
    },
    "optimization_time": {
        "title": "Total Optimization Time",
        "symbol": r":math:`t`",
        "description": "Total time required to complete the optimization.",
        "units": r":math:`\text{s}`",
        "notes": "Lower values indicate more efficient optimization algorithms or faster convergence.",
    },
    "quasisymmetry_average": {
        "title": "Average Quasisymmetry Error",
        "symbol": r":math:`\text{avg}(QS)`",
        "description": "Average two-term quasisymmetry error computed from VMEC equilibrium.",
        "math_forms": [r"QS = \frac{|\mathbf{B}|_{m,n}}{|\mathbf{B}|}"],
        "where": r"The two-term quasisymmetry error measures how well the magnetic field strength :math:`|\mathbf{B}|` is constant on flux surfaces by evaluating the ratio residual where :math:`(m,n)` is the target helicity.",
        "units": "dimensionless",
        "notes": "Lower values indicate better quasisymmetry, which is important for particle confinement in stellarators.",
    },
    "loss_fraction": {
        "title": "Loss Fraction",
        "symbol": r":math:`\text{LF}`",
        "description": "Final particle loss fraction from SIMPLE fast particle tracing.",
        "math_forms": [r"\text{LF} = 1 - f_c"],
        "where": r"where :math:`f_c` is the confined fraction (sum of confined passing and trapped particles).",
        "units": "dimensionless",
        "notes": "Lower values indicate better particle confinement. A value of 0 means all particles are confined, while a value of 1 means all particles are lost. This metric is computed by the SIMPLE code using Monte Carlo particle tracing.",
    },
    "fourier_continuation_orders": {
        "title": "Fourier Continuation (FC)",
        "description": "Sequence of Fourier orders used in continuation method. The optimization starts with a low-order representation, converges, then extends the solution to higher orders using the previous solution as initial condition. This helps achieve convergence for complex problems.",
        "notes": 'Format: comma-separated list of orders (e.g., "4,6,8" means optimization was performed at orders 4, 6, and 8 sequentially). If not used, the column shows "—".',
    },
    "total_superconductor_length_km": {
        "title": "Total Superconductor Length",
        "symbol": r":math:`L_{\text{SC}}`",
        "description": "Total length of superconducting tape required at reactor scale, accounting for the number of turns in each coil's winding pack.",
        "math_forms": [
            r"L_{\text{SC}} = \frac{1}{1000} \sum_{i=1}^{N_{\text{coils}}} N_{\text{turns},i} \times L_{\text{reactor},i}"
        ],
        "where": r"where :math:`N_{\text{turns},i} = \max(N_{F,i},\, N_{J_c,i})` is the number of turns per coil (driven by force limits or REBCO critical-current limits, whichever is larger), and :math:`L_{\text{reactor},i}` is the reactor-scale length of coil :math:`i`. The factor of 1/1000 converts from meters to kilometers.",
        "units": r":math:`\text{km}`",
        "notes": "Lower values indicate more economical coil designs requiring less superconducting material. This is a derived reactor-scale metric that combines the winding-pack turn count with the scaled coil lengths.",
    },
}
