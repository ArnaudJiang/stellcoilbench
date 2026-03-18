Metric Definitions
===================

The following metrics are used to evaluate coil optimization submissions. Notation: :math:`C_i` coil curve, :math:`S` plasma surface, :math:`\mathbf{r}_i` point on coil, :math:`\kappa_i` curvature, :math:`N` number of coils, :math:`\mathbf{B}` magnetic field, :math:`\mathbf{n}` surface normal.

Definitions
-----------

- :math:`\bar{B}_n`: Average of the absolute value of the normalized normal field component across the plasma surface. (dimensionless)
- :math:`\max(B_n)`: Maximum value of the normalized normal field component across the plasma surface. (dimensionless)
- :math:`J`: Variance of incremental arclength between coil segments. (:math:`\text{m}^2`)
- :math:`\kappa_\text{max}`: Maximum curvature value across all coils. (:math:`\text{m}^{-1}`)
- :math:`\text{MSC}`: Mean squared curvature per coil, averaged across all coils. (:math:`\text{m}^{-2}`)
- :math:`L`: Total length of all coils. (:math:`\text{m}`)
- **FC**: Sequence of Fourier orders used in continuation method. The optimization starts with a low-order representation,...
- :math:`N`: Number of base coils before applying stellarator symmetry. (dimensionless)
- :math:`L_{\text{SC}}`: Total length of superconducting tape required at reactor scale, accounting for the number of turns in each coil's... (:math:`\text{km}`)
- :math:`d_{cc}`: Minimum distance between any two coils. (:math:`\text{m}`)
- :math:`d_{cs}`: Minimum distance between any coil and the plasma surface. (:math:`\text{m}`)
- :math:`F_\text{max}`: Maximum force magnitude across all coils. (:math:`\text{N}/\text{m}`)
- :math:`\tau_\text{max}`: Maximum torque magnitude across all coils. (:math:`\text{N}`)
- :math:`\text{LN}`: Topological measure of how coils are linked together. (dimensionless)
- :math:`t`: Total time required to complete the optimization. (:math:`\text{s}`)

Composite Score
---------------

The **Score** column summarises reactor-scale engineering feasibility.

**Hard constraints** (any violation → score = 0): coil-surface linkage, coil-coil linking, finite-build clearance. **Soft constraints** contribute via margin exponents :math:`m_i`:

.. math::

   m_i = \begin{cases}
     1 - \text{value}_i\,/\,\text{bound}_i
       & \text{value} \leq \text{bound}\\[6pt]
     \text{value}_i\,/\,\text{bound}_i - 1
       & \text{value} \geq \text{bound}
   \end{cases}

Score :math:`= \exp\!\left(\frac{1}{n}\sum_{i=1}^{n} m_i\right)`. RMS curvature uses :math:`\sqrt{\text{MSC}}`; arclength variation uses :math:`\sqrt{\text{Var}}`.

.. list-table::
   :header-rows: 1
   :widths: 28 12 13 47

   * - Constraint
     - Direction
     - Bound
     - Margin :math:`m_i`
   * - avg :math:`\langle B{\cdot}n\rangle / \langle B\rangle` (``avg_BdotN_over_B``)
     - :math:`\leq`
     - 0.01
     - :math:`1 - \text{value}\;/\;0.01`
   * - Min coil-surface distance (``reactor_scale_min_cs_separation``)
     - :math:`\geq`
     - 1.3 m
     - :math:`\text{value}\;/\;1.3 - 1`
   * - Min coil-coil distance (``reactor_scale_min_cc_separation``)
     - :math:`\geq`
     - 0.7 m
     - :math:`\text{value}\;/\;0.7 - 1`
   * - Total coil length (``reactor_scale_total_length``)
     - :math:`\leq`
     - 220 m
     - :math:`1 - L\;/\;220`
   * - Max curvature :math:`\kappa` (``reactor_scale_max_curvature``)
     - :math:`\leq`
     - 1.0 m\ :sup:`-1`
     - :math:`1 - \kappa_{\max}\;/\;1.0`
   * - RMS curvature :math:`\sqrt{\text{MSC}}` (``reactor_scale_mean_squared_curvature``)
     - :math:`\leq`
     - 1.0 m\ :sup:`-1`
     - :math:`1 - \sqrt{\text{MSC}}\;/\;1.0`
   * - Arclength variation :math:`\sqrt{\text{Var}}` (``reactor_scale_arclength_variation``)
     - :math:`\leq`
     - 1.0 m
     - :math:`1 - \sqrt{\text{Var}}\;/\;1.0`
   * - Total superconductor length :math:`L_{\text{SC}}` (``total_superconductor_length_km``)
     - :math:`\leq`
     - 100 km
     - :math:`1 - L_{\text{SC}}\;/\;100`
   * - Max turns per coil :math:`N_{\text{turns}}` (``N_turns_per_coil``)
     - :math:`\leq`
     - 300 (turns)
     - :math:`1 - \max_i N_{\text{turns},i}\;/\;300`

**Interpretation:** Score = 0 → hard infeasible; 0 < Score < 1 → soft violated; Score ≥ 1 → constraints met. Entries sorted by score descending.

Reactor-Scale Constraints
-------------------------

Scaled to ARIES-CS (:math:`a = 1.7\,\text{m}`, :math:`B_0 = 5.7\,\text{T}`). Hard constraints (any violation → score = 0):

.. list-table::
   :header-rows: 1

   * - Constraint
     - Bound
     - Description
   * - Coils linked to plasma surface
     - = True
     - Every base coil must topologically encircle the plasma.
   * - Coil-coil linking number (:math:`|\text{LN}| \approx 0`)
     - ≤ 0.5 (dimensionless)
     - Coils must not interlink with one another.
   * - Finite-build coil-coil clearance (:math:`d_{\text{cc}} > w_{\text{WP}}`)
     - ≥ 0.0 m
     - Centreline distance :math:`d_{\text{cc,min}}` must exceed the largest winding-pack width :math:`w_{\text{WP,max}}` to prevent physical overlap of finite-build coils.

**Soft constraints** — margin factors; violations lower score but do not set to 0:

.. list-table::
   :header-rows: 1

   * - Metric
     - Bound
     - Direction
     - Units
   * - avg :math:`\bar{B}_n`
     - :math:`\leq 0.01`
     - max
     - (dimensionless)
   * - Minimum coil-surface distance
     - :math:`\geq 1.3`
     - min
     - m
   * - Minimum coil-coil distance
     - :math:`\geq 0.7`
     - min
     - m
   * - Total coil length
     - :math:`\leq 220.0`
     - max
     - m
   * - Max curvature :math:`\kappa`
     - :math:`\leq 1.0`
     - max
     - m⁻¹
   * - Max :math:`\sqrt{\text{MSC}}` (RMS curvature)
     - :math:`\leq 1.0`
     - max
     - m⁻¹
   * - Arclength variation :math:`\sqrt{\text{Var}}`
     - :math:`\leq 1.0`
     - max
     - m
   * - Total superconductor length :math:`L_{\text{SC}}`
     - :math:`\leq 100.0`
     - max
     - km
   * - Max turns per coil :math:`N_{\text{turns}}`
     - :math:`\leq 300`
     - max
     - (turns)
Winding-Pack Model
~~~~~~~~~~~~~~~~~~

The optimiser models coils as single filamentary turns carrying total current
:math:`I`. A real reactor winding pack has :math:`N_{\text{turns}}` turns per coil,
each carrying :math:`I/N_{\text{turns}}`, to keep per-turn Lorentz forces and
conductor field within limits. For each coil :math:`i` we compute two turn counts
and take the maximum:

:math:`N_{\text{turns},\,i} = \max\bigl(N^{(\text{force})}_i,\, N^{(J_c)}_i\bigr)`

**Force-based turns.** With :math:`N` turns, the force per unit length on each
turn is :math:`F_{\text{turn}} = F_{\text{reactor}}/N`. To keep
:math:`F_{\text{turn}} \leq 0.5` MN/m (structural limit):

:math:`N^{(\text{force})}_i = \lceil F_{\text{reactor},i} / (0.5\,\text{MN/m}) \rceil`

**Jc-based turns.** Ensures the HTS operates within its critical current envelope.
Uses a Kim-like REBCO :math:`J_c(B,T)` model (Stellaris params, Lion *et al.* 2025).

1. **Required ampere-turns** at reactor scale: :math:`NI_i = I_{\text{device},i} \times B_{\text{scale}} \times L_{\text{scale}}`
2. **Peak conductor field** (with winding-pack self-field factor 1.3):
   :math:`B_{\text{peak},i} = f_{\text{WP}} \times B_{\text{ext},i}`
3. **Critical current** of cable: :math:`I_{c,\text{cable}} = J_c(B_{\text{peak}}, T_{\text{op}}) \times A_{\text{HTS}}`
4. **Operating current per turn**: :math:`I_{\text{turn}} = \min(I_{\text{lead,max}},\, \eta \times I_{c,\text{cable}})`
5. **Turns from Jc**: :math:`N^{(J_c)}_i = \lceil NI_i / I_{\text{turn},i} \rceil`

**Soft constraint (bound 300).** :math:`\max_i N_{\text{turns},i}` contributes to the
composite score via a margin: designs with :math:`\max_i N_{\text{turns},i} < 300` are
rewarded, designs with :math:`\max_i N_{\text{turns},i} > 300` are penalized.

**Finite-build width.** Each turn occupies :math:`20\,\text{mm} \times 20\,\text{mm}`.
Winding-pack side length: :math:`w_{\text{WP}} = \sqrt{N_{\text{turns}}} \times 20\,\text{mm}`.
Clearance between coil packs: :math:`d_{\text{cc,min}} - w_{\text{WP,max}}`. Negative = infeasible.

**Per-turn force/torque.** :math:`F_{\text{turn}} = F_{\text{reactor}}/N_{\text{turns}}`,
:math:`\tau_{\text{turn}} = \tau_{\text{reactor}}/N_{\text{turns}}`. Reported on leaderboard.
