
Winding-Pack Turn-Count Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *simsopt* optimiser models each coil as a single filamentary turn
carrying the total current :math:`I`.  In a real reactor the winding pack
contains :math:`N_{\text{turns}}` turns, each carrying :math:`I / N_{\text{turns}}`.
We estimate the required number of turns from **two independent criteria**
and take the element-wise maximum:

.. math::

   N_{\text{turns},\,i}     = \max\!\bigl(N_{\text{turns},\,i}^{(\text{force})},\;                    N_{\text{turns},\,i}^{(J_c)}\bigr)

1. Force-based turns
^^^^^^^^^^^^^^^^^^^^

With :math:`N` turns the Lorentz force per unit length on each turn is

.. math::

   F_{\text{turn}} = \frac{F_{\text{reactor,single-turn}}}{N_{\text{turns}}}

For each coil we find the minimum :math:`N` to keep
:math:`F_{\text{turn}} \leq 0.5\,\text{MN/m}`:

.. math::

   N_{\text{turns},\,i}^{(\text{force})}     = \left\lceil \frac{F_{\text{reactor},\,i}}{0.5\;\text{MN/m}} \right\rceil

2. Critical-current-density-based turns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This criterion ensures the HTS superconductor operates within its critical
envelope.  The model follows the Stellaris winding-pack design
(Lion *et al.*, *Fusion Engineering and Design* **214**, 2025, 114868,
Table 7–8 and Section 2.9).

**REBCO tape-stack** :math:`J_c` **model.**
A Kim-like parametrisation calibrated to tape-stack data at 20 K
(field-aligned tapes):

.. math::

   J_c(B, T) = \frac{C_0}{1 + (B/B_0)^\alpha}     \;\times\;\frac{1 - T/T_c}{1 - T_{\text{ref}}/T_c}

with fitted constants at :math:`T_{\text{ref}} = 20\,\text{K}`:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - :math:`C_0`
     - :math:`5.0 \times 10^{9}\;\text{A/m}^2`
     - Zero-field engineering :math:`J_c` (≈ 5000 A/mm²)
   * - :math:`B_0`
     - 18.14 T
     - Characteristic field
   * - :math:`\alpha`
     - 0.902
     - Field exponent
   * - :math:`T_c`
     - 92 K
     - REBCO critical temperature

Validation against Stellaris Table 8:
:math:`B = 20\,\text{T} \rightarrow J_c \approx 2450\;\text{A/mm}^2`,
:math:`B = 25\,\text{T} \rightarrow J_c \approx 2200\;\text{A/mm}^2`.

**Stellaris winding-pack parameters.**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - :math:`T_{\text{op}}`
     - 20 K
     - Operating temperature
   * - :math:`\eta`
     - 0.80
     - Utilisation cap (:math:`J_{\text{op}} / J_c \leq \eta`)
   * - :math:`I_{\text{lead,max}}`
     - 50 kA
     - Current-lead limit
   * - :math:`A_{\text{HTS}}`
     - :math:`36\;\text{mm}^2` (6 mm × 6 mm)
     - HTS tape-stack cross-section per turn
   * - :math:`A_{\text{turn}}`
     - :math:`400\;\text{mm}^2` (20 mm × 20 mm)
     - Total turn cross-section (incl. stabiliser, insulation, structure)
   * - :math:`f_{\text{WP}}`
     - 1.3
     - Winding-pack self-field enhancement factor

**Algorithm for each coil** :math:`i`:

1. **Required ampere-turns** at reactor scale:

   .. math::

      NI_i = I_{\text{device},i} \times B_{\text{scale}} \times L_{\text{scale}}

   where :math:`I_{\text{device},i}` is the *simsopt* single-turn current.
   If per-coil currents are unavailable, :math:`I` is estimated from
   the force data: :math:`I \approx (F/L) / B_{\text{device}}`.

2. **Peak conductor field** estimate:

   .. math::

      B_{\text{ext},i} = \frac{(F/L)_{\text{device},i}}{I_{\text{device},i}} \times B_{\text{scale}},      \qquad      B_{\text{peak},i} = f_{\text{WP}} \times B_{\text{ext},i}

   The factor :math:`f_{\text{WP}} = 1.3` accounts for the additional
   self-field produced by the multi-turn winding pack at its inner edge.

3. **Critical current of the HTS cable**:

   .. math::

      I_{c,\text{cable}} = J_c(B_{\text{peak}},\; T_{\text{op}}) \times A_{\text{HTS}}

4. **Operating current per turn** (lead- or tape-limited):

   .. math::

      I_{\text{turn}} = \min\!\bigl(I_{\text{lead,max}},\; \eta \times I_{c,\text{cable}}\bigr)

5. **Number of turns** from :math:`J_c` requirements:

   .. math::

      N_{\text{turns},\,i}^{(J_c)} = \left\lceil \frac{NI_i}{I_{\text{turn},i}} \right\rceil

**Hard constraint.**
The final :math:`N_{\text{turns},i}` (element-wise maximum of force and
:math:`J_c` requirements) must satisfy
:math:`\max_i N_{\text{turns},\,i} \leq {{N_TURNS_MODEL}}`.

3. Finite-build (winding-pack) extent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With each turn occupying :math:`A_{\text{turn}} = 20\;\text{mm} \times
20\;\text{mm} = 400\;\text{mm}^2` (Table 7 of Lion *et al.*: this area
includes the REBCO tape stack, copper stabiliser, solder, steel jacket,
and helium cooling channel), a square winding pack with :math:`N` turns
has side length

.. math::

   w_{\text{WP}} = \sqrt{N_{\text{turns}}} \times 20\;\text{mm}

Validation against Stellaris Table 8:

- Coil 0: :math:`N = 324 \;\Rightarrow\; w = 18 \times 20\;\text{mm} = 360\;\text{mm} \;\checkmark`
- Coil 5: :math:`N = 225 \;\Rightarrow\; w = 15 \times 20\;\text{mm} = 300\;\text{mm} \;\checkmark`

The leaderboard reports :math:`w_{\text{WP}}` — the **maximum** winding-pack
side length across all coils (in metres).  This gives the finite-build extent
that must be accommodated by the coil-surface and coil-coil separation gaps.

4. Finite-build coil-coil intersection check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*simsopt*'s ``CurveCurveDistance`` penalty measures the **centreline-to-centreline**
distance between coil filaments.  Once the winding-pack extent is known, we
can check whether the finite-build coils would physically overlap.

Each coil's winding pack extends :math:`w_i / 2` from the centreline in every
direction.  For two coils *i* and *j* separated by centreline distance
:math:`d_{ij}`, the clearance between their outer edges is

.. math::

   \text{clearance}_{ij} = d_{ij} - \frac{w_i}{2} - \frac{w_j}{2}

Because we only store the **global minimum** coil-coil distance
:math:`d_{\text{cc,min}} = \min_{i<j} d_{ij}`, the most conservative
check uses the largest winding-pack width for both coils:

.. math::

   \text{clearance} = d_{\text{cc,min}} - w_{\text{WP,max}}

where :math:`w_{\text{WP,max}} = \max_i w_{\text{WP},i}`.  This is a **hard
constraint**: if the clearance is negative (:math:`d_{\text{cc,min}} <
w_{\text{WP,max}}`), the winding packs would intersect and the design is
infeasible (score = 0).

5. Per-turn force and torque
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once :math:`N_{\text{turns},i}` is known for each coil, the engineering-relevant
structural loads are the per-turn quantities:

.. math::

   F_{\text{turn},i} = \frac{F_{\text{reactor},i}}{N_{\text{turns},i}},   \qquad   \tau_{\text{turn},i} = \frac{\tau_{\text{reactor},i}}{N_{\text{turns},i}}

The leaderboard reports:

- :math:`F_{\text{turn}}` — :math:`\max_i F_{\text{turn},i}` (MN/m), the
  maximum per-turn force across all coils.
- :math:`\tau_{\text{turn}}` — :math:`\max_i \tau_{\text{turn},i}` (MN), the
  maximum per-turn torque across all coils.

These replace the single-turn :math:`F_{\max}` and :math:`\tau_{\max}` in the
reactor-scale leaderboard, since the single-turn values are not physically
meaningful for a multi-turn winding pack.
