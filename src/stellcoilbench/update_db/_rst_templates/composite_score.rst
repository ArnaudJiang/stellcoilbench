Composite Score
---------------

The **Score** column is a single-number summary of reactor-scale
engineering feasibility.  It is computed in two stages.

Stage 1: Hard Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

Hard constraints are evaluated first.  If **any** hard constraint is
violated the score is set to **0** and the entry is marked **FAIL**
(excluded from the main leaderboard).  Missing metrics are skipped.

Two hard constraints apply a transform before comparison:

- **Coil-coil linking number** — compared as :math:`|\text{LN}|`
  (absolute value), since the linking number may be negative.
- **Max turns per coil** — the element-wise maximum of the per-coil
  turn-count list is compared against the bound.

Stage 2: Soft Constraint Geometric Mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each soft constraint whose metric is available (and whose bound is
non-zero), a margin exponent :math:`m_i` is computed:

.. math::

   m_i = \begin{cases}
     1 - \text{value}_i\,/\,\text{bound}_i
       & \text{value} \leq \text{bound} \;(\text{upper-bound constraint})\\[6pt]
     \text{value}_i\,/\,\text{bound}_i - 1
       & \text{value} \geq \text{bound} \;(\text{lower-bound constraint})
   \end{cases}

Positive :math:`m_i` = constraint satisfied with margin; negative =
violated.

Two soft constraints apply a transform to the raw metric before the
margin calculation:

- **RMS curvature** — :math:`\sqrt{\text{MSC}}` is used (square root
  of the mean-squared curvature) so the comparison is in m\ :sup:`-1`.
- **Arclength variation** — :math:`\sqrt{\text{Var}}` is used (square
  root of the variance) so the comparison is in m.

The composite score is the geometric mean of the individual exponential
factors:

.. math::

   \text{Score}     = \exp\!\left(\frac{1}{n}\sum_{i=1}^{n} m_i\right)     = \left(\prod_{i=1}^{n} e^{m_i}\right)^{\!1/n}

where :math:`n` is the number of soft constraints for which metrics
were available.

Soft Constraints and Margin Formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table lists every soft constraint that enters the
composite score, together with its reactor-scale bound, the source
metric key, and the resulting margin formula.

.. list-table::
   :header-rows: 1
   :widths: 28 12 13 47

   * - Constraint
     - Direction
     - Bound
     - Margin :math:`m_i`
{{SOFT_CONSTRAINT_TABLE}}

Interpretation
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Score = 0**
     - Hard infeasibility (coils delinked from plasma, coils interlinked,
       winding packs overlap, or too many turns required).
   * - **0 < Score < 1**
     - Feasible but one or more soft constraints violated on average.
   * - **Score = 1**
     - All soft constraints met exactly on average.
   * - **Score > 1**
     - All constraints satisfied with engineering margin.
   * - **Score = None**
     - No soft-constraint metrics available (missing data).

Entries are sorted by composite score **descending** (higher is better).
Entries without a composite score fall back to
``score_primary`` (lower squared flux is better) and appear after
scored entries.

Worked Example
~~~~~~~~~~~~~~

Consider a reactor-scale design with:

- avg :math:`B_n/|B|` = 0.005, :math:`d_{cs}` = 1.5 m,
  :math:`d_{cc}` = 0.8 m, :math:`L` = 200 m,
  :math:`\kappa_{\max}` = 0.7, :math:`\sqrt{\text{MSC}}` = 0.6,
  :math:`\sqrt{\text{Var}}` = 0.4, :math:`L_{\text{SC}}` = 50 km

.. math::

   m_1 &= 1 - 0.005/0.01   &&= 0.500 \\
   m_2 &= 1.5/1.3 - 1      &&\approx 0.154 \\
   m_3 &= 0.8/0.7 - 1      &&\approx 0.143 \\
   m_4 &= 1 - 200/220       &&\approx 0.091 \\
   m_5 &= 1 - 0.7/1.0       &&= 0.300 \\
   m_6 &= 1 - 0.6/1.0       &&= 0.400 \\
   m_7 &= 1 - 0.4/1.0       &&= 0.600 \\
   m_8 &= 1 - 50/100         &&= 0.500

Mean margin :math:`= (0.500 + 0.154 + 0.143 + 0.091 + 0.300 + 0.400 + 0.600 + 0.500)\,/\,8 \approx 0.336`.

Score :math:`= \exp(0.336) \approx` **1.399** — all constraints
satisfied with comfortable engineering margin.
