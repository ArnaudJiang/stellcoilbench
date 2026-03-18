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
{{SOFT_CONSTRAINT_TABLE}}

**Interpretation:** Score = 0 → hard infeasible; 0 < Score < 1 → soft violated; Score ≥ 1 → constraints met. Entries sorted by score descending.
