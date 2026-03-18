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
