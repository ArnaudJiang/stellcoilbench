Cases
=====

Case files (YAML) define the optimization problem: plasma surface, coil configuration, optimizer, and objective terms.

**surface_params**
   - ``surface``: Name from ``plasma_surfaces/`` (e.g. ``input.LandremanPaul2021_QA``)
   - ``range``: ``"half period"`` or ``"full torus"``
   - ``virtual_casing``: Optional bool. Set ``true`` to use virtual-casing B-field from a VMEC wout file. Requires the ``virtual_casing`` Python package.

**coils_params**
   - ``ncoils``: Number of coils (4, 6, 8, …)
   - ``order``: Fourier order (4, 8, 16, …)

**optimizer_params** (keys optional; section required)
   Individual keys may be omitted; defaults: ``algorithm`` = ``"augmented_lagrangian"``, ``max_iterations`` = 30, ``max_iter_subopt`` = 10.
   - ``algorithm``: ``"augmented_lagrangian"`` (recommended; auto-tunes weights) or ``"L-BFGS-B"`` (weights set to reactor defaults, scaled by plasma surface minor radius)
   - ``max_iterations``: e.g. 200–1000
   - ``max_iter_subopt``: For augmented Lagrangian (e.g. 10–40)

**coil_objective_terms** (optional)
   If omitted, defaults to ``total_length`` (l2_threshold), ``coil_curvature`` (lp_threshold), ``coil_mean_squared_curvature`` (l2_threshold), ``linking_number`` (""), ``coil_arclength_variation`` (l2_threshold).
   Each term maps to a penalty type: ``l1``, ``l2``, ``lp``, ``l1_threshold``, ``l2_threshold``, ``lp_threshold``, or ``""``.
   Common terms: ``total_length``, ``coil_curvature``, ``coil_mean_squared_curvature``, ``coil_arclength_variation``, ``linking_number``, ``coil_coil_force``, ``coil_coil_torque``.
   ``coil_coil_distance`` and ``coil_surface_distance`` are always included; use ``cc_threshold`` and ``cs_threshold``.

**fourier_continuation** (optional)
   Progressive refinement by order: ``enabled: true``, ``orders: [4, 8, 16]``.

**post_processing_params** (optional)
   Flags for submission post-processing: ``run_vmec`` (VMEC equilibrium), ``run_simple`` (SIMPLE particle tracing), ``plot_poincare``, ``plot_boozer``, ``plot_finite_build``. Defaults: VMEC off, Poincaré off, Boozer on, finite-build off.

.. code-block:: yaml

   description: "Basic Landreman-Paul QA case"
   surface_params:
     surface: "input.LandremanPaul2021_QA"
   coils_params:
     ncoils: 4
     order: 4
   optimizer_params:
     max_iterations: 200
   coil_objective_terms:
     coil_coil_force: "lp_threshold"
     coil_coil_torque: "lp_threshold"
   # post_processing_params:  # optional; add to enable VMEC, Poincaré, etc.
   #   run_vmec: true
   #   plot_poincare: true

See :doc:`leaderboard/metric_definitions` for metric notation and :doc:`api` for full schema details. Run ``stellcoilbench update-db`` to refresh leaderboards after adding submissions.
