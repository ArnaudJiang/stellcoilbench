"""SIMPLE input file generation.

Builds the ``simple.in`` Namelist content from parameters and VMEC file path.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

DEFAULT_NPOIPER2 = 256


def _fortran_bool(val: bool) -> str:
    """Format a Python bool as a Fortran logical literal."""
    return ".True." if val else ".False."


def _format_fortran_double(val: float) -> str:
    """Format a float as Fortran double precision (e.g. ``1.0d-1``)."""
    formatted = f"{float(val):.6e}"
    return formatted.replace("e", "d") if "e" in formatted else formatted + "d0"


def _build_simple_input_content(
    params: Dict[str, Any],
    provided_params: set[str],
    netcdffile: str,
) -> str:
    """Build the ``simple.in`` Namelist content string.

    Parameters
    ----------
    params : dict
        All SIMPLE parameters (keys are parameter names, values are their
        resolved values after defaults and auto-scaling).
    provided_params : set[str]
        Parameter names explicitly passed by the caller (used to decide
        which optional parameters to emit).
    netcdffile : str
        Absolute path to the VMEC NetCDF file.

    Returns
    -------
    str
        Complete ``simple.in`` file content.
    """
    sbeg = params["sbeg"]
    if isinstance(sbeg, (list, tuple, np.ndarray)):
        sbeg_str = ", ".join(_format_fortran_double(s) for s in sbeg)
    else:
        sbeg_str = _format_fortran_double(sbeg)

    lines: list[str] = ["&config"]

    lines.append(
        f"trace_time = {_format_fortran_double(params['trace_time'])}        ! slowing down time, s"
    )
    lines.append(
        f"sbeg = {sbeg_str}     ! starting s (normalized toroidal flux) for particles"
    )
    lines.append(
        f"ntestpart = {params['ntestpart']}          ! number of test particles"
    )
    lines.append(f"n_d = {params['n_d']}")
    lines.append(f"n_e = {params['n_e']}")
    lines.append(f"facE_al = {_format_fortran_double(params['facE_al'])}")
    lines.append(f"vmec_B_scale = {_format_fortran_double(params['vmec_B_scale'])}")
    lines.append(f"netcdffile = '{netcdffile}'   ! name of VMEC file in NETCDF format")
    lines.append(
        f"isw_field_type = {params['isw_field_type']}       ! -1: Testing, 0: Canonical, 1: VMEC, 2: Boozer, 3: Meiss, 4: Albert"
    )

    _optional: list[tuple[str, str, Any, str]] = [
        (
            "notrace_passing",
            "notrace_passing",
            0,
            "! skip tracing passing prts if notrace_passing=1",
        ),
        ("nper", "nper", 1000, "! number of periods for initial field line"),
        ("npoiper", "npoiper", 100, "! number of points per period on this field line"),
        ("ntimstep", "ntimstep", 10000, "! number of time steps per slowing down time"),
        (
            "num_surf",
            "num_surf",
            1,
            "! number of flux surfaces. Value 0, distributes in volume.",
        ),
    ]
    for key, pname, default, comment in _optional:
        if key in provided_params or params[pname] != default:
            lines.append(f"{pname} = {params[pname]}              {comment}")

    _optional_doubles: list[tuple[str, str, float, str]] = [
        ("phibeg", "phibeg", 0.0, "! starting phi for field line"),
        ("thetabeg", "thetabeg", 0.0, "! starting theta for field line"),
        ("contr_pp", "contr_pp", -1.0, "! control of passing particle fraction"),
    ]
    for key, pname, default, comment in _optional_doubles:
        if key in provided_params or params[pname] != default:
            lines.append(
                f"{pname} = {_format_fortran_double(params[pname])}            {comment}"
            )

    if "npoiper2" in provided_params or params["npoiper2"] != DEFAULT_NPOIPER2:
        lines.append(
            f"npoiper2 = {params['npoiper2']}\t         ! points per period for integrator step"
        )
    if "ns_s" in provided_params or params["ns_s"] != 5:
        lines.append(
            f"ns_s = {params['ns_s']}                 ! spline order for 3D quantities over s variable"
        )
    if "ns_tp" in provided_params or params["ns_tp"] != 5:
        lines.append(
            f"ns_tp = {params['ns_tp']}                ! spline order for 3D quantities over theta and phi"
        )
    if "multharm" in provided_params or params["multharm"] != 5:
        lines.append(
            f"multharm = {params['multharm']}             ! angular grid factor"
        )
    if "vmec_RZ_scale" in provided_params or params["vmec_RZ_scale"] != 1.0:
        lines.append(
            f"vmec_RZ_scale = {_format_fortran_double(params['vmec_RZ_scale'])}    ! factor to scale the device size from VMEC"
        )
    if params.get("generate_start_only"):
        lines.append(
            f"generate_start_only = {_fortran_bool(params['generate_start_only'])} ! If .True., only initialisation is done"
        )
    if "startmode" in provided_params or params["startmode"] != 1:
        lines.append(
            f"startmode = {params['startmode']}            ! mode for initial conditions"
        )
    if "grid_density" in provided_params or params["grid_density"] != 0.0:
        lines.append(
            f"grid_density = {_format_fortran_double(params['grid_density'])}       ! for startmode 1 only"
        )
    if params.get("special_ants_file"):
        lines.append(
            f"special_ants_file = {_fortran_bool(params['special_ants_file'])}"
        )
    lines.append(
        f"integmode = {params['integmode']}            ! mode for integrator: 0=RK, 1=Euler1, 2=Euler2, 3=Midpoint"
    )
    if "relerr" in provided_params or params["relerr"] != 1e-13:
        lines.append(
            f"relerr = {_format_fortran_double(params['relerr'])}           ! tolerance for integrator"
        )
    if "tcut" in provided_params or params["tcut"] != -1.0:
        lines.append(
            f"tcut = {_format_fortran_double(params['tcut'])}              ! time when to do cut for classification"
        )
    if params.get("debug"):
        lines.append(
            f"debug = {_fortran_bool(params['debug'])}          ! produce debugging output"
        )
    if params.get("class_plot"):
        lines.append(
            f"class_plot = {_fortran_bool(params['class_plot'])}     ! write starting points at phi=const cut"
        )
    if "cut_in_per" in provided_params or params["cut_in_per"] != 0.0:
        lines.append(
            f"cut_in_per = {_format_fortran_double(params['cut_in_per'])}         ! normalized phi-cut position"
        )
    if params.get("fast_class"):
        lines.append(
            f"fast_class = {_fortran_bool(params['fast_class'])}     ! quit immediately after fast classification"
        )
    if params.get("swcoll"):
        lines.append(
            f"swcoll = {_fortran_bool(params['swcoll'])}         ! enables collisions"
        )
    if params.get("deterministic"):
        lines.append(
            f"deterministic = {_fortran_bool(params['deterministic'])}  ! put seed for the same random walk"
        )
    if not params.get("old_axis_healing", True):
        lines.append(f"old_axis_healing = {_fortran_bool(params['old_axis_healing'])}")
    if not params.get("old_axis_healing_boundary", True):
        lines.append(
            f"old_axis_healing_boundary = {_fortran_bool(params['old_axis_healing_boundary'])}"
        )
    if "batch_size" in provided_params or params["batch_size"] != 2000000000:
        lines.append(
            f"batch_size = {params['batch_size']} ! Use only a portion of all particles"
        )
    if "ran_seed" in provided_params or params["ran_seed"] != 12345:
        lines.append(f"ran_seed = {params['ran_seed']}   ! Random seed")
    if params.get("reuse_batch"):
        lines.append(f"reuse_batch = {_fortran_bool(params['reuse_batch'])}")
    if params.get("output_orbits_macrostep"):
        lines.append(
            f"output_orbits_macrostep = {_fortran_bool(params['output_orbits_macrostep'])}"
        )
    if params.get("output_error"):
        lines.append(f"output_error = {_fortran_bool(params['output_error'])}")
    if (
        "macrostep_time_grid" in provided_params
        or params["macrostep_time_grid"] != "linear"
    ):
        lines.append(f"macrostep_time_grid = '{params['macrostep_time_grid']}'")

    if params.get("swcoll"):
        for key, comment in [
            ("am1", "atomic mass of the first background species"),
            ("am2", "atomic mass of the second background species"),
            ("Z1", "charge number of the first background species"),
            ("Z2", "charge number of the second background species"),
            ("densi1", "density of the first background species (cm^-3)"),
            ("densi2", "density of the second background species (cm^-3)"),
            ("tempi1", "temperature of the first background species (eV)"),
            ("tempi2", "temperature of the second background species (eV)"),
            ("tempe", "electron temperature (eV)"),
        ]:
            lines.append(
                f"{key} = {_format_fortran_double(params[key])},             ! {comment}"
            )

    lines.append("/")
    return "\n".join(lines) + "\n"
