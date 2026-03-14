"""
Shared path resolution utilities for StellCoilBench.

Provides helpers for locating plasma_surfaces directories, surface files,
and case YAML files across the repository layout.
"""

from __future__ import annotations

from ._case_resolution import (
    CaseResolutionResult,
    ResolvedPaths,
    coils_json_path_from_dir,
    find_case_and_surface_path,
    resolve_all,
    resolve_case_and_surface,
    resolve_case_yaml_path,
)
from ._surface_extraction import get_surface_filename
from ._surface_io import get_reference_radii, load_surface_with_range
from ._path_search import (
    find_dir_up,
    find_file_up,
    find_plasma_surfaces_dir,
    find_repo_root,
)
from ._surface_resolution import (
    get_surface_search_base_dirs,
    get_target_B_from_surface,
    normalize_surface_id,
    resolve_surface_file_path,
    resolve_surface_path,
    surface_stem_from_filename,
)
from ._yaml import dump_yaml, load_yaml, load_yaml_safe

__all__ = [
    "CaseResolutionResult",
    "ResolvedPaths",
    "coils_json_path_from_dir",
    "get_surface_filename",
    "get_reference_radii",
    "load_surface_with_range",
    "dump_yaml",
    "find_case_and_surface_path",
    "find_dir_up",
    "find_file_up",
    "find_plasma_surfaces_dir",
    "find_repo_root",
    "get_surface_search_base_dirs",
    "get_target_B_from_surface",
    "normalize_surface_id",
    "load_yaml",
    "load_yaml_safe",
    "resolve_all",
    "resolve_case_and_surface",
    "resolve_case_yaml_path",
    "resolve_surface_file_path",
    "resolve_surface_path",
    "surface_stem_from_filename",
]
