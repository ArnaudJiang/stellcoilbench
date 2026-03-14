"""
Evaluation utilities for StellCoilBench.

This module provides dataclasses for evaluation results. Case loading is
handled by :mod:`stellcoilbench.case_loader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .config_scheme import SubmissionMetadata


@dataclass
class SubmissionResults:
    """
    Aggregated results for a single submission.

    Attributes
    ----------
    metadata : SubmissionMetadata
        Method name, version, contact, hardware.
    metrics : dict[str, Any]
        Evaluation metrics (scores, coil metrics, reactor-scale quantities).
    """

    metadata: SubmissionMetadata
    metrics: Dict[str, Any]
