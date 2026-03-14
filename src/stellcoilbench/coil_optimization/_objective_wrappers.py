"""
Base class for objective wrappers that delegate to an underlying simsopt objective.

Provides common delegation logic for __getattr__, x property, and _parent/_children
for simsopt compatibility. Subclasses implement J() and dJ().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Attributes that must not be delegated to the underlying objective.
# Subclasses extend this with their own wrapper-specific attributes.
_EXCLUDE_ATTRS = frozenset({"objective", "_parent", "_children", "J", "dJ"})


class _ObjectiveWrapperBase(ABC):
    """Abstract base for objective wrappers that delegate to an underlying objective.

    Provides __getattr__ delegation, x property, and _parent/_children for
    simsopt compatibility. Subclasses must implement J() and dJ().
    """

    _EXCLUDE_ATTRS: frozenset[str] = _EXCLUDE_ATTRS

    def __init__(
        self,
        objective: Any,
        *,
        _parent: Any = None,
        _children: list[Any] | None = None,
    ) -> None:
        """Initialize the wrapper.

        Parameters
        ----------
        objective : Any
            The underlying simsopt objective (must have J(), dJ(), and x).
        _parent : Any, optional
            Parent objective for simsopt graph (default None).
        _children : list, optional
            Child objectives for simsopt graph (default []).
        """
        self.objective = objective
        self._parent = _parent if _parent is not None else None
        self._children = list(_children) if _children is not None else []

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying objective for simsopt compatibility."""
        if name in type(self)._EXCLUDE_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.objective, name)

    @property
    def x(self) -> np.ndarray:
        """Get optimization variables from underlying objective."""
        return self.objective.x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        """Set optimization variables on underlying objective."""
        self.objective.x = value

    @abstractmethod
    def J(self) -> float:
        """Return the objective value. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def dJ(self, **kwargs: Any) -> Any:
        """Return the objective gradient. Must be implemented by subclasses."""
        ...
