"""A tiny name -> strategy registry, mirroring evalscope's own registry style."""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from evalscope_ext.pruning.base import PruningStrategy

_STRATEGIES: Dict[str, Type[PruningStrategy]] = {}


def register_pruning_strategy(name: str) -> Callable[[Type[PruningStrategy]], Type[PruningStrategy]]:
    """Class decorator registering a :class:`PruningStrategy` under ``name``."""

    def deco(cls: Type[PruningStrategy]) -> Type[PruningStrategy]:
        if name in _STRATEGIES:
            raise ValueError(f"Pruning strategy '{name}' already registered.")
        cls.name = name
        _STRATEGIES[name] = cls
        return cls

    return deco


def get_pruning_strategy(name: str) -> PruningStrategy:
    """Instantiate a registered strategy by name."""
    if name not in _STRATEGIES:
        raise ValueError(
            f"Pruning strategy '{name}' is not registered. Available: {sorted(_STRATEGIES)}"
        )
    return _STRATEGIES[name]()


def list_pruning_strategies() -> List[str]:
    return sorted(_STRATEGIES)
