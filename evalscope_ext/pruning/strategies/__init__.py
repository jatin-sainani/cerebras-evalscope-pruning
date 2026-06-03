"""Built-in pruning strategies. Importing this package registers them all."""

from evalscope_ext.pruning.strategies import (  # noqa: F401
    composite,
    discrimination,
    diversity,
    judge_noise,
    stratified,
)
