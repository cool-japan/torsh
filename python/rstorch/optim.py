"""Optimization algorithms."""

try:
    from . import _C
except ImportError:
    import rstorch._C as _C

# Import from Rust module
from ._C.optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
)

__all__ = [
    'Optimizer', 'SGD', 'Adam', 'AdamW',
]