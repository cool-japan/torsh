"""Optimization algorithms."""

try:
    from . import torsh_python as _C
except ImportError:
    import torsh_python as _C

# Import from Rust module
from torsh_python.optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
)

__all__ = [
    'Optimizer', 'SGD', 'Adam', 'AdamW',
]