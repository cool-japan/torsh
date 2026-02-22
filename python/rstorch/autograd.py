"""Automatic differentiation utilities."""

try:
    from . import _C
except ImportError:
    import rstorch._C as _C

# Import from Rust module
from ._C.autograd import (
    no_grad,
    enable_grad,
    set_grad_enabled,
    Function,
    grad,
    backward,
    is_grad_enabled,
    detect_anomaly,
)

__all__ = [
    'no_grad', 'enable_grad', 'set_grad_enabled', 'Function',
    'grad', 'backward', 'is_grad_enabled', 'detect_anomaly',
]