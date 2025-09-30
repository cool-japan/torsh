"""Automatic differentiation utilities."""

try:
    from . import torsh_python as _C
except ImportError:
    import torsh_python as _C

# Import from Rust module
from torsh_python.autograd import (
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