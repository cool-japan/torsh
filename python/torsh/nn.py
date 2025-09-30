"""Neural network modules and functions."""

from typing import Optional, Dict, Any, List
try:
    from . import torsh_python as _C
except ImportError:
    import torsh_python as _C

# Import from Rust module
from torsh_python.nn import (
    Module,
    Linear,
    Sequential,
    ModuleList,
    
    # Activation functions
    relu,
    sigmoid,
    tanh,
    softmax,
    log_softmax,
    
    # Loss functions
    mse_loss,
    cross_entropy,
)

# Parameter class (would need to implement)
class Parameter:
    """A kind of Tensor that is to be considered a module parameter."""
    
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad
    
    def __repr__(self):
        return f"Parameter containing:\n{self.data}"

# Convenience function
def parameter(data, requires_grad=True):
    """Create a parameter from data."""
    return Parameter(data, requires_grad)

__all__ = [
    'Module', 'Linear', 'Sequential', 'ModuleList', 'Parameter', 'parameter',
    'relu', 'sigmoid', 'tanh', 'softmax', 'log_softmax',
    'mse_loss', 'cross_entropy',
]