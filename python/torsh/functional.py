"""Functional interface for neural network operations."""

try:
    from . import torsh_python as _C
except ImportError:
    import torsh_python as _C

# Import all functional operations from Rust module
from torsh_python.functional import *

__all__ = [
    # Activation functions
    'relu', 'relu6', 'leaky_relu', 'elu', 'selu', 'gelu', 'silu', 'mish',
    'sigmoid', 'tanh', 'softmax', 'log_softmax', 'softplus', 'softsign',
    
    # Loss functions
    'mse_loss', 'cross_entropy', 'nll_loss', 'l1_loss', 'smooth_l1_loss',
    'huber_loss', 'binary_cross_entropy', 'binary_cross_entropy_with_logits',
    'kl_div',
    
    # Pooling functions
    'max_pool1d', 'max_pool2d', 'avg_pool1d', 'avg_pool2d',
    'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_max_pool1d', 'adaptive_max_pool2d',
    
    # Normalization functions
    'batch_norm', 'layer_norm', 'group_norm', 'instance_norm',
    
    # Dropout functions
    'dropout', 'dropout2d', 'dropout3d',
    
    # Linear algebra functions
    'linear', 'conv1d', 'conv2d',
]