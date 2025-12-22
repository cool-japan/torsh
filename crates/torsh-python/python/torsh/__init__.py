"""
ToRSh Python Bindings

PyTorch-compatible deep learning framework implemented in Rust.

This package provides Python bindings for the ToRSh deep learning framework,
offering high-performance tensor operations and neural network functionality.
"""

from .torsh import *  # noqa: F401, F403

__all__ = [
    # Version
    "__version__",
    # Main classes
    "PyDevice",
    "PyDType",
    "TorshError",
    # Device constants
    "cpu",
    # Device utility functions
    "device_count",
    "is_available",
    "cuda_is_available",
    "mps_is_available",
    "get_device_name",
    # DType constants
    "float32",
    "f32",
    "float64",
    "f64",
    "int8",
    "i8",
    "int16",
    "i16",
    "int32",
    "i32",
    "int64",
    "i64",
    "uint8",
    "u8",
    "uint32",
    "u32",
    "uint64",
    "u64",
    "bool",
    # PyTorch-style aliases
    "float",
    "double",
    "long",
    "int",
    "short",
    "byte",
    # DType utility functions
    "promote_types",
    "result_type",
    "can_operate",
]
