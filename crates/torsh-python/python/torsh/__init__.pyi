"""
ToRSh - PyTorch-compatible deep learning framework in Rust

Type stubs for the torsh module providing type hints and IDE support.
"""

from typing import Optional, Union, List, Tuple, Any
from enum import Enum

__version__: str

# ============================================================================
# Device Types
# ============================================================================

class Device:
    """
    Represents a computational device (CPU, CUDA, etc.)

    Examples:
        >>> device = Device("cpu")
        >>> device = Device("cuda")
        >>> device = Device("cuda:0")
    """

    def __init__(self, device: Union[str, 'Device']) -> None:
        """
        Create a new device.

        Args:
            device: Device specification as string ("cpu", "cuda", "cuda:0") or Device object
        """
        ...

    def __str__(self) -> str:
        """String representation of the device."""
        ...

    def __repr__(self) -> str:
        """Detailed representation of the device."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality with another device."""
        ...

    @property
    def type(self) -> str:
        """Get device type (cpu, cuda, etc.)."""
        ...

    @property
    def index(self) -> Optional[int]:
        """Get device index (for multi-GPU systems)."""
        ...

# Device constants
cpu: Device
cuda: Device

# ============================================================================
# Data Types
# ============================================================================

class DType:
    """
    Represents a tensor data type (float32, int64, etc.)

    Examples:
        >>> dtype = DType.float32
        >>> dtype = DType.int64
    """

    def __str__(self) -> str:
        """String representation of the dtype."""
        ...

    def __repr__(self) -> str:
        """Detailed representation of the dtype."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality with another dtype."""
        ...

    @property
    def name(self) -> str:
        """Get dtype name."""
        ...

    @property
    def itemsize(self) -> int:
        """Get size in bytes of this dtype."""
        ...

    @property
    def is_floating_point(self) -> bool:
        """Check if this is a floating point dtype."""
        ...

    @property
    def is_signed(self) -> bool:
        """Check if this is a signed dtype."""
        ...

    @property
    def is_complex(self) -> bool:
        """Check if this is a complex dtype."""
        ...

    @property
    def is_integer(self) -> bool:
        """Check if this is an integer dtype."""
        ...

    @property
    def numpy_dtype(self) -> str:
        """Get the NumPy-compatible dtype string."""
        ...

    def can_cast(self, other: 'DType') -> bool:
        """
        Check if this dtype can be safely cast to another dtype.

        Args:
            other: Target dtype to check casting compatibility

        Returns:
            True if safe cast is possible, False otherwise
        """
        ...

# DType constants
float32: DType
float64: DType
int32: DType
int64: DType
int8: DType
int16: DType
uint8: DType
uint32: DType
uint64: DType
bool: DType

# PyTorch-style dtype aliases
float: DType  # alias for float32
double: DType  # alias for float64
long: DType  # alias for int64
int: DType  # alias for int32
short: DType  # alias for int16
char: DType  # alias for int8
byte: DType  # alias for uint8

# DType utility functions
def promote_types(dtype1: DType, dtype2: DType) -> DType:
    """
    Promote two dtypes to a common dtype for operations.

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        Promoted dtype that can safely represent both inputs

    Examples:
        >>> result = promote_types(int32, float32)
        >>> print(result)  # float32
    """
    ...

def result_type(dtype1: DType, dtype2: DType) -> DType:
    """
    Get the result dtype for a binary operation between two dtypes.

    Args:
        dtype1: First operand dtype
        dtype2: Second operand dtype

    Returns:
        Result dtype for the operation
    """
    ...

def can_operate(dtype1: DType, dtype2: DType) -> bool:
    """
    Check if two dtypes are compatible for operations.

    Args:
        dtype1: First dtype
        dtype2: Second dtype

    Returns:
        True if dtypes can be used together in operations
    """
    ...

# ============================================================================
# Error Types
# ============================================================================

class TorshError(Exception):
    """Base exception class for ToRSh errors."""
    pass

class ShapeError(TorshError):
    """Error related to tensor shape operations."""
    pass

class DeviceError(TorshError):
    """Error related to device operations."""
    pass

class DTypeError(TorshError):
    """Error related to data type operations."""
    pass

class ValueError(TorshError):
    """Error related to invalid values."""
    pass

class RuntimeError(TorshError):
    """Runtime error in ToRSh operations."""
    pass

# ============================================================================
# Tensor Class (Currently Disabled)
# ============================================================================

# class Tensor:
#     """
#     Multi-dimensional array with automatic differentiation support.
#
#     Examples:
#         >>> t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
#         >>> t = torch.zeros([2, 3])
#         >>> t = torch.randn([3, 4], requires_grad=True)
#     """
#
#     def __init__(
#         self,
#         data: Any,
#         dtype: Optional[DType] = None,
#         device: Optional[Device] = None,
#         requires_grad: bool = False
#     ) -> None: ...
#
#     @property
#     def shape(self) -> Tuple[int, ...]: ...
#
#     @property
#     def dtype(self) -> DType: ...
#
#     @property
#     def device(self) -> Device: ...
#
#     @property
#     def requires_grad(self) -> bool: ...

# ============================================================================
# Tensor Creation Functions (Currently Disabled)
# ============================================================================

# def tensor(
#     data: Any,
#     dtype: Optional[DType] = None,
#     device: Optional[Device] = None,
#     requires_grad: bool = False
# ) -> Tensor:
#     """Create a tensor from data."""
#     ...
#
# def zeros(
#     size: List[int],
#     dtype: Optional[DType] = None,
#     device: Optional[Device] = None,
#     requires_grad: bool = False
# ) -> Tensor:
#     """Create a tensor filled with zeros."""
#     ...
#
# def ones(
#     size: List[int],
#     dtype: Optional[DType] = None,
#     device: Optional[Device] = None,
#     requires_grad: bool = False
# ) -> Tensor:
#     """Create a tensor filled with ones."""
#     ...
#
# def randn(
#     size: List[int],
#     dtype: Optional[DType] = None,
#     device: Optional[Device] = None,
#     requires_grad: bool = False
# ) -> Tensor:
#     """Create a tensor with random normal distribution."""
#     ...
#
# def rand(
#     size: List[int],
#     dtype: Optional[DType] = None,
#     device: Optional[Device] = None,
#     requires_grad: bool = False
# ) -> Tensor:
#     """Create a tensor with random uniform distribution."""
#     ...

# ============================================================================
# Module Submodules (Currently Disabled)
# ============================================================================

# class nn:
#     """Neural network modules and layers."""
#     pass
#
# class optim:
#     """Optimization algorithms."""
#     pass
#
# class F:
#     """Functional neural network operations."""
#     pass
#
# class autograd:
#     """Automatic differentiation utilities."""
#     pass
#
# class distributed:
#     """Distributed training utilities."""
#     pass
