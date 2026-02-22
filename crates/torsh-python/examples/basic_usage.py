"""
Basic usage examples for torsh-python

This example demonstrates the basic functionality currently available:
- Device management
- Data type handling
- Error handling

Note: Tensor operations are currently disabled due to dependency issues.
"""

import sys
import os

# Add the parent directory to path to import rstorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'debug'))

try:
    import rstorch_python as rstorch
except ImportError:
    print("Error: rstorch_python not built yet. Please run 'maturin develop' first.")
    sys.exit(1)


def demonstrate_devices():
    """Demonstrate device management"""
    print("=" * 60)
    print("DEVICE MANAGEMENT")
    print("=" * 60)

    # Create CPU device
    cpu = rstorch.PyDevice("cpu")
    print(f"CPU device: {cpu}")
    print(f"  Type: {cpu.type}")
    print(f"  Index: {cpu.index}")
    print()

    # Create CUDA devices with different indices
    cuda0 = rstorch.PyDevice("cuda:0")
    cuda1 = rstorch.PyDevice("cuda:1")
    print(f"CUDA device 0: {cuda0}")
    print(f"  Type: {cuda0.type}")
    print(f"  Index: {cuda0.index}")
    print()

    print(f"CUDA device 1: {cuda1}")
    print(f"  Type: {cuda1.type}")
    print(f"  Index: {cuda1.index}")
    print()

    # Create Metal device
    metal = rstorch.PyDevice("metal:0")
    print(f"Metal device: {metal}")
    print(f"  Type: {metal.type}")
    print(f"  Index: {metal.index}")
    print()

    # Device equality
    cpu2 = rstorch.PyDevice("cpu")
    print(f"Device equality test:")
    print(f"  cpu == cpu2: {cpu == cpu2}")
    print(f"  cpu == cuda0: {cpu == cuda0}")
    print()

    # Device constants
    print(f"Device constants:")
    print(f"  rstorch.cpu: {rstorch.cpu}")
    print()

    # Device utility functions
    print(f"Device utility functions:")
    print(f"  device_count(): {rstorch.device_count()}")
    print(f"  is_available(): {rstorch.is_available()}")
    print(f"  cuda_is_available(): {rstorch.cuda_is_available()}")
    print(f"  mps_is_available(): {rstorch.mps_is_available()}")
    print(f"  get_device_name(cpu): {rstorch.get_device_name(cpu)}")
    print()


def demonstrate_dtypes():
    """Demonstrate data type handling"""
    print("=" * 60)
    print("DATA TYPE HANDLING")
    print("=" * 60)

    # Create different data types
    float32 = rstorch.PyDType("float32")
    float64 = rstorch.PyDType("float64")
    int32 = rstorch.PyDType("int32")
    int64 = rstorch.PyDType("int64")
    bool_type = rstorch.PyDType("bool")

    print("Data types:")
    print(f"  float32: {float32} (size: {float32.itemsize} bytes)")
    print(f"  float64: {float64} (size: {float64.itemsize} bytes)")
    print(f"  int32: {int32} (size: {int32.itemsize} bytes)")
    print(f"  int64: {int64} (size: {int64.itemsize} bytes)")
    print(f"  bool: {bool_type} (size: {bool_type.itemsize} bytes)")
    print()

    # Check properties
    print("Float32 properties:")
    print(f"  Name: {float32.name}")
    print(f"  Is floating point: {float32.is_floating_point}")
    print(f"  Is signed: {float32.is_signed}")
    print()

    print("Int32 properties:")
    print(f"  Name: {int32.name}")
    print(f"  Is floating point: {int32.is_floating_point}")
    print(f"  Is signed: {int32.is_signed}")
    print()

    print("Bool properties:")
    print(f"  Name: {bool_type.name}")
    print(f"  Is floating point: {bool_type.is_floating_point}")
    print(f"  Is signed: {bool_type.is_signed}")
    print()

    # Type aliases
    f32_alias = rstorch.PyDType("f32")
    print(f"Type aliases:")
    print(f"  'float32' == 'f32': {float32 == f32_alias}")
    print()

    # Type constants
    print("Type constants:")
    print(f"  rstorch.float32: {rstorch.float32}")
    print(f"  rstorch.float64: {rstorch.float64}")
    print(f"  rstorch.int32: {rstorch.int32}")
    print(f"  rstorch.int64: {rstorch.int64}")
    print(f"  rstorch.bool: {rstorch.bool}")
    print()

    # PyTorch-style aliases
    print("PyTorch-style aliases:")
    print(f"  rstorch.float: {rstorch.float}")
    print(f"  rstorch.double: {rstorch.double}")
    print(f"  rstorch.long: {rstorch.long}")
    print(f"  rstorch.int: {rstorch.int}")
    print()


def demonstrate_error_handling():
    """Demonstrate error handling"""
    print("=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)

    # Custom error
    try:
        error = rstorch.TorshError("This is a test error")
        print(f"TorshError created: {error}")
        print(f"  Repr: {repr(error)}")
    except Exception as e:
        print(f"Error creating TorshError: {e}")
    print()

    # Invalid device
    try:
        invalid_device = rstorch.PyDevice("invalid_device")
    except ValueError as e:
        print(f"✓ Caught ValueError for invalid device: {e}")
    print()

    # Invalid CUDA device ID
    try:
        invalid_cuda = rstorch.PyDevice("cuda:abc")
    except ValueError as e:
        print(f"✓ Caught ValueError for invalid CUDA ID: {e}")
    print()

    # Negative device ID
    try:
        negative_device = rstorch.PyDevice(-1)
    except ValueError as e:
        print(f"✓ Caught ValueError for negative device ID: {e}")
    print()

    # Invalid dtype
    try:
        invalid_dtype = rstorch.PyDType("invalid_dtype")
    except ValueError as e:
        print(f"✓ Caught ValueError for invalid dtype: {e}")
    print()

    # Unsupported dtype (uint16)
    try:
        unsupported_dtype = rstorch.PyDType("uint16")
    except ValueError as e:
        print(f"✓ Caught ValueError for unsupported dtype: {e}")
    print()


def demonstrate_version():
    """Display version information"""
    print("=" * 60)
    print("VERSION INFORMATION")
    print("=" * 60)
    print(f"ToRSh Python version: {rstorch.__version__}")
    print()


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("TORSH PYTHON BINDINGS - BASIC USAGE EXAMPLES")
    print("=" * 60)
    print()

    demonstrate_version()
    demonstrate_devices()
    demonstrate_dtypes()
    demonstrate_error_handling()

    print("=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)
    print()

    print("Note: Tensor operations are currently disabled due to dependency issues.")
    print("      Please check the TODO.md file for the roadmap to re-enable them.")


if __name__ == "__main__":
    main()
