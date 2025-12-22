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

# Add the parent directory to path to import torsh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'debug'))

try:
    import torsh_python as torsh
except ImportError:
    print("Error: torsh_python not built yet. Please run 'maturin develop' first.")
    sys.exit(1)


def demonstrate_devices():
    """Demonstrate device management"""
    print("=" * 60)
    print("DEVICE MANAGEMENT")
    print("=" * 60)

    # Create CPU device
    cpu = torsh.PyDevice("cpu")
    print(f"CPU device: {cpu}")
    print(f"  Type: {cpu.type}")
    print(f"  Index: {cpu.index}")
    print()

    # Create CUDA devices with different indices
    cuda0 = torsh.PyDevice("cuda:0")
    cuda1 = torsh.PyDevice("cuda:1")
    print(f"CUDA device 0: {cuda0}")
    print(f"  Type: {cuda0.type}")
    print(f"  Index: {cuda0.index}")
    print()

    print(f"CUDA device 1: {cuda1}")
    print(f"  Type: {cuda1.type}")
    print(f"  Index: {cuda1.index}")
    print()

    # Create Metal device
    metal = torsh.PyDevice("metal:0")
    print(f"Metal device: {metal}")
    print(f"  Type: {metal.type}")
    print(f"  Index: {metal.index}")
    print()

    # Device equality
    cpu2 = torsh.PyDevice("cpu")
    print(f"Device equality test:")
    print(f"  cpu == cpu2: {cpu == cpu2}")
    print(f"  cpu == cuda0: {cpu == cuda0}")
    print()

    # Device constants
    print(f"Device constants:")
    print(f"  torsh.cpu: {torsh.cpu}")
    print()

    # Device utility functions
    print(f"Device utility functions:")
    print(f"  device_count(): {torsh.device_count()}")
    print(f"  is_available(): {torsh.is_available()}")
    print(f"  cuda_is_available(): {torsh.cuda_is_available()}")
    print(f"  mps_is_available(): {torsh.mps_is_available()}")
    print(f"  get_device_name(cpu): {torsh.get_device_name(cpu)}")
    print()


def demonstrate_dtypes():
    """Demonstrate data type handling"""
    print("=" * 60)
    print("DATA TYPE HANDLING")
    print("=" * 60)

    # Create different data types
    float32 = torsh.PyDType("float32")
    float64 = torsh.PyDType("float64")
    int32 = torsh.PyDType("int32")
    int64 = torsh.PyDType("int64")
    bool_type = torsh.PyDType("bool")

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
    f32_alias = torsh.PyDType("f32")
    print(f"Type aliases:")
    print(f"  'float32' == 'f32': {float32 == f32_alias}")
    print()

    # Type constants
    print("Type constants:")
    print(f"  torsh.float32: {torsh.float32}")
    print(f"  torsh.float64: {torsh.float64}")
    print(f"  torsh.int32: {torsh.int32}")
    print(f"  torsh.int64: {torsh.int64}")
    print(f"  torsh.bool: {torsh.bool}")
    print()

    # PyTorch-style aliases
    print("PyTorch-style aliases:")
    print(f"  torsh.float: {torsh.float}")
    print(f"  torsh.double: {torsh.double}")
    print(f"  torsh.long: {torsh.long}")
    print(f"  torsh.int: {torsh.int}")
    print()


def demonstrate_error_handling():
    """Demonstrate error handling"""
    print("=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)

    # Custom error
    try:
        error = torsh.TorshError("This is a test error")
        print(f"TorshError created: {error}")
        print(f"  Repr: {repr(error)}")
    except Exception as e:
        print(f"Error creating TorshError: {e}")
    print()

    # Invalid device
    try:
        invalid_device = torsh.PyDevice("invalid_device")
    except ValueError as e:
        print(f"✓ Caught ValueError for invalid device: {e}")
    print()

    # Invalid CUDA device ID
    try:
        invalid_cuda = torsh.PyDevice("cuda:abc")
    except ValueError as e:
        print(f"✓ Caught ValueError for invalid CUDA ID: {e}")
    print()

    # Negative device ID
    try:
        negative_device = torsh.PyDevice(-1)
    except ValueError as e:
        print(f"✓ Caught ValueError for negative device ID: {e}")
    print()

    # Invalid dtype
    try:
        invalid_dtype = torsh.PyDType("invalid_dtype")
    except ValueError as e:
        print(f"✓ Caught ValueError for invalid dtype: {e}")
    print()

    # Unsupported dtype (uint16)
    try:
        unsupported_dtype = torsh.PyDType("uint16")
    except ValueError as e:
        print(f"✓ Caught ValueError for unsupported dtype: {e}")
    print()


def demonstrate_version():
    """Display version information"""
    print("=" * 60)
    print("VERSION INFORMATION")
    print("=" * 60)
    print(f"ToRSh Python version: {torsh.__version__}")
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
