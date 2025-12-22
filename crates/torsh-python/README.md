# ToRSh Python Bindings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/torsh.svg)](https://pypi.org/project/torsh/)

Python bindings for **ToRSh** - a PyTorch-compatible deep learning framework built in pure Rust.

## 🚀 Quick Start

### Installation

#### From Source (Development)

```bash
# Install maturin
pip install maturin

# Clone the repository
git clone https://github.com/cool-japan/torsh.git
cd torsh/crates/torsh-python

# Build and install in development mode
maturin develop

# Or build in release mode for better performance
maturin develop --release
```

#### From PyPI (Coming Soon)

```bash
pip install torsh
```

### Basic Usage

```python
import torsh

# Device management
cpu = torsh.PyDevice("cpu")
cuda = torsh.PyDevice("cuda:0")
print(f"Device: {cpu.type}, Index: {cpu.index}")

# Data types
float32 = torsh.PyDType("float32")
int64 = torsh.PyDType("int64")
print(f"DType: {float32.name}, Size: {float32.itemsize} bytes")

# Check device availability
print(f"CUDA available: {torsh.cuda_is_available()}")
print(f"MPS available: {torsh.mps_is_available()}")
```

See [examples/basic_usage.py](examples/basic_usage.py) for more examples.

## 📚 Documentation

### Current Status

**Version**: 0.1.0-alpha.2

**Note**: This crate is in active development. Many features are currently disabled due to dependency conflicts with scirs2-autograd and are being re-enabled incrementally.

#### ✅ Available Features

- **Device Management**: CPU, CUDA, Metal device support with PyTorch-compatible API
- **Data Type Handling**: Complete dtype system with float32, int64, bool, etc.
- **Error Handling**: Comprehensive error types with helpful messages
- **Validation Utilities**: 25+ validation functions for input checking
- **Type Stubs**: Full `.pyi` type stubs for IDE support
- **Documentation**: Comprehensive documentation for all public APIs

#### ❌ Currently Disabled (Coming Soon)

- Tensor operations and creation functions
- Neural network layers (torsh.nn)
- Optimization algorithms (torsh.optim)
- Automatic differentiation (torsh.autograd)
- Distributed training (torsh.distributed)
- Functional operations (torsh.F)

See [TODO.md](TODO.md) for the full roadmap and progress tracking.

### API Reference

#### Device Management

```python
# Create devices
cpu = torsh.PyDevice("cpu")
cuda0 = torsh.PyDevice("cuda")      # Default to cuda:0
cuda1 = torsh.PyDevice("cuda:1")    # Specific GPU
metal = torsh.PyDevice("metal:0")   # Apple Silicon

# Device properties
print(cuda1.type)   # "cuda"
print(cuda1.index)  # 1

# Device equality
cpu1 = torsh.PyDevice("cpu")
cpu2 = torsh.PyDevice("cpu")
assert cpu1 == cpu2

# Utility functions
torsh.device_count()        # Number of devices
torsh.is_available()        # General availability
torsh.cuda_is_available()   # CUDA availability
torsh.mps_is_available()    # Metal Performance Shaders availability
```

#### Data Types

```python
# Create dtypes
float32 = torsh.PyDType("float32")  # or "f32"
float64 = torsh.PyDType("float64")  # or "f64"
int32 = torsh.PyDType("int32")      # or "i32"
int64 = torsh.PyDType("int64")      # or "i64"
bool_type = torsh.PyDType("bool")

# DType properties
print(float32.name)              # "float32"
print(float32.itemsize)          # 4 (bytes)
print(float32.is_floating_point) # True
print(float32.is_signed)         # True

# DType constants
torsh.float32  # Predefined dtype
torsh.float64
torsh.int32
torsh.int64
torsh.bool

# PyTorch-style aliases
torsh.float   # Same as float32
torsh.double  # Same as float64
torsh.long    # Same as int64
torsh.int     # Same as int32
```

#### Error Handling

```python
# Custom errors
error = torsh.TorshError("Custom error message")
print(str(error))    # "Custom error message"
print(repr(error))   # "TorshError('Custom error message')"

# Built-in validation with helpful errors
try:
    invalid_device = torsh.PyDevice("invalid")
except ValueError as e:
    print(f"Error: {e}")  # "Unknown device: invalid"
```

## 🏗️ Architecture

### Design Principles

1. **PyTorch Compatibility**: API designed to match PyTorch's interface for easy migration
2. **SciRS2 Integration**: Built on top of the SciRS2 scientific computing ecosystem
3. **Type Safety**: Leverages Rust's type system with Python type hints
4. **Performance**: Zero-cost abstractions and efficient memory management
5. **Modularity**: Clean separation between core functionality and Python bindings

### SciRS2 POLICY Compliance

This crate strictly follows the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md):

- **✅ REQUIRED**: All external dependencies accessed through `scirs2-core` abstractions
- **✅ REQUIRED**: No direct imports of `ndarray`, `rand`, `num-traits`, `rayon`, etc.
- **✅ REQUIRED**: Unified access through `scirs2_core::ndarray`, `scirs2_core::random`, etc.
- **✅ MANDATORY**: SIMD/parallel operations through `scirs2_core` only

See [SCIRS2_INTEGRATION_POLICY.md](../../SCIRS2_INTEGRATION_POLICY.md) for full details.

### Project Structure

```
torsh-python/
├── src/
│   ├── lib.rs              # Main module registration
│   ├── device.rs           # Device management
│   ├── dtype.rs            # Data type handling
│   ├── error.rs            # Error handling
│   ├── tensor/             # Tensor operations (disabled)
│   ├── nn/                 # Neural network layers (disabled)
│   ├── optim/              # Optimizers (disabled)
│   └── utils/              # Validation and utilities
├── python/
│   └── torsh/
│       ├── __init__.pyi    # Type stubs
│       └── py.typed        # PEP 561 marker
├── tests/                  # Integration tests
├── examples/               # Usage examples
├── pyproject.toml          # Python package metadata
├── Cargo.toml              # Rust package metadata
└── README.md               # This file
```

## 🧪 Testing

### Running Tests

```bash
# Run Rust unit tests
cargo test

# Run validation tests
cargo test --lib validation::tests

# Build Python extension for manual testing
maturin develop
python examples/basic_usage.py
```

### Test Coverage

- ✅ **Device Module**: 30+ comprehensive tests covering all functionality
- ✅ **DType Module**: 40+ tests for all data types and properties
- ✅ **Error Module**: Error creation and conversion tests
- ✅ **Validation Module**: 70+ tests for all validation functions

## 🛠️ Development

### Prerequisites

- Rust 1.70+ (for GAT support)
- Python 3.8+
- Maturin 1.0+

### Building from Source

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Build in debug mode
maturin develop

# Build in release mode
maturin develop --release

# Build wheel
maturin build --release
```

### Code Quality

```bash
# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy

# Format Python code
black examples/

# Type check Python code
mypy examples/
```

## 📊 Benchmarks

Benchmarks will be added once tensor operations are re-enabled.

## 🗺️ Roadmap

### v0.1.0-alpha.3 (Next Release)

- [ ] Re-enable tensor operations
- [ ] Re-enable basic neural network layers
- [ ] Add tensor creation functions (zeros, ones, randn)
- [ ] Add basic tensor operations (add, mul, matmul)

### v0.1.0-alpha.4

- [ ] Re-enable autograd support
- [ ] Re-enable optimizer implementations
- [ ] Add data loading utilities
- [ ] Performance benchmarks

### v0.1.0-beta.1

- [ ] Distributed training support
- [ ] Complete PyTorch API compatibility
- [ ] Comprehensive documentation
- [ ] Full test coverage

See [TODO.md](TODO.md) for detailed task breakdown.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **SciRS2 POLICY**: All code must follow the SciRS2 POLICY strictly
2. **Tests**: Add comprehensive tests for all new functionality
3. **Documentation**: Document all public APIs with examples
4. **Type Hints**: Include Python type stubs (.pyi files)
5. **Code Quality**: Run `cargo fmt` and `cargo clippy` before submitting

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- **PyTorch**: For the excellent API design that we strive to emulate
- **SciRS2**: For providing the scientific computing foundation
- **PyO3**: For excellent Rust-Python bindings

## 📞 Contact

- **Repository**: https://github.com/cool-japan/torsh
- **Issues**: https://github.com/cool-japan/torsh/issues

## 🔗 Related Projects

- [ToRSh](https://github.com/cool-japan/torsh) - Main ToRSh framework
- [SciRS2](https://github.com/cool-japan/scirs) - Scientific computing in Rust
- [NumRS2](https://github.com/cool-japan/numrs) - Numerical computing library
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python bindings

---

**Status**: 🚧 Active Development | **Version**: 0.1.0-alpha.2 | **Last Updated**: 2025-10-24
