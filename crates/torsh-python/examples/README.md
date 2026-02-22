# ToRSh Python Examples

This directory contains examples demonstrating the usage of torsh-python bindings.

## Available Examples

### basic_usage.py
Demonstrates the basic functionality currently available in torsh-python:
- Device management (CPU, CUDA, Metal)
- Data type handling (float32, int64, etc.)
- Error handling
- Version information

## Running Examples

First, build the Python extension using maturin:

```bash
# Install maturin if you haven't already
pip install maturin

# Build and install the extension in development mode
cd /path/to/torsh/crates/torsh-python
maturin develop

# Run the example
python examples/basic_usage.py
```

Alternatively, you can build in release mode for better performance:

```bash
maturin develop --release
python examples/basic_usage.py
```

## Current Status

**Note**: Many tensor operations are currently disabled due to dependency conflicts with scirs2-autograd. The following features are available:

✅ **Available**:
- Device management (PyDevice)
- Data type handling (PyDType)
- Error handling (TorshError)
- Utility functions

❌ **Currently Disabled**:
- Tensor operations
- Neural network layers (nn module)
- Optimization algorithms (optim module)
- Automatic differentiation (autograd module)
- Distributed training (distributed module)
- Functional operations (F module)

## Future Examples

Once tensor operations are re-enabled, we will add examples for:
- Tensor creation and manipulation
- Neural network training
- Optimization
- Automatic differentiation
- Distributed training
- Custom operations

## See Also

- [TODO.md](../TODO.md) - Project roadmap and progress
- [README.md](../README.md) - Installation and usage instructions
