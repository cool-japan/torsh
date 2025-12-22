# ToRSh Python Bindings - Developer Guide

This guide provides detailed information for developers working on torsh-python.

## Table of Contents

- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Build System](#build-system)
- [Testing Guide](#testing-guide)
- [Debugging](#debugging)
- [Performance Profiling](#performance-profiling)
- [Release Process](#release-process)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Project Structure

```
torsh-python/
├── .github/workflows/      # CI/CD workflows
│   ├── ci.yml             # Continuous integration
│   └── release.yml        # Release automation
│
├── examples/              # Python examples
│   ├── basic_usage.py    # Basic API demonstration
│   └── README.md         # Example documentation
│
├── python/torsh/         # Python package files
│   ├── __init__.pyi      # Type stubs
│   └── py.typed          # PEP 561 marker
│
├── src/                  # Rust source code
│   ├── lib.rs           # Main module registration
│   ├── device.rs        # Device management (✅ enabled)
│   ├── dtype.rs         # Data type handling (✅ enabled)
│   ├── error.rs         # Error handling (✅ enabled)
│   │
│   ├── tensor/          # Tensor operations (❌ disabled)
│   │   ├── mod.rs
│   │   ├── core.rs
│   │   └── creation.rs
│   │
│   ├── nn/              # Neural network layers (❌ disabled)
│   │   ├── mod.rs
│   │   ├── linear.rs
│   │   ├── conv.rs
│   │   ├── pooling.rs
│   │   ├── normalization.rs
│   │   ├── dropout.rs
│   │   ├── activation.rs
│   │   ├── container.rs
│   │   ├── loss.rs
│   │   └── module.rs
│   │
│   ├── optim/           # Optimizers (❌ disabled)
│   │   ├── mod.rs
│   │   ├── base.rs
│   │   ├── sgd.rs
│   │   ├── adam.rs
│   │   ├── adagrad.rs
│   │   └── rmsprop.rs
│   │
│   ├── utils/           # Utilities (✅ enabled)
│   │   ├── mod.rs
│   │   ├── validation.rs  # Input validation (25+ functions)
│   │   └── conversion.rs
│   │
│   ├── autograd.rs      # Autograd (❌ disabled)
│   ├── distributed.rs   # Distributed (❌ disabled)
│   └── functional.rs    # Functional ops (❌ disabled)
│
├── tests/               # Integration tests
│   ├── test_device.rs   # Device module tests (30+ tests)
│   ├── test_dtype.rs    # DType module tests (40+ tests)
│   └── test_error.rs    # Error handling tests
│
├── build.rs             # Build script
├── Cargo.toml           # Rust package manifest
├── pyproject.toml       # Python package manifest
├── README.md            # Project documentation
├── TODO.md              # Project roadmap
├── CONTRIBUTING.md      # Contribution guidelines
└── DEVELOPMENT.md       # This file
```

## Development Environment

### Required Tools

```bash
# Rust toolchain
rustup update stable
rustup component add rustfmt clippy

# Python
python3 --version  # 3.8+

# Maturin (Python-Rust bridge)
pip install maturin

# Development tools
pip install black ruff mypy pytest pytest-cov

# Pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Recommended VS Code Extensions

- `rust-analyzer` - Rust language server
- `Python` - Python language support
- `Even Better TOML` - TOML syntax highlighting
- `GitLens` - Git integration
- `Error Lens` - Inline error display

### Editor Configuration

```json
// .vscode/settings.json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.features": "all",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "editor.formatOnSave": true,
  "editor.rulers": [100]
}
```

## Build System

### Maturin Overview

ToRSh Python uses [Maturin](https://github.com/PyO3/maturin) to build Python wheels from Rust code.

```bash
# Development build (debug, fast compile)
maturin develop

# Release build (optimized, slow compile)
maturin develop --release

# Build wheel (for distribution)
maturin build --release

# Build wheel for specific Python version
maturin build --release --interpreter python3.11
```

### Build Profiles

**Debug Build** (`maturin develop`):
- Fast compilation
- No optimizations
- Debug symbols included
- Suitable for development

**Release Build** (`maturin develop --release`):
- Slow compilation
- Full optimizations
- No debug symbols
- Suitable for benchmarking

### Cargo Features

```toml
[features]
default = []
extension-module = ["pyo3/extension-module"]  # Required for Python module
```

## Testing Guide

### Test Organization

Tests are organized in three categories:

1. **Unit Tests** (inline in source files)
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_feature() {
           // Test implementation
       }
   }
   ```

2. **Integration Tests** (`tests/` directory)
   ```rust
   // tests/test_device.rs
   use pyo3::prelude::*;

   #[test]
   fn test_device_creation() {
       // Test using PyO3
   }
   ```

3. **Python Tests** (future - `python/tests/`)
   ```python
   # python/tests/test_device.py
   import torsh

   def test_device_creation():
       device = torsh.PyDevice("cpu")
       assert device.type == "cpu"
   ```

### Running Tests

```bash
# Run all Rust tests
cargo test --lib

# Run specific test file
cargo test --lib --test test_device

# Run specific test
cargo test --lib test_device_creation

# Run with output
cargo test --lib -- --nocapture

# Run with backtrace
RUST_BACKTRACE=1 cargo test --lib

# Run Python tests (future)
pytest python/tests/
```

### Writing Good Tests

#### Test Naming Convention

```rust
#[test]
fn test_{module}_{feature}_{scenario}() {
    // Examples:
    // test_device_creation_cpu()
    // test_dtype_equality_same()
    // test_validation_index_negative()
}
```

#### Test Structure

```rust
#[test]
fn test_feature() {
    // Arrange - Set up test data
    let input = create_test_input();

    // Act - Execute the code being tested
    let result = function_under_test(input);

    // Assert - Verify the result
    assert_eq!(result, expected);
}
```

#### Testing PyO3 Code

```rust
use pyo3::prelude::*;

#[test]
fn test_python_function() {
    Python::with_gil(|py| {
        let code = r#"
import sys
sys.path.insert(0, '{manifest_dir}')
import torsh_python as torsh

device = torsh.PyDevice("cpu")
result = str(device)
"#;
        let module = PyModule::from_code(
            py,
            &code.replace("{manifest_dir}", env!("CARGO_MANIFEST_DIR")),
            "",
            "",
        )
        .unwrap();

        let result: String = module.getattr("result").unwrap().extract().unwrap();
        assert_eq!(result, "cpu");
    });
}
```

## Debugging

### Rust Debugging

```bash
# Build with debug symbols
cargo build

# Run with debugger (lldb on macOS, gdb on Linux)
rust-lldb target/debug/torsh-python
```

### Python Debugging

```python
# Install Python debugger
pip install ipdb

# Add breakpoint in code
import ipdb; ipdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

### Logging

```rust
// Add logging to Rust code
use log::{debug, info, warn, error};

debug!("Debug message: {}", value);
info!("Info message");
warn!("Warning message");
error!("Error message");
```

```bash
# Set log level
RUST_LOG=debug maturin develop
```

## Performance Profiling

### Rust Profiling

```bash
# Install profiling tools
cargo install cargo-flamegraph
cargo install cargo-criterion

# Generate flamegraph
cargo flamegraph --bin torsh-python

# Run benchmarks
cargo bench
```

### Python Profiling

```python
# Profile Python code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
import torsh
device = torsh.PyDevice("cpu")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

### Memory Profiling

```bash
# Install memory profiler
cargo install cargo-valgrind

# Run with valgrind
cargo valgrind --bin torsh-python
```

## Release Process

### Version Bumping

1. Update version in `Cargo.toml`
2. Update version in `pyproject.toml`
3. Update CHANGELOG.md
4. Commit changes

```bash
# Update version
vim Cargo.toml pyproject.toml

# Commit
git add Cargo.toml pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.0-alpha.3"
```

### Creating a Release

```bash
# Tag the release
git tag v0.1.0-alpha.3
git push origin v0.1.0-alpha.3

# GitHub Actions will automatically:
# 1. Build wheels for all platforms
# 2. Create GitHub release
# 3. Upload wheels to release
```

### Manual Release (if needed)

```bash
# Build wheels for all platforms
maturin build --release --universal2

# Build source distribution
maturin sdist

# Upload to PyPI (requires credentials)
maturin publish
```

## Common Tasks

### Adding a New Python Function

1. **Write Rust implementation**:
   ```rust
   // src/device.rs
   #[pyfunction]
   fn new_function(arg: String) -> PyResult<String> {
       Ok(format!("Result: {}", arg))
   }
   ```

2. **Register function**:
   ```rust
   // src/device.rs
   pub fn register_device_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add_function(wrap_pyfunction!(new_function, m)?)?;
       Ok(())
   }
   ```

3. **Add tests**:
   ```rust
   #[test]
   fn test_new_function() {
       Python::with_gil(|py| {
           // Test implementation
       });
   }
   ```

4. **Update type stubs**:
   ```python
   # python/torsh/__init__.pyi
   def new_function(arg: str) -> str: ...
   ```

5. **Update documentation**:
   ```rust
   /// Description of new function
   ///
   /// # Arguments
   ///
   /// * `arg` - Description
   ///
   /// # Returns
   ///
   /// Description of return value
   #[pyfunction]
   fn new_function(arg: String) -> PyResult<String> {
       // Implementation
   }
   ```

### Adding a New Module

1. Create module file (e.g., `src/new_module.rs`)
2. Add module declaration in `src/lib.rs`
3. Implement PyO3 classes and functions
4. Register with Python module
5. Add tests
6. Update type stubs
7. Update documentation

### Updating Dependencies

```bash
# Update Cargo dependencies
cargo update

# Check for outdated dependencies
cargo install cargo-outdated
cargo outdated

# Update Python dependencies
pip install --upgrade -r requirements-dev.txt
```

## Troubleshooting

### Common Issues

#### PyO3 Symbol Not Found

**Problem**: `dyld: symbol not found: _PyExc_BaseException`

**Solution**: This is expected when running Rust tests without Python runtime. Tests will work correctly when run through Python.

#### Maturin Build Fails

**Problem**: Build fails with linking errors

**Solution**:
```bash
# Clean build
cargo clean
maturin develop --release
```

#### Import Error in Python

**Problem**: `ImportError: cannot import name 'torsh_python'`

**Solution**:
```bash
# Rebuild extension
maturin develop --release

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Type Stubs Not Working

**Problem**: IDE doesn't show type hints

**Solution**:
1. Ensure `py.typed` file exists
2. Check `__init__.pyi` is up to date
3. Restart IDE/language server

### Getting Help

1. Check [TODO.md](TODO.md) for known issues
2. Check [GitHub Issues](https://github.com/cool-japan/torsh/issues)
3. Read [PyO3 Documentation](https://pyo3.rs/)
4. Ask on GitHub Discussions

## Additional Resources

- [PyO3 User Guide](https://pyo3.rs/)
- [Maturin Documentation](https://github.com/PyO3/maturin)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Python C API](https://docs.python.org/3/c-api/)
- [SciRS2 Documentation](https://github.com/cool-japan/scirs)

---

**Last Updated**: 2025-10-24
