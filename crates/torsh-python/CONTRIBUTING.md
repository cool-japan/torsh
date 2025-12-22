# Contributing to ToRSh Python Bindings

Thank you for your interest in contributing to ToRSh! This document provides guidelines and instructions for contributing to the torsh-python crate.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [SciRS2 POLICY Compliance](#scirs2-policy-compliance)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Rust 1.70+ (for GAT support)
- Python 3.8+
- Maturin 1.0+
- Git

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/cool-japan/torsh.git
cd torsh/crates/torsh-python

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (if using)
pre-commit install

# Build in development mode
maturin develop

# Run tests
cargo test --lib
```

## Development Workflow

### 1. Fork and Branch

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/torsh.git
cd torsh/crates/torsh-python

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the [SciRS2 POLICY](#scirs2-policy-compliance) strictly
- Write tests for all new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 3. Test Your Changes

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-features -- -D warnings

# Run tests
cargo test --lib

# Build Python extension
maturin develop --release

# Test Python interface
python examples/basic_usage.py
```

### 4. Commit and Push

```bash
# Commit with descriptive message
git add .
git commit -m "feat: add new feature X"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and create a PR from your fork
- Fill out the PR template completely
- Link any related issues
- Wait for review

## SciRS2 POLICY Compliance

**CRITICAL**: All contributions MUST follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) strictly.

### ✅ Required Practices

1. **UNIFIED ndarray Access (v0.1.0-RC.1+)**
   ```rust
   // ✅ CORRECT
   use scirs2_core::ndarray::*;  // ALL array operations

   // ❌ WRONG
   use ndarray::{Array, array};  // POLICY VIOLATION
   ```

2. **UNIFIED random Access (v0.1.0-RC.1+)**
   ```rust
   // ✅ CORRECT
   use scirs2_core::random::*;  // ALL RNG and distributions

   // ❌ WRONG
   use rand::{thread_rng, Rng};  // POLICY VIOLATION
   ```

3. **UNIFIED numeric Access**
   ```rust
   // ✅ CORRECT
   use scirs2_core::numeric::*;  // ALL numerical traits

   // ❌ WRONG
   use num_traits::{Float, Zero};  // POLICY VIOLATION
   ```

4. **Performance Through scirs2-core**
   - SIMD: `scirs2_core::simd_ops::SimdUnifiedOps` (MANDATORY)
   - Parallel: `scirs2_core::parallel_ops::*` (MANDATORY)
   - GPU: `scirs2_core::gpu` (MANDATORY when using GPU)

### ❌ Prohibited Practices

```rust
// ❌ FORBIDDEN: Direct external imports
use ndarray::{Array, array};           // POLICY VIOLATION
use rand::{thread_rng, Rng};           // POLICY VIOLATION
use num_traits::{Float, Zero};         // POLICY VIOLATION
use rayon::prelude::*;                 // POLICY VIOLATION
```

**Any PR violating the SciRS2 POLICY will be rejected.**

## Code Style Guidelines

### Rust Code

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Maximum line length: 100 characters
- Use snake_case for variables and functions
- Use PascalCase for types and traits

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use `black` for formatting (line length: 100)
- Use `ruff` for linting
- Use type hints everywhere
- Maximum line length: 100 characters

### Documentation

- All public functions must have documentation
- Use Rust doc comments (`///`) for public APIs
- Include examples in documentation
- Document all parameters and return values
- Document error conditions

Example:
```rust
/// Create a new device from a string or integer specification.
///
/// # Arguments
///
/// * `device` - Device specification as string ("cpu", "cuda", "cuda:0", "metal:0")
///              or integer (for CUDA device index)
///
/// # Returns
///
/// New PyDevice instance
///
/// # Errors
///
/// Returns ValueError if:
/// - Device string is not recognized
/// - Device index is invalid (negative or malformed)
///
/// # Examples
///
/// ```python
/// cpu = torsh.PyDevice("cpu")
/// cuda = torsh.PyDevice("cuda:0")
/// ```
#[new]
fn new(device: &Bound<'_, PyAny>) -> PyResult<Self> {
    // Implementation
}
```

## Testing Requirements

### Test Coverage

- **All new functionality MUST have tests**
- Aim for 100% coverage of new code
- Test both success and error cases
- Test edge cases and boundary conditions

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_success() {
        // Test successful case
    }

    #[test]
    fn test_feature_error() {
        // Test error case
    }

    #[test]
    fn test_feature_edge_case() {
        // Test edge case
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test --lib

# Run specific test
cargo test --lib test_name

# Run with output
cargo test --lib -- --nocapture
```

## Documentation Standards

### Code Documentation

- All public modules need module-level documentation
- All public functions need function-level documentation
- Include usage examples in documentation
- Document all parameters, return values, and errors

### Type Stubs

When adding new Python-facing APIs, update:
- `python/torsh/__init__.pyi` - Add type definitions

### Examples

When adding significant new features:
- Add examples to `examples/`
- Update `examples/README.md`
- Ensure examples are well-commented

### README Updates

Update `README.md` when:
- Adding new major features
- Changing installation process
- Updating requirements
- Changing API significantly

## Pull Request Process

### Before Submitting

1. **Run all checks**:
   ```bash
   cargo fmt
   cargo clippy --all-features -- -D warnings
   cargo test --lib
   maturin develop --release
   python examples/basic_usage.py
   ```

2. **Update documentation**:
   - Add/update inline documentation
   - Update README if needed
   - Update TODO.md progress
   - Add type stubs if needed

3. **Write good commit messages**:
   ```
   feat: add new device utility function

   - Implement get_device_properties()
   - Add comprehensive tests
   - Update documentation
   ```

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## SciRS2 POLICY Compliance
- [ ] No direct external dependencies (ndarray, rand, etc.)
- [ ] Uses scirs2_core abstractions
- [ ] Follows unified access patterns

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Edge cases covered

## Documentation
- [ ] Inline documentation updated
- [ ] Type stubs updated
- [ ] README updated (if needed)
- [ ] Examples added/updated (if needed)

## Checklist
- [ ] Code formatted with cargo fmt
- [ ] Linted with cargo clippy
- [ ] All tests pass
- [ ] Documentation complete
- [ ] SciRS2 POLICY compliant
```

### Review Process

1. Automated CI checks must pass
2. At least one maintainer approval required
3. All review comments addressed
4. No merge conflicts

## Issue Guidelines

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.70.0]
- Python version: [e.g., 3.11]
- ToRSh version: [e.g., 0.1.0-alpha.2]

**Additional context**
Any other relevant information.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've considered.

**Additional context**
Any other relevant information.

**SciRS2 POLICY Compliance**
How will this feature maintain POLICY compliance?
```

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Build process or tooling changes

Examples:
```
feat: add GPU memory profiling support
fix: correct dtype size calculation for complex types
docs: update installation instructions
test: add comprehensive validation tests
refactor: simplify device creation logic
perf: optimize tensor allocation
chore: update CI workflow for Python 3.12
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Review documentation first

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).

---

Thank you for contributing to ToRSh! 🚀
