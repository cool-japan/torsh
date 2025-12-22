# Contributing to ToRSh

Thank you for your interest in contributing to ToRSh! This document provides guidelines for contributing to the project.

## 🎯 Project Status

ToRSh is currently in **alpha stage** (v0.1.0-alpha.2). We're actively working towards beta and production releases. Your contributions help us build a production-ready deep learning framework in pure Rust.

## 🚀 Getting Started

### Prerequisites

- Rust toolchain (1.75.0 or later)
- Git for version control
- Familiarity with Rust and/or PyTorch

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/cool-japan/torsh.git
cd torsh

# Build the project
make build

# Run tests
make test-fast

# Run lints
make lint

# Format code
make format
```

## 📋 Development Workflow

### Before Starting Work

1. **Check existing issues**: Look for open issues or create a new one
2. **Discuss your approach**: For significant changes, discuss in an issue first
3. **Review the codebase**: Familiarize yourself with relevant modules

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow coding standards**:
   - Follow Rust naming conventions (snake_case for functions/variables)
   - Add documentation for public APIs
   - Keep functions under 200 lines when possible
   - Follow the SciRS2 POLICY (see [SCIRS2_INTEGRATION_POLICY.md](./SCIRS2_INTEGRATION_POLICY.md))

3. **Write tests**:
   - Add unit tests for new functionality
   - Ensure all tests pass: `make test-fast`
   - Maintain or improve test coverage

4. **Run quality checks**:
   ```bash
   make check  # Runs format + lint + test-fast
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

### Submitting Changes

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request**:
   - Provide a clear description of changes
   - Reference related issues
   - Include test results
   - Update CHANGELOG.md if appropriate

3. **Code review**:
   - Address reviewer feedback
   - Keep commits focused and atomic
   - Squash commits if requested

## 🎨 Coding Standards

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (configured in project)
- Use `clippy` for linting (zero warnings policy)
- Document all public APIs with doc comments

### SciRS2 POLICY Compliance

**CRITICAL**: ToRSh must comply with the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md):

- **NEVER** import `ndarray`, `rand`, `num-traits` directly
- **ALWAYS** use `scirs2_core::ndarray::*` for array operations
- **ALWAYS** use `scirs2_core::random::*` for random number generation
- **ALWAYS** use `scirs2_core::numeric::*` for numerical traits

See [SCIRS2_INTEGRATION_POLICY.md](./SCIRS2_INTEGRATION_POLICY.md) for details.

### Documentation

- All public functions, structs, and modules must have doc comments
- Include usage examples in doc comments where helpful
- Update README.md files when adding new features
- Keep CHANGELOG.md updated

### Testing

- Write unit tests for all new functionality
- Add integration tests for cross-module features
- Use `#[cfg(test)]` modules for test organization
- Use deterministic random seeds in tests
- Follow the "no warnings" policy

## 🔍 Areas for Contribution

### High Priority

- **GPU Backend Development**: CUDA, Metal, WebGPU optimization
- **Performance Optimization**: SIMD improvements, kernel fusion
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Increase test coverage, edge cases
- **Bug Fixes**: Address open issues

### Medium Priority

- **Distributed Training**: Multi-node training improvements
- **Model Zoo**: Pre-trained model implementations
- **Quantization**: INT8/INT4 support
- **JIT Compilation**: Graph optimization passes

### Good First Issues

Look for issues labeled `good first issue` in the issue tracker. These are specifically chosen to be approachable for new contributors.

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Reproduction steps**: Minimal code to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - ToRSh version
   - Rust version
   - Operating system
   - Hardware (CPU/GPU)

## 💡 Suggesting Features

For feature requests:

1. **Check existing issues**: Avoid duplicates
2. **Provide context**: Why is this feature needed?
3. **PyTorch compatibility**: How does PyTorch handle this?
4. **Implementation ideas**: (Optional) How might it be implemented?

## 📖 Documentation Contributions

Documentation is crucial! You can help by:

- Fixing typos and improving clarity
- Adding code examples
- Writing tutorials
- Improving API documentation
- Creating guides for specific use cases

## 🧪 Testing Guidelines

### Running Tests

```bash
# Fast test suite (recommended for development)
make test-fast

# Full test suite (includes slower backend tests)
make test

# Test specific package
cargo test --package torsh-nn

# Test with output
cargo test -- --nocapture
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature() {
        // Arrange
        let input = array![1.0, 2.0, 3.0];

        // Act
        let result = my_function(&input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

## 📝 Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(tensor): add support for complex conjugate operation
fix(autograd): resolve gradient accumulation race condition
docs(nn): add examples for custom layer implementation
```

## ⚖️ License

By contributing to ToRSh, you agree that your contributions will be licensed under either:
- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))
- MIT license ([LICENSE-MIT](./LICENSE-MIT))

at the user's option (dual-licensed).

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions
- Collaborate openly

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks or trolling
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## 📞 Getting Help

- **Discord**: Join our community (link TBD)
- **GitHub Issues**: Ask questions in issues
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check [docs.rs/torsh](https://docs.rs/torsh)

## 🎓 Learning Resources

### For New Contributors

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SciRS2 Documentation](https://docs.rs/scirs2)
- [ToRSh Examples](./examples/)

### For Advanced Contributors

- [SCIRS2_INTEGRATION_POLICY.md](./SCIRS2_INTEGRATION_POLICY.md) - Integration guidelines
- [CLAUDE.md](./CLAUDE.md) - Project development guide
- [PYTHON_BINDINGS.md](./PYTHON_BINDINGS.md) - Python bindings implementation

## 🙏 Thank You

Every contribution, no matter how small, helps make ToRSh better. Whether you're fixing a typo, adding a feature, or reporting a bug - thank you for being part of the ToRSh community!

---

**Last Updated**: December 21, 2025
**Version**: 0.1.0-alpha.2
