# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.0-alpha.x   | :white_check_mark: |
| < 0.1.0-alpha.2 | :x:                |

## Reporting a Vulnerability

The ToRSh team and community take security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report a Security Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
- **Security Contact**: security@torsh-project.org (if available)
- **Alternative**: Open a private security advisory on GitHub

When reporting vulnerabilities, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** of the vulnerability
4. **Suggested fix** (if you have one)
5. **Your contact information** for follow-up

### What to Expect

- **Acknowledgment**: We will acknowledge your email within 48 hours
- **Updates**: We will send you regular updates about our progress
- **Timeline**: We aim to fix critical vulnerabilities within 7 days
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Security Update Process

1. **Verification**: We verify the reported vulnerability
2. **Assessment**: We assess the severity and impact
3. **Fix Development**: We develop and test a fix
4. **Release**: We release a security update
5. **Disclosure**: We publicly disclose the vulnerability (coordinated with reporter)

## Security Best Practices

When using ToRSh Python bindings, please follow these security best practices:

### Code Execution

- **Never execute untrusted Python code** with ToRSh
- **Validate all inputs** before processing
- **Use sandboxing** when processing untrusted data

### Dependencies

- **Keep dependencies updated** to get security patches
- **Review dependencies** regularly for known vulnerabilities
- **Use `cargo audit`** to check for vulnerabilities

```bash
# Install cargo-audit
cargo install cargo-audit

# Run audit
cargo audit
```

### Memory Safety

ToRSh leverages Rust's memory safety guarantees:

- **No buffer overflows** - Rust prevents out-of-bounds access
- **No use-after-free** - Ownership system prevents dangling pointers
- **No data races** - Borrow checker ensures thread safety

However, be aware of:
- **Unsafe blocks** - We minimize unsafe code, but it exists
- **FFI boundaries** - Python-Rust boundary requires careful handling

### Input Validation

Always validate inputs:

```python
import torsh

# Validate device string
try:
    device = torsh.PyDevice(user_input)
except ValueError as e:
    print(f"Invalid device: {e}")

# Validate dtype
try:
    dtype = torsh.PyDType(user_input)
except ValueError as e:
    print(f"Invalid dtype: {e}")
```

### Safe Defaults

ToRSh uses safe defaults:

- **CPU device by default** - Safer than GPU for untrusted code
- **Float32 dtype by default** - Balanced precision/performance
- **Validation enabled** - All inputs are validated

## Known Security Considerations

### Current Status (v0.1.0-alpha.x)

⚠️ **Alpha Release**: This is an alpha release and should not be used in production

environments without thorough testing and security review.

#### Disabled Modules

Currently disabled due to API conflicts (not security issues):
- Tensor operations
- Neural network modules
- Optimizers
- Autograd
- Distributed training

#### Active Security Features

✅ **Memory Safety**: Rust's ownership system prevents common vulnerabilities
✅ **Input Validation**: All public APIs validate inputs
✅ **Error Handling**: Comprehensive error handling prevents crashes
✅ **Type Safety**: Strong typing prevents type confusion attacks

### Potential Security Concerns

1. **Unvalidated Tensor Shapes** (when tensor ops are re-enabled)
   - Large shapes could cause memory exhaustion
   - Mitigation: Size limits and validation

2. **NumPy Array Conversion** (future)
   - Shared memory could lead to race conditions
   - Mitigation: Copy-on-write and ownership transfer

3. **GPU Memory** (when CUDA is enabled)
   - GPU memory exhaustion possible
   - Mitigation: Memory limits and monitoring

4. **Distributed Training** (when re-enabled)
   - Network communication could be intercepted
   - Mitigation: TLS encryption required

## Vulnerability Disclosure Policy

### Responsible Disclosure

We follow responsible disclosure practices:

1. **Private Reporting**: Vulnerabilities reported privately first
2. **Fix Development**: Fix developed before public disclosure
3. **Coordinated Disclosure**: Public disclosure coordinated with reporter
4. **Credit Given**: Credit given to reporter (if desired)

### Public Disclosure Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Acknowledgment sent
- **Day 3-7**: Verification and assessment
- **Day 7-30**: Fix development and testing
- **Day 30+**: Security update released
- **Day 30+**: Public disclosure (coordinated)

## Security Advisories

Security advisories will be published at:
- **GitHub Security Advisories**: https://github.com/cool-japan/torsh/security/advisories
- **CVE Database**: For critical vulnerabilities
- **Release Notes**: All security updates documented

## Automated Security Scanning

We use automated tools to detect vulnerabilities:

### Dependency Scanning

```bash
# Run cargo audit (in CI/CD)
cargo audit

# Update dependencies
cargo update
```

### Code Scanning

- **Clippy**: Rust linter catches common issues
- **CodeQL**: GitHub security scanning
- **Detect-secrets**: Pre-commit hook for secrets

### Continuous Monitoring

Our CI/CD pipeline includes:
- ✅ Dependency vulnerability scanning (cargo audit)
- ✅ Static analysis (clippy)
- ✅ Secret detection (detect-secrets)
- ✅ License compliance checks

## Security-Related Configuration

### Build Flags

For production builds, use:

```bash
# Enable optimizations
cargo build --release

# Enable overflow checks (production)
RUSTFLAGS="-C overflow-checks=on" cargo build --release
```

### Runtime Configuration

```python
# Safe configuration for untrusted environments
import torsh

# Always validate inputs
def safe_create_device(device_str: str) -> torsh.PyDevice:
    """Create device with validation"""
    allowed_devices = ["cpu", "cuda:0", "cuda:1"]
    if device_str not in allowed_devices:
        raise ValueError(f"Device {device_str} not in allowed list")
    return torsh.PyDevice(device_str)
```

## Contact

For security-related questions or concerns:

- **Email**: security@torsh-project.org (if available)
- **GitHub**: https://github.com/cool-japan/torsh/security
- **Issues**: For non-security bugs, use regular GitHub issues

## Resources

- [Rust Security Guidelines](https://anssi-fr.github.io/rust-guide/)
- [Python Security](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)

---

**Last Updated**: 2025-10-24
**Version**: 0.1.0-alpha.2
