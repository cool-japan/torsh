# Dependency Rationale - torsh-core

This document explains the rationale behind each dependency choice, alternatives considered, and version constraints.

## Table of Contents

- [Core Dependencies](#core-dependencies)
- [SciRS2 Ecosystem](#scirs2-ecosystem)
- [Optional Dependencies](#optional-dependencies)
- [Development Dependencies](#development-dependencies)
- [Dependency Policies](#dependency-policies)

## Core Dependencies

### scirs2-core (v0.1.0-rc.2)

**Purpose**: Foundation for scientific computing primitives, unified access to ndarray, random, and numeric operations.

**Rationale**:
- **SciRS2 POLICY Compliance**: All external dependencies MUST go through scirs2-core
- **Unified API**: Single import point for ndarray, rand, num-traits, etc.
- **Version Control**: Centralized version management reduces conflicts
- **Type Safety**: Consistent types across the ecosystem
- **Performance**: Optimized implementations with SIMD support

**Alternatives Considered**:
1. **Direct ndarray + rand + num-traits**
   - ❌ Version conflicts across crates
   - ❌ No unified API
   - ❌ Duplicated abstractions

2. **PyO3 + NumPy**
   - ❌ Python runtime dependency
   - ❌ Performance overhead
   - ❌ Not pure Rust

**Why Chosen**: Scirs2-core provides the best balance of performance, type safety, and ecosystem integration.

**Version Constraint**: `= 0.1.0-rc.2` (exact version for API stability during RC period)

---

### thiserror (v1.0)

**Purpose**: Ergonomic error type definitions with automatic Error trait implementation.

**Rationale**:
- **Boilerplate Reduction**: Automatic Display and Error impl
- **Compile-Time Checked**: Error messages validated at compile time
- **Source Chain**: Proper `#[source]` and `#[from]` support
- **Performance**: Zero runtime overhead
- **Widely Adopted**: De facto standard in Rust ecosystem

**Alternatives Considered**:
1. **Manual Error Implementation**
   ```rust
   // Manual implementation
   impl fmt::Display for TorshError {
       fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { ... }
   }
   impl Error for TorshError { ... }
   ```
   - ❌ Verbose and error-prone
   - ❌ Easy to forget source chains

2. **anyhow**
   - ❌ Type erasure loses specific error information
   - ❌ Not suitable for library code
   - ✅ Good for applications

3. **snafu**
   - ✅ Similar feature set to thiserror
   - ❌ Less widely adopted
   - ❌ More opinionated context management

**Why Chosen**: thiserror is the standard choice for library error types, providing exactly what we need without unnecessary features.

**Version Constraint**: `>= 1.0` (semver compatible, stable API)

---

### parking_lot (v0.12)

**Purpose**: High-performance synchronization primitives (Mutex, RwLock, Condvar).

**Rationale**:
- **Performance**: 2-5x faster than std::sync::Mutex in contended scenarios
- **Fairness**: Fair lock acquisition prevents starvation
- **API Compatibility**: Drop-in replacement for std::sync
- **Smaller**: More compact lock types (1 byte for Mutex)
- **Deadlock Detection**: Optional deadlock detection in debug builds

**Performance Comparison**:
```
Uncontended lock: std: 10ns, parking_lot: 8ns (20% faster)
Contended lock:   std: 50ns, parking_lot: 15ns (233% faster)
```

**Alternatives Considered**:
1. **std::sync**
   - ✅ No external dependency
   - ❌ Slower in contended scenarios
   - ❌ Larger lock types

2. **spin**
   - ✅ Very fast in low-contention
   - ❌ Poor performance under high contention
   - ❌ Can waste CPU cycles spinning

**Why Chosen**: Performance gain in multi-threaded tensor operations justifies the dependency.

**Version Constraint**: `>= 0.12` (stable API, semver compatible)

---

### once_cell (v1.19)

**Purpose**: Lazy static initialization and thread-safe OnceCell.

**Rationale**:
- **Lazy Initialization**: Defer expensive initialization until first use
- **Thread Safety**: Thread-safe lazy statics without macros
- **No Macros**: Clean syntax without `lazy_static!` macro
- **Standard Path**: Part of Rust 2024 edition (std::cell::LazyCell, std::sync::OnceLock)
- **Zero Cost**: Compiled away when not used

**Usage Examples**:
```rust
// Global stride cache
static STRIDE_CACHE: Lazy<Mutex<LruCache<Shape, Vec<usize>>>> =
    Lazy::new(|| Mutex::new(LruCache::new(1000)));

// Device registry
static DEVICE_REGISTRY: OnceLock<DeviceRegistry> = OnceLock::new();
```

**Alternatives Considered**:
1. **lazy_static**
   - ✅ Similar functionality
   - ❌ Macro-based (less clear)
   - ❌ Less actively maintained

2. **std::sync::OnceLock** (Rust 1.70+)
   - ✅ Standard library
   - ✅ Similar API to once_cell
   - ⚠️ once_cell provides Lazy which is more ergonomic

**Why Chosen**: Industry standard for lazy initialization, path to std adoption.

**Version Constraint**: `>= 1.19` (stable API)

## SciRS2 Ecosystem

### scirs2-linalg (v0.1.0-rc.2)

**Purpose**: Linear algebra operations (matrix multiplication, decomposition, etc.).

**Rationale**:
- **Integration**: Seamless integration with scirs2-core
- **Performance**: BLAS-accelerated operations
- **Completeness**: Full suite of linear algebra routines
- **Type Safety**: Strong typing for matrix operations

**Version Constraint**: `= 0.1.0-rc.2` (matches scirs2-core version)

---

### scirs2-stats (v0.1.0-rc.2)

**Purpose**: Statistical functions and distributions.

**Rationale**:
- **Statistical Tests**: Comprehensive test suite
- **Distributions**: Unified access to rand_distr through scirs2
- **Analysis**: Statistical analysis tools for ML
- **Benchmarking**: Performance statistics and regression detection

**Version Constraint**: `= 0.1.0-rc.2` (matches scirs2-core version)

---

### numrs2 (v0.1.0-beta.3)

**Purpose**: Numerical computing library for scientific operations.

**Rationale**:
- **Ecosystem Alignment**: Part of the cool-japan Rust scientific ecosystem
- **Numerical Algorithms**: Optimized numerical methods
- **Interoperability**: Works seamlessly with scirs2

**Version Constraint**: `= 0.1.0-beta.3` (beta stability acceptable for this functionality)

## Optional Dependencies

### serde (v1.0) - Feature: "serialize"

**Purpose**: Serialization and deserialization support.

**Rationale**:
- **Universal**: De facto standard for serialization in Rust
- **Format Agnostic**: Works with JSON, CBOR, oxicode, etc.
- **Derive Macros**: Automatic implementation for most types
- **Performance**: Efficient zero-copy deserialization
- **Ecosystem**: Massive ecosystem support

**Why Optional**: Not all users need serialization, reduces compile time and binary size.

**Alternatives Considered**:
1. **oxicode** (chosen for binary serialization)
   - ✅ Fast binary format with 100% bincode compatibility
   - ✅ Modern successor to bincode with advanced features
   - ✅ SIMD optimization, compression, streaming support

2. **rkyv**
   - ✅ Zero-copy deserialization
   - ❌ More complex, requires generated types
   - ❌ Less mature ecosystem

**Version Constraint**: `>= 1.0` (stable semver API)

---

### criterion (v0.5) - Dev Dependency

**Purpose**: Benchmarking framework with statistical analysis.

**Rationale**:
- **Statistical Rigor**: Detects performance regressions reliably
- **Visualization**: HTML reports with plots
- **Parameterized**: Easy to benchmark across inputs
- **Industry Standard**: Used by most Rust projects

**Alternatives Considered**:
1. **bencher** (deprecated)
   - ❌ No longer maintained

2. **Manual timing**
   - ❌ No statistical analysis
   - ❌ Hard to detect small regressions

**Version Constraint**: `>= 0.5` (current stable)

---

### proptest (v1.4) - Dev Dependency

**Purpose**: Property-based testing framework.

**Rationale**:
- **Comprehensive Testing**: Generates thousands of test cases
- **Shrinking**: Minimizes failing test cases
- **Reproducible**: Seed-based generation
- **Coverage**: Finds edge cases developers miss

**Use Cases**:
- Shape broadcasting edge cases
- Type promotion combinations
- Stride calculation correctness

**Alternatives Considered**:
1. **quickcheck**
   - ✅ Similar concept
   - ❌ Less maintained
   - ❌ Weaker shrinking

**Version Constraint**: `>= 1.0` (stable API)

## Development Dependencies

### approx (v0.5)

**Purpose**: Floating-point comparison with tolerance.

**Rationale**:
- **Necessity**: Exact floating-point comparison is unreliable
- **Flexibility**: Multiple comparison strategies (relative, absolute, ULP)
- **Testing**: Essential for numerical algorithm testing

**Usage**:
```rust
assert_relative_eq!(result, expected, epsilon = 1e-6);
```

**Version Constraint**: `>= 0.5` (stable API)

---

### tempfile (v3.0)

**Purpose**: Temporary file and directory creation for tests.

**Rationale**:
- **Testing**: Essential for testing file I/O operations
- **Cleanup**: Automatic cleanup on drop
- **Cross-Platform**: Works consistently across platforms

**Version Constraint**: `>= 3.0` (stable API)

## Dependency Policies

### Version Constraints Strategy

1. **Core Dependencies (scirs2-*)**: Exact version (`= x.y.z`) during RC/alpha
   - **Reason**: API stability during development
   - **Post-1.0**: Will use semver compatible (`>= x.y`)

2. **Standard Crates**: Semver compatible (`>= x.y`)
   - **Reason**: Leverage ecosystem improvements
   - **Safety**: Cargo ensures compatibility

3. **Dev Dependencies**: Permissive (`>= x.y`)
   - **Reason**: Only affects development, not users
   - **Benefit**: Get latest testing tools

### Dependency Update Policy

1. **Regular Updates**: Check for updates monthly
2. **Security Updates**: Apply immediately
3. **Breaking Changes**: Evaluate before upgrading
4. **Testing**: Run full test suite after updates
5. **Documentation**: Update CHANGELOG for dependency changes

### Dependency Addition Criteria

New dependencies must satisfy ALL of:

1. **Necessity**: Cannot reasonably implement ourselves
2. **Maintenance**: Actively maintained (commit in last 6 months)
3. **Quality**: High-quality code, good documentation
4. **License**: Compatible with our MIT/Apache-2.0 license
5. **Size**: Reasonable compile-time and binary size impact
6. **Alternatives**: Alternatives evaluated and documented

**Red Flags**:
- ❌ No activity in 12+ months
- ❌ Many open issues/PRs without response
- ❌ Questionable license
- ❌ Adds > 50 transitive dependencies
- ❌ Significantly increases compile time

### Minimizing Dependency Tree

**Strategies**:
1. **Feature Flags**: Make heavy dependencies optional
2. **Workspace Sharing**: Share dependencies across crates
3. **Version Alignment**: Use same version across workspace
4. **Audit**: Regular `cargo tree` audits
5. **Alternatives**: Consider lighter alternatives

**Current Stats** (as of v0.1.0-alpha.2):
```
Total dependencies: ~30 (including transitives)
Compile time: ~90 seconds (clean build)
Binary size impact: ~2MB (release build)
```

## Platform-Specific Dependencies

### Windows

- **windows-sys**: Low-level Windows API bindings
  - **Purpose**: Memory info, NUMA support
  - **Alternative**: winapi (deprecated)

### macOS

- **core-foundation-sys**: macOS frameworks
  - **Purpose**: Metal device detection
  - **Alternative**: objc (more complex)

### Linux

- **libc**: POSIX API bindings
  - **Purpose**: NUMA, CPU affinity
  - **Standard**: Part of Rust std development

## Future Dependency Considerations

### Potential Additions

1. **rayon** (parallel iterators)
   - **Consideration**: Excellent parallel processing
   - **Concern**: Must go through scirs2-core (POLICY)
   - **Status**: Use scirs2_core::parallel_ops

2. **ndarray** (N-dimensional arrays)
   - **Consideration**: Standard for numerical computing
   - **Concern**: Must go through scirs2-core (POLICY)
   - **Status**: Use scirs2_core::ndarray

3. **half** (f16/bf16 types)
   - **Consideration**: Hardware support growing
   - **Status**: Already implementing custom types

### Dependencies to Avoid

1. **tokio** (async runtime)
   - **Reason**: Synchronous operations are sufficient
   - **Impact**: Large dependency tree
   - **Alternative**: std::thread for now

2. **diesel** (ORM)
   - **Reason**: Not a database application
   - **Alternative**: N/A

## Dependency Health Monitoring

### Tools Used

1. **cargo-audit**: Security vulnerability scanning
2. **cargo-outdated**: Find outdated dependencies
3. **cargo-tree**: Analyze dependency tree
4. **cargo-deny**: Dependency policy enforcement

### Regular Checks

```bash
# Security audit (weekly)
cargo audit

# Check for updates (monthly)
cargo outdated

# Analyze dependency tree (quarterly)
cargo tree --duplicates

# Policy enforcement (CI/CD)
cargo deny check
```

## Conclusion

Our dependency choices prioritize:
1. **Performance**: parking_lot, criterion
2. **Safety**: thiserror, proptest
3. **Ecosystem**: SciRS2 integration
4. **Maintainability**: Well-maintained, standard crates
5. **Minimalism**: Only necessary dependencies

Every dependency is justified, alternatives are documented, and we maintain strict policies for additions and updates.

---

*Last Updated: 2025-10-23*
*Version: 0.1.0-alpha.2*
