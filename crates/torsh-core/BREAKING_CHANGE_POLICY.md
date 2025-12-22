# Breaking Change Policy

## Overview

This document defines ToRSh Core's policy for managing breaking changes, ensuring API stability while allowing necessary evolution. We follow **Semantic Versioning 2.0.0** (SemVer) with PyTorch-inspired stability guarantees.

## Semantic Versioning

ToRSh Core uses semantic versioning: `MAJOR.MINOR.PATCH` (e.g., `0.1.0`, `1.0.0`, `2.1.3`)

- **MAJOR**: Incremented for breaking changes (API incompatibilities)
- **MINOR**: Incremented for new features (backward-compatible additions)
- **PATCH**: Incremented for bug fixes (backward-compatible fixes)

### Pre-1.0 Status (Current: 0.1.0-alpha.2)

During the **0.x** series:
- **Breaking changes are allowed** but must be well-documented
- Each **0.MINOR** version may contain breaking changes
- Deprecation warnings are recommended but not strictly required
- Migration guides must be provided for significant breaking changes

### Post-1.0 Stability Guarantees

Once we reach **1.0.0**:
- **Breaking changes only in MAJOR versions** (e.g., 1.x → 2.0)
- **MINOR versions are backward-compatible** (e.g., 1.2 → 1.3)
- **PATCH versions are strictly fixes** (e.g., 1.2.3 → 1.2.4)

## What Constitutes a Breaking Change?

### Definite Breaking Changes

The following **always** constitute breaking changes:

1. **Removing public APIs** (functions, structs, traits, modules)
   ```rust
   // Breaking: Removing a public function
   // pub fn old_api() { }  // Removed
   ```

2. **Changing function signatures**
   ```rust
   // Breaking: Adding required parameter
   // Before: pub fn process(data: &[u8]) -> Result<()>
   // After:  pub fn process(data: &[u8], mode: Mode) -> Result<()>
   ```

3. **Renaming public items**
   ```rust
   // Breaking: Renaming struct
   // Before: pub struct TensorShape { ... }
   // After:  pub struct Shape { ... }
   ```

4. **Changing trait implementations**
   ```rust
   // Breaking: Removing trait implementation
   // Before: impl Clone for Shape { ... }
   // After:  // Clone removed
   ```

5. **Modifying error types** (when pattern-matched by users)
   ```rust
   // Breaking: Changing error variants
   // Before: pub enum TorshError { InvalidShape, ... }
   // After:  pub enum TorshError { ShapeError, ... }  // Renamed variant
   ```

6. **Changing struct field visibility**
   ```rust
   // Breaking: Making field public
   // Before: struct Shape { dims: Vec<usize> }
   // After:  struct Shape { pub dims: Vec<usize> }  // Exposes internal state
   ```

7. **Changing dependencies' major versions** (when re-exported)
   ```rust
   // Breaking: Upgrading re-exported dependency
   // Before: pub use ndarray;  // ndarray 0.15
   // After:  pub use ndarray;  // ndarray 0.16 (breaking changes in ndarray)
   ```

8. **Modifying behavior contracts** (documented behavior)
   ```rust
   // Breaking: Changing documented behavior
   // Before: "Returns elements in ascending order"
   // After:  "Returns elements in descending order"
   ```

### Non-Breaking Changes

The following are **NOT** considered breaking changes:

1. **Adding new public APIs** (functions, structs, traits)
   ```rust
   // Non-breaking: Adding new function
   pub fn new_feature() { }
   ```

2. **Adding default trait methods**
   ```rust
   // Non-breaking: Adding default method
   pub trait Device {
       // ... existing methods ...
       fn new_capability(&self) -> bool { false }  // Default implementation
   }
   ```

3. **Implementing new traits** for existing types
   ```rust
   // Non-breaking: Adding trait implementation
   impl Debug for Shape { ... }
   ```

4. **Adding new variants to non-exhaustive enums**
   ```rust
   // Non-breaking: Adding variant to #[non_exhaustive] enum
   #[non_exhaustive]
   pub enum DeviceType {
       Cpu,
       Cuda,
       Metal,  // New variant added
   }
   ```

5. **Internal implementation changes** (no public API impact)
   ```rust
   // Non-breaking: Changing internal implementation
   // Before: Vec<usize> internal storage
   // After:  SmallVec<[usize; 8]> internal storage
   ```

6. **Documentation improvements**
   - Clarifying documentation
   - Adding examples
   - Fixing typos

7. **Performance improvements** (without behavior changes)
   - SIMD optimizations
   - Memory pooling
   - Caching strategies

8. **Bug fixes** (correcting incorrect behavior)
   - Fixing logic errors
   - Correcting edge cases
   - Resolving undefined behavior

### Gray Areas (Case-by-Case Decision)

Some changes require careful evaluation:

1. **Tightening validation** (rejecting previously accepted invalid input)
   ```rust
   // Potentially breaking: Stricter validation
   // Before: Accepts dimensions up to usize::MAX
   // After:  Rejects dimensions > i64::MAX (overflow protection)
   // Decision: Treat as breaking if documented, non-breaking if fixing bug
   ```

2. **Relaxing constraints** (accepting previously rejected valid input)
   ```rust
   // Generally non-breaking: More permissive validation
   // Before: Requires power-of-two dimensions
   // After:  Accepts any positive dimension
   // Decision: Usually non-breaking, but document prominently
   ```

3. **Changing error messages** (diagnostic text only)
   ```rust
   // Non-breaking: Error message improvements
   // Before: "Invalid shape"
   // After:  "Invalid shape: expected 2D tensor, got 3D [2, 3, 4]"
   // Decision: Non-breaking (messages are not part of stable API)
   ```

4. **Deprecating APIs** (marked with #[deprecated])
   ```rust
   // Non-breaking: Deprecation with migration path
   #[deprecated(since = "0.2.0", note = "Use `new_api` instead")]
   pub fn old_api() { }
   pub fn new_api() { }
   // Decision: Non-breaking if migration path exists
   ```

## Deprecation Process

### Deprecation Timeline

1. **Announce deprecation** (e.g., version 1.5.0)
   - Add `#[deprecated]` attribute
   - Provide migration guide
   - Update documentation
   - Add changelog entry

2. **Maintain deprecated API** (minimum 2 MINOR versions)
   - Version 1.5.0: Deprecation announced
   - Version 1.6.0: Still available (warning)
   - Version 1.7.0: Still available (warning)

3. **Remove in next MAJOR version** (e.g., version 2.0.0)
   - Deprecated API removed
   - Breaking change documented in migration guide

### Deprecation Attributes

```rust
// Basic deprecation
#[deprecated(since = "0.2.0", note = "Use `Shape::new` instead")]
pub fn create_shape(dims: &[usize]) -> Shape {
    Shape::new(dims)
}

// With migration example
#[deprecated(
    since = "1.5.0",
    note = "Use `Device::try_new()` instead.
            Migration: `Device::new(id)` → `Device::try_new(id)?`"
)]
pub fn new(id: usize) -> Device {
    Device::try_new(id).expect("Device creation failed")
}

// Severity levels (using runtime deprecation system)
register_deprecation(
    "old_api",
    "2.0.0",
    DeprecationSeverity::Hard,  // Will be removed soon
    "Use new_api instead",
    "// Before:\nold_api();\n\n// After:\nnew_api();"
);
```

### Deprecation Severity

ToRSh Core uses a three-tier deprecation system (see `api_compat.rs`):

1. **Soft Deprecation**
   - API will be removed in distant future (3+ major versions)
   - Warnings can be disabled
   - Example: Internal utilities transitioning to better abstractions

2. **Hard Deprecation**
   - API will be removed in next major version
   - Warnings should not be disabled
   - Example: APIs with better alternatives available

3. **Critical Deprecation**
   - API is unsafe or fundamentally broken
   - Will be removed ASAP (possibly next minor version)
   - Example: Security vulnerabilities, undefined behavior

## Migration Guides

### Required Information

Every breaking change must include:

1. **What changed**
   - Clear description of the breaking change
   - Affected APIs and modules

2. **Why it changed**
   - Rationale for the breaking change
   - Benefits to users

3. **How to migrate**
   - Step-by-step migration instructions
   - Before/after code examples
   - Automated migration tools (if available)

4. **Timeline**
   - When the change was introduced
   - Deprecation timeline (if applicable)
   - Removal date (for deprecated APIs)

### Migration Guide Template

```markdown
## Breaking Change: [Title]

**Version**: X.Y.Z
**Severity**: [Minor | Major | Critical]
**Affected APIs**: [List of APIs]

### What Changed

[Clear description of the change]

### Why We Made This Change

[Rationale and benefits]

### Migration Guide

#### Before (Old API)
```rust
// Old code example
let shape = Shape::create(&[2, 3, 4]);
```

#### After (New API)
```rust
// New code example
let shape = Shape::new(&[2, 3, 4]);
```

### Automated Migration

[If available, describe automated migration tools]
```bash
cargo torsh-migrate --from 1.x --to 2.0
```

### Deprecation Timeline

- **1.5.0**: Deprecation announced, both APIs available
- **1.6.0 - 1.9.0**: Deprecated API still functional with warnings
- **2.0.0**: Old API removed, must migrate
```

## Version Compatibility Matrix

### SciRS2 POLICY Compliance

ToRSh Core follows the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for dependency management:

| ToRSh Core Version | scirs2-core Version | Rust MSRV |
|--------------------|---------------------|-----------|
| 0.1.0-alpha.2      | 0.1.0-beta.2        | 1.75+     |
| 0.1.0-alpha.2      | 0.1.0-RC.1          | 1.75+     |
| 0.1.0              | 0.1.0               | 1.75+     |
| 1.0.0              | 0.1.x or 1.0.x      | 1.80+     |

### Minimum Supported Rust Version (MSRV)

- **Current MSRV**: 1.75.0
- **MSRV changes** are considered breaking changes for stable (1.x) releases
- **MSRV policy**: Support Rust versions for at least 6 months after new release

## Breaking Change Approval Process

### Pre-1.0 (Current)

Breaking changes require:
1. **Justification document** explaining necessity
2. **Migration guide** with code examples
3. **Changelog entry** with clear breaking change marker
4. **Git commit** with `BREAKING CHANGE:` prefix or footer

Example commit message:
```
feat(shape)!: Refactor Shape API for better ergonomics

BREAKING CHANGE: Shape::create() renamed to Shape::new()

Migration:
- Replace all `Shape::create(&dims)` with `Shape::new(&dims)`
- No behavior changes, pure rename

Rationale: Aligns with Rust naming conventions (new vs create)
```

### Post-1.0 (Future)

Breaking changes require:
1. All pre-1.0 requirements
2. **RFC (Request for Comments)** for major breaking changes
3. **Community feedback period** (minimum 2 weeks)
4. **Core team approval** (majority vote)
5. **Deprecation period** (minimum 2 MINOR versions)

## Stability Guarantees

### Stable APIs (Post-1.0)

Once ToRSh Core reaches 1.0.0, the following are **guaranteed stable**:

1. **Core types**: `Shape`, `DType`, `Device`, `Storage`, `TorshError`
2. **Core traits**: `TensorElement`, `Device`, `Storage`
3. **Public modules**: `shape`, `dtype`, `device`, `storage`, `error`
4. **Re-exported SciRS2 modules**: `numeric`, `random`, `ndarray`, `parallel`, `simd`

### Unstable/Experimental APIs

APIs marked as experimental are **not** covered by stability guarantees:

```rust
#[cfg(feature = "experimental")]
pub mod experimental {
    // Experimental APIs - may change without notice
}
```

Features requiring opt-in are less stable:
```rust
#[cfg(feature = "unstable-advanced-features")]
pub fn bleeding_edge_api() { }
```

### Internal APIs

**Internal** modules (not re-exported in prelude) have weaker guarantees:
- May change in MINOR versions (with deprecation warnings)
- Documented as "internal use only"
- Users depending on internals do so at their own risk

## Communication Channels

### Breaking Change Announcements

1. **Changelog** (CHANGELOG.md)
   - All breaking changes listed under `## Breaking Changes` section
   - Clear migration instructions

2. **GitHub Releases**
   - Release notes with breaking change summaries
   - Links to migration guides

3. **Documentation**
   - Updated API docs with migration notes
   - Deprecation warnings in rustdoc

4. **Crates.io**
   - Version bump reflects breaking changes (MAJOR version)
   - Release notes include breaking change summary

### Getting Help

- **GitHub Issues**: Report migration problems
- **GitHub Discussions**: Ask questions about breaking changes
- **Migration Tools**: Automated migration assistance (when available)

## Examples

### Example 1: Renaming Function (Breaking)

**Version 0.2.0** (Pre-1.0)

```rust
// Old API (removed)
// pub fn create_shape(dims: &[usize]) -> Shape { ... }

// New API
pub fn new_shape(dims: &[usize]) -> Shape { ... }
```

**Migration Guide**:
```diff
- let shape = create_shape(&[2, 3, 4]);
+ let shape = new_shape(&[2, 3, 4]);
```

### Example 2: Adding Parameter with Default (Non-Breaking)

**Version 1.5.0** (Post-1.0)

```rust
// Old signature (still works via builder pattern)
pub fn allocate(size: usize) -> Storage { ... }

// New signature (with default)
pub fn allocate_with_options(size: usize, alignment: Option<usize>) -> Storage { ... }

// Old API remains as convenience wrapper
pub fn allocate(size: usize) -> Storage {
    allocate_with_options(size, None)
}
```

### Example 3: Deprecation with Migration Path

**Version 1.8.0** → **2.0.0**

```rust
// Version 1.8.0: Announce deprecation
#[deprecated(since = "1.8.0", note = "Use `Device::try_from_id()` instead")]
pub fn device_from_id(id: usize) -> Device {
    Device::try_from_id(id).expect("Invalid device ID")
}

// Version 1.9.0, 1.10.0: Warnings but still functional

// Version 2.0.0: Removed
// Function no longer available
```

## Tooling

### Automated Deprecation Tracking

ToRSh Core provides runtime deprecation tracking (see `api_compat.rs`):

```rust
use torsh_core::api_compat::{deprecation_warning, DeprecationSeverity};

pub fn old_api() {
    deprecation_warning(
        "old_api",
        "2.0.0",
        DeprecationSeverity::Hard,
        "Use new_api instead"
    );
    // ... existing implementation ...
}
```

### Deprecation Reports

Generate deprecation reports:

```rust
use torsh_core::api_compat::DeprecationReport;

let report = DeprecationReport::generate();
println!("{}", report.format());
// Outputs:
// Deprecation Report
// ==================
// Soft: 3 deprecations
// Hard: 2 deprecations
// Critical: 0 deprecations
```

### Cargo Integration

Check for deprecated API usage:

```bash
# Run with deprecation warnings as errors
RUSTFLAGS="-D deprecated" cargo build

# Generate deprecation report
cargo torsh-deprecation-report
```

## Exceptional Circumstances

### Security Vulnerabilities

Security fixes **may** introduce breaking changes even in PATCH versions:

1. **Document the vulnerability** (after responsible disclosure period)
2. **Explain the breaking change** necessity
3. **Provide migration path** if possible
4. **Bump PATCH version** for critical fixes, MINOR for larger changes

Example:
```
Version 1.2.1 (Security Fix)

SECURITY: Fixed buffer overflow in Shape::from_raw_parts (CVE-2024-XXXXX)

BREAKING CHANGE (Exceptional): Added bounds checking that may reject
previously accepted invalid inputs. This is a necessary security fix.

Migration: Ensure all Shape::from_raw_parts calls use valid pointers
and lengths. Invalid inputs will now return Err instead of causing UB.
```

### Critical Bugs

Critical bugs causing data corruption or undefined behavior may be fixed with breaking changes in PATCH versions:

1. **Document the bug** and its impact
2. **Explain why breaking change is necessary**
3. **Provide workarounds** for affected users
4. **Consider MINOR version** bump if impact is large

## Compliance Verification

### Pre-Release Checklist

Before releasing a version with breaking changes:

- [ ] All breaking changes documented in CHANGELOG.md
- [ ] Migration guide created for each breaking change
- [ ] Deprecation warnings added (if applicable)
- [ ] Version number follows SemVer (MAJOR bump for breaking changes)
- [ ] Release notes drafted
- [ ] Tests updated for new APIs
- [ ] Documentation updated
- [ ] Examples updated
- [ ] CI passes all tests

### Post-Release Monitoring

After releasing breaking changes:

- Monitor GitHub Issues for migration problems
- Update migration guides based on user feedback
- Provide additional examples if needed
- Consider blog post for major breaking changes (1.x → 2.0)

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Rust API Guidelines - Stability](https://rust-lang.github.io/api-guidelines/necessities.html#cargotoml-includes-all-common-metadata-c-metadata)
- [PyTorch Versioning](https://pytorch.org/docs/stable/notes/versioning.html)
- [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md)

## Revision History

- **2025-11-10**: Initial version (0.1.0-alpha.2 era)
  - Defined pre-1.0 and post-1.0 policies
  - Established deprecation process
  - Created migration guide templates
