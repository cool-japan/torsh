# Known Issues and Limitations

This document catalogs known issues, limitations, and workarounds for the torsh-functional crate. Understanding these limitations helps users write robust code and provides a roadmap for future improvements.

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [In-Place Operations](#in-place-operations)
3. [Numerical Stability](#numerical-stability)
4. [Performance Limitations](#performance-limitations)
5. [API Differences from PyTorch](#api-differences-from-pytorch)
6. [Memory Management](#memory-management)
7. [Backend Support](#backend-support)
8. [Future Work](#future-work)

---

## Linear Algebra

### Issue: Eigenvalue Decomposition for Degenerate Cases

**Status**: Known Limitation
**Severity**: Medium
**Affected Functions**: `linalg::eig`, `linalg::matrix_rank`, `linalg::cond`

**Description**:
The current eigenvalue decomposition implementation uses power iteration with deflation, which may not find all eigenvalues for degenerate cases where multiple eigenvalues are equal (e.g., identity matrices).

**Example**:
```rust
let identity = eye::<f32>(4)?;
let rank = matrix_rank(&identity, None)?;
// Returns 2 instead of expected 4 due to eigenvalue algorithm limitation
```

**Impact**:
- `matrix_rank` may underestimate rank for matrices with repeated eigenvalues
- `cond` (condition number) may be inaccurate for such matrices
- Affects matrices with high symmetry (identity, diagonal with equal values)

**Workaround**:
1. For identity matrices, use theoretical rank (matrix dimension)
2. For diagonal matrices, count non-zero diagonal elements
3. Use external libraries (scirs2-linalg with full LAPACK support) when available

**Planned Fix**:
- Implement QR algorithm for eigenvalue computation
- Add specialized handling for diagonal and symmetric matrices
- Integration with optimized BLAS/LAPACK libraries

**Reference**: `src/linalg/mod.rs:142`, `src/linalg/properties.rs`

---

### Issue: SVD Shape Conventions

**Status**: Resolved (Documented)
**Severity**: Low
**Affected Functions**: `linalg::svd`

**Description**:
The SVD implementation returns reduced form by default, which may differ from expectations for users familiar with full SVD.

**Example**:
```rust
let matrix = randn(&[4, 3])?;
let (u, s, vt) = svd(&matrix, false)?;
// U shape: [4, 3] (not [4, 4] - this is reduced SVD)
// S shape: [3]
// V^T shape: [3, 3]
```

**Workaround**:
Use `full_matrices=true` parameter for full SVD decomposition when needed.

**Documentation**: See `src/linalg/decompositions.rs` for detailed SVD shape documentation.

---

## In-Place Operations

### Issue: Limited In-Place Operation Support

**Status**: Known Limitation (Awaiting Tensor Mutation API)
**Severity**: Medium
**Affected Functions**: All activation functions with `_` suffix, dropout operations

**Description**:
Current tensor implementation doesn't support true in-place modification. Functions with `inplace=true` parameter currently create new tensors internally.

**Example**:
```rust
// Currently behaves the same as out-of-place version
let result = relu_(&input)?;  // Creates new tensor despite _ suffix
```

**Impact**:
- Higher memory usage than expected
- Performance degradation in memory-constrained environments
- API inconsistency with PyTorch conventions

**Workaround**:
1. Explicitly manage memory by reusing variables
2. Process data in smaller batches to reduce peak memory
3. Use explicit cloning when semantic clarity is needed

**Planned Fix**:
Implement true in-place operations once tensor mutation API is available in torsh-tensor.

**Reference**: `src/utils.rs:281`, `src/dropout.rs:39`, `src/activations/inplace.rs`

---

## Numerical Stability

### Issue: Softmax Overflow for Large Values

**Status**: Mitigated (Implementation Uses Max Subtraction)
**Severity**: Low
**Affected Functions**: `softmax`, `log_softmax`

**Description**:
Naive softmax implementation can overflow for large input values due to exponential computation.

**Current Implementation**:
The crate uses the numerically stable "max subtraction" technique:
```rust
max_val = max(input)
softmax = exp(input - max_val) / sum(exp(input - max_val))
```

**Still Possible Issues**:
- Underflow when input values are extremely small
- Precision loss for very large input ranges

**Best Practices**:
1. Normalize input values before softmax
2. Use `log_softmax` for numerical stability in loss computation
3. Clip extreme values if necessary

---

### Issue: LogSumExp Precision

**Status**: Resolved
**Severity**: Low
**Affected Functions**: `logsumexp`

**Description**:
Previous implementation had overflow issues. Now uses stable computation:
```rust
logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
```

**Reference**: Fixed in previous session (scalar tensor handling)

---

## Performance Limitations

### Issue: Suboptimal Convolution Performance for Small Kernels

**Status**: Known Limitation
**Severity**: Medium
**Affected Functions**: `conv1d`, `conv2d`, `conv3d`

**Description**:
Current implementation uses direct convolution without im2col optimization or Winograd transforms.

**Impact**:
- 2-5x slower than optimized libraries for 3×3 kernels
- Significant performance degradation for large batch sizes

**Workaround**:
1. Use backend with optimized convolution (when available)
2. Consider grouped convolutions for large channel counts
3. Use depthwise separable convolutions when applicable

**Planned Improvements**:
- Im2col transformation for better cache utilization
- Winograd convolution for 3×3 kernels
- Integration with cuDNN for GPU backends

---

### Issue: Limited SIMD Utilization

**Status**: Partial Implementation
**Severity**: Medium
**Affected Functions**: Element-wise operations, reductions

**Description**:
While some operations use SIMD through backends, many functional operations don't leverage vectorization.

**Impact**:
- 2-4x slower than hand-optimized SIMD code
- Suboptimal performance on modern CPUs

**Workaround**:
Enable backend SIMD features where available:
```toml
[features]
simd = ["dep:torsh-backend"]
```

**Planned Improvements**:
- Broader SIMD adoption through scirs2-core
- Auto-vectorization hints for compiler
- Platform-specific optimizations (AVX2, AVX-512, NEON)

---

### Issue: Flash Attention Not Implemented

**Status**: TODO
**Severity**: High (for Transformer Workloads)
**Affected Functions**: `scaled_dot_product_attention`, `flash_attention`

**Description**:
Current attention implementation uses standard attention mechanism without memory-efficient Flash Attention optimization.

**Impact**:
- O(n²) memory complexity instead of O(n)
- Slower for long sequences (>2048 tokens)
- Cannot handle very long sequences (>8192) without OOM

**Workaround**:
1. Use gradient checkpointing for training
2. Process in smaller chunks for inference
3. Use sliding window attention for very long sequences

**Planned Implementation**:
```rust
// Future API
let output = flash_attention(
    &query,
    &key,
    &value,
    block_size: 256,  // GPU block size optimization
    causal: true,
)?;
```

**Reference**: `src/attention.rs:516-531`

---

## API Differences from PyTorch

### Issue: Error Handling Style

**Status**: By Design
**Severity**: N/A
**Affected**: All functions

**Description**:
ToRSh uses Rust's `Result<T, E>` for error handling, while PyTorch raises exceptions.

**PyTorch**:
```python
try:
    result = torch.matmul(a, b)
except RuntimeError as e:
    print(f"Error: {e}")
```

**ToRSh**:
```rust
match matmul(&a, &b) {
    Ok(result) => { /* success */ },
    Err(e) => eprintln!("Error: {}", e),
}
// Or more commonly:
let result = matmul(&a, &b)?;
```

**Migration Tip**: Use `?` operator for ergonomic error propagation.

---

### Issue: Parameter Naming Conventions

**Status**: By Design (Rust Conventions)
**Severity**: Low

**Differences**:
| PyTorch | ToRSh | Reason |
|---------|-------|--------|
| `input` | `input` | Same |
| `keepdim` | `keep_dim` | Rust snake_case |
| `inplace` | `inplace` | Same |
| `dtype` | `dtype` (as enum) | Type safety |

**Migration**: Most names are similar; use IDE autocomplete for exact parameter names.

---

### Issue: Device Handling

**Status**: Simplified
**Severity**: Low

**PyTorch**:
```python
device = torch.device("cuda:0")
tensor = tensor.to(device)
```

**ToRSh**:
```rust
use torsh_core::device::DeviceType;
let tensor = tensor.to_device(DeviceType::Cuda)?;
```

**Difference**: ToRSh uses enum instead of string-based device specification for type safety.

---

## Memory Management

### Issue: No Automatic Memory Pooling

**Status**: Known Limitation
**Severity**: Medium
**Affected**: All tensor operations

**Description**:
Unlike PyTorch's caching allocator, ToRSh doesn't pool memory allocations, leading to more allocations.

**Impact**:
- Slower for workloads with many small tensors
- Potential memory fragmentation

**Workaround**:
1. Reuse tensors when possible
2. Pre-allocate tensors for iterative algorithms
3. Use in-place operations (when available)

**Example**:
```rust
// Instead of creating new tensors each iteration
for i in 0..1000 {
    let temp = operation(&input)?;  // Allocates
}

// Reuse output tensor
let mut output = zeros_like(&input)?;
for i in 0..1000 {
    operation_into(&input, &mut output)?;  // Reuses
}
```

---

### Issue: Peak Memory During Gradient Computation

**Status**: Expected Behavior
**Severity**: Medium
**Affected**: Training workflows

**Description**:
Autograd graph stores intermediate activations, causing peak memory during backward pass.

**Workaround**:
1. Use gradient checkpointing (recompute activations)
2. Process smaller batches
3. Clear gradients explicitly after optimizer step

---

## Backend Support

### Issue: Limited GPU Support

**Status**: In Development
**Severity**: High
**Affected**: All operations

**Description**:
CUDA backend is partially implemented. Many operations fall back to CPU.

**Current Status**:
- ✅ Basic tensor operations (add, mul, etc.)
- ⚠️ Matrix multiplication (basic, not optimized)
- ❌ Convolution operations
- ❌ Advanced operations (attention, etc.)

**Workaround**:
Explicitly check for GPU operation support and provide CPU fallback:
```rust
let device = if tensor.device().is_cuda() && supports_cuda_op() {
    DeviceType::Cuda
} else {
    DeviceType::Cpu
};
```

---

### Issue: No Metal Backend (Apple Silicon)

**Status**: Planned
**Severity**: Medium
**Affected**: macOS users with Apple Silicon

**Description**:
No native Metal Performance Shaders (MPS) backend for Apple Silicon GPUs.

**Workaround**:
Use CPU backend (which benefits from Apple's Accelerate framework on macOS).

---

### Issue: WebGPU Backend Not Available

**Status**: Future Work
**Severity**: Low
**Affected**: Web/WASM deployments

**Description**:
No WebGPU backend for browser-based inference.

**Alternative**:
Consider WASM compilation with CPU backend for web deployment.

---

## Future Work

### Planned Enhancements

1. **Linear Algebra Improvements**
   - QR algorithm for eigenvalue decomposition
   - Specialized handlers for symmetric matrices
   - Full LAPACK integration via scirs2-linalg

2. **Performance Optimizations**
   - Flash Attention implementation
   - Im2col for convolutions
   - Broader SIMD adoption
   - Memory pooling/caching allocator

3. **Backend Expansion**
   - Complete CUDA backend
   - Metal backend for Apple Silicon
   - WebGPU for browser deployment
   - ROCm for AMD GPUs

4. **API Enhancements**
   - True in-place operations
   - More comprehensive gradient checkpointing
   - Improved error messages with suggestions

5. **Testing & Validation**
   - Cross-platform compatibility tests
   - Performance regression detection
   - Numerical accuracy benchmarks vs PyTorch

---

## Reporting Issues

If you encounter an issue not listed here:

1. **Search existing issues**: Check [GitHub Issues](https://github.com/cool-japan/torsh/issues)
2. **Provide minimal reproduction**: Include code that demonstrates the issue
3. **Include environment details**: OS, Rust version, dependency versions
4. **Expected vs actual behavior**: Describe what you expected and what happened

**Template**:
```markdown
## Issue Summary
Brief description of the issue

## Environment
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.75.0]
- torsh-functional version: [e.g., 0.1.0-alpha.2]

## Reproduction
\`\`\`rust
// Minimal code to reproduce
\`\`\`

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Workaround (if known)
Any temporary solutions
```

---

## Contributing

Contributions to fix these issues are welcome! Please see:
- [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines
- [TODO.md](TODO.md) for prioritized work items
- [CLAUDE.md](../CLAUDE.md) for development guidance

When fixing an issue, please:
1. Add tests demonstrating the fix
2. Update this document to mark issue as resolved
3. Add migration notes if API changes are required
