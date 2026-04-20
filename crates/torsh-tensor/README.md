# torsh-tensor

[![version](https://img.shields.io/badge/version-0.1.2-blue)](https://crates.io/crates/torsh-tensor)

PyTorch-compatible tensor implementation for ToRSh, built on top of scirs2.

## Overview

This crate provides the core `Tensor` type with a familiar PyTorch-like API, wrapping scirs2's powerful autograd functionality.

## Features

- PyTorch-compatible tensor operations
- Automatic differentiation support
- Broadcasting and shape manipulation
- Comprehensive indexing and slicing
- Integration with scirs2 for optimized computation
- **`simd_ops_f32` module** (v0.1.2): zero-allocation SIMD f32 arithmetic (`add_into_f32`, `add_assign_f32`, etc.) and activation functions with PyTorch NaN semantics
- **Real SIMD dispatch** (v0.1.2): `Tensor::add`/`sub`/`mul`/`div` automatically use AVX2/NEON acceleration via scirs2_core for f32 tensors with ≥ 1024 elements
- **Zero-allocation in-place arithmetic** (v0.1.2): `add_`/`sub_`/`mul_`/`div_` dispatch through `simd_*_inplace` — no temporary buffers
- **In-place activation SIMD** (v0.1.2): `relu_`/`leaky_relu_`/`clamp_` route to SIMD helpers for maximum throughput
- **True buffer pool reuse** (v0.1.2): `GlobalMemoryPool::acquire_uninit::<T>()` returns `ReusedBuffer<T>` with zero copy on pool hit
- **`simd` and `parallel` features enabled by default** — no `--features` flag required

## Usage

### Basic Tensor Creation

```rust
use torsh_tensor::prelude::*;

// Create tensors using the tensor! macro
let a = tensor![1.0, 2.0, 3.0];
let b = tensor![[1.0, 2.0], [3.0, 4.0]];

// Create tensors with specific shapes
let zeros = zeros::<f32>(&[3, 4]);
let ones = ones::<f32>(&[2, 3]);
let eye = eye::<f32>(5);

// Random tensors
let uniform = rand::<f32>(&[3, 3]);
let normal = randn::<f32>(&[2, 4]);
```

### Tensor Operations

```rust
// Element-wise operations
let c = a.add(&b)?;
let d = a.mul(&b)?;

// Matrix multiplication
let e = a.matmul(&b)?;

// Reductions
let sum = a.sum();
let mean = a.mean();
let max = a.max();

// Activation functions
let relu = a.relu();
let sigmoid = a.sigmoid();
```

### Shape Manipulation

```rust
// Reshape
let reshaped = a.view(&[2, 3])?;

// Transpose
let transposed = a.t()?;

// Squeeze and unsqueeze
let squeezed = a.squeeze();
let unsqueezed = a.unsqueeze(0)?;
```

### Automatic Differentiation

```rust
// Enable gradient computation
let x = tensor![2.0].requires_grad_(true);

// Forward pass
let y = x.pow(2.0)?.add(&x.mul(&tensor![3.0])?)?;

// Backward pass
y.backward()?;

// Access gradient
let grad = x.grad().unwrap();
```

### Indexing and Slicing

```rust
// Basic indexing
let element = tensor.get(0)?;
let element_2d = tensor.get_2d(1, 2)?;

// Slicing with macros
let slice = tensor.index(&[s![1..5], s![..], s![0..10; 2]])?;

// Boolean masking
let mask = tensor.gt(&zeros)?;
let selected = tensor.masked_select(&mask)?;
```

## Performance

torsh-tensor routes hot arithmetic paths through SIMD automatically when the `simd` feature is active (default since v0.1.2).

- **Element-wise arithmetic** (`add`, `sub`, `mul`, `div`) on f32 tensors with ≥ 1024 elements dispatches through scirs2_core's AVX2 (x86-64) or NEON (AArch64) kernels.
- **In-place variants** (`add_`, `sub_`, `mul_`, `div_`) use `simd_*_inplace` — no intermediate allocation occurs at any tensor size.
- **Activation functions** (`relu_`, `leaky_relu_`, `clamp_`) take the same in-place SIMD path.
- The **global memory pool** (`GlobalMemoryPool`) returns slabs without copying when the requested size matches an existing free buffer (`acquire_uninit::<T>()`).

No special build flags are needed on supported targets; the feature-detection is done at runtime by scirs2_core.

## Recent Changes

### v0.1.2 — 2026-04-20

- Added `simd_ops_f32` module with zero-allocation SIMD f32 arithmetic and activations (PyTorch NaN semantics).
- Wired real SIMD dispatch into `Tensor::add`/`sub`/`mul`/`div` for f32 tensors ≥ 1024 elements (AVX2/NEON via scirs2_core).
- `add_`/`sub_`/`mul_`/`div_` now call `simd_*_inplace` — zero extra allocations.
- `relu_`/`leaky_relu_`/`clamp_` dispatch to SIMD helpers.
- `GlobalMemoryPool::acquire_uninit::<T>()` returns `ReusedBuffer<T>` with no copy on pool hit.
- `simd` and `parallel` features promoted to default features.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.