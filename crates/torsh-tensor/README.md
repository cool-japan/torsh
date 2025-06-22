# torsh-tensor

PyTorch-compatible tensor implementation for ToRSh, built on top of scirs2.

## Overview

This crate provides the core `Tensor` type with a familiar PyTorch-like API, wrapping scirs2's powerful autograd functionality.

## Features

- PyTorch-compatible tensor operations
- Automatic differentiation support
- Broadcasting and shape manipulation
- Comprehensive indexing and slicing
- Integration with scirs2 for optimized computation

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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.