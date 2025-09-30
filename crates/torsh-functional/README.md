# torsh-functional

Functional operations for ToRSh tensors, providing PyTorch-compatible functional API.

## Overview

This crate provides a comprehensive set of functional operations that work on tensors:

- **Mathematical Operations**: Element-wise, reduction, and special functions
- **Neural Network Functions**: Activations, normalization, loss functions
- **Linear Algebra**: Matrix operations, decompositions, solvers
- **Signal Processing**: FFT, convolution, filtering
- **Image Operations**: Transforms, filters, augmentations

Note: This crate integrates with various scirs2 modules (scirs2-linalg, scirs2-special, scirs2-signal, scirs2-fft) for optimized implementations.

## Usage

### Mathematical Operations

```rust
use torsh_functional as F;
use torsh_tensor::prelude::*;

// Element-wise operations
let a = tensor![[1.0, 2.0], [3.0, 4.0]];
let b = tensor![[5.0, 6.0], [7.0, 8.0]];

let sum = F::add(&a, &b)?;
let product = F::mul(&a, &b)?;
let power = F::pow(&a, 2.0)?;

// Trigonometric functions
let angles = tensor![0.0, PI/4.0, PI/2.0];
let sines = F::sin(&angles)?;
let cosines = F::cos(&angles)?;

// Reductions
let sum_all = F::sum(&a)?;
let mean = F::mean(&a)?;
let std = F::std(&a, true)?;  // unbiased
let max_vals = F::amax(&a, &[1], true)?;  // keepdim
```

### Neural Network Functions

```rust
// Activation functions
let x = randn(&[10, 20]);
let relu = F::relu(&x)?;
let sigmoid = F::sigmoid(&x)?;
let tanh = F::tanh(&x)?;
let gelu = F::gelu(&x)?;
let swish = F::silu(&x)?;

// Softmax with temperature
let logits = randn(&[32, 10]);
let probs = F::softmax(&logits, -1)?;
let log_probs = F::log_softmax(&logits, -1)?;

// Normalization
let normalized = F::layer_norm(&x, &[20], None, None, 1e-5)?;
let batch_normed = F::batch_norm(&x, None, None, None, None, true, 0.1, 1e-5)?;

// Dropout
let dropped = F::dropout(&x, 0.5, true)?;  // training mode
```

### Loss Functions

```rust
// Classification losses
let logits = model.forward(&input)?;
let targets = tensor![0, 1, 2, 3];

let ce_loss = F::cross_entropy(&logits, &targets, None, "mean", -100)?;
let nll_loss = F::nll_loss(&log_probs, &targets, None, "mean", -100)?;

// Regression losses
let predictions = model.forward(&input)?;
let targets = randn(&predictions.shape());

let mse = F::mse_loss(&predictions, &targets, "mean")?;
let mae = F::l1_loss(&predictions, &targets, "mean")?;
let huber = F::smooth_l1_loss(&predictions, &targets, "mean", 1.0)?;

// Binary classification
let binary_logits = model.forward(&input)?;
let binary_targets = rand(&binary_logits.shape())?;

let bce = F::binary_cross_entropy_with_logits(
    &binary_logits,
    &binary_targets,
    None,
    None,
    "mean"
)?;
```

### Convolution and Pooling

```rust
// 2D Convolution
let input = randn(&[1, 3, 224, 224]);  // NCHW
let weight = randn(&[64, 3, 7, 7]);
let bias = randn(&[64]);

let output = F::conv2d(
    &input,
    &weight,
    Some(&bias),
    &[2, 2],  // stride
    &[3, 3],  // padding
    &[1, 1],  // dilation
    1,        // groups
)?;

// Pooling operations
let pooled = F::max_pool2d(&input, &[2, 2], &[2, 2], &[0, 0], &[1, 1], false)?;
let avg_pooled = F::avg_pool2d(&input, &[2, 2], &[2, 2], &[0, 0], false, true)?;
let adaptive = F::adaptive_avg_pool2d(&input, &[1, 1])?;  // global pooling
```

### Linear Algebra

```rust
// Matrix operations (leveraging scirs2-linalg)
let a = randn(&[10, 20]);
let b = randn(&[20, 30]);

let c = F::matmul(&a, &b)?;
let det = F::det(&square_matrix)?;
let inv = F::inverse(&square_matrix)?;

// Eigenvalues and eigenvectors
let (eigenvalues, eigenvectors) = F::eig(&symmetric_matrix)?;

// SVD
let (u, s, v) = F::svd(&matrix, true, true)?;

// Solve linear systems
let x = F::solve(&a, &b)?;  // Solve Ax = b
```

### Signal Processing

```rust
// FFT operations (leveraging scirs2-signal)
let signal = randn(&[1024]);
let spectrum = F::fft(&signal)?;
let reconstructed = F::ifft(&spectrum)?;

// 2D FFT for images
let image = randn(&[1, 3, 256, 256]);
let freq_domain = F::fft2(&image)?;

// Convolution via FFT
let kernel = randn(&[32]);
let filtered = F::conv1d_fft(&signal, &kernel)?;
```

### Advanced Operations

```rust
// Interpolation
let upsampled = F::interpolate(
    &input,
    Some(&[224, 224]),  // size
    None,               // scale_factor
    "bilinear",         // mode
    Some(true),         // align_corners
)?;

// Affine grid
let theta = tensor![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
let grid = F::affine_grid(&theta, &[1, 1, 32, 32], false)?;

// Grid sampling
let sampled = F::grid_sample(&input, &grid, "bilinear", "zeros", Some(true))?;
```

### Utilities

```rust
// Tensor manipulation
let flattened = F::flatten(&tensor, 1, -1)?;
let reshaped = F::reshape(&tensor, &[-1, 10])?;
let permuted = F::permute(&tensor, &[0, 2, 3, 1])?;

// Padding
let padded = F::pad(&tensor, &[1, 1, 2, 2], "constant", Some(0.0))?;

// Concatenation and stacking
let tensors = vec![&a, &b, &c];
let concatenated = F::cat(&tensors, 0)?;
let stacked = F::stack(&tensors, 1)?;

// Splitting
let chunks = F::chunk(&tensor, 4, 0)?;
let splits = F::split(&tensor, &[10, 20, 30], 0)?;
```

## Integration with SciRS2

This crate leverages multiple scirs2 modules for optimized implementations:

- **scirs2-linalg**: For linear algebra operations (matrix multiplication, decompositions)
- **scirs2-special**: For special mathematical functions (bessel, gamma, etc.)
- **scirs2-signal**: For signal processing operations
- **scirs2-fft**: For Fast Fourier Transform operations
- **scirs2-core**: For SIMD operations and memory management
- **scirs2-neural**: For neural network specific operations

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.