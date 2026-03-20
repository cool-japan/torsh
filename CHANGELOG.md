# Changelog

All notable changes to ToRSh will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-03-17

### Changed
- Version bump to 0.1.1
- Updated all workspace crate versions

### Fixed
- Dependency version updates

## [0.1.0] - 2026-02-19

### Initial Release

ToRSh (Tensor Operations in Rust with Sharding) is a PyTorch-compatible deep learning framework built entirely in Rust. It provides a comprehensive set of tensor operations, automatic differentiation, neural network layers, and scientific computing integration through the SciRS2 ecosystem.

### Features

#### Core Components

- **Tensor Operations**: ~400 PyTorch-compatible operations
  - Arithmetic, matrix, and reduction operations
  - Advanced indexing, broadcasting, and shape manipulation
  - FFT, complex numbers, sorting, and histograms
  - 13 in-place operations (`add_`, `mul_`, `sub_`, `div_`, `relu_`, `sigmoid_`, `tanh_`, `gelu_`, `leaky_relu_`, `clamp_`, etc.) with method chaining and autograd safety
  - 11 manipulation operations (`stack`, `chunk`, `split`, `flip`, `fliplr`, `flipud`, `roll`, `rot90`, `tile`, `repeat_interleave`, `unflatten`)
  - 6 dimension manipulation operations (`movedim`, `moveaxis`, `swapaxes`, `swapdims`, `broadcast_to`, `expand_as`)
  - 5 advanced reduction operations (`argmax`, `argmin`, `cumsum`, `cumprod`, `prod`)
  - 4 statistical operations (`median`, `median_dim`, `mode`, `mode_dim`) with keepdim support
  - 5 NaN/Inf detection operations (`isnan`, `isinf`, `isfinite`, `allclose`, `isclose`)
  - 3 masked operations (`masked_fill`, `masked_fill_`, `nonzero`)
  - 3 triangular and diagonal operations (`tril`, `triu`, `diagonal`)
  - 3 index operations (`index_add`, `index_copy`, `index_fill`)
  - 9 scatter operations (`scatter`, `scatter_add`, `scatter_reduce`, `diagonal_scatter`, `select_scatter`, `slice_scatter`, `masked_scatter`, `index_put`, `put_`)
  - 2 repeating operations (`repeat`, `repeat_interleave`)
  - 2 utility operations (`unflatten`, `take_along_dim`)
  - Complete PyTorch scatter family coverage
- **Automatic Differentiation**: Complete reverse-mode AD with gradient computation
  - Computation graph tracking
  - Higher-order derivatives
  - Gradient checkpointing
- **Neural Network Layers**: All essential layers
  - Linear, Conv1d/2d/3d, ConvTranspose
  - BatchNorm, LayerNorm, GroupNorm, InstanceNorm
  - RNN, LSTM, GRU, Transformer, MultiheadAttention
  - All common activation functions (ReLU, GELU, SiLU, etc.)
  - Comprehensive pooling operations
- **Optimizers**: 70+ optimizers
  - SGD, Adam, AdamW, AdaGrad, RMSprop, LBFGS
  - Learning rate schedulers (CosineAnnealing, OneCycle, etc.)
  - Advanced optimizers from OptiRS
- **Data Loading**: Parallel data processing
  - Multi-worker DataLoader with csv-based tabular loading
  - Dataset abstractions (TensorDataset, ConcatDataset, etc.)
  - Sampling strategies (Random, Weighted, Distributed, etc.)

#### Scientific Computing (SciRS2 Integration)

- **18 SciRS2 Crates**: Complete scientific ecosystem
  - scirs2-core stable with OxiBLAS
  - scipy.linalg compatibility (35 functions: svd, eig, qr, lu, cholesky, etc.)
  - Graph Neural Networks (GCN, GAT, GraphSAGE)
  - Time Series Analysis (STL, SSA, Kalman filters)
  - Computer Vision operations
  - Sparse tensors (COO, CSR formats)
  - Special functions (Gamma, Bessel, error functions)

#### Advanced Features

- **JIT Compilation**: Cranelift-based compilation with kernel fusion
- **Quantization**: INT8 quantization, QAT, post-training quantization
- **Model Hub**: PyTorch model import, versioning, ONNX compatibility
- **Distributed Training**: DDP, FSDP, collective operations
- **Profiling**: Advanced profiling with metrics collection
- **Multiple Backends**: CPU, CUDA, Metal support

### Technical Architecture

- **29 Workspace Crates**: Modular architecture for flexibility
- **100% Pure Rust** by default (zero C/Fortran dependencies in default features)
- **SciRS2 Policy Compliance**: All numerical operations through scirs2-core
- **Parallel Operations**: Uses `scirs2_core::parallel_ops` for concurrent computation
- **No-warnings Policy**: Strict code quality standards
- **Comprehensive Testing**: Unit tests, integration tests, benchmarks
- System information via `sysinfo` crate (pure Rust)
- Linear algebra via OxiBLAS backend through scirs2-linalg

### Quality Metrics

- **Tests**: 9,000+ tests passing across all crates
- **Zero Warnings**: Clean build across all 29 workspace crates
- **Zero Errors**: All workspace packages compile successfully
- **Stable Dependencies**: All from crates.io (no local patches)

### Dependencies

Built on stable, production-ready dependencies:
- **SciRS2**: Scientific computing platform
- **OxiBLAS**: Optimized BLAS/LAPACK operations
- **OxiCode**: Modern binary serialization
- **OptiRS**: Advanced ML optimization algorithms

### Known Limitations

- **f16/bf16**: Half-precision floating point support coming in a future release
- **Distributed Training**: API stabilization ongoing
- **API Coverage**: Targeting 95%+ PyTorch API coverage for 1.0
- **CUDA**: Requires local CUDA toolkit installation

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
torsh = "0.1.1"
torsh-nn = "0.1.1"      # Neural networks
torsh-vision = "0.1.1"  # Computer vision
```
