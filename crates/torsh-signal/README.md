# torsh-signal

Signal processing operations for the ToRSh deep learning framework.

## Overview

This crate provides PyTorch-compatible signal processing functionality built on top of the **SciRS2 ecosystem** for high-performance scientific computing. It leverages scirs2-signal and scirs2-fft to deliver state-of-the-art signal processing operations with excellent performance and PyTorch compatibility.

## Features

### Core Signal Processing (Powered by SciRS2)
- **Advanced Spectral Operations**: STFT, ISTFT, spectrograms with scirs2-fft acceleration
- **Professional Window Functions**: Hamming, Hann, Blackman, Kaiser, Gaussian powered by scirs2-signal
- **High-Performance Filtering**: Convolution, correlation with scirs2-signal optimization
- **SciRS2 Integration**: Full access to scirs2-signal's advanced signal processing algorithms

### PyTorch Compatibility
- **Drop-in Replacement**: Compatible API for torch.signal operations
- **Seamless Migration**: Easy transition from PyTorch signal processing
- **Performance Benefits**: Rust-native performance with Python-like API

## Modules

### ToRSh Signal Processing Modules
- `spectral`: STFT, ISTFT, spectrograms, mel-scale filtering (scirs2-fft powered)
- `windows`: Various window functions for signal processing (scirs2-signal powered)
- `filters`: Convolution and correlation operations (scirs2-signal powered)

### SciRS2 Integration
- **scirs2-signal**: Advanced signal processing algorithms and optimizations
- **scirs2-fft**: High-performance FFT operations with SIMD acceleration
- **scirs2-core**: Scientific computing primitives and memory management

## Usage

### Basic Signal Processing with SciRS2 Integration

```rust
use torsh_signal::prelude::*;
use torsh_tensor::Tensor;

// Create a signal
let signal: Tensor<f32> = Tensor::randn(&[1000])?;

// Apply Hann window
let window = hann_window(256, false)?;
let windowed = signal.slice(0, 0, 256)? * &window;

// Compute STFT
let stft_result = stft(
    &signal,
    256,        // n_fft
    Some(64),   // hop_length
    Some(256),  // win_length
    Some(Window::Hann),
    true,       // center
    false,      // normalized
    true,       // onesided
    false,      // return_complex
)?;

// Compute spectrogram
let spec = spectrogram(
    &signal,
    512,        // n_fft
    Some(128),  // hop_length
    Some(512),  // win_length
    Some(Window::Hann),
    0.0,        // power (0.0 for magnitude)
    false,      // normalized
)?;
```

## Dependencies

### Core ToRSh Dependencies
- `torsh-core`: Core types and device abstraction
- `torsh-tensor`: Tensor operations and storage
- `torsh-functional`: Functional API components
- `torsh-linalg`: Linear algebra operations

### SciRS2 Ecosystem Dependencies
- **`scirs2-signal`**: Advanced signal processing algorithms and optimizations
- **`scirs2-fft`**: High-performance FFT operations with SIMD acceleration
- **`scirs2-core`**: Scientific computing primitives and memory management

### Supporting Libraries
- `num-complex`: Complex number support
- `num-traits`: Numeric trait abstractions
- `ndarray`: Multi-dimensional arrays (via scirs2-core)
- `thiserror`: Error handling

## Performance

torsh-signal delivers exceptional performance through the SciRS2 ecosystem:

### SciRS2-Powered Optimizations
- **SIMD-accelerated operations** through scirs2-signal and scirs2-core
- **High-performance FFT** via scirs2-fft with CPU-specific optimizations
- **Memory-efficient processing** with scirs2-core's advanced memory management
- **Vectorized operations** for window functions and filtering

### Advanced Features
- **Automatic optimization selection** based on input size and hardware
- **Parallel processing** capabilities for large-scale signal processing
- **Zero-copy operations** where possible to minimize memory overhead
- **Complex number optimizations** with proper magnitude calculations

## Compatibility

Designed to be a drop-in replacement for PyTorch's signal processing operations:
- `torch.stft` → `torsh_signal::stft`
- `torch.istft` → `torsh_signal::istft`
- `torch.spectrogram` → `torsh_signal::spectrogram`
- Window functions match PyTorch's implementation

## Examples

See the `examples/` directory for comprehensive usage examples and PyTorch compatibility demonstrations.