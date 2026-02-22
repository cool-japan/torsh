# Numerical Accuracy and Stability Documentation

## Overview

The `torsh-special` crate provides high-precision implementations of special mathematical functions with comprehensive accuracy guarantees and numerical stability considerations.

## Accuracy Specifications

### Gamma Functions
- **gamma(x)**: Relative error < 1e-6 for x > 0, < 1e-3 for edge cases
- **lgamma(x)**: Relative error < 1e-6, uses SciRS2 implementation for maximum stability
- **digamma(x)**: Relative error < 1e-6, maintains accuracy for all positive real inputs
- **polygamma(n, x)**: Currently supports n=0 (digamma), higher orders return 0.0 (placeholder)

### Error Functions  
- **erf(x)**: Relative error < 1e-6 for |x| < 10, handles asymptotic behavior correctly
- **erfc(x)**: Relative error < 1e-6, maintains precision for large positive arguments
- **erfcx(x)**: Scaled implementation prevents overflow, relative error < 1e-6
- **erfinv(x)**: Relative error < 1e-6 for x ∈ (-1, 1)

### Bessel Functions
- **J₀(x), J₁(x)**: Relative error < 1e-3 for |x| < 100, uses rational approximations
- **Jₙ(x)**: Forward/backward recurrence based on argument size, error < 1e-3
- **Y₀(x), Y₁(x)**: Handles singularities at x=0, error < 1e-3 for x > 0
- **I₀(x), I₁(x)**: Modified Bessel functions, error < 1e-3, handles large arguments
- **K₀(x), K₁(x)**: Corrected implementations with proper asymptotic behavior
- **Kₙ(x)**: Uses corrected recurrence relation K_{n+1}(x) = K_{n-1}(x) + (2n/x)*K_n(x)

### Special Trigonometric
- **sinc(x)**: Normalized sinc function, handles x=0 singularity, error < 1e-6
- **Fresnel integrals**: S(x) and C(x) with error < 1e-6

## Numerical Stability Features

### Edge Case Handling
- **Domain validation**: All functions validate input domains and return appropriate errors
- **Singularity management**: Proper handling of poles and branch cuts
- **Overflow prevention**: Use of scaled implementations (e.g., erfcx, lgamma)

### Algorithm Selection
- **Argument-dependent methods**: Different algorithms for small/large arguments
- **Asymptotic expansions**: Used for large arguments to maintain precision  
- **Series representations**: Used for small arguments to avoid cancellation

### Error Recovery
- **Input validation**: Comprehensive checking of input ranges
- **Graceful degradation**: Safe fallbacks for extreme inputs
- **NaN propagation**: Proper handling of invalid inputs

## Performance Optimizations

### SIMD Acceleration
- **AVX2/SSE4.1 support**: Vectorized implementations for hot-path functions
- **Fallback implementations**: Scalar versions for unsupported architectures
- **Function coverage**: gamma, erf, exp family with SIMD variants

### Lookup Tables
- **Precomputed values**: Common function values cached for instant retrieval
- **Polynomial coefficients**: Optimized coefficient tables for approximations
- **Memory efficiency**: Compact storage with fast access patterns

### Fast Approximations
- **Controlled accuracy**: Functions with ~0.01-0.1% error for performance-critical code
- **Alternative APIs**: _fast variants available for all major functions
- **Use cases**: Suitable for initialization, rough estimates, or real-time applications

### Smart Caching
- **TTL-based cache**: 5-minute expiration for function results
- **LRU eviction**: Automatic cleanup of old cache entries
- **Statistics tracking**: Hit/miss ratios and performance monitoring
- **Function-specific IDs**: Separate caches for different function types

## Testing and Validation

### Comprehensive Test Suite
- **Mathematical identities**: Verification using known mathematical relationships
- **Reference values**: Comparison against established mathematical constants
- **Boundary conditions**: Testing at domain boundaries and singularities
- **Cross-validation**: Verification between different function implementations

### Numerical Accuracy Tests
- **Relative error bounds**: Systematic verification of accuracy specifications
- **Convergence testing**: Validation of iterative algorithms
- **Stability analysis**: Testing with perturbed inputs
- **Edge case coverage**: Comprehensive testing of corner cases

## Implementation Details

### SciRS2 Integration
- **Core functions**: Gamma, error functions, and Fresnel integrals use SciRS2 backend
- **PyTorch compatibility**: Tensor-based API matching PyTorch special functions
- **Broadcasting support**: Full tensor broadcasting for all operations
- **Device agnostic**: CPU implementations with future GPU backend support

### Mathematical Accuracy
- **Reference implementations**: Based on established numerical recipes and mathematical literature
- **Coefficient optimization**: Polynomial approximations optimized for target precision
- **Recurrence relations**: Mathematically correct formulations (e.g., fixed Kₙ recurrence)
- **Special cases**: Proper handling of function-specific edge cases

## Usage Guidelines

### When to Use Standard Functions
- Default choice for most applications requiring high accuracy
- Scientific computing where precision is critical
- Applications requiring mathematical correctness

### When to Use Optimized Variants
- **SIMD versions**: Large tensor operations where vectorization helps
- **Fast approximations**: Real-time applications with relaxed accuracy requirements  
- **Cached versions**: Repeated evaluations of expensive functions

### Performance vs Accuracy Trade-offs
- Standard functions: Maximum accuracy, moderate performance
- Optimized functions: Good accuracy, better performance
- Fast approximations: Acceptable accuracy, maximum performance

## Future Improvements

### Planned Enhancements
- Complete polygamma function implementation for n > 0
- General Yₙ(x) and higher-order Bessel functions
- Complex number support for all functions
- Additional SIMD coverage for more functions

### Accuracy Improvements
- Higher precision coefficients for polynomial approximations
- Adaptive precision based on input ranges
- Extended precision support for critical applications
- Better asymptotic expansions for extreme arguments