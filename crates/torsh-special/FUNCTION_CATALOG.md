# ToRSh Special Functions Catalog

## Overview

This catalog provides a comprehensive overview of all special functions available in the `torsh-special` crate, including their mathematical definitions, performance characteristics, and usage examples.

## Function Categories

### 1. Gamma Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `gamma(x)` | Gamma function Γ(x) | 1e-6 | High |
| `lgamma(x)` | Log gamma function ln(Γ(x)) | 1e-6 | High |
| `digamma(x)` | Digamma function ψ(x) | 1e-6 | High |
| `polygamma(n, x)` | Polygamma function ψ⁽ⁿ⁾(x) | 1e-6 | Medium |
| `beta(a, b)` | Beta function B(a,b) | 1e-6 | High |

**Performance Benchmarks:**
- `gamma(1000 elements)`: ~50μs (SIMD optimized)
- `lgamma(1000 elements)`: ~45μs (SIMD optimized)
- `digamma(1000 elements)`: ~60μs
- `polygamma(n=5, 1000 elements)`: ~150μs
- `beta(1000 elements)`: ~100μs

### 2. Bessel Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `bessel_j0(x)` | Bessel function J₀(x) | 1e-6 | High |
| `bessel_j1(x)` | Bessel function J₁(x) | 1e-6 | High |
| `bessel_jn(n, x)` | Bessel function Jₙ(x) | 1e-6 | Medium |
| `bessel_y0(x)` | Bessel function Y₀(x) | 1e-6 | High |
| `bessel_y1(x)` | Bessel function Y₁(x) | 1e-6 | High |
| `bessel_yn(n, x)` | Bessel function Yₙ(x) | 1e-6 | Medium |
| `bessel_i0(x)` | Modified Bessel I₀(x) | 1e-6 | High |
| `bessel_i1(x)` | Modified Bessel I₁(x) | 1e-6 | High |
| `bessel_k0(x)` | Modified Bessel K₀(x) | 1e-3 | High |
| `bessel_k1(x)` | Modified Bessel K₁(x) | 1e-3 | High |

**Performance Benchmarks:**
- `bessel_j0(1000 elements)`: ~70μs (SIMD optimized)
- `bessel_j1(1000 elements)`: ~75μs (SIMD optimized)
- `bessel_jn(n=5, 1000 elements)`: ~120μs
- `bessel_y0(1000 elements)`: ~80μs
- `bessel_i0(1000 elements)`: ~65μs
- `bessel_k0(1000 elements)`: ~85μs

### 3. Spherical Bessel Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `spherical_j0(x)` | Spherical Bessel j₀(x) | 1e-6 | High |
| `spherical_j1(x)` | Spherical Bessel j₁(x) | 1e-6 | High |
| `spherical_jn(n, x)` | Spherical Bessel jₙ(x) | 1e-6 | Medium |
| `spherical_y0(x)` | Spherical Bessel y₀(x) | 1e-6 | High |
| `spherical_y1(x)` | Spherical Bessel y₁(x) | 1e-6 | High |
| `spherical_yn(n, x)` | Spherical Bessel yₙ(x) | 1e-6 | Medium |

**Performance Benchmarks:**
- `spherical_j0(1000 elements)`: ~55μs
- `spherical_j1(1000 elements)`: ~60μs
- `spherical_jn(n=5, 1000 elements)`: ~100μs

### 4. Error Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `erf(x)` | Error function erf(x) | 1e-6 | High |
| `erfc(x)` | Complementary error function erfc(x) | 1e-6 | High |
| `erfcx(x)` | Scaled complementary error function | 1e-6 | High |
| `erfinv(x)` | Inverse error function erf⁻¹(x) | 1e-6 | Medium |

**Performance Benchmarks:**
- `erf(1000 elements)`: ~35μs (SIMD optimized)
- `erfc(1000 elements)`: ~40μs (SIMD optimized)
- `erfcx(1000 elements)`: ~45μs
- `erfinv(1000 elements)`: ~120μs

### 5. Fresnel Integrals
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `fresnel_s(x)` | Fresnel sine integral S(x) | 1e-6 | Medium |
| `fresnel_c(x)` | Fresnel cosine integral C(x) | 1e-6 | Medium |

**Performance Benchmarks:**
- `fresnel_s(1000 elements)`: ~90μs
- `fresnel_c(1000 elements)`: ~90μs

### 6. Trigonometric Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `sinc(x)` | Normalized sinc function sinc(x) | 1e-6 | High |
| `sinc_unnormalized(x)` | Unnormalized sinc function | 1e-6 | High |

**Performance Benchmarks:**
- `sinc(1000 elements)`: ~30μs
- `sinc_unnormalized(1000 elements)`: ~30μs

### 7. Elliptic Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `elliptic_k(m)` | Complete elliptic integral K(m) | 1e-6 | Medium |
| `elliptic_e(m)` | Complete elliptic integral E(m) | 1e-6 | Medium |
| `elliptic_f(phi, m)` | Incomplete elliptic integral F(φ,m) | 1e-6 | Medium |
| `elliptic_pi(n, m)` | Complete elliptic integral Π(n,m) | 1e-6 | Medium |
| `jacobi_sn(u, m)` | Jacobi elliptic function sn(u,m) | 1e-6 | Medium |
| `jacobi_cn(u, m)` | Jacobi elliptic function cn(u,m) | 1e-6 | Medium |
| `jacobi_dn(u, m)` | Jacobi elliptic function dn(u,m) | 1e-6 | Medium |

**Performance Benchmarks:**
- `elliptic_k(1000 elements)`: ~80μs
- `elliptic_e(1000 elements)`: ~85μs
- `jacobi_sn(1000 elements)`: ~110μs

### 8. Exponential Integrals
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `exponential_integral_ei(x)` | Exponential integral Ei(x) | 1e-6 | Medium |
| `exponential_integral_en(n, x)` | Exponential integral Eₙ(x) | 1e-6 | Medium |
| `logarithmic_integral(x)` | Logarithmic integral li(x) | 1e-6 | Medium |
| `sine_integral(x)` | Sine integral Si(x) | 1e-6 | Medium |
| `cosine_integral(x)` | Cosine integral Ci(x) | 1e-6 | Medium |

**Performance Benchmarks:**
- `exponential_integral_ei(1000 elements)`: ~95μs
- `exponential_integral_en(1000 elements)`: ~120μs
- `logarithmic_integral(1000 elements)`: ~100μs

### 9. Hypergeometric Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `hypergeometric_2f1(a, b, c, z)` | Gauss hypergeometric ₂F₁ | 1e-6 | Low |
| `hypergeometric_1f1(a, b, z)` | Confluent hypergeometric ₁F₁ | 1e-6 | Low |
| `hypergeometric_u(a, b, z)` | Tricomi's confluent hypergeometric U | 1e-6 | Low |
| `hypergeometric_pfq(a, b, z)` | Generalized hypergeometric ₚFᵩ | 1e-6 | Low |

**Performance Benchmarks:**
- `hypergeometric_2f1(1000 elements)`: ~200μs
- `hypergeometric_1f1(1000 elements)`: ~150μs
- `hypergeometric_u(1000 elements)`: ~180μs

### 10. Orthogonal Polynomials
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `legendre_p(n, x)` | Legendre polynomial Pₙ(x) | 1e-6 | Medium |
| `legendre_q(n, x)` | Legendre function Qₙ(x) | 1e-6 | Medium |
| `associated_legendre_p(n, m, x)` | Associated Legendre Pₙᵐ(x) | 1e-6 | Medium |
| `chebyshev_t(n, x)` | Chebyshev polynomial Tₙ(x) | 1e-6 | Medium |
| `chebyshev_u(n, x)` | Chebyshev polynomial Uₙ(x) | 1e-6 | Medium |
| `hermite_h(n, x)` | Hermite polynomial Hₙ(x) | 1e-6 | Medium |
| `hermite_he(n, x)` | Probabilists' Hermite Heₙ(x) | 1e-6 | Medium |
| `laguerre_l(n, x)` | Laguerre polynomial Lₙ(x) | 1e-6 | Medium |
| `associated_laguerre_l(n, k, x)` | Associated Laguerre Lₙᵏ(x) | 1e-6 | Medium |

**Performance Benchmarks:**
- `legendre_p(n=10, 1000 elements)`: ~90μs
- `chebyshev_t(n=10, 1000 elements)`: ~80μs
- `hermite_h(n=10, 1000 elements)`: ~85μs
- `laguerre_l(n=10, 1000 elements)`: ~85μs

### 11. Advanced Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `riemann_zeta(s)` | Riemann zeta function ζ(s) | 1e-6 | Medium |
| `dirichlet_eta(s)` | Dirichlet eta function η(s) | 1e-6 | Medium |
| `hurwitz_zeta(s, a)` | Hurwitz zeta function ζ(s,a) | 1e-6 | Low |
| `polylogarithm(s, z)` | Polylogarithm Liₛ(z) | 1e-6 | Low |
| `barnes_g(z)` | Barnes G-function G(z) | 1e-6 | Low |

**Performance Benchmarks:**
- `riemann_zeta(1000 elements)`: ~130μs
- `dirichlet_eta(1000 elements)`: ~120μs
- `hurwitz_zeta(1000 elements)`: ~200μs
- `polylogarithm(1000 elements)`: ~180μs
- `barnes_g(1000 elements)`: ~250μs

### 12. Statistical Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `incomplete_beta(a, b, x)` | Incomplete beta function | 1e-6 | Medium |
| `student_t_cdf(t, df)` | Student's t CDF | 1e-6 | Medium |
| `chi_squared_cdf(x, df)` | Chi-squared CDF | 1e-6 | Medium |
| `f_distribution_cdf(x, df1, df2)` | F-distribution CDF | 1e-6 | Medium |
| `normal_cdf(x)` | Standard normal CDF | 1e-6 | High |
| `normal_pdf(x)` | Standard normal PDF | 1e-6 | High |

**Performance Benchmarks:**
- `incomplete_beta(1000 elements)`: ~140μs
- `student_t_cdf(1000 elements)`: ~120μs
- `chi_squared_cdf(1000 elements)`: ~110μs
- `normal_cdf(1000 elements)`: ~50μs
- `normal_pdf(1000 elements)`: ~45μs

### 13. Complex Functions
| Function | Description | Accuracy | Performance Level |
|----------|-------------|----------|------------------|
| `complex_gamma_c64(z)` | Complex gamma function (f64) | 1e-10 | Medium |
| `complex_gamma_c32(z)` | Complex gamma function (f32) | 1e-6 | Medium |
| `complex_zeta_c64(z)` | Complex zeta function (f64) | 1e-10 | Low |
| `complex_zeta_c32(z)` | Complex zeta function (f32) | 1e-6 | Low |
| `complex_erf_c64(z)` | Complex error function (f64) | 1e-10 | Medium |
| `complex_erf_c32(z)` | Complex error function (f32) | 1e-6 | Medium |

**Performance Benchmarks:**
- `complex_gamma_c64(1000 elements)`: ~180μs
- `complex_gamma_c32(1000 elements)`: ~120μs
- `complex_zeta_c64(1000 elements)`: ~300μs
- `complex_erf_c64(1000 elements)`: ~150μs

## Performance Optimization Features

### 1. SIMD Optimizations
**Available for:** `gamma`, `erf`, `bessel_j0`, `bessel_j1`
- Uses AVX2/SSE4.1 instructions when available
- 2-4x performance improvement on supported hardware
- Automatic fallback to scalar implementations

### 2. Lookup Tables
**Available for:** Common values of `gamma`, `erf`, `bessel_j0`, factorials
- Instant lookup for frequently used values
- Interpolation for nearby values
- Memory-efficient implementation

### 3. Fast Approximations
**Available for:** `gamma_fast`, `erf_fast`, `log_fast`, `exp_fast`, `sin_fast`, `cos_fast`
- 0.01-0.1% accuracy trade-off for 5-10x speed improvement
- Suitable for applications where speed is critical
- Polynomial and rational approximations

### 4. Smart Caching
**Available for:** Expensive functions like `barnes_g`, `polylogarithm`, `hurwitz_zeta`
- LRU cache with TTL (Time To Live)
- Automatic cache management
- Configurable cache sizes

## Usage Examples

### Basic Usage
```rust
use torsh_special::*;
use torsh_tensor::Tensor;
use torsh_core::device::DeviceType;

// Create input tensor
let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;

// Compute gamma function
let result = gamma(&x)?;
let values = result.data()?;
println!("Gamma values: {:?}", values);
```

### Performance Optimized Usage
```rust
use torsh_special::*;

// Use SIMD optimized version
let result = gamma_simd(&x)?;

// Use fast approximation
let result = gamma_fast(&x)?;

// Use cached computation
let result = cached_compute("gamma", &x, |x| gamma(x))?;
```

### Complex Number Support
```rust
use torsh_special::*;
use num_complex::Complex64;

// Create complex tensor
let z = Tensor::from_data(
    vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.3)], 
    vec![2], 
    DeviceType::Cpu
)?;

// Compute complex gamma
let result = complex_gamma_c64(&z)?;
```

## Accuracy Notes

- **High accuracy functions** (1e-6 to 1e-10): Suitable for scientific computing
- **Medium accuracy functions** (1e-3 to 1e-6): Good for most applications
- **Fast approximations** (0.01-0.1% error): Suitable for real-time applications
- **Complex functions** maintain accuracy across the complex plane
- **Edge cases** are handled with appropriate error values (NaN, infinity)

## Performance Tips

1. **Use SIMD versions** for large arrays when available
2. **Enable fast approximations** for performance-critical code
3. **Leverage caching** for repeated computations with expensive functions
4. **Use appropriate precision** (f32 vs f64) based on requirements
5. **Batch computations** rather than individual element processing
6. **Profile your code** to identify bottlenecks in special function usage

## Hardware Requirements

- **CPU**: Any x86_64 processor
- **SIMD Support**: AVX2/SSE4.1 recommended for optimal performance
- **Memory**: Minimal overhead, ~1MB for lookup tables
- **Precision**: IEEE 754 compliant floating-point arithmetic

## Integration with ToRSh Ecosystem

All functions are fully integrated with the ToRSh ecosystem:
- **Tensor API**: All functions work with ToRSh tensors
- **Device Support**: CPU backend with GPU support planned
- **Broadcasting**: Automatic broadcasting for tensor operations
- **Autograd**: Gradients available for differentiable functions
- **JIT Compilation**: Compatible with ToRSh JIT compiler
- **Distributed**: Works with ToRSh distributed training

This catalog is continuously updated as new functions are added and optimizations are implemented.