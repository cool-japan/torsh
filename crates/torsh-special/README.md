# torsh-special

Special mathematical functions for ToRSh, leveraging scirs2-special for optimized implementations.

## Overview

This crate provides a comprehensive collection of 135+ special mathematical functions organized into 19 mathematical families:

- **Bessel Functions**: J₀, J₁, Jₙ, Y₀, Y₁, Yₙ, I₀, I₁, Iₙ, K₀, K₁, Kₙ (cylindrical), spherical Bessel, Hankel functions
- **Gamma Functions**: Gamma, log-gamma, digamma, polygamma, beta
- **Error Functions**: erf, erfc, erfcx, erfinv, Fresnel integrals
- **Elliptic Functions**: Complete and incomplete elliptic integrals (K, E, F), Jacobi functions (sn, cn, dn), Weierstrass functions, theta functions
- **Exponential Integrals**: Ei, Eₙ, logarithmic integral, sine/cosine integrals
- **Hypergeometric Functions**: ₁F₁, ₂F₁, pFq, Meijer G, Appell F₁
- **Orthogonal Polynomials**: Legendre, Chebyshev, Hermite, Laguerre, Jacobi, Gegenbauer
- **Advanced Functions**: Riemann zeta, polylogarithm, Hurwitz zeta, Dirichlet eta, Barnes G
- **Advanced Special**: Dawson, Kelvin (ber, bei, ker, kei), parabolic cylinder, Spence, Struve, Voigt
- **Airy Functions**: Ai, Bi and their derivatives
- **Coulomb Wave Functions**: F_L(η,ρ), G_L(η,ρ) for quantum scattering
- **Mathieu Functions**: ce_n, se_n, characteristic values for periodic boundaries
- **Lommel Functions**: s_μ,ν, S_μ,ν for diffraction theory
- **Spheroidal Wave Functions**: Prolate and oblate angular/radial functions for electromagnetic scattering
- **Lambert W Functions**: Principal and secondary branches with applications
- **Statistical Functions**: Normal, Student's t, chi-squared, F-distribution CDFs/PDFs, incomplete beta
- **Complex Functions**: Complex gamma, zeta, erf, Bessel functions with branch cuts
- **Performance Optimizations**: SIMD-accelerated, fast approximations, smart caching, lookup tables
- **Visualization Tools**: Function analysis, accuracy comparison, ASCII plotting

## Usage

### Bessel Functions

```rust
use torsh_special::prelude::*;
use torsh_tensor::prelude::*;

// Bessel functions of the first kind
let x = tensor![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let j0 = special::j0(&x)?;  // J₀(x)
let j1 = special::j1(&x)?;  // J₁(x)
let jn = special::jn(3, &x)?; // J₃(x)

// Bessel functions of the second kind
let y0 = special::y0(&x)?;  // Y₀(x)
let y1 = special::y1(&x)?;  // Y₁(x)
let yn = special::yn(3, &x)?; // Y₃(x)

// Modified Bessel functions
let i0 = special::i0(&x)?;  // I₀(x)
let i1 = special::i1(&x)?;  // I₁(x)
let k0 = special::k0(&x)?;  // K₀(x)
let k1 = special::k1(&x)?;  // K₁(x)

// Spherical Bessel functions
let j0_spherical = special::spherical_jn(0, &x)?;
let y0_spherical = special::spherical_yn(0, &x)?;
```

### Gamma and Related Functions

```rust
// Gamma function
let x = tensor![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
let gamma_x = special::gamma(&x)?;

// Log-gamma function (more stable for large values)
let lgamma_x = special::lgamma(&x)?;

// Digamma (psi) function - derivative of log-gamma
let digamma_x = special::digamma(&x)?;

// Polygamma functions
let trigamma = special::polygamma(1, &x)?;  // ψ'(x)
let tetragamma = special::polygamma(2, &x)?; // ψ''(x)

// Beta function
let a = tensor![0.5, 1.0, 2.0];
let b = tensor![1.0, 2.0, 3.0];
let beta_ab = special::beta(&a, &b)?;

// Incomplete gamma functions
let lower_gamma = special::gammainc(&a, &x)?;    // γ(a,x)
let upper_gamma = special::gammaincc(&a, &x)?;   // Γ(a,x)
let regularized = special::gammaincp(&a, &x)?;   // P(a,x)
```

### Error Functions

```rust
// Error function and complementary error function
let x = tensor![-2.0, -1.0, 0.0, 1.0, 2.0];
let erf_x = special::erf(&x)?;
let erfc_x = special::erfc(&x)?;

// Scaled complementary error function (for large x)
let erfcx_x = special::erfcx(&x)?;  // exp(x²) * erfc(x)

// Inverse error functions
let p = tensor![0.1, 0.5, 0.9];
let erfinv_p = special::erfinv(&p)?;
let erfcinv_p = special::erfcinv(&p)?;

// Fresnel integrals
let (s, c) = special::fresnel(&x)?;  // S(x) and C(x)
```

### Elliptic Functions

```rust
// Complete elliptic integrals
let k = tensor![0.0, 0.5, 0.9, 0.99];
let ellipk = special::ellipk(&k)?;  // K(k)
let ellipe = special::ellipe(&k)?;  // E(k)

// Incomplete elliptic integrals
let phi = tensor![0.5, 1.0, 1.5];
let ellipf = special::ellipf(&phi, &k)?;  // F(φ,k)
let ellipe_inc = special::ellipeinc(&phi, &k)?;  // E(φ,k)

// Jacobi elliptic functions
let u = tensor![0.0, 0.5, 1.0, 1.5];
let m = tensor![0.0, 0.5, 0.9];
let (sn, cn, dn) = special::ellipj(&u, &m)?;
```

### Exponential and Logarithmic Integrals

```rust
// Exponential integral
let ei = special::expi(&x)?;  // Ei(x)

// Exponential integral E_n(x)
let e1 = special::expn(1, &x)?;  // E₁(x)
let e2 = special::expn(2, &x)?;  // E₂(x)

// Logarithmic integral
let li = special::li(&x)?;  // li(x)

// Sine and cosine integrals
let si = special::sici(&x)?.0;  // Si(x)
let ci = special::sici(&x)?.1;  // Ci(x)

// Hyperbolic sine and cosine integrals
let shi = special::shichi(&x)?.0;  // Shi(x)
let chi = special::shichi(&x)?.1;  // Chi(x)
```

### Other Special Functions

```rust
// Riemann zeta function
let s = tensor![0.5, 1.5, 2.0, 3.0];
let zeta_s = special::zeta(&s)?;

// Airy functions
let x = tensor![-2.0, -1.0, 0.0, 1.0, 2.0];
let (ai, aip, bi, bip) = special::airy(&x)?;

// Struve functions
let h0 = special::struve(0, &x)?;  // H₀(x)
let h1 = special::struve(1, &x)?;  // H₁(x)

// Hypergeometric functions
let a = tensor![0.5];
let b = tensor![1.0];
let c = tensor![1.5];
let z = tensor![0.1, 0.5, 0.9];
let hyp2f1 = special::hyp2f1(&a, &b, &c, &z)?;

// Legendre polynomials
let n = 3;
let x = tensor![-1.0, -0.5, 0.0, 0.5, 1.0];
let pn = special::legendre(n, &x)?;

// Associated Legendre functions
let m = 1;
let pmn = special::lpmv(m, n, &x)?;
```

### Spheroidal Wave Functions

```rust
use torsh_special::{prolate_angular, prolate_radial, oblate_angular, oblate_radial, spheroidal_eigenvalue};

// Prolate spheroidal wave functions (electromagnetic scattering)
let n = 2;  // Degree
let m = 0;  // Order
let c = 2.0; // Spheroidicity parameter

// Angular function S_nm(c, η) where η ∈ [-1, 1]
let eta = 0.5;
let s_value = prolate_angular(n, m, c, eta)?;

// Radial function R_nm(c, ξ) where ξ ∈ [1, ∞)
let xi = 2.0;
let r_value = prolate_radial(n, m, c, xi)?;

// Oblate spheroidal wave functions (acoustic cavities)
let xi_oblate = 0.5; // ξ ∈ [0, 1] for oblate
let s_oblate = oblate_angular(n, m, c, eta)?;
let r_oblate = oblate_radial(n, m, c, xi_oblate)?;

// Eigenvalues λ_nm(c)
let lambda_0 = spheroidal_eigenvalue(n, m, 0.0)?; // Spherical limit
let lambda = spheroidal_eigenvalue(n, m, c)?;    // Spheroidal
```

### Batch Operations

All functions support batched operations:

```rust
// Batch computation on 2D tensors
let batch_x = randn(&[32, 100]);  // 32 batches of 100 elements
let batch_gamma = special::gamma(&batch_x)?;
let batch_erf = special::erf(&batch_x)?;

// Broadcasting
let x = randn(&[10, 1]);
let y = randn(&[1, 20]);
let beta_xy = special::beta(&x, &y)?;  // Shape: [10, 20]
```

### Complex Number Support

Some functions support complex inputs:

```rust
#[cfg(feature = "complex")]
{
    use num_complex::Complex;
    
    let z = tensor![Complex::new(1.0, 0.5), Complex::new(2.0, -1.0)];
    let gamma_z = special::gamma_complex(&z)?;
    let zeta_z = special::zeta_complex(&z)?;
}
```

## Integration with SciRS2

This crate fully leverages scirs2-special for:
- Optimized implementations of all special functions
- Hardware acceleration where available
- Consistent numerical accuracy
- Efficient vectorized operations

## Numerical Considerations

- Functions are implemented with high numerical accuracy
- Appropriate algorithms are chosen for different input ranges
- Special care is taken near singularities and branch points
- Error bounds are documented for each function

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.