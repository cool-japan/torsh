//! Special mathematical functions for ToRSh

#![allow(clippy::result_large_err)]

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

use torsh_core::Result;
use torsh_tensor::Tensor;

/// Convenience type alias for Results in this crate
pub type TorshResult<T> = Result<T>;

pub mod advanced;
pub mod advanced_special;
pub mod airy;
pub mod bessel;
pub mod complex;
pub mod constants;
pub mod coulomb;
pub mod elliptic;
pub mod error_functions;
pub mod error_handling;
pub mod exponential_integrals;
pub mod fast_approximations;
pub mod gamma;
pub mod hypergeometric;
pub mod lambert_w;
pub mod lommel;
pub mod lookup_tables;
pub mod mathieu;
pub mod numerical_accuracy_tests;
pub mod orthogonal_polynomials;
pub mod scirs2_integration;
pub mod simd_optimizations;
pub mod smart_caching;
pub mod spheroidal;
pub mod statistical;
pub mod trigonometric;
pub mod utils;
pub mod visualization;

#[cfg(test)]
pub mod test_fix;

// Re-exports
pub use bessel::*;
// Use scirs2_integration versions for better PyTorch compatibility
pub use scirs2_integration::{
    bessel_i0_scirs2, bessel_i1_scirs2, bessel_j0_scirs2, bessel_j1_scirs2, bessel_jn_scirs2,
    bessel_k0_scirs2, bessel_k1_scirs2, bessel_y0_scirs2, bessel_y1_scirs2, bessel_yn_scirs2, beta,
    digamma, erf, erfc, erfcx, erfinv, fresnel, fresnel_c, fresnel_s, gamma, lgamma, polygamma,
    sinc,
};
// Export elliptic functions
pub use elliptic::{
    elliptic_e, elliptic_e_incomplete, elliptic_f, elliptic_k, jacobi_cn, jacobi_dn, jacobi_sn,
    theta_1, theta_2, theta_3, theta_4, weierstrass_p, weierstrass_sigma, weierstrass_zeta,
};
// Export exponential integrals
pub use exponential_integrals::{
    cosine_integral, exponential_integral_e1, exponential_integral_ei, exponential_integral_en,
    hyperbolic_cosine_integral, hyperbolic_sine_integral, logarithmic_integral, sine_integral,
};
// Export hypergeometric functions
pub use hypergeometric::{
    appell_f1, hypergeometric_1f1, hypergeometric_2f1, hypergeometric_pfq, hypergeometric_u,
    meijer_g,
};
// Export orthogonal polynomials
pub use orthogonal_polynomials::{
    chebyshev_t, chebyshev_u, gegenbauer_c, hermite_h, hermite_he, jacobi_p, laguerre_l,
    laguerre_l_associated, legendre_p, legendre_p_associated,
};
// Export advanced functions
pub use advanced::{barnes_g, dirichlet_eta, hurwitz_zeta, polylogarithm, riemann_zeta};
// Export advanced special functions
pub use advanced_special::{
    dawson, kelvin_bei, kelvin_ber, kelvin_kei, kelvin_ker, parabolic_cylinder_d, spence, struve_h,
    struve_l, voigt_profile,
};
// Export Airy functions
pub use airy::{airy_ai, airy_ai_prime, airy_bi, airy_bi_prime};
// Export Coulomb wave functions
pub use coulomb::{coulomb_f, coulomb_g, coulomb_sigma};
// Export Lommel functions
pub use lommel::{lommel_S, lommel_s, lommel_u, lommel_v};
// Export Mathieu functions
pub use mathieu::{mathieu_Ce, mathieu_Se, mathieu_a, mathieu_b, mathieu_ce, mathieu_se};
// Export Spheroidal Wave functions
pub use spheroidal::{
    oblate_angular, oblate_angular_tensor, oblate_radial, oblate_radial_tensor, prolate_angular,
    prolate_angular_tensor, prolate_radial, prolate_radial_tensor, spheroidal_eigenvalue,
};
// Export Lambert W functions
pub use lambert_w::{
    lambert_w, lambert_w_complex, lambert_w_derivative, lambert_w_principal, lambert_w_secondary,
};
// Export Lambert W applications
pub use lambert_w::applications::{
    solve_exponential_equation, solve_x_to_x_equals_c, tree_function, wright_omega,
};
// Export statistical functions
pub use statistical::{
    chi_squared_cdf, f_distribution_cdf, incomplete_beta, normal_cdf, normal_cdf_general,
    normal_pdf, normal_pdf_general, student_t_cdf,
};
// Export complex functions
pub use complex::{
    complex_airy_ai_c64,
    complex_airy_bi_c64,
    complex_bessel_j_c32,
    complex_bessel_j_c64,
    complex_bessel_y_c32,
    complex_bessel_y_c64,
    complex_beta_c32,
    complex_beta_c64,
    complex_erf_c32,
    complex_erf_c64,
    complex_erfc_c32,
    complex_erfc_c64,
    complex_gamma_c32,
    complex_gamma_c64,
    complex_incomplete_gamma_c64,
    complex_log_principal,
    complex_polygamma_c32,
    // New complex functions
    complex_polygamma_c64,
    complex_pow_principal,
    complex_sqrt_principal,
    complex_zeta_c32,
    complex_zeta_c64,
};
// Export enhanced error handling
pub use error_handling::{
    error_recovery::{clamp_to_finite, gradual_clamp, replace_problematic_values},
    safe_functions::{
        safe_bessel_j0, safe_bessel_j1, safe_bessel_y0, safe_bessel_y1, safe_erf, safe_erfc,
        safe_gamma, safe_lgamma,
    },
    DomainConstraints, InputValidation,
};
// Export performance optimizations
pub use fast_approximations::{
    atanh_fast, bessel_j0_fast, cos_fast, cosh_fast, erf_fast, erfc_fast, exp_fast, gamma_fast,
    log_fast, sin_fast, sinh_fast, tanh_fast,
};
pub use lookup_tables::{
    bessel_j0_optimized, erf_optimized, factorial, gamma_optimized, POLY_COEFFS,
};
pub use simd_optimizations::{erf_simd, exp_family_simd, gamma_simd, ExpVariant};
// Export smart caching functionality
pub use smart_caching::{
    cache_stats, cached_compute, clear_cache, function_ids, CacheStats, FloatKey, SmartCache,
};
// Export non-overlapping functions from other modules
pub use trigonometric::sinc as sinc_unnormalized;
// Keep error_functions and gamma modules available for reference
// but don't re-export to avoid conflicts

/// Exponential minus one (exp(x) - 1) for better numerical stability
pub fn expm1(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // expm1(x) = exp(x) - 1
    let exp_tensor = tensor.exp()?;
    let ones = tensor.ones_like()?;
    exp_tensor.sub(&ones)
}

/// Natural logarithm of one plus x (log(1 + x)) for better numerical stability  
pub fn log1p(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // log1p(x) = log(1 + x)
    let ones = tensor.ones_like()?;
    let one_plus_x = ones.add_op(tensor)?;
    one_plus_x.log()
}

/// Hyperbolic sine (using existing implementation)
pub fn sinh(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    tensor.sinh()
}

/// Hyperbolic cosine (using existing implementation)
pub fn cosh(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    tensor.cosh()
}

/// Hyperbolic tangent (using existing implementation)
pub fn tanh_special(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    tensor.tanh()
}

/// Inverse hyperbolic sine
pub fn asinh(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // asinh(x) = log(x + sqrt(x^2 + 1))
    let x_squared = tensor.mul_op(tensor)?;
    let ones = tensor.ones_like()?;
    let under_sqrt = x_squared.add_op(&ones)?;
    let sqrt_part = under_sqrt.sqrt()?;
    let sum = tensor.add_op(&sqrt_part)?;
    sum.log()
}

/// Inverse hyperbolic cosine
pub fn acosh(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // acosh(x) = log(x + sqrt(x^2 - 1))
    let x_squared = tensor.mul_op(tensor)?;
    let ones = tensor.ones_like()?;
    let under_sqrt = x_squared.sub(&ones)?;
    let sqrt_part = under_sqrt.sqrt()?;
    let sum = tensor.add_op(&sqrt_part)?;
    sum.log()
}

/// Inverse hyperbolic tangent
pub fn atanh(tensor: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // atanh(x) = 0.5 * log((1 + x) / (1 - x))
    let ones = tensor.ones_like()?;
    let one_plus_x = ones.add_op(tensor)?;
    let one_minus_x = ones.sub(tensor)?;
    let ratio = one_plus_x.div(&one_minus_x)?;
    let log_ratio = ratio.log()?;
    log_ratio.mul_scalar(0.5)
}

// Export visualization tools
pub use visualization::{
    analyze_function_behavior, benchmark_optimization_levels, compare_function_accuracy,
    generate_ascii_plot, print_accuracy_report, print_analysis_report, AccuracyComparison,
    FunctionAnalysis, Monotonicity, PlotData,
};
// Export mathematical constants
pub use constants::{
    APERY, CATALAN, DOTTIE, EULER_GAMMA, FEIGENBAUM_ALPHA, FEIGENBAUM_DELTA, GLAISHER_KINKELIN,
    GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL, KHINCHIN, LEVY, LN_2PI, LN_PI, MILLS, PI_SQUARED_OVER_6,
    PLASTIC_NUMBER, RAMANUJAN, SQRT_2PI,
};
// Export utility constants
pub use constants::utility::{
    HALF_LN_2PI, INV_2PI, INV_SQRT_2PI, LN_4PI, LOG_SQRT_2PI, PI_OVER_4, SQRT_PI_OVER_2,
    SQRT_PI_OVER_8, STIRLING_CONSTANT, THREE_PI_OVER_4, TWO_OVER_SQRT_PI,
};
// Export specialized constant collections for advanced users
pub use constants::{bernoulli, fractions, physics, zeta_values};
// Export utility functions
pub use utils::{
    chebyshev_expansion, chebyshev_expansion_tensor, continued_fraction, continued_fraction_tensor,
    double_factorial, factorial_safe, pade_approximant, pade_approximant_tensor,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        advanced::*, advanced_special::*, airy::*, bessel::*, coulomb::*, elliptic::*,
        exponential_integrals::*, hypergeometric::*, lambert_w::*, lommel::*, mathieu::*,
        orthogonal_polynomials::*, scirs2_integration::*, spheroidal::*,
    };
}
