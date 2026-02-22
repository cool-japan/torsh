//! Complex special functions module
//!
//! This module provides implementations of various complex special functions
//! organized by mathematical category for better maintainability.

pub mod bessel;
pub mod elementary;
pub mod error;
pub mod gamma;
pub mod special;
pub mod zeta;

// Re-export all public functions for backwards compatibility
pub use bessel::{
    complex_bessel_j_c32, complex_bessel_j_c64, complex_bessel_y_c32, complex_bessel_y_c64,
};
pub use elementary::{
    complex_cos, complex_exp, complex_log_principal, complex_pow_principal, complex_sin,
    complex_sqrt_principal,
};
pub use error::{complex_erf_c32, complex_erf_c64, complex_erfc_c32, complex_erfc_c64};
pub use gamma::{
    complex_beta_c32, complex_beta_c64, complex_gamma_c32, complex_gamma_c64,
    complex_incomplete_gamma_c64, complex_polygamma_c32, complex_polygamma_c64,
};
pub use special::{complex_airy_ai_c64, complex_airy_bi_c64};
pub use zeta::{complex_zeta_c32, complex_zeta_c64};
