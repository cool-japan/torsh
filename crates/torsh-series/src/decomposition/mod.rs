//! Time series decomposition methods
//!
//! This module provides various time series decomposition techniques:
//! - STL (Seasonal-Trend decomposition using LOESS)
//! - SSA (Singular Spectrum Analysis)
//! - MSTL (Multiple Seasonal-Trend decomposition using LOESS)
//! - Classical methods (X11, additive, multiplicative)

pub mod classical;
pub mod seasonal;
pub mod ssa;
pub mod stl;

// Re-export main types for easy access
pub use classical::{AdditiveDecomposition, MultiplicativeDecomposition, X11Decomposition};
pub use seasonal::{MSTLDecomposition, MSTLResult};
pub use ssa::SSA;
pub use stl::{STLDecomposition, STLResult};
