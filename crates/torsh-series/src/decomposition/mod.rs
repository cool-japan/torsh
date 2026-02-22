//! Time series decomposition methods
//!
//! This module provides various time series decomposition techniques:
//! - STL (Seasonal-Trend decomposition using LOESS)
//! - SSA (Singular Spectrum Analysis)
//! - MSTL (Multiple Seasonal-Trend decomposition using LOESS)
//! - Classical methods (X11, additive, multiplicative)
//! - Wavelet decomposition (DWT, CWT, wavelet packets)
//! - EMD (Empirical Mode Decomposition)
//! - VMD (Variational Mode Decomposition)

pub mod classical;
pub mod emd;
pub mod seasonal;
pub mod ssa;
pub mod stl;
pub mod vmd;
pub mod wavelet;

// Re-export main types for easy access
pub use classical::{AdditiveDecomposition, MultiplicativeDecomposition, X11Decomposition};
pub use emd::{eemd_decompose, emd_decompose, EMDConfig, EMDResult};
pub use seasonal::{MSTLDecomposition, MSTLResult};
pub use ssa::SSA;
pub use stl::{STLDecomposition, STLResult};
pub use vmd::{vmd_decompose, VMDConfig, VMDResult};
pub use wavelet::{
    CWTAnalyzer, CWTResult, WaveletDecomposer, WaveletDecomposition, WaveletPacketDecomposer,
    WaveletPacketTree,
};
