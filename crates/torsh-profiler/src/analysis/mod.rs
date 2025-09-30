//! Performance analysis and optimization recommendations

pub mod ml_analysis;
pub mod optimization;
pub mod regression;

// Re-export analysis types
pub use ml_analysis::*;
pub use optimization::*;
pub use regression::*;
