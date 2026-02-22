//! Auto-generated module structure

pub mod algebraicsimplificationpass_traits;
pub mod commonsubexpressioneliminationpass_traits;
pub mod constantfoldingpass_traits;
pub mod copyeliminationpass_traits;
pub mod deadcodeeliminationpass_traits;
pub mod functions;
pub mod layoutoptimizationpass_traits;
pub mod memoryallocationoptimizationpass_traits;
pub mod operationfusionpass_traits;
pub mod parallelizationanalysispass_traits;
pub mod types;
pub mod xlaconfig_traits;
pub mod xlapassmanager_traits;

// Re-export types and functions (trait impls are auto-registered via mod declarations)
pub use functions::*;
pub use types::*;
