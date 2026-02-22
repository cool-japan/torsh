//! Auto-generated module structure

pub mod autogradcontextguard_traits;
pub mod autogradresourcemanager_traits;
pub mod autogradscope_traits;
pub mod checkpointguard_traits;
pub mod computationgraphguard_traits;
pub mod computationgraphmanager_traits;
pub mod distributedcontextguard_traits;
pub mod enhancedresourcemanager_traits;
pub mod functions;
pub mod gradientstorageguard_traits;
pub mod gradientstoragemanager_traits;
pub mod memorybufferguard_traits;
pub mod memorybuffermanager_traits;
pub mod memorypressuremonitor_traits;
pub mod profilesessionguard_traits;
pub mod resourceleakdetector_traits;
pub mod tensorgradguard_traits;
pub mod types;
pub mod variableenvironmentguard_traits;

// Re-export types and functions (trait impls are auto-registered via mod declarations)
pub use functions::*;
pub use types::*;
