//! Foreign Function Interface for ToRSh
//!
//! This crate provides Python bindings and other FFI capabilities for the ToRSh
//! deep learning framework, enabling seamless integration with Python ecosystems
//! like NumPy, PyTorch migration scripts, and Jupyter notebooks.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
pub use python::*;

/// C FFI exports
pub mod c_api;

/// Error types for FFI operations
pub mod error;

pub use error::FfiError;

// Re-export commonly used types
pub mod prelude {
    pub use crate::error::FfiError;
    
    #[cfg(feature = "python")]
    pub use crate::python::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ffi_module_loads() {
        // Basic test to ensure the module compiles and loads
        assert!(true);
    }
}