//! CUDA types compatibility layer for cust API
//!
//! This module provides type aliases and re-exports to handle API changes
//! between different versions of the cust and cuda-sys crates.

// Re-export all cust prelude types
#[allow(unused_imports)]
pub use cust::prelude::*;

// Cust type re-exports for use as `cust::TypeName` pattern
pub mod cust_compat {
    pub use cust::context::{Context, ContextFlags};
    pub use cust::device::{Device, DeviceAttribute};
    pub use cust::error::CudaError;
    pub use cust::event::{Event, EventFlags};
    pub use cust::stream::{Stream, StreamFlags};
    pub use cust::memory::DevicePointer;

    // Re-export memory types
    pub mod memory {
        pub use cust::memory::DevicePointer;
    }
}

// CUDA system types from cuda_sys
pub mod cuda_sys {
    pub use cuda_sys::cudaError as cudaError_t;
    pub use cuda_sys::*;
}

pub use cust::context::{Context, ContextFlags};
pub use cust::device::{Device, DeviceAttribute};
pub use cust::event::{Event, EventFlags};
pub use cust::stream::{Stream, StreamFlags};
/// Direct re-exports for common use
pub use cust::memory::DevicePointer;

/// Type alias for CUDA error results from cust
pub type CustResult<T> = Result<T, cust::error::CudaError>;

// Re-export CUDA stream types from cuda_sys for interop
pub use cuda_sys::cudart::cudaStream_t;
pub use cuda_sys::cuda::CUstream;
