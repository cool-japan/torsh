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
    pub use cust::memory::DevicePointer;
    pub use cust::stream::{Stream, StreamFlags, StreamWaitEventFlags};

    // Re-export memory types
    pub mod memory {
        pub use cust::memory::DevicePointer;
    }

    // Re-export stream submodule
    pub mod stream {
        pub use cust::stream::{Stream, StreamFlags, StreamWaitEventFlags};
    }
}

// CUDA system types from cuda_sys - comprehensive compatibility layer
#[allow(non_upper_case_globals)]
pub mod cuda_sys_compat {
    // Re-export cudart module types
    pub use cuda_sys::cudart::*;

    // Re-export cuda driver API module
    pub mod cuda {
        pub use cuda_sys::cuda::*;
    }

    // CUstream compatibility - use cudart's cudaStream_t
    pub type CUstream = cuda_sys::cudart::cudaStream_t;

    // CUDA_SUCCESS compatibility - define as a constant if not available
    pub const CUDA_SUCCESS: u32 = 0;

    // cudaSuccess constant - represents successful CUDA operation
    // In cuda-sys 0.2.*, the enum variant is cudaError_t::Success (not cudaSuccess)
    pub const cudaSuccess: cuda_sys::cudart::cudaError_t = cuda_sys::cudart::cudaError_t::Success;

    // Re-export common functions from cudart
    pub use cuda_sys::cudart::{
        cudaFree, cudaFreeHost, cudaHostAlloc, cudaHostGetDevicePointer, cudaMalloc,
        cudaMallocManaged, cudaMemAdvise, cudaMemPrefetchAsync, cudaMemcpy, cudaMemcpyAsync,
        cudaMemset, cudaMemsetAsync, cudaStreamSynchronize,
    };

    // Re-export host allocation flags
    pub use cuda_sys::cudart::{
        cudaHostAllocMapped, cudaHostAllocPortable, cudaHostAllocWriteCombined,
    };

    // Re-export memcpy kinds
    pub use cuda_sys::cudart::cudaMemcpyKind;

    // Re-export cudaMemAttachGlobal
    pub use cuda_sys::cudart::cudaMemAttachGlobal;

    // cudaCpuDeviceId constant (this is typically -1 in CUDA)
    pub const cudaCpuDeviceId: i32 = -1;

    // Re-export memory advise constants
    pub use cuda_sys::cudart::cudaMemoryAdvise;

    // Memory advice constants - these are the actual enum values from cuda_sys
    pub const cudaMemAdviseSetReadMostly: u32 =
        cuda_sys::cudart::cudaMemoryAdvise_cudaMemAdviseSetReadMostly;
    pub const cudaMemAdviseUnsetReadMostly: u32 =
        cuda_sys::cudart::cudaMemoryAdvise_cudaMemAdviseUnsetReadMostly;
    pub const cudaMemAdviseSetPreferredLocation: u32 =
        cuda_sys::cudart::cudaMemoryAdvise_cudaMemAdviseSetPreferredLocation;
    pub const cudaMemAdviseUnsetPreferredLocation: u32 =
        cuda_sys::cudart::cudaMemoryAdvise_cudaMemAdviseUnsetPreferredLocation;
    pub const cudaMemAdviseSetAccessedBy: u32 =
        cuda_sys::cudart::cudaMemoryAdvise_cudaMemAdviseSetAccessedBy;
    pub const cudaMemAdviseUnsetAccessedBy: u32 =
        cuda_sys::cudart::cudaMemoryAdvise_cudaMemAdviseUnsetAccessedBy;
}

pub use cust::context::{Context, ContextFlags};
pub use cust::device::{Device, DeviceAttribute};
pub use cust::event::{Event, EventFlags};
/// Direct re-exports for common use
pub use cust::memory::DevicePointer;
pub use cust::stream::{Stream, StreamFlags};

/// Type alias for CUDA error results from cust
pub type CustResult<T> = Result<T, cust::error::CudaError>;

// Re-export CUDA stream types from cuda_sys for interop
// We create a compatibility alias using cudaStream_t
pub type CUstream = cuda_sys::cudart::cudaStream_t;

pub use cuda_sys::cudart::cudaError_t;
pub use cuda_sys::cudart::cudaStream_t;

// cudaSuccess constant - cudaError_t::Success in cuda-sys 0.2.*
// Using C-style naming for CUDA interoperability
#[allow(non_upper_case_globals)]
pub const cudaSuccess: cudaError_t = cuda_sys::cudart::cudaError_t::Success;
