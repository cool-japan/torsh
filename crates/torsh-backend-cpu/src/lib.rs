//! CPU backend implementation for ToRSh
//!
//! This crate provides high-performance CPU computing backend for ToRSh tensor operations.
//! It leverages multi-threading with Rayon, SIMD operations, and optimized memory layouts
//! to deliver maximum performance on CPU hardware.
//!
//! # Features
//!
//! - **Multi-threading**: Parallel tensor operations using Rayon
//! - **SIMD**: Vectorized operations for supported data types
//! - **Memory optimization**: Cache-friendly memory layouts
//! - **BLAS integration**: Optional BLAS backend for linear algebra
//! - **Cross-platform**: Works on all platforms supported by Rust

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod backend;
pub mod buffer;
pub mod device;
pub mod kernel;
pub mod memory;
pub mod optimized_kernels;
pub mod profiler;
pub mod simd;

// Re-exports
pub use backend::CpuBackend;
pub use buffer::CpuBuffer;
pub use device::CpuDevice;
pub use kernel::{CpuKernel, CpuKernelExecutor};
pub use memory::CpuMemoryManager;
pub use profiler::CpuProfiler;

use torsh_backends::backend::BackendFactory;
use torsh_backends::{Backend, BackendResult};
use torsh_core::device::DeviceType;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// CPU Backend Factory
pub struct CpuBackendFactory;

impl BackendFactory for CpuBackendFactory {
    fn create(&self) -> BackendResult<Box<dyn Backend>> {
        Ok(Box::new(CpuBackend::new()?))
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn is_available(&self) -> bool {
        true // CPU backend is always available
    }
}

/// Initialize and register the CPU backend
pub fn init() -> BackendResult<()> {
    let factory = CpuBackendFactory;
    if factory.is_available() {
        let backend = CpuBackend::new()?;
        torsh_backends::register_backend(backend)?;
    }
    Ok(())
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        init, CpuBackend, CpuBackendFactory, CpuBuffer, CpuDevice, CpuKernel, CpuMemoryManager,
        CpuProfiler,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_cpu_backend_creation() {
        let factory = CpuBackendFactory;
        assert!(factory.is_available());
        assert_eq!(factory.device_type(), DeviceType::Cpu);

        let backend = factory.create().unwrap();
        assert!(backend.is_available().unwrap());
    }

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
}
