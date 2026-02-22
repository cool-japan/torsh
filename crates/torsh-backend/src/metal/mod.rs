//! Metal GPU backend for ToRSh deep learning framework
//!
//! This module provides GPU acceleration for ToRSh on Apple Silicon (M1/M2/M3)
//! using the Metal framework and Metal Performance Shaders (MPS).

#![allow(unexpected_cfgs)]

pub mod backend;
pub mod buffer;
pub mod device;
pub mod error;
pub mod indirect_commands;
pub mod kernels;
pub mod mps;
pub mod neural_engine;
pub mod ops;

pub use backend::MetalBackend;
pub use buffer::MetalBuffer;
pub use device::MetalDevice;
pub use error::{MetalError, Result};
pub use indirect_commands::{
    CommandPattern, ConcurrentRequirements, IndexType, IndirectCommand,
    IndirectCommandBufferConfig, IndirectCommandCapabilities, IndirectCommandConfigBuilder,
    IndirectCommandManager, IndirectCommandMetrics, IndirectCommandType, MemoryAccessPattern,
    OptimizationResult, UpdateFrequency,
};
pub use neural_engine::{
    ModelFormat, NeuralEngineBuffer, NeuralEngineCapabilities, NeuralEngineContext,
    NeuralEngineOperation, NeuralEngineOpsBuilder,
};

/// Check if Metal backend is available
pub fn is_available() -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // Metal is available on Apple Silicon Macs
        true
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        false
    }
}

/// Get number of Metal devices
pub fn device_count() -> Option<usize> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // For Apple Silicon, typically there's 1 integrated GPU
        // In a real implementation, this would query the Metal device list
        if is_available() {
            Some(1)
        } else {
            Some(0)
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = MetalBackend::new();
        assert!(backend.is_ok());
    }
}
