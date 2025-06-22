//! CUDA backend for ToRSh deep learning framework
//!
//! This crate provides high-performance GPU acceleration for tensor operations
//! using NVIDIA CUDA and cuDNN. It integrates with the scirs2 ecosystem for
//! optimal performance and compatibility.

#![cfg_attr(docsrs, feature(doc_cfg))]

use torsh_core::{TensorError, DType};
use torsh_backends::{Backend, BackendError, Device, DeviceType};
use scirs2::gpu::CudaContext;

pub mod backend;
pub mod buffer;
pub mod device;
pub mod error;
pub mod kernels;
pub mod memory;
pub mod stream;

#[cfg(feature = "cudnn")]
pub mod cudnn;

pub mod neural_ops_enhanced;
pub mod mixed_precision;

pub use backend::CudaBackend;
pub use buffer::CudaBuffer;
pub use device::CudaDevice;
pub use error::CudaError;
pub use memory::CudaMemoryManager;
pub use stream::CudaStream;

#[cfg(feature = "cudnn")]
pub use cudnn::{CudnnHandle, CudnnOps, TensorDescriptor, FilterDescriptor, ConvolutionDescriptor, ActivationDescriptor};

pub use neural_ops_enhanced::EnhancedNeuralOps;
pub use mixed_precision::{GradientScaler, AmpContext, MixedPrecisionTrainer};

/// Re-export commonly used types
pub mod prelude {
    pub use super::{
        CudaBackend, CudaBuffer, CudaDevice, CudaError, 
        CudaMemoryManager, CudaStream, EnhancedNeuralOps,
        GradientScaler, AmpContext, MixedPrecisionTrainer
    };
    
    #[cfg(feature = "cudnn")]
    pub use super::{CudnnHandle, CudnnOps, TensorDescriptor, FilterDescriptor, ConvolutionDescriptor, ActivationDescriptor};
    
    pub use torsh_backends::prelude::*;
}

/// Initialize CUDA backend
pub fn init() -> Result<(), CudaError> {
    cust::init(cust::CudaFlags::empty())?;
    Ok(())
}

/// Check if CUDA is available
pub fn is_available() -> bool {
    match cust::init(cust::CudaFlags::empty()) {
        Ok(_) => {
            match cust::Device::get_count() {
                Ok(count) => count > 0,
                Err(_) => false,
            }
        },
        Err(_) => false,
    }
}

/// Get number of CUDA devices
pub fn device_count() -> Result<u32, CudaError> {
    Ok(cust::Device::get_count()?)
}

/// Get current CUDA device
pub fn current_device() -> Result<CudaDevice, CudaError> {
    let device = cust::Device::get_current()?;
    Ok(CudaDevice::new(device.as_device_ptr().0 as usize))
}

/// Set current CUDA device
pub fn set_device(device_id: usize) -> Result<(), CudaError> {
    let device = cust::Device::get_device(device_id as u32)?;
    device.set_current()?;
    Ok(())
}

/// Synchronize current device
pub fn synchronize() -> Result<(), CudaError> {
    cust::Context::synchronize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass if CUDA is available, skip if not
        if is_available() {
            assert!(device_count().unwrap() > 0);
        }
    }

    #[test] 
    fn test_cuda_init() {
        if is_available() {
            assert!(init().is_ok());
        }
    }
}