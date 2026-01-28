//! CUDA device management

use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::memory::CudaMemoryManager;
use crate::cuda::stream::CudaStream;
use cust::context::Context;
use cust::device::{Device as CustDevice, DeviceAttribute as CustDeviceAttribute};
use std::sync::Arc;
use torsh_core::DeviceType;

/// CUDA device implementation
#[derive(Debug, Clone)]
pub struct CudaDevice {
    device_id: usize,
    device: CustDevice,
    context: Arc<Context>,
    memory_manager: Arc<CudaMemoryManager>,
    default_stream: Arc<CudaStream>,
}

impl CudaDevice {
    /// Create new CUDA device
    pub fn new(device_id: usize) -> CudaResult<Self> {
        let device = CustDevice::get_device(device_id as u32).map_err(|e| CudaError::Context {
            message: format!("Failed to get CUDA device {}: {}", device_id, e),
        })?;
        let context = Context::new(device).map_err(|e| CudaError::Context {
            message: format!("Failed to create CUDA context: {}", e),
        })?;

        let memory_manager = Arc::new(CudaMemoryManager::new(device_id)?);
        let default_stream = Arc::new(CudaStream::default_stream()?);

        Ok(Self {
            device_id,
            device,
            context: Arc::new(context),
            memory_manager,
            default_stream,
        })
    }

    /// Get device ID
    pub fn id(&self) -> usize {
        self.device_id
    }

    /// Get CUDA device handle
    pub fn cuda_device(&self) -> CustDevice {
        self.device
    }

    /// Get CUDA context
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Get memory manager
    pub fn memory_manager(&self) -> &Arc<CudaMemoryManager> {
        &self.memory_manager
    }

    /// Get default stream
    pub fn default_stream(&self) -> &Arc<CudaStream> {
        &self.default_stream
    }

    /// Create new stream
    pub fn create_stream(&self) -> CudaResult<CudaStream> {
        CudaStream::new()
    }

    /// Get device properties
    pub fn properties(&self) -> CudaResult<DeviceProperties> {
        let name = self.device.name().map_err(|e| CudaError::Context {
            message: format!("Failed to get device name: {}", e),
        })?;
        let total_memory = self.device.total_memory().map_err(|e| CudaError::Context {
            message: format!("Failed to get total memory: {}", e),
        })?;

        let major = self
            .device
            .get_attribute(CustDeviceAttribute::ComputeCapabilityMajor)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get compute capability major: {}", e),
            })? as u32;
        let minor = self
            .device
            .get_attribute(CustDeviceAttribute::ComputeCapabilityMinor)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get compute capability minor: {}", e),
            })? as u32;
        let compute_capability = major * 10 + minor;

        let multiprocessor_count = self
            .device
            .get_attribute(CustDeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get multiprocessor count: {}", e),
            })? as u32;
        let max_threads_per_block = self
            .device
            .get_attribute(CustDeviceAttribute::MaxThreadsPerBlock)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get max threads per block: {}", e),
            })? as u32;
        let warp_size = self
            .device
            .get_attribute(CustDeviceAttribute::WarpSize)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get warp size: {}", e),
            })? as u32;

        let max_threads_per_multiprocessor = self
            .device
            .get_attribute(CustDeviceAttribute::MaxThreadsPerMultiprocessor)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get max threads per MP: {}", e),
            })? as u32;
        let shared_memory_per_multiprocessor = self
            .device
            .get_attribute(CustDeviceAttribute::MaxSharedMemoryPerMultiprocessor)
            .map_err(|e| CudaError::Context {
                message: format!("Failed to get shared memory per MP: {}", e),
            })? as usize;

        // MaxBlocksPerMultiprocessor not available in cust, use typical value based on compute capability
        let max_blocks_per_multiprocessor = if compute_capability >= 75 { 32 } else { 16 };

        // Registers per multiprocessor - typical values based on compute capability
        let registers_per_multiprocessor = if compute_capability >= 80 {
            65536 // Ampere and later
        } else if compute_capability >= 70 {
            65536 // Volta/Turing
        } else if compute_capability >= 50 {
            65536 // Maxwell/Pascal
        } else {
            32768 // Older architectures
        };

        Ok(DeviceProperties {
            name,
            total_memory,
            compute_capability,
            multiprocessor_count,
            max_threads_per_block,
            warp_size,
            max_threads_per_multiprocessor,
            shared_memory_per_multiprocessor,
            max_blocks_per_multiprocessor,
            registers_per_multiprocessor,
        })
    }

    /// Check if device supports feature
    pub fn supports_feature(&self, feature: CudaFeature) -> CudaResult<bool> {
        let props = self.properties()?;
        Ok(match feature {
            CudaFeature::DoublePrecision => props.compute_capability >= 13,
            CudaFeature::UnifiedAddressing => props.compute_capability >= 20,
            CudaFeature::ManagedMemory => props.compute_capability >= 30,
            CudaFeature::Tensor => props.compute_capability >= 70,
            CudaFeature::BFloat16 => props.compute_capability >= 80,
        })
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.device_id)
    }

    /// Check if device is available
    pub fn is_available(&self) -> bool {
        // Context is already created if this device exists
        true
    }

    /// Get device name
    pub fn name(&self) -> String {
        self.device
            .name()
            .unwrap_or_else(|_| format!("CUDA:{}", self.device_id))
    }

    /// Synchronize device
    pub fn synchronize(&self) -> Result<(), crate::BackendError> {
        // Use cust stream synchronization as context sync isn't directly available
        // The default stream synchronizes all work on the device
        cust::stream::Stream::synchronize(
            &cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None).map_err(|e| {
                crate::BackendError::Runtime {
                    message: format!("Failed to create stream for sync: {}", e),
                }
            })?,
        )
        .map_err(|e| crate::BackendError::Runtime {
            message: format!("CUDA synchronization failed: {}", e),
        })?;
        Ok(())
    }
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: u32,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
    pub warp_size: u32,
    pub max_threads_per_multiprocessor: u32,
    pub shared_memory_per_multiprocessor: usize,
    pub max_blocks_per_multiprocessor: u32,
    pub registers_per_multiprocessor: u32,
}

/// CUDA features
#[derive(Debug, Clone, Copy)]
pub enum CudaFeature {
    DoublePrecision,
    UnifiedAddressing,
    ManagedMemory,
    Tensor,
    BFloat16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires CUDA hardware - run with --ignored flag"]
    fn test_cuda_device_creation() {
        if crate::is_available() {
            let device = CudaDevice::new(0);
            assert!(device.is_ok());

            let device = device.unwrap();
            assert_eq!(device.id(), 0);
            assert_eq!(device.device_type(), DeviceType::Cuda(0));
        }
    }

    #[test]
    fn test_device_properties() {
        if crate::is_available() {
            let device = CudaDevice::new(0).unwrap();
            let props = device.properties().unwrap();

            assert!(!props.name.is_empty());
            assert!(props.total_memory > 0);
            assert!(props.multiprocessor_count > 0);
        }
    }

    #[test]
    fn test_feature_support() {
        if crate::is_available() {
            let device = CudaDevice::new(0).unwrap();

            // Most modern GPUs should support these
            assert!(device
                .supports_feature(CudaFeature::DoublePrecision)
                .unwrap());
            assert!(device
                .supports_feature(CudaFeature::UnifiedAddressing)
                .unwrap());
        }
    }
}
