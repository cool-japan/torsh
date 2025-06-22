//! CUDA device management

use std::sync::Arc;
use torsh_backends::{Device, DeviceType};
use crate::error::{CudaError, CudaResult};
use crate::memory::CudaMemoryManager;
use crate::stream::CudaStream;

/// CUDA device implementation
#[derive(Debug, Clone)]
pub struct CudaDevice {
    device_id: usize,
    device: cust::Device,
    context: Arc<cust::Context>,
    memory_manager: Arc<CudaMemoryManager>,
    default_stream: Arc<CudaStream>,
}

impl CudaDevice {
    /// Create new CUDA device
    pub fn new(device_id: usize) -> CudaResult<Self> {
        let device = cust::Device::get_device(device_id as u32)?;
        let context = cust::Context::create_and_push(
            cust::ContextFlags::MAP_HOST | cust::ContextFlags::SCHED_AUTO,
            device,
        )?;
        
        let memory_manager = Arc::new(CudaMemoryManager::new(device_id)?);
        let default_stream = Arc::new(CudaStream::default()?);
        
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
    pub fn cuda_device(&self) -> cust::Device {
        self.device
    }
    
    /// Get CUDA context
    pub fn context(&self) -> &Arc<cust::Context> {
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
        let name = self.device.name()?;
        let total_memory = self.device.total_memory()?;
        let compute_capability = self.device.get_attribute(cust::DeviceAttribute::ComputeCapabilityMajor)? as u32 * 10 +
                                 self.device.get_attribute(cust::DeviceAttribute::ComputeCapabilityMinor)? as u32;
        let multiprocessor_count = self.device.get_attribute(cust::DeviceAttribute::MultiprocessorCount)? as u32;
        let max_threads_per_block = self.device.get_attribute(cust::DeviceAttribute::MaxThreadsPerBlock)? as u32;
        let warp_size = self.device.get_attribute(cust::DeviceAttribute::WarpSize)? as u32;
        
        Ok(DeviceProperties {
            name,
            total_memory,
            compute_capability,
            multiprocessor_count,
            max_threads_per_block,
            warp_size,
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
}

impl Device for CudaDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.device_id)
    }
    
    fn is_available(&self) -> bool {
        self.context.set_current().is_ok()
    }
    
    fn name(&self) -> String {
        self.device.name().unwrap_or_else(|_| format!("CUDA:{}", self.device_id))
    }
    
    fn synchronize(&self) -> Result<(), torsh_backends::BackendError> {
        self.context.set_current()
            .map_err(|e| torsh_backends::BackendError::Runtime { 
                message: format!("Failed to set CUDA context: {}", e) 
            })?;
        cust::Context::synchronize()
            .map_err(|e| torsh_backends::BackendError::Runtime { 
                message: format!("CUDA synchronization failed: {}", e) 
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
            assert!(device.supports_feature(CudaFeature::DoublePrecision).unwrap());
            assert!(device.supports_feature(CudaFeature::UnifiedAddressing).unwrap());
        }
    }
}