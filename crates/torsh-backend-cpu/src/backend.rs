//! CPU Backend Implementation

use crate::{CpuBuffer, CpuDevice, CpuKernel, CpuMemoryManager, CpuProfiler};
use crate::buffer::BufferCpuExt;
use async_trait::async_trait;
use std::sync::Once;
use torsh_backends::{
    Backend, BackendError, BackendResult, Buffer, BufferDescriptor,
    Device, Kernel, KernelDescriptor, MemoryManager,
    Profiler,
};
use torsh_backends::backend::{BackendCapabilities, PerformanceHints};
use torsh_core::{device::DeviceType, dtype::DType};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Static initialization for global thread pool
static THREAD_POOL_INIT: Once = Once::new();

/// CPU compute backend implementation
pub struct CpuBackend {
    devices: Vec<CpuDevice>,
    memory_manager: CpuMemoryManager,
    profiler: CpuProfiler,
    initialized: bool,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> BackendResult<Self> {
        let num_cores = Self::detect_cpu_cores();
        
        // For CPU, we typically have one logical device that represents all cores
        let device = CpuDevice::new(0, num_cores)?;
        
        Ok(Self {
            devices: vec![device],
            memory_manager: CpuMemoryManager::new(),
            profiler: CpuProfiler::new(),
            initialized: false,
        })
    }
    
    /// Get the number of available CPU cores
    pub fn num_cores(&self) -> usize {
        Self::detect_cpu_cores()
    }
    
    /// Detect the number of CPU cores
    fn detect_cpu_cores() -> usize {
        #[cfg(feature = "std")]
        {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
        #[cfg(not(feature = "std"))]
        {
            // Fallback for no-std
            4 // Default assumption
        }
    }
}

#[async_trait]
impl Backend for CpuBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }
    
    fn name(&self) -> &str {
        "CPU Backend"
    }
    
    fn is_available(&self) -> BackendResult<bool> {
        Ok(true) // CPU backend is always available
    }
    
    async fn initialize(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        
        // Initialize Rayon thread pool if feature is enabled (only once globally)
        #[cfg(feature = "rayon-threads")]
        {
            let num_threads = self.num_cores();
            THREAD_POOL_INIT.call_once(|| {
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build_global();
            });
        }
        
        self.initialized = true;
        Ok(())
    }
    
    async fn shutdown(&mut self) -> BackendResult<()> {
        self.initialized = false;
        Ok(())
    }
    
    fn devices(&self) -> BackendResult<Vec<Device>> {
        Ok(self.devices.iter().map(|d| d.to_device()).collect())
    }
    
    fn default_device(&self) -> BackendResult<Device> {
        self.devices
            .first()
            .ok_or_else(|| BackendError::DeviceNotFound {
                device: "No CPU device available".to_string(),
            }.into())
            .map(|d| d.to_device())
    }
    
    fn create_device(&self, device_id: usize) -> BackendResult<Device> {
        if device_id >= self.devices.len() {
            return Err(BackendError::DeviceNotFound {
                device: format!("CPU device {}", device_id),
            }.into());
        }
        Ok(self.devices[device_id].to_device())
    }
    
    fn create_buffer(&self, device: &Device, descriptor: &BufferDescriptor) -> BackendResult<Buffer> {
        CpuBuffer::new_buffer(device.clone(), descriptor)
    }
    
    fn create_kernel(&self, device: &Device, descriptor: &KernelDescriptor) -> BackendResult<Kernel> {
        CpuKernel::new_kernel(device.clone(), descriptor)
    }
    
    fn memory_manager(&self, _device: &Device) -> BackendResult<Box<dyn MemoryManager>> {
        Ok(Box::new(self.memory_manager.clone()))
    }
    
    fn profiler(&self) -> BackendResult<Box<dyn Profiler>> {
        Ok(Box::new(self.profiler.clone()))
    }
    
    async fn synchronize(&self, _device: &Device) -> BackendResult<()> {
        // CPU operations are synchronous, so nothing to do
        Ok(())
    }
    
    async fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> BackendResult<()> {
        if !src.is_cpu() || !dst.is_cpu() {
            return Err(BackendError::UnsupportedOperation {
                operation: "copy_buffer with non-CPU buffers".to_string(),
            }.into());
        }
        
        let src_ptr = src.as_cpu_ptr().unwrap();
        let dst_ptr = dst.as_cpu_ptr().unwrap();
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_offset),
                dst_ptr.add(dst_offset),
                size,
            );
        }
        
        Ok(())
    }
    
    async fn copy_to_device(
        &self,
        src: &[u8],
        dst: &Buffer,
        dst_offset: usize,
    ) -> BackendResult<()> {
        if !dst.is_cpu() {
            return Err(BackendError::UnsupportedOperation {
                operation: "copy_to_device with non-CPU destination".to_string(),
            }.into());
        }
        
        let dst_ptr = dst.as_cpu_ptr().unwrap();
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst_ptr.add(dst_offset),
                src.len(),
            );
        }
        
        Ok(())
    }
    
    async fn copy_from_device(
        &self,
        src: &Buffer,
        dst: &mut [u8],
        src_offset: usize,
    ) -> BackendResult<()> {
        if !src.is_cpu() {
            return Err(BackendError::UnsupportedOperation {
                operation: "copy_from_device with non-CPU source".to_string(),
            }.into());
        }
        
        let src_ptr = src.as_cpu_ptr().unwrap();
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_offset),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        
        Ok(())
    }
    
    async fn execute_kernel(
        &self,
        _kernel: &Kernel,
        _buffers: &[&Buffer],
        _uniform_data: &[u8],
        _workgroup_size: (u32, u32, u32),
        _workgroup_count: (u32, u32, u32),
    ) -> BackendResult<()> {
        // For now, use the kernel name to dispatch to appropriate function
        // In a real implementation, you'd extract the CPU kernel implementation
        // from the abstract Kernel
        return Err(BackendError::UnsupportedOperation {
            operation: "execute_kernel not yet implemented for abstract kernels".to_string(),
        }.into());
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            max_buffer_size: usize::MAX, // Limited by available system memory
            max_compute_units: self.num_cores(),
            max_workgroup_size: (u32::MAX, 1, 1), // CPU doesn't have workgroup size limits
            supported_dtypes: vec![
                DType::F32, DType::F64, 
                DType::I32, DType::I64, 
                DType::I16, DType::I8, DType::U8,
                DType::Bool,
            ],
            supports_async: false, // CPU operations are synchronous
            supports_unified_memory: true, // CPU uses system memory
            supports_sub_buffers: true,
            supports_kernel_caching: true,
            memory_bandwidth_gbps: 50.0, // Typical DDR4 bandwidth
            compute_throughput_gflops: self.num_cores() as f32 * 10.0, // Rough estimate
        }
    }
    
    fn performance_hints(&self) -> PerformanceHints {
        PerformanceHints {
            preferred_workgroup_size: (self.num_cores() as u32, 1, 1),
            memory_alignment: 64, // Cache line size
            prefer_vectorized: cfg!(feature = "simd"),
            prefer_async: false, // CPU operations are synchronous
            optimal_batch_size: 1024,
            cache_kernels: true,
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create CPU backend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_cpu_backend_initialization() {
        let mut backend = CpuBackend::new().unwrap();
        assert!(!backend.initialized);
        
        backend.initialize().await.unwrap();
        assert!(backend.initialized);
        
        backend.shutdown().await.unwrap();
        assert!(!backend.initialized);
    }
    
    #[tokio::test]
    async fn test_cpu_backend_devices() {
        let backend = CpuBackend::new().unwrap();
        
        let devices = backend.devices().unwrap();
        assert!(!devices.is_empty());
        
        let default_device = backend.default_device().unwrap();
        assert_eq!(default_device.device_type(), DeviceType::Cpu);
    }
    
    #[test]
    fn test_cpu_backend_capabilities() {
        let backend = CpuBackend::new().unwrap();
        let caps = backend.capabilities();
        
        assert!(caps.max_compute_units > 0);
        assert!(caps.supports_unified_memory);
        assert!(!caps.supported_dtypes.is_empty());
    }
}