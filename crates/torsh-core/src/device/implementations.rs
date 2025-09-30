//! Concrete device implementations for different backends
//!
//! This module provides concrete implementations of the Device trait for
//! various compute backends including CPU, CUDA, Metal, and WebGPU.

use crate::device::core::DeviceContext;
use crate::device::{Device, DeviceCapabilities, DeviceType};
use crate::error::Result;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CPU device implementation
///
/// Provides CPU compute capabilities with SIMD optimizations and
/// multi-threading support through Rayon.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::CpuDevice;
///
/// let device = CpuDevice::new();
/// println!("CPU device: {}", device.name());
/// device.synchronize()?;
/// ```
#[derive(Debug)]
pub struct CpuDevice {
    context: DeviceContext,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
    simd_level: SimdLevel,
}

impl Clone for CpuDevice {
    fn clone(&self) -> Self {
        Self {
            context: DeviceContext::new(DeviceType::Cpu),
            thread_pool: self.thread_pool.clone(),
            simd_level: self.simd_level,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    None,
    Sse,
    Avx,
    Avx2,
    Avx512,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new() -> Self {
        let context = DeviceContext::new(DeviceType::Cpu);
        context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Initializing)
            .ok();

        let device = Self {
            context,
            thread_pool: Self::create_thread_pool(),
            simd_level: Self::detect_simd_level(),
        };

        device
            .context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Ready)
            .ok();
        device
    }

    /// Create with specific number of threads
    pub fn with_threads(num_threads: usize) -> Result<Self> {
        let context = DeviceContext::new(DeviceType::Cpu);
        context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Initializing)?;

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| {
                crate::error::TorshError::DeviceError(format!(
                    "Failed to create thread pool: {}",
                    e
                ))
            })?;

        let device = Self {
            context,
            thread_pool: Some(Arc::new(thread_pool)),
            simd_level: Self::detect_simd_level(),
        };

        device
            .context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Ready)?;
        Ok(device)
    }

    /// Get the thread pool
    pub fn thread_pool(&self) -> Option<&Arc<rayon::ThreadPool>> {
        self.thread_pool.as_ref()
    }

    /// Get SIMD level
    pub fn simd_level(&self) -> SimdLevel {
        self.simd_level
    }

    /// Execute work on the thread pool
    pub fn execute_parallel<F, T>(&self, work: F) -> T
    where
        F: FnOnce() -> T + Send,
        T: Send,
    {
        match &self.thread_pool {
            Some(pool) => pool.install(work),
            None => work(),
        }
    }

    fn create_thread_pool() -> Option<Arc<rayon::ThreadPool>> {
        rayon::ThreadPoolBuilder::new().build().map(Arc::new).ok()
    }

    fn detect_simd_level() -> SimdLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx512f") {
                SimdLevel::Avx512
            } else if cfg!(target_feature = "avx2") {
                SimdLevel::Avx2
            } else if cfg!(target_feature = "avx") {
                SimdLevel::Avx
            } else if cfg!(target_feature = "sse") {
                SimdLevel::Sse
            } else {
                SimdLevel::None
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            SimdLevel::None
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for CpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn is_available(&self) -> Result<bool> {
        Ok(self.context.lifecycle().is_ready())
    }

    fn capabilities(&self) -> Result<DeviceCapabilities> {
        DeviceCapabilities::detect(DeviceType::Cpu)
    }

    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous by default
        Ok(())
    }

    fn reset(&self) -> Result<()> {
        self.context.lifecycle().reset()?;
        self.context.clear_resources();
        self.context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Ready)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_device(&self) -> Result<Box<dyn Device>> {
        Ok(Box::new(CpuDevice::new()))
    }
}

/// CUDA device implementation (requires CUDA runtime)
#[derive(Debug)]
pub struct CudaDevice {
    context: DeviceContext,
    device_index: usize,
    cuda_context: Option<CudaContext>,
    #[allow(dead_code)] // CUDA stream management - future feature
    stream_manager: Arc<Mutex<CudaStreamManager>>,
}

#[derive(Debug)]
struct CudaContext {
    #[allow(dead_code)] // CUDA implementation - future feature
    device_handle: u32,
    #[allow(dead_code)] // CUDA implementation - future feature
    context_handle: u64,
    compute_capability: (u32, u32),
}

#[derive(Debug)]
struct CudaStreamManager {
    #[allow(dead_code)] // CUDA stream storage - future feature
    streams: HashMap<u32, CudaStream>,
    #[allow(dead_code)] // CUDA stream ID tracking - future feature
    next_stream_id: u32,
}

#[derive(Debug)]
struct CudaStream {
    #[allow(dead_code)] // CUDA stream implementation - future feature
    stream_id: u32,
    #[allow(dead_code)] // CUDA stream implementation - future feature
    stream_handle: u64,
    #[allow(dead_code)] // CUDA stream implementation - future feature
    is_default: bool,
}

impl CudaDevice {
    /// Create a new CUDA device
    pub fn new(device_index: usize) -> Result<Self> {
        let context = DeviceContext::new(DeviceType::Cuda(device_index));
        context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Initializing)?;

        #[cfg(feature = "cuda")]
        {
            let cuda_context = Self::initialize_cuda_context(device_index)?;
            let device = Self {
                context,
                device_index,
                cuda_context: Some(cuda_context),
                stream_manager: Arc::new(Mutex::new(CudaStreamManager::new())),
            };

            device
                .context
                .lifecycle()
                .set_state(crate::device::core::DeviceState::Ready)?;
            Ok(device)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError("CUDA support not compiled".to_string()),
            ))
        }
    }

    /// Get device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Get CUDA compute capability
    pub fn compute_capability(&self) -> Option<(u32, u32)> {
        self.cuda_context.as_ref().map(|ctx| ctx.compute_capability)
    }

    /// Create a new CUDA stream
    pub fn create_stream(&self) -> Result<u32> {
        #[cfg(feature = "cuda")]
        {
            let mut manager = self.stream_manager.lock().unwrap();
            let stream_id = manager.next_stream_id;
            manager.next_stream_id += 1;

            let stream = CudaStream {
                stream_id,
                stream_handle: self.create_cuda_stream_handle()?,
                is_default: false,
            };

            manager.streams.insert(stream_id, stream);
            Ok(stream_id)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::UnsupportedOperation {
                    op: "CUDA streams".to_string(),
                    dtype: "N/A".to_string(),
                },
            ))
        }
    }

    /// Synchronize device
    pub fn synchronize_device(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would call cudaDeviceSynchronize()
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(())
        }
    }

    #[cfg(feature = "cuda")]
    fn initialize_cuda_context(device_index: usize) -> Result<CudaContext> {
        // Mock CUDA context initialization
        // In a real implementation, this would:
        // 1. Call cudaSetDevice(device_index)
        // 2. Create CUDA context
        // 3. Query device properties
        Ok(CudaContext {
            device_handle: device_index as u32,
            context_handle: 0x12345678, // Mock handle
            compute_capability: (8, 6), // Mock compute capability
        })
    }

    #[cfg(feature = "cuda")]
    fn create_cuda_stream_handle(&self) -> Result<u64> {
        // Mock stream creation
        // In a real implementation, this would call cudaStreamCreate()
        Ok(0x87654321) // Mock stream handle
    }
}

impl CudaStreamManager {
    #[allow(dead_code)] // CUDA stream manager constructor - future feature
    fn new() -> Self {
        Self {
            streams: HashMap::new(),
            next_stream_id: 1, // 0 reserved for default stream
        }
    }
}

impl Device for CudaDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.device_index)
    }

    fn name(&self) -> &str {
        "CUDA Device"
    }

    fn is_available(&self) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            Ok(self.context.lifecycle().is_ready() && self.cuda_context.is_some())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(false)
        }
    }

    fn capabilities(&self) -> Result<DeviceCapabilities> {
        DeviceCapabilities::detect(DeviceType::Cuda(self.device_index))
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize_device()
    }

    fn reset(&self) -> Result<()> {
        self.context.lifecycle().reset()?;
        self.context.clear_resources();

        #[cfg(feature = "cuda")]
        {
            // Reset CUDA context
            let mut manager = self.stream_manager.lock().unwrap();
            manager.streams.clear();
            manager.next_stream_id = 1;
        }

        self.context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Ready)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_device(&self) -> Result<Box<dyn Device>> {
        CudaDevice::new(self.device_index).map(|d| Box::new(d) as Box<dyn Device>)
    }
}

/// Metal device implementation (macOS only)
#[derive(Debug)]
pub struct MetalDevice {
    context: DeviceContext,
    device_index: usize,
    metal_device: Option<MetalDeviceHandle>,
    #[allow(dead_code)] // Metal implementation - future feature
    command_queue: Option<MetalCommandQueue>,
}

#[derive(Debug)]
struct MetalDeviceHandle {
    #[allow(dead_code)] // Metal device handle - future feature
    device_id: u64,
    name: String,
    #[allow(dead_code)] // Metal device handle - future feature
    registry_id: u64,
}

#[derive(Debug)]
struct MetalCommandQueue {
    #[allow(dead_code)] // Metal command queue ID - future implementation
    queue_id: u64,
    #[allow(dead_code)] // Maximum command buffers per queue - future implementation
    max_command_buffers: usize,
}

impl MetalDevice {
    /// Create a new Metal device
    pub fn new(device_index: usize) -> Result<Self> {
        let context = DeviceContext::new(DeviceType::Metal(device_index));
        context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Initializing)?;

        #[cfg(target_os = "macos")]
        {
            let metal_device = Self::create_metal_device(device_index)?;
            let command_queue = Self::create_command_queue(&metal_device)?;

            let device = Self {
                context,
                device_index,
                metal_device: Some(metal_device),
                command_queue: Some(command_queue),
            };

            device
                .context
                .lifecycle()
                .set_state(crate::device::core::DeviceState::Ready)?;
            Ok(device)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError(
                    "Metal device only available on macOS".to_string(),
                ),
            ))
        }
    }

    /// Get device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Get Metal device name
    pub fn metal_device_name(&self) -> Option<&str> {
        self.metal_device.as_ref().map(|d| d.name.as_str())
    }

    /// Execute Metal compute shader
    pub fn execute_compute_shader(&self, _shader_source: &str) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            // Mock shader execution
            // In a real implementation, this would:
            // 1. Compile shader source
            // 2. Create compute pipeline
            // 3. Dispatch compute threads
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(crate::error::TorshError::NotImplemented(
                "Metal compute shaders not available".to_string(),
            ))
        }
    }

    #[cfg(target_os = "macos")]
    fn create_metal_device(device_index: usize) -> Result<MetalDeviceHandle> {
        // Mock Metal device creation
        // In a real implementation, this would use Metal-rs or similar
        Ok(MetalDeviceHandle {
            device_id: device_index as u64,
            name: format!("Apple GPU {}", device_index),
            registry_id: 0x1000 + device_index as u64,
        })
    }

    #[cfg(target_os = "macos")]
    fn create_command_queue(_device: &MetalDeviceHandle) -> Result<MetalCommandQueue> {
        // Mock command queue creation
        Ok(MetalCommandQueue {
            queue_id: 0x2000,
            max_command_buffers: 64,
        })
    }
}

impl Device for MetalDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Metal(self.device_index)
    }

    fn name(&self) -> &str {
        #[cfg(target_os = "macos")]
        {
            self.metal_device_name().unwrap_or("Metal Device")
        }
        #[cfg(not(target_os = "macos"))]
        {
            "Metal Device (Unavailable)"
        }
    }

    fn is_available(&self) -> Result<bool> {
        #[cfg(target_os = "macos")]
        {
            Ok(self.context.lifecycle().is_ready() && self.metal_device.is_some())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(false)
        }
    }

    fn capabilities(&self) -> Result<DeviceCapabilities> {
        DeviceCapabilities::detect(DeviceType::Metal(self.device_index))
    }

    fn synchronize(&self) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            // Metal command queue synchronization
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(())
        }
    }

    fn reset(&self) -> Result<()> {
        self.context.lifecycle().reset()?;
        self.context.clear_resources();
        self.context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Ready)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_device(&self) -> Result<Box<dyn Device>> {
        MetalDevice::new(self.device_index).map(|d| Box::new(d) as Box<dyn Device>)
    }
}

/// WebGPU device implementation
#[derive(Debug)]
pub struct WgpuDevice {
    context: DeviceContext,
    device_index: usize,
    #[allow(dead_code)] // WebGPU device handle - future feature
    wgpu_device: Option<WgpuDeviceHandle>,
    adapter_info: Option<WgpuAdapterInfo>,
}

#[derive(Debug)]
struct WgpuDeviceHandle {
    #[allow(dead_code)] // WebGPU device identifier - future implementation
    device_id: u64,
    #[allow(dead_code)] // WebGPU device limits - future implementation
    limits: WgpuLimits,
    #[allow(dead_code)] // WebGPU device features - future implementation
    features: Vec<String>,
}

#[derive(Debug)]
pub struct WgpuAdapterInfo {
    #[allow(dead_code)] // WebGPU adapter name - future feature
    name: String,
    #[allow(dead_code)] // WebGPU adapter vendor - future implementation
    vendor: String,
    #[allow(dead_code)] // WebGPU device type - future implementation
    device_type: WgpuDeviceType,
    #[allow(dead_code)] // WebGPU backend type - future implementation
    backend: WgpuBackend,
}

#[derive(Debug)]
struct WgpuLimits {
    #[allow(dead_code)] // Maximum bind groups limit - future implementation
    max_bind_groups: u32,
    #[allow(dead_code)] // Uniform buffer binding size limit - future implementation
    max_uniform_buffer_binding_size: u64,
    #[allow(dead_code)] // Storage buffer binding size limit - future implementation
    max_storage_buffer_binding_size: u64,
}

#[derive(Debug)]
enum WgpuDeviceType {
    #[allow(dead_code)] // WebGPU discrete GPU support - future feature
    DiscreteGpu,
    #[allow(dead_code)] // Integrated GPU support - future implementation
    IntegratedGpu,
    #[allow(dead_code)] // Virtual GPU support - future implementation
    VirtualGpu,
    #[allow(dead_code)] // CPU fallback support - future implementation
    Cpu,
}

#[derive(Debug)]
enum WgpuBackend {
    #[allow(dead_code)] // WebGPU Vulkan backend - future feature
    Vulkan,
    #[allow(dead_code)] // Metal backend support - future implementation
    Metal,
    #[allow(dead_code)] // DirectX 12 backend support - future implementation
    Dx12,
    #[allow(dead_code)] // DirectX 11 backend support - future implementation
    Dx11,
    #[allow(dead_code)] // OpenGL backend support - future implementation
    Gl,
    #[allow(dead_code)] // Browser WebGPU support - future implementation
    BrowserWebGpu,
}

impl WgpuDevice {
    /// Create a new WebGPU device
    pub fn new(device_index: usize) -> Result<Self> {
        let context = DeviceContext::new(DeviceType::Wgpu(device_index));
        context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Initializing)?;

        #[cfg(feature = "wgpu")]
        {
            let (wgpu_device, adapter_info) = Self::initialize_wgpu(device_index)?;

            let device = Self {
                context,
                device_index,
                wgpu_device: Some(wgpu_device),
                adapter_info: Some(adapter_info),
            };

            device
                .context
                .lifecycle()
                .set_state(crate::device::core::DeviceState::Ready)?;
            Ok(device)
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError("WebGPU support not compiled".to_string()),
            ))
        }
    }

    /// Get device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Get adapter information
    pub fn adapter_info(&self) -> Option<&WgpuAdapterInfo> {
        self.adapter_info.as_ref()
    }

    /// Execute compute shader
    pub fn execute_compute(&self, _shader_source: &str) -> Result<()> {
        #[cfg(feature = "wgpu")]
        {
            // Mock compute execution
            // In a real implementation, this would:
            // 1. Create compute pipeline
            // 2. Set up bind groups
            // 3. Dispatch compute workgroups
            Ok(())
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::UnsupportedOperation {
                    op: "WebGPU compute".to_string(),
                    dtype: "N/A".to_string(),
                },
            ))
        }
    }

    #[cfg(feature = "wgpu")]
    fn initialize_wgpu(device_index: usize) -> Result<(WgpuDeviceHandle, WgpuAdapterInfo)> {
        // Mock WebGPU initialization
        // In a real implementation, this would use wgpu-rs
        let device = WgpuDeviceHandle {
            device_id: device_index as u64,
            limits: WgpuLimits {
                max_bind_groups: 4,
                max_uniform_buffer_binding_size: 16384,
                max_storage_buffer_binding_size: 134217728,
            },
            features: vec!["compute-shaders".to_string()],
        };

        let adapter_info = WgpuAdapterInfo {
            name: format!("WebGPU Adapter {}", device_index),
            vendor: "Unknown".to_string(),
            device_type: WgpuDeviceType::DiscreteGpu,
            backend: WgpuBackend::Vulkan,
        };

        Ok((device, adapter_info))
    }
}

impl Device for WgpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Wgpu(self.device_index)
    }

    fn name(&self) -> &str {
        #[cfg(feature = "wgpu")]
        {
            self.adapter_info
                .as_ref()
                .map(|info| info.name.as_str())
                .unwrap_or("WebGPU Device")
        }
        #[cfg(not(feature = "wgpu"))]
        {
            "WebGPU Device (Unavailable)"
        }
    }

    fn is_available(&self) -> Result<bool> {
        #[cfg(feature = "wgpu")]
        {
            Ok(self.context.lifecycle().is_ready() && self.wgpu_device.is_some())
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Ok(false)
        }
    }

    fn capabilities(&self) -> Result<DeviceCapabilities> {
        DeviceCapabilities::detect(DeviceType::Wgpu(self.device_index))
    }

    fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "wgpu")]
        {
            // WebGPU device synchronization
            Ok(())
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Ok(())
        }
    }

    fn reset(&self) -> Result<()> {
        self.context.lifecycle().reset()?;
        self.context.clear_resources();
        self.context
            .lifecycle()
            .set_state(crate::device::core::DeviceState::Ready)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_device(&self) -> Result<Box<dyn Device>> {
        WgpuDevice::new(self.device_index).map(|d| Box::new(d) as Box<dyn Device>)
    }
}

/// Device factory for creating concrete device implementations
#[derive(Debug)]
pub struct DeviceFactory;

impl DeviceFactory {
    /// Create a device based on device type
    pub fn create_device(device_type: DeviceType) -> Result<Box<dyn Device>> {
        match device_type {
            DeviceType::Cpu => Ok(Box::new(CpuDevice::new())),
            DeviceType::Cuda(index) => {
                CudaDevice::new(index).map(|d| Box::new(d) as Box<dyn Device>)
            }
            DeviceType::Metal(index) => {
                MetalDevice::new(index).map(|d| Box::new(d) as Box<dyn Device>)
            }
            DeviceType::Wgpu(index) => {
                WgpuDevice::new(index).map(|d| Box::new(d) as Box<dyn Device>)
            }
        }
    }

    /// Create CPU device with specific thread count
    pub fn create_cpu_with_threads(num_threads: usize) -> Result<Box<dyn Device>> {
        CpuDevice::with_threads(num_threads).map(|d| Box::new(d) as Box<dyn Device>)
    }

    /// Check if a device type is available on this platform
    pub fn is_device_type_available(device_type: DeviceType) -> bool {
        match device_type {
            DeviceType::Cpu => true,
            DeviceType::Cuda(_) => cfg!(feature = "cuda"),
            DeviceType::Metal(_) => cfg!(target_os = "macos"),
            DeviceType::Wgpu(_) => cfg!(feature = "wgpu"),
        }
    }

    /// Get all available device types on this platform
    pub fn available_device_types() -> Vec<DeviceType> {
        let mut types = vec![DeviceType::Cpu];

        if cfg!(feature = "cuda") {
            types.push(DeviceType::Cuda(0));
        }

        if cfg!(target_os = "macos") {
            types.push(DeviceType::Metal(0));
        }

        if cfg!(feature = "wgpu") {
            types.push(DeviceType::Wgpu(0));
        }

        types
    }
}

/// Utility functions for device implementations
pub mod utils {
    use super::*;

    /// Cast a device to a specific implementation type
    pub fn cast_device<T: Device + 'static>(device: &dyn Device) -> Option<&T> {
        device.as_any().downcast_ref::<T>()
    }

    /// Cast a device mutably to a specific implementation type
    pub fn cast_device_mut<T: Device + 'static>(device: &mut dyn Device) -> Option<&mut T> {
        device.as_any_mut().downcast_mut::<T>()
    }

    /// Check if a device is a CPU device
    pub fn is_cpu_device(device: &dyn Device) -> bool {
        cast_device::<CpuDevice>(device).is_some()
    }

    /// Check if a device is a CUDA device
    pub fn is_cuda_device(device: &dyn Device) -> bool {
        cast_device::<CudaDevice>(device).is_some()
    }

    /// Check if a device is a Metal device
    pub fn is_metal_device(device: &dyn Device) -> bool {
        cast_device::<MetalDevice>(device).is_some()
    }

    /// Check if a device is a WebGPU device
    pub fn is_wgpu_device(device: &dyn Device) -> bool {
        cast_device::<WgpuDevice>(device).is_some()
    }

    /// Get device implementation name
    pub fn device_implementation_name(device: &dyn Device) -> &'static str {
        if is_cpu_device(device) {
            "CPU"
        } else if is_cuda_device(device) {
            "CUDA"
        } else if is_metal_device(device) {
            "Metal"
        } else if is_wgpu_device(device) {
            "WebGPU"
        } else {
            "Unknown"
        }
    }

    /// Create devices for all available types
    pub fn create_all_available_devices() -> Vec<Box<dyn Device>> {
        let mut devices = Vec::new();

        for device_type in DeviceFactory::available_device_types() {
            if let Ok(device) = DeviceFactory::create_device(device_type) {
                devices.push(device);
            }
        }

        devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = CpuDevice::new();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.name(), "CPU");
        assert!(device.is_available().unwrap());

        let cloned = device.clone_device().unwrap();
        assert_eq!(cloned.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_cpu_device_with_threads() {
        let device = CpuDevice::with_threads(4).unwrap();
        assert!(device.thread_pool().is_some());

        let result = device.execute_parallel(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_simd_level_detection() {
        let level = CpuDevice::detect_simd_level();
        // Just ensure it doesn't panic and returns a valid level
        match level {
            SimdLevel::None
            | SimdLevel::Sse
            | SimdLevel::Avx
            | SimdLevel::Avx2
            | SimdLevel::Avx512 => {}
        }
    }

    #[test]
    fn test_device_factory() {
        let cpu_device = DeviceFactory::create_device(DeviceType::Cpu).unwrap();
        assert_eq!(cpu_device.device_type(), DeviceType::Cpu);

        assert!(DeviceFactory::is_device_type_available(DeviceType::Cpu));

        let available_types = DeviceFactory::available_device_types();
        assert!(available_types.contains(&DeviceType::Cpu));
    }

    #[test]
    fn test_device_casting() {
        let device = CpuDevice::new();
        let device_ref: &dyn Device = &device;

        assert!(utils::is_cpu_device(device_ref));
        assert!(!utils::is_cuda_device(device_ref));
        assert!(!utils::is_metal_device(device_ref));
        assert!(!utils::is_wgpu_device(device_ref));

        let cpu_device = utils::cast_device::<CpuDevice>(device_ref);
        assert!(cpu_device.is_some());

        assert_eq!(utils::device_implementation_name(device_ref), "CPU");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_device() {
        if let Ok(device) = CudaDevice::new(0) {
            assert_eq!(device.device_type(), DeviceType::Cuda(0));
            assert_eq!(device.device_index(), 0);
            assert!(device.is_available().unwrap());

            if let Ok(stream_id) = device.create_stream() {
                assert!(stream_id > 0);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_metal_device() {
        if let Ok(device) = MetalDevice::new(0) {
            assert_eq!(device.device_type(), DeviceType::Metal(0));
            assert_eq!(device.device_index(), 0);
            assert!(device.is_available().unwrap());
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_wgpu_device() {
        if let Ok(device) = WgpuDevice::new(0) {
            assert_eq!(device.device_type(), DeviceType::Wgpu(0));
            assert_eq!(device.device_index(), 0);
            assert!(device.is_available().unwrap());
        }
    }

    #[test]
    fn test_create_all_available_devices() {
        let devices = utils::create_all_available_devices();
        assert!(!devices.is_empty()); // At least CPU should be available

        // Check that CPU device is present
        assert!(devices.iter().any(|d| d.device_type() == DeviceType::Cpu));
    }
}
