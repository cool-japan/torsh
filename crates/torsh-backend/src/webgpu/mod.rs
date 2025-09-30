//! WebGPU backend implementation for ToRSh
//!
//! This module provides a complete WebGPU backend that enables GPU acceleration
//! in web browsers and native applications through the WebGPU standard.

#[cfg(feature = "webgpu")]
use wgpu;

pub mod backend;
pub mod buffer;
pub mod device;
pub mod error;
pub mod kernels;
pub mod memory;
pub mod multi_device;
pub mod pipeline;
pub mod shader;

// Re-exports
pub use backend::{WebGpuBackend, WebGpuBackendBuilder};
pub use buffer::{WebGpuBuffer, WebGpuBufferPool};
pub use device::{
    DeviceMemoryInfo, FeatureCompatibilityReport, WebGpuDevice, WebGpuDeviceBuilder,
    WebGpuDeviceCapabilities,
};
pub use error::{WebGpuError, WebGpuResult};
pub use kernels::{WebGpuComputePipeline, WebGpuKernel, WebGpuKernelCache, WebGpuKernelExecutor};
pub use memory::{WebGpuMemoryManager, WebGpuMemoryPool};
pub use multi_device::{
    DeviceAssignment, DeviceFilter, DeviceMetrics, DeviceSelectionContext, LoadBalancingStrategy,
    ManagerStats, MultiDeviceConfig, MultiDeviceWebGpuManager, PerformanceMonitor, SystemMetrics,
    WorkDistributionPlan, WorkGranularity, WorkPartition, WorkPriority,
};
pub use pipeline::{ComputePipeline, PipelineCache, PipelineFactory};
pub use shader::{ShaderCompiler, ShaderModule, ShaderSource};

use parking_lot::RwLock;
use std::sync::Arc;

/// WebGPU adapter information
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: wgpu::DeviceType,
    pub driver_info: String,
    pub backend: wgpu::Backend,
}

impl From<wgpu::AdapterInfo> for AdapterInfo {
    fn from(info: wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: info.device_type,
            driver_info: info.driver_info,
            backend: info.backend,
        }
    }
}

/// Global WebGPU instance
static WEBGPU_INSTANCE: RwLock<Option<Arc<wgpu::Instance>>> = RwLock::new(None);

/// Initialize WebGPU
pub async fn init() -> WebGpuResult<()> {
    let mut instance_lock = WEBGPU_INSTANCE.write();
    if instance_lock.is_none() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        *instance_lock = Some(Arc::new(instance));
    }
    Ok(())
}

/// Get the global WebGPU instance
pub fn instance() -> Arc<wgpu::Instance> {
    let instance_lock = WEBGPU_INSTANCE.read();
    instance_lock
        .as_ref()
        .expect("WebGPU instance not initialized - call init() first")
        .clone()
}

/// Check if WebGPU is available
pub fn is_available() -> bool {
    // Check if WebGPU is available on this platform
    cfg!(feature = "webgpu") && {
        // Try to create an instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Check if we can enumerate adapters
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        !adapters.is_empty()
    }
}

/// Enumerate available WebGPU adapters
pub async fn enumerate_adapters() -> WebGpuResult<Vec<wgpu::Adapter>> {
    if !is_available() {
        return Err(WebGpuError::NotAvailable);
    }

    let instance = instance();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    Ok(adapters)
}

/// Get the best available adapter
pub async fn get_best_adapter() -> WebGpuResult<wgpu::Adapter> {
    if !is_available() {
        return Err(WebGpuError::NotAvailable);
    }

    let instance = instance();

    // Try to get a high-performance adapter first
    if let Ok(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        return Ok(adapter);
    }

    // Fall back to any available adapter
    if let Ok(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        return Ok(adapter);
    }

    Err(WebGpuError::NoAdapterFound)
}

/// Get adapter information
pub fn get_adapter_info(adapter: &wgpu::Adapter) -> AdapterInfo {
    adapter.get_info().into()
}

/// Get number of WebGPU devices/adapters
pub fn device_count() -> Option<usize> {
    if !is_available() {
        return Some(0);
    }

    // Create a temporary runtime to execute async code
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return Some(0),
    };

    rt.block_on(async {
        match enumerate_adapters().await {
            Ok(adapters) => Some(adapters.len()),
            Err(_) => Some(0),
        }
    })
}

/// WebGPU backend configuration
#[derive(Debug, Clone)]
pub struct WebGpuBackendConfig {
    pub adapter_index: Option<usize>,
    pub power_preference: wgpu::PowerPreference,
    pub debug_mode: bool,
    pub max_buffer_size: u64,
    pub enable_pipeline_cache: bool,
    pub preferred_workgroup_size: (u32, u32, u32),
}

impl Default for WebGpuBackendConfig {
    fn default() -> Self {
        Self {
            adapter_index: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            debug_mode: false,
            max_buffer_size: 2 * 1024 * 1024 * 1024, // 2GB
            enable_pipeline_cache: true,
            preferred_workgroup_size: (64, 1, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_webgpu_availability() {
        println!("WebGPU available: {}", is_available());

        if is_available() {
            let result = init().await;
            assert!(result.is_ok());

            let adapters = enumerate_adapters().await;
            if let Ok(adapters) = adapters {
                println!("Found {} WebGPU adapters", adapters.len());
                for (i, adapter) in adapters.iter().enumerate() {
                    let info = get_adapter_info(adapter);
                    println!("Adapter {}: {} ({:?})", i, info.name, info.device_type);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_best_adapter() {
        if is_available() {
            let _ = init().await;
            let result = get_best_adapter().await;
            if let Ok(adapter) = result {
                let info = get_adapter_info(&adapter);
                println!("Best adapter: {} ({:?})", info.name, info.device_type);
            }
        }
    }

    #[test]
    fn test_backend_config() {
        let config = WebGpuBackendConfig::default();
        assert_eq!(
            config.power_preference,
            wgpu::PowerPreference::HighPerformance
        );
        assert!(!config.debug_mode);
        assert_eq!(config.max_buffer_size, 2 * 1024 * 1024 * 1024);
        assert!(config.enable_pipeline_cache);
        assert_eq!(config.preferred_workgroup_size, (64, 1, 1));
    }
}
