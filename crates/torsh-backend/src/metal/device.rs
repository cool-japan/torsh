//! Metal device management

use metal::{CommandBuffer, CommandQueue, Device, DeviceRef, MTLResourceOptions};
use std::sync::{Arc, Mutex};
// use torsh_backends::BackendDevice; // TODO: This trait doesn't exist in current API

#[cfg(feature = "log")]
use log;

use crate::metal::error::{MetalError, Result};

/// Metal device wrapper
#[derive(Clone)]
pub struct MetalDevice {
    /// The underlying Metal device
    device: Device,
    /// Command queue for executing GPU commands
    command_queue: Arc<Mutex<CommandQueue>>,
    /// Device information
    info: DeviceInfo,
}

/// Information about a Metal device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub is_low_power: bool,
    pub is_removable: bool,
    pub has_unified_memory: bool,
    pub max_threadgroup_memory: u64,
    pub max_threads_per_threadgroup: usize,
}

impl MetalDevice {
    /// Create a new Metal device (uses default system device)
    pub fn new() -> Result<Self> {
        Self::with_device(
            Device::system_default()
                .ok_or_else(|| MetalError::InvalidArgument("No Metal device found".to_string()))?,
        )
    }

    /// Create a Metal device wrapper with a specific device
    pub fn with_device(device: Device) -> Result<Self> {
        let command_queue = device.new_command_queue();

        let info = DeviceInfo {
            name: device.name().to_string(),
            is_low_power: device.is_low_power(),
            is_removable: device.is_removable(),
            has_unified_memory: device.has_unified_memory(),
            max_threadgroup_memory: device.max_threadgroup_memory_length(),
            max_threads_per_threadgroup: device.max_threads_per_threadgroup().width as usize,
        };

        #[cfg(feature = "log")]
        {
            log::info!("Initialized Metal device: {}", info.name);
            log::info!("Unified memory: {}", info.has_unified_memory);
            log::info!(
                "Max threadgroup memory: {} bytes",
                info.max_threadgroup_memory
            );
        }

        Ok(Self {
            device,
            command_queue: Arc::new(Mutex::new(command_queue)),
            info,
        })
    }

    /// Get the underlying Metal device
    pub fn device(&self) -> &DeviceRef {
        &self.device
    }

    /// Get the underlying Metal device as &Device
    pub fn device_ref(&self) -> &Device {
        &self.device
    }

    /// Get device information
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get device name
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the maximum buffer length
    pub fn max_buffer_length(&self) -> usize {
        self.device.max_buffer_length() as usize
    }

    /// Get the maximum threads per threadgroup
    pub fn max_threads_per_threadgroup(&self) -> (usize, usize, usize) {
        let size = self.device.max_threads_per_threadgroup();
        (
            size.width as usize,
            size.height as usize,
            size.depth as usize,
        )
    }

    /// Create a new command buffer
    pub fn new_command_buffer(&self) -> Result<CommandBuffer> {
        let queue = self.command_queue.lock().map_err(|e| {
            MetalError::InvalidArgument(format!("Failed to lock command queue: {}", e))
        })?;

        let buffer = queue.new_command_buffer();
        Ok(buffer.to_owned())
    }

    /// Synchronize device (wait for all commands to complete)
    pub fn synchronize(&self) -> Result<()> {
        let buffer = self.new_command_buffer()?;
        buffer.commit();
        buffer.wait_until_completed();
        Ok(())
    }

    /// Check if a feature is supported
    pub fn supports_feature(&self, feature: MetalFeature) -> bool {
        match feature {
            MetalFeature::Float16 => true, // All Apple Silicon supports FP16
            MetalFeature::RayTracing => self.device.supports_raytracing(),
            MetalFeature::MemorylessRenderTargets => true,
            MetalFeature::TileShading => true,
        }
    }

    /// Get recommended resource options for buffer creation
    pub fn resource_options(&self) -> MTLResourceOptions {
        if self.info.has_unified_memory {
            // For Apple Silicon with unified memory
            MTLResourceOptions::StorageModeShared
        } else {
            // For discrete GPUs
            MTLResourceOptions::StorageModeManaged
        }
    }
}

/// Metal features that can be queried
pub enum MetalFeature {
    Float16,
    RayTracing,
    MemorylessRenderTargets,
    TileShading,
}

// TODO: BackendDevice trait doesn't exist in current API
// impl BackendDevice for MetalDevice {
//     fn is_cpu(&self) -> bool {
//         false
//     }

//     fn is_gpu(&self) -> bool {
//         true
//     }

//     fn synchronize(&self) -> anyhow::Result<()> {
//         MetalDevice::synchronize(self).map_err(Into::into)
//     }

//     fn memory_info(&self) -> anyhow::Result<(usize, usize)> {
//         // Metal doesn't provide direct memory queries
//         // Return approximate values based on system
//         if self.info.has_unified_memory {
//             // Approximate for Apple Silicon
//             let total = 8 * 1024 * 1024 * 1024; // 8GB default
//             let used = 0; // Can't query this easily
//             Ok((used, total))
//         } else {
//             Err(anyhow::anyhow!("Memory info not available for discrete GPUs"))
//         }
//     }
// }

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("name", &self.info.name)
            .field("unified_memory", &self.info.has_unified_memory)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        if Device::system_default().is_some() {
            let device = MetalDevice::new();
            assert!(device.is_ok());

            let device = device.unwrap();
            assert!(!device.info().name.is_empty());
        }
    }
}
