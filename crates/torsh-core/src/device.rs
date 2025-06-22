//! Device abstraction for ToRSh

use crate::error::{Result, TorshError};
use std::fmt;

/// Device types supported by ToRSh
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// CUDA GPU device
    Cuda(usize),
    /// Metal GPU device (Apple Silicon)
    Metal(usize),
    /// WebGPU device
    Wgpu(usize),
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda(id) => write!(f, "cuda:{}", id),
            DeviceType::Metal(id) => write!(f, "metal:{}", id),
            DeviceType::Wgpu(id) => write!(f, "wgpu:{}", id),
        }
    }
}

/// Device trait for backend abstraction
pub trait Device: Send + Sync + 'static {
    /// Get the device type
    fn device_type(&self) -> DeviceType;

    /// Check if the device is available
    fn is_available(&self) -> bool;

    /// Synchronize the device
    fn synchronize(&self) -> Result<()>;

    /// Get device name/description
    fn name(&self) -> &str;

    /// Get device memory info (free, total) in bytes
    fn memory_info(&self) -> Result<(usize, usize)>;
}

/// CPU device implementation
#[derive(Debug, Clone)]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous
        Ok(())
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn memory_info(&self) -> Result<(usize, usize)> {
        // For CPU, return system memory info
        // This is a placeholder - real implementation would query system
        Ok((8_000_000_000, 16_000_000_000)) // 8GB free, 16GB total
    }
}

/// Get the default device
pub fn default_device() -> impl Device {
    CpuDevice
}

/// Parse device string (e.g., "cuda:0", "cpu")
pub fn parse_device(device_str: &str) -> Result<DeviceType> {
    match device_str {
        "cpu" => Ok(DeviceType::Cpu),
        s if s.starts_with("cuda:") => {
            let id = s[5..]
                .parse::<usize>()
                .map_err(|_| TorshError::InvalidShape(format!("Invalid device: {}", s)))?;
            Ok(DeviceType::Cuda(id))
        }
        s if s.starts_with("metal:") => {
            let id = s[6..]
                .parse::<usize>()
                .map_err(|_| TorshError::InvalidShape(format!("Invalid device: {}", s)))?;
            Ok(DeviceType::Metal(id))
        }
        s if s.starts_with("wgpu:") => {
            let id = s[5..]
                .parse::<usize>()
                .map_err(|_| TorshError::InvalidShape(format!("Invalid device: {}", s)))?;
            Ok(DeviceType::Wgpu(id))
        }
        _ => Err(TorshError::InvalidShape(format!(
            "Unknown device: {}",
            device_str
        ))),
    }
}
