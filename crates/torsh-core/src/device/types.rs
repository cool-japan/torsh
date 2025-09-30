//! Core device types and fundamental definitions
//!
//! This module provides the fundamental types used throughout the device system,
//! including the core DeviceType enumeration and utility functions for device
//! identification and parsing.

use std::fmt;
use std::str::FromStr;

/// Device types supported by ToRSh
///
/// This enum represents the different types of compute devices that can be used
/// for tensor operations in ToRSh. Each variant includes device-specific information
/// such as device indices for multi-GPU systems.
///
/// # Examples
///
/// ```
/// use torsh_core::DeviceType;
///
/// // Create a CPU device
/// let cpu_device = DeviceType::Cpu;
/// println!("Device: {}", cpu_device); // "cpu"
///
/// // Create a CUDA device with index 0
/// let cuda_device = DeviceType::Cuda(0);
/// println!("Device: {}", cuda_device); // "cuda:0"
///
/// // Create a Metal device for Apple Silicon
/// let metal_device = DeviceType::Metal(0);
/// println!("Device: {}", metal_device); // "metal:0"
///
/// // Device types can be compared for equality
/// assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
/// assert_ne!(DeviceType::Cuda(0), DeviceType::Cuda(1));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceType {
    /// CPU device for host-side computations
    Cpu,
    /// CUDA GPU device with device index
    Cuda(usize),
    /// Metal GPU device (Apple Silicon) with device index
    Metal(usize),
    /// WebGPU device with device index
    Wgpu(usize),
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda(id) => write!(f, "cuda:{id}"),
            DeviceType::Metal(id) => write!(f, "metal:{id}"),
            DeviceType::Wgpu(id) => write!(f, "wgpu:{id}"),
        }
    }
}

impl FromStr for DeviceType {
    type Err = crate::error::TorshError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_device_string(s)
    }
}

impl DeviceType {
    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceType::Cpu)
    }

    /// Check if this is a GPU device (any GPU type)
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            DeviceType::Cuda(_) | DeviceType::Metal(_) | DeviceType::Wgpu(_)
        )
    }

    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self, DeviceType::Cuda(_))
    }

    /// Check if this is a Metal device
    pub fn is_metal(&self) -> bool {
        matches!(self, DeviceType::Metal(_))
    }

    /// Check if this is a WebGPU device
    pub fn is_wgpu(&self) -> bool {
        matches!(self, DeviceType::Wgpu(_))
    }

    /// Get the device index if applicable
    ///
    /// Returns Some(index) for indexed devices (GPU), None for CPU
    pub fn index(&self) -> Option<usize> {
        match self {
            DeviceType::Cpu => None,
            DeviceType::Cuda(id) | DeviceType::Metal(id) | DeviceType::Wgpu(id) => Some(*id),
        }
    }

    /// Get the backend name as a string
    pub fn backend_name(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Cuda(_) => "cuda",
            DeviceType::Metal(_) => "metal",
            DeviceType::Wgpu(_) => "wgpu",
        }
    }

    /// Create a new device with a different index
    ///
    /// For CPU devices, returns the same CPU device regardless of index
    pub fn with_index(&self, index: usize) -> DeviceType {
        match self {
            DeviceType::Cpu => DeviceType::Cpu,
            DeviceType::Cuda(_) => DeviceType::Cuda(index),
            DeviceType::Metal(_) => DeviceType::Metal(index),
            DeviceType::Wgpu(_) => DeviceType::Wgpu(index),
        }
    }

    /// Get all available device types for the current platform
    pub fn available_types() -> Vec<DeviceType> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut types = vec![DeviceType::Cpu];

        // Add GPU types based on platform support
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would query CUDA devices
            types.push(DeviceType::Cuda(0));
        }

        #[cfg(target_os = "macos")]
        {
            types.push(DeviceType::Metal(0));
        }

        #[cfg(feature = "wgpu")]
        {
            types.push(DeviceType::Wgpu(0));
        }

        types
    }

    /// Check if this device type requires special initialization
    pub fn requires_initialization(&self) -> bool {
        match self {
            DeviceType::Cpu => false,
            DeviceType::Cuda(_) | DeviceType::Metal(_) | DeviceType::Wgpu(_) => true,
        }
    }

    /// Get the typical memory alignment for this device type
    pub fn typical_alignment(&self) -> usize {
        match self {
            DeviceType::Cpu => 64,       // Cache line alignment
            DeviceType::Cuda(_) => 256,  // CUDA texture alignment
            DeviceType::Metal(_) => 256, // Metal buffer alignment
            DeviceType::Wgpu(_) => 256,  // WebGPU buffer alignment
        }
    }

    /// Check if devices of this type support peer-to-peer access
    pub fn supports_peer_access(&self) -> bool {
        match self {
            DeviceType::Cpu => false,
            DeviceType::Cuda(_) => true,   // CUDA supports P2P
            DeviceType::Metal(_) => false, // Metal doesn't support P2P
            DeviceType::Wgpu(_) => false,  // WebGPU doesn't support P2P
        }
    }
}

/// Parse device string (e.g., "cuda:0", "cpu", "metal:1")
///
/// This function parses device specifications from strings, supporting various
/// formats commonly used in tensor libraries.
///
/// # Supported formats
/// - "cpu" - CPU device
/// - "cuda" or "cuda:0" - CUDA device (defaults to index 0)
/// - "cuda:N" - CUDA device with specific index N
/// - "metal" or "metal:0" - Metal device (defaults to index 0)
/// - "metal:N" - Metal device with specific index N
/// - "wgpu" or "wgpu:0" - WebGPU device (defaults to index 0)
/// - "wgpu:N" - WebGPU device with specific index N
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::parse_device_string;
///
/// let cpu = parse_device_string("cpu").unwrap();
/// assert_eq!(cpu, DeviceType::Cpu);
///
/// let cuda = parse_device_string("cuda:1").unwrap();
/// assert_eq!(cuda, DeviceType::Cuda(1));
///
/// let metal = parse_device_string("metal").unwrap();
/// assert_eq!(metal, DeviceType::Metal(0));
/// ```
pub fn parse_device_string(device_str: &str) -> Result<DeviceType, crate::error::TorshError> {
    let device_str = device_str.trim().to_lowercase();

    if device_str == "cpu" {
        return Ok(DeviceType::Cpu);
    }

    // Handle GPU devices with optional indices
    if let Some(colon_pos) = device_str.find(':') {
        let (backend, index_str) = device_str.split_at(colon_pos);
        let index_str = &index_str[1..]; // Skip the colon

        let index = index_str.parse::<usize>().map_err(|_| {
            crate::error::TorshError::InvalidArgument(format!(
                "Invalid device index '{}' in device string '{}'",
                index_str, device_str
            ))
        })?;

        match backend {
            "cuda" => Ok(DeviceType::Cuda(index)),
            "metal" => Ok(DeviceType::Metal(index)),
            "wgpu" => Ok(DeviceType::Wgpu(index)),
            _ => Err(crate::error::TorshError::InvalidArgument(format!(
                "Unknown device backend '{}' in device string '{}'",
                backend, device_str
            ))),
        }
    } else {
        // Handle GPU devices without indices (default to 0)
        match device_str.as_str() {
            "cuda" => Ok(DeviceType::Cuda(0)),
            "metal" => Ok(DeviceType::Metal(0)),
            "wgpu" => Ok(DeviceType::Wgpu(0)),
            _ => Err(crate::error::TorshError::InvalidArgument(format!(
                "Unknown device string '{}'",
                device_str
            ))),
        }
    }
}

/// Device-related constants and limits
pub mod constants {
    /// Maximum number of devices per backend type
    pub const MAX_DEVICES_PER_BACKEND: usize = 16;

    /// Maximum device index value
    pub const MAX_DEVICE_INDEX: usize = 255;

    /// Default device index for unspecified GPU devices
    pub const DEFAULT_GPU_INDEX: usize = 0;

    /// Typical cache line size for CPU devices
    pub const CPU_CACHE_LINE_SIZE: usize = 64;

    /// Typical GPU memory alignment
    pub const GPU_MEMORY_ALIGNMENT: usize = 256;

    /// Maximum device name length
    pub const MAX_DEVICE_NAME_LENGTH: usize = 256;
}

/// Utility functions for device type operations
pub mod utils {
    use super::*;

    /// Check if two device types are compatible for operations
    pub fn devices_compatible(a: DeviceType, b: DeviceType) -> bool {
        match (a, b) {
            // Same devices are always compatible
            (DeviceType::Cpu, DeviceType::Cpu) => true,
            (DeviceType::Cuda(a_idx), DeviceType::Cuda(b_idx)) => a_idx == b_idx,
            (DeviceType::Metal(a_idx), DeviceType::Metal(b_idx)) => a_idx == b_idx,
            (DeviceType::Wgpu(a_idx), DeviceType::Wgpu(b_idx)) => a_idx == b_idx,
            // Different device types require explicit transfer
            _ => false,
        }
    }

    /// Get the transfer cost between two devices (arbitrary units)
    pub fn transfer_cost(from: DeviceType, to: DeviceType) -> u32 {
        if devices_compatible(from, to) {
            return 0; // No transfer needed
        }

        match (from, to) {
            // CPU to GPU transfers
            (DeviceType::Cpu, DeviceType::Cuda(_)) => 100,
            (DeviceType::Cpu, DeviceType::Metal(_)) => 80,
            (DeviceType::Cpu, DeviceType::Wgpu(_)) => 120,

            // GPU to CPU transfers
            (DeviceType::Cuda(_), DeviceType::Cpu) => 100,
            (DeviceType::Metal(_), DeviceType::Cpu) => 80,
            (DeviceType::Wgpu(_), DeviceType::Cpu) => 120,

            // GPU to GPU transfers (same type, different index)
            (DeviceType::Cuda(_), DeviceType::Cuda(_)) => 50,
            (DeviceType::Metal(_), DeviceType::Metal(_)) => 50,
            (DeviceType::Wgpu(_), DeviceType::Wgpu(_)) => 50,

            // Cross-GPU-type transfers (highest cost)
            _ => 200,
        }
    }

    /// Get a human-readable description of the device
    pub fn device_description(device: DeviceType) -> String {
        match device {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Cuda(idx) => format!("CUDA GPU #{}", idx),
            DeviceType::Metal(idx) => format!("Metal GPU #{}", idx),
            DeviceType::Wgpu(idx) => format!("WebGPU Device #{}", idx),
        }
    }

    /// Check if a device index is valid for the given backend
    pub fn is_valid_device_index(device: DeviceType) -> bool {
        match device.index() {
            Some(idx) => idx <= constants::MAX_DEVICE_INDEX,
            None => true, // CPU devices don't have indices
        }
    }

    /// Normalize device string for consistent parsing
    pub fn normalize_device_string(device_str: &str) -> String {
        device_str.trim().to_lowercase()
    }

    /// Generate all possible device combinations for multi-device operations
    pub fn generate_device_combinations(devices: &[DeviceType]) -> Vec<(DeviceType, DeviceType)> {
        let mut combinations = Vec::new();
        for (i, &device_a) in devices.iter().enumerate() {
            for &device_b in devices.iter().skip(i + 1) {
                combinations.push((device_a, device_b));
            }
        }
        combinations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Cuda(0).to_string(), "cuda:0");
        assert_eq!(DeviceType::Metal(1).to_string(), "metal:1");
        assert_eq!(DeviceType::Wgpu(2).to_string(), "wgpu:2");
    }

    #[test]
    fn test_device_type_properties() {
        assert!(DeviceType::Cpu.is_cpu());
        assert!(!DeviceType::Cpu.is_gpu());
        assert!(DeviceType::Cuda(0).is_gpu());
        assert!(DeviceType::Cuda(0).is_cuda());
        assert!(!DeviceType::Cuda(0).is_metal());

        assert_eq!(DeviceType::Cpu.index(), None);
        assert_eq!(DeviceType::Cuda(5).index(), Some(5));
    }

    #[test]
    fn test_parse_device_string() {
        assert_eq!(parse_device_string("cpu").unwrap(), DeviceType::Cpu);
        assert_eq!(parse_device_string("cuda:0").unwrap(), DeviceType::Cuda(0));
        assert_eq!(parse_device_string("cuda").unwrap(), DeviceType::Cuda(0));
        assert_eq!(
            parse_device_string("metal:1").unwrap(),
            DeviceType::Metal(1)
        );
        assert_eq!(parse_device_string("wgpu:2").unwrap(), DeviceType::Wgpu(2));

        // Test case insensitivity
        assert_eq!(parse_device_string("CPU").unwrap(), DeviceType::Cpu);
        assert_eq!(parse_device_string("CUDA:0").unwrap(), DeviceType::Cuda(0));

        // Test whitespace handling
        assert_eq!(parse_device_string("  cpu  ").unwrap(), DeviceType::Cpu);

        // Test invalid inputs
        assert!(parse_device_string("invalid").is_err());
        assert!(parse_device_string("cuda:abc").is_err());
    }

    #[test]
    fn test_device_compatibility() {
        use utils::*;

        assert!(devices_compatible(DeviceType::Cpu, DeviceType::Cpu));
        assert!(devices_compatible(DeviceType::Cuda(0), DeviceType::Cuda(0)));
        assert!(!devices_compatible(
            DeviceType::Cuda(0),
            DeviceType::Cuda(1)
        ));
        assert!(!devices_compatible(DeviceType::Cpu, DeviceType::Cuda(0)));
    }

    #[test]
    fn test_transfer_cost() {
        use utils::*;

        assert_eq!(transfer_cost(DeviceType::Cpu, DeviceType::Cpu), 0);
        assert_eq!(transfer_cost(DeviceType::Cuda(0), DeviceType::Cuda(0)), 0);
        assert!(transfer_cost(DeviceType::Cpu, DeviceType::Cuda(0)) > 0);
        assert!(transfer_cost(DeviceType::Cuda(0), DeviceType::Cuda(1)) > 0);
    }

    #[test]
    fn test_device_methods() {
        let cuda = DeviceType::Cuda(1);
        assert_eq!(cuda.backend_name(), "cuda");
        assert_eq!(cuda.with_index(5), DeviceType::Cuda(5));
        assert!(cuda.requires_initialization());
        assert_eq!(cuda.typical_alignment(), 256);

        let cpu = DeviceType::Cpu;
        assert_eq!(cpu.backend_name(), "cpu");
        assert_eq!(cpu.with_index(5), DeviceType::Cpu); // CPU ignores index
        assert!(!cpu.requires_initialization());
        assert_eq!(cpu.typical_alignment(), 64);
    }

    #[test]
    fn test_device_combinations() {
        use utils::*;

        let devices = vec![DeviceType::Cpu, DeviceType::Cuda(0), DeviceType::Metal(0)];
        let combinations = generate_device_combinations(&devices);

        assert_eq!(combinations.len(), 3); // 3 choose 2
        assert!(combinations.contains(&(DeviceType::Cpu, DeviceType::Cuda(0))));
        assert!(combinations.contains(&(DeviceType::Cpu, DeviceType::Metal(0))));
        assert!(combinations.contains(&(DeviceType::Cuda(0), DeviceType::Metal(0))));
    }

    #[test]
    fn test_from_str_trait() {
        let device: DeviceType = "cuda:1".parse().unwrap();
        assert_eq!(device, DeviceType::Cuda(1));

        let device: DeviceType = "cpu".parse().unwrap();
        assert_eq!(device, DeviceType::Cpu);
    }
}
