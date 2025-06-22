//! Device abstraction and management

use torsh_core::device::DeviceType as CoreDeviceType;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Device identifier and properties
#[derive(Debug, Clone)]
pub struct Device {
    /// Unique device ID within the backend
    pub id: usize,
    
    /// Device type (CPU, CUDA, Metal, etc.)
    pub device_type: CoreDeviceType,
    
    /// Human-readable device name
    pub name: String,
    
    /// Device information and capabilities
    pub info: DeviceInfo,
}

impl Device {
    /// Create a new device
    pub fn new(id: usize, device_type: CoreDeviceType, name: String, info: DeviceInfo) -> Self {
        Self {
            id,
            device_type,
            name,
            info,
        }
    }
    
    /// Get the device ID
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Get the device type
    pub fn device_type(&self) -> CoreDeviceType {
        self.device_type
    }
    
    /// Get the device name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get device information
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }
    
    /// Check if this device supports the given feature
    pub fn supports_feature(&self, feature: DeviceFeature) -> bool {
        self.info.features.contains(&feature)
    }
}

/// Detailed device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Vendor name (e.g., "NVIDIA", "AMD", "Apple")
    pub vendor: String,
    
    /// Driver version
    pub driver_version: String,
    
    /// Total memory in bytes
    pub total_memory: usize,
    
    /// Available memory in bytes
    pub available_memory: usize,
    
    /// Number of compute units (cores, SMs, etc.)
    pub compute_units: usize,
    
    /// Maximum work group size
    pub max_work_group_size: usize,
    
    /// Maximum work group dimensions
    pub max_work_group_dimensions: Vec<usize>,
    
    /// Clock frequency in MHz
    pub clock_frequency_mhz: u32,
    
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
    
    /// Peak compute performance in GFLOPS
    pub peak_gflops: f32,
    
    /// Supported features
    pub features: Vec<DeviceFeature>,
    
    /// Additional vendor-specific properties
    pub properties: Vec<(String, String)>,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            vendor: "Unknown".to_string(),
            driver_version: "Unknown".to_string(),
            total_memory: 0,
            available_memory: 0,
            compute_units: 1,
            max_work_group_size: 256,
            max_work_group_dimensions: vec![256, 1, 1],
            clock_frequency_mhz: 1000,
            memory_bandwidth_gbps: 10.0,
            peak_gflops: 100.0,
            features: Vec::new(),
            properties: Vec::new(),
        }
    }
}

/// Device features and capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceFeature {
    /// Supports double precision floating point
    DoublePrecision,
    
    /// Supports half precision floating point
    HalfPrecision,
    
    /// Supports unified memory between host and device
    UnifiedMemory,
    
    /// Supports atomic operations
    AtomicOperations,
    
    /// Supports sub-groups/warps
    SubGroups,
    
    /// Supports printf in kernels
    Printf,
    
    /// Supports profiling and debugging
    Profiling,
    
    /// Supports peer-to-peer memory access
    PeerToPeer,
    
    /// Supports concurrent kernel execution
    ConcurrentExecution,
    
    /// Supports asynchronous memory operations
    AsyncMemory,
    
    /// Supports texture/image operations
    ImageSupport,
    
    /// Supports fast math optimizations
    FastMath,
    
    /// Custom vendor-specific feature
    Custom(String),
}

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    
    /// NVIDIA CUDA GPU
    Cuda,
    
    /// Apple Metal GPU
    Metal,
    
    /// WebGPU device
    WebGpu,
    
    /// OpenCL device
    OpenCl,
    
    /// Vulkan Compute device
    Vulkan,
    
    /// Custom device type
    Custom,
}

impl From<CoreDeviceType> for DeviceType {
    fn from(core_type: CoreDeviceType) -> Self {
        match core_type {
            CoreDeviceType::Cpu => DeviceType::Cpu,
            CoreDeviceType::Cuda(_) => DeviceType::Cuda,
            CoreDeviceType::Metal(_) => DeviceType::Metal,
            CoreDeviceType::Wgpu(_) => DeviceType::WebGpu,
        }
    }
}

impl From<DeviceType> for CoreDeviceType {
    fn from(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Cpu => CoreDeviceType::Cpu,
            DeviceType::Cuda => CoreDeviceType::Cuda(0), // Default to device 0
            DeviceType::Metal => CoreDeviceType::Metal(0), // Default to device 0
            DeviceType::WebGpu => CoreDeviceType::Wgpu(0), // Default to device 0
            DeviceType::OpenCl => CoreDeviceType::Cpu, // Fallback
            DeviceType::Vulkan => CoreDeviceType::Cpu, // Fallback
            DeviceType::Custom => CoreDeviceType::Cpu, // Fallback
        }
    }
}

/// Device selection criteria
#[derive(Default)]
pub struct DeviceSelector {
    /// Preferred device type
    pub device_type: Option<DeviceType>,
    
    /// Minimum memory requirement in bytes
    pub min_memory: Option<usize>,
    
    /// Minimum compute units
    pub min_compute_units: Option<usize>,
    
    /// Required features
    pub required_features: Vec<DeviceFeature>,
    
    /// Preferred vendor
    pub preferred_vendor: Option<String>,
    
    /// Custom selection function
    #[allow(clippy::type_complexity)]
    pub custom_filter: Option<Box<dyn Fn(&Device) -> bool + Send + Sync>>,
}


impl DeviceSelector {
    /// Create a new device selector
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set preferred device type
    pub fn with_device_type(mut self, device_type: DeviceType) -> Self {
        self.device_type = Some(device_type);
        self
    }
    
    /// Set minimum memory requirement
    pub fn with_min_memory(mut self, min_memory: usize) -> Self {
        self.min_memory = Some(min_memory);
        self
    }
    
    /// Set minimum compute units
    pub fn with_min_compute_units(mut self, min_compute_units: usize) -> Self {
        self.min_compute_units = Some(min_compute_units);
        self
    }
    
    /// Add required feature
    pub fn with_feature(mut self, feature: DeviceFeature) -> Self {
        self.required_features.push(feature);
        self
    }
    
    /// Set preferred vendor
    pub fn with_vendor(mut self, vendor: String) -> Self {
        self.preferred_vendor = Some(vendor);
        self
    }
    
    /// Check if a device matches this selector
    pub fn matches(&self, device: &Device) -> bool {
        // Check device type
        if let Some(required_type) = &self.device_type {
            if device.device_type != (*required_type).into() {
                return false;
            }
        }
        
        // Check memory
        if let Some(min_memory) = self.min_memory {
            if device.info.total_memory < min_memory {
                return false;
            }
        }
        
        // Check compute units
        if let Some(min_compute_units) = self.min_compute_units {
            if device.info.compute_units < min_compute_units {
                return false;
            }
        }
        
        // Check required features
        for feature in &self.required_features {
            if !device.supports_feature(feature.clone()) {
                return false;
            }
        }
        
        // Check vendor
        if let Some(ref preferred_vendor) = self.preferred_vendor {
            if device.info.vendor != *preferred_vendor {
                return false;
            }
        }
        
        // Apply custom filter
        if let Some(ref filter) = self.custom_filter {
            if !filter(device) {
                return false;
            }
        }
        
        true
    }
}