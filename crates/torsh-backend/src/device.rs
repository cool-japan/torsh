//! Device abstraction and management

use torsh_core::device::DeviceType as CoreDeviceType;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Device identifier and properties
#[derive(Debug, Clone, PartialEq)]
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

    /// Builder pattern for creating devices
    pub fn builder() -> DeviceBuilder {
        DeviceBuilder::new()
    }

    /// Get the device ID
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Get the device type
    pub const fn device_type(&self) -> CoreDeviceType {
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

    /// Create a default CPU device
    pub fn cpu() -> crate::BackendResult<Self> {
        DeviceBuilder::new()
            .with_device_type(CoreDeviceType::Cpu)
            .with_name("CPU".to_string())
            .with_vendor("Generic".to_string())
            .with_compute_units(num_cpus::get())
            .build()
    }
}

/// Detailed device information
#[derive(Debug, Clone, PartialEq)]
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

    // WebGPU-specific features
    /// Supports timestamp queries for performance measurement
    TimestampQuery,

    /// Supports timestamp queries inside encoders
    TimestampQueryInsideEncoders,

    /// Supports pipeline statistics queries
    PipelineStatistics,

    /// Supports mappable primary buffers
    MappableBuffers,

    /// Supports buffer binding arrays
    BufferArrays,

    /// Supports storage resource binding arrays
    StorageArrays,

    /// Supports unsized binding arrays
    UnsizedBindingArray,

    /// Supports indirect first instance parameter
    IndirectFirstInstance,

    /// Supports 16-bit floating point in shaders
    ShaderF16,

    /// Supports 16-bit integers in shaders
    ShaderI16,

    /// Supports shader primitive index
    ShaderPrimitiveIndex,

    /// Supports early depth test in shaders
    ShaderEarlyDepthTest,

    /// Supports multi-draw indirect
    MultiDrawIndirect,

    /// Supports multi-draw indirect with count
    MultiDrawIndirectCount,

    /// Supports multisampled shading
    Multisampling,

    /// Supports texture clear operations
    ClearTexture,

    /// Supports SPIR-V shader passthrough
    SpirvShaderPassthrough,

    /// Custom vendor-specific feature
    Custom(String),
}

/// Device builder for constructing devices with validation
#[derive(Debug, Clone)]
pub struct DeviceBuilder {
    id: usize,
    device_type: Option<CoreDeviceType>,
    name: Option<String>,
    info: DeviceInfo,
}

impl DeviceBuilder {
    pub fn new() -> Self {
        Self {
            id: 0,
            device_type: None,
            name: None,
            info: DeviceInfo::default(),
        }
    }

    pub fn with_id(mut self, id: usize) -> Self {
        self.id = id;
        self
    }

    pub fn with_device_type(mut self, device_type: CoreDeviceType) -> Self {
        self.device_type = Some(device_type);
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn with_vendor(mut self, vendor: String) -> Self {
        self.info.vendor = vendor;
        self
    }

    pub fn with_driver_version(mut self, version: String) -> Self {
        self.info.driver_version = version;
        self
    }

    pub fn with_memory(mut self, total: usize, available: usize) -> Self {
        self.info.total_memory = total;
        self.info.available_memory = available;
        self
    }

    pub fn with_compute_units(mut self, units: usize) -> Self {
        self.info.compute_units = units;
        self
    }

    pub fn with_performance(mut self, gflops: f32, bandwidth_gbps: f32) -> Self {
        self.info.peak_gflops = gflops;
        self.info.memory_bandwidth_gbps = bandwidth_gbps;
        self
    }

    pub fn with_feature(mut self, feature: DeviceFeature) -> Self {
        self.info.features.push(feature);
        self
    }

    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.info.properties.push((key, value));
        self
    }

    pub fn build(self) -> crate::BackendResult<Device> {
        let device_type = self.device_type.ok_or_else(|| {
            torsh_core::error::TorshError::BackendError("Device type is required".to_string())
        })?;

        let name = self.name.ok_or_else(|| {
            torsh_core::error::TorshError::BackendError("Device name is required".to_string())
        })?;

        Ok(Device {
            id: self.id,
            device_type,
            name,
            info: self.info,
        })
    }
}

impl Default for DeviceBuilder {
    fn default() -> Self {
        Self::new()
    }
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
            DeviceType::OpenCl => CoreDeviceType::Cpu,   // Fallback
            DeviceType::Vulkan => CoreDeviceType::Cpu,   // Fallback
            DeviceType::Custom => CoreDeviceType::Cpu,   // Fallback
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

/// Unified device management interface for all backends
pub trait DeviceManager: Send + Sync {
    /// Enumerate all available devices for this backend type
    fn enumerate_devices(&self) -> crate::BackendResult<Vec<Device>>;

    /// Get detailed device information by ID
    fn get_device_info(&self, device_id: usize) -> crate::BackendResult<DeviceInfo>;

    /// Check if a device supports specific features
    fn check_device_features(
        &self,
        device_id: usize,
        features: &[DeviceFeature],
    ) -> crate::BackendResult<Vec<bool>>;

    /// Get optimal device configuration for the backend
    fn get_optimal_device_config(
        &self,
        device_id: usize,
    ) -> crate::BackendResult<DeviceConfiguration>;

    /// Validate device availability and readiness
    fn validate_device(&self, device_id: usize) -> crate::BackendResult<bool>;

    /// Get device performance characteristics
    fn get_performance_info(&self, device_id: usize)
        -> crate::BackendResult<DevicePerformanceInfo>;
}

/// Device configuration for optimal performance
#[derive(Debug, Clone)]
pub struct DeviceConfiguration {
    /// Optimal memory allocation size
    pub optimal_allocation_size: usize,

    /// Recommended workgroup/thread block size
    pub workgroup_size: (u32, u32, u32),

    /// Memory alignment requirements
    pub memory_alignment: usize,

    /// Concurrent operation limits
    pub max_concurrent_operations: u32,

    /// Backend-specific configuration
    pub backend_specific: std::collections::HashMap<String, crate::backend::CapabilityValue>,
}

/// Device performance characteristics
#[derive(Debug, Clone)]
pub struct DevicePerformanceInfo {
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,

    /// Compute throughput in GFLOPS
    pub compute_throughput_gflops: f32,

    /// Memory latency in nanoseconds
    pub memory_latency_ns: f32,

    /// Cache hierarchy information
    pub cache_hierarchy: Vec<CacheLevel>,

    /// Thermal information (if available)
    pub thermal_info: Option<ThermalInfo>,

    /// Power consumption information (if available)
    pub power_info: Option<PowerInfo>,
}

/// Cache level information
#[derive(Debug, Clone)]
pub struct CacheLevel {
    pub level: u8,
    pub size_bytes: usize,
    pub line_size_bytes: usize,
    pub associativity: Option<usize>,
}

/// Thermal monitoring information
#[derive(Debug, Clone)]
pub struct ThermalInfo {
    pub current_temperature_celsius: f32,
    pub max_temperature_celsius: f32,
    pub thermal_throttling_active: bool,
}

/// Power consumption information
#[derive(Debug, Clone)]
pub struct PowerInfo {
    pub current_power_watts: f32,
    pub max_power_watts: f32,
    pub power_limit_watts: f32,
}

/// Common device management utilities that can be shared across backends
pub struct DeviceUtils;

impl DeviceUtils {
    /// Validate device configuration parameters
    pub const fn validate_device_id(device_id: usize, max_devices: usize) -> bool {
        device_id < max_devices
    }

    /// Calculate device score for selection algorithms
    pub fn calculate_device_score(device: &Device, requirements: &DeviceRequirements) -> f32 {
        let mut score = 0.0;

        // Memory requirement scoring
        if let Some(min_memory) = requirements.min_memory {
            if device.info.total_memory >= min_memory {
                score += 20.0;
                // Bonus for having more memory than required
                score += (device.info.total_memory as f32 / min_memory as f32 - 1.0) * 5.0;
            } else {
                return 0.0; // Disqualify if insufficient memory
            }
        }

        // Compute units requirement scoring
        if let Some(min_compute_units) = requirements.min_compute_units {
            if device.info.compute_units >= min_compute_units {
                score += 15.0;
                score += (device.info.compute_units as f32 / min_compute_units as f32 - 1.0) * 3.0;
            } else {
                return 0.0;
            }
        }

        // Features requirement scoring
        for required_feature in &requirements.required_features {
            if device.supports_feature(required_feature.clone()) {
                score += 10.0;
            } else {
                return 0.0; // Disqualify if missing required feature
            }
        }

        // Performance scores
        score += device.info.peak_gflops / 1000.0; // Bonus for compute performance
        score += device.info.memory_bandwidth_gbps / 100.0; // Bonus for memory bandwidth

        // Backend preference
        match DeviceType::from(device.device_type) {
            DeviceType::Cuda => score += 15.0,  // Prefer CUDA
            DeviceType::Metal => score += 10.0, // Then Metal
            DeviceType::WebGpu => score += 5.0, // Then WebGPU
            DeviceType::Cpu => score += 1.0,    // CPU as fallback
            _ => score += 0.0,
        }

        score
    }

    /// Check if device meets minimum requirements
    pub fn meets_requirements(device: &Device, requirements: &DeviceRequirements) -> bool {
        // Check memory requirement
        if let Some(min_memory) = requirements.min_memory {
            if device.info.total_memory < min_memory {
                return false;
            }
        }

        // Check compute units requirement
        if let Some(min_compute_units) = requirements.min_compute_units {
            if device.info.compute_units < min_compute_units {
                return false;
            }
        }

        // Check required features
        for required_feature in &requirements.required_features {
            if !device.supports_feature(required_feature.clone()) {
                return false;
            }
        }

        // Check backend preference
        if let Some(preferred_backend) = requirements.preferred_backend {
            let device_backend = match DeviceType::from(device.device_type) {
                DeviceType::Cpu => crate::backend::BackendType::Cpu,
                DeviceType::Cuda => crate::backend::BackendType::Cuda,
                DeviceType::Metal => crate::backend::BackendType::Metal,
                DeviceType::WebGpu => crate::backend::BackendType::WebGpu,
                _ => return false,
            };
            if device_backend != preferred_backend {
                return false;
            }
        }

        true
    }

    /// Get optimal workgroup/thread block size for device
    pub fn get_optimal_workgroup_size(device: &Device, operation_type: &str) -> (u32, u32, u32) {
        match DeviceType::from(device.device_type) {
            DeviceType::Cuda => {
                // CUDA optimal sizes
                match operation_type {
                    "matrix_mul" => (16, 16, 1),
                    "element_wise" => (256, 1, 1),
                    "reduction" => (512, 1, 1),
                    _ => (32, 32, 1),
                }
            }
            DeviceType::Metal => {
                // Metal optimal sizes
                match operation_type {
                    "matrix_mul" => (16, 16, 1),
                    "element_wise" => (256, 1, 1),
                    "reduction" => (256, 1, 1),
                    _ => (32, 32, 1),
                }
            }
            DeviceType::WebGpu => {
                // WebGPU optimal sizes
                match operation_type {
                    "matrix_mul" => (8, 8, 1),
                    "element_wise" => (64, 1, 1),
                    "reduction" => (64, 1, 1),
                    _ => (8, 8, 1),
                }
            }
            _ => {
                // Default fallback
                (1, 1, 1)
            }
        }
    }
}

/// Common device discovery utilities
pub struct DeviceDiscovery;

impl DeviceDiscovery {
    /// Discover all available devices across all backends
    pub fn discover_all() -> crate::BackendResult<Vec<(crate::backend::BackendType, Vec<Device>)>> {
        let mut all_devices = Vec::new();

        // CPU devices (always available)
        if let Ok(cpu_devices) = Self::discover_cpu_devices() {
            all_devices.push((crate::backend::BackendType::Cpu, cpu_devices));
        }

        // CUDA devices
        #[cfg(feature = "cuda")]
        if let Ok(cuda_devices) = Self::discover_cuda_devices() {
            if !cuda_devices.is_empty() {
                all_devices.push((crate::backend::BackendType::Cuda, cuda_devices));
            }
        }

        // Metal devices
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Ok(metal_devices) = Self::discover_metal_devices() {
            if !metal_devices.is_empty() {
                all_devices.push((crate::backend::BackendType::Metal, metal_devices));
            }
        }

        // WebGPU devices
        #[cfg(feature = "webgpu")]
        if let Ok(webgpu_devices) = Self::discover_webgpu_devices() {
            if !webgpu_devices.is_empty() {
                all_devices.push((crate::backend::BackendType::WebGpu, webgpu_devices));
            }
        }

        Ok(all_devices)
    }

    /// Find the best device based on requirements
    pub fn find_best_device(
        requirements: &DeviceRequirements,
    ) -> crate::BackendResult<(crate::backend::BackendType, Device)> {
        let all_devices = Self::discover_all()?;

        let mut best_device = None;
        let mut best_score = 0.0;

        for (backend_type, devices) in all_devices {
            for device in devices {
                let score = Self::score_device(&device, requirements);
                if score > best_score {
                    best_score = score;
                    best_device = Some((backend_type, device));
                }
            }
        }

        best_device.ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "No suitable device found for requirements".to_string(),
            )
        })
    }

    /// Score a device based on requirements
    fn score_device(device: &Device, requirements: &DeviceRequirements) -> f32 {
        DeviceUtils::calculate_device_score(device, requirements)
    }

    /// Discover CPU devices
    fn discover_cpu_devices() -> crate::BackendResult<Vec<Device>> {
        let cpu_device = crate::cpu::CpuDevice::new(0, num_cpus::get())?;
        Ok(vec![cpu_device.to_device()])
    }

    /// Discover CUDA devices
    #[cfg(feature = "cuda")]
    fn discover_cuda_devices() -> crate::BackendResult<Vec<Device>> {
        // Implementation would query CUDA runtime for available devices
        // For now, return empty vector
        Ok(vec![])
    }

    /// Discover Metal devices
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn discover_metal_devices() -> crate::BackendResult<Vec<Device>> {
        // Implementation would query Metal framework for available devices
        // For now, return empty vector
        Ok(vec![])
    }

    /// Discover WebGPU devices
    #[cfg(feature = "webgpu")]
    fn discover_webgpu_devices() -> crate::BackendResult<Vec<Device>> {
        // Implementation would query WebGPU for available adapters
        // For now, return empty vector
        Ok(vec![])
    }
}

/// Device requirements for selection
#[derive(Debug, Clone, Default)]
pub struct DeviceRequirements {
    pub min_memory: Option<usize>,
    pub min_compute_units: Option<usize>,
    pub required_features: Vec<DeviceFeature>,
    pub preferred_backend: Option<crate::backend::BackendType>,
    pub max_power_consumption: Option<f32>,
    pub max_temperature: Option<f32>,
}

impl DeviceRequirements {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_min_memory(mut self, memory: usize) -> Self {
        self.min_memory = Some(memory);
        self
    }

    pub fn with_min_compute_units(mut self, units: usize) -> Self {
        self.min_compute_units = Some(units);
        self
    }

    pub fn with_feature(mut self, feature: DeviceFeature) -> Self {
        self.required_features.push(feature);
        self
    }

    pub fn with_preferred_backend(mut self, backend: crate::backend::BackendType) -> Self {
        self.preferred_backend = Some(backend);
        self
    }
}

// Manual implementations for Device to work around f32 fields in DeviceInfo
impl Eq for Device {}

impl std::hash::Hash for Device {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.device_type.hash(state);
        self.name.hash(state);
        // Skip DeviceInfo fields that contain f32 since they don't implement Hash
        self.info.vendor.hash(state);
        self.info.driver_version.hash(state);
        self.info.total_memory.hash(state);
        self.info.available_memory.hash(state);
        self.info.compute_units.hash(state);
        self.info.max_work_group_size.hash(state);
        self.info.max_work_group_dimensions.hash(state);
        self.info.clock_frequency_mhz.hash(state);
        // Skip memory_bandwidth_gbps and peak_gflops (f32 fields)
        self.info.features.hash(state);
        self.info.properties.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device_info() -> DeviceInfo {
        DeviceInfo {
            vendor: "Test Vendor".to_string(),
            driver_version: "1.0.0".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,     // 8GB
            available_memory: 6 * 1024 * 1024 * 1024, // 6GB
            compute_units: 32,
            max_work_group_size: 1024,
            max_work_group_dimensions: vec![1024, 1024, 64],
            clock_frequency_mhz: 1500,
            memory_bandwidth_gbps: 500.0,
            peak_gflops: 10000.0,
            features: vec![
                DeviceFeature::DoublePrecision,
                DeviceFeature::UnifiedMemory,
                DeviceFeature::AtomicOperations,
            ],
            properties: vec![
                ("compute_capability".to_string(), "7.5".to_string()),
                ("warp_size".to_string(), "32".to_string()),
            ],
        }
    }

    #[test]
    fn test_device_creation() {
        let info = create_test_device_info();
        let device = Device::new(
            0,
            CoreDeviceType::Cuda(0),
            "Test GPU".to_string(),
            info.clone(),
        );

        assert_eq!(device.id(), 0);
        assert_eq!(device.name(), "Test GPU");
        assert_eq!(device.device_type(), CoreDeviceType::Cuda(0));
        assert_eq!(device.info().vendor, "Test Vendor");
        assert_eq!(device.info().compute_units, 32);
    }

    #[test]
    fn test_device_feature_support() {
        let info = create_test_device_info();
        let device = Device::new(1, CoreDeviceType::Cpu, "Test CPU".to_string(), info);

        assert!(device.supports_feature(DeviceFeature::DoublePrecision));
        assert!(device.supports_feature(DeviceFeature::UnifiedMemory));
        assert!(device.supports_feature(DeviceFeature::AtomicOperations));
        assert!(!device.supports_feature(DeviceFeature::HalfPrecision));
        assert!(!device.supports_feature(DeviceFeature::SubGroups));
    }

    #[test]
    fn test_device_info_default() {
        let info = DeviceInfo::default();

        assert_eq!(info.vendor, "Unknown");
        assert_eq!(info.driver_version, "Unknown");
        assert_eq!(info.total_memory, 0);
        assert_eq!(info.available_memory, 0);
        assert_eq!(info.compute_units, 1);
        assert_eq!(info.max_work_group_size, 256);
        assert_eq!(info.max_work_group_dimensions, vec![256, 1, 1]);
        assert_eq!(info.clock_frequency_mhz, 1000);
        assert_eq!(info.memory_bandwidth_gbps, 10.0);
        assert_eq!(info.peak_gflops, 100.0);
        assert!(info.features.is_empty());
        assert!(info.properties.is_empty());
    }

    #[test]
    fn test_device_type_conversion() {
        assert_eq!(DeviceType::from(CoreDeviceType::Cpu), DeviceType::Cpu);
        assert_eq!(DeviceType::from(CoreDeviceType::Cuda(0)), DeviceType::Cuda);
        assert_eq!(
            DeviceType::from(CoreDeviceType::Metal(0)),
            DeviceType::Metal
        );
        assert_eq!(
            DeviceType::from(CoreDeviceType::Wgpu(0)),
            DeviceType::WebGpu
        );

        assert_eq!(CoreDeviceType::from(DeviceType::Cpu), CoreDeviceType::Cpu);
        assert_eq!(
            CoreDeviceType::from(DeviceType::Cuda),
            CoreDeviceType::Cuda(0)
        );
        assert_eq!(
            CoreDeviceType::from(DeviceType::Metal),
            CoreDeviceType::Metal(0)
        );
        assert_eq!(
            CoreDeviceType::from(DeviceType::WebGpu),
            CoreDeviceType::Wgpu(0)
        );

        // Fallback conversions
        assert_eq!(
            CoreDeviceType::from(DeviceType::OpenCl),
            CoreDeviceType::Cpu
        );
        assert_eq!(
            CoreDeviceType::from(DeviceType::Vulkan),
            CoreDeviceType::Cpu
        );
        assert_eq!(
            CoreDeviceType::from(DeviceType::Custom),
            CoreDeviceType::Cpu
        );
    }

    #[test]
    fn test_device_feature_variants() {
        let features = [
            DeviceFeature::DoublePrecision,
            DeviceFeature::HalfPrecision,
            DeviceFeature::UnifiedMemory,
            DeviceFeature::AtomicOperations,
            DeviceFeature::SubGroups,
            DeviceFeature::Printf,
            DeviceFeature::Profiling,
            DeviceFeature::PeerToPeer,
            DeviceFeature::ConcurrentExecution,
            DeviceFeature::AsyncMemory,
            DeviceFeature::ImageSupport,
            DeviceFeature::FastMath,
            DeviceFeature::Custom("CustomFeature".to_string()),
        ];

        // Ensure all features are distinct
        for (i, feature1) in features.iter().enumerate() {
            for (j, feature2) in features.iter().enumerate() {
                if i != j {
                    assert_ne!(feature1, feature2);
                }
            }
        }
    }

    #[test]
    fn test_device_selector_creation() {
        let selector = DeviceSelector::new();

        assert_eq!(selector.device_type, None);
        assert_eq!(selector.min_memory, None);
        assert_eq!(selector.min_compute_units, None);
        assert!(selector.required_features.is_empty());
        assert_eq!(selector.preferred_vendor, None);
        assert!(selector.custom_filter.is_none());
    }

    #[test]
    fn test_device_selector_builder() {
        let selector = DeviceSelector::new()
            .with_device_type(DeviceType::Cuda)
            .with_min_memory(4 * 1024 * 1024 * 1024) // 4GB
            .with_min_compute_units(16)
            .with_feature(DeviceFeature::DoublePrecision)
            .with_feature(DeviceFeature::AtomicOperations)
            .with_vendor("NVIDIA".to_string());

        assert_eq!(selector.device_type, Some(DeviceType::Cuda));
        assert_eq!(selector.min_memory, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(selector.min_compute_units, Some(16));
        assert_eq!(selector.required_features.len(), 2);
        assert!(selector
            .required_features
            .contains(&DeviceFeature::DoublePrecision));
        assert!(selector
            .required_features
            .contains(&DeviceFeature::AtomicOperations));
        assert_eq!(selector.preferred_vendor, Some("NVIDIA".to_string()));
    }

    #[test]
    fn test_device_selector_matching() {
        let mut info = create_test_device_info();
        info.vendor = "NVIDIA".to_string();
        info.total_memory = 8 * 1024 * 1024 * 1024; // 8GB
        info.compute_units = 32;

        let device = Device::new(0, CoreDeviceType::Cuda(0), "RTX 4090".to_string(), info);

        // Should match
        let selector1 = DeviceSelector::new()
            .with_device_type(DeviceType::Cuda)
            .with_min_memory(4 * 1024 * 1024 * 1024) // 4GB
            .with_min_compute_units(16)
            .with_feature(DeviceFeature::DoublePrecision)
            .with_vendor("NVIDIA".to_string());

        assert!(selector1.matches(&device));

        // Should not match - insufficient memory
        let selector2 = DeviceSelector::new().with_min_memory(16 * 1024 * 1024 * 1024); // 16GB

        assert!(!selector2.matches(&device));

        // Should not match - missing feature
        let selector3 = DeviceSelector::new().with_feature(DeviceFeature::HalfPrecision);

        assert!(!selector3.matches(&device));

        // Should not match - wrong vendor
        let selector4 = DeviceSelector::new().with_vendor("AMD".to_string());

        assert!(!selector4.matches(&device));
    }

    #[test]
    fn test_custom_device_feature() {
        let custom_feature1 = DeviceFeature::Custom("TensorCores".to_string());
        let custom_feature2 = DeviceFeature::Custom("TensorCores".to_string());
        let custom_feature3 = DeviceFeature::Custom("RTCores".to_string());

        assert_eq!(custom_feature1, custom_feature2);
        assert_ne!(custom_feature1, custom_feature3);

        let mut info = DeviceInfo::default();
        info.features.push(custom_feature1.clone());

        let device = Device::new(0, CoreDeviceType::Cuda(0), "Custom GPU".to_string(), info);
        assert!(device.supports_feature(custom_feature1));
        assert!(!device.supports_feature(custom_feature3));
    }
}
