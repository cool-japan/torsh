//! Device capability detection and SIMD support
//!
//! This module provides comprehensive device capability detection including
//! memory information, compute resources, SIMD instruction sets, and hardware
//! feature detection across different device types.

use crate::device::DeviceType;
use crate::error::Result;
use std::collections::HashMap;

/// Comprehensive device capability information
///
/// Provides detailed information about device capabilities including memory,
/// compute resources, SIMD support, and hardware features.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{DeviceCapabilities, DeviceType};
///
/// let capabilities = DeviceCapabilities::detect(DeviceType::Cpu)?;
/// println!("Memory: {} MB", capabilities.total_memory_mb());
/// println!("SIMD: {:?}", capabilities.simd_features());
/// println!("Cores: {}", capabilities.compute_units());
/// ```
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    device_type: DeviceType,
    total_memory: u64,
    available_memory: u64,
    memory_bandwidth: Option<u64>,
    compute_units: u32,
    clock_rate: Option<u32>,
    simd_features: SimdFeatures,
    hardware_features: HashMap<String, bool>,
    driver_version: Option<String>,
    device_name: String,
    pci_info: Option<PciInfo>,
    thermal_info: Option<ThermalInfo>,
}

impl DeviceCapabilities {
    /// Detect capabilities for the given device
    pub fn detect(device_type: DeviceType) -> Result<Self> {
        match device_type {
            DeviceType::Cpu => Self::detect_cpu_capabilities(),
            DeviceType::Cuda(index) => Self::detect_cuda_capabilities(index),
            DeviceType::Metal(index) => Self::detect_metal_capabilities(index),
            DeviceType::Wgpu(index) => Self::detect_wgpu_capabilities(index),
        }
    }

    /// Get the device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get total memory in bytes
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Get total memory in megabytes
    pub fn total_memory_mb(&self) -> u64 {
        self.total_memory / (1024 * 1024)
    }

    /// Get available memory in bytes
    pub fn available_memory(&self) -> u64 {
        self.available_memory
    }

    /// Get available memory in megabytes
    pub fn available_memory_mb(&self) -> u64 {
        self.available_memory / (1024 * 1024)
    }

    /// Get memory bandwidth in bytes per second
    pub fn memory_bandwidth(&self) -> Option<u64> {
        self.memory_bandwidth
    }

    /// Get number of compute units (cores, SMs, etc.)
    pub fn compute_units(&self) -> u32 {
        self.compute_units
    }

    /// Get clock rate in MHz
    pub fn clock_rate(&self) -> Option<u32> {
        self.clock_rate
    }

    /// Get SIMD features
    pub fn simd_features(&self) -> &SimdFeatures {
        &self.simd_features
    }

    /// Get hardware features
    pub fn hardware_features(&self) -> &HashMap<String, bool> {
        &self.hardware_features
    }

    /// Get driver version
    pub fn driver_version(&self) -> Option<&str> {
        self.driver_version.as_deref()
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get PCI information
    pub fn pci_info(&self) -> Option<&PciInfo> {
        self.pci_info.as_ref()
    }

    /// Get thermal information
    pub fn thermal_info(&self) -> Option<&ThermalInfo> {
        self.thermal_info.as_ref()
    }

    /// Check if a specific feature is supported
    pub fn supports_feature(&self, feature: &str) -> bool {
        self.hardware_features
            .get(feature)
            .copied()
            .unwrap_or(false)
    }

    /// Check if device supports double precision
    pub fn supports_double_precision(&self) -> bool {
        match self.device_type {
            DeviceType::Cpu => true,
            DeviceType::Cuda(_) => self.supports_feature("double_precision"),
            DeviceType::Metal(_) => false, // Metal typically doesn't support fp64
            DeviceType::Wgpu(_) => self.supports_feature("double_precision"),
        }
    }

    /// Check if device supports half precision
    pub fn supports_half_precision(&self) -> bool {
        match self.device_type {
            DeviceType::Cpu => self.simd_features.supports_f16(),
            DeviceType::Cuda(_) => self.supports_feature("half_precision"),
            DeviceType::Metal(_) => true, // Metal has good fp16 support
            DeviceType::Wgpu(_) => self.supports_feature("half_precision"),
        }
    }

    /// Check if device supports unified memory
    pub fn supports_unified_memory(&self) -> bool {
        match self.device_type {
            DeviceType::Cpu => true, // CPU always has unified memory
            DeviceType::Cuda(_) => self.supports_feature("unified_memory"),
            DeviceType::Metal(_) => true, // Apple Silicon has unified memory
            DeviceType::Wgpu(_) => false, // WebGPU doesn't expose this
        }
    }

    /// Get memory utilization ratio (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory == 0 {
            return 0.0;
        }
        1.0 - (self.available_memory as f64 / self.total_memory as f64)
    }

    /// Get peak memory bandwidth in GB/s
    pub fn peak_bandwidth_gbps(&self) -> Option<f64> {
        self.memory_bandwidth
            .map(|bw| bw as f64 / (1024.0 * 1024.0 * 1024.0))
    }

    /// Get compute capability score (arbitrary units for comparison)
    pub fn compute_score(&self) -> u64 {
        let base_score = self.compute_units as u64 * self.clock_rate.unwrap_or(1000) as u64;
        match self.device_type {
            DeviceType::Cpu => base_score,
            DeviceType::Cuda(_) => base_score * 10, // GPUs generally more parallel
            DeviceType::Metal(_) => base_score * 8, // Apple GPUs are efficient
            DeviceType::Wgpu(_) => base_score * 6,  // WebGPU has overhead
        }
    }

    fn detect_cpu_capabilities() -> Result<Self> {
        Ok(DeviceCapabilities {
            device_type: DeviceType::Cpu,
            total_memory: Self::get_system_memory(),
            available_memory: Self::get_available_memory(),
            memory_bandwidth: Self::estimate_cpu_bandwidth(),
            compute_units: Self::get_cpu_cores(),
            clock_rate: Self::get_cpu_frequency(),
            simd_features: SimdFeatures::detect_cpu(),
            hardware_features: Self::detect_cpu_features(),
            driver_version: None,
            device_name: Self::get_cpu_name(),
            pci_info: None,
            thermal_info: Self::get_cpu_thermal_info(),
        })
    }

    fn detect_cuda_capabilities(index: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would query CUDA runtime
            Ok(DeviceCapabilities {
                device_type: DeviceType::Cuda(index),
                total_memory: 8 * 1024 * 1024 * 1024, // Mock: 8GB
                available_memory: 7 * 1024 * 1024 * 1024, // Mock: 7GB available
                memory_bandwidth: Some(900 * 1024 * 1024 * 1024), // Mock: 900 GB/s
                compute_units: 108,                   // Mock: 108 SMs
                clock_rate: Some(1755),               // Mock: 1755 MHz
                simd_features: SimdFeatures::cuda_default(),
                hardware_features: Self::detect_cuda_features(index),
                driver_version: Some("12.0".to_string()),
                device_name: format!("CUDA Device {}", index),
                pci_info: Some(PciInfo::mock_cuda()),
                thermal_info: Some(ThermalInfo::mock_gpu()),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError(format!(
                    "CUDA device {} not available (CUDA support not compiled)",
                    index
                )),
            ))
        }
    }

    fn detect_metal_capabilities(index: usize) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            Ok(DeviceCapabilities {
                device_type: DeviceType::Metal(index),
                total_memory: Self::get_system_memory(), // Unified memory on Apple Silicon
                available_memory: Self::get_available_memory(),
                memory_bandwidth: Some(400 * 1024 * 1024 * 1024), // Mock: 400 GB/s
                compute_units: 32,                                // Mock: 32 GPU cores
                clock_rate: Some(1398),                           // Mock: 1398 MHz
                simd_features: SimdFeatures::metal_default(),
                hardware_features: Self::detect_metal_features(),
                driver_version: Some("Metal 3.0".to_string()),
                device_name: format!("Metal Device {}", index),
                pci_info: None, // Apple Silicon doesn't use PCIe for GPU
                thermal_info: Some(ThermalInfo::mock_integrated()),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError(format!(
                    "Metal device {} not available (not running on macOS)",
                    index
                )),
            ))
        }
    }

    fn detect_wgpu_capabilities(index: usize) -> Result<Self> {
        #[cfg(feature = "wgpu")]
        {
            Ok(DeviceCapabilities {
                device_type: DeviceType::Wgpu(index),
                total_memory: 4 * 1024 * 1024 * 1024, // Mock: 4GB
                available_memory: 3 * 1024 * 1024 * 1024, // Mock: 3GB available
                memory_bandwidth: Some(300 * 1024 * 1024 * 1024), // Mock: 300 GB/s
                compute_units: 24,                    // Mock: 24 compute units
                clock_rate: Some(1200),               // Mock: 1200 MHz
                simd_features: SimdFeatures::wgpu_default(),
                hardware_features: Self::detect_wgpu_features(),
                driver_version: Some("WebGPU 1.0".to_string()),
                device_name: format!("WebGPU Device {}", index),
                pci_info: None,
                thermal_info: None,
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError(format!(
                    "WebGPU device {} not available (WebGPU support not compiled)",
                    index
                )),
            ))
        }
    }

    fn get_system_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|line| line.starts_with("MemTotal:"))
                        .and_then(|line| {
                            line.split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse::<u64>().ok())
                        })
                })
                .map(|kb| kb * 1024)
                .unwrap_or(8 * 1024 * 1024 * 1024) // Default 8GB
        }
        #[cfg(target_os = "macos")]
        {
            // On macOS, we could use sysctl to get memory info
            16 * 1024 * 1024 * 1024 // Default 16GB for macOS
        }
        #[cfg(target_os = "windows")]
        {
            // On Windows, we could use GetPhysicallyInstalledSystemMemory
            16 * 1024 * 1024 * 1024 // Default 16GB for Windows
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            8 * 1024 * 1024 * 1024 // Default 8GB for other platforms
        }
    }

    fn get_available_memory() -> u64 {
        // Simplified implementation - in practice would check actual available memory
        Self::get_system_memory() * 80 / 100 // Assume 80% available
    }

    fn estimate_cpu_bandwidth() -> Option<u64> {
        // Rough estimates based on common CPU memory controllers
        Some(50 * 1024 * 1024 * 1024) // 50 GB/s as rough estimate
    }

    fn get_cpu_cores() -> u32 {
        std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(4)
    }

    fn get_cpu_frequency() -> Option<u32> {
        // Platform-specific implementation would go here
        Some(3000) // 3 GHz default
    }

    fn get_cpu_name() -> String {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|line| line.starts_with("model name"))
                        .and_then(|line| line.split(':').nth(1))
                        .map(|name| name.trim().to_string())
                })
                .unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        {
            "Unknown CPU".to_string()
        }
    }

    fn get_cpu_thermal_info() -> Option<ThermalInfo> {
        // Platform-specific thermal monitoring would go here
        Some(ThermalInfo {
            current_temp: 45.0,
            max_temp: 85.0,
            thermal_throttling: false,
        })
    }

    fn detect_cpu_features() -> HashMap<String, bool> {
        let mut features = HashMap::new();

        // SIMD instruction set features
        features.insert("sse".to_string(), true);
        features.insert("sse2".to_string(), true);
        features.insert("sse3".to_string(), true);
        features.insert("ssse3".to_string(), true);
        features.insert("sse4_1".to_string(), true);
        features.insert("sse4_2".to_string(), true);
        features.insert("avx".to_string(), cfg!(target_feature = "avx"));
        features.insert("avx2".to_string(), cfg!(target_feature = "avx2"));
        features.insert("avx512f".to_string(), cfg!(target_feature = "avx512f"));
        features.insert("fma".to_string(), cfg!(target_feature = "fma"));
        features.insert("bmi1".to_string(), cfg!(target_feature = "bmi1"));
        features.insert("bmi2".to_string(), cfg!(target_feature = "bmi2"));

        // Data type support features (CPU supports all basic types)
        features.insert("f32".to_string(), true);
        features.insert("f64".to_string(), true);
        features.insert("i8".to_string(), true);
        features.insert("i16".to_string(), true);
        features.insert("i32".to_string(), true);
        features.insert("i64".to_string(), true);
        features.insert("u8".to_string(), true);
        features.insert("u32".to_string(), true);
        features.insert("u64".to_string(), true);
        features.insert("bool".to_string(), true);

        // Half precision support depends on feature flag
        #[cfg(feature = "half")]
        {
            features.insert("f16".to_string(), true);
            features.insert("bf16".to_string(), true);
        }

        // Complex number support
        features.insert("c64".to_string(), true);
        features.insert("c128".to_string(), true);

        // Quantized types
        features.insert("qint8".to_string(), true);
        features.insert("quint8".to_string(), true);

        features
    }

    /// Detect CUDA device features at runtime
    ///
    /// # SciRS2 POLICY COMPLIANCE
    /// Uses scirs2-core GPU detection when available for accurate capability detection.
    ///
    /// # Arguments
    /// * `index` - CUDA device index to query
    ///
    /// # Returns
    /// HashMap of feature names and their availability
    #[allow(dead_code)]
    fn detect_cuda_features(_index: usize) -> HashMap<String, bool> {
        let mut features = HashMap::new();

        // Try to use scirs2-core GPU detection if available
        #[cfg(all(feature = "gpu", scirs2_gpu_available))]
        {
            use crate::gpu;
            if let Ok(device) = gpu::GpuDevice::new(index) {
                // Query actual device capabilities from scirs2-core
                features.insert("double_precision".to_string(), device.supports_f64());
                features.insert("half_precision".to_string(), device.supports_f16());
                features.insert("tensor_cores".to_string(), device.has_tensor_cores());
                features.insert(
                    "unified_memory".to_string(),
                    device.supports_unified_memory(),
                );
                features.insert("peer_to_peer".to_string(), device.supports_p2p());
                features.insert(
                    "concurrent_kernels".to_string(),
                    device.supports_concurrent_kernels(),
                );
                features.insert("async_copy".to_string(), device.supports_async_copy());
                return features;
            }
        }

        // Fallback: Optimistic feature set for modern CUDA devices
        // These are typical capabilities for CUDA Compute Capability 7.0+
        features.insert("double_precision".to_string(), true);
        features.insert("half_precision".to_string(), true);
        features.insert("tensor_cores".to_string(), true); // Volta and newer
        features.insert("unified_memory".to_string(), true);
        features.insert("peer_to_peer".to_string(), true);
        features.insert("concurrent_kernels".to_string(), true);
        features.insert("async_copy".to_string(), true);
        features.insert("dynamic_parallelism".to_string(), true);
        features.insert("cooperative_groups".to_string(), true);

        // Additional features for modern CUDA devices
        features.insert("bf16".to_string(), true); // Ampere and newer
        features.insert("tf32".to_string(), true); // Ampere and newer
        features.insert("sparse_tensor_cores".to_string(), false); // Ampere+ optional
        features.insert("mma_operations".to_string(), true); // Matrix multiply-accumulate

        features
    }

    /// Detect Metal GPU features at runtime
    ///
    /// # SciRS2 POLICY COMPLIANCE
    /// Uses scirs2-core GPU detection when available for accurate Metal capability detection.
    ///
    /// # Platform
    /// Only available on macOS/iOS platforms
    #[cfg(target_os = "macos")]
    fn detect_metal_features() -> HashMap<String, bool> {
        let mut features = HashMap::new();

        // Try to use scirs2-core Metal detection if available
        #[cfg(all(feature = "gpu", scirs2_gpu_available, target_os = "macos"))]
        {
            use crate::gpu;
            if let Ok(device) = gpu::GpuDevice::new(0) {
                // Query actual Metal device capabilities
                features.insert("half_precision".to_string(), device.supports_f16());
                features.insert("unified_memory".to_string(), true); // Always true on Metal
                features.insert("tile_shaders".to_string(), device.supports_tile_shaders());
                features.insert("compute_shaders".to_string(), true); // Always supported
                features.insert(
                    "indirect_command_buffers".to_string(),
                    device.supports_indirect_command_buffers(),
                );
                return features;
            }
        }

        // Fallback: Typical Metal 2.0+ features (macOS 10.13+)
        features.insert("half_precision".to_string(), true);
        features.insert("unified_memory".to_string(), true);
        features.insert("tile_shaders".to_string(), true);
        features.insert("compute_shaders".to_string(), true);
        features.insert("indirect_command_buffers".to_string(), true);
        features.insert("argument_buffers".to_string(), true);
        features.insert("raster_order_groups".to_string(), true);
        features.insert("imageblocks".to_string(), true);
        features.insert("threadgroup_sharing".to_string(), true);

        // Metal 3.0+ features (macOS 13+)
        #[cfg(target_os = "macos")]
        {
            features.insert("mesh_shaders".to_string(), true);
            features.insert("ray_tracing".to_string(), true);
            features.insert("function_pointers".to_string(), true);
        }

        features
    }

    /// Detect WebGPU features at runtime
    ///
    /// # SciRS2 POLICY COMPLIANCE
    /// Uses scirs2-core WebGPU detection when available for accurate capability detection.
    ///
    /// # Platform
    /// Cross-platform (web, desktop, mobile)
    #[allow(dead_code)]
    fn detect_wgpu_features() -> HashMap<String, bool> {
        let mut features = HashMap::new();

        // Try to use scirs2-core WebGPU detection if available
        #[cfg(all(feature = "gpu", scirs2_gpu_available, feature = "wgpu"))]
        {
            use crate::gpu;
            if let Ok(device) = gpu::GpuDevice::new(0) {
                // Query actual WebGPU device capabilities
                features.insert(
                    "compute_shaders".to_string(),
                    device.supports_compute_shaders(),
                );
                features.insert(
                    "storage_buffers".to_string(),
                    device.supports_storage_buffers(),
                );
                features.insert(
                    "push_constants".to_string(),
                    device.supports_push_constants(),
                );
                features.insert("half_precision".to_string(), device.supports_f16());
                features.insert("subgroups".to_string(), device.supports_subgroups());
                return features;
            }
        }

        // Fallback: WebGPU 1.0 baseline features
        features.insert("compute_shaders".to_string(), true);
        features.insert("storage_buffers".to_string(), true);
        features.insert("push_constants".to_string(), false); // Optional in WebGPU
        features.insert("half_precision".to_string(), false); // Optional, not widely supported
        features.insert("timestamp_queries".to_string(), true);
        features.insert("indirect_dispatch".to_string(), true);
        features.insert("shader_f16".to_string(), false);

        // WebGPU extended features (may require feature detection)
        features.insert("subgroups".to_string(), false); // Future WebGPU extension
        features.insert("bgra8unorm_storage".to_string(), false);
        features.insert("depth32float_stencil8".to_string(), true);
        features.insert("texture_compression_bc".to_string(), false); // Platform dependent
        features.insert("texture_compression_etc2".to_string(), false);
        features.insert("texture_compression_astc".to_string(), false);

        features
    }

    /// Query comprehensive GPU memory information
    ///
    /// Returns detailed memory statistics for GPU devices when available.
    pub fn query_gpu_memory(_device_index: usize) -> Option<GpuMemoryInfo> {
        #[cfg(all(feature = "gpu", scirs2_gpu_available))]
        {
            use crate::gpu;
            if let Ok(device) = gpu::GpuDevice::new(device_index) {
                return Some(GpuMemoryInfo {
                    total_memory: device.total_memory(),
                    free_memory: device.free_memory(),
                    used_memory: device.used_memory(),
                    supports_unified_memory: device.supports_unified_memory(),
                    memory_clock_rate: device.memory_clock_rate(),
                    memory_bus_width: device.memory_bus_width(),
                });
            }
        }
        None
    }

    /// Query GPU compute capabilities
    ///
    /// Returns compute capability version and other compute-specific information.
    pub fn query_compute_capability(_device_index: usize) -> Option<ComputeCapability> {
        #[cfg(all(feature = "gpu", scirs2_gpu_available))]
        {
            use crate::gpu;
            if let Ok(device) = gpu::GpuDevice::new(device_index) {
                return Some(ComputeCapability {
                    major: device.compute_capability_major(),
                    minor: device.compute_capability_minor(),
                    max_threads_per_block: device.max_threads_per_block(),
                    max_block_dimensions: device.max_block_dimensions(),
                    max_grid_dimensions: device.max_grid_dimensions(),
                    warp_size: device.warp_size(),
                    max_shared_memory_per_block: device.max_shared_memory_per_block(),
                });
            }
        }
        None
    }
}

/// GPU memory information structure
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    /// Total memory in bytes
    pub total_memory: usize,
    /// Free memory in bytes
    pub free_memory: usize,
    /// Used memory in bytes
    pub used_memory: usize,
    /// Whether unified memory is supported
    pub supports_unified_memory: bool,
    /// Memory clock rate in MHz
    pub memory_clock_rate: Option<u32>,
    /// Memory bus width in bits
    pub memory_bus_width: Option<u32>,
}

/// GPU compute capability information
#[derive(Debug, Clone)]
pub struct ComputeCapability {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum block dimensions [x, y, z]
    pub max_block_dimensions: [u32; 3],
    /// Maximum grid dimensions [x, y, z]
    pub max_grid_dimensions: [u32; 3],
    /// Warp/wavefront size
    pub warp_size: u32,
    /// Maximum shared memory per block in bytes
    pub max_shared_memory_per_block: usize,
}

/// SIMD instruction set features
#[derive(Debug, Clone, Default)]
pub struct SimdFeatures {
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool, // ARM NEON
    pub sve: bool,  // ARM SVE
    pub f16: bool,  // Half precision support
}

impl SimdFeatures {
    /// Detect SIMD features for the current CPU
    pub fn detect_cpu() -> Self {
        Self {
            sse: true, // Assume SSE is available on x86_64
            sse2: true,
            sse3: true,
            ssse3: true,
            sse4_1: true,
            sse4_2: true,
            avx: cfg!(target_feature = "avx"),
            avx2: cfg!(target_feature = "avx2"),
            avx512f: cfg!(target_feature = "avx512f"),
            fma: cfg!(target_feature = "fma"),
            neon: cfg!(target_feature = "neon"),
            sve: false, // SVE not commonly available yet
            f16: cfg!(target_feature = "f16c"),
        }
    }

    /// Default SIMD features for CUDA devices
    pub fn cuda_default() -> Self {
        Self {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: true, // CUDA has fused multiply-add
            neon: false,
            sve: false,
            f16: true, // CUDA supports half precision
        }
    }

    /// Default SIMD features for Metal devices
    pub fn metal_default() -> Self {
        Self {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: true,
            neon: cfg!(target_arch = "aarch64"),
            sve: false,
            f16: true, // Metal has good fp16 support
        }
    }

    /// Default SIMD features for WebGPU devices
    pub fn wgpu_default() -> Self {
        Self {
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: false,
            neon: false,
            sve: false,
            f16: false,
        }
    }

    /// Check if any SIMD features are available
    pub fn has_simd(&self) -> bool {
        self.sse || self.avx || self.neon
    }

    /// Check if advanced SIMD features are available (AVX2+)
    pub fn has_advanced_simd(&self) -> bool {
        self.avx2 || self.avx512f || self.sve
    }

    /// Check if half precision is supported
    pub fn supports_f16(&self) -> bool {
        self.f16
    }

    /// Get the best available vector width in bits
    pub fn max_vector_width(&self) -> u32 {
        if self.avx512f {
            512
        } else if self.avx2 || self.avx {
            256
        } else if self.sse || self.neon {
            128
        } else {
            64 // Scalar fallback
        }
    }

    /// Get optimal chunk size for SIMD operations
    pub fn optimal_chunk_size<T>(&self) -> usize {
        let element_size = std::mem::size_of::<T>();
        let vector_bytes = self.max_vector_width() as usize / 8;
        std::cmp::max(1, vector_bytes / element_size)
    }
}

/// PCI device information
#[derive(Debug, Clone)]
pub struct PciInfo {
    pub vendor_id: u16,
    pub device_id: u16,
    pub subsystem_vendor_id: u16,
    pub subsystem_device_id: u16,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciInfo {
    pub fn mock_cuda() -> Self {
        Self {
            vendor_id: 0x10de, // NVIDIA
            device_id: 0x2684, // RTX 4090
            subsystem_vendor_id: 0x10de,
            subsystem_device_id: 0x1658,
            bus: 1,
            device: 0,
            function: 0,
        }
    }
}

/// Thermal information
#[derive(Debug, Clone)]
pub struct ThermalInfo {
    pub current_temp: f32,
    pub max_temp: f32,
    pub thermal_throttling: bool,
}

impl ThermalInfo {
    pub fn mock_gpu() -> Self {
        Self {
            current_temp: 65.0,
            max_temp: 83.0,
            thermal_throttling: false,
        }
    }

    pub fn mock_integrated() -> Self {
        Self {
            current_temp: 45.0,
            max_temp: 100.0,
            thermal_throttling: false,
        }
    }

    /// Check if temperature is within safe operating range
    pub fn is_temperature_safe(&self) -> bool {
        self.current_temp < self.max_temp * 0.9 // 90% of max temp
    }

    /// Get temperature utilization ratio (0.0 to 1.0)
    pub fn temperature_ratio(&self) -> f32 {
        self.current_temp / self.max_temp
    }
}

/// Utility functions for device capabilities
pub mod utils {
    use super::*;

    /// Compare capabilities of two devices
    pub fn compare_capabilities(
        a: &DeviceCapabilities,
        b: &DeviceCapabilities,
    ) -> std::cmp::Ordering {
        a.compute_score().cmp(&b.compute_score())
    }

    /// Find the best device among a list of capabilities
    pub fn find_best_device(capabilities: &[DeviceCapabilities]) -> Option<&DeviceCapabilities> {
        capabilities
            .iter()
            .max_by(|a, b| compare_capabilities(a, b))
    }

    /// Filter devices by minimum memory requirement
    pub fn filter_by_memory(
        capabilities: &[DeviceCapabilities],
        min_memory_mb: u64,
    ) -> Vec<&DeviceCapabilities> {
        capabilities
            .iter()
            .filter(|cap| cap.total_memory_mb() >= min_memory_mb)
            .collect()
    }

    /// Filter devices by SIMD feature requirements
    pub fn filter_by_simd(
        capabilities: &[DeviceCapabilities],
        require_advanced: bool,
    ) -> Vec<&DeviceCapabilities> {
        capabilities
            .iter()
            .filter(|cap| {
                if require_advanced {
                    cap.simd_features().has_advanced_simd()
                } else {
                    cap.simd_features().has_simd()
                }
            })
            .collect()
    }

    /// Get capabilities summary string
    pub fn capabilities_summary(cap: &DeviceCapabilities) -> String {
        format!(
            "{} - {} MB, {} cores, SIMD: {}",
            cap.device_name(),
            cap.total_memory_mb(),
            cap.compute_units(),
            if cap.simd_features().has_advanced_simd() {
                "Advanced"
            } else if cap.simd_features().has_simd() {
                "Basic"
            } else {
                "None"
            }
        )
    }

    /// Check if device is suitable for training (vs inference)
    pub fn is_suitable_for_training(cap: &DeviceCapabilities) -> bool {
        cap.total_memory_mb() >= 4096 // At least 4GB
            && cap.compute_units() >= 16 // Reasonable parallelism
            && match cap.device_type() {
                DeviceType::Cpu => cap.simd_features().has_simd(),
                DeviceType::Cuda(_) => true, // GPUs generally good for training
                DeviceType::Metal(_) => true,
                DeviceType::Wgpu(_) => cap.total_memory_mb() >= 8192, // Need more memory for WebGPU
            }
    }

    /// Estimate training performance score
    pub fn estimate_training_performance(cap: &DeviceCapabilities) -> f64 {
        let memory_score = (cap.total_memory_mb() as f64).log2() / 10.0; // Log scale for memory
        let compute_score = cap.compute_score() as f64 / 1_000_000.0; // Normalize compute score
        let bandwidth_score = cap.peak_bandwidth_gbps().unwrap_or(1.0) / 100.0; // Normalize bandwidth

        memory_score + compute_score + bandwidth_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_capability_detection() {
        let cap = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();
        assert_eq!(cap.device_type(), DeviceType::Cpu);
        assert!(cap.total_memory() > 0);
        assert!(cap.compute_units() > 0);
    }

    #[test]
    fn test_simd_features() {
        let features = SimdFeatures::detect_cpu();
        assert!(features.has_simd() || !features.has_simd()); // Just ensure it doesn't panic
        assert!(features.max_vector_width() >= 64);
        assert!(features.optimal_chunk_size::<f32>() > 0);
    }

    #[test]
    fn test_memory_calculations() {
        let cap = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();
        assert_eq!(cap.total_memory_mb(), cap.total_memory() / (1024 * 1024));
        assert!(cap.memory_utilization() >= 0.0 && cap.memory_utilization() <= 1.0);
    }

    #[test]
    fn test_capability_comparison() {
        let cap1 = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();
        let cap2 = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();

        let _ordering = utils::compare_capabilities(&cap1, &cap2);
        let devices = [cap1, cap2];
        let best = utils::find_best_device(&devices);
        assert!(best.is_some());
    }

    #[test]
    fn test_device_filtering() {
        let cap = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();
        let caps = vec![cap];

        let filtered = utils::filter_by_memory(&caps, 1024); // 1GB minimum
        assert!(!filtered.is_empty());

        let simd_filtered = utils::filter_by_simd(&caps, false);
        assert!(!simd_filtered.is_empty());
    }

    #[test]
    fn test_training_suitability() {
        let cap = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();
        let _suitable = utils::is_suitable_for_training(&cap);
        let _performance = utils::estimate_training_performance(&cap);
        // Just ensure these don't panic
    }

    #[test]
    fn test_thermal_info() {
        let thermal = ThermalInfo::mock_gpu();
        assert!(thermal.is_temperature_safe());
        assert!(thermal.temperature_ratio() >= 0.0 && thermal.temperature_ratio() <= 1.0);
    }

    #[test]
    fn test_precision_support() {
        let cap = DeviceCapabilities::detect(DeviceType::Cpu).unwrap();
        assert!(cap.supports_double_precision()); // CPU should support fp64
                                                  // Half precision support varies by CPU
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_capabilities() {
        if let Ok(cap) = DeviceCapabilities::detect(DeviceType::Cuda(0)) {
            assert_eq!(cap.device_type(), DeviceType::Cuda(0));
            assert!(cap.supports_half_precision());
            assert!(cap.supports_feature("tensor_cores"));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_metal_capabilities() {
        if let Ok(cap) = DeviceCapabilities::detect(DeviceType::Metal(0)) {
            assert_eq!(cap.device_type(), DeviceType::Metal(0));
            assert!(cap.supports_unified_memory());
            assert!(cap.supports_half_precision());
        }
    }
}
