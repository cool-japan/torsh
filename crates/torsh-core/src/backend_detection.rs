//! Backend feature detection system for runtime capability discovery

use crate::device::{CpuDevice, Device, DeviceCapabilities, DeviceType, SimdFeatures};
use crate::error::Result;
use std::collections::HashMap;

/// Backend feature detection system for runtime capability discovery
#[derive(Debug, Clone)]
pub struct BackendFeatureDetector {
    /// Available devices discovered at runtime
    pub available_devices: Vec<DeviceInfo>,
    /// Runtime feature flags  
    pub runtime_features: RuntimeFeatures,
    /// Backend capabilities summary
    pub backend_summary: BackendSummary,
}

/// Information about a discovered device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type and ID
    pub device_type: DeviceType,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
    /// Whether the device is currently available for use
    pub is_available: bool,
    /// Device priority for automatic selection (higher is better)
    pub priority: u32,
    /// Additional device-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Runtime feature detection results
#[derive(Debug, Clone, Default)]
pub struct RuntimeFeatures {
    /// CPU features
    pub cpu_features: CpuFeatures,
    /// GPU features (if available)
    pub gpu_features: GpuFeatures,
    /// System features
    pub system_features: SystemFeatures,
    /// Compiler and build features
    pub build_features: BuildFeatures,
}

/// CPU-specific runtime features
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    /// Detected SIMD capabilities
    pub simd: SimdFeatures,
    /// Number of physical CPU cores
    pub physical_cores: usize,
    /// Number of logical CPU cores (including hyperthreading)
    pub logical_cores: usize,
    /// CPU architecture string
    pub architecture: String,
    /// CPU vendor (Intel, AMD, ARM, etc.)
    pub vendor: Option<String>,
    /// CPU model name
    pub model_name: Option<String>,
    /// CPU base frequency in Hz
    pub base_frequency: Option<u64>,
    /// Cache sizes (L1, L2, L3)
    pub cache_sizes: CacheSizes,
}

/// Cache size information
#[derive(Debug, Clone, Default)]
pub struct CacheSizes {
    /// L1 data cache size in bytes
    pub l1_data: Option<usize>,
    /// L1 instruction cache size in bytes
    pub l1_instruction: Option<usize>,
    /// L2 cache size in bytes
    pub l2: Option<usize>,
    /// L3 cache size in bytes
    pub l3: Option<usize>,
}

/// GPU-specific runtime features
#[derive(Debug, Clone, Default)]
pub struct GpuFeatures {
    /// CUDA support and version
    pub cuda_version: Option<String>,
    /// CUDA compute capability
    pub cuda_compute_capability: Option<(u32, u32)>,
    /// OpenCL support
    pub opencl_version: Option<String>,
    /// Vulkan support
    pub vulkan_version: Option<String>,
    /// Metal support (Apple)
    pub metal_version: Option<String>,
    /// WebGPU support
    pub webgpu_available: bool,
    /// Number of GPU devices detected
    pub gpu_count: usize,
}

/// System-level features
#[derive(Debug, Clone, Default)]
pub struct SystemFeatures {
    /// Operating system information
    pub os_info: OsInfo,
    /// Total system memory in bytes
    pub total_memory: usize,
    /// Page size in bytes
    pub page_size: usize,
    /// NUMA topology available
    pub numa_available: bool,
    /// Number of NUMA nodes
    pub numa_nodes: usize,
    /// Memory bandwidth estimate in bytes/sec
    pub memory_bandwidth: Option<u64>,
}

/// Operating system information
#[derive(Debug, Clone, Default)]
pub struct OsInfo {
    /// OS name (Linux, Windows, macOS, etc.)
    pub name: String,
    /// OS version
    pub version: Option<String>,
    /// OS architecture
    pub arch: String,
    /// Kernel version (for Linux/Unix systems)
    pub kernel_version: Option<String>,
}

/// Build and compiler features
#[derive(Debug, Clone, Default)]
pub struct BuildFeatures {
    /// Target triple for this build
    pub target_triple: String,
    /// Optimization level
    pub opt_level: Option<String>,
    /// Debug info available
    pub debug_info: bool,
    /// Feature flags enabled at compile time
    pub compile_features: Vec<String>,
    /// Cargo features enabled
    pub cargo_features: Vec<String>,
}

/// Summary of available backend capabilities
#[derive(Debug, Clone, Default)]
pub struct BackendSummary {
    /// Best available device for each type
    pub best_devices: HashMap<DeviceType, DeviceInfo>,
    /// Recommended device for general use
    pub recommended_device: Option<DeviceInfo>,
    /// Overall system performance tier (Low, Medium, High, Extreme)
    pub performance_tier: PerformanceTier,
    /// Features that may impact performance
    pub performance_notes: Vec<String>,
    /// Missing features or recommendations
    pub recommendations: Vec<String>,
}

/// System performance classification
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PerformanceTier {
    /// Basic performance for simple tasks
    Low,
    /// Good performance for most workloads
    #[default]
    Medium,
    /// High performance for demanding tasks
    High,
    /// Extreme performance for HPC workloads
    Extreme,
}

/// Workload type for device selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// General purpose computing
    GeneralCompute,
    /// High precision mathematical operations
    HighPrecisionMath,
    /// Large matrix operations
    LargeMatrices,
    /// Highly parallel workloads
    ParallelWorkloads,
}

impl BackendFeatureDetector {
    /// Create a new feature detector and perform comprehensive detection
    pub fn new() -> Result<Self> {
        let mut detector = Self {
            available_devices: Vec::new(),
            runtime_features: RuntimeFeatures::default(),
            backend_summary: BackendSummary::default(),
        };

        detector.detect_all_features()?;
        detector.discover_devices()?;
        detector.analyze_capabilities()?;

        Ok(detector)
    }

    /// Detect all runtime features
    fn detect_all_features(&mut self) -> Result<()> {
        self.runtime_features.cpu_features = self.detect_cpu_features()?;
        self.runtime_features.gpu_features = self.detect_gpu_features()?;
        self.runtime_features.system_features = self.detect_system_features()?;
        self.runtime_features.build_features = self.detect_build_features()?;
        Ok(())
    }

    /// Detect CPU features comprehensively
    fn detect_cpu_features(&self) -> Result<CpuFeatures> {
        let cpu_device = CpuDevice::new();
        let cpu_capabilities = cpu_device.capabilities()?;

        let features = CpuFeatures {
            simd: cpu_capabilities.simd_features().clone(),
            physical_cores: num_cpus::get_physical(),
            logical_cores: num_cpus::get(),
            architecture: std::env::consts::ARCH.to_string(),
            vendor: self.detect_cpu_vendor(),
            model_name: None,     // Custom properties access not available
            base_frequency: None, // Clock frequency access not available
            cache_sizes: self.detect_cache_sizes()?,
        };

        Ok(features)
    }

    /// Detect CPU vendor with detailed identification
    fn detect_cpu_vendor(&self) -> Option<String> {
        #[cfg(target_arch = "x86_64")]
        {
            self.detect_x86_cpu_vendor()
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.detect_arm_cpu_vendor()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            None
        }
    }

    /// Detect x86/x86_64 CPU vendor using CPUID
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_cpu_vendor(&self) -> Option<String> {
        use std::arch::x86_64::__cpuid;

        unsafe {
            let cpuid_result = __cpuid(0);
            let vendor_string = format!(
                "{}{}{}",
                std::str::from_utf8(&cpuid_result.ebx.to_le_bytes()).unwrap_or(""),
                std::str::from_utf8(&cpuid_result.edx.to_le_bytes()).unwrap_or(""),
                std::str::from_utf8(&cpuid_result.ecx.to_le_bytes()).unwrap_or("")
            );

            match vendor_string.as_str() {
                "GenuineIntel" => Some("Intel".to_string()),
                "AuthenticAMD" => Some("AMD".to_string()),
                "VIA VIA VIA " => Some("VIA".to_string()),
                "CyrixInstead" => Some("Cyrix".to_string()),
                "CentaurHauls" => Some("Centaur".to_string()),
                "NexGenDriven" => Some("NexGen".to_string()),
                "HygonGenuine" => Some("Hygon".to_string()),
                _ => Some(format!(
                    "Unknown ({})",
                    vendor_string.trim_end_matches('\0')
                )),
            }
        }
    }

    /// Detect ARM CPU vendor from /proc/cpuinfo
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_cpu_vendor(&self) -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if line.starts_with("CPU implementer") {
                        if let Some(implementer) = line.split(':').nth(1) {
                            let implementer = implementer.trim();
                            return match implementer {
                                "0x41" => Some("ARM".to_string()),
                                "0x42" => Some("Broadcom".to_string()),
                                "0x43" => Some("Cavium".to_string()),
                                "0x44" => Some("DEC".to_string()),
                                "0x46" => Some("Fujitsu".to_string()),
                                "0x48" => Some("HiSilicon".to_string()),
                                "0x49" => Some("Infineon".to_string()),
                                "0x4d" => Some("Motorola".to_string()),
                                "0x4e" => Some("NVIDIA".to_string()),
                                "0x50" => Some("Applied Micro".to_string()),
                                "0x51" => Some("Qualcomm".to_string()),
                                "0x56" => Some("Marvell".to_string()),
                                "0x61" => Some("Apple".to_string()),
                                _ => Some(format!("Unknown ARM implementer ({})", implementer)),
                            };
                        }
                    }
                    if line.starts_with("Hardware") && line.contains("BCM") {
                        return Some("Broadcom".to_string());
                    }
                    if line.starts_with("Hardware") && line.contains("Apple") {
                        return Some("Apple".to_string());
                    }
                }
            }
        }
        Some("ARM".to_string())
    }

    /// Detect cache sizes
    fn detect_cache_sizes(&self) -> Result<CacheSizes> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut cache_sizes = CacheSizes::default();

        #[cfg(target_os = "linux")]
        {
            // Try to read cache info from sysfs
            if let Ok(l1d) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size")
            {
                cache_sizes.l1_data = self.parse_cache_size(&l1d);
            }
            if let Ok(l1i) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index1/size")
            {
                cache_sizes.l1_instruction = self.parse_cache_size(&l1i);
            }
            if let Ok(l2) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
            {
                cache_sizes.l2 = self.parse_cache_size(&l2);
            }
            if let Ok(l3) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size")
            {
                cache_sizes.l3 = self.parse_cache_size(&l3);
            }
        }

        Ok(cache_sizes)
    }

    /// Parse cache size string (e.g., "32K" -> 32768)
    #[allow(dead_code)] // Used in platform-specific cache detection
    fn parse_cache_size(&self, size_str: &str) -> Option<usize> {
        let size_str = size_str.trim();
        if size_str.is_empty() {
            return None;
        }

        let (number_part, suffix) = if let Some(stripped) = size_str.strip_suffix('K') {
            (stripped, 1024)
        } else if let Some(stripped) = size_str.strip_suffix('M') {
            (stripped, 1024 * 1024)
        } else if let Some(stripped) = size_str.strip_suffix('G') {
            (stripped, 1024 * 1024 * 1024)
        } else {
            (size_str, 1)
        };

        number_part.parse::<usize>().ok().map(|n| n * suffix)
    }

    /// Detect GPU features
    fn detect_gpu_features(&self) -> Result<GpuFeatures> {
        let features = GpuFeatures::default();
        // GPU detection would be implemented here
        Ok(features)
    }

    /// Detect system features
    fn detect_system_features(&self) -> Result<SystemFeatures> {
        let cpu_device = CpuDevice::new();
        let memory_info = cpu_device.memory_info()?;
        let _cpu_capabilities = cpu_device.capabilities()?;

        let features = SystemFeatures {
            os_info: self.detect_os_info(),
            total_memory: memory_info.total as usize,
            page_size: self.detect_page_size(),
            numa_available: self.detect_numa_support(),
            numa_nodes: self.detect_numa_nodes(),
            memory_bandwidth: None, // Memory bandwidth not accessible
        };

        Ok(features)
    }

    /// Detect operating system information
    fn detect_os_info(&self) -> OsInfo {
        OsInfo {
            name: std::env::consts::OS.to_string(),
            version: self.get_os_version(),
            arch: std::env::consts::ARCH.to_string(),
            kernel_version: self.get_kernel_version(),
        }
    }

    /// Get OS version
    fn get_os_version(&self) -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/version")
                .ok()
                .and_then(|v| v.split_whitespace().nth(2).map(|s| s.to_string()))
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Get kernel version
    fn get_kernel_version(&self) -> Option<String> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            use std::process::Command;
            Command::new("uname")
                .arg("-r")
                .output()
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|v| v.trim().to_string())
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            None
        }
    }

    /// Detect system page size
    fn detect_page_size(&self) -> usize {
        4096 // Default page size
    }

    /// Detect NUMA support
    fn detect_numa_support(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/sys/devices/system/node").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Detect number of NUMA nodes
    fn detect_numa_nodes(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
                entries
                    .filter_map(|entry| {
                        entry.ok().and_then(|e| {
                            let name = e.file_name();
                            let name_str = name.to_string_lossy();
                            if name_str.starts_with("node")
                                && name_str[4..].chars().all(|c| c.is_ascii_digit())
                            {
                                Some(())
                            } else {
                                None
                            }
                        })
                    })
                    .count()
            } else {
                1
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            1
        }
    }

    /// Detect build features
    fn detect_build_features(&self) -> Result<BuildFeatures> {
        let features = BuildFeatures {
            target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            opt_level: std::env::var("OPT_LEVEL").ok(),
            debug_info: cfg!(debug_assertions),
            compile_features: self.get_compile_features(),
            cargo_features: self.get_cargo_features(),
        };

        Ok(features)
    }

    /// Get compile-time features
    fn get_compile_features(&self) -> Vec<String> {
        let mut features = Vec::new();

        if cfg!(feature = "std") {
            features.push("std".to_string());
        }
        if cfg!(feature = "no_std") {
            features.push("no_std".to_string());
        }
        if cfg!(feature = "serialize") {
            features.push("serialize".to_string());
        }
        if cfg!(feature = "half") {
            features.push("half".to_string());
        }
        if cfg!(feature = "avx512") {
            features.push("avx512".to_string());
        }
        if cfg!(feature = "simd") {
            features.push("simd".to_string());
        }
        if cfg!(feature = "parallel") {
            features.push("parallel".to_string());
        }
        if cfg!(feature = "fast-math") {
            features.push("fast-math".to_string());
        }

        features
    }

    /// Get cargo features
    fn get_cargo_features(&self) -> Vec<String> {
        vec!["default".to_string()]
    }

    /// Discover available devices
    fn discover_devices(&mut self) -> Result<()> {
        // Always add CPU device
        let cpu_device = CpuDevice::new();
        let cpu_capabilities = cpu_device.capabilities()?;

        self.available_devices.push(DeviceInfo {
            device_type: DeviceType::Cpu,
            capabilities: cpu_capabilities,
            is_available: cpu_device.is_available().unwrap_or(false),
            priority: 10, // Base priority for CPU
            metadata: HashMap::new(),
        });

        Ok(())
    }

    /// Analyze capabilities and generate summary
    fn analyze_capabilities(&mut self) -> Result<()> {
        // Find best device for each type
        for device in &self.available_devices {
            self.backend_summary
                .best_devices
                .insert(device.device_type, device.clone());
        }

        // Find recommended device (highest priority available device)
        self.backend_summary.recommended_device = self
            .available_devices
            .iter()
            .filter(|d| d.is_available)
            .max_by_key(|d| d.priority)
            .cloned();

        // Determine performance tier
        self.backend_summary.performance_tier = self.classify_performance_tier();

        // Generate performance notes and recommendations
        self.generate_performance_analysis();

        Ok(())
    }

    /// Classify overall system performance tier
    fn classify_performance_tier(&self) -> PerformanceTier {
        let cpu_features = &self.runtime_features.cpu_features;
        let system_features = &self.runtime_features.system_features;

        let memory_gb = system_features.total_memory / (1024 * 1024 * 1024);
        let core_count = cpu_features.logical_cores;
        let has_advanced_simd =
            cpu_features.simd.avx2 || cpu_features.simd.avx512f || cpu_features.simd.neon;

        if memory_gb >= 32 && core_count >= 16 && cpu_features.simd.avx512f {
            PerformanceTier::Extreme
        } else if memory_gb >= 16 && core_count >= 8 && has_advanced_simd {
            PerformanceTier::High
        } else if memory_gb >= 8 && core_count >= 4 {
            PerformanceTier::Medium
        } else {
            PerformanceTier::Low
        }
    }

    /// Generate performance analysis and recommendations
    fn generate_performance_analysis(&mut self) {
        let cpu_features = &self.runtime_features.cpu_features;
        let system_features = &self.runtime_features.system_features;

        // Performance notes
        if cpu_features.simd.avx512f {
            self.backend_summary
                .performance_notes
                .push("AVX-512 support detected - excellent SIMD performance".to_string());
        } else if cpu_features.simd.avx2 {
            self.backend_summary
                .performance_notes
                .push("AVX2 support detected - good SIMD performance".to_string());
        } else if cpu_features.simd.neon {
            self.backend_summary
                .performance_notes
                .push("NEON support detected - good ARM SIMD performance".to_string());
        }

        if system_features.numa_available {
            self.backend_summary.performance_notes.push(format!(
                "NUMA topology available with {} nodes",
                system_features.numa_nodes
            ));
        }

        // Recommendations
        if cpu_features.logical_cores < 4 {
            self.backend_summary.recommendations.push(
                "Consider upgrading to a CPU with more cores for better parallel performance"
                    .to_string(),
            );
        }

        if system_features.total_memory < 8 * 1024 * 1024 * 1024 {
            self.backend_summary
                .recommendations
                .push("Consider adding more RAM (minimum 8GB recommended)".to_string());
        }
    }

    /// Get the best available device for a specific workload type
    pub fn best_device_for_workload(&self, workload: WorkloadType) -> Option<&DeviceInfo> {
        match workload {
            WorkloadType::GeneralCompute => self.backend_summary.recommended_device.as_ref(),
            WorkloadType::HighPrecisionMath => {
                // Prefer devices with double precision support
                self.available_devices
                    .iter()
                    .filter(|d| d.capabilities.supports_double_precision())
                    .max_by_key(|d| d.priority)
            }
            WorkloadType::LargeMatrices => {
                // Prefer devices with lots of memory and good SIMD
                self.available_devices.iter().max_by_key(|d| {
                    (
                        d.capabilities.total_memory(),
                        if d.capabilities.simd_features().avx512f {
                            8
                        } else if d.capabilities.simd_features().avx2 {
                            4
                        } else {
                            1
                        },
                    )
                })
            }
            WorkloadType::ParallelWorkloads => {
                // Prefer devices with many cores
                self.available_devices
                    .iter()
                    .max_by_key(|d| d.capabilities.compute_units())
            }
        }
    }

    /// Check if a specific feature is available
    pub fn has_feature(&self, feature: &str) -> bool {
        match feature {
            "simd" => {
                let simd = &self.runtime_features.cpu_features.simd;
                simd.sse || simd.avx || simd.avx2 || simd.avx512f || simd.neon
            }
            "avx2" => self.runtime_features.cpu_features.simd.avx2,
            "avx512" => self.runtime_features.cpu_features.simd.avx512f,
            "neon" => self.runtime_features.cpu_features.simd.neon,
            "numa" => self.runtime_features.system_features.numa_available,
            "double_precision" => self
                .available_devices
                .iter()
                .any(|d| d.capabilities.supports_double_precision()),
            "half_precision" => self
                .available_devices
                .iter()
                .any(|d| d.capabilities.supports_half_precision()),
            _ => false,
        }
    }
}

impl Default for BackendFeatureDetector {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            available_devices: vec![],
            runtime_features: RuntimeFeatures::default(),
            backend_summary: BackendSummary::default(),
        })
    }
}
