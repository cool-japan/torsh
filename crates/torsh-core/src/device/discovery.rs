//! Device enumeration and selection algorithms
//!
//! This module provides intelligent device discovery, enumeration, and selection
//! algorithms that can choose optimal devices based on workload characteristics
//! and system constraints.

use crate::device::implementations::DeviceFactory;
use crate::device::{Device, DeviceCapabilities, DeviceType};
use crate::error::Result;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Device discovery engine for intelligent device selection
///
/// Provides advanced device discovery capabilities with smart selection algorithms
/// that consider workload characteristics, device capabilities, and system constraints.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{DeviceDiscovery, WorkloadProfile};
///
/// let discovery = DeviceDiscovery::new();
/// discovery.scan_devices()?;
///
/// // Select device for specific workload
/// let workload = WorkloadProfile::training_large();
/// let device = discovery.select_optimal_device(&workload)?;
/// ```
#[derive(Debug)]
pub struct DeviceDiscovery {
    discovered_devices: RwLock<Vec<DiscoveredDevice>>,
    device_cache: RwLock<HashMap<DeviceType, Arc<dyn Device>>>,
    selection_history: RwLock<Vec<SelectionRecord>>,
    config: DiscoveryConfig,
}

impl DeviceDiscovery {
    /// Create a new device discovery engine
    pub fn new() -> Self {
        Self {
            discovered_devices: RwLock::new(Vec::new()),
            device_cache: RwLock::new(HashMap::new()),
            selection_history: RwLock::new(Vec::new()),
            config: DiscoveryConfig::default(),
        }
    }

    /// Create discovery engine with custom configuration
    pub fn with_config(config: DiscoveryConfig) -> Self {
        Self {
            discovered_devices: RwLock::new(Vec::new()),
            device_cache: RwLock::new(HashMap::new()),
            selection_history: RwLock::new(Vec::new()),
            config,
        }
    }

    /// Scan for available devices
    pub fn scan_devices(&self) -> Result<usize> {
        let mut discovered = Vec::new();

        // Scan each device type
        if self.config.scan_cpu {
            discovered.extend(self.scan_cpu_devices()?);
        }

        if self.config.scan_cuda {
            discovered.extend(self.scan_cuda_devices()?);
        }

        if self.config.scan_metal {
            discovered.extend(self.scan_metal_devices()?);
        }

        if self.config.scan_wgpu {
            discovered.extend(self.scan_wgpu_devices()?);
        }

        let count = discovered.len();

        // Update discovered devices
        {
            let mut devices = self
                .discovered_devices
                .write()
                .expect("lock should not be poisoned");
            *devices = discovered;
        }

        // Populate device cache
        self.populate_device_cache()?;

        Ok(count)
    }

    /// Get all discovered devices
    pub fn get_discovered_devices(&self) -> Vec<DiscoveredDevice> {
        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        devices.clone()
    }

    /// Select optimal device for a specific workload
    pub fn select_optimal_device(
        &self,
        workload: &WorkloadProfile,
    ) -> Result<Option<Arc<dyn Device>>> {
        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        let cache = self
            .device_cache
            .read()
            .expect("lock should not be poisoned");

        if devices.is_empty() {
            return Ok(None);
        }

        let mut best_device = None;
        let mut best_score = 0.0;

        for discovered in devices.iter() {
            if !discovered.is_available {
                continue;
            }

            // Check workload compatibility
            if !self.is_workload_compatible(discovered, workload)? {
                continue;
            }

            // Calculate fitness score
            let score = self.calculate_fitness_score(discovered, workload)?;

            if score > best_score {
                best_score = score;
                best_device = cache.get(&discovered.device_type).cloned();
            }
        }

        // Record selection
        if let Some(ref device) = best_device {
            self.record_selection(device.device_type(), workload.clone(), best_score);
        }

        Ok(best_device)
    }

    /// Select multiple devices for distributed workload
    pub fn select_devices_for_distributed_workload(
        &self,
        workload: &WorkloadProfile,
        target_count: usize,
    ) -> Result<Vec<Arc<dyn Device>>> {
        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        let cache = self
            .device_cache
            .read()
            .expect("lock should not be poisoned");

        let mut candidates: Vec<_> = devices
            .iter()
            .filter(|d| d.is_available && self.is_workload_compatible(d, workload).unwrap_or(false))
            .collect();

        // Sort by fitness score
        candidates.sort_by(|a, b| {
            let score_a = self.calculate_fitness_score(a, workload).unwrap_or(0.0);
            let score_b = self.calculate_fitness_score(b, workload).unwrap_or(0.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected: Vec<_> = candidates
            .into_iter()
            .take(target_count)
            .filter_map(|d| cache.get(&d.device_type).cloned())
            .collect();

        Ok(selected)
    }

    /// Get devices by capability requirements
    pub fn get_devices_by_capabilities(
        &self,
        requirements: &CapabilityRequirements,
    ) -> Result<Vec<Arc<dyn Device>>> {
        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        let cache = self
            .device_cache
            .read()
            .expect("lock should not be poisoned");

        let mut matching_devices = Vec::new();

        for discovered in devices.iter() {
            if !discovered.is_available {
                continue;
            }

            if self.meets_capability_requirements(discovered, requirements)? {
                if let Some(device) = cache.get(&discovered.device_type) {
                    matching_devices.push(device.clone());
                }
            }
        }

        Ok(matching_devices)
    }

    /// Recommend device for specific use case
    pub fn recommend_device(&self, use_case: UseCase) -> Result<DeviceRecommendation> {
        let workload = match use_case {
            UseCase::Training => WorkloadProfile::training_large(),
            UseCase::Inference => WorkloadProfile::inference(),
            UseCase::Development => WorkloadProfile::development(),
            UseCase::Benchmarking => WorkloadProfile::benchmarking(),
            UseCase::Research => WorkloadProfile::research(),
        };

        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        let cache = self
            .device_cache
            .read()
            .expect("lock should not be poisoned");

        let mut recommendations = Vec::new();

        for discovered in devices.iter() {
            if !discovered.is_available {
                continue;
            }

            let score = self.calculate_fitness_score(discovered, &workload)?;
            let reasoning = self.generate_recommendation_reasoning(discovered, &workload, score);

            if let Some(device) = cache.get(&discovered.device_type) {
                recommendations.push(DeviceOption {
                    device: device.clone(),
                    score,
                    reasoning,
                    estimated_performance: self.estimate_performance(discovered, &workload)?,
                });
            }
        }

        // Sort by score
        recommendations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(DeviceRecommendation {
            use_case,
            options: recommendations,
            workload_profile: workload,
        })
    }

    /// Get device discovery statistics
    pub fn get_statistics(&self) -> DiscoveryStatistics {
        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        let history = self
            .selection_history
            .read()
            .expect("lock should not be poisoned");

        let total_devices = devices.len();
        let available_devices = devices.iter().filter(|d| d.is_available).count();

        let device_types: HashSet<_> = devices.iter().map(|d| d.device_type).collect();
        let unique_device_types = device_types.len();

        let total_memory: u64 = devices.iter().map(|d| d.capabilities.total_memory()).sum();

        DiscoveryStatistics {
            total_devices,
            available_devices,
            unique_device_types,
            total_memory_gb: total_memory / (1024 * 1024 * 1024),
            total_selections: history.len(),
        }
    }

    fn scan_cpu_devices(&self) -> Result<Vec<DiscoveredDevice>> {
        let mut devices = Vec::new();

        let device_type = DeviceType::Cpu;
        if DeviceFactory::is_device_type_available(device_type) {
            let capabilities = DeviceCapabilities::detect(device_type)?;
            let platform_info = self.detect_cpu_platform_info();

            devices.push(DiscoveredDevice {
                device_type,
                capabilities,
                is_available: true,
                platform_info,
                discovery_time: std::time::Instant::now(),
            });
        }

        Ok(devices)
    }

    fn scan_cuda_devices(&self) -> Result<Vec<DiscoveredDevice>> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut devices = Vec::new();

        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would query CUDA runtime
            for index in 0..2 {
                let device_type = DeviceType::Cuda(index);
                if DeviceFactory::is_device_type_available(device_type) {
                    if let Ok(capabilities) = DeviceCapabilities::detect(device_type) {
                        let platform_info = self.detect_cuda_platform_info(index);

                        devices.push(DiscoveredDevice {
                            device_type,
                            capabilities,
                            is_available: true,
                            platform_info,
                            discovery_time: std::time::Instant::now(),
                        });
                    }
                }
            }
        }

        Ok(devices)
    }

    fn scan_metal_devices(&self) -> Result<Vec<DiscoveredDevice>> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut devices = Vec::new();

        #[cfg(target_os = "macos")]
        {
            let device_type = DeviceType::Metal(0);
            if DeviceFactory::is_device_type_available(device_type) {
                if let Ok(capabilities) = DeviceCapabilities::detect(device_type) {
                    let platform_info = self.detect_metal_platform_info();

                    devices.push(DiscoveredDevice {
                        device_type,
                        capabilities,
                        is_available: true,
                        platform_info,
                        discovery_time: std::time::Instant::now(),
                    });
                }
            }
        }

        Ok(devices)
    }

    fn scan_wgpu_devices(&self) -> Result<Vec<DiscoveredDevice>> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut devices = Vec::new();

        #[cfg(feature = "wgpu")]
        {
            let device_type = DeviceType::Wgpu(0);
            if DeviceFactory::is_device_type_available(device_type) {
                if let Ok(capabilities) = DeviceCapabilities::detect(device_type) {
                    let platform_info = self.detect_wgpu_platform_info();

                    devices.push(DiscoveredDevice {
                        device_type,
                        capabilities,
                        is_available: true,
                        platform_info,
                        discovery_time: std::time::Instant::now(),
                    });
                }
            }
        }

        Ok(devices)
    }

    fn populate_device_cache(&self) -> Result<()> {
        let devices = self
            .discovered_devices
            .read()
            .expect("lock should not be poisoned");
        let mut cache = self
            .device_cache
            .write()
            .expect("lock should not be poisoned");

        cache.clear();

        for discovered in devices.iter() {
            if discovered.is_available {
                if let Ok(device) = DeviceFactory::create_device(discovered.device_type) {
                    let arc_device: Arc<dyn Device> = device.into();
                    cache.insert(discovered.device_type, arc_device);
                }
            }
        }

        Ok(())
    }

    fn is_workload_compatible(
        &self,
        device: &DiscoveredDevice,
        workload: &WorkloadProfile,
    ) -> Result<bool> {
        // Check memory requirements
        if device.capabilities.available_memory() < workload.min_memory_bytes {
            return Ok(false);
        }

        // Check compute requirements
        if device.capabilities.compute_units() < workload.min_compute_units {
            return Ok(false);
        }

        // Check precision requirements
        if workload.requires_fp64 && !device.capabilities.supports_double_precision() {
            return Ok(false);
        }

        if workload.requires_fp16 && !device.capabilities.supports_half_precision() {
            return Ok(false);
        }

        // Check device type constraints
        match workload.device_preference {
            DevicePreference::GpuOnly => {
                if device.device_type.is_cpu() {
                    return Ok(false);
                }
            }
            DevicePreference::CpuOnly => {
                if !device.device_type.is_cpu() {
                    return Ok(false);
                }
            }
            DevicePreference::CudaOnly => {
                if !device.device_type.is_cuda() {
                    return Ok(false);
                }
            }
            DevicePreference::Any => {}
        }

        Ok(true)
    }

    fn calculate_fitness_score(
        &self,
        device: &DiscoveredDevice,
        workload: &WorkloadProfile,
    ) -> Result<f64> {
        let mut score = 0.0;

        // Base performance score
        let perf_score = device.capabilities.compute_score() as f64 / 1_000_000.0;
        score += perf_score * workload.performance_weight;

        // Memory score
        let memory_ratio =
            device.capabilities.available_memory() as f64 / workload.min_memory_bytes as f64;
        let memory_score = memory_ratio.min(2.0); // Cap at 2x requirement
        score += memory_score * workload.memory_weight;

        // Efficiency score (performance per watt, estimated)
        let efficiency_score = self.estimate_efficiency(device)?;
        score += efficiency_score * workload.efficiency_weight;

        // Compatibility bonus
        if device.device_type.is_gpu() && workload.prefers_gpu {
            score += 0.5;
        }

        if device.capabilities.supports_double_precision() && workload.requires_fp64 {
            score += 0.3;
        }

        if device.capabilities.supports_half_precision() && workload.requires_fp16 {
            score += 0.2;
        }

        // Selection history bonus (prefer previously successful devices)
        let history_bonus = self.get_history_bonus(device.device_type, workload);
        score += history_bonus;

        Ok(score)
    }

    fn meets_capability_requirements(
        &self,
        device: &DiscoveredDevice,
        requirements: &CapabilityRequirements,
    ) -> Result<bool> {
        if let Some(min_memory) = requirements.min_memory_gb {
            if device.capabilities.total_memory_mb() < min_memory * 1024 {
                return Ok(false);
            }
        }

        if let Some(min_cores) = requirements.min_compute_units {
            if device.capabilities.compute_units() < min_cores {
                return Ok(false);
            }
        }

        if requirements.requires_gpu && device.device_type.is_cpu() {
            return Ok(false);
        }

        if requirements.requires_fp64 && !device.capabilities.supports_double_precision() {
            return Ok(false);
        }

        if requirements.requires_fp16 && !device.capabilities.supports_half_precision() {
            return Ok(false);
        }

        for feature in &requirements.required_features {
            if !device.capabilities.supports_feature(feature) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn estimate_efficiency(&self, device: &DiscoveredDevice) -> Result<f64> {
        // Estimate efficiency based on device type and capabilities
        match device.device_type {
            DeviceType::Cpu => Ok(0.5), // CPUs are generally less efficient for ML
            DeviceType::Cuda(_) => Ok(0.9), // CUDA GPUs are highly efficient
            DeviceType::Metal(_) => Ok(0.8), // Metal is efficient but slightly less than CUDA
            DeviceType::Wgpu(_) => Ok(0.6), // WebGPU has overhead
        }
    }

    fn estimate_performance(
        &self,
        device: &DiscoveredDevice,
        workload: &WorkloadProfile,
    ) -> Result<PerformanceEstimate> {
        let base_throughput = device.capabilities.compute_score() as f64;

        // Adjust for workload type
        let workload_multiplier = match workload.workload_type {
            WorkloadType::Training => 1.0,
            WorkloadType::Inference => 1.2, // Inference is typically faster
            WorkloadType::Validation => 1.1,
            WorkloadType::Benchmarking => 0.9, // Benchmarks might be more demanding
        };

        let estimated_throughput = base_throughput * workload_multiplier;

        // Estimate latency based on device type
        let estimated_latency_ms = match device.device_type {
            DeviceType::Cpu => 10.0,
            DeviceType::Cuda(_) => 2.0,
            DeviceType::Metal(_) => 3.0,
            DeviceType::Wgpu(_) => 5.0,
        };

        Ok(PerformanceEstimate {
            throughput: estimated_throughput,
            latency_ms: estimated_latency_ms,
            memory_bandwidth_gbps: device.capabilities.peak_bandwidth_gbps().unwrap_or(1.0),
        })
    }

    fn generate_recommendation_reasoning(
        &self,
        device: &DiscoveredDevice,
        workload: &WorkloadProfile,
        score: f64,
    ) -> String {
        let mut reasons = Vec::new();

        if device.device_type.is_gpu() && workload.prefers_gpu {
            reasons.push("GPU acceleration preferred for this workload".to_string());
        }

        if device.capabilities.total_memory_mb() > workload.min_memory_bytes / (1024 * 1024) {
            reasons.push(format!(
                "Sufficient memory ({:.1}GB available)",
                device.capabilities.total_memory_mb() as f64 / 1024.0
            ));
        }

        if device.capabilities.supports_half_precision() && workload.requires_fp16 {
            reasons.push("Supports required half-precision operations".to_string());
        }

        if score > 2.0 {
            reasons.push("High performance score for workload requirements".to_string());
        } else if score > 1.0 {
            reasons.push("Good performance score for workload requirements".to_string());
        }

        if reasons.is_empty() {
            "Meets basic requirements".to_string()
        } else {
            reasons.join("; ")
        }
    }

    fn get_history_bonus(&self, device_type: DeviceType, workload: &WorkloadProfile) -> f64 {
        let history = self
            .selection_history
            .read()
            .expect("lock should not be poisoned");

        let successful_selections = history
            .iter()
            .filter(|record| {
                record.device_type == device_type
                    && record.workload.workload_type == workload.workload_type
                    && record.success_score > 1.0
            })
            .count();

        // Small bonus for previously successful selections
        (successful_selections as f64) * 0.1
    }

    fn record_selection(&self, device_type: DeviceType, workload: WorkloadProfile, score: f64) {
        let mut history = self
            .selection_history
            .write()
            .expect("lock should not be poisoned");
        history.push(SelectionRecord {
            device_type,
            workload,
            success_score: score,
            timestamp: std::time::Instant::now(),
        });

        // Keep history bounded
        if history.len() > 1000 {
            history.remove(0);
        }
    }

    fn detect_cpu_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            vendor: "Unknown".to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            features: vec!["sse2".to_string(), "avx".to_string()],
            driver_version: None,
        }
    }

    #[allow(dead_code)]
    fn detect_cuda_platform_info(&self, _index: usize) -> PlatformInfo {
        PlatformInfo {
            vendor: "NVIDIA".to_string(),
            architecture: "CUDA".to_string(),
            features: vec!["compute_capability_8_6".to_string()],
            driver_version: Some("12.0".to_string()),
        }
    }

    #[allow(dead_code)] // Metal platform info - only used on macOS
    fn detect_metal_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            vendor: "Apple".to_string(),
            architecture: "Apple Silicon".to_string(),
            features: vec!["unified_memory".to_string(), "tile_shaders".to_string()],
            driver_version: Some("Metal 3.0".to_string()),
        }
    }

    #[allow(dead_code)]
    fn detect_wgpu_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            vendor: "WebGPU".to_string(),
            architecture: "WebGPU".to_string(),
            features: vec!["compute_shaders".to_string()],
            driver_version: Some("1.0".to_string()),
        }
    }
}

/// Discovered device information
#[derive(Debug, Clone)]
pub struct DiscoveredDevice {
    /// Type of the discovered device
    pub device_type: DeviceType,
    /// Capabilities of the device
    pub capabilities: DeviceCapabilities,
    /// Whether the device is currently available
    pub is_available: bool,
    /// Platform-specific information
    pub platform_info: PlatformInfo,
    /// When this device was discovered
    pub discovery_time: std::time::Instant,
}

/// Platform-specific device information
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Device vendor name
    pub vendor: String,
    /// Device architecture description
    pub architecture: String,
    /// List of supported features
    pub features: Vec<String>,
    /// Driver version if available
    pub driver_version: Option<String>,
}

/// Workload profile for device selection
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Type of workload (training, inference, etc.)
    pub workload_type: WorkloadType,
    /// Minimum required memory in bytes
    pub min_memory_bytes: u64,
    /// Minimum required compute units
    pub min_compute_units: u32,
    /// Whether 64-bit floating point is required
    pub requires_fp64: bool,
    /// Whether 16-bit floating point is required
    pub requires_fp16: bool,
    /// Whether GPU is preferred for this workload
    pub prefers_gpu: bool,
    /// Device preference policy
    pub device_preference: DevicePreference,
    /// Weight for performance in selection (0.0-1.0)
    pub performance_weight: f64,
    /// Weight for memory capacity in selection (0.0-1.0)
    pub memory_weight: f64,
    /// Weight for power efficiency in selection (0.0-1.0)
    pub efficiency_weight: f64,
}

impl WorkloadProfile {
    /// Profile for large model training
    pub fn training_large() -> Self {
        Self {
            workload_type: WorkloadType::Training,
            min_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            min_compute_units: 32,
            requires_fp64: false,
            requires_fp16: true,
            prefers_gpu: true,
            device_preference: DevicePreference::GpuOnly,
            performance_weight: 1.0,
            memory_weight: 0.8,
            efficiency_weight: 0.6,
        }
    }

    /// Profile for inference
    pub fn inference() -> Self {
        Self {
            workload_type: WorkloadType::Inference,
            min_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            min_compute_units: 8,
            requires_fp64: false,
            requires_fp16: true,
            prefers_gpu: true,
            device_preference: DevicePreference::Any,
            performance_weight: 0.8,
            memory_weight: 0.6,
            efficiency_weight: 1.0,
        }
    }

    /// Profile for development work
    pub fn development() -> Self {
        Self {
            workload_type: WorkloadType::Training,
            min_memory_bytes: 1024 * 1024 * 1024, // 1GB
            min_compute_units: 4,
            requires_fp64: false,
            requires_fp16: false,
            prefers_gpu: false,
            device_preference: DevicePreference::Any,
            performance_weight: 0.5,
            memory_weight: 0.7,
            efficiency_weight: 0.3,
        }
    }

    /// Profile for benchmarking
    pub fn benchmarking() -> Self {
        Self {
            workload_type: WorkloadType::Benchmarking,
            min_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            min_compute_units: 16,
            requires_fp64: true,
            requires_fp16: true,
            prefers_gpu: true,
            device_preference: DevicePreference::Any,
            performance_weight: 1.0,
            memory_weight: 0.5,
            efficiency_weight: 0.5,
        }
    }

    /// Profile for research work
    pub fn research() -> Self {
        Self {
            workload_type: WorkloadType::Training,
            min_memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            min_compute_units: 64,
            requires_fp64: true,
            requires_fp16: true,
            prefers_gpu: true,
            device_preference: DevicePreference::Any,
            performance_weight: 1.0,
            memory_weight: 1.0,
            efficiency_weight: 0.4,
        }
    }
}

/// Workload type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Model training workload
    Training,
    /// Model inference workload
    Inference,
    /// Validation workload
    Validation,
    /// Benchmarking workload
    Benchmarking,
}

/// Device preference for workload
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Any available device
    Any,
    /// GPU devices only
    GpuOnly,
    /// CPU devices only
    CpuOnly,
    /// CUDA devices only
    CudaOnly,
}

/// Use case for device recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// Model training use case
    Training,
    /// Model inference use case
    Inference,
    /// Development use case
    Development,
    /// Benchmarking use case
    Benchmarking,
    /// Research use case
    Research,
}

/// Capability requirements for device filtering
#[derive(Debug, Clone)]
pub struct CapabilityRequirements {
    /// Minimum memory in GB
    pub min_memory_gb: Option<u64>,
    /// Minimum compute units
    pub min_compute_units: Option<u32>,
    /// Whether GPU is required
    pub requires_gpu: bool,
    /// Whether 64-bit floating point is required
    pub requires_fp64: bool,
    /// Whether 16-bit floating point is required
    pub requires_fp16: bool,
    /// Required device features
    pub required_features: Vec<String>,
}

impl CapabilityRequirements {
    /// Create basic capability requirements with no restrictions
    pub fn basic() -> Self {
        Self {
            min_memory_gb: None,
            min_compute_units: None,
            requires_gpu: false,
            requires_fp64: false,
            requires_fp16: false,
            required_features: Vec::new(),
        }
    }

    /// Create capability requirements for GPU training
    pub fn gpu_training() -> Self {
        Self {
            min_memory_gb: Some(4),
            min_compute_units: Some(16),
            requires_gpu: true,
            requires_fp64: false,
            requires_fp16: true,
            required_features: vec!["tensor_cores".to_string()],
        }
    }
}

/// Device recommendation result
#[derive(Debug)]
pub struct DeviceRecommendation {
    /// Use case for this recommendation
    pub use_case: UseCase,
    /// List of recommended device options
    pub options: Vec<DeviceOption>,
    /// Workload profile used for recommendation
    pub workload_profile: WorkloadProfile,
}

/// Individual device option in recommendation
#[derive(Debug)]
pub struct DeviceOption {
    /// The recommended device
    pub device: Arc<dyn Device>,
    /// Fitness score for this device (0.0-1.0)
    pub score: f64,
    /// Human-readable reasoning for recommendation
    pub reasoning: String,
    /// Estimated performance characteristics
    pub estimated_performance: PerformanceEstimate,
}

/// Performance estimate for a device
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    /// Estimated throughput (operations/second)
    pub throughput: f64,
    /// Estimated latency in milliseconds
    pub latency_ms: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
}

/// Selection history record
#[derive(Debug, Clone)]
struct SelectionRecord {
    device_type: DeviceType,
    workload: WorkloadProfile,
    success_score: f64,
    #[allow(dead_code)] // Part of future device selection history tracking
    timestamp: std::time::Instant,
}

/// Discovery configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Whether to scan for CPU devices
    pub scan_cpu: bool,
    /// Whether to scan for CUDA devices
    pub scan_cuda: bool,
    /// Whether to scan for Metal devices
    pub scan_metal: bool,
    /// Whether to scan for WebGPU devices
    pub scan_wgpu: bool,
    /// Whether to cache discovered devices
    pub cache_discoveries: bool,
    /// Whether to track device selection history
    pub track_selection_history: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            scan_cpu: true,
            scan_cuda: cfg!(feature = "cuda"),
            scan_metal: cfg!(target_os = "macos"),
            scan_wgpu: cfg!(feature = "wgpu"),
            cache_discoveries: true,
            track_selection_history: true,
        }
    }
}

/// Discovery statistics
#[derive(Debug, Clone)]
pub struct DiscoveryStatistics {
    /// Total number of devices discovered
    pub total_devices: usize,
    /// Number of currently available devices
    pub available_devices: usize,
    /// Number of unique device types
    pub unique_device_types: usize,
    /// Total memory across all devices in GB
    pub total_memory_gb: u64,
    /// Total number of device selections made
    pub total_selections: usize,
}

impl Default for DeviceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for device discovery
pub mod utils {
    use super::*;

    /// Create discovery engine and perform initial scan
    pub fn create_and_scan() -> Result<DeviceDiscovery> {
        let discovery = DeviceDiscovery::new();
        discovery.scan_devices()?;
        Ok(discovery)
    }

    /// Quick device selection for common use cases
    pub fn quick_select_for_training() -> Result<Option<Arc<dyn Device>>> {
        let discovery = create_and_scan()?;
        let workload = WorkloadProfile::training_large();
        discovery.select_optimal_device(&workload)
    }

    /// Quick device selection for inference
    pub fn quick_select_for_inference() -> Result<Option<Arc<dyn Device>>> {
        let discovery = create_and_scan()?;
        let workload = WorkloadProfile::inference();
        discovery.select_optimal_device(&workload)
    }

    /// Get best GPU device if available
    pub fn get_best_gpu() -> Result<Option<Arc<dyn Device>>> {
        let discovery = create_and_scan()?;
        let requirements = CapabilityRequirements {
            min_memory_gb: Some(1),
            min_compute_units: Some(8),
            requires_gpu: true,
            requires_fp64: false,
            requires_fp16: false,
            required_features: Vec::new(),
        };

        let devices = discovery.get_devices_by_capabilities(&requirements)?;
        Ok(devices.into_iter().next())
    }

    /// Create summary of all discovered devices
    pub fn create_device_summary() -> Result<Vec<String>> {
        let discovery = create_and_scan()?;
        let devices = discovery.get_discovered_devices();

        let summary = devices
            .iter()
            .map(|device| {
                format!(
                    "{:?} - {:.1}GB, {} cores, {}",
                    device.device_type,
                    device.capabilities.total_memory_mb() as f64 / 1024.0,
                    device.capabilities.compute_units(),
                    if device.is_available {
                        "Available"
                    } else {
                        "Unavailable"
                    }
                )
            })
            .collect();

        Ok(summary)
    }

    /// Check if any high-performance devices are available
    pub fn has_high_performance_devices() -> Result<bool> {
        let discovery = create_and_scan()?;
        let requirements = CapabilityRequirements {
            min_memory_gb: Some(8),
            min_compute_units: Some(32),
            requires_gpu: true,
            requires_fp64: false,
            requires_fp16: true,
            required_features: Vec::new(),
        };

        let devices = discovery.get_devices_by_capabilities(&requirements)?;
        Ok(!devices.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_discovery() {
        let discovery = DeviceDiscovery::new();
        let count = discovery.scan_devices().unwrap();
        assert!(count > 0); // At least CPU should be discovered

        let devices = discovery.get_discovered_devices();
        assert!(!devices.is_empty());

        // CPU device should always be found
        assert!(devices.iter().any(|d| d.device_type == DeviceType::Cpu));
    }

    #[test]
    fn test_workload_profiles() {
        let training = WorkloadProfile::training_large();
        assert_eq!(training.workload_type, WorkloadType::Training);
        assert!(training.prefers_gpu);
        assert!(training.requires_fp16);

        let inference = WorkloadProfile::inference();
        assert_eq!(inference.workload_type, WorkloadType::Inference);
        assert!(inference.min_memory_bytes < training.min_memory_bytes);

        let development = WorkloadProfile::development();
        assert!(!development.prefers_gpu);
    }

    #[test]
    fn test_capability_requirements() {
        let basic = CapabilityRequirements::basic();
        assert!(!basic.requires_gpu);
        assert!(basic.required_features.is_empty());

        let gpu_training = CapabilityRequirements::gpu_training();
        assert!(gpu_training.requires_gpu);
        assert!(gpu_training.min_memory_gb.is_some());
        assert!(!gpu_training.required_features.is_empty());
    }

    #[test]
    fn test_device_selection() {
        let discovery = DeviceDiscovery::new();
        discovery.scan_devices().unwrap();

        let workload = WorkloadProfile::development();
        let device = discovery.select_optimal_device(&workload).unwrap();
        assert!(device.is_some()); // Should find at least CPU

        let distributed = discovery
            .select_devices_for_distributed_workload(&workload, 2)
            .unwrap();
        assert!(!distributed.is_empty());
    }

    #[test]
    fn test_device_recommendation() {
        let discovery = DeviceDiscovery::new();
        discovery.scan_devices().unwrap();

        let recommendation = discovery.recommend_device(UseCase::Development).unwrap();
        assert_eq!(recommendation.use_case, UseCase::Development);
        assert!(!recommendation.options.is_empty());

        for option in &recommendation.options {
            assert!(option.score >= 0.0);
            assert!(!option.reasoning.is_empty());
        }
    }

    #[test]
    fn test_discovery_statistics() {
        let discovery = DeviceDiscovery::new();
        discovery.scan_devices().unwrap();

        let stats = discovery.get_statistics();
        assert!(stats.total_devices > 0);
        assert!(stats.available_devices > 0);
        assert!(stats.unique_device_types > 0);
    }

    #[test]
    fn test_utils_functions() {
        let discovery = utils::create_and_scan().unwrap();
        assert!(!discovery.get_discovered_devices().is_empty());

        let _training_device = utils::quick_select_for_training().unwrap();
        // May or may not find a suitable device depending on system

        let _inference_device = utils::quick_select_for_inference().unwrap();
        // Should at least find CPU for inference

        let summary = utils::create_device_summary().unwrap();
        assert!(!summary.is_empty());

        let _has_hp = utils::has_high_performance_devices().unwrap();
        // Result depends on available hardware
    }

    #[test]
    fn test_platform_info() {
        let discovery = DeviceDiscovery::new();
        let cpu_info = discovery.detect_cpu_platform_info();
        assert!(!cpu_info.vendor.is_empty());
        assert!(!cpu_info.architecture.is_empty());
    }

    #[test]
    fn test_performance_estimate() {
        let discovery = DeviceDiscovery::new();
        discovery.scan_devices().unwrap();

        let devices = discovery.get_discovered_devices();
        if let Some(device) = devices.first() {
            let workload = WorkloadProfile::development();
            let estimate = discovery.estimate_performance(device, &workload).unwrap();
            assert!(estimate.throughput > 0.0);
            assert!(estimate.latency_ms > 0.0);
        }
    }
}
