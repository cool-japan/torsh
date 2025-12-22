//! Backend Capability Introspection System
//!
//! This module provides comprehensive runtime introspection of backend capabilities,
//! allowing users to query what operations are supported, performance characteristics,
//! hardware features, and optimization opportunities.
//!
//! ## Features
//!
//! - **Capability Discovery**: Query supported operations and data types
//! - **Performance Characteristics**: Get expected performance profiles
//! - **Hardware Features**: Inspect available hardware acceleration
//! - **Optimization Recommendations**: Get backend-specific optimization suggestions
//! - **Compatibility Checking**: Verify operation compatibility before execution
//! - **Resource Limits**: Query memory, compute, and bandwidth limits

use crate::backend::Backend;
use crate::error::BackendResult;
use crate::{BackendType, Device};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::String, vec::Vec};

/// Comprehensive capability information for a backend
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BackendCapabilities {
    /// Backend type
    pub backend_type: BackendType,
    /// Device information
    pub device_info: DeviceInfo,
    /// Supported operations
    pub operations: OperationSupport,
    /// Data type support
    pub data_types: DataTypeSupport,
    /// Memory capabilities
    pub memory: MemoryCapabilities,
    /// Compute capabilities
    pub compute: ComputeCapabilities,
    /// Optimization features
    pub optimizations: OptimizationFeatures,
    /// Performance characteristics
    pub performance: PerformanceProfile,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Extension support
    pub extensions: Vec<ExtensionInfo>,
}

impl BackendCapabilities {
    /// Create a detailed capability report
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "=== {} Backend Capabilities ===\n\n",
            self.backend_type
        ));

        // Device Information
        report.push_str(&format!("Device: {}\n", self.device_info.name));
        report.push_str(&format!("Vendor: {}\n", self.device_info.vendor));
        report.push_str(&format!(
            "Driver Version: {}\n\n",
            self.device_info.driver_version
        ));

        // Memory
        report.push_str(&format!("Memory:\n"));
        report.push_str(&format!(
            "  Total: {} GB\n",
            self.memory.total_memory / (1024 * 1024 * 1024)
        ));
        report.push_str(&format!(
            "  Available: {} GB\n",
            self.memory.available_memory / (1024 * 1024 * 1024)
        ));
        report.push_str(&format!("  Unified: {}\n", self.memory.unified_memory));
        report.push_str(&format!(
            "  Max Allocation: {} MB\n\n",
            self.limits.max_allocation_size / (1024 * 1024)
        ));

        // Compute
        report.push_str(&format!("Compute:\n"));
        report.push_str(&format!("  Cores: {}\n", self.compute.compute_units));
        report.push_str(&format!(
            "  Peak FLOPS: {:.2} TFLOPS\n",
            self.performance.peak_flops / 1e12
        ));
        report.push_str(&format!(
            "  Memory Bandwidth: {:.2} GB/s\n\n",
            self.performance.memory_bandwidth / 1e9
        ));

        // Optimizations
        report.push_str(&format!("Optimizations:\n"));
        report.push_str(&format!("  SIMD: {}\n", self.optimizations.simd_support));
        report.push_str(&format!(
            "  Tensor Cores: {}\n",
            self.optimizations.tensor_core_support
        ));
        report.push_str(&format!(
            "  Mixed Precision: {}\n",
            self.optimizations.mixed_precision_support
        ));
        report.push_str(&format!(
            "  Kernel Fusion: {}\n\n",
            self.optimizations.kernel_fusion_support
        ));

        // Extensions
        if !self.extensions.is_empty() {
            report.push_str("Extensions:\n");
            for ext in &self.extensions {
                report.push_str(&format!("  - {} (v{})\n", ext.name, ext.version));
            }
        }

        report
    }

    /// Check if an operation is supported
    pub fn supports_operation(&self, op: &str) -> bool {
        self.operations.supported_operations.contains(op)
    }

    /// Get optimization recommendations for this backend
    pub fn optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if self.optimizations.tensor_core_support {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Hardware,
                priority: RecommendationPriority::High,
                title: "Use Tensor Cores".to_string(),
                description: "This device supports tensor cores for accelerated matrix operations. Enable tensor core usage for matmul, conv2d, and attention operations.".to_string(),
                estimated_speedup: 5.0,
                implementation_difficulty: ImplementationDifficulty::Low,
            });
        }

        if self.optimizations.mixed_precision_support {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Precision,
                priority: RecommendationPriority::High,
                title: "Enable Mixed Precision Training".to_string(),
                description: "Use FP16 for forward/backward passes and FP32 for weight updates to improve performance while maintaining accuracy.".to_string(),
                estimated_speedup: 2.0,
                implementation_difficulty: ImplementationDifficulty::Medium,
            });
        }

        if self.memory.total_memory < 8 * 1024 * 1024 * 1024 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Memory,
                priority: RecommendationPriority::High,
                title: "Enable Gradient Checkpointing".to_string(),
                description: "Limited memory available. Use gradient checkpointing to trade compute for memory.".to_string(),
                estimated_speedup: 1.0,
                implementation_difficulty: ImplementationDifficulty::Medium,
            });
        }

        if self.optimizations.kernel_fusion_support {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Compute,
                priority: RecommendationPriority::Medium,
                title: "Enable Kernel Fusion".to_string(),
                description: "Fuse consecutive operations to reduce memory bandwidth and kernel launch overhead.".to_string(),
                estimated_speedup: 1.5,
                implementation_difficulty: ImplementationDifficulty::Low,
            });
        }

        if self.compute.compute_units > 64 {
            recommendations.push(OptimizationRecommendation {
                category: OptimizationCategory::Parallelism,
                priority: RecommendationPriority::Medium,
                title: "Increase Parallelism".to_string(),
                description: format!("Device has {} compute units. Increase batch size and use data parallelism for better utilization.", self.compute.compute_units),
                estimated_speedup: 1.3,
                implementation_difficulty: ImplementationDifficulty::Low,
            });
        }

        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        recommendations
    }

    /// Compare capabilities with another backend
    pub fn compare_with(&self, other: &BackendCapabilities) -> CapabilityComparison {
        CapabilityComparison {
            backend_a: self.backend_type.clone(),
            backend_b: other.backend_type.clone(),
            memory_ratio: self.memory.total_memory as f64 / other.memory.total_memory as f64,
            compute_ratio: self.performance.peak_flops / other.performance.peak_flops,
            bandwidth_ratio: self.performance.memory_bandwidth / other.performance.memory_bandwidth,
            unique_to_a: self.get_unique_features(other),
            unique_to_b: other.get_unique_features(self),
            recommendation: self.get_comparison_recommendation(other),
        }
    }

    fn get_unique_features(&self, other: &BackendCapabilities) -> Vec<String> {
        let mut unique = Vec::new();

        if self.optimizations.tensor_core_support && !other.optimizations.tensor_core_support {
            unique.push("Tensor Core Support".to_string());
        }
        if self.optimizations.simd_support && !other.optimizations.simd_support {
            unique.push("SIMD Support".to_string());
        }
        if self.memory.unified_memory && !other.memory.unified_memory {
            unique.push("Unified Memory".to_string());
        }

        unique
    }

    fn get_comparison_recommendation(&self, other: &BackendCapabilities) -> String {
        let compute_ratio = self.performance.peak_flops / other.performance.peak_flops;

        if compute_ratio > 2.0 {
            format!(
                "{} is significantly faster for compute-intensive workloads",
                self.backend_type
            )
        } else if compute_ratio < 0.5 {
            format!(
                "{} is significantly faster for compute-intensive workloads",
                other.backend_type
            )
        } else {
            "Backends have similar compute performance".to_string()
        }
    }
}

/// Device information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub driver_version: String,
    pub device_type: DeviceType,
    pub pci_bus_id: Option<String>,
    pub compute_capability: Option<(u32, u32)>,
}

/// Device type classification
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum DeviceType {
    CPU,
    DiscreteGPU,
    IntegratedGPU,
    VirtualGPU,
    Accelerator,
    Unknown,
}

/// Operation support matrix
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct OperationSupport {
    pub supported_operations: HashSet<String>,
    pub native_operations: HashSet<String>,
    pub emulated_operations: HashSet<String>,
    pub performance_tiers: HashMap<String, PerformanceTier>,
}

impl OperationSupport {
    /// Check if operation is natively supported (not emulated)
    pub fn is_native(&self, op: &str) -> bool {
        self.native_operations.contains(op)
    }

    /// Get performance tier for an operation
    pub fn performance_tier(&self, op: &str) -> Option<PerformanceTier> {
        self.performance_tiers.get(op).cloned()
    }
}

/// Performance tier for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum PerformanceTier {
    Poor,    // Emulated/slow path
    Fair,    // Standard implementation
    Good,    // Optimized implementation
    Optimal, // Hardware-accelerated
}

impl PartialOrd for PerformanceTier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PerformanceTier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use PerformanceTier::*;
        let self_value = match self {
            Poor => 0,
            Fair => 1,
            Good => 2,
            Optimal => 3,
        };
        let other_value = match other {
            Poor => 0,
            Fair => 1,
            Good => 2,
            Optimal => 3,
        };
        self_value.cmp(&other_value)
    }
}

impl fmt::Display for PerformanceTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerformanceTier::Optimal => write!(f, "Optimal"),
            PerformanceTier::Good => write!(f, "Good"),
            PerformanceTier::Fair => write!(f, "Fair"),
            PerformanceTier::Poor => write!(f, "Poor"),
        }
    }
}

/// Data type support information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct DataTypeSupport {
    pub float_types: Vec<FloatType>,
    pub integer_types: Vec<IntegerType>,
    pub complex_support: bool,
    pub quantized_support: bool,
    pub custom_types: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum FloatType {
    F16,
    BF16,
    TF32,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum IntegerType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// Memory capabilities
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MemoryCapabilities {
    pub total_memory: usize,
    pub available_memory: usize,
    pub unified_memory: bool,
    pub pinned_memory_support: bool,
    pub zero_copy_support: bool,
    pub peer_to_peer_support: bool,
    pub memory_pools: Vec<MemoryPoolInfo>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MemoryPoolInfo {
    pub name: String,
    pub size: usize,
    pub location: MemoryLocation,
    pub access_speed: MemoryAccessSpeed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum MemoryLocation {
    Device,
    Host,
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum MemoryAccessSpeed {
    VeryFast,
    Fast,
    Medium,
    Slow,
}

/// Compute capabilities
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ComputeCapabilities {
    pub compute_units: u32,
    pub max_threads_per_block: Option<u32>,
    pub max_workgroup_size: Option<(u32, u32, u32)>,
    pub warp_size: Option<u32>,
    pub shared_memory_per_block: Option<usize>,
    pub registers_per_block: Option<u32>,
    pub max_concurrent_kernels: Option<u32>,
}

/// Optimization features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct OptimizationFeatures {
    pub simd_support: bool,
    pub simd_width: Option<u32>,
    pub tensor_core_support: bool,
    pub mixed_precision_support: bool,
    pub kernel_fusion_support: bool,
    pub auto_tuning_support: bool,
    pub jit_compilation_support: bool,
    pub async_execution_support: bool,
}

/// Performance profile
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PerformanceProfile {
    pub peak_flops: f64,
    pub peak_flops_fp16: Option<f64>,
    pub peak_flops_int8: Option<f64>,
    pub memory_bandwidth: f64,
    pub cache_hierarchy: Vec<CacheLevel>,
    pub typical_latency_us: f64,
    pub throughput_estimate: f64,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CacheLevel {
    pub level: u32,
    pub size: usize,
    pub line_size: usize,
    pub associativity: u32,
}

/// Resource limits
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ResourceLimits {
    pub max_allocation_size: usize,
    pub max_buffer_size: usize,
    pub max_texture_size: Option<(u32, u32, u32)>,
    pub max_threads: u32,
    pub max_dispatch_size: Option<(u32, u32, u32)>,
    pub max_constant_buffer_size: Option<usize>,
}

/// Extension information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ExtensionInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub required: bool,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub estimated_speedup: f64,
    pub implementation_difficulty: ImplementationDifficulty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum OptimizationCategory {
    Hardware,
    Memory,
    Compute,
    Precision,
    Parallelism,
    IO,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl PartialOrd for RecommendationPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecommendationPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use RecommendationPriority::*;
        let self_value = match self {
            Low => 0,
            Medium => 1,
            High => 2,
            Critical => 3,
        };
        let other_value = match other {
            Low => 0,
            Medium => 1,
            High => 2,
            Critical => 3,
        };
        self_value.cmp(&other_value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum ImplementationDifficulty {
    Low,
    Medium,
    High,
}

/// Capability comparison result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CapabilityComparison {
    pub backend_a: BackendType,
    pub backend_b: BackendType,
    pub memory_ratio: f64,
    pub compute_ratio: f64,
    pub bandwidth_ratio: f64,
    pub unique_to_a: Vec<String>,
    pub unique_to_b: Vec<String>,
    pub recommendation: String,
}

impl CapabilityComparison {
    pub fn summary(&self) -> String {
        format!(
            "{} vs {}: Compute {:.2}x, Memory {:.2}x, Bandwidth {:.2}x",
            self.backend_a,
            self.backend_b,
            self.compute_ratio,
            self.memory_ratio,
            self.bandwidth_ratio
        )
    }
}

/// Backend introspection coordinator
pub struct BackendIntrospector {
    capabilities_cache: HashMap<String, BackendCapabilities>,
}

impl BackendIntrospector {
    pub fn new() -> Self {
        Self {
            capabilities_cache: HashMap::new(),
        }
    }

    /// Introspect a backend and return comprehensive capabilities
    pub fn introspect_backend(
        &mut self,
        backend: &dyn Backend,
        device: &Device,
    ) -> BackendResult<BackendCapabilities> {
        let cache_key = format!("{}_{}", backend.backend_type(), device.id());

        // Check cache first
        if let Some(cached) = self.capabilities_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Perform introspection
        let capabilities = self.perform_introspection(backend, device)?;

        // Cache result
        self.capabilities_cache
            .insert(cache_key, capabilities.clone());

        Ok(capabilities)
    }

    fn perform_introspection(
        &self,
        backend: &dyn Backend,
        device: &Device,
    ) -> BackendResult<BackendCapabilities> {
        // This is a simplified version - in production, each backend would provide
        // detailed capability information through the Backend trait

        let backend_type = backend.backend_type();

        // Device info (simplified)
        let device_info = DeviceInfo {
            name: format!("{} Device {}", backend_type, device.id()),
            vendor: self.detect_vendor(&backend_type),
            driver_version: "1.0.0".to_string(),
            device_type: self.classify_device(&backend_type),
            pci_bus_id: None,
            compute_capability: None,
        };

        // Operations (simplified)
        let mut supported_ops = HashSet::new();
        let mut native_ops = HashSet::new();

        // Common operations
        for op in &[
            "add",
            "sub",
            "mul",
            "div",
            "matmul",
            "conv2d",
            "relu",
            "sigmoid",
            "softmax",
            "layernorm",
            "batchnorm",
            "reduce_sum",
            "reduce_mean",
        ] {
            supported_ops.insert(op.to_string());
            native_ops.insert(op.to_string());
        }

        let operations = OperationSupport {
            supported_operations: supported_ops.clone(),
            native_operations: native_ops,
            emulated_operations: HashSet::new(),
            performance_tiers: supported_ops
                .iter()
                .map(|op| (op.clone(), PerformanceTier::Good))
                .collect(),
        };

        // Data types
        let data_types = DataTypeSupport {
            float_types: vec![FloatType::F32, FloatType::F64],
            integer_types: vec![IntegerType::I32, IntegerType::I64],
            complex_support: false,
            quantized_support: false,
            custom_types: Vec::new(),
        };

        // Memory (estimated based on backend type)
        let (total_mem, available_mem, unified) = match backend_type {
            BackendType::Cpu => (16 * 1024 * 1024 * 1024, 12 * 1024 * 1024 * 1024, true),
            BackendType::Cuda => (8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024, false),
            BackendType::Metal => (8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024, true),
            _ => (4 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024, false),
        };

        let memory = MemoryCapabilities {
            total_memory: total_mem,
            available_memory: available_mem,
            unified_memory: unified,
            pinned_memory_support: backend_type == BackendType::Cuda,
            zero_copy_support: unified,
            peer_to_peer_support: backend_type == BackendType::Cuda,
            memory_pools: vec![MemoryPoolInfo {
                name: "Main".to_string(),
                size: total_mem,
                location: if unified {
                    MemoryLocation::Shared
                } else {
                    MemoryLocation::Device
                },
                access_speed: MemoryAccessSpeed::Fast,
            }],
        };

        // Compute
        let compute = ComputeCapabilities {
            compute_units: match backend_type {
                BackendType::Cpu => 8,
                BackendType::Cuda => 80,
                BackendType::Metal => 16,
                _ => 4,
            },
            max_threads_per_block: match backend_type {
                BackendType::Cuda => Some(1024),
                _ => None,
            },
            max_workgroup_size: None,
            warp_size: match backend_type {
                BackendType::Cuda => Some(32),
                _ => None,
            },
            shared_memory_per_block: match backend_type {
                BackendType::Cuda => Some(48 * 1024),
                _ => None,
            },
            registers_per_block: None,
            max_concurrent_kernels: match backend_type {
                BackendType::Cuda => Some(32),
                _ => None,
            },
        };

        // Optimizations
        let optimizations = OptimizationFeatures {
            simd_support: backend_type == BackendType::Cpu,
            simd_width: if backend_type == BackendType::Cpu {
                Some(8)
            } else {
                None
            },
            tensor_core_support: backend_type == BackendType::Cuda,
            mixed_precision_support: backend_type == BackendType::Cuda
                || backend_type == BackendType::Metal,
            kernel_fusion_support: true,
            auto_tuning_support: true,
            jit_compilation_support: true,
            async_execution_support: backend_type != BackendType::Cpu,
        };

        // Performance
        let performance = PerformanceProfile {
            peak_flops: match backend_type {
                BackendType::Cpu => 1e12,    // 1 TFLOPS
                BackendType::Cuda => 20e12,  // 20 TFLOPS
                BackendType::Metal => 10e12, // 10 TFLOPS
                _ => 0.5e12,
            },
            peak_flops_fp16: match backend_type {
                BackendType::Cuda => Some(40e12),
                BackendType::Metal => Some(20e12),
                _ => None,
            },
            peak_flops_int8: match backend_type {
                BackendType::Cuda => Some(80e12),
                _ => None,
            },
            memory_bandwidth: match backend_type {
                BackendType::Cpu => 100e9,   // 100 GB/s
                BackendType::Cuda => 900e9,  // 900 GB/s
                BackendType::Metal => 400e9, // 400 GB/s
                _ => 50e9,
            },
            cache_hierarchy: vec![
                CacheLevel {
                    level: 1,
                    size: 32 * 1024,
                    line_size: 64,
                    associativity: 8,
                },
                CacheLevel {
                    level: 2,
                    size: 256 * 1024,
                    line_size: 64,
                    associativity: 8,
                },
                CacheLevel {
                    level: 3,
                    size: 8 * 1024 * 1024,
                    line_size: 64,
                    associativity: 16,
                },
            ],
            typical_latency_us: match backend_type {
                BackendType::Cpu => 1.0,
                BackendType::Cuda => 10.0,
                BackendType::Metal => 5.0,
                _ => 20.0,
            },
            throughput_estimate: 1.0,
        };

        // Limits
        let limits = ResourceLimits {
            max_allocation_size: total_mem / 2,
            max_buffer_size: total_mem / 2,
            max_texture_size: Some((16384, 16384, 2048)),
            max_threads: match backend_type {
                BackendType::Cpu => 64,
                BackendType::Cuda => 1024 * 80,
                _ => 256,
            },
            max_dispatch_size: Some((65535, 65535, 65535)),
            max_constant_buffer_size: Some(64 * 1024),
        };

        // Extensions
        let extensions = Vec::new();

        Ok(BackendCapabilities {
            backend_type,
            device_info,
            operations,
            data_types,
            memory,
            compute,
            optimizations,
            performance,
            limits,
            extensions,
        })
    }

    fn detect_vendor(&self, backend_type: &BackendType) -> String {
        match backend_type {
            BackendType::Cpu => "CPU Vendor".to_string(),
            BackendType::Cuda => "NVIDIA".to_string(),
            BackendType::Metal => "Apple".to_string(),
            BackendType::Rocm => "AMD".to_string(),
            BackendType::WebGpu => "WebGPU".to_string(),
            BackendType::Auto => "Auto".to_string(),
        }
    }

    fn classify_device(&self, backend_type: &BackendType) -> DeviceType {
        match backend_type {
            BackendType::Cpu => DeviceType::CPU,
            BackendType::Cuda => DeviceType::DiscreteGPU,
            BackendType::Metal => DeviceType::IntegratedGPU,
            BackendType::Rocm => DeviceType::DiscreteGPU,
            BackendType::WebGpu => DeviceType::VirtualGPU,
            BackendType::Auto => DeviceType::Unknown,
        }
    }

    /// Compare two backends
    pub fn compare_backends(
        &mut self,
        backend_a: &dyn Backend,
        device_a: &Device,
        backend_b: &dyn Backend,
        device_b: &Device,
    ) -> BackendResult<CapabilityComparison> {
        let caps_a = self.introspect_backend(backend_a, device_a)?;
        let caps_b = self.introspect_backend(backend_b, device_b)?;

        Ok(caps_a.compare_with(&caps_b))
    }

    /// Clear capability cache
    pub fn clear_cache(&mut self) {
        self.capabilities_cache.clear();
    }
}

impl Default for BackendIntrospector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_comparison() {
        let caps_a = create_test_capabilities(BackendType::Cuda);
        let caps_b = create_test_capabilities(BackendType::Cpu);

        let comparison = caps_a.compare_with(&caps_b);

        assert_eq!(comparison.backend_a, BackendType::Cuda);
        assert_eq!(comparison.backend_b, BackendType::Cpu);
        assert!(comparison.compute_ratio > 1.0); // CUDA should be faster
    }

    #[test]
    fn test_optimization_recommendations() {
        let caps = create_test_capabilities(BackendType::Cuda);
        let recommendations = caps.optimization_recommendations();

        assert!(!recommendations.is_empty());

        // Should recommend tensor cores for CUDA
        let has_tensor_core_rec = recommendations
            .iter()
            .any(|r| r.title.contains("Tensor Cores"));
        assert!(has_tensor_core_rec);
    }

    #[test]
    fn test_operation_support() {
        let mut ops = HashSet::new();
        ops.insert("matmul".to_string());

        let mut native = HashSet::new();
        native.insert("matmul".to_string());

        let support = OperationSupport {
            supported_operations: ops,
            native_operations: native,
            emulated_operations: HashSet::new(),
            performance_tiers: HashMap::new(),
        };

        assert!(support.is_native("matmul"));
        assert!(!support.is_native("custom_op"));
    }

    #[test]
    fn test_performance_tier_ordering() {
        assert!(PerformanceTier::Optimal > PerformanceTier::Good);
        assert!(PerformanceTier::Good > PerformanceTier::Fair);
        assert!(PerformanceTier::Fair > PerformanceTier::Poor);
    }

    #[test]
    fn test_detailed_report_generation() {
        let caps = create_test_capabilities(BackendType::Cuda);
        let report = caps.detailed_report();

        assert!(report.contains("Backend Capabilities"));
        assert!(report.contains("Memory:"));
        assert!(report.contains("Compute:"));
        assert!(report.contains("Optimizations:"));
    }

    #[test]
    fn test_backend_introspector() {
        let introspector = BackendIntrospector::new();
        assert_eq!(introspector.capabilities_cache.len(), 0);
    }

    #[test]
    fn test_recommendation_priority_ordering() {
        assert!(RecommendationPriority::Critical > RecommendationPriority::High);
        assert!(RecommendationPriority::High > RecommendationPriority::Medium);
        assert!(RecommendationPriority::Medium > RecommendationPriority::Low);
    }

    // Helper function
    fn create_test_capabilities(backend_type: BackendType) -> BackendCapabilities {
        BackendCapabilities {
            backend_type: backend_type.clone(),
            device_info: DeviceInfo {
                name: "Test Device".to_string(),
                vendor: "Test Vendor".to_string(),
                driver_version: "1.0".to_string(),
                device_type: DeviceType::DiscreteGPU,
                pci_bus_id: None,
                compute_capability: None,
            },
            operations: OperationSupport {
                supported_operations: HashSet::new(),
                native_operations: HashSet::new(),
                emulated_operations: HashSet::new(),
                performance_tiers: HashMap::new(),
            },
            data_types: DataTypeSupport {
                float_types: vec![FloatType::F32],
                integer_types: vec![IntegerType::I32],
                complex_support: false,
                quantized_support: false,
                custom_types: Vec::new(),
            },
            memory: MemoryCapabilities {
                total_memory: if backend_type == BackendType::Cuda {
                    8_000_000_000
                } else {
                    16_000_000_000
                },
                available_memory: 6_000_000_000,
                unified_memory: backend_type == BackendType::Cpu,
                pinned_memory_support: backend_type == BackendType::Cuda,
                zero_copy_support: false,
                peer_to_peer_support: false,
                memory_pools: Vec::new(),
            },
            compute: ComputeCapabilities {
                compute_units: if backend_type == BackendType::Cuda {
                    80
                } else {
                    8
                },
                max_threads_per_block: None,
                max_workgroup_size: None,
                warp_size: None,
                shared_memory_per_block: None,
                registers_per_block: None,
                max_concurrent_kernels: None,
            },
            optimizations: OptimizationFeatures {
                simd_support: backend_type == BackendType::Cpu,
                simd_width: None,
                tensor_core_support: backend_type == BackendType::Cuda,
                mixed_precision_support: backend_type == BackendType::Cuda,
                kernel_fusion_support: true,
                auto_tuning_support: true,
                jit_compilation_support: true,
                async_execution_support: true,
            },
            performance: PerformanceProfile {
                peak_flops: if backend_type == BackendType::Cuda {
                    20e12
                } else {
                    1e12
                },
                peak_flops_fp16: None,
                peak_flops_int8: None,
                memory_bandwidth: if backend_type == BackendType::Cuda {
                    900e9
                } else {
                    100e9
                },
                cache_hierarchy: Vec::new(),
                typical_latency_us: 1.0,
                throughput_estimate: 1.0,
            },
            limits: ResourceLimits {
                max_allocation_size: 4_000_000_000,
                max_buffer_size: 4_000_000_000,
                max_texture_size: None,
                max_threads: 1024,
                max_dispatch_size: None,
                max_constant_buffer_size: None,
            },
            extensions: Vec::new(),
        }
    }
}
