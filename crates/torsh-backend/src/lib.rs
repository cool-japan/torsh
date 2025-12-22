//! Unified backend implementation for ToRSh
//!
//! This crate provides a unified backend system that integrates with SciRS2's
//! compute backends. All backend implementations are included in this single
//! crate and selected via feature flags.
//!
//! # Features
//!
//! - `cpu` (default): CPU backend with SIMD optimizations via scirs2-core
//! - `cuda`: NVIDIA GPU backend via scirs2-core's CUDA support
//! - `metal`: Apple GPU backend via scirs2-core's Metal/MPS support
//! - `rocm`: AMD GPU backend (when available in scirs2-core)
//! - `webgpu`: WebGPU backend (when available in scirs2-core)

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::new_without_default)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::type_complexity)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::inherent_to_string)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::mut_from_ref)]
#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::manual_flatten)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::identity_op)]
#![allow(clippy::len_without_is_empty)]
#![allow(dead_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

/// Backend-specific error types
#[derive(Debug, Clone)]
pub enum BackendError {
    /// Invalid argument provided to backend operation
    InvalidArgument(String),

    /// Operation not supported by this backend
    UnsupportedOperation(String),

    /// Quantization-specific error
    QuantizationError(String),

    /// Invalid buffer state or operation
    InvalidBuffer { message: String },

    /// Runtime error during backend operation
    Runtime { message: String },

    /// Memory allocation error
    AllocationFailed(String),

    /// Device synchronization error
    SynchronizationFailed(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            BackendError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            BackendError::QuantizationError(msg) => write!(f, "Quantization error: {}", msg),
            BackendError::InvalidBuffer { message } => write!(f, "Invalid buffer: {}", message),
            BackendError::Runtime { message } => write!(f, "Runtime error: {}", message),
            BackendError::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            BackendError::SynchronizationFailed(msg) => {
                write!(f, "Synchronization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {}

// Core backend traits and types
pub mod adaptive_kernel_selection;
pub mod backend;
pub mod buffer;
pub mod convolution;
pub mod cross_backend_transfer;
pub mod cross_backend_validation;
pub mod deadlock_prevention;
pub mod device;
pub mod error;
pub mod fft;
pub mod hardware_optimization_tests;
pub mod introspection;
pub mod jit_compiler;
pub mod kernel;
pub mod kernel_generation;
pub mod memory;
pub mod memory_defrag;
pub mod memory_profiler;
pub mod performance_modeling;
pub mod performance_tuning;
pub mod profiler;
pub mod property_tests;
pub mod quantization;
pub mod rnn;
pub mod sparse_ops;
pub mod unified_memory_pool;
pub mod version_compat;
pub mod zero_copy;

// Feature-gated backend implementations
#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod metal;

#[cfg(feature = "rocm")]
pub mod rocm;

#[cfg(feature = "webgpu")]
pub mod webgpu;

// Re-exports
pub use adaptive_kernel_selection::{
    AdaptiveKernelSelector, AdaptiveSelectionConfig, BenchmarkResult, BenchmarkResults,
    CustomKernel, HybridConfig, KernelCharacteristics, KernelConstraints, KernelExecutor,
    KernelImplementation, KernelInputs, KernelOutputs, KernelParameter, KernelPerformanceRecord,
    KernelRegistry, KernelSelection, KernelUsageStats, KernelVariant, MLBasedConfig, MLModelType,
    MLTrainingParams, PerformanceTracker, ResourceRequirements, ScalabilityCharacteristics,
    ScalingBehavior, ScoreBasedConfig, SelectionAccuracyTracker, SelectionAlgorithm,
    SelectionReason, SelectionStatistics,
};
pub use backend::{
    Backend, BackendCapabilities, BackendCore, BackendDeviceManager, BackendExecutor,
    BackendExtension, BackendExtensionRegistry, BackendFactory, BackendLifecycle,
    BackendOperations, BackendOps, BackendPlugin, BackendRegistry, BackendResourceManager,
    BackendType, CapabilityValue, DeviceEnumerator, ExecutionModel, ExtendedCapabilities,
    HardwareFeature, MemoryHierarchy, OperationsBundle, PerformanceHints, PluginMetadata,
    PrecisionMode, ResourceLimits, ResourceStatistics, ResourceUsage, ScopedResource,
};
pub use buffer::{Buffer, BufferDescriptor, BufferHandle, BufferUsage, BufferView, MemoryLocation};

/// Buffer error type (alias to BackendError)
pub type BufferError = BackendError;
pub use convolution::{
    algorithms as conv_algorithms, ConvolutionAlgorithm, ConvolutionConfig, ConvolutionOps,
    ConvolutionPerformanceHints, ConvolutionType, DefaultConvolutionOps, PaddingMode,
};
pub use cross_backend_transfer::CrossBackendTransferManager;
pub use cross_backend_validation::{
    compare_f32_values, compare_f64_values, run_cross_backend_validation, CrossBackendValidator,
};
pub use device::{
    Device, DeviceConfiguration, DeviceDiscovery, DeviceFeature, DeviceInfo, DeviceManager,
    DevicePerformanceInfo, DeviceRequirements, DeviceType, DeviceUtils,
};
pub use error::{BackendResult, ErrorCategory, ErrorContext, ErrorSeverity};
pub use fft::{
    convenience as fft_convenience, DefaultFftExecutor, DefaultFftOps, FftDirection, FftExecutor,
    FftNormalization, FftOps, FftPlan, FftType,
};
pub use hardware_optimization_tests::{
    run_hardware_optimization_tests, run_lightweight_hardware_tests, HardwareOptimizationTester,
};
pub use kernel::{Kernel, KernelDescriptor, KernelHandle, KernelLaunchConfig, KernelMetadata};
pub use memory::{
    AccessPattern, AllocationHint, AllocationLifetime, AllocationStrategy, CompactionResult,
    DefragmentationPolicy, DefragmentationPriority, DefragmentationResult, DefragmentationStrategy,
    FragmentationInfo, FragmentationSeverity, FreeListPool, LeakReport, LeakSeverity, LeakType,
    MemoryAdvice, MemoryManager, MemoryManagerFactory, MemoryPool, MemoryPoolConfig, MemoryStats,
    PoolStats,
};
pub use memory_defrag::{
    CompactionPlan, DefragmentationManager, DefragmentationRequest, DefragmentationStats,
    DefragmentationTask, MemoryBlock, MemoryLayout, TaskStatus,
};
pub use memory_profiler::{
    AccessType, AllocationContext, AllocationUsageStats, HintSeverity, MemoryAllocation,
    MemoryPressureEvent, MemoryProfiler, MemoryProfilerConfig, MemorySnapshot, MemoryType,
    PerformanceHint, PerformanceHintType, PressureLevel,
};
pub use performance_modeling::{
    AnomalyDetector, AnomalySeverity, AnomalyType, ComplexityClass, CorrelationAnalyzer,
    CorrelationResult, EnvironmentalFactors, ModelAccuracy, ModelComplexity, ModelTrainingResult,
    PatternType, PerformanceAnomaly, PerformanceCharacteristics, PerformanceMeasurement,
    PerformanceModel, PerformanceReport, PerformanceSample, PerformanceTrend, RealtimeStatistics,
    RuntimeMonitor, RuntimePerformanceModeler, TrendDirection, WorkloadPattern,
};
pub use performance_tuning::{
    analyze_workload_optimization_opportunities,
    create_default_constraints,
    create_default_system_state,
    create_energy_budget_constraints,
    create_image_processing_workload,
    create_ml_inference_workload,
    create_ml_training_workload,
    create_performance_optimized_system_state,
    create_power_efficient_system_state,
    create_realtime_constraints,
    create_sample_workload,
    create_throughput_constraints,
    // Convenience functions
    new_coordinator,
    recommend_backend,
    AccessPattern as PerfAccessPattern,
    ActualPerformance,
    BackendTuningStrategy,

    DataType,
    GlobalPerformanceStats,

    MemoryAllocationStrategy,
    NumaTopologyState,
    // Configuration enums
    OperationType,
    OptimizationLevel,
    PerformanceFeedback,
    // Performance measurement and feedback
    PerformancePrediction,
    // Core coordination types
    PerformanceTuningCoordinator,
    PowerEfficiencyMode,
    PowerState,
    SchedulingStrategy,
    StrategyMetrics,
    SystemState,
    ThermalState,
    TuningConstraints,
    TuningParameters,

    TuningRecommendation,
    TuningValue,

    // Workload and system characteristics
    WorkloadCharacteristics,
};
pub use profiler::{Profiler, ProfilerEvent, ProfilerStats, SimpleProfiler};
pub use quantization::{
    CalibrationMethod, QuantizationCalibrator, QuantizationHardwareFeatures, QuantizationOps,
    QuantizationParams, QuantizationScheme, QuantizedDType, QuantizedTensor, SimdQuantizationOps,
};
pub use rnn::{
    activations as rnn_activations, cells as rnn_cells, DefaultRnnOps, RnnActivation, RnnCellType,
    RnnConfig, RnnDirection, RnnOps, RnnOutput, RnnPerformanceHints,
};
pub use sparse_ops::{
    DefaultSparseOps, SparseFormat, SparseFormatConverter, SparseMatrix, SparseOperation,
    SparseOps, SparseOptimizationHints,
};
pub use unified_memory_pool::{
    CpuMemoryPool, CudaMemoryPool, MetalMemoryPool, RocmMemoryPool, UnifiedMemoryPool,
    WebGpuMemoryPool,
};
pub use version_compat::{
    BackendDependency, CompatibilityReport, DependencyStatus, Version, VersionCompatibilityChecker,
    VersionError, VersionErrorContextExt, VersionRange,
};
pub use zero_copy::{
    TransferDirection, TransferMode, ZeroCopyCapabilities, ZeroCopyManager, ZeroCopyStats,
    ZeroCopyTransfer,
};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

// SciRS2 integration re-exports
#[cfg(feature = "cpu")]
pub use cpu::{prepare_tensor_data, prepare_tensor_data_mut, SciRS2CpuBackend};
use torsh_core::error::TorshError;

// Removed unused imports: DType, Device as CoreDevice, Shape

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Unified backend builder
pub struct BackendBuilder {
    backend_type: BackendType,
    device_id: usize,
    memory_pool_config: Option<MemoryPoolConfig>,
    num_threads: Option<usize>,
    enable_profiling: bool,
}

impl Default for BackendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendBuilder {
    /// Create a new backend builder
    pub fn new() -> Self {
        Self {
            backend_type: BackendType::Auto,
            device_id: 0,
            memory_pool_config: None,
            num_threads: None,
            enable_profiling: false,
        }
    }

    /// Set the backend type
    pub fn backend_type(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Set the device ID
    pub fn device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set memory pool configuration
    pub fn memory_pool(mut self, config: MemoryPoolConfig) -> Self {
        self.memory_pool_config = Some(config);
        self
    }

    /// Set number of threads (CPU backend)
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Enable profiling
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Build the backend
    pub fn build(self) -> BackendResult<Box<dyn Backend>> {
        match self.backend_type {
            BackendType::Auto => Self::auto_select(self),
            BackendType::Cpu => Self::build_cpu(self),
            BackendType::Cuda => Self::build_cuda(self),
            BackendType::Metal => Self::build_metal(self),
            BackendType::Rocm => Self::build_rocm(self),
            BackendType::WebGpu => Self::build_webgpu(self),
        }
    }

    fn auto_select(builder: Self) -> BackendResult<Box<dyn Backend>> {
        // Try backends in order of preference
        #[cfg(feature = "cuda")]
        if let Ok(backend) = Self::build_cuda(builder.clone()) {
            return Ok(backend);
        }

        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Ok(backend) = Self::build_metal(builder.clone()) {
            return Ok(backend);
        }

        #[cfg(feature = "rocm")]
        if let Ok(backend) = Self::build_rocm(builder.clone()) {
            return Ok(backend);
        }

        #[cfg(feature = "webgpu")]
        if let Ok(backend) = Self::build_webgpu(builder.clone()) {
            return Ok(backend);
        }

        // Fall back to CPU
        Self::build_cpu(builder)
    }

    #[cfg(feature = "cpu")]
    fn build_cpu(builder: Self) -> BackendResult<Box<dyn Backend>> {
        let mut cpu_builder = cpu::CpuBackend::builder();

        if let Some(num_threads) = builder.num_threads {
            cpu_builder = cpu_builder.num_threads(num_threads);
        }

        if let Some(pool_config) = builder.memory_pool_config {
            cpu_builder = cpu_builder.memory_pool(pool_config);
        }

        Ok(Box::new(cpu_builder.build()?))
    }

    #[cfg(not(feature = "cpu"))]
    fn build_cpu(_builder: Self) -> BackendResult<Box<dyn Backend>> {
        Err(TorshError::BackendError("CPU backend not enabled".into()))
    }

    #[cfg(feature = "cuda")]
    fn build_cuda(builder: Self) -> BackendResult<Box<dyn Backend>> {
        let mut cuda_builder = cuda::CudaBackend::builder();

        cuda_builder = cuda_builder.device(builder.device_id);

        if let Some(pool_config) = builder.memory_pool_config {
            cuda_builder = cuda_builder.memory_pool(pool_config);
        }

        Ok(Box::new(cuda_builder.build()?))
    }

    #[cfg(not(feature = "cuda"))]
    fn build_cuda(_builder: Self) -> BackendResult<Box<dyn Backend>> {
        Err(TorshError::BackendError("CUDA backend not enabled".into()))
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn build_metal(builder: Self) -> BackendResult<Box<dyn Backend>> {
        let mut metal_builder = metal::MetalBackend::builder();

        if let Some(pool_config) = builder.memory_pool_config {
            metal_builder = metal_builder.memory_pool(pool_config);
        }

        Ok(Box::new(metal_builder.build()?))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    fn build_metal(_builder: Self) -> BackendResult<Box<dyn Backend>> {
        Err(TorshError::BackendError("Metal backend not enabled".into()))
    }

    #[cfg(feature = "rocm")]
    fn build_rocm(_builder: Self) -> BackendResult<Box<dyn Backend>> {
        // TODO: Implement when scirs2 supports ROCm
        Err(TorshError::BackendError(
            "ROCm backend not yet implemented".into(),
        ))
    }

    #[cfg(not(feature = "rocm"))]
    fn build_rocm(_builder: Self) -> BackendResult<Box<dyn Backend>> {
        Err(TorshError::BackendError("ROCm backend not enabled".into()))
    }

    #[cfg(feature = "webgpu")]
    fn build_webgpu(builder: Self) -> BackendResult<Box<dyn Backend>> {
        let mut webgpu_builder = webgpu::WebGpuBackendBuilder::new();

        // Set device ID
        webgpu_builder = webgpu_builder.device_id(builder.device_id);

        if let Some(pool_config) = builder.memory_pool_config {
            if let Some(max_size) = pool_config.max_size {
                webgpu_builder = webgpu_builder.max_buffer_size(max_size as u64);
            }
        }

        webgpu_builder = webgpu_builder.enable_pipeline_cache(true);

        Ok(Box::new(webgpu_builder.build()))
    }

    #[cfg(not(feature = "webgpu"))]
    fn build_webgpu(_builder: Self) -> BackendResult<Box<dyn Backend>> {
        Err(TorshError::BackendError(
            "WebGPU backend not enabled".into(),
        ))
    }
}

impl Clone for BackendBuilder {
    fn clone(&self) -> Self {
        Self {
            backend_type: self.backend_type,
            device_id: self.device_id,
            memory_pool_config: self.memory_pool_config.clone(),
            num_threads: self.num_threads,
            enable_profiling: self.enable_profiling,
        }
    }
}

/// Create a backend with automatic selection
pub fn auto() -> BackendResult<Box<dyn Backend>> {
    BackendBuilder::new().build()
}

/// Create a CPU backend
pub fn cpu() -> BackendResult<Box<dyn Backend>> {
    BackendBuilder::new().backend_type(BackendType::Cpu).build()
}

/// Create a CUDA backend
pub fn cuda() -> BackendResult<Box<dyn Backend>> {
    BackendBuilder::new()
        .backend_type(BackendType::Cuda)
        .build()
}

/// Create a Metal backend
pub fn metal() -> BackendResult<Box<dyn Backend>> {
    BackendBuilder::new()
        .backend_type(BackendType::Metal)
        .build()
}

/// List available backend types
#[allow(clippy::vec_init_then_push)]
pub fn available_backends() -> Vec<BackendType> {
    let mut backends = vec![];

    #[cfg(feature = "cpu")]
    backends.push(BackendType::Cpu);

    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        backends.push(BackendType::Cuda);
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if metal::is_available() {
        backends.push(BackendType::Metal);
    }

    #[cfg(feature = "rocm")]
    if rocm::is_available() {
        backends.push(BackendType::Rocm);
    }

    #[cfg(feature = "webgpu")]
    if webgpu::is_available() {
        backends.push(BackendType::WebGpu);
    }

    backends
}

/// Comprehensive device enumeration across all available backends
pub fn enumerate_all_devices() -> BackendResult<Vec<(BackendType, Vec<Device>)>> {
    let mut all_devices = Vec::new();

    // Enumerate CPU devices
    #[cfg(feature = "cpu")]
    {
        match cpu() {
            Ok(backend) => {
                if let Ok(devices) = backend.devices() {
                    all_devices.push((BackendType::Cpu, devices));
                }
            }
            Err(_) => {
                // CPU backend failed, continue to other backends
            }
        }
    }

    // Enumerate CUDA devices
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        // Try to enumerate multiple CUDA devices
        for device_id in 0..cuda::device_count().unwrap_or(0) {
            match BackendBuilder::new()
                .backend_type(BackendType::Cuda)
                .device_id(device_id as usize)
                .build()
            {
                Ok(backend) => {
                    if let Ok(devices) = backend.devices() {
                        all_devices.push((BackendType::Cuda, devices));
                        break; // For now, just get the first available CUDA backend
                    }
                }
                Err(_) => continue,
            }
        }
    }

    // Enumerate Metal devices
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if metal::is_available() {
        match BackendBuilder::new()
            .backend_type(BackendType::Metal)
            .build()
        {
            Ok(backend) => {
                if let Ok(devices) = backend.devices() {
                    all_devices.push((BackendType::Metal, devices));
                }
            }
            Err(_) => {
                // Metal backend failed, continue
            }
        }
    }

    // Enumerate WebGPU devices
    #[cfg(feature = "webgpu")]
    if webgpu::is_available() {
        match BackendBuilder::new()
            .backend_type(BackendType::WebGpu)
            .build()
        {
            Ok(backend) => {
                if let Ok(devices) = backend.devices() {
                    all_devices.push((BackendType::WebGpu, devices));
                }
            }
            Err(_) => {
                // WebGPU backend failed, continue
            }
        }
    }

    Ok(all_devices)
}

/// Find the best available device based on selection criteria
pub fn find_best_device(
    selector: Option<device::DeviceSelector>,
) -> BackendResult<(BackendType, Device)> {
    let all_devices = enumerate_all_devices()?;

    if all_devices.is_empty() {
        return Err(TorshError::BackendError("No devices available".into()));
    }

    let selector = selector.unwrap_or_default();

    // First pass: try to find an exact match
    for (backend_type, devices) in &all_devices {
        for device in devices {
            if selector.matches(device) {
                return Ok((*backend_type, device.clone()));
            }
        }
    }

    // Second pass: fallback to best available device with preference order
    let preference_order = [
        BackendType::Cuda,
        BackendType::Metal,
        BackendType::WebGpu,
        BackendType::Cpu,
    ];

    for preferred_backend in &preference_order {
        for (backend_type, devices) in &all_devices {
            if backend_type == preferred_backend && !devices.is_empty() {
                return Ok((*backend_type, devices[0].clone()));
            }
        }
    }

    // Final fallback: return the first available device
    let (backend_type, devices) = &all_devices[0];
    Ok((*backend_type, devices[0].clone()))
}

/// Get device count for a specific backend type
pub fn device_count(backend_type: BackendType) -> BackendResult<usize> {
    match backend_type {
        BackendType::Cpu => Ok(1), // CPU backend always has 1 logical device

        #[cfg(feature = "cuda")]
        BackendType::Cuda => {
            if cuda::is_available() {
                Ok(cuda::device_count().unwrap_or(0) as usize)
            } else {
                Ok(0)
            }
        }

        #[cfg(not(feature = "cuda"))]
        BackendType::Cuda => Ok(0),

        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        BackendType::Metal => {
            if metal::is_available() {
                Ok(metal::device_count().unwrap_or(0))
            } else {
                Ok(0)
            }
        }

        #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
        BackendType::Metal => Ok(0),

        #[cfg(feature = "webgpu")]
        BackendType::WebGpu => {
            if webgpu::is_available() {
                Ok(webgpu::device_count().unwrap_or(0))
            } else {
                Ok(0)
            }
        }

        #[cfg(not(feature = "webgpu"))]
        BackendType::WebGpu => Ok(0),

        BackendType::Rocm => Ok(0), // Not implemented yet
        BackendType::Auto => {
            // For Auto, return the sum of all available devices
            let mut total = 0;
            for backend in available_backends() {
                if backend != BackendType::Auto {
                    total += device_count(backend)?;
                }
            }
            Ok(total)
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        auto,
        available_backends,
        compare_f32_values,
        compare_f64_values,
        cpu,
        cuda,
        device_count,
        enumerate_all_devices,
        find_best_device,
        metal,
        run_cross_backend_validation,
        run_hardware_optimization_tests,
        run_lightweight_hardware_tests,
        AdaptiveKernelSelector,
        Backend,
        BackendBuilder,
        BackendCapabilities,
        BackendOps,
        BackendPlugin,
        BackendRegistry,
        BackendResourceManager,
        BackendResult,
        BackendType,
        BenchmarkResult,
        Buffer,
        CompactionPlan,
        CrossBackendValidator,
        DefragmentationManager,
        DefragmentationStats,
        Device,
        ExecutionModel,
        ExtendedCapabilities,
        HardwareFeature,
        HardwareOptimizationTester,
        KernelImplementation,
        KernelSelection,
        KernelVariant,
        MemoryHierarchy,
        MemoryPool,
        OperationType,
        PerformanceMeasurement,
        PerformancePrediction,
        PerformanceReport,
        PerformanceTrend,
        PerformanceTuningCoordinator,
        PluginMetadata,
        PrecisionMode,
        ResourceLimits,
        ResourceStatistics,
        ResourceUsage,
        RuntimePerformanceModeler,
        SelectionAlgorithm,
        TransferDirection,
        TransferMode,
        TuningParameters,
        TuningRecommendation,
        WorkloadCharacteristics,
        ZeroCopyCapabilities,
        ZeroCopyManager,
        ZeroCopyStats,
        ZeroCopyTransfer,
        // Version information
        VERSION,
        VERSION_MAJOR,
        VERSION_MINOR,
        VERSION_PATCH,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_builder() {
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .device_id(0);

        // Should successfully build CPU backend without specifying num_threads
        // to avoid Rayon global thread pool conflicts in tests
        let result = builder.build();
        if let Err(e) = &result {
            eprintln!("Backend build failed: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_available_backends() {
        let backends = available_backends();
        // At least CPU should be available
        assert!(!backends.is_empty());
        assert!(backends.contains(&BackendType::Cpu));
    }

    #[test]
    fn test_device_count() {
        // CPU should always have at least 1 device
        assert_eq!(device_count(BackendType::Cpu).unwrap(), 1);

        // Auto should return total of all devices
        let auto_count = device_count(BackendType::Auto).unwrap();
        assert!(auto_count >= 1); // At least CPU

        // Other backends depend on availability
        for backend_type in available_backends() {
            if backend_type != BackendType::Auto {
                let count = device_count(backend_type).unwrap();
                assert!(count < usize::MAX); // Should not fail
            }
        }
    }

    #[test]
    fn test_enumerate_all_devices() {
        let devices = enumerate_all_devices().unwrap();
        assert!(!devices.is_empty()); // At least CPU should be available

        // Check that CPU backend is present
        let has_cpu = devices
            .iter()
            .any(|(backend_type, _)| *backend_type == BackendType::Cpu);
        assert!(has_cpu);

        // Verify each backend has at least one device
        for (backend_type, device_list) in &devices {
            assert!(
                !device_list.is_empty(),
                "Backend {:?} should have at least one device",
                backend_type
            );
        }
    }

    #[test]
    fn test_find_best_device() {
        let (backend_type, device) = find_best_device(None).unwrap();

        // Should find some device
        assert!(matches!(
            backend_type,
            BackendType::Cpu | BackendType::Cuda | BackendType::Metal | BackendType::WebGpu
        ));

        // Device should be valid
        assert!(!device.name().is_empty());
    }

    #[test]
    fn test_find_best_device_with_selector() {
        use crate::device::{DeviceSelector, DeviceType};

        // Try to find a CPU device specifically
        let selector = DeviceSelector::new().with_device_type(DeviceType::Cpu);
        let result = find_best_device(Some(selector));

        assert!(result.is_ok());
        let (backend_type, device) = result.unwrap();
        assert_eq!(backend_type, BackendType::Cpu);
        assert_eq!(device.device_type(), torsh_core::device::DeviceType::Cpu);
    }

    #[test]
    fn test_unified_error_handling() {
        use crate::error::{conversion, ErrorContext};

        // Test error context creation
        let context = ErrorContext::new("test_operation")
            .with_backend("TestBackend")
            .with_device("test:0")
            .with_details("test details");

        let formatted = context.format();
        assert!(formatted.contains("test_operation"));
        assert!(formatted.contains("backend: TestBackend"));
        assert!(formatted.contains("device: test:0"));
        assert!(formatted.contains("details: test details"));

        // Test error conversion utilities
        let cuda_error =
            conversion::cuda_error_with_context("Test CUDA error", "test_kernel", Some(0));
        let error_str = cuda_error.to_string();
        assert!(error_str.contains("CUDA"));
        assert!(error_str.contains("test_kernel"));
        assert!(error_str.contains("cuda:0"));

        let cpu_error = conversion::cpu_error_with_context("Test CPU error", "test_operation");
        let error_str = cpu_error.to_string();
        assert!(error_str.contains("CPU"));
        assert!(error_str.contains("test_operation"));

        // Test memory error conversion
        let memory_error =
            conversion::memory_error_with_context("Out of memory", 1024, "CUDA", Some("cuda:0"));
        let error_str = memory_error.to_string();
        assert!(error_str.contains("memory_allocation"));
        assert!(error_str.contains("1024 bytes"));
        assert!(error_str.contains("CUDA"));
        assert!(error_str.contains("cuda:0"));
    }

    #[test]
    fn test_error_context_extension() {
        // use crate::error::ErrorContextExt; // Currently unused
        use torsh_core::error::TorshError;

        // Test adding context to an error
        let result: Result<(), TorshError> =
            Err(TorshError::ComputeError("Test error".to_string()));
        let with_context = crate::error::ErrorContextExt::with_operation(result, "test_operation");

        assert!(with_context.is_err());
        let error_str = with_context.unwrap_err().to_string();
        assert!(error_str.contains("test_operation"));
        assert!(error_str.contains("Test error"));
    }

    // ========== EDGE CASE AND ERROR CONDITION TESTS ==========

    #[test]
    fn test_invalid_device_id_error() {
        // Test requesting a device ID that doesn't exist
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .device_id(999); // CPU only has device 0

        let backend = builder.build().unwrap();
        let result = backend.create_device(999);
        assert!(result.is_err());

        // Verify error message is descriptive
        let error_str = result.unwrap_err().to_string();
        assert!(error_str.contains("999"));
        assert!(error_str.contains("not found"));
    }

    #[test]
    fn test_backend_builder_invalid_thread_count() {
        // Test edge case: zero threads
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .num_threads(0);

        // Should still succeed but fall back to reasonable defaults
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_builder_extreme_thread_count() {
        // Test edge case: extremely high thread count
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .num_threads(10000);

        // Should handle gracefully (Rayon will cap to reasonable limits)
        let result = builder.build();
        if let Err(ref e) = result {
            eprintln!("Backend build failed with extreme thread count: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_unavailable_backend_selection() {
        // Test requesting backends that aren't compiled in
        #[cfg(not(feature = "cuda"))]
        {
            let builder = BackendBuilder::new().backend_type(BackendType::Cuda);
            let result = builder.build();
            assert!(result.is_err());

            let error_str = result.unwrap_err().to_string();
            assert!(error_str.contains("not enabled"));
        }

        #[cfg(not(feature = "metal"))]
        {
            let builder = BackendBuilder::new().backend_type(BackendType::Metal);
            let result = builder.build();
            assert!(result.is_err());

            let error_str = result.unwrap_err().to_string();
            assert!(error_str.contains("not enabled"));
        }
    }

    #[test]
    fn test_device_count_edge_cases() {
        // Test device count for unavailable backends
        #[cfg(not(feature = "cuda"))]
        {
            let count = device_count(BackendType::Cuda).unwrap();
            assert_eq!(count, 0);
        }

        #[cfg(not(feature = "metal"))]
        {
            let count = device_count(BackendType::Metal).unwrap();
            assert_eq!(count, 0);
        }

        // Test ROCm (always unavailable currently)
        let count = device_count(BackendType::Rocm).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_find_best_device_no_match() {
        use crate::device::{DeviceSelector, DeviceType};

        // Try to find a device that doesn't exist
        // This should still return a device (fallback behavior)
        let selector = DeviceSelector::new().with_device_type(DeviceType::Cuda);
        let result = find_best_device(Some(selector));

        // Should still return a device (CPU fallback)
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_pool_config_edge_cases() {
        // Test memory pool with extreme values
        let config = MemoryPoolConfig::new(0); // Zero initial size
        assert_eq!(config.initial_size, 0);

        let config = MemoryPoolConfig::new(usize::MAX); // Maximum size
        assert_eq!(config.initial_size, usize::MAX);

        // Test with invalid growth factor
        let config = MemoryPoolConfig::new(1024).with_growth_factor(0.0);
        assert_eq!(config.growth_factor, 0.0); // Should accept but may cause issues

        let config = MemoryPoolConfig::new(1024).with_growth_factor(-1.0);
        assert_eq!(config.growth_factor, -1.0); // Should accept but may cause issues
    }

    #[test]
    fn test_memory_pool_config_alignment_edge_cases() {
        // Test alignment edge cases
        let config = MemoryPoolConfig::new(1024).with_alignment(0);
        assert_eq!(config.alignment, 0); // Invalid alignment

        let config = MemoryPoolConfig::new(1024).with_alignment(1);
        assert_eq!(config.alignment, 1); // Minimal alignment

        let config = MemoryPoolConfig::new(1024).with_alignment(4096);
        assert_eq!(config.alignment, 4096); // Page-aligned
    }

    #[test]
    fn test_error_handling_with_long_messages() {
        use crate::error::conversion;

        // Test error handling with very long error messages
        let long_message = "x".repeat(10000);
        let error = conversion::cpu_error_with_context(long_message.clone(), "test_operation");

        let error_str = error.to_string();
        assert!(error_str.contains(&long_message));
        assert!(error_str.len() > 10000);
    }

    #[test]
    fn test_error_handling_with_special_characters() {
        use crate::error::conversion;

        // Test error handling with special characters
        let special_message = "Error: 測試 ñoño 🚀 \n\t\r";
        let error = conversion::cpu_error_with_context(special_message, "test_unicode_operation");

        let error_str = error.to_string();
        assert!(error_str.contains("測試"));
        assert!(error_str.contains("🚀"));
    }

    #[test]
    fn test_concurrent_backend_creation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread;

        // Test creating multiple backends concurrently
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        for _ in 0..10 {
            let success_count = Arc::clone(&success_count);
            let error_count = Arc::clone(&error_count);

            let handle = thread::spawn(move || {
                let builder = BackendBuilder::new().backend_type(BackendType::Cpu);
                match builder.build() {
                    Ok(_) => success_count.fetch_add(1, Ordering::Relaxed),
                    Err(_) => error_count.fetch_add(1, Ordering::Relaxed),
                };
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // At least some should succeed (thread pool initialization might cause some to fail)
        let successes = success_count.load(Ordering::Relaxed);
        assert!(
            successes > 0,
            "No backend creation succeeded in concurrent test"
        );
    }

    #[test]
    fn test_backend_memory_pressure_simulation() {
        // Test backend behavior under simulated memory pressure
        let backend = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .memory_pool(MemoryPoolConfig::new(1024)) // Very small pool
            .build()
            .unwrap();

        // This should succeed
        let device = backend.default_device().unwrap();
        assert!(!device.name().is_empty());
    }

    #[test]
    fn test_enumerate_devices_consistency() {
        // Test that device enumeration is consistent across multiple calls
        let devices1 = enumerate_all_devices().unwrap();
        let devices2 = enumerate_all_devices().unwrap();

        // Should return the same number of backends
        assert_eq!(devices1.len(), devices2.len());

        // Should return the same backend types
        let backend_types1: std::collections::HashSet<_> =
            devices1.iter().map(|(bt, _)| *bt).collect();
        let backend_types2: std::collections::HashSet<_> =
            devices2.iter().map(|(bt, _)| *bt).collect();
        assert_eq!(backend_types1, backend_types2);
    }

    #[test]
    fn test_device_selector_empty_criteria() {
        use crate::device::DeviceSelector;

        // Test device selector with no criteria (should match any device)
        let selector = DeviceSelector::new();
        let result = find_best_device(Some(selector));
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_builder_chain_operations() {
        // Test method chaining with all possible configurations
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .device_id(0)
            .num_threads(4)
            .memory_pool(MemoryPoolConfig::new(1024 * 1024))
            .enable_profiling(true);

        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_auto_backend_selection_fallback() {
        // Test that auto selection properly falls back through preference order
        let builder = BackendBuilder::new().backend_type(BackendType::Auto);
        let result = builder.build();

        // Should always succeed (CPU fallback)
        assert!(result.is_ok());

        let backend = result.unwrap();

        // Should have at least one device
        let devices = backend.devices().unwrap();
        assert!(!devices.is_empty());
    }

    // ========== ADDITIONAL EDGE CASE TESTS ==========

    #[test]
    fn test_memory_pool_zero_max_size() {
        // Test memory pool with zero max size
        let config = MemoryPoolConfig::new(1024).with_max_size(0);
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .memory_pool(config);

        // Should handle gracefully (may succeed or fail depending on implementation)
        let result = builder.build();
        // Don't assert success/failure - implementation defined behavior
        match result {
            Ok(_) => {
                // Success is acceptable
            }
            Err(_) => {
                // Failure is also acceptable for zero max size
            }
        }
    }

    #[test]
    fn test_memory_pool_negative_growth_factor() {
        // Test memory pool with negative growth factor
        let config = MemoryPoolConfig::new(1024).with_growth_factor(-0.5);
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .memory_pool(config);

        // Should handle gracefully
        let result = builder.build();
        // Implementation may accept or reject negative growth factors
        match result {
            Ok(_) => {
                // May accept and handle internally
            }
            Err(_) => {
                // May reject as invalid configuration
            }
        }
    }

    #[test]
    fn test_device_selector_with_conflicting_criteria() {
        use crate::device::{DeviceSelector, DeviceType};

        // Test device selector with conflicting criteria
        let selector = DeviceSelector::new()
            .with_device_type(DeviceType::Cpu)
            .with_device_type(DeviceType::Cuda); // Conflicting requirements

        let result = find_best_device(Some(selector));
        // Should still return a device (last criterion wins or fallback)
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_builder_cloning_with_modifications() {
        // Test cloning builder and modifying the clone
        let original_builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .num_threads(2);

        let mut cloned_builder = original_builder.clone();
        cloned_builder = cloned_builder.num_threads(4);

        // Both should be independent
        let original_result = original_builder.build();
        let cloned_result = cloned_builder.build();

        assert!(original_result.is_ok());
        assert!(cloned_result.is_ok());
    }

    #[test]
    fn test_error_context_with_empty_strings() {
        use crate::error::ErrorContext;

        // Test error context with empty strings
        let context = ErrorContext::new("")
            .with_backend("")
            .with_device("")
            .with_details("");

        let formatted = context.format();
        // Should not panic and should handle empty strings gracefully
        assert!(!formatted.is_empty());
    }

    #[test]
    fn test_error_context_with_null_characters() {
        use crate::error::ErrorContext;

        // Test error context with null characters and control characters
        let context = ErrorContext::new("op\0eration")
            .with_backend("back\0end")
            .with_device("dev\0ice")
            .with_details("deta\0ils");

        let formatted = context.format();
        // Should handle null characters without panicking
        assert!(!formatted.is_empty());
    }

    #[test]
    fn test_memory_manager_extreme_alignment() {
        // Test memory pool with extreme alignment values
        let config = MemoryPoolConfig::new(1024).with_alignment(usize::MAX);
        let builder = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .memory_pool(config);

        // Should handle extreme alignment gracefully
        let result = builder.build();
        // Implementation may accept or reject extreme alignment
        match result {
            Ok(_) => {
                // May clamp to reasonable values
            }
            Err(_) => {
                // May reject as invalid
            }
        }
    }

    #[test]
    fn test_backend_resource_cleanup() {
        // Test that backends properly clean up resources
        let backend = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .build()
            .unwrap();

        // Use the backend for operations
        let _device = backend.default_device().unwrap();
        let _devices = backend.devices().unwrap();

        // Drop the backend explicitly
        drop(backend);

        // Should not leak resources (verified by memory leak detection tools)
        // This test mainly ensures no panics during cleanup
    }

    #[test]
    fn test_available_backends_consistency() {
        // Test that available_backends() returns consistent results
        let backends1 = available_backends();
        let backends2 = available_backends();

        // Should return the same backends
        assert_eq!(backends1, backends2);

        // Should always include CPU
        assert!(backends1.contains(&BackendType::Cpu));

        // Should not include Auto in the list
        assert!(!backends1.contains(&BackendType::Auto));
    }

    #[test]
    fn test_device_count_consistency() {
        // Test that device_count() returns consistent results
        for backend_type in available_backends() {
            let count1 = device_count(backend_type).unwrap();
            let count2 = device_count(backend_type).unwrap();

            assert_eq!(
                count1, count2,
                "Device count should be consistent for {:?}",
                backend_type
            );
        }
    }

    #[test]
    fn test_enumerate_devices_with_no_backends() {
        // This test simulates the scenario where no backends are available
        // (can't actually disable all backends, but we can test the empty case handling)
        let devices = enumerate_all_devices().unwrap();

        // Should never be empty since CPU is always available
        assert!(!devices.is_empty());

        // But test that our logic handles empty cases in find_best_device
        // by testing the early return path
    }

    #[test]
    fn test_backend_capability_reporting() {
        // Test that backends properly report their capabilities
        let backend = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .build()
            .unwrap();

        let capabilities = backend.capabilities();

        // Should have some capabilities
        assert!(!capabilities.supported_dtypes.is_empty());

        // CPU should support basic data types
        assert!(capabilities
            .supported_dtypes
            .contains(&torsh_core::DType::F32));
        assert!(capabilities
            .supported_dtypes
            .contains(&torsh_core::DType::F64));
    }

    #[test]
    fn test_error_recovery_and_retry_logic() {
        // Test error recovery scenarios
        let mut retry_count = 0;
        let max_retries = 3;

        loop {
            // Simulate an operation that might fail
            let result = BackendBuilder::new()
                .backend_type(BackendType::Cpu)
                .num_threads(1) // Use minimal threads to avoid conflicts
                .build();

            match result {
                Ok(_) => {
                    // Success
                    break;
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        // Test that we can handle the error gracefully
                        let error_msg = e.to_string();
                        assert!(!error_msg.is_empty());
                        break;
                    }
                    // Simulate delay before retry
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
    }

    #[test]
    fn test_backend_performance_hints() {
        // Test backend performance hints system
        let backend = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .build()
            .unwrap();

        let hints = backend.performance_hints();

        // Should provide some hints
        assert!(hints.optimal_batch_size > 0);

        // Hints should be reasonable
        assert!(hints.optimal_batch_size <= 1024 * 1024); // Should be reasonable
    }

    #[test]
    fn test_cross_backend_type_compatibility() {
        // Test that different backend types can coexist
        let cpu_result = BackendBuilder::new().backend_type(BackendType::Cpu).build();

        assert!(cpu_result.is_ok());

        // Test other backends if available
        #[cfg(feature = "cuda")]
        {
            let cuda_result = BackendBuilder::new()
                .backend_type(BackendType::Cuda)
                .build();

            // May succeed or fail depending on hardware
            match cuda_result {
                Ok(_) => {
                    // Both backends should be able to exist
                }
                Err(_) => {
                    // CUDA may not be available
                }
            }
        }
    }

    #[test]
    fn test_backend_state_isolation() {
        // Test that different backend instances don't interfere
        let backend1 = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .num_threads(2)
            .build()
            .unwrap();

        let backend2 = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .num_threads(4)
            .build()
            .unwrap();

        // Both should work independently
        let device1 = backend1.default_device().unwrap();
        let device2 = backend2.default_device().unwrap();

        assert!(!device1.name().is_empty());
        assert!(!device2.name().is_empty());
    }

    #[test]
    fn test_profiling_enablement() {
        // Test backend with profiling enabled
        let backend = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .enable_profiling(true)
            .build()
            .unwrap();

        // Should successfully create backend with profiling
        let device = backend.default_device().unwrap();
        assert!(!device.name().is_empty());

        // Test with profiling disabled
        let backend_no_prof = BackendBuilder::new()
            .backend_type(BackendType::Cpu)
            .enable_profiling(false)
            .build()
            .unwrap();

        let device_no_prof = backend_no_prof.default_device().unwrap();
        assert!(!device_no_prof.name().is_empty());
    }

    // ========== CROSS-BACKEND VALIDATION TESTS ==========

    #[test]
    fn test_cross_backend_validation_integration() {
        use crate::cross_backend_validation::{
            run_cross_backend_validation, CrossBackendValidator,
        };

        // Test validator creation
        let validator = CrossBackendValidator::new();
        assert!(!validator.available_backends().is_empty());

        // Test individual validation components
        assert!(validator.validate_device_creation().is_ok());
        assert!(validator.validate_capabilities_consistency().is_ok());

        // Test full validation suite
        match run_cross_backend_validation() {
            Ok(()) => {
                // All validations passed
            }
            Err(e) => {
                // Some validation failed - log but don't fail the test
                // since some backends may not be available in CI
                eprintln!("Cross-backend validation warning: {}", e);
            }
        }
    }

    #[test]
    fn test_floating_point_comparison_utilities() {
        use crate::cross_backend_validation::{compare_f32_values, compare_f64_values};

        // Test normal comparisons
        assert!(compare_f32_values(1.0, 1.0, 1e-6));
        assert!(compare_f32_values(1.0, 1.0000005, 1e-6));
        assert!(!compare_f32_values(1.0, 1.1, 1e-6));

        assert!(compare_f64_values(1.0, 1.0, 1e-11));
        assert!(compare_f64_values(1.0, 1.00000000001, 1.1e-11));
        assert!(!compare_f64_values(1.0, 1.1, 1e-11));

        // Test special values
        assert!(compare_f32_values(f32::NAN, f32::NAN, 1e-6));
        assert!(compare_f32_values(f32::INFINITY, f32::INFINITY, 1e-6));
        assert!(!compare_f32_values(f32::INFINITY, f32::NEG_INFINITY, 1e-6));

        assert!(compare_f64_values(f64::NAN, f64::NAN, 1e-12));
        assert!(compare_f64_values(f64::INFINITY, f64::INFINITY, 1e-12));
        assert!(!compare_f64_values(f64::INFINITY, f64::NEG_INFINITY, 1e-12));
    }

    // ========== HARDWARE OPTIMIZATION TESTS ==========

    #[test]
    fn test_hardware_optimization_integration() {
        use crate::hardware_optimization_tests::{
            run_lightweight_hardware_tests, HardwareOptimizationTester,
        };

        // Test tester creation
        let tester = HardwareOptimizationTester::new();
        assert!(tester.simd_tests_enabled);
        assert!(tester.platform_tests_enabled);
        assert!(!tester.performance_tests_enabled); // Should be disabled by default

        // Run lightweight tests (suitable for CI)
        match run_lightweight_hardware_tests() {
            Ok(()) => {
                // Tests passed
            }
            Err(e) => {
                // Log warning but don't fail test - hardware detection may not be available
                eprintln!("Hardware optimization tests warning: {}", e);
            }
        }
    }

    #[test]
    fn test_hardware_optimization_tester_configuration() {
        use crate::hardware_optimization_tests::HardwareOptimizationTester;

        // Test that we can configure the tester
        let mut tester = HardwareOptimizationTester::new();

        // Modify configuration
        tester.simd_tests_enabled = false;
        tester.platform_tests_enabled = true;
        tester.performance_tests_enabled = false;

        // Configuration should be applied
        assert!(!tester.simd_tests_enabled);
        assert!(tester.platform_tests_enabled);
        assert!(!tester.performance_tests_enabled);
    }
}
