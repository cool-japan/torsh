//! Core types and traits for the `ToRSh` deep learning framework
//!
//! This crate provides fundamental building blocks used throughout `ToRSh`,
//! including error types, device abstractions, and core traits.

#![cfg_attr(not(feature = "std"), no_std)]
// Clippy allowances for acceptable patterns in torsh-core
#![allow(clippy::result_large_err)]
#![allow(clippy::type_complexity)]
#![allow(clippy::missing_safety_doc)]
// Allow forward-looking cfg conditions for future scirs2-core integration
#![allow(unexpected_cfgs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod alloc_optimizer;
pub mod api_compat;
pub mod backend_detection;
pub mod cache_oblivious; // Cache-oblivious algorithms for shape operations (NEW)
pub mod chunking;
pub mod compression;
pub mod debug_validation; // Debug-only validation checks (NEW)
pub mod device;
/// Distributed tensor metadata management for multi-node training
pub mod distributed; // Distributed tensor metadata management (NEW)
/// Data type representation and operations
pub mod dtype;
pub mod error;
pub mod error_codes; // Standard error codes for interoperability (NEW)
pub mod error_recovery;
pub mod examples;
/// Federated learning metadata management for privacy-preserving distributed learning
pub mod federated;
pub mod ffi;
pub mod gpu_shape_ops; // GPU-accelerated shape operations for very large tensors (NEW)
pub mod hdf5_metadata;
pub mod health;
pub mod ieee754_compliance;
pub mod inspector;
pub mod interop;
pub mod jax_transforms; // JAX-style functional transformations (NEW)
pub mod layout_optimizer; // Automatic memory layout optimization (NEW)
pub mod memory_debug;
pub mod memory_monitor;
pub mod memory_visualization; // Memory allocation visualization tools (NEW)
pub mod mlir_integration; // MLIR compiler infrastructure integration (NEW)
/// Neuromorphic computing data structures for spiking neural networks
pub mod neuromorphic; // Neuromorphic computing data structures (NEW)
pub mod op_trace;
pub mod perf_metrics;
pub mod perf_monitor; // Real-time performance monitoring (NEW)
pub mod perf_regression;
pub mod profiling;
pub mod runtime_config;
pub mod scirs2_bridge;
pub mod shape;
pub mod shape_debug;
pub mod shape_graph; // Graph-based shape inference for optimization (NEW)
pub mod shape_utils; // Shape utility functions for common patterns (NEW)
pub mod shape_validation;
pub mod simd_arm;
/// Sparse tensor representation and operations
pub mod sparse;
pub mod storage;
pub mod symbolic_shape;
pub mod telemetry;
pub mod tensor_expr; // Tensor expression templates for compile-time optimization (NEW)
pub mod tensor_network; // Tensor network representations for quantum computing (NEW)
/// Type-level automatic differentiation for compile-time gradient tracking
pub mod type_level_ad; // Type-level automatic differentiation (NEW)
pub mod type_level_shapes; // Advanced type-level shape verification (NEW)
/// WebGPU compute shader integration for web-based GPU acceleration
pub mod webgpu; // WebGPU compute shader integration (NEW)
pub mod xla_integration; // TensorFlow XLA compiler integration (NEW)

// Re-export commonly used items
#[cfg(feature = "std")]
pub use alloc_optimizer::{
    acquire_shape_buffer, get_allocation_stats, release_shape_buffer, reset_allocation_stats,
    track_allocation, BufferPool, ScopedBuffer,
};
pub use alloc_optimizer::{
    AllocationStats, CowShape, OptimizationRecommendations, StackShape, MAX_STACK_DIMS,
};
pub use api_compat::{
    clear_deprecation_counts, configure_deprecation_warnings, deprecation_warning,
    deprecation_warning_inline, get_all_deprecations, get_deprecation_info, get_deprecation_stats,
    register_deprecation, DeprecationInfo, DeprecationReport, DeprecationSeverity, Version,
    TORSH_VERSION,
};
pub use backend_detection::{
    BackendFeatureDetector, BackendSummary, DeviceInfo, PerformanceTier, RuntimeFeatures,
    WorkloadType,
};
pub use cache_oblivious::{
    CacheObliviousAnalyzer, CacheObliviousLayout, CacheObliviousMatMul, CacheObliviousReshape,
    CacheObliviousTranspose,
};
pub use chunking::{ChunkingRecommendation, ChunkingStrategy, ChunkingUtils, TensorChunkConfig};
pub use compression::{
    BitmapEncoded, CompressionAnalysis, CompressionEncoding, CompressionSelector, DeltaEncoded,
    MagnitudeThresholdCalculator, PruningMetadata, PruningStrategy, RunLengthEncoded,
};
pub use debug_validation::{
    validate_allocation_size, validate_broadcast_compatible, validate_dtype_compatibility,
    validate_dtype_supports_operation, validate_index_bounds, validate_shape_consistency,
    validate_shape_valid, validate_strides,
};
pub use device::{
    AllToAllTopology, CrossDeviceOp, Device, DeviceCapabilities, DeviceGroup, DeviceHandle,
    DeviceTopology, DeviceType, PeerToPeerOps, PhantomCpu, PhantomCuda, PhantomDevice,
    PhantomMetal, PhantomWgpu, RingTopology, TransferCompatible, TreeTopology, TypedDeviceAffinity,
};
pub use distributed::{
    CheckpointMetadata, CollectiveOp as DistributedCollectiveOp, CommBackend,
    CommunicationDescriptor, DeviceGroup as DistributedDeviceGroup,
    DeviceId as DistributedDeviceId, DeviceTopology as DistributedDeviceTopology,
    DistributedTensor, ReduceOp as DistributedReduceOp, Shard, ShardingStrategy,
};
pub use dtype::{
    AutoPromote, Complex32, Complex64, ComplexElement, DType, FloatElement, QInt8, QUInt8,
    TensorElement, TypePromotion,
};
pub use error::{ErrorLocation, Result, TorshError};
pub use error_codes::{ErrorCategory, ErrorCodeMapper, ErrorDetails, StandardErrorCode};
pub use federated::{
    AggregationStrategy, ClientId, ClientSelectionStrategy, ClientSelector, ClientUpdate,
    CompressionTechnique, CoordinatorStatistics, DataDistribution, FairnessMetrics,
    FederatedClient, FederatedCoordinator, PrivacyParameters, TrainingRound,
};
pub use ffi::{TorshDType, TorshDevice, TorshErrorCode, TorshShape};
#[cfg(feature = "std")]
pub use gpu_shape_ops::GpuShapeAccelerator;
pub use gpu_shape_ops::{AcceleratorConfig, AcceleratorStats};
pub use hdf5_metadata::{
    BloscCompressor, BloscShuffle, Hdf5AttributeValue, Hdf5ByteOrder, Hdf5Chunking,
    Hdf5DatasetMetadata, Hdf5Datatype, Hdf5DimensionScale, Hdf5FileMetadata, Hdf5Filter,
    Hdf5GroupMetadata, Hdf5TypeClass,
};
#[cfg(feature = "std")]
pub use health::{health_checker, init_health_checker, HealthChecker};
pub use health::{HealthCheckConfig, HealthCheckResult, HealthReport, HealthStatus};
pub use ieee754_compliance::{
    is_ieee754_compliant, validate_ieee754_operation, ComplianceChecker, Exception, IEEE754Float,
    RoundingMode, SpecialValue,
};
pub use interop::{
    ArrowDataType, ArrowTypeInfo, ConversionUtils, FromExternal, FromExternalZeroCopy, InteropDocs,
    NumpyArrayInfo, OnnxDataType, OnnxTensorInfo, ToExternal, ToExternalZeroCopy,
};
pub use jax_transforms::{
    CacheStats, ComposedTransform, GradTransform, JitTransform, Jittable, Parallelizable,
    PmapTransform, TransformId, TransformMetadata, TransformRegistry, TransformType, Vectorizable,
    VmapTransform,
};
pub use layout_optimizer::{
    AccessPattern, AccessStatistics, AccessTracker, LayoutOptimizer, LayoutRecommendation,
    TransformationCost,
};
pub use memory_debug::{
    detect_memory_leaks, generate_memory_report, get_memory_stats, init_memory_debugger,
    init_memory_debugger_with_config, record_allocation, record_deallocation, AllocationInfo,
    AllocationPattern, DebuggingAllocator, MemoryDebugConfig, MemoryDebugger, MemoryLeak,
    MemoryReport, MemoryStats, SystemDebuggingAllocator,
};
pub use memory_monitor::{
    AllocationStrategy, MemoryMonitorConfig, MemoryPressure, MemoryPressureThresholds,
    SystemMemoryMonitor, SystemMemoryStats,
};
pub use memory_visualization::{AllocationSummary, AllocationTimeline, MemoryMap, SizeHistogram};
pub use mlir_integration::{
    MlirAttributes, MlirBuilder, MlirDialect, MlirModule, MlirOp, MlirOpcode, MlirPass, MlirType,
    MlirValue,
};
pub use neuromorphic::{
    CoreUtilization, EventDrivenSimulation, IzhikevichNeuron, LIFNeuron, NeuromorphicCore,
    RateDecoder, RateEncoder, STDPSynapse, SpikeEncoding, SpikeEvent, SpikeTrain,
};
pub use op_trace::{
    trace_operation, trace_operation_result, OpTracer, OperationTrace, TensorMetadata,
    TraceBuilder, TraceConfig, TraceId, TraceStatistics,
};
pub use perf_metrics::{
    get_metrics_tracker, init_metrics_tracker, AdvancedMetricsConfig, AdvancedMetricsTracker,
    CacheEfficiencyMetrics, MemoryBandwidthMetrics, ParallelEfficiencyMetrics, RegressionDetection,
    SimdUtilizationMetrics,
};
#[cfg(feature = "std")]
pub use perf_regression::BenchmarkRunner;
pub use perf_regression::{
    PerfBaseline, PerfComparison, PerfMeasurement, PerfStatistics, RegressionReport,
    RegressionSeverity, RegressionTracker,
};
pub use profiling::{
    get_profiler, init_profiler, profile_closure, OperationContext, OperationHandle,
    OperationRecord, OperationStats, OperationType, PerformanceBottleneck, PerformanceProfiler,
    ProfilerConfig,
};
pub use runtime_config::{
    ConfigPreset, DebugLevel, MemoryTrackingConfig, MonitoringScope, OperationConfig,
    RuntimeConfig, RuntimeConfigSnapshot, ValidationLevel,
};
#[cfg(feature = "std")]
pub use scirs2_bridge::SharedBufferManager;
pub use scirs2_bridge::{
    ErrorMapper, SciRS2Bridge, TransferMetadata, TransferStrategy, ZeroCopyView,
};
pub use shape::Shape;
pub use shape_graph::{
    InferenceResult, NodeId, ShapeGraph, ShapeInferenceError, ShapeNode, ShapeOp,
};
pub use shape_utils::{
    are_compatible, batch_shape, expand_to_rank, flatten_from, image_shape, matrix_shape, numel,
    permute, scalar_shape, sequence_shape, squeeze, unsqueeze_at, vector_shape,
};
pub use simd_arm::ArmSimdOps;
pub use sparse::{
    CompressionStats, CooIndices, CooStorage, CsrIndices, CsrStorage, SparseFormat, SparseMetadata,
    SparseStorage,
};
pub use storage::{
    allocate_pooled, clear_pooled_memory, deallocate_pooled, pooled_memory_stats, MemoryFormat,
    MemoryPool, PoolStats, SharedStorage, Storage, StorageView,
};
pub use symbolic_shape::{
    DimExpression, ShapeInference, SymbolId, SymbolRegistry, SymbolicDim, SymbolicShape,
};
#[cfg(feature = "std")]
pub use telemetry::{init_telemetry, telemetry, Span, SpanEvent, SpanMetrics, TelemetrySystem};
pub use telemetry::{ErrorCode, LogEvent, LogLevel, TelemetryConfig};
pub use tensor_expr::{
    math::{AbsExpr, MathExpr, SqrExpr},
    AddOp, ArrayExpr, BinaryExpr, DivOp, ExprBuilder, MapExpr, MulOp, NegExpr, ScalarExpr, SubOp,
    TensorExpr,
};
pub use tensor_network::{
    EdgeId, IndexDim, MatrixProductState, NodeId as TensorNodeId, ProjectedEntangledPairState,
    TensorEdge, TensorNetwork, TensorNetworkError, TensorNode,
};
pub use type_level_ad::{
    stop_gradient, ADComputation, ADMode, BackwardOp, BinaryOp, ChainRule, CheckpointedOp,
    ComputeGradReq, ComputeHessian, ComputeJacobian, CustomGradient, ForwardMode, GradOp,
    GradState, Gradient, GradientAccumulator, GradientClipper, Hessian, HigherOrderDiff, Jacobian,
    NoGrad, RequiresGrad, ReverseMode, TypedTensor, UnaryGradOp,
};
pub use type_level_shapes::{
    Assert, AssertShapeEq, Batched, BroadcastCompatible, Concat, Conv2D, Dim, DimList, Flatten,
    ImageBatchNCHW, ImageBatchNHWC, MatMul, Matrix, Pool2D, Reshape, Reverse, Scalar, Squeeze,
    Tensor, Tensor3D, Tensor4D, Transpose2D, Unsqueeze, Vector,
};
pub use webgpu::{
    BindGroupEntry, BindGroupLayout, BufferUsage, ComputePipeline, GPUBuffer, PipelineCache,
    ResourceType, ShaderError, ShaderStage, WGSLShader, WorkgroupOptimizer,
};
pub use xla_integration::{
    AlgebraicSimplificationPass, CommonSubexpressionEliminationPass, ConstantFoldingPass,
    CopyEliminationPass, DeadCodeEliminationPass, HloOpcode, LayoutOptimizationPass,
    MemoryAllocationOptimizationPass, OperationFusionPass, ParallelizationAnalysisPass,
    PassStatistics, XlaBuilder, XlaComputation, XlaConfig, XlaMetadata, XlaNode, XlaNodeId,
    XlaPassManager, XlaTarget,
};

// NOTE: scirs2 meta crate removed - use specific scirs2-* sub-crates instead
// pub use scirs2; // REMOVED: Use specific sub-crates for better compilation performance

// SciRS2 POLICY COMPLIANCE: Unified access to scirs2-core features
// These re-exports provide PyTorch-compatible access to SciRS2 functionality

/// Numerical traits module - USE THIS instead of num-traits
///
/// Provides unified access to:
/// - Float, Zero, One, NumCast, ToPrimitive, FromPrimitive
/// - Integer traits, bounds checking, etc.
///
/// # SciRS2 POLICY
/// ✅ Use `torsh_core::numeric::*` for numerical operations
/// ❌ DO NOT use `num_traits` directly
pub mod numeric {
    #[cfg(feature = "std")]
    pub use scirs2_core::numeric::*;
}

/// Random number generation module - USE THIS instead of rand
///
/// Provides unified access to:
/// - RNG: thread_rng, seeded_rng, CoreRandom
/// - Distributions: Normal, Uniform, Beta, StudentT, Cauchy, etc.
///
/// # SciRS2 POLICY
/// ✅ Use `torsh_core::random::*` for random operations
/// ❌ DO NOT use `rand` or `rand_distr` directly
pub mod random {
    #[cfg(feature = "std")]
    pub use scirs2_core::random::*;
}

/// Array operations module - USE THIS instead of ndarray
///
/// Provides unified access to:
/// - Array types: Array, Array1, Array2, ArrayD
/// - Array operations: concatenate, stack, broadcast
/// - Macros: array!, arr1!, arr2!, s!
///
/// # SciRS2 POLICY
/// ✅ Use `torsh_core::ndarray::*` for array operations
/// ❌ DO NOT use `ndarray` directly
pub mod ndarray {
    #[cfg(feature = "std")]
    pub use scirs2_core::ndarray::*;
}

/// Parallel operations module - USE THIS instead of rayon
///
/// Provides unified access to:
/// - Parallel iterators with intelligent chunking
/// - CPU topology-aware processing
/// - Work-stealing optimization
///
/// # SciRS2 POLICY
/// ✅ Use `torsh_core::parallel::*` for parallel operations
/// ❌ DO NOT use `rayon` directly
#[cfg(feature = "parallel")]
pub mod parallel {
    pub use scirs2_core::parallel_ops::*;
}

/// SIMD operations module - USE THIS for aligned SIMD operations
///
/// Provides unified access to:
/// - AlignedVec for memory-aligned storage
/// - SIMD-accelerated operations (2-4x speedup)
/// - Automatic fallback for unsupported platforms
///
/// # SciRS2 POLICY
/// ✅ Use `torsh_core::simd::*` for SIMD operations
///
/// # Integration Status (Phase 3: Memory-Aligned SIMD)
/// - ✅ AlignedVec for 32-byte aligned memory (AVX2 optimized)
/// - ✅ SIMD unified operations trait (SimdUnifiedOps)
/// - ✅ Feature detection (AVX2, SSE, NEON)
/// - ✅ Automatic fallback for unsupported platforms
/// - ✅ Cache-optimized operations
/// - ✅ Fused multiply-add (FMA)
/// - ✅ Ultra-optimized variants for common operations
///
/// # Example
/// ```rust,ignore
/// use torsh_core::simd::*;
///
/// // Create aligned vectors for optimal SIMD performance
/// let mut a = AlignedVec::<f32>::with_capacity(1000)?;
/// let mut b = AlignedVec::<f32>::with_capacity(1000)?;
///
/// // Fill with data
/// for i in 0..1000 {
///     a.push(i as f32);
///     b.push((i * 2) as f32);
/// }
///
/// // Perform aligned SIMD addition (2-4x faster than scalar)
/// let result = simd_add_aligned_f32(a.as_slice(), b.as_slice())?;
///
/// // Use unified SIMD operations for arrays
/// use ndarray::Array1;
/// let x = Array1::<f32>::from_vec(vec![1.0, 2.0, 3.0]);
/// let y = Array1::<f32>::from_vec(vec![4.0, 5.0, 6.0]);
/// let sum = f32::simd_add(&x.view(), &y.view());  // Uses SIMD automatically
/// ```
#[cfg(feature = "simd")]
pub mod simd {
    // Re-export all SIMD operations
    pub use scirs2_core::simd_ops::*;

    // Re-export aligned memory operations
    pub use scirs2_core::simd_aligned::{AlignedVec, SIMD_ALIGNMENT};

    /// Convenience prelude for SIMD operations
    pub mod prelude {
        pub use scirs2_core::simd_aligned::{AlignedVec, SIMD_ALIGNMENT};
        pub use scirs2_core::simd_ops::SimdUnifiedOps;
    }
}

/// GPU operations module - USE THIS for GPU acceleration
///
/// Provides unified access to:
/// - Multi-backend GPU support (CUDA/Metal/WebGPU/ROCm/OpenCL)
/// - GPU kernels for neural networks, linear algebra, element-wise ops
/// - 10-100x speedup for large tensors
///
/// # SciRS2 POLICY
/// ✅ Use `torsh_core::gpu::*` for GPU operations
/// ❌ DO NOT use CUDA/Metal/WebGPU APIs directly
///
/// # Integration Status (Phase 2: GPU Kernel Integration)
/// Note: Full GPU support requires scirs2-core to be compiled with GPU features.
/// This is typically available in production builds but may be unavailable
/// in development/testing environments.
///
/// When scirs2-core GPU support is available:
/// - ✅ GPU backend support (CUDA/Metal/WebGPU/ROCm/OpenCL)
/// - ✅ Neural network kernels (GELU, LeakyReLU, Swish, ReLU, Sigmoid, Tanh)
/// - ✅ Element-wise operation kernels
/// - ✅ Linear algebra kernels (GEMV, BatchGEMV, GEMM, AXPY)
/// - ✅ Pooling kernels
/// - ✅ Softmax kernels
/// - ✅ Reduction kernels
/// - ✅ Transform kernels
/// - ✅ Complex number kernels
///
/// # Example
/// ```rust,ignore
/// #[cfg(all(feature = "gpu", feature = "std"))]
/// use torsh_core::gpu::*;
///
/// // Check if GPU support is available
/// if is_gpu_available() {
///     // Use GPU-accelerated operations
///     let device = GpuDevice::new(0)?;
///     // Perform GPU operations...
/// } else {
///     // Fall back to CPU
///     println!("GPU not available, using CPU");
/// }
/// ```
#[cfg(feature = "gpu")]
pub mod gpu {
    //! GPU acceleration through SciRS2
    //!
    //! This module provides a unified interface to GPU operations through scirs2-core.
    //! When scirs2-core is compiled with GPU support, this module re-exports all
    //! GPU functionality. Otherwise, it provides fallback stubs.

    // Attempt to use scirs2-core GPU support if available
    // Note: This will only compile if scirs2-core is built with GPU features
    #[cfg(scirs2_gpu_available)]
    pub use scirs2_core::gpu::*;

    // Fallback implementation when scirs2-core GPU is not available
    #[cfg(not(scirs2_gpu_available))]
    /// Fallback GPU implementations when scirs2-core GPU support is not available
    pub mod fallback {
        #[allow(unused_imports)]
        use super::*;

        /// Check if GPU support is available from scirs2-core
        pub fn is_gpu_available() -> bool {
            // Check if scirs2-core was compiled with GPU support
            // This can be detected at runtime if scirs2-core provides a detection function
            false
        }

        /// Placeholder GPU error type for fallback
        #[derive(Debug, Clone)]
        pub struct GpuError {
            message: String,
        }

        impl GpuError {
            /// Create a new GPU error with a custom message
            pub fn new(message: impl Into<String>) -> Self {
                Self {
                    message: message.into(),
                }
            }

            /// Create a GPU unavailable error
            pub fn unavailable() -> Self {
                Self::new("GPU support not available in this build")
            }
        }

        impl std::fmt::Display for GpuError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "GPU Error: {}", self.message)
            }
        }

        impl std::error::Error for GpuError {}

        /// GPU device stub for fallback
        #[derive(Debug, Clone)]
        pub struct GpuDevice {
            device_id: usize,
        }

        impl GpuDevice {
            /// Attempt to create a new GPU device (always fails in fallback mode)
            pub fn new(_device_id: usize) -> Result<Self, GpuError> {
                Err(GpuError::unavailable())
            }

            /// Get the device ID for this GPU device
            pub fn device_id(&self) -> usize {
                self.device_id
            }
        }
    }

    // Re-export fallback when scirs2-core GPU is not available
    #[cfg(not(scirs2_gpu_available))]
    pub use fallback::*;

    /// Information module for GPU availability
    pub mod info {
        /// Get information about GPU availability and configuration
        pub fn status() -> &'static str {
            #[cfg(scirs2_gpu_available)]
            {
                "GPU support enabled via scirs2-core with multi-backend support (CUDA/Metal/WebGPU/ROCm/OpenCL)"
            }
            #[cfg(not(scirs2_gpu_available))]
            {
                "GPU support requires scirs2-core compiled with 'gpu' feature. \
                 Current build does not have GPU support enabled. \
                 This is expected in development/testing environments. \
                 \
                 To enable GPU support:\n\
                 1. Ensure scirs2-core is compiled with 'gpu' feature\n\
                 2. Add appropriate backend features (cuda, metal, wgpu, rocm, opencl)\n\
                 3. Rebuild with: cargo build --features gpu"
            }
        }

        /// Check if GPU support is currently enabled
        pub fn is_enabled() -> bool {
            cfg!(scirs2_gpu_available)
        }

        /// Get list of available GPU backends (when GPU support is enabled)
        pub fn available_backends() -> &'static [&'static str] {
            #[cfg(scirs2_gpu_available)]
            {
                // This would come from scirs2-core in a real implementation
                &["CUDA", "Metal", "WebGPU"]
            }
            #[cfg(not(scirs2_gpu_available))]
            {
                &[]
            }
        }
    }

    /// Convenience function to check GPU availability
    pub fn is_gpu_available() -> bool {
        info::is_enabled()
    }
}

// Version information
/// ToRSh core version string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// ToRSh core major version number
pub const VERSION_MAJOR: u32 = 0;
/// ToRSh core minor version number
pub const VERSION_MINOR: u32 = 1;
/// ToRSh core patch version number
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
///
/// # SciRS2 POLICY COMPLIANCE
/// This prelude provides PyTorch-compatible types while ensuring
/// SciRS2 POLICY compliance through unified module access.
pub mod prelude {
    // Core ToRSh types
    pub use crate::device::{Device, DeviceCapabilities, DeviceType};
    pub use crate::dtype::{DType, TensorElement};
    pub use crate::error::{Result, TorshError};
    pub use crate::shape::Shape;

    // SciRS2 unified modules for new code
    #[cfg(feature = "std")]
    pub use crate::ndarray;
    #[cfg(feature = "std")]
    pub use crate::numeric;
    #[cfg(feature = "std")]
    pub use crate::random;

    #[cfg(feature = "parallel")]
    pub use crate::parallel;

    #[cfg(feature = "simd")]
    pub use crate::simd;

    // GPU support is optional and may not be available in all builds
    #[cfg(feature = "gpu")]
    pub use crate::gpu;

    #[cfg(feature = "std")]
    pub use crate::profiling;
}
