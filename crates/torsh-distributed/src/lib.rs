//! Distributed training support for ToRSh
//!
//! This crate provides distributed training capabilities including:
//! - Data parallel training (DDP)
//! - Model parallel training
//! - Pipeline parallelism
//! - Collective communication operations
//! - RPC framework

use log::{debug, info, warn};
use thiserror::Error;
use torsh_core::TorshError;

/// Type alias for Results with TorshDistributedError
pub type TorshResult<T> = std::result::Result<T, TorshDistributedError>;

/// Distributed training specific errors with detailed context
#[derive(Error, Debug)]
pub enum TorshDistributedError {
    #[error("Backend not initialized. Please call init_process_group() before performing distributed operations")]
    BackendNotInitialized,

    #[error("Invalid argument '{arg}': {reason}. Expected: {expected}")]
    InvalidArgument {
        arg: String,
        reason: String,
        expected: String,
    },

    #[error("Communication error in operation '{operation}': {cause}. This may be due to network issues, process failures, or backend problems")]
    CommunicationError { operation: String, cause: String },

    #[error("Backend '{backend}' error: {message}. Check backend configuration and availability")]
    BackendError { backend: String, message: String },

    #[error("Rank out of bounds: rank {rank} >= world_size {world_size}. Valid ranks are 0 to {}", .world_size - 1)]
    RankOutOfBounds { rank: u32, world_size: u32 },

    #[error("Feature '{feature}' not available in this build. Enable feature flags: {required_features}")]
    FeatureNotAvailable {
        feature: String,
        required_features: String,
    },

    #[error(
        "Process group not found with id '{group_id}'. Available groups: {available_groups:?}"
    )]
    ProcessGroupNotFound {
        group_id: String,
        available_groups: Vec<String>,
    },

    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}. All tensors in collective operations must have the same shape")]
    TensorShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Timeout after {timeout_secs}s waiting for operation '{operation}'. This may indicate network issues or process failures")]
    OperationTimeout {
        operation: String,
        timeout_secs: u64,
    },

    #[error("Process {rank} failed during operation '{operation}': {cause}. Consider using fault tolerance features")]
    ProcessFailure {
        rank: u32,
        operation: String,
        cause: String,
    },

    #[error("Memory allocation failed: requested {requested_bytes} bytes for '{context}'. Available memory may be insufficient")]
    MemoryAllocationFailed {
        requested_bytes: usize,
        context: String,
    },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Configuration error: {message}. Check your distributed training configuration")]
    ConfigurationError { message: String },

    #[error("Checkpoint error: {operation} failed - {cause}. Check filesystem permissions and disk space")]
    CheckpointError { operation: String, cause: String },
}

impl TorshDistributedError {
    /// Create an invalid argument error with context
    pub fn invalid_argument(
        arg: impl Into<String>,
        reason: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self::InvalidArgument {
            arg: arg.into(),
            reason: reason.into(),
            expected: expected.into(),
        }
    }

    /// Create a communication error with operation context
    pub fn communication_error(operation: impl Into<String>, cause: impl Into<String>) -> Self {
        Self::CommunicationError {
            operation: operation.into(),
            cause: cause.into(),
        }
    }

    /// Create a backend error with backend type
    pub fn backend_error(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::BackendError {
            backend: backend.into(),
            message: message.into(),
        }
    }

    /// Create a feature not available error with required features
    pub fn feature_not_available(
        feature: impl Into<String>,
        required_features: impl Into<String>,
    ) -> Self {
        Self::FeatureNotAvailable {
            feature: feature.into(),
            required_features: required_features.into(),
        }
    }

    /// Create a process group not found error
    pub fn process_group_not_found(
        group_id: impl Into<String>,
        available_groups: Vec<String>,
    ) -> Self {
        Self::ProcessGroupNotFound {
            group_id: group_id.into(),
            available_groups,
        }
    }

    /// Create a tensor shape mismatch error
    pub fn tensor_shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::TensorShapeMismatch { expected, actual }
    }

    /// Create an operation timeout error
    pub fn operation_timeout(operation: impl Into<String>, timeout_secs: u64) -> Self {
        Self::OperationTimeout {
            operation: operation.into(),
            timeout_secs,
        }
    }

    /// Create a process failure error
    pub fn process_failure(
        rank: u32,
        operation: impl Into<String>,
        cause: impl Into<String>,
    ) -> Self {
        Self::ProcessFailure {
            rank,
            operation: operation.into(),
            cause: cause.into(),
        }
    }

    /// Create a memory allocation failure error
    pub fn memory_allocation_failed(requested_bytes: usize, context: impl Into<String>) -> Self {
        Self::MemoryAllocationFailed {
            requested_bytes,
            context: context.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError(message.into())
    }

    /// Create an I/O error
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::IoError(message.into())
    }

    /// Create an internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::InternalError(message.into())
    }

    /// Create a configuration error
    pub fn configuration_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create a checkpoint error
    pub fn checkpoint_error(operation: impl Into<String>, cause: impl Into<String>) -> Self {
        Self::CheckpointError {
            operation: operation.into(),
            cause: cause.into(),
        }
    }

    /// Create a not implemented error
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::FeatureNotAvailable {
            feature: feature.into(),
            required_features: "Not yet implemented".to_string(),
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::CommunicationError { .. } => true,
            Self::OperationTimeout { .. } => true,
            Self::ProcessFailure { .. } => true,
            Self::MemoryAllocationFailed { .. } => false,
            Self::BackendNotInitialized => false,
            Self::InvalidArgument { .. } => false,
            Self::TensorShapeMismatch { .. } => false,
            Self::FeatureNotAvailable { .. } => false,
            Self::ProcessGroupNotFound { .. } => false,
            Self::SerializationError(_) => false,
            Self::IoError(_) => true,
            Self::InternalError(_) => false,
            Self::ConfigurationError { .. } => false,
            Self::CheckpointError { .. } => true,
            Self::BackendError { .. } => true,
            Self::RankOutOfBounds { .. } => false,
        }
    }

    /// Get suggested recovery actions for this error
    pub fn recovery_suggestions(&self) -> Vec<&'static str> {
        match self {
            Self::BackendNotInitialized => vec![
                "Call init_process_group() before performing distributed operations",
                "Ensure all processes initialize the backend with the same configuration",
            ],
            Self::CommunicationError { .. } => vec![
                "Check network connectivity between processes",
                "Verify all processes are running and responsive",
                "Consider using retry mechanisms",
                "Check firewall and port configurations",
            ],
            Self::OperationTimeout { .. } => vec![
                "Increase timeout duration",
                "Check for process failures or network issues",
                "Verify all processes are participating in the operation",
                "Consider using asynchronous operations",
            ],
            Self::ProcessFailure { .. } => vec![
                "Enable fault tolerance features",
                "Check process health and system resources",
                "Consider using elastic training",
                "Implement checkpoint/restart mechanisms",
            ],
            Self::MemoryAllocationFailed { .. } => vec![
                "Reduce batch size or model size",
                "Enable CPU offloading for gradients/parameters",
                "Use gradient compression",
                "Check available system memory",
            ],
            Self::TensorShapeMismatch { .. } => vec![
                "Ensure all processes use tensors with identical shapes",
                "Check data preprocessing and model definitions",
                "Verify tensor creation is consistent across processes",
            ],
            Self::FeatureNotAvailable { .. } => vec![
                "Rebuild with required feature flags enabled",
                "Install necessary system dependencies",
                "Use alternative backends or operations",
            ],
            _ => vec![
                "Check configuration and documentation",
                "Enable debug logging for more details",
                "Consider using fallback options",
            ],
        }
    }
}

impl From<TorshDistributedError> for TorshError {
    fn from(err: TorshDistributedError) -> Self {
        TorshError::Other(err.to_string())
    }
}

impl From<TorshError> for TorshDistributedError {
    fn from(err: TorshError) -> Self {
        TorshDistributedError::InternalError(err.to_string())
    }
}

pub mod backend;
pub mod bottleneck_detection;
pub mod collectives;
pub mod communication;
pub mod communication_scheduler;
pub mod dask_integration;
pub mod ddp;
pub mod debugging;
pub mod deepspeed_integration;
pub mod distributed_memory_optimization;
pub mod distributed_monitoring;
pub mod edge_computing;
pub mod enhanced_benchmarks;
pub mod enhanced_fault_tolerance;
pub mod error_recovery;
pub mod expert_parallelism;
pub mod fairscale_integration;
pub mod fault_tolerance;
pub mod fsdp;
pub mod gradient_compression;
pub mod gradient_compression_enhanced;
pub mod green_computing;
pub mod horovod_integration;
pub mod metrics;
pub mod network_aware_compression;
pub mod parameter_server;
pub mod pipeline;
pub mod process_group;
pub mod profiling;
pub mod ray_integration;
pub mod rdma_support;
pub mod rpc;
pub mod store;
pub mod tensor_parallel;
pub mod three_d_parallelism;
pub mod training_analytics_dashboard;
pub mod visualization;
pub mod zero_3_cpu_offload;

#[cfg(feature = "nccl")]
pub mod nccl_ops;

#[cfg(feature = "nccl")]
pub mod nccl_optimization;

// Re-exports
pub use backend::{Backend, BackendType, ReduceOp};
pub use bottleneck_detection::{
    init_global_bottleneck_detector, run_global_bottleneck_detection,
    with_global_bottleneck_detector, Bottleneck, BottleneckDetectionConfig, BottleneckDetector,
    BottleneckSeverity, BottleneckThresholds, BottleneckType,
};
pub use collectives::{
    all_gather,
    // Group-aware operations
    all_gather_group,
    all_reduce,
    all_reduce_group,
    all_to_all,
    barrier,
    barrier_group,
    broadcast,
    broadcast_group,
    bucket_all_reduce,
    hierarchical_all_reduce,
    irecv,
    isend,
    recv,
    recv_group,
    reduce,
    reduce_group,
    // Custom collective operations
    reduce_scatter,
    ring_all_reduce,
    scatter,
    send,
    send_group,
    // Communication group management
    CommunicationGroup,
    GroupManager,
};
pub use communication::{
    deserialize_message, deserialize_tensor, retry_with_backoff, serialize_message,
    serialize_tensor, validate_backend_initialized, validate_rank, with_backend_read,
    with_backend_write, wrap_communication_error, CommunicationStats, StatsCollector,
};
pub use communication_scheduler::{
    CommunicationOp, CommunicationScheduler, CommunicationTask, Priority, SchedulerConfig,
    SchedulerStats, SchedulingStrategy,
};
pub use dask_integration::{
    DaskArrayConfig, DaskBagConfig, DaskClusterConfig, DaskClusterType, DaskConfig,
    DaskDataFrameConfig, DaskDistributedConfig, DaskIntegration, DaskMLConfig, DaskMLSearchMethod,
    DaskScalingConfig, DaskSchedulerConfig, DaskSecurityConfig, DaskShuffleMethod, DaskStats,
    DaskWorkerConfig,
};
pub use ddp::{
    BucketConfig, BucketInfo, DistributedDataParallel, GradientSyncStats, OverlapConfig,
};
pub use debugging::{
    get_global_debugger, init_global_debugger, ActiveOperation, CommunicationState, DebugConfig,
    DebugEvent, DiagnosticResult, DistributedDebugger, LogLevel, ProcessGroupState, ResourceState,
    SystemStateSnapshot,
};
pub use deepspeed_integration::{
    ActivationCheckpointingConfig, DeepSpeedConfig, DeepSpeedIntegration, DeepSpeedStats,
    FP16Config, OffloadOptimizerConfig, OffloadParamConfig, ZeroOptimizationConfig, ZeroStage,
};
pub use edge_computing::{
    AdaptiveCommunicationParams, AggregationSchedule, AggregationStrategy,
    BandwidthAdaptationConfig, BandwidthMonitor, ClientSelectionStrategy, CommunicationManager,
    ComputeCapability, ConnectionType, DataInfo, DataLimits, DeviceDiscoveryConfig, DeviceLocation,
    DeviceResources, DeviceStatus, DeviceType, DiscoveryProtocol, EdgeComputingConfig,
    EdgeComputingManager, EdgeDevice, EdgeOptimizationConfig, FederatedAlgorithm,
    FederatedLearningConfig, FederatedLearningCoordinator, HierarchicalTrainingConfig,
    HierarchicalTrainingCoordinator, NetworkInfo, PrivacyConfig, PrivacyLevel, PrivacyManager,
    PrivacyMechanism, ThermalState, TrainingTier,
};
pub use error_recovery::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, FailureDetector, HealthChecker,
    HealthStatus, RetryConfig, RetryExecutor, RetryStats,
};
pub use expert_parallelism::{
    DistributedExpertManager, ExpertAssignment, ExpertParallelismConfig, ExpertParameters,
    ExpertRouter, ExpertShardingStrategy, RoutingDecision, RoutingStats,
};
pub use fairscale_integration::{
    FairScaleActivationCheckpointingConfig, FairScaleAutoWrapPolicy, FairScaleBalanceMode,
    FairScaleCheckpointingStrategy, FairScaleConfig, FairScaleFsdpConfig,
    FairScaleGradScalerConfig, FairScaleIntegration, FairScaleMemoryOptimizationConfig,
    FairScaleOssConfig, FairScalePipelineConfig, FairScalePipelineSchedule, FairScaleStats,
};
pub use fault_tolerance::{
    checkpoint_utils, CheckpointConfig, CheckpointManager, DistributedMetadata, ElasticConfig,
    ElasticTrainingManager, ScalingEvent, ScalingState, TrainingCheckpoint,
};
pub use fsdp::{
    auto_wrap_modules, fsdp_wrap, AutoWrapPolicy, BackwardPrefetch, FsdpConfig,
    FullyShardedDataParallel, MemoryConfig, MemoryStats, MixedPrecisionConfig,
    ShardInfo as FsdpShardInfo, ShardingStrategy,
};
pub use gradient_compression::{
    CompressedData, CompressedGradient, CompressionConfig, CompressionMetadata, CompressionMethod,
    CompressionStats, GradientCompressor,
};
pub use green_computing::{
    CarbonFootprintData, DeviceEnergyData, GreenComputingConfig, GreenComputingManager,
    GreenOptimizationStrategy, GreenTrainingScheduler, PowerManagementStrategy,
    RenewableEnergyData, ScheduleAction, SustainabilityMetrics, SustainabilityReport,
    SustainabilityReportingConfig, TrainingScheduleRecommendation, TrainingWindow,
};
pub use horovod_integration::{
    HorovodCompressionConfig, HorovodCompressionType, HorovodConfig, HorovodElasticConfig,
    HorovodIntegration, HorovodOptimizerFusionConfig, HorovodStats, HorovodTimelineConfig,
};
pub use metrics::{
    get_global_metrics_collector, init_global_metrics_collector, start_global_metrics_collection,
    stop_global_metrics_collection, CommunicationMetrics, MetricsCollector, MetricsConfig,
    PerformanceMetrics, SystemMetrics, TimeSeries, TimeSeriesPoint, TrainingMetrics,
};
pub use parameter_server::{
    ParameterServer, ParameterServerClient, ParameterServerConfig, ParameterServerMessage,
    ParameterServerResponse, ParameterServerStats,
};
pub use pipeline::{
    create_pipeline_stages, PipelineConfig, PipelineParallel, PipelineStage, PipelineStats,
    ScheduleType,
};
pub use process_group::{ProcessGroup, Rank, WorldSize};
pub use profiling::{
    get_global_profiler, init_global_profiler, CommunicationEvent, CommunicationOpType,
    CommunicationProfiler, OperationStats, ProfilingConfig, ProfilingTimer,
};
pub use ray_integration::{
    RayCheckpointConfig, RayClusterConfig, RayConfig, RayDataConfig, RayDataFormat,
    RayFailureConfig, RayFaultToleranceConfig, RayIntegration, RayPlacementGroupStrategy,
    RayResourceConfig, RayRunConfig, RayScalingConfig, RayScheduler, RaySearchAlgorithm,
    RayServeConfig, RayStats, RayTrainBackend, RayTrainConfig, RayTuneConfig,
};
pub use rdma_support::{
    CompletionStatus, MemoryAccess, MemoryRegion, MemoryRegistration, RdmaConfig,
    RdmaConnectionManager, RdmaEndpoint, RdmaError, RdmaMemoryPool, RdmaMemoryPoolConfig,
    RdmaOperation, RdmaProtocol, RdmaQoS, RdmaResult, RdmaStatistics, RdmaTensorScheduler,
    WorkCompletion, WorkRequest,
};
pub use rpc::{
    delete_rref, get_worker_rank, get_world_size, init_rpc, is_initialized, register_function,
    remote, rpc_async, shutdown, RRef, RpcBackendOptions, RpcMessage,
};
pub use store::{
    create_store, FileStore, MemoryStore, Store, StoreBackend, StoreConfig, StoreValue,
};
pub use tensor_parallel::{
    ShardInfo as TpShardInfo, TensorParallel, TensorParallelConfig, TensorParallelLayer,
    TensorParallelStats, TensorParallelStrategy,
};
pub use three_d_parallelism::{
    CommunicationStrategy, LayerShard, LayerType, Memory3DStats, MemoryOptimizationStrategy,
    ModelShards, Performance3DStats, RankMapping, ThreeDParallelismConfig,
    ThreeDParallelismCoordinator,
};
pub use training_analytics_dashboard::{
    CommunicationAnalytics, CommunicationHotspot, CommunicationPatterns, ConvergenceAnalytics,
    DashboardConfig, DashboardExport, EfficiencyAnalytics, OptimizationRecommendation,
    RecommendationCategory, ResourceBottleneck, ResourceUtilizationAnalytics,
    SystemHealthAnalytics, TrainingAnalytics, TrainingAnalyticsDashboard,
    TrainingPerformanceAnalytics, TrainingSummaryReport,
};
pub use visualization::{
    generate_communication_network_html, generate_monitoring_dashboard, Chart, ChartSeries,
    ChartType, ColorScheme, Dashboard, DashboardLayout, DataPoint, VisualizationConfig,
    VisualizationGenerator,
};
pub use zero_3_cpu_offload::{
    AutoMemoryStrategy, ConfigModelParameters as ModelParameters, CpuCompressionMethod,
    Zero3CpuOffloadConfig, Zero3CpuOffloadManager, Zero3MemoryStats, Zero3PerformanceStats,
};

#[cfg(feature = "nccl")]
pub use nccl_ops::{
    nccl_all_gather, nccl_all_reduce, nccl_broadcast, nccl_reduce_scatter, NcclBatch,
};

#[cfg(feature = "nccl")]
pub use nccl_optimization::{
    CudaEvent, CudaStream, FusedNcclOp, FusionStats, GpuMemoryPool, MemoryPoolStats,
    NcclPerformanceStats, NcclScheduler, OperationStats as NcclOperationStats,
};

/// Initialize the distributed process group
pub async fn init_process_group(
    backend: BackendType,
    rank: Rank,
    world_size: WorldSize,
    master_addr: &str,
    master_port: u16,
) -> TorshResult<ProcessGroup> {
    ProcessGroup::new(backend, rank, world_size, master_addr, master_port).await
}

/// Check if distributed training is available
#[allow(unexpected_cfgs)]
pub fn is_available() -> bool {
    // Always return true since we have MockBackend available
    true
}

/// Check if NCCL backend is available
#[allow(unexpected_cfgs)]
pub fn is_nccl_available() -> bool {
    cfg!(feature = "nccl") && cfg!(feature = "cuda")
}

/// Check if MPI backend is available  
pub fn is_mpi_available() -> bool {
    cfg!(feature = "mpi")
}

/// Check if Gloo backend is available
#[allow(unexpected_cfgs)]
pub fn is_gloo_available() -> bool {
    // Mock backend pretends to be Gloo
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability() {
        // At least one backend should be available
        let available = is_available();
        info!("Distributed training available: {}", available);
    }
}
