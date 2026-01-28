//! CUDA backend for ToRSh deep learning framework
//!
//! This module provides high-performance GPU acceleration for tensor operations
//! using NVIDIA CUDA and cuDNN. It integrates with the scirs2 ecosystem for
//! optimal performance and compatibility.

// Only compile CUDA modules when CUDA is available
#[cfg(cuda_available)]
pub mod backend;
#[cfg(cuda_available)]
pub mod buffer;
#[cfg(cuda_available)]
pub mod device;
#[cfg(cuda_available)]
pub mod error;
#[cfg(cuda_available)]
pub mod event_coordination;
#[cfg(cuda_available)]
pub mod types_compat;
// CUDA graph API not available in current cuda-sys version - using stub
#[cfg(cuda_available)]
pub mod graph {
    //! Stub module for CUDA graph (not available in current cuda-sys)
    pub use super::graph_stub::*;
}
#[cfg(cuda_available)]
mod graph_stub;
#[cfg(cuda_available)]
pub mod kernels;
#[cfg(cuda_available)]
pub mod memory;
#[cfg(cuda_available)]
pub mod stream;
#[cfg(cuda_available)]
pub mod stream_advanced;
#[cfg(cuda_available)]
pub mod unified_buffer;

#[cfg_attr(not(feature = "cudnn"), allow(unexpected_cfgs))]
#[cfg(all(feature = "cudnn", cuda_available))]
pub mod cudnn;

#[cfg(cuda_available)]
pub mod mixed_precision;
#[cfg(cuda_available)]
pub mod multi_gpu;
#[cfg(cuda_available)]
pub mod neural_ops_enhanced;
#[cfg(cuda_available)]
pub mod tensor_cores;

// CUDA Cooperative Groups support
#[cfg(cuda_available)]
pub mod cooperative_groups;

// Advanced multi-stream execution support
#[cfg(cuda_available)]
pub mod graph_execution;
#[cfg(cuda_available)]
pub mod intelligent_scheduler;
#[cfg(cuda_available)]
pub mod multi_stream_orchestrator;
#[cfg(cuda_available)]
pub mod multi_stream_usage_examples;

// CUDA Occupancy optimization and analysis
#[cfg(cuda_available)]
pub mod occupancy;

// Advanced Performance Optimization Modules
// TODO: temporarily disabled due to optimization module refactoring
// #[cfg(cuda_available)]
// pub mod high_performance_kernels;
// #[cfg(cuda_available)]
// pub mod intelligent_task_scheduler;
// #[cfg(cuda_available)]
// pub mod kernel_fusion_optimizer;
// #[cfg(cuda_available)]
// pub mod performance_optimization_coordinator;

// Fallback modules when CUDA is not available
#[cfg(not(cuda_available))]
pub mod fallback;

// Conditional re-exports based on CUDA availability
#[cfg(cuda_available)]
pub use backend::{CudaBackend, CudaBackendConfig};
#[cfg(cuda_available)]
pub use buffer::CudaBuffer;
#[cfg(cuda_available)]
pub use device::CudaDevice;
#[cfg(cuda_available)]
pub use error::{CudaError, CudaResult};
#[cfg(cuda_available)]
pub use event_coordination::{
    AsyncEventWaiter, CoordinationMetrics, CrossStreamBarrier, EventMetadata, EventPool,
    EventPriority, OperationCoordinator, OperationType,
};
#[cfg(cuda_available)]
pub use memory::{CudaMemoryManager, MemoryAdvice, UnifiedAllocation};
#[cfg(cuda_available)]
pub use stream::{CudaEvent, CudaStream, StreamMetrics, StreamPool, StreamPriority};
#[cfg(cuda_available)]
pub use stream_advanced::{
    AdvancedStreamPool, AllocationStrategy, MultiStreamCoordinator, PoolMetrics, ProfilingReport,
    StreamOrderedAllocator, StreamProfiler, StreamReport, WorkloadType,
};
#[cfg(cuda_available)]
pub use types_compat::{
    cudaError_t, cudaStream_t, cudaSuccess, CUstream, DevicePointer, Event, EventFlags, Stream,
    StreamFlags,
};

// Re-export cuda_sys_compat as a shorthand for cuda_sys compatibility
#[cfg(cuda_available)]
pub use types_compat::cuda_sys_compat;

// Re-export cust_compat for cust:: pattern compatibility
#[cfg(cuda_available)]
pub use types_compat::cust_compat;
#[cfg(cuda_available)]
pub use unified_buffer::UnifiedBuffer;

#[cfg(all(feature = "cudnn", cuda_available))]
pub use cudnn::{
    ActivationDescriptor, ConvolutionDescriptor, CudnnHandle, CudnnOps, FilterDescriptor,
    TensorDescriptor,
};

#[cfg(cuda_available)]
pub use cooperative_groups::{
    CooperationPattern, CooperativeGroupDescriptor, CooperativeGroupType,
    CooperativeGroupsCapabilities, CooperativeGroupsContext, CooperativeGroupsStats,
    CooperativeKernelConfig, CooperativeKernelConfigBuilder, CooperativeWorkload,
    KernelPerformanceMetrics, MemoryScope, SyncFrequency, SynchronizationType,
};
#[cfg(cuda_available)]
pub use mixed_precision::{AmpContext, GradientScaler, MixedPrecisionTrainer};
#[cfg(cuda_available)]
pub use multi_gpu::{DataParallel, MultiGpuContext, ReduceOp};
#[cfg(cuda_available)]
pub use neural_ops_enhanced::EnhancedNeuralOps;
#[cfg(cuda_available)]
pub use tensor_cores::{
    TensorCoreCapability, TensorCoreContext, TensorCoreDType, TensorCoreGemmConfig, TensorCoreOp,
    TensorCoreStats,
};

// Advanced multi-stream execution exports
// CUDA graph API not available in current cuda-sys version
// #[cfg(cuda_available)]
// pub use graph_execution::{
//     CudaGraph, CudaGraphExec, CudaKernelNodeParams, CudaMemcpyNodeParams, CudaMemsetNodeParams,
//     GraphCaptureSession, GraphExecutionManager, GraphExecutionStats, GraphMemoryPool,
//     GraphPerformanceSummary, MemoryPoolStats, PerformanceTrend,
// };
#[cfg(cuda_available)]
pub use intelligent_scheduler::{
    IntelligentStreamScheduler, MemoryAccessPattern, MultiOperationCoordinator, SchedulerMetrics,
    SchedulingDecision, SchedulingStrategy, SynchronizationRequirements, WorkloadCharacteristics,
};
#[cfg(cuda_available)]
pub use multi_stream_orchestrator::{
    ExecutionResult, MultiStreamOrchestrator, OptimizationResult, OrchestratorConfig,
    OrchestratorMetrics, RepeatingWorkloadResult,
};
#[cfg(cuda_available)]
pub use occupancy::{
    CudaDeviceOccupancy, CudaOccupancyAnalyzer, DeviceProperties, LimitingFactor, OccupancyResult,
    OptimizationHeuristics, OptimizedLaunchConfig, PerformanceMetrics, ResourceUsage,
};

// Advanced Performance Optimization Exports
// TODO: temporarily disabled due to optimization module refactoring
// #[cfg(cuda_available)]
// pub use high_performance_kernels::{
//     ActivationType, ConvolutionImplementation, HighPerformanceKernelManager,
//     KernelOptimizationConfig, MatMulImplementation, TensorCoreConfiguration, TensorCorePrecision,
// };
// #[cfg(cuda_available)]
// pub use intelligent_task_scheduler::{
//     DeviceCapability, ExecutionStrategyType as SchedulingStrategyType, IntelligentTaskScheduler,
//     SchedulableTask, SchedulingError, SchedulingStatus, TaskPriority, TaskSubmissionResult,
//     TaskType,
// };
// #[cfg(cuda_available)]
// pub use kernel_fusion_optimizer::{
//     AdvancedKernelFusionOptimizer, FusionKernel, FusionOperation, FusionOptimizationResult,
//     FusionPatternType, FusionStrategyType, KernelFusionStatus, OperationType as FusionOperationType,
// };
// #[cfg(cuda_available)]
// pub use performance_optimization_coordinator::{
//     ComprehensivePerformanceStatus, CoordinationError, CudaOperationRequest, CudaOperationResult,
//     CudaPerformanceOptimizationCoordinator, PerformanceCoordinatorConfig,
//     PerformanceMetrics as CoordinatorPerformanceMetrics,
// };

// Fallback exports when CUDA is not available
#[cfg(not(cuda_available))]
pub use fallback::*;

/// Re-export commonly used types
pub mod prelude {
    #[cfg(cuda_available)]
    pub use super::{
        // Core CUDA types
        AmpContext,
        CooperationPattern,
        CooperativeGroupDescriptor,
        CooperativeGroupType,
        CooperativeGroupsCapabilities,
        // Cooperative Groups types
        CooperativeGroupsContext,
        CooperativeKernelConfig,
        CooperativeKernelConfigBuilder,
        CooperativeWorkload,
        CrossStreamBarrier,
        CudaBackend,
        CudaBuffer,
        CudaDevice,
        CudaError,
        // CudaGraph,  // CUDA graph API not available
        // CudaGraphExec,  // CUDA graph API not available
        CudaMemoryManager,
        // Occupancy optimization types
        CudaOccupancyAnalyzer,
        CudaStream,
        EnhancedNeuralOps,
        EventPool,
        EventPriority,
        GradientScaler,
        // GraphCaptureSession,  // CUDA graph API not available
        // GraphExecutionManager,  // CUDA graph API not available
        IntelligentStreamScheduler,
        KernelPerformanceMetrics,
        LimitingFactor,
        MemoryAdvice,
        MixedPrecisionTrainer,
        MultiOperationCoordinator,
        // Multi-stream execution types
        MultiStreamOrchestrator,
        OccupancyResult,
        OperationCoordinator,
        OperationType,
        OptimizationHeuristics,
        OptimizedLaunchConfig,
        OrchestratorConfig,
        OrchestratorMetrics,
        PerformanceMetrics,
        RepeatingWorkloadResult,
        ResourceUsage,
        SchedulingDecision,
        SchedulingStrategy,
        SynchronizationType,
        TensorCoreCapability,
        TensorCoreContext,
        TensorCoreDType,
        TensorCoreGemmConfig,
        TensorCoreOp,
        TensorCoreStats,
        UnifiedAllocation,
        UnifiedBuffer,
        WorkloadCharacteristics,
    };

    #[cfg(all(feature = "cudnn", cuda_available))]
    pub use super::{
        ActivationDescriptor, ConvolutionDescriptor, CudnnHandle, CudnnOps, FilterDescriptor,
        TensorDescriptor,
    };

    pub use crate::prelude::*;
}

// Conditional compilation based on CUDA availability
#[cfg(cuda_available)]
mod cuda_impl {
    use super::*;
    use crate::cuda::error::CustResultExt;

    /// Initialize CUDA backend
    pub fn init() -> Result<(), CudaError> {
        cust::init(cust::CudaFlags::empty()).cuda_result()?;
        Ok(())
    }

    /// Check if CUDA is available
    pub fn is_available() -> bool {
        // Catch panics/segfaults from CUDA initialization issues
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let result = catch_unwind(AssertUnwindSafe(|| {
            match cust::init(cust::CudaFlags::empty()) {
                Ok(_) => {
                    let mut count: i32 = 0;
                    unsafe {
                        let result = cuda_sys::cudart::cudaGetDeviceCount(&mut count);
                        result == cudaSuccess && count > 0
                    }
                }
                Err(_) => false,
            }
        }));

        result.unwrap_or(false)
    }

    /// Get number of CUDA devices
    pub fn device_count() -> Result<u32, CudaError> {
        let mut count: i32 = 0;
        unsafe {
            let result = cuda_sys::cudart::cudaGetDeviceCount(&mut count);
            if result != cudaSuccess {
                return Err(CudaError::Context {
                    message: "Failed to get device count".to_string(),
                });
            }
        }
        Ok(count as u32)
    }

    /// Get current CUDA device
    pub fn current_device() -> Result<CudaDevice, CudaError> {
        let mut device_id: i32 = 0;
        unsafe {
            let result = cuda_sys::cudart::cudaGetDevice(&mut device_id);
            if result != cudaSuccess {
                return Err(CudaError::Context {
                    message: "Failed to get current device".to_string(),
                });
            }
        }
        CudaDevice::new(device_id as usize)
    }

    /// Set current CUDA device
    pub fn set_device(device_id: usize) -> Result<(), CudaError> {
        // In cust 0.4+, device selection is handled by context creation
        // Verify device is valid
        let _device = cust::device::Device::get_device(device_id as u32).cuda_result()?;
        Ok(())
    }

    /// Synchronize current device
    pub fn synchronize() -> Result<(), CudaError> {
        // Create a temporary stream and synchronize it to sync the device
        let stream =
            cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None).cuda_result()?;
        stream.synchronize().cuda_result()?;
        Ok(())
    }
}

#[cfg(not(cuda_available))]
mod cuda_impl {
    use super::*;

    /// Initialize CUDA backend (fallback - no CUDA available)
    pub fn init() -> Result<(), CudaError> {
        Err(CudaError::RuntimeError(
            "CUDA not available on this system".to_string(),
        ))
    }

    /// Check if CUDA is available (fallback - always false)
    pub fn is_available() -> bool {
        false
    }

    /// Get number of CUDA devices (fallback - no devices)
    pub fn device_count() -> Result<u32, CudaError> {
        Ok(0)
    }

    /// Get current CUDA device (fallback - error)
    pub fn current_device() -> Result<CudaDevice, CudaError> {
        Err(CudaError::RuntimeError(
            "CUDA not available on this system".to_string(),
        ))
    }

    /// Set current CUDA device (fallback - error)
    pub fn set_device(_device_id: usize) -> Result<(), CudaError> {
        Err(CudaError::RuntimeError(
            "CUDA not available on this system".to_string(),
        ))
    }

    /// Synchronize current device (fallback - no-op)
    pub fn synchronize() -> Result<(), CudaError> {
        Ok(()) // No-op when CUDA is not available
    }
}

// Re-export the implementation
pub use cuda_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass if CUDA is available, skip if not
        if is_available() {
            assert!(device_count().unwrap() > 0);
        }
    }

    #[test]
    fn test_cuda_init() {
        if is_available() {
            assert!(init().is_ok());
        }
    }
}
