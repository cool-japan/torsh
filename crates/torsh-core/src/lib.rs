//! Core types and traits for the `ToRSh` deep learning framework
//!
//! This crate provides fundamental building blocks used throughout `ToRSh`,
//! including error types, device abstractions, and core traits.

#![cfg_attr(not(feature = "std"), no_std)]
// Clippy allowances for acceptable patterns in torsh-core
#![allow(clippy::result_large_err)]
#![allow(clippy::type_complexity)]
#![allow(clippy::missing_safety_doc)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod backend_detection;
pub mod device;
pub mod dtype;
pub mod error;
pub mod error_recovery;
pub mod examples;
pub mod ffi;
pub mod inspector;
pub mod interop;
pub mod memory_debug;
pub mod memory_monitor;
pub mod profiling;
pub mod shape;
pub mod shape_debug;
pub mod simd_arm;
pub mod sparse;
pub mod storage;

// Re-export commonly used items
pub use backend_detection::{
    BackendFeatureDetector, BackendSummary, DeviceInfo, PerformanceTier, RuntimeFeatures,
    WorkloadType,
};
pub use device::{Device, DeviceCapabilities, DeviceType};
pub use dtype::{
    AutoPromote, Complex32, Complex64, ComplexElement, DType, FloatElement, QInt8, QUInt8,
    TensorElement, TypePromotion,
};
pub use error::{ErrorLocation, Result, TorshError};
pub use ffi::{TorshDType, TorshDevice, TorshErrorCode, TorshShape};
pub use interop::{
    ArrowDataType, ArrowTypeInfo, ConversionUtils, FromExternal, FromExternalZeroCopy, InteropDocs,
    NumpyArrayInfo, OnnxDataType, OnnxTensorInfo, ToExternal, ToExternalZeroCopy,
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
pub use profiling::{
    get_profiler, init_profiler, profile_closure, OperationContext, OperationHandle,
    OperationRecord, OperationStats, OperationType, PerformanceBottleneck, PerformanceProfiler,
    ProfilerConfig,
};
pub use shape::Shape;
pub use simd_arm::ArmSimdOps;
pub use sparse::{
    CompressionStats, CooIndices, CooStorage, CsrIndices, CsrStorage, SparseFormat, SparseMetadata,
    SparseStorage,
};
pub use storage::{
    allocate_pooled, clear_pooled_memory, deallocate_pooled, pooled_memory_stats, MemoryFormat,
    MemoryPool, PoolStats, SharedStorage, Storage, StorageView,
};

// Re-export scirs2 for use by other crates
pub use scirs2;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::device::{Device, DeviceCapabilities, DeviceType};
    pub use crate::dtype::{DType, TensorElement};
    pub use crate::error::{Result, TorshError};
    pub use crate::shape::Shape;
}
