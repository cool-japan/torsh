//! CPU backend implementation for ToRSh
//!
//! This module provides high-performance CPU computing backend for ToRSh tensor operations.
//! It leverages multi-threading with Rayon, SIMD operations, and optimized memory layouts
//! to deliver maximum performance on CPU hardware.
//!
//! # Features
//!
//! - **Multi-threading**: Parallel tensor operations using Rayon
//! - **SIMD**: Vectorized operations for supported data types
//! - **Memory optimization**: Cache-friendly memory layouts
//! - **BLAS integration**: Optional BLAS backend for linear algebra
//! - **Cross-platform**: Works on all platforms supported by Rust

pub mod advanced_rayon_optimizer;
pub mod autotuning;
pub mod backend;
pub mod buffer;
pub mod convolution;
pub mod device;
pub mod error;
pub mod feature_detection;
pub mod fft;
pub mod kernel;
pub mod memory;
pub mod memory_patterns;
pub mod optimizations;
pub mod optimized_kernels;
pub mod platform_optimization;
pub mod profiler;
pub mod riscv_vector;
pub mod rnn;
pub mod scirs2_integration;
pub mod simd;
pub mod wasm_simd;

// Re-exports
pub use advanced_rayon_optimizer::{
    AdvancedRayonOptimizer, ParallelStrategy, RayonOptimizationConfig, ThreadPriority, WorkloadType,
};
pub use autotuning::{AutoTuner, PerformanceMeasurement, TuningCache, TuningConfig, TuningResult};
pub use backend::CpuBackend;
pub use buffer::CpuBuffer;
pub use convolution::CpuConvolutionOps;
pub use device::CpuDevice;
pub use error::{BackendError, CpuResult};
pub use feature_detection::{
    cpu_arch_info, detected_features, global_detector, has_feature, CpuArch, CpuArchInfo,
    CpuFeature, CpuFeatureDetector, CpuKernelDispatcher, DynamicKernel,
};
pub use fft::{CpuFftExecutor, CpuFftOps};
pub use kernel::{CpuKernel, CpuKernelExecutor};
pub use memory::CpuMemoryManager;
pub use memory_patterns::{
    AccessPattern, AccessPatternOptimizer, CacheAlignedAllocator, CacheAlignedBuffer, CacheLevel,
    LayoutStrategy, LayoutTransformer, NumaPolicy, PrefetchStrategy,
};
pub use optimizations::{
    KernelFusionOptimizer, MemoryOptimizer, OptimizationLevel, OptimizationManager,
    ThreadPoolOptimizer, WorkStealingStats,
};
pub use platform_optimization::{
    ArmMicroarchitecture, CpuFeatures, CpuOptimizer, OptimizationCache, OptimizedOperations,
    PlatformOptimizer, X86Microarchitecture,
};
pub use profiler::CpuProfiler;
pub use riscv_vector::{RiscVVectorOps, RiscVVectorPerformanceInfo};
pub use rnn::{CpuRnnOps, calculate_weight_buffer_size_lstm};
pub use scirs2_integration::{prepare_tensor_data, prepare_tensor_data_mut, SciRS2CpuBackend};
pub use wasm_simd::{WasmSimdOps, WasmSimdPerformanceInfo};

/// Check if CPU backend is available (always true)
pub fn is_available() -> bool {
    true
}
