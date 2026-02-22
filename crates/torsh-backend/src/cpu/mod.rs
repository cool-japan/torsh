//! CPU backend implementation for ToRSh
//!
//! This module provides high-performance CPU computing backend for ToRSh tensor operations.
//! It leverages multi-threading with SciRS2 parallel operations, SIMD operations, and optimized
//! memory layouts to deliver maximum performance on CPU hardware.
//!
//! # Features
//!
//! - **Multi-threading**: Parallel tensor operations using SciRS2 parallel_ops (2-4x speedup)
//! - **SIMD**: Vectorized operations for supported data types (2-4x speedup)
//! - **Memory optimization**: Cache-friendly memory layouts with intelligent chunking
//! - **BLAS integration**: Optional BLAS backend for linear algebra
//! - **Cross-platform**: Works on all platforms supported by Rust
//!
//! # SciRS2 POLICY Compliance
//! This module follows the SciRS2 POLICY by using scirs2-core abstractions for:
//! - Parallel operations: `scirs2_parallel` module (replaces direct rayon usage)
//! - SIMD operations: `scirs2_core::simd_ops` (replaces direct wide usage)
//! - Numerical traits: `scirs2_core::numeric` (replaces num-traits)

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
pub mod scirs2_chunking; // Phase 4: SciRS2 intelligent chunking integration
pub mod scirs2_integration;
pub mod scirs2_parallel; // Phase 1: SciRS2 parallel operations integration
pub mod scirs2_simd; // Phase 3: SciRS2 memory-aligned SIMD integration
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
pub use rnn::{calculate_weight_buffer_size_lstm, CpuRnnOps};
pub use scirs2_chunking::prelude as scirs2_chunking_prelude; // SciRS2 intelligent chunking
pub use scirs2_integration::{prepare_tensor_data, prepare_tensor_data_mut, SciRS2CpuBackend};
pub use scirs2_parallel::prelude::*; // SciRS2 parallel operations
pub use scirs2_simd::prelude as scirs2_simd_prelude; // SciRS2 SIMD operations
pub use wasm_simd::{WasmSimdOps, WasmSimdPerformanceInfo};

/// Check if CPU backend is available (always true)
pub fn is_available() -> bool {
    true
}
