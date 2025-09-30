//! Utilities for ToRSh framework
//!
//! This crate provides various utilities including:
//! - TensorBoard integration
//! - Environment collection
//! - Model hub functionality
//! - Benchmarking utilities
//! - C++ extension building
//! - Mobile optimization

pub mod benchmark;
pub mod bottleneck;
pub mod collect_env;
pub mod cpp_extension;
pub mod mobile_optimizer;
pub mod model_zoo;

#[cfg(feature = "tensorboard")]
pub mod tensorboard;

/// Re-export commonly used items
pub mod prelude {
    #[cfg(feature = "tensorboard")]
    pub use crate::tensorboard::{SummaryWriter, TensorBoardWriter};

    pub use crate::benchmark::{benchmark_model, BenchmarkConfig, BenchmarkResult};
    pub use crate::bottleneck::{profile_bottlenecks, BottleneckReport};
    pub use crate::collect_env::{collect_env, EnvironmentInfo};
    pub use crate::cpp_extension::{build_cpp_extension, CppExtensionConfig};
    pub use crate::mobile_optimizer::{
        optimize_for_mobile, ExportFormat, MobileBackend, MobileOptimizerConfig,
    };
    pub use crate::model_zoo::{ModelInfo, ModelZoo};
}
