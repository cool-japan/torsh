//! # ToRSh Utilities
//!
//! `torsh-utils` provides essential utility functions and tools for the ToRSh deep learning framework.
//! This crate contains functionality for development, deployment, profiling, and model management.
//!
//! ## Features
//!
//! ### 🔧 Development Tools
//! - **Benchmarking**: Comprehensive model performance analysis with timing and memory tracking
//! - **Profiling**: Advanced bottleneck detection with flame graphs, memory profiling, and GPU analysis
//! - **Environment Collection**: System and framework information gathering for debugging
//!
//! ### 📊 Monitoring & Visualization
//! - **TensorBoard Integration**: Compatible logging for training metrics, histograms, and graphs
//! - **Interactive Visualizations**: D3.js-based model architecture visualization
//! - **Real-time Monitoring**: Live performance metrics and memory tracking
//!
//! ### 🚀 Deployment Tools
//! - **Mobile Optimization**: Model compression, quantization, and platform-specific optimizations
//! - **C++ Extensions**: Build custom CUDA kernels and operations
//! - **Model Zoo**: Pre-trained model management with HuggingFace Hub integration
//!
//! ## Quick Start
//!
//! Add to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! torsh-utils = { version = "0.1.0", features = ["tensorboard", "collect_env"] }
//! ```
//!
//! ### Basic Usage Examples
//!
//! #### Benchmarking a Model
//!
//! ```rust,no_run
//! use torsh_utils::prelude::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure benchmarking
//! let config = BenchmarkConfig {
//!     warmup_iterations: 10,
//!     benchmark_iterations: 100,
//!     batch_sizes: vec![1, 8, 16, 32],
//!     input_shapes: vec![vec![3, 224, 224]],
//!     profile_memory: true,
//!     ..Default::default()
//! };
//!
//! // Benchmark your model (model would be your actual neural network)
//! // let model = MyModel::new();
//! // let results = benchmark_model(&model, config)?;
//! // print_benchmark_results(&results);
//! # Ok(())
//! # }
//! ```
//!
//! #### Profiling Bottlenecks
//!
//! ```rust,no_run
//! use torsh_utils::prelude::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Profile model performance
//! // let model = MyModel::new();
//! // let report = profile_bottlenecks(
//! //     &model,
//! //     &[1, 3, 224, 224],
//! //     100,
//! //     true  // profile backward pass
//! // )?;
//! //
//! // // Print recommendations
//! // print_bottleneck_report(&report);
//! # Ok(())
//! # }
//! ```
//!
//! #### TensorBoard Logging
//!
//! ```rust,no_run
//! # #[cfg(feature = "tensorboard")]
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use torsh_utils::prelude::*;
//!
//! let mut writer = SummaryWriter::new("runs/experiment1")?;
//!
//! // Log training metrics
//! for epoch in 0..10 {
//!     let loss = 0.5 / (epoch + 1) as f32; // Example loss
//!     writer.add_scalar("loss/train", loss, Some(epoch as i64))?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! #### Mobile Optimization
//!
//! ```rust,no_run
//! use torsh_utils::prelude::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure mobile optimization
//! let config = MobileOptimizerConfig {
//!     quantize: true,
//!     fuse_ops: true,
//!     remove_dropout: true,
//!     fold_bn: true,
//!     backend: MobileBackend::Cpu,
//!     ..Default::default()
//! };
//!
//! // Optimize model for mobile (model would be your actual neural network)
//! // let optimized = optimize_for_mobile(&model, config)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Model Zoo Usage
//!
//! ```rust,no_run
//! use torsh_utils::prelude::*;
//! # use std::path::PathBuf;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create model zoo
//! let mut zoo = ModelZoo::new("~/.torsh/models")?;
//!
//! // Search for models
//! // let models = zoo.search_models("resnet", Some(ModelSearchQuery::new()))?;
//!
//! // Download a model
//! // let model_info = zoo.download_model("resnet50-imagenet", None)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Module Overview
//!
//! - [`benchmark`]: Model performance benchmarking with comprehensive metrics
//! - [`bottleneck`]: Advanced profiling for performance bottleneck detection
//! - [`collect_env`]: System and environment information collection
//! - [`cpp_extension`]: C++ and CUDA extension building utilities
//! - [`mobile_optimizer`]: Mobile deployment optimization tools
//! - [`model_zoo`]: Pre-trained model repository and management
//! - [`tensorboard`]: TensorBoard-compatible logging and visualization
//!
//! ## Feature Flags
//!
//! - `tensorboard` (default): Enable TensorBoard integration
//! - `collect_env` (default): Enable environment information collection
//!
//! ## Architecture
//!
//! This crate is designed to work seamlessly with the ToRSh ecosystem:
//! - Integrates with `torsh-core` for device abstraction
//! - Uses `torsh-tensor` for tensor operations
//! - Compatible with `torsh-nn` for neural network modules
//! - Leverages `torsh-profiler` for performance profiling
//!
//! ## Best Practices
//!
//! 1. **Benchmarking**: Always use warmup iterations to avoid cold start overhead
//! 2. **Profiling**: Profile with representative workloads
//! 3. **Mobile Optimization**: Test on actual target devices
//! 4. **TensorBoard**: Use meaningful tag names for easier analysis
//! 5. **Model Zoo**: Verify checksums for downloaded models
//!
//! ## PyTorch Migration
//!
//! For users migrating from PyTorch:
//! - `SummaryWriter` is compatible with `torch.utils.tensorboard.SummaryWriter`
//! - `benchmark_model` provides similar functionality to `torch.utils.benchmark`
//! - Mobile optimization is similar to `torch.utils.mobile_optimizer`
//!
//! ## Examples
//!
//! See the `examples/` directory for complete working examples:
//! - `benchmark_model.rs`: Complete benchmarking workflow
//! - `profile_bottlenecks.rs`: Performance profiling example
//! - `mobile_optimization.rs`: Model optimization for mobile deployment
//! - `tensorboard_logging.rs`: Training visualization with TensorBoard

pub mod benchmark;
pub mod bottleneck;
pub mod collect_env;
pub mod cpp_extension;
pub mod mobile_optimizer;
pub mod model_zoo;

#[cfg(feature = "tensorboard")]
pub mod tensorboard;

/// Re-export commonly used items for convenience.
///
/// This module provides quick access to the most frequently used types and functions
/// from the torsh-utils crate. Import this module to get started quickly:
///
/// ```rust
/// use torsh_utils::prelude::*;
/// ```
///
/// # Included Items
///
/// ## TensorBoard (with `tensorboard` feature)
/// - [`SummaryWriter`](crate::tensorboard::SummaryWriter): Main TensorBoard writer
/// - [`TensorBoardWriter`](crate::tensorboard::TensorBoardWriter): Enhanced writer with advanced features
///
/// ## Benchmarking
/// - [`benchmark_model`](crate::benchmark::benchmark_model): Benchmark a model
/// - [`BenchmarkConfig`](crate::benchmark::BenchmarkConfig): Benchmarking configuration
/// - [`BenchmarkResult`](crate::benchmark::BenchmarkResult): Benchmark results
///
/// ## Profiling
/// - [`profile_bottlenecks`](crate::bottleneck::profile_bottlenecks): Profile performance bottlenecks
/// - [`BottleneckReport`](crate::bottleneck::BottleneckReport): Profiling report
///
/// ## Environment
/// - [`collect_env`](crate::collect_env::collect_env): Collect environment information
/// - [`EnvironmentInfo`](crate::collect_env::EnvironmentInfo): Environment details
///
/// ## C++ Extensions
/// - [`build_cpp_extension`](crate::cpp_extension::build_cpp_extension): Build C++ extensions
/// - [`CppExtensionConfig`](crate::cpp_extension::CppExtensionConfig): Extension build configuration
///
/// ## Mobile Optimization
/// - [`optimize_for_mobile`](crate::mobile_optimizer::optimize_for_mobile): Optimize model for mobile
/// - [`ExportFormat`](crate::mobile_optimizer::ExportFormat): Mobile export formats
/// - [`MobileBackend`](crate::mobile_optimizer::MobileBackend): Target mobile backend
/// - [`MobileOptimizerConfig`](crate::mobile_optimizer::MobileOptimizerConfig): Optimization configuration
///
/// ## Model Zoo
/// - [`ModelInfo`](crate::model_zoo::ModelInfo): Model metadata
/// - [`ModelZoo`](crate::model_zoo::ModelZoo): Model repository manager
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
