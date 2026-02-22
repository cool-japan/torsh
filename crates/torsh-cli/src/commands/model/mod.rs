//! Model operation commands
//!
//! This module provides a comprehensive set of model operations including:
//! - Conversion between different model formats
//! - Optimization for deployment
//! - Quantization for efficiency
//! - Pruning for size reduction
//! - Analysis and inspection
//! - Benchmarking performance
//!
//! The implementation is modularized for better maintainability:
//! - `analysis`: Model inspection and validation
//! - `benchmarking`: Performance testing
//! - `conversion`: Format conversion and utility operations
//! - `optimization`: Model optimization, quantization, and pruning
//! - `types`: Shared data structures
//! - `args`: Command-line argument definitions

use anyhow::Result;
use clap::Subcommand;

use crate::config::Config;

// Import all sub-modules
pub mod analysis;
pub mod args;
pub mod benchmarking;
pub mod conversion;
pub mod enhanced_profiling;
pub mod enhanced_serialization;
pub mod optimization;
pub mod profiling;
pub mod pytorch_parser;
pub mod real_benchmarking;
pub mod serialization;
pub mod tensor_integration;
pub mod types;
pub mod validation;

// Re-export commonly used types for convenience
pub use args::*;

/// Model operation subcommands
#[derive(Subcommand)]
pub enum ModelCommands {
    /// Convert model between different formats
    Convert(ConvertArgs),

    /// Optimize model for deployment
    Optimize(OptimizeArgs),

    /// Quantize model to reduce size and improve performance
    Quantize(QuantizeArgs),

    /// Prune model to remove unnecessary parameters
    Prune(PruneArgs),

    /// Inspect model architecture and properties
    Inspect(InspectArgs),

    /// Validate model functionality and accuracy
    Validate(ValidateArgs),

    /// Benchmark model performance
    Benchmark(BenchmarkArgs),

    /// Compress model using various techniques
    Compress(CompressArgs),

    /// Extract model components (weights, architecture, etc.)
    Extract(ExtractArgs),

    /// Merge multiple models
    Merge(MergeArgs),
}

/// Execute model operation based on subcommand
pub async fn execute(cmd: ModelCommands, config: &Config, output_format: &str) -> Result<()> {
    match cmd {
        ModelCommands::Convert(args) => {
            conversion::convert_model(args, config, output_format).await
        }
        ModelCommands::Optimize(args) => {
            optimization::optimize_model(args, config, output_format).await
        }
        ModelCommands::Quantize(args) => {
            optimization::quantize_model(args, config, output_format).await
        }
        ModelCommands::Prune(args) => optimization::prune_model(args, config, output_format).await,
        ModelCommands::Inspect(args) => analysis::inspect_model(args, config, output_format).await,
        ModelCommands::Validate(args) => {
            analysis::validate_model(args, config, output_format).await
        }
        ModelCommands::Benchmark(args) => {
            benchmarking::benchmark_model(args, config, output_format).await
        }
        ModelCommands::Compress(args) => {
            conversion::compress_model(args, config, output_format).await
        }
        ModelCommands::Extract(args) => {
            conversion::extract_model(args, config, output_format).await
        }
        ModelCommands::Merge(args) => conversion::merge_model(args, config, output_format).await,
    }
}
