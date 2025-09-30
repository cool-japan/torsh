//! Command-line argument structures for model operations

use clap::Args;
use std::path::PathBuf;

/// Arguments for model conversion
#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output model file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Target format for conversion
    #[arg(short, long)]
    pub format: String,

    /// Optimization level (0-3)
    #[arg(long, default_value = "1")]
    pub optimization_level: u8,

    /// Preserve model metadata during conversion
    #[arg(long)]
    pub preserve_metadata: bool,

    /// Target device for optimization
    #[arg(long, default_value = "cpu")]
    pub target_device: String,

    /// Enable verbose output
    #[arg(long)]
    pub verbose: bool,
}

/// Arguments for model optimization
#[derive(Debug, Args)]
pub struct OptimizeArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output optimized model file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Optimization level (0-3)
    #[arg(long, default_value = "2")]
    pub level: u8,

    /// Target device for optimization
    #[arg(long, default_value = "cpu")]
    pub target: String,

    /// Enable operator fusion
    #[arg(long)]
    pub fusion: bool,

    /// Enable constant folding
    #[arg(long)]
    pub constant_folding: bool,

    /// Enable dead code elimination
    #[arg(long)]
    pub dead_code_elimination: bool,

    /// Memory optimization passes
    #[arg(long)]
    pub memory_optimization: bool,
}

/// Arguments for model quantization
#[derive(Debug, Args)]
pub struct QuantizeArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output quantized model file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Quantization method (dynamic, static, qat)
    #[arg(short, long, default_value = "dynamic")]
    pub method: String,

    /// Target precision (int8, int4, fp16)
    #[arg(long, default_value = "int8")]
    pub precision: String,

    /// Calibration dataset path for static quantization
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,

    /// Number of calibration samples
    #[arg(long, default_value = "100")]
    pub calibration_samples: usize,

    /// Accuracy threshold for validation
    #[arg(long, default_value = "0.95")]
    pub accuracy_threshold: f64,
}

/// Arguments for model pruning
#[derive(Debug, Args)]
pub struct PruneArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output pruned model file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Pruning method (magnitude, gradient, fisher)
    #[arg(short, long, default_value = "magnitude")]
    pub method: String,

    /// Sparsity ratio (0.0-1.0)
    #[arg(short, long, default_value = "0.5")]
    pub sparsity: f64,

    /// Structured pruning (channels, filters)
    #[arg(long)]
    pub structured: bool,

    /// Fine-tuning epochs after pruning
    #[arg(long, default_value = "10")]
    pub finetune_epochs: usize,

    /// Validation dataset path
    #[arg(long)]
    pub validation_data: Option<PathBuf>,
}

/// Arguments for model inspection
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Show detailed information
    #[arg(long)]
    pub detailed: bool,

    /// Show model statistics
    #[arg(long)]
    pub stats: bool,

    /// Show memory analysis
    #[arg(long)]
    pub memory: bool,

    /// Show computational complexity
    #[arg(long)]
    pub complexity: bool,

    /// Export information to file
    #[arg(long)]
    pub export: Option<PathBuf>,
}

/// Arguments for model validation
#[derive(Debug, Args)]
pub struct ValidateArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Validation dataset directory
    #[arg(short, long)]
    pub dataset: PathBuf,

    /// Number of samples to validate
    #[arg(short, long, default_value = "1000")]
    pub samples: usize,

    /// Target device for validation
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Batch size for validation
    #[arg(long, default_value = "32")]
    pub batch_size: usize,

    /// Accuracy threshold
    #[arg(long, default_value = "0.9")]
    pub accuracy_threshold: f64,
}

/// Arguments for model benchmarking
#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Target device for benchmarking
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Input shape for benchmarking
    #[arg(long, value_delimiter = ',')]
    pub input_shape: Vec<usize>,

    /// Batch sizes to test
    #[arg(long, value_delimiter = ',', default_values = ["1", "4", "8", "16"])]
    pub batch_sizes: Vec<usize>,

    /// Number of warmup iterations
    #[arg(long, default_value = "10")]
    pub warmup: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "100")]
    pub iterations: usize,

    /// Profile memory usage
    #[arg(long)]
    pub profile_memory: bool,

    /// Export results to file
    #[arg(long)]
    pub export: Option<PathBuf>,
}

/// Arguments for model compression
#[derive(Debug, Args)]
pub struct CompressArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output compressed model file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Compression algorithm
    #[arg(short, long, default_value = "gzip")]
    pub algorithm: String,

    /// Compression level (1-9)
    #[arg(long, default_value = "6")]
    pub level: u8,
}

/// Arguments for model component extraction
#[derive(Debug, Args)]
pub struct ExtractArgs {
    /// Input model file path
    #[arg(short, long)]
    pub input: PathBuf,

    /// Component to extract (weights, architecture, metadata)
    #[arg(short, long)]
    pub component: String,

    /// Output file path
    #[arg(short, long)]
    pub output: PathBuf,
}

/// Arguments for model merging
#[derive(Debug, Args)]
pub struct MergeArgs {
    /// Input model file paths
    #[arg(short, long)]
    pub inputs: Vec<PathBuf>,

    /// Output merged model file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Merge strategy (average, concatenate, ensemble)
    #[arg(short, long, default_value = "average")]
    pub strategy: String,

    /// Weights for merging (if using weighted average)
    #[arg(long, value_delimiter = ',')]
    pub weights: Vec<f64>,
}
