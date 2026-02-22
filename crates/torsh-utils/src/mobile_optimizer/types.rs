//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Export formats
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// Native ToRSh mobile format
    TorshMobile,
    /// TensorFlow Lite format
    TfLite,
    /// ONNX Runtime Mobile format
    Onnx,
    /// Core ML format
    CoreMl,
}
/// Optimization impact analysis
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    pub model_size_reduction: f32,
    pub latency_improvement: f32,
    pub memory_reduction: f32,
    pub energy_savings: Option<f32>,
    pub accuracy_impact: Option<f32>,
}
/// Backend-specific optimized data
#[derive(Debug)]
pub enum BackendData {
    /// TensorFlow Lite format
    TfLite(Vec<u8>),
    /// ONNX Runtime Mobile format
    OnnxMobile(Vec<u8>),
    /// Core ML format
    CoreMl(Vec<u8>),
    /// Custom format
    Custom(String, Vec<u8>),
}
/// Enhanced benchmark results with detailed metrics
#[derive(Debug, Clone)]
pub struct MobileBenchmarkResults {
    pub basic_results: BenchmarkResults,
    pub detailed_metrics: DetailedMetrics,
    pub platform_metrics: PlatformSpecificMetrics,
    pub optimization_impact: OptimizationImpact,
}
/// Optimized model representation
#[derive(Debug)]
pub struct OptimizedModel {
    /// Model graph representation
    pub graph: ModelGraph,
    /// Optimized weights
    pub weights: HashMap<String, Tensor>,
    /// Metadata about optimizations applied
    pub metadata: OptimizationMetadata,
    /// Backend-specific data
    pub backend_data: Option<BackendData>,
}
/// Metadata about optimizations applied
#[derive(Debug, Default)]
pub struct OptimizationMetadata {
    /// Original model size in bytes
    pub original_size: usize,
    /// Optimized model size in bytes
    pub optimized_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Applied optimization passes
    pub applied_passes: Vec<String>,
    /// Estimated speedup
    pub estimated_speedup: f32,
    /// Backend-specific metadata
    pub backend_metadata: HashMap<String, String>,
}
/// Size optimization configuration
#[derive(Debug, Clone, Default)]
pub struct SizeOptimizationConfig {
    /// Enable model pruning
    pub pruning: bool,
    /// Pruning sparsity ratio (0.0 to 1.0)
    pub pruning_sparsity: f32,
    /// Enable weight sharing/clustering
    pub weight_sharing: bool,
    /// Number of weight clusters
    pub weight_clusters: usize,
    /// Enable layer compression
    pub layer_compression: bool,
    /// Compression ratio target
    pub compression_ratio: f32,
    /// Enable knowledge distillation
    pub knowledge_distillation: bool,
    /// Teacher model path (for distillation)
    pub teacher_model_path: Option<String>,
}
/// Platform-specific benchmark information
#[derive(Debug, Clone)]
pub struct PlatformBenchmarkInfo {
    pub platform: MobilePlatform,
    pub device_model: String,
    pub os_version: String,
    pub cpu_info: CpuInfo,
    pub memory_info: MemoryInfo,
    pub thermal_design_power: Option<f32>,
}
/// Mobile backend target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobileBackend {
    /// CPU backend (default)
    Cpu,
    /// GPU backend (mobile GPU)
    Gpu,
    /// DSP backend (Hexagon, etc.)
    Dsp,
    /// NPU backend (Neural Processing Unit)
    Npu,
}
/// Advanced quantization strategies
#[derive(Debug, Clone)]
pub enum QuantizationStrategy {
    /// Static INT8 quantization (default)
    StaticInt8,
    /// Dynamic INT8 quantization
    DynamicInt8,
    /// Static INT4 quantization
    StaticInt4,
    /// Mixed precision quantization
    MixedPrecision {
        /// Layers to keep in FP16
        fp16_layers: Vec<String>,
        /// Layers to quantize to INT8
        int8_layers: Vec<String>,
        /// Layers to quantize to INT4
        int4_layers: Vec<String>,
    },
    /// QAT (Quantization Aware Training) style
    QAT {
        /// Calibration dataset size
        calibration_size: usize,
        /// Use symmetric quantization
        symmetric: bool,
    },
}
/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_mb: usize,
    pub bandwidth_gb_s: f32,
    pub memory_type: String,
}
/// Detailed performance metrics
#[derive(Debug, Clone)]
pub struct DetailedMetrics {
    pub latency_std_ms: f32,
    pub latency_p95_ms: f32,
    pub latency_p99_ms: f32,
    pub throughput_fps: f32,
    pub energy_efficiency: Option<f32>,
    pub thermal_state: ThermalState,
    pub cpu_utilization: f32,
    pub memory_bandwidth: f32,
    pub cache_hit_rate: f32,
}
/// CPU information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub cores_performance: usize,
    pub cores_efficiency: usize,
    pub max_frequency_ghz: f32,
    pub cache_l1_kb: usize,
    pub cache_l2_kb: usize,
    pub cache_l3_kb: Option<usize>,
}
/// Graph node representing an operation
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier
    pub id: String,
    /// Operation type
    pub op_type: OpType,
    /// Attributes
    pub attributes: HashMap<String, String>,
    /// Associated weights (if any)
    pub weights: Option<String>,
}
/// Operation types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpType {
    Conv2d,
    Linear,
    BatchNorm,
    ReLU,
    MaxPool,
    AvgPool,
    Add,
    Concat,
    Reshape,
    Transpose,
    Softmax,
    ConvBnReLU,
    LinearReLU,
    QuantizedConv2d,
    QuantizedLinear,
    Custom(String),
}
/// Android NNAPI accelerators
#[derive(Debug, Clone)]
pub enum NNAPIAccelerator {
    CPU,
    GPU,
    DSP,
    NPU,
    Custom(String),
}
/// Mobile platform types
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum MobilePlatform {
    iOS { chip: String, neural_engine: bool },
    Android { soc: String, npu_available: bool },
    Other(String),
}
/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Minimum latency in milliseconds
    pub min_latency_ms: f32,
    /// Maximum latency in milliseconds
    pub max_latency_ms: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Power usage in milliwatts (if available)
    pub power_usage_mw: Option<f32>,
}
/// Platform-specific optimization settings
#[derive(Debug, Clone)]
pub enum PlatformOptimization {
    /// No platform-specific optimizations
    None,
    /// iOS Core ML optimizations
    CoreML {
        /// Target iOS version
        ios_version: String,
        /// Enable compute units (CPU/GPU/ANE)
        compute_units: CoreMLComputeUnits,
    },
    /// Android NNAPI optimizations
    NNAPI {
        /// Target Android API level
        api_level: u32,
        /// Enable specific accelerators
        accelerators: Vec<NNAPIAccelerator>,
    },
    /// TensorFlow Lite optimizations
    TFLite {
        /// Use XNNPack delegate
        use_xnnpack: bool,
        /// Use GPU delegate
        use_gpu: bool,
    },
    /// ONNX Runtime Mobile optimizations
    ONNXMobile {
        /// Execution providers
        providers: Vec<String>,
        /// Graph optimization level
        optimization_level: u8,
    },
}
/// Core ML compute units
#[derive(Debug, Clone)]
pub enum CoreMLComputeUnits {
    All,
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
}
/// Model graph for optimization
#[derive(Debug)]
pub struct ModelGraph {
    /// Nodes in the graph
    pub nodes: Vec<GraphNode>,
    /// Edges between nodes
    pub edges: Vec<(usize, usize)>,
    /// Input node indices
    pub inputs: Vec<usize>,
    /// Output node indices
    pub outputs: Vec<usize>,
}
/// Configuration for mobile optimization
#[derive(Debug)]
pub struct MobileOptimizerConfig {
    /// Whether to apply quantization
    pub quantize: bool,
    /// Quantization strategy
    pub quantization_strategy: QuantizationStrategy,
    /// Whether to fuse operations
    pub fuse_ops: bool,
    /// Whether to remove dropout layers
    pub remove_dropout: bool,
    /// Whether to fold batch normalization
    pub fold_bn: bool,
    /// Whether to optimize for inference
    pub optimize_for_inference: bool,
    /// Target backend (cpu, gpu, dsp, npu)
    pub backend: MobileBackend,
    /// Platform-specific optimizations
    pub platform_optimization: PlatformOptimization,
    /// Size optimization configuration
    pub size_optimization: SizeOptimizationConfig,
    /// Preserve specific layers by name
    pub preserve_layers: Vec<String>,
    /// Custom optimization passes
    pub custom_passes: Vec<String>,
}
/// Platform-specific performance metrics
#[derive(Debug, Clone)]
pub struct PlatformSpecificMetrics {
    pub neural_engine_utilization: Option<f32>,
    pub gpu_utilization: Option<f32>,
    pub dsp_utilization: Option<f32>,
    pub memory_compression_ratio: Option<f32>,
    pub bandwidth_efficiency: f32,
}
/// Thermal state of device
#[derive(Debug, Clone)]
pub enum ThermalState {
    Normal,
    Warm,
    Hot,
    Critical,
}
