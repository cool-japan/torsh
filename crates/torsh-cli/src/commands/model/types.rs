//! Common types for model operations

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// SciRS2 ecosystem - MUST use instead of rand/ndarray (SCIRS2 POLICY COMPLIANT)

/// Model information structure
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub format: String,
    pub size: String,
    pub parameters: u64,
    pub layers: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub precision: String,
    pub device: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result structure for model operations
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelResult {
    pub operation: String,
    pub input_model: String,
    pub output_model: Option<String>,
    pub success: bool,
    pub duration: String,
    pub size_before: Option<String>,
    pub size_after: Option<String>,
    pub metrics: HashMap<String, serde_json::Value>,
    pub errors: Vec<String>,
}

/// Results from model timing benchmark
#[derive(Debug, Serialize)]
pub struct TimingResult {
    pub throughput_fps: f64,
    pub latency_ms: f64,
    pub memory_mb: f64,
    pub warmup_time_ms: f64,
    pub avg_inference_time_ms: f64,
    pub min_inference_time_ms: f64,
    pub max_inference_time_ms: f64,
    pub std_dev_ms: f64,
    pub device_utilization: Option<f64>,
}

/// Enhanced model structure for real ToRSh models
#[derive(Debug, Clone)]
pub struct TorshModel {
    pub layers: Vec<LayerInfo>,
    pub weights: HashMap<String, TensorInfo>,
    pub metadata: ModelMetadata,
}

/// Layer information with detailed structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameters: u64,
    pub trainable: bool,
    pub config: HashMap<String, serde_json::Value>,
}

/// Tensor information for weights and activations
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub requires_grad: bool,
    pub device: Device,
}

/// Data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    Bool,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::Bool => 1,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

/// Device enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Vulkan,
}

impl Device {
    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{}", id),
            Device::Metal(id) => format!("metal:{}", id),
            Device::Vulkan => "vulkan".to_string(),
        }
    }
}

/// Model metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub format: String,
    pub version: String,
    pub framework: String,
    pub created_at: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            format: "torsh".to_string(),
            version: "0.1.0".to_string(),
            framework: "torsh".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            description: None,
            author: None,
            license: None,
            tags: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

/// Model format detection from file extension
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Torsh,
    PyTorch,
    Onnx,
    TensorFlow,
    TensorFlowLite,
    SafeTensors,
    Unknown,
}

impl ModelFormat {
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|s| s.to_str()) {
            Some("torsh") => ModelFormat::Torsh,
            Some("pth") | Some("pt") => ModelFormat::PyTorch,
            Some("onnx") => ModelFormat::Onnx,
            Some("pb") => ModelFormat::TensorFlow,
            Some("tflite") => ModelFormat::TensorFlowLite,
            Some("safetensors") => ModelFormat::SafeTensors,
            _ => {
                if path.is_dir() {
                    // Check for TensorFlow SavedModel directory
                    if path.join("saved_model.pb").exists() {
                        return ModelFormat::TensorFlow;
                    }
                }
                ModelFormat::Unknown
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ModelFormat::Torsh => "torsh",
            ModelFormat::PyTorch => "pytorch",
            ModelFormat::Onnx => "onnx",
            ModelFormat::TensorFlow => "tensorflow",
            ModelFormat::TensorFlowLite => "tflite",
            ModelFormat::SafeTensors => "safetensors",
            ModelFormat::Unknown => "unknown",
        }
    }
}

/// Precision mode for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionMode {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    Mixed,
}

impl PrecisionMode {
    pub fn target_dtype(&self) -> DType {
        match self {
            PrecisionMode::FP32 => DType::F32,
            PrecisionMode::FP16 => DType::F16,
            PrecisionMode::BF16 => DType::BF16,
            PrecisionMode::INT8 => DType::I8,
            PrecisionMode::INT4 => DType::I8,   // INT4 stored as I8
            PrecisionMode::Mixed => DType::F16, // Default to FP16 for mixed
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            PrecisionMode::FP32 => "fp32",
            PrecisionMode::FP16 => "fp16",
            PrecisionMode::BF16 => "bf16",
            PrecisionMode::INT8 => "int8",
            PrecisionMode::INT4 => "int4",
            PrecisionMode::Mixed => "mixed",
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub mode: QuantizationMode,
    pub precision: PrecisionMode,
    pub per_channel: bool,
    pub symmetric: bool,
    pub calibration_samples: usize,
}

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    Dynamic,
    Static,
    QAT, // Quantization-Aware Training
}

impl QuantizationMode {
    pub fn name(&self) -> &'static str {
        match self {
            QuantizationMode::Dynamic => "dynamic",
            QuantizationMode::Static => "static",
            QuantizationMode::QAT => "qat",
        }
    }
}

/// Format bytes into human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Calculate model memory footprint
pub fn calculate_memory_footprint(model: &TorshModel) -> u64 {
    model
        .weights
        .values()
        .map(|tensor| {
            let elements: usize = tensor.shape.iter().product();
            (elements * tensor.dtype.size_bytes()) as u64
        })
        .sum()
}

/// Calculate FLOPs for common operations
pub fn estimate_flops(layer: &LayerInfo) -> u64 {
    let input_size: u64 = layer.input_shape.iter().map(|&x| x as u64).product();
    let output_size: u64 = layer.output_shape.iter().map(|&x| x as u64).product();

    match layer.layer_type.as_str() {
        "Linear" | "Dense" => {
            // Matrix multiplication: 2 * input * output
            2 * input_size * output_size
        }
        "Conv2d" => {
            // Convolution: 2 * kernel_size * output_size
            let kernel_h = layer
                .config
                .get("kernel_height")
                .and_then(|v| v.as_u64())
                .unwrap_or(3);
            let kernel_w = layer
                .config
                .get("kernel_width")
                .and_then(|v| v.as_u64())
                .unwrap_or(3);
            let channels = layer
                .config
                .get("in_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(3);

            2 * kernel_h * kernel_w * channels * output_size
        }
        "BatchNorm" | "LayerNorm" | "GroupNorm" => {
            // Normalization: ~5 ops per element (mean, var, normalize, scale, shift)
            5 * output_size
        }
        "ReLU" | "GELU" | "Sigmoid" | "Tanh" => {
            // Activation: 1 op per element
            output_size
        }
        "Attention" => {
            // Self-attention: Q@K^T + softmax + @V
            let seq_len = layer.input_shape.get(1).copied().unwrap_or(512) as u64;
            let dim = layer.input_shape.get(2).copied().unwrap_or(768) as u64;

            // QKV projection + attention matrix + output projection
            3 * 2 * seq_len * dim * dim + 2 * seq_len * seq_len * dim
        }
        _ => {
            // Conservative estimate: assume element-wise operation
            output_size
        }
    }
}

/// Comprehensive model statistics
#[derive(Debug, Clone, Serialize)]
pub struct ModelStatistics {
    /// Total number of parameters
    pub total_parameters: u64,
    /// Number of trainable parameters
    pub trainable_parameters: u64,
    /// Number of non-trainable parameters
    pub non_trainable_parameters: u64,
    /// Total memory footprint in bytes
    pub memory_footprint_bytes: u64,
    /// Total memory footprint in megabytes
    pub memory_footprint_mb: f64,
    /// Estimated FLOPs for one forward pass
    pub estimated_flops: u64,
    /// Estimated FLOPs in GFLOPs
    pub estimated_gflops: f64,
    /// Number of layers
    pub num_layers: usize,
    /// Number of tensors
    pub num_tensors: usize,
    /// Layer type distribution
    pub layer_distribution: HashMap<String, usize>,
    /// Average parameters per layer
    pub avg_parameters_per_layer: f64,
    /// Largest layer by parameters
    pub largest_layer: Option<String>,
    /// Smallest layer by parameters
    pub smallest_layer: Option<String>,
}

/// Calculate comprehensive model statistics
pub fn calculate_model_statistics(model: &TorshModel) -> ModelStatistics {
    let total_parameters: u64 = model.layers.iter().map(|l| l.parameters).sum();
    let trainable_parameters: u64 = model
        .layers
        .iter()
        .filter(|l| l.trainable)
        .map(|l| l.parameters)
        .sum();
    let non_trainable_parameters = total_parameters - trainable_parameters;

    let memory_footprint_bytes = calculate_memory_footprint(model);
    let memory_footprint_mb = memory_footprint_bytes as f64 / (1024.0 * 1024.0);

    let estimated_flops: u64 = model.layers.iter().map(|l| estimate_flops(l)).sum();
    let estimated_gflops = estimated_flops as f64 / 1_000_000_000.0;

    let num_layers = model.layers.len();
    let num_tensors = model.weights.len();

    let layer_distribution: HashMap<String, usize> =
        model.layers.iter().fold(HashMap::new(), |mut acc, layer| {
            *acc.entry(layer.layer_type.clone()).or_insert(0) += 1;
            acc
        });

    let avg_parameters_per_layer = if num_layers > 0 {
        total_parameters as f64 / num_layers as f64
    } else {
        0.0
    };

    // Find largest and smallest layers
    let (largest_layer, smallest_layer) = model.layers.iter().fold(
        (None, None),
        |(largest, smallest): (Option<&LayerInfo>, Option<&LayerInfo>), layer| {
            let new_largest = match largest {
                None => Some(layer),
                Some(l) if layer.parameters > l.parameters => Some(layer),
                _ => largest,
            };

            let new_smallest = match smallest {
                None => Some(layer),
                Some(l) if layer.parameters < l.parameters && layer.parameters > 0 => Some(layer),
                _ => smallest,
            };

            (new_largest, new_smallest)
        },
    );

    ModelStatistics {
        total_parameters,
        trainable_parameters,
        non_trainable_parameters,
        memory_footprint_bytes,
        memory_footprint_mb,
        estimated_flops,
        estimated_gflops,
        num_layers,
        num_tensors,
        layer_distribution,
        avg_parameters_per_layer,
        largest_layer: largest_layer.map(|l| l.name.clone()),
        smallest_layer: smallest_layer.map(|l| l.name.clone()),
    }
}

/// Memory breakdown by component
#[derive(Debug, Clone, Serialize)]
pub struct MemoryBreakdown {
    /// Memory used by model parameters
    pub parameters_mb: f64,
    /// Estimated memory for activations
    pub activations_mb: f64,
    /// Estimated memory for gradients
    pub gradients_mb: f64,
    /// Estimated memory for optimizer state
    pub optimizer_state_mb: f64,
    /// Total estimated memory
    pub total_mb: f64,
}

/// Calculate detailed memory breakdown for training
pub fn calculate_memory_breakdown(model: &TorshModel, batch_size: usize) -> MemoryBreakdown {
    let param_memory_bytes = calculate_memory_footprint(model);
    let parameters_mb = param_memory_bytes as f64 / (1024.0 * 1024.0);

    // Estimate activation memory based on layer outputs
    let activation_memory_bytes: u64 = model
        .layers
        .iter()
        .map(|layer| {
            let output_elements: u64 = layer.output_shape.iter().map(|&x| x as u64).product();
            output_elements * 4 * batch_size as u64 // Assuming f32
        })
        .sum();
    let activations_mb = activation_memory_bytes as f64 / (1024.0 * 1024.0);

    // Gradient memory is same as parameter memory for all trainable parameters
    let trainable_param_bytes: u64 = model
        .weights
        .values()
        .filter(|t| t.requires_grad)
        .map(|t| {
            let elements: usize = t.shape.iter().product();
            (elements * t.dtype.size_bytes()) as u64
        })
        .sum();
    let gradients_mb = trainable_param_bytes as f64 / (1024.0 * 1024.0);

    // Optimizer state (Adam requires 2x parameter memory for momentum and variance)
    let optimizer_state_mb = gradients_mb * 2.0;

    let total_mb = parameters_mb + activations_mb + gradients_mb + optimizer_state_mb;

    MemoryBreakdown {
        parameters_mb,
        activations_mb,
        gradients_mb,
        optimizer_state_mb,
        total_mb,
    }
}

/// Performance characteristics for a model
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceCharacteristics {
    /// Estimated inference time in milliseconds (for reference hardware)
    pub estimated_inference_ms: f64,
    /// Estimated throughput in samples per second
    pub estimated_throughput: f64,
    /// Memory bandwidth requirement in GB/s
    pub memory_bandwidth_gbs: f64,
    /// Compute intensity (FLOPs per byte)
    pub compute_intensity: f64,
    /// Model complexity category
    pub complexity_category: String,
}

/// Model architecture visualization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    /// ASCII art tree structure
    Ascii,
    /// Detailed text table
    Table,
    /// JSON structure
    Json,
    /// Graphviz DOT format
    Dot,
    /// Mermaid diagram
    Mermaid,
}

/// Estimate performance characteristics
pub fn estimate_performance_characteristics(
    model: &TorshModel,
    batch_size: usize,
) -> PerformanceCharacteristics {
    let stats = calculate_model_statistics(model);

    // Rough estimate based on typical hardware (e.g., RTX 3090: ~35 TFLOPS, ~936 GB/s)
    let reference_tflops = 35.0;
    let _reference_bandwidth_gbs = 936.0;

    // Estimate inference time based on FLOPs
    let estimated_inference_ms =
        (stats.estimated_gflops * 1000.0) / reference_tflops * batch_size as f64;

    let estimated_throughput = if estimated_inference_ms > 0.0 {
        1000.0 * batch_size as f64 / estimated_inference_ms
    } else {
        0.0
    };

    // Memory bandwidth requirement
    let memory_access_bytes = stats.memory_footprint_bytes as f64 * 2.0; // Read + write
    let memory_bandwidth_gbs =
        (memory_access_bytes / 1_000_000_000.0) / (estimated_inference_ms / 1000.0).max(0.001);

    // Compute intensity (FLOPs per byte transferred)
    let compute_intensity = if stats.memory_footprint_bytes > 0 {
        stats.estimated_flops as f64 / stats.memory_footprint_bytes as f64
    } else {
        0.0
    };

    // Categorize complexity
    let complexity_category = if stats.estimated_gflops < 1.0 {
        "Tiny".to_string()
    } else if stats.estimated_gflops < 10.0 {
        "Small".to_string()
    } else if stats.estimated_gflops < 50.0 {
        "Medium".to_string()
    } else if stats.estimated_gflops < 200.0 {
        "Large".to_string()
    } else {
        "Very Large".to_string()
    };

    PerformanceCharacteristics {
        estimated_inference_ms,
        estimated_throughput,
        memory_bandwidth_gbs,
        compute_intensity,
        complexity_category,
    }
}

/// Generate ASCII art visualization of model architecture
pub fn visualize_model_ascii(model: &TorshModel) -> String {
    let mut output = String::new();

    output.push_str("Model Architecture\n");
    output.push_str("==================\n\n");

    for (i, layer) in model.layers.iter().enumerate() {
        let prefix = if i == 0 {
            "┌─"
        } else if i == model.layers.len() - 1 {
            "└─"
        } else {
            "├─"
        };

        let connector = if i == model.layers.len() - 1 {
            "  "
        } else {
            "│ "
        };

        // Layer header
        output.push_str(&format!(
            "{} Layer {}: {} ({})\n",
            prefix, i, layer.name, layer.layer_type
        ));

        // Input shape
        output.push_str(&format!(
            "{}   Input:  {:?}\n",
            connector, layer.input_shape
        ));

        // Output shape
        output.push_str(&format!(
            "{}   Output: {:?}\n",
            connector, layer.output_shape
        ));

        // Parameters
        if layer.parameters > 0 {
            output.push_str(&format!(
                "{}   Params: {} ({})\n",
                connector,
                layer.parameters,
                format_bytes(layer.parameters * 4) // Assuming f32
            ));
        }

        // Trainable status
        if !layer.trainable {
            output.push_str(&format!("{}   Status: Frozen\n", connector));
        }

        if i < model.layers.len() - 1 {
            output.push_str("│\n");
        }
    }

    // Summary statistics
    let stats = calculate_model_statistics(model);
    output.push_str("\nModel Summary\n");
    output.push_str("=============\n");
    output.push_str(&format!("Total Parameters: {}\n", stats.total_parameters));
    output.push_str(&format!(
        "Trainable Parameters: {}\n",
        stats.trainable_parameters
    ));
    output.push_str(&format!(
        "Memory Footprint: {:.2} MB\n",
        stats.memory_footprint_mb
    ));
    output.push_str(&format!(
        "Estimated FLOPs: {:.2} GFLOPs\n",
        stats.estimated_gflops
    ));

    output
}

/// Generate table visualization of model architecture
pub fn visualize_model_table(model: &TorshModel) -> String {
    let mut output = String::new();

    // Header
    output.push_str("┌─────┬────────────────────┬──────────────┬────────────────┬────────────────┬──────────────┐\n");
    output.push_str("│ ID  │ Layer Name         │ Type         │ Input Shape    │ Output Shape   │ Parameters   │\n");
    output.push_str("├─────┼────────────────────┼──────────────┼────────────────┼────────────────┼──────────────┤\n");

    for (i, layer) in model.layers.iter().enumerate() {
        let name = format!(
            "{:18}",
            if layer.name.len() > 18 {
                format!("{}...", &layer.name[..15])
            } else {
                layer.name.clone()
            }
        );

        let layer_type = format!(
            "{:12}",
            if layer.layer_type.len() > 12 {
                format!("{}...", &layer.layer_type[..9])
            } else {
                layer.layer_type.clone()
            }
        );

        let input_shape = format!("{:14}", format!("{:?}", layer.input_shape));
        let output_shape = format!("{:14}", format!("{:?}", layer.output_shape));
        let params = format!("{:12}", format_number(layer.parameters));

        output.push_str(&format!(
            "│ {:3} │ {} │ {} │ {} │ {} │ {} │\n",
            i, name, layer_type, input_shape, output_shape, params
        ));
    }

    output.push_str("└─────┴────────────────────┴──────────────┴────────────────┴────────────────┴──────────────┘\n");

    output
}

/// Generate Graphviz DOT format visualization
pub fn visualize_model_dot(model: &TorshModel) -> String {
    let mut output = String::new();

    output.push_str("digraph Model {\n");
    output.push_str("    rankdir=TB;\n");
    output.push_str("    node [shape=box, style=rounded];\n\n");

    // Input node
    output.push_str(
        "    input [label=\"Input\", shape=ellipse, fillcolor=lightblue, style=filled];\n\n",
    );

    // Layer nodes
    for (i, layer) in model.layers.iter().enumerate() {
        let label = format!(
            "{}\\n{}\\nParams: {}",
            layer.name,
            layer.layer_type,
            format_number(layer.parameters)
        );

        let color = if layer.trainable {
            "lightgreen"
        } else {
            "lightgray"
        };

        output.push_str(&format!(
            "    layer_{} [label=\"{}\", fillcolor={}, style=filled];\n",
            i, label, color
        ));
    }

    // Output node
    output.push_str(
        "\n    output [label=\"Output\", shape=ellipse, fillcolor=lightcoral, style=filled];\n\n",
    );

    // Edges
    output.push_str("    input -> layer_0;\n");
    for i in 0..model.layers.len() - 1 {
        output.push_str(&format!("    layer_{} -> layer_{};\n", i, i + 1));
    }
    if !model.layers.is_empty() {
        output.push_str(&format!(
            "    layer_{} -> output;\n",
            model.layers.len() - 1
        ));
    }

    output.push_str("}\n");
    output
}

/// Generate Mermaid diagram visualization
pub fn visualize_model_mermaid(model: &TorshModel) -> String {
    let mut output = String::new();

    output.push_str("```mermaid\n");
    output.push_str("graph TD\n");

    // Input node
    output.push_str("    input[Input]\n");

    // Layer nodes
    for (i, layer) in model.layers.iter().enumerate() {
        let label = format!(
            "{}\\n{}\\n{} params",
            layer.name,
            layer.layer_type,
            format_number(layer.parameters)
        );

        output.push_str(&format!("    layer_{}[\"{}\"]", i, label));

        if !layer.trainable {
            output.push_str(":::frozen");
        }

        output.push('\n');
    }

    // Output node
    output.push_str("    output[Output]\n\n");

    // Edges
    output.push_str("    input --> layer_0\n");
    for i in 0..model.layers.len() - 1 {
        output.push_str(&format!("    layer_{} --> layer_{}\n", i, i + 1));
    }
    if !model.layers.is_empty() {
        output.push_str(&format!(
            "    layer_{} --> output\n",
            model.layers.len() - 1
        ));
    }

    // Style definitions
    output.push_str("\n    classDef frozen fill:#e0e0e0,stroke:#999\n");

    output.push_str("```\n");
    output
}

/// Generate JSON visualization of model architecture
pub fn visualize_model_json(model: &TorshModel) -> Result<String, serde_json::Error> {
    let visualization = serde_json::json!({
        "model": {
            "format": model.metadata.format,
            "version": model.metadata.version,
            "description": model.metadata.description,
        },
        "layers": model.layers.iter().enumerate().map(|(i, layer)| {
            serde_json::json!({
                "id": i,
                "name": layer.name,
                "type": layer.layer_type,
                "input_shape": layer.input_shape,
                "output_shape": layer.output_shape,
                "parameters": layer.parameters,
                "trainable": layer.trainable,
                "config": layer.config,
            })
        }).collect::<Vec<_>>(),
        "statistics": {
            "total_parameters": model.layers.iter().map(|l| l.parameters).sum::<u64>(),
            "num_layers": model.layers.len(),
            "num_tensors": model.weights.len(),
        }
    });

    serde_json::to_string_pretty(&visualization)
}

/// Format large numbers with K/M/B suffixes
fn format_number(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else if n < 1_000_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    }
}
