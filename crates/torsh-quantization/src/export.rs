//! Export support for quantized models

use crate::{QScheme, QuantConfig, TorshResult};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Export format for quantized models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// ONNX format for cross-platform deployment
    Onnx,
    /// TensorRT format for NVIDIA GPU inference
    TensorRT,
    /// Mobile format for on-device inference
    Mobile,
    /// TensorFlow Lite format
    TFLite,
    /// CoreML format for Apple devices
    CoreML,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub optimize_for_inference: bool,
    pub target_platform: TargetPlatform,
    pub compression_level: CompressionLevel,
    pub metadata: HashMap<String, String>,
}

/// Target platform for deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPlatform {
    /// Generic CPU deployment
    CPU,
    /// NVIDIA GPU deployment
    GPU,
    /// Mobile devices (ARM)
    Mobile,
    /// Edge devices with constraints
    Edge,
    /// Cloud deployment
    Cloud,
}

/// Compression level for export
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// No compression
    None,
    /// Basic compression
    Low,
    /// Moderate compression
    Medium,
    /// High compression with accuracy trade-off
    High,
    /// Maximum compression for extreme constraints
    Extreme,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Onnx,
            optimize_for_inference: true,
            target_platform: TargetPlatform::CPU,
            compression_level: CompressionLevel::Medium,
            metadata: HashMap::new(),
        }
    }
}

impl ExportConfig {
    /// Create a new export config
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set optimization for inference
    pub fn with_inference_optimization(mut self, optimize: bool) -> Self {
        self.optimize_for_inference = optimize;
        self
    }

    /// Set target platform
    pub fn with_target_platform(mut self, platform: TargetPlatform) -> Self {
        self.target_platform = platform;
        self
    }

    /// Set compression level
    pub fn with_compression(mut self, level: CompressionLevel) -> Self {
        self.compression_level = level;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Quantized model representation for export
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    /// Model parameters with quantization information
    pub parameters: HashMap<String, QuantizedParameter>,
    /// Model architecture definition
    pub architecture: ModelArchitecture,
    /// Quantization configuration used
    pub quant_config: QuantConfig,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

/// Quantized parameter with scale and zero point
#[derive(Debug, Clone)]
pub struct QuantizedParameter {
    pub data: Tensor,
    pub scale: f32,
    pub zero_point: i32,
    pub scheme: QScheme,
    pub dtype: torsh_core::DType,
}

/// Model architecture definition
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    pub layers: Vec<LayerDefinition>,
    pub connections: Vec<(String, String)>, // (from_layer, to_layer)
    pub input_shapes: HashMap<String, Vec<usize>>,
    pub output_shapes: HashMap<String, Vec<usize>>,
}

/// Layer definition for export
#[derive(Debug, Clone)]
pub struct LayerDefinition {
    pub name: String,
    pub layer_type: String,
    pub parameters: HashMap<String, String>,
    pub quantization_info: Option<LayerQuantizationInfo>,
}

/// Quantization information for a layer
#[derive(Debug, Clone)]
pub struct LayerQuantizationInfo {
    pub input_scale: f32,
    pub input_zero_point: i32,
    pub output_scale: f32,
    pub output_zero_point: i32,
    pub weight_scale: Option<f32>,
    pub weight_zero_point: Option<i32>,
}

/// Model exporter for different formats
pub struct ModelExporter {
    config: ExportConfig,
}

impl ModelExporter {
    /// Create a new model exporter
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export a quantized model to the specified format
    pub fn export_model(
        &self,
        model: &QuantizedModel,
        output_path: &Path,
    ) -> TorshResult<ExportResult> {
        match self.config.format {
            ExportFormat::Onnx => self.export_to_onnx(model, output_path),
            ExportFormat::TensorRT => self.export_to_tensorrt(model, output_path),
            ExportFormat::Mobile => self.export_to_mobile(model, output_path),
            ExportFormat::TFLite => self.export_to_tflite(model, output_path),
            ExportFormat::CoreML => self.export_to_coreml(model, output_path),
        }
    }

    /// Export to ONNX format
    fn export_to_onnx(
        &self,
        model: &QuantizedModel,
        output_path: &Path,
    ) -> TorshResult<ExportResult> {
        let mut onnx_content = String::new();

        // ONNX header
        onnx_content.push_str("# ONNX Model Export\n");
        onnx_content.push_str(&format!(
            "# Quantization Scheme: {:?}\n",
            model.quant_config.scheme
        ));
        onnx_content.push_str(&format!(
            "# Target Platform: {:?}\n",
            self.config.target_platform
        ));
        onnx_content.push('\n');

        // Model graph definition
        onnx_content.push_str("graph {\n");
        onnx_content.push_str("  name: \"quantized_model\"\n");

        // Add inputs
        for (name, shape) in &model.architecture.input_shapes {
            onnx_content.push_str(&format!(
                "  input {{ name: \"{name}\", shape: {shape:?} }}\n"
            ));
        }

        // Add layers
        for layer in &model.architecture.layers {
            onnx_content.push_str(&format!(
                "  node {{ name: \"{}\", op_type: \"{}\" }}\n",
                layer.name, layer.layer_type
            ));

            // Add quantization info if available
            if let Some(quant_info) = &layer.quantization_info {
                onnx_content.push_str(&format!(
                    "    # Quantization: input_scale={}, input_zero_point={}\n",
                    quant_info.input_scale, quant_info.input_zero_point
                ));
                onnx_content.push_str(&format!(
                    "    # Output: output_scale={}, output_zero_point={}\n",
                    quant_info.output_scale, quant_info.output_zero_point
                ));
            }
        }

        // Add outputs
        for (name, shape) in &model.architecture.output_shapes {
            onnx_content.push_str(&format!(
                "  output {{ name: \"{name}\", shape: {shape:?} }}\n"
            ));
        }

        onnx_content.push_str("}\n");

        // Write parameters
        onnx_content.push_str("\n# Parameters\n");
        for (name, param) in &model.parameters {
            onnx_content.push_str(&format!(
                "parameter \"{}\" {{ scale: {}, zero_point: {}, dtype: {:?} }}\n",
                name, param.scale, param.zero_point, param.dtype
            ));
        }

        // Write to file
        let mut file = File::create(output_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to create ONNX file: {e}")))?;
        file.write_all(onnx_content.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write ONNX file: {e}")))?;

        Ok(ExportResult {
            format: ExportFormat::Onnx,
            output_path: output_path.to_path_buf(),
            model_size_bytes: onnx_content.len(),
            compression_ratio: self.calculate_compression_ratio(model),
            export_metadata: self.create_export_metadata(model),
        })
    }

    /// Export to TensorRT format
    fn export_to_tensorrt(
        &self,
        model: &QuantizedModel,
        output_path: &Path,
    ) -> TorshResult<ExportResult> {
        let mut trt_content = String::new();

        // TensorRT header
        trt_content.push_str("# TensorRT Model Export\n");
        trt_content.push_str(&format!(
            "# Quantization: {:?}\n",
            model.quant_config.scheme
        ));
        trt_content.push_str(&format!("# Platform: {:?}\n", self.config.target_platform));
        trt_content.push('\n');

        // Engine configuration
        trt_content.push_str("engine_config {\n");
        trt_content.push_str("  precision_mode: \"INT8\"\n");
        trt_content.push_str(&format!(
            "  optimization_level: {:?}\n",
            self.config.compression_level
        ));
        trt_content.push_str("  enable_fp16: false\n");
        trt_content.push_str("  enable_int8: true\n");
        trt_content.push_str("}\n\n");

        // Network definition
        trt_content.push_str("network {\n");

        // Add layers with quantization calibration
        for layer in &model.architecture.layers {
            trt_content.push_str(&format!("  layer \"{}\" {{\n", layer.name));
            trt_content.push_str(&format!("    type: \"{}\"\n", layer.layer_type));

            if let Some(quant_info) = &layer.quantization_info {
                trt_content.push_str("    calibration {\n");
                trt_content.push_str(&format!("      input_scale: {}\n", quant_info.input_scale));
                trt_content.push_str(&format!(
                    "      input_zero_point: {}\n",
                    quant_info.input_zero_point
                ));
                trt_content.push_str(&format!(
                    "      output_scale: {}\n",
                    quant_info.output_scale
                ));
                trt_content.push_str(&format!(
                    "      output_zero_point: {}\n",
                    quant_info.output_zero_point
                ));
                trt_content.push_str("    }\n");
            }

            trt_content.push_str("  }\n");
        }

        trt_content.push_str("}\n");

        // Write to file
        let mut file = File::create(output_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create TensorRT file: {e}"))
        })?;
        file.write_all(trt_content.as_bytes()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write TensorRT file: {e}"))
        })?;

        Ok(ExportResult {
            format: ExportFormat::TensorRT,
            output_path: output_path.to_path_buf(),
            model_size_bytes: trt_content.len(),
            compression_ratio: self.calculate_compression_ratio(model),
            export_metadata: self.create_export_metadata(model),
        })
    }

    /// Export to mobile format
    fn export_to_mobile(
        &self,
        model: &QuantizedModel,
        output_path: &Path,
    ) -> TorshResult<ExportResult> {
        let mut mobile_content = String::new();

        // Mobile format header
        mobile_content.push_str("# Mobile Model Export\n");
        mobile_content.push_str("# Optimized for on-device inference\n");
        mobile_content.push_str(&format!(
            "# Compression: {:?}\n",
            self.config.compression_level
        ));
        mobile_content.push('\n');

        // Mobile-specific optimizations
        mobile_content.push_str("mobile_config {\n");
        mobile_content.push_str("  memory_optimization: true\n");
        mobile_content.push_str("  cpu_optimization: true\n");
        mobile_content.push_str("  battery_optimization: true\n");
        mobile_content.push_str(&format!(
            "  quantization_scheme: \"{:?}\"\n",
            model.quant_config.scheme
        ));
        mobile_content.push_str("}\n\n");

        // Compact layer representation
        mobile_content.push_str("layers [\n");
        for layer in &model.architecture.layers {
            mobile_content.push_str(&format!(
                "  {{ name: \"{}\", type: \"{}\", ",
                layer.name, layer.layer_type
            ));

            if let Some(quant_info) = &layer.quantization_info {
                mobile_content.push_str(&format!(
                    "quant: [scale: {}, zp: {}] ",
                    quant_info.input_scale, quant_info.input_zero_point
                ));
            }

            mobile_content.push_str("},\n");
        }
        mobile_content.push_str("]\n");

        // Compressed parameters
        mobile_content.push_str("\nparameters {\n");
        for (name, param) in &model.parameters {
            mobile_content.push_str(&format!(
                "  \"{}\": {{ s: {}, zp: {}, data: \"compressed\" }},\n",
                name, param.scale, param.zero_point
            ));
        }
        mobile_content.push_str("}\n");

        // Write to file
        let mut file = File::create(output_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create mobile file: {e}"))
        })?;
        file.write_all(mobile_content.as_bytes()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write mobile file: {e}"))
        })?;

        Ok(ExportResult {
            format: ExportFormat::Mobile,
            output_path: output_path.to_path_buf(),
            model_size_bytes: mobile_content.len(),
            compression_ratio: self.calculate_compression_ratio(model),
            export_metadata: self.create_export_metadata(model),
        })
    }

    /// Export to TensorFlow Lite format
    fn export_to_tflite(
        &self,
        model: &QuantizedModel,
        output_path: &Path,
    ) -> TorshResult<ExportResult> {
        let mut tflite_content = String::new();

        // TFLite header
        tflite_content.push_str("# TensorFlow Lite Model Export\n");
        tflite_content.push_str("# Quantized for edge deployment\n");
        tflite_content.push_str(&format!("# Scheme: {:?}\n", model.quant_config.scheme));
        tflite_content.push('\n');

        // TFLite configuration
        tflite_content.push_str("tflite_config {\n");
        tflite_content.push_str("  version: \"2.0\"\n");
        tflite_content.push_str("  quantization: \"FULL_INTEGER\"\n");
        tflite_content.push_str("  delegate: \"CPU\"\n");
        tflite_content.push_str(&format!(
            "  optimization: \"{:?}\"\n",
            self.config.compression_level
        ));
        tflite_content.push_str("}\n\n");

        // Model schema
        tflite_content.push_str("model {\n");
        tflite_content.push_str("  version: 3\n");
        tflite_content.push_str("  description: \"Quantized model for TFLite\"\n");

        // Subgraph definition
        tflite_content.push_str("  subgraphs [\n");
        tflite_content.push_str("    {\n");
        tflite_content.push_str("      tensors [\n");

        // Add tensor definitions
        for layer in model.architecture.layers.iter() {
            tflite_content.push_str(&format!(
                "        {{ name: \"{}\", shape: [1, 224, 224, 3], type: \"INT8\" }},\n",
                layer.name
            ));
        }

        tflite_content.push_str("      ]\n");
        tflite_content.push_str("      operators [\n");

        // Add operator definitions
        for layer in &model.architecture.layers {
            tflite_content.push_str(&format!(
                "        {{ opcode_index: 0, inputs: [0], outputs: [1], builtin_options_type: \"{}\" }},\n",
                layer.layer_type
            ));
        }

        tflite_content.push_str("      ]\n");
        tflite_content.push_str("    }\n");
        tflite_content.push_str("  ]\n");
        tflite_content.push_str("}\n");

        // Quantization metadata
        tflite_content.push_str("\nquantization_metadata {\n");
        for (name, param) in &model.parameters {
            tflite_content.push_str(&format!(
                "  tensor \"{}\" {{ scale: [{}], zero_point: [{}] }}\n",
                name, param.scale, param.zero_point
            ));
        }
        tflite_content.push_str("}\n");

        // Write to file
        let mut file = File::create(output_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create TFLite file: {e}"))
        })?;
        file.write_all(tflite_content.as_bytes()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write TFLite file: {e}"))
        })?;

        Ok(ExportResult {
            format: ExportFormat::TFLite,
            output_path: output_path.to_path_buf(),
            model_size_bytes: tflite_content.len(),
            compression_ratio: self.calculate_compression_ratio(model),
            export_metadata: self.create_export_metadata(model),
        })
    }

    /// Export to CoreML format
    fn export_to_coreml(
        &self,
        model: &QuantizedModel,
        output_path: &Path,
    ) -> TorshResult<ExportResult> {
        let mut coreml_content = String::new();

        // CoreML header
        coreml_content.push_str("# CoreML Model Export\n");
        coreml_content.push_str("# Optimized for Apple devices\n");
        coreml_content.push_str(&format!(
            "# Quantization: {:?}\n",
            model.quant_config.scheme
        ));
        coreml_content.push('\n');

        // CoreML model specification
        coreml_content.push_str("coreml_spec {\n");
        coreml_content.push_str("  version: \"7.0\"\n");
        coreml_content.push_str("  short_description: \"Quantized neural network\"\n");
        coreml_content.push_str("  author: \"ToRSh Quantization\"\n");
        coreml_content.push_str("  license: \"MIT\"\n");
        coreml_content.push_str("}\n\n");

        // Model interface
        coreml_content.push_str("model_interface {\n");

        // Input features
        for (name, shape) in &model.architecture.input_shapes {
            coreml_content.push_str(&format!(
                "  input {{ name: \"{name}\", shape: {shape:?}, type: \"multiArray\" }}\n"
            ));
        }

        // Output features
        for (name, shape) in &model.architecture.output_shapes {
            coreml_content.push_str(&format!(
                "  output {{ name: \"{name}\", shape: {shape:?}, type: \"multiArray\" }}\n"
            ));
        }

        coreml_content.push_str("}\n\n");

        // Neural network
        coreml_content.push_str("neural_network {\n");

        // Add layers
        for layer in &model.architecture.layers {
            coreml_content.push_str(&format!(
                "  layer {{\n    name: \"{}\"\n    type: \"{}\"\n",
                layer.name, layer.layer_type
            ));

            // Add quantization parameters
            if let Some(quant_info) = &layer.quantization_info {
                coreml_content.push_str("    quantization {\n");
                coreml_content.push_str(&format!("      scale: {}\n", quant_info.input_scale));
                coreml_content.push_str(&format!("      bias: {}\n", quant_info.input_zero_point));
                coreml_content.push_str("    }\n");
            }

            coreml_content.push_str("  }\n");
        }

        coreml_content.push_str("}\n");

        // Write to file
        let mut file = File::create(output_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create CoreML file: {e}"))
        })?;
        file.write_all(coreml_content.as_bytes()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write CoreML file: {e}"))
        })?;

        Ok(ExportResult {
            format: ExportFormat::CoreML,
            output_path: output_path.to_path_buf(),
            model_size_bytes: coreml_content.len(),
            compression_ratio: self.calculate_compression_ratio(model),
            export_metadata: self.create_export_metadata(model),
        })
    }

    /// Calculate compression ratio compared to original model
    fn calculate_compression_ratio(&self, model: &QuantizedModel) -> f32 {
        // Estimate original size (assuming FP32)
        let original_size: usize = model
            .parameters
            .values()
            .map(|param| param.data.numel() * 4) // 4 bytes per FP32
            .sum();

        // Estimate quantized size
        let quantized_size: usize = model
            .parameters
            .values()
            .map(|param| {
                match param.dtype {
                    torsh_core::DType::I8 | torsh_core::DType::U8 => param.data.numel(), // 1 byte
                    torsh_core::DType::F16 => param.data.numel() * 2,                    // 2 bytes
                    _ => param.data.numel() * 4,                                         // 4 bytes
                }
            })
            .sum();

        if quantized_size == 0 {
            1.0
        } else {
            original_size as f32 / quantized_size as f32
        }
    }

    /// Create export metadata
    fn create_export_metadata(&self, model: &QuantizedModel) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        metadata.insert(
            "export_format".to_string(),
            format!("{:?}", self.config.format),
        );
        metadata.insert(
            "target_platform".to_string(),
            format!("{:?}", self.config.target_platform),
        );
        metadata.insert(
            "compression_level".to_string(),
            format!("{:?}", self.config.compression_level),
        );
        metadata.insert(
            "quantization_scheme".to_string(),
            format!("{:?}", model.quant_config.scheme),
        );
        metadata.insert(
            "num_parameters".to_string(),
            model.parameters.len().to_string(),
        );
        metadata.insert(
            "num_layers".to_string(),
            model.architecture.layers.len().to_string(),
        );
        metadata.insert(
            "export_timestamp".to_string(),
            format!("{:?}", std::time::SystemTime::now()),
        );

        // Add custom metadata
        for (key, value) in &self.config.metadata {
            metadata.insert(key.clone(), value.clone());
        }

        metadata
    }
}

/// Result of model export operation
#[derive(Debug)]
pub struct ExportResult {
    pub format: ExportFormat,
    pub output_path: std::path::PathBuf,
    pub model_size_bytes: usize,
    pub compression_ratio: f32,
    pub export_metadata: HashMap<String, String>,
}

impl ExportResult {
    /// Get human-readable size
    pub fn get_size_string(&self) -> String {
        let size = self.model_size_bytes as f64;
        if size >= 1_000_000.0 {
            format!("{:.2} MB", size / 1_000_000.0)
        } else if size >= 1_000.0 {
            format!("{:.2} KB", size / 1_000.0)
        } else {
            format!("{size} bytes")
        }
    }

    /// Print export summary
    pub fn print_summary(&self) {
        println!("Export completed successfully!");
        println!("Format: {:?}", self.format);
        println!("Output: {}", self.output_path.display());
        println!("Size: {}", self.get_size_string());
        println!("Compression ratio: {:.2}x", self.compression_ratio);
        println!("Metadata:");
        for (key, value) in &self.export_metadata {
            println!("  {key}: {value}");
        }
    }
}

/// Utility functions for export
pub mod utils {
    use super::*;

    /// Create a sample quantized model for testing
    pub fn create_sample_model() -> TorshResult<QuantizedModel> {
        let mut parameters = HashMap::new();
        let mut metadata = HashMap::new();

        // Add sample parameters
        let sample_tensor = torsh_tensor::creation::tensor_1d(&[1.0, 2.0, 3.0, 4.0])?;
        parameters.insert(
            "layer1.weight".to_string(),
            QuantizedParameter {
                data: sample_tensor,
                scale: 0.1,
                zero_point: 0,
                scheme: QScheme::PerTensorAffine,
                dtype: torsh_core::DType::I8,
            },
        );

        // Create architecture
        let mut input_shapes = HashMap::new();
        let mut output_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 3, 224, 224]);
        output_shapes.insert("output".to_string(), vec![1, 1000]);

        let architecture = ModelArchitecture {
            layers: vec![
                LayerDefinition {
                    name: "conv1".to_string(),
                    layer_type: "conv2d".to_string(),
                    parameters: HashMap::new(),
                    quantization_info: Some(LayerQuantizationInfo {
                        input_scale: 0.1,
                        input_zero_point: 0,
                        output_scale: 0.2,
                        output_zero_point: 0,
                        weight_scale: Some(0.05),
                        weight_zero_point: Some(0),
                    }),
                },
                LayerDefinition {
                    name: "relu1".to_string(),
                    layer_type: "relu".to_string(),
                    parameters: HashMap::new(),
                    quantization_info: None,
                },
            ],
            connections: vec![("conv1".to_string(), "relu1".to_string())],
            input_shapes,
            output_shapes,
        };

        metadata.insert("model_name".to_string(), "sample_model".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        Ok(QuantizedModel {
            parameters,
            architecture,
            quant_config: QuantConfig::int8(),
            metadata,
        })
    }

    /// Validate export configuration
    pub fn validate_export_config(config: &ExportConfig) -> TorshResult<()> {
        match config.format {
            ExportFormat::TensorRT => {
                if config.target_platform != TargetPlatform::GPU {
                    return Err(TorshError::InvalidArgument(
                        "TensorRT export requires GPU target platform".to_string(),
                    ));
                }
            }
            ExportFormat::CoreML => {
                if config.target_platform == TargetPlatform::GPU {
                    return Err(TorshError::InvalidArgument(
                        "CoreML does not support GPU target platform".to_string(),
                    ));
                }
            }
            _ => {} // Other formats are flexible
        }

        Ok(())
    }

    /// Get recommended export format for target platform
    pub fn get_recommended_format(platform: TargetPlatform) -> ExportFormat {
        match platform {
            TargetPlatform::GPU => ExportFormat::TensorRT,
            TargetPlatform::Mobile => ExportFormat::Mobile,
            TargetPlatform::Edge => ExportFormat::TFLite,
            TargetPlatform::CPU | TargetPlatform::Cloud => ExportFormat::Onnx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_export_config() {
        let config = ExportConfig::new(ExportFormat::Onnx)
            .with_target_platform(TargetPlatform::CPU)
            .with_compression(CompressionLevel::High)
            .with_metadata("author".to_string(), "test".to_string());

        assert_eq!(config.format, ExportFormat::Onnx);
        assert_eq!(config.target_platform, TargetPlatform::CPU);
        assert_eq!(config.compression_level, CompressionLevel::High);
        assert_eq!(config.metadata.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_export_to_onnx() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("model.onnx");

        let model = utils::create_sample_model().unwrap();
        let config = ExportConfig::new(ExportFormat::Onnx);
        let exporter = ModelExporter::new(config);

        let result = exporter.export_model(&model, &output_path).unwrap();

        assert_eq!(result.format, ExportFormat::Onnx);
        assert!(result.output_path.exists());
        assert!(result.model_size_bytes > 0);
        assert!(result.compression_ratio > 0.0);
    }

    #[test]
    fn test_export_to_tensorrt() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("model.trt");

        let model = utils::create_sample_model().unwrap();
        let config =
            ExportConfig::new(ExportFormat::TensorRT).with_target_platform(TargetPlatform::GPU);
        let exporter = ModelExporter::new(config);

        let result = exporter.export_model(&model, &output_path).unwrap();

        assert_eq!(result.format, ExportFormat::TensorRT);
        assert!(result.output_path.exists());
        assert!(result.model_size_bytes > 0);
    }

    #[test]
    fn test_export_to_mobile() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("model.mobile");

        let model = utils::create_sample_model().unwrap();
        let config = ExportConfig::new(ExportFormat::Mobile)
            .with_target_platform(TargetPlatform::Mobile)
            .with_compression(CompressionLevel::High);
        let exporter = ModelExporter::new(config);

        let result = exporter.export_model(&model, &output_path).unwrap();

        assert_eq!(result.format, ExportFormat::Mobile);
        assert!(result.output_path.exists());
        assert!(result.model_size_bytes > 0);
    }

    #[test]
    fn test_export_to_tflite() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("model.tflite");

        let model = utils::create_sample_model().unwrap();
        let config =
            ExportConfig::new(ExportFormat::TFLite).with_target_platform(TargetPlatform::Edge);
        let exporter = ModelExporter::new(config);

        let result = exporter.export_model(&model, &output_path).unwrap();

        assert_eq!(result.format, ExportFormat::TFLite);
        assert!(result.output_path.exists());
        assert!(result.model_size_bytes > 0);
    }

    #[test]
    fn test_export_to_coreml() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("model.mlmodel");

        let model = utils::create_sample_model().unwrap();
        let config =
            ExportConfig::new(ExportFormat::CoreML).with_target_platform(TargetPlatform::Mobile);
        let exporter = ModelExporter::new(config);

        let result = exporter.export_model(&model, &output_path).unwrap();

        assert_eq!(result.format, ExportFormat::CoreML);
        assert!(result.output_path.exists());
        assert!(result.model_size_bytes > 0);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let model = utils::create_sample_model().unwrap();
        let config = ExportConfig::new(ExportFormat::Onnx);
        let exporter = ModelExporter::new(config);

        let ratio = exporter.calculate_compression_ratio(&model);
        assert!(ratio > 0.0);
        // Should be around 4x compression for I8 vs F32
        assert!((3.0..=5.0).contains(&ratio));
    }

    #[test]
    fn test_export_result_size_string() {
        let result = ExportResult {
            format: ExportFormat::Onnx,
            output_path: PathBuf::from("test.onnx"),
            model_size_bytes: 1_500_000,
            compression_ratio: 4.0,
            export_metadata: HashMap::new(),
        };

        let size_str = result.get_size_string();
        assert!(size_str.contains("1.50 MB"));
    }

    #[test]
    fn test_validation() {
        // Valid configuration
        let valid_config =
            ExportConfig::new(ExportFormat::TensorRT).with_target_platform(TargetPlatform::GPU);
        assert!(utils::validate_export_config(&valid_config).is_ok());

        // Invalid configuration - TensorRT on CPU
        let invalid_config =
            ExportConfig::new(ExportFormat::TensorRT).with_target_platform(TargetPlatform::CPU);
        assert!(utils::validate_export_config(&invalid_config).is_err());
    }

    #[test]
    fn test_recommended_formats() {
        assert_eq!(
            utils::get_recommended_format(TargetPlatform::GPU),
            ExportFormat::TensorRT
        );
        assert_eq!(
            utils::get_recommended_format(TargetPlatform::Mobile),
            ExportFormat::Mobile
        );
        assert_eq!(
            utils::get_recommended_format(TargetPlatform::Edge),
            ExportFormat::TFLite
        );
        assert_eq!(
            utils::get_recommended_format(TargetPlatform::CPU),
            ExportFormat::Onnx
        );
        assert_eq!(
            utils::get_recommended_format(TargetPlatform::Cloud),
            ExportFormat::Onnx
        );
    }

    #[test]
    fn test_sample_model_creation() {
        let model = utils::create_sample_model().unwrap();

        assert!(!model.parameters.is_empty());
        assert!(!model.architecture.layers.is_empty());
        assert!(!model.architecture.input_shapes.is_empty());
        assert!(!model.architecture.output_shapes.is_empty());
        assert!(!model.metadata.is_empty());
    }
}
