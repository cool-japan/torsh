//! Model export functionality
//!
//! This module provides export functionality for various formats including ONNX,
//! TorchScript compatibility, and deployment optimizations.
//!
//! Note: This module requires std for file operations and is only available with the "std" feature.

#[cfg(feature = "std")]
use crate::Module;
#[cfg(feature = "std")]
use std::{path::Path, string::String, vec::Vec};
#[cfg(feature = "std")]
use torsh_core::error::{Result, TorshError};

#[cfg(feature = "serialize")]
use serde_json;

/// Target device for model optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetDevice {
    /// CPU device
    Cpu,
    /// GPU device
    Gpu,
    /// CUDA GPU device
    Cuda,
    /// Mobile device
    Mobile,
    /// WebAssembly target
    Wasm,
    /// Web target
    Web,
    /// Custom device
    Custom(u32),
}

impl Default for TargetDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Export format for model serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// ONNX (Open Neural Network Exchange) format
    Onnx,
    /// TorchScript compatible format
    TorchScript,
    /// Custom binary format optimized for deployment
    TorshBinary,
    /// JSON format for easy inspection
    Json,
}

/// Export configuration for model serialization
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Target export format
    pub format: ExportFormat,
    /// Include training-specific parameters
    pub include_training: bool,
    /// Optimization level for deployment
    pub optimization_level: OptimizationLevel,
    /// Target device for optimized deployment
    pub target_device: TargetDevice,
    /// Include metadata and documentation
    pub include_metadata: bool,
    /// Input shapes for static optimization
    pub input_shapes: Vec<Vec<usize>>,
}

/// Optimization level for deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization, preserve exact behavior
    None,
    /// Basic optimizations that don't change semantics
    Basic,
    /// Aggressive optimizations for maximum performance
    Aggressive,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::TorshBinary,
            include_training: false,
            optimization_level: OptimizationLevel::Basic,
            target_device: TargetDevice::Cpu,
            include_metadata: true,
            input_shapes: vec![],
        }
    }
}

/// Model exporter for converting models to various formats
pub struct ModelExporter {
    config: ExportConfig,
}

impl ModelExporter {
    /// Create a new model exporter with the given configuration
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Create an exporter with default settings for ONNX format
    pub fn onnx() -> Self {
        Self::new(ExportConfig {
            format: ExportFormat::Onnx,
            ..Default::default()
        })
    }

    /// Create an exporter with default settings for TorchScript format
    pub fn torchscript() -> Self {
        Self::new(ExportConfig {
            format: ExportFormat::TorchScript,
            ..Default::default()
        })
    }

    /// Export a model to the specified path
    pub fn export<M: Module>(&self, model: &M, path: &Path) -> Result<()> {
        // Set model to evaluation mode for export
        // Note: We would need a mutable reference for this in practice
        // model.eval();

        match self.config.format {
            ExportFormat::Onnx => self.export_onnx(model, path),
            ExportFormat::TorchScript => self.export_torchscript(model, path),
            ExportFormat::TorshBinary => self.export_torsh_binary(model, path),
            ExportFormat::Json => self.export_json(model, path),
        }
    }

    /// Export model to ONNX format
    fn export_onnx<M: Module>(&self, model: &M, path: &Path) -> Result<()> {
        // ONNX export implementation
        // This would require integration with onnx crate or custom implementation

        // For now, we'll create a placeholder implementation
        let onnx_graph = self.build_onnx_graph(model)?;

        // Write to file
        std::fs::write(path, format!("ONNX Graph: {:?}", onnx_graph))
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Export model to TorchScript compatible format
    fn export_torchscript<M: Module>(&self, model: &M, path: &Path) -> Result<()> {
        // TorchScript export implementation
        // This would require integration with PyTorch JIT or custom implementation

        let script_module = self.build_torchscript_module(model)?;

        // Write to file
        std::fs::write(path, format!("TorchScript Module: {:?}", script_module))
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Export model to custom binary format optimized for Torsh
    fn export_torsh_binary<M: Module>(&self, model: &M, path: &Path) -> Result<()> {
        // Custom binary format implementation
        let binary_data = self.serialize_to_binary(model)?;

        std::fs::write(path, binary_data).map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Export model to JSON format for inspection
    fn export_json<M: Module>(&self, model: &M, path: &Path) -> Result<()> {
        let json_data = self.serialize_to_json(model)?;

        std::fs::write(path, json_data).map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Export model to bytes without writing to file (for benchmarking)
    pub fn export_to_bytes<M: Module>(&self, model: &M) -> Result<Vec<u8>> {
        match self.config.format {
            ExportFormat::Onnx => {
                let onnx_graph = self.build_onnx_graph(model)?;
                Ok(onnx_graph.into_bytes())
            }
            ExportFormat::TorchScript => {
                let script_module = self.build_torchscript_module(model)?;
                Ok(script_module.into_bytes())
            }
            ExportFormat::TorshBinary => self.serialize_to_binary(model),
            ExportFormat::Json => {
                let json_data = self.serialize_to_json(model)?;
                Ok(json_data.into_bytes())
            }
        }
    }

    /// Build ONNX graph representation (placeholder)
    fn build_onnx_graph<M: Module>(&self, model: &M) -> Result<String> {
        // This would build an actual ONNX graph
        // For now, return a placeholder
        Ok(format!(
            "ONNX graph for model with {} parameters",
            model.parameters().len()
        ))
    }

    /// Build TorchScript module representation (placeholder)
    fn build_torchscript_module<M: Module>(&self, model: &M) -> Result<String> {
        // This would build an actual TorchScript module
        // For now, return a placeholder
        Ok(format!(
            "TorchScript module for model with {} parameters",
            model.parameters().len()
        ))
    }

    /// Serialize model to binary format
    fn serialize_to_binary<M: Module>(&self, model: &M) -> Result<Vec<u8>> {
        // Custom binary serialization
        // This would use a format like Protocol Buffers, FlatBuffers, or custom format

        let mut data = Vec::new();

        // Write magic number
        data.extend_from_slice(b"TORSH_V1");

        // Write parameters
        let params = model.parameters();
        data.extend_from_slice(&(params.len() as u32).to_le_bytes());

        for (_name, param) in params {
            let tensor_arc = param.tensor();
            let tensor = tensor_arc.read();
            let tensor_shape = tensor.shape();
            let shape = tensor_shape.dims();

            // Write shape
            data.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                data.extend_from_slice(&(dim as u32).to_le_bytes());
            }

            // Write data (placeholder - would need actual tensor data serialization)
            let dummy_data = vec![0u8; shape.iter().product::<usize>() * 4]; // Assuming f32
            data.extend_from_slice(&dummy_data);
        }

        Ok(data)
    }

    /// Serialize model to JSON format
    #[cfg(feature = "serialize")]
    fn serialize_to_json<M: Module>(&self, model: &M) -> Result<String> {
        let mut json_obj = serde_json::Map::new();

        // Add metadata
        json_obj.insert(
            "format".to_string(),
            serde_json::Value::String("torsh_nn".to_string()),
        );
        json_obj.insert(
            "version".to_string(),
            serde_json::Value::String("0.1.0".to_string()),
        );

        // Add parameters info
        let params = model.parameters();
        let mut params_info = Vec::new();

        for (i, (name, param)) in params.iter().enumerate() {
            let tensor_arc = param.tensor();
            let tensor = tensor_arc.read();
            let shape_obj = tensor.shape();
            let shape = shape_obj.dims();

            let param_obj = serde_json::json!({
                "index": i,
                "name": name,
                "shape": shape,
                "numel": shape.iter().product::<usize>(),
                "requires_grad": param.requires_grad()
            });

            params_info.push(param_obj);
        }

        json_obj.insert(
            "parameters".to_string(),
            serde_json::Value::Array(params_info),
        );

        // Add configuration
        if self.config.include_metadata {
            let config_obj = serde_json::json!({
                "optimization_level": format!("{:?}", self.config.optimization_level),
                "target_device": format!("{:?}", self.config.target_device),
                "input_shapes": self.config.input_shapes
            });
            json_obj.insert("export_config".to_string(), config_obj);
        }

        serde_json::to_string_pretty(&json_obj)
            .map_err(|e| TorshError::SerializationError(e.to_string()))
    }

    /// Serialize model to JSON format (fallback when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    fn serialize_to_json<M: Module>(&self, _model: &M) -> Result<String> {
        Err(TorshError::ConfigError(
            "JSON serialization requires 'serialize' feature to be enabled".to_string(),
        ))
    }
}

/// Deployment optimization utilities
pub struct DeploymentOptimizer {
    target_device: TargetDevice,
    #[allow(dead_code)]
    optimization_level: OptimizationLevel,
}

impl DeploymentOptimizer {
    /// Create a new deployment optimizer
    pub fn new(target_device: TargetDevice, optimization_level: OptimizationLevel) -> Self {
        Self {
            target_device,
            optimization_level,
        }
    }

    /// Optimize model for deployment
    pub fn optimize<M: Module>(&self, model: &M) -> Result<OptimizedModel> {
        match self.target_device {
            TargetDevice::Cpu => self.optimize_for_cpu(model),
            TargetDevice::Gpu => self.optimize_for_cuda(model), // Use CUDA optimizations for GPU
            TargetDevice::Cuda => self.optimize_for_cuda(model),
            TargetDevice::Mobile => self.optimize_for_mobile(model),
            TargetDevice::Wasm => self.optimize_for_web(model), // Use web optimizations for WASM
            TargetDevice::Web => self.optimize_for_web(model),
            TargetDevice::Custom(_) => self.optimize_for_cpu(model), // Fallback to CPU optimizations
        }
    }

    /// CPU-specific optimizations
    fn optimize_for_cpu<M: Module>(&self, model: &M) -> Result<OptimizedModel> {
        // CPU optimizations:
        // - Loop fusion
        // - SIMD utilization
        // - Memory layout optimization
        // - Quantization if requested

        Ok(OptimizedModel::new(model, TargetDevice::Cpu))
    }

    /// CUDA-specific optimizations
    fn optimize_for_cuda<M: Module>(&self, model: &M) -> Result<OptimizedModel> {
        // CUDA optimizations:
        // - Kernel fusion
        // - Memory coalescing
        // - Tensor Core utilization
        // - Stream optimization

        Ok(OptimizedModel::new(model, TargetDevice::Cuda))
    }

    /// Mobile-specific optimizations
    fn optimize_for_mobile<M: Module>(&self, model: &M) -> Result<OptimizedModel> {
        // Mobile optimizations:
        // - Model pruning
        // - Quantization to INT8
        // - Memory usage reduction
        // - Battery optimization

        Ok(OptimizedModel::new(model, TargetDevice::Mobile))
    }

    /// Web/WASM-specific optimizations
    fn optimize_for_web<M: Module>(&self, model: &M) -> Result<OptimizedModel> {
        // Web optimizations:
        // - Size reduction
        // - WebGL shader optimization
        // - Memory efficiency
        // - Loading time optimization

        Ok(OptimizedModel::new(model, TargetDevice::Web))
    }
}

/// Optimized model for deployment
pub struct OptimizedModel {
    // This would contain the optimized model representation
    target_device: TargetDevice,
    optimizations_applied: Vec<String>,
}

impl OptimizedModel {
    fn new<M: Module>(_model: &M, target_device: TargetDevice) -> Self {
        Self {
            target_device,
            optimizations_applied: vec!["placeholder".to_string()],
        }
    }

    /// Get the target device for this optimized model
    pub fn target_device(&self) -> TargetDevice {
        self.target_device
    }

    /// Get the list of optimizations applied
    pub fn optimizations_applied(&self) -> &[String] {
        &self.optimizations_applied
    }
}

/// Benchmarking utilities for export and conversion performance
pub mod benchmarks {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    /// Export performance metrics
    #[derive(Debug, Clone)]
    pub struct ExportMetrics {
        /// Time taken for export operation
        pub export_time: Duration,
        /// Size of exported model in bytes
        pub export_size: usize,
        /// Memory usage during export
        pub peak_memory_mb: f32,
        /// Export throughput (ops/sec)
        pub throughput: f32,
        /// Compression ratio compared to original
        pub compression_ratio: f32,
        /// Target device used
        pub target_device: TargetDevice,
        /// Export format used
        pub export_format: ExportFormat,
    }

    /// Conversion performance metrics
    #[derive(Debug, Clone)]
    pub struct ConversionMetrics {
        /// Time taken for conversion
        pub conversion_time: Duration,
        /// Memory usage during conversion
        pub peak_memory_mb: f32,
        /// Number of layers converted
        pub layers_converted: usize,
        /// Number of parameters converted
        pub parameters_converted: usize,
        /// Conversion success rate
        pub success_rate: f32,
        /// Source and target formats
        pub source_format: String,
        pub target_format: String,
    }

    /// Comprehensive benchmark results
    #[derive(Debug, Clone)]
    pub struct BenchmarkResults {
        /// Export metrics for different configurations
        pub export_metrics: HashMap<String, ExportMetrics>,
        /// Conversion metrics for different paths
        pub conversion_metrics: HashMap<String, ConversionMetrics>,
        /// Overall benchmark summary
        pub summary: BenchmarkSummary,
    }

    /// Summary of benchmark results
    #[derive(Debug, Clone)]
    pub struct BenchmarkSummary {
        /// Total time for all benchmarks
        pub total_time: Duration,
        /// Fastest export configuration
        pub fastest_export: String,
        /// Most compact export configuration
        pub most_compact_export: String,
        /// Recommended configuration for deployment
        pub recommended_config: String,
    }

    /// Export performance benchmarker
    pub struct ExportBenchmarker {
        configurations: Vec<(String, ExportConfig)>,
        warmup_runs: usize,
        benchmark_runs: usize,
    }

    impl ExportBenchmarker {
        /// Create a new export benchmarker with default configurations
        pub fn new() -> Self {
            let mut configurations = Vec::new();

            // Add common configurations to benchmark
            configurations.push((
                "onnx_basic".to_string(),
                ExportConfig {
                    format: ExportFormat::Onnx,
                    optimization_level: OptimizationLevel::Basic,
                    target_device: TargetDevice::Cpu,
                    ..Default::default()
                },
            ));

            configurations.push((
                "onnx_aggressive".to_string(),
                ExportConfig {
                    format: ExportFormat::Onnx,
                    optimization_level: OptimizationLevel::Aggressive,
                    target_device: TargetDevice::Cpu,
                    ..Default::default()
                },
            ));

            configurations.push((
                "torchscript_basic".to_string(),
                ExportConfig {
                    format: ExportFormat::TorchScript,
                    optimization_level: OptimizationLevel::Basic,
                    target_device: TargetDevice::Cpu,
                    ..Default::default()
                },
            ));

            configurations.push((
                "binary_fast".to_string(),
                ExportConfig {
                    format: ExportFormat::TorshBinary,
                    optimization_level: OptimizationLevel::None,
                    target_device: TargetDevice::Cpu,
                    ..Default::default()
                },
            ));

            configurations.push((
                "json_debug".to_string(),
                ExportConfig {
                    format: ExportFormat::Json,
                    optimization_level: OptimizationLevel::None,
                    target_device: TargetDevice::Cpu,
                    include_metadata: true,
                    ..Default::default()
                },
            ));

            Self {
                configurations,
                warmup_runs: 3,
                benchmark_runs: 10,
            }
        }

        /// Add a custom configuration to benchmark
        pub fn add_configuration(&mut self, name: String, config: ExportConfig) {
            self.configurations.push((name, config));
        }

        /// Set the number of warmup and benchmark runs
        pub fn set_runs(&mut self, warmup_runs: usize, benchmark_runs: usize) {
            self.warmup_runs = warmup_runs;
            self.benchmark_runs = benchmark_runs;
        }

        /// Get the configurations
        pub fn configurations(&self) -> &Vec<(String, ExportConfig)> {
            &self.configurations
        }

        /// Get the number of warmup runs
        pub fn warmup_runs(&self) -> usize {
            self.warmup_runs
        }

        /// Get the number of benchmark runs
        pub fn benchmark_runs(&self) -> usize {
            self.benchmark_runs
        }

        /// Run comprehensive export benchmarks on a model
        pub fn benchmark_model<M: Module + Clone>(&self, model: &M) -> Result<BenchmarkResults> {
            let mut export_metrics = HashMap::new();
            let benchmark_start = Instant::now();

            for (config_name, config) in &self.configurations {
                println!("Benchmarking export configuration: {}", config_name);

                let metrics = self.benchmark_single_export(model, config)?;
                export_metrics.insert(config_name.clone(), metrics);
            }

            // Create summary
            let total_time = benchmark_start.elapsed();
            let summary = self.create_summary(&export_metrics, total_time);

            // For now, conversion metrics are empty - could be extended
            let conversion_metrics = HashMap::new();

            Ok(BenchmarkResults {
                export_metrics,
                conversion_metrics,
                summary,
            })
        }

        /// Benchmark a single export configuration
        fn benchmark_single_export<M: Module + Clone>(
            &self,
            model: &M,
            config: &ExportConfig,
        ) -> Result<ExportMetrics> {
            let exporter = ModelExporter::new(config.clone());

            // Warmup runs
            for _ in 0..self.warmup_runs {
                let _result = exporter.export_to_bytes(model)?;
            }

            // Benchmark runs
            let mut times = Vec::new();
            let mut export_size = 0;

            for _ in 0..self.benchmark_runs {
                let start = Instant::now();
                let exported_bytes = exporter.export_to_bytes(model)?;
                let elapsed = start.elapsed();

                times.push(elapsed);
                export_size = exported_bytes.len();
            }

            // Calculate statistics
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let throughput = 1.0 / avg_time.as_secs_f32();

            // Simulate memory usage and compression ratio
            let peak_memory_mb = (export_size as f32) / (1024.0 * 1024.0) * 1.5; // Rough estimate
            let compression_ratio = match config.optimization_level {
                OptimizationLevel::None => 1.0,
                OptimizationLevel::Basic => 1.2,
                OptimizationLevel::Aggressive => 1.8,
            };

            Ok(ExportMetrics {
                export_time: avg_time,
                export_size,
                peak_memory_mb,
                throughput,
                compression_ratio,
                target_device: config.target_device,
                export_format: config.format,
            })
        }

        /// Create benchmark summary
        fn create_summary(
            &self,
            export_metrics: &HashMap<String, ExportMetrics>,
            total_time: Duration,
        ) -> BenchmarkSummary {
            let mut fastest_export = String::new();
            let mut most_compact_export = String::new();
            let mut fastest_time = Duration::from_secs(u64::MAX);
            let mut smallest_size = usize::MAX;

            for (name, metrics) in export_metrics {
                if metrics.export_time < fastest_time {
                    fastest_time = metrics.export_time;
                    fastest_export = name.clone();
                }

                if metrics.export_size < smallest_size {
                    smallest_size = metrics.export_size;
                    most_compact_export = name.clone();
                }
            }

            // Simple heuristic for recommendation
            let recommended_config = if export_metrics.contains_key("onnx_basic") {
                "onnx_basic".to_string()
            } else {
                fastest_export.clone()
            };

            BenchmarkSummary {
                total_time,
                fastest_export,
                most_compact_export,
                recommended_config,
            }
        }
    }

    impl Default for ExportBenchmarker {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Conversion benchmarker for different model formats
    pub struct ConversionBenchmarker {
        conversion_paths: Vec<(String, String, String)>, // (name, source, target)
    }

    impl ConversionBenchmarker {
        /// Create a new conversion benchmarker
        pub fn new() -> Self {
            let conversion_paths = vec![
                (
                    "pytorch_to_onnx".to_string(),
                    "pytorch".to_string(),
                    "onnx".to_string(),
                ),
                (
                    "tensorflow_to_onnx".to_string(),
                    "tensorflow".to_string(),
                    "onnx".to_string(),
                ),
                (
                    "onnx_to_torsh".to_string(),
                    "onnx".to_string(),
                    "torsh".to_string(),
                ),
                (
                    "torsh_to_onnx".to_string(),
                    "torsh".to_string(),
                    "onnx".to_string(),
                ),
            ];

            Self { conversion_paths }
        }

        /// Benchmark model conversions (placeholder implementation)
        pub fn benchmark_conversions(&self) -> Result<HashMap<String, ConversionMetrics>> {
            let mut metrics = HashMap::new();

            for (name, source, target) in &self.conversion_paths {
                let start = Instant::now();

                // Simulate conversion time based on complexity
                std::thread::sleep(Duration::from_millis(10));

                let conversion_time = start.elapsed();

                let metric = ConversionMetrics {
                    conversion_time,
                    peak_memory_mb: 128.0,      // Placeholder
                    layers_converted: 10,       // Placeholder
                    parameters_converted: 1000, // Placeholder
                    success_rate: 0.95,         // Placeholder
                    source_format: source.clone(),
                    target_format: target.clone(),
                };

                metrics.insert(name.clone(), metric);
            }

            Ok(metrics)
        }
    }

    impl Default for ConversionBenchmarker {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Utility functions for benchmarking
    pub mod utils {
        use super::*;

        /// Create a comprehensive benchmark report
        pub fn create_benchmark_report(results: &BenchmarkResults) -> String {
            let mut report = String::new();

            report.push_str("# Export/Conversion Performance Benchmark Report\n\n");

            // Summary section
            report.push_str("## Summary\n");
            report.push_str(&format!(
                "- Total benchmark time: {:?}\n",
                results.summary.total_time
            ));
            report.push_str(&format!(
                "- Fastest export: {}\n",
                results.summary.fastest_export
            ));
            report.push_str(&format!(
                "- Most compact export: {}\n",
                results.summary.most_compact_export
            ));
            report.push_str(&format!(
                "- Recommended config: {}\n\n",
                results.summary.recommended_config
            ));

            // Export metrics section
            report.push_str("## Export Performance\n");
            for (name, metrics) in &results.export_metrics {
                report.push_str(&format!("### {}\n", name));
                report.push_str(&format!("- Export time: {:?}\n", metrics.export_time));
                report.push_str(&format!("- Export size: {} bytes\n", metrics.export_size));
                report.push_str(&format!(
                    "- Peak memory: {:.2} MB\n",
                    metrics.peak_memory_mb
                ));
                report.push_str(&format!(
                    "- Throughput: {:.2} exports/sec\n",
                    metrics.throughput
                ));
                report.push_str(&format!(
                    "- Compression ratio: {:.2}x\n\n",
                    metrics.compression_ratio
                ));
            }

            // Conversion metrics section
            if !results.conversion_metrics.is_empty() {
                report.push_str("## Conversion Performance\n");
                for (name, metrics) in &results.conversion_metrics {
                    report.push_str(&format!("### {}\n", name));
                    report.push_str(&format!(
                        "- Conversion time: {:?}\n",
                        metrics.conversion_time
                    ));
                    report.push_str(&format!(
                        "- Peak memory: {:.2} MB\n",
                        metrics.peak_memory_mb
                    ));
                    report.push_str(&format!(
                        "- Layers converted: {}\n",
                        metrics.layers_converted
                    ));
                    report.push_str(&format!(
                        "- Success rate: {:.1}%\n\n",
                        metrics.success_rate * 100.0
                    ));
                }
            }

            report
        }

        /// Compare two benchmark results
        pub fn compare_benchmarks(
            results1: &BenchmarkResults,
            results2: &BenchmarkResults,
            name1: &str,
            name2: &str,
        ) -> String {
            let mut comparison = String::new();

            comparison.push_str(&format!(
                "# Benchmark Comparison: {} vs {}\n\n",
                name1, name2
            ));

            // Compare export metrics
            for (config_name, metrics1) in &results1.export_metrics {
                if let Some(metrics2) = results2.export_metrics.get(config_name) {
                    comparison.push_str(&format!("## {}\n", config_name));

                    let time_ratio =
                        metrics2.export_time.as_secs_f32() / metrics1.export_time.as_secs_f32();
                    let size_ratio = metrics2.export_size as f32 / metrics1.export_size as f32;

                    comparison.push_str(&format!(
                        "- Export time: {:.2}x {}\n",
                        time_ratio,
                        if time_ratio > 1.0 { "slower" } else { "faster" }
                    ));
                    comparison.push_str(&format!(
                        "- Export size: {:.2}x {}\n",
                        size_ratio,
                        if size_ratio > 1.0 {
                            "larger"
                        } else {
                            "smaller"
                        }
                    ));
                    comparison.push_str("\n");
                }
            }

            comparison
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::TorshBinary);
        assert!(!config.include_training);
        assert_eq!(config.optimization_level, OptimizationLevel::Basic);
    }

    #[test]
    fn test_model_exporter_creation() {
        let exporter = ModelExporter::onnx();
        assert_eq!(exporter.config.format, ExportFormat::Onnx);

        let exporter = ModelExporter::torchscript();
        assert_eq!(exporter.config.format, ExportFormat::TorchScript);
    }

    #[test]
    fn test_deployment_optimizer() {
        let optimizer = DeploymentOptimizer::new(TargetDevice::Cpu, OptimizationLevel::Basic);

        assert_eq!(optimizer.target_device, TargetDevice::Cpu);
        assert_eq!(optimizer.optimization_level, OptimizationLevel::Basic);
    }

    #[test]
    fn test_export_benchmarker() {
        let benchmarker = benchmarks::ExportBenchmarker::new();
        assert!(!benchmarker.configurations().is_empty());
        assert_eq!(benchmarker.warmup_runs(), 3);
        assert_eq!(benchmarker.benchmark_runs(), 10);
    }

    #[test]
    fn test_conversion_benchmarker() {
        let benchmarker = benchmarks::ConversionBenchmarker::new();
        let results = benchmarker.benchmark_conversions().unwrap();
        assert!(!results.is_empty());

        for (name, metrics) in &results {
            assert!(!name.is_empty());
            assert!(metrics.conversion_time.as_millis() >= 10); // At least our sleep time
            assert!(metrics.success_rate > 0.0 && metrics.success_rate <= 1.0);
        }
    }

    #[test]
    fn test_benchmark_report_generation() {
        use benchmarks::*;
        use std::time::Duration;

        let mut export_metrics = HashMap::new();
        export_metrics.insert(
            "test_config".to_string(),
            ExportMetrics {
                export_time: Duration::from_millis(100),
                export_size: 1024,
                peak_memory_mb: 64.0,
                throughput: 10.0,
                compression_ratio: 1.5,
                target_device: TargetDevice::Cpu,
                export_format: ExportFormat::Onnx,
            },
        );

        let results = BenchmarkResults {
            export_metrics,
            conversion_metrics: HashMap::new(),
            summary: BenchmarkSummary {
                total_time: Duration::from_secs(1),
                fastest_export: "test_config".to_string(),
                most_compact_export: "test_config".to_string(),
                recommended_config: "test_config".to_string(),
            },
        };

        let report = utils::create_benchmark_report(&results);
        assert!(report.contains("Export/Conversion Performance Benchmark Report"));
        assert!(report.contains("test_config"));
        assert!(report.contains("100ms"));
    }

    #[test]
    fn test_benchmark_comparison() {
        use benchmarks::*;
        use std::time::Duration;

        let mut export_metrics1 = HashMap::new();
        export_metrics1.insert(
            "config1".to_string(),
            ExportMetrics {
                export_time: Duration::from_millis(100),
                export_size: 1024,
                peak_memory_mb: 64.0,
                throughput: 10.0,
                compression_ratio: 1.5,
                target_device: TargetDevice::Cpu,
                export_format: ExportFormat::Onnx,
            },
        );

        let mut export_metrics2 = HashMap::new();
        export_metrics2.insert(
            "config1".to_string(),
            ExportMetrics {
                export_time: Duration::from_millis(200),
                export_size: 2048,
                peak_memory_mb: 128.0,
                throughput: 5.0,
                compression_ratio: 1.5,
                target_device: TargetDevice::Cpu,
                export_format: ExportFormat::Onnx,
            },
        );

        let results1 = BenchmarkResults {
            export_metrics: export_metrics1,
            conversion_metrics: HashMap::new(),
            summary: BenchmarkSummary {
                total_time: Duration::from_secs(1),
                fastest_export: "config1".to_string(),
                most_compact_export: "config1".to_string(),
                recommended_config: "config1".to_string(),
            },
        };

        let results2 = BenchmarkResults {
            export_metrics: export_metrics2,
            conversion_metrics: HashMap::new(),
            summary: BenchmarkSummary {
                total_time: Duration::from_secs(2),
                fastest_export: "config1".to_string(),
                most_compact_export: "config1".to_string(),
                recommended_config: "config1".to_string(),
            },
        };

        let comparison = utils::compare_benchmarks(&results1, &results2, "baseline", "optimized");
        assert!(comparison.contains("Benchmark Comparison"));
        assert!(comparison.contains("2.00x slower"));
        assert!(comparison.contains("2.00x larger"));
    }
}
