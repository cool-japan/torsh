//! Model summary utilities for analyzing neural network architectures
//!
//! This module provides tools for printing detailed summaries of neural network models,
//! including layer information, parameter counts, and memory usage estimates.

use crate::{Module, Parameter};
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{
    collections::HashMap,
    fmt::{self, Display},
    string::String,
    time::{Duration, Instant},
    vec::Vec,
};

#[cfg(not(feature = "std"))]
use alloc::{
    fmt::{self, Display},
    string::String,
    vec::Vec,
};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Information about a single layer in the model
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name/identifier
    pub name: String,
    /// Layer type (e.g., "Linear", "Conv2d", "ReLU")
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of parameters
    pub param_count: usize,
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Memory usage estimate in bytes
    pub memory_bytes: usize,
}

impl LayerInfo {
    /// Create a new LayerInfo
    pub fn new(
        name: String,
        layer_type: String,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        param_count: usize,
        trainable_params: usize,
    ) -> Self {
        // Estimate memory usage (rough approximation)
        let input_elements: usize = input_shape.iter().product();
        let output_elements: usize = output_shape.iter().product();
        let memory_bytes = (input_elements + output_elements + param_count) * 4; // Assuming f32

        Self {
            name,
            layer_type,
            input_shape,
            output_shape,
            param_count,
            trainable_params,
            memory_bytes,
        }
    }
}

impl Display for LayerInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<20} {:<15} {:<20} {:<20} {:>10} {:>10}",
            self.name,
            self.layer_type,
            format!("{:?}", self.input_shape),
            format!("{:?}", self.output_shape),
            format_number(self.param_count),
            format_bytes(self.memory_bytes)
        )
    }
}

/// Complete model summary information
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Information for each layer
    pub layers: Vec<LayerInfo>,
    /// Total number of parameters
    pub total_params: usize,
    /// Total number of trainable parameters
    pub trainable_params: usize,
    /// Total memory usage estimate in bytes
    pub total_memory_bytes: usize,
    /// Model input shape
    pub input_shape: Vec<usize>,
    /// Model output shape
    pub output_shape: Vec<usize>,
}

impl ModelSummary {
    /// Create a new model summary
    pub fn new(layers: Vec<LayerInfo>, input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        let total_params = layers.iter().map(|l| l.param_count).sum();
        let trainable_params = layers.iter().map(|l| l.trainable_params).sum();
        let total_memory_bytes = layers.iter().map(|l| l.memory_bytes).sum();

        Self {
            layers,
            total_params,
            trainable_params,
            total_memory_bytes,
            input_shape,
            output_shape,
        }
    }

    /// Print a formatted summary to stdout
    pub fn print(&self) {
        println!("{}", self);
    }
}

impl Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "========================================================================================")?;
        writeln!(f, "Model Summary")?;
        writeln!(f, "========================================================================================")?;
        writeln!(f, "Input Shape: {:?}", self.input_shape)?;
        writeln!(f, "Output Shape: {:?}", self.output_shape)?;
        writeln!(f, "========================================================================================")?;
        writeln!(
            f,
            "{:<20} {:<15} {:<20} {:<20} {:>10} {:>10}",
            "Layer (type)", "Type", "Input Shape", "Output Shape", "Param #", "Memory"
        )?;
        writeln!(f, "========================================================================================")?;

        for layer in &self.layers {
            writeln!(f, "{}", layer)?;
        }

        writeln!(f, "========================================================================================")?;
        writeln!(f, "Total params: {}", format_number(self.total_params))?;
        writeln!(
            f,
            "Trainable params: {}",
            format_number(self.trainable_params)
        )?;
        writeln!(
            f,
            "Non-trainable params: {}",
            format_number(self.total_params - self.trainable_params)
        )?;
        writeln!(
            f,
            "Total memory usage: {}",
            format_bytes(self.total_memory_bytes)
        )?;
        writeln!(f, "========================================================================================")?;

        Ok(())
    }
}

/// Summary configuration options
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Maximum depth to traverse in nested modules
    pub max_depth: usize,
    /// Whether to show only trainable parameters
    pub trainable_only: bool,
    /// Whether to include memory estimates
    pub show_memory: bool,
    /// Whether to use verbose output
    pub verbose: bool,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            trainable_only: false,
            show_memory: true,
            verbose: false,
        }
    }
}

/// Create a summary of a model
pub fn summarize<M: Module>(
    model: &M,
    input_shape: &[usize],
    config: Option<SummaryConfig>,
) -> Result<ModelSummary> {
    let config = config.unwrap_or_default();

    // Create a dummy input tensor to trace through the model
    let dummy_input = torsh_tensor::creation::zeros(input_shape)?;

    // Get model output to determine output shape
    let output = model.forward(&dummy_input)?;
    let output_shape = output.shape().dims().to_vec();

    // Analyze the model structure
    let layers = analyze_model_structure(model, input_shape, &config)?;

    Ok(ModelSummary::new(
        layers,
        input_shape.to_vec(),
        output_shape,
    ))
}

/// Analyze the structure of a model and extract layer information
fn analyze_model_structure<M: Module>(
    model: &M,
    input_shape: &[usize],
    config: &SummaryConfig,
) -> Result<Vec<LayerInfo>> {
    let mut layers = Vec::new();

    // Get all parameters
    let parameters = model.parameters();
    let _named_parameters = model.named_parameters();

    // For simplicity, we'll create a single layer info for the entire model
    // In a more sophisticated implementation, we would traverse the module tree
    let total_params = count_parameters(&parameters);
    let trainable_params = count_trainable_parameters(&parameters);

    let layer_info = LayerInfo::new(
        "Model".to_string(),
        get_module_type_name(model),
        input_shape.to_vec(),
        input_shape.to_vec(), // Placeholder - would be computed from forward pass
        total_params,
        trainable_params,
    );

    layers.push(layer_info);

    // If the model has children, analyze them recursively
    if config.max_depth > 0 {
        let children = model.children();
        for (i, child) in children.iter().enumerate() {
            let _child_config = SummaryConfig {
                max_depth: config.max_depth - 1,
                ..config.clone()
            };

            let child_name = format!("child_{}", i);
            let child_params = child.parameters();
            let child_total_params = count_parameters(&child_params);
            let child_trainable_params = count_trainable_parameters(&child_params);

            let child_info = LayerInfo::new(
                child_name,
                get_module_type_name(*child),
                input_shape.to_vec(), // Simplified
                input_shape.to_vec(), // Simplified
                child_total_params,
                child_trainable_params,
            );

            layers.push(child_info);
        }
    }

    Ok(layers)
}

/// Count total parameters in a parameter map
fn count_parameters(parameters: &HashMap<String, Parameter>) -> usize {
    parameters
        .values()
        .map(|param| {
            let tensor_guard = param.tensor();
            let tensor = tensor_guard.read();
            tensor.shape().dims().iter().product::<usize>()
        })
        .sum()
}

/// Count trainable parameters in a parameter map
fn count_trainable_parameters(parameters: &HashMap<String, Parameter>) -> usize {
    // For now, assume all parameters are trainable
    // In a full implementation, this would check the requires_grad flag
    count_parameters(parameters)
}

/// Get the type name of a module (simplified implementation)
fn get_module_type_name<M: Module + ?Sized>(_module: &M) -> String {
    // This is a simplified implementation
    // In a full implementation, we would use type reflection or naming conventions
    "Module".to_string()
}

/// Format a number with appropriate units (K, M, B)
fn format_number(num: usize) -> String {
    if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

/// Format bytes with appropriate units (KB, MB, GB)
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.1} KB", bytes as f64 / 1_024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Estimate the memory usage of a tensor shape
pub fn estimate_tensor_memory(shape: &[usize], dtype_size: usize) -> usize {
    shape.iter().product::<usize>() * dtype_size
}

/// Advanced model profiler that can track memory usage and compute statistics
pub struct ModelProfiler {
    /// Whether to track memory usage
    pub track_memory: bool,
    /// Whether to track computation time
    pub track_time: bool,
    /// Whether to track activations
    pub track_activations: bool,
}

impl Default for ModelProfiler {
    fn default() -> Self {
        Self {
            track_memory: true,
            track_time: false,
            track_activations: false,
        }
    }
}

impl ModelProfiler {
    /// Create a new model profiler
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable memory tracking
    pub fn with_memory_tracking(mut self) -> Self {
        self.track_memory = true;
        self
    }

    /// Enable time tracking
    pub fn with_time_tracking(mut self) -> Self {
        self.track_time = true;
        self
    }

    /// Enable activation tracking
    pub fn with_activation_tracking(mut self) -> Self {
        self.track_activations = true;
        self
    }

    /// Profile a model with the given input
    pub fn profile<M: Module>(&self, model: &M, input: &Tensor) -> Result<ProfileResult> {
        let start_memory = if self.track_memory {
            Some(get_memory_usage())
        } else {
            None
        };

        let start_time = if self.track_time {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Run forward pass
        let output = model.forward(input)?;

        let end_time = start_time.map(|start| start.elapsed());
        let memory_used = start_memory.map(|start| get_memory_usage() - start);

        Ok(ProfileResult {
            input_shape: input.shape().dims().to_vec(),
            output_shape: output.shape().dims().to_vec(),
            memory_used,
            execution_time: end_time,
            parameter_count: count_parameters(&model.parameters()),
        })
    }
}

/// Result of model profiling
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Input tensor shape
    pub input_shape: Vec<usize>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Memory used during forward pass (if tracked)
    pub memory_used: Option<usize>,
    /// Execution time (if tracked)
    pub execution_time: Option<std::time::Duration>,
    /// Total parameter count
    pub parameter_count: usize,
}

impl Display for ProfileResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Profile Result:")?;
        writeln!(f, "  Input Shape: {:?}", self.input_shape)?;
        writeln!(f, "  Output Shape: {:?}", self.output_shape)?;
        writeln!(f, "  Parameters: {}", format_number(self.parameter_count))?;

        if let Some(memory) = self.memory_used {
            writeln!(f, "  Memory Used: {}", format_bytes(memory))?;
        }

        if let Some(time) = self.execution_time {
            writeln!(f, "  Execution Time: {:.3}ms", time.as_secs_f64() * 1000.0)?;
        }

        Ok(())
    }
}

/// Get current memory usage (simplified implementation)
fn get_memory_usage() -> usize {
    // This is a placeholder implementation
    // In practice, you would use system-specific APIs to get actual memory usage
    0
}

/// Utility functions for quick model analysis
pub mod utils {
    use super::*;

    /// Quick summary with default configuration
    pub fn quick_summary<M: Module>(model: &M, input_shape: &[usize]) -> Result<()> {
        let summary = summarize(model, input_shape, None)?;
        summary.print();
        Ok(())
    }

    /// Count total parameters in a model
    pub fn count_model_parameters<M: Module>(model: &M) -> usize {
        count_parameters(&model.parameters())
    }

    /// Get model size in MB (assuming f32 parameters)
    pub fn get_model_size_mb<M: Module>(model: &M) -> f64 {
        let param_count = count_model_parameters(model);
        (param_count * 4) as f64 / 1_048_576.0 // 4 bytes per f32, convert to MB
    }

    /// Check if model fits in given memory budget
    pub fn check_memory_budget<M: Module>(
        model: &M,
        input_shape: &[usize],
        budget_mb: f64,
    ) -> bool {
        let model_size = get_model_size_mb(model);
        let input_size = estimate_tensor_memory(input_shape, 4) as f64 / 1_048_576.0;
        let estimated_total = model_size + input_size * 2.0; // Factor for intermediate activations

        estimated_total <= budget_mb
    }
}

/// Enhanced profiling tools for comprehensive model analysis
pub mod profiling {
    use super::*;

    /// FLOPS counter for different layer types
    #[derive(Debug, Clone)]
    pub struct FLOPSCounter {
        pub total_flops: u64,
        pub layer_flops: HashMap<String, u64>,
    }

    impl FLOPSCounter {
        pub fn new() -> Self {
            Self {
                total_flops: 0,
                layer_flops: HashMap::new(),
            }
        }

        /// Estimate FLOPS for a linear layer
        pub fn count_linear_flops(
            &mut self,
            layer_name: String,
            input_size: usize,
            output_size: usize,
            batch_size: usize,
        ) {
            let flops = (batch_size * input_size * output_size * 2) as u64; // 2 for multiply-add
            self.layer_flops.insert(layer_name, flops);
            self.total_flops += flops;
        }

        /// Estimate FLOPS for a convolution layer
        pub fn count_conv_flops(
            &mut self,
            layer_name: String,
            input_shape: &[usize],
            kernel_size: &[usize],
            output_channels: usize,
        ) {
            let batch_size = input_shape[0];
            let input_channels = input_shape[1];
            let output_height = input_shape[2]; // Simplified
            let output_width = input_shape[3]; // Simplified

            let kernel_flops = kernel_size.iter().product::<usize>() * input_channels;
            let output_pixels = output_height * output_width * output_channels;
            let flops = (batch_size * output_pixels * kernel_flops * 2) as u64; // 2 for multiply-add

            self.layer_flops.insert(layer_name, flops);
            self.total_flops += flops;
        }

        /// Get formatted FLOPS string
        pub fn format_flops(flops: u64) -> String {
            if flops >= 1_000_000_000_000 {
                format!("{:.2} TFLOPS", flops as f64 / 1_000_000_000_000.0)
            } else if flops >= 1_000_000_000 {
                format!("{:.2} GFLOPS", flops as f64 / 1_000_000_000.0)
            } else if flops >= 1_000_000 {
                format!("{:.2} MFLOPS", flops as f64 / 1_000_000.0)
            } else if flops >= 1_000 {
                format!("{:.2} KFLOPS", flops as f64 / 1_000.0)
            } else {
                format!("{} FLOPS", flops)
            }
        }
    }

    /// Advanced model analyzer with detailed metrics
    #[derive(Debug, Clone)]
    pub struct ModelAnalyzer {
        pub config: AnalysisConfig,
    }

    #[derive(Debug, Clone)]
    pub struct AnalysisConfig {
        pub analyze_gradients: bool,
        pub analyze_activations: bool,
        pub analyze_flops: bool,
        pub analyze_memory: bool,
        pub batch_analysis: bool,
    }

    impl Default for AnalysisConfig {
        fn default() -> Self {
            Self {
                analyze_gradients: false,
                analyze_activations: false,
                analyze_flops: true,
                analyze_memory: true,
                batch_analysis: false,
            }
        }
    }

    impl ModelAnalyzer {
        pub fn new(config: AnalysisConfig) -> Self {
            Self { config }
        }

        pub fn default() -> Self {
            Self::new(AnalysisConfig::default())
        }

        /// Comprehensive model analysis
        pub fn analyze<M: Module>(
            &self,
            model: &M,
            input_shape: &[usize],
        ) -> Result<AnalysisReport> {
            let mut report = AnalysisReport::new();

            // Basic model information
            let parameters = model.parameters();
            report.parameter_count = count_parameters(&parameters);
            report.model_size_mb = (report.parameter_count * 4) as f64 / 1_048_576.0;

            // Memory analysis
            if self.config.analyze_memory {
                report.memory_analysis = Some(self.analyze_memory(model, input_shape)?);
            }

            // FLOPS analysis
            if self.config.analyze_flops {
                report.flops_analysis = Some(self.estimate_flops(model, input_shape)?);
            }

            Ok(report)
        }

        fn analyze_memory<M: Module>(
            &self,
            model: &M,
            input_shape: &[usize],
        ) -> Result<MemoryAnalysis> {
            let input_memory = estimate_tensor_memory(input_shape, 4);
            let param_memory = count_parameters(&model.parameters()) * 4;

            // Estimate intermediate activations (rough approximation)
            let intermediate_memory = input_memory * 3; // Rough estimate

            Ok(MemoryAnalysis {
                input_memory,
                parameter_memory: param_memory,
                intermediate_memory,
                total_memory: input_memory + param_memory + intermediate_memory,
            })
        }

        fn estimate_flops<M: Module>(
            &self,
            _model: &M,
            input_shape: &[usize],
        ) -> Result<FLOPSAnalysis> {
            // Simplified FLOPS estimation
            // In a full implementation, this would traverse the model structure
            let estimated_flops = input_shape.iter().product::<usize>() as u64 * 1000; // Rough estimate

            Ok(FLOPSAnalysis {
                total_flops: estimated_flops,
                flops_per_layer: HashMap::new(),
            })
        }
    }

    /// Memory analysis results
    #[derive(Debug, Clone)]
    pub struct MemoryAnalysis {
        pub input_memory: usize,
        pub parameter_memory: usize,
        pub intermediate_memory: usize,
        pub total_memory: usize,
    }

    /// FLOPS analysis results
    #[derive(Debug, Clone)]
    pub struct FLOPSAnalysis {
        pub total_flops: u64,
        pub flops_per_layer: HashMap<String, u64>,
    }

    /// Comprehensive analysis report
    #[derive(Debug, Clone)]
    pub struct AnalysisReport {
        pub parameter_count: usize,
        pub model_size_mb: f64,
        pub memory_analysis: Option<MemoryAnalysis>,
        pub flops_analysis: Option<FLOPSAnalysis>,
    }

    impl AnalysisReport {
        pub fn new() -> Self {
            Self {
                parameter_count: 0,
                model_size_mb: 0.0,
                memory_analysis: None,
                flops_analysis: None,
            }
        }

        /// Print detailed analysis report
        pub fn print_detailed(&self) {
            println!("=== Detailed Model Analysis ===");
            println!("Parameters: {}", format_number(self.parameter_count));
            println!("Model Size: {:.2} MB", self.model_size_mb);

            if let Some(memory) = &self.memory_analysis {
                println!("\n--- Memory Analysis ---");
                println!("Input Memory: {}", format_bytes(memory.input_memory));
                println!(
                    "Parameter Memory: {}",
                    format_bytes(memory.parameter_memory)
                );
                println!(
                    "Intermediate Memory: {}",
                    format_bytes(memory.intermediate_memory)
                );
                println!("Total Memory: {}", format_bytes(memory.total_memory));
            }

            if let Some(flops) = &self.flops_analysis {
                println!("\n--- FLOPS Analysis ---");
                println!(
                    "Total FLOPS: {}",
                    FLOPSCounter::format_flops(flops.total_flops)
                );

                if !flops.flops_per_layer.is_empty() {
                    println!("Per-layer FLOPS:");
                    for (layer, flops) in &flops.flops_per_layer {
                        println!("  {}: {}", layer, FLOPSCounter::format_flops(*flops));
                    }
                }
            }
        }
    }

    /// Batch profiler for statistical analysis
    pub struct BatchProfiler {
        config: BatchProfilingConfig,
    }

    #[derive(Debug, Clone)]
    pub struct BatchProfilingConfig {
        pub num_runs: usize,
        pub warmup_runs: usize,
        pub collect_stats: bool,
    }

    impl Default for BatchProfilingConfig {
        fn default() -> Self {
            Self {
                num_runs: 10,
                warmup_runs: 3,
                collect_stats: true,
            }
        }
    }

    impl BatchProfiler {
        pub fn new(config: BatchProfilingConfig) -> Self {
            Self { config }
        }

        /// Run batch profiling on a model
        #[cfg(feature = "std")]
        pub fn profile_batch<M: Module>(
            &self,
            model: &M,
            input: &Tensor,
        ) -> Result<BatchProfilingResult> {
            let mut times = Vec::new();

            // Warmup runs
            for _ in 0..self.config.warmup_runs {
                let _output = model.forward(input)?;
            }

            // Actual profiling runs
            for _ in 0..self.config.num_runs {
                let start = Instant::now();
                let _output = model.forward(input)?;
                let elapsed = start.elapsed();
                times.push(elapsed);
            }

            Ok(BatchProfilingResult::from_times(times))
        }
    }

    /// Results from batch profiling
    #[derive(Debug, Clone)]
    pub struct BatchProfilingResult {
        pub mean_time: f64,
        pub std_time: f64,
        pub min_time: f64,
        pub max_time: f64,
        pub median_time: f64,
        pub num_runs: usize,
    }

    #[cfg(feature = "std")]
    impl BatchProfilingResult {
        pub fn from_times(times: Vec<Duration>) -> Self {
            let times_ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

            let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
            let variance =
                times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times_ms.len() as f64;
            let std_dev = variance.sqrt();

            let mut sorted_times = times_ms.clone();
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = if sorted_times.len() % 2 == 0 {
                (sorted_times[sorted_times.len() / 2 - 1] + sorted_times[sorted_times.len() / 2])
                    / 2.0
            } else {
                sorted_times[sorted_times.len() / 2]
            };

            Self {
                mean_time: mean,
                std_time: std_dev,
                min_time: sorted_times[0],
                max_time: sorted_times[sorted_times.len() - 1],
                median_time: median,
                num_runs: times.len(),
            }
        }

        pub fn print_stats(&self) {
            println!("=== Batch Profiling Results ===");
            println!("Runs: {}", self.num_runs);
            println!("Mean: {:.3}ms", self.mean_time);
            println!("Std Dev: {:.3}ms", self.std_time);
            println!("Min: {:.3}ms", self.min_time);
            println!("Max: {:.3}ms", self.max_time);
            println!("Median: {:.3}ms", self.median_time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Linear;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_layer_info_creation() {
        let layer_info = LayerInfo::new(
            "linear1".to_string(),
            "Linear".to_string(),
            vec![10, 20],
            vec![10, 30],
            600, // 20 * 30 weights + 30 biases
            600,
        );

        assert_eq!(layer_info.name, "linear1");
        assert_eq!(layer_info.layer_type, "Linear");
        assert_eq!(layer_info.param_count, 600);
        assert_eq!(layer_info.trainable_params, 600);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1_500_000), "1.5M");
        assert_eq!(format_number(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1_572_864), "1.5 MB");
        assert_eq!(format_bytes(1_610_612_736), "1.5 GB");
    }

    #[test]
    fn test_model_summary() -> Result<()> {
        let model = Linear::new(128, 64, true);
        let input_shape = [10, 128];

        let summary = summarize(&model, &input_shape, None)?;

        assert_eq!(summary.input_shape, vec![10, 128]);
        assert!(!summary.layers.is_empty());
        assert!(summary.total_params > 0);

        Ok(())
    }

    #[test]
    fn test_model_profiler() -> Result<()> {
        let model = Linear::new(64, 32, true);
        let input = randn::<f32>(&[8, 64])?;

        let profiler = ModelProfiler::new().with_time_tracking();
        let result = profiler.profile(&model, &input)?;

        assert_eq!(result.input_shape, vec![8, 64]);
        assert_eq!(result.output_shape, vec![8, 32]);
        assert!(result.parameter_count > 0);

        Ok(())
    }

    #[test]
    fn test_utils_functions() -> Result<()> {
        let model = Linear::new(100, 50, true);

        let param_count = utils::count_model_parameters(&model);
        assert_eq!(param_count, 100 * 50 + 50); // weights + biases

        let model_size = utils::get_model_size_mb(&model);
        assert!(model_size > 0.0);

        let fits_budget = utils::check_memory_budget(&model, &[10, 100], 100.0);
        assert!(fits_budget); // Should easily fit in 100MB

        Ok(())
    }

    #[test]
    fn test_flops_counter() {
        let mut counter = profiling::FLOPSCounter::new();

        // Test linear layer FLOPS counting
        counter.count_linear_flops("linear1".to_string(), 128, 64, 32);
        assert_eq!(counter.total_flops, 32 * 128 * 64 * 2); // batch * input * output * 2

        // Test FLOPS formatting
        assert_eq!(profiling::FLOPSCounter::format_flops(1500), "1.50 KFLOPS");
        assert_eq!(
            profiling::FLOPSCounter::format_flops(1_500_000),
            "1.50 MFLOPS"
        );
        assert_eq!(
            profiling::FLOPSCounter::format_flops(1_500_000_000),
            "1.50 GFLOPS"
        );
    }

    #[test]
    fn test_model_analyzer() -> Result<()> {
        let model = Linear::new(128, 64, true);
        let input_shape = [10, 128];

        let analyzer = profiling::ModelAnalyzer::default();
        let report = analyzer.analyze(&model, &input_shape)?;

        assert!(report.parameter_count > 0);
        assert!(report.model_size_mb > 0.0);
        assert!(report.memory_analysis.is_some());
        assert!(report.flops_analysis.is_some());

        if let Some(memory) = &report.memory_analysis {
            assert!(memory.total_memory > 0);
            assert!(memory.parameter_memory > 0);
        }

        Ok(())
    }

    #[test]
    fn test_analysis_config() {
        let config = profiling::AnalysisConfig::default();
        assert!(!config.analyze_gradients);
        assert!(!config.analyze_activations);
        assert!(config.analyze_flops);
        assert!(config.analyze_memory);
        assert!(!config.batch_analysis);
    }

    #[test]
    fn test_batch_profiling_config() {
        let config = profiling::BatchProfilingConfig::default();
        assert_eq!(config.num_runs, 10);
        assert_eq!(config.warmup_runs, 3);
        assert!(config.collect_stats);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_batch_profiler() -> Result<()> {
        let model = Linear::new(64, 32, true);
        let input = randn::<f32>(&[8, 64])?;

        let config = profiling::BatchProfilingConfig {
            num_runs: 5,
            warmup_runs: 2,
            collect_stats: true,
        };

        let profiler = profiling::BatchProfiler::new(config);
        let result = profiler.profile_batch(&model, &input)?;

        assert_eq!(result.num_runs, 5);
        assert!(result.mean_time >= 0.0);
        assert!(result.std_time >= 0.0);
        assert!(result.min_time >= 0.0);
        assert!(result.max_time >= result.min_time);
        assert!(result.median_time >= 0.0);

        Ok(())
    }
}
