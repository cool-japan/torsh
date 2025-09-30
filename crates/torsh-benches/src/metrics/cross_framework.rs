//! Cross-framework metrics system for unified benchmark analysis
//!
//! This module provides comprehensive cross-framework comparison capabilities
//! including standardized metrics collection, framework comparison, and reporting.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Framework identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Framework {
    Torsh,
    TensorFlow,
    JAX,
    NumPy,
    Ndarray,
    PyTorch,
}

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Framework::Torsh => write!(f, "ToRSh"),
            Framework::TensorFlow => write!(f, "TensorFlow"),
            Framework::JAX => write!(f, "JAX"),
            Framework::NumPy => write!(f, "NumPy"),
            Framework::Ndarray => write!(f, "ndarray"),
            Framework::PyTorch => write!(f, "PyTorch"),
        }
    }
}

/// Operation type for standardized comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    MatrixMultiplication,
    ElementWiseAddition,
    ElementWiseMultiplication,
    Convolution2D,
    ReLU,
    Softmax,
    BatchNormalization,
    LinearLayer,
    BackwardPass,
    MemoryAllocation,
    DataLoading,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::MatrixMultiplication => write!(f, "Matrix Multiplication"),
            OperationType::ElementWiseAddition => write!(f, "Element-wise Addition"),
            OperationType::ElementWiseMultiplication => {
                write!(f, "Element-wise Multiplication")
            }
            OperationType::Convolution2D => write!(f, "2D Convolution"),
            OperationType::ReLU => write!(f, "ReLU Activation"),
            OperationType::Softmax => write!(f, "Softmax"),
            OperationType::BatchNormalization => write!(f, "Batch Normalization"),
            OperationType::LinearLayer => write!(f, "Linear Layer"),
            OperationType::BackwardPass => write!(f, "Backward Pass"),
            OperationType::MemoryAllocation => write!(f, "Memory Allocation"),
            OperationType::DataLoading => write!(f, "Data Loading"),
        }
    }
}

/// Unified benchmark metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMetrics {
    /// Framework that produced this metric
    pub framework: Framework,

    /// Operation type
    pub operation: OperationType,

    /// Input size/dimensions
    pub input_size: Vec<usize>,

    /// Execution time in nanoseconds
    pub execution_time_ns: f64,

    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: Option<u64>,

    /// Throughput (operations per second)
    pub throughput_ops: Option<f64>,

    /// FLOPS (floating point operations per second)
    pub flops: Option<f64>,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: Option<f64>,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,

    /// Device type (CPU, GPU, etc.)
    pub device_type: String,

    /// Data type (f32, f64, etc.)
    pub data_type: String,

    /// Framework version
    pub framework_version: Option<String>,

    /// Hardware information
    pub hardware_info: Option<HardwareInfo>,
}

/// Hardware information for benchmark context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_model: Option<String>,
    pub gpu_memory_gb: Option<f64>,
}

/// Cross-framework benchmark comparison
#[derive(Debug, Clone)]
pub struct CrossFrameworkComparison {
    metrics: Vec<UnifiedMetrics>,
}

impl CrossFrameworkComparison {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    /// Add benchmark metrics
    pub fn add_metrics(&mut self, metrics: UnifiedMetrics) {
        self.metrics.push(metrics);
    }

    /// Get metrics for a specific framework and operation
    pub fn get_metrics(
        &self,
        framework: &Framework,
        operation: &OperationType,
    ) -> Vec<&UnifiedMetrics> {
        self.metrics
            .iter()
            .filter(|m| m.framework == *framework && m.operation == *operation)
            .collect()
    }

    /// Compare frameworks for a specific operation
    pub fn compare_frameworks(&self, operation: &OperationType) -> FrameworkComparison {
        let mut framework_metrics: HashMap<Framework, Vec<&UnifiedMetrics>> = HashMap::new();

        for metric in &self.metrics {
            if metric.operation == *operation {
                framework_metrics
                    .entry(metric.framework.clone())
                    .or_default()
                    .push(metric);
            }
        }

        let mut comparisons = Vec::new();
        for (framework, metrics) in framework_metrics {
            if !metrics.is_empty() {
                let avg_time =
                    metrics.iter().map(|m| m.execution_time_ns).sum::<f64>() / metrics.len() as f64;
                let avg_memory = metrics
                    .iter()
                    .filter_map(|m| m.memory_usage_bytes)
                    .map(|m| m as f64)
                    .sum::<f64>()
                    / metrics.len() as f64;
                let avg_throughput = metrics.iter().filter_map(|m| m.throughput_ops).sum::<f64>()
                    / metrics.len() as f64;

                comparisons.push(FrameworkPerformance {
                    framework: framework.clone(),
                    average_time_ns: avg_time,
                    average_memory_bytes: if avg_memory > 0.0 {
                        Some(avg_memory as u64)
                    } else {
                        None
                    },
                    average_throughput_ops: if avg_throughput > 0.0 {
                        Some(avg_throughput)
                    } else {
                        None
                    },
                    sample_count: metrics.len(),
                });
            }
        }

        // Sort by average execution time (fastest first)
        comparisons.sort_by(|a, b| {
            a.average_time_ns
                .partial_cmp(&b.average_time_ns)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        FrameworkComparison {
            operation: operation.clone(),
            performances: comparisons,
        }
    }

    /// Generate scalability analysis for different input sizes
    pub fn analyze_scalability(
        &self,
        framework: &Framework,
        operation: &OperationType,
    ) -> ScalabilityAnalysis {
        let metrics = self.get_metrics(framework, operation);

        let mut size_performance: HashMap<usize, Vec<f64>> = HashMap::new();
        for metric in metrics {
            // Use the product of dimensions as the size metric
            let size = metric.input_size.iter().product();
            size_performance
                .entry(size)
                .or_default()
                .push(metric.execution_time_ns);
        }

        let mut points = Vec::new();
        for (size, times) in size_performance {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            points.push(ScalabilityPoint {
                input_size: size,
                average_time_ns: avg_time,
                sample_count: times.len(),
            });
        }

        // Sort by input size
        points.sort_by_key(|p| p.input_size);

        ScalabilityAnalysis {
            framework: framework.clone(),
            operation: operation.clone(),
            data_points: points,
        }
    }

    /// Export metrics to JSON
    pub fn export_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.metrics)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Import metrics from JSON
    pub fn import_json(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let metrics: Vec<UnifiedMetrics> = serde_json::from_str(&json)?;
        Ok(Self { metrics })
    }

    /// Generate comprehensive performance report
    pub fn generate_comprehensive_report(&self, output_dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // HTML report
        let html_path = format!("{}/cross_framework_report.html", output_dir);
        self.generate_html_report(&html_path)?;

        // CSV export
        let csv_path = format!("{}/cross_framework_metrics.csv", output_dir);
        self.export_csv(&csv_path)?;

        // JSON export
        let json_path = format!("{}/cross_framework_metrics.json", output_dir);
        self.export_json(&json_path)?;

        // Markdown summary
        let md_path = format!("{}/cross_framework_summary.md", output_dir);
        self.generate_markdown_summary(&md_path)?;

        println!(
            "📊 Comprehensive cross-framework report generated in {}",
            output_dir
        );
        Ok(())
    }

    /// Generate HTML report with charts
    fn generate_html_report(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html><head>")?;
        writeln!(file, "<title>Cross-Framework Benchmark Report</title>")?;
        writeln!(file, "<style>")?;
        writeln!(
            file,
            "body {{ font-family: Arial, sans-serif; margin: 20px; }}"
        )?;
        writeln!(
            file,
            "table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
        )?;
        writeln!(
            file,
            "th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
        )?;
        writeln!(file, "th {{ background-color: #f2f2f2; }}")?;
        writeln!(file, ".metric {{ margin: 20px 0; }}")?;
        writeln!(file, ".framework {{ color: #333; font-weight: bold; }}")?;
        writeln!(file, "</style>")?;
        writeln!(file, "</head><body>")?;

        writeln!(file, "<h1>🚀 Cross-Framework Benchmark Report</h1>")?;
        writeln!(
            file,
            "<p>Comprehensive performance comparison across tensor computing frameworks</p>"
        )?;

        // Generate comparison tables for each operation
        let operations = [
            OperationType::MatrixMultiplication,
            OperationType::ElementWiseAddition,
            OperationType::Convolution2D,
            OperationType::ReLU,
        ];

        for operation in &operations {
            writeln!(file, "<div class='metric'>")?;
            writeln!(file, "<h2>{}</h2>", operation)?;

            let comparison = self.compare_frameworks(operation);
            if !comparison.performances.is_empty() {
                writeln!(file, "<table>")?;
                writeln!(file, "<tr><th>Framework</th><th>Avg Time (μs)</th><th>Avg Memory (MB)</th><th>Throughput (ops/s)</th><th>Samples</th></tr>")?;

                for perf in &comparison.performances {
                    writeln!(file, "<tr>")?;
                    writeln!(file, "<td class='framework'>{}</td>", perf.framework)?;
                    writeln!(file, "<td>{:.2}</td>", perf.average_time_ns / 1000.0)?;
                    writeln!(
                        file,
                        "<td>{:.2}</td>",
                        perf.average_memory_bytes
                            .map(|m| m as f64 / 1_000_000.0)
                            .unwrap_or(0.0)
                    )?;
                    writeln!(
                        file,
                        "<td>{:.2}</td>",
                        perf.average_throughput_ops.unwrap_or(0.0)
                    )?;
                    writeln!(file, "<td>{}</td>", perf.sample_count)?;
                    writeln!(file, "</tr>")?;
                }

                writeln!(file, "</table>")?;
            } else {
                writeln!(file, "<p>No data available for this operation.</p>")?;
            }
            writeln!(file, "</div>")?;
        }

        writeln!(file, "</body></html>")?;
        Ok(())
    }

    /// Export to CSV format
    fn export_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "framework,operation,input_size,execution_time_ns,memory_usage_bytes,throughput_ops,device_type,data_type")?;

        for metric in &self.metrics {
            writeln!(
                file,
                "{},{},{:?},{},{},{},{},{}",
                metric.framework,
                metric.operation,
                metric.input_size,
                metric.execution_time_ns,
                metric.memory_usage_bytes.unwrap_or(0),
                metric.throughput_ops.unwrap_or(0.0),
                metric.device_type,
                metric.data_type
            )?;
        }

        Ok(())
    }

    /// Generate markdown summary
    fn generate_markdown_summary(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "# Cross-Framework Benchmark Summary\n")?;

        let operations = [
            OperationType::MatrixMultiplication,
            OperationType::ElementWiseAddition,
            OperationType::Convolution2D,
            OperationType::ReLU,
        ];

        for operation in &operations {
            writeln!(file, "## {}\n", operation)?;

            let comparison = self.compare_frameworks(operation);
            if !comparison.performances.is_empty() {
                writeln!(
                    file,
                    "| Framework | Avg Time (μs) | Relative Performance | Samples |"
                )?;
                writeln!(
                    file,
                    "|-----------|---------------|---------------------|---------|"
                )?;

                let fastest_time = comparison.performances[0].average_time_ns;
                for perf in &comparison.performances {
                    let relative_perf = fastest_time / perf.average_time_ns;
                    writeln!(
                        file,
                        "| {} | {:.2} | {:.2}x | {} |",
                        perf.framework,
                        perf.average_time_ns / 1000.0,
                        relative_perf,
                        perf.sample_count
                    )?;
                }
                writeln!(file)?;
            } else {
                writeln!(file, "No data available for this operation.\n")?;
            }
        }

        Ok(())
    }

    /// Get all unique frameworks in the dataset
    pub fn get_frameworks(&self) -> Vec<Framework> {
        let mut frameworks: Vec<Framework> = self
            .metrics
            .iter()
            .map(|m| m.framework.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        frameworks.sort_by_key(|f| format!("{}", f));
        frameworks
    }

    /// Get all unique operations in the dataset
    pub fn get_operations(&self) -> Vec<OperationType> {
        let mut operations: Vec<OperationType> = self
            .metrics
            .iter()
            .map(|m| m.operation.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        operations.sort_by_key(|o| format!("{}", o));
        operations
    }
}

impl Default for CrossFrameworkComparison {
    fn default() -> Self {
        Self::new()
    }
}

/// Framework performance summary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrameworkPerformance {
    pub framework: Framework,
    pub average_time_ns: f64,
    pub average_memory_bytes: Option<u64>,
    pub average_throughput_ops: Option<f64>,
    pub sample_count: usize,
}

/// Comparison between frameworks for a specific operation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrameworkComparison {
    pub operation: OperationType,
    pub performances: Vec<FrameworkPerformance>,
}

/// Scalability analysis data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalabilityAnalysis {
    pub framework: Framework,
    pub operation: OperationType,
    pub data_points: Vec<ScalabilityPoint>,
}

/// Single point in scalability analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalabilityPoint {
    pub input_size: usize,
    pub average_time_ns: f64,
    pub sample_count: usize,
}

/// Utility functions for converting benchmark results to unified metrics
pub mod converters {
    use super::*;

    /// Convert ToRSh result to UnifiedMetrics (simplified - would need actual BenchResult type)
    pub fn create_default_unified_metrics(
        framework: Framework,
        operation: OperationType,
        execution_time_ns: f64,
        input_size: Vec<usize>,
    ) -> UnifiedMetrics {
        UnifiedMetrics {
            framework,
            operation,
            input_size,
            execution_time_ns,
            memory_usage_bytes: None,
            peak_memory_bytes: None,
            throughput_ops: None,
            flops: None,
            memory_bandwidth_gbps: None,
            custom_metrics: HashMap::new(),
            device_type: "CPU".to_string(),
            data_type: "f32".to_string(),
            framework_version: None,
            hardware_info: None,
        }
    }
}
