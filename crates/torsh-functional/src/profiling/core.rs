//! Core profiling types and functionality
//!
//! This module provides the basic profiling infrastructure including
//! OperationMetrics and the main Profiler struct.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Performance metrics for an operation
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Name of the operation
    pub name: String,
    /// Execution time
    pub duration: Duration,
    /// Peak memory usage during operation (in bytes)
    pub peak_memory: Option<usize>,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Number of floating-point operations (estimated)
    pub flops: Option<u64>,
    /// Memory bandwidth utilization (bytes/second)
    pub memory_bandwidth: Option<f64>,
    /// CPU utilization percentage
    pub cpu_utilization: Option<f32>,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl OperationMetrics {
    /// Create new operation metrics
    pub fn new(name: String) -> Self {
        Self {
            name,
            duration: Duration::default(),
            peak_memory: None,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            flops: None,
            memory_bandwidth: None,
            cpu_utilization: None,
            custom_metrics: HashMap::new(),
        }
    }

    /// Add a custom metric
    pub fn add_metric(&mut self, key: String, value: f64) {
        self.custom_metrics.insert(key, value);
    }

    /// Get throughput in operations per second
    pub fn throughput(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            1.0 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get FLOPS (floating-point operations per second)
    pub fn flops_per_second(&self) -> Option<f64> {
        self.flops
            .map(|flops| flops as f64 / self.duration.as_secs_f64())
    }

    /// Get memory efficiency (fraction of peak bandwidth utilized)
    pub fn memory_efficiency(&self, peak_bandwidth_gbps: f64) -> Option<f64> {
        self.memory_bandwidth
            .map(|bw| bw / (peak_bandwidth_gbps * 1e9))
    }
}

/// Performance profiler for tracking operation metrics
pub struct Profiler {
    /// Collected metrics
    pub metrics: Vec<OperationMetrics>,
    /// Currently active profiling session
    current_session: Option<ProfilingSession>,
    /// Enable detailed memory tracking
    track_memory: bool,
    /// Enable FLOPS counting
    count_flops: bool,
    /// Custom profiling hooks
    hooks: Vec<Box<dyn Fn(&OperationMetrics) + Send + Sync>>,
}

#[derive(Debug)]
struct ProfilingSession {
    name: String,
    start_time: Instant,
    input_shapes: Vec<Vec<usize>>,
    initial_memory: Option<usize>,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            current_session: None,
            track_memory: false,
            count_flops: false,
            hooks: Vec::new(),
        }
    }

    /// Enable memory tracking
    pub fn enable_memory_tracking(&mut self) {
        self.track_memory = true;
    }

    /// Enable FLOPS counting
    pub fn enable_flops_counting(&mut self) {
        self.count_flops = true;
    }

    /// Add a profiling hook
    pub fn add_hook<F>(&mut self, hook: F)
    where
        F: Fn(&OperationMetrics) + Send + Sync + 'static,
    {
        self.hooks.push(Box::new(hook));
    }

    /// Start profiling an operation
    pub fn start_operation(&mut self, name: &str, inputs: &[&Tensor]) -> TorshResult<()> {
        if self.current_session.is_some() {
            return Err(TorshError::invalid_argument_with_context(
                "Cannot start operation while another is in progress",
                "Profiler::start_operation",
            ));
        }

        let input_shapes: Vec<Vec<usize>> =
            inputs.iter().map(|t| t.shape().dims().to_vec()).collect();

        let initial_memory = if self.track_memory {
            Some(get_current_memory_usage())
        } else {
            None
        };

        self.current_session = Some(ProfilingSession {
            name: name.to_string(),
            start_time: Instant::now(),
            input_shapes,
            initial_memory,
        });

        Ok(())
    }

    /// Finish profiling an operation
    pub fn finish_operation(&mut self, outputs: &[&Tensor]) -> TorshResult<()> {
        let session = self.current_session.take().ok_or_else(|| {
            TorshError::invalid_argument_with_context(
                "No operation in progress",
                "Profiler::finish_operation",
            )
        })?;

        let duration = session.start_time.elapsed();
        let output_shapes: Vec<Vec<usize>> =
            outputs.iter().map(|t| t.shape().dims().to_vec()).collect();

        let peak_memory = if self.track_memory {
            Some(get_current_memory_usage().saturating_sub(session.initial_memory.unwrap_or(0)))
        } else {
            None
        };

        let flops = if self.count_flops {
            Some(estimate_flops(
                &session.name,
                &session.input_shapes,
                &output_shapes,
            ))
        } else {
            None
        };

        let memory_bandwidth =
            calculate_memory_bandwidth(&session.input_shapes, &output_shapes, duration);

        let metrics = OperationMetrics {
            name: session.name,
            duration,
            peak_memory,
            input_shapes: session.input_shapes,
            output_shapes,
            flops,
            memory_bandwidth: Some(memory_bandwidth),
            cpu_utilization: None, // TODO: Implement CPU utilization tracking
            custom_metrics: HashMap::new(),
        };

        // Call hooks
        for hook in &self.hooks {
            hook(&metrics);
        }

        self.metrics.push(metrics);
        Ok(())
    }

    /// Get metrics for a specific operation
    pub fn get_metrics(&self, operation_name: &str) -> Vec<&OperationMetrics> {
        self.metrics
            .iter()
            .filter(|m| m.name == operation_name)
            .collect()
    }

    /// Get summary statistics for an operation
    pub fn get_summary(&self, operation_name: &str) -> Option<OperationSummary> {
        let metrics: Vec<_> = self.get_metrics(operation_name);
        if metrics.is_empty() {
            return None;
        }

        let count = metrics.len();
        let durations: Vec<f64> = metrics.iter().map(|m| m.duration.as_secs_f64()).collect();

        let mean_duration = durations.iter().sum::<f64>() / count as f64;
        let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_duration = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate standard deviation
        let variance = durations
            .iter()
            .map(|d| (d - mean_duration).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_duration = variance.sqrt();

        let total_flops: Option<u64> = metrics
            .iter()
            .try_fold(0u64, |acc, m| m.flops.map(|f| acc + f));

        let mean_throughput = metrics.iter().map(|m| m.throughput()).sum::<f64>() / count as f64;

        Some(OperationSummary {
            operation_name: operation_name.to_string(),
            count,
            mean_duration,
            std_duration,
            min_duration,
            max_duration,
            total_flops,
            mean_throughput,
        })
    }

    /// Clear all collected metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Export metrics to CSV format
    pub fn export_csv(&self) -> String {
        let mut csv = String::from(
            "operation,duration_ms,peak_memory_mb,input_shapes,output_shapes,flops,throughput\n",
        );

        for metric in &self.metrics {
            let input_shapes_str = format!("{:?}", metric.input_shapes);
            let output_shapes_str = format!("{:?}", metric.output_shapes);
            let peak_memory_mb = metric
                .peak_memory
                .map(|m| m as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);

            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                metric.name,
                metric.duration.as_millis(),
                peak_memory_mb,
                input_shapes_str,
                output_shapes_str,
                metric.flops.unwrap_or(0),
                metric.throughput()
            ));
        }

        csv
    }
}

/// Summary statistics for an operation
#[derive(Debug, Clone)]
pub struct OperationSummary {
    pub operation_name: String,
    pub count: usize,
    pub mean_duration: f64,
    pub std_duration: f64,
    pub min_duration: f64,
    pub max_duration: f64,
    pub total_flops: Option<u64>,
    pub mean_throughput: f64,
}

// Helper functions
pub fn get_current_memory_usage() -> usize {
    // Simplified memory usage tracking
    // In a real implementation, this would use platform-specific APIs
    0
}

pub fn estimate_flops(
    operation: &str,
    input_shapes: &[Vec<usize>],
    output_shapes: &[Vec<usize>],
) -> u64 {
    match operation {
        "matmul" | "bmm" => {
            if input_shapes.len() >= 2 {
                let a_shape = &input_shapes[0];
                let b_shape = &input_shapes[1];
                if a_shape.len() >= 2 && b_shape.len() >= 2 {
                    let m = a_shape[a_shape.len() - 2];
                    let k = a_shape[a_shape.len() - 1];
                    let n = b_shape[b_shape.len() - 1];
                    let batch_size = a_shape.iter().take(a_shape.len() - 2).product::<usize>();
                    return (2 * m * k * n * batch_size) as u64;
                }
            }
        }
        "conv2d" => {
            if !input_shapes.is_empty() && !output_shapes.is_empty() {
                let output_elements: usize = output_shapes[0].iter().product();
                // Rough estimate: 2 operations per output element per filter weight
                return (output_elements * 9 * 2) as u64; // Assuming 3x3 kernel
            }
        }
        "add" | "sub" | "mul" | "div" => {
            if !output_shapes.is_empty() {
                let elements: usize = output_shapes[0].iter().product();
                return elements as u64;
            }
        }
        _ => {}
    }
    0
}

fn calculate_memory_bandwidth(
    input_shapes: &[Vec<usize>],
    output_shapes: &[Vec<usize>],
    duration: Duration,
) -> f64 {
    let input_elements: usize = input_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .sum();
    let output_elements: usize = output_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .sum();

    let total_bytes = (input_elements + output_elements) * 4; // Assume f32
    total_bytes as f64 / duration.as_secs_f64()
}
