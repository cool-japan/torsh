//! Data structures for plotting and visualization

use std::time::SystemTime;

/// Data structure for plotting and visualization
#[derive(Debug, Clone)]
pub struct PlotData {
    /// Operation names for x-axis labels
    pub operation_names: Vec<String>,
    /// Average execution times in milliseconds
    pub avg_times: Vec<f64>,
    /// Throughput values (operations per second)
    pub throughputs: Vec<f64>,
    /// Memory usage values in bytes
    pub memory_usage: Vec<f64>,
    /// Timestamps for time-series data
    pub timestamps: Vec<SystemTime>,
}

impl PlotData {
    /// Create empty plot data
    pub fn new() -> Self {
        Self {
            operation_names: Vec::new(),
            avg_times: Vec::new(),
            throughputs: Vec::new(),
            memory_usage: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add a data point
    pub fn add_point(&mut self, operation: String, time_ms: f64, throughput: f64, memory: f64) {
        self.operation_names.push(operation);
        self.avg_times.push(time_ms);
        self.throughputs.push(throughput);
        self.memory_usage.push(memory);
        self.timestamps.push(SystemTime::now());
    }

    /// Get data for time-series plotting
    pub fn time_series_data(&self) -> Vec<(SystemTime, f64, f64, f64)> {
        self.timestamps
            .iter()
            .zip(&self.avg_times)
            .zip(&self.throughputs)
            .zip(&self.memory_usage)
            .map(|(((time, avg_time), throughput), memory)| {
                (*time, *avg_time, *throughput, *memory)
            })
            .collect()
    }
}

impl Default for PlotData {
    fn default() -> Self {
        Self::new()
    }
}
