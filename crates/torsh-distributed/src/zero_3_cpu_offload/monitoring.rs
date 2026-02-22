//! Monitoring and Performance Tracking for ZeRO-3 CPU Offloading
//!
//! This module provides comprehensive performance statistics, memory tracking,
//! and monitoring capabilities for ZeRO-3 distributed training. It implements
//! real-time metrics collection, historical trend analysis, and detailed
//! performance profiling for optimization and debugging.

use std::collections::{HashMap, VecDeque};
use log::{debug, info, warn};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Comprehensive performance statistics for ZeRO-3 training
///
/// Tracks all aspects of training performance including forward/backward passes,
/// optimizer steps, memory transfers, and layer-specific timings.
#[derive(Debug, Clone)]
pub struct Zero3PerformanceStats {
    pub forward_passes: u64,
    pub backward_passes: u64,
    pub optimizer_steps: u64,
    pub total_forward_time: Duration,
    pub total_backward_time: Duration,
    pub total_optimizer_time: Duration,
    pub parameter_transfer_time: Duration,
    pub gradient_sync_time: Duration,
    pub layer_timings: HashMap<String, Duration>,
    pub tokens_per_second: f64,
    pub memory_transfer_bandwidth_gbps: f64,
    pub prefetch_hit_rate: f64,
    pub cache_hit_rate: f64,
    pub compression_efficiency: f64,
    pub communication_overhead: Duration,
    pub memory_optimization_time: Duration,
}

impl Zero3PerformanceStats {
    /// Create new performance statistics
    pub fn new() -> Self {
        Self {
            forward_passes: 0,
            backward_passes: 0,
            optimizer_steps: 0,
            total_forward_time: Duration::ZERO,
            total_backward_time: Duration::ZERO,
            total_optimizer_time: Duration::ZERO,
            parameter_transfer_time: Duration::ZERO,
            gradient_sync_time: Duration::ZERO,
            layer_timings: HashMap::new(),
            tokens_per_second: 0.0,
            memory_transfer_bandwidth_gbps: 0.0,
            prefetch_hit_rate: 0.0,
            cache_hit_rate: 0.0,
            compression_efficiency: 1.0,
            communication_overhead: Duration::ZERO,
            memory_optimization_time: Duration::ZERO,
        }
    }

    /// Record a forward pass
    pub fn record_forward_pass(&mut self, duration: Duration, num_tokens: usize) {
        self.forward_passes += 1;
        self.total_forward_time += duration;

        if !self.total_forward_time.is_zero() {
            self.tokens_per_second = (num_tokens as f64 * self.forward_passes as f64)
                / self.total_forward_time.as_secs_f64();
        }
    }

    /// Record a backward pass
    pub fn record_backward_pass(&mut self, duration: Duration, _num_tokens: usize) {
        self.backward_passes += 1;
        self.total_backward_time += duration;
    }

    /// Record an optimizer step
    pub fn record_optimizer_step(&mut self, duration: Duration, _num_params: usize) {
        self.optimizer_steps += 1;
        self.total_optimizer_time += duration;
    }

    /// Record layer execution time
    pub fn record_layer_execution(&mut self, layer_name: String, duration: Duration) {
        *self
            .layer_timings
            .entry(layer_name)
            .or_insert(Duration::ZERO) += duration;
    }

    /// Record layer backward execution time
    pub fn record_layer_backward(&mut self, layer_name: String, duration: Duration) {
        let key = format!("{}_backward", layer_name);
        *self.layer_timings.entry(key).or_insert(Duration::ZERO) += duration;
    }

    /// Record parameter transfer time and bandwidth
    pub fn record_parameter_transfer(&mut self, duration: Duration, bytes_transferred: usize) {
        self.parameter_transfer_time += duration;

        if !duration.is_zero() {
            let gbps = (bytes_transferred as f64 * 8.0) / (duration.as_secs_f64() * 1_000_000_000.0);
            self.memory_transfer_bandwidth_gbps =
                (self.memory_transfer_bandwidth_gbps + gbps) / 2.0; // Running average
        }
    }

    /// Record gradient synchronization time
    pub fn record_gradient_sync(&mut self, duration: Duration) {
        self.gradient_sync_time += duration;
    }

    /// Record communication overhead
    pub fn record_communication_overhead(&mut self, duration: Duration) {
        self.communication_overhead += duration;
    }

    /// Record memory optimization operation
    pub fn record_memory_optimization(&mut self, duration: Duration) {
        self.memory_optimization_time += duration;
    }

    /// Update prefetch hit rate
    pub fn update_prefetch_hit_rate(&mut self, hits: u64, total: u64) {
        if total > 0 {
            self.prefetch_hit_rate = hits as f64 / total as f64;
        }
    }

    /// Update cache hit rate
    pub fn update_cache_hit_rate(&mut self, hits: u64, total: u64) {
        if total > 0 {
            self.cache_hit_rate = hits as f64 / total as f64;
        }
    }

    /// Update compression efficiency
    pub fn update_compression_efficiency(&mut self, original_size: usize, compressed_size: usize) {
        if original_size > 0 {
            self.compression_efficiency = 1.0 - (compressed_size as f64 / original_size as f64);
        }
    }

    /// Get average forward pass time
    pub fn average_forward_time(&self) -> Duration {
        if self.forward_passes > 0 {
            self.total_forward_time / self.forward_passes as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get average backward pass time
    pub fn average_backward_time(&self) -> Duration {
        if self.backward_passes > 0 {
            self.total_backward_time / self.backward_passes as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get average optimizer step time
    pub fn average_optimizer_time(&self) -> Duration {
        if self.optimizer_steps > 0 {
            self.total_optimizer_time / self.optimizer_steps as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get total training time
    pub fn total_training_time(&self) -> Duration {
        self.total_forward_time + self.total_backward_time + self.total_optimizer_time
    }

    /// Get efficiency ratio (training vs overhead)
    pub fn efficiency_ratio(&self) -> f64 {
        let training_time = self.total_training_time();
        let overhead_time = self.parameter_transfer_time + self.gradient_sync_time +
                           self.communication_overhead + self.memory_optimization_time;

        if !training_time.is_zero() {
            training_time.as_secs_f64() / (training_time + overhead_time).as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get the slowest layer
    pub fn slowest_layer(&self) -> Option<(String, Duration)> {
        self.layer_timings.iter()
            .max_by_key(|(_, &duration)| duration)
            .map(|(name, &duration)| (name.clone(), duration))
    }

    /// Get top N slowest layers
    pub fn top_slowest_layers(&self, n: usize) -> Vec<(String, Duration)> {
        let mut layers: Vec<_> = self.layer_timings.iter()
            .map(|(name, &duration)| (name.clone(), duration))
            .collect();

        layers.sort_by_key(|(_, duration)| *duration);
        layers.reverse();
        layers.truncate(n);
        layers
    }
}

impl Default for Zero3PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced memory statistics for ZeRO-3 with detailed tracking
#[derive(Debug, Clone)]
pub struct Zero3MemoryStats {
    pub cpu_memory_used: usize,
    pub gpu_memory_used: usize,
    pub total_parameters: usize,
    pub parameters_on_cpu: usize,
    pub parameters_on_gpu: usize,
    pub compression_ratio: f32,
    pub peak_cpu_memory: usize,
    pub peak_gpu_memory: usize,
    pub memory_fragmentation: f32,
    pub cache_memory_used: usize,
    pub buffer_memory_used: usize,
    pub optimizer_state_memory: usize,
    pub gradient_memory_used: usize,
}

impl Zero3MemoryStats {
    /// Create new memory statistics
    pub fn new() -> Self {
        Self {
            cpu_memory_used: 0,
            gpu_memory_used: 0,
            total_parameters: 0,
            parameters_on_cpu: 0,
            parameters_on_gpu: 0,
            compression_ratio: 1.0,
            peak_cpu_memory: 0,
            peak_gpu_memory: 0,
            memory_fragmentation: 0.0,
            cache_memory_used: 0,
            buffer_memory_used: 0,
            optimizer_state_memory: 0,
            gradient_memory_used: 0,
        }
    }

    /// Update memory usage and track peaks
    pub fn update_memory_usage(&mut self, cpu_used: usize, gpu_used: usize) {
        self.cpu_memory_used = cpu_used;
        self.gpu_memory_used = gpu_used;
        self.peak_cpu_memory = self.peak_cpu_memory.max(cpu_used);
        self.peak_gpu_memory = self.peak_gpu_memory.max(gpu_used);
    }

    /// Update parameter distribution
    pub fn update_parameter_distribution(&mut self, cpu_params: usize, gpu_params: usize) {
        self.parameters_on_cpu = cpu_params;
        self.parameters_on_gpu = gpu_params;
        self.total_parameters = cpu_params + gpu_params;
    }

    /// Update memory fragmentation
    pub fn update_fragmentation(&mut self, fragmentation: f32) {
        self.memory_fragmentation = fragmentation;
    }

    /// Get total memory used across all devices
    pub fn total_memory_used(&self) -> usize {
        self.cpu_memory_used + self.gpu_memory_used
    }

    /// Get memory distribution ratio (GPU / Total)
    pub fn gpu_memory_ratio(&self) -> f32 {
        let total = self.total_memory_used();
        if total == 0 {
            0.0
        } else {
            self.gpu_memory_used as f32 / total as f32
        }
    }

    /// Get parameter distribution ratio (GPU / Total)
    pub fn gpu_parameter_ratio(&self) -> f32 {
        if self.total_parameters == 0 {
            0.0
        } else {
            self.parameters_on_gpu as f32 / self.total_parameters as f32
        }
    }

    /// Get memory efficiency (actual usage vs peak)
    pub fn memory_efficiency(&self) -> f32 {
        let current_total = self.total_memory_used();
        let peak_total = self.peak_cpu_memory + self.peak_gpu_memory;

        if peak_total == 0 {
            1.0
        } else {
            current_total as f32 / peak_total as f32
        }
    }

    /// Get detailed memory breakdown
    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            parameters: self.parameters_on_cpu + self.parameters_on_gpu,
            gradients: self.gradient_memory_used,
            optimizer_states: self.optimizer_state_memory,
            cache: self.cache_memory_used,
            buffers: self.buffer_memory_used,
            other: self.total_memory_used().saturating_sub(
                self.gradient_memory_used + self.optimizer_state_memory +
                self.cache_memory_used + self.buffer_memory_used
            ),
        }
    }
}

impl Default for Zero3MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Breakdown of memory usage by category
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    pub parameters: usize,
    pub gradients: usize,
    pub optimizer_states: usize,
    pub cache: usize,
    pub buffers: usize,
    pub other: usize,
}

/// Performance monitor for ZeRO-3 training
///
/// Provides real-time monitoring, historical tracking, and trend analysis
/// for comprehensive performance insights.
pub struct Zero3PerformanceMonitor {
    current_stats: Arc<Mutex<Zero3PerformanceStats>>,
    memory_stats: Arc<Mutex<Zero3MemoryStats>>,
    historical_data: Arc<Mutex<HistoricalData>>,
    monitoring_enabled: bool,
    sample_interval: Duration,
    last_sample_time: Mutex<Instant>,
}

impl Zero3PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(sample_interval: Duration) -> Self {
        info!(" Performance Monitor initialized with {:?} sample interval", sample_interval);

        Self {
            current_stats: Arc::new(Mutex::new(Zero3PerformanceStats::new())),
            memory_stats: Arc::new(Mutex::new(Zero3MemoryStats::new())),
            historical_data: Arc::new(Mutex::new(HistoricalData::new())),
            monitoring_enabled: true,
            sample_interval,
            last_sample_time: Mutex::new(Instant::now()),
        }
    }

    /// Enable or disable monitoring
    pub fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
        info!(" Performance monitoring {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Record training step performance
    pub fn record_training_step(&self, step_stats: TrainingStepStats) {
        if !self.monitoring_enabled {
            return;
        }

        // Update current statistics
        {
            let mut stats = self.current_stats.lock().expect("lock should not be poisoned");

            if let Some(forward_time) = step_stats.forward_time {
                stats.record_forward_pass(forward_time, step_stats.num_tokens);
            }

            if let Some(backward_time) = step_stats.backward_time {
                stats.record_backward_pass(backward_time, step_stats.num_tokens);
            }

            if let Some(optimizer_time) = step_stats.optimizer_time {
                stats.record_optimizer_step(optimizer_time, step_stats.num_parameters);
            }

            if let Some((duration, bytes)) = step_stats.parameter_transfer {
                stats.record_parameter_transfer(duration, bytes);
            }

            if let Some(sync_time) = step_stats.gradient_sync_time {
                stats.record_gradient_sync(sync_time);
            }

            for (layer_name, duration) in step_stats.layer_timings {
                stats.record_layer_execution(layer_name, duration);
            }
        }

        // Sample historical data if enough time has passed
        self.maybe_sample_historical_data();
    }

    /// Update memory statistics
    pub fn update_memory_stats(&self, memory_stats: Zero3MemoryStats) {
        if !self.monitoring_enabled {
            return;
        }

        {
            let mut stats = self.memory_stats.lock().expect("lock should not be poisoned");
            *stats = memory_stats;
        }

        self.maybe_sample_historical_data();
    }

    /// Sample current statistics into historical data
    fn maybe_sample_historical_data(&self) {
        let mut last_sample = self.last_sample_time.lock().expect("lock should not be poisoned");
        let now = Instant::now();

        if now.duration_since(*last_sample) >= self.sample_interval {
            let current_perf = self.current_stats.lock().expect("lock should not be poisoned").clone();
            let current_mem = self.memory_stats.lock().expect("lock should not be poisoned").clone();

            {
                let mut historical = self.historical_data.lock().expect("lock should not be poisoned");
                historical.add_sample(HistoricalSample {
                    timestamp: now,
                    performance_stats: current_perf,
                    memory_stats: current_mem,
                });
            }

            *last_sample = now;
        }
    }

    /// Get current performance statistics
    pub fn get_current_stats(&self) -> Zero3PerformanceStats {
        self.current_stats.lock().expect("lock should not be poisoned").clone()
    }

    /// Get current memory statistics
    pub fn get_current_memory_stats(&self) -> Zero3MemoryStats {
        self.memory_stats.lock().expect("lock should not be poisoned").clone()
    }

    /// Get performance trends over time
    pub fn get_performance_trends(&self, duration: Duration) -> PerformanceTrends {
        let historical = self.historical_data.lock().expect("lock should not be poisoned");
        historical.analyze_trends(duration)
    }

    /// Get comprehensive monitoring report
    pub fn get_monitoring_report(&self) -> MonitoringReport {
        let current_perf = self.get_current_stats();
        let current_mem = self.get_current_memory_stats();
        let trends = self.get_performance_trends(Duration::from_secs(300)); // Last 5 minutes

        MonitoringReport {
            timestamp: Instant::now(),
            performance_stats: current_perf,
            memory_stats: current_mem,
            trends,
            monitoring_enabled: self.monitoring_enabled,
            sample_interval: self.sample_interval,
        }
    }

    /// Reset all statistics
    pub fn reset_stats(&self) {
        if self.monitoring_enabled {
            {
                let mut stats = self.current_stats.lock().expect("lock should not be poisoned");
                *stats = Zero3PerformanceStats::new();
            }

            {
                let mut stats = self.memory_stats.lock().expect("lock should not be poisoned");
                *stats = Zero3MemoryStats::new();
            }

            {
                let mut historical = self.historical_data.lock().expect("lock should not be poisoned");
                historical.clear();
            }

            info!(" Performance statistics reset");
        }
    }

    /// Get monitoring statistics
    pub fn get_monitor_stats(&self) -> MonitorStats {
        let historical = self.historical_data.lock().expect("lock should not be poisoned");

        MonitorStats {
            monitoring_enabled: self.monitoring_enabled,
            sample_interval: self.sample_interval,
            historical_samples: historical.sample_count(),
            memory_usage: std::mem::size_of::<Self>() + historical.memory_usage_estimate(),
        }
    }
}

/// Statistics for a single training step
#[derive(Debug, Clone)]
pub struct TrainingStepStats {
    pub forward_time: Option<Duration>,
    pub backward_time: Option<Duration>,
    pub optimizer_time: Option<Duration>,
    pub parameter_transfer: Option<(Duration, usize)>, // (duration, bytes)
    pub gradient_sync_time: Option<Duration>,
    pub layer_timings: HashMap<String, Duration>,
    pub num_tokens: usize,
    pub num_parameters: usize,
}

impl Default for TrainingStepStats {
    fn default() -> Self {
        Self {
            forward_time: None,
            backward_time: None,
            optimizer_time: None,
            parameter_transfer: None,
            gradient_sync_time: None,
            layer_timings: HashMap::new(),
            num_tokens: 0,
            num_parameters: 0,
        }
    }
}

/// Historical data storage and analysis
struct HistoricalData {
    samples: VecDeque<HistoricalSample>,
    max_samples: usize,
}

impl HistoricalData {
    fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: 1000, // Keep last 1000 samples
        }
    }

    fn add_sample(&mut self, sample: HistoricalSample) {
        self.samples.push_back(sample);

        // Remove old samples if we exceed the limit
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn clear(&mut self) {
        self.samples.clear();
    }

    fn memory_usage_estimate(&self) -> usize {
        self.samples.len() * std::mem::size_of::<HistoricalSample>()
    }

    fn analyze_trends(&self, duration: Duration) -> PerformanceTrends {
        let cutoff_time = Instant::now() - duration;
        let recent_samples: Vec<_> = self.samples.iter()
            .filter(|sample| sample.timestamp >= cutoff_time)
            .collect();

        if recent_samples.is_empty() {
            return PerformanceTrends::default();
        }

        // Calculate trends
        let forward_times: Vec<f64> = recent_samples.iter()
            .map(|s| s.performance_stats.average_forward_time().as_secs_f64())
            .collect();

        let backward_times: Vec<f64> = recent_samples.iter()
            .map(|s| s.performance_stats.average_backward_time().as_secs_f64())
            .collect();

        let memory_usage: Vec<f64> = recent_samples.iter()
            .map(|s| s.memory_stats.total_memory_used() as f64)
            .collect();

        PerformanceTrends {
            forward_time_trend: calculate_trend(&forward_times),
            backward_time_trend: calculate_trend(&backward_times),
            memory_usage_trend: calculate_trend(&memory_usage),
            sample_count: recent_samples.len(),
            duration,
        }
    }
}

/// A single historical sample
#[derive(Debug, Clone)]
struct HistoricalSample {
    timestamp: Instant,
    performance_stats: Zero3PerformanceStats,
    memory_stats: Zero3MemoryStats,
}

/// Performance trends over time
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    pub forward_time_trend: TrendDirection,
    pub backward_time_trend: TrendDirection,
    pub memory_usage_trend: TrendDirection,
    pub sample_count: usize,
    pub duration: Duration,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            forward_time_trend: TrendDirection::Stable,
            backward_time_trend: TrendDirection::Stable,
            memory_usage_trend: TrendDirection::Stable,
            sample_count: 0,
            duration: Duration::ZERO,
        }
    }
}

/// Direction of a performance trend
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,  // Values getting better (faster times, less memory)
    Degrading,  // Values getting worse (slower times, more memory)
    Stable,     // Values staying roughly the same
}

/// Calculate trend direction from a series of values
fn calculate_trend(values: &[f64]) -> TrendDirection {
    if values.len() < 2 {
        return TrendDirection::Stable;
    }

    let first_half = &values[..values.len() / 2];
    let second_half = &values[values.len() / 2..];

    let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
    let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;

    let change_ratio = (second_avg - first_avg) / first_avg.max(1e-10);

    if change_ratio > 0.05 {
        TrendDirection::Degrading
    } else if change_ratio < -0.05 {
        TrendDirection::Improving
    } else {
        TrendDirection::Stable
    }
}

/// Comprehensive monitoring report
#[derive(Debug, Clone)]
pub struct MonitoringReport {
    pub timestamp: Instant,
    pub performance_stats: Zero3PerformanceStats,
    pub memory_stats: Zero3MemoryStats,
    pub trends: PerformanceTrends,
    pub monitoring_enabled: bool,
    pub sample_interval: Duration,
}

/// Statistics about the monitor itself
#[derive(Debug, Clone)]
pub struct MonitorStats {
    pub monitoring_enabled: bool,
    pub sample_interval: Duration,
    pub historical_samples: usize,
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_stats() {
        let mut stats = Zero3PerformanceStats::new();

        // Test recording operations
        stats.record_forward_pass(Duration::from_millis(100), 1000);
        stats.record_backward_pass(Duration::from_millis(150), 1000);
        stats.record_optimizer_step(Duration::from_millis(50), 100000);

        assert_eq!(stats.forward_passes, 1);
        assert_eq!(stats.backward_passes, 1);
        assert_eq!(stats.optimizer_steps, 1);
        assert!(stats.tokens_per_second > 0.0);

        // Test averages
        assert_eq!(stats.average_forward_time(), Duration::from_millis(100));
        assert_eq!(stats.average_backward_time(), Duration::from_millis(150));
        assert_eq!(stats.average_optimizer_time(), Duration::from_millis(50));

        // Test layer timings
        stats.record_layer_execution("layer1".to_string(), Duration::from_millis(20));
        stats.record_layer_execution("layer2".to_string(), Duration::from_millis(30));

        let slowest = stats.slowest_layer().unwrap();
        assert_eq!(slowest.0, "layer2");
        assert_eq!(slowest.1, Duration::from_millis(30));

        let top_layers = stats.top_slowest_layers(2);
        assert_eq!(top_layers.len(), 2);
        assert_eq!(top_layers[0].0, "layer2");
        assert_eq!(top_layers[1].0, "layer1");
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = Zero3MemoryStats::new();

        stats.update_memory_usage(1000, 2000);
        stats.update_parameter_distribution(500, 1500);

        assert_eq!(stats.total_memory_used(), 3000);
        assert_eq!(stats.gpu_memory_ratio(), 2.0 / 3.0);
        assert_eq!(stats.gpu_parameter_ratio(), 0.75);

        // Test peak tracking
        stats.update_memory_usage(800, 2500);
        assert_eq!(stats.peak_cpu_memory, 1000);
        assert_eq!(stats.peak_gpu_memory, 2500);

        let breakdown = stats.memory_breakdown();
        assert_eq!(breakdown.parameters, 2000);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = Zero3PerformanceMonitor::new(Duration::from_millis(100));

        let step_stats = TrainingStepStats {
            forward_time: Some(Duration::from_millis(50)),
            backward_time: Some(Duration::from_millis(75)),
            num_tokens: 100,
            num_parameters: 10000,
            ..Default::default()
        };

        monitor.record_training_step(step_stats);

        let current_stats = monitor.get_current_stats();
        assert_eq!(current_stats.forward_passes, 1);
        assert_eq!(current_stats.backward_passes, 1);

        let monitor_stats = monitor.get_monitor_stats();
        assert!(monitor_stats.monitoring_enabled);
        assert_eq!(monitor_stats.sample_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_trend_calculation() {
        // Test improving trend
        let improving_values = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0];
        assert_eq!(calculate_trend(&improving_values), TrendDirection::Improving);

        // Test degrading trend
        let degrading_values = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(calculate_trend(&degrading_values), TrendDirection::Degrading);

        // Test stable trend
        let stable_values = vec![5.0, 5.1, 4.9, 5.0, 5.1, 4.9];
        assert_eq!(calculate_trend(&stable_values), TrendDirection::Stable);

        // Test empty values
        let empty_values = vec![];
        assert_eq!(calculate_trend(&empty_values), TrendDirection::Stable);
    }

    #[test]
    fn test_historical_data() {
        let mut historical = HistoricalData::new();

        // Add some samples
        for i in 0..5 {
            let mut perf_stats = Zero3PerformanceStats::new();
            perf_stats.record_forward_pass(Duration::from_millis(100 + i * 10), 1000);

            let sample = HistoricalSample {
                timestamp: Instant::now(),
                performance_stats: perf_stats,
                memory_stats: Zero3MemoryStats::new(),
            };

            historical.add_sample(sample);
        }

        assert_eq!(historical.sample_count(), 5);

        let trends = historical.analyze_trends(Duration::from_secs(10));
        assert!(trends.sample_count > 0);
    }

    #[test]
    fn test_training_step_stats() {
        let stats = TrainingStepStats {
            forward_time: Some(Duration::from_millis(100)),
            backward_time: Some(Duration::from_millis(150)),
            num_tokens: 1000,
            ..Default::default()
        };

        assert!(stats.forward_time.is_some());
        assert!(stats.backward_time.is_some());
        assert_eq!(stats.num_tokens, 1000);
    }

    #[test]
    fn test_memory_breakdown() {
        let breakdown = MemoryBreakdown {
            parameters: 1000,
            gradients: 500,
            optimizer_states: 300,
            cache: 200,
            buffers: 100,
            other: 50,
        };

        let total = breakdown.parameters + breakdown.gradients + breakdown.optimizer_states +
                   breakdown.cache + breakdown.buffers + breakdown.other;
        assert_eq!(total, 2150);
    }
}