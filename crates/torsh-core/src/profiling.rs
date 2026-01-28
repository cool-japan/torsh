//! Performance Profiling Hooks for ToRSh Operations
//!
//! This module provides comprehensive performance profiling capabilities for tensor operations,
//! including operation timing, memory bandwidth tracking, cache analysis, and performance
//! bottleneck identification.

use crate::error::{Result, TorshError};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

/// Global profiler instance
static PROFILER: OnceLock<Arc<Mutex<PerformanceProfiler>>> = OnceLock::new();

/// Performance profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Whether profiling is enabled
    pub enabled: bool,
    /// Maximum number of operation records to keep
    pub max_records: usize,
    /// Whether to capture stack traces for operations
    pub capture_stack_traces: bool,
    /// Whether to track memory bandwidth
    pub track_memory_bandwidth: bool,
    /// Whether to track cache performance
    pub track_cache_performance: bool,
    /// Minimum operation duration to record (filter out very fast operations)
    pub min_duration_ns: u64,
    /// Whether to aggregate similar operations
    pub aggregate_similar_ops: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_records: 10_000,
            capture_stack_traces: false,
            track_memory_bandwidth: true,
            track_cache_performance: true,
            min_duration_ns: 1_000, // 1 microsecond
            aggregate_similar_ops: true,
        }
    }
}

/// Type of operation being profiled
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Tensor creation operations
    Creation(String),
    /// Mathematical operations
    Math(String),
    /// Memory operations (copy, move, etc.)
    Memory(String),
    /// Shape operations (reshape, transpose, etc.)
    Shape(String),
    /// Reduction operations (sum, mean, etc.)
    Reduction(String),
    /// Neural network operations
    Neural(String),
    /// Backend operations
    Backend(String),
    /// Custom operation
    Custom(String),
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationType::Creation(name) => write!(f, "Creation::{name}"),
            OperationType::Math(name) => write!(f, "Math::{name}"),
            OperationType::Memory(name) => write!(f, "Memory::{name}"),
            OperationType::Shape(name) => write!(f, "Shape::{name}"),
            OperationType::Reduction(name) => write!(f, "Reduction::{name}"),
            OperationType::Neural(name) => write!(f, "Neural::{name}"),
            OperationType::Backend(name) => write!(f, "Backend::{name}"),
            OperationType::Custom(name) => write!(f, "Custom::{name}"),
        }
    }
}

/// Performance record for a single operation
#[derive(Debug, Clone)]
pub struct OperationRecord {
    /// Unique operation ID
    pub id: u64,
    /// Type of operation
    pub operation_type: OperationType,
    /// Duration of the operation
    pub duration: Duration,
    /// Memory bandwidth (bytes/second) if tracked
    pub memory_bandwidth: Option<f64>,
    /// Cache hit rate if tracked
    pub cache_hit_rate: Option<f64>,
    /// Input tensor sizes
    pub input_sizes: Vec<usize>,
    /// Output tensor size
    pub output_size: Option<usize>,
    /// Thread ID that executed the operation
    pub thread_id: thread::ThreadId,
    /// Timestamp when operation started
    pub timestamp: Instant,
    /// Stack trace if captured
    pub stack_trace: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Aggregated performance statistics for operation types
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation type
    pub operation_type: OperationType,
    /// Number of times this operation was executed
    pub count: u64,
    /// Total time spent in this operation
    pub total_duration: Duration,
    /// Minimum execution time
    pub min_duration: Duration,
    /// Maximum execution time
    pub max_duration: Duration,
    /// Average execution time
    pub avg_duration: Duration,
    /// 50th percentile (median) execution time
    pub p50_duration: Duration,
    /// 95th percentile execution time
    pub p95_duration: Duration,
    /// 99th percentile execution time
    pub p99_duration: Duration,
    /// Average memory bandwidth
    pub avg_memory_bandwidth: Option<f64>,
    /// Average cache hit rate
    pub avg_cache_hit_rate: Option<f64>,
    /// Total bytes processed
    pub total_bytes: usize,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Operation type causing the bottleneck
    pub operation_type: OperationType,
    /// Percentage of total time spent in this operation
    pub time_percentage: f64,
    /// Number of times this operation was called
    pub call_count: u64,
    /// Average duration per call
    pub avg_duration: Duration,
    /// Suggested optimization
    pub optimization_suggestion: String,
}

/// Main performance profiler
pub struct PerformanceProfiler {
    /// Configuration
    config: ProfilerConfig,
    /// Operation records
    records: VecDeque<OperationRecord>,
    /// Aggregated statistics
    stats: HashMap<OperationType, OperationStats>,
    /// Next operation ID
    next_id: u64,
    /// Total profiling overhead
    overhead_ns: u64,
    /// Profiler start time
    start_time: Instant,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            records: VecDeque::new(),
            stats: HashMap::new(),
            next_id: 1,
            overhead_ns: 0,
            start_time: Instant::now(),
        }
    }

    /// Start profiling an operation
    pub fn start_operation(&mut self, operation_type: OperationType) -> OperationHandle {
        if !self.config.enabled {
            return OperationHandle::disabled();
        }

        let start_time = Instant::now();
        let id = self.next_id;
        self.next_id += 1;

        OperationHandle {
            id,
            operation_type,
            start_time,
            enabled: true,
        }
    }

    /// Finish profiling an operation
    pub fn finish_operation(&mut self, handle: OperationHandle, context: OperationContext) {
        if !handle.enabled || !self.config.enabled {
            return;
        }

        let profile_start = Instant::now();
        let duration = handle.start_time.elapsed();

        // Filter out very fast operations if configured
        if duration.as_nanos() < self.config.min_duration_ns as u128 {
            self.overhead_ns += profile_start.elapsed().as_nanos() as u64;
            return;
        }

        let memory_bandwidth = if self.config.track_memory_bandwidth {
            context.calculate_memory_bandwidth(duration)
        } else {
            None
        };

        let cache_hit_rate = if self.config.track_cache_performance {
            context.cache_hit_rate
        } else {
            None
        };

        let stack_trace = if self.config.capture_stack_traces {
            Some(capture_stack_trace())
        } else {
            None
        };

        let record = OperationRecord {
            id: handle.id,
            operation_type: handle.operation_type.clone(),
            duration,
            memory_bandwidth,
            cache_hit_rate,
            input_sizes: context.input_sizes,
            output_size: context.output_size,
            thread_id: thread::current().id(),
            timestamp: handle.start_time,
            stack_trace,
            metadata: context.metadata,
        };

        // Add to records
        self.records.push_back(record.clone());

        // Maintain max records limit
        if self.records.len() > self.config.max_records {
            self.records.pop_front();
        }

        // Update aggregated statistics
        self.update_stats(&record);

        self.overhead_ns += profile_start.elapsed().as_nanos() as u64;
    }

    /// Get aggregated statistics for all operations
    pub fn get_stats(&self) -> HashMap<OperationType, OperationStats> {
        self.stats.clone()
    }

    /// Get all operation records
    pub fn get_records(&self) -> Vec<OperationRecord> {
        self.records.iter().cloned().collect()
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Performance Profile Report ===\n\n");

        let total_duration = self.start_time.elapsed();
        report.push_str(&format!("Profiling Duration: {total_duration:.2?}\n"));
        let total_ops = self.records.len();
        report.push_str(&format!("Total Operations: {total_ops}\n"));
        let overhead_us = self.overhead_ns as f64 / 1000.0;
        report.push_str(&format!("Profiling Overhead: {overhead_us:.2} µs\n"));

        // Top operations by total time
        let mut sorted_stats: Vec<_> = self.stats.values().collect();
        sorted_stats.sort_by(|a, b| b.total_duration.cmp(&a.total_duration));

        report.push_str("\nTop Operations by Total Time:\n");
        for (i, stat) in sorted_stats.iter().take(10).enumerate() {
            let percentage =
                (stat.total_duration.as_nanos() as f64 / total_duration.as_nanos() as f64) * 100.0;
            let idx = i + 1;
            let op_type = &stat.operation_type;
            let total_dur = stat.total_duration;
            let count = stat.count;
            let avg_dur = stat.avg_duration;
            report.push_str(&format!(
                "  {idx}. {op_type} - {total_dur:.2?} ({percentage:.1}%, {count} calls, avg: {avg_dur:.2?})\n"
            ));
        }

        // Performance bottlenecks
        let bottlenecks = self.identify_bottlenecks();
        if !bottlenecks.is_empty() {
            report.push_str("\nPerformance Bottlenecks:\n");
            for bottleneck in bottlenecks.iter().take(5) {
                let op_type = &bottleneck.operation_type;
                let time_pct = bottleneck.time_percentage;
                let call_count = bottleneck.call_count;
                let suggestion = &bottleneck.optimization_suggestion;
                report.push_str(&format!(
                    "  - {op_type}: {time_pct:.1}% of total time ({call_count} calls)\n"
                ));
                report.push_str(&format!("    Suggestion: {suggestion}\n"));
            }
        }

        // Memory bandwidth analysis
        let avg_bandwidth = self.calculate_average_bandwidth();
        if let Some(bandwidth) = avg_bandwidth {
            report.push_str(&format!(
                "\nAverage Memory Bandwidth: {bandwidth:.2} GB/s\n"
            ));
        }

        // Cache performance
        let avg_cache_hit_rate = self.calculate_average_cache_hit_rate();
        if let Some(hit_rate) = avg_cache_hit_rate {
            let hit_rate_percent = hit_rate * 100.0;
            report.push_str(&format!("Average Cache Hit Rate: {hit_rate_percent:.1}%\n"));
        }

        report
    }

    /// Reset profiler state
    pub fn reset(&mut self) {
        self.records.clear();
        self.stats.clear();
        self.next_id = 1;
        self.overhead_ns = 0;
        self.start_time = Instant::now();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ProfilerConfig) {
        self.config = config;
    }

    fn update_stats(&mut self, record: &OperationRecord) {
        let entry = self
            .stats
            .entry(record.operation_type.clone())
            .or_insert_with(|| OperationStats {
                operation_type: record.operation_type.clone(),
                count: 0,
                total_duration: Duration::ZERO,
                min_duration: Duration::MAX,
                max_duration: Duration::ZERO,
                avg_duration: Duration::ZERO,
                p50_duration: Duration::ZERO,
                p95_duration: Duration::ZERO,
                p99_duration: Duration::ZERO,
                avg_memory_bandwidth: None,
                avg_cache_hit_rate: None,
                total_bytes: 0,
            });

        entry.count += 1;
        entry.total_duration += record.duration;
        entry.min_duration = entry.min_duration.min(record.duration);
        entry.max_duration = entry.max_duration.max(record.duration);
        entry.avg_duration = entry.total_duration / entry.count as u32;

        if let Some(bandwidth) = record.memory_bandwidth {
            entry.avg_memory_bandwidth = Some(
                entry.avg_memory_bandwidth.unwrap_or(0.0)
                    + (bandwidth - entry.avg_memory_bandwidth.unwrap_or(0.0)) / entry.count as f64,
            );
        }

        if let Some(cache_rate) = record.cache_hit_rate {
            entry.avg_cache_hit_rate = Some(
                entry.avg_cache_hit_rate.unwrap_or(0.0)
                    + (cache_rate - entry.avg_cache_hit_rate.unwrap_or(0.0)) / entry.count as f64,
            );
        }

        // Update percentiles (simplified calculation)
        let durations: Vec<Duration> = self
            .records
            .iter()
            .filter(|r| r.operation_type == record.operation_type)
            .map(|r| r.duration)
            .collect();

        if !durations.is_empty() {
            let mut sorted_durations = durations.clone();
            sorted_durations.sort();

            let p50_idx = (sorted_durations.len() * 50) / 100;
            let p95_idx = (sorted_durations.len() * 95) / 100;
            let p99_idx = (sorted_durations.len() * 99) / 100;

            entry.p50_duration = sorted_durations
                .get(p50_idx)
                .copied()
                .unwrap_or(Duration::ZERO);
            entry.p95_duration = sorted_durations
                .get(p95_idx)
                .copied()
                .unwrap_or(Duration::ZERO);
            entry.p99_duration = sorted_durations
                .get(p99_idx)
                .copied()
                .unwrap_or(Duration::ZERO);
        }

        // Update total bytes
        let total_input_bytes: usize = record.input_sizes.iter().sum();
        let total_bytes = total_input_bytes + record.output_size.unwrap_or(0);
        entry.total_bytes += total_bytes;
    }

    fn identify_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        let total_time = self.start_time.elapsed();
        let mut bottlenecks = Vec::new();

        for stat in self.stats.values() {
            let time_percentage =
                (stat.total_duration.as_nanos() as f64 / total_time.as_nanos() as f64) * 100.0;

            if time_percentage > 5.0 {
                // Consider anything >5% of total time as a potential bottleneck
                let suggestion = generate_optimization_suggestion(&stat.operation_type, stat);

                bottlenecks.push(PerformanceBottleneck {
                    operation_type: stat.operation_type.clone(),
                    time_percentage,
                    call_count: stat.count,
                    avg_duration: stat.avg_duration,
                    optimization_suggestion: suggestion,
                });
            }
        }

        bottlenecks.sort_by(|a, b| {
            b.time_percentage
                .partial_cmp(&a.time_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bottlenecks
    }

    fn calculate_average_bandwidth(&self) -> Option<f64> {
        let bandwidths: Vec<f64> = self
            .records
            .iter()
            .filter_map(|r| r.memory_bandwidth)
            .collect();

        if bandwidths.is_empty() {
            None
        } else {
            Some(bandwidths.iter().sum::<f64>() / bandwidths.len() as f64)
        }
    }

    fn calculate_average_cache_hit_rate(&self) -> Option<f64> {
        let hit_rates: Vec<f64> = self
            .records
            .iter()
            .filter_map(|r| r.cache_hit_rate)
            .collect();

        if hit_rates.is_empty() {
            None
        } else {
            Some(hit_rates.iter().sum::<f64>() / hit_rates.len() as f64)
        }
    }
}

/// Handle for an operation being profiled
pub struct OperationHandle {
    id: u64,
    operation_type: OperationType,
    start_time: Instant,
    enabled: bool,
}

impl OperationHandle {
    fn disabled() -> Self {
        Self {
            id: 0,
            operation_type: OperationType::Custom("disabled".to_string()),
            start_time: Instant::now(),
            enabled: false,
        }
    }
}

/// Context information for an operation
pub struct OperationContext {
    /// Input tensor sizes in bytes
    pub input_sizes: Vec<usize>,
    /// Output tensor size in bytes
    pub output_size: Option<usize>,
    /// Cache hit rate if available
    pub cache_hit_rate: Option<f64>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl OperationContext {
    pub fn new() -> Self {
        Self {
            input_sizes: Vec::new(),
            output_size: None,
            cache_hit_rate: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_input_size(mut self, size: usize) -> Self {
        self.input_sizes.push(size);
        self
    }

    pub fn with_output_size(mut self, size: usize) -> Self {
        self.output_size = Some(size);
        self
    }

    pub fn with_cache_hit_rate(mut self, rate: f64) -> Self {
        self.cache_hit_rate = Some(rate);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    fn calculate_memory_bandwidth(&self, duration: Duration) -> Option<f64> {
        let total_bytes: usize =
            self.input_sizes.iter().sum::<usize>() + self.output_size.unwrap_or(0);

        if total_bytes == 0 || duration.is_zero() {
            return None;
        }

        let duration_secs = duration.as_secs_f64();
        let bandwidth_bytes_per_sec = total_bytes as f64 / duration_secs;
        let bandwidth_gb_per_sec = bandwidth_bytes_per_sec / 1_000_000_000.0;

        Some(bandwidth_gb_per_sec)
    }
}

impl Default for OperationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate optimization suggestions based on operation type and statistics
fn generate_optimization_suggestion(op_type: &OperationType, stats: &OperationStats) -> String {
    match op_type {
        OperationType::Math(name) => {
            if stats.avg_duration > Duration::from_millis(10) {
                format!("Consider using SIMD optimizations for {name} operations")
            } else if let Some(bandwidth) = stats.avg_memory_bandwidth {
                if bandwidth < 10.0 {
                    "Memory bandwidth is low - consider batching operations".to_string()
                } else {
                    "Consider using more efficient algorithms or caching".to_string()
                }
            } else {
                "Consider optimizing algorithm or using specialized libraries".to_string()
            }
        }
        OperationType::Memory(name) => {
            if let Some(bandwidth) = stats.avg_memory_bandwidth {
                if bandwidth < 20.0 {
                    format!(
                        "Memory bandwidth for {name} is low - consider memory layout optimization"
                    )
                } else {
                    "Consider reducing memory allocations or using memory pools".to_string()
                }
            } else {
                "Consider optimizing memory access patterns".to_string()
            }
        }
        OperationType::Shape(name) => {
            if stats.count > 1000 {
                format!("High frequency {name} operations - consider caching or batching")
            } else {
                "Consider optimizing shape operations with compile-time checks".to_string()
            }
        }
        OperationType::Neural(name) => {
            format!("Consider using specialized neural network libraries for {name} operations")
        }
        _ => "Consider profiling individual sub-operations to identify bottlenecks".to_string(),
    }
}

/// Capture stack trace (simplified implementation)
fn capture_stack_trace() -> String {
    // In a real implementation, this would capture the actual stack trace
    // For now, we'll return a placeholder
    let binding = std::thread::current();
    let thread_name = binding.name().unwrap_or("unknown");
    format!("Stack trace captured at {thread_name}")
}

/// Global profiler access functions
pub fn get_profiler() -> Arc<Mutex<PerformanceProfiler>> {
    PROFILER
        .get_or_init(|| {
            Arc::new(Mutex::new(PerformanceProfiler::new(
                ProfilerConfig::default(),
            )))
        })
        .clone()
}

/// Initialize the global profiler with custom configuration
pub fn init_profiler(config: ProfilerConfig) -> Result<()> {
    if PROFILER.get().is_some() {
        return Err(TorshError::InvalidState(
            "Profiler already initialized".to_string(),
        ));
    }

    PROFILER
        .set(Arc::new(Mutex::new(PerformanceProfiler::new(config))))
        .map_err(|_| TorshError::InvalidState("Failed to initialize profiler".to_string()))?;

    Ok(())
}

/// Convenience macro for profiling operations
#[macro_export]
macro_rules! profile_operation {
    ($op_type:expr, $context:expr, $body:expr) => {{
        let profiler = $crate::profiling::get_profiler();
        let handle = {
            let mut p = profiler.lock().expect("lock should not be poisoned");
            p.start_operation($op_type)
        };

        let result = $body;

        {
            let mut p = profiler.lock().expect("lock should not be poisoned");
            p.finish_operation(handle, $context);
        }

        result
    }};
}

/// Convenience function for profiling a closure
pub fn profile_closure<F, R>(op_type: OperationType, context: OperationContext, closure: F) -> R
where
    F: FnOnce() -> R,
{
    let profiler = get_profiler();
    let handle = {
        let mut p = profiler.lock().expect("lock should not be poisoned");
        p.start_operation(op_type)
    };

    let result = closure();

    {
        let mut p = profiler.lock().expect("lock should not be poisoned");
        p.finish_operation(handle, context);
    }

    result
}

/// Shape-specific performance metrics collection
#[derive(Debug, Clone, Default)]
pub struct ShapeMetrics {
    /// Number of dimensions
    pub ndim: usize,
    /// Total number of elements
    pub numel: usize,
    /// Memory layout efficiency (0.0-1.0)
    pub layout_efficiency: f64,
    /// Broadcasting complexity score
    pub broadcast_complexity: f64,
    /// SIMD vectorization efficiency
    pub simd_efficiency: Option<f64>,
    /// Cache locality score
    pub cache_locality: Option<f64>,
}

impl ShapeMetrics {
    /// Create new shape metrics
    pub fn new(ndim: usize, numel: usize) -> Self {
        Self {
            ndim,
            numel,
            layout_efficiency: 1.0,    // Default to perfect efficiency
            broadcast_complexity: 0.0, // No broadcasting
            simd_efficiency: None,
            cache_locality: None,
        }
    }

    /// Set layout efficiency score
    pub fn with_layout_efficiency(mut self, efficiency: f64) -> Self {
        self.layout_efficiency = efficiency.clamp(0.0, 1.0);
        self
    }

    /// Set broadcasting complexity score
    pub fn with_broadcast_complexity(mut self, complexity: f64) -> Self {
        self.broadcast_complexity = complexity.max(0.0);
        self
    }

    /// Set SIMD efficiency score
    pub fn with_simd_efficiency(mut self, efficiency: f64) -> Self {
        self.simd_efficiency = Some(efficiency.clamp(0.0, 1.0));
        self
    }

    /// Set cache locality score
    pub fn with_cache_locality(mut self, locality: f64) -> Self {
        self.cache_locality = Some(locality.clamp(0.0, 1.0));
        self
    }

    /// Calculate overall performance score
    pub fn performance_score(&self) -> f64 {
        let mut score = self.layout_efficiency;

        // Penalize for broadcasting complexity
        score *= 1.0 - (self.broadcast_complexity / 10.0).min(0.5);

        // Boost for SIMD efficiency
        if let Some(simd) = self.simd_efficiency {
            score *= 1.0 + simd * 0.2;
        }

        // Boost for cache locality
        if let Some(cache) = self.cache_locality {
            score *= 1.0 + cache * 0.1;
        }

        score.clamp(0.0, 1.0)
    }
}

/// Shape operation performance tracker
#[derive(Debug)]
pub struct ShapePerformanceTracker {
    /// Shape operation records
    records: VecDeque<ShapeOperationRecord>,
    /// Maximum number of records to keep
    max_records: usize,
    /// Aggregate statistics by operation type
    aggregates: HashMap<String, ShapeOperationAggregate>,
}

/// Record for a shape operation
#[derive(Debug, Clone)]
pub struct ShapeOperationRecord {
    /// Operation name
    pub operation: String,
    /// Operation duration
    pub duration: Duration,
    /// Shape metrics
    pub metrics: ShapeMetrics,
    /// Timestamp
    pub timestamp: Instant,
    /// Thread ID
    pub thread_id: std::thread::ThreadId,
}

/// Aggregate statistics for a shape operation type
#[derive(Debug, Clone)]
pub struct ShapeOperationAggregate {
    /// Number of operations
    pub count: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Average duration
    pub avg_duration: Duration,
    /// Min duration
    pub min_duration: Duration,
    /// Max duration
    pub max_duration: Duration,
    /// Average performance score
    pub avg_performance_score: f64,
    /// Best performance score
    pub best_performance_score: f64,
    /// Worst performance score
    pub worst_performance_score: f64,
}

impl ShapePerformanceTracker {
    /// Create a new shape performance tracker
    pub fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records),
            max_records,
            aggregates: HashMap::new(),
        }
    }

    /// Record a shape operation
    pub fn record_operation(
        &mut self,
        operation: String,
        duration: Duration,
        metrics: ShapeMetrics,
    ) {
        let record = ShapeOperationRecord {
            operation: operation.clone(),
            duration,
            metrics: metrics.clone(),
            timestamp: Instant::now(),
            thread_id: std::thread::current().id(),
        };

        // Add to records (with size limit)
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(record);

        // Update aggregates
        let performance_score = metrics.performance_score();
        let aggregate =
            self.aggregates
                .entry(operation)
                .or_insert_with(|| ShapeOperationAggregate {
                    count: 0,
                    total_duration: Duration::ZERO,
                    avg_duration: Duration::ZERO,
                    min_duration: duration,
                    max_duration: duration,
                    avg_performance_score: performance_score,
                    best_performance_score: performance_score,
                    worst_performance_score: performance_score,
                });

        aggregate.count += 1;
        aggregate.total_duration += duration;
        aggregate.avg_duration = aggregate.total_duration / aggregate.count as u32;
        aggregate.min_duration = aggregate.min_duration.min(duration);
        aggregate.max_duration = aggregate.max_duration.max(duration);

        // Update performance scores
        let total_score =
            aggregate.avg_performance_score * (aggregate.count - 1) as f64 + performance_score;
        aggregate.avg_performance_score = total_score / aggregate.count as f64;
        aggregate.best_performance_score = aggregate.best_performance_score.max(performance_score);
        aggregate.worst_performance_score =
            aggregate.worst_performance_score.min(performance_score);
    }

    /// Get recent records
    pub fn get_records(&self) -> Vec<ShapeOperationRecord> {
        self.records.iter().cloned().collect()
    }

    /// Get aggregate statistics
    pub fn get_aggregates(&self) -> &HashMap<String, ShapeOperationAggregate> {
        &self.aggregates
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Shape Operations Performance Report ===\n\n");

        report.push_str(&format!("Total Records: {}\n", self.records.len()));
        report.push_str(&format!("Operation Types: {}\n\n", self.aggregates.len()));

        // Sort aggregates by average performance score (worst first)
        let mut sorted_ops: Vec<_> = self.aggregates.iter().collect();
        sorted_ops.sort_by(|a, b| {
            a.1.avg_performance_score
                .partial_cmp(&b.1.avg_performance_score)
                .expect("performance scores should be comparable (no NaN)")
        });

        report.push_str("Performance Summary (worst to best):\n");
        for (op_name, aggregate) in sorted_ops {
            report.push_str(&format!(
                "  {}: {:.3} avg score, {:.2}ms avg time, {} calls\n",
                op_name,
                aggregate.avg_performance_score,
                aggregate.avg_duration.as_secs_f64() * 1000.0,
                aggregate.count
            ));
        }

        report.push_str("\nDetailed Statistics:\n");
        for (op_name, aggregate) in &self.aggregates {
            report.push_str(&format!("\n{op_name}:\n"));
            report.push_str(&format!("  Count: {}\n", aggregate.count));
            report.push_str(&format!(
                "  Avg Duration: {:.2}ms\n",
                aggregate.avg_duration.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Min Duration: {:.2}ms\n",
                aggregate.min_duration.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Max Duration: {:.2}ms\n",
                aggregate.max_duration.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Avg Performance: {:.3}\n",
                aggregate.avg_performance_score
            ));
            report.push_str(&format!(
                "  Best Performance: {:.3}\n",
                aggregate.best_performance_score
            ));
            report.push_str(&format!(
                "  Worst Performance: {:.3}\n",
                aggregate.worst_performance_score
            ));
        }

        report
    }

    /// Find performance bottlenecks
    pub fn find_bottlenecks(&self) -> Vec<(String, String)> {
        let mut bottlenecks = Vec::new();

        for (op_name, aggregate) in &self.aggregates {
            // Check for poor performance scores
            if aggregate.avg_performance_score < 0.5 {
                bottlenecks.push((
                    op_name.clone(),
                    format!(
                        "Low performance score: {:.3}",
                        aggregate.avg_performance_score
                    ),
                ));
            }

            // Check for high variance in execution time
            let duration_ratio =
                aggregate.max_duration.as_secs_f64() / aggregate.min_duration.as_secs_f64();
            if duration_ratio > 5.0 && aggregate.count > 10 {
                bottlenecks.push((
                    op_name.clone(),
                    format!(
                        "High variance: {duration_ratio:.1}x difference between min/max duration"
                    ),
                ));
            }

            // Check for frequent operations that could benefit from optimization
            if aggregate.count > 100 && aggregate.avg_duration.as_millis() > 1 {
                bottlenecks.push((
                    op_name.clone(),
                    format!(
                        "Frequent expensive operation: {} calls, {:.2}ms avg",
                        aggregate.count,
                        aggregate.avg_duration.as_secs_f64() * 1000.0
                    ),
                ));
            }
        }

        bottlenecks
    }

    /// Get optimization suggestions
    pub fn get_optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        let bottlenecks = self.find_bottlenecks();

        for (op_name, issue) in bottlenecks {
            if issue.contains("Low performance score") {
                suggestions.push(format!(
                    "Consider optimizing {op_name} - check memory layout and broadcasting efficiency"
                ));
            } else if issue.contains("High variance") {
                suggestions.push(format!(
                    "Investigate {op_name} for inconsistent performance - possible cache/memory pressure issues"
                ));
            } else if issue.contains("Frequent expensive") {
                suggestions.push(format!(
                    "Profile {op_name} for optimization opportunities - consider caching or vectorization"
                ));
            }
        }

        if suggestions.is_empty() {
            suggestions.push("No performance issues detected - good job!".to_string());
        }

        suggestions
    }
}

/// Global shape performance tracker
static SHAPE_TRACKER: OnceLock<Arc<Mutex<ShapePerformanceTracker>>> = OnceLock::new();

/// Get or initialize the global shape performance tracker
pub fn get_shape_tracker() -> &'static Arc<Mutex<ShapePerformanceTracker>> {
    SHAPE_TRACKER.get_or_init(|| Arc::new(Mutex::new(ShapePerformanceTracker::new(10_000))))
}

/// Profile a shape operation with automatic metrics collection
pub fn profile_shape_operation<F, R>(operation_name: &str, ndim: usize, numel: usize, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();

    let metrics = ShapeMetrics::new(ndim, numel);

    let tracker = get_shape_tracker();
    if let Ok(mut tracker) = tracker.lock() {
        tracker.record_operation(operation_name.to_string(), duration, metrics);
    }

    result
}

/// Profile a shape operation with custom metrics
pub fn profile_shape_operation_with_metrics<F, R>(
    operation_name: &str,
    metrics: ShapeMetrics,
    f: F,
) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();

    let tracker = get_shape_tracker();
    if let Ok(mut tracker) = tracker.lock() {
        tracker.record_operation(operation_name.to_string(), duration, metrics);
    }

    result
}

/// Macro for easy shape operation profiling
#[macro_export]
macro_rules! profile_shape_op {
    ($op_name:expr, $ndim:expr, $numel:expr, $body:expr) => {
        $crate::profiling::profile_shape_operation($op_name, $ndim, $numel, || $body)
    };
    ($op_name:expr, $metrics:expr, $body:expr) => {
        $crate::profiling::profile_shape_operation_with_metrics($op_name, $metrics, || $body)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_creation() {
        let profiler = PerformanceProfiler::new(ProfilerConfig::default());
        assert_eq!(profiler.records.len(), 0);
        assert_eq!(profiler.stats.len(), 0);
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());
        let op_type = OperationType::Math("add".to_string());

        let handle = profiler.start_operation(op_type.clone());
        thread::sleep(Duration::from_millis(1));

        let context = OperationContext::new()
            .with_input_size(1000)
            .with_output_size(1000);

        profiler.finish_operation(handle, context);

        assert_eq!(profiler.records.len(), 1);
        assert!(profiler.stats.contains_key(&op_type));
    }

    #[test]
    fn test_profiler_statistics() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());
        let op_type = OperationType::Math("multiply".to_string());

        // Profile multiple operations
        for _ in 0..3 {
            let handle = profiler.start_operation(op_type.clone());
            thread::sleep(Duration::from_millis(1));

            let context = OperationContext::new()
                .with_input_size(500)
                .with_output_size(500);

            profiler.finish_operation(handle, context);
        }

        let stats = profiler.get_stats();
        let multiply_stats = stats.get(&op_type).unwrap();

        assert_eq!(multiply_stats.count, 3);
        assert!(multiply_stats.total_duration > Duration::ZERO);
        assert!(multiply_stats.avg_duration > Duration::ZERO);
    }

    #[test]
    fn test_bottleneck_identification() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());
        let slow_op = OperationType::Math("slow_operation".to_string());
        let fast_op = OperationType::Math("fast_operation".to_string());

        // Create a slow operation
        let handle = profiler.start_operation(slow_op.clone());
        thread::sleep(Duration::from_millis(10));
        profiler.finish_operation(handle, OperationContext::new());

        // Create fast operations
        for _ in 0..5 {
            let handle = profiler.start_operation(fast_op.clone());
            thread::sleep(Duration::from_millis(1));
            profiler.finish_operation(handle, OperationContext::new());
        }

        let bottlenecks = profiler.identify_bottlenecks();
        assert!(!bottlenecks.is_empty());

        // The slow operation should be identified as a bottleneck
        assert!(bottlenecks.iter().any(|b| b.operation_type == slow_op));
    }

    #[test]
    fn test_memory_bandwidth_calculation() {
        let context = OperationContext::new()
            .with_input_size(1000)
            .with_output_size(1000);

        let duration = Duration::from_millis(1);
        let bandwidth = context.calculate_memory_bandwidth(duration);

        assert!(bandwidth.is_some());
        assert!(bandwidth.unwrap() > 0.0);
    }

    #[test]
    fn test_profile_closure() {
        let _profiler = get_profiler();

        let result = profile_closure(
            OperationType::Math("test".to_string()),
            OperationContext::new(),
            || {
                thread::sleep(Duration::from_millis(1));
                42
            },
        );

        assert_eq!(result, 42);

        // Check that the operation was recorded
        let profiler = get_profiler();
        let records = {
            let p = profiler.lock().expect("lock should not be poisoned");
            p.get_records()
        };

        assert!(!records.is_empty());
    }

    #[test]
    fn test_profiler_report_generation() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());

        // Add some operations
        let handle = profiler.start_operation(OperationType::Math("add".to_string()));
        thread::sleep(Duration::from_millis(1));
        profiler.finish_operation(handle, OperationContext::new());

        let report = profiler.generate_report();
        assert!(report.contains("Performance Profile Report"));
        assert!(report.contains("Total Operations: 1"));
        assert!(report.contains("Math::add"));
    }
}
