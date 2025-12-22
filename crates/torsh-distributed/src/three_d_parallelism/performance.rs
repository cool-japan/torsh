//! Performance monitoring and statistics for 3D parallelism
//!
//! This module provides comprehensive performance monitoring, metrics collection,
//! and analysis capabilities for 3D parallelism operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::config::RankMapping;

/// Performance monitor for 3D parallelism operations
pub struct Performance3DMonitor {
    /// Rank mapping for context
    rank_mapping: RankMapping,
    /// Performance statistics
    stats: Arc<Mutex<Performance3DStats>>,
    /// Detailed timing measurements
    timing_history: Arc<Mutex<TimingHistory>>,
    /// Memory usage tracking
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Communication metrics
    communication_metrics: Arc<Mutex<CommunicationMetrics>>,
}

impl Performance3DMonitor {
    /// Create new performance monitor
    pub fn new(rank_mapping: &RankMapping) -> Self {
        Self {
            rank_mapping: rank_mapping.clone(),
            stats: Arc::new(Mutex::new(Performance3DStats::new())),
            timing_history: Arc::new(Mutex::new(TimingHistory::new())),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::new())),
            communication_metrics: Arc::new(Mutex::new(CommunicationMetrics::new())),
        }
    }

    /// Record forward pass performance
    pub async fn record_forward_pass(&self, duration: Duration, num_tokens: usize) {
        let mut stats = self.stats.lock().unwrap();
        stats.forward_passes += 1;
        stats.total_forward_time += duration;
        stats.total_tokens_processed += num_tokens as u64;

        // Calculate tokens per second
        if !stats.total_forward_time.is_zero() {
            stats.tokens_per_second =
                stats.total_tokens_processed as f64 / stats.total_forward_time.as_secs_f64();
        }

        // Record in timing history
        let mut history = self.timing_history.lock().unwrap();
        history.record_forward_pass(duration, num_tokens);

        // Update computation time
        stats.computation_time += duration;
    }

    /// Record backward pass performance
    pub async fn record_backward_pass(&self, duration: Duration, num_tokens: usize) {
        let mut stats = self.stats.lock().unwrap();
        stats.backward_passes += 1;
        stats.total_backward_time += duration;

        // Record in timing history
        let mut history = self.timing_history.lock().unwrap();
        history.record_backward_pass(duration, num_tokens);

        // Update computation time
        stats.computation_time += duration;
    }

    /// Record communication event
    pub async fn record_communication(
        &self,
        comm_type: CommunicationType,
        duration: Duration,
        bytes: usize,
    ) {
        let mut stats = self.stats.lock().unwrap();
        stats.communication_time += duration;

        let mut comm_metrics = self.communication_metrics.lock().unwrap();
        comm_metrics.record_communication(comm_type, duration, bytes);
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, usage_mb: f64) {
        let mut stats = self.stats.lock().unwrap();
        stats.memory_usage_mb = usage_mb;

        let mut memory_tracker = self.memory_tracker.lock().unwrap();
        memory_tracker.record_usage(usage_mb);
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> Performance3DStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get detailed performance analysis
    pub fn get_performance_analysis(&self) -> PerformanceAnalysis {
        let stats = self.stats.lock().unwrap();
        let timing_history = self.timing_history.lock().unwrap();
        let memory_tracker = self.memory_tracker.lock().unwrap();
        let comm_metrics = self.communication_metrics.lock().unwrap();

        PerformanceAnalysis {
            overall_throughput: stats.tokens_per_second,
            forward_pass_avg_ms: timing_history.avg_forward_time_ms(),
            backward_pass_avg_ms: timing_history.avg_backward_time_ms(),
            communication_overhead_percent: self.calculate_communication_overhead(&stats),
            memory_efficiency: memory_tracker.efficiency(),
            pipeline_utilization: self.calculate_pipeline_utilization(&timing_history),
            tensor_parallel_efficiency: self.calculate_tp_efficiency(&comm_metrics),
            data_parallel_efficiency: self.calculate_dp_efficiency(&comm_metrics),
            bottlenecks: self.identify_bottlenecks(&stats, &timing_history, &comm_metrics),
        }
    }

    /// Calculate communication overhead percentage
    fn calculate_communication_overhead(&self, stats: &Performance3DStats) -> f32 {
        let total_time = stats.computation_time + stats.communication_time;
        if total_time.is_zero() {
            0.0
        } else {
            (stats.communication_time.as_secs_f32() / total_time.as_secs_f32()) * 100.0
        }
    }

    /// Calculate pipeline utilization
    fn calculate_pipeline_utilization(&self, timing_history: &TimingHistory) -> f32 {
        // Simplified calculation - would analyze pipeline bubble time
        let ideal_time = timing_history.total_forward_time + timing_history.total_backward_time;
        if ideal_time.is_zero() {
            0.0
        } else {
            let actual_time = timing_history.wall_clock_time;
            (ideal_time.as_secs_f32() / actual_time.as_secs_f32()).min(1.0) * 100.0
        }
    }

    /// Calculate tensor parallel efficiency
    fn calculate_tp_efficiency(&self, comm_metrics: &CommunicationMetrics) -> f32 {
        // Efficiency based on all-reduce vs all-gather patterns
        if self.rank_mapping.config.tp_size <= 1 {
            100.0
        } else {
            let tp_comm_time = comm_metrics.get_communication_time(CommunicationType::AllReduceTP);
            let total_comm_time = comm_metrics.total_communication_time();

            if total_comm_time.is_zero() {
                100.0
            } else {
                let ideal_ratio = 1.0 / self.rank_mapping.config.tp_size as f32;
                let actual_ratio = tp_comm_time.as_secs_f32() / total_comm_time.as_secs_f32();
                ((ideal_ratio / actual_ratio.max(ideal_ratio)) * 100.0).min(100.0)
            }
        }
    }

    /// Calculate data parallel efficiency
    fn calculate_dp_efficiency(&self, comm_metrics: &CommunicationMetrics) -> f32 {
        if self.rank_mapping.config.dp_size <= 1 {
            100.0
        } else {
            // Efficiency based on gradient synchronization patterns
            let dp_comm_time = comm_metrics.get_communication_time(CommunicationType::AllReduceDP);
            let computation_time = self.stats.lock().unwrap().computation_time;

            if computation_time.is_zero() {
                100.0
            } else {
                let comm_ratio = dp_comm_time.as_secs_f32() / computation_time.as_secs_f32();
                ((1.0 / (1.0 + comm_ratio)) * 100.0).min(100.0)
            }
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(
        &self,
        stats: &Performance3DStats,
        timing_history: &TimingHistory,
        comm_metrics: &CommunicationMetrics,
    ) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Check communication overhead
        let comm_overhead = self.calculate_communication_overhead(stats);
        if comm_overhead > 30.0 {
            bottlenecks.push(PerformanceBottleneck {
                category: "Communication".to_string(),
                description: format!("High communication overhead: {:.1}%", comm_overhead),
                severity: BottleneckSeverity::High,
                suggested_fix:
                    "Consider increasing micro-batch size or optimizing communication patterns"
                        .to_string(),
            });
        }

        // Check memory usage
        if stats.memory_usage_mb
            > 0.9 * (self.rank_mapping.config.max_memory_per_device as f64) * 1024.0
        {
            bottlenecks.push(PerformanceBottleneck {
                category: "Memory".to_string(),
                description: "Memory usage near capacity".to_string(),
                severity: BottleneckSeverity::Critical,
                suggested_fix: "Enable gradient checkpointing or reduce model size".to_string(),
            });
        }

        // Check pipeline utilization
        let pipeline_util = self.calculate_pipeline_utilization(timing_history);
        if pipeline_util < 70.0 {
            bottlenecks.push(PerformanceBottleneck {
                category: "Pipeline".to_string(),
                description: format!("Low pipeline utilization: {:.1}%", pipeline_util),
                severity: BottleneckSeverity::Medium,
                suggested_fix: "Adjust micro-batch size or pipeline schedule".to_string(),
            });
        }

        // Check tensor parallel efficiency
        let tp_efficiency = self.calculate_tp_efficiency(comm_metrics);
        if tp_efficiency < 80.0 && self.rank_mapping.config.tp_size > 1 {
            bottlenecks.push(PerformanceBottleneck {
                category: "TensorParallel".to_string(),
                description: format!("Low tensor parallel efficiency: {:.1}%", tp_efficiency),
                severity: BottleneckSeverity::Medium,
                suggested_fix: "Optimize tensor parallel communication or reduce TP size"
                    .to_string(),
            });
        }

        bottlenecks
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let analysis = self.get_performance_analysis();
        let stats = self.get_stats();

        format!(
            "🚀 3D Parallelism Performance Report\n\
             ===================================\n\
             \n\
             📊 Overall Performance:\n\
             • Throughput: {:.1} tokens/second\n\
             • Forward Pass: {:.2}ms avg\n\
             • Backward Pass: {:.2}ms avg\n\
             • Communication Overhead: {:.1}%\n\
             \n\
             💾 Memory Metrics:\n\
             • Current Usage: {:.1} MB\n\
             • Memory Efficiency: {:.1}%\n\
             \n\
             🔄 Parallelism Efficiency:\n\
             • Pipeline Utilization: {:.1}%\n\
             • Tensor Parallel Efficiency: {:.1}%\n\
             • Data Parallel Efficiency: {:.1}%\n\
             \n\
             ⚠️ Bottlenecks Identified:\n\
             {}\n\
             \n\
             📈 Statistics:\n\
             • Forward Passes: {}\n\
             • Backward Passes: {}\n\
             • Total Tokens Processed: {}\n\
             • Total Computation Time: {:.2}s\n\
             • Total Communication Time: {:.2}s\n",
            analysis.overall_throughput,
            analysis.forward_pass_avg_ms,
            analysis.backward_pass_avg_ms,
            analysis.communication_overhead_percent,
            stats.memory_usage_mb,
            analysis.memory_efficiency,
            analysis.pipeline_utilization,
            analysis.tensor_parallel_efficiency,
            analysis.data_parallel_efficiency,
            self.format_bottlenecks(&analysis.bottlenecks),
            stats.forward_passes,
            stats.backward_passes,
            stats.total_tokens_processed,
            stats.computation_time.as_secs_f64(),
            stats.communication_time.as_secs_f64()
        )
    }

    /// Format bottlenecks for display
    fn format_bottlenecks(&self, bottlenecks: &[PerformanceBottleneck]) -> String {
        if bottlenecks.is_empty() {
            "No significant bottlenecks detected".to_string()
        } else {
            bottlenecks
                .iter()
                .map(|b| {
                    format!(
                        "• {}: {} ({})",
                        b.category,
                        b.description,
                        b.severity.as_str()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = Performance3DStats::new();

        let mut history = self.timing_history.lock().unwrap();
        *history = TimingHistory::new();

        let mut memory_tracker = self.memory_tracker.lock().unwrap();
        *memory_tracker = MemoryTracker::new();

        let mut comm_metrics = self.communication_metrics.lock().unwrap();
        *comm_metrics = CommunicationMetrics::new();
    }
}

/// Performance statistics for 3D parallelism
#[derive(Debug, Clone)]
pub struct Performance3DStats {
    pub forward_passes: u64,
    pub backward_passes: u64,
    pub total_forward_time: Duration,
    pub total_backward_time: Duration,
    pub total_tokens_processed: u64,
    pub tokens_per_second: f64,
    pub communication_time: Duration,
    pub computation_time: Duration,
    pub memory_usage_mb: f64,
}

impl Default for Performance3DStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Performance3DStats {
    pub fn new() -> Self {
        Self {
            forward_passes: 0,
            backward_passes: 0,
            total_forward_time: Duration::ZERO,
            total_backward_time: Duration::ZERO,
            total_tokens_processed: 0,
            tokens_per_second: 0.0,
            communication_time: Duration::ZERO,
            computation_time: Duration::ZERO,
            memory_usage_mb: 0.0,
        }
    }
}

/// Detailed performance analysis
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub overall_throughput: f64,
    pub forward_pass_avg_ms: f32,
    pub backward_pass_avg_ms: f32,
    pub communication_overhead_percent: f32,
    pub memory_efficiency: f32,
    pub pipeline_utilization: f32,
    pub tensor_parallel_efficiency: f32,
    pub data_parallel_efficiency: f32,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub category: String,
    pub description: String,
    pub severity: BottleneckSeverity,
    pub suggested_fix: String,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl BottleneckSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Critical => "Critical",
        }
    }
}

/// Timing history tracker
#[derive(Debug, Clone)]
struct TimingHistory {
    forward_times: Vec<Duration>,
    backward_times: Vec<Duration>,
    total_forward_time: Duration,
    total_backward_time: Duration,
    wall_clock_time: Duration,
    start_time: Option<Instant>,
}

impl TimingHistory {
    fn new() -> Self {
        Self {
            forward_times: Vec::new(),
            backward_times: Vec::new(),
            total_forward_time: Duration::ZERO,
            total_backward_time: Duration::ZERO,
            wall_clock_time: Duration::ZERO,
            start_time: Some(Instant::now()),
        }
    }

    fn record_forward_pass(&mut self, duration: Duration, _num_tokens: usize) {
        self.forward_times.push(duration);
        self.total_forward_time += duration;
        self.update_wall_clock_time();

        // Keep only recent measurements
        if self.forward_times.len() > 1000 {
            self.forward_times.remove(0);
        }
    }

    fn record_backward_pass(&mut self, duration: Duration, _num_tokens: usize) {
        self.backward_times.push(duration);
        self.total_backward_time += duration;
        self.update_wall_clock_time();

        // Keep only recent measurements
        if self.backward_times.len() > 1000 {
            self.backward_times.remove(0);
        }
    }

    fn update_wall_clock_time(&mut self) {
        if let Some(start) = self.start_time {
            self.wall_clock_time = start.elapsed();
        }
    }

    fn avg_forward_time_ms(&self) -> f32 {
        if self.forward_times.is_empty() {
            0.0
        } else {
            let total: Duration = self.forward_times.iter().sum();
            total.as_secs_f32() * 1000.0 / self.forward_times.len() as f32
        }
    }

    fn avg_backward_time_ms(&self) -> f32 {
        if self.backward_times.is_empty() {
            0.0
        } else {
            let total: Duration = self.backward_times.iter().sum();
            total.as_secs_f32() * 1000.0 / self.backward_times.len() as f32
        }
    }
}

/// Memory usage tracker
#[derive(Debug, Clone)]
struct MemoryTracker {
    usage_history: Vec<f64>,
    peak_usage: f64,
    average_usage: f64,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            usage_history: Vec::new(),
            peak_usage: 0.0,
            average_usage: 0.0,
        }
    }

    fn record_usage(&mut self, usage_mb: f64) {
        self.usage_history.push(usage_mb);
        self.peak_usage = self.peak_usage.max(usage_mb);

        // Update average
        if !self.usage_history.is_empty() {
            self.average_usage =
                self.usage_history.iter().sum::<f64>() / self.usage_history.len() as f64;
        }

        // Keep only recent measurements
        if self.usage_history.len() > 1000 {
            self.usage_history.remove(0);
        }
    }

    fn efficiency(&self) -> f32 {
        if self.peak_usage == 0.0 {
            100.0
        } else {
            (self.average_usage / self.peak_usage * 100.0) as f32
        }
    }
}

/// Communication metrics tracker
#[derive(Debug, Clone)]
struct CommunicationMetrics {
    communication_times: HashMap<CommunicationType, Vec<Duration>>,
    bytes_transferred: HashMap<CommunicationType, Vec<usize>>,
}

impl CommunicationMetrics {
    fn new() -> Self {
        Self {
            communication_times: HashMap::new(),
            bytes_transferred: HashMap::new(),
        }
    }

    fn record_communication(
        &mut self,
        comm_type: CommunicationType,
        duration: Duration,
        bytes: usize,
    ) {
        self.communication_times
            .entry(comm_type)
            .or_default()
            .push(duration);

        self.bytes_transferred
            .entry(comm_type)
            .or_default()
            .push(bytes);
    }

    fn get_communication_time(&self, comm_type: CommunicationType) -> Duration {
        self.communication_times
            .get(&comm_type)
            .map(|times| times.iter().sum())
            .unwrap_or(Duration::ZERO)
    }

    fn total_communication_time(&self) -> Duration {
        self.communication_times
            .values()
            .flat_map(|times| times.iter())
            .sum()
    }
}

/// Communication operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommunicationType {
    AllReduceDP,
    AllReduceTP,
    AllGatherTP,
    ReduceScatterTP,
    Send,
    Recv,
}

/// Memory statistics for 3D parallelism
#[derive(Debug, Clone)]
pub struct Memory3DStats {
    pub model_memory: usize,
    pub activation_memory: usize,
    pub gradient_memory: usize,
    pub optimizer_memory: usize,
    pub total_memory: usize,
    pub peak_memory: usize,
    pub memory_efficiency: f32,
}

impl Default for Memory3DStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory3DStats {
    pub fn new() -> Self {
        Self {
            model_memory: 0,
            activation_memory: 0,
            gradient_memory: 0,
            optimizer_memory: 0,
            total_memory: 0,
            peak_memory: 0,
            memory_efficiency: 0.0,
        }
    }
}
