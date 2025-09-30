//! Metrics and statistics collection for distributed autograd operations

use crate::distributed::common::types::SyncOperationType;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Statistics for distributed gradient accumulation
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total number of communication operations
    pub total_communications: usize,
    /// Total communication time
    pub total_comm_time: Duration,
    /// Total data communicated (bytes)
    pub total_data_communicated: usize,
    /// Average communication bandwidth (MB/s)
    pub avg_bandwidth_mbps: f64,
    /// Number of gradient accumulation steps
    pub accumulation_steps: usize,
    /// Compression ratio (if compression is used)
    pub compression_ratio: f64,
    /// Synchronization overhead
    pub sync_overhead: Duration,
}

impl DistributedStats {
    /// Create new default statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update communication statistics
    pub fn update_communication(&mut self, data_size: usize, duration: Duration) {
        self.total_communications += 1;
        self.total_comm_time += duration;
        self.total_data_communicated += data_size;

        // Update average bandwidth
        if self.total_comm_time.as_secs_f64() > 0.0 {
            self.avg_bandwidth_mbps = (self.total_data_communicated as f64 / (1024.0 * 1024.0))
                / self.total_comm_time.as_secs_f64();
        }
    }

    /// Update accumulation step count
    pub fn increment_accumulation_step(&mut self) {
        self.accumulation_steps += 1;
    }

    /// Set compression ratio
    pub fn set_compression_ratio(&mut self, ratio: f64) {
        self.compression_ratio = ratio;
    }

    /// Add synchronization overhead
    pub fn add_sync_overhead(&mut self, overhead: Duration) {
        self.sync_overhead += overhead;
    }

    /// Get efficiency metrics
    pub fn efficiency_metrics(&self) -> EfficiencyMetrics {
        EfficiencyMetrics {
            communication_efficiency: if self.accumulation_steps > 0 {
                self.avg_bandwidth_mbps / self.accumulation_steps as f64
            } else {
                0.0
            },
            compression_effectiveness: self.compression_ratio,
            overhead_ratio: if self.total_comm_time.as_secs_f64() > 0.0 {
                self.sync_overhead.as_secs_f64() / self.total_comm_time.as_secs_f64()
            } else {
                0.0
            },
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Efficiency metrics derived from statistics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub communication_efficiency: f64,
    pub compression_effectiveness: f64,
    pub overhead_ratio: f64,
}

/// Gradient statistics for consistency checking
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Mean magnitude
    pub mean_magnitude: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Checksum for verification
    pub checksum: u64,
    /// Min/max values
    pub min_value: f64,
    pub max_value: f64,
    /// Number of elements
    pub num_elements: usize,
    /// Sparsity ratio (fraction of near-zero elements)
    pub sparsity: f64,
}

impl GradientStats {
    /// Create statistics from gradient data
    pub fn from_gradient_slice<T: torsh_core::dtype::FloatElement>(data: &[T]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut zero_count = 0;
        let mut checksum = 0u64;

        for (i, &value) in data.iter().enumerate() {
            let val = torsh_core::TensorElement::to_f64(&value).unwrap_or(0.0);

            sum += val;
            sum_squares += val * val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);

            if val.abs()
                < T::from(1e-8)
                    .map(|x| torsh_core::TensorElement::to_f64(&x).unwrap_or(1e-8))
                    .unwrap_or(1e-8)
            {
                zero_count += 1;
            }

            // Simple checksum calculation
            checksum ^= (val.to_bits() as u64).wrapping_mul(i as u64 + 1);
        }

        let count = data.len() as f64;
        let mean = sum / count;
        let variance = (sum_squares / count) - (mean * mean);
        let std_dev = variance.sqrt();

        Self {
            mean_magnitude: mean.abs(),
            std_dev,
            checksum,
            min_value: min_val,
            max_value: max_val,
            num_elements: data.len(),
            sparsity: zero_count as f64 / count,
        }
    }

    /// Check if statistics are within tolerance of expected values
    pub fn is_consistent_with(&self, expected: &Self, tolerance: f64) -> bool {
        let mean_diff = (self.mean_magnitude - expected.mean_magnitude).abs();
        let std_diff = (self.std_dev - expected.std_dev).abs();

        mean_diff <= tolerance * expected.mean_magnitude.max(1e-8)
            && std_diff <= tolerance * expected.std_dev.max(1e-8)
            && self.checksum == expected.checksum
    }

    /// Get a summary string of the statistics
    pub fn summary(&self) -> String {
        format!(
            "GradientStats {{ mean_mag: {:.6}, std: {:.6}, range: [{:.6}, {:.6}], sparsity: {:.3}, elements: {} }}",
            self.mean_magnitude, self.std_dev, self.min_value, self.max_value, self.sparsity, self.num_elements
        )
    }
}

impl Default for GradientStats {
    fn default() -> Self {
        Self {
            mean_magnitude: 0.0,
            std_dev: 0.0,
            checksum: 0,
            min_value: 0.0,
            max_value: 0.0,
            num_elements: 0,
            sparsity: 0.0,
        }
    }
}

/// Comprehensive synchronization statistics
#[derive(Debug, Clone, Default)]
pub struct SyncStatistics {
    /// Total synchronization operations
    pub total_operations: usize,
    /// Successful operations
    pub successful_operations: usize,
    /// Failed operations
    pub failed_operations: usize,
    /// Average synchronization time
    pub avg_sync_time: Duration,
    /// Bandwidth utilization
    pub avg_bandwidth_utilization: f64,
    /// Consistency check results
    pub consistency_success_rate: f64,
    /// Operation breakdown by type
    pub operation_breakdown: HashMap<SyncOperationType, usize>,
    /// Historical sync times
    sync_times: VecDeque<Duration>,
    /// Total consistency checks
    total_consistency_checks: usize,
    /// Successful consistency checks
    successful_consistency_checks: usize,
}

impl SyncStatistics {
    /// Create new sync statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful operation
    pub fn record_success(&mut self, operation_type: SyncOperationType, duration: Duration) {
        self.total_operations += 1;
        self.successful_operations += 1;

        *self.operation_breakdown.entry(operation_type).or_insert(0) += 1;

        // Update average sync time
        self.sync_times.push_back(duration);
        if self.sync_times.len() > 100 {
            self.sync_times.pop_front();
        }

        let sum: Duration = self.sync_times.iter().sum();
        self.avg_sync_time = sum / self.sync_times.len() as u32;
    }

    /// Record a failed operation
    pub fn record_failure(&mut self, operation_type: SyncOperationType) {
        self.total_operations += 1;
        self.failed_operations += 1;

        *self.operation_breakdown.entry(operation_type).or_insert(0) += 1;
    }

    /// Record consistency check result
    pub fn record_consistency_check(&mut self, success: bool) {
        self.total_consistency_checks += 1;
        if success {
            self.successful_consistency_checks += 1;
        }

        self.consistency_success_rate = if self.total_consistency_checks > 0 {
            self.successful_consistency_checks as f64 / self.total_consistency_checks as f64
        } else {
            0.0
        };
    }

    /// Update bandwidth utilization
    pub fn update_bandwidth_utilization(&mut self, utilization: f64) {
        self.avg_bandwidth_utilization = utilization;
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_operations > 0 {
            self.successful_operations as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }

    /// Get failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.total_operations > 0 {
            self.failed_operations as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }

    /// Get the most common operation type
    pub fn most_common_operation(&self) -> Option<(SyncOperationType, usize)> {
        self.operation_breakdown
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(op, &count)| (op.clone(), count))
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            success_rate: self.success_rate(),
            avg_sync_time: self.avg_sync_time,
            total_operations: self.total_operations,
            bandwidth_utilization: self.avg_bandwidth_utilization,
            consistency_rate: self.consistency_success_rate,
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Performance summary metrics
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub success_rate: f64,
    pub avg_sync_time: Duration,
    pub total_operations: usize,
    pub bandwidth_utilization: f64,
    pub consistency_rate: f64,
}

impl PerformanceSummary {
    /// Get overall performance score (0.0 to 1.0)
    pub fn overall_score(&self) -> f64 {
        let success_weight = 0.4;
        let bandwidth_weight = 0.3;
        let consistency_weight = 0.2;
        let latency_weight = 0.1;

        // Normalize latency score (lower is better)
        let latency_score = if self.avg_sync_time.as_millis() > 0 {
            1.0 / (1.0 + self.avg_sync_time.as_millis() as f64 / 1000.0)
        } else {
            1.0
        };

        success_weight * self.success_rate
            + bandwidth_weight * self.bandwidth_utilization
            + consistency_weight * self.consistency_rate
            + latency_weight * latency_score
    }

    /// Get human-readable performance rating
    pub fn rating(&self) -> &'static str {
        let score = self.overall_score();
        match score {
            s if s >= 0.9 => "Excellent",
            s if s >= 0.8 => "Good",
            s if s >= 0.7 => "Fair",
            s if s >= 0.6 => "Poor",
            _ => "Critical",
        }
    }
}

/// Metrics collector for aggregating various statistics
#[derive(Debug)]
pub struct MetricsCollector {
    distributed_stats: DistributedStats,
    sync_stats: SyncStatistics,
    gradient_stats_history: VecDeque<GradientStats>,
    collection_start: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            distributed_stats: DistributedStats::new(),
            sync_stats: SyncStatistics::new(),
            gradient_stats_history: VecDeque::new(),
            collection_start: Instant::now(),
        }
    }

    /// Get distributed statistics
    pub fn distributed_stats(&self) -> &DistributedStats {
        &self.distributed_stats
    }

    /// Get mutable distributed statistics
    pub fn distributed_stats_mut(&mut self) -> &mut DistributedStats {
        &mut self.distributed_stats
    }

    /// Get sync statistics
    pub fn sync_stats(&self) -> &SyncStatistics {
        &self.sync_stats
    }

    /// Get mutable sync statistics
    pub fn sync_stats_mut(&mut self) -> &mut SyncStatistics {
        &mut self.sync_stats
    }

    /// Record gradient statistics
    pub fn record_gradient_stats(&mut self, stats: GradientStats) {
        self.gradient_stats_history.push_back(stats);
        if self.gradient_stats_history.len() > 1000 {
            self.gradient_stats_history.pop_front();
        }
    }

    /// Get latest gradient statistics
    pub fn latest_gradient_stats(&self) -> Option<&GradientStats> {
        self.gradient_stats_history.back()
    }

    /// Get gradient statistics history
    pub fn gradient_stats_history(&self) -> &VecDeque<GradientStats> {
        &self.gradient_stats_history
    }

    /// Get collection duration
    pub fn collection_duration(&self) -> Duration {
        self.collection_start.elapsed()
    }

    /// Get comprehensive metrics report
    pub fn report(&self) -> MetricsReport {
        MetricsReport {
            distributed_stats: self.distributed_stats.clone(),
            performance_summary: self.sync_stats.performance_summary(),
            latest_gradient_stats: self.latest_gradient_stats().cloned(),
            collection_duration: self.collection_duration(),
            efficiency_metrics: self.distributed_stats.efficiency_metrics(),
        }
    }

    /// Reset all collected metrics
    pub fn reset(&mut self) {
        self.distributed_stats.reset();
        self.sync_stats.reset();
        self.gradient_stats_history.clear();
        self.collection_start = Instant::now();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive metrics report
#[derive(Debug, Clone)]
pub struct MetricsReport {
    pub distributed_stats: DistributedStats,
    pub performance_summary: PerformanceSummary,
    pub latest_gradient_stats: Option<GradientStats>,
    pub collection_duration: Duration,
    pub efficiency_metrics: EfficiencyMetrics,
}

impl MetricsReport {
    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("=== Distributed Training Metrics Report ===\n"));
        summary.push_str(&format!(
            "Collection Duration: {:.2}s\n",
            self.collection_duration.as_secs_f64()
        ));
        summary.push_str(&format!(
            "Performance Rating: {}\n",
            self.performance_summary.rating()
        ));
        summary.push_str(&format!(
            "Overall Score: {:.3}\n\n",
            self.performance_summary.overall_score()
        ));

        summary.push_str(&format!("Communication Stats:\n"));
        summary.push_str(&format!(
            "  - Operations: {}\n",
            self.distributed_stats.total_communications
        ));
        summary.push_str(&format!(
            "  - Data Transferred: {:.2} MB\n",
            self.distributed_stats.total_data_communicated as f64 / (1024.0 * 1024.0)
        ));
        summary.push_str(&format!(
            "  - Avg Bandwidth: {:.2} MB/s\n",
            self.distributed_stats.avg_bandwidth_mbps
        ));
        summary.push_str(&format!(
            "  - Compression Ratio: {:.3}\n\n",
            self.distributed_stats.compression_ratio
        ));

        summary.push_str(&format!("Performance Stats:\n"));
        summary.push_str(&format!(
            "  - Success Rate: {:.1}%\n",
            self.performance_summary.success_rate * 100.0
        ));
        summary.push_str(&format!(
            "  - Avg Sync Time: {:.2}ms\n",
            self.performance_summary.avg_sync_time.as_millis()
        ));
        summary.push_str(&format!(
            "  - Bandwidth Utilization: {:.1}%\n",
            self.performance_summary.bandwidth_utilization * 100.0
        ));
        summary.push_str(&format!(
            "  - Consistency Rate: {:.1}%\n",
            self.performance_summary.consistency_rate * 100.0
        ));

        if let Some(ref grad_stats) = self.latest_gradient_stats {
            summary.push_str(&format!("\nLatest Gradient Stats:\n"));
            summary.push_str(&format!("  - {}\n", grad_stats.summary()));
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_stats() {
        let mut stats = DistributedStats::new();

        stats.update_communication(1024, Duration::from_millis(10));
        assert_eq!(stats.total_communications, 1);
        assert_eq!(stats.total_data_communicated, 1024);

        stats.increment_accumulation_step();
        assert_eq!(stats.accumulation_steps, 1);

        let efficiency = stats.efficiency_metrics();
        assert!(efficiency.communication_efficiency > 0.0);
    }

    #[test]
    fn test_gradient_stats() {
        let data = vec![1.0f32, 2.0, 3.0, 0.0, -1.0];
        let stats = GradientStats::from_gradient_slice(&data);

        assert_eq!(stats.num_elements, 5);
        assert!(stats.mean_magnitude > 0.0);
        assert!(stats.sparsity > 0.0);
        assert_eq!(stats.min_value, -1.0);
        assert_eq!(stats.max_value, 3.0);
    }

    #[test]
    fn test_sync_statistics() {
        let mut stats = SyncStatistics::new();

        stats.record_success(
            SyncOperationType::AllReduce(
                crate::distributed::common::types::AllReduceAlgorithm::Ring,
            ),
            Duration::from_millis(5),
        );
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);
        assert_eq!(stats.success_rate(), 1.0);

        stats.record_failure(SyncOperationType::AllReduce(
            crate::distributed::common::types::AllReduceAlgorithm::Ring,
        ));
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.success_rate(), 0.5);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        collector
            .distributed_stats_mut()
            .increment_accumulation_step();
        collector.sync_stats_mut().record_success(
            SyncOperationType::AllReduce(
                crate::distributed::common::types::AllReduceAlgorithm::Ring,
            ),
            Duration::from_millis(10),
        );

        let report = collector.report();
        assert_eq!(report.distributed_stats.accumulation_steps, 1);
        assert!(report.performance_summary.success_rate > 0.0);
    }

    #[test]
    fn test_performance_summary() {
        let summary = PerformanceSummary {
            success_rate: 0.95,
            avg_sync_time: Duration::from_millis(10),
            total_operations: 100,
            bandwidth_utilization: 0.85,
            consistency_rate: 0.98,
        };

        assert!(summary.overall_score() > 0.8);
        // With high metrics (0.95, 0.85, 0.98, 10ms), should get "Excellent"
        assert_eq!(summary.rating(), "Excellent");
    }
}
