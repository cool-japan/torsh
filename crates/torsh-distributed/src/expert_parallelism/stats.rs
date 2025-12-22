//! Routing statistics and monitoring for expert parallelism
//!
//! This module provides comprehensive statistics tracking and monitoring capabilities
//! for expert routing decisions, load balancing effectiveness, and system performance.

use super::router::RoutingDecision;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Routing statistics for monitoring and debugging
///
/// Tracks various metrics related to expert routing performance, efficiency,
/// and utilization patterns over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingStats {
    /// Total number of routing operations performed
    pub total_routings: u64,
    /// Total number of tokens processed
    pub total_tokens: u64,
    /// Total number of tokens dropped due to capacity constraints
    pub tokens_dropped: u64,
    /// Expert utilization rates (running average per expert)
    pub expert_utilization: Vec<f32>,
    /// Average load balance loss across all routing operations
    pub average_load_balance_loss: f32,
    /// Average router z-loss for numerical stability
    pub average_router_z_loss: f32,
    /// Overall routing efficiency as a percentage
    pub routing_efficiency: f32,
    /// Per-expert token assignment counts
    pub expert_token_counts: Vec<u64>,
    /// Routing latency statistics
    pub routing_latency_stats: LatencyStats,
    /// Expert load variance over time
    pub load_variance_history: Vec<f32>,
    /// Capacity utilization statistics
    pub capacity_stats: CapacityStats,
}

impl RoutingStats {
    /// Create a new routing statistics tracker
    pub fn new() -> Self {
        Self {
            total_routings: 0,
            total_tokens: 0,
            tokens_dropped: 0,
            expert_utilization: Vec::new(),
            average_load_balance_loss: 0.0,
            average_router_z_loss: 0.0,
            routing_efficiency: 0.0,
            expert_token_counts: Vec::new(),
            routing_latency_stats: LatencyStats::new(),
            load_variance_history: Vec::new(),
            capacity_stats: CapacityStats::new(),
        }
    }

    /// Record a routing decision and update statistics
    ///
    /// # Arguments
    ///
    /// * `routing_decision` - The routing decision to record
    pub fn record_routing(&mut self, routing_decision: &RoutingDecision) {
        self.total_routings += 1;
        self.total_tokens += routing_decision.total_tokens as u64;
        self.tokens_dropped += routing_decision.tokens_dropped as u64;

        // Update running averages using exponential moving average
        let alpha = 1.0 / self.total_routings as f32;
        self.average_load_balance_loss = alpha * routing_decision.load_balance_loss
            + (1.0 - alpha) * self.average_load_balance_loss;
        self.average_router_z_loss =
            alpha * routing_decision.router_z_loss + (1.0 - alpha) * self.average_router_z_loss;

        // Calculate routing efficiency
        if self.total_tokens > 0 {
            self.routing_efficiency =
                (self.total_tokens - self.tokens_dropped) as f32 / self.total_tokens as f32 * 100.0;
        }

        // Update expert utilization
        if self.expert_utilization.len() != routing_decision.expert_capacities.len() {
            self.expert_utilization = vec![0.0; routing_decision.expert_capacities.len()];
            self.expert_token_counts = vec![0; routing_decision.expert_capacities.len()];
        }

        for (i, &capacity) in routing_decision.expert_capacities.iter().enumerate() {
            if i < self.expert_utilization.len() {
                let utilization = if routing_decision.total_tokens > 0 {
                    capacity as f32 / routing_decision.total_tokens as f32
                } else {
                    0.0
                };
                self.expert_utilization[i] =
                    alpha * utilization + (1.0 - alpha) * self.expert_utilization[i];
                self.expert_token_counts[i] += capacity as u64;
            }
        }

        // Calculate and record load variance
        let load_variance = self.calculate_load_variance(&routing_decision.expert_capacities);
        self.load_variance_history.push(load_variance);

        // Limit history size
        if self.load_variance_history.len() > 1000 {
            self.load_variance_history.remove(0);
        }

        // Update capacity statistics
        self.capacity_stats.update(routing_decision);
    }

    /// Record routing latency for performance monitoring
    ///
    /// # Arguments
    ///
    /// * `latency` - The latency of the routing operation
    pub fn record_routing_latency(&mut self, latency: Duration) {
        self.routing_latency_stats.record_latency(latency);
    }

    /// Calculate load variance for a given set of expert capacities
    fn calculate_load_variance(&self, capacities: &[usize]) -> f32 {
        if capacities.is_empty() {
            return 0.0;
        }

        let mean = capacities.iter().sum::<usize>() as f32 / capacities.len() as f32;
        let variance = capacities
            .iter()
            .map(|&cap| {
                let diff = cap as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / capacities.len() as f32;

        variance
    }

    /// Get the coefficient of variation for expert utilization
    pub fn utilization_cv(&self) -> f32 {
        if self.expert_utilization.is_empty() {
            return 0.0;
        }

        let mean =
            self.expert_utilization.iter().sum::<f32>() / self.expert_utilization.len() as f32;
        if mean <= 0.0 {
            return 0.0;
        }

        let variance = self
            .expert_utilization
            .iter()
            .map(|&util| {
                let diff = util - mean;
                diff * diff
            })
            .sum::<f32>()
            / self.expert_utilization.len() as f32;

        variance.sqrt() / mean
    }

    /// Get the most utilized expert
    pub fn most_utilized_expert(&self) -> Option<(usize, f32)> {
        self.expert_utilization
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, &util)| (idx, util))
    }

    /// Get the least utilized expert
    pub fn least_utilized_expert(&self) -> Option<(usize, f32)> {
        self.expert_utilization
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, &util)| (idx, util))
    }

    /// Get utilization statistics summary
    pub fn utilization_summary(&self) -> HashMap<String, f32> {
        let mut summary = HashMap::new();

        if !self.expert_utilization.is_empty() {
            let mean =
                self.expert_utilization.iter().sum::<f32>() / self.expert_utilization.len() as f32;
            let min = self
                .expert_utilization
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let max = self
                .expert_utilization
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            summary.insert("mean_utilization".to_string(), mean);
            summary.insert("min_utilization".to_string(), min);
            summary.insert("max_utilization".to_string(), max);
            summary.insert("utilization_cv".to_string(), self.utilization_cv());
        }

        summary.insert("routing_efficiency".to_string(), self.routing_efficiency);
        summary.insert(
            "average_load_balance_loss".to_string(),
            self.average_load_balance_loss,
        );
        summary.insert(
            "average_router_z_loss".to_string(),
            self.average_router_z_loss,
        );

        summary
    }

    /// Get recent load variance trend
    pub fn recent_load_variance_trend(&self, window: usize) -> f32 {
        if self.load_variance_history.len() < 2 {
            return 0.0;
        }

        let start_idx = self.load_variance_history.len().saturating_sub(window);
        let recent_variances = &self.load_variance_history[start_idx..];

        if recent_variances.len() < 2 {
            return 0.0;
        }

        // Simple linear trend calculation
        let n = recent_variances.len() as f32;
        let sum_x: f32 = (0..recent_variances.len()).map(|i| i as f32).sum();
        let sum_y: f32 = recent_variances.iter().sum();
        let sum_xy: f32 = recent_variances
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let sum_x2: f32 = (0..recent_variances.len())
            .map(|i| (i as f32).powi(2))
            .sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f32::EPSILON {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get throughput statistics
    pub fn throughput_stats(&self) -> ThroughputStats {
        ThroughputStats {
            total_tokens: self.total_tokens,
            total_routings: self.total_routings,
            tokens_per_routing: if self.total_routings > 0 {
                self.total_tokens as f32 / self.total_routings as f32
            } else {
                0.0
            },
            routing_efficiency: self.routing_efficiency,
            average_latency: self.routing_latency_stats.average_latency(),
        }
    }
}

impl Default for RoutingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency statistics for routing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Total latency measurements
    pub total_measurements: u64,
    /// Sum of all latencies in milliseconds
    pub total_latency_ms: f64,
    /// Minimum observed latency
    pub min_latency_ms: f64,
    /// Maximum observed latency
    pub max_latency_ms: f64,
    /// Recent latency measurements for percentile calculation
    pub recent_latencies: Vec<f64>,
}

impl LatencyStats {
    /// Create new latency statistics
    pub fn new() -> Self {
        Self {
            total_measurements: 0,
            total_latency_ms: 0.0,
            min_latency_ms: f64::INFINITY,
            max_latency_ms: 0.0,
            recent_latencies: Vec::new(),
        }
    }

    /// Record a latency measurement
    pub fn record_latency(&mut self, latency: Duration) {
        let latency_ms = latency.as_secs_f64() * 1000.0;

        self.total_measurements += 1;
        self.total_latency_ms += latency_ms;
        self.min_latency_ms = self.min_latency_ms.min(latency_ms);
        self.max_latency_ms = self.max_latency_ms.max(latency_ms);

        self.recent_latencies.push(latency_ms);

        // Keep only recent measurements for percentile calculation
        if self.recent_latencies.len() > 1000 {
            self.recent_latencies.remove(0);
        }
    }

    /// Get average latency in milliseconds
    pub fn average_latency(&self) -> f64 {
        if self.total_measurements > 0 {
            self.total_latency_ms / self.total_measurements as f64
        } else {
            0.0
        }
    }

    /// Get latency percentile
    pub fn percentile(&self, p: f64) -> f64 {
        if self.recent_latencies.is_empty() {
            return 0.0;
        }

        let mut sorted_latencies = self.recent_latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((p / 100.0) * (sorted_latencies.len() - 1) as f64) as usize;
        sorted_latencies[index.min(sorted_latencies.len() - 1)]
    }

    /// Get 95th percentile latency
    pub fn p95_latency(&self) -> f64 {
        self.percentile(95.0)
    }

    /// Get 99th percentile latency
    pub fn p99_latency(&self) -> f64 {
        self.percentile(99.0)
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Capacity utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityStats {
    /// Average capacity utilization across all experts
    pub average_utilization: f32,
    /// Peak capacity utilization observed
    pub peak_utilization: f32,
    /// Number of times capacity was exceeded
    pub capacity_exceeded_count: u64,
    /// Total available capacity across all experts
    pub total_capacity: u64,
    /// Total used capacity across all experts
    pub total_used: u64,
}

impl CapacityStats {
    /// Create new capacity statistics
    pub fn new() -> Self {
        Self {
            average_utilization: 0.0,
            peak_utilization: 0.0,
            capacity_exceeded_count: 0,
            total_capacity: 0,
            total_used: 0,
        }
    }

    /// Update capacity statistics with a routing decision
    pub fn update(&mut self, routing_decision: &RoutingDecision) {
        let current_utilization = routing_decision.expert_capacities.iter().sum::<usize>() as f32
            / (routing_decision.expert_capacities.len() as f32 * 100.0); // Assuming capacity of 100 per expert

        // Update average utilization
        let alpha = 0.1; // Smoothing factor
        self.average_utilization =
            alpha * current_utilization + (1.0 - alpha) * self.average_utilization;

        // Update peak utilization
        self.peak_utilization = self.peak_utilization.max(current_utilization);

        // Count capacity exceeded events
        if routing_decision.tokens_dropped > 0 {
            self.capacity_exceeded_count += 1;
        }

        // Update totals
        self.total_used += routing_decision.expert_capacities.iter().sum::<usize>() as u64;
        self.total_capacity += routing_decision.expert_capacities.len() as u64 * 100;
        // Assuming capacity of 100 per expert
    }

    /// Get overall utilization percentage
    pub fn overall_utilization(&self) -> f32 {
        if self.total_capacity > 0 {
            (self.total_used as f32 / self.total_capacity as f32) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for CapacityStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total routing operations
    pub total_routings: u64,
    /// Average tokens per routing operation
    pub tokens_per_routing: f32,
    /// Routing efficiency percentage
    pub routing_efficiency: f32,
    /// Average routing latency in milliseconds
    pub average_latency: f64,
}

impl ThroughputStats {
    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        if self.average_latency > 0.0 {
            (self.tokens_per_routing as f64 * 1000.0) / self.average_latency
        } else {
            0.0
        }
    }

    /// Calculate routings per second
    pub fn routings_per_second(&self) -> f64 {
        if self.average_latency > 0.0 {
            1000.0 / self.average_latency
        } else {
            0.0
        }
    }
}

/// Performance monitoring utilities
pub mod monitoring {
    use super::*;

    /// Performance monitor for expert routing system
    pub struct PerformanceMonitor {
        stats: RoutingStats,
        start_time: Instant,
        last_report_time: Instant,
        report_interval: Duration,
    }

    impl PerformanceMonitor {
        /// Create a new performance monitor
        pub fn new(report_interval: Duration) -> Self {
            let now = Instant::now();
            Self {
                stats: RoutingStats::new(),
                start_time: now,
                last_report_time: now,
                report_interval,
            }
        }

        /// Record a routing decision
        pub fn record_routing(&mut self, routing_decision: &RoutingDecision, latency: Duration) {
            self.stats.record_routing(routing_decision);
            self.stats.record_routing_latency(latency);

            // Check if it's time for a report
            if self.last_report_time.elapsed() >= self.report_interval {
                self.print_report();
                self.last_report_time = Instant::now();
            }
        }

        /// Print performance report
        pub fn print_report(&self) {
            let uptime = self.start_time.elapsed();
            let throughput = self.stats.throughput_stats();

            info!("🔍 Expert Routing Performance Report");
            info!("  Uptime: {:.2}s", uptime.as_secs_f64());
            info!("  Total routings: {}", self.stats.total_routings);
            info!("  Total tokens: {}", self.stats.total_tokens);
            info!(
                "  Routing efficiency: {:.2}%",
                self.stats.routing_efficiency
            );
            info!("  Tokens/second: {:.2}", throughput.tokens_per_second());
            info!(
                "  Average latency: {:.2}ms",
                self.stats.routing_latency_stats.average_latency()
            );
            info!(
                "  P95 latency: {:.2}ms",
                self.stats.routing_latency_stats.p95_latency()
            );
            info!("  Utilization CV: {:.3}", self.stats.utilization_cv());

            if let Some((idx, util)) = self.stats.most_utilized_expert() {
                info!("  Most utilized expert: {} ({:.2}%)", idx, util * 100.0);
            }
            if let Some((idx, util)) = self.stats.least_utilized_expert() {
                info!("  Least utilized expert: {} ({:.2}%)", idx, util * 100.0);
            }
        }

        /// Get current statistics
        pub fn stats(&self) -> &RoutingStats {
            &self.stats
        }

        /// Reset statistics
        pub fn reset(&mut self) {
            self.stats.reset();
            self.start_time = Instant::now();
            self.last_report_time = Instant::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_parallelism::router::{ExpertAssignment, RoutingDecision};

    #[test]
    fn test_routing_stats_creation() {
        let stats = RoutingStats::new();
        assert_eq!(stats.total_routings, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.routing_efficiency, 0.0);
    }

    #[test]
    fn test_routing_stats_recording() {
        let mut stats = RoutingStats::new();

        let routing_decision = RoutingDecision {
            expert_assignments: vec![vec![ExpertAssignment::new(0, 0.8, 0, 0)]],
            expert_capacities: vec![5, 3, 2, 0],
            total_tokens: 10,
            tokens_dropped: 0,
            load_balance_loss: 0.1,
            router_z_loss: 0.05,
            auxiliary_loss: 0.15,
        };

        stats.record_routing(&routing_decision);

        assert_eq!(stats.total_routings, 1);
        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.tokens_dropped, 0);
        assert_eq!(stats.routing_efficiency, 100.0);
        assert_eq!(stats.expert_utilization.len(), 4);
    }

    #[test]
    fn test_latency_stats() {
        let mut latency_stats = LatencyStats::new();

        latency_stats.record_latency(Duration::from_millis(10));
        latency_stats.record_latency(Duration::from_millis(20));
        latency_stats.record_latency(Duration::from_millis(30));

        assert_eq!(latency_stats.total_measurements, 3);
        assert_eq!(latency_stats.average_latency(), 20.0);
        assert_eq!(latency_stats.min_latency_ms, 10.0);
        assert_eq!(latency_stats.max_latency_ms, 30.0);
    }

    #[test]
    fn test_utilization_cv() {
        let mut stats = RoutingStats::new();
        stats.expert_utilization = vec![0.1, 0.2, 0.3, 0.4]; // Varied utilization

        let cv = stats.utilization_cv();
        assert!(cv > 0.0); // Should have some variance
    }

    #[test]
    fn test_capacity_stats() {
        let mut capacity_stats = CapacityStats::new();

        let routing_decision = RoutingDecision {
            expert_assignments: vec![],
            expert_capacities: vec![50, 75, 25, 100], // Mixed utilization
            total_tokens: 250,
            tokens_dropped: 0,
            load_balance_loss: 0.0,
            router_z_loss: 0.0,
            auxiliary_loss: 0.0,
        };

        capacity_stats.update(&routing_decision);
        assert!(capacity_stats.average_utilization > 0.0);
        assert!(capacity_stats.peak_utilization > 0.0);
    }

    #[test]
    fn test_throughput_stats() {
        let throughput = ThroughputStats {
            total_tokens: 1000,
            total_routings: 10,
            tokens_per_routing: 100.0,
            routing_efficiency: 95.0,
            average_latency: 50.0, // 50ms
        };

        assert_eq!(throughput.tokens_per_second(), 2000.0); // (100 * 1000) / 50
        assert_eq!(throughput.routings_per_second(), 20.0); // 1000 / 50
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = monitoring::PerformanceMonitor::new(Duration::from_secs(1));

        let routing_decision = RoutingDecision {
            expert_assignments: vec![],
            expert_capacities: vec![10, 20, 30],
            total_tokens: 60,
            tokens_dropped: 0,
            load_balance_loss: 0.1,
            router_z_loss: 0.05,
            auxiliary_loss: 0.15,
        };

        monitor.record_routing(&routing_decision, Duration::from_millis(25));

        assert_eq!(monitor.stats().total_routings, 1);
        assert_eq!(monitor.stats().total_tokens, 60);
    }
}
