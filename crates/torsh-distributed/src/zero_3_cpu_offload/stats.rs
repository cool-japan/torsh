//! Performance and Memory Statistics for ZeRO-3 CPU Offloading
//!
//! This module provides comprehensive statistics collection and analysis for ZeRO-3
//! (Zero Redundancy Optimizer Stage 3) with CPU offloading. It tracks performance
//! metrics, memory usage patterns, throughput statistics, and provides detailed
//! insights for optimization and monitoring.

use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive performance statistics for ZeRO-3 operations
///
/// Tracks all aspects of ZeRO-3 performance including:
/// - Forward and backward pass timing
/// - Parameter transfer and optimization statistics
/// - Memory management performance
/// - Distributed synchronization metrics
/// - Throughput and efficiency measurements
#[derive(Debug, Clone)]
pub struct Zero3PerformanceStats {
    /// Number of forward passes completed
    pub forward_passes: u64,
    /// Number of backward passes completed
    pub backward_passes: u64,
    /// Number of optimizer steps completed
    pub optimizer_steps: u64,
    /// Total time spent in forward passes
    pub total_forward_time: Duration,
    /// Total time spent in backward passes
    pub total_backward_time: Duration,
    /// Total time spent in optimizer steps
    pub total_optimizer_time: Duration,
    /// Time spent transferring parameters between CPU/GPU
    pub parameter_transfer_time: Duration,
    /// Time spent synchronizing gradients across ranks
    pub gradient_sync_time: Duration,
    /// Per-layer execution timings
    pub layer_timings: HashMap<String, LayerTimingStats>,
    /// Throughput metrics
    pub throughput_metrics: ThroughputMetrics,
    /// Memory transfer performance
    pub memory_transfer_metrics: MemoryTransferMetrics,
    /// Distributed communication statistics
    pub communication_stats: CommunicationStats,
    /// Optimization efficiency metrics
    pub optimization_efficiency: OptimizationEfficiency,
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
            throughput_metrics: ThroughputMetrics::new(),
            memory_transfer_metrics: MemoryTransferMetrics::new(),
            communication_stats: CommunicationStats::new(),
            optimization_efficiency: OptimizationEfficiency::new(),
        }
    }

    /// Record a completed forward pass
    pub fn record_forward_pass(&mut self, duration: Duration, num_tokens: usize) {
        self.forward_passes += 1;
        self.total_forward_time += duration;
        self.throughput_metrics
            .record_forward_pass(duration, num_tokens);
        self.optimization_efficiency.record_forward_pass(duration);
    }

    /// Record a completed backward pass
    pub fn record_backward_pass(&mut self, duration: Duration, num_tokens: usize) {
        self.backward_passes += 1;
        self.total_backward_time += duration;
        self.throughput_metrics
            .record_backward_pass(duration, num_tokens);
        self.optimization_efficiency.record_backward_pass(duration);
    }

    /// Record a completed optimizer step
    pub fn record_optimizer_step(&mut self, duration: Duration, num_params: usize) {
        self.optimizer_steps += 1;
        self.total_optimizer_time += duration;
        self.optimization_efficiency
            .record_optimizer_step(duration, num_params);
    }

    /// Record layer execution timing
    pub fn record_layer_execution(&mut self, layer_name: String, duration: Duration) {
        let layer_stats = self.layer_timings.entry(layer_name.clone()).or_default();
        layer_stats.record_forward_execution(duration);
    }

    /// Record layer backward pass timing
    pub fn record_layer_backward(&mut self, layer_name: String, duration: Duration) {
        let layer_stats = self.layer_timings.entry(layer_name).or_default();
        layer_stats.record_backward_execution(duration);
    }

    /// Record parameter transfer operation
    pub fn record_parameter_transfer(
        &mut self,
        duration: Duration,
        bytes_transferred: usize,
        direction: TransferDirection,
    ) {
        self.parameter_transfer_time += duration;
        self.memory_transfer_metrics
            .record_transfer(duration, bytes_transferred, direction);
    }

    /// Record gradient synchronization
    pub fn record_gradient_sync(
        &mut self,
        duration: Duration,
        num_gradients: usize,
        world_size: usize,
    ) {
        self.gradient_sync_time += duration;
        self.communication_stats
            .record_gradient_sync(duration, num_gradients, world_size);
    }

    /// Record distributed communication operation
    pub fn record_communication(
        &mut self,
        operation: CommunicationOperation,
        duration: Duration,
        bytes: usize,
    ) {
        self.communication_stats
            .record_operation(operation, duration, bytes);
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

    /// Get tokens per second throughput
    pub fn get_tokens_per_second(&self) -> f64 {
        self.throughput_metrics.get_tokens_per_second()
    }

    /// Get memory transfer bandwidth in GB/s
    pub fn get_memory_bandwidth_gbps(&self) -> f64 {
        self.memory_transfer_metrics.get_bandwidth_gbps()
    }

    /// Get communication efficiency metrics
    pub fn get_communication_efficiency(&self) -> f64 {
        self.communication_stats.get_efficiency()
    }

    /// Get overall training efficiency score (0.0 to 1.0)
    pub fn get_training_efficiency(&self) -> f64 {
        self.optimization_efficiency.get_overall_efficiency()
    }

    /// Get detailed performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_operations: self.forward_passes + self.backward_passes + self.optimizer_steps,
            average_forward_time: self.average_forward_time(),
            average_backward_time: self.average_backward_time(),
            average_optimizer_time: self.average_optimizer_time(),
            tokens_per_second: self.get_tokens_per_second(),
            memory_bandwidth_gbps: self.get_memory_bandwidth_gbps(),
            communication_efficiency: self.get_communication_efficiency(),
            training_efficiency: self.get_training_efficiency(),
            memory_transfer_efficiency: self.memory_transfer_metrics.get_efficiency(),
            layer_performance: self.get_layer_performance_summary(),
        }
    }

    /// Get layer performance summary
    fn get_layer_performance_summary(&self) -> HashMap<String, LayerPerformanceSummary> {
        self.layer_timings
            .iter()
            .map(|(name, stats)| {
                (
                    name.clone(),
                    LayerPerformanceSummary {
                        total_executions: stats.forward_executions + stats.backward_executions,
                        average_forward_time: stats.average_forward_time(),
                        average_backward_time: stats.average_backward_time(),
                        total_time: stats.total_forward_time + stats.total_backward_time,
                    },
                )
            })
            .collect()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merge statistics from another instance (useful for distributed aggregation)
    pub fn merge(&mut self, other: &Zero3PerformanceStats) {
        self.forward_passes += other.forward_passes;
        self.backward_passes += other.backward_passes;
        self.optimizer_steps += other.optimizer_steps;
        self.total_forward_time += other.total_forward_time;
        self.total_backward_time += other.total_backward_time;
        self.total_optimizer_time += other.total_optimizer_time;
        self.parameter_transfer_time += other.parameter_transfer_time;
        self.gradient_sync_time += other.gradient_sync_time;

        // Merge layer timings
        for (layer_name, other_stats) in &other.layer_timings {
            let layer_stats = self.layer_timings.entry(layer_name.clone()).or_default();
            layer_stats.merge(other_stats);
        }

        self.throughput_metrics.merge(&other.throughput_metrics);
        self.memory_transfer_metrics
            .merge(&other.memory_transfer_metrics);
        self.communication_stats.merge(&other.communication_stats);
        self.optimization_efficiency
            .merge(&other.optimization_efficiency);
    }
}

impl Default for Zero3PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-layer timing statistics
#[derive(Debug, Clone)]
pub struct LayerTimingStats {
    /// Number of forward executions
    pub forward_executions: u64,
    /// Number of backward executions
    pub backward_executions: u64,
    /// Total time spent in forward passes
    pub total_forward_time: Duration,
    /// Total time spent in backward passes
    pub total_backward_time: Duration,
    /// Minimum forward execution time
    pub min_forward_time: Duration,
    /// Maximum forward execution time
    pub max_forward_time: Duration,
    /// Minimum backward execution time
    pub min_backward_time: Duration,
    /// Maximum backward execution time
    pub max_backward_time: Duration,
}

impl LayerTimingStats {
    pub fn new() -> Self {
        Self {
            forward_executions: 0,
            backward_executions: 0,
            total_forward_time: Duration::ZERO,
            total_backward_time: Duration::ZERO,
            min_forward_time: Duration::MAX,
            max_forward_time: Duration::ZERO,
            min_backward_time: Duration::MAX,
            max_backward_time: Duration::ZERO,
        }
    }

    pub fn record_forward_execution(&mut self, duration: Duration) {
        self.forward_executions += 1;
        self.total_forward_time += duration;
        self.min_forward_time = self.min_forward_time.min(duration);
        self.max_forward_time = self.max_forward_time.max(duration);
    }

    pub fn record_backward_execution(&mut self, duration: Duration) {
        self.backward_executions += 1;
        self.total_backward_time += duration;
        self.min_backward_time = self.min_backward_time.min(duration);
        self.max_backward_time = self.max_backward_time.max(duration);
    }

    pub fn average_forward_time(&self) -> Duration {
        if self.forward_executions > 0 {
            self.total_forward_time / self.forward_executions as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn average_backward_time(&self) -> Duration {
        if self.backward_executions > 0 {
            self.total_backward_time / self.backward_executions as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn merge(&mut self, other: &LayerTimingStats) {
        self.forward_executions += other.forward_executions;
        self.backward_executions += other.backward_executions;
        self.total_forward_time += other.total_forward_time;
        self.total_backward_time += other.total_backward_time;
        self.min_forward_time = self.min_forward_time.min(other.min_forward_time);
        self.max_forward_time = self.max_forward_time.max(other.max_forward_time);
        self.min_backward_time = self.min_backward_time.min(other.min_backward_time);
        self.max_backward_time = self.max_backward_time.max(other.max_backward_time);
    }
}

impl Default for LayerTimingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Throughput metrics for training operations
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Total tokens processed in forward passes
    pub total_forward_tokens: usize,
    /// Total tokens processed in backward passes
    pub total_backward_tokens: usize,
    /// Total time spent in forward passes
    pub total_forward_time: Duration,
    /// Total time spent in backward passes
    pub total_backward_time: Duration,
    /// Peak tokens per second observed
    pub peak_tokens_per_second: f64,
    /// Average tokens per second (rolling window)
    pub rolling_average_tps: f64,
    /// Number of samples in rolling average
    pub rolling_samples: u32,
}

impl ThroughputMetrics {
    pub fn new() -> Self {
        Self {
            total_forward_tokens: 0,
            total_backward_tokens: 0,
            total_forward_time: Duration::ZERO,
            total_backward_time: Duration::ZERO,
            peak_tokens_per_second: 0.0,
            rolling_average_tps: 0.0,
            rolling_samples: 0,
        }
    }

    pub fn record_forward_pass(&mut self, duration: Duration, num_tokens: usize) {
        self.total_forward_tokens += num_tokens;
        self.total_forward_time += duration;
        self.update_rolling_average(duration, num_tokens);
    }

    pub fn record_backward_pass(&mut self, duration: Duration, num_tokens: usize) {
        self.total_backward_tokens += num_tokens;
        self.total_backward_time += duration;
        self.update_rolling_average(duration, num_tokens);
    }

    fn update_rolling_average(&mut self, duration: Duration, num_tokens: usize) {
        if !duration.is_zero() {
            let current_tps = num_tokens as f64 / duration.as_secs_f64();
            self.peak_tokens_per_second = self.peak_tokens_per_second.max(current_tps);

            // Update rolling average with exponential decay
            let alpha = 0.1; // Smoothing factor
            if self.rolling_samples == 0 {
                self.rolling_average_tps = current_tps;
            } else {
                self.rolling_average_tps =
                    alpha * current_tps + (1.0 - alpha) * self.rolling_average_tps;
            }
            self.rolling_samples += 1;
        }
    }

    pub fn get_tokens_per_second(&self) -> f64 {
        let total_time = self.total_forward_time + self.total_backward_time;
        let total_tokens = self.total_forward_tokens + self.total_backward_tokens;

        if !total_time.is_zero() && total_tokens > 0 {
            total_tokens as f64 / total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_forward_tps(&self) -> f64 {
        if !self.total_forward_time.is_zero() && self.total_forward_tokens > 0 {
            self.total_forward_tokens as f64 / self.total_forward_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_backward_tps(&self) -> f64 {
        if !self.total_backward_time.is_zero() && self.total_backward_tokens > 0 {
            self.total_backward_tokens as f64 / self.total_backward_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn merge(&mut self, other: &ThroughputMetrics) {
        self.total_forward_tokens += other.total_forward_tokens;
        self.total_backward_tokens += other.total_backward_tokens;
        self.total_forward_time += other.total_forward_time;
        self.total_backward_time += other.total_backward_time;
        self.peak_tokens_per_second = self
            .peak_tokens_per_second
            .max(other.peak_tokens_per_second);

        // Merge rolling averages (weighted by sample count)
        let total_samples = self.rolling_samples + other.rolling_samples;
        if total_samples > 0 {
            let self_weight = self.rolling_samples as f64 / total_samples as f64;
            let other_weight = other.rolling_samples as f64 / total_samples as f64;
            self.rolling_average_tps =
                self_weight * self.rolling_average_tps + other_weight * other.rolling_average_tps;
            self.rolling_samples = total_samples;
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory transfer performance metrics
#[derive(Debug, Clone)]
pub struct MemoryTransferMetrics {
    /// Total bytes transferred CPU to GPU
    pub cpu_to_gpu_bytes: usize,
    /// Total bytes transferred GPU to CPU
    pub gpu_to_cpu_bytes: usize,
    /// Time spent in CPU to GPU transfers
    pub cpu_to_gpu_time: Duration,
    /// Time spent in GPU to CPU transfers
    pub gpu_to_cpu_time: Duration,
    /// Number of CPU to GPU transfers
    pub cpu_to_gpu_transfers: u64,
    /// Number of GPU to CPU transfers
    pub gpu_to_cpu_transfers: u64,
    /// Peak transfer bandwidth observed (bytes/sec)
    pub peak_bandwidth: f64,
    /// Transfer efficiency (actual vs theoretical bandwidth)
    pub transfer_efficiency: f64,
}

impl MemoryTransferMetrics {
    pub fn new() -> Self {
        Self {
            cpu_to_gpu_bytes: 0,
            gpu_to_cpu_bytes: 0,
            cpu_to_gpu_time: Duration::ZERO,
            gpu_to_cpu_time: Duration::ZERO,
            cpu_to_gpu_transfers: 0,
            gpu_to_cpu_transfers: 0,
            peak_bandwidth: 0.0,
            transfer_efficiency: 1.0,
        }
    }

    pub fn record_transfer(
        &mut self,
        duration: Duration,
        bytes: usize,
        direction: TransferDirection,
    ) {
        if !duration.is_zero() {
            let bandwidth = bytes as f64 / duration.as_secs_f64();
            self.peak_bandwidth = self.peak_bandwidth.max(bandwidth);
        }

        match direction {
            TransferDirection::CpuToGpu => {
                self.cpu_to_gpu_bytes += bytes;
                self.cpu_to_gpu_time += duration;
                self.cpu_to_gpu_transfers += 1;
            }
            TransferDirection::GpuToCpu => {
                self.gpu_to_cpu_bytes += bytes;
                self.gpu_to_cpu_time += duration;
                self.gpu_to_cpu_transfers += 1;
            }
        }

        self.update_efficiency();
    }

    fn update_efficiency(&mut self) {
        // Estimate efficiency based on achieved vs theoretical bandwidth
        // This is a simplified calculation; real implementation would use hardware specs
        let theoretical_bandwidth = 1_000_000_000.0; // 1 GB/s theoretical
        let actual_bandwidth = self.get_bandwidth_bps();

        if theoretical_bandwidth > 0.0 {
            self.transfer_efficiency = (actual_bandwidth / theoretical_bandwidth).min(1.0);
        }
    }

    pub fn get_bandwidth_gbps(&self) -> f64 {
        self.get_bandwidth_bps() / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn get_bandwidth_bps(&self) -> f64 {
        let total_bytes = self.cpu_to_gpu_bytes + self.gpu_to_cpu_bytes;
        let total_time = self.cpu_to_gpu_time + self.gpu_to_cpu_time;

        if !total_time.is_zero() && total_bytes > 0 {
            total_bytes as f64 / total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_cpu_to_gpu_bandwidth(&self) -> f64 {
        if !self.cpu_to_gpu_time.is_zero() && self.cpu_to_gpu_bytes > 0 {
            self.cpu_to_gpu_bytes as f64 / self.cpu_to_gpu_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_gpu_to_cpu_bandwidth(&self) -> f64 {
        if !self.gpu_to_cpu_time.is_zero() && self.gpu_to_cpu_bytes > 0 {
            self.gpu_to_cpu_bytes as f64 / self.gpu_to_cpu_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_efficiency(&self) -> f64 {
        self.transfer_efficiency
    }

    pub fn merge(&mut self, other: &MemoryTransferMetrics) {
        self.cpu_to_gpu_bytes += other.cpu_to_gpu_bytes;
        self.gpu_to_cpu_bytes += other.gpu_to_cpu_bytes;
        self.cpu_to_gpu_time += other.cpu_to_gpu_time;
        self.gpu_to_cpu_time += other.gpu_to_cpu_time;
        self.cpu_to_gpu_transfers += other.cpu_to_gpu_transfers;
        self.gpu_to_cpu_transfers += other.gpu_to_cpu_transfers;
        self.peak_bandwidth = self.peak_bandwidth.max(other.peak_bandwidth);
        self.update_efficiency();
    }
}

impl Default for MemoryTransferMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Direction of memory transfer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    CpuToGpu,
    GpuToCpu,
}

/// Distributed communication statistics
#[derive(Debug, Clone)]
pub struct CommunicationStats {
    /// Number of all-reduce operations
    pub allreduce_operations: u64,
    /// Total time spent in all-reduce
    pub allreduce_time: Duration,
    /// Total bytes all-reduced
    pub allreduce_bytes: usize,
    /// Number of broadcast operations
    pub broadcast_operations: u64,
    /// Total time spent in broadcast
    pub broadcast_time: Duration,
    /// Total bytes broadcast
    pub broadcast_bytes: usize,
    /// Number of point-to-point communications
    pub p2p_operations: u64,
    /// Total time spent in point-to-point communication
    pub p2p_time: Duration,
    /// Total bytes in point-to-point communication
    pub p2p_bytes: usize,
    /// Communication efficiency (achieved vs theoretical)
    pub communication_efficiency: f64,
}

impl CommunicationStats {
    pub fn new() -> Self {
        Self {
            allreduce_operations: 0,
            allreduce_time: Duration::ZERO,
            allreduce_bytes: 0,
            broadcast_operations: 0,
            broadcast_time: Duration::ZERO,
            broadcast_bytes: 0,
            p2p_operations: 0,
            p2p_time: Duration::ZERO,
            p2p_bytes: 0,
            communication_efficiency: 1.0,
        }
    }

    pub fn record_gradient_sync(
        &mut self,
        duration: Duration,
        num_gradients: usize,
        world_size: usize,
    ) {
        // Gradient sync typically uses all-reduce
        self.allreduce_operations += 1;
        self.allreduce_time += duration;
        // Estimate bytes (simplified calculation)
        let estimated_bytes = num_gradients * 4 * world_size; // Assuming f32 gradients
        self.allreduce_bytes += estimated_bytes;
        self.update_efficiency();
    }

    pub fn record_operation(
        &mut self,
        operation: CommunicationOperation,
        duration: Duration,
        bytes: usize,
    ) {
        match operation {
            CommunicationOperation::AllReduce => {
                self.allreduce_operations += 1;
                self.allreduce_time += duration;
                self.allreduce_bytes += bytes;
            }
            CommunicationOperation::Broadcast => {
                self.broadcast_operations += 1;
                self.broadcast_time += duration;
                self.broadcast_bytes += bytes;
            }
            CommunicationOperation::PointToPoint => {
                self.p2p_operations += 1;
                self.p2p_time += duration;
                self.p2p_bytes += bytes;
            }
        }
        self.update_efficiency();
    }

    fn update_efficiency(&mut self) {
        // Simplified efficiency calculation
        let total_time = self.allreduce_time + self.broadcast_time + self.p2p_time;
        let total_bytes = self.allreduce_bytes + self.broadcast_bytes + self.p2p_bytes;

        if !total_time.is_zero() && total_bytes > 0 {
            let achieved_bandwidth = total_bytes as f64 / total_time.as_secs_f64();
            let theoretical_bandwidth = 10_000_000_000.0; // 10 GB/s theoretical network
            self.communication_efficiency = (achieved_bandwidth / theoretical_bandwidth).min(1.0);
        }
    }

    pub fn get_efficiency(&self) -> f64 {
        self.communication_efficiency
    }

    pub fn get_allreduce_bandwidth(&self) -> f64 {
        if !self.allreduce_time.is_zero() && self.allreduce_bytes > 0 {
            self.allreduce_bytes as f64 / self.allreduce_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_broadcast_bandwidth(&self) -> f64 {
        if !self.broadcast_time.is_zero() && self.broadcast_bytes > 0 {
            self.broadcast_bytes as f64 / self.broadcast_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn merge(&mut self, other: &CommunicationStats) {
        self.allreduce_operations += other.allreduce_operations;
        self.allreduce_time += other.allreduce_time;
        self.allreduce_bytes += other.allreduce_bytes;
        self.broadcast_operations += other.broadcast_operations;
        self.broadcast_time += other.broadcast_time;
        self.broadcast_bytes += other.broadcast_bytes;
        self.p2p_operations += other.p2p_operations;
        self.p2p_time += other.p2p_time;
        self.p2p_bytes += other.p2p_bytes;
        self.update_efficiency();
    }
}

impl Default for CommunicationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of distributed communication operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationOperation {
    AllReduce,
    Broadcast,
    PointToPoint,
}

/// Optimization efficiency metrics
#[derive(Debug, Clone)]
pub struct OptimizationEfficiency {
    /// Time spent in computation vs communication
    pub compute_time: Duration,
    /// Time spent in communication
    pub communication_time: Duration,
    /// Memory utilization efficiency (0.0 to 1.0)
    pub memory_efficiency: f64,
    /// Parameter update efficiency
    pub parameter_update_efficiency: f64,
    /// Overall training efficiency score
    pub overall_efficiency: f64,
    /// Number of efficiency measurements
    pub measurements: u32,
}

impl OptimizationEfficiency {
    pub fn new() -> Self {
        Self {
            compute_time: Duration::ZERO,
            communication_time: Duration::ZERO,
            memory_efficiency: 1.0,
            parameter_update_efficiency: 1.0,
            overall_efficiency: 1.0,
            measurements: 0,
        }
    }

    pub fn record_forward_pass(&mut self, duration: Duration) {
        self.compute_time += duration;
        self.update_efficiency();
    }

    pub fn record_backward_pass(&mut self, duration: Duration) {
        self.compute_time += duration;
        self.update_efficiency();
    }

    pub fn record_optimizer_step(&mut self, duration: Duration, _num_params: usize) {
        self.compute_time += duration;
        self.update_efficiency();
    }

    pub fn record_communication(&mut self, duration: Duration) {
        self.communication_time += duration;
        self.update_efficiency();
    }

    fn update_efficiency(&mut self) {
        self.measurements += 1;

        // Calculate compute vs communication ratio
        let total_time = self.compute_time + self.communication_time;
        let compute_ratio = if !total_time.is_zero() {
            self.compute_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            1.0
        };

        // Overall efficiency is weighted combination of different factors
        self.overall_efficiency = 0.5 * compute_ratio
            + 0.3 * self.memory_efficiency
            + 0.2 * self.parameter_update_efficiency;
        self.overall_efficiency = self.overall_efficiency.clamp(0.0, 1.0);
    }

    pub fn update_memory_efficiency(&mut self, efficiency: f64) {
        self.memory_efficiency = efficiency.clamp(0.0, 1.0);
        self.update_efficiency();
    }

    pub fn update_parameter_efficiency(&mut self, efficiency: f64) {
        self.parameter_update_efficiency = efficiency.clamp(0.0, 1.0);
        self.update_efficiency();
    }

    pub fn get_compute_ratio(&self) -> f64 {
        let total_time = self.compute_time + self.communication_time;
        if !total_time.is_zero() {
            self.compute_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            1.0
        }
    }

    pub fn get_communication_ratio(&self) -> f64 {
        let total_time = self.compute_time + self.communication_time;
        if !total_time.is_zero() {
            self.communication_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn get_overall_efficiency(&self) -> f64 {
        self.overall_efficiency
    }

    pub fn merge(&mut self, other: &OptimizationEfficiency) {
        self.compute_time += other.compute_time;
        self.communication_time += other.communication_time;
        self.measurements += other.measurements;

        // Merge efficiency metrics (weighted average)
        let total_measurements = self.measurements as f64;
        if total_measurements > 0.0 {
            let self_weight = (self.measurements - other.measurements) as f64 / total_measurements;
            let other_weight = other.measurements as f64 / total_measurements;

            self.memory_efficiency =
                self_weight * self.memory_efficiency + other_weight * other.memory_efficiency;
            self.parameter_update_efficiency = self_weight * self.parameter_update_efficiency
                + other_weight * other.parameter_update_efficiency;
        }

        self.update_efficiency();
    }
}

impl Default for OptimizationEfficiency {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Total number of operations (forward + backward + optimizer)
    pub total_operations: u64,
    /// Average forward pass time
    pub average_forward_time: Duration,
    /// Average backward pass time
    pub average_backward_time: Duration,
    /// Average optimizer step time
    pub average_optimizer_time: Duration,
    /// Tokens processed per second
    pub tokens_per_second: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Communication efficiency (0.0 to 1.0)
    pub communication_efficiency: f64,
    /// Overall training efficiency (0.0 to 1.0)
    pub training_efficiency: f64,
    /// Memory transfer efficiency (0.0 to 1.0)
    pub memory_transfer_efficiency: f64,
    /// Per-layer performance summary
    pub layer_performance: HashMap<String, LayerPerformanceSummary>,
}

/// Per-layer performance summary
#[derive(Debug, Clone)]
pub struct LayerPerformanceSummary {
    /// Total number of executions (forward + backward)
    pub total_executions: u64,
    /// Average forward execution time
    pub average_forward_time: Duration,
    /// Average backward execution time
    pub average_backward_time: Duration,
    /// Total time spent in this layer
    pub total_time: Duration,
}

/// Memory statistics for ZeRO-3 (re-exported from memory_management module)
pub use super::memory_management::Zero3MemoryStats;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_stats_creation() {
        let stats = Zero3PerformanceStats::new();
        assert_eq!(stats.forward_passes, 0);
        assert_eq!(stats.backward_passes, 0);
        assert_eq!(stats.optimizer_steps, 0);
        assert_eq!(stats.total_forward_time, Duration::ZERO);
    }

    #[test]
    fn test_record_forward_pass() {
        let mut stats = Zero3PerformanceStats::new();
        stats.record_forward_pass(Duration::from_millis(100), 1000);

        assert_eq!(stats.forward_passes, 1);
        assert_eq!(stats.total_forward_time, Duration::from_millis(100));
        assert_eq!(stats.average_forward_time(), Duration::from_millis(100));
    }

    #[test]
    fn test_layer_timing_stats() {
        let mut layer_stats = LayerTimingStats::new();

        layer_stats.record_forward_execution(Duration::from_millis(50));
        layer_stats.record_backward_execution(Duration::from_millis(75));

        assert_eq!(layer_stats.forward_executions, 1);
        assert_eq!(layer_stats.backward_executions, 1);
        assert_eq!(
            layer_stats.average_forward_time(),
            Duration::from_millis(50)
        );
        assert_eq!(
            layer_stats.average_backward_time(),
            Duration::from_millis(75)
        );
    }

    #[test]
    fn test_throughput_metrics() {
        let mut metrics = ThroughputMetrics::new();

        metrics.record_forward_pass(Duration::from_secs(1), 1000);
        assert_eq!(metrics.get_tokens_per_second(), 1000.0);

        metrics.record_backward_pass(Duration::from_secs(1), 1000);
        assert_eq!(metrics.get_tokens_per_second(), 1000.0); // 2000 tokens in 2 seconds
    }

    #[test]
    fn test_memory_transfer_metrics() {
        let mut metrics = MemoryTransferMetrics::new();

        metrics.record_transfer(Duration::from_secs(1), 1000, TransferDirection::CpuToGpu);
        assert_eq!(metrics.cpu_to_gpu_bytes, 1000);
        assert_eq!(metrics.cpu_to_gpu_transfers, 1);
        assert_eq!(metrics.get_cpu_to_gpu_bandwidth(), 1000.0);
    }

    #[test]
    fn test_communication_stats() {
        let mut stats = CommunicationStats::new();

        stats.record_operation(
            CommunicationOperation::AllReduce,
            Duration::from_millis(100),
            1000,
        );
        assert_eq!(stats.allreduce_operations, 1);
        assert_eq!(stats.allreduce_bytes, 1000);
        assert_eq!(stats.get_allreduce_bandwidth(), 10000.0); // 1000 bytes / 0.1 seconds = 10000 bytes/sec
    }

    #[test]
    fn test_optimization_efficiency() {
        let mut efficiency = OptimizationEfficiency::new();

        efficiency.record_forward_pass(Duration::from_millis(800));
        efficiency.record_communication(Duration::from_millis(200));

        assert_eq!(efficiency.get_compute_ratio(), 0.8);
        assert_eq!(efficiency.get_communication_ratio(), 0.2);
    }

    #[test]
    fn test_stats_merging() {
        let mut stats1 = Zero3PerformanceStats::new();
        stats1.record_forward_pass(Duration::from_millis(100), 1000);

        let mut stats2 = Zero3PerformanceStats::new();
        stats2.record_forward_pass(Duration::from_millis(200), 2000);

        stats1.merge(&stats2);
        assert_eq!(stats1.forward_passes, 2);
        assert_eq!(stats1.total_forward_time, Duration::from_millis(300));
    }

    #[test]
    fn test_performance_summary() {
        let mut stats = Zero3PerformanceStats::new();
        stats.record_forward_pass(Duration::from_millis(100), 1000);
        stats.record_backward_pass(Duration::from_millis(150), 1000);
        stats.record_optimizer_step(Duration::from_millis(50), 100);

        let summary = stats.get_performance_summary();
        assert_eq!(summary.total_operations, 3);
        assert!(summary.tokens_per_second > 0.0);
    }

    #[test]
    fn test_transfer_direction() {
        assert_eq!(TransferDirection::CpuToGpu, TransferDirection::CpuToGpu);
        assert_ne!(TransferDirection::CpuToGpu, TransferDirection::GpuToCpu);
    }

    #[test]
    fn test_communication_operation() {
        assert_eq!(
            CommunicationOperation::AllReduce,
            CommunicationOperation::AllReduce
        );
        assert_ne!(
            CommunicationOperation::AllReduce,
            CommunicationOperation::Broadcast
        );
    }
}
