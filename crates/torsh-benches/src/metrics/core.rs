//! Core system metrics collection
//!
//! This module provides the fundamental metrics collection infrastructure including
//! system-level metrics, timing, and basic performance measurement capabilities.

use std::time::{Duration, Instant};

/// System metrics collector
pub struct MetricsCollector {
    start_time: Option<Instant>,
    memory_tracker: super::tracking::MemoryTracker,
    cpu_tracker: super::tracking::CpuTracker,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: None,
            memory_tracker: super::tracking::MemoryTracker::new(),
            cpu_tracker: super::tracking::CpuTracker::new(),
        }
    }

    /// Start collecting metrics
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.memory_tracker.start();
        self.cpu_tracker.start();
    }

    /// Stop collecting metrics and return results
    pub fn stop(&mut self) -> SystemMetrics {
        let elapsed = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        let memory_stats = self.memory_tracker.stop();
        let cpu_stats = self.cpu_tracker.stop();

        SystemMetrics {
            elapsed_time: elapsed,
            memory_stats,
            cpu_stats,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete system metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemMetrics {
    pub elapsed_time: Duration,
    pub memory_stats: MemoryStats,
    pub cpu_stats: CpuStats,
}

impl SystemMetrics {
    /// Get memory efficiency (operations per MB)
    pub fn memory_efficiency(&self, operations: usize) -> f64 {
        if self.memory_stats.peak_usage_mb > 0.0 {
            operations as f64 / self.memory_stats.peak_usage_mb
        } else {
            0.0
        }
    }

    /// Get CPU utilization percentage
    pub fn cpu_utilization(&self) -> f64 {
        self.cpu_stats.average_usage_percent
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStats {
    /// Initial memory usage in MB
    pub initial_usage_mb: f64,

    /// Peak memory usage in MB
    pub peak_usage_mb: f64,

    /// Final memory usage in MB
    pub final_usage_mb: f64,

    /// Memory allocated during benchmark in MB
    pub allocated_mb: f64,

    /// Memory deallocated during benchmark in MB
    pub deallocated_mb: f64,

    /// Number of allocations
    pub allocation_count: usize,

    /// Number of deallocations
    pub deallocation_count: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            initial_usage_mb: 0.0,
            peak_usage_mb: 0.0,
            final_usage_mb: 0.0,
            allocated_mb: 0.0,
            deallocated_mb: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }
}

/// CPU usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuStats {
    /// Average CPU usage percentage
    pub average_usage_percent: f64,

    /// Peak CPU usage percentage
    pub peak_usage_percent: f64,

    /// Number of CPU cores used
    pub cores_used: usize,

    /// Context switches during benchmark
    pub context_switches: usize,
}

impl Default for CpuStats {
    fn default() -> Self {
        Self {
            average_usage_percent: 0.0,
            peak_usage_percent: 0.0,
            cores_used: 1,
            context_switches: 0,
        }
    }
}
