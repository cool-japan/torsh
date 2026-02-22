//! Metrics collection functionality for dashboard
//!
//! This module provides comprehensive metrics collection capabilities for the
//! ToRSh performance dashboard, including performance, memory, and system metrics.

use crate::{MemoryProfiler, ProfileEvent, Profiler, TorshResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use torsh_core::TorshError;

use super::types::{
    DashboardAlert, DashboardConfig, DashboardData, MemoryMetrics, OperationSummary,
    PerformanceMetrics, SystemMetrics,
};

// =============================================================================
// Main Data Collection Functions
// =============================================================================

/// Continuous data collection loop for dashboard metrics
pub fn data_collection_loop(
    data_history: Arc<Mutex<Vec<DashboardData>>>,
    _alerts: Arc<Mutex<Vec<DashboardAlert>>>,
    config: DashboardConfig,
    running: Arc<Mutex<bool>>,
    profiler: Arc<Profiler>,
    memory_profiler: Arc<MemoryProfiler>,
) {
    while {
        let is_running = running.lock().map(|r| *r).unwrap_or(false);
        is_running
    } {
        // Collect current metrics
        if let Ok(data) = collect_dashboard_data(&profiler, &memory_profiler) {
            if let Ok(mut history) = data_history.lock() {
                history.push(data);

                // Keep only recent data points
                if history.len() > config.max_data_points {
                    history.remove(0);
                }
            }
        }

        // Sleep for refresh interval
        thread::sleep(Duration::from_secs(config.refresh_interval));
    }
}

/// Collect current dashboard data snapshot
pub fn collect_dashboard_data(
    profiler: &Profiler,
    memory_profiler: &MemoryProfiler,
) -> TorshResult<DashboardData> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TorshError::RuntimeError("Failed to get timestamp".to_string()))?
        .as_secs();

    // Collect performance metrics
    let performance_metrics = collect_performance_metrics(profiler)?;

    // Collect memory metrics
    let memory_metrics = collect_memory_metrics(memory_profiler)?;

    // Collect system metrics
    let system_metrics = collect_system_metrics()?;

    // Get top operations
    let top_operations = get_top_operations(profiler)?;

    Ok(DashboardData {
        timestamp,
        performance_metrics,
        memory_metrics,
        system_metrics,
        alerts: Vec::new(), // Alerts are managed separately
        top_operations,
    })
}

// =============================================================================
// Performance Metrics Collection
// =============================================================================

/// Collect comprehensive performance metrics from profiler
pub fn collect_performance_metrics(profiler: &Profiler) -> TorshResult<PerformanceMetrics> {
    let events = profiler.events();

    if events.is_empty() {
        return Ok(PerformanceMetrics {
            total_operations: 0,
            average_duration_ms: 0.0,
            operations_per_second: 0.0,
            total_flops: 0,
            gflops_per_second: 0.0,
            cpu_utilization: 0.0,
            thread_count: 0,
        });
    }

    let total_operations = events.len() as u64;
    let total_duration_us: u64 = events.iter().map(|e| e.duration_us).sum();
    let average_duration_ms = (total_duration_us as f64) / (events.len() as f64) / 1000.0;

    let total_flops: u64 = events.iter().map(|e| e.flops.unwrap_or(0)).sum();

    let total_time_seconds = total_duration_us as f64 / 1_000_000.0;
    let operations_per_second = if total_time_seconds > 0.0 {
        total_operations as f64 / total_time_seconds
    } else {
        0.0
    };

    let gflops_per_second = if total_time_seconds > 0.0 {
        (total_flops as f64) / total_time_seconds / 1_000_000_000.0
    } else {
        0.0
    };

    let unique_threads: std::collections::HashSet<_> = events.iter().map(|e| e.thread_id).collect();
    let thread_count = unique_threads.len();

    let cpu_utilization = (thread_count as f64) / (num_cpus::get() as f64) * 100.0;

    Ok(PerformanceMetrics {
        total_operations,
        average_duration_ms,
        operations_per_second,
        total_flops,
        gflops_per_second,
        cpu_utilization,
        thread_count,
    })
}

/// Calculate advanced performance statistics
pub fn collect_advanced_performance_metrics(
    profiler: &Profiler,
) -> TorshResult<AdvancedPerformanceMetrics> {
    let events = profiler.events();

    if events.is_empty() {
        return Ok(AdvancedPerformanceMetrics::default());
    }

    // Calculate percentiles
    let mut durations: Vec<u64> = events.iter().map(|e| e.duration_us).collect();
    durations.sort_unstable();

    let p50 = percentile(&durations, 50.0);
    let p95 = percentile(&durations, 95.0);
    let p99 = percentile(&durations, 99.0);

    // Calculate efficiency metrics
    let total_cpu_time_us: u64 = events.iter().map(|e| e.duration_us).sum();
    let wall_clock_time_us = events
        .iter()
        .map(|e| e.start_us + e.duration_us)
        .max()
        .unwrap_or(0)
        - events.iter().map(|e| e.start_us).min().unwrap_or(0);

    let parallel_efficiency = if wall_clock_time_us > 0 {
        (total_cpu_time_us as f64) / (wall_clock_time_us as f64)
    } else {
        0.0
    };

    // Cache efficiency metrics - would need additional ProfileEvent fields
    let cache_hit_rate = 0.0; // Placeholder - would calculate from cache performance data

    Ok(AdvancedPerformanceMetrics {
        percentile_50_ms: p50 as f64 / 1000.0,
        percentile_95_ms: p95 as f64 / 1000.0,
        percentile_99_ms: p99 as f64 / 1000.0,
        parallel_efficiency,
        cache_hit_rate,
        total_cpu_time_seconds: total_cpu_time_us as f64 / 1_000_000.0,
        wall_clock_time_seconds: wall_clock_time_us as f64 / 1_000_000.0,
    })
}

// =============================================================================
// Memory Metrics Collection
// =============================================================================

/// Collect memory metrics from memory profiler
pub fn collect_memory_metrics(memory_profiler: &MemoryProfiler) -> TorshResult<MemoryMetrics> {
    let stats = memory_profiler.get_stats()?;

    Ok(MemoryMetrics {
        current_usage_mb: stats.allocated as f64 / (1024.0 * 1024.0),
        peak_usage_mb: stats.peak as f64 / (1024.0 * 1024.0),
        total_allocations: stats.allocations as u64,
        total_deallocations: stats.deallocations as u64,
        active_allocations: stats.allocations.saturating_sub(stats.deallocations) as u64,
        fragmentation_ratio: calculate_fragmentation_ratio(&stats),
        allocation_rate: calculate_allocation_rate(&stats),
    })
}

/// Calculate memory fragmentation ratio
fn calculate_fragmentation_ratio(stats: &crate::MemoryStats) -> f64 {
    // Simplified fragmentation calculation
    // In a real implementation, this would analyze memory layout
    if stats.allocated > 0 {
        1.0 - (stats.allocated as f64 / stats.peak as f64)
    } else {
        0.0
    }
}

/// Calculate current allocation rate (allocations per second)
fn calculate_allocation_rate(stats: &crate::MemoryStats) -> f64 {
    // This would need time-based tracking in a real implementation
    // For now, return a placeholder
    stats.allocations as f64 / 60.0 // Rough estimate: allocations per minute / 60
}

/// Collect detailed memory analysis
pub fn collect_detailed_memory_metrics(
    memory_profiler: &MemoryProfiler,
) -> TorshResult<DetailedMemoryMetrics> {
    let stats = memory_profiler.get_stats()?;

    // Calculate memory trends
    let allocation_trend = calculate_allocation_trend(memory_profiler)?;
    let deallocation_trend = calculate_deallocation_trend(memory_profiler)?;

    // Calculate memory efficiency
    let efficiency = if stats.allocations > 0 {
        stats.deallocations as f64 / stats.allocations as f64
    } else {
        0.0
    };

    Ok(DetailedMemoryMetrics {
        heap_usage_mb: stats.allocated as f64 / (1024.0 * 1024.0),
        stack_usage_mb: 0.0, // Would need stack tracking
        allocation_trend,
        deallocation_trend,
        memory_efficiency: efficiency,
        largest_allocation_mb: 0.0, // Would need tracking
        memory_pressure: calculate_memory_pressure(&stats),
    })
}

// =============================================================================
// System Metrics Collection
// =============================================================================

/// Collect system-level metrics
pub fn collect_system_metrics() -> TorshResult<SystemMetrics> {
    // This is a simplified implementation
    // In a real system, you would read from /proc, use system APIs, or system crates

    #[cfg(target_os = "linux")]
    {
        return collect_linux_system_metrics();
    }

    #[cfg(target_os = "macos")]
    {
        return collect_macos_system_metrics();
    }

    #[cfg(target_os = "windows")]
    {
        return collect_windows_system_metrics();
    }

    // Default fallback for other platforms
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Ok(SystemMetrics {
            uptime_seconds: 0,
            load_average: 0.0,
            available_memory_mb: get_available_memory_mb(),
            disk_usage_percent: get_disk_usage_percent(),
            network_io_mbps: 0.0,
        })
    }
}

#[cfg(target_os = "linux")]
fn collect_linux_system_metrics() -> TorshResult<SystemMetrics> {
    use std::fs;

    // Read /proc/uptime
    let uptime_seconds = fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|content| {
            content
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0) as u64;

    // Read /proc/loadavg
    let load_average = fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|content| {
            content
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0);

    Ok(SystemMetrics {
        uptime_seconds,
        load_average,
        available_memory_mb: get_available_memory_mb(),
        disk_usage_percent: get_disk_usage_percent(),
        network_io_mbps: get_network_io_mbps(),
    })
}

#[cfg(target_os = "macos")]
fn collect_macos_system_metrics() -> TorshResult<SystemMetrics> {
    // Use system calls or external tools for macOS
    Ok(SystemMetrics {
        uptime_seconds: get_macos_uptime(),
        load_average: get_macos_load_average(),
        available_memory_mb: get_available_memory_mb(),
        disk_usage_percent: get_disk_usage_percent(),
        network_io_mbps: get_network_io_mbps(),
    })
}

#[cfg(target_os = "windows")]
fn collect_windows_system_metrics() -> TorshResult<SystemMetrics> {
    // Use Windows APIs for system metrics
    Ok(SystemMetrics {
        uptime_seconds: get_windows_uptime(),
        load_average: get_windows_cpu_usage(),
        available_memory_mb: get_available_memory_mb(),
        disk_usage_percent: get_disk_usage_percent(),
        network_io_mbps: get_network_io_mbps(),
    })
}

// =============================================================================
// Operation Analysis
// =============================================================================

/// Get top operations by duration and other metrics
pub fn get_top_operations(profiler: &Profiler) -> TorshResult<Vec<OperationSummary>> {
    let events = profiler.events();

    if events.is_empty() {
        return Ok(Vec::new());
    }

    let mut operation_stats: HashMap<String, (u64, u64)> = HashMap::new();
    let total_duration: u64 = events.iter().map(|e| e.duration_us).sum();

    // Aggregate by operation name
    for event in events {
        let entry = operation_stats.entry(event.name.clone()).or_insert((0, 0));
        entry.0 += 1; // count
        entry.1 += event.duration_us; // total duration
    }

    // Convert to OperationSummary and sort by total duration
    let mut summaries: Vec<OperationSummary> = operation_stats
        .into_iter()
        .map(|(name, (count, total_duration_us))| {
            let total_duration_ms = total_duration_us as f64 / 1000.0;
            let average_duration_ms = total_duration_ms / count as f64;
            let percentage_of_total = if total_duration > 0 {
                (total_duration_us as f64 / total_duration as f64) * 100.0
            } else {
                0.0
            };

            OperationSummary {
                name: name.clone(),
                category: categorize_operation(&name),
                count,
                total_duration_ms,
                average_duration_ms,
                percentage_of_total,
            }
        })
        .collect();

    summaries.sort_by(|a, b| {
        b.total_duration_ms
            .partial_cmp(&a.total_duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top 10
    summaries.truncate(10);

    Ok(summaries)
}

/// Analyze operation patterns and trends
pub fn analyze_operation_patterns(profiler: &Profiler) -> TorshResult<OperationPatternAnalysis> {
    let events = profiler.events();

    if events.is_empty() {
        return Ok(OperationPatternAnalysis::default());
    }

    // Group by operation type
    let mut patterns = HashMap::new();
    for event in events {
        let category = categorize_operation(&event.name);
        let entry = patterns.entry(category).or_insert(Vec::new());
        entry.push(event);
    }

    // Analyze each category
    let mut category_analysis = HashMap::new();
    for (category, events) in patterns {
        let total_duration: u64 = events.iter().map(|e| e.duration_us).sum();
        let avg_duration = total_duration as f64 / events.len() as f64;
        let total_flops: u64 = events.iter().map(|e| e.flops.unwrap_or(0)).sum();

        category_analysis.insert(
            category.clone(),
            CategoryMetrics {
                operation_count: events.len() as u64,
                total_duration_ms: total_duration as f64 / 1000.0,
                average_duration_ms: avg_duration / 1000.0,
                total_flops,
            },
        );
    }

    Ok(OperationPatternAnalysis {
        categories: category_analysis.clone(),
        total_unique_operations: events.len() as u64,
        dominant_category: find_dominant_category(&category_analysis),
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[u64], percentile: f64) -> u64 {
    if sorted_data.is_empty() {
        return 0;
    }

    let index = (percentile / 100.0) * (sorted_data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        let lower_val = sorted_data[lower] as f64;
        let upper_val = sorted_data[upper] as f64;
        (lower_val + weight * (upper_val - lower_val)) as u64
    }
}

/// Categorize operations by name patterns
fn categorize_operation(name: &str) -> String {
    if name.contains("conv") || name.contains("convolution") {
        "Convolution".to_string()
    } else if name.contains("matmul") || name.contains("gemm") {
        "Matrix Operations".to_string()
    } else if name.contains("add") || name.contains("sub") || name.contains("mul") {
        "Arithmetic".to_string()
    } else if name.contains("relu") || name.contains("sigmoid") || name.contains("tanh") {
        "Activation".to_string()
    } else if name.contains("batch_norm") || name.contains("layer_norm") {
        "Normalization".to_string()
    } else if name.contains("dropout") {
        "Regularization".to_string()
    } else {
        "Other".to_string()
    }
}

/// Calculate memory allocation trend
fn calculate_allocation_trend(memory_profiler: &MemoryProfiler) -> TorshResult<f64> {
    // This would analyze historical allocation data
    // For now, return a placeholder
    Ok(0.0)
}

/// Calculate memory deallocation trend
fn calculate_deallocation_trend(memory_profiler: &MemoryProfiler) -> TorshResult<f64> {
    // This would analyze historical deallocation data
    // For now, return a placeholder
    Ok(0.0)
}

/// Calculate memory pressure indicator
fn calculate_memory_pressure(stats: &crate::MemoryStats) -> f64 {
    // Simple memory pressure calculation
    if stats.peak > 0 {
        stats.allocated as f64 / stats.peak as f64
    } else {
        0.0
    }
}

/// Find the dominant operation category
fn find_dominant_category(analysis: &HashMap<String, CategoryMetrics>) -> String {
    analysis
        .iter()
        .max_by(|a, b| {
            a.1.total_duration_ms
                .partial_cmp(&b.1.total_duration_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(category, _)| category.clone())
        .unwrap_or_else(|| "Unknown".to_string())
}

// Platform-specific system metric helpers
#[cfg(target_os = "macos")]
fn get_macos_uptime() -> u64 {
    // Implementation would use sysctl or similar
    0
}

#[cfg(target_os = "macos")]
fn get_macos_load_average() -> f64 {
    // Implementation would use sysctl or similar
    0.0
}

#[cfg(target_os = "windows")]
fn get_windows_uptime() -> u64 {
    // Implementation would use Windows APIs
    0
}

#[cfg(target_os = "windows")]
fn get_windows_cpu_usage() -> f64 {
    // Implementation would use Windows APIs
    0.0
}

fn get_available_memory_mb() -> f64 {
    // Platform-agnostic memory detection
    // This would use system APIs or /proc/meminfo
    8192.0 // Placeholder: 8GB
}

fn get_disk_usage_percent() -> f64 {
    // Platform-agnostic disk usage
    50.0 // Placeholder: 50%
}

fn get_network_io_mbps() -> f64 {
    // Platform-agnostic network I/O
    0.0 // Placeholder
}

// =============================================================================
// Extended Metrics Types
// =============================================================================

/// Advanced performance metrics
#[derive(Debug, Clone)]
pub struct AdvancedPerformanceMetrics {
    pub percentile_50_ms: f64,
    pub percentile_95_ms: f64,
    pub percentile_99_ms: f64,
    pub parallel_efficiency: f64,
    pub cache_hit_rate: f64,
    pub total_cpu_time_seconds: f64,
    pub wall_clock_time_seconds: f64,
}

impl Default for AdvancedPerformanceMetrics {
    fn default() -> Self {
        Self {
            percentile_50_ms: 0.0,
            percentile_95_ms: 0.0,
            percentile_99_ms: 0.0,
            parallel_efficiency: 0.0,
            cache_hit_rate: 0.0,
            total_cpu_time_seconds: 0.0,
            wall_clock_time_seconds: 0.0,
        }
    }
}

/// Detailed memory metrics
#[derive(Debug, Clone)]
pub struct DetailedMemoryMetrics {
    pub heap_usage_mb: f64,
    pub stack_usage_mb: f64,
    pub allocation_trend: f64,
    pub deallocation_trend: f64,
    pub memory_efficiency: f64,
    pub largest_allocation_mb: f64,
    pub memory_pressure: f64,
}

/// Operation pattern analysis results
#[derive(Debug, Clone)]
pub struct OperationPatternAnalysis {
    pub categories: HashMap<String, CategoryMetrics>,
    pub total_unique_operations: u64,
    pub dominant_category: String,
}

impl Default for OperationPatternAnalysis {
    fn default() -> Self {
        Self {
            categories: HashMap::new(),
            total_unique_operations: 0,
            dominant_category: "Unknown".to_string(),
        }
    }
}

/// Metrics for a category of operations
#[derive(Debug, Clone)]
pub struct CategoryMetrics {
    pub operation_count: u64,
    pub total_duration_ms: f64,
    pub average_duration_ms: f64,
    pub total_flops: u64,
}

// =============================================================================
// MetricsCollector - Missing Implementation
// =============================================================================

/// Centralized metrics collector for dashboard data
pub struct MetricsCollector {
    /// Configuration for metrics collection
    pub config: MetricsCollectorConfig,
}

/// Configuration for the metrics collector
#[derive(Debug, Clone)]
pub struct MetricsCollectorConfig {
    /// Enable detailed memory tracking
    pub enable_memory_tracking: bool,
    /// Enable system metrics collection
    pub enable_system_metrics: bool,
    /// Sample rate for performance metrics
    pub sample_rate: f64,
}

impl Default for MetricsCollectorConfig {
    fn default() -> Self {
        Self {
            enable_memory_tracking: true,
            enable_system_metrics: true,
            sample_rate: 1.0,
        }
    }
}

impl MetricsCollector {
    /// Create a new metrics collector with default configuration
    pub fn new() -> Self {
        Self {
            config: MetricsCollectorConfig::default(),
        }
    }

    /// Create a new metrics collector with custom configuration
    pub fn new_with_config(config: MetricsCollectorConfig) -> Self {
        Self { config }
    }

    /// Collect comprehensive dashboard data
    pub fn collect_dashboard_data(
        &self,
        profiler: &Profiler,
        memory_profiler: &MemoryProfiler,
    ) -> TorshResult<DashboardData> {
        collect_dashboard_data(profiler, memory_profiler)
    }

    /// Collect performance metrics
    pub fn collect_performance_metrics(
        &self,
        profiler: &Profiler,
    ) -> TorshResult<PerformanceMetrics> {
        collect_performance_metrics(profiler)
    }

    /// Collect memory metrics
    pub fn collect_memory_metrics(
        &self,
        memory_profiler: &MemoryProfiler,
    ) -> TorshResult<MemoryMetrics> {
        collect_memory_metrics(memory_profiler)
    }

    /// Collect system metrics
    pub fn collect_system_metrics(&self) -> TorshResult<SystemMetrics> {
        collect_system_metrics()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
