//! Memory and performance profiling utilities for clustering algorithms
//!
//! This module provides utilities for monitoring memory usage, performance metrics,
//! and resource consumption during clustering operations on large datasets.

use crate::error::{ClusterError, ClusterResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Number of allocations
    pub allocations: usize,
}

impl MemoryStats {
    /// Create new memory stats
    pub fn new() -> Self {
        Self {
            memory_bytes: 0,
            peak_memory_bytes: 0,
            allocations: 0,
        }
    }

    /// Update memory usage
    pub fn update(&mut self, bytes: usize) {
        self.memory_bytes = bytes;
        self.peak_memory_bytes = self.peak_memory_bytes.max(bytes);
        self.allocations += 1;
    }

    /// Get memory usage in MB
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn peak_memory_mb(&self) -> f64 {
        self.peak_memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiling context for clustering operations
#[derive(Debug)]
pub struct ProfilingContext {
    /// Start time
    start_time: Instant,
    /// Memory statistics
    memory_stats: MemoryStats,
    /// Operation timings
    timings: HashMap<String, Duration>,
    /// Operation counts
    counts: HashMap<String, usize>,
}

impl ProfilingContext {
    /// Create a new profiling context
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            memory_stats: MemoryStats::new(),
            timings: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_operation(&mut self, name: &str) -> OperationTimer {
        OperationTimer::new(name.to_string())
    }

    /// Record operation timing
    pub fn record_timing(&mut self, name: String, duration: Duration) {
        *self.timings.entry(name).or_insert(Duration::ZERO) += duration;
    }

    /// Increment operation count
    pub fn increment_count(&mut self, name: &str) {
        *self.counts.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Record memory usage
    pub fn record_memory(&mut self, bytes: usize) {
        self.memory_stats.update(bytes);
    }

    /// Get elapsed time since context creation
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> &MemoryStats {
        &self.memory_stats
    }

    /// Get timing for a specific operation
    pub fn get_timing(&self, name: &str) -> Option<Duration> {
        self.timings.get(name).copied()
    }

    /// Get count for a specific operation
    pub fn get_count(&self, name: &str) -> Option<usize> {
        self.counts.get(name).copied()
    }

    /// Get all timings
    pub fn all_timings(&self) -> &HashMap<String, Duration> {
        &self.timings
    }

    /// Get all counts
    pub fn all_counts(&self) -> &HashMap<String, usize> {
        &self.counts
    }

    /// Generate profiling report
    pub fn report(&self) -> ProfilingReport {
        ProfilingReport {
            total_time: self.elapsed(),
            memory_stats: self.memory_stats.clone(),
            timings: self.timings.clone(),
            counts: self.counts.clone(),
        }
    }
}

impl Default for ProfilingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation timer for profiling individual operations
#[derive(Debug)]
pub struct OperationTimer {
    name: String,
    start: Instant,
}

impl OperationTimer {
    /// Create a new operation timer
    pub fn new(name: String) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }

    /// Stop the timer and get the elapsed duration
    pub fn stop(self) -> (String, Duration) {
        (self.name, self.start.elapsed())
    }
}

/// Profiling report with comprehensive statistics
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Total execution time
    pub total_time: Duration,
    /// Memory statistics
    pub memory_stats: MemoryStats,
    /// Operation timings
    pub timings: HashMap<String, Duration>,
    /// Operation counts
    pub counts: HashMap<String, usize>,
}

impl ProfilingReport {
    /// Print formatted report to stdout
    pub fn print(&self) {
        println!("\n=== Clustering Profiling Report ===");
        println!("Total Time: {:.3}s", self.total_time.as_secs_f64());
        println!("Peak Memory: {:.2} MB", self.memory_stats.peak_memory_mb());
        println!("Current Memory: {:.2} MB", self.memory_stats.memory_mb());
        println!("\nOperation Timings:");
        for (name, duration) in &self.timings {
            println!("  {}: {:.3}s", name, duration.as_secs_f64());
        }
        println!("\nOperation Counts:");
        for (name, count) in &self.counts {
            println!("  {}: {}", name, count);
        }
        println!("===================================\n");
    }

    /// Export as JSON string
    pub fn to_json(&self) -> String {
        // Simple JSON formatting (no dependencies)
        let mut json = String::from("{\n");
        json.push_str(&format!(
            "  \"total_time_s\": {},\n",
            self.total_time.as_secs_f64()
        ));
        json.push_str(&format!(
            "  \"peak_memory_mb\": {},\n",
            self.memory_stats.peak_memory_mb()
        ));
        json.push_str(&format!(
            "  \"current_memory_mb\": {},\n",
            self.memory_stats.memory_mb()
        ));
        json.push_str("  \"timings\": {\n");
        for (i, (name, duration)) in self.timings.iter().enumerate() {
            json.push_str(&format!(
                "    \"{}\": {}{}",
                name,
                duration.as_secs_f64(),
                if i < self.timings.len() - 1 {
                    ",\n"
                } else {
                    "\n"
                }
            ));
        }
        json.push_str("  },\n");
        json.push_str("  \"counts\": {\n");
        for (i, (name, count)) in self.counts.iter().enumerate() {
            json.push_str(&format!(
                "    \"{}\": {}{}",
                name,
                count,
                if i < self.counts.len() - 1 {
                    ",\n"
                } else {
                    "\n"
                }
            ));
        }
        json.push_str("  }\n");
        json.push_str("}\n");
        json
    }
}

/// Estimate memory usage for clustering operations
pub fn estimate_memory_usage(n_samples: usize, n_features: usize, n_clusters: usize) -> usize {
    // Data storage: n_samples * n_features * size_of(f32)
    let data_size = n_samples * n_features * std::mem::size_of::<f32>();

    // Centroids/means: n_clusters * n_features * size_of(f32)
    let centroids_size = n_clusters * n_features * std::mem::size_of::<f32>();

    // Labels: n_samples * size_of(i32)
    let labels_size = n_samples * std::mem::size_of::<i32>();

    // Distance matrix (for some algorithms): n_samples * n_clusters * size_of(f32)
    let distance_matrix_size = n_samples * n_clusters * std::mem::size_of::<f32>();

    // Total estimate with 20% overhead for temporary allocations
    let total = data_size + centroids_size + labels_size + distance_matrix_size;
    (total as f64 * 1.2) as usize
}

/// Check if dataset fits in available memory
pub fn check_memory_feasibility(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
) -> ClusterResult<()> {
    let estimated = estimate_memory_usage(n_samples, n_features, n_clusters);

    // Get system memory (simplified - actual implementation would query OS)
    // For now, warn if estimated usage is very large (> 8GB)
    const WARNING_THRESHOLD: usize = 8 * 1024 * 1024 * 1024; // 8 GB

    if estimated > WARNING_THRESHOLD {
        return Err(ClusterError::ConfigError(format!(
            "Estimated memory usage ({:.2} GB) exceeds recommended threshold. \
             Consider using mini-batch or incremental algorithms.",
            estimated as f64 / (1024.0 * 1024.0 * 1024.0)
        )));
    }

    Ok(())
}

/// Suggest optimal algorithm based on dataset characteristics
pub fn suggest_algorithm(n_samples: usize, n_features: usize, n_clusters: usize) -> &'static str {
    // Memory-efficient threshold
    if n_samples > 100_000 {
        return "MiniBatch K-Means (for memory efficiency)";
    }

    // High-dimensional data
    if n_features > 100 {
        return "K-Means or Mini-Batch K-Means (efficient for high dimensions)";
    }

    // Large number of clusters
    if n_clusters > 50 {
        return "Elkan's K-Means (optimized for large k)";
    }

    // Small dataset
    if n_samples < 1000 {
        return "Any algorithm (dataset is small)";
    }

    // Default recommendation
    "K-Means Lloyd (standard, good all-around performance)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();
        assert_eq!(stats.memory_bytes, 0);
        assert_eq!(stats.peak_memory_bytes, 0);

        stats.update(1024 * 1024); // 1 MB
        assert_eq!(stats.memory_mb(), 1.0);

        stats.update(2 * 1024 * 1024); // 2 MB
        assert_eq!(stats.peak_memory_mb(), 2.0);

        stats.update(512 * 1024); // 0.5 MB
        assert_eq!(stats.peak_memory_mb(), 2.0); // Peak remains
    }

    #[test]
    fn test_profiling_context() {
        let mut ctx = ProfilingContext::new();

        // Record some operations
        ctx.increment_count("e_step");
        ctx.increment_count("e_step");
        ctx.increment_count("m_step");

        assert_eq!(ctx.get_count("e_step"), Some(2));
        assert_eq!(ctx.get_count("m_step"), Some(1));

        // Record memory
        ctx.record_memory(1024 * 1024);
        assert_eq!(ctx.memory_stats().memory_mb(), 1.0);
    }

    #[test]
    fn test_operation_timer() {
        let timer = OperationTimer::new("test_op".to_string());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let (name, duration) = timer.stop();

        assert_eq!(name, "test_op");
        assert!(duration.as_millis() >= 10);
    }

    #[test]
    fn test_memory_estimation() {
        let memory = estimate_memory_usage(1000, 10, 5);
        // Should be reasonable: ~1000*10*4 + 5*10*4 + 1000*4 + 1000*5*4 bytes
        // = 40000 + 200 + 4000 + 20000 = 64200 bytes * 1.2 overhead = ~77k bytes
        assert!(memory > 70_000 && memory < 100_000);
    }

    #[test]
    fn test_algorithm_suggestion() {
        assert_eq!(
            suggest_algorithm(100, 10, 5),
            "Any algorithm (dataset is small)"
        );
        assert_eq!(
            suggest_algorithm(200_000, 10, 5),
            "MiniBatch K-Means (for memory efficiency)"
        );
        assert_eq!(
            suggest_algorithm(5_000, 200, 5),
            "K-Means or Mini-Batch K-Means (efficient for high dimensions)"
        );
        assert_eq!(
            suggest_algorithm(5_000, 10, 100),
            "Elkan's K-Means (optimized for large k)"
        );
    }

    #[test]
    fn test_profiling_report() {
        let mut ctx = ProfilingContext::new();
        ctx.increment_count("test");
        ctx.record_memory(1024 * 1024);

        let report = ctx.report();
        assert!(report.total_time.as_secs_f64() >= 0.0);
        assert_eq!(report.counts.get("test"), Some(&1));

        // Test JSON export
        let json = report.to_json();
        assert!(json.contains("total_time_s"));
        assert!(json.contains("peak_memory_mb"));
    }
}
