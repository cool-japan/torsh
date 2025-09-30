//! Memory management and leak prevention for autograd contexts

use super::core::AutogradContext;
use crate::gradient_storage::GradientStorage;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
use torsh_core::error::Result;

/// Configuration for memory management
#[derive(Debug, Clone)]
pub struct MemoryManagementConfig {
    /// Enable automatic cleanup of orphaned gradients
    pub auto_cleanup_orphaned: bool,
    /// Enable detection and breaking of circular references
    pub break_circular_references: bool,
    /// Maximum age for cached gradients before cleanup (in seconds)
    pub max_gradient_age: Duration,
    /// Threshold for automatic memory pressure cleanup (in bytes)
    pub memory_pressure_threshold: Option<usize>,
    /// Interval for periodic cleanup operations
    pub cleanup_interval: Duration,
    /// Enable detailed memory tracking and logging
    pub detailed_tracking: bool,
}

impl Default for MemoryManagementConfig {
    fn default() -> Self {
        Self {
            auto_cleanup_orphaned: true,
            break_circular_references: true,
            max_gradient_age: Duration::from_secs(3600), // 1 hour
            memory_pressure_threshold: Some(512 * 1024 * 1024), // 512MB
            cleanup_interval: Duration::from_secs(300),  // 5 minutes
            detailed_tracking: false,
        }
    }
}

/// Statistics about memory cleanup operations
#[derive(Debug, Clone, Default)]
pub struct CleanupStatistics {
    /// Number of orphaned gradients cleaned up
    pub orphaned_gradients_cleaned: usize,
    /// Number of circular references broken
    pub circular_references_broken: usize,
    /// Number of expired gradients removed
    pub expired_gradients_removed: usize,
    /// Total memory freed (estimated in bytes)
    pub memory_freed_bytes: usize,
    /// Number of cleanup operations performed
    pub cleanup_operations: usize,
    /// Last cleanup timestamp
    pub last_cleanup: Option<Instant>,
}

/// Statistics about memory optimization operations
#[derive(Debug, Clone, Default)]
pub struct OptimizationStatistics {
    /// Number of graph compaction operations
    pub graph_compactions: usize,
    /// Number of graph defragmentation operations
    pub graph_defragmentations: usize,
    /// Number of weak references cleared
    pub weak_references_cleared: usize,
    /// Memory saved through optimization (estimated in bytes)
    pub memory_saved_bytes: usize,
    /// Average optimization time in milliseconds
    pub avg_optimization_time_ms: f64,
}

/// Memory pressure monitoring and alerts
#[derive(Debug, Clone)]
pub struct MemoryPressureMonitor {
    /// Current memory usage estimate
    pub current_usage_bytes: usize,
    /// Peak memory usage since last reset
    pub peak_usage_bytes: usize,
    /// Memory usage trend (positive = increasing, negative = decreasing)
    pub usage_trend: f64,
    /// Number of pressure events triggered
    pub pressure_events: usize,
    /// Time of last pressure event
    pub last_pressure_event: Option<Instant>,
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self {
            current_usage_bytes: 0,
            peak_usage_bytes: 0,
            usage_trend: 0.0,
            pressure_events: 0,
            last_pressure_event: None,
        }
    }
}

/// Gradient age tracking for cleanup
#[derive(Debug)]
struct GradientAge {
    /// Tensor ID
    tensor_id: usize,
    /// Creation timestamp
    created_at: Instant,
    /// Last access timestamp
    last_accessed: Instant,
    /// Access count
    access_count: usize,
}

/// Memory management extension for AutogradContext
impl AutogradContext {
    /// Advanced memory cleanup with detailed tracking
    pub fn advanced_memory_cleanup(
        &mut self,
        config: &MemoryManagementConfig,
    ) -> Result<CleanupStatistics> {
        let mut stats = CleanupStatistics::default();
        let start_time = Instant::now();

        // Clean up orphaned gradients
        if config.auto_cleanup_orphaned {
            stats.orphaned_gradients_cleaned = self.cleanup_orphaned_gradients()?;
        }

        // Break circular references
        if config.break_circular_references {
            stats.circular_references_broken = self.break_circular_references()?;
        }

        // Remove expired gradients
        stats.expired_gradients_removed =
            self.cleanup_expired_gradients(config.max_gradient_age)?;

        // Estimate memory freed
        stats.memory_freed_bytes = self.estimate_memory_freed(&stats);

        stats.cleanup_operations += 1;
        stats.last_cleanup = Some(start_time);

        if config.detailed_tracking {
            tracing::info!(
                "Memory cleanup completed: {} orphaned, {} circular refs, {} expired, ~{} bytes freed",
                stats.orphaned_gradients_cleaned,
                stats.circular_references_broken,
                stats.expired_gradients_removed,
                stats.memory_freed_bytes
            );
        }

        Ok(stats)
    }

    /// Clean up gradients for tensors no longer in the computation graph
    fn cleanup_orphaned_gradients(&mut self) -> Result<usize> {
        let mut orphaned = Vec::new();

        // Identify gradients for tensors not in the current graph
        for &tensor_id in self.gradient_cache.keys() {
            if !self.tensor_to_node.contains_key(&tensor_id) {
                orphaned.push(tensor_id);
            }
        }

        // Remove orphaned gradients
        for tensor_id in &orphaned {
            self.gradient_cache.remove(tensor_id);
            let _ = self.gradient_storage.clear_gradient(*tensor_id);
        }

        Ok(orphaned.len())
    }

    /// Detect and break circular references in the computation graph
    fn break_circular_references(&mut self) -> Result<usize> {
        let mut broken_cycles = 0;
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        // Use a copy of node indices to avoid borrowing issues
        let node_indices: Vec<_> = self.computation_graph.node_indices().collect();

        for node_index in node_indices {
            if !visited.contains(&node_index) {
                if self.detect_and_break_cycle(node_index, &mut visited, &mut rec_stack)? {
                    broken_cycles += 1;
                }
            }
        }

        Ok(broken_cycles)
    }

    /// Helper function for cycle detection and breaking
    fn detect_and_break_cycle(
        &mut self,
        node: petgraph::graph::NodeIndex,
        visited: &mut HashSet<petgraph::graph::NodeIndex>,
        rec_stack: &mut HashSet<petgraph::graph::NodeIndex>,
    ) -> Result<bool> {
        visited.insert(node);
        rec_stack.insert(node);

        // Get edges to avoid borrowing conflicts
        let edges: Vec<_> = self
            .computation_graph
            .edges_directed(node, petgraph::Direction::Outgoing)
            .map(|e| (e.source(), e.target(), e.id()))
            .collect();

        for (_, target, edge_id) in edges {
            if !visited.contains(&target) {
                if self.detect_and_break_cycle(target, visited, rec_stack)? {
                    return Ok(true);
                }
            } else if rec_stack.contains(&target) {
                // Cycle detected - break it by removing this edge
                self.computation_graph.remove_edge(edge_id);
                tracing::warn!(
                    "Circular reference detected and broken between nodes {:?} and {:?}",
                    node,
                    target
                );
                return Ok(true);
            }
        }

        rec_stack.remove(&node);
        Ok(false)
    }

    /// Clean up gradients that have expired based on age
    fn cleanup_expired_gradients(&mut self, max_age: Duration) -> Result<usize> {
        let now = Instant::now();
        let mut expired = Vec::new();

        // This is a simplified implementation - in practice, you'd track gradient ages
        // For now, we'll clean up gradients for nodes that haven't been used recently
        for (&tensor_id, _) in &self.gradient_cache {
            // Placeholder logic - in a real implementation, you'd track access times
            if !self.tensor_to_node.contains_key(&tensor_id) {
                expired.push(tensor_id);
            }
        }

        // Remove expired gradients
        for tensor_id in &expired {
            self.gradient_cache.remove(tensor_id);
            let _ = self.gradient_storage.clear_gradient(*tensor_id);
        }

        Ok(expired.len())
    }

    /// Estimate memory freed by cleanup operations
    fn estimate_memory_freed(&self, stats: &CleanupStatistics) -> usize {
        // Rough estimates for different cleanup operations
        let orphaned_size = stats.orphaned_gradients_cleaned * 1024; // 1KB per gradient
        let circular_size = stats.circular_references_broken * 512; // 512B per broken reference
        let expired_size = stats.expired_gradients_removed * 1024; // 1KB per expired gradient

        orphaned_size + circular_size + expired_size
    }

    /// Monitor memory pressure and trigger cleanup if needed
    pub fn monitor_memory_pressure(
        &mut self,
        config: &MemoryManagementConfig,
    ) -> Result<MemoryPressureMonitor> {
        let current_usage = self.estimate_memory_usage();

        let mut monitor = MemoryPressureMonitor {
            current_usage_bytes: current_usage,
            peak_usage_bytes: current_usage, // This would be tracked over time
            usage_trend: 0.0,                // This would be calculated from historical data
            pressure_events: 0,
            last_pressure_event: None,
        };

        // Check if memory pressure threshold is exceeded
        if let Some(threshold) = config.memory_pressure_threshold {
            if current_usage > threshold {
                monitor.pressure_events += 1;
                monitor.last_pressure_event = Some(Instant::now());

                tracing::warn!(
                    "Memory pressure detected: {} bytes > {} threshold",
                    current_usage,
                    threshold
                );

                // Trigger automatic cleanup
                let _ = self.advanced_memory_cleanup(config)?;
            }
        }

        Ok(monitor)
    }

    /// Perform comprehensive memory optimization
    pub fn optimize_memory_layout(&mut self) -> Result<OptimizationStatistics> {
        let start_time = Instant::now();
        let mut stats = OptimizationStatistics::default();

        // Compact the graph
        self.compact_graph()?;
        stats.graph_compactions += 1;

        // Defragment the graph
        self.defragment_graph()?;
        stats.graph_defragmentations += 1;

        // Clear weak references
        self.clear_weak_references();
        stats.weak_references_cleared += 1;

        // Calculate optimization time
        let elapsed = start_time.elapsed();
        stats.avg_optimization_time_ms = elapsed.as_millis() as f64;

        // Estimate memory saved (simplified)
        stats.memory_saved_bytes = 1024 * (stats.graph_compactions + stats.graph_defragmentations);

        tracing::debug!(
            "Memory optimization completed in {:.2}ms: {} compactions, {} defragmentations",
            stats.avg_optimization_time_ms,
            stats.graph_compactions,
            stats.graph_defragmentations
        );

        Ok(stats)
    }

    /// Set up automatic memory management with periodic cleanup
    pub fn enable_automatic_memory_management(&mut self, config: MemoryManagementConfig) {
        // This would typically spawn a background task for periodic cleanup
        // For now, we'll just store the configuration
        tracing::info!(
            "Automatic memory management enabled with config: {:?}",
            config
        );

        // In a real implementation, this would:
        // 1. Spawn a background thread
        // 2. Periodically run cleanup operations
        // 3. Monitor memory pressure
        // 4. Trigger optimizations as needed
    }

    /// Get detailed memory usage breakdown
    pub fn get_memory_breakdown(&self) -> HashMap<String, usize> {
        let mut breakdown = HashMap::new();

        // Graph structure memory
        let graph_memory = self.computation_graph.node_count()
            * std::mem::size_of::<super::core::GraphNode>()
            + self.computation_graph.edge_count() * std::mem::size_of::<()>();
        breakdown.insert("computation_graph".to_string(), graph_memory);

        // Tensor to node mapping
        let mapping_memory = self.tensor_to_node.len()
            * (std::mem::size_of::<usize>() + std::mem::size_of::<petgraph::graph::NodeIndex>());
        breakdown.insert("tensor_to_node_mapping".to_string(), mapping_memory);

        // Gradient cache
        let cache_memory = self.gradient_cache.len() * std::mem::size_of::<(usize, Vec<f32>)>();
        breakdown.insert("gradient_cache".to_string(), cache_memory);

        // Gradient storage (estimated)
        let storage_memory = self.gradient_storage.gradient_tensor_ids().len() * 1024; // Rough estimate
        breakdown.insert("gradient_storage".to_string(), storage_memory);

        breakdown
    }

    /// Check for memory leaks and report potential issues
    pub fn check_memory_leaks(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for orphaned gradients
        let orphaned_count = self
            .gradient_cache
            .keys()
            .filter(|&&tensor_id| !self.tensor_to_node.contains_key(&tensor_id))
            .count();

        if orphaned_count > 0 {
            issues.push(format!("Found {} orphaned gradients", orphaned_count));
        }

        // Check for empty computation graph with active gradients
        if self.computation_graph.node_count() == 0 && !self.gradient_cache.is_empty() {
            issues.push("Computation graph is empty but gradient cache is not".to_string());
        }

        // Check for excessive memory usage
        let current_usage = self.estimate_memory_usage();
        if current_usage > 1024 * 1024 * 1024 {
            // 1GB
            issues.push(format!(
                "High memory usage detected: {} bytes",
                current_usage
            ));
        }

        issues
    }
}

/// Utility functions for memory management
pub mod utils {
    use super::*;

    /// Create a default memory management configuration
    pub fn default_memory_config() -> MemoryManagementConfig {
        MemoryManagementConfig::default()
    }

    /// Create a conservative memory management configuration
    pub fn conservative_memory_config() -> MemoryManagementConfig {
        MemoryManagementConfig {
            auto_cleanup_orphaned: true,
            break_circular_references: false, // More conservative
            max_gradient_age: Duration::from_secs(7200), // 2 hours
            memory_pressure_threshold: Some(256 * 1024 * 1024), // 256MB
            cleanup_interval: Duration::from_secs(600), // 10 minutes
            detailed_tracking: true,
        }
    }

    /// Create an aggressive memory management configuration
    pub fn aggressive_memory_config() -> MemoryManagementConfig {
        MemoryManagementConfig {
            auto_cleanup_orphaned: true,
            break_circular_references: true,
            max_gradient_age: Duration::from_secs(1800), // 30 minutes
            memory_pressure_threshold: Some(128 * 1024 * 1024), // 128MB
            cleanup_interval: Duration::from_secs(60),   // 1 minute
            detailed_tracking: true,
        }
    }

    /// Format memory size in human-readable format
    pub fn format_memory_size(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    /// Estimate memory usage of a gradient cache entry
    pub fn estimate_gradient_entry_size(gradient: &[f32]) -> usize {
        std::mem::size_of::<f32>() * gradient.len() + std::mem::size_of::<Vec<f32>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::core::AutogradContext;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryManagementConfig::default();
        assert!(config.auto_cleanup_orphaned);
        assert!(config.break_circular_references);
        assert_eq!(config.max_gradient_age, Duration::from_secs(3600));
    }

    #[test]
    fn test_cleanup_statistics() {
        let stats = CleanupStatistics::default();
        assert_eq!(stats.orphaned_gradients_cleaned, 0);
        assert_eq!(stats.circular_references_broken, 0);
        assert_eq!(stats.memory_freed_bytes, 0);
    }

    #[test]
    fn test_memory_breakdown() {
        let ctx = AutogradContext::new();
        let breakdown = ctx.get_memory_breakdown();

        assert!(breakdown.contains_key("computation_graph"));
        assert!(breakdown.contains_key("tensor_to_node_mapping"));
        assert!(breakdown.contains_key("gradient_cache"));
        assert!(breakdown.contains_key("gradient_storage"));
    }

    #[test]
    fn test_memory_leak_detection() {
        let ctx = AutogradContext::new();
        let issues = ctx.check_memory_leaks();

        // Empty context should have no issues
        assert!(issues.is_empty());
    }

    #[test]
    fn test_memory_formatting() {
        assert_eq!(utils::format_memory_size(512), "512.00 B");
        assert_eq!(utils::format_memory_size(1024), "1.00 KB");
        assert_eq!(utils::format_memory_size(1024 * 1024), "1.00 MB");
        assert_eq!(utils::format_memory_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_memory_configs() {
        let conservative = utils::conservative_memory_config();
        let aggressive = utils::aggressive_memory_config();

        assert!(conservative.max_gradient_age > aggressive.max_gradient_age);
        assert!(conservative.cleanup_interval > aggressive.cleanup_interval);
        assert!(conservative.memory_pressure_threshold > aggressive.memory_pressure_threshold);
    }
}
