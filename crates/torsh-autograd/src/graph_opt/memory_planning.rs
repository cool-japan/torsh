//! Memory Usage Planning and Optimization
//!
//! This module provides memory planning capabilities for efficient execution
//! of computation graphs, including memory usage simulation, lifetime analysis,
//! and budget management.

use super::graph_types::*;
use std::collections::HashSet;
use torsh_core::error::Result;

impl OptimizedGraph {
    /// Plan memory usage for optimal execution
    ///
    /// Simulates graph execution to predict memory usage patterns and identify
    /// opportunities for memory optimization. Tracks memory allocations and
    /// deallocations throughout the execution timeline.
    ///
    /// # Algorithm
    /// 1. Get optimal execution order for the graph
    /// 2. Simulate execution step by step
    /// 3. Track memory allocations for each node output
    /// 4. Identify when tensors can be safely deallocated
    /// 5. Monitor peak memory usage and budget compliance
    ///
    /// # Memory Management Strategy
    /// - Allocate memory when a node produces output
    /// - Deallocate memory when a tensor has no future uses
    /// - Track peak memory usage for optimization
    /// - Warn when memory budget is exceeded
    ///
    /// # Returns
    /// * `Result<()>` - Ok if planning succeeds, error otherwise
    pub fn plan_memory_usage(&mut self) -> Result<()> {
        tracing::debug!("Planning memory usage");

        let execution_order = self.get_execution_order()?;
        let mut memory_tracker = self.memory_tracker.lock();

        // Reset memory tracking state
        memory_tracker.node_memory.clear();
        memory_tracker.current_memory = 0;
        memory_tracker.peak_memory = 0;
        memory_tracker.memory_timeline.clear();

        // Simulate execution and track memory usage
        let mut active_tensors = HashSet::new();

        for &node_id in &execution_order {
            let node_idx = self.node_lookup[&node_id];
            let node = &self.graph[node_idx];

            // Allocate memory for this node's output
            memory_tracker.current_memory += node.memory_usage;
            memory_tracker
                .node_memory
                .insert(node_id, node.memory_usage);
            memory_tracker
                .memory_timeline
                .push((node_id, node.memory_usage, true)); // true = allocation
            active_tensors.insert(node_id);

            // Update peak memory usage
            memory_tracker.peak_memory = memory_tracker
                .peak_memory
                .max(memory_tracker.current_memory);

            // Check if we can deallocate input tensors
            for &input_id in &node.inputs {
                if !self.has_future_uses(
                    input_id,
                    &execution_order,
                    &execution_order.iter().position(|&x| x == node_id).unwrap(),
                )? && active_tensors.remove(&input_id)
                {
                    if let Some(&input_memory) = memory_tracker.node_memory.get(&input_id) {
                        memory_tracker.current_memory -= input_memory;
                        memory_tracker
                            .memory_timeline
                            .push((input_id, input_memory, false)); // false = deallocation
                    }
                }
            }

            // Check memory budget compliance
            if memory_tracker.current_memory > self.config.memory_budget {
                tracing::warn!(
                    "Memory usage {} bytes exceeds budget {} bytes at node {}",
                    memory_tracker.current_memory,
                    self.config.memory_budget,
                    node_id
                );
            }
        }

        // Update graph statistics
        self.stats
            .write()
            .expect("lock should not be poisoned")
            .peak_memory_bytes = memory_tracker.peak_memory;

        tracing::debug!(
            "Memory planning complete - Peak usage: {} bytes, Budget: {} bytes, Efficiency: {:.2}%",
            memory_tracker.peak_memory,
            self.config.memory_budget,
            (memory_tracker.peak_memory as f64 / self.config.memory_budget as f64) * 100.0
        );

        Ok(())
    }

    /// Check if a tensor has future uses in the execution order
    ///
    /// Determines whether a tensor will be needed as input to future operations,
    /// which is crucial for deciding when it's safe to deallocate memory.
    ///
    /// # Arguments
    /// * `tensor_id` - ID of the tensor to check
    /// * `execution_order` - Complete execution order of the graph
    /// * `current_pos` - Current position in the execution order
    ///
    /// # Returns
    /// * `Result<bool>` - True if tensor has future uses, false if it can be deallocated
    pub fn has_future_uses(
        &self,
        tensor_id: NodeId,
        execution_order: &[NodeId],
        current_pos: &usize,
    ) -> Result<bool> {
        // Check all remaining nodes in execution order
        for &node_id in &execution_order[current_pos + 1..] {
            let node_idx = self.node_lookup[&node_id];
            let node = &self.graph[node_idx];

            // If any future node uses this tensor as input, it has future uses
            if node.inputs.contains(&tensor_id) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Analyze memory usage patterns
    ///
    /// Provides detailed analysis of memory usage patterns including
    /// allocation/deallocation timeline and efficiency metrics.
    ///
    /// # Returns
    /// * `MemoryAnalysis` - Detailed memory usage analysis
    pub fn analyze_memory_patterns(&self) -> MemoryAnalysis {
        let memory_tracker = self.memory_tracker.lock();

        let mut total_allocations = 0;
        let mut total_deallocations = 0;
        let mut allocation_timeline = Vec::new();

        for &(node_id, size, is_allocation) in &memory_tracker.memory_timeline {
            if is_allocation {
                total_allocations += size;
                allocation_timeline.push((node_id, size, true));
            } else {
                total_deallocations += size;
                allocation_timeline.push((node_id, size, false));
            }
        }

        MemoryAnalysis {
            peak_memory: memory_tracker.peak_memory,
            total_allocations,
            total_deallocations,
            allocation_count: memory_tracker
                .memory_timeline
                .iter()
                .filter(|(_, _, is_alloc)| *is_alloc)
                .count(),
            deallocation_count: memory_tracker
                .memory_timeline
                .iter()
                .filter(|(_, _, is_alloc)| !*is_alloc)
                .count(),
            memory_efficiency: memory_tracker.peak_memory as f64 / self.config.memory_budget as f64,
            fragmentation_ratio: self.estimate_fragmentation_ratio(),
        }
    }

    /// Estimate memory fragmentation ratio
    ///
    /// Provides an estimate of memory fragmentation based on allocation patterns.
    ///
    /// # Returns
    /// * `f64` - Estimated fragmentation ratio (0.0 = no fragmentation, 1.0 = high fragmentation)
    fn estimate_fragmentation_ratio(&self) -> f64 {
        let memory_tracker = self.memory_tracker.lock();

        if memory_tracker.memory_timeline.is_empty() {
            return 0.0;
        }

        // Simple heuristic: fragmentation increases with allocation/deallocation frequency
        let total_operations = memory_tracker.memory_timeline.len() as f64;
        let unique_tensors = memory_tracker.node_memory.len() as f64;

        if unique_tensors == 0.0 {
            return 0.0;
        }

        // Higher ratio of operations to unique tensors suggests more fragmentation
        let fragmentation_estimate = (total_operations / unique_tensors - 2.0).max(0.0) / 10.0;
        fragmentation_estimate.min(1.0)
    }

    /// Get memory usage timeline
    ///
    /// Returns the complete timeline of memory allocations and deallocations.
    ///
    /// # Returns
    /// * `Vec<(NodeId, usize, bool)>` - Timeline entries (node_id, size, is_allocation)
    pub fn get_memory_timeline(&self) -> Vec<(NodeId, usize, bool)> {
        self.memory_tracker.lock().memory_timeline.clone()
    }

    /// Predict memory requirements for execution
    ///
    /// Estimates the total memory requirements without actually executing the graph.
    ///
    /// # Returns
    /// * `MemoryRequirements` - Predicted memory requirements
    pub fn predict_memory_requirements(&self) -> MemoryRequirements {
        let mut total_memory = 0;
        let mut max_single_allocation = 0;
        let node_count = self.graph.node_count();

        for node_idx in self.graph.node_indices() {
            let node = &self.graph[node_idx];
            total_memory += node.memory_usage;
            max_single_allocation = max_single_allocation.max(node.memory_usage);
        }

        MemoryRequirements {
            total_memory,
            max_single_allocation,
            estimated_peak: (total_memory as f64 * 0.6) as usize, // Heuristic: ~60% concurrent usage
            node_count,
        }
    }
}

/// Memory usage analysis results
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    /// Peak memory usage during execution
    pub peak_memory: usize,
    /// Total memory allocated
    pub total_allocations: usize,
    /// Total memory deallocated
    pub total_deallocations: usize,
    /// Number of allocation operations
    pub allocation_count: usize,
    /// Number of deallocation operations
    pub deallocation_count: usize,
    /// Memory efficiency ratio (peak / budget)
    pub memory_efficiency: f64,
    /// Estimated memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Memory requirements prediction
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Total memory if all tensors exist simultaneously
    pub total_memory: usize,
    /// Largest single tensor allocation
    pub max_single_allocation: usize,
    /// Estimated peak memory usage
    pub estimated_peak: usize,
    /// Number of nodes in the graph
    pub node_count: usize,
}
