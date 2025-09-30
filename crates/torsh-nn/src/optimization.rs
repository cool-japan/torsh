//! Performance optimization utilities for neural networks
//!
//! This module provides utilities for optimizing neural network performance,
//! including kernel fusion, memory optimization, and computation graph optimizations.

use crate::Module;
use std::collections::{HashMap, HashSet, VecDeque};
use torsh_core::error::{Result, TorshError};

/// Optimization strategy for neural networks
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStrategy {
    /// Fuse compatible operations into single kernels
    KernelFusion,
    /// Optimize memory allocation patterns
    MemoryOptimization,
    /// Remove redundant computations
    DeadCodeElimination,
    /// Reorder operations for better cache locality
    OperationReordering,
    /// Inline small functions
    InlineOptimization,
}

/// Fusion pattern for common operation combinations
#[derive(Debug, Clone)]
pub enum FusionPattern {
    /// Conv + BatchNorm + ReLU
    ConvBnRelu,
    /// Linear + ReLU
    LinearRelu,
    /// Linear + Dropout
    LinearDropout,
    /// Add + ReLU (residual connections)
    AddRelu,
    /// Mul + Add (scale and shift)
    MulAdd,
    /// Softmax + CrossEntropy
    SoftmaxCrossEntropy,
}

/// Memory optimization hint
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryHint {
    /// Prefer in-place operations
    InPlace,
    /// Use memory pooling
    Pooled,
    /// Stream computation to reduce peak memory
    Streaming,
    /// Use gradient checkpointing
    Checkpointing,
}

/// Operation node in computation graph
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Operation ID
    pub id: usize,
    /// Operation type
    pub op_type: String,
    /// Input nodes
    pub inputs: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Memory requirement in bytes
    pub memory_bytes: usize,
    /// Computation cost estimate
    pub flops: u64,
    /// Whether operation can be fused
    pub fusable: bool,
}

/// Computation graph for optimization analysis
#[derive(Debug)]
pub struct ComputationGraph {
    /// All operation nodes
    pub nodes: HashMap<usize, OpNode>,
    /// Graph topology (adjacency list)
    pub adjacency: HashMap<usize, Vec<usize>>,
    /// Input nodes (no dependencies)
    pub inputs: Vec<usize>,
    /// Output nodes (no dependents)
    pub outputs: Vec<usize>,
    /// Next available node ID
    next_id: usize,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, op_type: String, output_shape: Vec<usize>, flops: u64) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let memory_bytes = output_shape.iter().product::<usize>() * 4; // Assume f32

        let node = OpNode {
            id,
            op_type,
            inputs: Vec::new(),
            output_shape,
            memory_bytes,
            flops,
            fusable: true,
        };

        self.nodes.insert(id, node);
        self.adjacency.insert(id, Vec::new());

        id
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<()> {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return Err(TorshError::InvalidArgument(
                "Cannot add edge to non-existent nodes".to_string(),
            ));
        }

        self.adjacency.get_mut(&from).unwrap().push(to);
        self.nodes.get_mut(&to).unwrap().inputs.push(from);

        Ok(())
    }

    /// Compute topological order of nodes
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut in_degree: HashMap<usize, usize> = HashMap::new();

        // Initialize in-degrees
        for &node_id in self.nodes.keys() {
            in_degree.insert(node_id, self.nodes[&node_id].inputs.len());
        }

        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Add nodes with no incoming edges
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            // Process neighbors
            if let Some(neighbors) = self.adjacency.get(&node_id) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(TorshError::InvalidArgument(
                "Graph contains cycles".to_string(),
            ));
        }

        Ok(result)
    }

    /// Find fusable operation sequences
    pub fn find_fusion_candidates(&self) -> Vec<Vec<usize>> {
        let mut candidates = Vec::new();
        let visited = &mut HashSet::new();

        for &node_id in self.nodes.keys() {
            if !visited.contains(&node_id) {
                let sequence = self.find_fusion_sequence(node_id, visited);
                if sequence.len() > 1 {
                    candidates.push(sequence);
                }
            }
        }

        candidates
    }

    /// Find a sequence of fusable operations starting from a node
    fn find_fusion_sequence(&self, start: usize, visited: &mut HashSet<usize>) -> Vec<usize> {
        let mut sequence = Vec::new();
        let mut current = start;

        loop {
            if visited.contains(&current) || !self.nodes[&current].fusable {
                break;
            }

            visited.insert(current);
            sequence.push(current);

            // Check if we can continue the sequence
            let successors = self.adjacency.get(&current).unwrap();
            if successors.len() != 1 {
                break; // Multiple outputs, can't fuse
            }

            let next = successors[0];
            if self.nodes[&next].inputs.len() != 1 {
                break; // Multiple inputs to next node, can't fuse
            }

            current = next;
        }

        sequence
    }

    /// Estimate memory usage for the graph
    pub fn estimate_memory_usage(&self) -> usize {
        // Simple estimation: sum of all intermediate results
        self.nodes.values().map(|node| node.memory_bytes).sum()
    }

    /// Estimate total computation cost
    pub fn estimate_flops(&self) -> u64 {
        self.nodes.values().map(|node| node.flops).sum()
    }
}

/// Neural network optimizer
pub struct NetworkOptimizer {
    strategies: Vec<OptimizationStrategy>,
    fusion_patterns: Vec<FusionPattern>,
    memory_hints: Vec<MemoryHint>,
}

impl NetworkOptimizer {
    /// Create a new optimizer with default strategies
    pub fn new() -> Self {
        Self {
            strategies: vec![
                OptimizationStrategy::KernelFusion,
                OptimizationStrategy::MemoryOptimization,
                OptimizationStrategy::DeadCodeElimination,
            ],
            fusion_patterns: vec![
                FusionPattern::ConvBnRelu,
                FusionPattern::LinearRelu,
                FusionPattern::AddRelu,
            ],
            memory_hints: vec![MemoryHint::InPlace, MemoryHint::Pooled],
        }
    }

    /// Create an optimizer with custom configuration
    pub fn with_config(
        strategies: Vec<OptimizationStrategy>,
        fusion_patterns: Vec<FusionPattern>,
        memory_hints: Vec<MemoryHint>,
    ) -> Self {
        Self {
            strategies,
            fusion_patterns,
            memory_hints,
        }
    }

    /// Optimize a module
    pub fn optimize_module<M: Module>(&self, module: &M) -> Result<OptimizationReport> {
        let graph = self.build_computation_graph(module)?;
        let original_memory = graph.estimate_memory_usage();
        let original_flops = graph.estimate_flops();

        let mut optimizations = Vec::new();

        // Apply fusion optimizations
        if self
            .strategies
            .contains(&OptimizationStrategy::KernelFusion)
        {
            let fusion_results = self.apply_kernel_fusion(&graph)?;
            optimizations.extend(fusion_results);
        }

        // Apply memory optimizations
        if self
            .strategies
            .contains(&OptimizationStrategy::MemoryOptimization)
        {
            let memory_results = self.apply_memory_optimization(&graph)?;
            optimizations.extend(memory_results);
        }

        // Estimate improvements
        let optimized_memory = self.estimate_optimized_memory(&graph, &optimizations);
        let optimized_flops = self.estimate_optimized_flops(&graph, &optimizations);

        Ok(OptimizationReport {
            original_memory,
            optimized_memory,
            memory_reduction: original_memory - optimized_memory,
            original_flops,
            optimized_flops,
            flops_reduction: original_flops - optimized_flops,
            optimizations,
        })
    }

    /// Build computation graph from module (simplified)
    fn build_computation_graph<M: Module>(&self, _module: &M) -> Result<ComputationGraph> {
        // This is a simplified implementation
        // In practice, you'd traverse the module's computation graph
        let mut graph = ComputationGraph::new();

        // Add some example nodes
        let input_id = graph.add_node("input".to_string(), vec![1, 3, 224, 224], 0);
        let conv_id = graph.add_node("conv2d".to_string(), vec![1, 64, 112, 112], 1_000_000);
        let bn_id = graph.add_node("batch_norm".to_string(), vec![1, 64, 112, 112], 100_000);
        let relu_id = graph.add_node("relu".to_string(), vec![1, 64, 112, 112], 50_000);

        graph.add_edge(input_id, conv_id)?;
        graph.add_edge(conv_id, bn_id)?;
        graph.add_edge(bn_id, relu_id)?;

        Ok(graph)
    }

    /// Apply kernel fusion optimizations
    fn apply_kernel_fusion(&self, graph: &ComputationGraph) -> Result<Vec<OptimizationApplied>> {
        let mut optimizations = Vec::new();
        let fusion_candidates = graph.find_fusion_candidates();

        for candidate in fusion_candidates {
            if candidate.len() >= 2 {
                let ops: Vec<String> = candidate
                    .iter()
                    .map(|&id| graph.nodes[&id].op_type.clone())
                    .collect();

                // Check for known fusion patterns
                if self.matches_fusion_pattern(&ops) {
                    optimizations.push(OptimizationApplied {
                        optimization_type: "kernel_fusion".to_string(),
                        description: format!("Fused operations: {}", ops.join(" + ")),
                        memory_saved: self.estimate_fusion_memory_savings(&candidate, graph),
                        flops_saved: self.estimate_fusion_flops_savings(&candidate, graph),
                    });
                }
            }
        }

        Ok(optimizations)
    }

    /// Apply memory optimizations
    fn apply_memory_optimization(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<OptimizationApplied>> {
        let mut optimizations = Vec::new();

        // Look for in-place operation opportunities
        if self.memory_hints.contains(&MemoryHint::InPlace) {
            for node in graph.nodes.values() {
                if self.can_be_inplace(&node.op_type) {
                    optimizations.push(OptimizationApplied {
                        optimization_type: "inplace_operation".to_string(),
                        description: format!("Made {} operation in-place", node.op_type),
                        memory_saved: node.memory_bytes,
                        flops_saved: 0,
                    });
                }
            }
        }

        Ok(optimizations)
    }

    /// Check if operation sequence matches known fusion patterns
    fn matches_fusion_pattern(&self, ops: &[String]) -> bool {
        for pattern in &self.fusion_patterns {
            match pattern {
                FusionPattern::ConvBnRelu => {
                    if ops.len() == 3
                        && ops[0] == "conv2d"
                        && ops[1] == "batch_norm"
                        && ops[2] == "relu"
                    {
                        return true;
                    }
                }
                FusionPattern::LinearRelu => {
                    if ops.len() == 2 && ops[0] == "linear" && ops[1] == "relu" {
                        return true;
                    }
                }
                FusionPattern::AddRelu => {
                    if ops.len() == 2 && ops[0] == "add" && ops[1] == "relu" {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Check if operation can be performed in-place
    fn can_be_inplace(&self, op_type: &str) -> bool {
        matches!(op_type, "relu" | "dropout" | "batch_norm" | "layer_norm")
    }

    /// Estimate memory savings from fusion
    fn estimate_fusion_memory_savings(&self, _nodes: &[usize], _graph: &ComputationGraph) -> usize {
        // Simplified: assume we save one intermediate buffer
        1024 * 1024 // 1MB placeholder
    }

    /// Estimate FLOPS savings from fusion
    fn estimate_fusion_flops_savings(&self, _nodes: &[usize], _graph: &ComputationGraph) -> u64 {
        // Simplified: assume small overhead reduction
        1000
    }

    /// Estimate optimized memory usage
    fn estimate_optimized_memory(
        &self,
        _graph: &ComputationGraph,
        optimizations: &[OptimizationApplied],
    ) -> usize {
        let savings: usize = optimizations.iter().map(|opt| opt.memory_saved).sum();
        _graph.estimate_memory_usage().saturating_sub(savings)
    }

    /// Estimate optimized FLOPS
    fn estimate_optimized_flops(
        &self,
        _graph: &ComputationGraph,
        optimizations: &[OptimizationApplied],
    ) -> u64 {
        let savings: u64 = optimizations.iter().map(|opt| opt.flops_saved).sum();
        _graph.estimate_flops().saturating_sub(savings)
    }
}

impl Default for NetworkOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Applied optimization result
#[derive(Debug, Clone)]
pub struct OptimizationApplied {
    /// Type of optimization
    pub optimization_type: String,
    /// Human-readable description
    pub description: String,
    /// Memory saved in bytes
    pub memory_saved: usize,
    /// FLOPS saved
    pub flops_saved: u64,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Original memory usage in bytes
    pub original_memory: usize,
    /// Optimized memory usage in bytes
    pub optimized_memory: usize,
    /// Memory reduction in bytes
    pub memory_reduction: usize,
    /// Original FLOPS count
    pub original_flops: u64,
    /// Optimized FLOPS count
    pub optimized_flops: u64,
    /// FLOPS reduction
    pub flops_reduction: u64,
    /// List of applied optimizations
    pub optimizations: Vec<OptimizationApplied>,
}

impl OptimizationReport {
    /// Get memory reduction percentage
    pub fn memory_reduction_percent(&self) -> f64 {
        if self.original_memory == 0 {
            0.0
        } else {
            (self.memory_reduction as f64 / self.original_memory as f64) * 100.0
        }
    }

    /// Get FLOPS reduction percentage
    pub fn flops_reduction_percent(&self) -> f64 {
        if self.original_flops == 0 {
            0.0
        } else {
            (self.flops_reduction as f64 / self.original_flops as f64) * 100.0
        }
    }

    /// Format report as string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Neural Network Optimization Report ===\n");
        report.push_str(&format!("Memory Usage:\n"));
        report.push_str(&format!(
            "  Original: {} MB\n",
            self.original_memory / (1024 * 1024)
        ));
        report.push_str(&format!(
            "  Optimized: {} MB\n",
            self.optimized_memory / (1024 * 1024)
        ));
        report.push_str(&format!(
            "  Reduction: {} MB ({:.1}%)\n",
            self.memory_reduction / (1024 * 1024),
            self.memory_reduction_percent()
        ));

        report.push_str(&format!("\nComputation Cost:\n"));
        report.push_str(&format!(
            "  Original: {} GFLOPS\n",
            self.original_flops / 1_000_000_000
        ));
        report.push_str(&format!(
            "  Optimized: {} GFLOPS\n",
            self.optimized_flops / 1_000_000_000
        ));
        report.push_str(&format!(
            "  Reduction: {} GFLOPS ({:.1}%)\n",
            self.flops_reduction / 1_000_000_000,
            self.flops_reduction_percent()
        ));

        report.push_str(&format!("\nOptimizations Applied:\n"));
        for opt in &self.optimizations {
            report.push_str(&format!(
                "  - {}: {}\n",
                opt.optimization_type, opt.description
            ));
        }

        report
    }
}

/// Memory profiler for tracking memory usage patterns
pub struct MemoryProfiler {
    allocations: HashMap<String, usize>,
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            peak_usage: 0,
            current_usage: 0,
        }
    }

    /// Record a memory allocation
    pub fn allocate(&mut self, name: String, size: usize) {
        self.allocations.insert(name, size);
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
    }

    /// Record a memory deallocation
    pub fn deallocate(&mut self, name: &str) {
        if let Some(size) = self.allocations.remove(name) {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Reset profiler
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.peak_usage = 0;
        self.current_usage = 0;
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for optimization
pub fn optimize_module<M: Module>(module: &M) -> Result<OptimizationReport> {
    let optimizer = NetworkOptimizer::new();
    optimizer.optimize_module(module)
}

pub fn optimize_for_inference<M: Module>(module: &M) -> Result<OptimizationReport> {
    let optimizer = NetworkOptimizer::with_config(
        vec![
            OptimizationStrategy::KernelFusion,
            OptimizationStrategy::MemoryOptimization,
            OptimizationStrategy::InlineOptimization,
        ],
        vec![
            FusionPattern::ConvBnRelu,
            FusionPattern::LinearRelu,
            FusionPattern::AddRelu,
            FusionPattern::MulAdd,
        ],
        vec![MemoryHint::InPlace, MemoryHint::Pooled],
    );
    optimizer.optimize_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_computation_graph() {
        let mut graph = ComputationGraph::new();

        let node1 = graph.add_node("input".to_string(), vec![1, 3, 224, 224], 0);
        let node2 = graph.add_node("conv2d".to_string(), vec![1, 64, 112, 112], 1000000);
        let node3 = graph.add_node("relu".to_string(), vec![1, 64, 112, 112], 50000);

        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        assert_eq!(graph.nodes.len(), 3);

        let topo_order = graph.topological_sort().unwrap();
        assert_eq!(topo_order, vec![node1, node2, node3]);

        let fusion_candidates = graph.find_fusion_candidates();
        assert!(!fusion_candidates.is_empty());
    }

    #[test]
    fn test_network_optimizer() {
        let optimizer = NetworkOptimizer::new();
        assert_eq!(optimizer.strategies.len(), 3);
        assert_eq!(optimizer.fusion_patterns.len(), 3);
        assert_eq!(optimizer.memory_hints.len(), 2);
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();

        profiler.allocate("tensor1".to_string(), 1024);
        assert_eq!(profiler.current_usage(), 1024);
        assert_eq!(profiler.peak_usage(), 1024);

        profiler.allocate("tensor2".to_string(), 2048);
        assert_eq!(profiler.current_usage(), 3072);
        assert_eq!(profiler.peak_usage(), 3072);

        profiler.deallocate("tensor1");
        assert_eq!(profiler.current_usage(), 2048);
        assert_eq!(profiler.peak_usage(), 3072);
    }

    #[test]
    fn test_optimization_report() {
        let report = OptimizationReport {
            original_memory: 1024 * 1024 * 10, // 10MB
            optimized_memory: 1024 * 1024 * 8, // 8MB
            memory_reduction: 1024 * 1024 * 2, // 2MB
            original_flops: 1_000_000_000,     // 1 GFLOP
            optimized_flops: 800_000_000,      // 0.8 GFLOP
            flops_reduction: 200_000_000,      // 0.2 GFLOP
            optimizations: vec![OptimizationApplied {
                optimization_type: "kernel_fusion".to_string(),
                description: "Fused conv + relu".to_string(),
                memory_saved: 1024 * 1024,
                flops_saved: 100_000_000,
            }],
        };

        assert_eq!(report.memory_reduction_percent(), 20.0);
        assert_eq!(report.flops_reduction_percent(), 20.0);

        let formatted = report.format_report();
        assert!(formatted.contains("Memory Usage:"));
        assert!(formatted.contains("Computation Cost:"));
        assert!(formatted.contains("Optimizations Applied:"));
    }
}
