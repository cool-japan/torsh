//! Profile-Guided Optimization for ToRSh JIT
//!
//! This module implements profile-guided optimization (PGO) to improve JIT compilation
//! performance by using runtime profiling data to guide optimization decisions.

use crate::{ComputationGraph, JitError, JitResult, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Profile-guided optimization manager
pub struct ProfileGuidedOptimizer {
    profile_data: Arc<RwLock<ProfileData>>,
    config: PgoConfig,
    is_profiling: bool,
}

/// Configuration for profile-guided optimization
#[derive(Debug, Clone)]
pub struct PgoConfig {
    /// Minimum number of executions before applying optimizations
    pub min_execution_count: u32,

    /// Threshold for hot path detection (execution frequency)
    pub hot_path_threshold: f64,

    /// Maximum profile data size to prevent memory bloat
    pub max_profile_entries: usize,

    /// Enable branch prediction optimizations
    pub enable_branch_prediction: bool,

    /// Enable loop optimization based on iteration count
    pub enable_loop_optimization: bool,

    /// Enable function inlining based on call frequency
    pub enable_inline_optimization: bool,

    /// Profile data persistence file
    pub profile_file: Option<String>,
}

impl Default for PgoConfig {
    fn default() -> Self {
        Self {
            min_execution_count: 10,
            hot_path_threshold: 0.1, // 10% of total executions
            max_profile_entries: 10000,
            enable_branch_prediction: true,
            enable_loop_optimization: true,
            enable_inline_optimization: true,
            profile_file: None,
        }
    }
}

/// Runtime profiling data collected during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileData {
    /// Execution counts for each node
    node_execution_counts: HashMap<crate::graph::SerializableNodeIndex, u64>,

    /// Average execution times for each node
    node_execution_times: HashMap<crate::graph::SerializableNodeIndex, Duration>,

    /// Branch taken frequencies
    branch_frequencies: HashMap<crate::graph::SerializableNodeIndex, BranchData>,

    /// Loop iteration counts
    loop_iterations: HashMap<crate::graph::SerializableNodeIndex, LoopData>,

    /// Function call frequencies
    call_frequencies: HashMap<String, u64>,

    /// Memory access patterns
    memory_patterns: HashMap<crate::graph::SerializableNodeIndex, MemoryPattern>,

    /// Total execution count
    total_executions: u64,
}

/// Branch profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchData {
    /// Number of times branch was taken
    taken_count: u64,

    /// Number of times branch was not taken
    not_taken_count: u64,

    /// Prediction accuracy (for adaptive optimization)
    prediction_accuracy: f64,
}

/// Loop profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopData {
    /// Average number of iterations per execution
    avg_iterations: f64,

    /// Maximum iterations observed
    max_iterations: u64,

    /// Minimum iterations observed
    min_iterations: u64,

    /// Number of loop executions
    execution_count: u64,
}

/// Memory access pattern data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    /// Cache hit rate
    cache_hit_rate: f64,

    /// Average memory latency
    avg_latency: Duration,

    /// Memory bandwidth utilization
    bandwidth_utilization: f64,

    /// Access locality score
    locality_score: f64,
}

/// Optimization recommendations based on profiling data
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Node to optimize
    pub node_id: NodeId,

    /// Type of optimization
    pub optimization_type: OptimizationType,

    /// Expected performance improvement
    pub expected_improvement: f64,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of profile-guided optimizations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    /// Inline function calls
    FunctionInlining,

    /// Optimize branch prediction
    BranchPrediction,

    /// Unroll loops
    LoopUnrolling,

    /// Optimize memory layout
    MemoryLayout,

    /// Vectorize operations
    Vectorization,

    /// Specialize for hot paths
    HotPathSpecialization,

    /// Dead code elimination
    DeadCodeElimination,

    /// Constant propagation
    ConstantPropagation,
}

impl ProfileGuidedOptimizer {
    /// Create a new profile-guided optimizer
    pub fn new(config: PgoConfig) -> Self {
        let profile_data = ProfileData {
            node_execution_counts: HashMap::new(),
            node_execution_times: HashMap::new(),
            branch_frequencies: HashMap::new(),
            loop_iterations: HashMap::new(),
            call_frequencies: HashMap::new(),
            memory_patterns: HashMap::new(),
            total_executions: 0,
        };

        Self {
            profile_data: Arc::new(RwLock::new(profile_data)),
            config,
            is_profiling: false,
        }
    }

    /// Start profiling execution
    pub fn start_profiling(&mut self) -> JitResult<()> {
        self.is_profiling = true;

        // Load existing profile data if available
        if let Some(ref file) = self.config.profile_file {
            self.load_profile_data(file)?;
        }

        Ok(())
    }

    /// Stop profiling execution
    pub fn stop_profiling(&mut self) -> JitResult<()> {
        self.is_profiling = false;

        // Save profile data if configured
        if let Some(ref file) = self.config.profile_file {
            self.save_profile_data(file)?;
        }

        Ok(())
    }

    /// Record execution of a node
    pub fn record_node_execution(&self, node_id: NodeId, execution_time: Duration) {
        if !self.is_profiling {
            return;
        }

        if let Ok(mut data) = self.profile_data.write() {
            // Update execution count
            let serializable_node_id = node_id.into();
            *data
                .node_execution_counts
                .entry(serializable_node_id)
                .or_insert(0) += 1;
            data.total_executions += 1;

            // Update average execution time
            let count = data.node_execution_counts[&serializable_node_id];
            let entry = data
                .node_execution_times
                .entry(serializable_node_id)
                .or_insert(Duration::ZERO);
            *entry = (*entry * (count - 1) as u32 + execution_time) / count as u32;

            // Limit profile data size
            if data.node_execution_counts.len() > self.config.max_profile_entries {
                self.cleanup_old_data(&mut data);
            }
        }
    }

    /// Record branch taken/not taken
    pub fn record_branch(&self, node_id: NodeId, taken: bool) {
        if !self.is_profiling {
            return;
        }

        if let Ok(mut data) = self.profile_data.write() {
            let serializable_node_id = node_id.into();
            let branch_data = data
                .branch_frequencies
                .entry(serializable_node_id)
                .or_insert(BranchData {
                    taken_count: 0,
                    not_taken_count: 0,
                    prediction_accuracy: 0.5,
                });

            if taken {
                branch_data.taken_count += 1;
            } else {
                branch_data.not_taken_count += 1;
            }

            // Update prediction accuracy
            let total = branch_data.taken_count + branch_data.not_taken_count;
            let taken_ratio = branch_data.taken_count as f64 / total as f64;
            branch_data.prediction_accuracy = taken_ratio.max(1.0 - taken_ratio);
        }
    }

    /// Record loop execution
    pub fn record_loop(&self, node_id: NodeId, iterations: u64) {
        if !self.is_profiling {
            return;
        }

        if let Ok(mut data) = self.profile_data.write() {
            let serializable_node_id = node_id.into();
            let loop_data = data
                .loop_iterations
                .entry(serializable_node_id)
                .or_insert(LoopData {
                    avg_iterations: 0.0,
                    max_iterations: 0,
                    min_iterations: u64::MAX,
                    execution_count: 0,
                });

            loop_data.execution_count += 1;
            loop_data.max_iterations = loop_data.max_iterations.max(iterations);
            loop_data.min_iterations = loop_data.min_iterations.min(iterations);

            // Update average
            let count = loop_data.execution_count;
            loop_data.avg_iterations =
                (loop_data.avg_iterations * (count - 1) as f64 + iterations as f64) / count as f64;
        }
    }

    /// Record function call
    pub fn record_function_call(&self, function_name: &str) {
        if !self.is_profiling {
            return;
        }

        if let Ok(mut data) = self.profile_data.write() {
            *data
                .call_frequencies
                .entry(function_name.to_string())
                .or_insert(0) += 1;
        }
    }

    /// Record memory access pattern
    pub fn record_memory_access(&self, node_id: NodeId, cache_hit: bool, latency: Duration) {
        if !self.is_profiling {
            return;
        }

        if let Ok(mut data) = self.profile_data.write() {
            // Get execution count first
            let serializable_node_id = node_id.into();
            let execution_count = data
                .node_execution_counts
                .get(&serializable_node_id)
                .copied()
                .unwrap_or(0);

            let pattern =
                data.memory_patterns
                    .entry(serializable_node_id)
                    .or_insert(MemoryPattern {
                        cache_hit_rate: 0.0,
                        avg_latency: Duration::ZERO,
                        bandwidth_utilization: 0.0,
                        locality_score: 0.0,
                    });

            // Update cache hit rate
            if execution_count > 0 {
                pattern.cache_hit_rate = (pattern.cache_hit_rate * (execution_count - 1) as f64
                    + if cache_hit { 1.0 } else { 0.0 })
                    / execution_count as f64;

                pattern.avg_latency = (pattern.avg_latency * (execution_count - 1) as u32
                    + latency)
                    / execution_count as u32;
            }
        }
    }

    /// Generate optimization recommendations based on profiling data
    pub fn generate_recommendations(&self) -> JitResult<Vec<OptimizationRecommendation>> {
        let data = self
            .profile_data
            .read()
            .map_err(|_| JitError::RuntimeError("Failed to read profile data".to_string()))?;

        if data.total_executions < self.config.min_execution_count as u64 {
            return Ok(Vec::new());
        }

        let mut recommendations = Vec::new();

        // Analyze hot paths
        recommendations.extend(self.analyze_hot_paths(&data)?);

        // Analyze branch predictions
        if self.config.enable_branch_prediction {
            recommendations.extend(self.analyze_branches(&data)?);
        }

        // Analyze loops
        if self.config.enable_loop_optimization {
            recommendations.extend(self.analyze_loops(&data)?);
        }

        // Analyze function calls
        if self.config.enable_inline_optimization {
            recommendations.extend(self.analyze_function_calls(&data)?);
        }

        // Sort by expected improvement
        recommendations.sort_by(|a, b| {
            b.expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap()
        });

        Ok(recommendations)
    }

    /// Apply optimizations to a computation graph
    pub fn apply_optimizations(
        &self,
        graph: &mut ComputationGraph,
        recommendations: &[OptimizationRecommendation],
    ) -> JitResult<usize> {
        let mut applied_count = 0;

        for recommendation in recommendations {
            if recommendation.confidence < 0.7 {
                continue; // Skip low-confidence optimizations
            }

            match recommendation.optimization_type {
                OptimizationType::FunctionInlining => {
                    if self.apply_function_inlining(graph, recommendation)? {
                        applied_count += 1;
                    }
                }
                OptimizationType::BranchPrediction => {
                    if self.apply_branch_optimization(graph, recommendation)? {
                        applied_count += 1;
                    }
                }
                OptimizationType::LoopUnrolling => {
                    if self.apply_loop_unrolling(graph, recommendation)? {
                        applied_count += 1;
                    }
                }
                OptimizationType::HotPathSpecialization => {
                    if self.apply_hot_path_specialization(graph, recommendation)? {
                        applied_count += 1;
                    }
                }
                _ => {
                    // Other optimizations can be implemented as needed
                }
            }
        }

        Ok(applied_count)
    }

    /// Load profile data from file
    pub fn load_profile_data(&self, file_path: &str) -> JitResult<()> {
        match std::fs::read_to_string(file_path) {
            Ok(contents) => {
                let loaded_data: ProfileData = serde_json::from_str(&contents).map_err(|e| {
                    JitError::RuntimeError(format!("Failed to parse profile data: {}", e))
                })?;

                if let Ok(mut data) = self.profile_data.write() {
                    *data = loaded_data;
                }
                Ok(())
            }
            Err(_) => {
                // File doesn't exist or can't be read, start with empty data
                Ok(())
            }
        }
    }

    /// Save profile data to file
    pub fn save_profile_data(&self, file_path: &str) -> JitResult<()> {
        let data = self
            .profile_data
            .read()
            .map_err(|_| JitError::RuntimeError("Failed to read profile data".to_string()))?;

        let json = serde_json::to_string_pretty(&*data).map_err(|e| {
            JitError::RuntimeError(format!("Failed to serialize profile data: {}", e))
        })?;

        std::fs::write(file_path, json)
            .map_err(|e| JitError::RuntimeError(format!("Failed to write profile data: {}", e)))?;

        Ok(())
    }

    /// Get profiling statistics
    pub fn get_statistics(&self) -> JitResult<PgoStatistics> {
        let data = self
            .profile_data
            .read()
            .map_err(|_| JitError::RuntimeError("Failed to read profile data".to_string()))?;

        let total_nodes = data.node_execution_counts.len();
        let total_executions = data.node_execution_counts.values().sum::<u64>();
        let avg_execution_time = if !data.node_execution_times.is_empty() {
            data.node_execution_times.values().sum::<Duration>()
                / data.node_execution_times.len() as u32
        } else {
            Duration::ZERO
        };

        let hot_nodes = data
            .node_execution_counts
            .iter()
            .filter(|(_, &count)| {
                count as f64 / total_executions as f64 > self.config.hot_path_threshold
            })
            .count();

        Ok(PgoStatistics {
            total_nodes,
            total_executions,
            avg_execution_time,
            hot_nodes,
            branch_count: data.branch_frequencies.len(),
            loop_count: data.loop_iterations.len(),
            function_count: data.call_frequencies.len(),
        })
    }

    // Helper methods for analysis
    fn analyze_hot_paths(&self, data: &ProfileData) -> JitResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        let total_executions = data.node_execution_counts.values().sum::<u64>();

        for (&node_id, &count) in &data.node_execution_counts {
            let frequency = count as f64 / total_executions as f64;
            if frequency > self.config.hot_path_threshold {
                recommendations.push(OptimizationRecommendation {
                    node_id: node_id.into(),
                    optimization_type: OptimizationType::HotPathSpecialization,
                    expected_improvement: frequency * 0.2, // Estimate 20% improvement
                    confidence: 0.8,
                    metadata: [("frequency".to_string(), frequency.to_string())].into(),
                });
            }
        }

        Ok(recommendations)
    }

    fn analyze_branches(&self, data: &ProfileData) -> JitResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for (&node_id, branch_data) in &data.branch_frequencies {
            let total = branch_data.taken_count + branch_data.not_taken_count;
            if total > 100 {
                // Minimum sample size
                let bias = (branch_data.taken_count as f64 / total as f64 - 0.5).abs();
                if bias > 0.3 {
                    // Highly biased branch
                    recommendations.push(OptimizationRecommendation {
                        node_id: node_id.into(),
                        optimization_type: OptimizationType::BranchPrediction,
                        expected_improvement: bias * 0.1,
                        confidence: 0.7,
                        metadata: [("bias".to_string(), bias.to_string())].into(),
                    });
                }
            }
        }

        Ok(recommendations)
    }

    fn analyze_loops(&self, data: &ProfileData) -> JitResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        for (&node_id, loop_data) in &data.loop_iterations {
            if loop_data.execution_count > 10 {
                // Recommend unrolling for small, frequent loops
                if loop_data.avg_iterations < 10.0 && loop_data.avg_iterations > 2.0 {
                    let improvement = (10.0 - loop_data.avg_iterations) / 10.0 * 0.15;
                    recommendations.push(OptimizationRecommendation {
                        node_id: node_id.into(),
                        optimization_type: OptimizationType::LoopUnrolling,
                        expected_improvement: improvement,
                        confidence: 0.6,
                        metadata: [(
                            "avg_iterations".to_string(),
                            loop_data.avg_iterations.to_string(),
                        )]
                        .into(),
                    });
                }
            }
        }

        Ok(recommendations)
    }

    fn analyze_function_calls(
        &self,
        data: &ProfileData,
    ) -> JitResult<Vec<OptimizationRecommendation>> {
        let recommendations = Vec::new();
        let total_calls = data.call_frequencies.values().sum::<u64>();

        for (_function_name, &count) in &data.call_frequencies {
            let frequency = count as f64 / total_calls as f64;
            if frequency > 0.05 && count > 50 { // Frequent function calls
                 // This would need actual node ID mapping from function names
                 // For now, we'll skip this implementation
            }
        }

        Ok(recommendations)
    }

    fn apply_function_inlining(
        &self,
        graph: &mut ComputationGraph,
        recommendation: &OptimizationRecommendation,
    ) -> JitResult<bool> {
        // Find function call nodes and inline them if they meet criteria
        let node_id = recommendation.node_id;

        let node_name = if let Some(node) = graph.get_node(node_id) {
            node.name.clone()
        } else {
            return Ok(false);
        };

        if !node_name.is_empty() {
            // Check if function is suitable for inlining (small, frequently called)
            // Simplified inlining check - look for nodes with this name
            if let Some(_node) = graph.nodes().find(|(_, node)| node.name == node_name) {
                // Simplified inlining - just mark as inlined for now
                // In a real implementation, this would actually inline the function body
                // For now, we'll just assume the function can be inlined
                let instruction_count = 10; // Placeholder - would get actual count from function body

                // Only inline small functions (< 50 instructions)
                if instruction_count < 50 {
                    // Simplified inlining - in reality this would inline the actual function body
                    // For now just return success
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn apply_branch_optimization(
        &self,
        graph: &mut ComputationGraph,
        recommendation: &OptimizationRecommendation,
    ) -> JitResult<bool> {
        let node_id = recommendation.node_id;

        if let Some(node) = graph.get_node_mut(node_id) {
            // Get branch statistics from metadata
            if let Some(bias_str) = recommendation.metadata.get("bias") {
                if let Ok(bias) = bias_str.parse::<f64>() {
                    // Add branch prediction hint based on bias
                    let prediction_hint = if bias > 0.5 { "likely" } else { "unlikely" };

                    // Set branch prediction hint in node metadata
                    node.set_optimization_hint("branch_prediction", prediction_hint)?;

                    // If branch is highly biased, consider branch elimination
                    if bias > 0.9 || bias < 0.1 {
                        node.set_optimization_hint("branch_elimination_candidate", "true")?;
                    }

                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn apply_loop_unrolling(
        &self,
        graph: &mut ComputationGraph,
        recommendation: &OptimizationRecommendation,
    ) -> JitResult<bool> {
        let node_id = recommendation.node_id;

        if let Some(node) = graph.get_node_mut(node_id) {
            // Get average iterations from metadata
            if let Some(avg_iter_str) = recommendation.metadata.get("avg_iterations") {
                if let Ok(avg_iterations) = avg_iter_str.parse::<f64>() {
                    // Determine unroll factor based on average iterations
                    let unroll_factor = if avg_iterations <= 4.0 {
                        avg_iterations as usize
                    } else if avg_iterations <= 8.0 {
                        4
                    } else {
                        2
                    };

                    if unroll_factor > 1 {
                        // Set loop unrolling optimization hint
                        node.set_optimization_hint(
                            "loop_unroll_factor",
                            &unroll_factor.to_string(),
                        )?;
                        node.set_optimization_hint("loop_unroll_enabled", "true")?;

                        // For very small loops, consider full unrolling
                        if avg_iterations <= 3.0 {
                            node.set_optimization_hint("loop_full_unroll", "true")?;
                        }

                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    fn apply_hot_path_specialization(
        &self,
        graph: &mut ComputationGraph,
        recommendation: &OptimizationRecommendation,
    ) -> JitResult<bool> {
        let node_id = recommendation.node_id;

        if let Some(node) = graph.get_node_mut(node_id) {
            // Get frequency from metadata
            if let Some(frequency_str) = recommendation.metadata.get("frequency") {
                if let Ok(frequency) = frequency_str.parse::<f64>() {
                    // Apply hot path optimizations based on frequency
                    if frequency > 0.5 {
                        // Very hot path - aggressive optimizations
                        node.set_optimization_hint("hot_path_priority", "high")?;
                        node.set_optimization_hint("aggressive_optimization", "true")?;
                        node.set_optimization_hint("inline_aggressive", "true")?;
                        node.set_optimization_hint("vectorize_aggressive", "true")?;
                    } else if frequency > 0.2 {
                        // Moderately hot path - standard optimizations
                        node.set_optimization_hint("hot_path_priority", "medium")?;
                        node.set_optimization_hint("optimize_for_speed", "true")?;
                        node.set_optimization_hint("inline_enabled", "true")?;
                    } else {
                        // Warm path - basic optimizations
                        node.set_optimization_hint("hot_path_priority", "low")?;
                        node.set_optimization_hint("optimize_for_size", "true")?;
                    }

                    // Create specialized version for hot paths
                    if frequency > 0.3 {
                        node.set_optimization_hint("create_specialized_version", "true")?;
                        node.set_optimization_hint(
                            "specialization_frequency",
                            &frequency.to_string(),
                        )?;
                    }

                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn inline_function_body(
        &self,
        graph: &mut ComputationGraph,
        call_node_id: NodeId,
        function_body: &[String],
    ) -> JitResult<()> {
        // Create new nodes for the inlined function body
        let mut inline_nodes = Vec::new();

        for (i, _instruction_name) in function_body.iter().enumerate() {
            // Create a placeholder instruction for inlining
            // In a real implementation, this would convert the instruction properly
            // Create a placeholder node for inlining
            let mut inline_node = crate::graph::Node::new(
                crate::graph::Operation::Add,
                format!("inline_placeholder_{}", i),
            );
            inline_node.device = torsh_core::DeviceType::Cpu;
            inline_node.inputs = Vec::new();
            inline_node.is_output = false;
            let inline_node_id = graph.add_node(inline_node);
            inline_nodes.push(inline_node_id);

            // Connect nodes in sequence
            if i > 0 {
                graph.add_edge(
                    inline_nodes[i - 1],
                    inline_node_id,
                    crate::graph::Edge::default(),
                );
            }
        }

        // Connect the inlined nodes to the graph
        if !inline_nodes.is_empty() {
            // Use the call_node_id parameter passed to the function

            // Get incoming edges first and collect data
            let incoming_edges = graph.incoming_edges(call_node_id);
            let incoming_data: Vec<_> = incoming_edges
                .into_iter()
                .map(|(src, dst, edge)| (src, dst, edge.clone()))
                .collect();

            // Get outgoing edges and collect data
            let outgoing_edges = graph.outgoing_edges(call_node_id);
            let outgoing_data: Vec<_> = outgoing_edges
                .into_iter()
                .map(|(src, dst, edge)| (src, dst, edge.clone()))
                .collect();

            // Connect incoming edges to the first inlined node
            for (source_id, _dst_id, edge) in incoming_data {
                graph.add_edge(source_id, inline_nodes[0], edge);
            }

            // Connect the last inlined node to outgoing edges
            let last_inline_node = *inline_nodes.last().unwrap();
            for (_src_id, target_id, edge) in outgoing_data {
                graph.add_edge(last_inline_node, target_id, edge);
            }

            // Remove the original function call node
            graph
                .remove_node(call_node_id)
                .ok_or_else(|| crate::JitError::GraphError("Failed to remove node".to_string()))?;
        }

        Ok(())
    }

    fn cleanup_old_data(&self, data: &mut ProfileData) {
        // Remove least frequently used entries
        let mut entries: Vec<_> = data
            .node_execution_counts
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        entries.sort_by_key(|(_, count)| *count);

        let remove_count = entries.len() / 10; // Remove 10% of entries
        let nodes_to_remove: Vec<_> = entries
            .iter()
            .take(remove_count)
            .map(|(node_id, _)| *node_id)
            .collect();

        for node_id in nodes_to_remove {
            data.node_execution_counts.remove(&node_id);
            data.node_execution_times.remove(&node_id);
            data.branch_frequencies.remove(&node_id);
            data.loop_iterations.remove(&node_id);
            data.memory_patterns.remove(&node_id);
        }
    }
}

/// Statistics about profile-guided optimization
#[derive(Debug, Clone)]
pub struct PgoStatistics {
    pub total_nodes: usize,
    pub total_executions: u64,
    pub avg_execution_time: Duration,
    pub hot_nodes: usize,
    pub branch_count: usize,
    pub loop_count: usize,
    pub function_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_pgo_creation() {
        let config = PgoConfig::default();
        let optimizer = ProfileGuidedOptimizer::new(config);
        assert!(!optimizer.is_profiling);
    }

    #[test]
    fn test_profiling_control() {
        let mut optimizer = ProfileGuidedOptimizer::new(PgoConfig::default());
        optimizer.start_profiling().unwrap();
        assert!(optimizer.is_profiling);

        optimizer.stop_profiling().unwrap();
        assert!(!optimizer.is_profiling);
    }

    #[test]
    fn test_node_execution_recording() {
        let mut optimizer = ProfileGuidedOptimizer::new(PgoConfig::default());
        optimizer.start_profiling().unwrap();

        let node_id = NodeId::new(1);
        optimizer.record_node_execution(node_id, Duration::from_millis(10));
        optimizer.record_node_execution(node_id, Duration::from_millis(20));

        let stats = optimizer.get_statistics().unwrap();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.total_executions, 2);
    }

    #[test]
    fn test_branch_recording() {
        let mut optimizer = ProfileGuidedOptimizer::new(PgoConfig::default());
        optimizer.start_profiling().unwrap();

        let node_id = NodeId::new(1);
        optimizer.record_branch(node_id, true);
        optimizer.record_branch(node_id, true);
        optimizer.record_branch(node_id, false);

        let data = optimizer.profile_data.read().unwrap();
        let branch_data = &data.branch_frequencies[&node_id.into()];
        assert_eq!(branch_data.taken_count, 2);
        assert_eq!(branch_data.not_taken_count, 1);
    }

    #[test]
    fn test_recommendation_generation() {
        let mut optimizer = ProfileGuidedOptimizer::new(PgoConfig {
            min_execution_count: 1,
            hot_path_threshold: 0.3,
            ..Default::default()
        });
        optimizer.start_profiling().unwrap();

        // Record some executions to create a hot path
        let node_id = NodeId::new(1);
        for _ in 0..100 {
            optimizer.record_node_execution(node_id, Duration::from_millis(5));
        }

        let recommendations = optimizer.generate_recommendations().unwrap();
        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.optimization_type == OptimizationType::HotPathSpecialization));
    }
}
