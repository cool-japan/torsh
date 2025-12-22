//! Performance optimization and analysis utilities for FX graphs
//!
//! This module provides advanced performance optimization features including:
//! - Parallel graph traversal for large graphs
//! - Graph caching and memoization for repeated operations
//! - Graph compression techniques for reduced memory usage
//! - Automatic performance profiling and bottleneck detection

use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
// SCIRS2 POLICY COMPLIANCE: Use scirs2_core::parallel_ops instead of direct rayon
use scirs2_core::parallel_ops::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Parallel graph traversal utilities for large graphs
pub struct ParallelTraversal {
    graph: Arc<FxGraph>,
    thread_pool_size: Option<usize>,
}

impl ParallelTraversal {
    /// Create a new parallel traversal instance
    pub fn new(graph: Arc<FxGraph>) -> Self {
        Self {
            graph,
            thread_pool_size: None,
        }
    }

    /// Set custom thread pool size (default: number of CPU cores)
    pub fn with_thread_pool_size(mut self, size: usize) -> Self {
        self.thread_pool_size = Some(size);
        self
    }

    /// Perform parallel topological traversal of the graph
    pub fn parallel_topological_traverse<F>(&self, visitor: F) -> TorshResult<()>
    where
        F: Fn(NodeIndex, &Node) + Send + Sync,
    {
        // Build dependency graph for topological ordering
        let dependencies = self.build_dependency_map();
        let visited = HashSet::new();
        let mut ready_nodes = Vec::new();

        // Find nodes with no dependencies
        for (idx, _) in self.graph.nodes() {
            if dependencies.get(&idx).map_or(true, |deps| deps.is_empty()) {
                ready_nodes.push(idx);
            }
        }

        let visited = Arc::new(Mutex::new(visited));
        let dependencies = Arc::new(dependencies);

        while !ready_nodes.is_empty() {
            // Process ready nodes in parallel
            ready_nodes.par_iter().for_each(|&idx| {
                if let Some(node) = self.graph.get_node(idx) {
                    visitor(idx, node);
                    visited.lock().unwrap().insert(idx);
                }
            });

            // Find newly ready nodes - optimized to avoid cloning the entire visited set
            ready_nodes.clear();

            for (idx, deps) in dependencies.iter() {
                let visited_guard = visited.lock().unwrap();
                if !visited_guard.contains(idx) {
                    if deps.iter().all(|dep| visited_guard.contains(dep)) {
                        ready_nodes.push(*idx);
                    }
                }
                // Release the lock early by dropping the guard
                drop(visited_guard);
            }
        }

        Ok(())
    }

    /// Perform parallel depth-first search with work stealing
    pub fn parallel_dfs<F>(&self, start_nodes: Vec<NodeIndex>, visitor: F) -> TorshResult<()>
    where
        F: Fn(NodeIndex, &Node) + Send + Sync,
    {
        let visited = Arc::new(Mutex::new(HashSet::new()));
        let visitor = Arc::new(visitor);

        start_nodes.into_par_iter().for_each(|start| {
            self.dfs_worker(start, visited.clone(), visitor.clone());
        });

        Ok(())
    }

    /// Build a dependency map for topological traversal
    fn build_dependency_map(&self) -> HashMap<NodeIndex, Vec<NodeIndex>> {
        let mut dependencies = HashMap::new();

        for (idx, _) in self.graph.nodes() {
            dependencies.insert(idx, Vec::new());
        }

        // Build reverse dependency map
        for edge_ref in self.graph.graph.edge_references() {
            use petgraph::visit::EdgeRef;
            let target = edge_ref.target();
            let source = edge_ref.source();
            dependencies.get_mut(&target).unwrap().push(source);
        }

        dependencies
    }

    /// DFS worker for parallel traversal
    fn dfs_worker<F>(
        &self,
        node: NodeIndex,
        visited: Arc<Mutex<HashSet<NodeIndex>>>,
        visitor: Arc<F>,
    ) where
        F: Fn(NodeIndex, &Node) + Send + Sync,
    {
        let mut stack = vec![node];

        while let Some(current) = stack.pop() {
            let already_visited = {
                let mut v = visited.lock().unwrap();
                if v.contains(&current) {
                    true
                } else {
                    v.insert(current);
                    false
                }
            };

            if already_visited {
                continue;
            }

            if let Some(node_data) = self.graph.get_node(current) {
                visitor(current, node_data);
            }

            // Add neighbors to stack
            for edge in self.graph.graph.edges(current) {
                stack.push(edge.target());
            }
        }
    }
}

/// Graph caching and memoization system
#[derive(Debug)]
pub struct GraphCache {
    operation_cache: RwLock<HashMap<String, CachedResult>>,
    subgraph_cache: RwLock<HashMap<String, Arc<FxGraph>>>,
    cache_stats: Arc<Mutex<CacheStatistics>>,
    max_cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    pub result: String,
    pub computation_time: Duration,
    pub access_count: u64,
    pub last_accessed: std::time::SystemTime,
}

#[derive(Debug, Default, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_computation_time_saved: Duration,
}

impl GraphCache {
    /// Create a new graph cache with specified maximum size
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            operation_cache: RwLock::new(HashMap::new()),
            subgraph_cache: RwLock::new(HashMap::new()),
            cache_stats: Arc::new(Mutex::new(CacheStatistics::default())),
            max_cache_size,
        }
    }

    /// Get cached operation result
    pub fn get_operation(&self, key: &str) -> Option<CachedResult> {
        let mut cache = self.operation_cache.write().unwrap();
        if let Some(result) = cache.get_mut(key) {
            result.access_count += 1;
            result.last_accessed = std::time::SystemTime::now();
            self.cache_stats.lock().unwrap().hits += 1;
            Some(result.clone())
        } else {
            self.cache_stats.lock().unwrap().misses += 1;
            None
        }
    }

    /// Cache operation result
    pub fn cache_operation(&self, key: String, result: String, computation_time: Duration) {
        let mut cache = self.operation_cache.write().unwrap();

        // Evict oldest entries if cache is full
        if cache.len() >= self.max_cache_size {
            self.evict_lru_operation(&mut cache);
        }

        let cached_result = CachedResult {
            result,
            computation_time,
            access_count: 1,
            last_accessed: std::time::SystemTime::now(),
        };

        cache.insert(key, cached_result);
    }

    /// Get cached subgraph
    pub fn get_subgraph(&self, key: &str) -> Option<Arc<FxGraph>> {
        let cache = self.subgraph_cache.read().unwrap();
        if let Some(graph) = cache.get(key) {
            self.cache_stats.lock().unwrap().hits += 1;
            Some(graph.clone())
        } else {
            self.cache_stats.lock().unwrap().misses += 1;
            None
        }
    }

    /// Cache subgraph
    pub fn cache_subgraph(&self, key: String, graph: Arc<FxGraph>) {
        let mut cache = self.subgraph_cache.write().unwrap();

        // Evict oldest entries if cache is full
        if cache.len() >= self.max_cache_size {
            self.evict_lru_subgraph(&mut cache);
        }

        cache.insert(key, graph);
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.cache_stats.lock().unwrap().clone()
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.operation_cache.write().unwrap().clear();
        self.subgraph_cache.write().unwrap().clear();
        *self.cache_stats.lock().unwrap() = CacheStatistics::default();
    }

    /// Evict least recently used operation
    fn evict_lru_operation(&self, cache: &mut HashMap<String, CachedResult>) {
        if let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, result)| result.last_accessed)
            .map(|(key, _)| key.clone())
        {
            cache.remove(&lru_key);
            self.cache_stats.lock().unwrap().evictions += 1;
        }
    }

    /// Evict least recently used subgraph (simplified LRU)
    fn evict_lru_subgraph(&self, cache: &mut HashMap<String, Arc<FxGraph>>) {
        if let Some(key) = cache.keys().next().cloned() {
            cache.remove(&key);
            self.cache_stats.lock().unwrap().evictions += 1;
        }
    }
}

/// Graph compression utilities
pub struct GraphCompression;

impl GraphCompression {
    /// Compress graph using operation deduplication
    pub fn deduplicate_operations(graph: &FxGraph) -> TorshResult<FxGraph> {
        let mut compressed_graph = FxGraph::new();
        let mut operation_map: HashMap<String, NodeIndex> = HashMap::new();
        let mut node_mapping: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        // First pass: deduplicate operations
        for (old_idx, node) in graph.nodes() {
            let operation_key = Self::operation_key(node);

            if let Some(&existing_idx) = operation_map.get(&operation_key) {
                // Reuse existing operation
                node_mapping.insert(old_idx, existing_idx);
            } else {
                // Create new operation
                let new_idx = compressed_graph.graph.add_node(node.clone());
                operation_map.insert(operation_key, new_idx);
                node_mapping.insert(old_idx, new_idx);
            }
        }

        // Second pass: rebuild edges
        for edge_ref in graph.graph.edge_references() {
            use petgraph::visit::EdgeRef;
            let old_source = edge_ref.source();
            let old_target = edge_ref.target();

            if let (Some(&new_source), Some(&new_target)) =
                (node_mapping.get(&old_source), node_mapping.get(&old_target))
            {
                // Avoid duplicate edges
                if !compressed_graph
                    .graph
                    .edges_connecting(new_source, new_target)
                    .next()
                    .is_some()
                {
                    compressed_graph.graph.add_edge(
                        new_source,
                        new_target,
                        edge_ref.weight().clone(),
                    );
                }
            }
        }

        // Update inputs and outputs
        compressed_graph.inputs = graph
            .inputs()
            .iter()
            .filter_map(|&idx| node_mapping.get(&idx).copied())
            .collect();
        compressed_graph.outputs = graph
            .outputs()
            .iter()
            .filter_map(|&idx| node_mapping.get(&idx).copied())
            .collect();

        Ok(compressed_graph)
    }

    /// Create operation key for deduplication
    fn operation_key(node: &Node) -> String {
        match node {
            Node::Input(name) => format!("input:{name}"),
            Node::Call(op, args) => {
                let args_str = args.join(",");
                format!("call:{op}:{args_str}")
            }
            Node::Output => "output".into(),
            Node::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                format!(
                    "conditional:{}:{}:{}",
                    condition,
                    then_branch.join(","),
                    else_branch.join(",")
                )
            }
            Node::Loop {
                condition,
                body,
                loop_vars,
            } => {
                format!(
                    "loop:{}:{}:{}",
                    condition,
                    body.join(","),
                    loop_vars.join(",")
                )
            }
            Node::Merge { inputs } => {
                let inputs_str = inputs.join(",");
                format!("merge:{inputs_str}")
            }
            Node::GetAttr { target, attr } => format!("getattr:{target}:{attr}"),
        }
    }

    /// Compress graph by removing redundant nodes
    pub fn remove_redundant_nodes(graph: &FxGraph) -> TorshResult<FxGraph> {
        let mut compressed_graph = FxGraph::new();
        let mut node_mapping: HashMap<NodeIndex, Option<NodeIndex>> = HashMap::new();

        // Identify redundant nodes (e.g., identity operations)
        for (old_idx, node) in graph.nodes() {
            if Self::is_redundant_node(node) {
                node_mapping.insert(old_idx, None); // Mark for removal
            } else {
                let new_idx = compressed_graph.graph.add_node(node.clone());
                node_mapping.insert(old_idx, Some(new_idx));
            }
        }

        // Rebuild edges, skipping redundant nodes
        for edge_ref in graph.graph.edge_references() {
            use petgraph::visit::EdgeRef;
            let old_source = edge_ref.source();
            let old_target = edge_ref.target();

            // Find actual source and target (skipping redundant nodes)
            let new_source = Self::find_actual_node(old_source, &node_mapping, graph);
            let new_target = Self::find_actual_node(old_target, &node_mapping, graph);

            if let (Some(source), Some(target)) = (new_source, new_target) {
                if source != target {
                    // Avoid self-loops
                    compressed_graph
                        .graph
                        .add_edge(source, target, edge_ref.weight().clone());
                }
            }
        }

        // Update inputs and outputs
        compressed_graph.inputs = graph
            .inputs()
            .iter()
            .filter_map(|&idx| Self::find_actual_node(idx, &node_mapping, graph))
            .collect();
        compressed_graph.outputs = graph
            .outputs()
            .iter()
            .filter_map(|&idx| Self::find_actual_node(idx, &node_mapping, graph))
            .collect();

        Ok(compressed_graph)
    }

    /// Check if a node is redundant (e.g., identity operation)
    fn is_redundant_node(node: &Node) -> bool {
        match node {
            Node::Call(op, _) => op == "identity" || op == "noop",
            _ => false,
        }
    }

    /// Find the actual node after skipping redundant nodes
    fn find_actual_node(
        start_idx: NodeIndex,
        node_mapping: &HashMap<NodeIndex, Option<NodeIndex>>,
        _graph: &FxGraph,
    ) -> Option<NodeIndex> {
        node_mapping.get(&start_idx).and_then(|&idx| idx)
    }
}

/// Automatic performance profiling and bottleneck detection
#[derive(Debug)]
pub struct PerformanceProfiler {
    operation_times: RwLock<HashMap<String, Vec<Duration>>>,
    bottlenecks: RwLock<Vec<PerformanceBottleneck>>,
    profiling_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub operation: String,
    pub average_time: Duration,
    pub frequency: u64,
    pub impact_score: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_operations: u64,
    pub total_time: Duration,
    pub average_operation_time: Duration,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub optimization_suggestions: Vec<String>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            operation_times: RwLock::new(HashMap::new()),
            bottlenecks: RwLock::new(Vec::new()),
            profiling_enabled: true,
        }
    }

    /// Enable or disable profiling
    pub fn set_profiling_enabled(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    /// Record operation execution time
    pub fn record_operation(&self, operation: &str, duration: Duration) {
        if !self.profiling_enabled {
            return;
        }

        let mut times = self.operation_times.write().unwrap();
        times
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Profile graph execution
    pub fn profile_graph_execution<F>(
        &self,
        graph: &FxGraph,
        executor: F,
    ) -> TorshResult<PerformanceReport>
    where
        F: FnOnce(&FxGraph) -> TorshResult<()>,
    {
        let start_time = Instant::now();

        // Execute the graph
        executor(graph)?;

        let total_time = start_time.elapsed();

        // Analyze performance data
        self.analyze_bottlenecks();

        // Generate report
        let report = self.generate_report(total_time);
        Ok(report)
    }

    /// Detect and analyze performance bottlenecks
    fn analyze_bottlenecks(&self) {
        let times = self.operation_times.read().unwrap();
        let mut bottlenecks = Vec::new();

        for (operation, durations) in times.iter() {
            if durations.is_empty() {
                continue;
            }

            let total_time: Duration = durations.iter().sum();
            let average_time = total_time / durations.len() as u32;
            let frequency = durations.len() as u64;

            // Calculate impact score (average time * frequency)
            let impact_score = average_time.as_secs_f64() * frequency as f64;

            // Generate recommendations
            let recommendations = self.generate_recommendations(operation, average_time, frequency);

            if impact_score > 0.1 {
                // Threshold for considering as bottleneck
                bottlenecks.push(PerformanceBottleneck {
                    operation: operation.clone(),
                    average_time,
                    frequency,
                    impact_score,
                    recommendations,
                });
            }
        }

        // Sort by impact score
        bottlenecks.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap());

        *self.bottlenecks.write().unwrap() = bottlenecks;
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        operation: &str,
        avg_time: Duration,
        frequency: u64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if avg_time.as_millis() > 100 {
            recommendations.push(format!(
                "Consider optimizing '{}' operation - high execution time",
                operation
            ));
        }

        if frequency > 1000 {
            recommendations.push(format!(
                "Operation '{}' is called frequently - consider caching",
                operation
            ));
        }

        if operation.contains("conv") && avg_time.as_millis() > 50 {
            recommendations.push(
                "Consider using optimized convolution algorithms or GPU acceleration".to_string(),
            );
        }

        if operation.contains("matmul") && avg_time.as_millis() > 20 {
            recommendations.push(
                "Consider using BLAS libraries or tensor cores for matrix multiplication"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Performance seems adequate for this operation".to_string());
        }

        recommendations
    }

    /// Generate comprehensive performance report
    fn generate_report(&self, total_time: Duration) -> PerformanceReport {
        let times = self.operation_times.read().unwrap();
        let bottlenecks = self.bottlenecks.read().unwrap().clone();

        let total_operations: u64 = times.values().map(|v| v.len() as u64).sum();
        let average_operation_time = if total_operations > 0 {
            total_time / total_operations as u32
        } else {
            Duration::from_millis(0)
        };

        let optimization_suggestions = self.generate_global_optimizations(&bottlenecks);

        PerformanceReport {
            total_operations,
            total_time,
            average_operation_time,
            bottlenecks,
            optimization_suggestions,
        }
    }

    /// Generate global optimization suggestions
    fn generate_global_optimizations(&self, bottlenecks: &[PerformanceBottleneck]) -> Vec<String> {
        let mut suggestions = Vec::new();

        if bottlenecks.len() > 5 {
            suggestions.push(
                "Consider using graph optimization passes to reduce operation count".to_string(),
            );
        }

        if bottlenecks
            .iter()
            .any(|b| b.operation.contains("copy") || b.operation.contains("transpose"))
        {
            suggestions
                .push("Consider memory layout optimizations to reduce data movement".to_string());
        }

        if bottlenecks.iter().any(|b| b.frequency > 100) {
            suggestions
                .push("Enable operation caching for frequently used computations".to_string());
        }

        suggestions
            .push("Consider using parallel execution for independent operations".to_string());
        suggestions
            .push("Enable compiler optimizations and use release build for production".to_string());

        suggestions
    }

    /// Clear all profiling data
    pub fn clear(&self) {
        self.operation_times.write().unwrap().clear();
        self.bottlenecks.write().unwrap().clear();
    }

    /// Get current bottlenecks
    pub fn bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        self.bottlenecks.read().unwrap().clone()
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, FxGraph, Node};
    use std::sync::Arc;

    #[test]
    fn test_parallel_traversal() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        let parallel_traversal = ParallelTraversal::new(Arc::new(graph));
        let visited_nodes = Vec::new();
        let visited_nodes = Arc::new(Mutex::new(visited_nodes));

        let result = parallel_traversal.parallel_topological_traverse(|idx, _node| {
            visited_nodes.lock().unwrap().push(idx);
        });

        assert!(result.is_ok());
        assert_eq!(visited_nodes.lock().unwrap().len(), 3);
    }

    #[test]
    fn test_graph_cache() {
        let cache = GraphCache::new(100);

        // Test operation caching
        assert!(cache.get_operation("test_op").is_none());

        cache.cache_operation(
            "test_op".to_string(),
            "result".to_string(),
            Duration::from_millis(100),
        );

        let cached = cache.get_operation("test_op").unwrap();
        assert_eq!(cached.result, "result");
        assert_eq!(cached.access_count, 2);

        // Test cache hit
        let cached_again = cache.get_operation("test_op").unwrap();
        assert_eq!(cached_again.access_count, 3);

        let stats = cache.statistics();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_graph_compression() {
        let mut graph = FxGraph::new();
        let input1 = graph.graph.add_node(Node::Input("x".to_string()));
        let input2 = graph.graph.add_node(Node::Input("x".to_string())); // Duplicate
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input1,
            relu,
            Edge {
                name: "x1".to_string(),
            },
        );
        graph.graph.add_edge(
            input2,
            relu,
            Edge {
                name: "x2".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );

        let compressed = GraphCompression::deduplicate_operations(&graph).unwrap();

        // Should have fewer nodes due to deduplication
        assert!(compressed.node_count() < graph.node_count());
    }

    #[test]
    fn test_performance_profiler() {
        let profiler = PerformanceProfiler::new();

        // Record some operations
        profiler.record_operation("conv2d", Duration::from_millis(100));
        profiler.record_operation("conv2d", Duration::from_millis(120));
        profiler.record_operation("relu", Duration::from_millis(10));

        // Create a simple graph for testing
        let graph = FxGraph::new();

        let report = profiler
            .profile_graph_execution(&graph, |_| Ok(()))
            .unwrap();

        assert_eq!(report.total_operations, 3);
        assert!(!report.bottlenecks.is_empty());
        assert!(!report.optimization_suggestions.is_empty());
    }

    #[test]
    fn test_cache_statistics() {
        let cache = GraphCache::new(2); // Small cache for testing eviction

        cache.cache_operation(
            "op1".to_string(),
            "result1".to_string(),
            Duration::from_millis(50),
        );
        // Small delay to ensure different timestamps
        std::thread::sleep(Duration::from_millis(1));

        cache.cache_operation(
            "op2".to_string(),
            "result2".to_string(),
            Duration::from_millis(75),
        );
        std::thread::sleep(Duration::from_millis(1));

        cache.cache_operation(
            "op3".to_string(),
            "result3".to_string(),
            Duration::from_millis(100),
        ); // Should trigger eviction

        let stats = cache.statistics();
        assert_eq!(stats.evictions, 1);

        // Verify cache contents - op1 should be evicted as it was oldest
        assert!(cache.get_operation("op1").is_none());
        assert!(cache.get_operation("op2").is_some() || cache.get_operation("op3").is_some());

        // Verify we have exactly 2 items cached
        let op2_exists = cache.get_operation("op2").is_some();
        let op3_exists = cache.get_operation("op3").is_some();
        assert_eq!((op2_exists as usize) + (op3_exists as usize), 2);
    }
}
