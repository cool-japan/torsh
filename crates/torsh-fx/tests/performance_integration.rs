//! Integration tests for performance optimization features

use std::sync::Arc;
use tempfile::NamedTempFile;
use torsh_fx::*;

/// Test comprehensive performance features
#[test]
fn test_comprehensive_performance_features() {
    // Create a test graph
    let mut graph = FxGraph::new();
    let input = graph.add_node(Node::Input("x".to_string()));
    let relu1 = graph.add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
    let relu2 = graph.add_node(Node::Call("relu".to_string(), vec!["relu1".to_string()]));
    let output = graph.add_node(Node::Output);

    graph.add_edge(
        input,
        relu1,
        Edge {
            name: "x".to_string(),
        },
    );
    graph.add_edge(
        relu1,
        relu2,
        Edge {
            name: "relu1".to_string(),
        },
    );
    graph.add_edge(
        relu2,
        output,
        Edge {
            name: "relu2".to_string(),
        },
    );
    graph.add_input(input);
    graph.add_output(output);

    // Test graph linting
    let lint_report = graph.lint();
    println!(
        "Lint Report: {} issues found, score: {:.2}",
        lint_report.total_issues, lint_report.overall_score
    );
    assert!(lint_report.overall_score > 0.5);

    // Test memory analysis
    let memory_report = graph.analyze_memory();
    println!(
        "Memory Usage: {} bytes, efficiency: {:.2}",
        memory_report.total_size_bytes, memory_report.memory_efficiency
    );
    assert!(memory_report.total_size_bytes > 0);

    // Test graph metrics
    let metrics = graph.metrics();
    println!(
        "Graph Metrics - Nodes: {}, Edges: {}, Complexity: {:.2}",
        metrics.node_count, metrics.edge_count, metrics.complexity_score
    );
    assert_eq!(metrics.node_count, 4);
    assert_eq!(metrics.edge_count, 3);

    // Test pattern detection
    let patterns = graph.detect_patterns();
    println!("Detected {} patterns", patterns.len());
    assert!(!patterns.is_empty()); // Should detect linear chain pattern

    // Test graph compression
    let compressed = graph.compress().unwrap();
    println!(
        "Compressed graph: {} -> {} nodes",
        graph.node_count(),
        compressed.node_count()
    );

    // Test parallel traversal
    let graph_arc = Arc::new(graph.clone());
    let parallel_traversal = graph_arc.parallel_traversal();

    let result = parallel_traversal.parallel_topological_traverse(|_idx, _node| {
        // Just process the node, don't count
    });
    assert!(result.is_ok());

    // Test graph caching
    let cache = GraphCache::new(100);
    cache.cache_operation(
        "test_op".to_string(),
        "result".to_string(),
        std::time::Duration::from_millis(50),
    );
    let cached_result = cache.get_operation("test_op");
    assert!(cached_result.is_some());

    // Test performance profiler
    let profiler = PerformanceProfiler::new();
    profiler.record_operation("relu", std::time::Duration::from_millis(10));
    profiler.record_operation("relu", std::time::Duration::from_millis(12));

    let perf_report = profiler
        .profile_graph_execution(&graph, |_| Ok(()))
        .unwrap();
    println!(
        "Performance Report: {} operations",
        perf_report.total_operations
    );
    assert!(perf_report.total_operations > 0);
}

/// Test memory-mapped graph storage
#[test]
fn test_memory_mapped_storage() {
    let temp_file = NamedTempFile::new().unwrap();
    let mut mmap_graph = MemoryMappedGraph::new(temp_file.path(), 1000).unwrap();

    // Create a simple test graph
    let mut graph = FxGraph::new();
    let input = graph.add_node(Node::Input("x".to_string()));
    let output = graph.add_node(Node::Output);
    graph.add_edge(
        input,
        output,
        Edge {
            name: "direct".to_string(),
        },
    );
    graph.add_input(input);
    graph.add_output(output);

    // Test save and load
    let save_result = mmap_graph.save_graph(&graph);
    assert!(save_result.is_ok());

    let loaded_graph = mmap_graph.load_graph().unwrap();
    assert_eq!(loaded_graph.node_count(), graph.node_count());
}

/// Test adaptive memory management
#[test]
fn test_adaptive_memory_management() {
    let manager =
        AdaptiveMemoryManager::new(AllocationStrategy::Adaptive).with_memory_limit(10_000_000); // 10MB limit

    let mut graph = FxGraph::new();
    let input = graph.add_node(Node::Input("x".to_string()));
    let output = graph.add_node(Node::Output);
    graph.add_edge(
        input,
        output,
        Edge {
            name: "direct".to_string(),
        },
    );

    let layout = manager.allocate_graph_memory(&graph).unwrap();
    println!(
        "Memory layout - Size: {}, Memory mapping: {}, Compression: {}",
        layout.total_size, layout.use_memory_mapping, layout.compression_enabled
    );

    assert!(layout.total_size > 0);
    manager.deallocate_graph_memory(&layout);
}

/// Test graph diffing functionality
#[test]
fn test_graph_diffing() {
    // Create original graph
    let mut graph1 = FxGraph::new();
    let input1 = graph1.add_node(Node::Input("x".to_string()));
    let relu1 = graph1.add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
    let output1 = graph1.add_node(Node::Output);
    graph1.add_edge(
        input1,
        relu1,
        Edge {
            name: "x".to_string(),
        },
    );
    graph1.add_edge(
        relu1,
        output1,
        Edge {
            name: "relu".to_string(),
        },
    );

    // Create modified graph
    let mut graph2 = FxGraph::new();
    let input2 = graph2.add_node(Node::Input("x".to_string()));
    let relu2 = graph2.add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
    let sigmoid2 = graph2.add_node(Node::Call("sigmoid".to_string(), vec!["relu".to_string()]));
    let output2 = graph2.add_node(Node::Output);
    graph2.add_edge(
        input2,
        relu2,
        Edge {
            name: "x".to_string(),
        },
    );
    graph2.add_edge(
        relu2,
        sigmoid2,
        Edge {
            name: "relu".to_string(),
        },
    );
    graph2.add_edge(
        sigmoid2,
        output2,
        Edge {
            name: "sigmoid".to_string(),
        },
    );

    // Test diffing
    let diff = graph1.diff(&graph2);
    println!(
        "Graph diff - Added: {}, Removed: {}, Modified: {}",
        diff.added_nodes.len(),
        diff.removed_nodes.len(),
        diff.modified_nodes.len()
    );

    assert_eq!(diff.added_nodes.len(), 1); // sigmoid node added
}

/// Test graph linting with problematic graph
#[test]
fn test_graph_linting_issues() {
    let mut graph = FxGraph::new();

    // Create disconnected node (should trigger warning)
    let _disconnected = graph.add_node(Node::Call("orphan".to_string(), vec![]));

    // No inputs or outputs (should trigger errors)

    let linter = GraphLinter::new();
    let report = linter.lint_graph(&graph);

    println!("Linting problematic graph: {} issues", report.total_issues);
    assert!(report.total_issues > 0);
    assert!(report.overall_score < 1.0);
    assert!(!report.recommendations.is_empty());
}

/// Test performance profiling with bottleneck detection
#[test]
fn test_performance_profiling() {
    let profiler = PerformanceProfiler::new();

    // Simulate some operations with different performance characteristics
    profiler.record_operation("fast_op", std::time::Duration::from_millis(1));
    profiler.record_operation("slow_op", std::time::Duration::from_millis(100));
    profiler.record_operation("frequent_op", std::time::Duration::from_millis(10));
    profiler.record_operation("frequent_op", std::time::Duration::from_millis(12));
    profiler.record_operation("frequent_op", std::time::Duration::from_millis(11));

    let graph = FxGraph::new();
    let report = profiler
        .profile_graph_execution(&graph, |_| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok(())
        })
        .unwrap();

    println!(
        "Performance profiling - {} operations, {} bottlenecks",
        report.total_operations,
        report.bottlenecks.len()
    );

    assert!(report.total_operations > 0);
    assert!(!report.optimization_suggestions.is_empty());
}

/// Test cache statistics and LRU eviction
#[test]
fn test_cache_lru_eviction() {
    let cache = GraphCache::new(2); // Very small cache for testing eviction

    // Add items to fill cache
    cache.cache_operation(
        "op1".to_string(),
        "result1".to_string(),
        std::time::Duration::from_millis(10),
    );
    cache.cache_operation(
        "op2".to_string(),
        "result2".to_string(),
        std::time::Duration::from_millis(20),
    );

    // Add third item to trigger eviction
    cache.cache_operation(
        "op3".to_string(),
        "result3".to_string(),
        std::time::Duration::from_millis(30),
    );

    let stats = cache.statistics();
    println!(
        "Cache stats - Hits: {}, Misses: {}, Evictions: {}",
        stats.hits, stats.misses, stats.evictions
    );

    assert_eq!(stats.evictions, 1);

    // op1 should be evicted (oldest)
    assert!(cache.get_operation("op1").is_none());
    assert!(cache.get_operation("op2").is_some());
    assert!(cache.get_operation("op3").is_some());
}
