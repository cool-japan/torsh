//! Integration tests for torsh-graph

use approx::assert_relative_eq;
use torsh_core::device::DeviceType;
use torsh_graph::{
    conv::{AggregationType, GATConv, GCNConv, GINConv, GraphTransformer, MPNNConv, SAGEConv},
    functional::{elu, gelu, leaky_relu, mish, swish},
    pool::{global, hierarchical},
    scirs2_integration::{algorithms, generation, spatial},
    utils::{connectivity, graph_laplacian, metrics},
    GraphData, GraphLayer,
};
use torsh_tensor::creation::{from_vec, randn, zeros};

/// Create a simple test graph
fn create_test_graph() -> GraphData {
    // Create a simple 4-node graph with 5 edges
    let node_features = vec![
        1.0, 0.5, 0.2, // Node 0
        0.8, 1.0, 0.1, // Node 1
        0.3, 0.7, 1.0, // Node 2
        0.9, 0.2, 0.8, // Node 3
    ];
    let x = from_vec(node_features, &[4, 3], DeviceType::Cpu).unwrap();

    // Edges: (0,1), (1,2), (2,3), (3,0), (1,3)
    let edges = vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 0.0, 3.0];
    let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();

    GraphData::new(x, edge_index)
}

/// Create a larger test graph for more complex tests
fn create_large_test_graph() -> GraphData {
    let num_nodes = 10;
    let features_per_node = 8;

    // Random node features
    let x = randn(&[num_nodes, features_per_node]).unwrap();

    // Create a more connected graph
    let mut edges: Vec<f32> = Vec::new();
    for i in 0..num_nodes {
        for j in (i + 1)..num_nodes {
            if (i * 7 + j * 3) % 4 == 0 {
                // Pseudo-random edge selection
                edges.extend_from_slice(&[i as f32, j as f32]);
            }
        }
    }

    let num_edges = edges.len() / 2;
    let edge_index = from_vec(edges, &[2, num_edges], DeviceType::Cpu).unwrap();

    GraphData::new(x, edge_index)
}

#[test]
fn test_graph_data_creation() {
    let graph = create_test_graph();

    assert_eq!(graph.num_nodes, 4);
    assert_eq!(graph.num_edges, 5);
    assert_eq!(graph.x.shape().dims(), &[4, 3]);
    assert_eq!(graph.edge_index.shape().dims(), &[2, 5]);
}

#[test]
fn test_gcn_layer() {
    let graph = create_test_graph();
    let gcn = GCNConv::new(3, 16, true);

    let output_graph = gcn.forward(&graph);

    assert_eq!(output_graph.num_nodes, graph.num_nodes);
    assert_eq!(output_graph.num_edges, graph.num_edges);
    assert_eq!(output_graph.x.shape().dims(), &[4, 16]);

    // Check that parameters are accessible
    let params = gcn.parameters();
    assert_eq!(params.len(), 2); // weight + bias
}

// #[test]
// fn test_gat_layer() {
//     let graph = create_test_graph();
//     let gat = GATConv::new(3, 8, 2, 0.1, true); // 2 heads, 8 out features per head

//     let output_graph = gat.forward(&graph);

//     assert_eq!(output_graph.num_nodes, graph.num_nodes);
//     assert_eq!(output_graph.num_edges, graph.num_edges);
//     assert_eq!(output_graph.x.shape().dims(), &[4, 16]); // 2 heads * 8 features

//     // Check parameters
//     let params = gat.parameters();
//     assert_eq!(params.len(), 3); // weight + attention + bias
// }

#[test]
fn test_sage_layer() {
    let graph = create_test_graph();
    let sage = SAGEConv::new(3, 12, true);

    let output_graph = sage.forward(&graph);

    assert_eq!(output_graph.num_nodes, graph.num_nodes);
    assert_eq!(output_graph.num_edges, graph.num_edges);
    assert_eq!(output_graph.x.shape().dims(), &[4, 12]);

    // Check parameters
    let params = sage.parameters();
    assert_eq!(params.len(), 3); // weight_neighbor + weight_self + bias
}

#[test]
fn test_gin_layer() {
    let graph = create_test_graph();
    let gin = GINConv::new(3, 10, 0.0, false, true);

    let output_graph = gin.forward(&graph);

    assert_eq!(output_graph.num_nodes, graph.num_nodes);
    assert_eq!(output_graph.num_edges, graph.num_edges);
    assert_eq!(output_graph.x.shape().dims(), &[4, 10]);

    // Check parameters (2 MLP layers + bias, no trainable eps)
    let params = gin.parameters();
    assert_eq!(params.len(), 3);
}

#[test]
fn test_graph_transformer() {
    let graph = create_test_graph();
    let transformer = GraphTransformer::new(3, 12, 3, 2, 0.1, true); // 3 heads

    let output_graph = transformer.forward(&graph);

    assert_eq!(output_graph.num_nodes, graph.num_nodes);
    assert_eq!(output_graph.num_edges, graph.num_edges);
    assert_eq!(output_graph.x.shape().dims(), &[4, 12]);

    // Check parameters
    let params = transformer.parameters();
    assert_eq!(params.len(), 6); // Q, K, V, edge, output weights + bias
}

#[test]
fn test_activation_functions() {
    let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu).unwrap();

    // Test LeakyReLU
    let leaky_output = leaky_relu(&input, 0.01);
    let leaky_vals = leaky_output.to_vec().unwrap();
    assert_relative_eq!(leaky_vals[0], -0.02, epsilon = 1e-6);
    assert_relative_eq!(leaky_vals[4], 2.0, epsilon = 1e-6);

    // Test ELU
    let elu_output = elu(&input, 1.0);
    let elu_vals = elu_output.to_vec().unwrap();
    assert_relative_eq!(elu_vals[2], 0.0, epsilon = 1e-6); // ELU(0) = 0
    assert_relative_eq!(elu_vals[4], 2.0, epsilon = 1e-6); // ELU(2) = 2

    // Test Swish
    let swish_output = swish(&input);
    let swish_vals = swish_output.to_vec().unwrap();
    assert_relative_eq!(swish_vals[2], 0.0, epsilon = 1e-6); // Swish(0) = 0

    // Test GELU
    let gelu_output = gelu(&input);
    let gelu_vals = gelu_output.to_vec().unwrap();
    assert!(gelu_vals.iter().all(|&x| x.is_finite()));

    // Test Mish
    let mish_output = mish(&input);
    let mish_vals = mish_output.to_vec().unwrap();
    assert!(mish_vals.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_global_pooling() {
    let graph = create_test_graph();

    // Test global mean pooling
    let mean_pooled = global::global_mean_pool(&graph);
    assert_eq!(mean_pooled.shape().dims(), &[3]); // Feature dimension

    // Test global max pooling
    let max_pooled = global::global_max_pool(&graph);
    assert_eq!(max_pooled.shape().dims(), &[3]);

    // Test global sum pooling
    let sum_pooled = global::global_sum_pool(&graph);
    assert_eq!(sum_pooled.shape().dims(), &[3]);

    // Verify that max >= mean >= some reasonable lower bound
    let mean_vals = mean_pooled.to_vec().unwrap();
    let max_vals = max_pooled.to_vec().unwrap();

    for i in 0..3 {
        assert!(max_vals[i] >= mean_vals[i] - 1e-6);
    }
}

#[test]
fn test_global_attention_pooling() {
    let graph = create_test_graph();
    let attention_pool = global::GlobalAttentionPool::new(3, 8);

    let pooled = attention_pool.forward(&graph);
    assert_eq!(pooled.shape().dims(), &[8]);

    // Check parameters
    let params = attention_pool.parameters();
    assert_eq!(params.len(), 2); // gate_nn + feat_nn
}

#[test]
fn test_set2set_pooling() {
    let graph = create_test_graph();
    let set2set = global::Set2Set::new(3, 16, 1, 3); // 3 iterations

    let pooled = set2set.forward(&graph);
    assert_eq!(pooled.shape().dims(), &[16]);

    // Check parameters
    let params = set2set.parameters();
    assert_eq!(params.len(), 3); // lstm_weights (1 layer) + attention_weights + projection_weights
}

#[test]
fn test_topk_pooling() {
    let graph = create_large_test_graph();
    let topk_pool = hierarchical::TopKPool::new(8, 0.5, None); // Keep 50% of nodes

    let pooled_graph = topk_pool.forward(&graph);

    assert!(pooled_graph.num_nodes <= graph.num_nodes);
    assert!(pooled_graph.num_nodes >= 1);
    assert_eq!(pooled_graph.x.shape().dims()[1], graph.x.shape().dims()[1]); // Same feature dim

    // Check parameters
    let params = topk_pool.parameters();
    assert_eq!(params.len(), 1); // score_layer
}

#[test]
fn test_diffpool() {
    let graph = create_test_graph();
    let diffpool = hierarchical::DiffPool::new(3, 2); // Pool to 2 clusters

    let (pooled_graph, aux_loss) = diffpool.forward(&graph);

    assert_eq!(pooled_graph.num_nodes, 2);
    assert_eq!(pooled_graph.x.shape().dims(), &[2, 3]);
    assert!(aux_loss.to_vec().unwrap()[0].is_finite());

    // Check parameters
    let params = diffpool.parameters();
    assert_eq!(params.len(), 2); // embed_gnn + assign_gnn
}

#[test]
fn test_mincut_pooling() {
    let graph = create_test_graph();
    let mincut_pool = hierarchical::MinCutPool::new(3, 2);

    let (pooled_graph, loss) = mincut_pool.forward(&graph);

    assert_eq!(pooled_graph.num_nodes, 2);
    assert_eq!(pooled_graph.x.shape().dims(), &[2, 3]);
    assert!(loss.to_vec().unwrap()[0].is_finite());

    // Check parameters
    let params = mincut_pool.parameters();
    assert_eq!(params.len(), 1); // assignment_layer
}

#[test]
fn test_graph_laplacian() {
    let graph = create_test_graph();

    // Test normalized Laplacian
    let normalized_laplacian = graph_laplacian(&graph.edge_index, graph.num_nodes, true);
    assert_eq!(normalized_laplacian.shape().dims(), &[4, 4]);

    // Test unnormalized Laplacian
    let unnormalized_laplacian = graph_laplacian(&graph.edge_index, graph.num_nodes, false);
    assert_eq!(unnormalized_laplacian.shape().dims(), &[4, 4]);

    // Laplacian should be symmetric (approximately)
    let lap_data = normalized_laplacian.to_vec().unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let idx_ij = i * 4 + j;
            let idx_ji = j * 4 + i;
            assert_relative_eq!(lap_data[idx_ij], lap_data[idx_ji], epsilon = 1e-6);
        }
    }
}

#[test]
fn test_graph_connectivity() {
    let graph = create_test_graph();

    // Test connectivity
    let is_connected = connectivity::is_connected(&graph.edge_index, graph.num_nodes);
    assert!(is_connected); // Our test graph should be connected

    // Test connected components
    let components = connectivity::connected_components(&graph.edge_index, graph.num_nodes);
    assert_eq!(components.len(), 1); // Should be one component if connected
    assert_eq!(components[0].len(), 4); // All nodes in one component

    // Test largest component
    let largest = connectivity::largest_component(&graph.edge_index, graph.num_nodes);
    assert_eq!(largest.len(), 4);
}

#[test]
fn test_graph_metrics() {
    let graph = create_test_graph();

    // Test centrality measures
    let centrality = metrics::node_centrality(&graph.edge_index, graph.num_nodes);
    assert_eq!(centrality.degree.shape().dims(), &[4]);
    assert_eq!(centrality.betweenness.shape().dims(), &[4]);
    assert_eq!(centrality.closeness.shape().dims(), &[4]);
    assert_eq!(centrality.eigenvector.shape().dims(), &[4]);

    // Test clustering coefficient
    let clustering = metrics::clustering_coefficient(&graph.edge_index, graph.num_nodes);
    assert_eq!(clustering.shape().dims(), &[4]);

    let clustering_vals = clustering.to_vec().unwrap();
    assert!(clustering_vals.iter().all(|&x| x >= 0.0 && x <= 1.0));

    // Test graph diameter
    let diameter = metrics::graph_diameter(&graph.edge_index, graph.num_nodes);
    assert!(diameter > 0 && diameter < graph.num_nodes);
}

#[test]
fn test_layer_chaining() {
    let graph = create_test_graph();

    // Chain multiple layers
    let gcn1 = GCNConv::new(3, 8, true);
    let gat = GATConv::new(8, 4, 2, 0.1, true); // 2 heads * 4 = 8 features
    let gcn2 = GCNConv::new(8, 5, false);

    let intermediate1 = gcn1.forward(&graph);
    let intermediate2 = gat.forward(&intermediate1);
    let final_output = gcn2.forward(&intermediate2);

    assert_eq!(final_output.x.shape().dims(), &[4, 5]);
    assert_eq!(final_output.num_nodes, graph.num_nodes);
    assert_eq!(final_output.num_edges, graph.num_edges);
}

#[test]
fn test_gradient_flow_simulation() {
    // Simulate gradient flow by checking that parameters can be accessed
    let graph = create_test_graph();
    let gcn = GCNConv::new(3, 16, true);

    let output_graph = gcn.forward(&graph);
    let params = gcn.parameters();

    // Simulate a simple gradient update
    for param in params {
        let param_data = param.to_vec().unwrap();
        assert!(param_data.iter().all(|&x| x.is_finite()));
    }

    // Check output is finite
    let output_data = output_graph.x.to_vec().unwrap();
    assert!(output_data.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_numerical_stability() {
    // Test with extreme values
    let extreme_features = vec![
        1000.0, -1000.0, 0.001, -0.001, 1e6, -1e6, 0.0, 1e-8, -1e-8, 42.0, -42.0, 3.14,
    ];
    let x = from_vec(extreme_features, &[4, 3], DeviceType::Cpu).unwrap();
    let edges = vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 0.0, 3.0];
    let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
    let extreme_graph = GraphData::new(x, edge_index);

    let gcn = GCNConv::new(3, 8, true);
    let output = gcn.forward(&extreme_graph);

    // Check that output is finite
    let output_vals = output.x.to_vec().unwrap();
    assert!(output_vals.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_empty_graph_handling() {
    // Test with minimal graph (single node, no edges)
    let x = zeros(&[1, 3]).unwrap();
    let edge_index = zeros(&[2, 0]).unwrap();
    let minimal_graph = GraphData::new(x, edge_index);

    let gcn = GCNConv::new(3, 5, true);
    let output = gcn.forward(&minimal_graph);

    assert_eq!(output.num_nodes, 1);
    assert_eq!(output.num_edges, 0);
    assert_eq!(output.x.shape().dims(), &[1, 5]);
}

#[test]
fn test_normalization_functions() {
    use torsh_graph::functional::normalization;

    let graph = create_test_graph();

    // Test layer normalization
    let layer_normed = normalization::layer_norm(&graph.x, 1e-8);
    assert_eq!(layer_normed.shape().dims(), graph.x.shape().dims());

    // Test graph normalization
    let graph_normed = normalization::graph_norm(&graph.x, &graph.edge_index, graph.num_nodes);
    assert_eq!(graph_normed.shape().dims(), graph.x.shape().dims());

    // Test batch normalization
    let batch_normed = normalization::batch_norm(&graph.x, 1e-8);
    assert_eq!(batch_normed.shape().dims(), graph.x.shape().dims());
}

#[test]
fn test_dropout_behavior() {
    use torsh_graph::functional::dropout;

    let input = randn(&[100]).unwrap();

    // Test training mode
    let dropped_training = dropout(&input, 0.5, true);
    let dropped_values = dropped_training.to_vec().unwrap();
    let num_zeros = dropped_values.iter().filter(|&&x| x == 0.0).count();

    // Should have some zeros (with high probability)
    assert!(num_zeros > 10 && num_zeros < 90); // Roughly 50% with some variance

    // Test eval mode (no dropout)
    let dropped_eval = dropout(&input, 0.5, false);
    let original_values = input.to_vec().unwrap();
    let eval_values = dropped_eval.to_vec().unwrap();

    for (orig, eval) in original_values.iter().zip(eval_values.iter()) {
        assert_relative_eq!(orig, eval, epsilon = 1e-10);
    }
}

#[test]
fn test_mpnn_layer() {
    let graph = create_test_graph();

    // Add edge attributes for MPNN
    let edge_attr = randn(&[graph.num_edges, 2]).unwrap();
    let graph_with_attrs = graph.clone().with_edge_attr(edge_attr);

    // Test different aggregation types
    let aggregation_types = [
        AggregationType::Sum,
        AggregationType::Mean,
        AggregationType::Max,
    ];

    for &agg_type in &aggregation_types {
        let mpnn = MPNNConv::new(3, 8, 2, 16, 16, agg_type, true);

        let output_graph = mpnn.forward(&graph_with_attrs);

        assert_eq!(output_graph.num_nodes, graph.num_nodes);
        assert_eq!(output_graph.num_edges, graph.num_edges);
        assert_eq!(output_graph.x.shape().dims(), &[4, 8]);

        // Check parameters
        let params = mpnn.parameters();
        assert!(params.len() >= 4); // At least message and update layers

        // Check output is finite
        let output_vals = output_graph.x.to_vec().unwrap();
        assert!(output_vals.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_mpnn_without_edge_attributes() {
    let graph = create_test_graph();
    let mpnn = MPNNConv::new(3, 6, 0, 12, 12, AggregationType::Mean, false);

    let output_graph = mpnn.forward(&graph);

    assert_eq!(output_graph.num_nodes, graph.num_nodes);
    assert_eq!(output_graph.x.shape().dims(), &[4, 6]);

    // Should work without edge attributes
    let output_vals = output_graph.x.to_vec().unwrap();
    assert!(output_vals.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_comprehensive_layer_pipeline() {
    // Create a comprehensive pipeline using all graph layers
    let graph = create_test_graph();

    // Stage 1: GCN for initial feature transformation
    let gcn = GCNConv::new(3, 16, true);
    let stage1 = gcn.forward(&graph);
    assert_eq!(stage1.x.shape().dims(), &[4, 16]);

    // Stage 2: GAT for attention-based refinement
    let gat = GATConv::new(16, 8, 2, 0.1, true); // 2 heads * 8 = 16 features
    let stage2 = gat.forward(&stage1);
    assert_eq!(stage2.x.shape().dims(), &[4, 16]);

    // Stage 3: MPNN for message passing
    let mpnn = MPNNConv::new(16, 12, 0, 24, 24, AggregationType::Mean, true);
    let stage3 = mpnn.forward(&stage2);
    assert_eq!(stage3.x.shape().dims(), &[4, 12]);

    // Stage 4: GIN for final representation
    let gin = GINConv::new(12, 8, 0.0, false, true);
    let stage4 = gin.forward(&stage3);
    assert_eq!(stage4.x.shape().dims(), &[4, 8]);

    // Stage 5: Global pooling to graph-level representation
    let graph_repr = global::global_mean_pool(&stage4);
    assert_eq!(graph_repr.shape().dims(), &[8]);

    // Verify all intermediate outputs are finite
    let stages = [&stage1, &stage2, &stage3, &stage4];
    for stage in &stages {
        let vals = stage.x.to_vec().unwrap();
        assert!(vals.iter().all(|&x| x.is_finite()));
    }

    let final_vals = graph_repr.to_vec().unwrap();
    assert!(final_vals.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_scirs2_algorithms_integration() {
    let graph = create_test_graph();

    // Test PageRank
    let pagerank_scores = algorithms::pagerank(&graph, 0.85, 50);
    assert_eq!(pagerank_scores.shape().dims(), &[4]);

    let pr_vals = pagerank_scores.to_vec().unwrap();
    let sum: f32 = pr_vals.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-3); // PageRank scores should sum to 1
    assert!(pr_vals.iter().all(|&x| x > 0.0)); // All scores should be positive

    // Test community detection
    let communities = algorithms::community_detection(&graph, 1.0);
    assert_eq!(communities.len(), 4);
    assert!(communities.iter().all(|&c| c < 4)); // Community IDs should be valid

    // Test spectral clustering
    let clusters = algorithms::spectral_clustering(&graph, 2);
    assert_eq!(clusters.len(), 4);
    assert!(clusters.iter().all(|&c| c < 2)); // Should be 2 clusters

    // Test centrality measures
    let betweenness = algorithms::betweenness_centrality(&graph);
    assert_eq!(betweenness.shape().dims(), &[4]);

    let closeness = algorithms::closeness_centrality(&graph);
    assert_eq!(closeness.shape().dims(), &[4]);

    let eigenvector = algorithms::eigenvector_centrality(&graph, 100);
    assert_eq!(eigenvector.shape().dims(), &[4]);

    // Test connectivity analysis
    let (components, is_connected) = algorithms::graph_connectivity(&graph);
    assert!(is_connected); // Our test graph should be connected
    assert_eq!(components.len(), 1); // Should be one component

    // Test graph properties
    let density = algorithms::compute_graph_density(&graph);
    assert!(density > 0.0 && density <= 1.0);

    let diameter = algorithms::compute_diameter(&graph);
    assert!(diameter.is_some());
    if let Some(d) = diameter {
        assert!(d > 0 && d < graph.num_nodes);
    }
}

#[test]
fn test_scirs2_graph_generation() {
    // Test Erdős-Rényi graph generation
    let er_graph = generation::erdos_renyi(8, 0.3);
    assert_eq!(er_graph.num_nodes, 8);
    assert_eq!(er_graph.x.shape().dims(), &[8, 16]); // Default features
                                                     // num_edges is usize, always >= 0

    // Test Barabási-Albert graph generation
    let ba_graph = generation::barabasi_albert(10, 3);
    assert_eq!(ba_graph.num_nodes, 10);
    assert_eq!(ba_graph.x.shape().dims(), &[10, 16]);
    assert!(ba_graph.num_edges > 0); // Should have edges due to preferential attachment

    // Test Watts-Strogatz graph generation
    let ws_graph = generation::watts_strogatz(12, 4, 0.2);
    assert_eq!(ws_graph.num_nodes, 12);
    assert_eq!(ws_graph.x.shape().dims(), &[12, 16]);
    assert!(ws_graph.num_edges > 0);

    // Test complete graph generation
    let complete_graph = generation::complete(5);
    assert_eq!(complete_graph.num_nodes, 5);
    assert_eq!(complete_graph.x.shape().dims(), &[5, 16]);
    // Complete graph stores bidirectional edges: n*(n-1) directed edges
    assert_eq!(complete_graph.num_edges, 20); // 5*4 = 20 directed edges
}

#[test]
fn test_scirs2_spatial_graphs() {
    // Create 2D points
    let points = from_vec(
        vec![
            0.0, 0.0, // Point 0
            1.0, 0.0, // Point 1
            0.0, 1.0, // Point 2
            1.0, 1.0, // Point 3
            0.5, 0.5, // Point 4 (center)
        ],
        &[5, 2],
        DeviceType::Cpu,
    )
    .unwrap();

    // Test k-NN graph construction
    let knn_graph = spatial::knn_graph(&points, 2);
    assert_eq!(knn_graph.num_nodes, 5);
    assert!(knn_graph.num_edges > 0);
    assert_eq!(knn_graph.x.shape().dims(), &[5, 2]); // Should preserve point coordinates

    // Test radius graph construction
    let radius_graph = spatial::radius_graph(&points, 1.5);
    assert_eq!(radius_graph.num_nodes, 5);
    assert!(radius_graph.num_edges > 0);

    // Test Delaunay triangulation (approximation)
    let delaunay_graph = spatial::delaunay_graph(&points);
    assert_eq!(delaunay_graph.num_nodes, 5);
    // num_edges is usize, always >= 0
}

#[test]
fn test_advanced_pooling_with_generated_graphs() {
    // Test pooling on a larger generated graph
    let large_graph = generation::erdos_renyi(20, 0.15);

    // Test hierarchical pooling
    let diffpool = hierarchical::DiffPool::new(16, 5); // Pool from 20 to 5 nodes
    let (pooled_graph, aux_loss) = diffpool.forward(&large_graph);

    assert_eq!(pooled_graph.num_nodes, 5);
    assert_eq!(pooled_graph.x.shape().dims(), &[5, 16]);
    assert!(aux_loss.to_vec().unwrap()[0].is_finite());

    // Test TopK pooling
    let topk_pool = hierarchical::TopKPool::new(16, 0.4, Some(0.1)); // Keep 40% with minimum score
    let topk_pooled = topk_pool.forward(&large_graph);

    assert!(topk_pooled.num_nodes <= large_graph.num_nodes);
    assert!(topk_pooled.num_nodes >= 1);
    assert_eq!(topk_pooled.x.shape().dims()[1], 16);
}

#[test]
fn test_numerical_robustness_comprehensive() {
    // Test numerical stability across the entire pipeline
    let mut graph = create_test_graph();

    // Add some numerical challenges
    let extreme_features = from_vec(
        vec![
            1e6, -1e6, 0.0, // Node 0: extreme values
            1e-8, 1e-8, 1e-8, // Node 1: very small values
            -1e-8, -1e-8, -1e-8, // Node 2: very small negative values
            42.0, -42.0, 3.14, // Node 3: normal values
        ],
        &[4, 3],
        DeviceType::Cpu,
    )
    .unwrap();

    graph.x = extreme_features;

    // Test through multiple layers
    let gcn = GCNConv::new(3, 8, true);
    let sage = SAGEConv::new(8, 6, true);
    let mpnn = MPNNConv::new(6, 4, 0, 12, 12, AggregationType::Mean, true);

    let stage1 = gcn.forward(&graph);
    let stage2 = sage.forward(&stage1);
    let stage3 = mpnn.forward(&stage2);

    // All outputs should be finite
    let stages = [&stage1, &stage2, &stage3];
    for stage in &stages {
        let vals = stage.x.to_vec().unwrap();
        assert!(vals.iter().all(|&x| x.is_finite()));
    }

    // Test pooling on extreme values
    let pooled = global::global_mean_pool(&stage3);
    let pooled_vals = pooled.to_vec().unwrap();
    assert!(pooled_vals.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_graph_level_prediction_pipeline() {
    // Simulate a complete graph-level prediction pipeline
    let graphs = vec![
        create_test_graph(),
        generation::erdos_renyi(6, 0.4),
        generation::barabasi_albert(5, 2),
        generation::complete(4),
    ];

    let mut graph_embeddings = Vec::new();

    for graph in graphs {
        // Feature extraction pipeline
        let gcn1 = GCNConv::new(
            if graph.x.shape().dims()[1] == 3 {
                3
            } else {
                16
            }, // Handle different input dims
            32,
            true,
        );
        let gat = GATConv::new(32, 8, 4, 0.1, true); // 4 heads * 8 = 32 features
        let mpnn = MPNNConv::new(32, 16, 0, 48, 48, AggregationType::Sum, true);

        let h1 = gcn1.forward(&graph);
        let h2 = gat.forward(&h1);
        let h3 = mpnn.forward(&h2);

        // Graph-level pooling
        let graph_embedding = global::global_mean_pool(&h3);
        graph_embeddings.push(graph_embedding);
    }

    // Verify all embeddings have the same dimension
    assert!(graph_embeddings
        .iter()
        .all(|emb| emb.shape().dims() == &[16]));

    // Verify all embeddings are finite
    for embedding in graph_embeddings {
        let vals = embedding.to_vec().unwrap();
        assert!(vals.iter().all(|&x| x.is_finite()));
    }
}

// =============================================================================
// Foundation Model Integration Tests
// =============================================================================

#[test]
fn test_foundation_model_creation() {
    use torsh_graph::foundation::{
        FoundationModelConfig, GraphFoundationModel, PretrainingObjective,
    };

    let config = FoundationModelConfig {
        model_dim: 64,
        num_layers: 2,
        num_heads: 4,
        ff_dim: 256,
        max_seq_length: 50,
        vocab_size: 100,
        dropout: 0.1,
        pretraining_objectives: vec![
            PretrainingObjective::MaskedNodeModeling,
            PretrainingObjective::GraphContrastive,
        ],
    };

    let model_result = GraphFoundationModel::new(config);
    assert!(model_result.is_ok());

    let model = model_result.unwrap();
    assert_eq!(model.config.model_dim, 64);
    assert_eq!(model.config.num_layers, 2);
    assert_eq!(model.pretraining_head.active_objectives.len(), 2);
}

#[test]
fn test_foundation_model_pretraining() {
    use torsh_graph::foundation::{
        FoundationModelConfig, GraphFoundationModel, PretrainingObjective,
    };

    let config = FoundationModelConfig {
        model_dim: 32,
        num_layers: 1,
        num_heads: 2,
        ff_dim: 128,
        max_seq_length: 20,
        vocab_size: 50,
        dropout: 0.0,
        pretraining_objectives: vec![
            PretrainingObjective::MaskedNodeModeling,
            PretrainingObjective::GraphContrastive,
        ],
    };

    let mut model = GraphFoundationModel::new(config).unwrap();

    // Create small graphs for pretraining with correct feature dimension
    let graphs = vec![
        create_foundation_test_graph(5, 32),
        create_foundation_test_graph(6, 32),
        create_foundation_test_graph(4, 32),
    ];

    // Run pretraining for 2 epochs
    let stats_result = model.pretrain(&graphs, 2);
    assert!(stats_result.is_ok());

    let stats = stats_result.unwrap();
    assert_eq!(stats.current_epoch, 1); // 0-indexed
    assert_eq!(stats.epoch_losses.len(), 2);
    assert!(stats.total_samples > 0);
    assert!(stats.pretraining_completed);

    // Verify all losses are finite
    for loss in &stats.epoch_losses {
        assert!(loss.is_finite());
    }
}

#[test]
fn test_foundation_model_finetuning() {
    use std::collections::HashMap;
    use torsh_graph::foundation::{
        FoundationModelConfig, GraphFoundationModel, PretrainingObjective, TaskConfig, TaskType,
    };

    let config = FoundationModelConfig {
        model_dim: 32,
        num_layers: 1,
        num_heads: 2,
        ff_dim: 128,
        max_seq_length: 20,
        vocab_size: 50,
        dropout: 0.0,
        pretraining_objectives: vec![PretrainingObjective::MaskedNodeModeling],
    };

    let mut model = GraphFoundationModel::new(config).unwrap();

    // Create labeled data for finetuning
    let train_data = vec![
        (
            create_foundation_test_graph(4, 32),
            from_vec(vec![0.0; 4], &[4], DeviceType::Cpu).unwrap(),
        ),
        (
            create_foundation_test_graph(5, 32),
            from_vec(vec![1.0; 5], &[5], DeviceType::Cpu).unwrap(),
        ),
    ];

    let val_data = vec![(
        create_foundation_test_graph(4, 32),
        from_vec(vec![0.0; 4], &[4], DeviceType::Cpu).unwrap(),
    )];

    let task_config = TaskConfig {
        task_type: TaskType::NodeClassification { num_classes: 2 },
        num_epochs: 2,
        learning_rate: 0.001,
        freeze_pretrained: false,
        task_params: HashMap::new(),
    };

    let finetune_result = model.finetune("test_task", &train_data, &val_data, task_config);
    assert!(finetune_result.is_ok());

    let stats = finetune_result.unwrap();
    assert_eq!(stats.train_losses.len(), 2);
    assert_eq!(stats.val_losses.len(), 2);
    assert_eq!(stats.val_accuracies.len(), 2);

    // Verify all metrics are finite and in valid ranges
    for loss in &stats.train_losses {
        assert!(loss.is_finite());
        assert!(*loss >= 0.0);
    }
    for acc in &stats.val_accuracies {
        assert!(acc.is_finite());
        assert!(*acc >= 0.0 && *acc <= 1.0);
    }
}

#[test]
fn test_foundation_model_tokenization() {
    use torsh_graph::foundation::GraphTokenizer;

    let tokenizer_result = GraphTokenizer::new(100);
    assert!(tokenizer_result.is_ok());

    let tokenizer = tokenizer_result.unwrap();

    // Test tokenization on a graph
    let graph = create_foundation_test_graph(5, 16);
    let tokens_result = tokenizer.tokenize(&graph);
    assert!(tokens_result.is_ok());

    let tokens = tokens_result.unwrap();
    assert!(!tokens.is_empty());

    // Verify tokens are in valid range
    for token in &tokens {
        assert!(*token < 100); // Within vocab size
    }
}

#[test]
fn test_foundation_model_task_heads() {
    use torsh_graph::foundation::{
        GraphClassificationHead, GraphRegressionHead, LinkPredictionHead, NodeClassificationHead,
        TaskHead,
    };

    // Test node classification head
    let node_head_result = NodeClassificationHead::new(64, 5);
    assert!(node_head_result.is_ok());
    let node_head = node_head_result.unwrap();
    assert_eq!(node_head.parameters().len(), 2); // classifier + bias

    // Test graph classification head
    let graph_head_result = GraphClassificationHead::new(64, 3);
    assert!(graph_head_result.is_ok());
    let graph_head = graph_head_result.unwrap();
    assert_eq!(graph_head.parameters().len(), 3); // pooling + classifier + bias

    // Test link prediction head
    let link_head_result = LinkPredictionHead::new(64);
    assert!(link_head_result.is_ok());
    let link_head = link_head_result.unwrap();
    assert_eq!(link_head.parameters().len(), 1); // edge_predictor

    // Test graph regression head
    let regression_head_result = GraphRegressionHead::new(64);
    assert!(regression_head_result.is_ok());
    let regression_head = regression_head_result.unwrap();
    assert_eq!(regression_head.parameters().len(), 2); // regressor + bias

    // Test forward passes with dummy data
    let dummy_embeddings = randn(&[5, 64]).unwrap();

    let node_output = node_head.forward(&dummy_embeddings);
    assert!(node_output.is_ok());

    let graph_output = graph_head.forward(&dummy_embeddings);
    assert!(graph_output.is_ok());

    // Link prediction expects concatenated node pairs [num_pairs, 128]
    let link_embeddings = randn(&[3, 128]).unwrap(); // 3 node pairs
    let link_output = link_head.forward(&link_embeddings);
    assert!(link_output.is_ok());

    let regression_output = regression_head.forward(&dummy_embeddings);
    assert!(regression_output.is_ok());
}

#[test]
fn test_foundation_model_pretraining_objectives() {
    use torsh_graph::foundation::PretrainingObjective;

    let objectives = vec![
        PretrainingObjective::MaskedNodeModeling,
        PretrainingObjective::MaskedEdgeModeling,
        PretrainingObjective::GraphContrastive,
        PretrainingObjective::NodeContrastive,
        PretrainingObjective::StructurePrediction,
        PretrainingObjective::MotifPrediction,
        PretrainingObjective::PropertyPrediction,
        PretrainingObjective::GraphDenoising,
    ];

    // Verify all objectives can be cloned and debugged
    for objective in &objectives {
        let cloned = objective.clone();
        let debug_str = format!("{:?}", cloned);
        assert!(!debug_str.is_empty());
    }
}

/// Helper function to create a test graph for foundation models
fn create_foundation_test_graph(num_nodes: usize, num_features: usize) -> GraphData {
    let features = randn(&[num_nodes, num_features]).unwrap();

    // Create a simple cycle graph
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        edges.push(i as f32);
        edges.push(((i + 1) % num_nodes) as f32);
    }

    let edge_index = from_vec(edges, &[2, num_nodes], DeviceType::Cpu).unwrap();
    GraphData::new(features, edge_index)
}
