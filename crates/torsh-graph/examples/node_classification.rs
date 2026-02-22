//! Node Classification Example
//!
//! This example demonstrates how to use torsh-graph for node classification tasks,
//! showcasing graph creation, augmentation, and training with different GNN architectures.

use torsh_graph::{
    conv::{gat::GATConv, gcn::GCNConv, sage::SAGEConv},
    data::{
        augmentation::{add_self_loops, feature_noise, normalize_features},
        converters::from_edge_list,
    },
    GraphData, GraphLayer,
};
use torsh_tensor::creation::randn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Graph: Node Classification Example ===\n");

    // Step 1: Create a synthetic graph (Zachary's Karate Club-like structure)
    println!("1. Creating graph...");
    let graph = create_karate_club_graph()?;
    println!(
        "   Graph created: {} nodes, {} edges",
        graph.num_nodes, graph.num_edges
    );

    // Step 2: Apply graph augmentations
    println!("\n2. Applying augmentations...");
    let mut augmented_graph = graph.clone();

    // Add self-loops for better GCN performance
    add_self_loops(&mut augmented_graph)?;
    println!(
        "   Added self-loops: {} edges total",
        augmented_graph.num_edges
    );

    // Normalize features
    normalize_features(&mut augmented_graph)?;
    println!("   Normalized node features");

    // Add small noise for robustness
    feature_noise(&mut augmented_graph, 0.01)?;
    println!("   Added feature noise");

    // Step 3: Create and test GCN model
    println!("\n3. Testing GCN model...");
    test_gcn_model(&augmented_graph)?;

    // Step 4: Create and test GAT model
    println!("\n4. Testing GAT model...");
    test_gat_model(&augmented_graph)?;

    // Step 5: Create and test GraphSAGE model
    println!("\n5. Testing GraphSAGE model...");
    test_sage_model(&augmented_graph)?;

    // Step 6: Demonstrate graph utilities
    println!("\n6. Graph statistics:");
    show_graph_statistics(&augmented_graph)?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Create a small graph similar to Zachary's Karate Club
fn create_karate_club_graph() -> Result<GraphData, Box<dyn std::error::Error>> {
    let num_nodes = 34;

    // Create edge list (simplified version of karate club)
    let edges = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ];

    let mut graph = from_edge_list(&edges, num_nodes)?;

    // Create random node features (8-dimensional)
    let x = randn::<f32>(&[num_nodes, 8])?;
    graph.x = x;

    Ok(graph)
}

/// Test GCN model on the graph
fn test_gcn_model(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2-layer GCN
    let gcn1 = GCNConv::new(8, 16, true); // 8 -> 16 features
    let gcn2 = GCNConv::new(16, 4, true); // 16 -> 4 features (4 classes)

    // Forward pass
    let hidden = gcn1.forward(graph);
    println!("   GCN Layer 1 output shape: {:?}", hidden.x.shape().dims());

    let output = gcn2.forward(&hidden);
    println!("   GCN Layer 2 output shape: {:?}", output.x.shape().dims());
    println!("   ✓ GCN forward pass successful");

    // Show number of parameters
    let params1 = gcn1.parameters();
    let params2 = gcn2.parameters();
    println!(
        "   Parameters: GCN1={}, GCN2={}",
        params1.len(),
        params2.len()
    );

    Ok(())
}

/// Test GAT model on the graph
fn test_gat_model(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    // Create GAT with 4 attention heads
    let gat = GATConv::new(8, 16, 4, 0.1, true); // 8 -> 16 features, 4 heads

    // Forward pass
    let output = gat.forward(graph);
    println!("   GAT output shape: {:?}", output.x.shape().dims());
    println!("   ✓ GAT forward pass successful (4 attention heads)");

    // Show number of parameters
    let params = gat.parameters();
    println!("   Parameters: {} tensors", params.len());

    Ok(())
}

/// Test GraphSAGE model on the graph
fn test_sage_model(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    // Create GraphSAGE layer
    let sage = SAGEConv::new(8, 16, true); // 8 -> 16 features

    // Forward pass
    let output = sage.forward(graph);
    println!("   SAGE output shape: {:?}", output.x.shape().dims());
    println!("   ✓ GraphSAGE forward pass successful");

    // Show number of parameters
    let params = sage.parameters();
    println!(
        "   Parameters: {} tensors (neighbor + self weights)",
        params.len()
    );

    Ok(())
}

/// Show graph statistics
fn show_graph_statistics(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Nodes: {}", graph.num_nodes);
    println!("   Edges: {}", graph.num_edges);
    println!(
        "   Avg degree: {:.2}",
        (graph.num_edges as f32) / (graph.num_nodes as f32)
    );

    let memory_stats = graph.memory_stats();
    println!("   Memory usage:");
    println!("     Total: {} bytes", memory_stats.total_bytes);
    println!(
        "     Node features: {} bytes",
        memory_stats.node_features_bytes
    );
    println!("     Edge index: {} bytes", memory_stats.edge_index_bytes);

    Ok(())
}
