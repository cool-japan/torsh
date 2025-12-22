//! Graph Augmentation Example
//!
//! This example demonstrates various graph augmentation techniques available in torsh-graph,
//! which are useful for data augmentation, regularization, and improving model generalization.

use torsh_graph::{
    data::{augmentation::*, converters::from_edge_list},
    GraphData,
};
use torsh_tensor::creation::randn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Graph: Augmentation Techniques Example ===\n");

    // Create a simple test graph
    let graph = create_test_graph()?;
    println!(
        "Original graph: {} nodes, {} edges\n",
        graph.num_nodes, graph.num_edges
    );

    // Demonstrate each augmentation technique
    demonstrate_self_loops(&graph)?;
    demonstrate_feature_normalization(&graph)?;
    demonstrate_edge_dropout(&graph)?;
    demonstrate_node_dropout(&graph)?;
    demonstrate_feature_masking(&graph)?;
    demonstrate_feature_noise(&graph)?;
    demonstrate_random_walk_subgraph(&graph)?;
    demonstrate_isolated_node_removal()?;

    println!("\n=== All augmentation techniques demonstrated successfully! ===");
    Ok(())
}

fn create_test_graph() -> Result<GraphData, Box<dyn std::error::Error>> {
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (1, 3),
        (2, 4),
        (0, 3),
        (1, 4),
        (2, 5),
    ];

    let mut graph = from_edge_list(&edges, 6)?;
    graph.x = randn::<f32>(&[6, 4])?;

    Ok(graph)
}

fn demonstrate_self_loops(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Adding Self-Loops");
    println!("   Original edges: {}", graph.num_edges);

    let mut augmented = graph.clone();
    add_self_loops(&mut augmented)?;

    println!("   After adding self-loops: {}", augmented.num_edges);
    println!(
        "   Self-loops added: {}",
        augmented.num_edges - graph.num_edges
    );
    println!("   ✓ Self-loops improve GCN performance by including node's own features\n");

    Ok(())
}

fn demonstrate_feature_normalization(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Feature Normalization");

    let mut augmented = graph.clone();
    let original_data = augmented.x.to_vec()?;

    normalize_features(&mut augmented)?;

    let normalized_data = augmented.x.to_vec()?;

    // Calculate norms of first node
    let orig_norm: f32 = original_data[0..4]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let norm_norm: f32 = normalized_data[0..4]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    println!("   Original norm (node 0): {:.4}", orig_norm);
    println!("   Normalized norm (node 0): {:.4}", norm_norm);
    println!("   ✓ Features normalized to unit L2 norm for scale invariance\n");

    Ok(())
}

fn demonstrate_edge_dropout(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Edge Dropout (drop_rate=0.3)");
    println!("   Original edges: {}", graph.num_edges);

    let mut augmented = graph.clone();
    edge_dropout(&mut augmented, 0.3)?;

    println!("   After dropout: {}", augmented.num_edges);
    println!(
        "   Edges removed: ~{}",
        graph.num_edges - augmented.num_edges
    );
    println!("   ✓ Edge dropout prevents overfitting and improves generalization\n");

    Ok(())
}

fn demonstrate_node_dropout(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Node Dropout (drop_rate=0.3)");
    println!(
        "   Original nodes: {}, edges: {}",
        graph.num_nodes, graph.num_edges
    );

    let mut augmented = graph.clone();
    node_dropout(&mut augmented, 0.3)?;

    println!(
        "   After dropout: nodes={}, edges={}",
        augmented.num_nodes, augmented.num_edges
    );
    println!("   ✓ Node dropout creates subgraphs for mini-batch training\n");

    Ok(())
}

fn demonstrate_feature_masking(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Feature Masking (mask_rate=0.2)");

    let mut augmented = graph.clone();
    let original_data = augmented.x.to_vec()?;

    feature_masking(&mut augmented, 0.2)?;

    let masked_data = augmented.x.to_vec()?;
    let zeros_count = masked_data.iter().filter(|&&x| x == 0.0).count();
    let original_zeros = original_data.iter().filter(|&&x| x == 0.0).count();

    println!("   Original zero features: {}", original_zeros);
    println!("   After masking: {} (~20% masked)", zeros_count);
    println!("   ✓ Feature masking useful for contrastive learning and robustness\n");

    Ok(())
}

fn demonstrate_feature_noise(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Feature Noise (std=0.1)");

    let mut augmented = graph.clone();
    let original_data = augmented.x.to_vec()?;

    feature_noise(&mut augmented, 0.1)?;

    let noisy_data = augmented.x.to_vec()?;

    // Calculate average absolute difference
    let avg_diff: f32 = original_data
        .iter()
        .zip(noisy_data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / original_data.len() as f32;

    println!("   Average feature change: {:.6}", avg_diff);
    println!("   ✓ Gaussian noise improves model robustness\n");

    Ok(())
}

fn demonstrate_random_walk_subgraph(graph: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Random Walk Subgraph Sampling");
    println!(
        "   Original: {} nodes, {} edges",
        graph.num_nodes, graph.num_edges
    );

    // Perform 3 random walks of length 5
    let subgraph = random_walk_subgraph(graph, 3, 5)?;

    println!(
        "   Subgraph: {} nodes, {} edges",
        subgraph.num_nodes, subgraph.num_edges
    );
    println!("   ✓ Random walks preserve local graph structure better than random sampling\n");

    Ok(())
}

fn demonstrate_isolated_node_removal() -> Result<(), Box<dyn std::error::Error>> {
    println!("8. Removing Isolated Nodes");

    // Create graph with isolated node
    let edges = vec![(0, 1), (1, 2), (3, 4)]; // Node 5 will be isolated
    let mut graph = from_edge_list(&edges, 6)?;
    graph.x = randn::<f32>(&[6, 4])?;

    println!(
        "   Original: {} nodes (node 5 is isolated)",
        graph.num_nodes
    );

    let mapping = remove_isolated_nodes(&mut graph)?;

    println!("   After removal: {} nodes", graph.num_nodes);
    println!("   Isolated node mapping: {:?}", mapping[5]);
    println!("   ✓ Removing isolated nodes reduces memory and computation\n");

    Ok(())
}
