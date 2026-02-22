//! Graph Foundation Model Pre-training Example
//!
//! This example demonstrates how to use the foundation model module for
//! self-supervised pre-training on graph data.

use torsh_core::device::DeviceType;
use torsh_graph::foundation::{
    FoundationModelConfig, GraphFoundationModel, PretrainingObjective, TaskConfig, TaskType,
};
use torsh_graph::GraphData;
use torsh_tensor::creation::from_vec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Graph Foundation Model Pre-training Example");
    println!("{}", "=".repeat(60));

    // Step 1: Create a simple graph dataset for pre-training
    println!("\n📊 Step 1: Creating synthetic graph dataset...");
    let graphs = create_synthetic_graphs()?;
    println!("   Created {} graphs for pre-training", graphs.len());

    // Step 2: Configure the foundation model
    println!("\n⚙️  Step 2: Configuring foundation model...");
    let config = FoundationModelConfig {
        model_dim: 128,
        num_layers: 3,
        num_heads: 4,
        ff_dim: 512,
        max_seq_length: 100,
        vocab_size: 500,
        dropout: 0.1,
        pretraining_objectives: vec![
            PretrainingObjective::MaskedNodeModeling,
            PretrainingObjective::GraphContrastive,
            PretrainingObjective::NodeContrastive,
        ],
    };
    println!("   Model dimension: {}", config.model_dim);
    println!("   Number of layers: {}", config.num_layers);
    println!(
        "   Pre-training objectives: {}",
        config.pretraining_objectives.len()
    );

    // Step 3: Initialize the foundation model
    println!("\n🔧 Step 3: Initializing foundation model...");
    let mut model = GraphFoundationModel::new(config)?;
    println!("   ✓ Model initialized successfully");

    // Step 4: Pre-train the model
    println!("\n🏋️  Step 4: Pre-training the model...");
    let num_epochs = 5;
    println!("   Running {} epochs of pre-training...", num_epochs);

    let stats = model.pretrain(&graphs, num_epochs)?;

    println!("\n📈 Pre-training Results:");
    println!("   Total samples processed: {}", stats.total_samples);
    println!("   Epochs completed: {}", stats.current_epoch + 1);
    println!("   Epoch losses:");
    for (epoch, loss) in stats.epoch_losses.iter().enumerate() {
        println!("      Epoch {}: loss = {:.4}", epoch + 1, loss);
    }

    // Step 5: Fine-tune on a downstream task
    println!("\n🎯 Step 5: Fine-tuning on node classification...");

    // Create labeled data for fine-tuning
    let (train_data, val_data) = create_labeled_data()?;

    let task_config = TaskConfig {
        task_type: TaskType::NodeClassification { num_classes: 3 },
        num_epochs: 3,
        learning_rate: 0.001,
        freeze_pretrained: false,
        task_params: std::collections::HashMap::new(),
    };

    let finetune_stats =
        model.finetune("node_classification", &train_data, &val_data, task_config)?;

    println!("\n📊 Fine-tuning Results:");
    println!(
        "   Final validation accuracy: {:.2}%",
        finetune_stats.val_accuracies.last().unwrap_or(&0.0) * 100.0
    );
    println!("   Training losses: {:?}", finetune_stats.train_losses);
    println!(
        "   Validation accuracies: {:?}",
        finetune_stats.val_accuracies
    );

    // Step 6: Summary
    println!("\n✅ Example completed successfully!");
    println!("\n📝 Summary:");
    println!("   • Pre-trained foundation model with {} objectives", 3);
    println!("   • Processed {} training samples", stats.total_samples);
    println!("   • Fine-tuned on node classification task");
    println!(
        "   • Achieved {:.1}% validation accuracy",
        finetune_stats.val_accuracies.last().unwrap_or(&0.0) * 100.0
    );

    Ok(())
}

/// Create synthetic graphs for pre-training
fn create_synthetic_graphs() -> Result<Vec<GraphData>, Box<dyn std::error::Error>> {
    let mut graphs = Vec::new();

    // Create 10 small graphs with different structures
    for i in 0..10 {
        let num_nodes = 5 + i % 3; // 5-7 nodes
        let num_features = 128; // Must match model_dim

        // Create node features
        let features: Vec<f32> = (0..num_nodes * num_features)
            .map(|j| ((j + i) as f32 * 0.1).sin())
            .collect();

        let x = from_vec(features, &[num_nodes, num_features], DeviceType::Cpu)?;

        // Create a simple graph structure (cycle + some extra edges)
        let mut edges = Vec::new();
        for node in 0..num_nodes {
            // Cycle edges
            edges.push(node as f32);
            edges.push(((node + 1) % num_nodes) as f32);

            // Add reverse edge
            edges.push(((node + 1) % num_nodes) as f32);
            edges.push(node as f32);
        }

        let edge_index = from_vec(edges, &[2, num_nodes * 2], DeviceType::Cpu)?;

        graphs.push(GraphData::new(x, edge_index));
    }

    Ok(graphs)
}

/// Create labeled data for fine-tuning
fn create_labeled_data() -> Result<
    (
        Vec<(GraphData, torsh_tensor::Tensor)>,
        Vec<(GraphData, torsh_tensor::Tensor)>,
    ),
    Box<dyn std::error::Error>,
> {
    let mut train_data = Vec::new();
    let mut val_data = Vec::new();

    // Create 6 training samples
    for i in 0..6 {
        let num_nodes = 5;
        let num_features = 128; // Must match model_dim

        let features: Vec<f32> = (0..num_nodes * num_features)
            .map(|j| ((j + i * 10) as f32 * 0.1).cos())
            .collect();

        let x = from_vec(features, &[num_nodes, num_features], DeviceType::Cpu)?;

        let edges = vec![
            0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0,
        ];
        let edge_index = from_vec(edges, &[2, 8], DeviceType::Cpu)?;

        let graph = GraphData::new(x, edge_index);

        // Create dummy labels (3 classes)
        let labels = from_vec(
            vec![(i % 3) as f32; num_nodes],
            &[num_nodes],
            DeviceType::Cpu,
        )?;

        train_data.push((graph, labels));
    }

    // Create 2 validation samples
    for i in 0..2 {
        let num_nodes = 5;
        let num_features = 128; // Must match model_dim

        let features: Vec<f32> = (0..num_nodes * num_features)
            .map(|j| ((j + i * 100) as f32 * 0.1).sin())
            .collect();

        let x = from_vec(features, &[num_nodes, num_features], DeviceType::Cpu)?;

        let edges = vec![
            0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0,
        ];
        let edge_index = from_vec(edges, &[2, 8], DeviceType::Cpu)?;

        let graph = GraphData::new(x, edge_index);

        let labels = from_vec(
            vec![(i % 3) as f32; num_nodes],
            &[num_nodes],
            DeviceType::Cpu,
        )?;

        val_data.push((graph, labels));
    }

    Ok((train_data, val_data))
}
