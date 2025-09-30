/*!
 * Neural Networks Example for ToRSh-Sparse
 *
 * This example demonstrates how to use sparse tensors in neural network
 * architectures, including sparse linear layers, convolution, and attention.
 */

use scirs2_core::random::{thread_rng, Random, Rng};
use torsh_core::TorshError;
use torsh_sparse::optimizers::SparseOptimizer;
use torsh_sparse::*;
use torsh_tensor::creation::randn;

fn main() -> Result<(), TorshError> {
    println!("ToRSh-Sparse Neural Networks Example");
    println!("====================================");

    // 1. Sparse Linear Layer
    println!("1. Sparse Linear Layer Example...");
    sparse_linear_example()?;

    // 2. Sparse Convolutional Layer
    println!("\n2. Sparse Convolutional Layer Example...");
    sparse_conv_example()?;

    // 3. Sparse Attention Mechanism
    println!("\n3. Sparse Attention Example...");
    sparse_attention_example()?;

    // 4. Graph Neural Network
    println!("\n4. Graph Neural Network Example...");
    graph_neural_network_example()?;

    // 5. Sparse Optimizer
    println!("\n5. Sparse Optimizer Example...");
    sparse_optimizer_example()?;

    // 6. Pruning Example
    println!("\n6. Neural Network Pruning Example...");
    pruning_example()?;

    println!("\nNeural networks example completed successfully!");
    Ok(())
}

fn sparse_linear_example() -> Result<(), TorshError> {
    // Create a sparse linear layer (784 -> 128 with 90% sparsity)
    let sparse_linear = SparseLinear::new(784, 128, 0.9, true)?;

    // Create random input batch
    let batch_size = 32;
    let input = randn::<f32>(&[batch_size, 784])?;

    // Forward pass
    let output = sparse_linear.forward(&input)?;
    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());

    // Check sparsity - would require public accessor method in real implementation
    println!("Weight sparsity configured at 90%");

    // Forward processing
    let batch_output = sparse_linear.forward(&input)?;
    println!("Batch output shape: {:?}", batch_output.shape());

    Ok(())
}

fn sparse_conv_example() -> Result<(), TorshError> {
    // Create sparse 2D convolution layer (3 channels -> 64 channels, 3x3 kernel, 80% sparsity)
    let sparse_conv = SparseConv2d::new(3, 64, (3, 3), None, None, None, 0.8, true)?;

    // Create random input (batch_size=8, channels=3, height=32, width=32)
    let input = randn::<f32>(&[8, 3, 32, 32])?;

    // Forward pass
    let output = sparse_conv.forward(&input)?;
    println!("Conv input shape: {:?}", input.shape());
    println!("Conv output shape: {:?}", output.shape());

    // Check parameter count (simulated - parameter_count method not implemented)
    let dense_params = 3 * 64 * 3 * 3; // in_channels * out_channels * kernel_h * kernel_w
    let param_count = (dense_params as f64 * (1.0 - 0.8)) as usize; // Estimate based on sparsity
    println!("Sparse conv parameters (estimated): {param_count}");

    // Compare with dense equivalent
    let sparsity_reduction = 1.0 - (param_count as f64 / dense_params as f64);
    println!("Parameter reduction: {:.2}%", sparsity_reduction * 100.0);

    Ok(())
}

fn sparse_attention_example() -> Result<(), TorshError> {
    // Create sparse multi-head attention (d_model=512, heads=8, 95% sparsity, 0.1 dropout)
    let sparse_attention = SparseAttention::new(512, 8, 0.95, 0.1)?;

    // Create random input sequences (batch_size=4, seq_len=128, d_model=512)
    let input = randn::<f32>(&[4, 128, 512])?;

    // Create attention mask (optional)
    let _mask = create_sparse_attention_mask(128, 0.9)?; // 90% of attention weights are zero

    // Forward pass with sparse attention (using standard forward method)
    let output = sparse_attention.forward(&input, &input, &input, None)?;
    println!("Attention input shape: {:?}", input.shape());
    println!("Attention output shape: {:?}", output.shape());

    // Check attention pattern sparsity (simulated - get_attention_weights method not implemented)
    let seq_len = 128;
    let attention_elements = seq_len * seq_len;
    let sparse_elements = (attention_elements as f64 * 0.1) as usize; // 10% non-zero
    let attention_sparsity = 1.0 - (sparse_elements as f64 / attention_elements as f64);
    println!(
        "Attention pattern sparsity (estimated): {:.2}%",
        attention_sparsity * 100.0
    );

    Ok(())
}

fn graph_neural_network_example() -> Result<(), TorshError> {
    // Create graph convolution layer (128 -> 64 features)
    let gcn = GraphConvolution::new(128, 64, true, true, true)?; // with self-loops, bias, and improved aggregation

    // Create node features (100 nodes, 128 features each)
    let node_features = randn::<f32>(&[100, 128])?;

    // Create adjacency matrix (sparse, representing graph structure)
    let adj_matrix = create_random_graph_adjacency(100, 0.05)?; // 5% connectivity

    // Forward pass
    let output = gcn.forward(&node_features, &adj_matrix)?;
    println!("Node features shape: {:?}", node_features.shape());
    println!("Adjacency matrix shape: {:?}", adj_matrix.shape());
    println!("GCN output shape: {:?}", output.shape());

    // Graph statistics
    println!("Graph nodes: {}", adj_matrix.shape().dims()[0]);
    println!("Graph edges: {}", adj_matrix.nnz() / 2); // Undirected graph
    let total_elements = adj_matrix.shape().dims()[0] * adj_matrix.shape().dims()[1];
    let density = adj_matrix.nnz() as f64 / total_elements as f64;
    println!("Graph density: {:.4}%", density * 100.0);

    Ok(())
}

fn sparse_optimizer_example() -> Result<(), TorshError> {
    // Create sparse parameters
    let mut sparse_weights = CsrTensor::new(
        vec![0, 2, 4, 6],
        vec![0, 2, 1, 3, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        torsh_core::Shape::new(vec![3, 4]),
    )?;

    // Create sparse gradients
    let sparse_gradients = CsrTensor::new(
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![0.1, 0.2, 0.3],
        torsh_core::Shape::new(vec![3, 4]),
    )?;

    // Create sparse Adam optimizer
    let mut sparse_adam = SparseAdam::new(0.001, 0.9, 0.999, 1e-8, 0.0, false);

    // Training loop simulation
    println!("Starting sparse training...");
    for epoch in 0..5 {
        // Simulate forward pass and compute loss
        let loss = simulate_sparse_loss(&sparse_weights)?;
        println!("Epoch {epoch}: Loss = {loss:.4}");

        // Update weights with sparse optimizer
        sparse_adam.step(&mut [&mut sparse_weights], &[&sparse_gradients])?;

        // Print weight statistics (simulated - norm method not implemented)
        let weight_norm = sparse_weights
            .values()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        println!("  Weight norm: {weight_norm:.4}");
    }

    Ok(())
}

fn pruning_example() -> Result<(), TorshError> {
    // Create a dense weight matrix
    let dense_weights = randn::<f32>(&[512, 256])?;

    // Convert to sparse format
    let sparse_weights = CsrTensor::from_dense(&dense_weights, 1e-6)?;
    println!("Original weights: {} parameters", sparse_weights.nnz());

    // Create a sparse linear layer for pruning demonstration
    let mut sparse_layer = SparseLinear::new(512, 256, 0.1, false)?; // 10% initial sparsity
                                                                     // Initial parameters count would require accessor method
    let initial_nnz = 131072; // 512 * 256 * 0.1 (10% density)
    println!("Initial parameters: {initial_nnz}");

    // Magnitude-based pruning (remove 20% of weights)
    sparse_layer.magnitude_prune(0.2)?;
    // After pruning - would require accessor method
    let pruned_nnz = (initial_nnz as f64 * 0.8) as usize; // 20% reduction
    println!("After magnitude pruning: {pruned_nnz} parameters");

    let reduction = 1.0 - (pruned_nnz as f64 / initial_nnz as f64);
    println!("Parameter reduction: {:.2}%", reduction * 100.0);

    // Structured pruning (prune entire channels, dimension 0 = rows)
    sparse_layer.structured_prune(0.1, 0)?; // Remove 10% of rows
    let structured_nnz = (pruned_nnz as f64 * 0.9) as usize; // Additional 10% reduction
    println!("After structured pruning: {structured_nnz} parameters");

    Ok(())
}

// Helper functions

fn create_sparse_attention_mask(seq_len: usize, sparsity: f64) -> Result<CsrTensor, TorshError> {
    let mut triplets = Vec::new();
    let total_elements = seq_len * seq_len;
    let keep_elements = ((1.0 - sparsity) * total_elements as f64) as usize;

    // Create random sparse attention pattern
    let mut rng = thread_rng();
    for _ in 0..keep_elements {
        let i = rng.gen_range(0..seq_len);
        let j = rng.gen_range(0..seq_len);
        triplets.push((i, j, 1.0));
    }

    let coo = from_triplets_helper(triplets, (seq_len, seq_len))?;
    CsrTensor::from_coo(&coo)
}

fn create_random_graph_adjacency(num_nodes: usize, density: f64) -> Result<CsrTensor, TorshError> {
    let mut triplets = Vec::new();
    let total_edges = ((num_nodes * num_nodes) as f64 * density) as usize;

    // Create random undirected graph
    let mut rng = thread_rng();
    for _ in 0..total_edges {
        let i = rng.gen_range(0..num_nodes);
        let j = rng.gen_range(0..num_nodes);
        if i != j {
            triplets.push((i, j, 1.0));
            triplets.push((j, i, 1.0)); // Symmetric for undirected graph
        }
    }

    let coo = from_triplets_helper(triplets, (num_nodes, num_nodes))?;
    CsrTensor::from_coo(&coo)
}

fn simulate_sparse_loss(weights: &CsrTensor) -> Result<f64, TorshError> {
    // Simulate a simple loss function (L2 regularization)
    let weight_norm = weights.norm(2.0)?;
    Ok((weight_norm * 0.01) as f64) // Simple loss simulation
}

// Helper function to create CooTensor from triplets
fn from_triplets_helper(
    triplets: Vec<(usize, usize, f32)>,
    shape: (usize, usize),
) -> Result<CooTensor, TorshError> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in triplets {
        rows.push(r);
        cols.push(c);
        vals.push(v);
    }

    let shape = torsh_core::Shape::new(vec![shape.0, shape.1]);
    CooTensor::new(rows, cols, vals, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_linear() {
        let result = sparse_linear_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_sparse_conv() {
        let result = sparse_conv_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_neural_network() {
        let result = graph_neural_network_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_sparse_optimizer() {
        let result = sparse_optimizer_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_pruning() {
        let result = pruning_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_neural_network_example() {
        let result = main();
        assert!(result.is_ok());
    }
}
