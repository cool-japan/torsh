//! Neural Architecture Search (NAS) operations
//!
//! This module provides tools for automated neural architecture search including:
//! - Architecture encoding and decoding
//! - DARTS (Differentiable Architecture Search) operations
//! - Architecture performance prediction
//! - Evolutionary search operations

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{
    creation::{rand, randn},
    Tensor,
};

/// Encode neural network architecture into a compact representation
///
/// Creates a tensor representation of a neural architecture that can be used
/// for architecture search, comparison, and optimization.
///
/// ## Architecture Encoding Format
///
/// The encoding combines:
/// 1. **Operation encoding**: One-hot vectors for each layer's operation
/// 2. **Connection encoding**: Adjacency matrix flattened
/// 3. **Skip connections**: Binary indicators for residual connections
///
/// ## Mathematical Representation
///
/// For an architecture with L layers and O operations:
/// ```text
/// encoding = [op₁, op₂, ..., opₗ, conn₁₁, conn₁₂, ..., connₗₗ]
/// ```
/// where opᵢ ∈ {0,1}^O is one-hot and connᵢⱼ ∈ {0,1} indicates connections.
///
/// # Arguments
/// * `operations` - Vector of operation indices for each layer
/// * `connections` - Adjacency matrix tensor [L, L]
/// * `num_ops` - Total number of possible operations
///
/// # Returns
/// Encoded architecture tensor
pub fn encode_architecture(
    operations: &[usize],
    connections: &Tensor,
    num_ops: usize,
) -> TorshResult<Tensor> {
    let num_layers = operations.len();

    // Create one-hot encoding for operations
    let mut op_encoding = Vec::with_capacity(num_layers * num_ops);
    for &op_idx in operations {
        let mut one_hot = vec![0.0f32; num_ops];
        if op_idx < num_ops {
            one_hot[op_idx] = 1.0;
        }
        op_encoding.extend(one_hot);
    }

    // Create operation encoding tensor
    let op_tensor = Tensor::from_data(
        op_encoding,
        vec![num_layers, num_ops],
        torsh_core::device::DeviceType::Cpu,
    )?;

    // Flatten connections for concatenation
    let connections_flat = connections.view(&[-1])?;

    // Concatenate operation encoding and connections
    let op_view = op_tensor.view(&[-1])?;
    let encoding = Tensor::cat(&[&op_view, &connections_flat], 0)?;

    Ok(encoding)
}

/// Decode architecture representation back to operations and connections
///
/// Reconstructs the original architecture specification from its encoded
/// representation for network instantiation.
///
/// ## Decoding Process
///
/// 1. **Split encoding**: Separate operation and connection components
/// 2. **Reshape operations**: Convert flat encoding to [L, O] matrix
/// 3. **Argmax operations**: Extract operation indices from one-hot
/// 4. **Reshape connections**: Restore adjacency matrix [L, L]
///
/// # Arguments
/// * `encoding` - Encoded architecture tensor
/// * `num_layers` - Number of layers in the architecture
/// * `num_ops` - Total number of possible operations
///
/// # Returns
/// Tuple of (operations, connections)
pub fn decode_architecture(
    encoding: &Tensor,
    num_layers: usize,
    num_ops: usize,
) -> TorshResult<(Vec<usize>, Tensor)> {
    let encoding_data = encoding.data()?;
    let ops_size = num_layers * num_ops;
    let connections_size = num_layers * num_layers;

    if encoding_data.len() != ops_size + connections_size {
        return Err(TorshError::invalid_argument_with_context(
            "Encoding size doesn't match expected dimensions",
            "decode_architecture",
        ));
    }

    // Decode operations
    let mut operations = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let start_idx = layer * num_ops;
        let end_idx = start_idx + num_ops;
        let layer_ops = &encoding_data[start_idx..end_idx];

        // Find argmax (operation with highest probability/value)
        let op_idx = layer_ops
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        operations.push(op_idx);
    }

    // Decode connections
    let connections_data = &encoding_data[ops_size..];
    let connections = Tensor::from_data(
        connections_data.to_vec(),
        vec![num_layers, num_layers],
        torsh_core::device::DeviceType::Cpu,
    )?;

    Ok((operations, connections))
}

/// DARTS (Differentiable Architecture Search) operation
///
/// Implements the continuous relaxation of architecture search as used in DARTS.
/// This allows architecture search to be performed via gradient descent.
///
/// ## Mathematical Formulation
///
/// DARTS represents each edge as a weighted combination of operations:
/// ```text
/// o(x) = Σᵢ αᵢ opᵢ(x)
/// ```
/// where αᵢ = softmax(wᵢ) are the architecture weights.
///
/// ## Continuous Relaxation
///
/// Instead of discrete operation selection, DARTS uses:
/// ```text
/// mixed_op(x) = Σₒ (exp(α_o) / Σₒ' exp(α_o')) · op_o(x)
/// ```
///
/// ## Benefits
///
/// 1. **Differentiable**: Architecture weights can be optimized via gradients
/// 2. **Efficient**: Avoids expensive discrete search
/// 3. **End-to-end**: Joint optimization of weights and architecture
/// 4. **Memory efficient**: Shares computation across operations
///
/// # Arguments
/// * `x` - Input tensor
/// * `alpha` - Architecture weights \[num_operations\]
/// * `operations` - List of operation tensors to mix
///
/// # Returns
/// Mixed operation output
pub fn darts_operation(_x: &Tensor, alpha: &Tensor, operations: &[Tensor]) -> TorshResult<Tensor> {
    if operations.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "Operations list cannot be empty",
            "darts_operation",
        ));
    }

    // Apply softmax to architecture weights
    let alpha_softmax = alpha.softmax(0)?;
    let alpha_data = alpha_softmax.data()?;

    // Initialize result with first operation
    let mut result = operations[0].mul_scalar(*alpha_data.get(0).unwrap_or(&0.0))?;

    // Add weighted contributions from remaining operations
    for (i, op_output) in operations.iter().enumerate().skip(1) {
        let weight = *alpha_data.get(i).unwrap_or(&0.0);
        let weighted_op = op_output.mul_scalar(weight)?;
        result = result.add(&weighted_op)?;
    }

    Ok(result)
}

/// Predict architecture performance without full training
///
/// Uses a performance predictor network to estimate the validation accuracy
/// of an architecture without expensive training.
///
/// ## Performance Prediction Approaches
///
/// 1. **Encoding-based**: Use architecture encoding as input to MLP
/// 2. **Graph-based**: Use graph neural networks on architecture graphs
/// 3. **Zero-shot**: Predict based on architecture statistics
/// 4. **Transfer learning**: Leverage performance from similar architectures
///
/// ## Features Used
///
/// Common features for performance prediction:
/// - Number of parameters
/// - FLOPs (floating-point operations)
/// - Depth and width statistics
/// - Skip connection patterns
/// - Operation type distributions
///
/// # Arguments
/// * `encoding` - Encoded architecture representation
/// * `predictor_weights` - Weights of the performance predictor network
///
/// # Returns
/// Predicted performance score (typically validation accuracy)
pub fn predict_architecture_performance(
    encoding: &Tensor,
    predictor_weights: &Tensor,
) -> TorshResult<Tensor> {
    // Simple linear predictor: performance = weights^T * encoding + bias
    let prediction = predictor_weights.matmul(encoding)?;

    // Apply sigmoid to get performance in [0, 1] range
    prediction.sigmoid()
}

/// Mutate architecture for evolutionary search
///
/// Applies random mutations to an architecture for evolutionary neural
/// architecture search algorithms.
///
/// ## Mutation Operations
///
/// 1. **Operation mutation**: Change operation type for random layers
/// 2. **Connection mutation**: Add/remove skip connections
/// 3. **Depth mutation**: Add/remove layers
/// 4. **Width mutation**: Change channel dimensions
///
/// ## Mutation Strategy
///
/// The mutation rate controls the probability of each type of mutation:
/// ```text
/// P(mutation) = mutation_rate
/// P(operation_change | mutation) = 0.6
/// P(connection_change | mutation) = 0.3
/// P(depth_change | mutation) = 0.1
/// ```
///
/// # Arguments
/// * `operations` - Current operation indices
/// * `connections` - Current connection matrix
/// * `mutation_rate` - Probability of mutation [0, 1]
/// * `num_ops` - Total number of possible operations
///
/// # Returns
/// Tuple of (mutated_operations, mutated_connections)
pub fn mutate_architecture(
    operations: &[usize],
    connections: &Tensor,
    mutation_rate: f32,
    num_ops: usize,
) -> TorshResult<(Vec<usize>, Tensor)> {
    let mut mutated_ops = operations.to_vec();
    let mut _mutated_connections = connections.clone();

    // Mutate operations
    for op in &mut mutated_ops {
        let mutate_data = rand(&[1])?.data()?;
        if *mutate_data.get(0).unwrap_or(&1.0) < mutation_rate {
            let new_op_data = rand(&[1])?.data()?;
            *op = (*new_op_data.get(0).unwrap_or(&0.5) * num_ops as f32) as usize % num_ops;
        }
    }

    // Mutate connections (simplified - just add noise and threshold)
    let noise = randn(connections.shape().dims())?;
    let noisy_connections = connections.add(&noise.mul_scalar(mutation_rate)?)?;
    let mutated_connections = noisy_connections.sigmoid()?; // Threshold with sigmoid

    Ok((mutated_ops, mutated_connections))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, randn};

    #[test]
    fn test_encode_decode_architecture() -> TorshResult<()> {
        let operations = vec![0, 1, 2, 1]; // 4 layers
        let connections = ones(&[4, 4])?;
        let num_ops = 3;

        // Encode
        let encoding = encode_architecture(&operations, &connections, num_ops)?;

        // Decode
        let (decoded_ops, decoded_connections) = decode_architecture(&encoding, 4, num_ops)?;

        // Check operations match
        assert_eq!(operations, decoded_ops);

        // Check connection shapes match
        assert_eq!(
            connections.shape().dims(),
            decoded_connections.shape().dims()
        );

        Ok(())
    }

    #[test]
    fn test_darts_operation() -> TorshResult<()> {
        let x = randn(&[2, 4])?;
        let alpha = randn(&[3])?;

        // Create mock operations (just different scalings of input for simplicity)
        let op1 = x.mul_scalar(1.0)?;
        let op2 = x.mul_scalar(2.0)?;
        let op3 = x.mul_scalar(0.5)?;
        let operations = vec![op1, op2, op3];

        let result = darts_operation(&x, &alpha, &operations)?;

        // Check shape is preserved
        assert_eq!(x.shape().dims(), result.shape().dims());

        Ok(())
    }

    #[test]
    fn test_architecture_mutation() -> TorshResult<()> {
        let operations = vec![0, 1, 2];
        let connections = ones(&[3, 3])?;
        let num_ops = 4;

        let (mutated_ops, mutated_connections) =
            mutate_architecture(&operations, &connections, 0.5, num_ops)?;

        // Check dimensions are preserved
        assert_eq!(operations.len(), mutated_ops.len());
        assert_eq!(
            connections.shape().dims(),
            mutated_connections.shape().dims()
        );

        // Check operations are in valid range
        for &op in &mutated_ops {
            assert!(op < num_ops);
        }

        Ok(())
    }
}
