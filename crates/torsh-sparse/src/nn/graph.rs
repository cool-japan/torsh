//! Sparse graph neural network layers
//!
//! This module provides graph neural network layers optimized for sparse graph data.
//! These layers are designed to work efficiently with sparse adjacency matrices and
//! large-scale graph datasets commonly found in social networks, knowledge graphs,
//! and molecular modeling applications.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use torsh_core::TorshError;
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Convolutional Network (GCN) layer
///
/// Implements the standard GCN operation from Kipf & Welling (2017):
/// H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
///
/// Where:
/// - A is the adjacency matrix (with optional self-loops)
/// - D is the degree matrix
/// - H is the node feature matrix
/// - W is the learnable weight matrix
/// - σ is an activation function (applied externally)
///
/// This implementation is optimized for sparse adjacency matrices and supports
/// both normalized and unnormalized graph convolutions.
#[derive(Debug, Clone)]
pub struct GraphConvolution {
    /// Weight matrix for feature transformation
    weight: Tensor,
    /// Optional bias vector
    bias: Option<Tensor>,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Whether to add self-loops to adjacency matrix
    add_self_loops: bool,
    /// Whether to normalize adjacency matrix
    normalize: bool,
}

impl GraphConvolution {
    /// Create a new graph convolution layer
    ///
    /// # Arguments
    /// * `in_features` - Number of input node features
    /// * `out_features` - Number of output node features
    /// * `use_bias` - Whether to include learnable bias
    /// * `add_self_loops` - Whether to add self-loops to adjacency matrix
    /// * `normalize` - Whether to apply symmetric normalization
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New graph convolution layer or error
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::graph::GraphConvolution;
    ///
    /// // Create GCN layer: 16 input features -> 32 output features
    /// let gcn = GraphConvolution::new(16, 32, true, true, true).unwrap();
    /// ```
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        add_self_loops: bool,
        normalize: bool,
    ) -> TorshResult<Self> {
        // Initialize weight matrix with Xavier/Glorot initialization
        let _std_dev = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = randn::<f32>(&[in_features, out_features])?;

        // Initialize bias if requested
        let bias = if use_bias {
            Some(zeros::<f32>(&[out_features])?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            add_self_loops,
            normalize,
        })
    }

    /// Forward pass through the graph convolution layer
    ///
    /// # Arguments
    /// * `node_features` - Node feature matrix (num_nodes x in_features)
    /// * `adjacency` - Sparse adjacency matrix (num_nodes x num_nodes)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Output node features (num_nodes x out_features)
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::graph::GraphConvolution;
    /// use torsh_tensor::creation::randn;
    /// use torsh_sparse::CsrTensor;
    ///
    /// let gcn = GraphConvolution::new(4, 2, false, true, true).unwrap();
    /// let features = randn::<f32>(&[10, 4]).unwrap(); // 10 nodes, 4 features each
    /// // adjacency would be a 10x10 sparse matrix
    /// // let output = gcn.forward(&features, &adjacency).unwrap();
    /// ```
    pub fn forward(&self, node_features: &Tensor, adjacency: &CsrTensor) -> TorshResult<Tensor> {
        // Validate input dimensions
        let feature_shape = node_features.shape();
        if feature_shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Node features must be 2D tensor (num_nodes x in_features)".to_string(),
            ));
        }

        let num_nodes = feature_shape.dims()[0];
        let input_features = feature_shape.dims()[1];

        if input_features != self.in_features {
            return Err(TorshError::InvalidArgument(format!(
                "Input features {} don't match layer input features {}",
                input_features, self.in_features
            )));
        }

        // Validate adjacency matrix
        let adj_shape = adjacency.shape();
        if adj_shape.dims() != [num_nodes, num_nodes] {
            return Err(TorshError::InvalidArgument(
                "Adjacency matrix must be square and match number of nodes".to_string(),
            ));
        }

        // Prepare adjacency matrix (add self-loops if requested)
        let adj_processed = if self.add_self_loops {
            self.add_self_loops_to_adjacency(adjacency)?
        } else {
            adjacency.clone()
        };

        // Normalize adjacency matrix if requested
        let adj_normalized = if self.normalize {
            self.normalize_adjacency(&adj_processed)?
        } else {
            adj_processed
        };

        // Apply linear transformation: H * W
        let transformed_features = zeros::<f32>(&[num_nodes, self.out_features])?;
        for i in 0..num_nodes {
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for k in 0..self.in_features {
                    sum += node_features.get(&[i, k])? * self.weight.get(&[k, j])?;
                }
                transformed_features.set(&[i, j], sum)?;
            }
        }

        // Apply graph convolution: A_norm * (H * W)
        let output = zeros::<f32>(&[num_nodes, self.out_features])?;
        for i in 0..num_nodes {
            let (neighbors, weights) = adj_normalized.get_row(i)?;
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for (&neighbor, &weight) in neighbors.iter().zip(weights.iter()) {
                    sum += weight * transformed_features.get(&[neighbor, j])?;
                }
                output.set(&[i, j], sum)?;
            }
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for i in 0..num_nodes {
                for j in 0..self.out_features {
                    let current = output.get(&[i, j])?;
                    output.set(&[i, j], current + bias.get(&[j])?)?;
                }
            }
        }

        Ok(output)
    }

    /// Add self-loops to adjacency matrix
    ///
    /// Self-loops allow nodes to aggregate their own features, which is often
    /// beneficial for graph learning tasks.
    fn add_self_loops_to_adjacency(&self, adjacency: &CsrTensor) -> TorshResult<CsrTensor> {
        let coo = adjacency.to_coo()?;
        let mut triplets = coo.triplets();
        let num_nodes = adjacency.shape().dims()[0];

        // Add self-loops (diagonal entries with value 1.0)
        let mut self_loop_set = std::collections::HashSet::new();
        for (row, col, _) in &triplets {
            if row == col {
                self_loop_set.insert(*row);
            }
        }

        // Add missing self-loops
        for i in 0..num_nodes {
            if !self_loop_set.contains(&i) {
                triplets.push((i, i, 1.0));
            }
        }

        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            triplets.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        let new_coo = CooTensor::new(row_indices, col_indices, values, adjacency.shape().clone())?;
        CsrTensor::from_coo(&new_coo)
    }

    /// Normalize adjacency matrix using symmetric normalization: D^(-1/2) * A * D^(-1/2)
    ///
    /// This normalization scheme helps stabilize training and ensures that the
    /// eigenvalues of the normalized Laplacian are bounded, which is important
    /// for preventing exploding gradients in deep graph networks.
    fn normalize_adjacency(&self, adjacency: &CsrTensor) -> TorshResult<CsrTensor> {
        let num_nodes = adjacency.shape().dims()[0];

        // Calculate degree for each node
        let mut degrees = vec![0.0; num_nodes];
        let coo = adjacency.to_coo()?;
        let triplets = coo.triplets();

        for (row, _col, val) in &triplets {
            degrees[*row] += val;
        }

        // Calculate D^(-1/2)
        let inv_sqrt_degrees: Vec<f32> = degrees
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Apply normalization: D^(-1/2) * A * D^(-1/2)
        let normalized_triplets: Vec<_> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let normalized_val = inv_sqrt_degrees[row] * val * inv_sqrt_degrees[col];
                (row, col, normalized_val)
            })
            .collect();

        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            normalized_triplets.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        let normalized_coo =
            CooTensor::new(row_indices, col_indices, values, adjacency.shape().clone())?;
        CsrTensor::from_coo(&normalized_coo)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = self.bias.as_ref().map_or(0, |b| b.shape().numel());
        weight_params + bias_params
    }

    /// Get input feature dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Check if self-loops are added
    pub fn adds_self_loops(&self) -> bool {
        self.add_self_loops
    }

    /// Check if adjacency matrix is normalized
    pub fn normalizes(&self) -> bool {
        self.normalize
    }

    /// Get weight matrix (for inspection/analysis)
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get bias vector (for inspection/analysis)
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

/// Graph Attention Network (GAT) layer
///
/// Implements the attention mechanism from Velickovic et al. (2018).
/// This will be a simplified version for the sparse framework.
#[derive(Debug, Clone)]
pub struct GraphAttention {
    /// Weight matrix for feature transformation
    weight: Tensor,
    /// Attention weights
    attention_weights: Tensor,
    /// Optional bias vector
    bias: Option<Tensor>,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dropout probability for attention
    dropout: f32,
}

impl GraphAttention {
    /// Create a new graph attention layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        dropout: f32,
        use_bias: bool,
    ) -> TorshResult<Self> {
        if !(0.0..=1.0).contains(&dropout) {
            return Err(TorshError::InvalidArgument(
                "Dropout must be between 0.0 and 1.0".to_string(),
            ));
        }

        if num_heads == 0 {
            return Err(TorshError::InvalidArgument(
                "Number of heads must be greater than 0".to_string(),
            ));
        }

        let weight = randn::<f32>(&[in_features, out_features * num_heads])?;
        let attention_weights = randn::<f32>(&[2 * out_features, num_heads])?;

        let bias = if use_bias {
            Some(zeros::<f32>(&[out_features * num_heads])?)
        } else {
            None
        };

        Ok(Self {
            weight,
            attention_weights,
            bias,
            in_features,
            out_features,
            num_heads,
            dropout,
        })
    }

    /// Forward pass (simplified implementation)
    pub fn forward(&self, node_features: &Tensor, _adjacency: &CsrTensor) -> TorshResult<Tensor> {
        // This is a simplified implementation - full GAT would require more complex attention computation
        let feature_shape = node_features.shape();
        if feature_shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Node features must be 2D tensor".to_string(),
            ));
        }

        let num_nodes = feature_shape.dims()[0];

        // Basic linear transformation (full attention mechanism would be more complex)
        let output = zeros::<f32>(&[num_nodes, self.out_features * self.num_heads])?;

        for i in 0..num_nodes {
            for j in 0..(self.out_features * self.num_heads) {
                let mut sum = 0.0;
                for k in 0..self.in_features {
                    sum += node_features.get(&[i, k])? * self.weight.get(&[k, j])?;
                }
                output.set(&[i, j], sum)?;
            }
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for i in 0..num_nodes {
                for j in 0..(self.out_features * self.num_heads) {
                    let current = output.get(&[i, j])?;
                    output.set(&[i, j], current + bias.get(&[j])?)?;
                }
            }
        }

        Ok(output)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features * self.num_heads;
        let attention_params = 2 * self.out_features * self.num_heads;
        let bias_params = self.bias.as_ref().map_or(0, |b| b.shape().numel());
        weight_params + attention_params + bias_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooTensor, CsrTensor};
    use torsh_core::Shape;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_graph_convolution_creation() {
        let gcn = GraphConvolution::new(16, 32, true, true, true).unwrap();
        assert_eq!(gcn.in_features(), 16);
        assert_eq!(gcn.out_features(), 32);
        assert!(gcn.adds_self_loops());
        assert!(gcn.normalizes());
        assert!(gcn.num_parameters() > 0);
    }

    #[test]
    fn test_graph_convolution_forward() {
        let gcn = GraphConvolution::new(4, 2, false, false, false).unwrap();
        let features = ones::<f32>(&[3, 4]).unwrap();

        // Create simple adjacency matrix (3x3)
        let row_indices = vec![0, 1, 2, 0, 1, 2];
        let col_indices = vec![1, 2, 0, 0, 1, 2];
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let shape = Shape::new(vec![3, 3]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape).unwrap();
        let adjacency = CsrTensor::from_coo(&coo).unwrap();

        let output = gcn.forward(&features, &adjacency).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_graph_attention_creation() {
        let gat = GraphAttention::new(8, 16, 4, 0.1, true).unwrap();
        assert!(gat.num_parameters() > 0);
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(GraphAttention::new(8, 16, 0, 0.1, true).is_err()); // zero heads
        assert!(GraphAttention::new(8, 16, 4, 1.5, true).is_err()); // invalid dropout
    }

    #[test]
    fn test_self_loop_addition() {
        let gcn = GraphConvolution::new(2, 2, false, true, false).unwrap();

        // Create adjacency without self-loops
        let row_indices = vec![0, 1];
        let col_indices = vec![1, 0];
        let values = vec![1.0, 1.0];
        let shape = Shape::new(vec![2, 2]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape).unwrap();
        let adjacency = CsrTensor::from_coo(&coo).unwrap();

        let features = ones::<f32>(&[2, 2]).unwrap();
        let _output = gcn.forward(&features, &adjacency).unwrap();
    }

    #[test]
    fn test_normalization() {
        let gcn = GraphConvolution::new(3, 3, false, false, true).unwrap();

        // Create simple adjacency matrix
        let row_indices = vec![0, 0, 1, 1, 2, 2];
        let col_indices = vec![0, 1, 0, 1, 1, 2];
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let shape = Shape::new(vec![3, 3]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape).unwrap();
        let adjacency = CsrTensor::from_coo(&coo).unwrap();

        let features = ones::<f32>(&[3, 3]).unwrap();
        let _output = gcn.forward(&features, &adjacency).unwrap();
    }

    #[test]
    fn test_dimension_validation() {
        let gcn = GraphConvolution::new(4, 2, false, false, false).unwrap();
        let wrong_features = ones::<f32>(&[3, 5]).unwrap(); // Wrong feature dim

        let row_indices = vec![0, 1, 2];
        let col_indices = vec![1, 2, 0];
        let values = vec![1.0, 1.0, 1.0];
        let shape = Shape::new(vec![3, 3]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape).unwrap();
        let adjacency = CsrTensor::from_coo(&coo).unwrap();

        assert!(gcn.forward(&wrong_features, &adjacency).is_err());
    }
}
