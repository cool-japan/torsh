//! Graph Neural Network components for ToRSh
//!
//! This module provides PyTorch-compatible graph neural network layers and operations,
//! built on top of SciRS2's graph algorithms and spectral methods.
//!
//! # Enhanced Features:
//! - GPU acceleration for graph operations
//! - Memory-efficient sparse representations
//! - Graph attention visualization
//! - Batch processing capabilities

pub mod classification;
// pub mod continuous_time; // Continuous-time graph networks (TGN, Neural ODE) - TODO: Fix API compatibility
pub mod conv;
pub mod data;
pub mod datasets;
// pub mod diffusion; // Graph diffusion models (DDPM, DDIM, discrete diffusion) - TODO: Fix API compatibility
pub mod distributed; // Distributed graph neural networks
pub mod enhanced_scirs2_integration; // Full SciRS2 algorithm suite
                                     // pub mod equivariant; // Equivariant graph neural networks (EGNN, SchNet) - TODO: Fix API compatibility
pub mod explainability;
pub mod foundation; // Graph foundation models and self-supervised learning
pub mod functional;
pub mod generative; // Graph generation models (VAE, GAN)
pub mod geometric; // Geometric graph neural networks
pub mod hypergraph;
pub mod jit;
pub mod lottery_ticket; // Graph lottery ticket hypothesis and pruning
pub mod matching; // Graph matching and similarity learning
pub mod multimodal;
pub mod neural_operators;
pub mod neuromorphic;
pub mod optimal_transport; // Graph optimal transport (Gromov-Wasserstein, Sinkhorn)
pub mod parameter;
pub mod pool;
pub mod quantum;
pub mod scirs2_integration;
pub mod spectral; // Spectral graph methods
pub mod temporal;
pub mod utils;

use torsh_tensor::Tensor;
// Enhanced SciRS2 integration for performance optimization
// use scirs2_core::gpu::{GpuContext, GpuBuffer}; // Will be used when available
// use scirs2_core::memory_efficient::MemoryMappedArray; // Will be used when available

/// Graph data structure for GNNs
#[derive(Debug, Clone)]
pub struct GraphData {
    /// Node feature matrix (num_nodes x num_features)
    pub x: Tensor,
    /// Edge index matrix (2 x num_edges)
    pub edge_index: Tensor,
    /// Edge features (optional)
    pub edge_attr: Option<Tensor>,
    /// Batch assignment vector (optional, for batched graphs)
    pub batch: Option<Tensor>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
}

impl GraphData {
    /// Create a new graph data structure
    ///
    /// # Arguments
    /// * `x` - Node feature matrix with shape `[num_nodes, num_features]`
    /// * `edge_index` - Edge connectivity with shape `[2, num_edges]`
    ///
    /// # Returns
    /// A new `GraphData` instance
    ///
    /// # Example
    /// ```
    /// use torsh_graph::GraphData;
    /// use torsh_tensor::creation::from_vec;
    /// use torsh_core::device::DeviceType;
    ///
    /// // Create a simple triangle graph
    /// let x = from_vec(vec![1.0, 2.0, 3.0], &[3, 1], DeviceType::Cpu).unwrap();
    /// let edge_index = from_vec(
    ///     vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0], // src, dst
    ///     &[2, 3],
    ///     DeviceType::Cpu
    /// ).unwrap();
    ///
    /// let graph = GraphData::new(x, edge_index);
    /// assert_eq!(graph.num_nodes, 3);
    /// assert_eq!(graph.num_edges, 3);
    /// ```
    pub fn new(x: Tensor, edge_index: Tensor) -> Self {
        let num_nodes = x.shape().dims()[0];
        let num_edges = edge_index.shape().dims()[1];

        Self {
            x,
            edge_index,
            edge_attr: None,
            batch: None,
            num_nodes,
            num_edges,
        }
    }

    /// Add edge attributes
    pub fn with_edge_attr(mut self, edge_attr: Tensor) -> Self {
        self.edge_attr = Some(edge_attr);
        self
    }

    /// Add batch assignment
    pub fn with_batch(mut self, batch: Tensor) -> Self {
        self.batch = Some(batch);
        self
    }

    /// Add edge attributes (optional chaining)
    pub fn with_edge_attr_opt(mut self, edge_attr: Option<Tensor>) -> Self {
        self.edge_attr = edge_attr;
        self
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> GraphMemoryStats {
        let x_bytes = self.x.numel() * std::mem::size_of::<f32>(); // Assuming f32
        let edge_index_bytes = self.edge_index.numel() * std::mem::size_of::<f32>(); // Changed to f32
        let edge_attr_bytes = self
            .edge_attr
            .as_ref()
            .map(|t| t.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let batch_bytes = self
            .batch
            .as_ref()
            .map(|t| t.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);

        GraphMemoryStats {
            total_bytes: x_bytes + edge_index_bytes + edge_attr_bytes + batch_bytes,
            node_features_bytes: x_bytes,
            edge_index_bytes,
            edge_attr_bytes,
            batch_bytes,
        }
    }

    /// Validate graph structure
    ///
    /// Checks that:
    /// - All edge indices refer to valid nodes
    /// - Edge attributes (if present) match the number of edges
    ///
    /// # Returns
    /// * `Ok(())` - Graph structure is valid
    /// * `Err(GraphValidationError)` - Validation failed
    ///
    /// # Example
    /// ```
    /// use torsh_graph::GraphData;
    /// use torsh_tensor::creation::from_vec;
    /// use torsh_core::device::DeviceType;
    ///
    /// let x = from_vec(vec![1.0, 2.0], &[2, 1], DeviceType::Cpu).unwrap();
    /// let edge_index = from_vec(vec![0.0, 1.0], &[2, 1], DeviceType::Cpu).unwrap();
    /// let graph = GraphData::new(x, edge_index);
    ///
    /// assert!(graph.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), GraphValidationError> {
        // Check edge indices are within node range
        if let Ok(edge_data) = self.edge_index.to_vec() {
            let max_node_id = edge_data.iter().fold(0.0f32, |a, &b| a.max(b));
            if max_node_id >= self.num_nodes as f32 {
                return Err(GraphValidationError::InvalidNodeIndex {
                    node_id: max_node_id as i64,
                    num_nodes: self.num_nodes,
                });
            }
        }

        // Validate edge attributes shape if present
        if let Some(ref edge_attr) = self.edge_attr {
            if edge_attr.shape().dims()[0] != self.num_edges {
                return Err(GraphValidationError::EdgeAttrSizeMismatch {
                    expected: self.num_edges,
                    actual: edge_attr.shape().dims()[0],
                });
            }
        }

        Ok(())
    }
}

/// Memory usage statistics for a graph
#[derive(Debug, Clone)]
pub struct GraphMemoryStats {
    pub total_bytes: usize,
    pub node_features_bytes: usize,
    pub edge_index_bytes: usize,
    pub edge_attr_bytes: usize,
    pub batch_bytes: usize,
}

/// Graph validation errors
#[derive(Debug, thiserror::Error)]
pub enum GraphValidationError {
    #[error("Invalid node index {node_id}, graph has only {num_nodes} nodes")]
    InvalidNodeIndex { node_id: i64, num_nodes: usize },

    #[error("Edge attribute size mismatch: expected {expected}, got {actual}")]
    EdgeAttrSizeMismatch { expected: usize, actual: usize },

    #[error("Tensor operation error: {0}")]
    TensorError(String),
}

/// Trait for graph neural network layers
pub trait GraphLayer: std::fmt::Debug {
    /// Forward pass through the layer
    fn forward(&self, graph: &GraphData) -> GraphData;

    /// Get layer parameters
    fn parameters(&self) -> Vec<Tensor>;
}

/// Graph attention visualization utilities
pub mod attention_viz {

    use torsh_tensor::Tensor;

    /// Attention weights for visualization
    #[derive(Debug, Clone)]
    pub struct AttentionWeights {
        pub edge_weights: Tensor,         // [num_edges]
        pub node_weights: Option<Tensor>, // [num_nodes]
        pub layer_name: String,
        pub head_index: Option<usize>,
    }

    impl AttentionWeights {
        pub fn new(edge_weights: Tensor, layer_name: String) -> Self {
            Self {
                edge_weights,
                node_weights: None,
                layer_name,
                head_index: None,
            }
        }

        pub fn with_node_weights(mut self, node_weights: Tensor) -> Self {
            self.node_weights = Some(node_weights);
            self
        }

        pub fn with_head_index(mut self, head_index: usize) -> Self {
            self.head_index = Some(head_index);
            self
        }

        /// Normalize attention weights for visualization
        pub fn normalize(&self) -> Self {
            // Simplified normalization - just return clone for now due to tensor API limitations
            Self {
                edge_weights: self.edge_weights.clone(),
                node_weights: self.node_weights.clone(),
                layer_name: self.layer_name.clone(),
                head_index: self.head_index,
            }
        }
    }
}

/// Node importance analysis utilities
pub mod importance_analysis {

    use torsh_tensor::Tensor;

    /// Node importance metrics
    #[derive(Debug, Clone)]
    pub struct NodeImportance {
        pub centrality_scores: Tensor,           // [num_nodes]
        pub gradient_norm: Option<Tensor>,       // [num_nodes]
        pub attention_sum: Option<Tensor>,       // [num_nodes]
        pub feature_attribution: Option<Tensor>, // [num_nodes, num_features]
    }

    impl NodeImportance {
        pub fn new(centrality_scores: Tensor) -> Self {
            Self {
                centrality_scores,
                gradient_norm: None,
                attention_sum: None,
                feature_attribution: None,
            }
        }

        /// Combine multiple importance metrics
        pub fn combined_importance(
            &self,
            weights: &[f32],
        ) -> Result<Tensor, Box<dyn std::error::Error>> {
            // Simplified implementation - just return weighted centrality for now
            Ok(self.centrality_scores.mul_scalar(weights[0])?)
        }
    }
}
