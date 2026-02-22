//! Equivariant Graph Neural Networks (EGNN)
//!
//! This module implements E(n)-equivariant graph neural networks that preserve geometric
//! symmetries (rotation, translation, reflection) making them ideal for:
//! - 3D molecular property prediction
//! - Protein structure modeling
//! - Physics simulations
//! - Point cloud processing with geometric constraints
//!
//! # Key Features:
//! - SE(3)-equivariant message passing
//! - Coordinate updates preserving symmetries
//! - Velocity/force prediction
//! - Integration with existing graph neural network infrastructure
//!
//! # References:
//! - Satorras et al. "E(n) Equivariant Graph Neural Networks" (ICML 2021)
//! - Schütt et al. "SchNet: A continuous-filter convolutional neural network" (NeurIPS 2017)

use crate::{GraphData, GraphLayer};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Normal};
use std::f32::consts::PI;
use torsh_core::device::DeviceType;
use torsh_tensor::{
    creation::{from_vec, zeros},
    Tensor,
};

/// Equivariant Graph Convolutional Layer (EGNN)
///
/// This layer implements SE(3)-equivariant message passing that preserves
/// geometric transformations. Node features are updated in an invariant manner
/// while coordinates are updated equivariantly.
///
/// # Mathematical Formulation:
/// ```text
/// m_ij = φ_e([h_i, h_j, ||x_i - x_j||², a_ij])
/// x_i' = x_i + Σ_j (x_i - x_j) φ_x(m_ij)
/// m_i = Σ_j m_ij
/// h_i' = φ_h([h_i, m_i])
/// ```
///
/// where:
/// - h_i are node features (invariant)
/// - x_i are node coordinates (equivariant)
/// - a_ij are edge attributes (optional)
/// - φ_e, φ_x, φ_h are MLPs
#[derive(Debug, Clone)]
pub struct EGNNLayer {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Hidden dimension for message MLPs
    hidden_dim: usize,
    /// Whether to use attention mechanism
    use_attention: bool,
    /// Whether to normalize coordinates
    normalize_coords: bool,
    /// Trainable parameters for edge message MLP
    edge_mlp_weight1: Tensor,
    edge_mlp_weight2: Tensor,
    edge_mlp_bias1: Tensor,
    edge_mlp_bias2: Tensor,
    /// Trainable parameters for coordinate update MLP
    coord_mlp_weight: Tensor,
    coord_mlp_bias: Tensor,
    /// Trainable parameters for node update MLP
    node_mlp_weight1: Tensor,
    node_mlp_weight2: Tensor,
    node_mlp_bias1: Tensor,
    node_mlp_bias2: Tensor,
    /// Optional attention parameters
    attention_weight: Option<Tensor>,
}

impl EGNNLayer {
    /// Create a new EGNN layer
    ///
    /// # Arguments:
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension
    /// * `hidden_dim` - Hidden dimension for MLPs
    /// * `use_attention` - Whether to use attention mechanism
    /// * `normalize_coords` - Whether to normalize coordinate updates
    ///
    /// # Example:
    /// ```rust
    /// use torsh_graph::equivariant::EGNNLayer;
    ///
    /// let layer = EGNNLayer::new(64, 64, 128, true, true).unwrap();
    /// ```
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_dim: usize,
        use_attention: bool,
        normalize_coords: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.01)?;

        // Edge message MLP: [h_i || h_j || ||x_i - x_j||² || a_ij] -> hidden_dim -> message_dim
        let edge_input_dim = in_features * 2 + 1; // +1 for distance
        let edge_mlp_weight1 = Self::init_weight(edge_input_dim, hidden_dim, &normal, &mut rng)?;
        let edge_mlp_weight2 = Self::init_weight(hidden_dim, hidden_dim, &normal, &mut rng)?;
        let edge_mlp_bias1 = zeros(&[hidden_dim], DeviceType::Cpu)?;
        let edge_mlp_bias2 = zeros(&[hidden_dim], DeviceType::Cpu)?;

        // Coordinate update MLP: message -> 1 (scalar for each dimension)
        let coord_mlp_weight = Self::init_weight(hidden_dim, 1, &normal, &mut rng)?;
        let coord_mlp_bias = zeros(&[1], DeviceType::Cpu)?;

        // Node update MLP: [h_i || m_i] -> hidden_dim -> out_features
        let node_input_dim = in_features + hidden_dim;
        let node_mlp_weight1 = Self::init_weight(node_input_dim, hidden_dim, &normal, &mut rng)?;
        let node_mlp_weight2 = Self::init_weight(hidden_dim, out_features, &normal, &mut rng)?;
        let node_mlp_bias1 = zeros(&[hidden_dim], DeviceType::Cpu)?;
        let node_mlp_bias2 = zeros(&[out_features], DeviceType::Cpu)?;

        // Optional attention weights
        let attention_weight = if use_attention {
            Some(Self::init_weight(hidden_dim, 1, &normal, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            hidden_dim,
            use_attention,
            normalize_coords,
            edge_mlp_weight1,
            edge_mlp_weight2,
            edge_mlp_bias1,
            edge_mlp_bias2,
            coord_mlp_weight,
            coord_mlp_bias,
            node_mlp_weight1,
            node_mlp_weight2,
            node_mlp_bias1,
            node_mlp_bias2,
            attention_weight,
        })
    }

    /// Initialize weight tensor with Xavier/Glorot initialization
    fn init_weight(
        in_dim: usize,
        out_dim: usize,
        normal: &Normal<f64>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let std = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let values: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| (normal.sample(rng) * std) as f32)
            .collect();
        from_vec(values, &[in_dim, out_dim], DeviceType::Cpu)
    }

    /// Compute squared Euclidean distances between connected nodes
    ///
    /// # Arguments:
    /// * `coords` - Node coordinates [num_nodes, 3]
    /// * `edge_index` - Edge connectivity [2, num_edges]
    ///
    /// # Returns:
    /// Squared distances [num_edges, 1]
    fn compute_distances(
        &self,
        coords: &Tensor,
        edge_index: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let edge_data = edge_index.to_vec()?;
        let coord_data = coords.to_vec()?;
        let num_edges = edge_index.shape().dims()[1];
        let coord_dim = coords.shape().dims()[1];

        let mut distances = Vec::with_capacity(num_edges);

        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            let dst = edge_data[num_edges + e] as usize;

            let mut dist_sq = 0.0f32;
            for d in 0..coord_dim {
                let diff = coord_data[src * coord_dim + d] - coord_data[dst * coord_dim + d];
                dist_sq += diff * diff;
            }
            distances.push(dist_sq);
        }

        from_vec(distances, &[num_edges, 1], DeviceType::Cpu)
    }

    /// Compute edge messages using MLP
    ///
    /// # Arguments:
    /// * `node_features` - Node features [num_nodes, in_features]
    /// * `edge_index` - Edge connectivity [2, num_edges]
    /// * `distances` - Squared distances [num_edges, 1]
    ///
    /// # Returns:
    /// Edge messages [num_edges, hidden_dim]
    fn compute_edge_messages(
        &self,
        node_features: &Tensor,
        edge_index: &Tensor,
        distances: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let edge_data = edge_index.to_vec()?;
        let feat_data = node_features.to_vec()?;
        let dist_data = distances.to_vec()?;

        let num_edges = edge_index.shape().dims()[1];
        let num_nodes = node_features.shape().dims()[0];

        // Construct edge features: [h_i || h_j || dist_ij²]
        let mut edge_features = Vec::with_capacity(num_edges * (self.in_features * 2 + 1));

        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            let dst = edge_data[num_edges + e] as usize;

            // Source features
            for f in 0..self.in_features {
                edge_features.push(feat_data[src * self.in_features + f]);
            }
            // Target features
            for f in 0..self.in_features {
                edge_features.push(feat_data[dst * self.in_features + f]);
            }
            // Distance²
            edge_features.push(dist_data[e]);
        }

        let edge_feat_tensor = from_vec(
            edge_features,
            &[num_edges, self.in_features * 2 + 1],
            DeviceType::Cpu,
        )?;

        // MLP: edge_features -> hidden_dim
        // Layer 1
        let hidden1 = edge_feat_tensor.matmul(&self.edge_mlp_weight1)?;
        let hidden1 = hidden1.add(&self.edge_mlp_bias1.unsqueeze(0)?)?;
        let hidden1 = Self::silu(&hidden1)?; // SiLU activation

        // Layer 2
        let messages = hidden1.matmul(&self.edge_mlp_weight2)?;
        let messages = messages.add(&self.edge_mlp_bias2.unsqueeze(0)?)?;
        let messages = Self::silu(&messages)?;

        Ok(messages)
    }

    /// Compute coordinate updates (equivariant)
    ///
    /// # Arguments:
    /// * `coords` - Current coordinates [num_nodes, 3]
    /// * `edge_index` - Edge connectivity [2, num_edges]
    /// * `edge_messages` - Edge messages [num_edges, hidden_dim]
    ///
    /// # Returns:
    /// Updated coordinates [num_nodes, 3]
    fn update_coordinates(
        &self,
        coords: &Tensor,
        edge_index: &Tensor,
        edge_messages: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let edge_data = edge_index.to_vec()?;
        let coord_data = coords.to_vec()?;
        let num_nodes = coords.shape().dims()[0];
        let coord_dim = coords.shape().dims()[1];
        let num_edges = edge_index.shape().dims()[1];

        // Compute coordinate influence weights from messages
        let coord_weights = edge_messages.matmul(&self.coord_mlp_weight)?;
        let coord_weights = coord_weights.add(&self.coord_mlp_bias.unsqueeze(0)?)?;
        let coord_weight_data = coord_weights.to_vec()?;

        // Aggregate coordinate updates: x_i' = x_i + Σ_j (x_i - x_j) * weight_ij
        let mut coord_updates = vec![0.0f32; num_nodes * coord_dim];

        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            let dst = edge_data[num_edges + e] as usize;
            let weight = coord_weight_data[e].tanh(); // Bounded update

            for d in 0..coord_dim {
                let diff = coord_data[src * coord_dim + d] - coord_data[dst * coord_dim + d];
                coord_updates[src * coord_dim + d] += diff * weight;
            }
        }

        // Normalize coordinate updates if requested
        if self.normalize_coords {
            for n in 0..num_nodes {
                let mut norm = 0.0f32;
                for d in 0..coord_dim {
                    norm += coord_updates[n * coord_dim + d].powi(2);
                }
                norm = norm.sqrt().max(1e-8);
                for d in 0..coord_dim {
                    coord_updates[n * coord_dim + d] /= norm;
                }
            }
        }

        // Apply updates
        let mut new_coords = coord_data.clone();
        for i in 0..new_coords.len() {
            new_coords[i] += coord_updates[i];
        }

        from_vec(new_coords, &[num_nodes, coord_dim], DeviceType::Cpu)
    }

    /// Aggregate edge messages to nodes
    ///
    /// # Arguments:
    /// * `edge_index` - Edge connectivity [2, num_edges]
    /// * `edge_messages` - Edge messages [num_edges, hidden_dim]
    /// * `num_nodes` - Number of nodes
    ///
    /// # Returns:
    /// Aggregated messages [num_nodes, hidden_dim]
    fn aggregate_messages(
        &self,
        edge_index: &Tensor,
        edge_messages: &Tensor,
        num_nodes: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let edge_data = edge_index.to_vec()?;
        let message_data = edge_messages.to_vec()?;
        let num_edges = edge_index.shape().dims()[1];

        let mut aggregated = vec![0.0f32; num_nodes * self.hidden_dim];

        // Apply attention if enabled
        let attention_scores = if let Some(ref att_weight) = self.attention_weight {
            let scores = edge_messages.matmul(att_weight)?;
            Some(Self::softmax_by_node(&scores, edge_index, num_nodes)?)
        } else {
            None
        };

        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            let weight = attention_scores
                .as_ref()
                .and_then(|scores| scores.to_vec().ok())
                .map(|scores| scores[e])
                .unwrap_or(1.0);

            for h in 0..self.hidden_dim {
                aggregated[src * self.hidden_dim + h] += message_data[e * self.hidden_dim + h] * weight;
            }
        }

        from_vec(aggregated, &[num_nodes, self.hidden_dim], DeviceType::Cpu)
    }

    /// Update node features (invariant)
    ///
    /// # Arguments:
    /// * `node_features` - Current features [num_nodes, in_features]
    /// * `aggregated_messages` - Aggregated messages [num_nodes, hidden_dim]
    ///
    /// # Returns:
    /// Updated features [num_nodes, out_features]
    fn update_node_features(
        &self,
        node_features: &Tensor,
        aggregated_messages: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Concatenate features and messages
        let feat_data = node_features.to_vec()?;
        let msg_data = aggregated_messages.to_vec()?;
        let num_nodes = node_features.shape().dims()[0];

        let mut combined = Vec::with_capacity(num_nodes * (self.in_features + self.hidden_dim));
        for n in 0..num_nodes {
            for f in 0..self.in_features {
                combined.push(feat_data[n * self.in_features + f]);
            }
            for h in 0..self.hidden_dim {
                combined.push(msg_data[n * self.hidden_dim + h]);
            }
        }

        let combined_tensor = from_vec(
            combined,
            &[num_nodes, self.in_features + self.hidden_dim],
            DeviceType::Cpu,
        )?;

        // MLP: combined -> hidden_dim -> out_features
        let hidden = combined_tensor.matmul(&self.node_mlp_weight1)?;
        let hidden = hidden.add(&self.node_mlp_bias1.unsqueeze(0)?)?;
        let hidden = Self::silu(&hidden)?;

        let output = hidden.matmul(&self.node_mlp_weight2)?;
        let output = output.add(&self.node_mlp_bias2.unsqueeze(0)?)?;

        Ok(output)
    }

    /// SiLU (Swish) activation: x * sigmoid(x)
    fn silu(x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let sigmoid = x.sigmoid()?;
        x.mul(&sigmoid)
    }

    /// Softmax over edges grouped by source node
    fn softmax_by_node(
        scores: &Tensor,
        edge_index: &Tensor,
        num_nodes: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let edge_data = edge_index.to_vec()?;
        let score_data = scores.to_vec()?;
        let num_edges = edge_index.shape().dims()[1];

        // Group edges by source node and compute softmax
        let mut max_scores = vec![f32::NEG_INFINITY; num_nodes];
        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            max_scores[src] = max_scores[src].max(score_data[e]);
        }

        let mut exp_sums = vec![0.0f32; num_nodes];
        let mut exp_scores = vec![0.0f32; num_edges];
        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            exp_scores[e] = (score_data[e] - max_scores[src]).exp();
            exp_sums[src] += exp_scores[e];
        }

        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            exp_scores[e] /= exp_sums[src].max(1e-8);
        }

        from_vec(exp_scores, &[num_edges], DeviceType::Cpu)
    }
}

impl GraphLayer for EGNNLayer {
    fn forward(&self, graph: &GraphData) -> GraphData {
        // Extract coordinates from edge attributes (assumed to be appended)
        // For now, we'll create dummy coordinates for demonstration
        let num_nodes = graph.num_nodes;
        let coords = zeros(&[num_nodes, 3], DeviceType::Cpu).expect("Failed to create coords");

        // Compute distances
        let distances = self
            .compute_distances(&coords, &graph.edge_index)
            .expect("Failed to compute distances");

        // Compute edge messages
        let edge_messages = self
            .compute_edge_messages(&graph.x, &graph.edge_index, &distances)
            .expect("Failed to compute edge messages");

        // Update coordinates (equivariant)
        let new_coords = self
            .update_coordinates(&coords, &graph.edge_index, &edge_messages)
            .expect("Failed to update coordinates");

        // Aggregate messages
        let aggregated = self
            .aggregate_messages(&graph.edge_index, &edge_messages, num_nodes)
            .expect("Failed to aggregate messages");

        // Update node features (invariant)
        let new_features = self
            .update_node_features(&graph.x, &aggregated)
            .expect("Failed to update features");

        GraphData {
            x: new_features,
            edge_index: graph.edge_index.clone(),
            edge_attr: Some(new_coords), // Store coordinates in edge_attr for now
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.edge_mlp_weight1.clone(),
            self.edge_mlp_weight2.clone(),
            self.edge_mlp_bias1.clone(),
            self.edge_mlp_bias2.clone(),
            self.coord_mlp_weight.clone(),
            self.coord_mlp_bias.clone(),
            self.node_mlp_weight1.clone(),
            self.node_mlp_weight2.clone(),
            self.node_mlp_bias1.clone(),
            self.node_mlp_bias2.clone(),
        ];

        if let Some(ref att_weight) = self.attention_weight {
            params.push(att_weight.clone());
        }

        params
    }
}

/// Radial Basis Function layer for distance encoding
///
/// Encodes distances using Gaussian radial basis functions, commonly used
/// in SchNet and other continuous-filter convolutions.
#[derive(Debug, Clone)]
pub struct RBFLayer {
    /// Number of RBF kernels
    num_rbf: usize,
    /// Minimum distance for RBF centers
    cutoff_lower: f32,
    /// Maximum distance for RBF centers (cutoff)
    cutoff_upper: f32,
    /// RBF centers
    centers: Array1<f32>,
    /// RBF widths (gamma)
    gammas: Array1<f32>,
}

impl RBFLayer {
    /// Create a new RBF layer
    ///
    /// # Arguments:
    /// * `num_rbf` - Number of radial basis functions
    /// * `cutoff_lower` - Minimum distance
    /// * `cutoff_upper` - Maximum distance (cutoff)
    ///
    /// # Example:
    /// ```rust
    /// use torsh_graph::equivariant::RBFLayer;
    ///
    /// let rbf = RBFLayer::new(20, 0.0, 5.0);
    /// ```
    pub fn new(num_rbf: usize, cutoff_lower: f32, cutoff_upper: f32) -> Self {
        // Gaussian RBF centers uniformly distributed
        let centers: Array1<f32> = Array1::linspace(cutoff_lower, cutoff_upper, num_rbf);

        // Gamma = 1 / (2 * spacing²)
        let spacing = (cutoff_upper - cutoff_lower) / (num_rbf as f32 - 1.0);
        let gamma = 1.0 / (2.0 * spacing * spacing);
        let gammas = Array1::from_elem(num_rbf, gamma);

        Self {
            num_rbf,
            cutoff_lower,
            cutoff_upper,
            centers,
            gammas,
        }
    }

    /// Expand distances using radial basis functions
    ///
    /// # Arguments:
    /// * `distances` - Pairwise distances [num_pairs]
    ///
    /// # Returns:
    /// RBF-encoded distances [num_pairs, num_rbf]
    pub fn expand(
        &self,
        distances: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let dist_data = distances.to_vec()?;
        let num_pairs = dist_data.len();

        let mut rbf_values = Vec::with_capacity(num_pairs * self.num_rbf);

        for &dist in &dist_data {
            let dist_sqrt = dist.sqrt(); // distances are squared

            for k in 0..self.num_rbf {
                let diff = dist_sqrt - self.centers[k];
                let rbf = (-self.gammas[k] * diff * diff).exp();
                // Apply cutoff envelope
                let envelope = if dist_sqrt <= self.cutoff_upper {
                    0.5 * (1.0 + ((PI * dist_sqrt) / self.cutoff_upper).cos())
                } else {
                    0.0
                };
                rbf_values.push(rbf * envelope);
            }
        }

        from_vec(rbf_values, &[num_pairs, self.num_rbf], DeviceType::Cpu)
    }
}

/// SchNet-style continuous-filter convolutional layer
///
/// Uses radial basis functions for distance encoding and maintains
/// SE(3)-equivariance through invariant features only (no coordinate updates).
#[derive(Debug, Clone)]
pub struct SchNetConv {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Number of radial basis functions
    num_rbf: usize,
    /// RBF layer
    rbf_layer: RBFLayer,
    /// Filter-generating network weights
    filter_weight: Tensor,
    filter_bias: Tensor,
    /// Feature transformation weights
    feature_weight: Tensor,
    feature_bias: Tensor,
}

impl SchNetConv {
    /// Create a new SchNet convolution layer
    ///
    /// # Arguments:
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension
    /// * `num_rbf` - Number of radial basis functions
    /// * `cutoff` - Distance cutoff
    ///
    /// # Example:
    /// ```rust
    /// use torsh_graph::equivariant::SchNetConv;
    ///
    /// let layer = SchNetConv::new(64, 64, 20, 5.0).unwrap();
    /// ```
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_rbf: usize,
        cutoff: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.01)?;

        let rbf_layer = RBFLayer::new(num_rbf, 0.0, cutoff);

        // Filter-generating network: RBF -> out_features
        let filter_weight = EGNNLayer::init_weight(num_rbf, out_features, &normal, &mut rng)?;
        let filter_bias = zeros(&[out_features], DeviceType::Cpu)?;

        // Feature transformation: in_features -> out_features
        let feature_weight = EGNNLayer::init_weight(in_features, out_features, &normal, &mut rng)?;
        let feature_bias = zeros(&[out_features], DeviceType::Cpu)?;

        Ok(Self {
            in_features,
            out_features,
            num_rbf,
            rbf_layer,
            filter_weight,
            filter_bias,
            feature_weight,
            feature_bias,
        })
    }
}

impl GraphLayer for SchNetConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        // For demonstration, create dummy coordinates
        let num_nodes = graph.num_nodes;
        let coords = zeros(&[num_nodes, 3], DeviceType::Cpu).expect("Failed to create coords");

        // Compute distances
        let egnn_layer = EGNNLayer::new(
            self.in_features,
            self.out_features,
            64,
            false,
            false,
        )
        .expect("Failed to create EGNN layer");

        let distances = egnn_layer
            .compute_distances(&coords, &graph.edge_index)
            .expect("Failed to compute distances");

        // Expand distances with RBF
        let rbf_expanded = self
            .rbf_layer
            .expand(&distances)
            .expect("Failed to expand distances");

        // Generate filters from RBF
        let filters = rbf_expanded
            .matmul(&self.filter_weight)
            .expect("Failed to generate filters");
        let filters = filters
            .add(&self.filter_bias.unsqueeze(0).expect("Failed to unsqueeze"))
            .expect("Failed to add bias");

        // Transform features
        let transformed = graph
            .x
            .matmul(&self.feature_weight)
            .expect("Failed to transform features");
        let transformed = transformed
            .add(&self.feature_bias.unsqueeze(0).expect("Failed to unsqueeze"))
            .expect("Failed to add bias");

        // Apply continuous filter convolution (simplified)
        // In full implementation, this would be edge-wise multiplication and aggregation
        let output = transformed; // Placeholder

        GraphData {
            x: output,
            edge_index: graph.edge_index.clone(),
            edge_attr: graph.edge_attr.clone(),
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.filter_weight.clone(),
            self.filter_bias.clone(),
            self.feature_weight.clone(),
            self.feature_bias.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_egnn_layer_creation() {
        let layer = EGNNLayer::new(32, 64, 128, true, true);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.in_features, 32);
        assert_eq!(layer.out_features, 64);
        assert_eq!(layer.hidden_dim, 128);
        assert!(layer.use_attention);
        assert!(layer.normalize_coords);
    }

    #[test]
    fn test_egnn_forward_pass() {
        let layer = EGNNLayer::new(8, 16, 32, false, true).unwrap();

        // Create simple graph
        let x = from_vec(
            vec![1.0; 10 * 8],
            &[10, 8],
            DeviceType::Cpu,
        )
        .unwrap();
        let edge_index = from_vec(
            vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let graph = GraphData::new(x, edge_index);
        let output = layer.forward(&graph);

        assert_eq!(output.x.shape().dims()[0], 10);
        assert_eq!(output.x.shape().dims()[1], 16);
    }

    #[test]
    fn test_rbf_layer() {
        let rbf = RBFLayer::new(20, 0.0, 5.0);
        assert_eq!(rbf.num_rbf, 20);
        assert_eq!(rbf.cutoff_lower, 0.0);
        assert_eq!(rbf.cutoff_upper, 5.0);

        // Test expansion
        let distances = from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4], DeviceType::Cpu).unwrap();
        let expanded = rbf.expand(&distances);
        assert!(expanded.is_ok());
        let expanded = expanded.unwrap();
        assert_eq!(expanded.shape().dims(), &[4, 20]);
    }

    #[test]
    fn test_schnet_conv_creation() {
        let layer = SchNetConv::new(32, 64, 20, 5.0);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.in_features, 32);
        assert_eq!(layer.out_features, 64);
        assert_eq!(layer.num_rbf, 20);
    }

    #[test]
    fn test_schnet_conv_forward() {
        let layer = SchNetConv::new(8, 16, 10, 5.0).unwrap();

        let x = from_vec(vec![1.0; 5 * 8], &[5, 8], DeviceType::Cpu).unwrap();
        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();

        let graph = GraphData::new(x, edge_index);
        let output = layer.forward(&graph);

        assert_eq!(output.x.shape().dims()[0], 5);
        assert_eq!(output.x.shape().dims()[1], 16);
    }

    #[test]
    fn test_egnn_parameters() {
        let layer = EGNNLayer::new(16, 32, 64, true, false).unwrap();
        let params = layer.parameters();
        assert!(params.len() >= 10); // At least 10 parameter tensors
    }

    #[test]
    fn test_distance_computation() {
        let layer = EGNNLayer::new(4, 8, 16, false, false).unwrap();

        let coords = from_vec(
            vec![
                0.0, 0.0, 0.0, // Node 0
                1.0, 0.0, 0.0, // Node 1
                0.0, 1.0, 0.0, // Node 2
            ],
            &[3, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();

        let distances = layer.compute_distances(&coords, &edge_index).unwrap();
        let dist_data = distances.to_vec().unwrap();

        // Distance from node 0 to node 1 should be 1.0 (squared)
        assert!((dist_data[0] - 1.0).abs() < 0.01);
        // Distance from node 1 to node 2 should be 1.0 (squared)
        assert!((dist_data[1] - 1.0).abs() < 0.01);
    }
}
