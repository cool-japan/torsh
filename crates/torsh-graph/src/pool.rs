//! Graph pooling layers

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::parameter::Parameter;
use crate::GraphData;
use torsh_tensor::{
    creation::{from_vec, randn, zeros},
    Tensor,
};

/// Global pooling operations for graphs
pub mod global {
    use super::*;

    /// Global mean pooling
    pub fn global_mean_pool(graph: &GraphData) -> Tensor {
        // Average node features across the graph
        graph.x.mean(Some(&[0]), false).unwrap()
    }

    /// Global max pooling
    pub fn global_max_pool(graph: &GraphData) -> Tensor {
        // Max node features across the graph - simplified using max without indices
        graph.x.max(Some(0), false).unwrap()
    }

    /// Global sum pooling
    pub fn global_sum_pool(graph: &GraphData) -> Tensor {
        // Sum node features across the graph (along node dimension)
        graph.x.sum_dim(&[0], false).unwrap()
    }

    /// Global attention pooling
    pub struct GlobalAttentionPool {
        gate_nn: Parameter,
        feat_nn: Parameter,
    }

    impl GlobalAttentionPool {
        /// Create a new global attention pooling layer
        pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
            let gate_nn = Parameter::new(randn(&[input_dim, hidden_dim]).unwrap());
            let feat_nn = Parameter::new(randn(&[input_dim, hidden_dim]).unwrap());

            Self { gate_nn, feat_nn }
        }

        /// Apply attention-based global pooling
        pub fn forward(&self, graph: &GraphData) -> Tensor {
            // Compute gate and feature transformations
            let gate = graph
                .x
                .matmul(&self.gate_nn.clone_data())
                .unwrap()
                .sigmoid()
                .unwrap();
            let feat = graph.x.matmul(&self.feat_nn.clone_data()).unwrap();

            // Apply attention weights
            let weighted_features = feat.mul(&gate).unwrap();

            // Sum over nodes (axis 0), preserving feature dimension
            weighted_features.sum_dim(&[0], false).unwrap()
        }

        /// Get parameters
        pub fn parameters(&self) -> Vec<Tensor> {
            vec![self.gate_nn.clone_data(), self.feat_nn.clone_data()]
        }
    }

    /// Set2Set pooling for variable-sized graphs
    pub struct Set2Set {
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_iters: usize,
        lstm_weights: Vec<Parameter>,
        attention_weights: Parameter,
        projection_weights: Parameter,
    }

    impl Set2Set {
        /// Create a new Set2Set pooling layer
        pub fn new(
            input_dim: usize,
            hidden_dim: usize,
            num_layers: usize,
            num_iters: usize,
        ) -> Self {
            // Simple LSTM-like weights (simplified implementation)
            let mut lstm_weights = Vec::new();
            for _ in 0..num_layers {
                lstm_weights.push(Parameter::new(
                    randn(&[hidden_dim * 4, hidden_dim + input_dim]).unwrap(),
                ));
            }

            let attention_weights = Parameter::new(randn(&[hidden_dim, input_dim]).unwrap());
            let projection_weights = Parameter::new(randn(&[input_dim, hidden_dim]).unwrap());

            Self {
                input_dim,
                hidden_dim,
                num_layers,
                num_iters,
                lstm_weights,
                attention_weights,
                projection_weights,
            }
        }

        /// Apply Set2Set pooling
        pub fn forward(&self, graph: &GraphData) -> Tensor {
            let _num_nodes = graph.num_nodes;
            let mut query = zeros(&[1, self.hidden_dim]).unwrap();

            // Simplified Set2Set implementation
            for _ in 0..self.num_iters {
                // Compute attention scores
                let scores = query
                    .matmul(&self.attention_weights.clone_data())
                    .unwrap()
                    .matmul(&graph.x.t().unwrap())
                    .unwrap()
                    .softmax(-1)
                    .unwrap();

                // Weighted sum of node features
                let attended = scores.matmul(&graph.x).unwrap();

                // Project attended features to hidden dimension
                let projected_attended = attended
                    .matmul(&self.projection_weights.clone_data())
                    .unwrap();

                // Update query (simplified LSTM step)
                query = query.add(&projected_attended).unwrap();
            }

            query.squeeze(0).unwrap()
        }

        /// Get parameters
        pub fn parameters(&self) -> Vec<Tensor> {
            let mut params: Vec<Tensor> =
                self.lstm_weights.iter().map(|p| p.clone_data()).collect();
            params.push(self.attention_weights.clone_data());
            params.push(self.projection_weights.clone_data());
            params
        }
    }
}

/// Hierarchical pooling layers
pub mod hierarchical {
    use super::*;

    /// DiffPool: Differentiable graph pooling
    pub struct DiffPool {
        embed_dim: usize,
        assign_dim: usize,
        embed_gnn: Parameter,
        assign_gnn: Parameter,
        link_pred_loss_weight: f64,
        entropy_loss_weight: f64,
    }

    impl DiffPool {
        /// Create a new DiffPool layer
        pub fn new(embed_dim: usize, assign_dim: usize) -> Self {
            let embed_gnn = Parameter::new(randn(&[embed_dim, embed_dim]).unwrap());
            let assign_gnn = Parameter::new(randn(&[embed_dim, assign_dim]).unwrap());

            Self {
                embed_dim,
                assign_dim,
                embed_gnn,
                assign_gnn,
                link_pred_loss_weight: 1.0,
                entropy_loss_weight: 1.0,
            }
        }

        /// Apply differentiable pooling
        pub fn forward(&self, graph: &GraphData) -> (GraphData, Tensor) {
            let num_nodes = graph.num_nodes;

            // Generate node embeddings
            let node_embeddings = graph.x.matmul(&self.embed_gnn.clone_data()).unwrap();

            // Generate assignment matrix (soft clustering)
            let assignment_logits = graph.x.matmul(&self.assign_gnn.clone_data()).unwrap();
            let assignment_matrix = assignment_logits.softmax(-1).unwrap();

            // Pool node features using assignment matrix
            let pooled_features = assignment_matrix
                .t()
                .unwrap()
                .matmul(&node_embeddings)
                .unwrap();

            // Create new adjacency matrix
            let adjacency = self.compute_adjacency_matrix(&graph.edge_index, num_nodes);
            let pooled_adj = assignment_matrix
                .t()
                .unwrap()
                .matmul(&adjacency)
                .unwrap()
                .matmul(&assignment_matrix)
                .unwrap();

            // Extract edges from pooled adjacency matrix
            let (new_edge_index, _) = self.adjacency_to_edge_index(&pooled_adj);

            // Compute auxiliary losses for training
            let link_pred_loss = self.compute_link_prediction_loss(&adjacency, &assignment_matrix);
            let entropy_loss = self.compute_entropy_loss(&assignment_matrix);
            let total_aux_loss = link_pred_loss
                .mul_scalar(self.link_pred_loss_weight as f32)
                .unwrap()
                .add(
                    &entropy_loss
                        .mul_scalar(self.entropy_loss_weight as f32)
                        .unwrap(),
                )
                .unwrap();

            let pooled_graph = GraphData {
                x: pooled_features,
                edge_index: new_edge_index,
                edge_attr: None,
                batch: None,
                num_nodes: self.assign_dim,
                num_edges: 0, // Will be computed from edge_index
            };

            (pooled_graph, total_aux_loss)
        }

        /// Compute adjacency matrix from edge index
        fn compute_adjacency_matrix(&self, edge_index: &Tensor, num_nodes: usize) -> Tensor {
            let mut adjacency = zeros(&[num_nodes, num_nodes]).unwrap();
            let edge_data = edge_index.to_vec().unwrap();
            let edge_list: Vec<Vec<i64>> = vec![
                edge_data[0..edge_data.len() / 2]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                edge_data[edge_data.len() / 2..]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
            ];

            for j in 0..edge_list[0].len() {
                let src = edge_list[0][j] as usize;
                let dst = edge_list[1][j] as usize;
                if src < num_nodes && dst < num_nodes {
                    // Simplified adjacency matrix setting - use direct indexing approach
                    let mut adj_data = adjacency.to_vec().unwrap();
                    adj_data[src * num_nodes + dst] = 1.0;
                    adjacency = torsh_tensor::creation::from_vec(
                        adj_data,
                        &[num_nodes, num_nodes],
                        torsh_core::device::DeviceType::Cpu,
                    )
                    .unwrap();
                }
            }

            adjacency
        }

        /// Convert adjacency matrix to edge index
        fn adjacency_to_edge_index(&self, adjacency: &Tensor) -> (Tensor, usize) {
            let adj_data = adjacency.to_vec().unwrap();
            let mut edges = Vec::new();

            // Convert flattened vector to 2D indexing using tensor shape
            let shape = adjacency.shape();
            let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    if idx < adj_data.len() && adj_data[idx] > 0.5 {
                        // Threshold for edge existence
                        edges.push([i as f32, j as f32]);
                    }
                }
            }

            if edges.is_empty() {
                (zeros(&[2, 0]).unwrap(), 0)
            } else {
                let num_edges = edges.len();
                let mut edge_vec = Vec::with_capacity(2 * num_edges);

                for edge in &edges {
                    edge_vec.push(edge[0]);
                }
                for edge in &edges {
                    edge_vec.push(edge[1]);
                }

                (
                    from_vec(
                        edge_vec.iter().map(|&x| x as f32).collect(),
                        &[2, num_edges],
                        torsh_core::device::DeviceType::Cpu,
                    )
                    .unwrap(),
                    num_edges,
                )
            }
        }

        /// Compute link prediction auxiliary loss
        fn compute_link_prediction_loss(&self, adjacency: &Tensor, assignment: &Tensor) -> Tensor {
            // Predict adjacency matrix from assignment
            let predicted_adj = assignment.matmul(&assignment.t().unwrap()).unwrap();

            // Compute binary cross-entropy loss
            let eps = 1e-8;
            let eps_tensor = torsh_tensor::creation::ones_like(adjacency)
                .unwrap()
                .mul_scalar(eps as f32)
                .unwrap();
            let one_tensor = torsh_tensor::creation::ones_like(adjacency).unwrap();
            let pos_loss = adjacency
                .mul(&predicted_adj.add(&eps_tensor).unwrap().ln().unwrap())
                .unwrap();
            let neg_loss = one_tensor
                .sub(adjacency)
                .unwrap()
                .mul(
                    &one_tensor
                        .sub(&predicted_adj)
                        .unwrap()
                        .add(&eps_tensor)
                        .unwrap()
                        .ln()
                        .unwrap(),
                )
                .unwrap();

            pos_loss
                .add(&neg_loss)
                .unwrap()
                .mean(None, false)
                .unwrap()
                .neg()
                .unwrap()
        }

        /// Compute entropy auxiliary loss to encourage discrete assignments
        fn compute_entropy_loss(&self, assignment: &Tensor) -> Tensor {
            let eps = 1e-8;
            let eps_tensor = torsh_tensor::creation::ones_like(assignment)
                .unwrap()
                .mul_scalar(eps as f32)
                .unwrap();
            let entropy = assignment
                .mul(&assignment.add(&eps_tensor).unwrap().ln().unwrap())
                .unwrap()
                .sum()
                .unwrap()
                .mean(None, false)
                .unwrap()
                .neg()
                .unwrap();
            entropy
        }

        /// Get parameters
        pub fn parameters(&self) -> Vec<Tensor> {
            vec![self.embed_gnn.clone_data(), self.assign_gnn.clone_data()]
        }
    }

    /// TopK pooling
    pub struct TopKPool {
        ratio: f32,
        min_score: Option<f32>,
        score_layer: Parameter,
    }

    impl TopKPool {
        /// Create a new TopK pooling layer
        pub fn new(input_dim: usize, ratio: f32, min_score: Option<f32>) -> Self {
            let score_layer = Parameter::new(randn(&[input_dim, 1]).unwrap());

            Self {
                ratio,
                min_score,
                score_layer,
            }
        }

        /// Apply TopK pooling
        pub fn forward(&self, graph: &GraphData) -> GraphData {
            let num_nodes = graph.num_nodes;
            let k = (num_nodes as f32 * self.ratio).ceil() as usize;

            // Compute node importance scores
            let scores = graph
                .x
                .matmul(&self.score_layer.clone_data())
                .unwrap()
                .squeeze(-1)
                .unwrap();

            // Get top-k node indices
            let (top_scores, top_indices) = self.topk(&scores, k);

            // Filter nodes based on minimum score if specified
            let (selected_indices, _selected_scores) = if let Some(min_score) = self.min_score {
                let valid_mask = top_scores.gt_scalar(min_score).unwrap();
                // Convert boolean mask to f32 for compatibility
                let mask_data = valid_mask.to_vec().unwrap();
                let mask_f32 = mask_data
                    .iter()
                    .map(|&x| if x { 1.0 } else { 0.0 })
                    .collect();
                let mask_tensor = from_vec(
                    mask_f32,
                    valid_mask.shape().dims(),
                    torsh_core::device::DeviceType::Cpu,
                )
                .unwrap();
                let valid_indices = self.masked_select(&top_indices, &mask_tensor);
                let valid_scores = self.masked_select(&top_scores, &mask_tensor);
                (valid_indices, valid_scores)
            } else {
                (top_indices, top_scores)
            };

            // Extract features for selected nodes
            let selected_features = self.index_select(&graph.x, &selected_indices, 0);

            // Filter edges to only include those between selected nodes
            let (new_edge_index, new_num_edges) =
                self.filter_edges(&graph.edge_index, &selected_indices);

            GraphData {
                x: selected_features,
                edge_index: new_edge_index,
                edge_attr: graph.edge_attr.clone(), // Could be filtered similarly
                batch: None,                        // Batch information would need to be updated
                num_nodes: selected_indices.shape().dims()[0],
                num_edges: new_num_edges,
            }
        }

        /// Compute top-k indices and values
        fn topk(&self, tensor: &Tensor, k: usize) -> (Tensor, Tensor) {
            let values = tensor.to_vec().unwrap();
            let mut indexed_values: Vec<(f32, usize)> = values
                .into_iter()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();

            // Sort by value in descending order
            indexed_values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Take top k
            indexed_values.truncate(k);

            let top_values: Vec<f32> = indexed_values.iter().map(|(v, _)| *v).collect();
            let top_indices: Vec<f32> = indexed_values.iter().map(|(_, i)| *i as f32).collect();

            let values_tensor =
                from_vec(top_values, &[k], torsh_core::device::DeviceType::Cpu).unwrap();
            let indices_tensor =
                from_vec(top_indices, &[k], torsh_core::device::DeviceType::Cpu).unwrap();

            (values_tensor, indices_tensor)
        }

        /// Select elements based on a boolean mask
        fn masked_select(&self, tensor: &Tensor, mask: &Tensor) -> Tensor {
            let values = tensor.to_vec().unwrap();
            let mask_values = mask.to_vec().unwrap();

            let selected: Vec<f32> = values
                .into_iter()
                .zip(mask_values.into_iter())
                .filter_map(|(v, m)| if m > 0.5 { Some(v) } else { None })
                .collect();

            let selected_len = selected.len();
            from_vec(
                selected,
                &[selected_len],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap()
        }

        /// Select rows/columns from a tensor based on indices
        fn index_select(&self, tensor: &Tensor, indices: &Tensor, dim: i64) -> Tensor {
            let idx_values = indices.to_vec().unwrap();

            if dim == 0 {
                // Select rows
                let tensor_data = tensor.to_vec().unwrap();
                let shape = tensor.shape();
                let cols = shape.dims()[1];
                let original_data: Vec<Vec<f32>> = tensor_data
                    .chunks(cols)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                let mut selected_rows = Vec::new();

                for &idx in &idx_values {
                    let idx_usize = idx as usize;
                    if idx_usize < original_data.len() {
                        selected_rows.extend_from_slice(&original_data[idx_usize]);
                    }
                }

                let num_rows = idx_values.len();
                let num_cols = if num_rows > 0 {
                    selected_rows.len() / num_rows
                } else {
                    0
                };

                from_vec(
                    selected_rows,
                    &[num_rows, num_cols],
                    torsh_core::device::DeviceType::Cpu,
                )
                .unwrap()
            } else {
                // For simplicity, only implement row selection
                tensor.clone()
            }
        }

        /// Filter edges to only include those between selected nodes
        fn filter_edges(&self, edge_index: &Tensor, selected_nodes: &Tensor) -> (Tensor, usize) {
            let edge_data = edge_index.to_vec().unwrap();
            let edges = vec![
                edge_data[0..edge_data.len() / 2]
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<i64>>(),
                edge_data[edge_data.len() / 2..]
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<i64>>(),
            ];
            let selected_indices = selected_nodes.to_vec().unwrap();

            // Create a mapping from old node indices to new ones
            let mut node_mapping = std::collections::HashMap::new();
            for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
                node_mapping.insert(old_idx as i64, new_idx as i64);
            }

            // Filter and remap edges
            let mut filtered_edges = Vec::new();
            for j in 0..edges[0].len() {
                let src = edges[0][j];
                let dst = edges[1][j];

                if let (Some(&new_src), Some(&new_dst)) =
                    (node_mapping.get(&src), node_mapping.get(&dst))
                {
                    filtered_edges.push([new_src, new_dst]);
                }
            }

            if filtered_edges.is_empty() {
                (zeros(&[2, 0]).unwrap(), 0)
            } else {
                let num_edges = filtered_edges.len();
                let mut edge_vec = Vec::with_capacity(2 * num_edges);

                for edge in &filtered_edges {
                    edge_vec.push(edge[0]);
                }
                for edge in &filtered_edges {
                    edge_vec.push(edge[1]);
                }

                (
                    from_vec(
                        edge_vec.iter().map(|&x| x as f32).collect(),
                        &[2, num_edges],
                        torsh_core::device::DeviceType::Cpu,
                    )
                    .unwrap(),
                    num_edges,
                )
            }
        }

        /// Get parameters
        pub fn parameters(&self) -> Vec<Tensor> {
            vec![self.score_layer.clone_data()]
        }
    }

    /// MinCut pooling for graph coarsening
    pub struct MinCutPool {
        input_dim: usize,
        output_dim: usize,
        assignment_layer: Parameter,
    }

    impl MinCutPool {
        /// Create a new MinCut pooling layer
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let assignment_layer = Parameter::new(randn(&[input_dim, output_dim]).unwrap());

            Self {
                input_dim,
                output_dim,
                assignment_layer,
            }
        }

        /// Apply MinCut pooling
        pub fn forward(&self, graph: &GraphData) -> (GraphData, Tensor) {
            // Compute soft assignment matrix
            let assignment_logits = graph.x.matmul(&self.assignment_layer.clone_data()).unwrap();
            let assignment_matrix = assignment_logits.softmax(-1).unwrap();

            // Pool node features
            let pooled_features = assignment_matrix.t().unwrap().matmul(&graph.x).unwrap();

            // Compute adjacency matrix
            let adjacency = self.compute_adjacency_matrix(&graph.edge_index, graph.num_nodes);

            // Pool adjacency matrix
            let pooled_adj = assignment_matrix
                .t()
                .unwrap()
                .matmul(&adjacency)
                .unwrap()
                .matmul(&assignment_matrix)
                .unwrap();

            // Create new edge index
            let (new_edge_index, new_num_edges) = self.adjacency_to_edge_index(&pooled_adj);

            // Compute MinCut loss
            let mincut_loss = self.compute_mincut_loss(&adjacency, &assignment_matrix);
            let orthogonality_loss = self.compute_orthogonality_loss(&assignment_matrix);
            let total_loss = mincut_loss.add(&orthogonality_loss).unwrap();

            let pooled_graph = GraphData {
                x: pooled_features,
                edge_index: new_edge_index,
                edge_attr: None,
                batch: None,
                num_nodes: self.output_dim,
                num_edges: new_num_edges,
            };

            (pooled_graph, total_loss)
        }

        /// Compute adjacency matrix from edge index
        fn compute_adjacency_matrix(&self, edge_index: &Tensor, num_nodes: usize) -> Tensor {
            let mut adjacency = zeros(&[num_nodes, num_nodes]).unwrap();
            let edge_data = edge_index.to_vec().unwrap();
            let edge_list: Vec<Vec<i64>> = vec![
                edge_data[0..edge_data.len() / 2]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                edge_data[edge_data.len() / 2..]
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
            ];

            for j in 0..edge_list[0].len() {
                let src = edge_list[0][j] as usize;
                let dst = edge_list[1][j] as usize;
                if src < num_nodes && dst < num_nodes {
                    // Simplified adjacency matrix setting - use direct indexing approach
                    let mut adj_data = adjacency.to_vec().unwrap();
                    adj_data[src * num_nodes + dst] = 1.0;
                    adjacency = torsh_tensor::creation::from_vec(
                        adj_data,
                        &[num_nodes, num_nodes],
                        torsh_core::device::DeviceType::Cpu,
                    )
                    .unwrap();
                }
            }

            adjacency
        }

        /// Convert adjacency matrix to edge index
        fn adjacency_to_edge_index(&self, adjacency: &Tensor) -> (Tensor, usize) {
            let adj_data = adjacency.to_vec().unwrap();
            let mut edges = Vec::new();

            // Convert flattened vector to 2D indexing using tensor shape
            let shape = adjacency.shape();
            let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    if idx < adj_data.len() && adj_data[idx] > 0.1 {
                        // Threshold for edge existence
                        edges.push([i as f32, j as f32]);
                    }
                }
            }

            if edges.is_empty() {
                (zeros(&[2, 0]).unwrap(), 0)
            } else {
                let num_edges = edges.len();
                let mut edge_vec = Vec::with_capacity(2 * num_edges);

                for edge in &edges {
                    edge_vec.push(edge[0]);
                }
                for edge in &edges {
                    edge_vec.push(edge[1]);
                }

                (
                    from_vec(
                        edge_vec.iter().map(|&x| x as f32).collect(),
                        &[2, num_edges],
                        torsh_core::device::DeviceType::Cpu,
                    )
                    .unwrap(),
                    num_edges,
                )
            }
        }

        /// Compute MinCut loss
        fn compute_mincut_loss(&self, adjacency: &Tensor, assignment: &Tensor) -> Tensor {
            // MinCut loss encourages nodes in different clusters to have few connections
            let cut = assignment
                .t()
                .unwrap()
                .matmul(adjacency)
                .unwrap()
                .matmul(assignment)
                .unwrap();
            // Compute degree for each cluster (sum along node dimension)
            let degree = assignment.sum_dim(&[0], false).unwrap();

            // Normalized cut - outer product of degrees
            let degree_unsqueezed = degree.unsqueeze(0).unwrap();
            let degree_t = degree.unsqueeze(1).unwrap();
            let degree_product = degree_t.matmul(&degree_unsqueezed).unwrap();
            let eps_tensor = torsh_tensor::creation::ones_like(&degree_product)
                .unwrap()
                .mul_scalar(1e-8_f32)
                .unwrap();
            let normalized_cut = cut.div(&degree_product.add(&eps_tensor).unwrap()).unwrap();
            // Simplified trace computation - sum of diagonal elements
            let diag_sum = normalized_cut.sum().unwrap();
            diag_sum.neg().unwrap()
        }

        /// Compute orthogonality loss to encourage balanced clusters
        fn compute_orthogonality_loss(&self, assignment: &Tensor) -> Tensor {
            let cluster_sizes = assignment.sum().unwrap();
            let normalized_sizes = cluster_sizes.div(&cluster_sizes.sum().unwrap()).unwrap();

            // Entropy loss to encourage balanced clusters
            let eps = 1e-8;
            let eps_tensor = torsh_tensor::creation::ones_like(&normalized_sizes)
                .unwrap()
                .mul_scalar(eps as f32)
                .unwrap();
            let entropy_loss = normalized_sizes
                .mul(&normalized_sizes.add(&eps_tensor).unwrap().ln().unwrap())
                .unwrap()
                .sum()
                .unwrap()
                .neg()
                .unwrap();
            entropy_loss.neg().unwrap()
        }

        /// Get parameters
        pub fn parameters(&self) -> Vec<Tensor> {
            vec![self.assignment_layer.clone_data()]
        }
    }
}
