//! Graph Attention Network (GAT) layer implementation

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use scirs2_core::random::Random;
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Attention Network (GAT) layer
pub struct GATConv {
    in_features: usize,
    out_features: usize,
    heads: usize,
    weight: Parameter,
    attention: Parameter,
    bias: Option<Parameter>,
    dropout: f32,
}

impl GATConv {
    /// Create a new GAT convolution layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        heads: usize,
        dropout: f32,
        bias: bool,
    ) -> Self {
        let weight = Parameter::new(randn(&[in_features, heads * out_features]).unwrap());
        let attention = Parameter::new(randn(&[heads, 2 * out_features]).unwrap());
        let bias = if bias {
            Some(Parameter::new(zeros(&[heads * out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            heads,
            weight,
            attention,
            bias,
            dropout,
        }
    }

    /// Apply graph attention convolution
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        let num_nodes = graph.num_nodes;

        // Transform node features: X @ W
        let x_transformed = graph.x.matmul(&self.weight.clone_data()).unwrap();

        // Reshape to separate heads: [num_nodes, heads, out_features]
        let x_reshaped = x_transformed
            .view(&[
                num_nodes as i32,
                self.heads as i32,
                self.out_features as i32,
            ])
            .unwrap();

        // Get edge indices - flatten and interpret as pairs
        let edge_flat = graph.edge_index.to_vec().unwrap();
        let num_edges = graph.num_edges;

        // Extract source and destination nodes (edge_index is [2, num_edges] stored row-major)
        let src_nodes: Vec<usize> = (0..num_edges).map(|i| edge_flat[i] as usize).collect();
        let dst_nodes: Vec<usize> = (0..num_edges)
            .map(|i| edge_flat[i + num_edges] as usize)
            .collect();

        // Initialize output
        let mut output = zeros(&[num_nodes, self.heads * self.out_features]).unwrap();

        // Process each head independently
        for head in 0..self.heads {
            // Extract attention parameters for this head
            let attention_head = self
                .attention
                .clone_data()
                .slice_tensor(0, head, head + 1)
                .unwrap()
                .squeeze_tensor(0)
                .unwrap();

            // Compute attention scores for all edges
            let mut attention_scores = Vec::with_capacity(num_edges);

            for edge_idx in 0..num_edges {
                let src = src_nodes[edge_idx];
                let dst = dst_nodes[edge_idx];

                // Get source and destination node features for this head
                let src_feat = x_reshaped
                    .slice_tensor(0, src, src + 1)
                    .unwrap()
                    .slice_tensor(1, head, head + 1)
                    .unwrap()
                    .squeeze_tensor(0)
                    .unwrap()
                    .squeeze_tensor(0)
                    .unwrap();

                let dst_feat = x_reshaped
                    .slice_tensor(0, dst, dst + 1)
                    .unwrap()
                    .slice_tensor(1, head, head + 1)
                    .unwrap()
                    .squeeze_tensor(0)
                    .unwrap()
                    .squeeze_tensor(0)
                    .unwrap();

                // Concatenate source and destination features
                let concat_feat = Tensor::cat(&[&src_feat, &dst_feat], 0).unwrap();

                // Compute attention coefficient: a^T [h_i || h_j]
                // Element-wise multiplication and sum to get scalar
                let attention_coeff = attention_head.mul(&concat_feat).unwrap().sum().unwrap();

                // Apply LeakyReLU activation
                let coeff_val = attention_coeff.to_vec().unwrap()[0] as f64;
                let activated_val = if coeff_val > 0.0 {
                    coeff_val
                } else {
                    0.2 * coeff_val // LeakyReLU with alpha=0.2
                };

                attention_scores.push((src, dst, activated_val));
            }

            // Apply softmax normalization for each destination node
            let mut normalized_scores = vec![0.0; num_edges];
            for node in 0..num_nodes {
                // Find edges pointing to this node
                let mut node_edge_indices = Vec::new();
                let mut node_scores = Vec::new();

                for (edge_idx, (_, dst, score)) in attention_scores.iter().enumerate() {
                    if *dst == node {
                        node_edge_indices.push(edge_idx);
                        node_scores.push(*score);
                    }
                }

                if !node_scores.is_empty() {
                    // Apply softmax
                    let max_score = node_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_scores: Vec<f64> =
                        node_scores.iter().map(|s| (*s - max_score).exp()).collect();
                    let sum_exp: f64 = exp_scores.iter().sum();

                    for (i, &edge_idx) in node_edge_indices.iter().enumerate() {
                        normalized_scores[edge_idx] = exp_scores[i] / sum_exp;
                    }
                }
            }

            // Aggregate features using attention weights
            let mut head_output = zeros(&[num_nodes, self.out_features]).unwrap();

            for node in 0..num_nodes {
                let mut node_output = zeros(&[self.out_features]).unwrap();

                for (edge_idx, (src, dst, _)) in attention_scores.iter().enumerate() {
                    if *dst == node {
                        let weight = normalized_scores[edge_idx];
                        if weight > 0.0 {
                            let src_feat = x_reshaped
                                .slice_tensor(0, *src, *src + 1)
                                .unwrap()
                                .slice_tensor(1, head, head + 1)
                                .unwrap()
                                .squeeze_tensor(0)
                                .unwrap()
                                .squeeze_tensor(0)
                                .unwrap();

                            let weighted_feat = src_feat.mul_scalar(weight as f32).unwrap();
                            node_output = node_output.add(&weighted_feat).unwrap();
                        }
                    }
                }

                // Set the aggregated features for this node
                let mut node_slice = head_output.slice_tensor(0, node, node + 1).unwrap();
                node_slice.copy_(&node_output.unsqueeze_tensor(0).unwrap());
            }

            // Place head output into the appropriate slice of the final output
            let start_feat = head * self.out_features;
            let end_feat = (head + 1) * self.out_features;
            let mut output_slice = output.slice_tensor(1, start_feat, end_feat).unwrap();
            output_slice.copy_(&head_output);
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        // Apply dropout if in training mode (placeholder for now)
        if self.dropout > 0.0 {
            // Note: For now, we'll skip dropout implementation to focus on core functionality
            // In a complete implementation, this would apply dropout during training
        }

        GraphData {
            x: output,
            edge_index: graph.edge_index.clone(),
            edge_attr: graph.edge_attr.clone(),
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        }
    }
}

impl GraphLayer for GATConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone_data(), self.attention.clone_data()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_gat_creation() {
        let gat = GATConv::new(16, 8, 4, 0.1, true);
        let params = gat.parameters();
        assert_eq!(params.len(), 3); // weight + attention + bias
        assert_eq!(gat.heads, 4);
    }

    #[test]
    fn test_gat_forward() {
        let gat = GATConv::new(3, 4, 2, 0.0, false);

        // Create simple test graph
        let x = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let edge_index =
            from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0], &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(x, edge_index);

        let output = gat.forward(&graph);
        assert_eq!(output.x.shape().dims(), &[3, 8]); // 2 heads * 4 features
        assert_eq!(output.num_nodes, 3);
    }
}
