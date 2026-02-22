//! Graph Isomorphism Network (GIN) layer implementation
//! Based on the paper "How Powerful are Graph Neural Networks?"

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Isomorphism Network (GIN) layer
#[derive(Debug)]
pub struct GINConv {
    in_features: usize,
    out_features: usize,
    eps: f64,
    train_eps: bool,
    eps_param: Option<Parameter>,
    mlp: Vec<Parameter>, // Simple MLP: Linear -> ReLU -> Linear
    bias: Option<Parameter>,
}

impl GINConv {
    /// Create a new GIN convolution layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        eps: f64,
        train_eps: bool,
        bias: bool,
    ) -> Self {
        let eps_param = if train_eps {
            Some(Parameter::new(
                torsh_tensor::creation::tensor_scalar(eps as f32)
                    .expect("failed to create epsilon scalar"),
            ))
        } else {
            None
        };

        // Create a simple 2-layer MLP
        let hidden_dim = (in_features + out_features) / 2;
        let mlp = vec![
            Parameter::new(
                randn(&[in_features, hidden_dim]).expect("failed to create MLP layer 1 weights"),
            ),
            Parameter::new(
                randn(&[hidden_dim, out_features]).expect("failed to create MLP layer 2 weights"),
            ),
        ];

        let bias = if bias {
            Some(Parameter::new(
                zeros(&[out_features]).expect("failed to create bias tensor"),
            ))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            eps,
            train_eps,
            eps_param,
            mlp,
            bias,
        }
    }

    /// Apply GIN convolution
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        let num_nodes = graph.num_nodes;
        let edge_flat = graph
            .edge_index
            .to_vec()
            .expect("conversion should succeed");
        let num_edges = edge_flat.len() / 2;
        let edge_data = vec![
            edge_flat[0..num_edges].to_vec(),
            edge_flat[num_edges..].to_vec(),
        ];

        // Build adjacency list for efficient neighbor aggregation
        let mut adjacency_list: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;
            if src < num_nodes && dst < num_nodes {
                adjacency_list[dst].push(src);
            }
        }

        // Aggregate neighbor features (sum aggregation for GIN)
        let neighbor_features = zeros(&[num_nodes, self.in_features])
            .expect("failed to create neighbor features tensor");

        for node in 0..num_nodes {
            let mut aggregated =
                zeros(&[self.in_features]).expect("failed to create aggregated features tensor");

            // Sum all neighbor features
            for &neighbor in &adjacency_list[node] {
                let neighbor_feat = graph
                    .x
                    .slice_tensor(0, neighbor, neighbor + 1)
                    .expect("failed to slice neighbor features")
                    .squeeze_tensor(0)
                    .expect("failed to squeeze neighbor features");
                aggregated = aggregated
                    .add(&neighbor_feat)
                    .expect("operation should succeed");
            }

            let mut node_slice = neighbor_features
                .slice_tensor(0, node, node + 1)
                .expect("failed to slice neighbor features");
            let _ = node_slice.copy_(
                &aggregated
                    .unsqueeze_tensor(0)
                    .expect("failed to unsqueeze aggregated features"),
            );
        }

        // Get epsilon value
        let epsilon = if let Some(ref eps_param) = self.eps_param {
            eps_param
                .clone_data()
                .to_vec()
                .expect("conversion should succeed")[0] as f64
        } else {
            self.eps
        };

        // Combine self and neighbor features: (1 + eps) * h_i + sum(h_j)
        let self_weighted = graph
            .x
            .mul_scalar((1.0 + epsilon) as f32)
            .expect("failed to scale self features");
        let combined_features = self_weighted
            .add(&neighbor_features)
            .expect("operation should succeed");

        // Apply MLP
        let mut output = combined_features
            .matmul(&self.mlp[0].clone_data())
            .expect("operation should succeed");

        // Apply ReLU activation (using max with zero tensor)
        let zero_tensor =
            zeros(output.shape().dims()).expect("failed to create zero tensor for ReLU");
        output = output
            .maximum(&zero_tensor)
            .expect("failed to apply ReLU activation");

        // Second layer
        output = output
            .matmul(&self.mlp[1].clone_data())
            .expect("operation should succeed");

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output
                .add(&bias.clone_data())
                .expect("operation should succeed");
        }

        // Create output graph
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

impl GraphLayer for GINConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.mlp[0].clone_data(), self.mlp[1].clone_data()];

        if let Some(ref eps_param) = self.eps_param {
            params.push(eps_param.clone_data());
        }

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
    fn test_gin_creation() {
        let gin = GINConv::new(8, 16, 0.5, true, true);
        let params = gin.parameters();
        assert!(params.len() >= 2); // At least MLP weights
        assert!(params.len() <= 4); // At most MLP + eps + bias
    }

    #[test]
    fn test_gin_forward() {
        let gin = GINConv::new(4, 6, 0.0, false, false);

        // Create test graph
        let x = from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // node 0
                5.0, 6.0, 7.0, 8.0, // node 1
                9.0, 10.0, 11.0, 12.0, // node 2
            ],
            &[3, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let edge_index =
            from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0], &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(x, edge_index);

        let output = gin.forward(&graph);
        assert_eq!(output.x.shape().dims(), &[3, 6]);
        assert_eq!(output.num_nodes, 3);
    }

    #[test]
    fn test_gin_trainable_eps() {
        let gin_fixed = GINConv::new(4, 8, 1.0, false, false);
        let gin_trainable = GINConv::new(4, 8, 1.0, true, false);

        let fixed_params = gin_fixed.parameters();
        let trainable_params = gin_trainable.parameters();

        // Trainable eps version should have one more parameter
        assert_eq!(trainable_params.len(), fixed_params.len() + 1);
    }
}
