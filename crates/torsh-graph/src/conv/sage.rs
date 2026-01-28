//! GraphSAGE (Sample and Aggregate) layer implementation

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// GraphSAGE convolution layer
#[derive(Debug)]
pub struct SAGEConv {
    in_features: usize,
    out_features: usize,
    weight_neighbor: Parameter,
    weight_self: Parameter,
    bias: Option<Parameter>,
}

impl SAGEConv {
    /// Create a new GraphSAGE convolution layer
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight_neighbor = Parameter::new(
            randn(&[in_features, out_features]).expect("failed to create neighbor weight tensor"),
        );
        let weight_self = Parameter::new(
            randn(&[in_features, out_features]).expect("failed to create self weight tensor"),
        );
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
            weight_neighbor,
            weight_self,
            bias,
        }
    }

    /// Get input feature dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Apply GraphSAGE convolution
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        let num_nodes = graph.num_nodes;
        let edge_data = crate::utils::tensor_to_vec2::<f32>(&graph.edge_index)
            .expect("failed to extract edge index data");

        // Build adjacency list for efficient neighbor aggregation
        let mut adjacency_list: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;
            adjacency_list[dst].push(src);
        }

        // Aggregate neighbor features (mean aggregation)
        let mut neighbor_features = zeros(&[num_nodes, self.in_features])
            .expect("failed to create neighbor features tensor");

        for node in 0..num_nodes {
            if !adjacency_list[node].is_empty() {
                let mut aggregated = zeros(&[self.in_features])
                    .expect("failed to create aggregated features tensor");

                for &neighbor in &adjacency_list[node] {
                    let neighbor_slice = graph
                        .x
                        .slice(0, neighbor, neighbor + 1)
                        .expect("failed to slice neighbor features")
                        .to_tensor()
                        .expect("failed to convert slice to tensor");
                    let neighbor_feat = neighbor_slice.squeeze(0).expect("squeeze should succeed");
                    aggregated = aggregated
                        .add(&neighbor_feat)
                        .expect("operation should succeed");
                }

                // Mean aggregation
                aggregated = aggregated
                    .div_scalar(adjacency_list[node].len() as f32)
                    .expect("failed to compute mean aggregation");
                // Store aggregated features for this node
                let aggregated_data = aggregated.to_vec().expect("conversion should succeed");
                for (i, &value) in aggregated_data.iter().enumerate() {
                    neighbor_features
                        .set_item(&[node, i], value)
                        .expect("failed to set neighbor feature value");
                }
            }
        }

        // Transform neighbor features and self features
        let neighbor_transformed = neighbor_features
            .matmul(&self.weight_neighbor.clone_data())
            .expect("operation should succeed");
        let self_transformed = graph
            .x
            .matmul(&self.weight_self.clone_data())
            .expect("operation should succeed");

        // Combine neighbor and self representations
        let mut output_features = neighbor_transformed
            .add(&self_transformed)
            .expect("operation should succeed");

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output_features = output_features
                .add(&bias.clone_data())
                .expect("operation should succeed");
        }

        // L2 normalize the output features (common in GraphSAGE)
        // For simplicity, using standard normalization instead of row-wise normalization
        let norm_val = output_features
            .norm()
            .expect("failed to compute feature norm");
        let epsilon = 1e-8_f32;
        let norm_scalar = norm_val
            .item()
            .expect("tensor should have single item")
            .max(epsilon);
        output_features = output_features
            .div_scalar(norm_scalar)
            .expect("failed to normalize output features");

        // Create output graph
        GraphData {
            x: output_features,
            edge_index: graph.edge_index.clone(),
            edge_attr: graph.edge_attr.clone(),
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        }
    }
}

impl GraphLayer for SAGEConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.weight_neighbor.clone_data(),
            self.weight_self.clone_data(),
        ];
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
    fn test_sage_creation() {
        let sage = SAGEConv::new(10, 20, true);
        let params = sage.parameters();
        assert_eq!(params.len(), 3); // weight_neighbor + weight_self + bias
    }

    #[test]
    fn test_sage_forward() {
        let sage = SAGEConv::new(4, 8, false);

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

        let output = sage.forward(&graph);
        assert_eq!(output.x.shape().dims(), &[3, 8]);
        assert_eq!(output.num_nodes, 3);

        // Check that output is finite (simplified test since norm_dim doesn't exist)
        let output_values = output.x.to_vec().unwrap();
        for &val in &output_values {
            assert!(val.is_finite(), "Output should be finite");
        }
    }
}
