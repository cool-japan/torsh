//! Graph Convolutional Network (GCN) layer implementation

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Convolutional Network (GCN) layer
#[derive(Debug)]
pub struct GCNConv {
    in_features: usize,
    out_features: usize,
    weight: Parameter,
    bias: Option<Parameter>,
}

impl GCNConv {
    /// Create a new GCN convolution layer
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Parameter::new(randn(&[in_features, out_features]).unwrap());
        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            weight,
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

    /// Apply graph convolution
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        // Compute normalized Laplacian matrix
        let laplacian = crate::utils::graph_laplacian(&graph.edge_index, graph.num_nodes, true);

        // Apply graph convolution: L @ X @ W
        let x_transformed = graph.x.matmul(&self.weight.clone_data()).unwrap();
        let mut output_features = laplacian.matmul(&x_transformed).unwrap();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output_features = output_features.add(&bias.clone_data()).unwrap();
        }

        // Create output graph with transformed features
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

impl GraphLayer for GCNConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone_data()];
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
    fn test_gcn_creation() {
        let gcn = GCNConv::new(8, 16, true);
        let params = gcn.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_gcn_forward() {
        let gcn = GCNConv::new(3, 8, false);

        // Create simple test graph
        let x = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DeviceType::Cpu).unwrap();
        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 0.0], &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(x, edge_index);

        let output = gcn.forward(&graph);
        assert_eq!(output.x.shape().dims(), &[2, 8]);
        assert_eq!(output.num_nodes, 2);
    }
}
