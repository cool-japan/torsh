//! Graph-level classification networks
//!
//! Implementation of complete graph classification architectures
//! for various graph-level prediction tasks as specified in TODO.md

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::conv::{GATConv, GCNConv, GINConv, SAGEConv};
use crate::parameter::Parameter;
use crate::pool::global::{global_max_pool, global_mean_pool, GlobalAttentionPool};
use crate::{GraphData, GraphLayer};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph classification model trait
pub trait GraphClassifier {
    /// Forward pass for graph classification
    fn forward(&self, graph: &GraphData) -> Tensor;

    /// Get model parameters
    fn parameters(&self) -> Vec<Tensor>;

    /// Get number of output classes
    fn num_classes(&self) -> usize;
}

/// Multi-layer Graph Convolutional Network for graph classification
pub struct GraphClassificationGCN {
    /// GCN layers
    layers: Vec<GCNConv>,
    /// Final classification layer
    classifier: Parameter,
    /// Bias for classifier
    bias: Option<Parameter>,
    /// Pooling type
    pooling_type: PoolingType,
    /// Global attention pool (if used)
    attention_pool: Option<GlobalAttentionPool>,
    /// Number of classes
    num_classes: usize,
    /// Dropout rate
    dropout: f32,
}

/// Pooling strategies for graph-level representations
#[derive(Debug, Clone)]
pub enum PoolingType {
    Mean,
    Max,
    Sum,
    Attention,
}

impl GraphClassificationGCN {
    /// Create new graph classification GCN
    pub fn new(
        layer_dims: Vec<usize>,
        num_classes: usize,
        pooling_type: PoolingType,
        dropout: f32,
    ) -> Self {
        assert!(
            layer_dims.len() >= 2,
            "Need at least input and output dimensions"
        );

        let mut layers = Vec::new();
        for i in 0..layer_dims.len() - 1 {
            layers.push(GCNConv::new(layer_dims[i], layer_dims[i + 1], true));
        }

        let final_dim = layer_dims.last().unwrap();
        let classifier = Parameter::new(randn(&[*final_dim, num_classes]).unwrap());
        let bias = Some(Parameter::new(zeros(&[num_classes]).unwrap()));

        let attention_pool = if matches!(pooling_type, PoolingType::Attention) {
            Some(GlobalAttentionPool::new(*final_dim, *final_dim))
        } else {
            None
        };

        Self {
            layers,
            classifier,
            bias,
            pooling_type,
            attention_pool,
            num_classes,
            dropout,
        }
    }

    /// Apply pooling to get graph-level representation
    fn pool_graph(&self, graph: &GraphData) -> Tensor {
        match &self.pooling_type {
            PoolingType::Mean => global_mean_pool(graph),
            PoolingType::Max => global_max_pool(graph),
            PoolingType::Sum => {
                // Sum pooling
                let num_features = graph.x.shape().dims()[1];
                let mut sum_features = zeros(&[num_features]).unwrap();

                for node in 0..graph.num_nodes {
                    let node_feat = graph
                        .x
                        .slice_tensor(0, node, node + 1)
                        .unwrap()
                        .squeeze_tensor(0)
                        .unwrap();
                    sum_features = sum_features.add(&node_feat).unwrap();
                }

                sum_features
            }
            PoolingType::Attention => {
                if let Some(ref pool) = self.attention_pool {
                    pool.forward(graph)
                } else {
                    global_mean_pool(graph) // Fallback
                }
            }
        }
    }
}

impl GraphClassifier for GraphClassificationGCN {
    fn forward(&self, graph: &GraphData) -> Tensor {
        let mut current_graph = graph.clone();

        // Forward through GCN layers
        for (i, layer) in self.layers.iter().enumerate() {
            current_graph = layer.forward(&current_graph);

            // Apply ReLU activation (except for last layer)
            if i < self.layers.len() - 1 {
                let zero_tensor = zeros(current_graph.x.shape().dims()).unwrap();
                current_graph.x = current_graph.x.maximum(&zero_tensor).unwrap();

                // Apply dropout (simplified - would need proper training/eval mode)
                if self.dropout > 0.0 {
                    // Placeholder for dropout implementation
                }
            }
        }

        // Pool to graph-level representation
        let graph_embedding = self.pool_graph(&current_graph);

        // Classification layer
        // Ensure graph_embedding is 2D for matrix multiplication
        let graph_embedding_2d = if graph_embedding.shape().dims().len() == 1 {
            graph_embedding.unsqueeze_tensor(0).unwrap()
        } else if graph_embedding.shape().dims().len() == 2 {
            graph_embedding
        } else {
            // Flatten to 2D if more than 2 dimensions
            let total_features = graph_embedding.shape().dims().iter().product::<usize>();
            graph_embedding.view(&[1, total_features as i32]).unwrap()
        };

        let mut logits = graph_embedding_2d
            .matmul(&self.classifier.clone_data())
            .unwrap();

        // Squeeze to 1D if needed
        if logits.shape().dims().len() == 2 && logits.shape().dims()[0] == 1 {
            logits = logits.squeeze_tensor(0).unwrap();
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            logits = logits.add(&bias.clone_data()).unwrap();
        }

        logits
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // GCN layer parameters
        for layer in &self.layers {
            params.extend(layer.parameters());
        }

        // Classifier parameters
        params.push(self.classifier.clone_data());
        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        // Attention pool parameters
        if let Some(ref pool) = self.attention_pool {
            params.extend(pool.parameters());
        }

        params
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Graph Attention Network for classification
pub struct GraphClassificationGAT {
    /// GAT layers
    gat_layers: Vec<GATConv>,
    /// Final classification layers
    classifier: Vec<Parameter>,
    /// Pooling strategy
    pooling_type: PoolingType,
    /// Global attention pool
    attention_pool: Option<GlobalAttentionPool>,
    /// Number of classes
    num_classes: usize,
}

impl GraphClassificationGAT {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        num_classes: usize,
        dropout: f32,
    ) -> Self {
        let mut gat_layers = Vec::new();

        // First layer
        gat_layers.push(GATConv::new(
            input_dim, hidden_dim, num_heads, dropout, true,
        ));

        // Hidden layers
        for _ in 1..num_layers {
            gat_layers.push(GATConv::new(
                hidden_dim * num_heads,
                hidden_dim,
                num_heads,
                dropout,
                true,
            ));
        }

        // Classifier
        let final_dim = hidden_dim * num_heads;
        let classifier = vec![
            Parameter::new(randn(&[final_dim, final_dim / 2]).unwrap()),
            Parameter::new(randn(&[final_dim / 2, num_classes]).unwrap()),
        ];

        Self {
            gat_layers,
            classifier,
            pooling_type: PoolingType::Attention,
            attention_pool: Some(GlobalAttentionPool::new(final_dim, final_dim)),
            num_classes,
        }
    }
}

impl GraphClassifier for GraphClassificationGAT {
    fn forward(&self, graph: &GraphData) -> Tensor {
        let mut current_graph = graph.clone();

        // Forward through GAT layers
        for (i, layer) in self.gat_layers.iter().enumerate() {
            current_graph = layer.forward(&current_graph);

            // Apply activation (except for last layer)
            if i < self.gat_layers.len() - 1 {
                let zero_tensor = zeros(current_graph.x.shape().dims()).unwrap();
                current_graph.x = current_graph.x.maximum(&zero_tensor).unwrap();
            }
        }

        // Pool to graph-level representation
        let graph_embedding = if let Some(ref pool) = self.attention_pool {
            pool.forward(&current_graph)
        } else {
            global_mean_pool(&current_graph)
        };

        // Two-layer classifier
        // Ensure graph_embedding is 2D for matrix multiplication
        let graph_embedding_2d = if graph_embedding.shape().dims().len() == 1 {
            graph_embedding.unsqueeze_tensor(0).unwrap()
        } else if graph_embedding.shape().dims().len() == 2 {
            graph_embedding
        } else {
            // Flatten to 2D if more than 2 dimensions
            let total_features = graph_embedding.shape().dims().iter().product::<usize>();
            graph_embedding.view(&[1, total_features as i32]).unwrap()
        };

        let mut hidden = graph_embedding_2d
            .matmul(&self.classifier[0].clone_data())
            .unwrap();

        // Squeeze to 1D if needed
        if hidden.shape().dims().len() == 2 && hidden.shape().dims()[0] == 1 {
            hidden = hidden.squeeze_tensor(0).unwrap();
        }

        // ReLU activation
        let zero_tensor = zeros(hidden.shape().dims()).unwrap();
        hidden = hidden.maximum(&zero_tensor).unwrap();

        // Final classification
        let hidden_2d = if hidden.shape().dims().len() == 1 {
            hidden.unsqueeze_tensor(0).unwrap()
        } else {
            hidden
        };

        let mut logits = hidden_2d.matmul(&self.classifier[1].clone_data()).unwrap();

        // Squeeze to 1D if needed
        if logits.shape().dims().len() == 2 && logits.shape().dims()[0] == 1 {
            logits = logits.squeeze_tensor(0).unwrap();
        }

        logits
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // GAT layer parameters
        for layer in &self.gat_layers {
            params.extend(layer.parameters());
        }

        // Classifier parameters
        for layer in &self.classifier {
            params.push(layer.clone_data());
        }

        // Attention pool parameters
        if let Some(ref pool) = self.attention_pool {
            params.extend(pool.parameters());
        }

        params
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Hierarchical Graph Classification Network
/// Uses multiple scales of graph representations
pub struct HierarchicalGraphClassifier {
    /// Local feature extractors
    local_layers: Vec<GCNConv>,
    /// Global context layers
    global_layers: Vec<SAGEConv>,
    /// Cross-scale attention
    attention_weights: Parameter,
    /// Final classifier
    classifier: Parameter,
    /// Number of classes
    num_classes: usize,
}

impl HierarchicalGraphClassifier {
    pub fn new(
        input_dim: usize,
        local_hidden: usize,
        global_hidden: usize,
        num_classes: usize,
    ) -> Self {
        // Local feature extraction
        let local_layers = vec![
            GCNConv::new(input_dim, local_hidden, true),
            GCNConv::new(local_hidden, local_hidden, true),
        ];

        // Global context modeling
        let global_layers = vec![
            SAGEConv::new(input_dim, global_hidden, true),
            SAGEConv::new(global_hidden, global_hidden, true),
        ];

        // Cross-scale attention
        let combined_dim = local_hidden + global_hidden;
        let attention_weights = Parameter::new(randn(&[combined_dim, 1]).unwrap());

        // Final classifier
        let classifier = Parameter::new(randn(&[combined_dim, num_classes]).unwrap());

        Self {
            local_layers,
            global_layers,
            attention_weights,
            classifier,
            num_classes,
        }
    }
}

impl GraphClassifier for HierarchicalGraphClassifier {
    fn forward(&self, graph: &GraphData) -> Tensor {
        // Local pathway
        let mut local_graph = graph.clone();
        for layer in &self.local_layers {
            local_graph = layer.forward(&local_graph);
            let zero_tensor = zeros(local_graph.x.shape().dims()).unwrap();
            local_graph.x = local_graph.x.maximum(&zero_tensor).unwrap();
        }
        let local_repr = global_mean_pool(&local_graph);

        // Global pathway
        let mut global_graph = graph.clone();
        for layer in &self.global_layers {
            global_graph = layer.forward(&global_graph);
            let zero_tensor = zeros(global_graph.x.shape().dims()).unwrap();
            global_graph.x = global_graph.x.maximum(&zero_tensor).unwrap();
        }
        let global_repr = global_mean_pool(&global_graph);

        // Combine representations
        // Ensure both representations are 1D first, then expand to 2D
        let local_1d = if local_repr.shape().dims().len() == 1 {
            local_repr
        } else {
            local_repr
                .view(&[local_repr.shape().dims().iter().product::<usize>() as i32])
                .unwrap()
        };
        let global_1d = if global_repr.shape().dims().len() == 1 {
            global_repr
        } else {
            global_repr
                .view(&[global_repr.shape().dims().iter().product::<usize>() as i32])
                .unwrap()
        };

        // Concatenate the 1D tensors along feature dimension, then expand to 2D
        let combined_1d = Tensor::cat(&[&local_1d, &global_1d], 0).unwrap();
        let combined = combined_1d.unsqueeze_tensor(0).unwrap();

        // Apply cross-scale attention
        // Ensure combined tensor is 2D for matrix multiplication
        let combined_2d = if combined.shape().dims().len() == 1 {
            combined.unsqueeze_tensor(0).unwrap()
        } else {
            combined.clone()
        };

        let mut attention_scores = combined_2d
            .matmul(&self.attention_weights.clone_data())
            .unwrap();

        // Squeeze to appropriate dimensions
        if attention_scores.shape().dims().len() == 2 && attention_scores.shape().dims()[1] == 1 {
            attention_scores = attention_scores.squeeze_tensor(1).unwrap();
        }

        // Softmax attention
        let exp_scores = attention_scores.exp().unwrap();
        let sum_exp = exp_scores.sum().unwrap();
        let normalized_scores = exp_scores.div_scalar(sum_exp.to_vec().unwrap()[0]).unwrap();

        // Weighted combination - ensure proper broadcasting
        let scores_expanded = normalized_scores.unsqueeze_tensor(1).unwrap();
        let combined_shape = combined_2d.shape();
        let scores_broadcasted = scores_expanded.expand(combined_shape.dims()).unwrap();
        let attended = combined_2d
            .mul(&scores_broadcasted)
            .unwrap()
            .sum_dim(&[0], false) // Sum along batch dimension, keep feature dimension
            .unwrap();

        // Classification
        let attended_2d = if attended.shape().dims().len() == 1 {
            attended.unsqueeze_tensor(0).unwrap()
        } else {
            attended
        };

        let mut logits = attended_2d.matmul(&self.classifier.clone_data()).unwrap();

        // Squeeze to 1D if needed
        if logits.shape().dims().len() == 2 && logits.shape().dims()[0] == 1 {
            logits = logits.squeeze_tensor(0).unwrap();
        }

        logits
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // Local pathway parameters
        for layer in &self.local_layers {
            params.extend(layer.parameters());
        }

        // Global pathway parameters
        for layer in &self.global_layers {
            params.extend(layer.parameters());
        }

        // Attention and classifier parameters
        params.push(self.attention_weights.clone_data());
        params.push(self.classifier.clone_data());

        params
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Graph-level regression for continuous targets
pub struct GraphRegressor {
    /// Feature extraction layers
    feature_layers: Vec<GINConv>,
    /// Regression head
    regressor: Parameter,
    /// Bias
    bias: Option<Parameter>,
    /// Output dimension (1 for scalar regression, >1 for multivariate)
    output_dim: usize,
}

impl GraphRegressor {
    pub fn new(layer_dims: Vec<usize>, output_dim: usize, bias: bool) -> Self {
        let mut feature_layers = Vec::new();
        for i in 0..layer_dims.len() - 1 {
            feature_layers.push(GINConv::new(
                layer_dims[i],
                layer_dims[i + 1],
                0.0,
                false,
                true,
            ));
        }

        let final_dim = layer_dims.last().unwrap();
        let regressor = Parameter::new(randn(&[*final_dim, output_dim]).unwrap());
        let bias_param = if bias {
            Some(Parameter::new(zeros(&[output_dim]).unwrap()))
        } else {
            None
        };

        Self {
            feature_layers,
            regressor,
            bias: bias_param,
            output_dim,
        }
    }

    pub fn forward(&self, graph: &GraphData) -> Tensor {
        let mut current_graph = graph.clone();

        // Forward through feature layers
        for layer in &self.feature_layers {
            current_graph = layer.forward(&current_graph);

            // ReLU activation
            let zero_tensor = zeros(current_graph.x.shape().dims()).unwrap();
            current_graph.x = current_graph.x.maximum(&zero_tensor).unwrap();
        }

        // Pool to graph-level representation
        let graph_embedding = global_mean_pool(&current_graph);

        // Regression
        let graph_embedding_2d = if graph_embedding.shape().dims().len() == 1 {
            graph_embedding.unsqueeze_tensor(0).unwrap()
        } else {
            graph_embedding
        };

        let mut output = graph_embedding_2d
            .matmul(&self.regressor.clone_data())
            .unwrap();

        // Squeeze to 1D if needed
        if output.shape().dims().len() == 2 && output.shape().dims()[0] == 1 {
            output = output.squeeze_tensor(0).unwrap();
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        output
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // Feature layer parameters
        for layer in &self.feature_layers {
            params.extend(layer.parameters());
        }

        // Regression parameters
        params.push(self.regressor.clone_data());
        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

/// Multi-task graph learning for simultaneous classification and regression
pub struct MultiTaskGraphNetwork {
    /// Shared feature extractor
    shared_layers: Vec<GCNConv>,
    /// Classification head
    classifier: Parameter,
    /// Regression head
    regressor: Parameter,
    /// Task-specific biases
    classification_bias: Parameter,
    regression_bias: Parameter,
    /// Dimensions
    num_classes: usize,
    regression_dim: usize,
}

impl MultiTaskGraphNetwork {
    pub fn new(layer_dims: Vec<usize>, num_classes: usize, regression_dim: usize) -> Self {
        let mut shared_layers = Vec::new();
        for i in 0..layer_dims.len() - 1 {
            shared_layers.push(GCNConv::new(layer_dims[i], layer_dims[i + 1], true));
        }

        let final_dim = layer_dims.last().unwrap();

        let classifier = Parameter::new(randn(&[*final_dim, num_classes]).unwrap());
        let regressor = Parameter::new(randn(&[*final_dim, regression_dim]).unwrap());

        let classification_bias = Parameter::new(zeros(&[num_classes]).unwrap());
        let regression_bias = Parameter::new(zeros(&[regression_dim]).unwrap());

        Self {
            shared_layers,
            classifier,
            regressor,
            classification_bias,
            regression_bias,
            num_classes,
            regression_dim,
        }
    }

    pub fn forward(&self, graph: &GraphData) -> (Tensor, Tensor) {
        let mut current_graph = graph.clone();

        // Forward through shared layers
        for layer in &self.shared_layers {
            current_graph = layer.forward(&current_graph);

            let zero_tensor = zeros(current_graph.x.shape().dims()).unwrap();
            current_graph.x = current_graph.x.maximum(&zero_tensor).unwrap();
        }

        // Pool to graph-level representation
        let graph_embedding = global_mean_pool(&current_graph);

        // Classification output
        let graph_embedding_2d = if graph_embedding.shape().dims().len() == 1 {
            graph_embedding.unsqueeze_tensor(0).unwrap()
        } else {
            graph_embedding.clone()
        };

        let mut classification_logits = graph_embedding_2d
            .matmul(&self.classifier.clone_data())
            .unwrap();

        // Squeeze to 1D if needed
        if classification_logits.shape().dims().len() == 2
            && classification_logits.shape().dims()[0] == 1
        {
            classification_logits = classification_logits.squeeze_tensor(0).unwrap();
        }

        let classification_logits = classification_logits
            .add(&self.classification_bias.clone_data())
            .unwrap();

        // Regression output
        let graph_embedding_2d_reg = if graph_embedding.shape().dims().len() == 1 {
            graph_embedding.unsqueeze_tensor(0).unwrap()
        } else {
            graph_embedding
        };

        let mut regression_output = graph_embedding_2d_reg
            .matmul(&self.regressor.clone_data())
            .unwrap();

        // Squeeze to 1D if needed
        if regression_output.shape().dims().len() == 2 && regression_output.shape().dims()[0] == 1 {
            regression_output = regression_output.squeeze_tensor(0).unwrap();
        }

        let regression_output = regression_output
            .add(&self.regression_bias.clone_data())
            .unwrap();

        (classification_logits, regression_output)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // Shared layer parameters
        for layer in &self.shared_layers {
            params.extend(layer.parameters());
        }

        // Task-specific parameters
        params.push(self.classifier.clone_data());
        params.push(self.regressor.clone_data());
        params.push(self.classification_bias.clone_data());
        params.push(self.regression_bias.clone_data());

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scirs2_integration::generation;

    #[test]
    fn test_graph_classification_gcn() {
        let graph = generation::erdos_renyi(10, 0.3);
        let classifier = GraphClassificationGCN::new(
            vec![16, 32, 16], // layer dimensions
            3,                // num_classes
            PoolingType::Mean,
            0.1, // dropout
        );

        let logits = classifier.forward(&graph);
        assert_eq!(logits.shape().dims(), &[3]);

        let params = classifier.parameters();
        assert!(params.len() > 0);
        assert_eq!(classifier.num_classes(), 3);
    }

    #[test]
    fn test_different_pooling_strategies() {
        let graph = generation::complete(5);
        let pooling_types = vec![
            PoolingType::Mean,
            PoolingType::Max,
            PoolingType::Sum,
            PoolingType::Attention,
        ];

        for pooling in pooling_types {
            let classifier = GraphClassificationGCN::new(vec![16, 8], 2, pooling, 0.0);

            let logits = classifier.forward(&graph);
            assert_eq!(logits.shape().dims(), &[2]);

            let logit_vals = logits.to_vec().unwrap();
            assert!(logit_vals.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_graph_regressor() {
        let graph = generation::barabasi_albert(8, 2);
        let regressor = GraphRegressor::new(
            vec![16, 12, 8],
            2, // bivariate regression
            true,
        );

        let output = regressor.forward(&graph);
        assert_eq!(output.shape().dims(), &[2]);

        let output_vals = output.to_vec().unwrap();
        assert!(output_vals.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_hierarchical_classifier() {
        let graph = generation::watts_strogatz(12, 4, 0.3);
        let classifier = HierarchicalGraphClassifier::new(
            16, // input_dim
            8,  // local_hidden
            8,  // global_hidden
            4,  // num_classes
        );

        let logits = classifier.forward(&graph);
        assert_eq!(logits.shape().dims(), &[4]);

        let params = classifier.parameters();
        assert!(params.len() > 0);
    }

    #[test]
    fn test_multi_task_network() {
        let graph = generation::erdos_renyi(6, 0.5);
        let multi_task = MultiTaskGraphNetwork::new(
            vec![16, 12],
            3, // classification classes
            2, // regression dimensions
        );

        let (class_logits, reg_output) = multi_task.forward(&graph);

        assert_eq!(class_logits.shape().dims(), &[3]);
        assert_eq!(reg_output.shape().dims(), &[2]);

        let class_vals = class_logits.to_vec().unwrap();
        let reg_vals = reg_output.to_vec().unwrap();

        assert!(class_vals.iter().all(|&x| x.is_finite()));
        assert!(reg_vals.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_parameter_count_consistency() {
        let graph = generation::complete(4);

        // Test that parameter access is consistent
        let classifiers: Vec<Box<dyn GraphClassifier>> = vec![Box::new(
            GraphClassificationGCN::new(vec![16, 8], 2, PoolingType::Mean, 0.0),
        )];

        for classifier in classifiers {
            let params1 = classifier.parameters();
            let params2 = classifier.parameters();

            assert_eq!(params1.len(), params2.len());

            // Test forward pass consistency
            let logits1 = classifier.forward(&graph);
            let logits2 = classifier.forward(&graph);

            let vals1 = logits1.to_vec().unwrap();
            let vals2 = logits2.to_vec().unwrap();

            for (v1, v2) in vals1.iter().zip(vals2.iter()) {
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
    }
}
