//! Multi-Modal Graph Learning
//!
//! Advanced implementation of multi-modal graph neural networks for learning
//! from heterogeneous data modalities including text, images, audio, and
//! structured data on graph structures.
//!
//! # Features:
//! - Cross-modal graph attention mechanisms
//! - Multi-modal graph fusion strategies
//! - Modality-specific encoders and decoders
//! - Graph-based contrastive learning across modalities
//! - Multi-modal graph pre-training
//! - Zero-shot graph learning with multi-modal embeddings

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use std::collections::{HashMap, HashSet};
use torsh_tensor::{
    creation::{from_vec, ones, randn, zeros},
    Tensor,
};

/// Supported data modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Tabular,
    Graph,
    Video,
    TimeSeries,
}

/// Multi-modal data for a single node
#[derive(Debug, Clone)]
pub struct MultiModalNodeData {
    pub modalities: HashMap<Modality, Tensor>,
    pub node_id: usize,
    pub labels: Option<Tensor>,
}

impl MultiModalNodeData {
    /// Create new multi-modal node data
    pub fn new(node_id: usize) -> Self {
        Self {
            modalities: HashMap::new(),
            node_id,
            labels: None,
        }
    }

    /// Add data for a specific modality
    pub fn add_modality(mut self, modality: Modality, data: Tensor) -> Self {
        self.modalities.insert(modality, data);
        self
    }

    /// Add labels
    pub fn with_labels(mut self, labels: Tensor) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Get available modalities
    pub fn available_modalities(&self) -> Vec<Modality> {
        self.modalities.keys().copied().collect()
    }

    /// Check if modality is available
    pub fn has_modality(&self, modality: Modality) -> bool {
        self.modalities.contains_key(&modality)
    }
}

/// Multi-modal graph data structure
#[derive(Debug, Clone)]
pub struct MultiModalGraphData {
    /// Base graph structure
    pub graph: GraphData,
    /// Multi-modal data for each node
    pub node_data: HashMap<usize, MultiModalNodeData>,
    /// Available modalities in the dataset
    pub available_modalities: HashSet<Modality>,
    /// Modality-specific feature dimensions
    pub modality_dims: HashMap<Modality, usize>,
}

impl MultiModalGraphData {
    /// Create new multi-modal graph data
    pub fn new(graph: GraphData) -> Self {
        Self {
            graph,
            node_data: HashMap::new(),
            available_modalities: HashSet::new(),
            modality_dims: HashMap::new(),
        }
    }

    /// Add multi-modal data for a node
    pub fn add_node_data(&mut self, node_data: MultiModalNodeData) {
        let node_id = node_data.node_id;

        // Update available modalities
        for modality in node_data.available_modalities() {
            self.available_modalities.insert(modality);

            // Update modality dimensions
            if let Some(data) = node_data.modalities.get(&modality) {
                let dim = data.shape().dims().iter().product::<usize>();
                self.modality_dims.insert(modality, dim);
            }
        }

        self.node_data.insert(node_id, node_data);
    }

    /// Get node data for specific modalities
    pub fn get_modality_data(&self, modality: Modality) -> Vec<(usize, &Tensor)> {
        self.node_data
            .iter()
            .filter_map(|(&node_id, data)| {
                data.modalities
                    .get(&modality)
                    .map(|tensor| (node_id, tensor))
            })
            .collect()
    }

    /// Get nodes that have all specified modalities
    pub fn get_complete_nodes(&self, modalities: &[Modality]) -> Vec<usize> {
        self.node_data
            .iter()
            .filter(|(_, data)| {
                modalities
                    .iter()
                    .all(|&modality| data.has_modality(modality))
            })
            .map(|(&node_id, _)| node_id)
            .collect()
    }

    /// Get statistics about modality coverage
    pub fn modality_statistics(&self) -> HashMap<Modality, f32> {
        let total_nodes = self.graph.num_nodes;
        let mut stats = HashMap::new();

        for &modality in &self.available_modalities {
            let count = self
                .node_data
                .values()
                .filter(|data| data.has_modality(modality))
                .count();

            let coverage = count as f32 / total_nodes as f32;
            stats.insert(modality, coverage);
        }

        stats
    }
}

/// Cross-modal graph attention layer
#[derive(Debug)]
pub struct CrossModalGraphAttention {
    modalities: Vec<Modality>,
    feature_dim: usize,
    attention_dim: usize,
    num_heads: usize,

    // Modality-specific projections
    modality_projections: HashMap<Modality, Parameter>,

    // Cross-modal attention weights
    query_weights: Parameter,
    key_weights: Parameter,
    value_weights: Parameter,

    // Output projection
    output_projection: Parameter,

    // Layer normalization parameters
    layer_norm_weight: Parameter,
    layer_norm_bias: Parameter,

    dropout: f32,
}

impl CrossModalGraphAttention {
    /// Create new cross-modal graph attention layer
    pub fn new(
        modalities: Vec<Modality>,
        modality_dims: HashMap<Modality, usize>,
        feature_dim: usize,
        attention_dim: usize,
        num_heads: usize,
        dropout: f32,
    ) -> Self {
        let mut modality_projections = HashMap::new();

        // Create projection layers for each modality
        for modality in &modalities {
            let input_dim = modality_dims.get(modality).copied().unwrap_or(feature_dim);
            modality_projections.insert(
                *modality,
                Parameter::new(randn(&[input_dim, feature_dim]).unwrap()),
            );
        }

        let query_weights = Parameter::new(randn(&[feature_dim, attention_dim]).unwrap());
        let key_weights = Parameter::new(randn(&[feature_dim, attention_dim]).unwrap());
        let value_weights = Parameter::new(randn(&[feature_dim, attention_dim]).unwrap());
        let output_projection = Parameter::new(randn(&[attention_dim, feature_dim]).unwrap());

        let layer_norm_weight = Parameter::new(ones(&[feature_dim]).unwrap());
        let layer_norm_bias = Parameter::new(zeros::<f32>(&[feature_dim]).unwrap());

        Self {
            modalities,
            feature_dim,
            attention_dim,
            num_heads,
            modality_projections,
            query_weights,
            key_weights,
            value_weights,
            output_projection,
            layer_norm_weight,
            layer_norm_bias,
            dropout,
        }
    }

    /// Forward pass through cross-modal attention
    pub fn forward(&self, mm_graph: &MultiModalGraphData) -> Tensor {
        let num_nodes = mm_graph.graph.num_nodes;

        // Project each modality to common feature space
        let mut modality_features = HashMap::new();

        for &modality in &self.modalities {
            let projection = &self.modality_projections[&modality];
            let modality_data = mm_graph.get_modality_data(modality);

            if !modality_data.is_empty() {
                let features =
                    self.project_modality_features(&modality_data, projection, num_nodes);
                modality_features.insert(modality, features);
            }
        }

        // Apply cross-modal attention
        let attended_features = self.apply_cross_modal_attention(&modality_features);

        // Layer normalization
        self.layer_norm(&attended_features)
    }

    /// Project modality-specific features to common space
    fn project_modality_features(
        &self,
        modality_data: &[(usize, &Tensor)],
        projection: &Parameter,
        num_nodes: usize,
    ) -> Tensor {
        let mut projected_data = vec![0.0f32; num_nodes * self.feature_dim];

        for &(node_id, features) in modality_data {
            if node_id < num_nodes {
                let feature_data = features.to_vec().unwrap();
                let input_tensor = from_vec(
                    feature_data,
                    &[1, features.shape().dims().iter().product::<usize>()],
                    torsh_core::device::DeviceType::Cpu,
                )
                .unwrap();

                let projected = input_tensor.matmul(&projection.clone_data()).unwrap();
                let projected_data_vec = projected.to_vec().unwrap();

                for (i, &val) in projected_data_vec.iter().enumerate() {
                    if i < self.feature_dim {
                        projected_data[node_id * self.feature_dim + i] = val;
                    }
                }
            }
        }

        from_vec(
            projected_data,
            &[num_nodes, self.feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Apply cross-modal attention mechanism
    fn apply_cross_modal_attention(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        if modality_features.is_empty() {
            return zeros::<f32>(&[1, self.feature_dim]).unwrap();
        }

        // For simplicity, use the first modality as the base
        let first_modality = modality_features.keys().next().unwrap();
        let base_features = &modality_features[first_modality];
        let _num_nodes = base_features.shape().dims()[0];

        // Compute queries, keys, and values
        let queries = base_features
            .matmul(&self.query_weights.clone_data())
            .unwrap();
        let _keys = base_features
            .matmul(&self.key_weights.clone_data())
            .unwrap();
        let values = base_features
            .matmul(&self.value_weights.clone_data())
            .unwrap();

        // Apply attention across all modalities
        let mut attended_values = values.clone();

        for (modality, features) in modality_features {
            if *modality != *first_modality {
                let modal_keys = features.matmul(&self.key_weights.clone_data()).unwrap();
                let modal_values = features.matmul(&self.value_weights.clone_data()).unwrap();

                // Simplified attention computation
                let attention_scores = queries.matmul(&modal_keys.t().unwrap()).unwrap();
                let attention_weights = self.softmax(&attention_scores);
                let attended = attention_weights.matmul(&modal_values).unwrap();

                attended_values = attended_values.add(&attended).unwrap();
            }
        }

        // Output projection
        attended_values
            .matmul(&self.output_projection.clone_data())
            .unwrap()
    }

    /// Softmax activation
    fn softmax(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let exp_data: Vec<f32> = data.iter().map(|&val| (val - max_val).exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();

        let softmax_data: Vec<f32> = exp_data.iter().map(|&val| val / sum_exp).collect();

        from_vec(
            softmax_data,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let num_features = self.feature_dim;
        let num_samples = data.len() / num_features;

        let mut normalized_data = Vec::new();

        for sample in 0..num_samples {
            let start_idx = sample * num_features;
            let end_idx = start_idx + num_features;
            let sample_data = &data[start_idx..end_idx];

            // Compute mean and std
            let mean: f32 = sample_data.iter().sum::<f32>() / num_features as f32;
            let variance: f32 =
                sample_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_features as f32;
            let std = (variance + 1e-5).sqrt();

            // Normalize
            for &val in sample_data {
                let normalized = (val - mean) / std;
                normalized_data.push(normalized);
            }
        }

        let normalized_tensor = from_vec(
            normalized_data,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Apply learned parameters
        normalized_tensor
            .mul(&self.layer_norm_weight.clone_data())
            .unwrap()
            .add(&self.layer_norm_bias.clone_data())
            .unwrap()
    }
}

impl GraphLayer for CrossModalGraphAttention {
    fn forward(&self, graph: &GraphData) -> GraphData {
        // Create a simple multi-modal graph with only graph modality
        let mut mm_graph = MultiModalGraphData::new(graph.clone());

        for node_id in 0..graph.num_nodes {
            let node_features = graph.x.slice_tensor(0, node_id, node_id + 1).unwrap();
            let node_data =
                MultiModalNodeData::new(node_id).add_modality(Modality::Graph, node_features);
            mm_graph.add_node_data(node_data);
        }

        let output_features = self.forward(&mm_graph);

        let mut output_graph = graph.clone();
        output_graph.x = output_features;
        output_graph
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.query_weights.clone_data(),
            self.key_weights.clone_data(),
            self.value_weights.clone_data(),
            self.output_projection.clone_data(),
            self.layer_norm_weight.clone_data(),
            self.layer_norm_bias.clone_data(),
        ];

        for projection in self.modality_projections.values() {
            params.push(projection.clone_data());
        }

        params
    }
}

/// Multi-modal graph fusion strategies
#[derive(Debug)]
pub struct MultiModalFusion {
    fusion_strategy: FusionStrategy,
    modalities: Vec<Modality>,
    feature_dim: usize,
    fusion_weights: Option<Parameter>,
    gating_network: Option<Vec<Parameter>>,
}

#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    Concatenation,
    ElementwiseSum,
    WeightedSum,
    AttentionFusion,
    GatedFusion,
}

impl MultiModalFusion {
    /// Create new multi-modal fusion layer
    pub fn new(
        fusion_strategy: FusionStrategy,
        modalities: Vec<Modality>,
        feature_dim: usize,
    ) -> Self {
        let fusion_weights = match fusion_strategy {
            FusionStrategy::WeightedSum => Some(Parameter::new(ones(&[modalities.len()]).unwrap())),
            _ => None,
        };

        let gating_network = match fusion_strategy {
            FusionStrategy::GatedFusion => {
                let mut gates = Vec::new();
                for _ in 0..modalities.len() {
                    gates.push(Parameter::new(randn(&[feature_dim, 1]).unwrap()));
                }
                Some(gates)
            }
            _ => None,
        };

        Self {
            fusion_strategy,
            modalities,
            feature_dim,
            fusion_weights,
            gating_network,
        }
    }

    /// Fuse multi-modal features
    pub fn fuse_features(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        match self.fusion_strategy {
            FusionStrategy::Concatenation => self.concatenate_features(modality_features),
            FusionStrategy::ElementwiseSum => self.elementwise_sum_features(modality_features),
            FusionStrategy::WeightedSum => self.weighted_sum_features(modality_features),
            FusionStrategy::AttentionFusion => self.attention_fusion_features(modality_features),
            FusionStrategy::GatedFusion => self.gated_fusion_features(modality_features),
        }
    }

    /// Concatenate features from different modalities
    fn concatenate_features(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        let mut concatenated_data = Vec::new();

        for &modality in &self.modalities {
            if let Some(features) = modality_features.get(&modality) {
                concatenated_data.extend(features.to_vec().unwrap());
            } else {
                // Pad with zeros for missing modalities
                concatenated_data.extend(vec![0.0f32; self.feature_dim]);
            }
        }

        let num_nodes = modality_features
            .values()
            .next()
            .map(|t| t.shape().dims()[0])
            .unwrap_or(1);

        from_vec(
            concatenated_data,
            &[num_nodes, self.modalities.len() * self.feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Element-wise sum of features
    fn elementwise_sum_features(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        let mut sum_features: Option<Tensor> = None;

        for &modality in &self.modalities {
            if let Some(features) = modality_features.get(&modality) {
                if let Some(ref sum) = sum_features {
                    sum_features = Some(sum.add(features).unwrap());
                } else {
                    sum_features = Some(features.clone());
                }
            }
        }

        sum_features.unwrap_or_else(|| zeros::<f32>(&[1, self.feature_dim]).unwrap())
    }

    /// Weighted sum of features
    fn weighted_sum_features(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        let weights = self
            .fusion_weights
            .as_ref()
            .unwrap()
            .clone_data()
            .to_vec()
            .unwrap();
        let mut weighted_sum: Option<Tensor> = None;

        for (i, &modality) in self.modalities.iter().enumerate() {
            if let Some(features) = modality_features.get(&modality) {
                let weight = weights.get(i).copied().unwrap_or(1.0);
                let weighted_features = features.mul_scalar(weight).unwrap();

                if let Some(ref sum) = weighted_sum {
                    weighted_sum = Some(sum.add(&weighted_features).unwrap());
                } else {
                    weighted_sum = Some(weighted_features);
                }
            }
        }

        weighted_sum.unwrap_or_else(|| zeros::<f32>(&[1, self.feature_dim]).unwrap())
    }

    /// Attention-based fusion
    fn attention_fusion_features(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        // Simplified attention-based fusion
        let available_features: Vec<&Tensor> = modality_features.values().collect();

        if available_features.is_empty() {
            return zeros::<f32>(&[1, self.feature_dim]).unwrap();
        }

        // Compute attention weights based on feature norms
        let mut attention_weights = Vec::new();
        let mut total_norm = 0.0;

        for features in &available_features {
            let data = features.to_vec().unwrap();
            let norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
            attention_weights.push(norm);
            total_norm += norm;
        }

        // Normalize attention weights
        if total_norm > 0.0 {
            for weight in &mut attention_weights {
                *weight /= total_norm;
            }
        }

        // Apply attention weights
        let mut attended_features: Option<Tensor> = None;
        for (features, &weight) in available_features.iter().zip(attention_weights.iter()) {
            let weighted = features.mul_scalar(weight).unwrap();

            if let Some(ref sum) = attended_features {
                attended_features = Some(sum.add(&weighted).unwrap());
            } else {
                attended_features = Some(weighted);
            }
        }

        attended_features.unwrap_or_else(|| zeros::<f32>(&[1, self.feature_dim]).unwrap())
    }

    /// Gated fusion
    fn gated_fusion_features(&self, modality_features: &HashMap<Modality, Tensor>) -> Tensor {
        let gates = self.gating_network.as_ref().unwrap();
        let mut gated_sum: Option<Tensor> = None;

        for (i, &modality) in self.modalities.iter().enumerate() {
            if let Some(features) = modality_features.get(&modality) {
                let gate = &gates[i];
                let gate_values = features.matmul(&gate.clone_data()).unwrap();
                let gate_probs = self.sigmoid(&gate_values);

                // Apply gating
                let gated_features = features.mul(&gate_probs).unwrap();

                if let Some(ref sum) = gated_sum {
                    gated_sum = Some(sum.add(&gated_features).unwrap());
                } else {
                    gated_sum = Some(gated_features);
                }
            }
        }

        gated_sum.unwrap_or_else(|| zeros::<f32>(&[1, self.feature_dim]).unwrap())
    }

    /// Sigmoid activation
    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let sigmoid_data: Vec<f32> = data.iter().map(|&val| 1.0 / (1.0 + (-val).exp())).collect();

        from_vec(
            sigmoid_data,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }
}

/// Contrastive learning for multi-modal graphs
#[derive(Debug)]
pub struct MultiModalContrastiveLearning {
    temperature: f32,
    projection_dim: usize,
    modality_projectors: HashMap<Modality, Parameter>,
}

impl MultiModalContrastiveLearning {
    /// Create new contrastive learning module
    pub fn new(
        modalities: Vec<Modality>,
        modality_dims: HashMap<Modality, usize>,
        projection_dim: usize,
        temperature: f32,
    ) -> Self {
        let mut modality_projectors = HashMap::new();

        for modality in modalities {
            let input_dim = modality_dims.get(&modality).copied().unwrap_or(128);
            modality_projectors.insert(
                modality,
                Parameter::new(randn(&[input_dim, projection_dim]).unwrap()),
            );
        }

        Self {
            temperature,
            projection_dim,
            modality_projectors,
        }
    }

    /// Compute contrastive loss between modalities
    pub fn contrastive_loss(
        &self,
        modality1: Modality,
        features1: &Tensor,
        modality2: Modality,
        features2: &Tensor,
    ) -> f32 {
        // Project features to common space
        let proj1 = features1
            .matmul(&self.modality_projectors[&modality1].clone_data())
            .unwrap();
        let proj2 = features2
            .matmul(&self.modality_projectors[&modality2].clone_data())
            .unwrap();

        // Compute similarity matrix
        let similarity = proj1.matmul(&proj2.t().unwrap()).unwrap();
        let scaled_similarity = similarity.div_scalar(self.temperature).unwrap();

        // Simplified contrastive loss computation
        let sim_data = scaled_similarity.to_vec().unwrap();
        let max_sim = sim_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sims: Vec<f32> = sim_data.iter().map(|&x| (x - max_sim).exp()).collect();
        let sum_exp: f32 = exp_sims.iter().sum();

        // Negative log likelihood of positive pairs (diagonal elements)
        let num_samples = proj1.shape().dims()[0];
        let mut loss = 0.0;

        for i in 0..num_samples {
            let positive_sim = exp_sims[i * num_samples + i];
            loss -= (positive_sim / sum_exp).ln();
        }

        loss / num_samples as f32
    }

    /// Generate positive and negative pairs for contrastive learning
    pub fn generate_contrastive_pairs(
        &self,
        mm_graph: &MultiModalGraphData,
        modality1: Modality,
        modality2: Modality,
    ) -> Vec<(Tensor, Tensor, bool)> {
        let mut pairs = Vec::new();

        let data1 = mm_graph.get_modality_data(modality1);
        let data2 = mm_graph.get_modality_data(modality2);

        // Positive pairs (same node, different modalities)
        for (node_id, features1) in &data1 {
            if let Some((_, features2)) = data2.iter().find(|(id, _)| id == node_id) {
                pairs.push(((*features1).clone(), (*features2).clone(), true));
            }
        }

        // Negative pairs (different nodes, different modalities)
        for (node_id1, features1) in &data1 {
            for (node_id2, features2) in &data2 {
                if node_id1 != node_id2 {
                    pairs.push(((*features1).clone(), (*features2).clone(), false));

                    // Limit number of negative pairs to avoid explosion
                    if pairs.len() > 1000 {
                        break;
                    }
                }
            }
            if pairs.len() > 1000 {
                break;
            }
        }

        pairs
    }
}

/// Multi-modal graph utilities
pub mod utils {
    use super::*;

    /// Create synthetic multi-modal graph data
    pub fn create_synthetic_multimodal_graph(
        num_nodes: usize,
        base_feature_dim: usize,
        modalities: Vec<Modality>,
    ) -> MultiModalGraphData {
        let mut rng = scirs2_core::random::thread_rng();

        // Create base graph
        let base_features = randn(&[num_nodes, base_feature_dim]).unwrap();
        let mut edge_data = Vec::new();

        for _ in 0..(num_nodes * 2) {
            let src = rng.gen_range(0..num_nodes) as f32;
            let dst = rng.gen_range(0..num_nodes) as f32;
            edge_data.push(src);
            edge_data.push(dst);
        }

        let edge_index = from_vec(
            edge_data,
            &[2, num_nodes * 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let graph = GraphData::new(base_features, edge_index);
        let mut mm_graph = MultiModalGraphData::new(graph);

        // Add multi-modal data for each node
        for node_id in 0..num_nodes {
            let mut node_data = MultiModalNodeData::new(node_id);

            for &modality in &modalities {
                // Generate modality-specific features with different dimensions
                let feature_dim = match modality {
                    Modality::Text => 768,   // BERT-like embeddings
                    Modality::Image => 2048, // ResNet-like features
                    Modality::Audio => 128,  // Audio features
                    Modality::Tabular => 64, // Structured data
                    Modality::Graph => base_feature_dim,
                    Modality::Video => 1024,     // Video features
                    Modality::TimeSeries => 256, // Time series features
                };

                // Only add modality data with some probability for missing modality simulation
                if rng.gen_range(0.0..1.0) < 0.8 {
                    let features = randn(&[feature_dim]).unwrap();
                    node_data = node_data.add_modality(modality, features);
                }
            }

            mm_graph.add_node_data(node_data);
        }

        mm_graph
    }

    /// Evaluate multi-modal representation quality
    pub fn evaluate_multimodal_quality(
        mm_graph: &MultiModalGraphData,
        representations: &HashMap<Modality, Tensor>,
    ) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();

        // Coverage metrics
        let modality_stats = mm_graph.modality_statistics();
        for (modality, coverage) in modality_stats {
            metrics.insert(format!("{:?}_coverage", modality), coverage);
        }

        // Representation diversity (simplified)
        for (modality, tensor) in representations {
            let data = tensor.to_vec().unwrap();
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            let variance: f32 =
                data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

            metrics.insert(format!("{:?}_mean", modality), mean);
            metrics.insert(format!("{:?}_variance", modality), variance);
        }

        // Cross-modal consistency (simplified)
        if representations.len() > 1 {
            let modalities: Vec<_> = representations.keys().collect();
            for i in 0..modalities.len() {
                for j in (i + 1)..modalities.len() {
                    let rep1 = &representations[modalities[i]];
                    let rep2 = &representations[modalities[j]];

                    let consistency = compute_tensor_similarity(rep1, rep2);
                    metrics.insert(
                        format!("{:?}_{:?}_consistency", modalities[i], modalities[j]),
                        consistency,
                    );
                }
            }
        }

        metrics
    }

    /// Compute similarity between two tensors
    fn compute_tensor_similarity(tensor1: &Tensor, tensor2: &Tensor) -> f32 {
        let data1 = tensor1.to_vec().unwrap();
        let data2 = tensor2.to_vec().unwrap();

        if data1.len() != data2.len() {
            return 0.0;
        }

        // Cosine similarity
        let dot_product: f32 = data1.iter().zip(data2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = data1.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = data2.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Generate cross-modal alignment tasks
    pub fn generate_alignment_tasks(
        mm_graph: &MultiModalGraphData,
        source_modality: Modality,
        target_modality: Modality,
        num_tasks: usize,
    ) -> Vec<(usize, Tensor, Tensor)> {
        let source_data = mm_graph.get_modality_data(source_modality);
        let target_data = mm_graph.get_modality_data(target_modality);

        let mut tasks = Vec::new();
        let mut rng = scirs2_core::random::thread_rng();

        // Find nodes that have both modalities
        let common_nodes: Vec<usize> = source_data
            .iter()
            .filter_map(|&(node_id, _)| {
                if target_data.iter().any(|&(id, _)| id == node_id) {
                    Some(node_id)
                } else {
                    None
                }
            })
            .collect();

        for _ in 0..num_tasks.min(common_nodes.len()) {
            let &node_id = common_nodes.choose(&mut rng).unwrap();

            let source_features = source_data
                .iter()
                .find(|&&(id, _)| id == node_id)
                .map(|(_, tensor)| (*tensor).clone())
                .unwrap();

            let target_features = target_data
                .iter()
                .find(|&&(id, _)| id == node_id)
                .map(|(_, tensor)| (*tensor).clone())
                .unwrap();

            tasks.push((node_id, source_features, target_features));
        }

        tasks
    }
}

// Implement choose method for Vec<T> (simplified random selection)
trait RandomChoice<T> {
    fn choose(
        &self,
        rng: &mut scirs2_core::random::CoreRandom<scirs2_core::rngs::ThreadRng>,
    ) -> Option<&T>;
}

impl<T> RandomChoice<T> for Vec<T> {
    fn choose(
        &self,
        rng: &mut scirs2_core::random::CoreRandom<scirs2_core::rngs::ThreadRng>,
    ) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            let index = rng.gen_range(0..self.len());
            self.get(index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_multimodal_node_data_creation() {
        let text_features = randn(&[768]).unwrap();
        let image_features = randn(&[2048]).unwrap();

        let node_data = MultiModalNodeData::new(0)
            .add_modality(Modality::Text, text_features)
            .add_modality(Modality::Image, image_features);

        assert_eq!(node_data.node_id, 0);
        assert!(node_data.has_modality(Modality::Text));
        assert!(node_data.has_modality(Modality::Image));
        assert!(!node_data.has_modality(Modality::Audio));
        assert_eq!(node_data.available_modalities().len(), 2);
    }

    #[test]
    fn test_multimodal_graph_data() {
        let features = randn(&[3, 4]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];
        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let mut mm_graph = MultiModalGraphData::new(graph);

        // Add multi-modal data for nodes
        for i in 0..3 {
            let node_data = MultiModalNodeData::new(i)
                .add_modality(Modality::Text, randn(&[768]).unwrap())
                .add_modality(Modality::Image, randn(&[2048]).unwrap());
            mm_graph.add_node_data(node_data);
        }

        assert_eq!(mm_graph.available_modalities.len(), 2);
        assert_eq!(mm_graph.get_modality_data(Modality::Text).len(), 3);
        assert_eq!(
            mm_graph
                .get_complete_nodes(&[Modality::Text, Modality::Image])
                .len(),
            3
        );

        let stats = mm_graph.modality_statistics();
        assert_eq!(stats[&Modality::Text], 1.0); // 100% coverage
        assert_eq!(stats[&Modality::Image], 1.0); // 100% coverage
    }

    #[test]
    fn test_cross_modal_attention() {
        let modalities = vec![Modality::Text, Modality::Image];
        let mut modality_dims = HashMap::new();
        modality_dims.insert(Modality::Text, 768);
        modality_dims.insert(Modality::Image, 2048);

        let attention = CrossModalGraphAttention::new(
            modalities,
            modality_dims,
            256, // feature_dim
            128, // attention_dim
            4,   // num_heads
            0.1, // dropout
        );

        assert_eq!(attention.feature_dim, 256);
        assert_eq!(attention.attention_dim, 128);
        assert_eq!(attention.num_heads, 4);
    }

    #[test]
    fn test_multimodal_fusion() {
        let modalities = vec![Modality::Text, Modality::Image];
        let fusion = MultiModalFusion::new(FusionStrategy::WeightedSum, modalities, 128);

        let mut modality_features = HashMap::new();
        modality_features.insert(Modality::Text, randn(&[3, 128]).unwrap());
        modality_features.insert(Modality::Image, randn(&[3, 128]).unwrap());

        let fused = fusion.fuse_features(&modality_features);
        assert_eq!(fused.shape().dims(), &[3, 128]);
    }

    #[test]
    fn test_contrastive_learning() {
        let modalities = vec![Modality::Text, Modality::Image];
        let mut modality_dims = HashMap::new();
        modality_dims.insert(Modality::Text, 768);
        modality_dims.insert(Modality::Image, 2048);

        let contrastive = MultiModalContrastiveLearning::new(
            modalities,
            modality_dims,
            256,  // projection_dim
            0.07, // temperature
        );

        let text_features = randn(&[4, 768]).unwrap();
        let image_features = randn(&[4, 2048]).unwrap();

        let loss = contrastive.contrastive_loss(
            Modality::Text,
            &text_features,
            Modality::Image,
            &image_features,
        );

        assert!(loss > 0.0);
    }

    #[test]
    fn test_synthetic_multimodal_graph() {
        let modalities = vec![Modality::Text, Modality::Image, Modality::Audio];
        let mm_graph = utils::create_synthetic_multimodal_graph(5, 64, modalities);

        assert_eq!(mm_graph.graph.num_nodes, 5);
        assert!(mm_graph.available_modalities.len() <= 3);

        // Check that some nodes have multi-modal data
        assert!(!mm_graph.node_data.is_empty());

        let stats = mm_graph.modality_statistics();
        for coverage in stats.values() {
            assert!(*coverage >= 0.0 && *coverage <= 1.0);
        }
    }

    #[test]
    fn test_multimodal_quality_evaluation() {
        let modalities = vec![Modality::Text, Modality::Image];
        let mm_graph = utils::create_synthetic_multimodal_graph(4, 32, modalities);

        let mut representations = HashMap::new();
        representations.insert(Modality::Text, randn(&[4, 128]).unwrap());
        representations.insert(Modality::Image, randn(&[4, 128]).unwrap());

        let metrics = utils::evaluate_multimodal_quality(&mm_graph, &representations);

        assert!(metrics.contains_key("Text_mean"));
        assert!(metrics.contains_key("Image_variance"));

        // Check for cross-modal consistency metrics
        let consistency_keys: Vec<_> = metrics
            .keys()
            .filter(|k| k.contains("consistency"))
            .collect();
        assert!(!consistency_keys.is_empty());
    }

    #[test]
    fn test_alignment_task_generation() {
        let modalities = vec![Modality::Text, Modality::Image];
        let mm_graph = utils::create_synthetic_multimodal_graph(3, 32, modalities);

        let tasks = utils::generate_alignment_tasks(&mm_graph, Modality::Text, Modality::Image, 5);

        // Should have some alignment tasks (depending on random generation)
        assert!(tasks.len() <= 5);

        for (node_id, source, target) in &tasks {
            assert!(*node_id < 3);
            assert!(!source.to_vec().unwrap().is_empty());
            assert!(!target.to_vec().unwrap().is_empty());
        }
    }
}
