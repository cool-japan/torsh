//! Heterogeneous Graph Neural Networks
//!
//! Implementation of multi-relational GNNs for heterogeneous graphs
//! with different node types and edge types, as specified in TODO.md

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use std::collections::HashMap;
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Node type identifier
pub type NodeType = String;

/// Edge type identifier (source_type, relation, target_type)
pub type EdgeType = (NodeType, String, NodeType);

/// Heterogeneous graph data structure
#[derive(Debug, Clone)]
pub struct HeteroGraphData {
    /// Node features for each node type
    pub node_features: HashMap<NodeType, Tensor>,
    /// Edge indices for each edge type
    pub edge_indices: HashMap<EdgeType, Tensor>,
    /// Edge attributes for each edge type (optional)
    pub edge_attributes: HashMap<EdgeType, Option<Tensor>>,
    /// Number of nodes per type
    pub num_nodes: HashMap<NodeType, usize>,
}

impl HeteroGraphData {
    /// Create a new heterogeneous graph
    pub fn new() -> Self {
        Self {
            node_features: HashMap::new(),
            edge_indices: HashMap::new(),
            edge_attributes: HashMap::new(),
            num_nodes: HashMap::new(),
        }
    }

    /// Add node type with features
    pub fn add_node_type(&mut self, node_type: NodeType, features: Tensor) -> &mut Self {
        let num_nodes = features.shape().dims()[0];
        self.node_features.insert(node_type.clone(), features);
        self.num_nodes.insert(node_type, num_nodes);
        self
    }

    /// Add edge type with indices
    pub fn add_edge_type(
        &mut self,
        edge_type: EdgeType,
        edge_index: Tensor,
        edge_attr: Option<Tensor>,
    ) -> &mut Self {
        self.edge_indices.insert(edge_type.clone(), edge_index);
        self.edge_attributes.insert(edge_type, edge_attr);
        self
    }

    /// Get all node types
    pub fn node_types(&self) -> Vec<&NodeType> {
        self.node_features.keys().collect()
    }

    /// Get all edge types
    pub fn edge_types(&self) -> Vec<&EdgeType> {
        self.edge_indices.keys().collect()
    }
}

/// Heterogeneous Graph Neural Network layer
pub struct HeteroGNN {
    node_types: Vec<NodeType>,
    edge_types: Vec<EdgeType>,
    /// Type-specific transformation layers
    node_transformations: HashMap<NodeType, Parameter>,
    /// Relation-specific message functions
    edge_transformations: HashMap<EdgeType, Parameter>,
    /// Output dimension
    out_features: usize,
    /// Whether to use bias
    bias: bool,
    /// Bias parameters per node type
    biases: HashMap<NodeType, Option<Parameter>>,
}

impl HeteroGNN {
    /// Create a new heterogeneous GNN layer
    pub fn new(
        node_type_dims: HashMap<NodeType, usize>,
        edge_types: Vec<EdgeType>,
        out_features: usize,
        bias: bool,
    ) -> Self {
        let mut node_transformations = HashMap::new();
        let mut biases = HashMap::new();

        // Create transformation matrices for each node type
        for (node_type, in_features) in &node_type_dims {
            let weight = Parameter::new(randn(&[*in_features, out_features]).unwrap());
            node_transformations.insert(node_type.clone(), weight);

            let bias_param = if bias {
                Some(Parameter::new(zeros(&[out_features]).unwrap()))
            } else {
                None
            };
            biases.insert(node_type.clone(), bias_param);
        }

        // Create edge transformation matrices
        let mut edge_transformations = HashMap::new();
        for edge_type in &edge_types {
            // Use output features as the message dimension
            let weight = Parameter::new(randn(&[out_features, out_features]).unwrap());
            edge_transformations.insert(edge_type.clone(), weight);
        }

        Self {
            node_types: node_type_dims.keys().cloned().collect(),
            edge_types,
            node_transformations,
            edge_transformations,
            out_features,
            bias,
            biases,
        }
    }

    /// Forward pass through heterogeneous GNN
    pub fn forward(&self, hetero_graph: &HeteroGraphData) -> HeteroGraphData {
        let mut output_features = HashMap::new();

        // Step 1: Transform node features for each node type
        let mut transformed_features = HashMap::new();
        for node_type in &self.node_types {
            if let Some(features) = hetero_graph.node_features.get(node_type) {
                if let Some(transform) = self.node_transformations.get(node_type) {
                    let mut transformed = features.matmul(&transform.clone_data()).unwrap();

                    // Add bias if present
                    if let Some(Some(bias)) = self.biases.get(node_type) {
                        transformed = transformed.add(&bias.clone_data()).unwrap();
                    }

                    transformed_features.insert(node_type.clone(), transformed);
                }
            }
        }

        // Step 2: Message passing for each edge type
        let mut aggregated_messages = HashMap::new();

        for edge_type in &self.edge_types {
            let (src_type, relation, dst_type) = edge_type;

            if let (Some(edge_index), Some(src_features), Some(edge_transform)) = (
                hetero_graph.edge_indices.get(edge_type),
                transformed_features.get(src_type),
                self.edge_transformations.get(edge_type),
            ) {
                // Get edge connections
                let edge_flat = edge_index.to_vec().unwrap();
                let num_edges = edge_flat.len() / 2;

                if num_edges > 0 {
                    let src_indices = &edge_flat[0..num_edges];
                    let dst_indices = &edge_flat[num_edges..];

                    // Initialize aggregated messages for destination nodes
                    let dst_num_nodes = hetero_graph.num_nodes.get(dst_type).unwrap_or(&0);
                    let mut messages = zeros(&[*dst_num_nodes, self.out_features]).unwrap();

                    // Compute and aggregate messages
                    for edge_idx in 0..num_edges {
                        let src_node = src_indices[edge_idx] as usize;
                        let dst_node = dst_indices[edge_idx] as usize;

                        // Extract source node features
                        let src_feat = src_features
                            .slice_tensor(0, src_node, src_node + 1)
                            .unwrap()
                            .squeeze_tensor(0)
                            .unwrap();

                        // Apply relation-specific transformation
                        let message = src_feat
                            .unsqueeze_tensor(0)
                            .unwrap()
                            .matmul(&edge_transform.clone_data())
                            .unwrap()
                            .squeeze_tensor(0)
                            .unwrap();

                        // Aggregate to destination node
                        let mut dst_slice =
                            messages.slice_tensor(0, dst_node, dst_node + 1).unwrap();
                        let current_msg = dst_slice.squeeze_tensor(0).unwrap();
                        let updated_msg = current_msg.add(&message).unwrap();
                        dst_slice.copy_(&updated_msg.unsqueeze_tensor(0).unwrap());
                    }

                    // Store aggregated messages
                    aggregated_messages.insert(
                        (src_type.clone(), relation.clone(), dst_type.clone()),
                        messages,
                    );
                }
            }
        }

        // Step 3: Combine self-features with aggregated messages
        for node_type in &self.node_types {
            let mut node_output = if let Some(self_features) = transformed_features.get(node_type) {
                self_features.clone()
            } else {
                continue;
            };

            // Add messages from all relevant edge types
            for edge_type in &self.edge_types {
                let (_, _, dst_type) = edge_type;
                if dst_type == node_type {
                    if let Some(messages) = aggregated_messages.get(edge_type) {
                        node_output = node_output.add(messages).unwrap();
                    }
                }
            }

            // Apply activation (ReLU)
            let zero_tensor = zeros(node_output.shape().dims()).unwrap();
            node_output = node_output.maximum(&zero_tensor).unwrap();

            output_features.insert(node_type.clone(), node_output);
        }

        // Create output heterogeneous graph
        let mut output = HeteroGraphData::new();
        output.node_features = output_features;
        output.edge_indices = hetero_graph.edge_indices.clone();
        output.edge_attributes = hetero_graph.edge_attributes.clone();
        output.num_nodes = hetero_graph.num_nodes.clone();

        output
    }

    /// Get all parameters for optimization
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // Add node transformation parameters
        for transform in self.node_transformations.values() {
            params.push(transform.clone_data());
        }

        // Add edge transformation parameters
        for transform in self.edge_transformations.values() {
            params.push(transform.clone_data());
        }

        // Add bias parameters
        for bias_opt in self.biases.values() {
            if let Some(bias) = bias_opt {
                params.push(bias.clone_data());
            }
        }

        params
    }
}

/// Heterogeneous Graph Attention Network
pub struct HeteroGAT {
    node_types: Vec<NodeType>,
    edge_types: Vec<EdgeType>,
    /// Type-specific query/key/value transformations
    query_transforms: HashMap<NodeType, Parameter>,
    key_transforms: HashMap<NodeType, Parameter>,
    value_transforms: HashMap<NodeType, Parameter>,
    /// Relation-specific attention parameters
    relation_attentions: HashMap<EdgeType, Parameter>,
    /// Attention heads
    heads: usize,
    /// Output features per head
    out_features: usize,
    /// Dropout rate
    dropout: f32,
}

impl HeteroGAT {
    /// Create a new heterogeneous GAT layer
    pub fn new(
        node_type_dims: HashMap<NodeType, usize>,
        edge_types: Vec<EdgeType>,
        out_features: usize,
        heads: usize,
        dropout: f32,
    ) -> Self {
        let mut query_transforms = HashMap::new();
        let mut key_transforms = HashMap::new();
        let mut value_transforms = HashMap::new();

        // Create Q, K, V transformations for each node type
        for (node_type, in_features) in &node_type_dims {
            let q = Parameter::new(randn(&[*in_features, heads * out_features]).unwrap());
            let k = Parameter::new(randn(&[*in_features, heads * out_features]).unwrap());
            let v = Parameter::new(randn(&[*in_features, heads * out_features]).unwrap());

            query_transforms.insert(node_type.clone(), q);
            key_transforms.insert(node_type.clone(), k);
            value_transforms.insert(node_type.clone(), v);
        }

        // Create relation-specific attention parameters
        let mut relation_attentions = HashMap::new();
        for edge_type in &edge_types {
            let attention = Parameter::new(randn(&[heads, 2 * out_features]).unwrap());
            relation_attentions.insert(edge_type.clone(), attention);
        }

        Self {
            node_types: node_type_dims.keys().cloned().collect(),
            edge_types,
            query_transforms,
            key_transforms,
            value_transforms,
            relation_attentions,
            heads,
            out_features,
            dropout,
        }
    }

    /// Forward pass with heterogeneous attention
    pub fn forward(&self, hetero_graph: &HeteroGraphData) -> HeteroGraphData {
        let mut output_features = HashMap::new();

        // Step 1: Compute Q, K, V for all node types
        let mut queries = HashMap::new();
        let mut keys = HashMap::new();
        let mut values = HashMap::new();

        for node_type in &self.node_types {
            if let Some(features) = hetero_graph.node_features.get(node_type) {
                let q = features
                    .matmul(&self.query_transforms[node_type].clone_data())
                    .unwrap();
                let k = features
                    .matmul(&self.key_transforms[node_type].clone_data())
                    .unwrap();
                let v = features
                    .matmul(&self.value_transforms[node_type].clone_data())
                    .unwrap();

                // Reshape for multi-head attention [num_nodes, heads, out_features]
                let num_nodes = features.shape().dims()[0];
                let q_reshaped = q
                    .view(&[
                        num_nodes as i32,
                        self.heads as i32,
                        self.out_features as i32,
                    ])
                    .unwrap();
                let k_reshaped = k
                    .view(&[
                        num_nodes as i32,
                        self.heads as i32,
                        self.out_features as i32,
                    ])
                    .unwrap();
                let v_reshaped = v
                    .view(&[
                        num_nodes as i32,
                        self.heads as i32,
                        self.out_features as i32,
                    ])
                    .unwrap();

                queries.insert(node_type.clone(), q_reshaped);
                keys.insert(node_type.clone(), k_reshaped);
                values.insert(node_type.clone(), v_reshaped);
            }
        }

        // Step 2: Compute attention and aggregate for each edge type
        for dst_type in &self.node_types {
            let dst_num_nodes = hetero_graph.num_nodes.get(dst_type).unwrap_or(&0);
            let mut aggregated_output =
                zeros(&[*dst_num_nodes, self.heads * self.out_features]).unwrap();

            // Aggregate from all edge types that target this node type
            for edge_type in &self.edge_types {
                let (src_type, _relation, target_type) = edge_type;

                if target_type != dst_type {
                    continue;
                }

                if let (
                    Some(edge_index),
                    Some(src_queries),
                    Some(dst_keys),
                    Some(src_values),
                    Some(attention_params),
                ) = (
                    hetero_graph.edge_indices.get(edge_type),
                    queries.get(src_type),
                    keys.get(dst_type),
                    values.get(src_type),
                    self.relation_attentions.get(edge_type),
                ) {
                    // For simplicity, use mean aggregation with attention weights
                    // In a full implementation, this would compute proper attention scores

                    let edge_flat = edge_index.to_vec().unwrap();
                    let num_edges = edge_flat.len() / 2;

                    if num_edges > 0 {
                        let src_indices = &edge_flat[0..num_edges];
                        let dst_indices = &edge_flat[num_edges..];

                        // Simple aggregation (placeholder for full attention mechanism)
                        for edge_idx in 0..num_edges {
                            let src_node = src_indices[edge_idx] as usize;
                            let dst_node = dst_indices[edge_idx] as usize;

                            // Extract source value for aggregation
                            let src_value = src_values
                                .slice_tensor(0, src_node, src_node + 1)
                                .unwrap()
                                .view(&[1, (self.heads * self.out_features) as i32])
                                .unwrap()
                                .squeeze_tensor(0)
                                .unwrap();

                            // Add to destination (simple sum for now)
                            let mut dst_slice = aggregated_output
                                .slice_tensor(0, dst_node, dst_node + 1)
                                .unwrap();
                            let current = dst_slice.squeeze_tensor(0).unwrap();
                            let updated = current.add(&src_value).unwrap();
                            dst_slice.copy_(&updated.unsqueeze_tensor(0).unwrap());
                        }
                    }
                }
            }

            output_features.insert(dst_type.clone(), aggregated_output);
        }

        // Create output
        let mut output = HeteroGraphData::new();
        output.node_features = output_features;
        output.edge_indices = hetero_graph.edge_indices.clone();
        output.edge_attributes = hetero_graph.edge_attributes.clone();
        output.num_nodes = hetero_graph.num_nodes.clone();

        output
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // Add Q, K, V parameters
        for transform in self.query_transforms.values() {
            params.push(transform.clone_data());
        }
        for transform in self.key_transforms.values() {
            params.push(transform.clone_data());
        }
        for transform in self.value_transforms.values() {
            params.push(transform.clone_data());
        }

        // Add attention parameters
        for attention in self.relation_attentions.values() {
            params.push(attention.clone_data());
        }

        params
    }
}

/// Knowledge Graph Embedding layer
pub struct KnowledgeGraphEmbedding {
    entity_types: Vec<NodeType>,
    relation_types: Vec<String>,
    /// Entity embeddings
    entity_embeddings: HashMap<NodeType, Parameter>,
    /// Relation embeddings
    relation_embeddings: HashMap<String, Parameter>,
    /// Embedding dimension
    embedding_dim: usize,
}

impl KnowledgeGraphEmbedding {
    /// Create new knowledge graph embeddings
    pub fn new(
        entity_types: Vec<NodeType>,
        relation_types: Vec<String>,
        num_entities: HashMap<NodeType, usize>,
        embedding_dim: usize,
    ) -> Self {
        let mut entity_embeddings = HashMap::new();
        let mut relation_embeddings = HashMap::new();

        // Create entity embeddings
        for entity_type in &entity_types {
            let num = num_entities.get(entity_type).unwrap_or(&100);
            let embeddings = Parameter::new(randn(&[*num, embedding_dim]).unwrap());
            entity_embeddings.insert(entity_type.clone(), embeddings);
        }

        // Create relation embeddings
        for relation in &relation_types {
            let embeddings = Parameter::new(randn(&[embedding_dim, embedding_dim]).unwrap());
            relation_embeddings.insert(relation.clone(), embeddings);
        }

        Self {
            entity_types,
            relation_types,
            entity_embeddings,
            relation_embeddings,
            embedding_dim,
        }
    }

    /// Get entity embedding
    pub fn get_entity_embedding(&self, entity_type: &NodeType, entity_id: usize) -> Option<Tensor> {
        if let Some(embeddings) = self.entity_embeddings.get(entity_type) {
            Some(
                embeddings
                    .clone_data()
                    .slice_tensor(0, entity_id, entity_id + 1)
                    .unwrap()
                    .squeeze_tensor(0)
                    .unwrap(),
            )
        } else {
            None
        }
    }

    /// Compute triple score (head, relation, tail)
    pub fn triple_score(
        &self,
        head_type: &NodeType,
        head_id: usize,
        relation: &String,
        tail_type: &NodeType,
        tail_id: usize,
    ) -> Option<f64> {
        if let (Some(head_emb), Some(tail_emb), Some(rel_emb)) = (
            self.get_entity_embedding(head_type, head_id),
            self.get_entity_embedding(tail_type, tail_id),
            self.relation_embeddings.get(relation),
        ) {
            // Simple TransE-style scoring: ||h + r - t||
            let head_plus_rel = head_emb
                .unsqueeze_tensor(0)
                .unwrap()
                .matmul(&rel_emb.clone_data())
                .unwrap()
                .squeeze_tensor(0)
                .unwrap();

            let diff = head_plus_rel.sub(&tail_emb).unwrap();
            let score_tensor = diff.dot(&diff).unwrap();
            let score = score_tensor.to_vec().unwrap()[0] as f64;

            Some(-score) // Negative distance as score
        } else {
            None
        }
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for emb in self.entity_embeddings.values() {
            params.push(emb.clone_data());
        }

        for emb in self.relation_embeddings.values() {
            params.push(emb.clone_data());
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
    fn test_hetero_graph_creation() {
        let mut hetero_graph = HeteroGraphData::new();

        // Add user nodes
        let user_features = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu).unwrap();
        hetero_graph.add_node_type("user".to_string(), user_features);

        // Add item nodes
        let item_features = from_vec(
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        hetero_graph.add_node_type("item".to_string(), item_features);

        // Add user-item edges
        let edge_index = from_vec(vec![0.0, 1.0, 0.0, 1.0], &[2, 2], DeviceType::Cpu).unwrap();
        hetero_graph.add_edge_type(
            ("user".to_string(), "likes".to_string(), "item".to_string()),
            edge_index,
            None,
        );

        assert_eq!(hetero_graph.node_types().len(), 2);
        assert_eq!(hetero_graph.edge_types().len(), 1);
    }

    #[test]
    fn test_hetero_gnn_creation() {
        let mut node_dims = HashMap::new();
        node_dims.insert("user".to_string(), 2);
        node_dims.insert("item".to_string(), 3);

        let edge_types = vec![("user".to_string(), "likes".to_string(), "item".to_string())];

        let hetero_gnn = HeteroGNN::new(node_dims, edge_types, 8, true);
        let params = hetero_gnn.parameters();

        // Should have transformations for 2 node types + 1 edge type + biases
        assert!(params.len() >= 4);
    }

    #[test]
    fn test_knowledge_graph_embeddings() {
        let entity_types = vec!["person".to_string(), "company".to_string()];
        let relation_types = vec!["works_at".to_string(), "founded".to_string()];

        let mut num_entities = HashMap::new();
        num_entities.insert("person".to_string(), 10);
        num_entities.insert("company".to_string(), 5);

        let kg_emb = KnowledgeGraphEmbedding::new(entity_types, relation_types, num_entities, 50);

        // Test embedding retrieval
        let person_emb = kg_emb.get_entity_embedding(&"person".to_string(), 0);
        assert!(person_emb.is_some());

        let emb = person_emb.unwrap();
        assert_eq!(emb.shape().dims(), &[50]);

        // Test triple scoring
        let score = kg_emb.triple_score(
            &"person".to_string(),
            0,
            &"works_at".to_string(),
            &"company".to_string(),
            0,
        );
        assert!(score.is_some());
        assert!(score.unwrap().is_finite());
    }
}
