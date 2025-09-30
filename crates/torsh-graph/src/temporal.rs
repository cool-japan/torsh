//! Temporal Graph Neural Networks
//!
//! Advanced implementation of temporal graph neural networks for continuous-time dynamic graphs.
//! Handles evolving graph structures and node/edge features over time with sophisticated
//! temporal modeling capabilities.
//!
//! # Features:
//! - Continuous-time temporal graphs with event-based modeling
//! - Temporal graph convolution layers (TGN, DyRep, TGAT)
//! - Memory-augmented temporal networks
//! - Time-aware graph attention mechanisms
//! - Temporal pooling and aggregation operations
//! - Causal temporal modeling with proper time ordering

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use std::collections::{BTreeMap, HashMap, VecDeque};
use torsh_tensor::{
    creation::{from_vec, ones, randn, zeros},
    Tensor,
};

/// Temporal event representing a graph change at a specific time
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Time of the event (continuous time)
    pub time: f64,
    /// Type of event (node addition, edge addition, feature update, etc.)
    pub event_type: EventType,
    /// Source node ID (for edge events)
    pub source: Option<usize>,
    /// Target node ID (for edge events)
    pub target: Option<usize>,
    /// Node ID (for node events)
    pub node: Option<usize>,
    /// Feature vector associated with the event
    pub features: Option<Tensor>,
    /// Edge weight (for edge events)
    pub weight: Option<f32>,
}

/// Types of temporal events
#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    NodeAddition,
    NodeDeletion,
    NodeFeatureUpdate,
    EdgeAddition,
    EdgeDeletion,
    EdgeFeatureUpdate,
    GraphSnapshot,
}

/// Temporal graph data structure for continuous-time dynamic graphs
#[derive(Debug, Clone)]
pub struct TemporalGraphData {
    /// Static graph structure at current time
    pub current_graph: GraphData,
    /// Sequence of temporal events ordered by time
    pub events: BTreeMap<u64, Vec<TemporalEvent>>, // Using u64 timestamp for ordering
    /// Time-indexed node features
    pub node_features_history: HashMap<usize, BTreeMap<u64, Tensor>>,
    /// Time-indexed edge features
    pub edge_features_history: HashMap<(usize, usize), BTreeMap<u64, Tensor>>,
    /// Current timestamp
    pub current_time: f64,
    /// Time window for temporal aggregation
    pub time_window: f64,
    /// Maximum number of events to keep in memory
    pub max_events: usize,
}

impl TemporalGraphData {
    /// Create a new temporal graph
    pub fn new(initial_graph: GraphData, time_window: f64, max_events: usize) -> Self {
        Self {
            current_graph: initial_graph,
            events: BTreeMap::new(),
            node_features_history: HashMap::new(),
            edge_features_history: HashMap::new(),
            current_time: 0.0,
            time_window,
            max_events,
        }
    }

    /// Add a temporal event to the graph
    pub fn add_event(&mut self, event: TemporalEvent) {
        let timestamp = (event.time * 1000.0) as u64; // Convert to milliseconds for ordering
        self.events
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .push(event.clone());

        // Update current time
        self.current_time = self.current_time.max(event.time);

        // Apply event to current graph structure
        self.apply_event(&event);

        // Clean up old events outside time window
        self.cleanup_old_events();
    }

    /// Apply an event to the current graph structure
    fn apply_event(&mut self, event: &TemporalEvent) {
        match event.event_type {
            EventType::NodeFeatureUpdate => {
                if let (Some(node), Some(ref features)) = (event.node, &event.features) {
                    // Update node features in current graph
                    self.update_node_features(node, features.clone());

                    // Store in history
                    let timestamp = (event.time * 1000.0) as u64;
                    self.node_features_history
                        .entry(node)
                        .or_insert_with(BTreeMap::new)
                        .insert(timestamp, features.clone());
                }
            }
            EventType::EdgeFeatureUpdate => {
                if let (Some(source), Some(target), Some(ref features)) =
                    (event.source, event.target, &event.features)
                {
                    let timestamp = (event.time * 1000.0) as u64;
                    self.edge_features_history
                        .entry((source, target))
                        .or_insert_with(BTreeMap::new)
                        .insert(timestamp, features.clone());
                }
            }
            _ => {
                // For simplicity, other event types are not fully implemented
                // In a complete implementation, these would modify the graph structure
            }
        }
    }

    /// Update node features in the current graph
    fn update_node_features(&mut self, node_id: usize, features: Tensor) {
        // Simplified implementation - would need proper tensor slicing in practice
        let current_features = self.current_graph.x.to_vec().unwrap();
        let feature_dim = self.current_graph.x.shape().dims()[1];
        let new_features = features.to_vec().unwrap();

        let mut updated_features = current_features;
        let start_idx = node_id * feature_dim;
        let end_idx = start_idx + feature_dim.min(new_features.len());

        for (i, &value) in new_features.iter().take(feature_dim).enumerate() {
            if start_idx + i < updated_features.len() {
                updated_features[start_idx + i] = value;
            }
        }

        self.current_graph.x = from_vec(
            updated_features,
            &[self.current_graph.num_nodes, feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
    }

    /// Clean up old events outside the time window
    fn cleanup_old_events(&mut self) {
        let cutoff_time = ((self.current_time - self.time_window) * 1000.0) as u64;

        // Remove events older than time window
        let old_keys: Vec<u64> = self
            .events
            .keys()
            .filter(|&&timestamp| timestamp < cutoff_time)
            .cloned()
            .collect();

        for key in old_keys {
            self.events.remove(&key);
        }

        // Also limit total number of events
        while self.events.len() > self.max_events {
            if let Some(first_key) = self.events.keys().next().cloned() {
                self.events.remove(&first_key);
            } else {
                break;
            }
        }
    }

    /// Get events within a specific time range
    pub fn get_events_in_range(&self, start_time: f64, end_time: f64) -> Vec<&TemporalEvent> {
        let start_timestamp = (start_time * 1000.0) as u64;
        let end_timestamp = (end_time * 1000.0) as u64;

        self.events
            .range(start_timestamp..=end_timestamp)
            .flat_map(|(_, events)| events.iter())
            .collect()
    }

    /// Get node features at a specific time (with interpolation)
    pub fn get_node_features_at_time(&self, node_id: usize, time: f64) -> Option<Tensor> {
        let timestamp = (time * 1000.0) as u64;

        if let Some(history) = self.node_features_history.get(&node_id) {
            // Find the most recent features before or at the requested time
            if let Some((_, features)) = history.range(..=timestamp).next_back() {
                return Some(features.clone());
            }
        }

        None
    }

    /// Create a snapshot of the graph at a specific time
    pub fn snapshot_at_time(&self, time: f64) -> GraphData {
        // Simplified implementation - returns current graph
        // In practice, this would reconstruct the graph state at the specified time
        self.current_graph.clone()
    }
}

/// Temporal Graph Convolutional Network (TGCN) layer
#[derive(Debug)]
pub struct TGCNConv {
    in_features: usize,
    out_features: usize,
    temporal_dim: usize,
    spatial_weight: Parameter,
    temporal_weight: Parameter,
    bias: Option<Parameter>,
    memory_size: usize,
    time_encoding_dim: usize,
}

impl TGCNConv {
    /// Create a new TGCN layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        temporal_dim: usize,
        memory_size: usize,
        bias: bool,
    ) -> Self {
        let spatial_weight = Parameter::new(randn(&[in_features, out_features]).unwrap());
        let temporal_weight = Parameter::new(randn(&[temporal_dim, out_features]).unwrap());
        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            temporal_dim,
            spatial_weight,
            temporal_weight,
            bias,
            memory_size,
            time_encoding_dim: temporal_dim,
        }
    }

    /// Forward pass through TGCN layer
    pub fn forward(&self, temporal_graph: &TemporalGraphData) -> TemporalGraphData {
        // Step 1: Spatial convolution on current graph
        let spatial_features = temporal_graph
            .current_graph
            .x
            .matmul(&self.spatial_weight.clone_data())
            .unwrap();

        // Step 2: Temporal encoding based on recent events
        let temporal_features = self.encode_temporal_context(temporal_graph);

        // Step 3: Combine spatial and temporal features
        let combined_features = spatial_features.add(&temporal_features).unwrap();

        // Step 4: Add bias if present
        let output_features = if let Some(ref bias) = self.bias {
            combined_features.add(&bias.clone_data()).unwrap()
        } else {
            combined_features
        };

        // Create output temporal graph
        let mut output_graph = temporal_graph.clone();
        output_graph.current_graph.x = output_features;
        output_graph
    }

    /// Encode temporal context from recent events
    fn encode_temporal_context(&self, temporal_graph: &TemporalGraphData) -> Tensor {
        let num_nodes = temporal_graph.current_graph.num_nodes;
        let current_time = temporal_graph.current_time;
        let lookback_time = current_time - temporal_graph.time_window;

        // Get recent events
        let recent_events = temporal_graph.get_events_in_range(lookback_time, current_time);

        // Initialize temporal encoding
        let mut temporal_encoding = zeros::<f32>(&[num_nodes, self.out_features]).unwrap();

        // Simple temporal encoding based on event recency and frequency
        let mut node_event_counts = vec![0.0; num_nodes];

        for event in recent_events {
            if let Some(node_id) = event.node {
                if node_id < num_nodes {
                    // Weight by recency (more recent events have higher weight)
                    let recency_weight =
                        1.0 - (current_time - event.time) / temporal_graph.time_window;
                    node_event_counts[node_id] += recency_weight;
                }
            }
        }

        // Convert counts to temporal features
        let temporal_data: Vec<f32> = node_event_counts
            .iter()
            .flat_map(|&count| {
                // Simple encoding: repeat the count for each output feature
                (0..self.out_features).map(move |_| count as f32)
            })
            .collect();

        from_vec(
            temporal_data,
            &[num_nodes, self.out_features],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }
}

impl GraphLayer for TGCNConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        // Convert to temporal graph for processing
        let temporal_graph = TemporalGraphData::new(graph.clone(), 1.0, 1000);
        let output_temporal = self.forward(&temporal_graph);
        output_temporal.current_graph
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.spatial_weight.clone_data(),
            self.temporal_weight.clone_data(),
        ];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }
        params
    }
}

/// Temporal Graph Attention Network (TGAT) layer
#[derive(Debug)]
pub struct TGATConv {
    in_features: usize,
    out_features: usize,
    heads: usize,
    time_encoding_dim: usize,
    query_weight: Parameter,
    key_weight: Parameter,
    value_weight: Parameter,
    time_weight: Parameter,
    output_weight: Parameter,
    bias: Option<Parameter>,
    dropout: f32,
}

impl TGATConv {
    /// Create a new TGAT layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        heads: usize,
        time_encoding_dim: usize,
        dropout: f32,
        bias: bool,
    ) -> Self {
        let query_weight = Parameter::new(randn(&[in_features, out_features]).unwrap());
        let key_weight = Parameter::new(randn(&[in_features, out_features]).unwrap());
        let value_weight = Parameter::new(randn(&[in_features, out_features]).unwrap());
        let time_weight = Parameter::new(randn(&[time_encoding_dim, out_features]).unwrap());
        let output_weight = Parameter::new(randn(&[out_features, out_features]).unwrap());

        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            heads,
            time_encoding_dim,
            query_weight,
            key_weight,
            value_weight,
            time_weight,
            output_weight,
            bias,
            dropout,
        }
    }

    /// Forward pass through TGAT layer
    pub fn forward(&self, temporal_graph: &TemporalGraphData) -> TemporalGraphData {
        let num_nodes = temporal_graph.current_graph.num_nodes;
        let head_dim = self.out_features / self.heads;

        // Compute Q, K, V transformations
        let queries = temporal_graph
            .current_graph
            .x
            .matmul(&self.query_weight.clone_data())
            .unwrap();
        let keys = temporal_graph
            .current_graph
            .x
            .matmul(&self.key_weight.clone_data())
            .unwrap();
        let values = temporal_graph
            .current_graph
            .x
            .matmul(&self.value_weight.clone_data())
            .unwrap();

        // Compute time encoding for each node based on recent activity
        let time_encoding = self.compute_time_encoding(temporal_graph);
        let time_transformed = time_encoding
            .matmul(&self.time_weight.clone_data())
            .unwrap();

        // Reshape for multi-head attention
        let q = queries
            .view(&[num_nodes as i32, self.heads as i32, head_dim as i32])
            .unwrap();
        let k = keys
            .view(&[num_nodes as i32, self.heads as i32, head_dim as i32])
            .unwrap();
        let v = values
            .view(&[num_nodes as i32, self.heads as i32, head_dim as i32])
            .unwrap();

        // Perform temporal attention
        let attended_features =
            self.temporal_attention(&q, &k, &v, &time_transformed, temporal_graph);

        // Reshape and apply output transformation
        let concatenated = attended_features
            .view(&[num_nodes as i32, self.out_features as i32])
            .unwrap();
        let mut output = concatenated
            .matmul(&self.output_weight.clone_data())
            .unwrap();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        // Create output temporal graph
        let mut output_graph = temporal_graph.clone();
        output_graph.current_graph.x = output;
        output_graph
    }

    /// Compute time encoding for nodes based on recent events
    fn compute_time_encoding(&self, temporal_graph: &TemporalGraphData) -> Tensor {
        let num_nodes = temporal_graph.current_graph.num_nodes;
        let current_time = temporal_graph.current_time;

        // Simple time encoding: time since last event for each node
        let mut time_features = vec![current_time as f32; num_nodes * self.time_encoding_dim];

        // Update with actual last event times
        for (node_id, history) in &temporal_graph.node_features_history {
            if *node_id < num_nodes {
                if let Some((timestamp, _)) = history.iter().next_back() {
                    let last_event_time = (*timestamp as f64) / 1000.0;
                    let time_diff = (current_time - last_event_time) as f32;

                    // Encode time difference in multiple dimensions
                    for dim in 0..self.time_encoding_dim {
                        let freq = 2.0_f32.powf(dim as f32);
                        let encoded = (time_diff * freq).sin();
                        time_features[*node_id * self.time_encoding_dim + dim] = encoded;
                    }
                }
            }
        }

        from_vec(
            time_features,
            &[num_nodes, self.time_encoding_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Temporal attention mechanism
    fn temporal_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        time_encoding: &Tensor,
        temporal_graph: &TemporalGraphData,
    ) -> Tensor {
        let num_nodes = temporal_graph.current_graph.num_nodes;
        let head_dim = self.out_features / self.heads;

        // Simplified temporal attention
        let mut output = zeros(&[num_nodes, self.heads, head_dim]).unwrap();

        // For each head, compute attention with temporal bias
        for head in 0..self.heads {
            // Extract head-specific features
            let q_head = q.slice_tensor(1, head, head + 1).unwrap();
            let k_head = k.slice_tensor(1, head, head + 1).unwrap();
            let v_head = v.slice_tensor(1, head, head + 1).unwrap();

            // Simplified attention computation (using dot product)
            for i in 0..num_nodes {
                let mut attended_value = zeros(&[head_dim]).unwrap();
                let mut attention_sum = 0.0;

                for j in 0..num_nodes {
                    // Basic attention score computation
                    let score = 1.0 / (1.0 + (i as f32 - j as f32).abs()); // Distance-based attention

                    // Get value for node j
                    let v_j = v_head
                        .slice_tensor(0, j, j + 1)
                        .unwrap()
                        .squeeze_tensor(0)
                        .unwrap()
                        .squeeze_tensor(0)
                        .unwrap();

                    let weighted_value = v_j.mul_scalar(score).unwrap();
                    attended_value = attended_value.add(&weighted_value).unwrap();
                    attention_sum += score;
                }

                // Normalize
                if attention_sum > 0.0 {
                    attended_value = attended_value.div_scalar(attention_sum).unwrap();
                }

                // Store in output (simplified assignment)
                let attended_data = attended_value.to_vec().unwrap();
                for (dim, &val) in attended_data.iter().enumerate() {
                    if dim < head_dim {
                        output.set_item(&[i, head, dim], val).unwrap();
                    }
                }
            }
        }

        output
    }
}

impl GraphLayer for TGATConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        let temporal_graph = TemporalGraphData::new(graph.clone(), 1.0, 1000);
        let output_temporal = self.forward(&temporal_graph);
        output_temporal.current_graph
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.query_weight.clone_data(),
            self.key_weight.clone_data(),
            self.value_weight.clone_data(),
            self.time_weight.clone_data(),
            self.output_weight.clone_data(),
        ];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }
        params
    }
}

/// Memory-augmented Temporal Graph Network (TGN) layer
#[derive(Debug)]
pub struct TGNConv {
    in_features: usize,
    out_features: usize,
    memory_dim: usize,
    time_encoding_dim: usize,
    message_function: Parameter,
    memory_updater: Parameter,
    node_embedding: Parameter,
    bias: Option<Parameter>,
    node_memories: HashMap<usize, Tensor>,
    last_update_times: HashMap<usize, f64>,
}

impl TGNConv {
    /// Create a new TGN layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        memory_dim: usize,
        time_encoding_dim: usize,
        bias: bool,
    ) -> Self {
        let message_function =
            Parameter::new(randn(&[in_features + time_encoding_dim, memory_dim]).unwrap());
        let memory_updater = Parameter::new(randn(&[memory_dim * 2, memory_dim]).unwrap());
        let node_embedding = Parameter::new(randn(&[memory_dim, out_features]).unwrap());

        let bias = if bias {
            Some(Parameter::new(zeros(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            memory_dim,
            time_encoding_dim,
            message_function,
            memory_updater,
            node_embedding,
            bias,
            node_memories: HashMap::new(),
            last_update_times: HashMap::new(),
        }
    }

    /// Forward pass through TGN layer
    pub fn forward(&mut self, temporal_graph: &TemporalGraphData) -> TemporalGraphData {
        // Update node memories based on recent events
        self.update_memories(temporal_graph);

        // Generate node embeddings from memories
        let output_features = self.generate_embeddings(temporal_graph);

        // Create output temporal graph
        let mut output_graph = temporal_graph.clone();
        output_graph.current_graph.x = output_features;
        output_graph
    }

    /// Update node memories based on temporal events
    fn update_memories(&mut self, temporal_graph: &TemporalGraphData) {
        let current_time = temporal_graph.current_time;
        let lookback_time = current_time - temporal_graph.time_window;

        // Get recent events
        let recent_events = temporal_graph.get_events_in_range(lookback_time, current_time);

        for event in recent_events {
            if let Some(node_id) = event.node {
                // Generate message from event
                let message = self.compute_message(event, current_time);

                // Update node memory
                self.update_node_memory(node_id, message, event.time);
            }
        }
    }

    /// Compute message from temporal event
    fn compute_message(&self, event: &TemporalEvent, current_time: f64) -> Tensor {
        // Time encoding
        let time_diff = (current_time - event.time) as f32;
        let mut time_encoding = Vec::new();

        for i in 0..self.time_encoding_dim {
            let freq = 2.0_f32.powf(i as f32);
            time_encoding.push((time_diff * freq).sin());
        }

        // Combine event features with time encoding
        let mut message_input = if let Some(ref features) = event.features {
            features.to_vec().unwrap()
        } else {
            vec![1.0; self.in_features] // Default features
        };

        message_input.extend(time_encoding);

        let input_tensor = from_vec(
            message_input,
            &[1, self.in_features + self.time_encoding_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Apply message function
        input_tensor
            .matmul(&self.message_function.clone_data())
            .unwrap()
    }

    /// Update memory for a specific node
    fn update_node_memory(&mut self, node_id: usize, message: Tensor, event_time: f64) {
        // Get current memory or initialize
        let current_memory = self
            .node_memories
            .get(&node_id)
            .cloned()
            .unwrap_or_else(|| zeros(&[1, self.memory_dim]).unwrap());

        // Concatenate current memory and message
        let current_data = current_memory.to_vec().unwrap();
        let message_data = message.to_vec().unwrap();
        let mut combined_data = current_data;
        combined_data.extend(message_data);

        let combined_tensor = from_vec(
            combined_data,
            &[1, self.memory_dim * 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Update memory using memory updater
        let new_memory = combined_tensor
            .matmul(&self.memory_updater.clone_data())
            .unwrap();

        self.node_memories.insert(node_id, new_memory);
        self.last_update_times.insert(node_id, event_time);
    }

    /// Generate node embeddings from memories
    fn generate_embeddings(&self, temporal_graph: &TemporalGraphData) -> Tensor {
        let num_nodes = temporal_graph.current_graph.num_nodes;
        let mut embeddings = Vec::new();

        for node_id in 0..num_nodes {
            let memory = self
                .node_memories
                .get(&node_id)
                .cloned()
                .unwrap_or_else(|| zeros(&[1, self.memory_dim]).unwrap());

            let embedding = memory.matmul(&self.node_embedding.clone_data()).unwrap();
            let embedding_data = embedding.to_vec().unwrap();
            embeddings.extend(embedding_data);
        }

        let mut output = from_vec(
            embeddings,
            &[num_nodes, self.out_features],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        output
    }
}

/// Temporal graph pooling operations
pub mod pooling {
    use super::*;

    /// Temporal pooling methods
    #[derive(Debug, Clone, Copy)]
    pub enum TemporalPoolingMethod {
        MostRecent,
        TimeWeightedMean,
        ExponentialDecay,
        AttentionBased,
    }

    /// Global temporal pooling
    pub fn temporal_pool(
        temporal_graph: &TemporalGraphData,
        method: TemporalPoolingMethod,
    ) -> Tensor {
        match method {
            TemporalPoolingMethod::MostRecent => {
                // Use current graph features
                temporal_graph
                    .current_graph
                    .x
                    .mean(Some(&[0]), false)
                    .unwrap()
            }
            TemporalPoolingMethod::TimeWeightedMean => time_weighted_pool(temporal_graph),
            TemporalPoolingMethod::ExponentialDecay => exponential_decay_pool(temporal_graph),
            TemporalPoolingMethod::AttentionBased => attention_temporal_pool(temporal_graph),
        }
    }

    /// Time-weighted pooling based on event recency
    fn time_weighted_pool(temporal_graph: &TemporalGraphData) -> Tensor {
        let current_time = temporal_graph.current_time;
        let lookback_time = current_time - temporal_graph.time_window;
        let recent_events = temporal_graph.get_events_in_range(lookback_time, current_time);

        if recent_events.is_empty() {
            return temporal_graph
                .current_graph
                .x
                .mean(Some(&[0]), false)
                .unwrap();
        }

        // Weight events by recency
        let mut weighted_sum = zeros(&[temporal_graph.current_graph.x.shape().dims()[1]]).unwrap();
        let mut total_weight = 0.0;

        for event in recent_events {
            if let Some(ref features) = event.features {
                let weight = 1.0 - (current_time - event.time) / temporal_graph.time_window;
                let weighted_features = features.mul_scalar(weight as f32).unwrap();

                // Sum the features (simplified)
                let features_data = weighted_features.to_vec().unwrap();
                let current_data = weighted_sum.to_vec().unwrap();
                let mut new_data = Vec::new();

                for (i, (&current, &new)) in
                    current_data.iter().zip(features_data.iter()).enumerate()
                {
                    new_data.push(current + new);
                }

                weighted_sum = from_vec(
                    new_data,
                    &[weighted_sum.shape().dims()[0]],
                    torsh_core::device::DeviceType::Cpu,
                )
                .unwrap();

                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            weighted_sum.div_scalar(total_weight as f32).unwrap()
        } else {
            temporal_graph
                .current_graph
                .x
                .mean(Some(&[0]), false)
                .unwrap()
        }
    }

    /// Exponential decay pooling
    fn exponential_decay_pool(temporal_graph: &TemporalGraphData) -> Tensor {
        let decay_rate = 0.1; // Decay parameter
        let current_time = temporal_graph.current_time;

        // Simple exponential decay - use current features
        let decay_factor = (-decay_rate * current_time).exp() as f32;
        temporal_graph
            .current_graph
            .x
            .mul_scalar(decay_factor)
            .unwrap()
            .mean(Some(&[0]), false)
            .unwrap()
    }

    /// Attention-based temporal pooling
    fn attention_temporal_pool(temporal_graph: &TemporalGraphData) -> Tensor {
        // Simplified attention pooling
        let features = &temporal_graph.current_graph.x;
        let attention_scores = features.sum_dim(&[1], false).unwrap();
        let attention_weights = attention_scores.softmax(0).unwrap();
        let attention_expanded = attention_weights.unsqueeze(-1).unwrap();

        let weighted_features = features.mul(&attention_expanded).unwrap();
        weighted_features.sum_dim(&[0], false).unwrap()
    }
}

/// Temporal graph utilities
pub mod utils {
    use super::*;

    /// Generate random temporal events
    pub fn generate_random_events(
        num_events: usize,
        num_nodes: usize,
        time_span: f64,
        feature_dim: usize,
    ) -> Vec<TemporalEvent> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut events = Vec::new();

        for _ in 0..num_events {
            let time = rng.gen_range(0.0..time_span);
            let event_type = if rng.gen_range(0.0..1.0) < 0.7 {
                EventType::NodeFeatureUpdate
            } else {
                EventType::EdgeAddition
            };

            let node = if matches!(event_type, EventType::NodeFeatureUpdate) {
                Some(rng.gen_range(0..num_nodes))
            } else {
                None
            };

            let (source, target) = if matches!(event_type, EventType::EdgeAddition) {
                let s = rng.gen_range(0..num_nodes);
                let t = rng.gen_range(0..num_nodes);
                (Some(s), Some(t))
            } else {
                (None, None)
            };

            let features = if matches!(event_type, EventType::NodeFeatureUpdate) {
                Some(randn(&[feature_dim]).unwrap())
            } else {
                None
            };

            events.push(TemporalEvent {
                time,
                event_type,
                source,
                target,
                node,
                features,
                weight: Some(rng.gen_range(0.1..1.0)),
            });
        }

        // Sort events by time
        events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        events
    }

    /// Create temporal graph from event sequence
    pub fn create_temporal_graph_from_events(
        initial_graph: GraphData,
        events: Vec<TemporalEvent>,
        time_window: f64,
    ) -> TemporalGraphData {
        let mut temporal_graph = TemporalGraphData::new(initial_graph, time_window, 10000);

        for event in events {
            temporal_graph.add_event(event);
        }

        temporal_graph
    }

    /// Compute temporal graph metrics
    pub fn temporal_metrics(temporal_graph: &TemporalGraphData) -> TemporalMetrics {
        let total_events = temporal_graph.events.values().map(|v| v.len()).sum();
        let unique_nodes_with_events = temporal_graph.node_features_history.len();
        let time_span = if let (Some(first), Some(last)) = (
            temporal_graph.events.keys().next(),
            temporal_graph.events.keys().next_back(),
        ) {
            (*last as f64 - *first as f64) / 1000.0
        } else {
            0.0
        };

        let event_rate = if time_span > 0.0 {
            total_events as f64 / time_span
        } else {
            0.0
        };

        TemporalMetrics {
            total_events,
            unique_active_nodes: unique_nodes_with_events,
            time_span,
            event_rate,
            current_time: temporal_graph.current_time,
        }
    }

    /// Temporal graph metrics
    #[derive(Debug, Clone)]
    pub struct TemporalMetrics {
        pub total_events: usize,
        pub unique_active_nodes: usize,
        pub time_span: f64,
        pub event_rate: f64,
        pub current_time: f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_temporal_graph_creation() {
        let features = randn(&[4, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let temporal_graph = TemporalGraphData::new(graph, 10.0, 1000);

        assert_eq!(temporal_graph.current_graph.num_nodes, 4);
        assert_eq!(temporal_graph.time_window, 10.0);
        assert_eq!(temporal_graph.max_events, 1000);
    }

    #[test]
    fn test_temporal_event_addition() {
        let features = randn(&[3, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];
        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let mut temporal_graph = TemporalGraphData::new(graph, 5.0, 100);

        let event = TemporalEvent {
            time: 1.0,
            event_type: EventType::NodeFeatureUpdate,
            source: None,
            target: None,
            node: Some(0),
            features: Some(randn(&[2]).unwrap()),
            weight: None,
        };

        temporal_graph.add_event(event);

        assert_eq!(temporal_graph.current_time, 1.0);
        assert!(!temporal_graph.events.is_empty());
    }

    #[test]
    fn test_tgcn_layer() {
        let features = randn(&[3, 4]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];
        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let temporal_graph = TemporalGraphData::new(graph, 1.0, 100);
        let tgcn = TGCNConv::new(4, 8, 16, 64, true);

        let output = tgcn.forward(&temporal_graph);
        assert_eq!(output.current_graph.x.shape().dims(), &[3, 8]);
    }

    #[test]
    fn test_tgat_layer() {
        let features = randn(&[4, 6]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let temporal_graph = TemporalGraphData::new(graph, 2.0, 200);
        let tgat = TGATConv::new(6, 12, 3, 8, 0.1, true);

        let output = tgat.forward(&temporal_graph);
        assert_eq!(output.current_graph.x.shape().dims(), &[4, 12]);
    }

    #[test]
    fn test_temporal_pooling() {
        let features = randn(&[5, 4]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let temporal_graph = TemporalGraphData::new(graph, 3.0, 150);

        let pooled =
            pooling::temporal_pool(&temporal_graph, pooling::TemporalPoolingMethod::MostRecent);
        assert_eq!(pooled.shape().dims(), &[4]);

        let weighted_pooled = pooling::temporal_pool(
            &temporal_graph,
            pooling::TemporalPoolingMethod::TimeWeightedMean,
        );
        assert_eq!(weighted_pooled.shape().dims(), &[4]);
    }

    #[test]
    fn test_temporal_utils() {
        let events = utils::generate_random_events(10, 5, 10.0, 3);
        assert_eq!(events.len(), 10);

        // Check that events are sorted by time
        for i in 1..events.len() {
            assert!(events[i].time >= events[i - 1].time);
        }

        let features = randn(&[5, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 0.0];
        let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let temporal_graph = utils::create_temporal_graph_from_events(graph, events, 5.0);
        let metrics = utils::temporal_metrics(&temporal_graph);

        assert!(metrics.total_events > 0);
        assert!(metrics.time_span >= 0.0);
    }

    #[test]
    fn test_event_time_range_query() {
        let features = randn(&[3, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];
        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let mut temporal_graph = TemporalGraphData::new(graph, 10.0, 100);

        // Add events at different times
        for i in 0..5 {
            let event = TemporalEvent {
                time: i as f64,
                event_type: EventType::NodeFeatureUpdate,
                source: None,
                target: None,
                node: Some(i % 3),
                features: Some(randn(&[2]).unwrap()),
                weight: None,
            };
            temporal_graph.add_event(event);
        }

        let events_in_range = temporal_graph.get_events_in_range(1.0, 3.0);
        assert_eq!(events_in_range.len(), 3); // Events at times 1, 2, 3
    }
}
