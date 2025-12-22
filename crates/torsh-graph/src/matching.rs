//! Graph Matching and Similarity Learning
//!
//! Advanced implementation of graph matching algorithms and graph similarity
//! learning methods for comparing and aligning graph structures.
//!
//! # Features:
//! - Graph isomorphism testing and subgraph matching
//! - Graph edit distance computation
//! - Graph kernel methods for similarity
//! - Neural graph matching networks
//! - Graph alignment and correspondence learning
//! - Siamese and triplet networks for graph similarity

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::parameter::Parameter;
use crate::GraphData;
use std::collections::{HashMap, HashSet, VecDeque};
use torsh_tensor::{
    creation::{from_vec, randn, zeros},
    Tensor,
};

/// Graph Edit Distance (GED) computation
pub struct GraphEditDistance {
    /// Cost for node insertion/deletion
    pub node_cost: f32,
    /// Cost for edge insertion/deletion
    pub edge_cost: f32,
    /// Cost for node substitution
    pub node_subst_cost: f32,
    /// Cost for edge substitution
    pub edge_subst_cost: f32,
}

impl GraphEditDistance {
    /// Create a new GED calculator with default costs
    pub fn new() -> Self {
        Self {
            node_cost: 1.0,
            edge_cost: 1.0,
            node_subst_cost: 1.0,
            edge_subst_cost: 1.0,
        }
    }

    /// Compute approximate graph edit distance between two graphs
    pub fn compute(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        let n1 = graph1.num_nodes;
        let n2 = graph2.num_nodes;

        // Node operations cost
        let node_ops = ((n1 as i32 - n2 as i32).abs() as f32) * self.node_cost;

        // Edge operations cost (simplified)
        let e1 = graph1.num_edges;
        let e2 = graph2.num_edges;
        let edge_ops = ((e1 as i32 - e2 as i32).abs() as f32) * self.edge_cost;

        // Feature dissimilarity (using L2 distance)
        let feature_cost = self.compute_feature_distance(graph1, graph2);

        node_ops + edge_ops + feature_cost
    }

    /// Compute feature distance between graphs
    fn compute_feature_distance(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        let f1_data = graph1.x.to_vec().unwrap();
        let f2_data = graph2.x.to_vec().unwrap();

        let min_len = f1_data.len().min(f2_data.len());
        let mut dist = 0.0;

        for i in 0..min_len {
            dist += (f1_data[i] - f2_data[i]).powi(2);
        }

        // Add penalty for size mismatch
        dist += ((f1_data.len() as i32 - f2_data.len() as i32).abs() as f32) * self.node_subst_cost;

        dist.sqrt()
    }

    /// Find approximate node correspondence between two graphs
    pub fn node_correspondence(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
    ) -> Vec<(usize, usize)> {
        let mut correspondences = Vec::new();
        let n1 = graph1.num_nodes;
        let n2 = graph2.num_nodes;

        // Simple greedy matching based on feature similarity
        let mut matched_nodes2 = HashSet::new();

        for i in 0..n1 {
            let mut best_match = None;
            let mut best_similarity = f32::NEG_INFINITY;

            for j in 0..n2 {
                if matched_nodes2.contains(&j) {
                    continue;
                }

                let similarity = self.node_similarity(graph1, i, graph2, j);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_match = Some(j);
                }
            }

            if let Some(j) = best_match {
                correspondences.push((i, j));
                matched_nodes2.insert(j);
            }
        }

        correspondences
    }

    /// Compute similarity between two nodes
    fn node_similarity(
        &self,
        graph1: &GraphData,
        node1: usize,
        graph2: &GraphData,
        node2: usize,
    ) -> f32 {
        let f1 = graph1.x.slice_tensor(0, node1, node1 + 1).unwrap();
        let f2 = graph2.x.slice_tensor(0, node2, node2 + 1).unwrap();

        // Cosine similarity
        let dot = f1.dot(&f2.t().unwrap()).unwrap().item().unwrap();
        let norm1 = f1.norm().unwrap().item().unwrap();
        let norm2 = f2.norm().unwrap().item().unwrap();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

impl Default for GraphEditDistance {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph Kernel methods for similarity computation
pub struct GraphKernel {
    kernel_type: GraphKernelType,
}

#[derive(Debug, Clone, Copy)]
pub enum GraphKernelType {
    /// Random walk kernel
    RandomWalk,
    /// Shortest path kernel
    ShortestPath,
    /// Weisfeiler-Lehman kernel
    WeisfeilerLehman,
    /// Graphlet kernel
    Graphlet,
}

impl GraphKernel {
    /// Create a new graph kernel
    pub fn new(kernel_type: GraphKernelType) -> Self {
        Self { kernel_type }
    }

    /// Compute kernel similarity between two graphs
    pub fn compute(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        match self.kernel_type {
            GraphKernelType::RandomWalk => self.random_walk_kernel(graph1, graph2),
            GraphKernelType::ShortestPath => self.shortest_path_kernel(graph1, graph2),
            GraphKernelType::WeisfeilerLehman => self.wl_kernel(graph1, graph2),
            GraphKernelType::Graphlet => self.graphlet_kernel(graph1, graph2),
        }
    }

    /// Random walk kernel
    fn random_walk_kernel(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        // Simplified: count common random walk patterns
        let walks1 = self.sample_random_walks(graph1, 10, 5);
        let walks2 = self.sample_random_walks(graph2, 10, 5);

        let mut common_count = 0;
        for w1 in &walks1 {
            if walks2.contains(w1) {
                common_count += 1;
            }
        }

        common_count as f32 / (walks1.len() + walks2.len()) as f32
    }

    /// Sample random walks from a graph
    fn sample_random_walks(
        &self,
        graph: &GraphData,
        num_walks: usize,
        walk_length: usize,
    ) -> Vec<Vec<usize>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut walks = Vec::new();
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;
                adj_list.entry(src).or_insert_with(Vec::new).push(dst);
            }
        }

        // Sample walks
        for _ in 0..num_walks {
            if graph.num_nodes == 0 {
                break;
            }

            let mut walk = Vec::new();
            let mut current_node = rng.gen_range(0..graph.num_nodes);
            walk.push(current_node);

            for _ in 0..walk_length {
                if let Some(neighbors) = adj_list.get(&current_node) {
                    if neighbors.is_empty() {
                        break;
                    }
                    let idx = rng.gen_range(0..neighbors.len());
                    current_node = neighbors[idx];
                    walk.push(current_node);
                } else {
                    break;
                }
            }

            walks.push(walk);
        }

        walks
    }

    /// Shortest path kernel
    fn shortest_path_kernel(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        // Compare shortest path distributions
        let sp1 = self.compute_shortest_paths_distribution(graph1);
        let sp2 = self.compute_shortest_paths_distribution(graph2);

        // Compute histogram intersection
        let mut intersection = 0.0;
        for i in 0..sp1.len().min(sp2.len()) {
            intersection += sp1[i].min(sp2[i]);
        }

        intersection
    }

    /// Compute distribution of shortest path lengths
    fn compute_shortest_paths_distribution(&self, graph: &GraphData) -> Vec<f32> {
        let max_path_len = 10;
        let mut distribution = vec![0.0; max_path_len];

        // Simplified: use BFS to compute some shortest paths
        let edge_data = graph.edge_index.to_vec().unwrap();
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;
                adj_list.entry(src).or_insert_with(Vec::new).push(dst);
            }
        }

        // BFS from a few random nodes
        let num_samples = graph.num_nodes.min(5);
        for start in 0..num_samples {
            let path_lengths = self.bfs_shortest_paths(&adj_list, start, graph.num_nodes);
            for length in path_lengths {
                if length < max_path_len {
                    distribution[length] += 1.0;
                }
            }
        }

        // Normalize
        let sum: f32 = distribution.iter().sum();
        if sum > 0.0 {
            for val in &mut distribution {
                *val /= sum;
            }
        }

        distribution
    }

    /// BFS to compute shortest path lengths
    fn bfs_shortest_paths(
        &self,
        adj_list: &HashMap<usize, Vec<usize>>,
        start: usize,
        num_nodes: usize,
    ) -> Vec<usize> {
        let mut distances = vec![usize::MAX; num_nodes];
        let mut queue = VecDeque::new();

        distances[start] = 0;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    if neighbor < num_nodes && distances[neighbor] == usize::MAX {
                        distances[neighbor] = distances[node] + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        distances.into_iter().filter(|&d| d != usize::MAX).collect()
    }

    /// Weisfeiler-Lehman kernel
    fn wl_kernel(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        // Simplified WL: compare node label histograms after one iteration
        let labels1 = self.wl_iteration(graph1);
        let labels2 = self.wl_iteration(graph2);

        // Compute label histogram similarity
        let mut hist1: HashMap<usize, f32> = HashMap::new();
        let mut hist2: HashMap<usize, f32> = HashMap::new();

        for &label in &labels1 {
            *hist1.entry(label).or_insert(0.0) += 1.0;
        }
        for &label in &labels2 {
            *hist2.entry(label).or_insert(0.0) += 1.0;
        }

        // Histogram intersection
        let all_labels: HashSet<_> = hist1.keys().chain(hist2.keys()).collect();
        let mut intersection = 0.0;

        for &&label in &all_labels {
            let count1 = hist1.get(&label).copied().unwrap_or(0.0);
            let count2 = hist2.get(&label).copied().unwrap_or(0.0);
            intersection += count1.min(count2);
        }

        intersection / (labels1.len() + labels2.len()) as f32
    }

    /// One iteration of Weisfeiler-Lehman relabeling
    fn wl_iteration(&self, graph: &GraphData) -> Vec<usize> {
        let num_nodes = graph.num_nodes;
        let labels = vec![0; num_nodes]; // Initial labels

        // Build adjacency list
        let edge_data = graph.edge_index.to_vec().unwrap();
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;
                adj_list.entry(src).or_insert_with(Vec::new).push(dst);
            }
        }

        // Update labels based on neighborhood
        let mut new_labels = vec![0; num_nodes];
        for node in 0..num_nodes {
            let mut neighbor_labels = vec![labels[node]];
            if let Some(neighbors) = adj_list.get(&node) {
                for &neighbor in neighbors {
                    if neighbor < num_nodes {
                        neighbor_labels.push(labels[neighbor]);
                    }
                }
            }
            neighbor_labels.sort_unstable();

            // Hash neighbor labels to create new label (simplified)
            new_labels[node] = neighbor_labels
                .iter()
                .fold(0usize, |acc, &l| acc.wrapping_mul(31).wrapping_add(l));
        }

        new_labels
    }

    /// Graphlet kernel
    fn graphlet_kernel(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        // Simplified: count small subgraph patterns (triangles, stars, etc.)
        let graphlets1 = self.count_graphlets(graph1);
        let graphlets2 = self.count_graphlets(graph2);

        // Compare graphlet counts
        let mut similarity = 0.0;
        for (pattern, &count1) in &graphlets1 {
            if let Some(&count2) = graphlets2.get(pattern) {
                similarity += count1.min(count2);
            }
        }

        similarity / (graph1.num_nodes + graph2.num_nodes) as f32
    }

    /// Count small graphlet patterns
    fn count_graphlets(&self, graph: &GraphData) -> HashMap<String, f32> {
        let mut counts = HashMap::new();

        // Build adjacency list
        let edge_data = graph.edge_index.to_vec().unwrap();
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;
                adj_list.entry(src).or_insert_with(Vec::new).push(dst);
            }
        }

        // Count triangles
        let mut triangles = 0.0;
        for (_node, neighbors) in &adj_list {
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let n1 = neighbors[i];
                    let n2 = neighbors[j];

                    if let Some(n1_neighbors) = adj_list.get(&n1) {
                        if n1_neighbors.contains(&n2) {
                            triangles += 1.0;
                        }
                    }
                }
            }
        }
        counts.insert("triangle".to_string(), triangles / 3.0); // Each triangle counted 3 times

        // Count stars (nodes with degree >= 3)
        let mut stars = 0.0;
        for neighbors in adj_list.values() {
            if neighbors.len() >= 3 {
                stars += 1.0;
            }
        }
        counts.insert("star".to_string(), stars);

        counts
    }
}

/// Neural Graph Matching Network
#[derive(Debug)]
pub struct GraphMatchingNetwork {
    node_embedding_dim: usize,
    hidden_dim: usize,

    // Node embedding layers
    node_encoder1: Parameter,
    node_encoder2: Parameter,

    // Cross-graph attention
    attention_query: Parameter,
    attention_key: Parameter,
    attention_value: Parameter,

    // Matching score layers
    matching_layer1: Parameter,
    matching_layer2: Parameter,
    output_layer: Parameter,

    bias: Option<Parameter>,
}

impl GraphMatchingNetwork {
    /// Create a new graph matching network
    pub fn new(node_embedding_dim: usize, hidden_dim: usize, use_bias: bool) -> Self {
        let node_encoder1 = Parameter::new(randn(&[node_embedding_dim, hidden_dim]).unwrap());
        let node_encoder2 = Parameter::new(randn(&[hidden_dim, hidden_dim]).unwrap());

        let attention_query = Parameter::new(randn(&[hidden_dim, hidden_dim]).unwrap());
        let attention_key = Parameter::new(randn(&[hidden_dim, hidden_dim]).unwrap());
        let attention_value = Parameter::new(randn(&[hidden_dim, hidden_dim]).unwrap());

        let matching_layer1 = Parameter::new(randn(&[hidden_dim * 2, hidden_dim]).unwrap());
        let matching_layer2 = Parameter::new(randn(&[hidden_dim, (hidden_dim / 2)]).unwrap());
        let output_layer = Parameter::new(randn(&[(hidden_dim / 2), 1]).unwrap());

        let bias = if use_bias {
            Some(Parameter::new(zeros(&[1]).unwrap()))
        } else {
            None
        };

        Self {
            node_embedding_dim,
            hidden_dim,
            node_encoder1,
            node_encoder2,
            attention_query,
            attention_key,
            attention_value,
            matching_layer1,
            matching_layer2,
            output_layer,
            bias,
        }
    }

    /// Compute similarity score between two graphs
    pub fn compute_similarity(&self, graph1: &GraphData, graph2: &GraphData) -> f32 {
        // Encode both graphs
        let h1 = self.encode_graph(&graph1.x);
        let h2 = self.encode_graph(&graph2.x);

        // Cross-graph attention
        let attended1 = self.cross_attention(&h1, &h2);
        let attended2 = self.cross_attention(&h2, &h1);

        // Pool to graph-level representations
        let g1 = attended1.mean(Some(&[0]), false).unwrap();
        let g2 = attended2.mean(Some(&[0]), false).unwrap();

        // Concatenate
        let g1_data = g1.to_vec().unwrap();
        let g2_data = g2.to_vec().unwrap();
        let mut concat_data = g1_data;
        concat_data.extend(g2_data);

        let concat = from_vec(
            concat_data,
            &[1, self.hidden_dim * 2],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Matching layers
        let mut h = concat.matmul(&self.matching_layer1.clone_data()).unwrap();
        h = self.relu(&h);

        h = h.matmul(&self.matching_layer2.clone_data()).unwrap();
        h = self.relu(&h);

        let mut score = h.matmul(&self.output_layer.clone_data()).unwrap();
        if let Some(ref bias) = self.bias {
            score = score.add(&bias.clone_data()).unwrap();
        }

        // Sigmoid activation
        let score_val = score.item().unwrap();
        1.0 / (1.0 + (-score_val).exp())
    }

    /// Encode graph features
    fn encode_graph(&self, x: &Tensor) -> Tensor {
        let mut h = x.matmul(&self.node_encoder1.clone_data()).unwrap();
        h = self.relu(&h);
        h = h.matmul(&self.node_encoder2.clone_data()).unwrap();
        self.relu(&h)
    }

    /// Cross-graph attention mechanism
    fn cross_attention(&self, query_graph: &Tensor, key_value_graph: &Tensor) -> Tensor {
        let _q = query_graph
            .matmul(&self.attention_query.clone_data())
            .unwrap();
        let _k = key_value_graph
            .matmul(&self.attention_key.clone_data())
            .unwrap();
        let v = key_value_graph
            .matmul(&self.attention_value.clone_data())
            .unwrap();

        // Simplified attention: mean pooling
        // In practice, would compute q @ k^T / sqrt(d), then softmax, then @ v
        v.mean(Some(&[0]), false).unwrap().unsqueeze(0).unwrap()
    }

    fn relu(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let activated: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.node_encoder1.clone_data(),
            self.node_encoder2.clone_data(),
            self.attention_query.clone_data(),
            self.attention_key.clone_data(),
            self.attention_value.clone_data(),
            self.matching_layer1.clone_data(),
            self.matching_layer2.clone_data(),
            self.output_layer.clone_data(),
        ];

        if let Some(ref b) = self.bias {
            params.push(b.clone_data());
        }

        params
    }
}

/// Siamese Graph Network for similarity learning
#[derive(Debug)]
pub struct SiameseGraphNetwork {
    embedding_network: Parameter,
    hidden_dim: usize,
    output_dim: usize,
}

impl SiameseGraphNetwork {
    /// Create a new Siamese graph network
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let embedding_network = Parameter::new(randn(&[input_dim, hidden_dim]).unwrap());

        Self {
            embedding_network,
            hidden_dim,
            output_dim,
        }
    }

    /// Compute embeddings for a graph
    pub fn embed(&self, graph: &GraphData) -> Tensor {
        let mut h = graph
            .x
            .matmul(&self.embedding_network.clone_data())
            .unwrap();
        h = self.relu(&h);

        // Global pooling
        h.mean(Some(&[0]), false).unwrap()
    }

    /// Compute contrastive loss between similar and dissimilar pairs
    pub fn contrastive_loss(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
        is_similar: bool,
        margin: f32,
    ) -> f32 {
        let emb1 = self.embed(graph1);
        let emb2 = self.embed(graph2);

        // Euclidean distance
        let diff = emb1.sub(&emb2).unwrap();
        let dist_sq = diff.dot(&diff).unwrap().item().unwrap();
        let dist = dist_sq.sqrt();

        if is_similar {
            // Pull similar graphs closer
            dist_sq
        } else {
            // Push dissimilar graphs apart
            (margin - dist).max(0.0).powi(2)
        }
    }

    fn relu(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let activated: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.embedding_network.clone_data()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_graph_edit_distance() {
        let features1 = randn(&[4, 3]).unwrap();
        let features2 = randn(&[5, 3]).unwrap();
        let edges1 = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edges2 = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        let edge_index1 = from_vec(edges1, &[2, 3], DeviceType::Cpu).unwrap();
        let edge_index2 = from_vec(edges2, &[2, 4], DeviceType::Cpu).unwrap();

        let graph1 = GraphData::new(features1, edge_index1);
        let graph2 = GraphData::new(features2, edge_index2);

        let ged = GraphEditDistance::new();
        let distance = ged.compute(&graph1, &graph2);

        assert!(distance > 0.0);
    }

    #[test]
    fn test_node_correspondence() {
        let features1 = randn(&[3, 4]).unwrap();
        let features2 = randn(&[3, 4]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];

        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let ged = GraphEditDistance::new();
        let correspondences = ged.node_correspondence(&graph1, &graph2);

        assert_eq!(correspondences.len(), 3);
    }

    #[test]
    fn test_random_walk_kernel() {
        let features1 = randn(&[4, 3]).unwrap();
        let features2 = randn(&[4, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];

        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let kernel = GraphKernel::new(GraphKernelType::RandomWalk);
        let similarity = kernel.compute(&graph1, &graph2);

        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_shortest_path_kernel() {
        let features1 = randn(&[5, 3]).unwrap();
        let features2 = randn(&[5, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let kernel = GraphKernel::new(GraphKernelType::ShortestPath);
        let similarity = kernel.compute(&graph1, &graph2);

        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_weisfeiler_lehman_kernel() {
        let features1 = randn(&[4, 3]).unwrap();
        let features2 = randn(&[4, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];

        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let kernel = GraphKernel::new(GraphKernelType::WeisfeilerLehman);
        let similarity = kernel.compute(&graph1, &graph2);

        assert!(similarity >= 0.0);
    }

    #[test]
    fn test_graph_matching_network() {
        let features1 = randn(&[4, 8]).unwrap();
        let features2 = randn(&[5, 8]).unwrap();
        let edges1 = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edges2 = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

        let edge_index1 = from_vec(edges1, &[2, 3], DeviceType::Cpu).unwrap();
        let edge_index2 = from_vec(edges2, &[2, 4], DeviceType::Cpu).unwrap();

        let graph1 = GraphData::new(features1, edge_index1);
        let graph2 = GraphData::new(features2, edge_index2);

        let gmn = GraphMatchingNetwork::new(8, 16, true);
        let similarity = gmn.compute_similarity(&graph1, &graph2);

        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_siamese_network() {
        let features1 = randn(&[3, 6]).unwrap();
        let features2 = randn(&[3, 6]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];

        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let siamese = SiameseGraphNetwork::new(6, 12, 8);

        let emb1 = siamese.embed(&graph1);
        let emb2 = siamese.embed(&graph2);

        assert_eq!(emb1.shape().dims(), &[12]);
        assert_eq!(emb2.shape().dims(), &[12]);

        // Test contrastive loss for similar graphs
        let loss_similar = siamese.contrastive_loss(&graph1, &graph2, true, 1.0);
        assert!(loss_similar >= 0.0);

        // Test contrastive loss for dissimilar graphs
        let loss_dissimilar = siamese.contrastive_loss(&graph1, &graph2, false, 1.0);
        assert!(loss_dissimilar >= 0.0);
    }

    #[test]
    fn test_graphlet_kernel() {
        let features1 = randn(&[5, 3]).unwrap();
        let features2 = randn(&[5, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 2.0, 3.0, 3.0, 4.0];

        let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let kernel = GraphKernel::new(GraphKernelType::Graphlet);
        let similarity = kernel.compute(&graph1, &graph2);

        assert!(similarity >= 0.0);
    }
}
