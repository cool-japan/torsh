//! Graph data loading and manipulation utilities

use crate::GraphData;
use torsh_core::device::DeviceType;
use torsh_tensor::{
    creation::{from_vec, zeros},
    Tensor,
};
// Direct implementation for now

/// Graph data loader
pub struct GraphDataLoader {
    batch_size: usize,
    shuffle: bool,
}

impl GraphDataLoader {
    /// Create a new graph data loader with validation
    ///
    /// # Arguments
    /// * `batch_size` - Size of batches, must be > 0
    /// * `shuffle` - Whether to shuffle data
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully created data loader
    /// * `Err` - If batch_size is 0
    ///
    /// # Example
    /// ```
    /// use torsh_graph::data::GraphDataLoader;
    /// let loader = GraphDataLoader::new(32, true).unwrap();
    /// assert_eq!(loader.batch_size(), 32);
    /// ```
    pub fn new(batch_size: usize, shuffle: bool) -> Result<Self, Box<dyn std::error::Error>> {
        if batch_size == 0 {
            return Err("Batch size must be greater than 0".into());
        }

        Ok(Self {
            batch_size,
            shuffle,
        })
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get shuffle setting
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Load graphs from a directory
    pub fn from_directory(&self, _path: &str) -> Vec<GraphData> {
        // Skeleton implementation
        // Would load graph files and convert to GraphData
        Vec::new()
    }
}

/// Convert between different graph representations
pub mod converters {
    use super::*;

    // TODO: Implement when scirs2_graph API is stable
    // pub fn from_scirs2_graph(graph: &Graph) -> GraphData { ... }

    /// Convert from edge list to GraphData with validation
    ///
    /// # Arguments
    /// * `edges` - List of (source, target) node pairs
    /// * `num_nodes` - Total number of nodes in the graph
    ///
    /// # Returns
    /// * `Ok(GraphData)` - Successfully created graph
    /// * `Err` - If edges contain invalid node indices
    ///
    /// # Example
    /// ```
    /// use torsh_graph::data::converters;
    /// let edges = vec![(0, 1), (1, 2), (2, 0)];
    /// let graph = converters::from_edge_list(&edges, 3).unwrap();
    /// assert_eq!(graph.num_nodes, 3);
    /// assert_eq!(graph.num_edges, 3);
    /// ```
    pub fn from_edge_list(
        edges: &[(usize, usize)],
        num_nodes: usize,
    ) -> Result<GraphData, Box<dyn std::error::Error>> {
        if num_nodes == 0 {
            return Err("Number of nodes must be greater than 0".into());
        }

        // Validate edge indices
        for (i, (src, dst)) in edges.iter().enumerate() {
            if *src >= num_nodes {
                return Err(format!(
                    "Edge {} has invalid source node {} (max: {})",
                    i,
                    src,
                    num_nodes - 1
                )
                .into());
            }
            if *dst >= num_nodes {
                return Err(format!(
                    "Edge {} has invalid target node {} (max: {})",
                    i,
                    dst,
                    num_nodes - 1
                )
                .into());
            }
        }

        let num_edges = edges.len();
        let mut edge_index = vec![0i64; 2 * num_edges];

        for (i, (src, dst)) in edges.iter().enumerate() {
            edge_index[i] = *src as i64;
            edge_index[num_edges + i] = *dst as i64;
        }

        let x = zeros(&[num_nodes, 1])?;
        let edge_index = from_vec(edge_index, &[2, num_edges], DeviceType::Cpu)?;

        Ok(GraphData::new(x, edge_index.to_f32_simd()?))
    }

    /// Convert from adjacency matrix to GraphData
    ///
    /// Converts a dense adjacency matrix to GraphData format.
    /// Non-zero entries in the matrix are treated as edges.
    ///
    /// # Arguments
    /// * `adj` - Square adjacency matrix (num_nodes x num_nodes)
    ///
    /// # Returns
    /// GraphData with edges extracted from non-zero matrix entries
    pub fn from_adjacency_matrix(adj: &Tensor) -> Result<GraphData, Box<dyn std::error::Error>> {
        let shape = adj.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err("Adjacency matrix must be 2D".into());
        }

        let num_nodes = dims[0];
        if dims[0] != dims[1] {
            return Err("Adjacency matrix must be square".into());
        }

        // Extract edges from adjacency matrix
        let adj_data = adj.to_vec()?;
        let mut edges = Vec::new();

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let idx = i * num_nodes + j;
                if idx < adj_data.len() && adj_data[idx].abs() > 1e-8 {
                    edges.push((i, j));
                }
            }
        }

        let num_edges = edges.len();

        // Create edge index tensor
        let mut edge_index_vec = Vec::new();
        for &(src, _dst) in &edges {
            edge_index_vec.push(src as f32);
        }
        for &(_src, dst) in &edges {
            edge_index_vec.push(dst as f32);
        }

        let edge_index = from_vec(edge_index_vec, &[2, num_edges], DeviceType::Cpu)?;
        let x = zeros(&[num_nodes, 1])?;

        Ok(GraphData::new(x, edge_index))
    }

    /// Convert GraphData to adjacency matrix
    ///
    /// Creates a dense adjacency matrix from a GraphData structure.
    ///
    /// # Arguments
    /// * `graph` - Input graph data
    ///
    /// # Returns
    /// Dense adjacency matrix (num_nodes x num_nodes)
    pub fn to_adjacency_matrix(graph: &GraphData) -> Result<Tensor, Box<dyn std::error::Error>> {
        let num_nodes = graph.num_nodes;
        let mut adj_data = vec![0.0f32; num_nodes * num_nodes];

        let edge_data = graph.edge_index.to_vec()?;

        for i in 0..graph.num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[graph.num_edges + i] as usize;

            if src < num_nodes && dst < num_nodes {
                adj_data[src * num_nodes + dst] = 1.0;
            }
        }

        Ok(from_vec(
            adj_data,
            &[num_nodes, num_nodes],
            DeviceType::Cpu,
        )?)
    }

    /// Convert GraphData to weighted adjacency matrix
    ///
    /// Creates a weighted adjacency matrix using edge attributes.
    ///
    /// # Arguments
    /// * `graph` - Input graph data with edge attributes
    ///
    /// # Returns
    /// Weighted adjacency matrix (num_nodes x num_nodes)
    pub fn to_weighted_adjacency_matrix(
        graph: &GraphData,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let num_nodes = graph.num_nodes;
        let mut adj_data = vec![0.0f32; num_nodes * num_nodes];

        let edge_data = graph.edge_index.to_vec()?;

        if let Some(ref edge_attr) = graph.edge_attr {
            let weights = edge_attr.to_vec()?;

            for i in 0..graph.num_edges {
                let src = edge_data[i] as usize;
                let dst = edge_data[graph.num_edges + i] as usize;

                if src < num_nodes && dst < num_nodes && i < weights.len() {
                    adj_data[src * num_nodes + dst] = weights[i];
                }
            }
        } else {
            // No edge attributes, use binary adjacency
            for i in 0..graph.num_edges {
                let src = edge_data[i] as usize;
                let dst = edge_data[graph.num_edges + i] as usize;

                if src < num_nodes && dst < num_nodes {
                    adj_data[src * num_nodes + dst] = 1.0;
                }
            }
        }

        Ok(from_vec(
            adj_data,
            &[num_nodes, num_nodes],
            DeviceType::Cpu,
        )?)
    }
}

/// Graph augmentation utilities
pub mod augmentation {
    use super::*;
    use scirs2_core::random::thread_rng;
    use torsh_tensor::creation::randn;

    /// Add self-loops to a graph
    ///
    /// Adds self-connections (i->i) for all nodes in the graph.
    /// This is commonly used in GCN-style architectures.
    pub fn add_self_loops(graph: &mut GraphData) -> Result<(), Box<dyn std::error::Error>> {
        let num_nodes = graph.num_nodes;
        let existing_edges = graph.edge_index.to_vec()?;

        // Create self-loop edges
        let mut self_loops = Vec::new();
        for i in 0..num_nodes {
            self_loops.push(i as f32); // source
        }
        for i in 0..num_nodes {
            self_loops.push(i as f32); // destination
        }

        // Concatenate existing edges with self-loops
        let mut all_edges = existing_edges;
        all_edges.extend(self_loops);

        let new_num_edges = graph.num_edges + num_nodes;
        graph.edge_index = from_vec(all_edges, &[2, new_num_edges], DeviceType::Cpu)?;
        graph.num_edges = new_num_edges;

        Ok(())
    }

    /// Remove isolated nodes from the graph
    ///
    /// Removes nodes that have no incoming or outgoing edges.
    /// Returns the mapping from old node indices to new node indices.
    pub fn remove_isolated_nodes(
        graph: &mut GraphData,
    ) -> Result<Vec<Option<usize>>, Box<dyn std::error::Error>> {
        let edge_data = graph.edge_index.to_vec()?;
        let num_nodes = graph.num_nodes;

        // Find which nodes have edges
        let mut has_edge = vec![false; num_nodes];
        for i in 0..graph.num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[graph.num_edges + i] as usize;
            if src < num_nodes {
                has_edge[src] = true;
            }
            if dst < num_nodes {
                has_edge[dst] = true;
            }
        }

        // Create mapping from old to new indices
        let mut old_to_new = vec![None; num_nodes];
        let mut new_idx = 0;
        for (old_idx, &has_edges) in has_edge.iter().enumerate() {
            if has_edges {
                old_to_new[old_idx] = Some(new_idx);
                new_idx += 1;
            }
        }

        let new_num_nodes = new_idx;

        // Remap edges
        let mut new_edges = Vec::new();
        for i in 0..graph.num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[graph.num_edges + i] as usize;

            if src < num_nodes && dst < num_nodes {
                if let (Some(new_src), Some(new_dst)) = (old_to_new[src], old_to_new[dst]) {
                    new_edges.push(new_src as f32);
                    new_edges.push(new_dst as f32);
                }
            }
        }

        let new_num_edges = new_edges.len() / 2;

        // Remap node features
        let feature_dim = graph.x.shape().dims()[1];
        let mut new_features = Vec::new();
        for old_idx in 0..num_nodes {
            if old_to_new[old_idx].is_some() {
                // Extract features for this node
                for f in 0..feature_dim {
                    let idx = old_idx * feature_dim + f;
                    if let Ok(feat_data) = graph.x.to_vec() {
                        if idx < feat_data.len() {
                            new_features.push(feat_data[idx]);
                        }
                    }
                }
            }
        }

        // Update graph
        graph.x = from_vec(new_features, &[new_num_nodes, feature_dim], DeviceType::Cpu)?;
        graph.edge_index = from_vec(new_edges, &[2, new_num_edges], DeviceType::Cpu)?;
        graph.num_nodes = new_num_nodes;
        graph.num_edges = new_num_edges;

        Ok(old_to_new)
    }

    /// Normalize node features to unit norm
    ///
    /// Normalizes each node's feature vector to have L2 norm = 1.
    /// This helps stabilize training and makes features scale-invariant.
    pub fn normalize_features(graph: &mut GraphData) -> Result<(), Box<dyn std::error::Error>> {
        let feature_data = graph.x.to_vec()?;
        let num_nodes = graph.num_nodes;
        let feature_dim = graph.x.shape().dims()[1];

        let mut normalized = Vec::new();

        for node in 0..num_nodes {
            let start = node * feature_dim;
            let end = start + feature_dim;
            let node_features = &feature_data[start..end];

            // Compute L2 norm
            let norm: f32 = node_features.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm = norm.max(1e-8); // Avoid division by zero

            // Normalize
            for &feat in node_features {
                normalized.push(feat / norm);
            }
        }

        graph.x = from_vec(normalized, &[num_nodes, feature_dim], DeviceType::Cpu)?;
        Ok(())
    }

    /// Drop edges randomly for augmentation
    ///
    /// Randomly removes a fraction of edges from the graph.
    /// This is useful for regularization and data augmentation.
    pub fn edge_dropout(
        graph: &mut GraphData,
        drop_rate: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let edge_data = graph.edge_index.to_vec()?;

        let mut kept_edges = Vec::new();
        let mut kept_count = 0;

        for i in 0..graph.num_edges {
            if rng.gen_range(0.0..1.0) > drop_rate {
                kept_edges.push(edge_data[i]); // source
                kept_count += 1;
            }
        }

        // Add corresponding destinations
        let mut dest_edges = Vec::new();
        let mut kept_idx = 0;
        for i in 0..graph.num_edges {
            if rng.gen_range(0.0..1.0) > drop_rate && kept_idx < kept_count {
                dest_edges.push(edge_data[graph.num_edges + i]); // destination
                kept_idx += 1;
            }
        }

        kept_edges.extend(dest_edges);

        graph.edge_index = from_vec(kept_edges, &[2, kept_count], DeviceType::Cpu)?;
        graph.num_edges = kept_count;

        Ok(())
    }

    /// Mask node features randomly for augmentation
    ///
    /// Randomly masks (zeros out) a fraction of node features.
    /// Useful for contrastive learning and robustness.
    pub fn feature_masking(
        graph: &mut GraphData,
        mask_rate: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let mut feature_data = graph.x.to_vec()?;

        for feat in feature_data.iter_mut() {
            if rng.gen_range(0.0..1.0) < mask_rate {
                *feat = 0.0;
            }
        }

        let shape = graph.x.shape().dims().to_vec();
        graph.x = from_vec(feature_data, &shape, DeviceType::Cpu)?;

        Ok(())
    }

    /// Add random noise to node features
    ///
    /// Adds Gaussian noise to node features for augmentation.
    /// This improves model robustness and generalization.
    pub fn feature_noise(
        graph: &mut GraphData,
        noise_std: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let shape_binding = graph.x.shape();
        let shape = shape_binding.dims();
        let noise: Tensor = randn::<f32>(shape)?;

        let noise_data = noise.to_vec()?;
        let mut feature_data = graph.x.to_vec()?;

        for (feat, &n) in feature_data.iter_mut().zip(noise_data.iter()) {
            let noise_val: f32 = n * noise_std;
            *feat = *feat + noise_val;
        }

        graph.x = from_vec(feature_data, shape, DeviceType::Cpu)?;

        Ok(())
    }

    /// Drop nodes randomly (subgraph sampling)
    ///
    /// Randomly removes a fraction of nodes and their associated edges.
    /// Useful for creating mini-batches from large graphs.
    pub fn node_dropout(
        graph: &mut GraphData,
        drop_rate: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let num_nodes = graph.num_nodes;

        // Determine which nodes to keep
        let mut keep_node = vec![false; num_nodes];
        for i in 0..num_nodes {
            if rng.gen_range(0.0..1.0) > drop_rate {
                keep_node[i] = true;
            }
        }

        // Create mapping from old to new indices
        let mut old_to_new = vec![None; num_nodes];
        let mut new_idx = 0;
        for (old_idx, &keep) in keep_node.iter().enumerate() {
            if keep {
                old_to_new[old_idx] = Some(new_idx);
                new_idx += 1;
            }
        }

        let new_num_nodes = new_idx;

        // Filter edges and remap
        let edge_data = graph.edge_index.to_vec()?;
        let mut new_edges = Vec::new();

        for i in 0..graph.num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[graph.num_edges + i] as usize;

            if src < num_nodes && dst < num_nodes && keep_node[src] && keep_node[dst] {
                if let (Some(new_src), Some(new_dst)) = (old_to_new[src], old_to_new[dst]) {
                    new_edges.push(new_src as f32);
                    new_edges.push(new_dst as f32);
                }
            }
        }

        let new_num_edges = new_edges.len() / 2;

        // Filter node features
        let feature_dim = graph.x.shape().dims()[1];
        let feature_data = graph.x.to_vec()?;
        let mut new_features = Vec::new();

        for old_idx in 0..num_nodes {
            if keep_node[old_idx] {
                let start = old_idx * feature_dim;
                let end = start + feature_dim;
                new_features.extend_from_slice(&feature_data[start..end]);
            }
        }

        // Update graph
        graph.x = from_vec(new_features, &[new_num_nodes, feature_dim], DeviceType::Cpu)?;
        graph.edge_index = from_vec(new_edges, &[2, new_num_edges], DeviceType::Cpu)?;
        graph.num_nodes = new_num_nodes;
        graph.num_edges = new_num_edges;

        Ok(())
    }

    /// Apply random walk-based subgraph sampling
    ///
    /// Samples a subgraph by performing random walks from random starting nodes.
    /// This preserves local graph structure better than random node dropout.
    pub fn random_walk_subgraph(
        graph: &GraphData,
        num_walks: usize,
        walk_length: usize,
    ) -> Result<GraphData, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let edge_data = graph.edge_index.to_vec()?;

        // Build adjacency list
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); graph.num_nodes];
        for i in 0..graph.num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[graph.num_edges + i] as usize;
            if src < graph.num_nodes && dst < graph.num_nodes {
                adj_list[src].push(dst);
            }
        }

        // Perform random walks
        let mut visited_nodes = std::collections::HashSet::new();

        for _ in 0..num_walks {
            let mut current = rng.gen_range(0..graph.num_nodes);
            visited_nodes.insert(current);

            for _ in 0..walk_length {
                if adj_list[current].is_empty() {
                    break;
                }
                let next_idx = rng.gen_range(0..adj_list[current].len());
                current = adj_list[current][next_idx];
                visited_nodes.insert(current);
            }
        }

        // Create subgraph with visited nodes
        let visited: Vec<usize> = visited_nodes.into_iter().collect();
        let mut old_to_new = vec![None; graph.num_nodes];
        for (new_idx, &old_idx) in visited.iter().enumerate() {
            old_to_new[old_idx] = Some(new_idx);
        }

        let new_num_nodes = visited.len();

        // Extract edges
        let mut new_edges = Vec::new();
        for i in 0..graph.num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[graph.num_edges + i] as usize;

            if src < graph.num_nodes && dst < graph.num_nodes {
                if let (Some(new_src), Some(new_dst)) = (old_to_new[src], old_to_new[dst]) {
                    new_edges.push(new_src as f32);
                    new_edges.push(new_dst as f32);
                }
            }
        }

        let new_num_edges = new_edges.len() / 2;

        // Extract features
        let feature_dim = graph.x.shape().dims()[1];
        let feature_data = graph.x.to_vec()?;
        let mut new_features = Vec::new();

        for &old_idx in &visited {
            let start = old_idx * feature_dim;
            let end = start + feature_dim;
            if end <= feature_data.len() {
                new_features.extend_from_slice(&feature_data[start..end]);
            }
        }

        // Create new graph
        let x = from_vec(new_features, &[new_num_nodes, feature_dim], DeviceType::Cpu)?;
        let edge_index = from_vec(new_edges, &[2, new_num_edges], DeviceType::Cpu)?;

        Ok(GraphData::new(x, edge_index))
    }
}

#[cfg(test)]
mod tests {
    use super::augmentation::*;
    use super::converters::*;
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_add_self_loops() {
        let x = randn(&[3, 4]).unwrap();
        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let mut graph = GraphData::new(x, edge_index);

        let original_edges = graph.num_edges;
        add_self_loops(&mut graph).unwrap();

        assert_eq!(graph.num_edges, original_edges + 3); // Added 3 self-loops
    }

    #[test]
    fn test_normalize_features() {
        let x = randn(&[5, 8]).unwrap();
        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let mut graph = GraphData::new(x, edge_index);

        normalize_features(&mut graph).unwrap();

        // Check that features are normalized
        let normalized_data = graph.x.to_vec().unwrap();
        let feature_dim = 8;

        for node in 0..5 {
            let start = node * feature_dim;
            let end = start + feature_dim;
            let node_features = &normalized_data[start..end];
            let norm: f32 = node_features.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Features should be normalized to unit norm"
            );
        }
    }

    #[test]
    fn test_edge_dropout() {
        let x = randn(&[4, 3]).unwrap();
        let edge_index = from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0],
            &[2, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let mut graph = GraphData::new(x, edge_index);

        let original_edges = graph.num_edges;
        edge_dropout(&mut graph, 0.5).unwrap();

        assert!(graph.num_edges <= original_edges);
        assert_eq!(graph.edge_index.shape().dims()[1], graph.num_edges);
    }

    #[test]
    fn test_feature_masking() {
        let x = randn(&[3, 5]).unwrap();
        let edge_index = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let mut graph = GraphData::new(x.clone(), edge_index);

        feature_masking(&mut graph, 0.3).unwrap();

        let masked_data = graph.x.to_vec().unwrap();
        let zero_count = masked_data.iter().filter(|&&x| x == 0.0).count();

        // Some features should be masked
        assert!(zero_count > 0, "Some features should be masked");
    }

    #[test]
    fn test_node_dropout() {
        let x = randn(&[5, 4]).unwrap();
        let edge_index = from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            &[2, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let mut graph = GraphData::new(x, edge_index);

        let original_nodes = graph.num_nodes;
        node_dropout(&mut graph, 0.3).unwrap();

        assert!(graph.num_nodes <= original_nodes);
        assert_eq!(graph.x.shape().dims()[0], graph.num_nodes);
    }

    #[test]
    fn test_remove_isolated_nodes() {
        let x = randn(&[5, 3]).unwrap();
        // Create graph where node 2 is isolated
        let edge_index = from_vec(
            vec![0.0, 1.0, 3.0, 4.0, 1.0, 0.0, 4.0, 3.0],
            &[2, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let mut graph = GraphData::new(x, edge_index);

        let mapping = remove_isolated_nodes(&mut graph).unwrap();

        assert_eq!(graph.num_nodes, 4); // Node 2 should be removed
        assert!(mapping[2].is_none()); // Node 2 should have no mapping
    }

    #[test]
    fn test_from_adjacency_matrix() {
        // Create a simple 3x3 adjacency matrix
        let adj_data = vec![
            0.0, 1.0, 0.0, // Node 0 connects to node 1
            1.0, 0.0, 1.0, // Node 1 connects to nodes 0 and 2
            0.0, 1.0, 0.0, // Node 2 connects to node 1
        ];
        let adj = from_vec(adj_data, &[3, 3], DeviceType::Cpu).unwrap();

        let graph = from_adjacency_matrix(&adj).unwrap();

        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 4); // 4 directed edges
    }

    #[test]
    fn test_to_adjacency_matrix() {
        let x = randn(&[3, 2]).unwrap();
        let edge_index =
            from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0], &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(x, edge_index);

        let adj = to_adjacency_matrix(&graph).unwrap();

        assert_eq!(adj.shape().dims(), &[3, 3]);

        // Check that edges are present in adjacency matrix
        let adj_data = adj.to_vec().unwrap();
        assert_eq!(adj_data[0 * 3 + 1], 1.0); // Edge 0->1
        assert_eq!(adj_data[1 * 3 + 2], 1.0); // Edge 1->2
        assert_eq!(adj_data[2 * 3 + 0], 1.0); // Edge 2->0
    }

    #[test]
    fn test_from_edge_list() {
        let edges = vec![(0, 1), (1, 2), (2, 0), (1, 0)];
        let graph = from_edge_list(&edges, 3).unwrap();

        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 4);
    }

    #[test]
    fn test_adjacency_round_trip() {
        // Create graph from edge list
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let graph1 = from_edge_list(&edges, 4).unwrap();

        // Convert to adjacency matrix and back
        let adj = to_adjacency_matrix(&graph1).unwrap();
        let graph2 = from_adjacency_matrix(&adj).unwrap();

        assert_eq!(graph1.num_nodes, graph2.num_nodes);
        assert_eq!(graph1.num_edges, graph2.num_edges);
    }
}
