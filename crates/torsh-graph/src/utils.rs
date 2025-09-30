//! Graph utilities and algorithms

use torsh_core::device::DeviceType;
use torsh_tensor::{
    creation::{eye, from_vec, zeros},
    Tensor,
};
// Direct implementation of graph algorithms for now

/// Helper function to convert tensor to 2D Vec format
/// Assumes tensor is of shape [rows, cols] and returns Vec<Vec<T>>
pub fn tensor_to_vec2<T: Clone + torsh_core::TensorElement>(
    tensor: &Tensor<T>,
) -> Result<Vec<Vec<T>>, Box<dyn std::error::Error>> {
    let data = tensor.to_vec()?;
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();

    if shape.len() != 2 {
        return Err("Tensor must be 2D".into());
    }

    let rows = shape[0];
    let cols = shape[1];
    let mut result = Vec::with_capacity(rows);

    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        result.push(data[start..end].to_vec());
    }

    Ok(result)
}

/// Compute graph Laplacian
pub fn graph_laplacian(edge_index: &Tensor, num_nodes: usize, normalized: bool) -> Tensor {
    // Simplified graph Laplacian implementation for compilation compatibility
    let edge_data = tensor_to_vec2::<f32>(edge_index).unwrap();

    // Build adjacency matrix manually with proper indexing
    let mut adjacency_data = vec![0.0; num_nodes * num_nodes];
    let mut degrees = vec![0.0; num_nodes];

    // Build adjacency matrix and count degrees
    for j in 0..edge_data[0].len() {
        let src = edge_data[0][j] as usize;
        let dst = edge_data[1][j] as usize;

        if src < num_nodes && dst < num_nodes {
            // Set A[src][dst] = 1 (and A[dst][src] = 1 for undirected graphs)
            adjacency_data[src * num_nodes + dst] = 1.0;
            adjacency_data[dst * num_nodes + src] = 1.0;
            degrees[src] += 1.0;
            degrees[dst] += 1.0;
        }
    }

    // Create tensors
    let adjacency = from_vec(
        adjacency_data.clone(),
        &[num_nodes, num_nodes],
        DeviceType::Cpu,
    )
    .unwrap();

    if normalized {
        // Normalized Laplacian: L = I - D^(-1/2) @ A @ D^(-1/2)
        let mut laplacian_data = vec![0.0; num_nodes * num_nodes];

        // Compute normalized laplacian manually
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    laplacian_data[i * num_nodes + j] = 1.0; // Identity diagonal
                    if degrees[i] > 0.0 {
                        // Subtract normalized adjacency
                        laplacian_data[i * num_nodes + j] -=
                            adjacency_data[i * num_nodes + j] / (degrees[i] as f32).sqrt();
                    }
                } else if adjacency_data[i * num_nodes + j] > 0.0 {
                    // Off-diagonal normalized elements
                    if degrees[i] > 0.0 && degrees[j] > 0.0 {
                        laplacian_data[i * num_nodes + j] =
                            -1.0 / ((degrees[i] as f32).sqrt() * (degrees[j] as f32).sqrt());
                    }
                }
            }
        }

        from_vec(laplacian_data, &[num_nodes, num_nodes], DeviceType::Cpu).unwrap()
    } else {
        // Unnormalized Laplacian: L = D - A
        let mut laplacian_data = vec![0.0; num_nodes * num_nodes];

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i == j {
                    laplacian_data[i * num_nodes + j] = degrees[i]; // Degree on diagonal
                } else {
                    laplacian_data[i * num_nodes + j] = -adjacency_data[i * num_nodes + j];
                    // -A off diagonal
                }
            }
        }

        from_vec(laplacian_data, &[num_nodes, num_nodes], DeviceType::Cpu).unwrap()
    }
}

/// Compute degree matrix
pub fn degree_matrix(edge_index: &Tensor, num_nodes: usize) -> Tensor {
    // Count degrees for each node
    let mut degrees = vec![0.0; num_nodes];
    let edge_data = tensor_to_vec2::<f32>(edge_index).unwrap();

    // Count edges (both directions for undirected graphs)
    for j in 0..edge_data[0].len() {
        let src = edge_data[0][j] as usize;
        let dst = edge_data[1][j] as usize;

        degrees[src] += 1.0;
        degrees[dst] += 1.0;
    }

    // Create diagonal degree matrix
    let mut degree_matrix = zeros(&[num_nodes, num_nodes]).unwrap();
    for i in 0..num_nodes {
        // Set diagonal element to degree value
        degree_matrix.set_item(&[i, i], degrees[i]).unwrap();
    }

    degree_matrix
}

/// Graph connectivity utilities
pub mod connectivity {
    use super::*;
    use std::collections::{HashSet, VecDeque};

    /// Check if graph is connected
    pub fn is_connected(edge_index: &Tensor, num_nodes: usize) -> bool {
        if num_nodes <= 1 {
            return true;
        }

        // Build adjacency list
        let adjacency_list = build_adjacency_list(edge_index, num_nodes);

        // Perform BFS from node 0
        let mut visited = vec![false; num_nodes];
        let mut queue = VecDeque::new();
        queue.push_back(0);
        visited[0] = true;
        let mut visited_count = 1;

        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency_list[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    visited_count += 1;
                    queue.push_back(neighbor);
                }
            }
        }

        visited_count == num_nodes
    }

    /// Get connected components using DFS
    pub fn connected_components(edge_index: &Tensor, num_nodes: usize) -> Vec<Vec<usize>> {
        let adjacency_list = build_adjacency_list(edge_index, num_nodes);
        let mut visited = vec![false; num_nodes];
        let mut components = Vec::new();

        for start_node in 0..num_nodes {
            if !visited[start_node] {
                let mut component = Vec::new();
                let mut stack = vec![start_node];

                while let Some(node) = stack.pop() {
                    if !visited[node] {
                        visited[node] = true;
                        component.push(node);

                        for &neighbor in &adjacency_list[node] {
                            if !visited[neighbor] {
                                stack.push(neighbor);
                            }
                        }
                    }
                }

                components.push(component);
            }
        }

        components
    }

    /// Get largest connected component
    pub fn largest_component(edge_index: &Tensor, num_nodes: usize) -> Vec<usize> {
        let components = connected_components(edge_index, num_nodes);
        components
            .into_iter()
            .max_by_key(|component| component.len())
            .unwrap_or_default()
    }

    /// Build adjacency list from edge index
    pub(crate) fn build_adjacency_list(edge_index: &Tensor, num_nodes: usize) -> Vec<Vec<usize>> {
        let mut adjacency_list = vec![Vec::new(); num_nodes];
        let edge_data = tensor_to_vec2::<f32>(edge_index).unwrap();

        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;

            if src < num_nodes && dst < num_nodes {
                adjacency_list[src].push(dst);
                adjacency_list[dst].push(src); // Undirected graph
            }
        }

        // Remove duplicates and sort
        for neighbors in &mut adjacency_list {
            neighbors.sort();
            neighbors.dedup();
        }

        adjacency_list
    }
}

/// Graph metrics and statistics
pub mod metrics {
    use super::*;
    use std::collections::{HashMap, VecDeque};

    /// Compute node centrality measures
    pub fn node_centrality(edge_index: &Tensor, num_nodes: usize) -> CentralityMeasures {
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes);

        // Degree centrality
        let degree_values: Vec<f64> = adjacency_list
            .iter()
            .map(|neighbors| neighbors.len() as f64)
            .collect();
        let degree_f32: Vec<f32> = degree_values.into_iter().map(|x| x as f32).collect();
        let degree = from_vec(degree_f32, &[num_nodes], DeviceType::Cpu).unwrap();

        // Betweenness centrality (simplified implementation)
        let betweenness_values = compute_betweenness_centrality(&adjacency_list, num_nodes);
        let betweenness_f32: Vec<f32> = betweenness_values.into_iter().map(|x| x as f32).collect();
        let betweenness = from_vec(betweenness_f32, &[num_nodes], DeviceType::Cpu).unwrap();

        // Closeness centrality
        let closeness_values = compute_closeness_centrality(&adjacency_list, num_nodes);
        let closeness_f32: Vec<f32> = closeness_values.into_iter().map(|x| x as f32).collect();
        let closeness = from_vec(closeness_f32, &[num_nodes], DeviceType::Cpu).unwrap();

        // Eigenvector centrality (power iteration approximation)
        let eigenvector_values = compute_eigenvector_centrality(&adjacency_list, num_nodes);
        let eigenvector_f32: Vec<f32> = eigenvector_values.into_iter().map(|x| x as f32).collect();
        let eigenvector = from_vec(eigenvector_f32, &[num_nodes], DeviceType::Cpu).unwrap();

        CentralityMeasures {
            degree,
            betweenness,
            closeness,
            eigenvector,
        }
    }

    /// Centrality measures container
    pub struct CentralityMeasures {
        pub degree: Tensor,
        pub betweenness: Tensor,
        pub closeness: Tensor,
        pub eigenvector: Tensor,
    }

    /// Compute clustering coefficient
    pub fn clustering_coefficient(edge_index: &Tensor, num_nodes: usize) -> Tensor {
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes);
        let mut clustering_coeffs = Vec::with_capacity(num_nodes);

        for node in 0..num_nodes {
            let neighbors = &adjacency_list[node];
            let degree = neighbors.len();

            if degree < 2 {
                clustering_coeffs.push(0.0);
                continue;
            }

            // Count triangles
            let mut triangles = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let neighbor1 = neighbors[i];
                    let neighbor2 = neighbors[j];

                    // Check if neighbor1 and neighbor2 are connected
                    if adjacency_list[neighbor1].contains(&neighbor2) {
                        triangles += 1;
                    }
                }
            }

            // Clustering coefficient = 2 * triangles / (degree * (degree - 1))
            let max_edges = degree * (degree - 1) / 2;
            let clustering = if max_edges > 0 {
                triangles as f64 / max_edges as f64
            } else {
                0.0
            };

            clustering_coeffs.push(clustering);
        }

        let coeffs_f32: Vec<f32> = clustering_coeffs.into_iter().map(|x| x as f32).collect();
        from_vec(coeffs_f32, &[num_nodes], DeviceType::Cpu).unwrap()
    }

    /// Compute graph diameter
    pub fn graph_diameter(edge_index: &Tensor, num_nodes: usize) -> usize {
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes);
        let mut max_distance = 0;

        for start_node in 0..num_nodes {
            let distances = bfs_distances(&adjacency_list, start_node, num_nodes);
            for &distance in &distances {
                if distance != usize::MAX && distance > max_distance {
                    max_distance = distance;
                }
            }
        }

        max_distance
    }

    /// Compute betweenness centrality using Brandes' algorithm (simplified)
    fn compute_betweenness_centrality(adjacency_list: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
        let mut betweenness = vec![0.0; num_nodes];

        for s in 0..num_nodes {
            // Single-source shortest-path problem
            let mut stack = Vec::new();
            let mut paths = vec![Vec::new(); num_nodes];
            let mut sigma = vec![0.0; num_nodes];
            let mut distances = vec![-1; num_nodes];
            let mut delta = vec![0.0; num_nodes];

            sigma[s] = 1.0;
            distances[s] = 0;

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);

                for &w in &adjacency_list[v] {
                    // First time we found shortest path to w?
                    if distances[w] < 0 {
                        queue.push_back(w);
                        distances[w] = distances[v] + 1;
                    }
                    // Shortest path to w via v?
                    if distances[w] == distances[v] + 1 {
                        sigma[w] += sigma[v];
                        paths[w].push(v);
                    }
                }
            }

            // Accumulation
            while let Some(w) = stack.pop() {
                for &v in &paths[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    betweenness[w] += delta[w];
                }
            }
        }

        // Normalize
        let normalization = if num_nodes > 2 {
            2.0 / ((num_nodes - 1) * (num_nodes - 2)) as f64
        } else {
            1.0
        };

        betweenness.iter().map(|&x| x * normalization).collect()
    }

    /// Compute closeness centrality
    fn compute_closeness_centrality(adjacency_list: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
        let mut closeness = Vec::with_capacity(num_nodes);

        for node in 0..num_nodes {
            let distances = bfs_distances(adjacency_list, node, num_nodes);
            let mut sum_distances = 0.0;
            let mut reachable_nodes = 0;

            for &distance in &distances {
                if distance != usize::MAX {
                    sum_distances += distance as f64;
                    reachable_nodes += 1;
                }
            }

            let closeness_value = if sum_distances > 0.0 && reachable_nodes > 1 {
                (reachable_nodes - 1) as f64 / sum_distances
            } else {
                0.0
            };

            closeness.push(closeness_value);
        }

        closeness
    }

    /// Compute eigenvector centrality using power iteration
    fn compute_eigenvector_centrality(adjacency_list: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
        let mut centrality = vec![1.0; num_nodes];
        let iterations = 100;
        let tolerance = 1e-6;

        for _ in 0..iterations {
            let mut new_centrality = vec![0.0; num_nodes];

            // Matrix-vector multiplication with adjacency matrix
            for node in 0..num_nodes {
                for &neighbor in &adjacency_list[node] {
                    new_centrality[node] += centrality[neighbor];
                }
            }

            // Normalize
            let norm: f64 = new_centrality.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for value in &mut new_centrality {
                    *value /= norm;
                }
            }

            // Check convergence
            let diff: f64 = centrality
                .iter()
                .zip(&new_centrality)
                .map(|(&old, &new)| (old - new).abs())
                .sum();

            centrality = new_centrality;

            if diff < tolerance {
                break;
            }
        }

        centrality
    }

    /// Compute shortest distances from a source node using BFS
    fn bfs_distances(adjacency_list: &[Vec<usize>], source: usize, num_nodes: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; num_nodes];
        let mut queue = VecDeque::new();

        distances[source] = 0;
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency_list[node] {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[node] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }
}

/// Graph sampling utilities
pub mod sampling {
    use super::*;
    use scirs2_core::random::{Random, Rng};

    /// Sample neighbors for GraphSAGE
    pub fn neighbor_sampling(
        edge_index: &Tensor,
        node_idx: &[usize],
        num_neighbors: usize,
    ) -> Vec<Vec<usize>> {
        let num_nodes = node_idx.iter().max().unwrap_or(&0) + 1;
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes);
        let mut rng = Random::seed(42);
        let mut sampled_neighbors = Vec::with_capacity(node_idx.len());

        for &node in node_idx {
            let neighbors = &adjacency_list[node];

            if neighbors.len() <= num_neighbors {
                // Return all neighbors if we have fewer than requested
                sampled_neighbors.push(neighbors.clone());
            } else {
                // Randomly sample without replacement
                let mut indices: Vec<usize> = (0..neighbors.len()).collect();

                // Fisher-Yates shuffle for first num_neighbors elements
                for i in 0..num_neighbors {
                    let j = rng.gen_range(i..indices.len());
                    indices.swap(i, j);
                }

                let sampled: Vec<usize> = indices[..num_neighbors]
                    .iter()
                    .map(|&i| neighbors[i])
                    .collect();

                sampled_neighbors.push(sampled);
            }
        }

        sampled_neighbors
    }

    /// Random walk sampling
    pub fn random_walk(
        edge_index: &Tensor,
        start_nodes: &[usize],
        walk_length: usize,
    ) -> Vec<Vec<usize>> {
        let num_nodes = start_nodes.iter().max().unwrap_or(&0) + 1;
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes);
        let mut rng = Random::seed(42);
        let mut walks = Vec::with_capacity(start_nodes.len());

        for &start_node in start_nodes {
            let mut walk = vec![start_node];
            let mut current_node = start_node;

            for _ in 1..walk_length {
                let neighbors = &adjacency_list[current_node];

                if neighbors.is_empty() {
                    break; // Dead end
                }

                // Randomly select next node
                let next_idx = rng.gen_range(0..neighbors.len());
                current_node = neighbors[next_idx];
                walk.push(current_node);
            }

            walks.push(walk);
        }

        walks
    }

    /// Subgraph sampling using random node sampling
    pub fn subgraph_sampling(
        edge_index: &Tensor,
        num_nodes: usize,
        sample_size: usize,
    ) -> (Tensor, Vec<usize>) {
        let mut rng = Random::seed(42);

        // Sample nodes randomly
        let mut sampled_nodes: Vec<usize> = (0..num_nodes).collect();

        // Fisher-Yates shuffle
        for i in 0..sample_size.min(num_nodes) {
            let j = rng.gen_range(i..sampled_nodes.len());
            sampled_nodes.swap(i, j);
        }
        sampled_nodes.truncate(sample_size.min(num_nodes));
        sampled_nodes.sort();

        // Create node mapping
        let mut node_map = std::collections::HashMap::new();
        for (new_idx, &old_idx) in sampled_nodes.iter().enumerate() {
            node_map.insert(old_idx, new_idx);
        }

        // Extract subgraph edges
        let edge_data = tensor_to_vec2::<f32>(edge_index).unwrap();
        let mut subgraph_edges = Vec::new();

        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;

            // Include edge if both nodes are in the sampled set
            if let (Some(&new_src), Some(&new_dst)) = (node_map.get(&src), node_map.get(&dst)) {
                subgraph_edges.push([new_src as i64, new_dst as i64]);
            }
        }

        // Create edge tensor
        let subgraph_edge_index = if subgraph_edges.is_empty() {
            zeros(&[2, 0]).unwrap()
        } else {
            let num_edges = subgraph_edges.len();
            let mut edge_vec = Vec::with_capacity(2 * num_edges);

            for edge in &subgraph_edges {
                edge_vec.push(edge[0] as f32);
            }
            for edge in &subgraph_edges {
                edge_vec.push(edge[1] as f32);
            }

            from_vec(edge_vec, &[2, num_edges], DeviceType::Cpu).unwrap()
        };

        (subgraph_edge_index, sampled_nodes)
    }

    /// FastGCN sampling for efficient graph convolution
    pub fn fastgcn_sampling(
        edge_index: &Tensor,
        layer_sizes: &[usize],
        num_nodes: usize,
    ) -> Vec<Vec<usize>> {
        let mut rng = Random::seed(42);
        let adjacency_list = super::connectivity::build_adjacency_list(edge_index, num_nodes);

        // Compute importance sampling probabilities based on node degrees
        let degrees: Vec<f64> = adjacency_list
            .iter()
            .map(|neighbors| neighbors.len() as f64)
            .collect();

        let total_degree: f64 = degrees.iter().sum();
        let probabilities: Vec<f64> = degrees.iter().map(|&d| d / total_degree).collect();

        let mut sampled_layers = Vec::with_capacity(layer_sizes.len());

        for &layer_size in layer_sizes {
            let mut sampled_nodes = Vec::new();

            // Importance sampling
            for _ in 0..layer_size.min(num_nodes) {
                let mut cumsum = 0.0;
                let random_val = rng.gen::<f64>();

                for (node, &prob) in probabilities.iter().enumerate() {
                    cumsum += prob;
                    if random_val <= cumsum {
                        sampled_nodes.push(node);
                        break;
                    }
                }
            }

            sampled_nodes.sort();
            sampled_nodes.dedup();
            sampled_layers.push(sampled_nodes);
        }

        sampled_layers
    }
}

// Memory-efficient graph operations optimized with SciRS2
// Temporarily commented out due to API compatibility issues
/*
pub mod memory_efficient {
    use super::*;
    use crate::{GraphData, GraphMemoryStats};
    use torsh_tensor::{creation::{from_vec, zeros}, Tensor};
    use torsh_core::device::DeviceType;

    /// Sparse graph representation for memory efficiency
    #[derive(Debug, Clone)]
    pub struct SparseGraph {
        pub edge_list: Vec<(usize, usize)>,
        pub node_features: Option<Tensor>,
        pub edge_weights: Option<Vec<f32>>,
        pub num_nodes: usize,
        pub num_edges: usize,
    }

    impl SparseGraph {
        /// Create from dense adjacency matrix with sparsity threshold
        pub fn from_dense(adjacency: &Tensor, threshold: f32) -> Self {
            let binding = adjacency.shape();
            let shape = binding.dims();
            let num_nodes = shape[0];
            let data = adjacency.to_vec().unwrap();
            let mut edge_list = Vec::new();
            let mut edge_weights = Vec::new();

            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    let weight = data[i * num_nodes + j];
                    if weight.abs() > threshold {
                        edge_list.push((i, j));
                        edge_weights.push(weight);
                    }
                }
            }

            let num_edges = edge_list.len();
            Self {
                edge_list,
                node_features: None,
                edge_weights: Some(edge_weights),
                num_nodes,
                num_edges,
            }
        }

        /// Convert to dense edge_index format
        pub fn to_edge_index(&self) -> Tensor {
            if self.edge_list.is_empty() {
                return zeros(&[2, 0]).unwrap();
            }

            let mut edge_vec = Vec::with_capacity(2 * self.num_edges);

            for &(src, _) in &self.edge_list {
                edge_vec.push(src as i64);
            }
            for &(_, dst) in &self.edge_list {
                edge_vec.push(dst as i64);
            }

            from_vec(edge_vec, &[2, self.num_edges], DeviceType::Cpu).unwrap()
        }

        /// Get memory footprint in bytes
        pub fn memory_footprint(&self) -> usize {
            let edge_list_size = self.edge_list.len() * std::mem::size_of::<(usize, usize)>();
            let features_size = self.node_features.as_ref()
                .map(|t| t.numel() * std::mem::size_of::<f32>())
                .unwrap_or(0);
            let weights_size = self.edge_weights.as_ref()
                .map(|w| w.len() * std::mem::size_of::<f32>())
                .unwrap_or(0);

            edge_list_size + features_size + weights_size
        }
    }

    /// Memory-efficient graph laplacian computation
    pub fn sparse_laplacian(edge_index: &Tensor, num_nodes: usize, normalized: bool) -> SparseGraph {
        // Use coordinate format (COO) to build Laplacian efficiently
        let edge_data = tensor_to_vec2::<f32>(edge_index).unwrap();
        let mut edges = Vec::new();
        let mut values = Vec::new();

        // Count node degrees first
        let mut degrees = vec![0; num_nodes];
        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;

            degrees[src] += 1;
            degrees[dst] += 1;
        }

        if normalized {
            // Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
            // Add diagonal entries (I)
            for i in 0..num_nodes {
                edges.push((i, i));
                values.push(1.0);
            }

            // Add off-diagonal entries with normalization
            for j in 0..edge_data[0].len() {
                let src = edge_data[0][j] as usize;
                let dst = edge_data[1][j] as usize;

                let norm_factor = if degrees[src] > 0 && degrees[dst] > 0 {
                    -1.0 / ((degrees[src] as f32).sqrt() * (degrees[dst] as f32).sqrt())
                } else {
                    0.0
                };

                edges.push((src, dst));
                values.push(norm_factor);

                if src != dst {
                    edges.push((dst, src));
                    values.push(norm_factor);
                }
            }
        } else {
            // Unnormalized Laplacian: L = D - A
            // Add diagonal entries (D)
            for i in 0..num_nodes {
                edges.push((i, i));
                values.push(degrees[i] as f32);
            }

            // Subtract adjacency entries (-A)
            for j in 0..edge_data[0].len() {
                let src = edge_data[0][j] as usize;
                let dst = edge_data[1][j] as usize;

                edges.push((src, dst));
                values.push(-1.0);

                if src != dst {
                    edges.push((dst, src));
                    values.push(-1.0);
                }
            }
        }

        SparseGraph {
            edge_list: edges,
            node_features: None,
            edge_weights: Some(values),
            num_nodes,
            num_edges: edges.len(),
        }
    }

    /// Chunk-based processing for large graphs
    pub fn chunked_graph_processing<F, R>(
        graph: &GraphData,
        chunk_size: usize,
        processor: F,
    ) -> Vec<R>
    where
        F: Fn(&GraphData) -> R,
    {
        let mut results = Vec::new();
        let total_nodes = graph.num_nodes;

        for chunk_start in (0..total_nodes).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_nodes);

            // Extract subgraph for this chunk
            let chunk_nodes: Vec<usize> = (chunk_start..chunk_end).collect();
            let (chunk_edge_index, _) = super::sampling::subgraph_sampling(
                &graph.edge_index,
                total_nodes,
                chunk_end - chunk_start,
            );

            // Extract node features for this chunk
            let chunk_features = graph.x
                .slice_tensor(0, chunk_start as i64, chunk_end as i64)
                .unwrap();

            let chunk_graph = GraphData::new(chunk_features, chunk_edge_index);
            results.push(processor(&chunk_graph));
        }

        results
    }

    /// Memory-efficient batch processing with automatic memory management
    pub fn memory_aware_batch_processing(
        graphs: &[GraphData],
        memory_limit_mb: usize,
        process_fn: impl Fn(&[GraphData]) -> Vec<Tensor>,
    ) -> Vec<Tensor> {
        let memory_limit_bytes = memory_limit_mb * 1024 * 1024;
        let mut results = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_memory = 0;

        for graph in graphs {
            let graph_memory = graph.memory_stats().total_bytes;

            // If adding this graph would exceed memory limit, process current batch
            if current_memory + graph_memory > memory_limit_bytes && !current_batch.is_empty() {
                let batch_results = process_fn(&current_batch);
                results.extend(batch_results);

                current_batch.clear();
                current_memory = 0;
            }

            current_batch.push(graph.clone());
            current_memory += graph_memory;
        }

        // Process remaining batch
        if !current_batch.is_empty() {
            let batch_results = process_fn(&current_batch);
            results.extend(batch_results);
        }

        results
    }

    /// Adaptive graph coarsening for memory efficiency
    pub fn adaptive_coarsening(graph: &GraphData, target_nodes: usize) -> GraphData {
        if graph.num_nodes <= target_nodes {
            return graph.clone();
        }

        // Simple coarsening: merge nodes based on similarity
        let adjacency_list = super::connectivity::build_adjacency_list(&graph.edge_index, graph.num_nodes);
        let mut node_clusters = Vec::new();
        let mut visited = vec![false; graph.num_nodes];

        for node in 0..graph.num_nodes {
            if !visited[node] && node_clusters.len() < target_nodes {
                let mut cluster = vec![node];
                visited[node] = true;

                // Add similar neighbors to cluster (simple heuristic)
                for &neighbor in &adjacency_list[node] {
                    if !visited[neighbor] && cluster.len() < graph.num_nodes / target_nodes {
                        cluster.push(neighbor);
                        visited[neighbor] = true;
                    }
                }

                node_clusters.push(cluster);
            }
        }

        // Create coarsened graph
        let coarsened_nodes = node_clusters.len();
        let mut coarsened_features = Vec::new();
        let mut coarsened_edges = Vec::new();

        // Aggregate node features by averaging within clusters
        for cluster in &node_clusters {
            let mut cluster_features = vec![0.0; graph.x.shape().dims()[1]];

            for &node in cluster {
                let node_features = graph.x
                    .slice_tensor(0, node as i64, (node + 1) as i64)
                    .unwrap()
                    .to_vec()
                    .unwrap();

                for (i, &feat) in node_features.iter().enumerate() {
                    cluster_features[i] += feat;
                }
            }

            // Average the features
            for feat in &mut cluster_features {
                *feat /= cluster.len() as f32;
            }

            coarsened_features.extend(cluster_features);
        }

        // Create cluster mapping
        let mut node_to_cluster = vec![0; graph.num_nodes];
        for (cluster_id, cluster) in node_clusters.iter().enumerate() {
            for &node in cluster {
                node_to_cluster[node] = cluster_id;
            }
        }

        // Build coarsened edges
        let edge_data = tensor_to_vec2::<i64>(&graph.edge_index).unwrap();
        let mut edge_set = std::collections::HashSet::new();

        for j in 0..edge_data[0].len() {
            let src = edge_data[0][j] as usize;
            let dst = edge_data[1][j] as usize;

            let src_cluster = node_to_cluster[src];
            let dst_cluster = node_to_cluster[dst];

            // Only add edges between different clusters
            if src_cluster != dst_cluster {
                edge_set.insert((src_cluster.min(dst_cluster), src_cluster.max(dst_cluster)));
            }
        }

        // Convert to edge format
        for (src, dst) in edge_set {
            coarsened_edges.push(src as i64);
            coarsened_edges.push(dst as i64);
        }

        let num_coarsened_edges = coarsened_edges.len() / 2;

        let coarsened_x = from_vec(
            coarsened_features,
            &[coarsened_nodes, graph.x.shape().dims()[1]],
            DeviceType::Cpu
        ).unwrap();

        let coarsened_edge_index = if num_coarsened_edges > 0 {
            from_vec(coarsened_edges, &[2, num_coarsened_edges], DeviceType::Cpu).unwrap()
        } else {
            zeros(&[2, 0]).unwrap()
        };

        GraphData::new(coarsened_x, coarsened_edge_index)
    }
}
*/
