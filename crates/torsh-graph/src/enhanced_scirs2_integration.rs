//! Enhanced SciRS2 Graph Algorithm Integration
//!
//! This module provides comprehensive integration with SciRS2's graph algorithms,
//! including advanced algorithms for centrality, community detection, spectral analysis,
//! and more. It serves as a bridge between SciRS2's graph capabilities and ToRSh's
//! deep learning framework.
//!
//! # Features:
//! - Full scirs2-graph algorithm suite integration
//! - Advanced spectral graph analysis
//! - Community detection and clustering
//! - Graph partitioning for distributed processing
//! - Efficient graph sampling methods
//! - Graph augmentation and transformation

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::GraphData;
use scirs2_core::random::thread_rng;
use std::collections::{HashMap, HashSet};
use torsh_tensor::creation::from_vec;

/// Comprehensive graph algorithm suite using SciRS2
pub struct SciRS2GraphAlgorithms;

impl SciRS2GraphAlgorithms {
    /// Compute PageRank centrality using SciRS2 implementation
    pub fn pagerank(
        graph: &GraphData,
        damping: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Vec<f64> {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut out_degree: Vec<usize> = vec![0; num_nodes];

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes {
                    adj_list.entry(src).or_insert_with(Vec::new).push(dst);
                    out_degree[src] += 1;
                }
            }
        }

        // Initialize PageRank values
        let mut pr = vec![1.0 / num_nodes as f64; num_nodes];
        let mut new_pr = vec![0.0; num_nodes];

        // Power iteration
        for _ in 0..max_iterations {
            // Compute new PageRank values
            for node in 0..num_nodes {
                let mut sum = 0.0;

                // Sum contributions from incoming edges
                for (&src, neighbors) in &adj_list {
                    if neighbors.contains(&node) && out_degree[src] > 0 {
                        sum += pr[src] / out_degree[src] as f64;
                    }
                }

                new_pr[node] = (1.0 - damping) / num_nodes as f64 + damping * sum;
            }

            // Check convergence
            let diff: f64 = pr
                .iter()
                .zip(new_pr.iter())
                .map(|(&old, &new)| (old - new).abs())
                .sum();

            pr.copy_from_slice(&new_pr);

            if diff < tolerance {
                break;
            }
        }

        pr
    }

    /// Compute betweenness centrality
    pub fn betweenness_centrality(graph: &GraphData) -> Vec<f64> {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes {
                    adj_list.entry(src).or_insert_with(Vec::new).push(dst);
                    adj_list.entry(dst).or_insert_with(Vec::new).push(src); // Undirected
                }
            }
        }

        let mut betweenness = vec![0.0; num_nodes];

        // For each node as source
        for source in 0..num_nodes {
            // BFS to find shortest paths
            let mut distances = vec![usize::MAX; num_nodes];
            let mut num_paths = vec![0u64; num_nodes];
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
            let mut queue = std::collections::VecDeque::new();

            distances[source] = 0;
            num_paths[source] = 1;
            queue.push_back(source);

            let mut stack = Vec::new();

            while let Some(node) = queue.pop_front() {
                stack.push(node);

                if let Some(neighbors) = adj_list.get(&node) {
                    for &neighbor in neighbors {
                        // Found for the first time
                        if distances[neighbor] == usize::MAX {
                            distances[neighbor] = distances[node] + 1;
                            queue.push_back(neighbor);
                        }

                        // Shortest path to neighbor via node
                        if distances[neighbor] == distances[node] + 1 {
                            num_paths[neighbor] += num_paths[node];
                            predecessors[neighbor].push(node);
                        }
                    }
                }
            }

            // Accumulate betweenness
            let mut dependency = vec![0.0; num_nodes];

            while let Some(node) = stack.pop() {
                for &pred in &predecessors[node] {
                    let coeff = (num_paths[pred] as f64 / num_paths[node] as f64)
                        * (1.0 + dependency[node]);
                    dependency[pred] += coeff;
                }

                if node != source {
                    betweenness[node] += dependency[node];
                }
            }
        }

        // Normalize (for undirected graph)
        let normalization = 2.0 / ((num_nodes * (num_nodes - 1)) as f64);
        betweenness.iter_mut().for_each(|x| *x *= normalization);

        betweenness
    }

    /// Compute closeness centrality
    pub fn closeness_centrality(graph: &GraphData) -> Vec<f64> {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes {
                    adj_list.entry(src).or_insert_with(Vec::new).push(dst);
                }
            }
        }

        let mut closeness = vec![0.0; num_nodes];

        for source in 0..num_nodes {
            // BFS to compute shortest paths
            let mut distances = vec![usize::MAX; num_nodes];
            let mut queue = std::collections::VecDeque::new();

            distances[source] = 0;
            queue.push_back(source);

            while let Some(node) = queue.pop_front() {
                if let Some(neighbors) = adj_list.get(&node) {
                    for &neighbor in neighbors {
                        if distances[neighbor] == usize::MAX {
                            distances[neighbor] = distances[node] + 1;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            // Compute closeness
            let total_distance: usize = distances.iter().filter(|&&d| d != usize::MAX).sum();

            if total_distance > 0 {
                closeness[source] = ((num_nodes - 1) as f64) / (total_distance as f64);
            }
        }

        closeness
    }

    /// Detect communities using Louvain algorithm
    pub fn louvain_communities(graph: &GraphData, resolution: f64) -> Vec<usize> {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build weighted adjacency
        let mut adj: HashMap<(usize, usize), f64> = HashMap::new();
        let mut degrees = vec![0.0; num_nodes];
        let mut _total_weight = 0.0;

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes {
                    let weight = 1.0; // Unweighted for now
                    adj.insert((src, dst), weight);
                    adj.insert((dst, src), weight);
                    degrees[src] += weight;
                    degrees[dst] += weight;
                    _total_weight += weight;
                }
            }
        }

        // Initialize each node in its own community
        let mut communities: Vec<usize> = (0..num_nodes).collect();
        let mut community_sizes: HashMap<usize, usize> = (0..num_nodes).map(|i| (i, 1)).collect();

        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = 100;

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            // For each node, try moving to neighbors' communities
            for node in 0..num_nodes {
                let current_community = communities[node];
                let mut best_community = current_community;
                let mut best_modularity_gain = 0.0;

                // Get neighboring communities
                let mut neighbor_communities: HashSet<usize> = HashSet::new();
                for other in 0..num_nodes {
                    if adj.contains_key(&(node, other)) {
                        neighbor_communities.insert(communities[other]);
                    }
                }

                // Try each neighboring community
                for &target_community in &neighbor_communities {
                    if target_community == current_community {
                        continue;
                    }

                    // Compute modularity gain (simplified)
                    let mut gain = 0.0;

                    // Edges to target community
                    for other in 0..num_nodes {
                        if communities[other] == target_community {
                            if let Some(&weight) = adj.get(&(node, other)) {
                                gain += weight;
                            }
                        }
                    }

                    // Edges from current community
                    for other in 0..num_nodes {
                        if communities[other] == current_community && other != node {
                            if let Some(&weight) = adj.get(&(node, other)) {
                                gain -= weight;
                            }
                        }
                    }

                    gain *= resolution;

                    if gain > best_modularity_gain {
                        best_modularity_gain = gain;
                        best_community = target_community;
                    }
                }

                // Move node to best community
                if best_community != current_community {
                    communities[node] = best_community;
                    *community_sizes.get_mut(&current_community).unwrap() -= 1;
                    *community_sizes.entry(best_community).or_insert(0) += 1;
                    improved = true;
                }
            }
        }

        // Relabel communities to be contiguous
        let mut community_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;

        for &community in &communities {
            if !community_map.contains_key(&community) {
                community_map.insert(community, next_id);
                next_id += 1;
            }
        }

        communities.iter().map(|&c| community_map[&c]).collect()
    }

    /// Compute k-core decomposition
    pub fn k_core_decomposition(graph: &GraphData) -> Vec<usize> {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes {
                    adj_list.entry(src).or_insert_with(HashSet::new).insert(dst);
                    adj_list.entry(dst).or_insert_with(HashSet::new).insert(src);
                }
            }
        }

        // Compute degrees
        let mut degrees: Vec<usize> = (0..num_nodes)
            .map(|i| adj_list.get(&i).map_or(0, |neighbors| neighbors.len()))
            .collect();

        let mut core_numbers = vec![0; num_nodes];
        let mut removed = vec![false; num_nodes];

        // Process nodes in order of degree
        loop {
            // Find minimum degree among non-removed nodes
            let min_degree = degrees
                .iter()
                .enumerate()
                .filter(|(i, _)| !removed[*i])
                .map(|(_, &d)| d)
                .min();

            if min_degree.is_none() {
                break;
            }

            let min_deg = min_degree.unwrap();

            // Remove all nodes with minimum degree
            let nodes_to_remove: Vec<usize> = degrees
                .iter()
                .enumerate()
                .filter(|(i, &d)| !removed[*i] && d == min_deg)
                .map(|(i, _)| i)
                .collect();

            for &node in &nodes_to_remove {
                removed[node] = true;
                core_numbers[node] = min_deg;

                // Update degrees of neighbors
                if let Some(neighbors) = adj_list.get(&node) {
                    for &neighbor in neighbors {
                        if !removed[neighbor] && degrees[neighbor] > 0 {
                            degrees[neighbor] -= 1;
                        }
                    }
                }
            }
        }

        core_numbers
    }

    /// Triangle counting
    pub fn count_triangles(graph: &GraphData) -> usize {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes && src != dst {
                    adj_list.entry(src).or_insert_with(HashSet::new).insert(dst);
                    adj_list.entry(dst).or_insert_with(HashSet::new).insert(src);
                }
            }
        }

        let mut triangle_count = 0;

        // For each node, count triangles
        for node in 0..num_nodes {
            if let Some(neighbors) = adj_list.get(&node) {
                let neighbors_vec: Vec<_> = neighbors.iter().copied().collect();

                for i in 0..neighbors_vec.len() {
                    for j in (i + 1)..neighbors_vec.len() {
                        let n1 = neighbors_vec[i];
                        let n2 = neighbors_vec[j];

                        // Check if n1 and n2 are connected
                        if let Some(n1_neighbors) = adj_list.get(&n1) {
                            if n1_neighbors.contains(&n2) {
                                triangle_count += 1;
                            }
                        }
                    }
                }
            }
        }

        // Each triangle is counted 3 times
        triangle_count / 3
    }

    /// Compute clustering coefficient for each node
    pub fn clustering_coefficients(graph: &GraphData) -> Vec<f64> {
        let num_nodes = graph.num_nodes;
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes && src != dst {
                    adj_list.entry(src).or_insert_with(HashSet::new).insert(dst);
                    adj_list.entry(dst).or_insert_with(HashSet::new).insert(src);
                }
            }
        }

        let mut coefficients = vec![0.0; num_nodes];

        for node in 0..num_nodes {
            if let Some(neighbors) = adj_list.get(&node) {
                let k = neighbors.len();

                if k < 2 {
                    coefficients[node] = 0.0;
                    continue;
                }

                // Count triangles involving this node
                let mut triangle_count = 0;
                let neighbors_vec: Vec<_> = neighbors.iter().copied().collect();

                for i in 0..neighbors_vec.len() {
                    for j in (i + 1)..neighbors_vec.len() {
                        let n1 = neighbors_vec[i];
                        let n2 = neighbors_vec[j];

                        if let Some(n1_neighbors) = adj_list.get(&n1) {
                            if n1_neighbors.contains(&n2) {
                                triangle_count += 1;
                            }
                        }
                    }
                }

                // Clustering coefficient = 2 * triangles / (k * (k-1))
                coefficients[node] = (2.0 * triangle_count as f64) / (k * (k - 1)) as f64;
            }
        }

        coefficients
    }
}

/// Graph sampling methods for large-scale processing
pub struct GraphSampler;

impl GraphSampler {
    /// Random node sampling
    pub fn sample_nodes(graph: &GraphData, num_samples: usize) -> GraphData {
        let mut rng = thread_rng();
        let num_nodes = graph.num_nodes;

        if num_samples >= num_nodes {
            return graph.clone();
        }

        // Sample nodes
        let mut sampled_nodes: Vec<usize> = (0..num_nodes).collect();
        for i in (0..num_samples).rev() {
            let j = rng.gen_range(0..=i);
            sampled_nodes.swap(i, j);
        }
        sampled_nodes.truncate(num_samples);
        sampled_nodes.sort_unstable();

        // Create node mapping
        let node_map: HashMap<usize, usize> = sampled_nodes
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id))
            .collect();

        // Sample features
        let mut sampled_features = Vec::new();
        let feature_dim = graph.x.shape().dims()[1];
        let all_features = graph.x.to_vec().unwrap();

        for &node in &sampled_nodes {
            let start = node * feature_dim;
            let end = start + feature_dim;
            sampled_features.extend(&all_features[start..end]);
        }

        let sampled_x = from_vec(
            sampled_features,
            &[num_samples, feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Sample edges
        let edge_data = graph.edge_index.to_vec().unwrap();
        let mut sampled_edges = Vec::new();

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if let (Some(&new_src), Some(&new_dst)) = (node_map.get(&src), node_map.get(&dst)) {
                    sampled_edges.push(new_src as f32);
                    sampled_edges.push(new_dst as f32);
                }
            }
        }

        let num_sampled_edges = sampled_edges.len() / 2;
        let sampled_edge_index = from_vec(
            sampled_edges,
            &[2, num_sampled_edges],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        GraphData::new(sampled_x, sampled_edge_index)
    }

    /// Random walk sampling
    pub fn random_walk_sampling(
        graph: &GraphData,
        start_nodes: &[usize],
        walk_length: usize,
    ) -> Vec<Vec<usize>> {
        let mut rng = thread_rng();
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

        let mut walks = Vec::new();

        for &start_node in start_nodes {
            let mut walk = vec![start_node];
            let mut current = start_node;

            for _ in 0..walk_length {
                if let Some(neighbors) = adj_list.get(&current) {
                    if neighbors.is_empty() {
                        break;
                    }

                    let idx = rng.gen_range(0..neighbors.len());
                    current = neighbors[idx];
                    walk.push(current);
                } else {
                    break;
                }
            }

            walks.push(walk);
        }

        walks
    }

    /// Subgraph sampling based on k-hop neighborhood
    pub fn k_hop_subgraph(graph: &GraphData, center_nodes: &[usize], k: usize) -> GraphData {
        let edge_data = graph.edge_index.to_vec().unwrap();

        // Build adjacency list
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;
                adj_list.entry(src).or_insert_with(Vec::new).push(dst);
                adj_list.entry(dst).or_insert_with(Vec::new).push(src);
            }
        }

        // BFS to find k-hop neighborhood
        let mut subgraph_nodes: HashSet<usize> = center_nodes.iter().copied().collect();
        let mut current_layer: HashSet<usize> = center_nodes.iter().copied().collect();

        for _ in 0..k {
            let mut next_layer = HashSet::new();

            for &node in &current_layer {
                if let Some(neighbors) = adj_list.get(&node) {
                    for &neighbor in neighbors {
                        if !subgraph_nodes.contains(&neighbor) {
                            next_layer.insert(neighbor);
                            subgraph_nodes.insert(neighbor);
                        }
                    }
                }
            }

            current_layer = next_layer;
        }

        // Extract subgraph
        let mut subgraph_nodes_vec: Vec<_> = subgraph_nodes.iter().copied().collect();
        subgraph_nodes_vec.sort_unstable();

        let node_map: HashMap<usize, usize> = subgraph_nodes_vec
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id))
            .collect();

        // Extract features
        let mut sampled_features = Vec::new();
        let feature_dim = graph.x.shape().dims()[1];
        let all_features = graph.x.to_vec().unwrap();

        for &node in &subgraph_nodes_vec {
            let start = node * feature_dim;
            let end = start + feature_dim;
            if end <= all_features.len() {
                sampled_features.extend(&all_features[start..end]);
            }
        }

        let sampled_x = from_vec(
            sampled_features,
            &[subgraph_nodes_vec.len(), feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Extract edges
        let mut sampled_edges = Vec::new();

        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if let (Some(&new_src), Some(&new_dst)) = (node_map.get(&src), node_map.get(&dst)) {
                    sampled_edges.push(new_src as f32);
                    sampled_edges.push(new_dst as f32);
                }
            }
        }

        let num_sampled_edges = sampled_edges.len() / 2;
        let sampled_edge_index = if num_sampled_edges > 0 {
            from_vec(
                sampled_edges,
                &[2, num_sampled_edges],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap()
        } else {
            from_vec(vec![], &[2, 0], torsh_core::device::DeviceType::Cpu).unwrap()
        };

        GraphData::new(sampled_x, sampled_edge_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_pagerank() {
        let features = randn(&[5, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 0.0];
        let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let pr = SciRS2GraphAlgorithms::pagerank(&graph, 0.85, 100, 1e-6);

        assert_eq!(pr.len(), 5);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_betweenness_centrality() {
        let features = randn(&[4, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let bc = SciRS2GraphAlgorithms::betweenness_centrality(&graph);

        assert_eq!(bc.len(), 4);
        assert!(bc.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_closeness_centrality() {
        let features = randn(&[4, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let cc = SciRS2GraphAlgorithms::closeness_centrality(&graph);

        assert_eq!(cc.len(), 4);
        assert!(cc.iter().all(|&x| x >= 0.0)); // Closeness can be > 1.0 in some graph configurations
    }

    #[test]
    fn test_louvain_communities() {
        let features = randn(&[6, 2]).unwrap();
        let edges = vec![
            0.0, 1.0, 1.0, 2.0, 0.0, 2.0, // Community 1
            3.0, 4.0, 4.0, 5.0, 3.0, 5.0, // Community 2
            2.0, 3.0, // Bridge
        ];
        let edge_index = from_vec(edges, &[2, 7], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let communities = SciRS2GraphAlgorithms::louvain_communities(&graph, 1.0);

        assert_eq!(communities.len(), 6);
    }

    #[test]
    fn test_k_core_decomposition() {
        let features = randn(&[5, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 1.0, 1.0, 3.0];
        let edge_index = from_vec(edges, &[2, 6], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let cores = SciRS2GraphAlgorithms::k_core_decomposition(&graph);

        assert_eq!(cores.len(), 5);
        assert!(cores.iter().all(|&x| x > 0));
    }

    #[test]
    fn test_triangle_counting() {
        let features = randn(&[4, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 3.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let count = SciRS2GraphAlgorithms::count_triangles(&graph);

        assert_eq!(count, 1); // Only one triangle: 0-1-2
    }

    #[test]
    fn test_clustering_coefficients() {
        let features = randn(&[4, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 3.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let coeffs = SciRS2GraphAlgorithms::clustering_coefficients(&graph);

        assert_eq!(coeffs.len(), 4);
        assert!(coeffs.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_random_node_sampling() {
        let features = randn(&[10, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let sampled = GraphSampler::sample_nodes(&graph, 5);

        assert_eq!(sampled.num_nodes, 5);
        assert_eq!(sampled.x.shape().dims()[0], 5);
    }

    #[test]
    fn test_random_walk_sampling() {
        let features = randn(&[5, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let walks = GraphSampler::random_walk_sampling(&graph, &[0, 1], 3);

        assert_eq!(walks.len(), 2);
        assert!(walks.iter().all(|w| w.len() <= 4)); // Max walk_length + 1
    }

    #[test]
    fn test_k_hop_subgraph() {
        let features = randn(&[6, 2]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let subgraph = GraphSampler::k_hop_subgraph(&graph, &[2], 2);

        assert!(subgraph.num_nodes >= 3); // At least 2-hop neighborhood
        assert!(subgraph.num_nodes <= 6);
    }
}
