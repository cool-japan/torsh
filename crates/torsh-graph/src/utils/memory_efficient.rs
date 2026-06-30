//! Memory-efficient graph operations.
//!
//! This module provides utilities that keep memory usage proportional to the
//! number of *edges* rather than the number of *node pairs*:
//!
//! - [`SparseGraph`] stores a graph in coordinate (COO) form, extracting only
//!   the non-negligible entries of a dense adjacency matrix.
//! - [`sparse_laplacian`] builds the (optionally symmetric-normalized) graph
//!   Laplacian directly in COO form without ever materializing the dense
//!   `num_nodes x num_nodes` matrix.
//! - [`adaptive_coarsening`] reduces a graph to a target number of supernodes
//!   using greedy edge-contraction (union-find), averaging the node features of
//!   each contracted cluster.
//! - [`chunked_neighbor_aggregation`] performs mean neighbor aggregation using a
//!   sparse adjacency list (`O(E)` memory) and processes destination nodes in
//!   bounded-size chunks for cache locality.

use std::collections::{HashMap, HashSet};

use super::tensor_to_vec2;
use crate::GraphData;
use torsh_core::device::DeviceType;
use torsh_tensor::{
    creation::{from_vec, zeros},
    Tensor,
};

/// Sparse coordinate (COO) representation of a graph or graph operator.
///
/// Each stored entry is a `(row, col)` coordinate together with an associated
/// weight. This is used both for sparsified adjacency matrices (see
/// [`SparseGraph::from_dense`]) and for sparse operators such as the graph
/// Laplacian (see [`sparse_laplacian`]).
#[derive(Debug, Clone)]
pub struct SparseGraph {
    /// Stored `(row, col)` coordinates of non-zero entries.
    pub edge_list: Vec<(usize, usize)>,
    /// Optional dense node-feature matrix associated with the graph.
    pub node_features: Option<Tensor>,
    /// Weight associated with each coordinate in `edge_list` (same length).
    pub edge_weights: Option<Vec<f32>>,
    /// Number of nodes the operator is defined over.
    pub num_nodes: usize,
    /// Number of stored entries (`edge_list.len()`).
    pub num_edges: usize,
}

impl SparseGraph {
    /// Build a sparse graph from a dense adjacency matrix, keeping only entries
    /// whose magnitude strictly exceeds `threshold`.
    ///
    /// The matrix is interpreted as a `rows x cols` tensor; `num_nodes` is taken
    /// from the number of rows. Entries are scanned in row-major order so the
    /// resulting `edge_list` is sorted by `(row, col)`.
    pub fn from_dense(adjacency: &Tensor, threshold: f32) -> Self {
        let shape = adjacency.shape();
        let dims = shape.dims();
        let rows = dims[0];
        let cols = if dims.len() > 1 { dims[1] } else { 1 };
        let num_nodes = rows;

        let data = adjacency
            .to_vec()
            .expect("dense adjacency tensor must be convertible to a vector");

        let mut edge_list = Vec::new();
        let mut edge_weights = Vec::new();

        for i in 0..rows {
            let row_base = i * cols;
            for j in 0..cols {
                let weight = data[row_base + j];
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

    /// Convert the stored coordinates into a `[2, num_edges]` edge-index tensor.
    ///
    /// The first row holds source indices and the second holds destinations,
    /// matching the convention used by [`GraphData`]. An empty graph yields a
    /// `[2, 0]` tensor.
    pub fn to_edge_index(&self) -> Tensor {
        if self.edge_list.is_empty() {
            return zeros(&[2, 0]).expect("zeros([2, 0]) is a valid empty edge index");
        }

        let mut edge_vec = Vec::with_capacity(2 * self.edge_list.len());
        for &(src, _) in &self.edge_list {
            edge_vec.push(src as f32);
        }
        for &(_, dst) in &self.edge_list {
            edge_vec.push(dst as f32);
        }

        from_vec(edge_vec, &[2, self.edge_list.len()], DeviceType::Cpu)
            .expect("edge buffer length matches [2, num_edges]")
    }

    /// Total memory occupied by this representation, in bytes.
    ///
    /// This counts the inline size of the struct itself plus all heap
    /// allocations it owns (the coordinate list, the optional node-feature
    /// tensor, and the optional weight vector). The value is therefore always
    /// strictly positive, even for a graph with no stored edges, because the
    /// structure's own fields still occupy memory.
    pub fn memory_footprint(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let edge_list_bytes = self.edge_list.capacity() * std::mem::size_of::<(usize, usize)>();
        let feature_bytes = self
            .node_features
            .as_ref()
            .map(|t| t.numel() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let weight_bytes = self
            .edge_weights
            .as_ref()
            .map(|w| w.capacity() * std::mem::size_of::<f32>())
            .unwrap_or(0);

        base + edge_list_bytes + feature_bytes + weight_bytes
    }

    /// Fraction of possible directed entries that are actually stored.
    ///
    /// Returns `num_edges / num_nodes^2`, or `0.0` for an empty node set.
    pub fn density(&self) -> f32 {
        if self.num_nodes == 0 {
            return 0.0;
        }
        let possible = self.num_nodes as f32 * self.num_nodes as f32;
        self.num_edges as f32 / possible
    }
}

/// Build the graph Laplacian directly in sparse COO form.
///
/// With `normalized == false` this returns the combinatorial Laplacian
/// `L = D - A`; with `normalized == true` it returns the symmetric normalized
/// Laplacian `L = I - D^{-1/2} A D^{-1/2}`.
///
/// Node degrees are accumulated treating each column of `edge_index` as an
/// undirected edge (both endpoints gain one degree), and each undirected edge
/// contributes the matching symmetric off-diagonal entries. The result matches
/// the dense [`graph_laplacian`](super::graph_laplacian) for self-loop-free
/// graphs while only storing the non-zero entries.
pub fn sparse_laplacian(edge_index: &Tensor, num_nodes: usize, normalized: bool) -> SparseGraph {
    let edge_data =
        tensor_to_vec2::<f32>(edge_index).expect("edge_index must be a 2 x num_edges tensor");

    let src_row: &[f32] = edge_data.first().map(|r| r.as_slice()).unwrap_or(&[]);
    let dst_row: &[f32] = edge_data.get(1).map(|r| r.as_slice()).unwrap_or(&[]);

    // Undirected degree accumulation (matches the dense reference).
    let mut degrees = vec![0usize; num_nodes];
    for (&src, &dst) in src_row.iter().zip(dst_row.iter()) {
        let s = src as usize;
        let d = dst as usize;
        if s < num_nodes && d < num_nodes {
            degrees[s] += 1;
            degrees[d] += 1;
        }
    }

    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut values: Vec<f32> = Vec::new();

    if normalized {
        // Diagonal: identity.
        for i in 0..num_nodes {
            edges.push((i, i));
            values.push(1.0);
        }
        // Off-diagonal: -1 / sqrt(deg_i * deg_j).
        for (&src, &dst) in src_row.iter().zip(dst_row.iter()) {
            let s = src as usize;
            let d = dst as usize;
            if s >= num_nodes || d >= num_nodes {
                continue;
            }
            let weight = if degrees[s] > 0 && degrees[d] > 0 {
                -1.0 / ((degrees[s] as f32).sqrt() * (degrees[d] as f32).sqrt())
            } else {
                0.0
            };
            edges.push((s, d));
            values.push(weight);
            if s != d {
                edges.push((d, s));
                values.push(weight);
            }
        }
    } else {
        // Diagonal: node degree.
        for (i, &deg) in degrees.iter().enumerate() {
            edges.push((i, i));
            values.push(deg as f32);
        }
        // Off-diagonal: -1 for each adjacency entry.
        for (&src, &dst) in src_row.iter().zip(dst_row.iter()) {
            let s = src as usize;
            let d = dst as usize;
            if s >= num_nodes || d >= num_nodes {
                continue;
            }
            edges.push((s, d));
            values.push(-1.0);
            if s != d {
                edges.push((d, s));
                values.push(-1.0);
            }
        }
    }

    let num_edges = edges.len();
    SparseGraph {
        edge_list: edges,
        node_features: None,
        edge_weights: Some(values),
        num_nodes,
        num_edges,
    }
}

/// Find the representative of `x` with path compression.
fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    // Path compression: point every node on the path directly at the root.
    let mut cursor = x;
    while parent[cursor] != root {
        let next = parent[cursor];
        parent[cursor] = root;
        cursor = next;
    }
    root
}

/// Union the two *roots* `a` and `b` using union-by-rank.
fn uf_union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    if a == b {
        return;
    }
    match rank[a].cmp(&rank[b]) {
        std::cmp::Ordering::Less => parent[a] = b,
        std::cmp::Ordering::Greater => parent[b] = a,
        std::cmp::Ordering::Equal => {
            parent[b] = a;
            rank[a] += 1;
        }
    }
}

/// Coarsen `graph` down to at most `target_nodes` supernodes.
///
/// Coarsening uses greedy edge-contraction: adjacent components are repeatedly
/// merged (via union-find) until the component count reaches the target. For
/// disconnected graphs where edge-contraction alone cannot reach the target,
/// the remaining components are merged deterministically. Each resulting
/// supernode's feature vector is the **mean** of the original node features it
/// absorbed, so finite inputs always yield finite outputs.
///
/// If `graph.num_nodes <= target_nodes` the graph is returned unchanged. A
/// `target_nodes` of `0` is treated as `1` so the result is never empty.
pub fn adaptive_coarsening(graph: &GraphData, target_nodes: usize) -> GraphData {
    let n = graph.num_nodes;
    // Never collapse to an empty graph.
    let target = target_nodes.max(1);

    if n <= target {
        return graph.clone();
    }

    let num_features = graph.x.shape().dims()[1];

    // --- Greedy edge-contraction via union-find -------------------------------
    let adjacency = super::connectivity::build_adjacency_list(&graph.edge_index, n);
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];
    let mut num_components = n;

    loop {
        if num_components <= target {
            break;
        }
        let mut progressed = false;
        for (u, neighbors) in adjacency.iter().enumerate() {
            if num_components <= target {
                break;
            }
            for &v in neighbors {
                if num_components <= target {
                    break;
                }
                let ru = uf_find(&mut parent, u);
                let rv = uf_find(&mut parent, v);
                if ru != rv {
                    uf_union(&mut parent, &mut rank, ru, rv);
                    num_components -= 1;
                    progressed = true;
                }
            }
        }
        if !progressed {
            // Disconnected: no adjacent components remain to contract.
            break;
        }
    }

    // Force-merge leftover components for disconnected graphs so the output
    // always honors `num_nodes <= target`.
    if num_components > target {
        let mut representatives: Vec<usize> = Vec::new();
        for node in 0..n {
            let root = uf_find(&mut parent, node);
            if !representatives.contains(&root) {
                representatives.push(root);
            }
        }
        let anchor = representatives[0];
        let mut idx = representatives.len();
        while num_components > target && idx > 1 {
            idx -= 1;
            let root = uf_find(&mut parent, representatives[idx]);
            let anchor_root = uf_find(&mut parent, anchor);
            if root != anchor_root {
                uf_union(&mut parent, &mut rank, root, anchor_root);
                num_components -= 1;
            }
        }
    }

    // --- Assign contiguous cluster ids (first-appearance order) ---------------
    let mut root_to_cluster: HashMap<usize, usize> = HashMap::new();
    let mut node_to_cluster = vec![0usize; n];
    for (node, slot) in node_to_cluster.iter_mut().enumerate() {
        let root = uf_find(&mut parent, node);
        let next_id = root_to_cluster.len();
        let cluster_id = *root_to_cluster.entry(root).or_insert(next_id);
        *slot = cluster_id;
    }
    let num_coarse = root_to_cluster.len();

    // --- Mean-aggregate features within each cluster --------------------------
    let x_flat = graph
        .x
        .to_vec()
        .expect("node feature tensor must be convertible to a vector");
    let mut coarse_features = vec![0.0f32; num_coarse * num_features];
    let mut cluster_sizes = vec![0usize; num_coarse];
    for (node, &cluster_id) in node_to_cluster.iter().enumerate() {
        cluster_sizes[cluster_id] += 1;
        let src_base = node * num_features;
        let dst_base = cluster_id * num_features;
        for f in 0..num_features {
            coarse_features[dst_base + f] += x_flat[src_base + f];
        }
    }
    for (cluster_id, &size) in cluster_sizes.iter().enumerate() {
        if size > 1 {
            let inv = 1.0 / size as f32;
            let base = cluster_id * num_features;
            for value in &mut coarse_features[base..base + num_features] {
                *value *= inv;
            }
        }
    }

    // --- Build coarsened edges (undirected, de-duplicated, no self-loops) -----
    let edge_data = tensor_to_vec2::<f32>(&graph.edge_index)
        .expect("edge_index must be a 2 x num_edges tensor");
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    if edge_data.len() >= 2 {
        for (&src, &dst) in edge_data[0].iter().zip(edge_data[1].iter()) {
            let s = src as usize;
            let d = dst as usize;
            if s < n && d < n {
                let cs = node_to_cluster[s];
                let cd = node_to_cluster[d];
                if cs != cd {
                    edge_set.insert((cs.min(cd), cs.max(cd)));
                }
            }
        }
    }
    let mut coarse_edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    coarse_edges.sort_unstable();
    let num_coarse_edges = coarse_edges.len();

    let coarse_x = from_vec(
        coarse_features,
        &[num_coarse, num_features],
        DeviceType::Cpu,
    )
    .expect("coarse feature buffer matches [num_coarse, num_features]");

    let coarse_edge_index = if num_coarse_edges > 0 {
        let mut edge_vec = Vec::with_capacity(2 * num_coarse_edges);
        for &(src, _) in &coarse_edges {
            edge_vec.push(src as f32);
        }
        for &(_, dst) in &coarse_edges {
            edge_vec.push(dst as f32);
        }
        from_vec(edge_vec, &[2, num_coarse_edges], DeviceType::Cpu)
            .expect("coarse edge buffer matches [2, num_coarse_edges]")
    } else {
        zeros(&[2, 0]).expect("zeros([2, 0]) is a valid empty edge index")
    };

    GraphData::new(coarse_x, coarse_edge_index)
}

/// Mean-aggregate each node's neighbor features.
///
/// This uses a sparse adjacency list (`O(E)` memory) instead of a dense
/// `O(N^2)` adjacency matrix, and processes destination nodes in batches of
/// `chunk_size` (clamped to at least `1`) to bound the per-step working set and
/// improve cache locality. A node with no neighbors retains its own features so
/// the output is always well defined and finite for finite inputs.
///
/// The returned tensor has shape `[num_nodes, num_features]`.
pub fn chunked_neighbor_aggregation(graph: &GraphData, chunk_size: usize) -> Tensor {
    let n = graph.num_nodes;
    let num_features = if n == 0 { 0 } else { graph.x.shape().dims()[1] };

    let x_flat = graph
        .x
        .to_vec()
        .expect("node feature tensor must be convertible to a vector");
    let adjacency = super::connectivity::build_adjacency_list(&graph.edge_index, n);
    let chunk = chunk_size.max(1);
    let mut out = vec![0.0f32; n * num_features];

    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        for (offset, neighbors) in adjacency[start..end].iter().enumerate() {
            let node = start + offset;
            let out_base = node * num_features;
            if neighbors.is_empty() {
                let in_base = node * num_features;
                out[out_base..out_base + num_features]
                    .copy_from_slice(&x_flat[in_base..in_base + num_features]);
            } else {
                for &neighbor in neighbors {
                    let nb_base = neighbor * num_features;
                    for f in 0..num_features {
                        out[out_base + f] += x_flat[nb_base + f];
                    }
                }
                let inv = 1.0 / neighbors.len() as f32;
                for value in &mut out[out_base..out_base + num_features] {
                    *value *= inv;
                }
            }
        }
        start = end;
    }

    from_vec(out, &[n, num_features], DeviceType::Cpu)
        .expect("aggregated feature buffer matches [num_nodes, num_features]")
}

#[cfg(test)]
mod tests {
    use super::{adaptive_coarsening, chunked_neighbor_aggregation, sparse_laplacian, SparseGraph};
    use crate::utils::{graph_laplacian, tensor_to_vec2};
    use crate::GraphData;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::{from_vec, zeros};
    use torsh_tensor::Tensor;

    /// Undirected 4-cycle (0-1, 1-2, 2-3, 3-0), each edge listed once.
    fn cycle4_edge_index() -> Tensor {
        from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0],
            &[2, 4],
            DeviceType::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn from_dense_extracts_entries_above_threshold() {
        let dense = from_vec(
            vec![
                0.0, 0.5, 0.0, // (0,1) = 0.5
                0.0, 0.0, 0.05, // (1,2) = 0.05 (below threshold)
                -0.8, 0.0, 0.0, // (2,0) = -0.8
            ],
            &[3, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let sparse = SparseGraph::from_dense(&dense, 0.1);
        assert_eq!(sparse.num_nodes, 3);
        assert_eq!(sparse.num_edges, 2);
        assert_eq!(sparse.edge_list, vec![(0, 1), (2, 0)]);

        let weights = sparse.edge_weights.as_ref().unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 0.5).abs() < 1e-6);
        assert!((weights[1] + 0.8).abs() < 1e-6);
    }

    #[test]
    fn empty_sparse_graph_still_has_positive_footprint() {
        // An all-zero adjacency produces no stored edges, but the structure
        // itself still occupies memory.
        let dense = zeros(&[4, 4]).unwrap();
        let sparse = SparseGraph::from_dense(&dense, 0.1);
        assert_eq!(sparse.num_edges, 0);
        assert!(sparse.memory_footprint() > 0);
        assert_eq!(sparse.density(), 0.0);
    }

    #[test]
    fn footprint_increases_with_stored_edges() {
        let empty = SparseGraph::from_dense(&zeros(&[4, 4]).unwrap(), 0.1);
        let dense = from_vec(
            vec![
                0.0, 1.0, 1.0, 1.0, //
                1.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, 0.0, //
            ],
            &[4, 4],
            DeviceType::Cpu,
        )
        .unwrap();
        let full = SparseGraph::from_dense(&dense, 0.1);
        assert_eq!(full.num_edges, 12);
        assert!(full.memory_footprint() > empty.memory_footprint());
        assert!((full.density() - 12.0 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn to_edge_index_round_trips_coordinates() {
        let dense = from_vec(
            vec![
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, //
            ],
            &[3, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let sparse = SparseGraph::from_dense(&dense, 0.5);
        let edge_index = sparse.to_edge_index();
        assert_eq!(edge_index.shape().dims(), &[2, 3]);
        let rows = tensor_to_vec2::<f32>(&edge_index).unwrap();
        assert_eq!(rows[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(rows[1], vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn to_edge_index_empty_is_two_by_zero() {
        let sparse = SparseGraph::from_dense(&zeros(&[3, 3]).unwrap(), 0.5);
        let edge_index = sparse.to_edge_index();
        assert_eq!(edge_index.shape().dims(), &[2, 0]);
    }

    #[test]
    fn sparse_unnormalized_laplacian_matches_dense_reference() {
        let edge_index = cycle4_edge_index();
        let sparse = sparse_laplacian(&edge_index, 4, false);
        let weights = sparse.edge_weights.as_ref().unwrap();
        assert!(weights.iter().all(|w| w.is_finite()));

        // Reconstruct the dense matrix from COO and compare against the
        // independent dense implementation.
        let mut reconstructed = vec![0.0f32; 16];
        for (&(row, col), &value) in sparse.edge_list.iter().zip(weights.iter()) {
            reconstructed[row * 4 + col] += value;
        }
        let reference = graph_laplacian(&edge_index, 4, false).to_vec().unwrap();
        for (got, want) in reconstructed.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-6, "sparse {got} vs dense {want}");
        }
    }

    #[test]
    fn sparse_normalized_laplacian_matches_dense_reference() {
        let edge_index = cycle4_edge_index();
        let sparse = sparse_laplacian(&edge_index, 4, true);
        let weights = sparse.edge_weights.as_ref().unwrap();
        assert!(weights.iter().all(|w| w.is_finite()));

        let mut reconstructed = vec![0.0f32; 16];
        for (&(row, col), &value) in sparse.edge_list.iter().zip(weights.iter()) {
            reconstructed[row * 4 + col] += value;
        }
        let reference = graph_laplacian(&edge_index, 4, true).to_vec().unwrap();
        for (got, want) in reconstructed.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-6, "sparse {got} vs dense {want}");
        }
    }

    #[test]
    fn coarsening_is_noop_when_already_at_or_below_target() {
        let x = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu).unwrap();
        let edge_index = from_vec(vec![0.0, 1.0], &[2, 1], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(x, edge_index);
        let coarsened = adaptive_coarsening(&graph, 5);
        assert_eq!(coarsened.num_nodes, 2);
    }

    #[test]
    fn coarsening_reaches_target_and_averages_features() {
        // Features chosen so the cluster means are exactly predictable.
        let x = from_vec(
            vec![
                0.0, 0.0, // node 0
                2.0, 4.0, // node 1
                10.0, 10.0, // node 2
                4.0, 8.0, // node 3
            ],
            &[4, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let graph = GraphData::new(x, cycle4_edge_index());
        let coarsened = adaptive_coarsening(&graph, 2);

        assert_eq!(coarsened.num_nodes, 2);
        let vals = coarsened.x.to_vec().unwrap();
        assert_eq!(vals.len(), 4);
        assert!(vals.iter().all(|v| v.is_finite()));

        // Greedy contraction merges {0,1,3} and leaves {2}, with cluster 0
        // appearing first. mean(node0,1,3) = (2.0, 4.0); node 2 = (10, 10).
        assert!((vals[0] - 2.0).abs() < 1e-6);
        assert!((vals[1] - 4.0).abs() < 1e-6);
        assert!((vals[2] - 10.0).abs() < 1e-6);
        assert!((vals[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn coarsening_to_one_node_averages_everything() {
        let x = from_vec(vec![1.0, 3.0, 5.0, 7.0], &[4, 1], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(x, cycle4_edge_index());
        let coarsened = adaptive_coarsening(&graph, 1);
        assert_eq!(coarsened.num_nodes, 1);
        let vals = coarsened.x.to_vec().unwrap();
        assert!((vals[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn coarsening_force_merges_disconnected_graph() {
        // Four isolated nodes: edge-contraction cannot reach the target on its
        // own, so the force-merge path must still produce exactly two nodes.
        let x = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4, 1], DeviceType::Cpu).unwrap();
        let edge_index = zeros(&[2, 0]).unwrap();
        let graph = GraphData::new(x, edge_index);
        let coarsened = adaptive_coarsening(&graph, 2);
        assert_eq!(coarsened.num_nodes, 2);
        let vals = coarsened.x.to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn chunked_aggregation_matches_naive_for_all_chunk_sizes() {
        let x = from_vec(
            vec![
                1.0, 1.0, //
                2.0, 2.0, //
                3.0, 3.0, //
                4.0, 4.0, //
            ],
            &[4, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let graph = GraphData::new(x, cycle4_edge_index());

        // adjacency: 0:{1,3} 1:{0,2} 2:{1,3} 3:{0,2}
        let naive = vec![
            3.0, 3.0, // node 0: mean(2,4)
            2.0, 2.0, // node 1: mean(1,3)
            3.0, 3.0, // node 2: mean(2,4)
            2.0, 2.0, // node 3: mean(1,3)
        ];

        for chunk in [1usize, 2, 3, 100] {
            let out = chunked_neighbor_aggregation(&graph, chunk)
                .to_vec()
                .unwrap();
            assert_eq!(out.len(), naive.len());
            for (got, want) in out.iter().zip(naive.iter()) {
                assert!((got - want).abs() < 1e-6, "chunk {chunk}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn chunked_aggregation_isolated_node_keeps_own_features() {
        let x = from_vec(vec![5.0, 9.0], &[2, 1], DeviceType::Cpu).unwrap();
        let edge_index = zeros(&[2, 0]).unwrap();
        let graph = GraphData::new(x, edge_index);
        let out = chunked_neighbor_aggregation(&graph, 1).to_vec().unwrap();
        assert_eq!(out, vec![5.0, 9.0]);
    }
}
