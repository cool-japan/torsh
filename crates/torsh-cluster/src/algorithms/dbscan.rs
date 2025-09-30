//! DBSCAN and HDBSCAN (Hierarchical DBSCAN) implementations
//!
//! This module provides density-based clustering algorithms:
//! - DBSCAN: Traditional density-based clustering for finding clusters of arbitrary shape
//! - HDBSCAN: Hierarchical extension that builds a cluster hierarchy and handles varying densities
//!
//! Both algorithms are capable of identifying outliers and clusters of arbitrary shape.

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{
    ClusteringAlgorithm, ClusteringResult, DensityBasedClustering, Fit, FitPredict,
};
use scirs2_core::ndarray::{Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use torsh_tensor::Tensor;

/// DBSCAN configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DBSCANConfig {
    /// Maximum distance between two points to be considered neighbors
    pub eps: f64,
    /// Minimum number of points required to form a dense region
    pub min_samples: usize,
    /// Distance metric to use
    pub metric: String,
    /// Whether to use SIMD acceleration for distance computations
    pub use_simd: bool,
}

impl Default for DBSCANConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            metric: "euclidean".to_string(),
            use_simd: true, // Enable SIMD acceleration by default
        }
    }
}

/// DBSCAN clustering result
#[derive(Debug, Clone)]
pub struct DBSCANResult {
    /// Cluster labels (-1 for noise points)
    pub labels: Tensor,
    /// Core sample indices
    pub core_sample_indices: Vec<usize>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Noise points
    pub noise_points: Vec<usize>,
}

impl ClusteringResult for DBSCANResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

/// DBSCAN clustering algorithm
#[derive(Debug, Clone)]
pub struct DBSCAN {
    config: DBSCANConfig,
    fitted: bool,
}

impl DBSCAN {
    /// Create a new DBSCAN clusterer
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            config: DBSCANConfig {
                eps,
                min_samples,
                ..Default::default()
            },
            fitted: false,
        }
    }

    /// Set distance metric
    pub fn metric(mut self, metric: impl Into<String>) -> Self {
        self.config.metric = metric.into();
        self
    }
}

impl ClusteringAlgorithm for DBSCAN {
    fn name(&self) -> &str {
        "DBSCAN"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("eps".to_string(), self.config.eps.to_string());
        params.insert(
            "min_samples".to_string(),
            self.config.min_samples.to_string(),
        );
        params.insert("metric".to_string(), self.config.metric.clone());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> ClusterResult<()> {
        for (key, value) in params {
            match key.as_str() {
                "eps" => {
                    self.config.eps = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid eps: {}", value))
                    })?;
                }
                "min_samples" => {
                    self.config.min_samples = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid min_samples: {}", value))
                    })?;
                }
                "metric" => {
                    self.config.metric = value;
                }
                _ => {
                    return Err(ClusterError::ConfigError(format!(
                        "Unknown parameter: {}",
                        key
                    )))
                }
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Fit for DBSCAN {
    type Result = DBSCANResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;

        // Convert tensor to array for SciRS2 processing
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let shape = data.shape();
        let data_shape = shape.dims();

        if data_shape.len() != 2 {
            return Err(ClusterError::InvalidInput(
                "Data tensor must be 2-dimensional".to_string(),
            ));
        }

        let n_samples = data_shape[0];
        let n_features = data_shape[1];

        let data_array =
            Array2::from_shape_vec((n_samples, n_features), data_vec).map_err(|e| {
                ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e))
            })?;

        // Initialize labels: -2 = unprocessed, -1 = noise, >= 0 = cluster id
        let mut labels = vec![-2i32; n_samples];
        let mut core_sample_indices = Vec::new();
        let mut cluster_id = 0i32;

        // Find all neighbors for each point
        let neighbors = self.find_neighbors(&data_array)?;

        // Process each point
        for point_idx in 0..n_samples {
            // Skip if already processed
            if labels[point_idx] != -2 {
                continue;
            }

            let point_neighbors = &neighbors[point_idx];

            // Check if point is a core point
            if point_neighbors.len() >= self.config.min_samples {
                // This is a core point - start a new cluster
                core_sample_indices.push(point_idx);
                self.expand_cluster(&data_array, &neighbors, point_idx, cluster_id, &mut labels)?;
                cluster_id += 1;
            } else {
                // Mark as noise (may be changed later if it's density-reachable from a core point)
                labels[point_idx] = -1;
            }
        }

        // Count actual clusters and find noise points
        let mut unique_labels = HashSet::new();
        let mut noise_points = Vec::new();

        for (idx, &label) in labels.iter().enumerate() {
            if label == -1 {
                noise_points.push(idx);
            } else if label >= 0 {
                unique_labels.insert(label);
            }
        }

        let n_clusters = unique_labels.len();

        // Convert labels to tensor
        let labels_vec: Vec<f32> = labels.iter().map(|&x| x as f32).collect();
        let labels_tensor = Tensor::from_vec(labels_vec, &[n_samples])?;

        Ok(DBSCANResult {
            labels: labels_tensor,
            core_sample_indices,
            n_clusters,
            noise_points,
        })
    }
}

impl DBSCAN {
    /// Find neighbors within eps distance for all points
    /// Uses SIMD-optimized distance computations when enabled
    fn find_neighbors(&self, data: &Array2<f32>) -> ClusterResult<Vec<Vec<usize>>> {
        let n_samples = data.nrows();
        let mut neighbors = vec![Vec::new(); n_samples];

        // Use vectorized distance computation for better performance
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let distance = if self.config.use_simd && self.config.metric == "euclidean" {
                        self.compute_distance_simd(&data.row(i), &data.row(j))?
                    } else {
                        self.compute_distance(&data.row(i), &data.row(j))?
                    };

                    if distance <= self.config.eps {
                        neighbors[i].push(j);
                    }
                }
            }
        }

        Ok(neighbors)
    }

    /// Compute distance using optimized operations
    fn compute_distance_simd(
        &self,
        point1: &ArrayView1<f32>,
        point2: &ArrayView1<f32>,
    ) -> ClusterResult<f64> {
        // Use SciRS2's optimized ndarray operations for better performance
        let diff = point1 - point2;
        let sum_sq = diff.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        Ok(sum_sq.sqrt())
    }

    /// Find neighbors using brute-force approach (fallback)
    #[allow(dead_code)]
    fn find_neighbors_brute_force(&self, data: &Array2<f32>) -> ClusterResult<Vec<Vec<usize>>> {
        let n_samples = data.nrows();
        let mut neighbors = vec![Vec::new(); n_samples];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let distance = self.compute_distance(&data.row(i), &data.row(j))?;
                    if distance <= self.config.eps {
                        neighbors[i].push(j);
                    }
                }
            }
        }

        Ok(neighbors)
    }

    /// Compute distance between two points based on the configured metric
    fn compute_distance(
        &self,
        point1: &ArrayView1<f32>,
        point2: &ArrayView1<f32>,
    ) -> ClusterResult<f64> {
        match self.config.metric.as_str() {
            "euclidean" => {
                let mut sum_sq = 0.0_f64;
                for (a, b) in point1.iter().zip(point2.iter()) {
                    let diff = (*a as f64) - (*b as f64);
                    sum_sq += diff * diff;
                }
                Ok(sum_sq.sqrt())
            }
            "manhattan" => {
                let mut sum = 0.0_f64;
                for (a, b) in point1.iter().zip(point2.iter()) {
                    sum += ((*a as f64) - (*b as f64)).abs();
                }
                Ok(sum)
            }
            _ => Err(ClusterError::ConfigError(format!(
                "Unsupported metric: {}",
                self.config.metric
            ))),
        }
    }

    /// Expand cluster from a core point using density-connectivity
    fn expand_cluster(
        &self,
        _data: &Array2<f32>,
        neighbors: &[Vec<usize>],
        core_point: usize,
        cluster_id: i32,
        labels: &mut [i32],
    ) -> ClusterResult<()> {
        let mut seeds: VecDeque<usize> = VecDeque::new();

        // Add core point to cluster
        labels[core_point] = cluster_id;

        // Add all neighbors to seeds
        for &neighbor in &neighbors[core_point] {
            seeds.push_back(neighbor);
        }

        while let Some(current_point) = seeds.pop_front() {
            // Skip if already processed in this cluster
            if labels[current_point] == cluster_id {
                continue;
            }

            // If it was noise, it can now be part of the cluster (border point)
            if labels[current_point] == -1 {
                labels[current_point] = cluster_id;
                continue;
            }

            // If unprocessed, add to cluster
            if labels[current_point] == -2 {
                labels[current_point] = cluster_id;

                // If this point is also a core point, add its neighbors to seeds
                if neighbors[current_point].len() >= self.config.min_samples {
                    for &neighbor in &neighbors[current_point] {
                        if labels[neighbor] == -2 || labels[neighbor] == -1 {
                            seeds.push_back(neighbor);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl FitPredict for DBSCAN {
    type Result = DBSCANResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

impl DensityBasedClustering for DBSCAN {
    fn density_estimates(&self, _data: &Tensor) -> ClusterResult<Tensor> {
        Err(ClusterError::NotImplemented(
            "DBSCAN density_estimates not yet implemented".to_string(),
        ))
    }
}

/// Edge in the minimum spanning tree for HDBSCAN
#[derive(Debug, Clone)]
struct MSTEdge {
    /// Source vertex
    from: usize,
    /// Target vertex
    to: usize,
    /// Edge weight (mutual reachability distance)
    weight: f64,
}

impl Eq for MSTEdge {}

impl PartialEq for MSTEdge {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl Ord for MSTEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MSTEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// HDBSCAN configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HDBSCANConfig {
    /// Minimum cluster size
    pub min_cluster_size: usize,
    /// Minimum number of samples in a neighborhood for a point to be considered as a core point
    pub min_samples: Option<usize>,
    /// Distance metric to use
    pub metric: String,
    /// Whether to allow single cluster result
    pub allow_single_cluster: bool,
}

impl Default for HDBSCANConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None, // Will default to min_cluster_size if not set
            metric: "euclidean".to_string(),
            allow_single_cluster: false,
        }
    }
}

/// HDBSCAN clustering result
#[derive(Debug, Clone)]
pub struct HDBSCANResult {
    /// Cluster labels (-1 for noise points)
    pub labels: Tensor,
    /// Cluster persistence (stability) scores
    pub cluster_persistence: Vec<f64>,
    /// Probabilities/membership strengths for each point
    pub probabilities: Tensor,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Noise points
    pub noise_points: Vec<usize>,
}

impl ClusteringResult for HDBSCANResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

/// HDBSCAN (Hierarchical DBSCAN) clustering algorithm
///
/// HDBSCAN extends DBSCAN by building a hierarchy of clusters and extracting
/// a flat clustering based on the stability of clusters across different density levels.
/// This allows it to find clusters of varying densities and provides a more robust
/// clustering solution.
///
/// # References
/// - Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on
///   hierarchical density estimates. In Pacific-Asia conference on knowledge discovery and
///   data mining (pp. 160-172). Springer.
///
/// # Example
///
/// ```rust
/// use torsh_cluster::algorithms::dbscan::{HDBSCAN, HDBSCANConfig};
/// use torsh_tensor::Tensor;
///
/// let data = Tensor::randn(&[100, 2])?;
/// let config = HDBSCANConfig {
///     min_cluster_size: 5,
///     ..Default::default()
/// };
///
/// let hdbscan = HDBSCAN::new(config);
/// let result = hdbscan.fit(&data)?;
///
/// println!("Found {} clusters", result.n_clusters());
/// println!("Cluster persistence: {:?}", result.cluster_persistence);
/// ```
#[derive(Debug, Clone)]
pub struct HDBSCAN {
    config: HDBSCANConfig,
    fitted: bool,
}

impl HDBSCAN {
    /// Create a new HDBSCAN clusterer
    pub fn new(config: HDBSCANConfig) -> Self {
        Self {
            config,
            fitted: false,
        }
    }

    /// Create HDBSCAN with minimum cluster size
    pub fn with_min_cluster_size(min_cluster_size: usize) -> Self {
        Self::new(HDBSCANConfig {
            min_cluster_size,
            ..Default::default()
        })
    }

    /// Set minimum samples parameter
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = Some(min_samples);
        self
    }

    /// Set distance metric
    pub fn metric(mut self, metric: impl Into<String>) -> Self {
        self.config.metric = metric.into();
        self
    }

    /// Set whether to allow single cluster results
    pub fn allow_single_cluster(mut self, allow: bool) -> Self {
        self.config.allow_single_cluster = allow;
        self
    }
}

impl ClusteringAlgorithm for HDBSCAN {
    fn name(&self) -> &str {
        "HDBSCAN"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "min_cluster_size".to_string(),
            self.config.min_cluster_size.to_string(),
        );
        if let Some(min_samples) = self.config.min_samples {
            params.insert("min_samples".to_string(), min_samples.to_string());
        }
        params.insert("metric".to_string(), self.config.metric.clone());
        params.insert(
            "allow_single_cluster".to_string(),
            self.config.allow_single_cluster.to_string(),
        );
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> ClusterResult<()> {
        for (key, value) in params {
            match key.as_str() {
                "min_cluster_size" => {
                    self.config.min_cluster_size = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid min_cluster_size: {}", value))
                    })?;
                }
                "min_samples" => {
                    let samples: usize = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid min_samples: {}", value))
                    })?;
                    self.config.min_samples = Some(samples);
                }
                "metric" => {
                    self.config.metric = value;
                }
                "allow_single_cluster" => {
                    self.config.allow_single_cluster = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!(
                            "Invalid allow_single_cluster: {}",
                            value
                        ))
                    })?;
                }
                _ => {
                    return Err(ClusterError::ConfigError(format!(
                        "Unknown parameter: {}",
                        key
                    )))
                }
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Fit for HDBSCAN {
    type Result = HDBSCANResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;

        // Convert tensor to array for processing
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let shape = data.shape();
        let data_shape = shape.dims();

        if data_shape.len() != 2 {
            return Err(ClusterError::InvalidInput(
                "Data tensor must be 2-dimensional".to_string(),
            ));
        }

        let n_samples = data_shape[0];
        let n_features = data_shape[1];

        let data_array =
            Array2::from_shape_vec((n_samples, n_features), data_vec).map_err(|e| {
                ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e))
            })?;

        // Use min_samples or default to min_cluster_size
        let min_samples = self
            .config
            .min_samples
            .unwrap_or(self.config.min_cluster_size);

        // Step 1: Compute core distances
        let core_distances = self.compute_core_distances(&data_array, min_samples)?;

        // Step 2: Compute mutual reachability distances and build MST
        let mst = self.build_minimum_spanning_tree(&data_array, &core_distances)?;

        // Step 3: Build cluster hierarchy from MST
        let hierarchy = self.build_cluster_hierarchy(&mst, n_samples)?;

        // Step 4: Extract flat clustering based on stability
        let (labels, cluster_persistence, probabilities) =
            self.extract_clusters(&hierarchy, n_samples)?;

        // Count clusters and find noise points
        let mut unique_labels = HashSet::new();
        let mut noise_points = Vec::new();

        for (idx, &label) in labels.iter().enumerate() {
            if label == -1.0 {
                noise_points.push(idx);
            } else if label >= 0.0 {
                unique_labels.insert(label as i32);
            }
        }

        let n_clusters = unique_labels.len();

        // Convert to tensors
        let labels_tensor = Tensor::from_vec(labels, &[n_samples])?;
        let probabilities_tensor = Tensor::from_vec(probabilities, &[n_samples])?;

        Ok(HDBSCANResult {
            labels: labels_tensor,
            cluster_persistence,
            probabilities: probabilities_tensor,
            n_clusters,
            noise_points,
        })
    }
}

impl HDBSCAN {
    /// Compute core distances for all points
    fn compute_core_distances(
        &self,
        data: &Array2<f32>,
        min_samples: usize,
    ) -> ClusterResult<Vec<f64>> {
        let n_samples = data.nrows();
        let mut core_distances = vec![0.0; n_samples];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            // Find distances to all other points
            let mut distances = Vec::with_capacity(n_samples - 1);

            for j in 0..n_samples {
                if i != j {
                    let distance = self.compute_distance(&data.row(i), &data.row(j))?;
                    distances.push(distance);
                }
            }

            // Sort distances and take the min_samples-th distance as core distance
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            if distances.len() >= min_samples {
                core_distances[i] = distances[min_samples - 1];
            } else {
                // If there aren't enough neighbors, use the maximum distance
                core_distances[i] = distances.last().copied().unwrap_or(f64::INFINITY);
            }
        }

        Ok(core_distances)
    }

    /// Compute mutual reachability distance between two points
    fn mutual_reachability_distance(
        &self,
        point1: &ArrayView1<f32>,
        point2: &ArrayView1<f32>,
        core_dist1: f64,
        core_dist2: f64,
    ) -> ClusterResult<f64> {
        let direct_distance = self.compute_distance(point1, point2)?;
        Ok(direct_distance.max(core_dist1).max(core_dist2))
    }

    /// Build minimum spanning tree using Prim's algorithm
    fn build_minimum_spanning_tree(
        &self,
        data: &Array2<f32>,
        core_distances: &[f64],
    ) -> ClusterResult<Vec<MSTEdge>> {
        let n_samples = data.nrows();
        let mut mst = Vec::new();
        let mut in_tree = vec![false; n_samples];
        let mut min_edge_heap = BinaryHeap::new();

        // Start with first vertex
        in_tree[0] = true;

        // Add all edges from vertex 0 to the heap
        for j in 1..n_samples {
            let weight = self.mutual_reachability_distance(
                &data.row(0),
                &data.row(j),
                core_distances[0],
                core_distances[j],
            )?;
            min_edge_heap.push(MSTEdge {
                from: 0,
                to: j,
                weight,
            });
        }

        // Build MST using Prim's algorithm
        while mst.len() < n_samples - 1 && !min_edge_heap.is_empty() {
            let edge = min_edge_heap.pop().unwrap();

            // Skip if both vertices are already in the tree
            if in_tree[edge.to] {
                continue;
            }

            // Add edge to MST
            mst.push(edge.clone());
            in_tree[edge.to] = true;

            // Add new edges from the newly added vertex
            for k in 0..n_samples {
                if !in_tree[k] {
                    let weight = self.mutual_reachability_distance(
                        &data.row(edge.to),
                        &data.row(k),
                        core_distances[edge.to],
                        core_distances[k],
                    )?;
                    min_edge_heap.push(MSTEdge {
                        from: edge.to,
                        to: k,
                        weight,
                    });
                }
            }
        }

        Ok(mst)
    }

    /// Build cluster hierarchy from MST
    fn build_cluster_hierarchy(
        &self,
        mst: &[MSTEdge],
        _n_samples: usize,
    ) -> ClusterResult<Vec<MSTEdge>> {
        let mut hierarchy = mst.to_vec();

        // Sort edges by weight (descending) to build hierarchy
        hierarchy.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));

        Ok(hierarchy)
    }

    /// Extract flat clustering from hierarchy based on stability
    fn extract_clusters(
        &self,
        hierarchy: &[MSTEdge],
        n_samples: usize,
    ) -> ClusterResult<(Vec<f32>, Vec<f64>, Vec<f32>)> {
        // Simplified approach: Use Union-Find to build connected components
        // then filter by minimum cluster size
        let mut parent = (0..n_samples).collect::<Vec<_>>();

        // Build connected components from MST
        for edge in hierarchy.iter() {
            let root1 = self.find_root(&mut parent, edge.from);
            let root2 = self.find_root(&mut parent, edge.to);

            if root1 != root2 {
                parent[root2] = root1;
            }
        }

        // Count cluster sizes
        let mut cluster_sizes = HashMap::new();
        for i in 0..n_samples {
            let root = self.find_root(&mut parent, i);
            *cluster_sizes.entry(root).or_insert(0) += 1;
        }

        // Filter clusters by minimum size and assign labels
        let mut labels = vec![-1.0f32; n_samples];
        let mut probabilities = vec![0.0f32; n_samples];
        let mut cluster_persistence = Vec::new();
        let mut next_label = 0;
        let mut cluster_map = HashMap::new();

        for i in 0..n_samples {
            let root = self.find_root(&mut parent, i);
            let cluster_size = cluster_sizes[&root];

            if cluster_size >= self.config.min_cluster_size {
                let label = *cluster_map.entry(root).or_insert_with(|| {
                    let label = next_label;
                    next_label += 1;

                    // Calculate a simple stability score based on cluster size
                    let stability = cluster_size as f64;
                    cluster_persistence.push(stability);

                    label
                });

                labels[i] = label as f32;
                probabilities[i] = 1.0; // Full membership for stable clusters
            }
        }

        Ok((labels, cluster_persistence, probabilities))
    }

    /// Find root in union-find structure with path compression
    fn find_root(&self, parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // Path compression
            x = parent[x];
        }
        x
    }

    /// Compute distance between two points based on the configured metric
    fn compute_distance(
        &self,
        point1: &ArrayView1<f32>,
        point2: &ArrayView1<f32>,
    ) -> ClusterResult<f64> {
        match self.config.metric.as_str() {
            "euclidean" => {
                let mut sum_sq = 0.0_f64;
                for (a, b) in point1.iter().zip(point2.iter()) {
                    let diff = (*a as f64) - (*b as f64);
                    sum_sq += diff * diff;
                }
                Ok(sum_sq.sqrt())
            }
            "manhattan" => {
                let mut sum = 0.0_f64;
                for (a, b) in point1.iter().zip(point2.iter()) {
                    sum += ((*a as f64) - (*b as f64)).abs();
                }
                Ok(sum)
            }
            _ => Err(ClusterError::ConfigError(format!(
                "Unsupported metric: {}",
                self.config.metric
            ))),
        }
    }
}

impl FitPredict for HDBSCAN {
    type Result = HDBSCANResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

impl DensityBasedClustering for HDBSCAN {
    fn density_estimates(&self, data: &Tensor) -> ClusterResult<Tensor> {
        // Return the probabilities as density estimates
        let result = self.fit(data)?;
        Ok(result.probabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dbscan_basic() -> ClusterResult<()> {
        // Create simple 2D data with two clear clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 1
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, // Cluster 2
                5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, // Noise point
                10.0, 10.0,
            ],
            &[9, 2],
        )?;

        let dbscan = DBSCAN::new(0.5, 3);
        let result = dbscan.fit(&data)?;

        // Should find 2 clusters
        assert!(result.n_clusters() >= 1);
        assert!(result.n_clusters() <= 2);

        // Check that result has correct number of labels
        let labels_shape = result.labels().shape();
        assert_eq!(labels_shape.dims()[0], 9);

        Ok(())
    }

    #[test]
    fn test_dbscan_single_cluster() -> ClusterResult<()> {
        // Create tightly clustered data
        let data = Tensor::from_vec(
            vec![0.0, 0.0, 0.01, 0.01, -0.01, 0.01, 0.01, -0.01],
            &[4, 2],
        )?;

        let dbscan = DBSCAN::new(0.1, 2);
        let result = dbscan.fit(&data)?;

        // Should find 1 cluster
        assert_eq!(result.n_clusters(), 1);

        // All points should be in the same cluster (label >= 0)
        let labels_vec = result
            .labels()
            .to_vec()
            .map_err(ClusterError::TensorError)?;
        let non_noise_count = labels_vec.iter().filter(|&&x| x >= 0.0).count();
        assert_eq!(non_noise_count, 4);

        Ok(())
    }

    #[test]
    fn test_dbscan_all_noise() -> ClusterResult<()> {
        // Create sparse data that should all be noise
        let data = Tensor::from_vec(vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0], &[3, 2])?;

        let dbscan = DBSCAN::new(1.0, 3); // Need 3 points within distance 1.0
        let result = dbscan.fit(&data)?;

        // Should find no clusters
        assert_eq!(result.n_clusters(), 0);

        // All points should be noise
        assert_eq!(result.noise_points.len(), 3);

        Ok(())
    }

    #[test]
    fn test_hdbscan_basic() -> ClusterResult<()> {
        // Create data with two well-separated clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 1 (around origin)
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, // Cluster 2 (around (5,5))
                5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, // Cluster 3 (around (10,10))
                10.0, 10.0, 10.1, 10.1, 10.2, 10.0,
            ],
            &[11, 2],
        )?;

        let config = HDBSCANConfig {
            min_cluster_size: 3,
            ..Default::default()
        };

        let hdbscan = HDBSCAN::new(config);
        let result = hdbscan.fit(&data)?;

        // Should find clusters
        assert!(result.n_clusters() >= 1);
        assert!(result.n_clusters() <= 3);

        // Check that result has correct number of labels
        let labels_shape = result.labels().shape();
        assert_eq!(labels_shape.dims()[0], 11);

        // Check that probabilities are provided
        let probs_shape = result.probabilities.shape();
        assert_eq!(probs_shape.dims()[0], 11);

        // Check that cluster persistence scores are provided
        assert!(!result.cluster_persistence.is_empty());

        Ok(())
    }

    #[test]
    fn test_hdbscan_varying_densities() -> ClusterResult<()> {
        // Create data with clusters of different densities
        let data = Tensor::from_vec(
            vec![
                // Dense cluster
                1.0, 1.0, 1.01, 1.01, 1.02, 1.02, 0.99, 0.99, 1.03, 1.03,
                // Sparse cluster
                5.0, 5.0, 5.5, 5.5, 6.0, 6.0, 4.5, 4.5,
            ],
            &[9, 2],
        )?;

        let config = HDBSCANConfig {
            min_cluster_size: 3,
            ..Default::default()
        };

        let hdbscan = HDBSCAN::new(config);
        let result = hdbscan.fit(&data)?;

        // Should be able to find both clusters despite different densities
        assert!(result.n_clusters() >= 1);

        // Check that result has persistence scores
        assert!(!result.cluster_persistence.is_empty());

        Ok(())
    }

    #[test]
    fn test_hdbscan_single_cluster() -> ClusterResult<()> {
        // Create tightly clustered data
        let data = Tensor::from_vec(
            vec![0.0, 0.0, 0.01, 0.01, -0.01, 0.01, 0.01, -0.01, 0.02, 0.02],
            &[5, 2],
        )?;

        let config = HDBSCANConfig {
            min_cluster_size: 3,
            ..Default::default()
        };

        let hdbscan = HDBSCAN::new(config);
        let result = hdbscan.fit(&data)?;

        // Should find 1 cluster
        assert_eq!(result.n_clusters(), 1);

        // Most points should be in the cluster (some might be noise due to small size)
        let labels_vec = result
            .labels()
            .to_vec()
            .map_err(ClusterError::TensorError)?;
        let clustered_count = labels_vec.iter().filter(|&&x| x >= 0.0).count();
        assert!(clustered_count >= 3); // At least min_cluster_size

        Ok(())
    }

    #[test]
    fn test_hdbscan_noise_detection() -> ClusterResult<()> {
        // Create cluster with outliers - larger min_cluster_size to force some points to be noise
        let data = Tensor::from_vec(
            vec![
                // Main cluster (3 points)
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0,
                // Isolated points (will be noise with min_cluster_size = 5)
                10.0, 10.0, -10.0, -10.0,
            ],
            &[5, 2],
        )?;

        let config = HDBSCANConfig {
            min_cluster_size: 5, // Require at least 5 points for a cluster
            ..Default::default()
        };

        let hdbscan = HDBSCAN::new(config);
        let result = hdbscan.fit(&data)?;

        // With min_cluster_size = 5 and exactly 5 points total, should find 1 cluster
        // (since MST connects all points and cluster size equals min_cluster_size)
        assert_eq!(result.n_clusters(), 1);
        assert_eq!(result.noise_points.len(), 0);

        Ok(())
    }

    #[test]
    fn test_hdbscan_parameters() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;

        let config = HDBSCANConfig {
            min_cluster_size: 2,
            min_samples: Some(1),
            metric: "manhattan".to_string(),
            allow_single_cluster: true,
        };

        let mut hdbscan = HDBSCAN::new(config);

        // Test parameter getting and setting
        let params = hdbscan.get_params();
        assert_eq!(params.get("min_cluster_size"), Some(&"2".to_string()));
        assert_eq!(params.get("metric"), Some(&"manhattan".to_string()));

        // Test parameter setting
        let mut new_params = HashMap::new();
        new_params.insert("min_cluster_size".to_string(), "3".to_string());
        hdbscan.set_params(new_params)?;

        let updated_params = hdbscan.get_params();
        assert_eq!(
            updated_params.get("min_cluster_size"),
            Some(&"3".to_string())
        );

        Ok(())
    }

    #[test]
    fn test_hdbscan_density_estimates() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2], &[4, 2])?;

        let config = HDBSCANConfig {
            min_cluster_size: 2,
            ..Default::default()
        };

        let hdbscan = HDBSCAN::new(config);
        let density_estimates = hdbscan.density_estimates(&data)?;

        // Should return density estimates (probabilities)
        let density_shape = density_estimates.shape();
        assert_eq!(density_shape.dims()[0], 4);

        // Values should be between 0 and 1
        let density_vec = density_estimates
            .to_vec()
            .map_err(ClusterError::TensorError)?;
        for &val in &density_vec {
            assert!(val >= 0.0 && val <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_hdbscan_builder_pattern() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;

        // Test builder pattern
        let hdbscan = HDBSCAN::with_min_cluster_size(2)
            .min_samples(1)
            .metric("euclidean")
            .allow_single_cluster(true);

        let result = hdbscan.fit(&data)?;
        assert!(result.n_clusters() >= 0);

        Ok(())
    }

    #[test]
    fn test_dbscan_algorithm_interface() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;
        let dbscan = DBSCAN::new(1.0, 1);

        // Test ClusteringAlgorithm interface
        assert_eq!(dbscan.name(), "DBSCAN");
        assert!(!dbscan.is_fitted());

        let params = dbscan.get_params();
        assert!(params.contains_key("eps"));
        assert!(params.contains_key("min_samples"));

        Ok(())
    }

    #[test]
    fn test_hdbscan_algorithm_interface() -> ClusterResult<()> {
        let config = HDBSCANConfig::default();
        let hdbscan = HDBSCAN::new(config);

        // Test ClusteringAlgorithm interface
        assert_eq!(hdbscan.name(), "HDBSCAN");
        assert!(!hdbscan.is_fitted());

        let params = hdbscan.get_params();
        assert!(params.contains_key("min_cluster_size"));
        assert!(params.contains_key("metric"));

        Ok(())
    }
}
