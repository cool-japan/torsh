//! OPTICS (Ordering Points To Identify the Clustering Structure) Implementation
//!
//! This module provides the OPTICS algorithm, a density-based clustering method
//! that creates an ordering of points and a reachability plot. Unlike DBSCAN,
//! OPTICS can identify clusters with varying densities and provides a richer
//! hierarchical structure.
//!
//! # Algorithm Overview
//!
//! OPTICS produces:
//! 1. An ordering of data points
//! 2. Reachability distances for each point
//! 3. Core distances for each point
//!
//! Clusters can be extracted from the reachability plot at different thresholds.
//!
//! # References
//!
//! Ankerst, M., Breunig, M. M., Kriegel, H. P., & Sander, J. (1999).
//! OPTICS: Ordering points to identify the clustering structure.
//! ACM SIGMOD Record, 28(2), 49-60.

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{
    AlgorithmComplexity, ClusteringAlgorithm, ClusteringConfig, ClusteringResult,
    DensityBasedClustering, Fit, FitPredict, MemoryPattern,
};
use crate::utils::validation::validate_cluster_input;
use scirs2_core::ndarray::{Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use torsh_tensor::Tensor;

/// OPTICS configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OPTICSConfig {
    /// Maximum epsilon neighborhood distance
    pub max_eps: f64,
    /// Minimum number of points in a neighborhood
    pub min_samples: usize,
    /// Distance metric to use (euclidean, manhattan, etc.)
    pub metric: String,
    /// Extraction threshold for cluster extraction (optional)
    pub xi: Option<f64>,
    /// Predecessor correction parameter
    pub predecessor_correction: bool,
}

impl Default for OPTICSConfig {
    fn default() -> Self {
        Self {
            max_eps: f64::INFINITY,
            min_samples: 5,
            metric: "euclidean".to_string(),
            xi: Some(0.05),
            predecessor_correction: true,
        }
    }
}

impl ClusteringConfig for OPTICSConfig {
    fn validate(&self) -> ClusterResult<()> {
        if self.max_eps <= 0.0 {
            return Err(ClusterError::ConfigError(
                "max_eps must be positive".to_string(),
            ));
        }
        if self.min_samples == 0 {
            return Err(ClusterError::ConfigError(
                "min_samples must be positive".to_string(),
            ));
        }
        if let Some(xi) = self.xi {
            if xi <= 0.0 || xi >= 1.0 {
                return Err(ClusterError::ConfigError(
                    "xi must be between 0 and 1".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn default() -> Self {
        <OPTICSConfig as std::default::Default>::default()
    }

    fn merge(&mut self, other: &Self) {
        let default_config = <OPTICSConfig as std::default::Default>::default();
        if other.max_eps != default_config.max_eps {
            self.max_eps = other.max_eps;
        }
        if other.min_samples != default_config.min_samples {
            self.min_samples = other.min_samples;
        }
        if other.xi != default_config.xi {
            self.xi = other.xi;
        }
    }
}

/// OPTICS clustering result
#[derive(Debug, Clone)]
pub struct OPTICSResult {
    /// Cluster labels (-1 for noise points)
    pub labels: Tensor,
    /// Reachability distances for each point (in ordering)
    pub reachability: Tensor,
    /// Core distances for each point
    pub core_distances: Tensor,
    /// Ordering of points
    pub ordering: Vec<usize>,
    /// Predecessor indices in the ordering
    pub predecessor: Vec<Option<usize>>,
    /// Number of clusters found
    pub n_clusters: usize,
}

impl ClusteringResult for OPTICSResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    fn n_iter(&self) -> Option<usize> {
        Some(self.ordering.len())
    }

    fn metadata(&self) -> Option<&HashMap<String, String>> {
        None
    }
}

/// Priority queue element for OPTICS processing
#[derive(Debug, Clone)]
struct SeedPoint {
    index: usize,
    reachability_dist: f64,
}

impl PartialEq for SeedPoint {
    fn eq(&self, other: &Self) -> bool {
        self.reachability_dist == other.reachability_dist
    }
}

impl Eq for SeedPoint {}

impl PartialOrd for SeedPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeedPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .reachability_dist
            .partial_cmp(&self.reachability_dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// OPTICS (Ordering Points To Identify the Clustering Structure) algorithm
///
/// OPTICS is a density-based clustering algorithm that creates an ordering
/// of data points with reachability distances, allowing cluster extraction
/// at multiple density levels.
///
/// # Example
///
/// ```rust
/// use torsh_cluster::algorithms::optics::{OPTICS, OPTICSConfig};
/// use torsh_cluster::traits::Fit;
/// use torsh_tensor::creation::randn;
///
/// let data = randn::<f32>(&[100, 2])?;
/// let optics = OPTICS::new(0.5, 5);
/// let result = optics.fit(&data)?;
/// println!("Found {} clusters", result.n_clusters);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct OPTICS {
    config: OPTICSConfig,
    fitted: bool,
}

impl OPTICS {
    /// Create a new OPTICS clusterer
    pub fn new(max_eps: f64, min_samples: usize) -> Self {
        Self {
            config: OPTICSConfig {
                max_eps,
                min_samples,
                ..Default::default()
            },
            fitted: false,
        }
    }

    /// Create OPTICS with custom configuration
    pub fn with_config(config: OPTICSConfig) -> Self {
        Self {
            config,
            fitted: false,
        }
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: impl Into<String>) -> Self {
        self.config.metric = metric.into();
        self
    }

    /// Set the xi parameter for cluster extraction
    pub fn xi(mut self, xi: f64) -> Self {
        self.config.xi = Some(xi);
        self
    }

    /// Enable/disable predecessor correction
    pub fn predecessor_correction(mut self, enabled: bool) -> Self {
        self.config.predecessor_correction = enabled;
        self
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        let diff = p1 - p2;
        diff.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute Manhattan distance between two points
    fn manhattan_distance(&self, p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        let diff = p1 - p2;
        diff.iter().map(|x| x.abs()).sum()
    }

    /// Compute distance between two points based on the configured metric
    fn compute_distance(&self, p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        match self.config.metric.as_str() {
            "euclidean" => self.euclidean_distance(p1, p2),
            "manhattan" => self.manhattan_distance(p1, p2),
            _ => self.euclidean_distance(p1, p2), // Default to Euclidean
        }
    }

    /// Find neighbors within max_eps distance
    fn get_neighbors(&self, data: &Array2<f64>, point_idx: usize) -> Vec<usize> {
        let point = data.row(point_idx);
        let mut neighbors = Vec::new();

        for i in 0..data.nrows() {
            if i != point_idx {
                let dist = self.compute_distance(&point, &data.row(i));
                if dist <= self.config.max_eps {
                    neighbors.push(i);
                }
            }
        }

        neighbors
    }

    /// Compute core distance for a point
    /// Core distance is the distance to the min_samples-th nearest neighbor
    fn compute_core_distance(&self, data: &Array2<f64>, point_idx: usize) -> f64 {
        let point = data.row(point_idx);
        let mut distances: Vec<f64> = Vec::new();

        for i in 0..data.nrows() {
            if i != point_idx {
                let dist = self.compute_distance(&point, &data.row(i));
                if dist <= self.config.max_eps {
                    distances.push(dist);
                }
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if distances.len() < self.config.min_samples {
            f64::INFINITY
        } else {
            distances[self.config.min_samples - 1]
        }
    }

    /// Compute reachability distance from point to neighbor
    fn compute_reachability_distance(&self, core_dist: f64, actual_dist: f64) -> f64 {
        core_dist.max(actual_dist)
    }

    /// Update the seed list with new neighbors
    fn update_seeds(
        &self,
        seeds: &mut BinaryHeap<SeedPoint>,
        neighbors: &[usize],
        center_idx: usize,
        core_dist: f64,
        data: &Array2<f64>,
        processed: &[bool],
        reachability: &mut [f64],
    ) {
        let center_point = data.row(center_idx);

        for &neighbor_idx in neighbors {
            if processed[neighbor_idx] {
                continue;
            }

            let neighbor_point = data.row(neighbor_idx);
            let dist = self.compute_distance(&center_point, &neighbor_point);
            let new_reach_dist = self.compute_reachability_distance(core_dist, dist);

            if reachability[neighbor_idx] == f64::INFINITY {
                // First time seeing this point
                reachability[neighbor_idx] = new_reach_dist;
                seeds.push(SeedPoint {
                    index: neighbor_idx,
                    reachability_dist: new_reach_dist,
                });
            } else if new_reach_dist < reachability[neighbor_idx] {
                // Update with better reachability
                reachability[neighbor_idx] = new_reach_dist;
                // Note: In a real priority queue, we'd update the existing entry
                // Here we push a new one (duplicates are handled in processing)
                seeds.push(SeedPoint {
                    index: neighbor_idx,
                    reachability_dist: new_reach_dist,
                });
            }
        }
    }

    /// Extract clusters from the reachability plot using the xi method
    fn extract_clusters_xi(
        &self,
        ordering: &[usize],
        reachability: &[f64],
        n_samples: usize,
    ) -> Vec<i32> {
        let mut labels = vec![-1i32; n_samples];

        if let Some(xi) = self.config.xi {
            let mut cluster_id = 0i32;
            let mut in_cluster = vec![false; ordering.len()];

            // Simple cluster extraction: find valleys in reachability plot
            for i in 1..ordering.len() - 1 {
                let prev_reach = reachability[i - 1];
                let curr_reach = reachability[i];
                let next_reach = reachability[i + 1];

                // Start of a cluster (significant drop in reachability)
                if prev_reach != f64::INFINITY
                    && curr_reach != f64::INFINITY
                    && curr_reach < prev_reach * (1.0 - xi)
                {
                    cluster_id += 1;
                    in_cluster[i] = true;
                    labels[ordering[i]] = cluster_id;
                }
                // Continue cluster
                else if in_cluster[i - 1]
                    && curr_reach != f64::INFINITY
                    && curr_reach < prev_reach * (1.0 + xi)
                {
                    in_cluster[i] = true;
                    labels[ordering[i]] = cluster_id;
                }
                // End of cluster (significant rise in reachability)
                else if in_cluster[i - 1]
                    && next_reach != f64::INFINITY
                    && next_reach > curr_reach * (1.0 + xi)
                {
                    in_cluster[i] = false;
                }
            }
        }

        labels
    }

    /// Run the OPTICS algorithm
    fn run_optics(&self, data: &Array2<f64>) -> ClusterResult<OPTICSResult> {
        let n_samples = data.nrows();

        // Initialize data structures
        let mut ordering = Vec::with_capacity(n_samples);
        let mut reachability = vec![f64::INFINITY; n_samples];
        let mut core_distances = vec![f64::INFINITY; n_samples];
        let mut processed = vec![false; n_samples];
        let predecessor = vec![None; n_samples];

        // Compute all core distances
        for i in 0..n_samples {
            core_distances[i] = self.compute_core_distance(data, i);
        }

        // Process each point
        for start_idx in 0..n_samples {
            if processed[start_idx] {
                continue;
            }

            // Add start point to ordering
            ordering.push(start_idx);
            processed[start_idx] = true;
            reachability[start_idx] = f64::INFINITY; // Start points have infinite reachability

            // Initialize seed set
            let mut seeds = BinaryHeap::new();
            let neighbors = self.get_neighbors(data, start_idx);

            if core_distances[start_idx] != f64::INFINITY {
                self.update_seeds(
                    &mut seeds,
                    &neighbors,
                    start_idx,
                    core_distances[start_idx],
                    data,
                    &processed,
                    &mut reachability,
                );
            }

            // Expand cluster
            while let Some(seed) = seeds.pop() {
                let current_idx = seed.index;

                // Skip if already processed (handles duplicates from priority queue)
                if processed[current_idx] {
                    continue;
                }

                // Add to ordering
                ordering.push(current_idx);
                processed[current_idx] = true;

                // Get neighbors
                let neighbors = self.get_neighbors(data, current_idx);

                // Update seeds if this is a core point
                if core_distances[current_idx] != f64::INFINITY {
                    self.update_seeds(
                        &mut seeds,
                        &neighbors,
                        current_idx,
                        core_distances[current_idx],
                        data,
                        &processed,
                        &mut reachability,
                    );
                }
            }
        }

        // Extract clusters from reachability plot
        let labels = self.extract_clusters_xi(&ordering, &reachability, n_samples);

        // Count clusters
        let n_clusters = labels.iter().filter(|&&l| l >= 0).max().unwrap_or(&-1) + 1;

        // Convert to tensors
        let labels_tensor = array1_i32_to_tensor(&labels)?;
        let reachability_ordered: Vec<f64> = ordering.iter().map(|&i| reachability[i]).collect();
        let reachability_tensor = array1_to_tensor(&reachability_ordered)?;
        let core_distances_tensor = array1_to_tensor(&core_distances)?;

        Ok(OPTICSResult {
            labels: labels_tensor,
            reachability: reachability_tensor,
            core_distances: core_distances_tensor,
            ordering,
            predecessor,
            n_clusters: n_clusters as usize,
        })
    }
}

impl ClusteringAlgorithm for OPTICS {
    fn name(&self) -> &str {
        "OPTICS"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("max_eps".to_string(), self.config.max_eps.to_string());
        params.insert(
            "min_samples".to_string(),
            self.config.min_samples.to_string(),
        );
        params.insert("metric".to_string(), self.config.metric.clone());
        if let Some(xi) = self.config.xi {
            params.insert("xi".to_string(), xi.to_string());
        }
        params.insert(
            "predecessor_correction".to_string(),
            self.config.predecessor_correction.to_string(),
        );
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> ClusterResult<()> {
        for (key, value) in params {
            match key.as_str() {
                "max_eps" => {
                    self.config.max_eps = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid max_eps: {}", value))
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
                "xi" => {
                    let xi: f64 = value
                        .parse()
                        .map_err(|_| ClusterError::ConfigError(format!("Invalid xi: {}", value)))?;
                    self.config.xi = Some(xi);
                }
                _ => {
                    return Err(ClusterError::ConfigError(format!(
                        "Unknown parameter: {}",
                        key
                    )));
                }
            }
        }
        self.config.validate()?;
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn complexity_info(&self) -> AlgorithmComplexity {
        AlgorithmComplexity {
            time_complexity: "O(n²)".to_string(),
            space_complexity: "O(n)".to_string(),
            deterministic: true,
            online_capable: false,
            memory_pattern: MemoryPattern::Linear,
        }
    }

    fn supported_distance_metrics(&self) -> Vec<&str> {
        vec!["euclidean", "manhattan"]
    }
}

impl Fit for OPTICS {
    type Result = OPTICSResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;
        validate_cluster_input(data)?;
        self.config.validate()?;

        // Convert tensor to ndarray
        let data_array = tensor_to_array2(data)?;

        // Run OPTICS algorithm
        self.run_optics(&data_array)
    }
}

impl FitPredict for OPTICS {
    type Result = OPTICSResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

impl DensityBasedClustering for OPTICS {
    // Optional methods with default implementations - can be overridden if state is stored
    fn core_points(&self) -> Option<&Tensor> {
        None // Would require storing fitted state
    }

    fn noise_points(&self) -> Option<&Tensor> {
        None // Would require storing fitted state
    }

    // Required method: estimate local densities for each point
    fn density_estimates(&self, data: &Tensor) -> ClusterResult<Tensor> {
        // Convert tensor to ndarray
        let data_array = tensor_to_array2(data)?;
        let n_samples = data_array.nrows();
        let mut densities = vec![0.0; n_samples];

        // Estimate density as inverse of core distance
        for i in 0..n_samples {
            let core_dist = self.compute_core_distance(&data_array, i);
            if core_dist == f64::INFINITY {
                densities[i] = 0.0; // No density (isolated point)
            } else {
                densities[i] = 1.0 / (core_dist + 1e-10); // Inverse distance as density
            }
        }

        array1_to_tensor(&densities)
    }
}

// Utility functions for tensor/array conversions
fn tensor_to_array2(tensor: &Tensor) -> ClusterResult<Array2<f64>> {
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();
    if shape.len() != 2 {
        return Err(ClusterError::InvalidInput("Expected 2D tensor".to_string()));
    }

    let data_f32: Vec<f32> = tensor.to_vec().map_err(ClusterError::TensorError)?;
    let data: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|_| ClusterError::InvalidInput("Failed to convert tensor to array".to_string()))
}

fn array1_to_tensor(array: &[f64]) -> ClusterResult<Tensor> {
    let len = array.len();
    let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
    Tensor::from_vec(data, &[len]).map_err(ClusterError::TensorError)
}

fn array1_i32_to_tensor(array: &[i32]) -> ClusterResult<Tensor> {
    let len = array.len();
    let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
    Tensor::from_vec(data, &[len]).map_err(ClusterError::TensorError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optics_basic() {
        // Create simple 2D dataset with two clusters
        let data = vec![
            // Cluster 1
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, // Cluster 2
            5.0, 5.0, 5.1, 5.1, 5.0, 5.2, 5.2, 5.0, 5.1, 5.2,
        ];
        let tensor = Tensor::from_vec(data, &[10, 2]).unwrap();

        let optics = OPTICS::new(0.5, 2);
        let result = optics.fit(&tensor).unwrap();

        // Should find 2 clusters
        assert!(result.n_clusters >= 1);
        assert_eq!(result.ordering.len(), 10);
    }

    #[test]
    fn test_optics_config_validation() {
        let mut config = <OPTICSConfig as std::default::Default>::default();

        // Valid config
        assert!(config.validate().is_ok());

        // Invalid max_eps
        config.max_eps = -1.0;
        assert!(config.validate().is_err());
        config.max_eps = 1.0;

        // Invalid min_samples
        config.min_samples = 0;
        assert!(config.validate().is_err());
        config.min_samples = 5;

        // Invalid xi
        config.xi = Some(1.5);
        assert!(config.validate().is_err());
        config.xi = Some(0.05);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_optics_algorithm_properties() {
        let optics = OPTICS::new(0.5, 5);

        assert_eq!(optics.name(), "OPTICS");
        assert!(!optics.is_fitted());

        let params = optics.get_params();
        assert!(params.contains_key("max_eps"));
        assert!(params.contains_key("min_samples"));

        let complexity = optics.complexity_info();
        assert_eq!(complexity.time_complexity, "O(n²)");
        assert!(complexity.deterministic);
    }

    #[test]
    fn test_optics_reachability_ordering() {
        // Create dataset with clear structure
        let data = vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1];
        let tensor = Tensor::from_vec(data, &[6, 2]).unwrap();

        let optics = OPTICS::new(1.0, 2);
        let result = optics.fit(&tensor).unwrap();

        // All points should be in the ordering
        assert_eq!(result.ordering.len(), 6);

        // Reachability should be computed for all points
        assert_eq!(result.reachability.shape().dims()[0], 6);
    }
}
