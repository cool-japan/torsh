//! K-Means clustering algorithm implementation
//!
//! This module provides a high-performance K-Means implementation built on SciRS2,
//! offering multiple initialization strategies and convergence criteria.
//!
//! # Mathematical Formulation
//!
//! K-Means clustering minimizes within-cluster sum of squares (WCSS):
//!
//! ```text
//! argmin_S ∑_{i=1}^{k} ∑_{x ∈ S_i} ||x - μ_i||²
//! ```
//!
//! ## Algorithms
//!
//! - **Lloyd**: Standard algorithm, O(nkdi) complexity
//! - **Elkan**: Triangle inequality optimization for large k
//! - **Mini-Batch**: Stochastic updates, O(bkd) where b << n
//!
//! ## Optimizations
//!
//! - Automatic parallel processing for n ≥ 1000
//! - SIMD-accelerated distance computations
//! - K-means++ initialization

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{ClusteringAlgorithm, ClusteringResult, Fit, FitPredict};
use crate::utils::parallel;
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{seeded_rng, CoreRandom};
// Using SciRS2 re-exported StdRng to avoid direct rand dependency (SciRS2 POLICY)
use scirs2_core::random::rngs::StdRng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_tensor::{creation::zeros, Tensor};

/// K-Means initialization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InitMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization for better convergence
    KMeansPlusPlus,
    /// Forgy initialization
    Forgy,
    /// Random partition initialization
    RandomPartition,
}

impl Default for InitMethod {
    fn default() -> Self {
        Self::KMeansPlusPlus
    }
}

/// K-Means clustering configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iters: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Initialization method
    pub init_method: InitMethod,
    /// Number of random initializations (best result kept)
    pub n_init: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Algorithm variant
    pub algorithm: KMeansAlgorithm,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iters: 300,
            tolerance: 1e-4,
            init_method: InitMethod::KMeansPlusPlus,
            n_init: 10,
            random_state: None,
            algorithm: KMeansAlgorithm::Lloyd,
        }
    }
}

/// K-Means algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum KMeansAlgorithm {
    /// Lloyd's algorithm (standard)
    Lloyd,
    /// Elkan's algorithm (optimized for large k)
    Elkan,
    /// Mini-batch K-Means for large datasets
    MiniBatch,
}

impl Default for KMeansAlgorithm {
    fn default() -> Self {
        Self::Lloyd
    }
}

/// K-Means clustering result
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster labels for each data point
    pub labels: Tensor,
    /// Cluster centroids
    pub centroids: Tensor,
    /// Within-cluster sum of squares (inertia)
    pub inertia: f64,
    /// Number of iterations to convergence
    pub n_iter: usize,
    /// Whether algorithm converged
    pub converged: bool,
    /// Final centroids change
    pub final_change: f64,
}

impl ClusteringResult for KMeansResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.centroids.shape().dims()[0]
    }

    fn centers(&self) -> Option<&Tensor> {
        Some(&self.centroids)
    }

    fn inertia(&self) -> Option<f64> {
        Some(self.inertia)
    }

    fn n_iter(&self) -> Option<usize> {
        Some(self.n_iter)
    }

    fn converged(&self) -> bool {
        self.converged
    }
}

/// K-Means clustering algorithm
#[derive(Debug, Clone)]
pub struct KMeans {
    config: KMeansConfig,
    #[allow(dead_code)]
    centroids: Option<Tensor>,
    fitted: bool,
}

impl KMeans {
    /// Create a new K-Means clusterer
    pub fn new(n_clusters: usize) -> Self {
        Self {
            config: KMeansConfig {
                n_clusters,
                ..Default::default()
            },
            centroids: None,
            fitted: false,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.config.max_iters = max_iters;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Set initialization method
    pub fn init_method(mut self, init_method: InitMethod) -> Self {
        self.config.init_method = init_method;
        self
    }

    /// Set number of random initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.config.n_init = n_init;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set algorithm variant
    pub fn algorithm(mut self, algorithm: KMeansAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Initialize centroids using the specified method
    fn initialize_centroids(
        &self,
        data: &Tensor,
        rng: &mut CoreRandom<StdRng>,
    ) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let _n_features = data.shape().dims()[1];

        if self.config.n_clusters > n_samples {
            return Err(ClusterError::InvalidClusters(self.config.n_clusters));
        }

        match self.config.init_method {
            InitMethod::Random => self.init_random(data, rng),
            InitMethod::KMeansPlusPlus => self.init_kmeans_plus_plus(data, rng),
            InitMethod::Forgy => self.init_forgy(data, rng),
            InitMethod::RandomPartition => self.init_random_partition(data, rng),
        }
    }

    /// Random initialization
    fn init_random(&self, data: &Tensor, rng: &mut CoreRandom<StdRng>) -> ClusterResult<Tensor> {
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let mut centroids_data = Vec::with_capacity(self.config.n_clusters * n_features);

        // Compute data range for each feature
        let mut min_vals = vec![f32::INFINITY; n_features];
        let mut max_vals = vec![f32::NEG_INFINITY; n_features];

        for i in 0..data.shape().dims()[0] {
            for j in 0..n_features {
                let val = data_vec[i * n_features + j];
                min_vals[j] = min_vals[j].min(val);
                max_vals[j] = max_vals[j].max(val);
            }
        }

        // Generate random centroids within data range
        for _ in 0..self.config.n_clusters {
            for j in 0..n_features {
                let val = rng.random::<f32>() * (max_vals[j] - min_vals[j]) + min_vals[j];
                centroids_data.push(val);
            }
        }

        Tensor::from_vec(centroids_data, &[self.config.n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    /// K-means++ initialization for better initial centroids
    fn init_kmeans_plus_plus(
        &self,
        data: &Tensor,
        rng: &mut CoreRandom<StdRng>,
    ) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        let mut centroids_data = Vec::with_capacity(self.config.n_clusters * n_features);

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..n_samples);
        for j in 0..n_features {
            centroids_data.push(data_vec[first_idx * n_features + j]);
        }

        // Choose remaining centroids using K-means++ strategy
        for k in 1..self.config.n_clusters {
            let mut distances = vec![f32::INFINITY; n_samples];

            // Compute minimum distance to existing centroids for each point
            for i in 0..n_samples {
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..n_features {
                        let diff =
                            data_vec[i * n_features + j] - centroids_data[c * n_features + j];
                        dist += diff * diff;
                    }
                    distances[i] = distances[i].min(dist);
                }
            }

            // Choose next centroid with probability proportional to squared distance
            let total_dist: f32 = distances.iter().sum();
            if total_dist <= 0.0 {
                // Fallback to random selection
                let idx = rng.gen_range(0..n_samples);
                for j in 0..n_features {
                    centroids_data.push(data_vec[idx * n_features + j]);
                }
            } else {
                let threshold = rng.random::<f32>() * total_dist;
                let mut cumsum = 0.0;
                let mut selected_idx = 0;

                #[allow(clippy::needless_range_loop)]
                for i in 0..n_samples {
                    cumsum += distances[i];
                    if cumsum >= threshold {
                        selected_idx = i;
                        break;
                    }
                }

                for j in 0..n_features {
                    centroids_data.push(data_vec[selected_idx * n_features + j]);
                }
            }
        }

        Tensor::from_vec(centroids_data, &[self.config.n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    /// Forgy initialization - randomly select k data points as centroids
    fn init_forgy(&self, data: &Tensor, rng: &mut CoreRandom<StdRng>) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        let mut selected = std::collections::HashSet::new();
        let mut centroids_data = Vec::with_capacity(self.config.n_clusters * n_features);

        for _ in 0..self.config.n_clusters {
            let mut idx = rng.gen_range(0..n_samples);
            while selected.contains(&idx) {
                idx = rng.gen_range(0..n_samples);
            }
            selected.insert(idx);

            for j in 0..n_features {
                centroids_data.push(data_vec[idx * n_features + j]);
            }
        }

        Tensor::from_vec(centroids_data, &[self.config.n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    /// Random partition initialization
    fn init_random_partition(
        &self,
        data: &Tensor,
        rng: &mut CoreRandom<StdRng>,
    ) -> ClusterResult<Tensor> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        // Randomly assign points to clusters
        let mut cluster_assignments = Vec::new();
        for _ in 0..n_samples {
            cluster_assignments.push(rng.gen_range(0..self.config.n_clusters));
        }

        // Compute centroid for each cluster
        let mut centroids_data = vec![0.0; self.config.n_clusters * n_features];
        let mut cluster_counts = vec![0; self.config.n_clusters];

        for i in 0..n_samples {
            let cluster = cluster_assignments[i];
            cluster_counts[cluster] += 1;
            for j in 0..n_features {
                centroids_data[cluster * n_features + j] += data_vec[i * n_features + j];
            }
        }

        // Average to get centroids
        for k in 0..self.config.n_clusters {
            if cluster_counts[k] > 0 {
                for j in 0..n_features {
                    centroids_data[k * n_features + j] /= cluster_counts[k] as f32;
                }
            }
        }

        Tensor::from_vec(centroids_data, &[self.config.n_clusters, n_features])
            .map_err(ClusterError::TensorError)
    }

    /// K-means run with Lloyd's algorithm (standard implementation)
    /// Enhanced with parallel distance computations when beneficial
    fn fit_lloyd(&self, data: &Tensor, mut rng: CoreRandom<StdRng>) -> ClusterResult<KMeansResult> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];

        // Use parallel implementation for larger datasets
        if n_samples >= 1000 && parallel::parallel_enabled() {
            return self.fit_lloyd_parallel(data, rng);
        }

        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, &mut rng)?;
        let mut labels = zeros(&[n_samples])?;
        let mut converged = false;
        let mut n_iter = 0;
        let mut final_change = f64::INFINITY;

        for iter in 0..self.config.max_iters {
            n_iter = iter + 1;

            // Assignment step: assign each point to nearest centroid
            let centroids_vec = centroids.to_vec().map_err(ClusterError::TensorError)?;
            let mut labels_vec = vec![0.0; n_samples];

            for i in 0..n_samples {
                let mut min_dist = f32::INFINITY;
                let mut best_cluster = 0;

                for k in 0..self.config.n_clusters {
                    let mut dist = 0.0;
                    for j in 0..n_features {
                        let diff = data_vec[i * n_features + j] - centroids_vec[k * n_features + j];
                        dist += diff * diff;
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = k;
                    }
                }

                labels_vec[i] = best_cluster as f32;
            }

            labels = Tensor::from_vec(labels_vec.clone(), &[n_samples])
                .map_err(ClusterError::TensorError)?;

            // Update step: recompute centroids
            let old_centroids = centroids.clone();
            let mut new_centroids_data = vec![0.0; self.config.n_clusters * n_features];
            let mut cluster_counts = vec![0; self.config.n_clusters];

            for i in 0..n_samples {
                let cluster = labels_vec[i] as usize;
                cluster_counts[cluster] += 1;
                for j in 0..n_features {
                    new_centroids_data[cluster * n_features + j] += data_vec[i * n_features + j];
                }
            }

            // Average to get new centroids
            for k in 0..self.config.n_clusters {
                if cluster_counts[k] > 0 {
                    for j in 0..n_features {
                        new_centroids_data[k * n_features + j] /= cluster_counts[k] as f32;
                    }
                }
            }

            centroids = Tensor::from_vec(new_centroids_data, &[self.config.n_clusters, n_features])
                .map_err(ClusterError::TensorError)?;

            // Check convergence
            let change = self.compute_centroid_change(&old_centroids, &centroids)?;
            final_change = change;

            if change < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute inertia (within-cluster sum of squares)
        let inertia = self.compute_inertia(data, &labels, &centroids)?;

        Ok(KMeansResult {
            labels,
            centroids,
            inertia,
            n_iter,
            converged,
            final_change,
        })
    }

    /// Parallel Lloyd's algorithm using SciRS2 parallel utilities
    fn fit_lloyd_parallel(
        &self,
        data: &Tensor,
        mut rng: CoreRandom<StdRng>,
    ) -> ClusterResult<KMeansResult> {
        let n_samples = data.shape().dims()[0];

        // Convert to Array2 for parallel operations
        let data_array = tensor_to_array2(data)?;

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, &mut rng)?;
        let mut converged = false;
        let mut n_iter = 0;
        let mut final_change = f64::INFINITY;

        for iter in 0..self.config.max_iters {
            n_iter = iter + 1;

            // Convert centroids to Array2
            let centroids_array = tensor_to_array2(&centroids)?;

            // Parallel K-means iteration using optimized parallel utilities
            let (new_centroids_array, _labels_array, _inertia_f32) =
                parallel::parallel_kmeans_iteration_f32(&data_array, &centroids_array)?;

            // Convert back to tensors
            let new_centroids = array2_to_tensor(&new_centroids_array)?;

            // Check convergence
            let change = self.compute_centroid_change(&centroids, &new_centroids)?;
            final_change = change;

            centroids = new_centroids;

            if change < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Final assignment and inertia computation
        let centroids_array = tensor_to_array2(&centroids)?;
        let (_, labels_array, inertia_f32) =
            parallel::parallel_kmeans_iteration_f32(&data_array, &centroids_array)?;

        let labels_usize = labels_array.to_vec();
        let labels = Tensor::from_vec(
            labels_usize.into_iter().map(|x| x as f32).collect(),
            &[n_samples],
        )
        .map_err(ClusterError::TensorError)?;

        Ok(KMeansResult {
            labels,
            centroids,
            inertia: inertia_f32 as f64,
            n_iter,
            converged,
            final_change,
        })
    }

    /// Compute change in centroids between iterations
    fn compute_centroid_change(
        &self,
        old_centroids: &Tensor,
        new_centroids: &Tensor,
    ) -> ClusterResult<f64> {
        let old_vec = old_centroids.to_vec().map_err(ClusterError::TensorError)?;
        let new_vec = new_centroids.to_vec().map_err(ClusterError::TensorError)?;

        let mut total_change = 0.0;
        for i in 0..old_vec.len() {
            let diff = (new_vec[i] - old_vec[i]) as f64;
            total_change += diff * diff;
        }

        Ok(total_change.sqrt())
    }

    /// Compute within-cluster sum of squares (inertia)
    fn compute_inertia(
        &self,
        data: &Tensor,
        labels: &Tensor,
        centroids: &Tensor,
    ) -> ClusterResult<f64> {
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let labels_vec = labels.to_vec().map_err(ClusterError::TensorError)?;
        let centroids_vec = centroids.to_vec().map_err(ClusterError::TensorError)?;

        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let mut inertia = 0.0;

        for i in 0..n_samples {
            let cluster = labels_vec[i] as usize;
            for j in 0..n_features {
                let diff =
                    (data_vec[i * n_features + j] - centroids_vec[cluster * n_features + j]) as f64;
                inertia += diff * diff;
            }
        }

        Ok(inertia)
    }

    /// K-means run with Elkan's algorithm (optimized for large k)
    /// Uses triangle inequality to reduce distance computations
    fn fit_elkan(&self, data: &Tensor, mut rng: CoreRandom<StdRng>) -> ClusterResult<KMeansResult> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, &mut rng)?;
        let mut labels = zeros(&[n_samples])?;
        let mut converged = false;
        let mut n_iter = 0;
        let mut final_change = f64::INFINITY;

        // Elkan's algorithm specific data structures
        let mut upper_bounds = vec![f32::INFINITY; n_samples]; // Upper bound on distance to assigned centroid
        let mut lower_bounds = vec![vec![0.0f32; self.config.n_clusters]; n_samples]; // Lower bounds to each centroid
        let mut center_distances =
            vec![vec![0.0f32; self.config.n_clusters]; self.config.n_clusters]; // Distances between centroids

        for iter in 0..self.config.max_iters {
            n_iter = iter + 1;

            // Compute distances between all pairs of centroids
            let centroids_vec = centroids.to_vec().map_err(ClusterError::TensorError)?;
            for i in 0..self.config.n_clusters {
                for j in i + 1..self.config.n_clusters {
                    let mut dist = 0.0;
                    for k in 0..n_features {
                        let diff =
                            centroids_vec[i * n_features + k] - centroids_vec[j * n_features + k];
                        dist += diff * diff;
                    }
                    let sqrt_dist = dist.sqrt();
                    center_distances[i][j] = sqrt_dist;
                    center_distances[j][i] = sqrt_dist;
                }
            }

            // Assignment step with triangle inequality optimization
            let mut labels_vec = vec![0.0; n_samples];
            for i in 0..n_samples {
                let current_label = if iter == 0 { 0 } else { labels_vec[i] as usize };
                let mut min_dist = upper_bounds[i];
                let mut best_cluster = current_label;

                // Use triangle inequality to avoid distance computations
                for k in 0..self.config.n_clusters {
                    if k == current_label
                        || upper_bounds[i] <= lower_bounds[i][k]
                        || upper_bounds[i] <= 0.5 * center_distances[current_label][k]
                    {
                        continue;
                    }

                    // Need to compute actual distance
                    let mut dist = 0.0;
                    for j in 0..n_features {
                        let diff = data_vec[i * n_features + j] - centroids_vec[k * n_features + j];
                        dist += diff * diff;
                    }
                    let sqrt_dist = dist.sqrt();

                    lower_bounds[i][k] = sqrt_dist;
                    if sqrt_dist < min_dist {
                        min_dist = sqrt_dist;
                        best_cluster = k;
                    }
                }

                labels_vec[i] = best_cluster as f32;
                upper_bounds[i] = min_dist;
            }

            labels = Tensor::from_vec(labels_vec.clone(), &[n_samples])
                .map_err(ClusterError::TensorError)?;

            // Update step: recompute centroids
            let old_centroids = centroids.clone();
            let mut new_centroids_data = vec![0.0; self.config.n_clusters * n_features];
            let mut cluster_counts = vec![0; self.config.n_clusters];

            for i in 0..n_samples {
                let cluster = labels_vec[i] as usize;
                cluster_counts[cluster] += 1;
                for j in 0..n_features {
                    new_centroids_data[cluster * n_features + j] += data_vec[i * n_features + j];
                }
            }

            // Average to get new centroids and compute movement
            let mut centroid_movements = vec![0.0; self.config.n_clusters];
            #[allow(clippy::needless_range_loop)]
            for k in 0..self.config.n_clusters {
                if cluster_counts[k] > 0 {
                    let mut movement = 0.0;
                    for j in 0..n_features {
                        let old_val = centroids_vec[k * n_features + j];
                        let new_val =
                            new_centroids_data[k * n_features + j] / cluster_counts[k] as f32;
                        new_centroids_data[k * n_features + j] = new_val;
                        let diff = new_val - old_val;
                        movement += diff * diff;
                    }
                    centroid_movements[k] = movement.sqrt();
                }
            }

            centroids = Tensor::from_vec(new_centroids_data, &[self.config.n_clusters, n_features])
                .map_err(ClusterError::TensorError)?;

            // Update bounds based on centroid movements
            for i in 0..n_samples {
                let assigned_cluster = labels_vec[i] as usize;
                upper_bounds[i] += centroid_movements[assigned_cluster];

                #[allow(clippy::needless_range_loop)]
                for k in 0..self.config.n_clusters {
                    lower_bounds[i][k] = (lower_bounds[i][k] - centroid_movements[k]).max(0.0);
                }
            }

            // Check convergence
            let change = self.compute_centroid_change(&old_centroids, &centroids)?;
            final_change = change;

            if change < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute inertia
        let inertia = self.compute_inertia(data, &labels, &centroids)?;

        Ok(KMeansResult {
            labels,
            centroids,
            inertia,
            n_iter,
            converged,
            final_change,
        })
    }

    /// K-means run with Mini-batch algorithm (for large datasets)
    /// Uses random mini-batches to update centroids incrementally
    fn fit_minibatch(
        &self,
        data: &Tensor,
        mut rng: CoreRandom<StdRng>,
    ) -> ClusterResult<KMeansResult> {
        let n_samples = data.shape().dims()[0];
        let n_features = data.shape().dims()[1];
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

        // Mini-batch size (adaptive based on dataset size)
        let batch_size = (n_samples / 10).clamp(100, 1000);

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, &mut rng)?;
        let mut converged = false;
        let mut n_iter = 0;
        let mut final_change = f64::INFINITY;

        // Mini-batch specific: track per-center counts for learning rate
        let mut center_counts = vec![0.0f32; self.config.n_clusters];

        for iter in 0..self.config.max_iters {
            n_iter = iter + 1;

            // Sample random mini-batch
            let mut batch_indices = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                batch_indices.push(rng.gen_range(0..n_samples));
            }

            let old_centroids = centroids.clone();
            let mut centroids_vec = centroids.to_vec().map_err(ClusterError::TensorError)?;

            // Process mini-batch
            for &idx in &batch_indices {
                // Find closest centroid for this point
                let mut min_dist = f32::INFINITY;
                let mut best_cluster = 0;

                for k in 0..self.config.n_clusters {
                    let mut dist = 0.0;
                    for j in 0..n_features {
                        let diff =
                            data_vec[idx * n_features + j] - centroids_vec[k * n_features + j];
                        dist += diff * diff;
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = k;
                    }
                }

                // Update centroid with adaptive learning rate
                center_counts[best_cluster] += 1.0;
                let learning_rate = 1.0 / center_counts[best_cluster];

                for j in 0..n_features {
                    let current_centroid = centroids_vec[best_cluster * n_features + j];
                    let data_point = data_vec[idx * n_features + j];
                    centroids_vec[best_cluster * n_features + j] =
                        current_centroid + learning_rate * (data_point - current_centroid);
                }
            }

            centroids = Tensor::from_vec(centroids_vec, &[self.config.n_clusters, n_features])
                .map_err(ClusterError::TensorError)?;

            // Check convergence every few iterations (less frequent for efficiency)
            if iter % 5 == 0 {
                let change = self.compute_centroid_change(&old_centroids, &centroids)?;
                final_change = change;

                if change < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        }

        // Final assignment of all points to nearest centroids
        let centroids_vec = centroids.to_vec().map_err(ClusterError::TensorError)?;
        let mut labels_vec = vec![0.0; n_samples];

        for i in 0..n_samples {
            let mut min_dist = f32::INFINITY;
            let mut best_cluster = 0;

            for k in 0..self.config.n_clusters {
                let mut dist = 0.0;
                for j in 0..n_features {
                    let diff = data_vec[i * n_features + j] - centroids_vec[k * n_features + j];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            labels_vec[i] = best_cluster as f32;
        }

        let labels =
            Tensor::from_vec(labels_vec, &[n_samples]).map_err(ClusterError::TensorError)?;

        // Compute inertia
        let inertia = self.compute_inertia(data, &labels, &centroids)?;

        Ok(KMeansResult {
            labels,
            centroids,
            inertia,
            n_iter,
            converged,
            final_change,
        })
    }
}

impl ClusteringAlgorithm for KMeans {
    fn name(&self) -> &str {
        "K-Means"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.config.n_clusters.to_string());
        params.insert("max_iters".to_string(), self.config.max_iters.to_string());
        params.insert("tolerance".to_string(), self.config.tolerance.to_string());
        params.insert(
            "init_method".to_string(),
            format!("{:?}", self.config.init_method),
        );
        params.insert("n_init".to_string(), self.config.n_init.to_string());
        params.insert(
            "algorithm".to_string(),
            format!("{:?}", self.config.algorithm),
        );
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> ClusterResult<()> {
        for (key, value) in params {
            match key.as_str() {
                "n_clusters" => {
                    self.config.n_clusters = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid n_clusters: {}", value))
                    })?;
                }
                "max_iters" => {
                    self.config.max_iters = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid max_iters: {}", value))
                    })?;
                }
                "tolerance" => {
                    self.config.tolerance = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid tolerance: {}", value))
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

impl Fit for KMeans {
    type Result = KMeansResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;

        let base_seed = match self.config.random_state {
            Some(seed) => seed,
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time should be after UNIX_EPOCH")
                    .as_secs()
            }
        };

        // Select algorithm implementation based on configuration - first run
        let rng_0 = seeded_rng(base_seed);
        let mut best_result = match self.config.algorithm {
            KMeansAlgorithm::Lloyd => self.fit_lloyd(data, rng_0)?,
            KMeansAlgorithm::Elkan => self.fit_elkan(data, rng_0)?,
            KMeansAlgorithm::MiniBatch => self.fit_minibatch(data, rng_0)?,
        };
        let mut best_inertia = best_result.inertia;

        // Run multiple times and keep best result
        for i in 1..self.config.n_init {
            let rng_i = seeded_rng(base_seed.wrapping_add(i as u64));
            let result = match self.config.algorithm {
                KMeansAlgorithm::Lloyd => self.fit_lloyd(data, rng_i)?,
                KMeansAlgorithm::Elkan => self.fit_elkan(data, rng_i)?,
                KMeansAlgorithm::MiniBatch => self.fit_minibatch(data, rng_i)?,
            };
            if result.inertia < best_inertia {
                best_inertia = result.inertia;
                best_result = result;
            }
        }

        Ok(best_result)
    }
}

impl FitPredict for KMeans {
    type Result = KMeansResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

// Utility functions for tensor/array conversions
fn tensor_to_array2(tensor: &Tensor) -> ClusterResult<Array2<f32>> {
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();
    if shape.len() != 2 {
        return Err(ClusterError::InvalidInput("Expected 2D tensor".to_string()));
    }

    let data_f32: Vec<f32> = tensor.to_vec().map_err(ClusterError::TensorError)?;
    Array2::from_shape_vec((shape[0], shape[1]), data_f32)
        .map_err(|_| ClusterError::InvalidInput("Failed to convert tensor to array".to_string()))
}

fn array2_to_tensor(array: &Array2<f32>) -> ClusterResult<Tensor> {
    let (rows, cols) = array.dim();
    let data_f32: Vec<f32> = array.iter().copied().collect();
    Tensor::from_vec(data_f32, &[rows, cols]).map_err(ClusterError::TensorError)
}
