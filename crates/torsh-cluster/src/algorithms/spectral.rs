//! Spectral clustering implementation
//!
//! This module provides a complete implementation of spectral clustering,
//! which uses eigendecomposition of a similarity matrix for dimensionality
//! reduction before applying K-means clustering.
//!
//! # Algorithm Overview
//!
//! Spectral clustering works in three main steps:
//! 1. **Affinity Matrix**: Compute similarity matrix between all data points
//! 2. **Graph Laplacian**: Compute normalized Laplacian matrix
//! 3. **Spectral Embedding**: Use eigenvectors for dimensionality reduction
//! 4. **K-means**: Apply K-means to the embedded data

use crate::algorithms::kmeans::KMeans;
use crate::error::{ClusterError, ClusterResult};
use crate::traits::{
    AlgorithmComplexity, ClusteringAlgorithm, ClusteringConfig, ClusteringResult, Fit, FitPredict,
    MemoryPattern,
};
use crate::utils::validation::{validate_cluster_input, validate_n_clusters};
use scirs2_autograd::{self as ag, tensor_ops::linear_algebra::*, tensor_ops::*};
use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Affinity methods for spectral clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AffinityType {
    /// Radial Basis Function (RBF) kernel
    Rbf,
    /// K-nearest neighbors
    KNearest,
    /// Fully connected with RBF kernel
    FullyConnected,
    /// Precomputed affinity matrix
    Precomputed,
}

impl Default for AffinityType {
    fn default() -> Self {
        Self::Rbf
    }
}

impl std::fmt::Display for AffinityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rbf => write!(f, "rbf"),
            Self::KNearest => write!(f, "nearest_neighbors"),
            Self::FullyConnected => write!(f, "fully_connected"),
            Self::Precomputed => write!(f, "precomputed"),
        }
    }
}

/// Spectral clustering configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpectralConfig {
    /// Number of clusters to find
    pub n_clusters: usize,
    /// Affinity method for building similarity matrix
    pub affinity: AffinityType,
    /// Gamma parameter for RBF kernel
    pub gamma: f64,
    /// Number of nearest neighbors (for k-nearest neighbors affinity)
    pub n_neighbors: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Whether to assign labels to original data points
    pub assign_labels: bool,
    /// Eigenvalue solver tolerance
    pub eigen_tolerance: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            affinity: AffinityType::Rbf,
            gamma: 1.0,
            n_neighbors: 10,
            random_state: None,
            assign_labels: true,
            eigen_tolerance: 0.0,
        }
    }
}

impl ClusteringConfig for SpectralConfig {
    fn validate(&self) -> ClusterResult<()> {
        if self.n_clusters == 0 {
            return Err(ClusterError::InvalidClusters(self.n_clusters));
        }
        if self.gamma <= 0.0 {
            return Err(ClusterError::ConfigError(
                "gamma must be positive".to_string(),
            ));
        }
        if self.n_neighbors == 0 {
            return Err(ClusterError::ConfigError(
                "n_neighbors must be positive".to_string(),
            ));
        }
        if self.eigen_tolerance < 0.0 {
            return Err(ClusterError::ConfigError(
                "eigen_tolerance must be non-negative".to_string(),
            ));
        }
        Ok(())
    }

    fn default() -> Self {
        <SpectralConfig as std::default::Default>::default()
    }

    fn merge(&mut self, other: &Self) {
        let default_config = <SpectralConfig as std::default::Default>::default();
        if other.n_clusters != default_config.n_clusters {
            self.n_clusters = other.n_clusters;
        }
        if other.gamma != default_config.gamma {
            self.gamma = other.gamma;
        }
        if other.n_neighbors != default_config.n_neighbors {
            self.n_neighbors = other.n_neighbors;
        }
    }
}

/// Spectral clustering result
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Cluster assignments for each sample
    pub labels: Tensor,
    /// Affinity (similarity) matrix [n_samples, n_samples]
    pub affinity_matrix: Tensor,
    /// Spectral embedding [n_samples, n_clusters]
    pub embedding: Tensor,
    /// Eigenvalues from the Laplacian
    pub eigenvalues: Tensor,
    /// Whether the embedding was successful
    pub embedding_success: bool,
    /// Number of iterations for K-means on embedding
    pub kmeans_iterations: usize,
}

impl ClusteringResult for SpectralResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        // Count unique labels
        if let Ok(labels_vec) = self.labels.to_vec() {
            let mut unique_labels: Vec<i32> = labels_vec.iter().map(|&x| x as i32).collect();
            unique_labels.sort_unstable();
            unique_labels.dedup();
            unique_labels.len()
        } else {
            0
        }
    }

    fn converged(&self) -> bool {
        self.embedding_success
    }

    fn n_iter(&self) -> Option<usize> {
        Some(self.kmeans_iterations)
    }

    fn metadata(&self) -> Option<&HashMap<String, String>> {
        None // Could be enhanced to return eigenvalue information
    }
}

/// Spectral clustering algorithm
///
/// Spectral clustering performs dimensionality reduction using eigenvectors
/// of a similarity matrix (graph Laplacian), followed by K-means clustering
/// on the reduced spectral embedding.
///
/// # Mathematical Formulation
///
/// ## Graph Construction
///
/// Given data points X = {x₁, ..., xₙ}, construct a similarity graph where
/// vertices represent data points and edges represent similarities.
///
/// ### Affinity Matrix
///
/// **RBF (Radial Basis Function) Kernel:**
/// ```text
/// W_{ij} = exp(-γ ||x_i - x_j||²)
/// ```
///
/// **K-Nearest Neighbors:**
/// ```text
/// W_{ij} = { 1  if x_j is among k nearest neighbors of x_i
///          { 0  otherwise
/// ```
///
/// where γ is the kernel coefficient.
///
/// ## Graph Laplacian
///
/// The normalized graph Laplacian is constructed as:
///
/// ```text
/// L_norm = I - D^(-1/2) W D^(-1/2)
/// ```
///
/// where:
/// - `W` is the affinity (similarity) matrix
/// - `D` is the degree matrix: `D_{ii} = Σ_j W_{ij}`
/// - `I` is the identity matrix
///
/// **Alternative (Symmetric) Normalization:**
/// ```text
/// L_sym = D^(-1/2) W D^(-1/2)
/// ```
///
/// ## Eigendecomposition
///
/// Compute the k smallest eigenvectors of L_norm (or k largest of L_sym):
///
/// ```text
/// L_norm v_i = λ_i v_i,  for i = 1, ..., k
/// ```
///
/// where:
/// - `λ₁ ≤ λ₂ ≤ ... ≤ λₙ` are eigenvalues
/// - `v_i` are corresponding eigenvectors
///
/// ## Spectral Embedding
///
/// Form the embedding matrix U ∈ ℝⁿˣᵏ from the k eigenvectors:
///
/// ```text
/// U = [v₁ | v₂ | ... | vₖ]
/// ```
///
/// Each row u_i of U represents the embedding of point x_i in the k-dimensional space.
///
/// ### Row Normalization
///
/// Normalize each row to unit length:
///
/// ```text
/// u_i ← u_i / ||u_i||₂
/// ```
///
/// This ensures points lie on the unit sphere in embedding space.
///
/// ## K-Means Clustering
///
/// Apply K-means to the embedded points U to obtain final cluster assignments:
///
/// ```text
/// {C₁, ..., Cₖ} = KMeans(U, k)
/// ```
///
/// # Theoretical Properties
///
/// ## Graph Cut Interpretation
///
/// Spectral clustering approximates the normalized cut objective:
///
/// ```text
/// NCut(C₁, ..., Cₖ) = Σᵢ cut(Cᵢ, C̄ᵢ) / vol(Cᵢ)
/// ```
///
/// where:
/// - `cut(A, B) = Σ_{i∈A, j∈B} W_{ij}`
/// - `vol(A) = Σ_{i∈A} d_i`
///
/// ## Spectral Gap
///
/// The quality of clustering is related to the spectral gap:
///
/// ```text
/// gap = λₖ₊₁ - λₖ
/// ```
///
/// A larger gap suggests k is a good choice for the number of clusters.
///
/// # Advantages
///
/// - Can identify non-convex clusters (unlike K-means)
/// - Works well when clusters have complex shapes
/// - Based on graph theory and has strong theoretical foundations
/// - No assumption about cluster shape or density
///
/// # When to Use
///
/// - Data has non-linearly separable clusters
/// - Clusters have arbitrary shapes (moons, circles, etc.)
/// - Graph structure is meaningful for your data
/// - Computational resources allow eigendecomposition (O(n³) complexity)
///
/// # Example
///
/// ```rust
/// use torsh_cluster::algorithms::spectral::{SpectralClustering, AffinityType};
/// use torsh_cluster::traits::Fit;
/// use torsh_tensor::creation::randn;
///
/// let data = randn::<f32>(&[100, 2])?;
/// let spectral = SpectralClustering::new(3)
///     .affinity(AffinityType::Rbf)
///     .gamma(1.0)
///     .n_neighbors(10);
/// let result = spectral.fit(&data)?;
/// println!("Embedding shape: {:?}", result.embedding.shape());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct SpectralClustering {
    config: SpectralConfig,
}

impl SpectralClustering {
    /// Create a new spectral clustering algorithm
    pub fn new(n_clusters: usize) -> Self {
        Self {
            config: SpectralConfig {
                n_clusters,
                ..Default::default()
            },
        }
    }

    /// Set the affinity method
    pub fn affinity(mut self, affinity: AffinityType) -> Self {
        self.config.affinity = affinity;
        self
    }

    /// Set the gamma parameter for RBF kernel
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Set the number of nearest neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Set eigenvalue solver tolerance
    pub fn eigen_tolerance(mut self, tolerance: f64) -> Self {
        self.config.eigen_tolerance = tolerance;
        self
    }
}

impl ClusteringAlgorithm for SpectralClustering {
    fn name(&self) -> &str {
        "Spectral Clustering"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.config.n_clusters.to_string());
        params.insert("affinity".to_string(), self.config.affinity.to_string());
        params.insert("gamma".to_string(), self.config.gamma.to_string());
        params.insert(
            "n_neighbors".to_string(),
            self.config.n_neighbors.to_string(),
        );
        params.insert(
            "eigen_tolerance".to_string(),
            self.config.eigen_tolerance.to_string(),
        );
        if let Some(seed) = self.config.random_state {
            params.insert("random_state".to_string(), seed.to_string());
        }
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
                "gamma" => {
                    self.config.gamma = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid gamma: {}", value))
                    })?;
                }
                "n_neighbors" => {
                    self.config.n_neighbors = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid n_neighbors: {}", value))
                    })?;
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
        false // For now, always return false since we don't store fitted state
    }

    fn complexity_info(&self) -> AlgorithmComplexity {
        AlgorithmComplexity {
            time_complexity: "O(n³)".to_string(), // Dominated by eigendecomposition
            space_complexity: "O(n²)".to_string(), // Affinity matrix storage
            deterministic: false,
            online_capable: false,
            memory_pattern: MemoryPattern::Quadratic,
        }
    }

    fn supported_distance_metrics(&self) -> Vec<&str> {
        vec!["rbf", "cosine", "euclidean", "precomputed"]
    }
}

impl Fit for SpectralClustering {
    type Result = SpectralResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;
        validate_n_clusters(self.config.n_clusters, data.shape().dims()[0])?;
        validate_cluster_input(data)?;
        self.config.validate()?;

        let _n_samples = data.shape().dims()[0];
        let _n_features = data.shape().dims()[1];

        // Convert tensor to ndarray for computation
        let data_array = tensor_to_array2(data)?;

        // Step 1: Compute affinity matrix
        let affinity_matrix = self.compute_affinity_matrix(&data_array)?;

        // Step 2: Compute graph Laplacian
        let laplacian = self.compute_normalized_laplacian(&affinity_matrix)?;

        // Step 3: Compute eigendecomposition
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&laplacian)?;

        // Step 4: Create spectral embedding using smallest eigenvalues
        let embedding = self.create_spectral_embedding(&eigenvalues, &eigenvectors)?;

        // Step 5: Apply K-means to the embedding
        let (labels, kmeans_iterations) = self.cluster_embedding(&embedding)?;

        Ok(SpectralResult {
            labels: array1_i32_to_tensor(&labels)?,
            affinity_matrix: array2_to_tensor(&affinity_matrix)?,
            embedding: array2_to_tensor(&embedding)?,
            eigenvalues: array1_to_tensor(&eigenvalues)?,
            embedding_success: true,
            kmeans_iterations,
        })
    }
}

impl FitPredict for SpectralClustering {
    type Result = SpectralResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

// Core spectral clustering implementation
impl SpectralClustering {
    /// Compute affinity matrix based on the specified affinity type
    fn compute_affinity_matrix(&self, data: &Array2<f64>) -> ClusterResult<Array2<f64>> {
        let (n_samples, _) = data.dim();
        let mut affinity = Array2::<f64>::zeros((n_samples, n_samples));

        match self.config.affinity {
            AffinityType::Rbf | AffinityType::FullyConnected => {
                // RBF (Gaussian) kernel: exp(-gamma * ||x_i - x_j||^2)
                for i in 0..n_samples {
                    for j in i..n_samples {
                        let diff = &data.row(i) - &data.row(j);
                        let dist_sq = diff.dot(&diff);
                        let similarity = (-self.config.gamma * dist_sq).exp();
                        affinity[[i, j]] = similarity;
                        affinity[[j, i]] = similarity;
                    }
                }
            }
            AffinityType::KNearest => {
                // K-nearest neighbors affinity
                for i in 0..n_samples {
                    let mut distances: Vec<(f64, usize)> = Vec::new();

                    for j in 0..n_samples {
                        if i != j {
                            let diff = &data.row(i) - &data.row(j);
                            let dist = diff.dot(&diff).sqrt();
                            distances.push((dist, j));
                        }
                    }

                    // Sort by distance and take k nearest
                    distances
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let k = self.config.n_neighbors.min(distances.len());

                    for &(dist, j) in distances.iter().take(k) {
                        let similarity = (-self.config.gamma * dist * dist).exp();
                        affinity[[i, j]] = similarity;
                        affinity[[j, i]] = similarity; // Make symmetric
                    }
                }
            }
            AffinityType::Precomputed => {
                // For precomputed, we assume the data is already an affinity matrix
                if data.nrows() != data.ncols() {
                    return Err(ClusterError::InvalidAffinityMatrix(
                        "Precomputed affinity matrix must be square".to_string(),
                    ));
                }
                affinity = data.clone();
            }
        }

        Ok(affinity)
    }

    /// Compute normalized graph Laplacian
    fn compute_normalized_laplacian(&self, affinity: &Array2<f64>) -> ClusterResult<Array2<f64>> {
        let n_samples = affinity.nrows();
        let mut laplacian = Array2::<f64>::zeros((n_samples, n_samples));

        // Compute degree matrix (sum of each row)
        let mut degrees = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            degrees[i] = affinity.row(i).sum();
            if degrees[i] == 0.0 {
                degrees[i] = 1.0; // Avoid division by zero
            }
        }

        // Compute normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    laplacian[[i, j]] =
                        1.0 - affinity[[i, j]] / (degrees[i].sqrt() * degrees[j].sqrt());
                } else {
                    laplacian[[i, j]] = -affinity[[i, j]] / (degrees[i].sqrt() * degrees[j].sqrt());
                }
            }
        }

        Ok(laplacian)
    }

    /// Compute eigendecomposition using SciRS2's symmetric eigendecomposition
    fn compute_eigendecomposition(
        &self,
        laplacian: &Array2<f64>,
    ) -> ClusterResult<(Array1<f64>, Array2<f64>)> {
        // Use SciRS2's autograd system for eigendecomposition
        let result = ag::run(|g| -> Result<_, String> {
            // Convert ndarray to SciRS2 autograd tensor (as f32)
            let laplacian_f32 = laplacian.mapv(|x| x as f32);
            let laplacian_tensor = convert_to_tensor(laplacian_f32, g);

            // Use symmetric eigendecomposition (eigh) since Laplacian is symmetric
            // eigh returns eigenvalues in ascending order (smallest first), which is what we want
            let (eigenvalues, eigenvectors) = eigh(&laplacian_tensor);

            // Evaluate the results with error handling
            let vals = eigenvalues
                .eval(g)
                .map_err(|e| format!("Failed to evaluate eigenvalues: {:?}", e))?;
            let vecs = eigenvectors
                .eval(g)
                .map_err(|e| format!("Failed to evaluate eigenvectors: {:?}", e))?;

            Ok((vals, vecs))
        });

        // Handle the result and convert potential errors
        let (vals_f32, vecs_f32) = result
            .map_err(|e| ClusterError::SciRS2Error(format!("Eigendecomposition failed: {}", e)))?;

        // Convert results back to f64 ndarray with proper dimensions
        let eigenvalues = vals_f32.mapv(|x| x as f64);
        let eigenvectors = vecs_f32.mapv(|x| x as f64);

        // Validate the dimensions and convert to fixed-size arrays
        let n = laplacian.nrows();
        if eigenvalues.len() != n || eigenvectors.shape()[0] != n {
            return Err(ClusterError::SciRS2Error(
                "Eigendecomposition returned invalid dimensions".to_string(),
            ));
        }

        // Convert dynamic arrays to fixed dimension arrays
        let eigenvals_array = eigenvalues
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .map_err(|e| {
                ClusterError::SciRS2Error(format!(
                    "Failed to convert eigenvalues to 1D array: {:?}",
                    e
                ))
            })?;

        let eigenvecs_array = eigenvectors
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| {
                ClusterError::SciRS2Error(format!(
                    "Failed to convert eigenvectors to 2D array: {:?}",
                    e
                ))
            })?;

        Ok((eigenvals_array, eigenvecs_array))
    }

    /// Create spectral embedding using the smallest eigenvectors
    fn create_spectral_embedding(
        &self,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
    ) -> ClusterResult<Array2<f64>> {
        let n_samples = eigenvectors.nrows();
        let n_clusters = self.config.n_clusters;

        // Take the eigenvectors corresponding to the smallest eigenvalues
        // (excluding the first one which is typically all ones)
        let start_idx = 1.min(eigenvalues.len());
        let end_idx = (start_idx + n_clusters).min(eigenvalues.len());

        if end_idx <= start_idx {
            return Err(ClusterError::ConfigError(
                "Not enough eigenvalues for the specified number of clusters".to_string(),
            ));
        }

        let actual_clusters = end_idx - start_idx;
        let mut embedding = Array2::<f64>::zeros((n_samples, actual_clusters));

        for (col_idx, eigen_idx) in (start_idx..end_idx).enumerate() {
            for row_idx in 0..n_samples {
                embedding[[row_idx, col_idx]] = eigenvectors[[row_idx, eigen_idx]];
            }
        }

        // Normalize the embedding (L2 norm per row)
        for i in 0..n_samples {
            let row = embedding.row(i);
            let norm = row.dot(&row).sqrt();
            if norm > 1e-10 {
                for j in 0..actual_clusters {
                    embedding[[i, j]] /= norm;
                }
            }
        }

        Ok(embedding)
    }

    /// Apply K-means clustering to the spectral embedding
    fn cluster_embedding(&self, embedding: &Array2<f64>) -> ClusterResult<(Array1<i32>, usize)> {
        // Convert embedding to tensor for K-means
        let embedding_tensor = array2_to_tensor(embedding)?;

        // Configure K-means using builder pattern
        let n_clusters = self.config.n_clusters.min(embedding.ncols());
        let mut kmeans = KMeans::new(n_clusters).max_iters(100).tolerance(1e-4);

        if let Some(seed) = self.config.random_state {
            kmeans = kmeans.random_state(seed);
        }

        let kmeans_result = kmeans.fit(&embedding_tensor)?;

        // Convert labels back to ndarray
        let labels_tensor = kmeans_result.labels();
        let labels_vec: Vec<f32> = labels_tensor.to_vec().map_err(ClusterError::TensorError)?;
        let labels = Array1::from_vec(labels_vec.into_iter().map(|x| x as i32).collect());

        let iterations = kmeans_result.n_iter().unwrap_or(100);

        Ok((labels, iterations))
    }
}

// Utility functions for tensor/array conversions (reuse from GMM)
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

fn array2_to_tensor(array: &Array2<f64>) -> ClusterResult<Tensor> {
    let (rows, cols) = array.dim();
    let data_f64: Vec<f64> = array.iter().copied().collect();
    let data: Vec<f32> = data_f64.into_iter().map(|x| x as f32).collect();
    Tensor::from_vec(data, &[rows, cols]).map_err(ClusterError::TensorError)
}

fn array1_to_tensor(array: &Array1<f64>) -> ClusterResult<Tensor> {
    let len = array.len();
    let data_f64: Vec<f64> = array.iter().copied().collect();
    let data: Vec<f32> = data_f64.into_iter().map(|x| x as f32).collect();
    Tensor::from_vec(data, &[len]).map_err(ClusterError::TensorError)
}

fn array1_i32_to_tensor(array: &Array1<i32>) -> ClusterResult<Tensor> {
    let len = array.len();
    let data_i32: Vec<i32> = array.iter().copied().collect();
    let data: Vec<f32> = data_i32.into_iter().map(|x| x as f32).collect();
    Tensor::from_vec(data, &[len]).map_err(ClusterError::TensorError)
}
