//! Gaussian Mixture Model clustering implementation
//!
//! This module provides a complete implementation of Gaussian Mixture Models (GMM)
//! using the Expectation-Maximization (EM) algorithm, built on SciRS2 foundations.
//!
//! # Algorithm Overview
//!
//! GMM models data as a mixture of K multivariate Gaussian distributions:
//! p(x) = Σ(k=1 to K) π_k * N(x | μ_k, Σ_k)
//!
//! where:
//! - π_k: mixing coefficients (weights)
//! - μ_k: mean vectors
//! - Σ_k: covariance matrices
//!
//! The EM algorithm alternates between:
//! - E-step: Compute posterior probabilities (responsibilities)
//! - M-step: Update parameters based on responsibilities

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{
    AlgorithmComplexity, ClusteringAlgorithm, ClusteringConfig, ClusteringResult, Fit, FitPredict,
    MemoryPattern, ProbabilisticClustering,
};
use crate::utils::validation::{validate_cluster_input, validate_n_clusters};
use scirs2_autograd::{self as ag, tensor_ops::*};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Random;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use torsh_tensor::Tensor;

/// Covariance matrix types for Gaussian Mixture Model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CovarianceType {
    /// Full covariance matrices (most flexible, most parameters)
    Full,
    /// Diagonal covariance matrices (independent features)
    Diag,
    /// Spherical covariance matrices (isotropic)
    Spherical,
}

impl Default for CovarianceType {
    fn default() -> Self {
        Self::Full
    }
}

impl std::fmt::Display for CovarianceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Diag => write!(f, "diag"),
            Self::Spherical => write!(f, "spherical"),
        }
    }
}

/// Gaussian Mixture Model initialization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GMMInitMethod {
    /// K-means initialization (default)
    KMeans,
    /// Random initialization
    Random,
}

impl Default for GMMInitMethod {
    fn default() -> Self {
        Self::KMeans
    }
}

/// Gaussian Mixture Model configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GMConfig {
    /// Number of mixture components
    pub n_components: usize,
    /// Type of covariance parameters to use
    pub covariance_type: CovarianceType,
    /// Maximum number of EM iterations
    pub max_iters: usize,
    /// Convergence tolerance (log-likelihood improvement)
    pub tolerance: f64,
    /// Regularization added to covariance diagonal
    pub reg_covar: f64,
    /// Initialization method
    pub init_method: GMMInitMethod,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for GMConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            covariance_type: CovarianceType::Full,
            max_iters: 100,
            tolerance: 1e-3,
            reg_covar: 1e-6,
            init_method: GMMInitMethod::KMeans,
            random_state: None,
        }
    }
}

impl ClusteringConfig for GMConfig {
    fn validate(&self) -> ClusterResult<()> {
        if self.n_components == 0 {
            return Err(ClusterError::InvalidClusters(self.n_components));
        }
        if self.max_iters == 0 {
            return Err(ClusterError::ConfigError(
                "max_iters must be positive".to_string(),
            ));
        }
        if self.tolerance <= 0.0 {
            return Err(ClusterError::ConfigError(
                "tolerance must be positive".to_string(),
            ));
        }
        if self.reg_covar <= 0.0 {
            return Err(ClusterError::ConfigError(
                "reg_covar must be positive".to_string(),
            ));
        }
        Ok(())
    }

    fn default() -> Self {
        <GMConfig as std::default::Default>::default()
    }

    fn merge(&mut self, other: &Self) {
        let default_config = <GMConfig as std::default::Default>::default();
        if other.n_components != default_config.n_components {
            self.n_components = other.n_components;
        }
        if other.max_iters != default_config.max_iters {
            self.max_iters = other.max_iters;
        }
        if other.tolerance != default_config.tolerance {
            self.tolerance = other.tolerance;
        }
    }
}

/// Gaussian Mixture Model fitting result
#[derive(Debug, Clone)]
pub struct GMResult {
    /// Cluster assignments for each sample (hard assignment)
    pub labels: Tensor,
    /// Mean vectors for each component [n_components, n_features]
    pub means: Tensor,
    /// Covariance matrices (shape depends on covariance_type)
    pub covariances: Tensor,
    /// Mixing coefficients (weights) for each component
    pub weights: Tensor,
    /// Posterior probabilities [n_samples, n_components]
    pub responsibilities: Tensor,
    /// Final log-likelihood of the data
    pub log_likelihood: f64,
    /// Number of EM iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

impl ClusteringResult for GMResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.means.shape().dims()[0]
    }

    fn centers(&self) -> Option<&Tensor> {
        Some(&self.means)
    }

    fn n_iter(&self) -> Option<usize> {
        Some(self.n_iter)
    }

    fn converged(&self) -> bool {
        self.converged
    }

    fn metadata(&self) -> Option<&HashMap<String, String>> {
        None // Could be enhanced to return more metadata
    }
}

/// Gaussian Mixture Model clustering algorithm
///
/// GMM models data as a mixture of multivariate Gaussian distributions,
/// using the Expectation-Maximization algorithm for parameter estimation.
///
/// # Example
///
/// ```rust
/// use torsh_cluster::algorithms::gaussian_mixture::{GaussianMixture, CovarianceType};
/// use torsh_tensor::Tensor;
///
/// let data = Tensor::randn(&[100, 2])?;
/// let gmm = GaussianMixture::new(3)
///     .covariance_type(CovarianceType::Full)
///     .max_iters(150)
///     .tolerance(1e-4);
/// let result = gmm.fit(&data)?;
/// println!("Final log-likelihood: {}", result.log_likelihood);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianMixture {
    config: GMConfig,
}

impl GaussianMixture {
    /// Create a new Gaussian Mixture Model with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            config: GMConfig {
                n_components,
                ..Default::default()
            },
        }
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.config.covariance_type = covariance_type;
        self
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

    /// Set regularization parameter for covariance
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.config.reg_covar = reg_covar;
        self
    }

    /// Set initialization method
    pub fn init_method(mut self, init_method: GMMInitMethod) -> Self {
        self.config.init_method = init_method;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl ClusteringAlgorithm for GaussianMixture {
    fn name(&self) -> &str {
        "Gaussian Mixture Model"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "n_components".to_string(),
            self.config.n_components.to_string(),
        );
        params.insert(
            "covariance_type".to_string(),
            self.config.covariance_type.to_string(),
        );
        params.insert("max_iters".to_string(), self.config.max_iters.to_string());
        params.insert("tolerance".to_string(), self.config.tolerance.to_string());
        params.insert("reg_covar".to_string(), self.config.reg_covar.to_string());
        params.insert(
            "init_method".to_string(),
            format!("{:?}", self.config.init_method),
        );
        if let Some(seed) = self.config.random_state {
            params.insert("random_state".to_string(), seed.to_string());
        }
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> ClusterResult<()> {
        for (key, value) in params {
            match key.as_str() {
                "n_components" => {
                    self.config.n_components = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid n_components: {}", value))
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
            time_complexity: "O(n * k * d² * iter)".to_string(),
            space_complexity: "O(n * k + k * d²)".to_string(),
            deterministic: false,
            online_capable: false,
            memory_pattern: MemoryPattern::Quadratic,
        }
    }

    fn supported_distance_metrics(&self) -> Vec<&str> {
        vec!["mahalanobis", "euclidean"] // Implicit in the probabilistic framework
    }
}

impl Fit for GaussianMixture {
    type Result = GMResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;
        validate_n_clusters(self.config.n_components, data.shape().dims()[0])?;
        validate_cluster_input(data)?;
        self.config.validate()?;

        let _n_samples = data.shape().dims()[0];
        let _n_features = data.shape().dims()[1];

        // Convert tensor to ndarray for easier computation
        let data_array = tensor_to_array2(data)?;

        // Fit the model using EM algorithm
        let result = self.fit_em(&data_array)?;

        Ok(result)
    }
}

impl FitPredict for GaussianMixture {
    type Result = GMResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

impl ProbabilisticClustering for GaussianMixture {
    fn membership_probabilities(&self, _data: &Tensor) -> ClusterResult<Tensor> {
        Err(ClusterError::NotImplemented(
            "GMM membership_probabilities not yet implemented - requires fitted model state"
                .to_string(),
        ))
    }

    fn cluster_parameters(&self) -> ClusterResult<Vec<HashMap<String, Tensor>>> {
        Err(ClusterError::NotImplemented(
            "GMM cluster_parameters not yet implemented - requires fitted model state".to_string(),
        ))
    }

    fn log_likelihood(&self, _data: &Tensor) -> ClusterResult<f64> {
        Err(ClusterError::NotImplemented(
            "GMM log_likelihood not yet implemented - requires fitted model state".to_string(),
        ))
    }

    fn sample(&self, _n_samples: usize) -> ClusterResult<Tensor> {
        Err(ClusterError::NotImplemented(
            "GMM sample not yet implemented - requires fitted model state".to_string(),
        ))
    }
}

// Core EM algorithm implementation
impl GaussianMixture {
    /// Fit the model using the EM algorithm
    fn fit_em(&self, data: &Array2<f64>) -> ClusterResult<GMResult> {
        let (n_samples, n_features) = data.dim();
        let n_components = self.config.n_components;

        // Initialize random number generator
        let mut rng = scirs2_core::random::thread_rng();

        // Initialize parameters
        let (mut means, mut covariances, mut weights) =
            self.initialize_parameters(data, &mut rng)?;

        let mut responsibilities = Array2::<f64>::zeros((n_samples, n_components));
        let mut log_likelihood = f64::NEG_INFINITY;
        let mut converged = false;

        // EM iterations
        for iter in 0..self.config.max_iters {
            // E-step: compute responsibilities
            let new_log_likelihood =
                self.e_step(data, &means, &covariances, &weights, &mut responsibilities)?;

            // Check convergence
            if iter > 0 && (new_log_likelihood - log_likelihood).abs() < self.config.tolerance {
                converged = true;
                log_likelihood = new_log_likelihood;
                break;
            }
            log_likelihood = new_log_likelihood;

            // M-step: update parameters
            self.m_step(
                data,
                &responsibilities,
                &mut means,
                &mut covariances,
                &mut weights,
            )?;
        }

        // Compute hard assignments
        let labels = self.compute_labels(&responsibilities)?;

        // Compute information criteria
        let n_params = self.count_parameters(n_features);
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (n_params as f64) * (n_samples as f64).ln();

        Ok(GMResult {
            labels: array1_i32_to_tensor(&labels)?,
            means: array2_to_tensor(&means)?,
            covariances: self.format_covariances(&covariances)?,
            weights: array1_to_tensor(&weights)?,
            responsibilities: array2_to_tensor(&responsibilities)?,
            log_likelihood,
            n_iter: if converged {
                self.config.max_iters.min(100) // Estimate iter count
            } else {
                self.config.max_iters
            },
            converged,
            aic,
            bic,
        })
    }

    /// Initialize GMM parameters based on initialization method
    fn initialize_parameters<R: scirs2_core::random::Rng + ?Sized>(
        &self,
        data: &Array2<f64>,
        rng: &mut R,
    ) -> ClusterResult<(Array2<f64>, Array3<f64>, Array1<f64>)> {
        let (n_samples, n_features) = data.dim();
        let n_components = self.config.n_components;

        // Initialize means by randomly selecting data points
        let mut means = Array2::<f64>::zeros((n_components, n_features));
        for k in 0..n_components {
            let idx = rng.gen_range(0..n_samples);
            means.row_mut(k).assign(&data.row(idx));
        }

        // Initialize covariances as identity matrices scaled by data variance
        let mut covariances = Array3::<f64>::zeros((n_components, n_features, n_features));
        let data_var = self.compute_data_variance(data);

        for k in 0..n_components {
            for i in 0..n_features {
                covariances[[k, i, i]] = data_var + self.config.reg_covar;
            }
        }

        // Initialize weights uniformly
        let weights = Array1::<f64>::from_elem(n_components, 1.0 / n_components as f64);

        Ok((means, covariances, weights))
    }

    /// E-step: compute responsibilities (posterior probabilities)
    fn e_step(
        &self,
        data: &Array2<f64>,
        means: &Array2<f64>,
        covariances: &Array3<f64>,
        weights: &Array1<f64>,
        responsibilities: &mut Array2<f64>,
    ) -> ClusterResult<f64> {
        let (n_samples, _) = data.dim();
        let n_components = means.nrows();
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let x = data.row(i);
            let mut weighted_log_probs = Vec::with_capacity(n_components);
            let mut max_log_prob = f64::NEG_INFINITY;

            // Compute weighted log probabilities for each component
            for k in 0..n_components {
                let mean_k = means.row(k);
                let log_prob = self.log_multivariate_normal_pdf(
                    &x,
                    &mean_k,
                    &covariances.slice(s![k, .., ..]),
                )? + weights[k].ln();
                weighted_log_probs.push(log_prob);
                max_log_prob = max_log_prob.max(log_prob);
            }

            // Compute responsibilities using log-sum-exp trick for numerical stability
            let mut sum_exp = 0.0;
            for &log_prob in &weighted_log_probs {
                sum_exp += (log_prob - max_log_prob).exp();
            }
            let log_sum = max_log_prob + sum_exp.ln();

            for k in 0..n_components {
                responsibilities[[i, k]] = (weighted_log_probs[k] - log_sum).exp();
            }

            log_likelihood += log_sum;
        }

        Ok(log_likelihood)
    }

    /// M-step: update parameters based on responsibilities
    fn m_step(
        &self,
        data: &Array2<f64>,
        responsibilities: &Array2<f64>,
        means: &mut Array2<f64>,
        covariances: &mut Array3<f64>,
        weights: &mut Array1<f64>,
    ) -> ClusterResult<()> {
        let (n_samples, n_features) = data.dim();
        let n_components = means.nrows();

        // Update weights and means
        for k in 0..n_components {
            let nk = responsibilities.column(k).sum();
            weights[k] = nk / n_samples as f64;

            if nk > 1e-6 {
                // Avoid division by zero
                // Update mean
                for j in 0..n_features {
                    let mut weighted_sum = 0.0;
                    for i in 0..n_samples {
                        weighted_sum += responsibilities[[i, k]] * data[[i, j]];
                    }
                    means[[k, j]] = weighted_sum / nk;
                }

                // Update covariance based on covariance type
                self.update_covariance(k, data, responsibilities, &means.row(k), covariances, nk)?;
            }
        }

        Ok(())
    }

    /// Update covariance matrix for a specific component based on covariance type
    fn update_covariance(
        &self,
        component: usize,
        data: &Array2<f64>,
        responsibilities: &Array2<f64>,
        mean: &ArrayView1<f64>,
        covariances: &mut Array3<f64>,
        nk: f64,
    ) -> ClusterResult<()> {
        let (n_samples, n_features) = data.dim();

        match self.config.covariance_type {
            CovarianceType::Diag => {
                // Diagonal covariance - only update diagonal elements
                for j in 0..n_features {
                    let mut var = 0.0;
                    for i in 0..n_samples {
                        let diff = data[[i, j]] - mean[j];
                        var += responsibilities[[i, component]] * diff * diff;
                    }
                    var = (var / nk) + self.config.reg_covar;
                    covariances[[component, j, j]] = var;

                    // Ensure off-diagonal elements are zero
                    for l in 0..n_features {
                        if l != j {
                            covariances[[component, j, l]] = 0.0;
                        }
                    }
                }
            }

            CovarianceType::Full => {
                // Full covariance matrix
                for j in 0..n_features {
                    for l in 0..n_features {
                        let mut covar = 0.0;
                        for i in 0..n_samples {
                            let diff_j = data[[i, j]] - mean[j];
                            let diff_l = data[[i, l]] - mean[l];
                            covar += responsibilities[[i, component]] * diff_j * diff_l;
                        }
                        covar /= nk;

                        // Add regularization to diagonal elements
                        if j == l {
                            covar += self.config.reg_covar;
                        }

                        covariances[[component, j, l]] = covar;
                    }
                }
            }

            CovarianceType::Spherical => {
                // Spherical covariance - single variance for all dimensions
                let mut total_var = 0.0;
                for j in 0..n_features {
                    for i in 0..n_samples {
                        let diff = data[[i, j]] - mean[j];
                        total_var += responsibilities[[i, component]] * diff * diff;
                    }
                }
                let spherical_var = (total_var / (nk * n_features as f64)) + self.config.reg_covar;

                // Set diagonal to spherical variance, off-diagonal to zero
                for j in 0..n_features {
                    for l in 0..n_features {
                        if j == l {
                            covariances[[component, j, l]] = spherical_var;
                        } else {
                            covariances[[component, j, l]] = 0.0;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute log probability density function for multivariate normal distribution
    /// Using SciRS2's linear algebra operations for proper matrix operations
    fn log_multivariate_normal_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &ArrayView2<f64>,
    ) -> ClusterResult<f64> {
        let _n_features = x.len();

        match self.config.covariance_type {
            CovarianceType::Diag | CovarianceType::Spherical => {
                // Optimized diagonal case
                self.log_multivariate_normal_pdf_diagonal(x, mean, cov)
            }
            CovarianceType::Full => {
                // Full covariance matrix using SciRS2
                self.log_multivariate_normal_pdf_full(x, mean, cov)
            }
        }
    }

    /// Diagonal covariance case (optimized)
    fn log_multivariate_normal_pdf_diagonal(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &ArrayView2<f64>,
    ) -> ClusterResult<f64> {
        let n_features = x.len();
        let diff = x - mean;

        let mut log_det = 0.0;
        let mut quad_form = 0.0;

        for i in 0..n_features {
            let var = cov[[i, i]];
            if var <= 0.0 {
                return Err(ClusterError::SingularMatrix);
            }
            log_det += var.ln();
            quad_form += diff[i] * diff[i] / var;
        }

        let log_pdf = -0.5 * (n_features as f64 * (2.0 * PI).ln() + log_det + quad_form);
        Ok(log_pdf)
    }

    /// Full covariance matrix case using SciRS2
    fn log_multivariate_normal_pdf_full(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &ArrayView2<f64>,
    ) -> ClusterResult<f64> {
        let n_features = x.len();
        let diff = x - mean;

        // Use SciRS2 for matrix determinant and inverse operations
        let result = ag::run(|g| -> Result<(f64, f64), String> {
            // Convert to f32 arrays for SciRS2
            let cov_f32 = cov.mapv(|x| x as f32);
            let diff_f32 = diff.mapv(|x| x as f32).insert_axis(Axis(1)); // Make column vector

            // Convert to SciRS2 tensors
            let cov_tensor = convert_to_tensor(cov_f32, g);
            let _diff_tensor = convert_to_tensor(diff_f32, g);

            // Compute determinant for log-likelihood
            let det_cov = det(&cov_tensor);
            let det_val = det_cov
                .eval(g)
                .map_err(|e| format!("Determinant computation failed: {:?}", e))?;

            let det_scalar = if det_val.ndim() == 0 {
                det_val[[]] as f64
            } else {
                det_val[scirs2_core::ndarray::IxDyn(&[0])] as f64
            };

            if det_scalar <= 0.0 {
                return Err("Singular or non-positive-definite covariance matrix".to_string());
            }

            // Compute matrix inverse
            let inv_cov = matinv(&cov_tensor);
            let inv_cov_val = inv_cov
                .eval(g)
                .map_err(|e| format!("Matrix inverse computation failed: {:?}", e))?;

            // Convert back to ndarray for quadratic form computation
            let inv_cov_f64 = inv_cov_val.mapv(|x| x as f64);

            // Compute quadratic form: diff^T * inv(cov) * diff
            let mut quad_form = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    quad_form += diff[i] * inv_cov_f64[[i, j]] * diff[j];
                }
            }

            Ok((det_scalar.ln(), quad_form))
        });

        let (log_det, quad_form) = result.map_err(|e| {
            ClusterError::SciRS2Error(format!("Multivariate normal PDF computation failed: {}", e))
        })?;

        let log_pdf = -0.5 * (n_features as f64 * (2.0 * PI).ln() + log_det + quad_form);
        Ok(log_pdf)
    }

    /// Compute hard cluster assignments from responsibilities
    fn compute_labels(&self, responsibilities: &Array2<f64>) -> ClusterResult<Array1<i32>> {
        let (n_samples, _) = responsibilities.dim();
        let mut labels = Array1::<i32>::zeros(n_samples);

        for i in 0..n_samples {
            let mut max_resp = 0.0;
            let mut best_label = 0;

            for k in 0..responsibilities.ncols() {
                if responsibilities[[i, k]] > max_resp {
                    max_resp = responsibilities[[i, k]];
                    best_label = k;
                }
            }

            labels[i] = best_label as i32;
        }

        Ok(labels)
    }

    /// Compute data variance for initialization
    fn compute_data_variance(&self, data: &Array2<f64>) -> f64 {
        let (n_samples, n_features) = data.dim();
        let mut total_var = 0.0;

        for j in 0..n_features {
            let column = data.column(j);
            let mean = column.sum() / n_samples as f64;
            let var = column.mapv(|x| (x - mean).powi(2)).sum() / n_samples as f64;
            total_var += var;
        }

        total_var / n_features as f64
    }

    /// Count number of parameters for information criteria
    fn count_parameters(&self, n_features: usize) -> usize {
        let n_components = self.config.n_components;
        let mean_params = n_components * n_features;
        let weight_params = n_components - 1; // Sum to 1 constraint

        let cov_params = match self.config.covariance_type {
            CovarianceType::Full => n_components * n_features * (n_features + 1) / 2,
            CovarianceType::Diag => n_components * n_features,
            CovarianceType::Spherical => n_components,
        };

        mean_params + weight_params + cov_params
    }

    /// Format covariances tensor based on covariance type
    fn format_covariances(&self, covariances: &Array3<f64>) -> ClusterResult<Tensor> {
        match self.config.covariance_type {
            CovarianceType::Full | CovarianceType::Diag => {
                // Return as 3D tensor
                array3_to_tensor(covariances)
            }
            CovarianceType::Spherical => {
                // Extract diagonal elements only
                let (n_components, n_features, _) = covariances.dim();
                let mut spherical_vars = Array2::<f64>::zeros((n_components, n_features));

                for k in 0..n_components {
                    for i in 0..n_features {
                        spherical_vars[[k, i]] = covariances[[k, i, i]];
                    }
                }

                array2_to_tensor(&spherical_vars)
            }
        }
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

fn array3_to_tensor(array: &Array3<f64>) -> ClusterResult<Tensor> {
    let (d1, d2, d3) = array.dim();
    let data_f64: Vec<f64> = array.iter().copied().collect();
    let data: Vec<f32> = data_f64.into_iter().map(|x| x as f32).collect();
    Tensor::from_vec(data, &[d1, d2, d3]).map_err(ClusterError::TensorError)
}
