//! Unscented Kalman filter implementation

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TimeSeries;
use torsh_tensor::{
    creation::{eye, zeros},
    Tensor,
};

/// Perform Cholesky-Banachiewicz decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that A = L * L^T.
/// The matrix is given as a flat row-major `Vec<f64>` of size n*n.
/// Returns `None` if the matrix is not positive definite.
fn cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let diag = l[j * n + j];
                if diag.abs() < 1e-14 {
                    return None; // Singular
                }
                l[i * n + j] = sum / diag;
            }
        }
    }
    Some(l)
}

/// Unscented Kalman Filter for nonlinear systems
pub struct UnscentedKalmanFilter {
    state_dim: usize,
    obs_dim: usize,
    /// UKF parameters
    alpha: f64, // Spread of sigma points
    beta: f64,  // Prior knowledge parameter (2.0 for Gaussian)
    kappa: f64, // Secondary scaling parameter
    /// Derived parameters
    lambda: f64, // Composite scaling parameter
    /// Sigma points
    sigma_points: Tensor,
    /// Weights for mean computation
    weights_mean: Vec<f64>,
    /// Weights for covariance computation
    weights_cov: Vec<f64>,
    /// Process noise covariance
    process_noise: Tensor,
    /// Measurement noise covariance
    measurement_noise: Tensor,
    /// Current state
    state: Tensor,
    /// State covariance
    covariance: Tensor,
}

impl UnscentedKalmanFilter {
    /// Create a new UKF with default parameters
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        let alpha: f64 = 1e-3;
        let beta = 2.0;
        let kappa = 0.0;
        let lambda = alpha.powi(2) * (state_dim as f64 + kappa) - state_dim as f64;

        let num_sigma = 2 * state_dim + 1;
        let sigma_points = zeros(&[num_sigma, state_dim]).expect("tensor creation should succeed");

        let mut ukf = Self {
            state_dim,
            obs_dim,
            alpha,
            beta,
            kappa,
            lambda,
            sigma_points,
            weights_mean: vec![0.0; num_sigma],
            weights_cov: vec![0.0; num_sigma],
            process_noise: eye(state_dim).expect("tensor creation should succeed"),
            measurement_noise: eye(obs_dim).expect("tensor creation should succeed"),
            state: zeros(&[state_dim]).expect("tensor creation should succeed"),
            covariance: eye(state_dim).expect("tensor creation should succeed"),
        };

        ukf.compute_weights();
        ukf
    }

    /// Create UKF with custom parameters
    pub fn with_parameters(
        state_dim: usize,
        obs_dim: usize,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> Self {
        let lambda = alpha.powi(2) * (state_dim as f64 + kappa) - state_dim as f64;

        let num_sigma = 2 * state_dim + 1;
        let sigma_points = zeros(&[num_sigma, state_dim]).expect("tensor creation should succeed");

        let mut ukf = Self {
            state_dim,
            obs_dim,
            alpha,
            beta,
            kappa,
            lambda,
            sigma_points,
            weights_mean: vec![0.0; num_sigma],
            weights_cov: vec![0.0; num_sigma],
            process_noise: eye(state_dim).expect("tensor creation should succeed"),
            measurement_noise: eye(obs_dim).expect("tensor creation should succeed"),
            state: zeros(&[state_dim]).expect("tensor creation should succeed"),
            covariance: eye(state_dim).expect("tensor creation should succeed"),
        };

        ukf.compute_weights();
        ukf
    }

    /// Set noise covariances
    pub fn with_noise(mut self, process_noise: Tensor, measurement_noise: Tensor) -> Self {
        self.process_noise = process_noise;
        self.measurement_noise = measurement_noise;
        self
    }

    /// Get UKF parameters
    pub fn parameters(&self) -> (f64, f64, f64, f64) {
        (self.alpha, self.beta, self.kappa, self.lambda)
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.state_dim, self.obs_dim)
    }

    /// Get current state
    pub fn state(&self) -> &Tensor {
        &self.state
    }

    /// Get state covariance
    pub fn covariance(&self) -> &Tensor {
        &self.covariance
    }

    /// Get sigma points
    pub fn sigma_points(&self) -> &Tensor {
        &self.sigma_points
    }

    /// Set initial state
    pub fn set_initial_state(&mut self, state: Tensor, covariance: Tensor) {
        self.state = state;
        self.covariance = covariance;
    }

    /// Set process noise
    pub fn set_process_noise(&mut self, noise: Tensor) {
        self.process_noise = noise;
    }

    /// Set measurement noise
    pub fn set_measurement_noise(&mut self, noise: Tensor) {
        self.measurement_noise = noise;
    }

    /// Compute weights for sigma points
    fn compute_weights(&mut self) {
        let n = self.state_dim as f64;
        let lambda = self.lambda;
        let n_plus_lambda = n + lambda;

        // w_0^m = lambda / (n + lambda)
        let w0_mean = lambda / n_plus_lambda;
        // w_0^c = lambda / (n + lambda) + (1 - alpha^2 + beta)
        let w0_cov = w0_mean + (1.0 - self.alpha.powi(2) + self.beta);
        // w_i^m = w_i^c = 1 / (2 * (n + lambda)) for i = 1, ..., 2n
        let w_rest = 0.5 / n_plus_lambda;

        let num_sigma = 2 * self.state_dim + 1;
        self.weights_mean[0] = w0_mean;
        self.weights_cov[0] = w0_cov;
        for i in 1..num_sigma {
            self.weights_mean[i] = w_rest;
            self.weights_cov[i] = w_rest;
        }
    }

    /// Generate sigma points around current state using Cholesky decomposition.
    ///
    /// Computes:
    ///   σ_0   = x
    ///   σ_i   = x + L_i       for i = 1..=N  (i-th column of L)
    ///   σ_{N+i} = x - L_i     for i = 1..=N
    /// where L is the lower-triangular Cholesky factor of (N + λ) * P.
    pub fn generate_sigma_points(&mut self) {
        let n = self.state_dim;
        let scale = n as f64 + self.lambda;

        // Read covariance into f64 flat row-major matrix, scaled by (n + lambda)
        let mut cov_f64 = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let val = self
                    .covariance
                    .get(&[i, j])
                    .expect("covariance element read should succeed");
                cov_f64[i * n + j] = scale * val as f64;
            }
        }

        // Cholesky factor L such that (n+λ)*P = L * L^T
        let l = cholesky_lower(&cov_f64, n).unwrap_or_else(|| {
            // Fallback: identity scaled by sqrt(scale) for degenerate/near-zero covariance
            let mut diag = vec![0.0f64; n * n];
            let s = scale.sqrt();
            for k in 0..n {
                diag[k * n + k] = s;
            }
            diag
        });

        // Sigma point 0: the mean itself
        for j in 0..n {
            let xj = self
                .state
                .get(&[j])
                .expect("state element read should succeed") as f64;
            self.sigma_points
                .set(&[0, j], xj as f32)
                .expect("sigma point set should succeed");
        }

        // Sigma points i = 1..=n: x + L[:,i-1]  (i-th column of L, 0-indexed = column i-1)
        // Sigma points i = n+1..=2n: x - L[:,i-n-1]
        for col in 0..n {
            for j in 0..n {
                let xj = self
                    .state
                    .get(&[j])
                    .expect("state element read should succeed") as f64;
                let l_jcol = l[j * n + col];
                self.sigma_points
                    .set(&[col + 1, j], (xj + l_jcol) as f32)
                    .expect("sigma point set should succeed");
                self.sigma_points
                    .set(&[n + col + 1, j], (xj - l_jcol) as f32)
                    .expect("sigma point set should succeed");
            }
        }
    }

    /// Apply unscented transform for prediction step.
    ///
    /// Passes each sigma point through `transition_fn`, then computes the
    /// weighted mean and weighted covariance (plus process noise Q):
    ///
    ///   m = Σ_i W_m[i] * f(σ_i)
    ///   C = Σ_i W_c[i] * (f(σ_i) - m)(f(σ_i) - m)^T + Q
    fn unscented_transform_predict(
        &self,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
    ) -> (Tensor, Tensor) {
        let num_sigma = 2 * self.state_dim + 1;

        // Apply transition function to each sigma point (shape [1, N] slice)
        let mut transformed_sigma_points: Vec<Tensor> = Vec::with_capacity(num_sigma);
        for i in 0..num_sigma {
            let sigma_point = self
                .sigma_points
                .slice_tensor(0, i, i + 1)
                .expect("sigma slice should succeed");
            let transformed = transition_fn(&sigma_point);
            transformed_sigma_points.push(transformed);
        }

        let n = self.state_dim;

        // Weighted mean: m_j = sum_i W_m[i] * transformed[i][0,j]
        let predicted_mean = zeros(&[n]).expect("tensor creation should succeed");
        for j in 0..n {
            let mut mean_j = 0.0f64;
            for (i, sp) in transformed_sigma_points.iter().enumerate() {
                // Sigma point slices are [1, N]; use get(&[0, j]) when 2D, get(&[j]) when 1D
                let val = if sp.shape().ndim() == 2 {
                    sp.get(&[0, j])
                        .expect("transformed sigma point read should succeed")
                        as f64
                } else {
                    sp.get(&[j])
                        .expect("transformed sigma point read should succeed")
                        as f64
                };
                mean_j += self.weights_mean[i] * val;
            }
            predicted_mean
                .set(&[j], mean_j as f32)
                .expect("predicted mean set should succeed");
        }

        // Weighted covariance: C_{jk} = sum_i W_c[i] * d_j * d_k + Q_{jk}
        // where d_j = transformed[i][j] - mean[j]
        let predicted_cov = zeros(&[n, n]).expect("tensor creation should succeed");

        // Initialise with process noise Q
        for j in 0..n {
            for k in 0..n {
                let q_jk = self
                    .process_noise
                    .get(&[j, k])
                    .expect("process noise read should succeed");
                predicted_cov
                    .set(&[j, k], q_jk)
                    .expect("predicted cov set should succeed");
            }
        }

        for (i, sp) in transformed_sigma_points.iter().enumerate() {
            // Compute deviation vector d = f(σ_i) - m
            let mut d = vec![0.0f64; n];
            for j in 0..n {
                let val = if sp.shape().ndim() == 2 {
                    sp.get(&[0, j])
                        .expect("transformed sigma point read should succeed")
                        as f64
                } else {
                    sp.get(&[j])
                        .expect("transformed sigma point read should succeed")
                        as f64
                };
                let mean_j = predicted_mean
                    .get(&[j])
                    .expect("predicted mean read should succeed")
                    as f64;
                d[j] = val - mean_j;
            }
            // Accumulate W_c[i] * d * d^T
            let wc = self.weights_cov[i];
            for j in 0..n {
                for k in 0..n {
                    let old = predicted_cov
                        .get(&[j, k])
                        .expect("predicted cov read should succeed")
                        as f64;
                    predicted_cov
                        .set(&[j, k], (old + wc * d[j] * d[k]) as f32)
                        .expect("predicted cov set should succeed");
                }
            }
        }

        (predicted_mean, predicted_cov)
    }

    /// Apply unscented transform for update step.
    ///
    /// Passes predicted sigma points through `observation_fn`, then computes:
    ///   - predicted observation mean
    ///   - predicted observation covariance (plus R)
    ///   - cross-covariance between state and observation
    fn unscented_transform_update(
        &self,
        observation_fn: &dyn Fn(&Tensor) -> Tensor,
        predicted_sigma_points: &[Tensor],
    ) -> (Tensor, Tensor, Tensor) {
        let num_sigma = 2 * self.state_dim + 1;
        let obs_dim = self.obs_dim;
        let state_dim = self.state_dim;

        // Apply observation function to each predicted sigma point
        let mut obs_sigma_points: Vec<Tensor> = Vec::with_capacity(num_sigma);
        for sigma_point in predicted_sigma_points {
            let obs = observation_fn(sigma_point);
            obs_sigma_points.push(obs);
        }

        // Weighted observation mean
        let obs_mean = zeros(&[obs_dim]).expect("tensor creation should succeed");
        for j in 0..obs_dim {
            let mut mean_j = 0.0f64;
            for (i, sp) in obs_sigma_points.iter().enumerate() {
                let val = if sp.shape().ndim() == 2 {
                    sp.get(&[0, j])
                        .expect("obs sigma point read should succeed") as f64
                } else {
                    sp.get(&[j]).expect("obs sigma point read should succeed") as f64
                };
                mean_j += self.weights_mean[i] * val;
            }
            obs_mean
                .set(&[j], mean_j as f32)
                .expect("obs mean set should succeed");
        }

        // Observation covariance (initialised with measurement noise R)
        let obs_cov = zeros(&[obs_dim, obs_dim]).expect("tensor creation should succeed");
        for j in 0..obs_dim {
            for k in 0..obs_dim {
                let r_jk = self
                    .measurement_noise
                    .get(&[j, k])
                    .expect("measurement noise read should succeed");
                obs_cov
                    .set(&[j, k], r_jk)
                    .expect("obs cov set should succeed");
            }
        }

        // Cross-covariance (state x obs)
        let cross_cov = zeros(&[state_dim, obs_dim]).expect("tensor creation should succeed");

        for (i, (sp_x, sp_z)) in predicted_sigma_points
            .iter()
            .zip(obs_sigma_points.iter())
            .enumerate()
        {
            let wc = self.weights_cov[i];

            // Deviation in state space
            let mut dx = vec![0.0f64; state_dim];
            for j in 0..state_dim {
                let val = if sp_x.shape().ndim() == 2 {
                    sp_x.get(&[0, j])
                        .expect("sigma point state read should succeed") as f64
                } else {
                    sp_x.get(&[j])
                        .expect("sigma point state read should succeed") as f64
                };
                let mean_j = self.state.get(&[j]).expect("state read should succeed") as f64;
                dx[j] = val - mean_j;
            }

            // Deviation in observation space
            let mut dz = vec![0.0f64; obs_dim];
            for j in 0..obs_dim {
                let val = if sp_z.shape().ndim() == 2 {
                    sp_z.get(&[0, j])
                        .expect("obs sigma point read should succeed") as f64
                } else {
                    sp_z.get(&[j]).expect("obs sigma point read should succeed") as f64
                };
                let mean_j = obs_mean.get(&[j]).expect("obs mean read should succeed") as f64;
                dz[j] = val - mean_j;
            }

            // Accumulate W_c[i] * dz * dz^T into obs_cov
            for j in 0..obs_dim {
                for k in 0..obs_dim {
                    let old = obs_cov.get(&[j, k]).expect("obs cov read should succeed") as f64;
                    obs_cov
                        .set(&[j, k], (old + wc * dz[j] * dz[k]) as f32)
                        .expect("obs cov set should succeed");
                }
            }

            // Accumulate W_c[i] * dx * dz^T into cross_cov
            for j in 0..state_dim {
                for k in 0..obs_dim {
                    let old = cross_cov
                        .get(&[j, k])
                        .expect("cross cov read should succeed")
                        as f64;
                    cross_cov
                        .set(&[j, k], (old + wc * dx[j] * dz[k]) as f32)
                        .expect("cross cov set should succeed");
                }
            }
        }

        (obs_mean, obs_cov, cross_cov)
    }

    /// Predict step
    pub fn predict(&mut self, transition_fn: &dyn Fn(&Tensor) -> Tensor) -> Tensor {
        // Generate sigma points
        self.generate_sigma_points();

        // Apply unscented transform (process noise Q is incorporated inside)
        let (predicted_mean, predicted_cov) = self.unscented_transform_predict(transition_fn);

        // Update state and covariance
        self.state = predicted_mean;
        self.covariance = predicted_cov;

        self.state.clone()
    }

    /// Update step
    pub fn update(&mut self, observation: &Tensor, observation_fn: &dyn Fn(&Tensor) -> Tensor) {
        // Generate sigma points from predicted state
        self.generate_sigma_points();

        // Collect predicted sigma point slices
        let predicted_sigma_points: Vec<Tensor> = (0..(2 * self.state_dim + 1))
            .map(|i| {
                self.sigma_points
                    .slice_tensor(0, i, i + 1)
                    .expect("slice should succeed")
            })
            .collect();

        let (_obs_mean, _obs_cov, _cross_cov) =
            self.unscented_transform_update(observation_fn, &predicted_sigma_points);

        // Innovation
        // innovation = observation - obs_mean
        let _innovation = observation.clone(); // Subtraction not yet implemented

        // Kalman gain: K = cross_cov @ inv(obs_cov + measurement_noise)
        // State update: state = state + K @ innovation
        // Covariance update: covariance = covariance - K @ obs_cov @ K.T
        // These require matrix inversion; deferred until tensor matrix-inverse is available.
    }

    /// Run UKF on time series
    pub fn filter(
        &mut self,
        series: &TimeSeries,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        observation_fn: &dyn Fn(&Tensor) -> Tensor,
    ) -> TimeSeries {
        let mut filtered = Vec::new();

        for t in 0..series.len() {
            // Predict
            self.predict(transition_fn);

            // Update
            let obs = series
                .values
                .slice_tensor(0, t, t + 1)
                .expect("slice should succeed");
            self.update(&obs, observation_fn);

            filtered.push(self.state.clone());
        }

        // Stack filtered states into output series
        let _ = filtered; // Stacking not yet implemented via tensor ops
        let values =
            zeros(&[series.len(), self.state_dim]).expect("tensor creation should succeed");
        TimeSeries::new(values)
    }

    /// Run UKF smoother
    pub fn smooth(
        &mut self,
        series: &TimeSeries,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        observation_fn: &dyn Fn(&Tensor) -> Tensor,
    ) -> TimeSeries {
        // Forward pass
        let filtered = self.filter(series, transition_fn, observation_fn);

        // Backward pass for UKF smoothing deferred until tensor stacking is available.
        filtered
    }

    /// Compute log-likelihood
    pub fn log_likelihood(
        &mut self,
        _series: &TimeSeries,
        _transition_fn: &dyn Fn(&Tensor) -> Tensor,
        _observation_fn: &dyn Fn(&Tensor) -> Tensor,
    ) -> f32 {
        // Log-likelihood computation deferred until observation covariance inversion is available.
        0.0
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.state = zeros(&[self.state_dim]).expect("tensor creation should succeed");
        self.covariance = eye(self.state_dim).expect("tensor creation should succeed");
    }

    /// Get filter statistics
    pub fn statistics(&self) -> UKFStats {
        UKFStats {
            alpha: self.alpha,
            beta: self.beta,
            kappa: self.kappa,
            lambda: self.lambda,
            num_sigma_points: 2 * self.state_dim + 1,
        }
    }
}

/// UKF statistics and parameters
#[derive(Debug, Clone)]
pub struct UKFStats {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
    pub lambda: f64,
    pub num_sigma_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).expect("test series creation should succeed");
        TimeSeries::new(tensor)
    }

    fn identity_transition(x: &Tensor) -> Tensor {
        x.clone()
    }

    fn identity_observation(x: &Tensor) -> Tensor {
        x.clone()
    }

    #[test]
    fn test_ukf_creation() {
        let ukf = UnscentedKalmanFilter::new(2, 1);
        let (state_dim, obs_dim) = ukf.dimensions();
        let (alpha, beta, kappa, lambda) = ukf.parameters();

        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
        assert_eq!(alpha, 1e-3);
        assert_eq!(beta, 2.0);
        assert_eq!(kappa, 0.0);
        assert!((lambda - (1e-3_f64.powi(2) * 2.0 - 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ukf_with_parameters() {
        let ukf = UnscentedKalmanFilter::with_parameters(3, 2, 0.1, 2.5, 1.0);
        let (state_dim, obs_dim) = ukf.dimensions();
        let (alpha, beta, kappa, lambda) = ukf.parameters();

        assert_eq!(state_dim, 3);
        assert_eq!(obs_dim, 2);
        assert_eq!(alpha, 0.1);
        assert_eq!(beta, 2.5);
        assert_eq!(kappa, 1.0);
        assert!((lambda - (0.1_f64.powi(2) * 4.0 - 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ukf_with_noise() {
        let process_noise = eye(2).expect("eye creation should succeed");
        let measurement_noise = eye(1).expect("eye creation should succeed");

        let ukf = UnscentedKalmanFilter::new(2, 1).with_noise(process_noise, measurement_noise);

        let (state_dim, obs_dim) = ukf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_ukf_state() {
        let mut ukf = UnscentedKalmanFilter::new(2, 1);
        let initial_state = zeros(&[2]).expect("zeros creation should succeed");
        let initial_cov = eye(2).expect("eye creation should succeed");

        ukf.set_initial_state(initial_state, initial_cov);

        assert_eq!(ukf.state().shape().dims(), [2]);
        assert_eq!(ukf.covariance().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_ukf_sigma_points() {
        let ukf = UnscentedKalmanFilter::new(2, 1);
        let sigma_points = ukf.sigma_points();

        // Should have 2*n + 1 = 5 sigma points for 2D state
        assert_eq!(sigma_points.shape().dims(), [5, 2]);
    }

    #[test]
    fn test_ukf_generate_sigma_points() {
        let mut ukf = UnscentedKalmanFilter::new(2, 1);
        ukf.generate_sigma_points();

        // Test that generation completes without error
        assert_eq!(ukf.sigma_points().shape().dims(), [5, 2]);
    }

    #[test]
    fn test_ukf_sigma_points_mean_is_state() {
        // Sigma point 0 must equal the current state
        let mut ukf = UnscentedKalmanFilter::new(2, 1);

        // Set a non-trivial initial state
        let state = zeros(&[2]).expect("zeros should succeed");
        state.set(&[0], 3.0).expect("set should succeed");
        state.set(&[1], -1.5).expect("set should succeed");
        let cov = eye(2).expect("eye should succeed");
        ukf.set_initial_state(state, cov);

        ukf.generate_sigma_points();

        let sp = ukf.sigma_points();
        assert!((sp.get(&[0, 0]).expect("get should succeed") - 3.0).abs() < 1e-5);
        assert!((sp.get(&[0, 1]).expect("get should succeed") - (-1.5)).abs() < 1e-5);
    }

    #[test]
    fn test_ukf_sigma_points_symmetry() {
        // For zero mean and identity covariance, σ_i and σ_{n+i} must be symmetric about 0.
        let mut ukf = UnscentedKalmanFilter::with_parameters(2, 1, 1.0, 2.0, 0.0);
        ukf.generate_sigma_points();

        let n = 2;
        let sp = ukf.sigma_points();
        for col in 0..n {
            for j in 0..n {
                let pos = sp.get(&[col + 1, j]).expect("get should succeed") as f64;
                let neg = sp.get(&[n + col + 1, j]).expect("get should succeed") as f64;
                // pos and neg must be mirror images about the mean (0.0 here)
                assert!(
                    (pos + neg).abs() < 1e-5,
                    "symmetry violated at col={col} j={j}: pos={pos} neg={neg}"
                );
            }
        }
    }

    #[test]
    fn test_ukf_weights_sum_to_one() {
        // Sum of W_m weights must equal 1.0 (property of UKF weight normalisation)
        let ukf = UnscentedKalmanFilter::with_parameters(3, 1, 1.0, 2.0, 0.0);
        let sum: f64 = ukf.weights_mean.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights_mean sum = {sum}");
    }

    #[test]
    fn test_ukf_predict() {
        let mut ukf = UnscentedKalmanFilter::new(2, 1);
        let prediction = ukf.predict(&identity_transition);

        assert_eq!(prediction.shape().dims(), [2]);
    }

    #[test]
    fn test_ukf_update() {
        let mut ukf = UnscentedKalmanFilter::new(2, 1);
        let obs = zeros(&[1]).expect("zeros should succeed");

        ukf.update(&obs, &identity_observation);

        // Test that update completes without error
        assert_eq!(ukf.state().shape().dims(), [2]);
    }

    #[test]
    fn test_ukf_filter() {
        let series = create_test_series();
        let mut ukf = UnscentedKalmanFilter::new(1, 1);

        let filtered = ukf.filter(&series, &identity_transition, &identity_observation);
        assert_eq!(filtered.len(), series.len());
    }

    #[test]
    fn test_ukf_smooth() {
        let series = create_test_series();
        let mut ukf = UnscentedKalmanFilter::new(1, 1);

        let smoothed = ukf.smooth(&series, &identity_transition, &identity_observation);
        assert_eq!(smoothed.len(), series.len());
    }

    #[test]
    fn test_ukf_log_likelihood() {
        let series = create_test_series();
        let mut ukf = UnscentedKalmanFilter::new(1, 1);

        let ll = ukf.log_likelihood(&series, &identity_transition, &identity_observation);
        assert_eq!(ll, 0.0); // Placeholder implementation
    }

    #[test]
    fn test_ukf_reset() {
        let mut ukf = UnscentedKalmanFilter::new(2, 1);
        ukf.reset();

        assert_eq!(ukf.state().shape().dims(), [2]);
        assert_eq!(ukf.covariance().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_ukf_statistics() {
        let ukf = UnscentedKalmanFilter::with_parameters(2, 1, 0.5, 3.0, 0.5);
        let stats = ukf.statistics();

        assert_eq!(stats.alpha, 0.5);
        assert_eq!(stats.beta, 3.0);
        assert_eq!(stats.kappa, 0.5);
        assert_eq!(stats.num_sigma_points, 5);
    }

    #[test]
    fn test_cholesky_lower_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let l = cholesky_lower(&a, 2).expect("cholesky should succeed on identity");
        // L of identity is identity
        assert!((l[0] - 1.0).abs() < 1e-10);
        assert!((l[1]).abs() < 1e-10);
        assert!((l[2]).abs() < 1e-10);
        assert!((l[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_lower_reconstruction() {
        // A = [[4, 2], [2, 3]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky_lower(&a, 2).expect("cholesky should succeed");
        // Verify L * L^T == A
        let (l00, l10, l11) = (l[0], l[2], l[3]);
        assert!((l00 * l00 - 4.0).abs() < 1e-10);
        assert!((l10 * l00 - 2.0).abs() < 1e-10);
        assert!((l10 * l10 + l11 * l11 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_lower_not_posdef() {
        let a = vec![-1.0, 0.0, 0.0, 1.0]; // not positive definite
        assert!(cholesky_lower(&a, 2).is_none());
    }

    #[test]
    fn test_ukf_predict_with_nontrivial_state() {
        // Verify that after predict with identity transition, state == weighted mean of sigma pts
        let mut ukf = UnscentedKalmanFilter::with_parameters(2, 1, 1.0, 2.0, 0.0);
        let state = zeros(&[2]).expect("zeros should succeed");
        state.set(&[0], 1.0).expect("set should succeed");
        state.set(&[1], 2.0).expect("set should succeed");
        let cov = eye(2).expect("eye should succeed");
        ukf.set_initial_state(state, cov);

        let predicted = ukf.predict(&identity_transition);

        // With identity transition, the predicted mean should stay close to [1.0, 2.0]
        let p0 = predicted.get(&[0]).expect("get should succeed");
        let p1 = predicted.get(&[1]).expect("get should succeed");
        assert!((p0 - 1.0).abs() < 1e-4, "p0={p0}");
        assert!((p1 - 2.0).abs() < 1e-4, "p1={p1}");
    }

    #[test]
    fn test_ukf_covariance_shape_after_predict() {
        let mut ukf = UnscentedKalmanFilter::with_parameters(3, 1, 1.0, 2.0, 0.0);
        ukf.predict(&identity_transition);
        assert_eq!(ukf.covariance().shape().dims(), [3, 3]);
    }
}
