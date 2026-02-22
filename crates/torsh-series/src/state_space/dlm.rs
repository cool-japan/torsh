//! Dynamic Linear Models (DLM)
//!
//! DLMs provide a flexible framework for state-space modeling with time-varying parameters.
//! They generalize Kalman filters to handle discount factors, polynomial trends, seasonal
//! components, and regression effects.
//!
//! # References
//! - West, M., & Harrison, J. (1997). Bayesian forecasting and dynamic models. Springer.
//! - Petris, G., Petrone, S., & Campagnoli, P. (2009). Dynamic linear models with R. Springer.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

/// Dynamic Linear Model
///
/// The DLM is specified by the following equations:
/// - Observation equation: y_t = F_t' * θ_t + ν_t, where ν_t ~ N(0, V_t)
/// - State equation: θ_t = G_t * θ_{t-1} + ω_t, where ω_t ~ N(0, W_t)
///
/// where:
/// - y_t: observation at time t
/// - θ_t: state vector at time t
/// - F_t: observation matrix (design matrix)
/// - G_t: evolution matrix (system matrix)
/// - V_t: observation variance
/// - W_t: evolution covariance matrix
#[derive(Debug, Clone)]
pub struct DynamicLinearModel {
    /// State dimension
    pub state_dim: usize,
    /// Observation dimension
    pub obs_dim: usize,
    /// Current state mean (m_t)
    pub state_mean: Array1<f64>,
    /// Current state covariance (C_t)
    pub state_cov: Array2<f64>,
    /// Evolution matrix G_t (default: identity for random walk)
    pub evolution_matrix: Array2<f64>,
    /// Observation matrix F_t
    pub observation_matrix: Array2<f64>,
    /// Observation variance V_t
    pub observation_variance: f64,
    /// Evolution covariance W_t
    pub evolution_cov: Array2<f64>,
    /// Discount factor (0 < δ < 1) for adaptive estimation
    pub discount_factor: Option<f64>,
    /// Time index
    pub time_step: usize,
}

impl DynamicLinearModel {
    /// Create a new DLM with specified dimensions
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of the state vector
    /// * `obs_dim` - Dimension of the observations
    ///
    /// # Example
    /// ```
    /// use torsh_series::state_space::DynamicLinearModel;
    ///
    /// let dlm = DynamicLinearModel::new(2, 1);
    /// ```
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        // Create observation matrix with proper shape
        let mut observation_matrix = Array2::zeros((obs_dim, state_dim));
        for i in 0..obs_dim.min(state_dim) {
            observation_matrix[[i, i]] = 1.0;
        }

        Self {
            state_dim,
            obs_dim,
            state_mean: Array1::zeros(state_dim),
            state_cov: Array2::eye(state_dim),
            evolution_matrix: Array2::eye(state_dim),
            observation_matrix,
            observation_variance: 1.0,
            evolution_cov: Array2::eye(state_dim),
            discount_factor: None,
            time_step: 0,
        }
    }

    /// Create a polynomial trend DLM of order p
    ///
    /// For a polynomial trend of order p, the state has p+1 components:
    /// [level, slope, acceleration, ...] and the evolution matrix has the structure
    /// of a Jordan block.
    ///
    /// # Example
    /// ```
    /// use torsh_series::state_space::DynamicLinearModel;
    ///
    /// // Linear trend (order 1): state = [level, slope]
    /// let dlm = DynamicLinearModel::polynomial_trend(1, 1.0, 0.1);
    /// ```
    pub fn polynomial_trend(order: usize, obs_variance: f64, evolution_variance: f64) -> Self {
        let state_dim = order + 1;
        let obs_dim = 1;

        // Evolution matrix for polynomial trend (Jordan block structure)
        let mut g = Array2::zeros((state_dim, state_dim));
        for i in 0..state_dim {
            for j in i..state_dim {
                if j == i {
                    g[[i, j]] = 1.0;
                } else if j == i + 1 {
                    g[[i, j]] = 1.0;
                }
            }
        }

        // Observation matrix: [1, 0, 0, ...]
        let mut f = Array2::zeros((obs_dim, state_dim));
        f[[0, 0]] = 1.0;

        // Evolution covariance
        let mut w = Array2::zeros((state_dim, state_dim));
        w[[state_dim - 1, state_dim - 1]] = evolution_variance;

        Self {
            state_dim,
            obs_dim,
            state_mean: Array1::zeros(state_dim),
            state_cov: Array2::eye(state_dim) * 10.0, // Diffuse prior
            evolution_matrix: g,
            observation_matrix: f,
            observation_variance: obs_variance,
            evolution_cov: w,
            discount_factor: None,
            time_step: 0,
        }
    }

    /// Create a seasonal DLM with specified period
    ///
    /// The seasonal component uses a Fourier form representation.
    ///
    /// # Example
    /// ```
    /// use torsh_series::state_space::DynamicLinearModel;
    ///
    /// // Monthly seasonality (period=12)
    /// let dlm = DynamicLinearModel::seasonal(12, 1.0, 0.01);
    /// ```
    pub fn seasonal(period: usize, obs_variance: f64, evolution_variance: f64) -> Self {
        // For a seasonal component with period p, we need p-1 states
        let state_dim = period - 1;
        let obs_dim = 1;

        // Evolution matrix: cyclic permutation with sign change
        let mut g = Array2::zeros((state_dim, state_dim));
        for i in 0..state_dim - 1 {
            g[[i, i + 1]] = 1.0;
        }
        // Last row: -1 for all previous states
        for j in 0..state_dim {
            g[[state_dim - 1, j]] = -1.0;
        }

        // Observation matrix: [1, 0, 0, ...]
        let mut f = Array2::zeros((obs_dim, state_dim));
        f[[0, 0]] = 1.0;

        // Evolution covariance
        let w = Array2::eye(state_dim) * evolution_variance;

        Self {
            state_dim,
            obs_dim,
            state_mean: Array1::zeros(state_dim),
            state_cov: Array2::eye(state_dim) * 10.0,
            evolution_matrix: g,
            observation_matrix: f,
            observation_variance: obs_variance,
            evolution_cov: w,
            discount_factor: None,
            time_step: 0,
        }
    }

    /// Set discount factor for adaptive estimation
    ///
    /// The discount factor δ (0 < δ < 1) controls how quickly the model adapts to changes.
    /// Smaller values allow faster adaptation but increase variance.
    /// Common values: 0.95-0.99 for slow adaptation, 0.8-0.9 for fast adaptation.
    pub fn with_discount_factor(mut self, discount: f64) -> Self {
        assert!(
            discount > 0.0 && discount < 1.0,
            "Discount factor must be in (0, 1)"
        );
        self.discount_factor = Some(discount);
        self
    }

    /// Set initial state
    pub fn with_initial_state(mut self, mean: Array1<f64>, cov: Array2<f64>) -> Self {
        assert_eq!(mean.len(), self.state_dim);
        assert_eq!(cov.shape(), &[self.state_dim, self.state_dim]);
        self.state_mean = mean;
        self.state_cov = cov;
        self
    }

    /// Set evolution matrix G_t
    pub fn with_evolution_matrix(mut self, g: Array2<f64>) -> Self {
        assert_eq!(g.shape(), &[self.state_dim, self.state_dim]);
        self.evolution_matrix = g;
        self
    }

    /// Set observation matrix F_t
    pub fn with_observation_matrix(mut self, f: Array2<f64>) -> Self {
        assert_eq!(f.shape(), &[self.obs_dim, self.state_dim]);
        self.observation_matrix = f;
        self
    }

    /// Predict step (prior distribution at time t)
    ///
    /// Computes:
    /// - a_t = G_t * m_{t-1} (prior state mean)
    /// - R_t = G_t * C_{t-1} * G_t' + W_t (prior state covariance)
    ///
    /// If discount factor is used:
    /// - R_t = G_t * C_{t-1} * G_t' / δ
    pub fn predict(&mut self) -> (Array1<f64>, Array2<f64>) {
        // Prior state mean: a_t = G * m_{t-1}
        let a_t = self.evolution_matrix.dot(&self.state_mean);

        // Prior state covariance
        let r_t = if let Some(delta) = self.discount_factor {
            // With discount factor: R_t = G * C * G' / δ
            let gc = self.evolution_matrix.dot(&self.state_cov);
            let gcg = gc.dot(&self.evolution_matrix.t()) / delta;
            gcg
        } else {
            // Without discount: R_t = G * C * G' + W
            let gc = self.evolution_matrix.dot(&self.state_cov);
            let gcg = gc.dot(&self.evolution_matrix.t());
            &gcg + &self.evolution_cov
        };

        (a_t, r_t)
    }

    /// Update step (posterior distribution given observation y_t)
    ///
    /// Computes:
    /// - f_t = F_t' * a_t (one-step ahead forecast mean)
    /// - q_t = F_t' * R_t * F_t + V_t (one-step ahead forecast variance)
    /// - A_t = R_t * F_t / q_t (Kalman gain)
    /// - m_t = a_t + A_t * (y_t - f_t) (posterior state mean)
    /// - C_t = R_t - A_t * q_t * A_t' (posterior state covariance)
    pub fn update(&mut self, observation: f64) -> Result<f64, String> {
        // Predict
        let (a_t, r_t) = self.predict();

        // One-step ahead forecast
        let f_t_vec = self.observation_matrix.row(0).to_owned();
        let f_t = f_t_vec.dot(&a_t);

        // One-step ahead forecast variance
        let rf = r_t.dot(&f_t_vec);
        let q_t = f_t_vec.dot(&rf) + self.observation_variance;

        if q_t <= 0.0 {
            return Err("Forecast variance must be positive".to_string());
        }

        // Forecast error
        let e_t = observation - f_t;

        // Adaptive coefficient (Kalman gain)
        let a_kalman = &rf / q_t;

        // Posterior state mean
        let m_t = &a_t + &(&a_kalman * e_t);

        // Posterior state covariance
        let aa_q = a_kalman
            .clone()
            .into_shape_with_order((self.state_dim, 1))
            .expect("reshape should succeed for compatible dimensions");
        let aa_q_t = aa_q.dot(&aa_q.t()) * q_t;
        let c_t = &r_t - &aa_q_t;

        // Update state
        self.state_mean = m_t;
        self.state_cov = c_t;
        self.time_step += 1;

        Ok(f_t)
    }

    /// Forecast k steps ahead
    ///
    /// Returns the k-step ahead forecast mean and variance
    pub fn forecast(&self, steps: usize) -> Vec<(f64, f64)> {
        let mut forecasts = Vec::with_capacity(steps);
        let mut state_mean = self.state_mean.clone();
        let mut state_cov = self.state_cov.clone();

        for _ in 0..steps {
            // Evolve state
            state_mean = self.evolution_matrix.dot(&state_mean);

            state_cov = if let Some(delta) = self.discount_factor {
                self.evolution_matrix
                    .dot(&state_cov)
                    .dot(&self.evolution_matrix.t())
                    / delta
            } else {
                let temp = self
                    .evolution_matrix
                    .dot(&state_cov)
                    .dot(&self.evolution_matrix.t());
                &temp + &self.evolution_cov
            };

            // Forecast mean
            let f_t_vec = self.observation_matrix.row(0).to_owned();
            let f_t = f_t_vec.dot(&state_mean);

            // Forecast variance
            let rf = state_cov.dot(&f_t_vec);
            let q_t = f_t_vec.dot(&rf) + self.observation_variance;

            forecasts.push((f_t, q_t));
        }

        forecasts
    }

    /// Generate samples from the forecast distribution
    pub fn forecast_samples(&self, steps: usize, num_samples: usize) -> Vec<Vec<f64>> {
        let mut rng = thread_rng();
        let mut samples = vec![Vec::with_capacity(steps); num_samples];

        for sample_idx in 0..num_samples {
            let mut state = self.state_mean.clone();

            for _ in 0..steps {
                // Evolve state with noise
                let evolution_noise = if let Some(_delta) = self.discount_factor {
                    // For discount factor, scale covariance
                    Array1::zeros(self.state_dim) // Simplified
                } else {
                    // Sample from evolution noise using standard normal and scaling
                    Array1::from_vec(
                        (0..self.state_dim)
                            .map(|i| {
                                let std = self.evolution_cov[[i, i]].sqrt();
                                let z: f64 = rng.gen_range(-3.0..3.0); // Simplified random sampling
                                z * std
                            })
                            .collect(),
                    )
                };

                state = &self.evolution_matrix.dot(&state) + &evolution_noise;

                // Observation with noise
                let f_t_vec = self.observation_matrix.row(0).to_owned();
                let y_mean = f_t_vec.dot(&state);
                let obs_std = self.observation_variance.sqrt();
                let z: f64 = rng.gen_range(-3.0..3.0); // Simplified random sampling
                let y_sample = y_mean + z * obs_std;

                samples[sample_idx].push(y_sample);
            }
        }

        samples
    }

    /// Fit DLM to observed time series
    ///
    /// Performs sequential updating for each observation
    pub fn fit(&mut self, observations: &[f64]) -> Result<Vec<f64>, String> {
        let mut forecasts = Vec::with_capacity(observations.len());

        for &obs in observations {
            let forecast = self.update(obs)?;
            forecasts.push(forecast);
        }

        Ok(forecasts)
    }

    /// Reset the DLM to initial state
    pub fn reset(&mut self) {
        self.state_mean = Array1::zeros(self.state_dim);
        self.state_cov = Array2::eye(self.state_dim);
        self.time_step = 0;
    }

    /// Get current state estimate
    pub fn get_state(&self) -> (&Array1<f64>, &Array2<f64>) {
        (&self.state_mean, &self.state_cov)
    }

    /// Get filtered values (smoothed estimates) using forward-backward algorithm
    ///
    /// Note: This is a simplified implementation. For full Kalman smoothing,
    /// use RTS (Rauch-Tung-Striebel) smoother.
    pub fn smooth(&self, observations: &[f64]) -> Result<Vec<Array1<f64>>, String> {
        // Forward pass (filtering)
        let mut filtered_means = Vec::with_capacity(observations.len());
        let mut filtered_covs = Vec::with_capacity(observations.len());

        let mut dlm_copy = self.clone();

        for &obs in observations {
            dlm_copy.update(obs)?;
            filtered_means.push(dlm_copy.state_mean.clone());
            filtered_covs.push(dlm_copy.state_cov.clone());
        }

        // For now, return filtered estimates
        // Full RTS smoothing would require backward pass
        Ok(filtered_means)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlm_creation() {
        let dlm = DynamicLinearModel::new(2, 1);
        assert_eq!(dlm.state_dim, 2);
        assert_eq!(dlm.obs_dim, 1);
        assert_eq!(dlm.time_step, 0);
    }

    #[test]
    fn test_polynomial_trend() {
        let dlm = DynamicLinearModel::polynomial_trend(1, 1.0, 0.1);
        assert_eq!(dlm.state_dim, 2); // level + slope
        assert_eq!(dlm.evolution_matrix[[0, 0]], 1.0);
        assert_eq!(dlm.evolution_matrix[[0, 1]], 1.0);
    }

    #[test]
    fn test_seasonal_dlm() {
        let dlm = DynamicLinearModel::seasonal(4, 1.0, 0.01);
        assert_eq!(dlm.state_dim, 3); // period - 1
        assert_eq!(dlm.observation_matrix[[0, 0]], 1.0);
    }

    #[test]
    fn test_predict() {
        let mut dlm = DynamicLinearModel::new(2, 1);
        let (a_t, r_t) = dlm.predict();
        assert_eq!(a_t.len(), 2);
        assert_eq!(r_t.shape(), &[2, 2]);
    }

    #[test]
    fn test_update() {
        let mut dlm = DynamicLinearModel::polynomial_trend(0, 1.0, 0.1);
        let forecast = dlm.update(1.5).unwrap();
        assert!(forecast.is_finite());
    }

    #[test]
    fn test_forecast() {
        let dlm = DynamicLinearModel::polynomial_trend(0, 1.0, 0.1);
        let forecasts = dlm.forecast(5);
        assert_eq!(forecasts.len(), 5);
        for (mean, var) in forecasts {
            assert!(mean.is_finite());
            assert!(var > 0.0);
        }
    }

    #[test]
    fn test_fit() {
        let mut dlm = DynamicLinearModel::polynomial_trend(1, 1.0, 0.1);
        let observations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let forecasts = dlm.fit(&observations).unwrap();
        assert_eq!(forecasts.len(), 5);
    }

    #[test]
    fn test_discount_factor() {
        let dlm = DynamicLinearModel::polynomial_trend(0, 1.0, 0.1).with_discount_factor(0.95);
        assert_eq!(dlm.discount_factor, Some(0.95));
    }

    #[test]
    fn test_initial_state() {
        let mean = Array1::from_vec(vec![1.0, 2.0]);
        let cov = Array2::eye(2);
        let dlm = DynamicLinearModel::new(2, 1).with_initial_state(mean.clone(), cov.clone());
        assert_eq!(dlm.state_mean, mean);
        assert_eq!(dlm.state_cov, cov);
    }

    #[test]
    fn test_reset() {
        let mut dlm = DynamicLinearModel::polynomial_trend(1, 1.0, 0.1);
        dlm.update(1.0).unwrap();
        dlm.update(2.0).unwrap();
        assert_eq!(dlm.time_step, 2);

        dlm.reset();
        assert_eq!(dlm.time_step, 0);
    }

    #[test]
    fn test_forecast_samples() {
        let dlm = DynamicLinearModel::polynomial_trend(0, 1.0, 0.1);
        let samples = dlm.forecast_samples(5, 10);
        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 5);
    }

    #[test]
    fn test_smooth() {
        let dlm = DynamicLinearModel::polynomial_trend(1, 1.0, 0.1);
        let observations = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = dlm.smooth(&observations).unwrap();
        assert_eq!(smoothed.len(), 5);
    }

    #[test]
    fn test_linear_trend_fitting() {
        // Test with a linear trend
        let mut dlm = DynamicLinearModel::polynomial_trend(1, 0.1, 0.01);
        let true_observations = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let forecasts = dlm.fit(&true_observations).unwrap();

        // After seeing linear trend, forecasts should improve
        let last_forecast = forecasts.last().unwrap();
        assert!(
            (last_forecast - 8.0).abs() < 2.0,
            "Last forecast should be close to 8.0"
        );
    }
}
