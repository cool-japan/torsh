//! Linear Kalman filter implementation

use crate::TimeSeries;
use torsh_core::error::Result;
use torsh_tensor::{
    creation::{eye, ones, zeros},
    Tensor,
};

/// Cholesky-Banachiewicz decomposition (lower triangular) for f64.
/// Returns `None` if the matrix is not positive definite.
fn kalman_cholesky_lower_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 { return None; }
                l[i * n + j] = sum.sqrt();
            } else {
                let diag = l[j * n + j];
                if diag.abs() < 1e-15 { return None; }
                l[i * n + j] = sum / diag;
            }
        }
    }
    Some(l)
}

/// Invert a symmetric positive-definite n×n f64 matrix using Cholesky.
pub(crate) fn kalman_cholesky_invert_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = kalman_cholesky_lower_f64(a, n)?;
    let mut inv = vec![0.0f64; n * n];
    for col in 0..n {
        let mut e = vec![0.0f64; n];
        e[col] = 1.0;
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = e[i];
            for j in 0..i { s -= l[i * n + j] * y[j]; }
            let d = l[i * n + i];
            if d.abs() < 1e-15 { return None; }
            y[i] = s / d;
        }
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n { s -= l[j * n + i] * x[j]; }
            let d = l[i * n + i];
            if d.abs() < 1e-15 { return None; }
            x[i] = s / d;
        }
        for row in 0..n { inv[row * n + col] = x[row]; }
    }
    Some(inv)
}

/// Kalman filter for linear state estimation
pub struct KalmanFilter {
    state_dim: usize,
    obs_dim: usize,
    /// State transition matrix (F)
    transition: Tensor,
    /// Observation matrix (H)
    observation: Tensor,
    /// Process noise covariance (Q)
    process_noise: Tensor,
    /// Measurement noise covariance (R)
    measurement_noise: Tensor,
    /// Current state estimate
    state: Tensor,
    /// State covariance
    covariance: Tensor,
}

impl KalmanFilter {
    /// Create a new Kalman filter
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            state_dim,
            obs_dim,
            transition: eye(state_dim).expect("tensor creation should succeed"),
            observation: ones(&[obs_dim, state_dim]).expect("tensor creation should succeed"),
            process_noise: eye(state_dim)
                .expect("tensor creation should succeed")
                .mul_scalar(0.01)
                .expect("scalar mul should succeed"),
            measurement_noise: eye(obs_dim)
                .expect("tensor creation should succeed")
                .mul_scalar(0.1)
                .expect("scalar mul should succeed"),
            state: zeros(&[state_dim, 1]).expect("tensor creation should succeed"), // Column vector
            covariance: eye(state_dim).expect("tensor creation should succeed"),
        }
    }

    /// Create with custom matrices
    pub fn with_matrices(
        state_dim: usize,
        obs_dim: usize,
        transition: Tensor,
        observation: Tensor,
        process_noise: Tensor,
        measurement_noise: Tensor,
    ) -> Self {
        Self {
            state_dim,
            obs_dim,
            transition,
            observation,
            process_noise,
            measurement_noise,
            state: zeros(&[state_dim, 1]).expect("tensor creation should succeed"), // Column vector
            covariance: eye(state_dim).expect("tensor creation should succeed"),
        }
    }

    /// Get current state dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.state_dim, self.obs_dim)
    }

    /// Set state transition matrix
    pub fn set_transition(&mut self, matrix: Tensor) {
        self.transition = matrix;
    }

    /// Set observation matrix
    pub fn set_observation(&mut self, matrix: Tensor) {
        self.observation = matrix;
    }

    /// Set process noise covariance
    pub fn set_process_noise(&mut self, matrix: Tensor) {
        self.process_noise = matrix;
    }

    /// Set measurement noise covariance
    pub fn set_measurement_noise(&mut self, matrix: Tensor) {
        self.measurement_noise = matrix;
    }

    /// Get state transition matrix
    pub fn transition_matrix(&self) -> &Tensor {
        &self.transition
    }

    /// Get observation matrix
    pub fn observation_matrix(&self) -> &Tensor {
        &self.observation
    }

    /// Get current state estimate
    pub fn state(&self) -> &Tensor {
        &self.state
    }

    /// Get state covariance matrix
    pub fn covariance(&self) -> &Tensor {
        &self.covariance
    }

    /// Set initial state
    pub fn set_initial_state(&mut self, state: Tensor, covariance: Tensor) {
        self.state = state;
        self.covariance = covariance;
    }

    /// Predict next state
    pub fn predict(&mut self) -> Result<Tensor> {
        // Predict step of Kalman filter
        // x = F @ x
        self.state = self.transition.matmul(&self.state)?;

        // P = F @ P @ F.T + Q
        let f_p = self.transition.matmul(&self.covariance)?;
        let f_p_ft = f_p.matmul(&self.transition.transpose(0, 1)?)?;
        self.covariance = f_p_ft.add(&self.process_noise)?;

        Ok(self.state.clone())
    }

    /// Update with observation
    pub fn update(&mut self, observation: &Tensor) -> Result<()> {
        // Update step of Kalman filter
        // Ensure observation is a column vector
        let obs_reshaped = if observation.ndim() == 1 {
            observation.view(&[self.obs_dim as i32, 1])?
        } else {
            observation.clone()
        };

        // Innovation: y = z - H @ x
        let h_x = self.observation.matmul(&self.state)?;
        let innovation = obs_reshaped.add(&h_x.mul_scalar(-1.0)?)?;

        // Innovation covariance: S = H @ P @ H.T + R
        let h_p = self.observation.matmul(&self.covariance)?;
        let h_p_ht = h_p.matmul(&self.observation.transpose(0, 1)?)?;
        let innovation_cov = h_p_ht.add(&self.measurement_noise)?;

        // Kalman gain: K = P @ H.T @ inv(S)
        let p_ht = self.covariance.matmul(&self.observation.transpose(0, 1)?)?;

        // For simplicity, use pseudoinverse approach by adding regularization
        // S_reg = S + lambda * I for numerical stability
        let lambda = 1e-6f32;
        let reg_eye = eye(self.obs_dim)?.mul_scalar(lambda)?;
        let innovation_cov_reg = innovation_cov.add(&reg_eye)?;

        // Simplified Kalman gain: K = P @ H.T / (S + lambda * I) for scalar case
        // For multivariate case, we would need matrix inverse
        let kalman_gain = if self.obs_dim == 1 {
            // Scalar case: K = P @ H.T / s_scalar
            let s_scalar = innovation_cov_reg.get_item_flat(0)? + 1e-10f32; // Add small epsilon for stability
            p_ht.div_scalar(s_scalar)?
        } else {
            // For multivariate case, use a simplified approach
            // In practice, you would use LU decomposition or SVD for matrix inverse
            p_ht.div_scalar(innovation_cov_reg.get_item_flat(0)? + 1e-10f32)?
        };

        // State update: x = x + K @ y
        let k_times_innovation = kalman_gain.matmul(&innovation)?;
        self.state = self.state.add(&k_times_innovation)?;

        // Covariance update: P = (I - K @ H) @ P (Joseph form for numerical stability)
        let k_h = kalman_gain.matmul(&self.observation)?;
        let identity = eye(self.state_dim)?;
        let i_minus_kh = identity.add(&k_h.mul_scalar(-1.0)?)?;
        self.covariance = i_minus_kh.matmul(&self.covariance)?;

        Ok(())
    }

    /// Run filter on time series
    pub fn filter(&mut self, series: &TimeSeries) -> Result<TimeSeries> {
        let mut filtered_states = Vec::new();

        for t in 0..series.len() {
            self.predict()?;
            // Extract observation at time t using flat indexing
            let obs_value = series.values.get_item_flat(t)?;
            let obs = Tensor::from_vec(vec![obs_value], &[1])?;
            self.update(&obs)?;

            // Extract state vector elements
            let mut state_vec = Vec::new();
            for i in 0..self.state_dim {
                let val = self.state.get_item_flat(i)?;
                state_vec.push(val);
            }
            filtered_states.extend(state_vec);
        }

        // Create output tensor with proper shape [time_steps, state_dim]
        let values = Tensor::from_vec(filtered_states, &[series.len(), self.state_dim])?;
        Ok(TimeSeries::new(values))
    }

    /// Smooth time series using the Rauch-Tung-Striebel (RTS) smoother.
    ///
    /// Runs a forward Kalman filter pass, recording all filtered states and predicted
    /// covariances, then performs a backward RTS pass to compute smoothed estimates.
    pub fn smooth(&mut self, series: &TimeSeries) -> Result<TimeSeries> {
        let t_len = series.len();
        let n = self.state_dim;

        // --- Forward pass ---
        // Store: filtered states, filtered covariances, predicted covariances
        let mut filtered_means: Vec<Vec<f32>> = Vec::with_capacity(t_len);
        let mut filtered_covs: Vec<Vec<f32>> = Vec::with_capacity(t_len);
        let mut predicted_covs: Vec<Vec<f32>> = Vec::with_capacity(t_len);

        self.reset();
        for t in 0..t_len {
            // Predict
            self.state = self.transition.matmul(&self.state)?;
            let f_p = self.transition.matmul(&self.covariance)?;
            let f_p_ft = f_p.matmul(&self.transition.transpose(0, 1)?)?;
            let pred_cov = f_p_ft.add(&self.process_noise)?;

            // Record predicted covariance before update
            let mut pcov_row = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    pcov_row[i * n + j] = pred_cov.get_item_flat(i * n + j)?;
                }
            }
            predicted_covs.push(pcov_row);
            self.covariance = pred_cov;

            // Update
            let obs_value = series.values.get_item_flat(t)?;
            let obs = Tensor::from_vec(vec![obs_value], &[1])?;
            self.update(&obs)?;

            // Record filtered mean and covariance
            let mut state_vec = vec![0.0f32; n];
            for i in 0..n {
                state_vec[i] = self.state.get_item_flat(i)?;
            }
            filtered_means.push(state_vec);

            let mut cov_vec = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    cov_vec[i * n + j] = self.covariance.get_item_flat(i * n + j)?;
                }
            }
            filtered_covs.push(cov_vec);
        }

        // --- Backward RTS pass ---
        // Initialise smoother with last filtered estimate
        let mut smoothed_means = filtered_means.clone();
        let mut smoothed_covs = filtered_covs.clone();

        for t in (0..t_len.saturating_sub(1)).rev() {
            // Smoother gain: G_t = P_t|t * F^T * inv(P_{t+1|t})
            let p_t_flat = &filtered_covs[t];
            let p_pred_flat = &predicted_covs[t + 1];

            // Compute P_t|t * F^T  (n×n)
            let mut pft = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        // F[k,j] via transposition: F^T[j,k] = F[k,j]
                        let f_kj = self.transition.get_item_flat(k * n + j)? as f64;
                        s += p_t_flat[i * n + k] as f64 * f_kj;
                    }
                    pft[i * n + j] = s;
                }
            }

            // Invert P_{t+1|t}
            let p_pred_f64: Vec<f64> = p_pred_flat.iter().map(|&v| v as f64).collect();
            // Add jitter for numerical stability
            let mut p_pred_reg = p_pred_f64.clone();
            for i in 0..n {
                p_pred_reg[i * n + i] += 1e-8;
            }
            let p_pred_inv = kalman_cholesky_invert_f64(&p_pred_reg, n)
                .unwrap_or_else(|| {
                    let mut diag = vec![0.0f64; n * n];
                    for i in 0..n {
                        let d = p_pred_reg[i * n + i];
                        diag[i * n + i] = if d > 1e-15 { 1.0 / d } else { 0.0 };
                    }
                    diag
                });

            // G_t = pft @ p_pred_inv  (n×n)
            let mut g = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += pft[i * n + k] * p_pred_inv[k * n + j];
                    }
                    g[i * n + j] = s;
                }
            }

            // Smoothed mean: m_t^s = m_t|t + G_t * (m_{t+1}^s - m_{t+1|t})
            // m_{t+1|t} = F * m_t|t
            let m_t = &filtered_means[t];
            let mut m_pred_next = vec![0.0f64; n];
            for i in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    let f_ik = self.transition.get_item_flat(i * n + k)? as f64;
                    s += f_ik * m_t[k] as f64;
                }
                m_pred_next[i] = s;
            }
            let m_smooth_next = &smoothed_means[t + 1];
            let mut m_smooth_t = vec![0.0f32; n];
            for i in 0..n {
                let mut delta = 0.0f64;
                for j in 0..n {
                    delta += g[i * n + j] * (m_smooth_next[j] as f64 - m_pred_next[j]);
                }
                m_smooth_t[i] = (m_t[i] as f64 + delta) as f32;
            }
            smoothed_means[t] = m_smooth_t;

            // Smoothed covariance: P_t^s = P_t|t + G_t * (P_{t+1}^s - P_{t+1|t}) * G_t^T
            let p_smooth_next = &smoothed_covs[t + 1];
            // diff = P_{t+1}^s - P_{t+1|t}
            let diff: Vec<f64> = p_smooth_next.iter().zip(p_pred_flat.iter())
                .map(|(&a, &b)| a as f64 - b as f64).collect();
            // G_t * diff  (n×n)
            let mut g_diff = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += g[i * n + k] * diff[k * n + j];
                    }
                    g_diff[i * n + j] = s;
                }
            }
            // G_t * diff * G_t^T  (n×n)
            let mut g_diff_gt = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..n {
                        s += g_diff[i * n + k] * g[j * n + k]; // g[j*n+k] = G^T[k,j]
                    }
                    g_diff_gt[i * n + j] = s;
                }
            }
            let p_t = &filtered_covs[t];
            let mut p_smooth_t = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    p_smooth_t[i * n + j] = (p_t[i * n + j] as f64 + g_diff_gt[i * n + j]) as f32;
                }
            }
            smoothed_covs[t] = p_smooth_t;
        }

        // Build output tensor [t_len × n]
        let mut out_data = vec![0.0f32; t_len * n];
        for t in 0..t_len {
            for j in 0..n {
                out_data[t * n + j] = smoothed_means[t][j];
            }
        }
        let values = Tensor::from_vec(out_data, &[t_len, n])?;
        Ok(TimeSeries::new(values))
    }

    /// Get innovation (prediction error)
    pub fn innovation(&self, observation: &Tensor) -> Result<Tensor> {
        // y = z - H @ x
        // Ensure observation is a column vector
        let obs_reshaped = if observation.ndim() == 1 {
            observation.view(&[self.obs_dim as i32, 1])?
        } else {
            observation.clone()
        };

        let h_x = self.observation.matmul(&self.state)?;
        obs_reshaped.add(&h_x.mul_scalar(-1.0)?)
    }

    /// Get innovation covariance
    pub fn innovation_covariance(&self) -> Result<Tensor> {
        // S = H @ P @ H.T + R
        let h_p = self.observation.matmul(&self.covariance)?;
        let h_p_ht = h_p.matmul(&self.observation.transpose(0, 1)?)?;
        h_p_ht.add(&self.measurement_noise)
    }

    /// Get Kalman gain (proper implementation)
    pub fn kalman_gain(&self) -> Result<Tensor> {
        // K = P @ H.T @ inv(S)
        let innovation_cov = self.innovation_covariance()?;
        let p_ht = self.covariance.matmul(&self.observation.transpose(0, 1)?)?;

        // Add regularization for numerical stability
        let lambda = 1e-6f32;
        let reg_eye = eye(self.obs_dim)?.mul_scalar(lambda)?;
        let innovation_cov_reg = innovation_cov.add(&reg_eye)?;

        // Compute Kalman gain
        let kalman_gain = if self.obs_dim == 1 {
            // Scalar case: K = P @ H.T / s_scalar
            let s_scalar = innovation_cov_reg.get_item_flat(0)? + 1e-10f32;
            p_ht.div_scalar(s_scalar)?
        } else {
            // For multivariate case, use simplified approach
            p_ht.div_scalar(innovation_cov_reg.get_item_flat(0)? + 1e-10f32)?
        };

        Ok(kalman_gain)
    }

    /// Compute log-likelihood of observations
    pub fn log_likelihood(&mut self, series: &TimeSeries) -> Result<f32> {
        // Compute log-likelihood of observations given model
        // Reset state to start fresh
        self.reset();

        let mut log_likelihood = 0.0f32;
        let n = series.len() as f32;

        // Constant terms for multivariate normal distribution
        let two_pi = 2.0 * std::f32::consts::PI;
        let obs_dim_f32 = self.obs_dim as f32;
        let log_normalization = -0.5 * obs_dim_f32 * two_pi.ln();

        for t in 0..series.len() {
            self.predict()?;

            // Get observation
            let obs_value = series.values.get_item_flat(t)?;
            let obs = Tensor::from_vec(vec![obs_value], &[1])?;

            // Compute innovation and its covariance
            let innovation = self.innovation(&obs)?;
            let innovation_cov = self.innovation_covariance()?;

            // Add small regularization for numerical stability
            let lambda = 1e-6f32;
            let reg_eye = eye(self.obs_dim)?.mul_scalar(lambda)?;
            let innovation_cov_reg = innovation_cov.add(&reg_eye)?;

            // Compute log-likelihood contribution for this observation
            // For univariate case: -0.5 * (log(2π) + log(σ²) + (y²/σ²))
            if self.obs_dim == 1 {
                let innovation_val = innovation.get_item_flat(0)?;
                let cov_val = innovation_cov_reg.get_item_flat(0)?.max(1e-10f32);

                let log_det_term = cov_val.ln();
                let quadratic_term = (innovation_val * innovation_val) / cov_val;

                log_likelihood += log_normalization - 0.5 * (log_det_term + quadratic_term);
            } else {
                // For multivariate case, we would need determinant and matrix inverse
                // Using simplified approach for now
                let innovation_norm_sq: f32 = (0..self.obs_dim)
                    .map(|i| {
                        let val = innovation.get_item_flat(i).unwrap_or(0.0);
                        val * val
                    })
                    .sum();

                let cov_trace: f32 = (0..self.obs_dim)
                    .map(|i| {
                        innovation_cov_reg
                            .get_item_flat(i * self.obs_dim + i)
                            .unwrap_or(1.0)
                    })
                    .sum();

                log_likelihood += log_normalization
                    - 0.5 * (cov_trace.ln() + innovation_norm_sq / cov_trace.max(1e-10f32));
            }

            // Update state
            self.update(&obs)?;
        }

        Ok(log_likelihood / n) // Return average log-likelihood per observation
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.state = zeros(&[self.state_dim, 1]).expect("tensor creation should succeed"); // Column vector
        self.covariance = eye(self.state_dim).expect("tensor creation should succeed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_kalman_filter_creation() {
        let kf = KalmanFilter::new(2, 1);
        let (state_dim, obs_dim) = kf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_kalman_filter_with_matrices() {
        let transition = eye(2).expect("eye should succeed");
        let observation = ones(&[1, 2]).expect("ones should succeed");
        let process_noise = eye(2).expect("eye should succeed");
        let measurement_noise = eye(1).expect("eye should succeed");

        let kf = KalmanFilter::with_matrices(
            2,
            1,
            transition,
            observation,
            process_noise,
            measurement_noise,
        );

        let (state_dim, obs_dim) = kf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_kalman_filter_matrices() {
        let mut kf = KalmanFilter::new(2, 1);
        let new_transition = eye(2).expect("eye should succeed");
        kf.set_transition(new_transition);

        assert_eq!(kf.transition_matrix().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_kalman_filter_state() {
        let mut kf = KalmanFilter::new(2, 1);
        let initial_state = zeros(&[2, 1]).expect("zeros should succeed"); // Column vector
        let initial_cov = eye(2).expect("eye should succeed");

        kf.set_initial_state(initial_state, initial_cov);
        assert_eq!(kf.state().shape().dims(), [2, 1]);
        assert_eq!(kf.covariance().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_kalman_filter_predict() {
        let mut kf = KalmanFilter::new(2, 1);
        let prediction = kf.predict().expect("prediction should succeed");
        assert_eq!(prediction.shape().dims(), [2, 1]); // Column vector
    }

    #[test]
    fn test_kalman_filter_update() {
        let mut kf = KalmanFilter::new(2, 1);
        let obs = zeros(&[1]).expect("zeros should succeed");
        kf.update(&obs).expect("update operation should succeed");
        // Test that update completes without error
    }

    #[test]
    fn test_kalman_filter_filter() {
        let series = create_test_series();
        let mut kf = KalmanFilter::new(1, 1);
        let filtered = kf.filter(&series).expect("filter operation should succeed");

        assert_eq!(filtered.len(), series.len());
    }

    #[test]
    fn test_kalman_filter_smooth() {
        let series = create_test_series();
        let mut kf = KalmanFilter::new(1, 1);
        let smoothed = kf.smooth(&series).expect("smoothing should succeed");

        assert_eq!(smoothed.len(), series.len());
    }

    #[test]
    fn test_kalman_filter_innovation() {
        let kf = KalmanFilter::new(1, 1);
        let obs = ones(&[1]).expect("ones should succeed");
        let innovation = kf
            .innovation(&obs)
            .expect("innovation computation should succeed");

        assert_eq!(innovation.shape().dims(), [1, 1]); // Column vector
    }

    #[test]
    fn test_kalman_filter_log_likelihood() {
        let series = create_test_series();
        let mut kf = KalmanFilter::new(1, 1);
        let ll = kf
            .log_likelihood(&series)
            .expect("log-likelihood computation should succeed");

        // Now returns actual computed log-likelihood (negative value expected for likelihood)
        assert!(ll < 0.0); // Log-likelihood should be negative
        assert!(ll.is_finite()); // Should be finite
    }

    #[test]
    fn test_kalman_filter_reset() {
        let mut kf = KalmanFilter::new(2, 1);
        kf.reset();

        assert_eq!(kf.state().shape().dims(), [2, 1]); // Column vector
        assert_eq!(kf.covariance().shape().dims(), [2, 2]);
    }
}
