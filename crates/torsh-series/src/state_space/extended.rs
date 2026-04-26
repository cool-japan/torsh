//! Extended Kalman filter implementation for nonlinear systems

use crate::TimeSeries;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{
    creation::{eye, zeros},
    Tensor,
};

/// Compute the inverse of a square matrix represented as a flat row-major `Vec<f32>`.
///
/// Uses Gauss-Jordan elimination with partial pivoting.
/// Returns an error if the matrix is singular or the input is not square.
fn mat_inv_f32(data: &[f32], n: usize) -> Result<Vec<f32>> {
    if data.len() != n * n {
        return Err(TorshError::InvalidArgument(format!(
            "mat_inv_f32: expected {} elements for {}×{} matrix, got {}",
            n * n,
            n,
            n,
            data.len()
        )));
    }

    // Augment [A | I]
    let mut aug = vec![0.0_f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = data[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0_f32;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot row
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * 2 * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-12_f32 {
            return Err(TorshError::Other(
                "mat_inv_f32: matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows col <-> max_row
        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }

        // Scale pivot row so pivot == 1
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }

        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                let pivot_val = aug[col * 2 * n + j];
                aug[row * 2 * n + j] -= factor * pivot_val;
            }
        }
    }

    // Extract right half [I | A^{-1}] → A^{-1}
    let mut inv = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}

/// Extended Kalman Filter for nonlinear systems
pub struct ExtendedKalmanFilter {
    state_dim: usize,
    obs_dim: usize,
    /// State transition function f(x)
    transition_fn: Box<dyn Fn(&Tensor) -> Tensor>,
    /// Observation function h(x)
    observation_fn: Box<dyn Fn(&Tensor) -> Tensor>,
    /// Jacobian of transition function
    transition_jacobian_fn: Option<Box<dyn Fn(&Tensor) -> Tensor>>,
    /// Jacobian of observation function
    observation_jacobian_fn: Option<Box<dyn Fn(&Tensor) -> Tensor>>,
    /// Process noise covariance (Q)
    process_noise: Tensor,
    /// Measurement noise covariance (R)
    measurement_noise: Tensor,
    /// Current state
    state: Tensor,
    /// State covariance
    covariance: Tensor,
}

impl ExtendedKalmanFilter {
    /// Create a new EKF with nonlinear functions
    pub fn new(
        state_dim: usize,
        obs_dim: usize,
        transition_fn: Box<dyn Fn(&Tensor) -> Tensor>,
        observation_fn: Box<dyn Fn(&Tensor) -> Tensor>,
    ) -> Self {
        Self {
            state_dim,
            obs_dim,
            transition_fn,
            observation_fn,
            transition_jacobian_fn: None,
            observation_jacobian_fn: None,
            process_noise: eye(state_dim).expect("tensor creation should succeed"),
            measurement_noise: eye(obs_dim).expect("tensor creation should succeed"),
            state: zeros(&[state_dim]).expect("tensor creation should succeed"),
            covariance: eye(state_dim).expect("tensor creation should succeed"),
        }
    }

    /// Create EKF with custom noise covariances
    pub fn with_noise(
        state_dim: usize,
        obs_dim: usize,
        transition_fn: Box<dyn Fn(&Tensor) -> Tensor>,
        observation_fn: Box<dyn Fn(&Tensor) -> Tensor>,
        process_noise: Tensor,
        measurement_noise: Tensor,
    ) -> Self {
        Self {
            state_dim,
            obs_dim,
            transition_fn,
            observation_fn,
            transition_jacobian_fn: None,
            observation_jacobian_fn: None,
            process_noise,
            measurement_noise,
            state: zeros(&[state_dim]).expect("tensor creation should succeed"),
            covariance: eye(state_dim).expect("tensor creation should succeed"),
        }
    }

    /// Set Jacobian functions for better accuracy
    pub fn with_jacobians(
        mut self,
        transition_jacobian: Box<dyn Fn(&Tensor) -> Tensor>,
        observation_jacobian: Box<dyn Fn(&Tensor) -> Tensor>,
    ) -> Self {
        self.transition_jacobian_fn = Some(transition_jacobian);
        self.observation_jacobian_fn = Some(observation_jacobian);
        self
    }

    /// Get state dimensions
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

    /// Set initial state
    pub fn set_initial_state(&mut self, state: Tensor, covariance: Tensor) {
        self.state = state;
        self.covariance = covariance;
    }

    /// Set process noise covariance
    pub fn set_process_noise(&mut self, noise: Tensor) {
        self.process_noise = noise;
    }

    /// Set measurement noise covariance
    pub fn set_measurement_noise(&mut self, noise: Tensor) {
        self.measurement_noise = noise;
    }

    /// Compute numerical Jacobian using central finite differences.
    ///
    /// Returns J where J[i, j] = (f(x + eps*e_j)[i] - f(x - eps*e_j)[i]) / (2*eps).
    ///
    /// The perturbation vectors are built from `to_vec()` so that no shared
    /// Arc storage is mutated through a clone.
    fn numerical_jacobian(
        &self,
        f: &dyn Fn(&Tensor) -> Tensor,
        x: &Tensor,
        output_dim: usize,
    ) -> Result<Tensor> {
        // eps chosen as cube-root of f32 machine epsilon (~6e-4 * sqrt(2))
        let eps = 1e-4_f32;
        let two_eps = 2.0_f32 * eps;

        let x_data = x.to_vec()?;
        let n = self.state_dim;

        // Jacobian stored row-major: J[output_dim x state_dim]
        let mut jac_data = vec![0.0_f32; output_dim * n];

        for j in 0..n {
            // x + eps * e_j
            let mut x_plus = x_data.clone();
            x_plus[j] += eps;
            let t_plus = Tensor::from_vec(x_plus, &[n])?;

            // x - eps * e_j
            let mut x_minus = x_data.clone();
            x_minus[j] -= eps;
            let t_minus = Tensor::from_vec(x_minus, &[n])?;

            let f_plus = f(&t_plus).to_vec()?;
            let f_minus = f(&t_minus).to_vec()?;

            for i in 0..output_dim {
                let fi_plus = *f_plus.get(i).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("f_plus index {} out of range", i))
                })?;
                let fi_minus = *f_minus.get(i).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("f_minus index {} out of range", i))
                })?;
                jac_data[i * n + j] = (fi_plus - fi_minus) / two_eps;
            }
        }

        Tensor::from_vec(jac_data, &[output_dim, n])
    }

    /// Get transition Jacobian at current state
    fn transition_jacobian(&self, state: &Tensor) -> Tensor {
        if let Some(ref jacobian_fn) = self.transition_jacobian_fn {
            jacobian_fn(state)
        } else {
            self.numerical_jacobian(&*self.transition_fn, state, self.state_dim)
                .expect("numerical Jacobian computation should succeed")
        }
    }

    /// Get observation Jacobian at current state
    fn observation_jacobian(&self, state: &Tensor) -> Tensor {
        if let Some(ref jacobian_fn) = self.observation_jacobian_fn {
            jacobian_fn(state)
        } else {
            self.numerical_jacobian(&*self.observation_fn, state, self.obs_dim)
                .expect("numerical Jacobian computation should succeed")
        }
    }

    /// Predict step: propagate state and covariance through the nonlinear transition.
    ///
    /// - State: `x = f(x)`
    /// - Covariance: `P = F * P * F^T + Q`  (F is the transition Jacobian)
    pub fn predict(&mut self) -> Tensor {
        self.predict_impl()
            .expect("EKF predict step should succeed")
    }

    fn predict_impl(&mut self) -> Result<Tensor> {
        // Nonlinear state prediction: x = f(x)
        self.state = (self.transition_fn)(&self.state);

        // Jacobian of transition function at new state
        let f_jac = self.transition_jacobian(&self.state);

        // P = F * P * F^T + Q
        // f_jac: [state_dim x state_dim], covariance: [state_dim x state_dim]
        let fp = f_jac.matmul(&self.covariance)?;
        let f_jac_t = f_jac.transpose(0, 1)?;
        let fp_ft = fp.matmul(&f_jac_t)?;
        self.covariance = fp_ft.add(&self.process_noise)?;

        Ok(self.state.clone())
    }

    /// Update step: incorporate a new observation.
    ///
    /// - Innovation: `y = z - h(x)`
    /// - Innovation covariance: `S = H * P * H^T + R`
    /// - Kalman gain: `K = P * H^T * S^{-1}`
    /// - State update: `x = x + K * y`
    /// - Covariance update: `P = (I - K * H) * P`
    pub fn update(&mut self, observation: &Tensor) {
        self.update_impl(observation)
            .expect("EKF update step should succeed")
    }

    fn update_impl(&mut self, observation: &Tensor) -> Result<()> {
        // Predicted observation: z_pred = h(x)
        let predicted_obs = (self.observation_fn)(&self.state);

        // Innovation: y = z - z_pred   shape: [obs_dim]
        let innovation = observation.sub(&predicted_obs)?;

        // Observation Jacobian: H = ∂h/∂x   shape: [obs_dim x state_dim]
        let h_jac = self.observation_jacobian(&self.state);

        // Innovation covariance: S = H * P * H^T + R   shape: [obs_dim x obs_dim]
        let hp = h_jac.matmul(&self.covariance)?;
        let h_jac_t = h_jac.transpose(0, 1)?;
        let hp_ht = hp.matmul(&h_jac_t)?;
        let s = hp_ht.add(&self.measurement_noise)?;

        // Kalman gain: K = P * H^T * S^{-1}   shape: [state_dim x obs_dim]
        // Invert the innovation covariance S using Gauss-Jordan elimination
        let s_shape = s.shape();
        let s_dims = s_shape.dims();
        let s_n = s_dims[0];
        let s_data = s.to_vec()?;
        let s_inv_data = mat_inv_f32(&s_data, s_n)?;
        let s_inv = Tensor::from_vec(s_inv_data, &[s_n, s_n])?;
        let ph_t = self.covariance.matmul(&h_jac_t)?;
        let kalman_gain = ph_t.matmul(&s_inv)?;

        // State update: x = x + K * y   shape: [state_dim]
        // K: [state_dim x obs_dim], y: [obs_dim] → reshape y to [obs_dim x 1] for matmul,
        // then squeeze back to 1D [state_dim]
        let obs_dim = innovation.shape().dims()[0];
        let innovation_col = innovation.reshape(&[obs_dim as i32, 1])?;
        let k_y_col = kalman_gain.matmul(&innovation_col)?;
        // k_y_col has shape [state_dim, 1] — squeeze to [state_dim]
        let k_y = k_y_col.reshape(&[self.state_dim as i32])?;
        self.state = self.state.add(&k_y)?;

        // Covariance update: P = (I - K * H) * P
        // I: [state_dim x state_dim]
        let identity = eye::<f32>(self.state_dim)?;
        let kh = kalman_gain.matmul(&h_jac)?;
        let i_minus_kh = identity.sub(&kh)?;
        self.covariance = i_minus_kh.matmul(&self.covariance)?;

        Ok(())
    }

    /// Run EKF on time series
    pub fn filter(&mut self, series: &TimeSeries) -> TimeSeries {
        self.filter_impl(series)
            .expect("EKF filter step should succeed")
    }

    fn filter_impl(&mut self, series: &TimeSeries) -> Result<TimeSeries> {
        let mut filtered_states: Vec<Tensor> = Vec::with_capacity(series.len());

        for t in 0..series.len() {
            self.predict_impl()?;
            let obs = series.values.slice_tensor(0, t, t + 1)?;
            self.update_impl(&obs)?;
            filtered_states.push(self.state.clone());
        }

        // Stack all filtered states along a new leading dimension → [T x state_dim]
        let values = Tensor::stack(&filtered_states, 0)?;
        Ok(TimeSeries::new(values))
    }

    /// Run EKF smoother (forward-backward)
    pub fn smooth(&mut self, series: &TimeSeries) -> TimeSeries {
        // Forward pass — backward RTS smoother not yet implemented
        self.filter(series)
    }

    /// Compute log-likelihood
    pub fn log_likelihood(&mut self, _series: &TimeSeries) -> f32 {
        // Log-likelihood computation deferred; requires storing innovation sequences
        0.0
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.state = zeros(&[self.state_dim]).expect("tensor creation should succeed");
        self.covariance = eye(self.state_dim).expect("tensor creation should succeed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).expect("tensor creation should succeed");
        TimeSeries::new(tensor)
    }

    fn linear_transition(x: &Tensor) -> Tensor {
        // Identity transition for testing (x_new = x)
        x.clone()
    }

    fn linear_observation_1d(x: &Tensor) -> Tensor {
        // Observe first component of state, producing obs_dim=1 output
        x.slice_tensor(0, 0, 1).expect("slice should succeed")
    }

    #[test]
    fn test_ekf_creation() {
        let ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        );

        let (state_dim, obs_dim) = ekf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_ekf_with_noise() {
        let process_noise = eye(2).expect("eye creation should succeed");
        let measurement_noise = eye(1).expect("eye creation should succeed");

        let ekf = ExtendedKalmanFilter::with_noise(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
            process_noise,
            measurement_noise,
        );

        let (state_dim, obs_dim) = ekf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_ekf_with_jacobians() {
        let transition_jac = Box::new(|_x: &Tensor| eye(2).expect("eye creation should succeed"));
        let observation_jac =
            Box::new(|_x: &Tensor| ones(&[1, 2]).expect("ones creation should succeed"));

        let ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        )
        .with_jacobians(transition_jac, observation_jac);

        let (state_dim, obs_dim) = ekf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_ekf_state() {
        let mut ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        );

        let initial_state = zeros(&[2]).expect("zeros creation should succeed");
        let initial_cov = eye(2).expect("eye creation should succeed");
        ekf.set_initial_state(initial_state, initial_cov);

        assert_eq!(ekf.state().shape().dims(), [2]);
        assert_eq!(ekf.covariance().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_ekf_predict() {
        let mut ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        );

        let prediction = ekf.predict();
        assert_eq!(prediction.shape().dims(), [2]);
    }

    #[test]
    fn test_ekf_update() {
        let mut ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        );

        // Observation has obs_dim=1 to match the observation function output
        let obs = zeros(&[1]).expect("zeros creation should succeed");
        ekf.update(&obs);
        // State must remain 1D with shape [state_dim]
        assert_eq!(ekf.state().shape().dims(), [2]);
        assert_eq!(ekf.covariance().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_ekf_filter() {
        let series = create_test_series();
        let mut ekf = ExtendedKalmanFilter::new(
            1,
            1,
            Box::new(linear_transition),
            Box::new(linear_transition),
        );

        let filtered = ekf.filter(&series);
        assert_eq!(filtered.len(), series.len());
    }

    #[test]
    fn test_ekf_smooth() {
        let series = create_test_series();
        let mut ekf = ExtendedKalmanFilter::new(
            1,
            1,
            Box::new(linear_transition),
            Box::new(linear_transition),
        );

        let smoothed = ekf.smooth(&series);
        assert_eq!(smoothed.len(), series.len());
    }

    #[test]
    fn test_ekf_log_likelihood() {
        let series = create_test_series();
        let mut ekf = ExtendedKalmanFilter::new(
            1,
            1,
            Box::new(linear_transition),
            Box::new(linear_transition),
        );

        let ll = ekf.log_likelihood(&series);
        assert_eq!(ll, 0.0); // Placeholder implementation
    }

    #[test]
    fn test_ekf_reset() {
        let mut ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        );

        ekf.reset();
        assert_eq!(ekf.state().shape().dims(), [2]);
        assert_eq!(ekf.covariance().shape().dims(), [2, 2]);
    }

    #[test]
    fn test_ekf_numerical_jacobian() {
        // Verify the numerical Jacobian approximates the analytical one for a linear function
        let ekf = ExtendedKalmanFilter::new(
            2,
            2,
            Box::new(linear_transition),
            Box::new(linear_transition),
        );

        let x = zeros(&[2]).expect("zeros creation should succeed");
        // f(x) = x is identity, so Jacobian should be close to I
        let jac = ekf
            .numerical_jacobian(&|v: &Tensor| v.clone(), &x, 2)
            .expect("numerical Jacobian should succeed");

        assert_eq!(jac.shape().dims(), [2, 2]);

        // Diagonal elements should be close to 1.0, off-diagonal close to 0.0
        let jac_data = jac.to_vec().expect("to_vec should succeed");
        let diag_0 = jac_data[0]; // J[0,0]
        let diag_1 = jac_data[3]; // J[1,1]
        assert!(
            (diag_0 - 1.0_f32).abs() < 1e-3,
            "J[0,0]={diag_0} expected ~1"
        );
        assert!(
            (diag_1 - 1.0_f32).abs() < 1e-3,
            "J[1,1]={diag_1} expected ~1"
        );
    }

    #[test]
    fn test_ekf_predict_covariance() {
        // After one predict with identity transition on identity covariance + Q=I:
        // P_new = I*I*I + I = 2*I
        let mut ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation_1d),
        );
        // Provide analytical Jacobians: F = I for identity transition
        ekf = ekf.with_jacobians(
            Box::new(|_x: &Tensor| eye(2).expect("eye creation should succeed")),
            Box::new(|_x: &Tensor| {
                let mut h = zeros(&[1, 2]).expect("zeros creation should succeed");
                h.set_item(&[0, 0], 1.0_f32)
                    .expect("set_item should succeed");
                h
            }),
        );

        ekf.predict();
        let cov_data = ekf.covariance().to_vec().expect("to_vec should succeed");
        // P = I + I = 2*I → diagonals = 2.0, off-diagonals = 0.0
        assert!(
            (cov_data[0] - 2.0_f32).abs() < 1e-5,
            "cov[0,0]={}",
            cov_data[0]
        );
        assert!(
            (cov_data[3] - 2.0_f32).abs() < 1e-5,
            "cov[1,1]={}",
            cov_data[3]
        );
        assert!(cov_data[1].abs() < 1e-5, "cov[0,1]={}", cov_data[1]);
        assert!(cov_data[2].abs() < 1e-5, "cov[1,0]={}", cov_data[2]);
    }
}
