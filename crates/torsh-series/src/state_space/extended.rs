//! Extended Kalman filter implementation for nonlinear systems

use crate::TimeSeries;
use torsh_tensor::{
    creation::{eye, zeros},
    Tensor,
};

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
            process_noise: eye(state_dim).unwrap(),
            measurement_noise: eye(obs_dim).unwrap(),
            state: zeros(&[state_dim]).unwrap(),
            covariance: eye(state_dim).unwrap(),
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
            state: zeros(&[state_dim]).unwrap(),
            covariance: eye(state_dim).unwrap(),
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

    /// Compute numerical Jacobian if analytical not provided
    fn numerical_jacobian(
        &self,
        f: &dyn Fn(&Tensor) -> Tensor,
        x: &Tensor,
        output_dim: usize,
    ) -> Tensor {
        // Finite difference approximation of Jacobian
        // TODO: Implement when tensor operations are available
        zeros(&[output_dim, self.state_dim]).unwrap()
    }

    /// Get transition Jacobian at current state
    fn transition_jacobian(&self, state: &Tensor) -> Tensor {
        if let Some(ref jacobian_fn) = self.transition_jacobian_fn {
            jacobian_fn(state)
        } else {
            // Use numerical differentiation
            self.numerical_jacobian(&*self.transition_fn, state, self.state_dim)
        }
    }

    /// Get observation Jacobian at current state
    fn observation_jacobian(&self, state: &Tensor) -> Tensor {
        if let Some(ref jacobian_fn) = self.observation_jacobian_fn {
            jacobian_fn(state)
        } else {
            // Use numerical differentiation
            self.numerical_jacobian(&*self.observation_fn, state, self.obs_dim)
        }
    }

    /// Predict step
    pub fn predict(&mut self) -> Tensor {
        // Nonlinear prediction: x = f(x)
        self.state = (self.transition_fn)(&self.state);

        // Covariance prediction: P = F @ P @ F.T + Q
        // where F is the Jacobian of transition function
        let f_jacobian = self.transition_jacobian(&self.state);
        // TODO: Implement matrix operations when tensor API is complete
        // self.covariance = &f_jacobian @ &self.covariance @ &f_jacobian.transpose() + &self.process_noise;

        self.state.clone()
    }

    /// Update step
    pub fn update(&mut self, observation: &Tensor) {
        // Predicted observation: z_pred = h(x)
        let predicted_obs = (self.observation_fn)(&self.state);

        // Innovation: y = z - z_pred
        // TODO: Implement subtraction when tensor API is complete
        let innovation = observation.clone();

        // Observation Jacobian: H = ∂h/∂x
        let h_jacobian = self.observation_jacobian(&self.state);

        // Innovation covariance: S = H @ P @ H.T + R
        // TODO: Implement matrix operations when tensor API is complete

        // Kalman gain: K = P @ H.T @ inv(S)
        // TODO: Implement matrix operations when tensor API is complete

        // State update: x = x + K @ y
        // TODO: Implement when tensor operations are available

        // Covariance update: P = (I - K @ H) @ P
        // TODO: Implement when tensor operations are available
    }

    /// Run EKF on time series
    pub fn filter(&mut self, series: &TimeSeries) -> TimeSeries {
        let mut filtered = Vec::new();

        for t in 0..series.len() {
            self.predict();
            let obs = series.values.slice_tensor(0, t, t + 1).unwrap();
            self.update(&obs);
            filtered.push(self.state.clone());
        }

        // TODO: Stack filtered states
        let values = zeros(&[series.len(), self.state_dim]).unwrap();
        TimeSeries::new(values)
    }

    /// Run EKF smoother (forward-backward)
    pub fn smooth(&mut self, series: &TimeSeries) -> TimeSeries {
        // Forward pass
        let filtered = self.filter(series);

        // TODO: Implement backward pass for EKF smoothing
        filtered
    }

    /// Compute log-likelihood
    pub fn log_likelihood(&mut self, series: &TimeSeries) -> f32 {
        // TODO: Implement log-likelihood computation
        0.0
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.state = zeros(&[self.state_dim]).unwrap();
        self.covariance = eye(self.state_dim).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        TimeSeries::new(tensor)
    }

    fn linear_transition(x: &Tensor) -> Tensor {
        // Simple linear transition for testing
        x.clone()
    }

    fn linear_observation(x: &Tensor) -> Tensor {
        // Simple linear observation for testing
        x.clone()
    }

    #[test]
    fn test_ekf_creation() {
        let ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation),
        );

        let (state_dim, obs_dim) = ekf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_ekf_with_noise() {
        let process_noise = eye(2).unwrap();
        let measurement_noise = eye(1).unwrap();

        let ekf = ExtendedKalmanFilter::with_noise(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation),
            process_noise,
            measurement_noise,
        );

        let (state_dim, obs_dim) = ekf.dimensions();
        assert_eq!(state_dim, 2);
        assert_eq!(obs_dim, 1);
    }

    #[test]
    fn test_ekf_with_jacobians() {
        let transition_jac = Box::new(|_x: &Tensor| eye(2).unwrap());
        let observation_jac = Box::new(|_x: &Tensor| ones(&[1, 2]).unwrap());

        let ekf = ExtendedKalmanFilter::new(
            2,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation),
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
            Box::new(linear_observation),
        );

        let initial_state = zeros(&[2]).unwrap();
        let initial_cov = eye(2).unwrap();
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
            Box::new(linear_observation),
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
            Box::new(linear_observation),
        );

        let obs = zeros(&[1]).unwrap();
        ekf.update(&obs);
        // Test that update completes without error
    }

    #[test]
    fn test_ekf_filter() {
        let series = create_test_series();
        let mut ekf = ExtendedKalmanFilter::new(
            1,
            1,
            Box::new(linear_transition),
            Box::new(linear_observation),
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
            Box::new(linear_observation),
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
            Box::new(linear_observation),
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
            Box::new(linear_observation),
        );

        ekf.reset();
        assert_eq!(ekf.state().shape().dims(), [2]);
        assert_eq!(ekf.covariance().shape().dims(), [2, 2]);
    }
}
