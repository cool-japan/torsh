//! Particle filter implementation for non-linear/non-Gaussian systems

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TimeSeries;
use scirs2_core::random::{thread_rng, Distribution, Uniform};
use torsh_tensor::{
    creation::{ones, randn, zeros},
    Tensor,
};

/// Particle filter for non-linear/non-Gaussian systems
pub struct ParticleFilter {
    num_particles: usize,
    state_dim: usize,
    /// Particles (samples)
    particles: Tensor,
    /// Particle weights
    weights: Tensor,
    /// Effective sample size threshold for resampling
    ess_threshold: f64,
}

impl ParticleFilter {
    /// Create a new particle filter
    pub fn new(num_particles: usize, state_dim: usize) -> Self {
        let particles: Tensor<f32> = randn(&[num_particles, state_dim]).unwrap();
        // TODO: Implement tensor scalar division
        let weights = ones(&[num_particles]).unwrap(); // / num_particles as f32;

        Self {
            num_particles,
            state_dim,
            particles,
            weights,
            ess_threshold: 0.5,
        }
    }

    /// Create with custom initial particles
    pub fn with_initial_particles(particles: Tensor, weights: Option<Tensor>) -> Self {
        let shape = particles.shape();
        let num_particles = shape.dims()[0];
        let state_dim = shape.dims()[1];

        let weights = weights.unwrap_or_else(|| {
            // Equal weights if not provided
            ones(&[num_particles]).unwrap()
        });

        Self {
            num_particles,
            state_dim,
            particles,
            weights,
            ess_threshold: 0.5,
        }
    }

    /// Set effective sample size threshold for resampling
    pub fn with_ess_threshold(mut self, threshold: f64) -> Self {
        self.ess_threshold = threshold;
        self
    }

    /// Get filter configuration
    pub fn config(&self) -> (usize, usize, f64) {
        (self.num_particles, self.state_dim, self.ess_threshold)
    }

    /// Get particles
    pub fn particles(&self) -> &Tensor {
        &self.particles
    }

    /// Get weights
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Set particles
    pub fn set_particles(&mut self, particles: Tensor) {
        self.particles = particles;
    }

    /// Set weights
    pub fn set_weights(&mut self, weights: Tensor) {
        self.weights = weights;
        self.normalize_weights();
    }

    /// Normalize weights to sum to 1
    fn normalize_weights(&mut self) {
        // Extract weights as vector
        let mut weights_vec = self.weights.to_vec().unwrap_or_default();

        // Compute sum
        let weight_sum: f32 = weights_vec.iter().sum();

        if weight_sum > 1e-10 {
            // Normalize each weight
            for w in &mut weights_vec {
                *w /= weight_sum;
            }

            // Create normalized weights tensor
            self.weights = Tensor::from_vec(weights_vec, &[self.num_particles]).unwrap();
        }
    }

    /// Compute effective sample size
    ///
    /// ESS = 1 / sum(w_i^2)
    /// Measures the degeneracy of the particle filter.
    /// Lower ESS indicates that few particles have significant weights.
    fn effective_sample_size(&self) -> f64 {
        let weights_vec = self.weights.to_vec().unwrap_or_default();

        // Compute sum of squared weights
        let sum_squared: f64 = weights_vec.iter().map(|&w| (w as f64) * (w as f64)).sum();

        if sum_squared > 1e-10 {
            1.0 / sum_squared
        } else {
            self.num_particles as f64
        }
    }

    /// Systematic resampling
    ///
    /// Low-variance resampling method that uses deterministic spacing
    /// between samples to reduce variance compared to multinomial resampling.
    ///
    /// Algorithm:
    /// 1. Generate a random starting point u ~ U[0, 1/N]
    /// 2. Sample at evenly spaced points: u + k/N for k = 0, ..., N-1
    /// 3. Select particles proportional to cumulative weights
    fn systematic_resample(&mut self) {
        let weights_vec = self.weights.to_vec().unwrap_or_default();
        let particles_vec = self.particles.to_vec().unwrap_or_default();

        // Compute cumulative weights
        let mut cumulative_weights = Vec::with_capacity(self.num_particles);
        let mut cumsum = 0.0f64;
        for &w in &weights_vec {
            cumsum += w as f64;
            cumulative_weights.push(cumsum);
        }

        // Normalize cumulative weights (should sum to 1.0 if weights were normalized)
        let total_weight = cumulative_weights.last().copied().unwrap_or(1.0);
        if total_weight > 1e-10 {
            for cw in &mut cumulative_weights {
                *cw /= total_weight;
            }
        }

        // Generate systematic sample points
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0 / self.num_particles as f64).unwrap();
        let u0 = dist.sample(&mut rng);

        let mut resampled_indices = Vec::with_capacity(self.num_particles);
        let mut j = 0;

        for i in 0..self.num_particles {
            let u = u0 + (i as f64) / (self.num_particles as f64);

            // Find index where cumulative weight exceeds u
            while j < cumulative_weights.len() && cumulative_weights[j] < u {
                j += 1;
            }

            // Clamp to valid range
            let idx = j.min(self.num_particles - 1);
            resampled_indices.push(idx);
        }

        // Resample particles
        let mut resampled_particles = Vec::with_capacity(self.num_particles * self.state_dim);
        for &idx in &resampled_indices {
            for d in 0..self.state_dim {
                let particle_idx = idx * self.state_dim + d;
                resampled_particles.push(particles_vec[particle_idx]);
            }
        }

        // Update particles
        self.particles =
            Tensor::from_vec(resampled_particles, &[self.num_particles, self.state_dim]).unwrap();
    }

    /// Multinomial resampling
    ///
    /// Standard resampling method that samples N particles independently
    /// from the discrete distribution defined by the particle weights.
    ///
    /// Algorithm:
    /// 1. Compute cumulative weight distribution
    /// 2. For each new particle, sample u ~ U[0,1]
    /// 3. Select particle i where CDF[i-1] < u <= CDF[i]
    fn multinomial_resample(&mut self) {
        let weights_vec = self.weights.to_vec().unwrap_or_default();
        let particles_vec = self.particles.to_vec().unwrap_or_default();

        // Compute cumulative weights
        let mut cumulative_weights = Vec::with_capacity(self.num_particles);
        let mut cumsum = 0.0f64;
        for &w in &weights_vec {
            cumsum += w as f64;
            cumulative_weights.push(cumsum);
        }

        // Normalize cumulative weights
        let total_weight = cumulative_weights.last().copied().unwrap_or(1.0);
        if total_weight > 1e-10 {
            for cw in &mut cumulative_weights {
                *cw /= total_weight;
            }
        }

        // Sample particles
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0).unwrap();

        let mut resampled_indices = Vec::with_capacity(self.num_particles);
        for _ in 0..self.num_particles {
            let u = dist.sample(&mut rng);

            // Binary search for the index where u falls in cumulative distribution
            let idx = cumulative_weights
                .iter()
                .position(|&cw| cw >= u)
                .unwrap_or(self.num_particles - 1);

            resampled_indices.push(idx);
        }

        // Resample particles
        let mut resampled_particles = Vec::with_capacity(self.num_particles * self.state_dim);
        for &idx in &resampled_indices {
            for d in 0..self.state_dim {
                let particle_idx = idx * self.state_dim + d;
                resampled_particles.push(particles_vec[particle_idx]);
            }
        }

        // Update particles
        self.particles =
            Tensor::from_vec(resampled_particles, &[self.num_particles, self.state_dim]).unwrap();
    }

    /// Resample particles based on weights
    pub fn resample(&mut self, method: ResamplingMethod) {
        let ess = self.effective_sample_size();
        let threshold = self.ess_threshold * self.num_particles as f64;

        if ess < threshold {
            match method {
                ResamplingMethod::Systematic => self.systematic_resample(),
                ResamplingMethod::Multinomial => self.multinomial_resample(),
            }

            // Reset weights to uniform after resampling
            self.weights = ones(&[self.num_particles]).unwrap();
            self.normalize_weights();
        }
    }

    /// Predict step with transition function
    pub fn predict(&mut self, transition_fn: &dyn Fn(&Tensor) -> Tensor) {
        // Apply transition to each particle
        for i in 0..self.num_particles {
            let particle = self.particles.slice_tensor(0, i, i + 1).unwrap();
            let _predicted = transition_fn(&particle);
            // TODO: Update particle in place when tensor operations are available
        }
    }

    /// Predict with additive noise
    pub fn predict_with_noise(
        &mut self,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        _noise_std: f32,
    ) {
        // Apply transition and add noise to each particle
        for i in 0..self.num_particles {
            let particle = self.particles.slice_tensor(0, i, i + 1).unwrap();
            let _predicted = transition_fn(&particle);

            // Add noise
            let _noise: Tensor<f32> = randn(&[1, self.state_dim]).unwrap();
            // TODO: Implement scalar multiplication and addition
            // let noisy_predicted = &predicted + &(&noise * noise_std);

            // TODO: Update particle in place
        }
    }

    /// Update weights based on observation likelihood
    pub fn update(
        &mut self,
        observation: &Tensor,
        likelihood_fn: &dyn Fn(&Tensor, &Tensor) -> f64,
    ) {
        // Update particle weights based on likelihood
        for i in 0..self.num_particles {
            let particle = self.particles.slice_tensor(0, i, i + 1).unwrap();
            let _likelihood = likelihood_fn(&particle, observation);

            // TODO: Update weight in place when tensor indexing is available
            // self.weights[i] *= likelihood as f32;
        }

        self.normalize_weights();
    }

    /// Get state estimate (weighted mean of particles)
    pub fn estimate(&self) -> Tensor {
        // Weighted average of particles
        // TODO: Implement when tensor operations are available
        // let weighted_sum = (&self.particles * &self.weights.unsqueeze(1)).sum(0);
        // weighted_sum

        // For now, return simple mean
        zeros(&[self.state_dim]).unwrap()
    }

    /// Get state covariance estimate
    pub fn covariance(&self) -> Tensor {
        // Weighted covariance of particles
        // TODO: Implement when tensor operations are available
        zeros(&[self.state_dim, self.state_dim]).unwrap()
    }

    /// Run particle filter on time series
    pub fn filter(
        &mut self,
        series: &TimeSeries,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        likelihood_fn: &dyn Fn(&Tensor, &Tensor) -> f64,
    ) -> TimeSeries {
        let mut estimates = Vec::new();

        for t in 0..series.len() {
            // Predict
            self.predict(transition_fn);

            // Update
            let obs = series.values.slice_tensor(0, t, t + 1).unwrap();
            self.update(&obs, likelihood_fn);

            // Resample
            self.resample(ResamplingMethod::Systematic);

            // Store estimate
            estimates.push(self.estimate());
        }

        // TODO: Stack estimates into tensor
        let values = zeros(&[series.len(), self.state_dim]).unwrap();
        TimeSeries::new(values)
    }

    /// Run particle smoother (forward-backward)
    pub fn smooth(
        &mut self,
        series: &TimeSeries,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        likelihood_fn: &dyn Fn(&Tensor, &Tensor) -> f64,
    ) -> TimeSeries {
        // Forward pass
        let filtered = self.filter(series, transition_fn, likelihood_fn);

        // TODO: Implement backward pass for particle smoothing
        filtered
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.particles = randn::<f32>(&[self.num_particles, self.state_dim]).unwrap();
        self.weights = ones(&[self.num_particles]).unwrap();
        self.normalize_weights();
    }

    /// Get particle statistics
    pub fn statistics(&self) -> ParticleStats {
        ParticleStats {
            num_particles: self.num_particles,
            ess: self.effective_sample_size(),
            mean: self.estimate(),
            covariance: self.covariance(),
        }
    }
}

/// Resampling methods for particle filter
pub enum ResamplingMethod {
    /// Systematic resampling (lower variance)
    Systematic,
    /// Multinomial resampling
    Multinomial,
}

/// Particle filter statistics
pub struct ParticleStats {
    pub num_particles: usize,
    pub ess: f64,
    pub mean: Tensor,
    pub covariance: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        TimeSeries::new(tensor)
    }

    fn identity_transition(x: &Tensor) -> Tensor {
        x.clone()
    }

    fn gaussian_likelihood(_particle: &Tensor, _obs: &Tensor) -> f64 {
        // Simple Gaussian likelihood for testing
        1.0
    }

    #[test]
    fn test_particle_filter_creation() {
        let pf = ParticleFilter::new(100, 2);
        let (num_particles, state_dim, ess_threshold) = pf.config();

        assert_eq!(num_particles, 100);
        assert_eq!(state_dim, 2);
        assert_eq!(ess_threshold, 0.5);
    }

    #[test]
    fn test_particle_filter_with_initial() {
        let particles: Tensor<f32> = randn(&[50, 3]).unwrap();
        let weights = ones(&[50]).unwrap();

        let pf = ParticleFilter::with_initial_particles(particles, Some(weights));
        let (num_particles, state_dim, _) = pf.config();

        assert_eq!(num_particles, 50);
        assert_eq!(state_dim, 3);
    }

    #[test]
    fn test_particle_filter_ess_threshold() {
        let pf = ParticleFilter::new(100, 2).with_ess_threshold(0.3);
        let (_, _, threshold) = pf.config();

        assert_eq!(threshold, 0.3);
    }

    #[test]
    fn test_particle_filter_particles() {
        let pf = ParticleFilter::new(100, 2);

        assert_eq!(pf.particles().shape().dims(), [100, 2]);
        assert_eq!(pf.weights().shape().dims(), [100]);
    }

    #[test]
    fn test_particle_filter_set_particles() {
        let mut pf = ParticleFilter::new(50, 2);
        let new_particles = zeros(&[50, 2]).unwrap();

        pf.set_particles(new_particles);
        assert_eq!(pf.particles().shape().dims(), [50, 2]);
    }

    #[test]
    fn test_particle_filter_set_weights() {
        let mut pf = ParticleFilter::new(50, 2);
        let new_weights = ones(&[50]).unwrap();

        pf.set_weights(new_weights);
        assert_eq!(pf.weights().shape().dims(), [50]);
    }

    #[test]
    fn test_particle_filter_predict() {
        let mut pf = ParticleFilter::new(10, 2);
        pf.predict(&identity_transition);

        // Test that predict completes without error
        assert_eq!(pf.particles().shape().dims(), [10, 2]);
    }

    #[test]
    fn test_particle_filter_predict_with_noise() {
        let mut pf = ParticleFilter::new(10, 2);
        pf.predict_with_noise(&identity_transition, 0.1);

        // Test that predict with noise completes without error
        assert_eq!(pf.particles().shape().dims(), [10, 2]);
    }

    #[test]
    fn test_particle_filter_update() {
        let mut pf = ParticleFilter::new(10, 2);
        let obs = zeros(&[1]).unwrap();

        pf.update(&obs, &gaussian_likelihood);

        // Test that update completes without error
        assert_eq!(pf.weights().shape().dims(), [10]);
    }

    #[test]
    fn test_particle_filter_resample() {
        let mut pf = ParticleFilter::new(10, 2);

        pf.resample(ResamplingMethod::Systematic);
        pf.resample(ResamplingMethod::Multinomial);

        // Test that resampling completes without error
        assert_eq!(pf.particles().shape().dims(), [10, 2]);
    }

    #[test]
    fn test_particle_filter_estimate() {
        let pf = ParticleFilter::new(10, 2);
        let estimate = pf.estimate();

        assert_eq!(estimate.shape().dims(), [2]);
    }

    #[test]
    fn test_particle_filter_covariance() {
        let pf = ParticleFilter::new(10, 2);
        let cov = pf.covariance();

        assert_eq!(cov.shape().dims(), [2, 2]);
    }

    #[test]
    fn test_particle_filter_filter() {
        let series = create_test_series();
        let mut pf = ParticleFilter::new(20, 1);

        let filtered = pf.filter(&series, &identity_transition, &gaussian_likelihood);
        assert_eq!(filtered.len(), series.len());
    }

    #[test]
    fn test_particle_filter_smooth() {
        let series = create_test_series();
        let mut pf = ParticleFilter::new(20, 1);

        let smoothed = pf.smooth(&series, &identity_transition, &gaussian_likelihood);
        assert_eq!(smoothed.len(), series.len());
    }

    #[test]
    fn test_particle_filter_reset() {
        let mut pf = ParticleFilter::new(10, 2);
        pf.reset();

        assert_eq!(pf.particles().shape().dims(), [10, 2]);
        assert_eq!(pf.weights().shape().dims(), [10]);
    }

    #[test]
    fn test_particle_filter_statistics() {
        let pf = ParticleFilter::new(10, 2);
        let stats = pf.statistics();

        assert_eq!(stats.num_particles, 10);
        assert_eq!(stats.mean.shape().dims(), [2]);
        assert_eq!(stats.covariance.shape().dims(), [2, 2]);
    }
}
