//! Particle filter implementation for non-linear/non-Gaussian systems

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TimeSeries;
use scirs2_core::random::{thread_rng, Distribution, Normal, Random, Uniform};
use torsh_tensor::{
    creation::{ones, randn, zeros},
    Tensor,
};

/// Fixed seed for the particle smoother's forward pass, so smoothing is
/// reproducible across runs given the same inputs.
const SMOOTHER_SEED: u64 = 0x5EED_5C0F;

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
        let particles: Tensor<f32> =
            randn(&[num_particles, state_dim]).expect("tensor creation should succeed");
        let weights = ones::<f32>(&[num_particles])
            .expect("tensor creation should succeed")
            .div_scalar(num_particles as f32)
            .expect("weight normalization should succeed");

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
            ones(&[num_particles]).expect("tensor creation should succeed")
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
            self.weights = Tensor::from_vec(weights_vec, &[self.num_particles])
                .expect("tensor creation should succeed");
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
        let dist = Uniform::new(0.0, 1.0 / self.num_particles as f64)
            .expect("distribution should succeed");
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
            Tensor::from_vec(resampled_particles, &[self.num_particles, self.state_dim])
                .expect("tensor creation should succeed");
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
        let dist = Uniform::new(0.0, 1.0).expect("distribution should succeed");

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
            Tensor::from_vec(resampled_particles, &[self.num_particles, self.state_dim])
                .expect("tensor creation should succeed");
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
            self.weights = ones(&[self.num_particles]).expect("tensor creation should succeed");
            self.normalize_weights();
        }
    }

    /// Predict step with transition function
    pub fn predict(&mut self, transition_fn: &dyn Fn(&Tensor) -> Tensor) {
        // Apply transition to each particle, building a new particles buffer
        let mut new_particles = Vec::with_capacity(self.num_particles * self.state_dim);

        for i in 0..self.num_particles {
            let particle = self
                .particles
                .slice_tensor(0, i, i + 1)
                .expect("slice should succeed");
            let predicted = transition_fn(&particle);
            let row = predicted
                .to_vec()
                .expect("particle data extraction should succeed");
            new_particles.extend_from_slice(&row);
        }

        self.particles = Tensor::from_vec(new_particles, &[self.num_particles, self.state_dim])
            .expect("particle tensor reconstruction should succeed");
    }

    /// Predict with additive noise
    pub fn predict_with_noise(
        &mut self,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        noise_std: f32,
    ) {
        // Apply transition and add noise to each particle, building a new particles buffer
        let mut new_particles = Vec::with_capacity(self.num_particles * self.state_dim);

        for i in 0..self.num_particles {
            let particle = self
                .particles
                .slice_tensor(0, i, i + 1)
                .expect("slice should succeed");
            let predicted = transition_fn(&particle);

            // Scale a standard-normal noise tensor by noise_std and add to predicted
            let noise: Tensor<f32> =
                randn(&[1, self.state_dim]).expect("tensor creation should succeed");
            let scaled_noise = noise
                .mul_scalar(noise_std)
                .expect("noise scaling should succeed");
            let noisy_predicted = predicted
                .add(&scaled_noise)
                .expect("noise addition should succeed");

            let row = noisy_predicted
                .to_vec()
                .expect("particle data extraction should succeed");
            new_particles.extend_from_slice(&row);
        }

        self.particles = Tensor::from_vec(new_particles, &[self.num_particles, self.state_dim])
            .expect("particle tensor reconstruction should succeed");
    }

    /// Update weights based on observation likelihood
    pub fn update(
        &mut self,
        observation: &Tensor,
        likelihood_fn: &dyn Fn(&Tensor, &Tensor) -> f64,
    ) {
        // Collect current weights into a vec, update each, then rebuild the tensor
        let mut weights_vec = self
            .weights
            .to_vec()
            .expect("weight data extraction should succeed");

        for i in 0..self.num_particles {
            let particle = self
                .particles
                .slice_tensor(0, i, i + 1)
                .expect("slice should succeed");
            let likelihood = likelihood_fn(&particle, observation);
            weights_vec[i] *= likelihood as f32;
        }

        self.weights = Tensor::from_vec(weights_vec, &[self.num_particles])
            .expect("weight tensor reconstruction should succeed");

        self.normalize_weights();
    }

    /// Get state estimate (weighted mean of particles)
    ///
    /// Returns the weighted mean: μ = Σ w_i * x_i
    pub fn estimate(&self) -> Tensor {
        let weights_vec = self.weights.to_vec().unwrap_or_default();
        let particles_vec = self.particles.to_vec().unwrap_or_default();

        let mut mean = vec![0.0f32; self.state_dim];

        for i in 0..self.num_particles {
            let w = weights_vec.get(i).copied().unwrap_or(0.0);
            for d in 0..self.state_dim {
                let idx = i * self.state_dim + d;
                mean[d] += w * particles_vec.get(idx).copied().unwrap_or(0.0);
            }
        }

        Tensor::from_vec(mean, &[self.state_dim]).expect("tensor creation should succeed")
    }

    /// Get state covariance estimate (weighted sample covariance)
    ///
    /// Returns Σ w_i * (x_i - μ)(x_i - μ)^T
    pub fn covariance(&self) -> Tensor {
        let weights_vec = self.weights.to_vec().unwrap_or_default();
        let particles_vec = self.particles.to_vec().unwrap_or_default();
        let mean_tensor = self.estimate();
        let mean_vec = mean_tensor.to_vec().unwrap_or_default();

        let d = self.state_dim;
        let mut cov = vec![0.0f32; d * d];

        for i in 0..self.num_particles {
            let w = weights_vec.get(i).copied().unwrap_or(0.0);
            for r in 0..d {
                let xr = particles_vec.get(i * d + r).copied().unwrap_or(0.0)
                    - mean_vec.get(r).copied().unwrap_or(0.0);
                for c in 0..d {
                    let xc = particles_vec.get(i * d + c).copied().unwrap_or(0.0)
                        - mean_vec.get(c).copied().unwrap_or(0.0);
                    cov[r * d + c] += w * xr * xc;
                }
            }
        }

        Tensor::from_vec(cov, &[d, d]).expect("tensor creation should succeed")
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
            let obs = series
                .values
                .slice_tensor(0, t, t + 1)
                .expect("slice should succeed");
            self.update(&obs, likelihood_fn);

            // Resample
            self.resample(ResamplingMethod::Systematic);

            // Store estimate
            estimates.push(self.estimate());
        }

        // Stack per-timestep estimates into [time_steps, state_dim]
        let mut stacked: Vec<f32> = Vec::with_capacity(series.len() * self.state_dim);
        for est in &estimates {
            let row = est
                .to_vec()
                .unwrap_or_else(|_| vec![0.0f32; self.state_dim]);
            stacked.extend_from_slice(&row);
        }

        let values = if stacked.is_empty() {
            zeros(&[series.len(), self.state_dim]).expect("tensor creation should succeed")
        } else {
            Tensor::from_vec(stacked, &[series.len(), self.state_dim]).unwrap_or_else(|_| {
                zeros(&[series.len(), self.state_dim]).expect("zeros should succeed")
            })
        };
        TimeSeries::new(values)
    }

    /// Run a particle smoother using Forward-Filter Backward-Smoother (FFBSm)
    /// reweighting (Godsill, Doucet & West, 2004).
    ///
    /// A bootstrap particle filter is run forward, storing at every step the
    /// filtering particles `{x_t^i}` with normalised weights `{w_t^i}` and the
    /// transition means `f(x_t^i)`. The backward pass reweights each filtering
    /// distribution with the marginal smoothing recursion
    ///
    /// ```text
    /// w_{t|T}^i = Σ_j w_{t+1|T}^j · ( w_t^i p(x_{t+1}^j | x_t^i) ) / ( Σ_l w_t^l p(x_{t+1}^j | x_t^l) )
    /// ```
    ///
    /// where the transition density is the isotropic Gaussian
    /// `p(x_{t+1} | x_t) = N(x_{t+1}; f(x_t), process_noise_std² I)` implied by the
    /// additive process noise. The smoothed estimate at time `t` is the weighted
    /// mean `Σ_i w_{t|T}^i x_t^i`.
    ///
    /// `process_noise_std` must be strictly positive (it is the std of the additive
    /// process noise); a non-positive value is floored to a tiny constant since the
    /// transition density is otherwise singular and smoothing is undefined.
    pub fn smooth(
        &mut self,
        series: &TimeSeries,
        transition_fn: &dyn Fn(&Tensor) -> Tensor,
        likelihood_fn: &dyn Fn(&Tensor, &Tensor) -> f64,
        process_noise_std: f32,
    ) -> TimeSeries {
        let t_len = series.len();
        let n = self.num_particles;
        let d = self.state_dim;

        if t_len == 0 {
            return TimeSeries::new(zeros(&[0, d]).expect("tensor creation should succeed"));
        }

        let sigma = (process_noise_std as f64).max(1e-6);
        let inv_two_var = 1.0 / (2.0 * sigma * sigma);

        // --- Forward filter with trajectory/weight storage ---
        // fwd_particles[t]: N*d filtering particle locations x_t^i
        // fwd_weights[t]:   N   normalised filtering weights w_t^i
        // fwd_pred[t]:      N*d transition means f(x_t^i) (for the transition density)
        let mut fwd_particles: Vec<Vec<f32>> = Vec::with_capacity(t_len);
        let mut fwd_weights: Vec<Vec<f32>> = Vec::with_capacity(t_len);
        let mut fwd_pred: Vec<Vec<f32>> = Vec::with_capacity(t_len);

        let mut rng = Random::seed(SMOOTHER_SEED);
        let noise_dist = Normal::new(0.0f64, sigma).expect("normal distribution should succeed");
        let unit = Uniform::new(0.0f64, 1.0).expect("distribution should succeed");

        // Working particle set starts from the filter's current particles.
        let mut particles: Vec<f32> = self
            .particles
            .to_vec()
            .unwrap_or_else(|_| vec![0.0f32; n * d]);

        // Number of observation features per time step. The t-th observation is read
        // by contiguous flat index from the original (offset-0) series tensor; a
        // sliced view cannot be used because `get_item_flat` ignores the view offset.
        let obs_feat = (series.values.numel() / t_len).max(1);

        for t in 0..t_len {
            // Predict: x_t^i = f(resampled x_{t-1}^i) + noise.
            let mut predicted = vec![0.0f32; n * d];
            for i in 0..n {
                let particle = Tensor::from_vec(particles[i * d..(i + 1) * d].to_vec(), &[1, d])
                    .expect("tensor creation should succeed");
                let moved = transition_fn(&particle);
                let row = moved
                    .to_vec()
                    .expect("transition output extraction should succeed");
                predicted[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
            }
            for v in predicted.iter_mut() {
                *v += noise_dist.sample(&mut rng) as f32;
            }

            // Update: weights proportional to the observation likelihood. Particles
            // entering each step are equally weighted (resampled below), so the
            // bootstrap weight is just the likelihood.
            let mut obs_data = Vec::with_capacity(obs_feat);
            for fidx in 0..obs_feat {
                obs_data.push(
                    series
                        .values
                        .get_item_flat(t * obs_feat + fidx)
                        .expect("observation extraction should succeed"),
                );
            }
            let obs =
                Tensor::from_vec(obs_data, &[obs_feat]).expect("tensor creation should succeed");
            let mut weights = vec![0.0f64; n];
            for i in 0..n {
                let particle = Tensor::from_vec(predicted[i * d..(i + 1) * d].to_vec(), &[1, d])
                    .expect("tensor creation should succeed");
                weights[i] = likelihood_fn(&particle, &obs).max(0.0);
            }
            let wsum: f64 = weights.iter().sum();
            if wsum > 1e-300 {
                for w in weights.iter_mut() {
                    *w /= wsum;
                }
            } else {
                let uniform_w = 1.0 / n as f64;
                for w in weights.iter_mut() {
                    *w = uniform_w;
                }
            }

            // Store the filtering distribution and the transition means f(x_t^i).
            let mut pred_means = vec![0.0f32; n * d];
            for i in 0..n {
                let particle = Tensor::from_vec(predicted[i * d..(i + 1) * d].to_vec(), &[1, d])
                    .expect("tensor creation should succeed");
                let moved = transition_fn(&particle);
                let row = moved
                    .to_vec()
                    .expect("transition output extraction should succeed");
                pred_means[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
            }
            fwd_particles.push(predicted.clone());
            fwd_weights.push(weights.iter().map(|&w| w as f32).collect());
            fwd_pred.push(pred_means);

            // Systematic resampling to obtain equally-weighted particles for t+1.
            let mut cumulative = vec![0.0f64; n];
            let mut acc = 0.0f64;
            for (i, &w) in weights.iter().enumerate() {
                acc += w;
                cumulative[i] = acc;
            }
            let total = cumulative[n - 1];
            if total > 0.0 {
                for c in cumulative.iter_mut() {
                    *c /= total;
                }
            }
            let u0 = unit.sample(&mut rng) / n as f64;
            let mut next = vec![0.0f32; n * d];
            let mut jdx = 0usize;
            for i in 0..n {
                let u = u0 + i as f64 / n as f64;
                while jdx < n - 1 && cumulative[jdx] < u {
                    jdx += 1;
                }
                next[i * d..(i + 1) * d].copy_from_slice(&predicted[jdx * d..(jdx + 1) * d]);
            }
            particles = next;
        }

        // --- Backward FFBSm pass ---
        let smoothed = self.ffbs_backward(&fwd_particles, &fwd_weights, &fwd_pred, inv_two_var);

        // Smoothed estimate at each time: weighted mean of the filtering particles.
        let mut out = vec![0.0f32; t_len * d];
        for t in 0..t_len {
            let parts = &fwd_particles[t];
            let sw = &smoothed[t];
            let mut acc = vec![0.0f64; d];
            for i in 0..n {
                let w = sw[i];
                for k in 0..d {
                    acc[k] += w * parts[i * d + k] as f64;
                }
            }
            for k in 0..d {
                out[t * d + k] = acc[k] as f32;
            }
        }

        let values = Tensor::from_vec(out, &[t_len, d]).expect("tensor creation should succeed");
        TimeSeries::new(values)
    }

    /// Backward FFBSm reweighting. Returns the smoothed weights `w_{t|T}^i` for each
    /// time step (each row sums to one).
    ///
    /// The smoothed weights at the final step equal the filtering weights; for
    /// earlier steps they are computed from the marginal smoothing recursion using
    /// the stored Gaussian transition means. Each backward kernel is normalised in
    /// log-space (log-sum-exp) for numerical stability.
    fn ffbs_backward(
        &self,
        fwd_particles: &[Vec<f32>],
        fwd_weights: &[Vec<f32>],
        fwd_pred: &[Vec<f32>],
        inv_two_var: f64,
    ) -> Vec<Vec<f64>> {
        let t_len = fwd_weights.len();
        let n = self.num_particles;
        let d = self.state_dim;

        let mut smoothed: Vec<Vec<f64>> = vec![Vec::new(); t_len];
        if t_len == 0 {
            return smoothed;
        }
        smoothed[t_len - 1] = fwd_weights[t_len - 1].iter().map(|&w| w as f64).collect();

        for t in (0..t_len - 1).rev() {
            let log_w_t: Vec<f64> = fwd_weights[t]
                .iter()
                .map(|&w| {
                    let wf = w as f64;
                    if wf > 0.0 {
                        wf.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect();
            let pred_t = &fwd_pred[t]; // f(x_t^i)
            let part_next = &fwd_particles[t + 1]; // x_{t+1}^j
            let sw_next = &smoothed[t + 1];
            let mut sw_t = vec![0.0f64; n];
            let mut log_a = vec![0.0f64; n];

            for j in 0..n {
                // Backward kernel over i: a_{j->i} ∝ w_t^i · N(x_{t+1}^j; f(x_t^i), σ²)
                //   log a_{ji} = log w_t^i − ‖x_{t+1}^j − f(x_t^i)‖² / (2σ²).
                let mut max_log = f64::NEG_INFINITY;
                for i in 0..n {
                    let mut dist2 = 0.0f64;
                    for k in 0..d {
                        let diff = part_next[j * d + k] as f64 - pred_t[i * d + k] as f64;
                        dist2 += diff * diff;
                    }
                    let l = log_w_t[i] - dist2 * inv_two_var;
                    log_a[i] = l;
                    if l > max_log {
                        max_log = l;
                    }
                }
                if !max_log.is_finite() {
                    continue; // degenerate kernel for this child; contributes nothing
                }
                let mut z = 0.0f64;
                for a in log_a.iter_mut() {
                    let e = (*a - max_log).exp();
                    *a = e;
                    z += e;
                }
                if z <= 0.0 {
                    continue;
                }
                let factor = sw_next[j] / z;
                for i in 0..n {
                    sw_t[i] += factor * log_a[i];
                }
            }
            smoothed[t] = sw_t;
        }

        smoothed
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.particles = randn::<f32>(&[self.num_particles, self.state_dim])
            .expect("tensor creation should succeed");
        self.weights = ones(&[self.num_particles]).expect("tensor creation should succeed");
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
        let tensor = Tensor::from_vec(data, &[5]).expect("Tensor should succeed");
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
        let particles: Tensor<f32> = randn(&[50, 3]).expect("randn should succeed");
        let weights = ones(&[50]).expect("ones should succeed");

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
        let new_particles = zeros(&[50, 2]).expect("zeros should succeed");

        pf.set_particles(new_particles);
        assert_eq!(pf.particles().shape().dims(), [50, 2]);
    }

    #[test]
    fn test_particle_filter_set_weights() {
        let mut pf = ParticleFilter::new(50, 2);
        let new_weights = ones(&[50]).expect("ones should succeed");

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
        let obs = zeros(&[1]).expect("zeros should succeed");

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

        let smoothed = pf.smooth(&series, &identity_transition, &gaussian_likelihood, 0.5);
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

    /// When all particles are equal, the weighted mean should equal that value
    /// and the covariance should be zero.
    #[test]
    fn test_particle_filter_estimate_uniform_particles() {
        // Build a filter via new() (weights are normalised), then set all particles
        // to the same constant vector [3.0, 5.0].
        let num_particles = 8;
        let state_dim = 2;
        let mut pf = ParticleFilter::new(num_particles, state_dim);

        let data: Vec<f32> = std::iter::repeat([3.0f32, 5.0f32])
            .take(num_particles)
            .flatten()
            .collect();
        let particles = Tensor::from_vec(data, &[num_particles, state_dim])
            .expect("tensor creation should succeed");
        pf.set_particles(particles);

        let mean = pf.estimate();
        let mean_vec = mean.to_vec().expect("mean extraction should succeed");

        // With uniform normalised weights the weighted mean equals the particle value
        assert!(
            (mean_vec[0] - 3.0).abs() < 1e-3,
            "mean[0] should be ~3.0, got {}",
            mean_vec[0]
        );
        assert!(
            (mean_vec[1] - 5.0).abs() < 1e-3,
            "mean[1] should be ~5.0, got {}",
            mean_vec[1]
        );

        // Covariance should be zero for identical particles
        let cov = pf.covariance();
        let cov_vec = cov.to_vec().expect("covariance extraction should succeed");
        for &v in &cov_vec {
            assert!(
                v.abs() < 1e-3,
                "covariance should be ~0 for identical particles, got {}",
                v
            );
        }
    }

    /// Verify that predict() actually moves particles through the transition function.
    #[test]
    fn test_particle_filter_predict_updates_particles() {
        let num_particles = 4;
        let state_dim = 2;
        // All particles start at zero
        let data = vec![0.0f32; num_particles * state_dim];
        let particles = Tensor::from_vec(data, &[num_particles, state_dim])
            .expect("tensor creation should succeed");

        let mut pf = ParticleFilter::with_initial_particles(particles, None);

        // Transition that adds 1.0 to every element
        let add_one =
            |x: &Tensor| -> Tensor { x.add_scalar(1.0_f32).expect("add scalar should succeed") };

        pf.predict(&add_one);

        let updated = pf
            .particles()
            .to_vec()
            .expect("particle extraction should succeed");
        for &v in &updated {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "each particle element should be 1.0 after predict, got {}",
                v
            );
        }
    }

    /// filter() should return a TimeSeries whose length matches the input.
    #[test]
    fn test_particle_filter_filter_output_shape() {
        let series = create_test_series();
        let mut pf = ParticleFilter::new(15, 1);
        let filtered = pf.filter(&series, &identity_transition, &gaussian_likelihood);

        // The stacked output should have shape [time_steps, state_dim]
        let dims = filtered.values.shape().dims().to_vec();
        assert_eq!(dims[0], series.len(), "first dim should equal time steps");
        assert_eq!(dims[1], 1, "second dim should equal state_dim");
    }

    /// Exact scalar Kalman filter + RTS smoother for the 1-D linear-Gaussian model
    /// x_t = a·x_{t-1} + w_t (w ~ N(0,q)), y_t = c·x_t + v_t (v ~ N(0,r)),
    /// with prior x ~ N(m0, p0) before the first predict.
    /// Returns (filtered_means, smoothed_means).
    fn analytic_kf_rts(
        ys: &[f64],
        a: f64,
        c: f64,
        q: f64,
        r: f64,
        m0: f64,
        p0: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let t_len = ys.len();
        let mut m_filt = vec![0.0f64; t_len];
        let mut p_filt = vec![0.0f64; t_len];
        let mut m_pred = vec![0.0f64; t_len];
        let mut p_pred = vec![0.0f64; t_len];

        let mut m = m0;
        let mut p = p0;
        for t in 0..t_len {
            // Predict.
            let mp = a * m;
            let pp = a * a * p + q;
            m_pred[t] = mp;
            p_pred[t] = pp;
            // Update.
            let s = c * c * pp + r;
            let k = c * pp / s;
            m = mp + k * (ys[t] - c * mp);
            p = (1.0 - k * c) * pp;
            m_filt[t] = m;
            p_filt[t] = p;
        }

        // RTS backward pass.
        let mut m_smooth = m_filt.clone();
        for t in (0..t_len.saturating_sub(1)).rev() {
            let g = p_filt[t] * a / p_pred[t + 1];
            m_smooth[t] = m_filt[t] + g * (m_smooth[t + 1] - m_pred[t + 1]);
        }
        (m_filt, m_smooth)
    }

    /// KNOWN-ANSWER: on a linear-Gaussian model the particle smoother's smoothed
    /// means must match the exact RTS-smoother means within Monte-Carlo tolerance,
    /// and must be demonstrably closer to the RTS *smoother* than to the RTS
    /// *filter* (proving the backward pass actually runs).
    #[test]
    fn test_particle_smoother_matches_rts() {
        let a = 0.9f64;
        let c = 1.0f64;
        let sigma_proc = 0.4f64;
        let sigma_obs = 1.0f64;
        let q = sigma_proc * sigma_proc;
        let r = sigma_obs * sigma_obs;
        let t_len = 20usize;

        // Generate a synthetic trajectory + observations (seeded, deterministic).
        let mut rng = Random::seed(20_240_607);
        let wdist = Normal::new(0.0f64, sigma_proc).expect("normal should succeed");
        let vdist = Normal::new(0.0f64, sigma_obs).expect("normal should succeed");
        let mut x = 0.0f64;
        let mut ys = Vec::with_capacity(t_len);
        for _ in 0..t_len {
            x = a * x + wdist.sample(&mut rng);
            ys.push(c * x + vdist.sample(&mut rng));
        }

        // Exact filter/smoother reference (prior N(0,1) matches the PF init below).
        let (m_filt, m_smooth) = analytic_kf_rts(&ys, a, c, q, r, 0.0, 1.0);

        // Build the observation series and run the particle smoother.
        let ys_f32: Vec<f32> = ys.iter().map(|&y| y as f32).collect();
        let series = TimeSeries::new(
            Tensor::from_vec(ys_f32, &[t_len]).expect("tensor creation should succeed"),
        );
        let mut pf = ParticleFilter::new(1500, 1); // initial particles ~ N(0,1)
        let transition =
            move |xt: &Tensor| -> Tensor { xt.mul_scalar(a as f32).expect("mul should succeed") };
        let likelihood = move |particle: &Tensor, obs: &Tensor| -> f64 {
            let xp = particle.get_item_flat(0).unwrap_or(0.0) as f64;
            let yo = obs.get_item_flat(0).unwrap_or(0.0) as f64;
            let diff = yo - c * xp;
            (-0.5 * diff * diff / r).exp()
        };
        let smoothed = pf.smooth(&series, &transition, &likelihood, sigma_proc as f32);
        let pf_smooth = smoothed
            .values
            .to_vec()
            .expect("smoothed extraction should succeed");
        assert_eq!(pf_smooth.len(), t_len);

        // The smoothing must be non-trivial (otherwise the test is vacuous).
        let mean_gap: f64 = (0..t_len)
            .map(|t| (m_filt[t] - m_smooth[t]).abs())
            .sum::<f64>()
            / t_len as f64;
        assert!(
            mean_gap > 0.05,
            "filter/smoother gap too small to test against ({mean_gap})"
        );

        // Per-time absolute correctness against the exact RTS smoother.
        let mut err_to_smooth = 0.0f64;
        let mut err_to_filter = 0.0f64;
        for t in 0..t_len {
            let est = pf_smooth[t] as f64;
            err_to_smooth += (est - m_smooth[t]).abs();
            err_to_filter += (est - m_filt[t]).abs();
            assert!(
                (est - m_smooth[t]).abs() < 0.25,
                "t={t}: particle-smoothed {est:.4} vs RTS-smoothed {:.4}",
                m_smooth[t]
            );
        }
        err_to_smooth /= t_len as f64;
        err_to_filter /= t_len as f64;

        // The particle smoother must be the SMOOTHER, not the filter: it is clearly
        // closer to the RTS smoother than to the RTS filter.
        assert!(
            err_to_smooth < 0.5 * err_to_filter,
            "particle smoother not closer to RTS-smoother (err_smooth={err_to_smooth:.4}, err_filter={err_to_filter:.4})"
        );
    }
}
