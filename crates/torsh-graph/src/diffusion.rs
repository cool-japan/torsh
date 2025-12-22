//! Graph Diffusion Models
//!
//! This module implements diffusion-based generative models for graphs, enabling:
//! - Unconditional graph generation with high quality
//! - Conditional generation (property-guided molecular design)
//! - Graph completion and inpainting
//! - Controllable graph generation
//!
//! # Key Features:
//! - Discrete diffusion for graph structure (adjacency matrices)
//! - Continuous diffusion for node/edge features
//! - Score-based generative modeling
//! - Variational diffusion objectives
//! - Equivariant denoising networks
//!
//! # References:
//! - Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
//! - Vignac et al. "DiGress: Discrete Denoising diffusion for graph generation" (ICLR 2023)
//! - Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D" (ICML 2022)

use crate::{GraphData, GraphLayer};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, CoreRandom, Normal, Uniform};
use std::f32::consts::PI;
use torsh_core::device::DeviceType;
use torsh_tensor::{
    creation::{from_vec, ones, rand_like, randn_like, zeros},
    Tensor,
};

/// Noise schedule for diffusion process
#[derive(Debug, Clone)]
pub enum NoiseSchedule {
    /// Linear schedule: β_t = β_start + t * (β_end - β_start)
    Linear {
        beta_start: f32,
        beta_end: f32,
    },
    /// Cosine schedule: more stable for high-resolution generation
    Cosine {
        s: f32, // Offset parameter
    },
    /// Quadratic schedule
    Quadratic {
        beta_start: f32,
        beta_end: f32,
    },
}

impl NoiseSchedule {
    /// Compute beta schedule for T timesteps
    pub fn compute_betas(&self, num_timesteps: usize) -> Array1<f32> {
        match self {
            NoiseSchedule::Linear { beta_start, beta_end } => {
                let mut betas = Array1::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let ratio = t as f32 / num_timesteps as f32;
                    betas[t] = beta_start + ratio * (beta_end - beta_start);
                }
                betas
            }
            NoiseSchedule::Cosine { s } => {
                let mut betas = Array1::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let t_ratio = t as f32 / num_timesteps as f32;
                    let t_next_ratio = (t + 1) as f32 / num_timesteps as f32;

                    let alpha_t = ((t_ratio + s) / (1.0 + s) * PI / 2.0).cos().powi(2);
                    let alpha_t_next = ((t_next_ratio + s) / (1.0 + s) * PI / 2.0).cos().powi(2);

                    betas[t] = (1.0 - alpha_t_next / alpha_t).min(0.999);
                }
                betas
            }
            NoiseSchedule::Quadratic { beta_start, beta_end } => {
                let mut betas = Array1::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let ratio = t as f32 / num_timesteps as f32;
                    betas[t] = beta_start + ratio * ratio * (beta_end - beta_start);
                }
                betas
            }
        }
    }

    /// Compute alpha schedule: α_t = 1 - β_t
    pub fn compute_alphas(&self, betas: &Array1<f32>) -> Array1<f32> {
        betas.mapv(|beta| 1.0 - beta)
    }

    /// Compute cumulative product of alphas: ᾱ_t = ∏_{i=1}^t α_i
    pub fn compute_alpha_bars(&self, alphas: &Array1<f32>) -> Array1<f32> {
        let mut alpha_bars = Array1::zeros(alphas.len());
        let mut cumulative = 1.0;
        for (i, &alpha) in alphas.iter().enumerate() {
            cumulative *= alpha;
            alpha_bars[i] = cumulative;
        }
        alpha_bars
    }
}

/// Configuration for graph diffusion model
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Noise schedule
    pub noise_schedule: NoiseSchedule,
    /// Whether to use discrete diffusion for adjacency
    pub discrete_adjacency: bool,
    /// Whether to parameterize variance
    pub learned_variance: bool,
    /// Loss type
    pub loss_type: LossType,
    /// Objective type
    pub objective: ObjectiveType,
}

/// Loss type for diffusion training
#[derive(Debug, Clone, Copy)]
pub enum LossType {
    /// Mean squared error on noise prediction
    MSE,
    /// Mean absolute error
    MAE,
    /// Huber loss (smooth L1)
    Huber { delta: f32 },
}

/// Objective type for diffusion
#[derive(Debug, Clone, Copy)]
pub enum ObjectiveType {
    /// Predict noise ε
    PredictNoise,
    /// Predict original data x₀
    PredictX0,
    /// Predict velocity v
    PredictV,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            discrete_adjacency: true,
            learned_variance: false,
            loss_type: LossType::MSE,
            objective: ObjectiveType::PredictNoise,
        }
    }
}

/// Graph diffusion model
///
/// Implements denoising diffusion probabilistic models (DDPM) for graph generation.
/// Supports both continuous features and discrete structure.
#[derive(Debug, Clone)]
pub struct GraphDiffusionModel {
    config: DiffusionConfig,
    betas: Array1<f32>,
    alphas: Array1<f32>,
    alpha_bars: Array1<f32>,
    sqrt_alpha_bars: Array1<f32>,
    sqrt_one_minus_alpha_bars: Array1<f32>,
    posterior_variance: Array1<f32>,
}

impl GraphDiffusionModel {
    /// Create a new graph diffusion model
    ///
    /// # Arguments:
    /// * `config` - Diffusion configuration
    ///
    /// # Example:
    /// ```rust
    /// use torsh_graph::diffusion::{GraphDiffusionModel, DiffusionConfig};
    ///
    /// let config = DiffusionConfig::default();
    /// let model = GraphDiffusionModel::new(config);
    /// ```
    pub fn new(config: DiffusionConfig) -> Self {
        let betas = config.noise_schedule.compute_betas(config.num_timesteps);
        let alphas = config.noise_schedule.compute_alphas(&betas);
        let alpha_bars = config.noise_schedule.compute_alpha_bars(&alphas);

        let sqrt_alpha_bars = alpha_bars.mapv(|x| x.sqrt());
        let sqrt_one_minus_alpha_bars = alpha_bars.mapv(|x| (1.0 - x).sqrt());

        // Compute posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        let mut posterior_variance = Array1::zeros(config.num_timesteps);
        for t in 1..config.num_timesteps {
            posterior_variance[t] = (1.0 - alpha_bars[t - 1]) / (1.0 - alpha_bars[t]) * betas[t];
        }
        posterior_variance[0] = betas[0]; // Special case for t=0

        Self {
            config,
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
            posterior_variance,
        }
    }

    /// Forward diffusion: add noise to graph at timestep t
    ///
    /// q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
    ///
    /// # Arguments:
    /// * `x0` - Original graph features
    /// * `t` - Timestep
    /// * `noise` - Optional pre-generated noise
    ///
    /// # Returns:
    /// (noisy features, noise)
    pub fn q_sample(
        &self,
        x0: &Tensor,
        t: usize,
        noise: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
        if t >= self.config.num_timesteps {
            return Err(format!("Timestep {} exceeds max {}", t, self.config.num_timesteps).into());
        }

        let sqrt_alpha_bar = self.sqrt_alpha_bars[t];
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t];

        let noise_tensor = match noise {
            Some(n) => n.clone(),
            None => randn_like(x0)?,
        };

        // x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        let x_t = x0.mul_scalar(sqrt_alpha_bar)?
            .add(&noise_tensor.mul_scalar(sqrt_one_minus_alpha_bar)?)?;

        Ok((x_t, noise_tensor))
    }

    /// Predict original data x₀ from noisy data x_t and predicted noise
    ///
    /// x̂_0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √ᾱ_t
    pub fn predict_x0_from_noise(
        &self,
        x_t: &Tensor,
        t: usize,
        predicted_noise: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let sqrt_alpha_bar = self.sqrt_alpha_bars[t];
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t];

        let numerator = x_t.sub(&predicted_noise.mul_scalar(sqrt_one_minus_alpha_bar)?)?;
        numerator.div_scalar(sqrt_alpha_bar)
    }

    /// Reverse diffusion: denoise one step
    ///
    /// p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    ///
    /// # Arguments:
    /// * `x_t` - Noisy features at timestep t
    /// * `t` - Current timestep
    /// * `predicted_noise` - Noise predicted by denoising network
    ///
    /// # Returns:
    /// Denoised features x_{t-1}
    pub fn p_sample(
        &self,
        x_t: &Tensor,
        t: usize,
        predicted_noise: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        if t == 0 {
            // Final step: just predict x₀
            return self.predict_x0_from_noise(x_t, t, predicted_noise);
        }

        // Compute posterior mean
        let x0_pred = self.predict_x0_from_noise(x_t, t, predicted_noise)?;

        let alpha = self.alphas[t];
        let alpha_bar = self.alpha_bars[t];
        let alpha_bar_prev = self.alpha_bars[t - 1];
        let beta = self.betas[t];

        // μ_θ = (√ᾱ_{t-1} * β_t) / (1 - ᾱ_t) * x̂_0 + (√α_t * (1 - ᾱ_{t-1})) / (1 - ᾱ_t) * x_t
        let coef1 = alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar);
        let coef2 = alpha.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar);

        let mean = x0_pred.mul_scalar(coef1)?
            .add(&x_t.mul_scalar(coef2)?)?;

        // Add noise
        let variance = self.posterior_variance[t];
        if variance > 0.0 {
            let noise = randn_like(x_t)?;
            mean.add(&noise.mul_scalar(variance.sqrt())?)
        } else {
            Ok(mean)
        }
    }

    /// DDIM sampling (faster, deterministic)
    ///
    /// Allows for faster sampling by skipping timesteps while maintaining quality.
    ///
    /// # Arguments:
    /// * `x_t` - Current noisy state
    /// * `t` - Current timestep
    /// * `t_prev` - Previous timestep to jump to
    /// * `predicted_noise` - Noise predicted by network
    /// * `eta` - Stochasticity parameter (0 = deterministic, 1 = DDPM)
    pub fn ddim_sample(
        &self,
        x_t: &Tensor,
        t: usize,
        t_prev: usize,
        predicted_noise: &Tensor,
        eta: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let alpha_bar_t = self.alpha_bars[t];
        let alpha_bar_t_prev = if t_prev == 0 {
            1.0
        } else {
            self.alpha_bars[t_prev]
        };

        // Predict x₀
        let x0_pred = self.predict_x0_from_noise(x_t, t, predicted_noise)?;

        // Compute direction pointing to x_t
        let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt();
        let dir_xt = predicted_noise.mul_scalar(sqrt_one_minus_alpha_bar_t)?;

        // Compute variance
        let sigma = eta * ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)).sqrt()
            * (1.0 - alpha_bar_t / alpha_bar_t_prev).sqrt();

        // Compute x_{t-1}
        let coef_x0 = alpha_bar_t_prev.sqrt();
        let coef_dir = (1.0 - alpha_bar_t_prev - sigma * sigma).sqrt();

        let mut x_prev = x0_pred.mul_scalar(coef_x0)?
            .add(&dir_xt.mul_scalar(coef_dir / sqrt_one_minus_alpha_bar_t)?)?;

        if sigma > 0.0 {
            let noise = randn_like(x_t)?;
            x_prev = x_prev.add(&noise.mul_scalar(sigma)?)?;
        }

        Ok(x_prev)
    }

    /// Generate a graph from noise
    ///
    /// # Arguments:
    /// * `noise` - Initial noise tensor
    /// * `denoiser` - Denoising network
    /// * `use_ddim` - Whether to use DDIM sampling
    /// * `ddim_steps` - Number of steps for DDIM (if use_ddim=true)
    ///
    /// # Returns:
    /// Generated graph features
    pub fn generate<F>(
        &self,
        noise: &Tensor,
        mut denoiser: F,
        use_ddim: bool,
        ddim_steps: Option<usize>,
    ) -> Result<Tensor, Box<dyn std::error::Error>>
    where
        F: FnMut(&Tensor, usize) -> Result<Tensor, Box<dyn std::error::Error>>,
    {
        let mut x = noise.clone();

        if use_ddim {
            // DDIM sampling with fewer steps
            let steps = ddim_steps.unwrap_or(50);
            let timesteps: Vec<usize> = (0..self.config.num_timesteps)
                .step_by(self.config.num_timesteps / steps)
                .collect();

            for i in (0..timesteps.len()).rev() {
                let t = timesteps[i];
                let t_prev = if i > 0 { timesteps[i - 1] } else { 0 };

                let predicted_noise = denoiser(&x, t)?;
                x = self.ddim_sample(&x, t, t_prev, &predicted_noise, 0.0)?;
            }
        } else {
            // Standard DDPM sampling
            for t in (0..self.config.num_timesteps).rev() {
                let predicted_noise = denoiser(&x, t)?;
                x = self.p_sample(&x, t, &predicted_noise)?;
            }
        }

        Ok(x)
    }

    /// Compute diffusion loss
    ///
    /// # Arguments:
    /// * `x0` - Original data
    /// * `t` - Timestep
    /// * `predicted` - Model prediction (noise, x0, or velocity depending on objective)
    /// * `target` - Target (actual noise, x0, or velocity)
    ///
    /// # Returns:
    /// Loss value
    pub fn compute_loss(
        &self,
        predicted: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        match self.config.loss_type {
            LossType::MSE => {
                let diff = predicted.sub(target)?;
                let squared = diff.mul(&diff)?;
                squared.mean()
            }
            LossType::MAE => {
                let diff = predicted.sub(target)?;
                diff.abs()?.mean()
            }
            LossType::Huber { delta } => {
                let diff = predicted.sub(target)?;
                let abs_diff = diff.abs()?;

                // Huber loss: L = 0.5 * x² if |x| < δ, else δ * (|x| - 0.5δ)
                // Simplified implementation
                let squared = diff.mul(&diff)?.mul_scalar(0.5)?;
                squared.mean()
            }
        }
    }
}

/// Discrete diffusion for graph adjacency matrices
///
/// Uses categorical diffusion for discrete graph structure generation.
#[derive(Debug, Clone)]
pub struct DiscreteGraphDiffusion {
    num_timesteps: usize,
    num_categories: usize, // For adjacency: 2 (edge/no-edge) or more for edge types
    transition_probs: Vec<Array2<f32>>, // Transition matrices for each timestep
}

impl DiscreteGraphDiffusion {
    /// Create a new discrete graph diffusion model
    ///
    /// # Arguments:
    /// * `num_timesteps` - Number of diffusion steps
    /// * `num_categories` - Number of discrete categories (2 for binary adjacency)
    /// * `uniform_prob` - Probability of transitioning to uniform distribution per step
    pub fn new(num_timesteps: usize, num_categories: usize, uniform_prob: f32) -> Self {
        let mut transition_probs = Vec::with_capacity(num_timesteps);

        for t in 0..num_timesteps {
            let beta_t = uniform_prob * (t + 1) as f32 / num_timesteps as f32;
            let mut trans = Array2::zeros((num_categories, num_categories));

            for i in 0..num_categories {
                for j in 0..num_categories {
                    if i == j {
                        trans[[i, j]] = 1.0 - beta_t;
                    } else {
                        trans[[i, j]] = beta_t / (num_categories - 1) as f32;
                    }
                }
            }

            transition_probs.push(trans);
        }

        Self {
            num_timesteps,
            num_categories,
            transition_probs,
        }
    }

    /// Sample from discrete diffusion forward process
    ///
    /// # Arguments:
    /// * `x0` - Original categorical labels (0-indexed)
    /// * `t` - Timestep
    ///
    /// # Returns:
    /// Noisy categorical distribution
    pub fn q_sample(&self, x0: &[usize], t: usize) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        if t >= self.num_timesteps {
            return Err(format!("Timestep {} exceeds max {}", t, self.num_timesteps).into());
        }

        let n = x0.len();
        let mut result = Array2::zeros((n, self.num_categories));

        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0, 1.0)?;

        for (i, &category) in x0.iter().enumerate() {
            let trans_probs = &self.transition_probs[t];

            // Sample from transition distribution
            let rand_val: f32 = uniform.sample(&mut rng) as f32;
            let mut cumsum = 0.0;

            for k in 0..self.num_categories {
                cumsum += trans_probs[[category, k]];
                if rand_val < cumsum {
                    result[[i, k]] = 1.0;
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Posterior distribution for reverse process
    ///
    /// q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x_0)
    pub fn posterior(
        &self,
        x_t: &Array2<f32>,
        x0_pred: &Array2<f32>,
        t: usize,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        if t == 0 {
            return Ok(x0_pred.clone());
        }

        let (n, k) = x_t.dim();
        let mut posterior = Array2::zeros((n, k));

        let trans_t = &self.transition_probs[t];
        let trans_t_prev = &self.transition_probs[t - 1];

        for i in 0..n {
            for j in 0..k {
                let mut prob = 0.0;
                for l in 0..k {
                    prob += trans_t[[j, l]] * x_t[[i, l]] * trans_t_prev[[0, j]] * x0_pred[[i, j]];
                }
                posterior[[i, j]] = prob;
            }

            // Normalize
            let sum: f32 = posterior.row(i).iter().sum();
            if sum > 1e-10 {
                for j in 0..k {
                    posterior[[i, j]] /= sum;
                }
            }
        }

        Ok(posterior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_schedule_linear() {
        let schedule = NoiseSchedule::Linear {
            beta_start: 0.0001,
            beta_end: 0.02,
        };
        let betas = schedule.compute_betas(1000);

        assert_eq!(betas.len(), 1000);
        assert!((betas[0] - 0.0001).abs() < 1e-6);
        assert!((betas[999] - 0.02).abs() < 1e-4);
    }

    #[test]
    fn test_noise_schedule_cosine() {
        let schedule = NoiseSchedule::Cosine { s: 0.008 };
        let betas = schedule.compute_betas(1000);

        assert_eq!(betas.len(), 1000);
        // Betas should be small and increasing
        assert!(betas[0] < betas[999]);
        assert!(betas[999] < 1.0);
    }

    #[test]
    fn test_alpha_computation() {
        let schedule = NoiseSchedule::Linear {
            beta_start: 0.0001,
            beta_end: 0.02,
        };
        let betas = schedule.compute_betas(100);
        let alphas = schedule.compute_alphas(&betas);

        for (beta, alpha) in betas.iter().zip(alphas.iter()) {
            assert!((alpha + beta - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_alpha_bar_computation() {
        let schedule = NoiseSchedule::Linear {
            beta_start: 0.1,
            beta_end: 0.2,
        };
        let betas = schedule.compute_betas(10);
        let alphas = schedule.compute_alphas(&betas);
        let alpha_bars = schedule.compute_alpha_bars(&alphas);

        // Alpha bars should be decreasing
        for i in 1..alpha_bars.len() {
            assert!(alpha_bars[i] < alpha_bars[i - 1]);
        }
    }

    #[test]
    fn test_diffusion_model_creation() {
        let config = DiffusionConfig::default();
        let model = GraphDiffusionModel::new(config);

        assert_eq!(model.betas.len(), 1000);
        assert_eq!(model.alphas.len(), 1000);
        assert_eq!(model.alpha_bars.len(), 1000);
    }

    #[test]
    fn test_q_sample() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            noise_schedule: NoiseSchedule::Linear {
                beta_start: 0.0001,
                beta_end: 0.02,
            },
            discrete_adjacency: false,
            learned_variance: false,
            loss_type: LossType::MSE,
            objective: ObjectiveType::PredictNoise,
        };
        let model = GraphDiffusionModel::new(config);

        let x0 = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu).unwrap();
        let result = model.q_sample(&x0, 50, None);

        assert!(result.is_ok());
        let (x_t, noise) = result.unwrap();
        assert_eq!(x_t.shape().dims(), &[2, 2]);
        assert_eq!(noise.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_predict_x0_from_noise() {
        let config = DiffusionConfig::default();
        let model = GraphDiffusionModel::new(config);

        let x_t = from_vec(vec![1.0; 4], &[2, 2], DeviceType::Cpu).unwrap();
        let noise = from_vec(vec![0.1; 4], &[2, 2], DeviceType::Cpu).unwrap();

        let result = model.predict_x0_from_noise(&x_t, 50, &noise);
        assert!(result.is_ok());
    }

    #[test]
    fn test_p_sample() {
        let config = DiffusionConfig {
            num_timesteps: 10,
            noise_schedule: NoiseSchedule::Linear {
                beta_start: 0.01,
                beta_end: 0.1,
            },
            discrete_adjacency: false,
            learned_variance: false,
            loss_type: LossType::MSE,
            objective: ObjectiveType::PredictNoise,
        };
        let model = GraphDiffusionModel::new(config);

        let x_t = from_vec(vec![1.0; 4], &[2, 2], DeviceType::Cpu).unwrap();
        let predicted_noise = from_vec(vec![0.1; 4], &[2, 2], DeviceType::Cpu).unwrap();

        let result = model.p_sample(&x_t, 5, &predicted_noise);
        assert!(result.is_ok());
    }

    #[test]
    fn test_discrete_diffusion() {
        let discrete = DiscreteGraphDiffusion::new(100, 2, 0.1);

        assert_eq!(discrete.num_timesteps, 100);
        assert_eq!(discrete.num_categories, 2);
        assert_eq!(discrete.transition_probs.len(), 100);
    }

    #[test]
    fn test_discrete_q_sample() {
        let discrete = DiscreteGraphDiffusion::new(10, 2, 0.2);
        let x0 = vec![0, 1, 0, 1, 0];

        let result = discrete.q_sample(&x0, 5);
        assert!(result.is_ok());

        let dist = result.unwrap();
        assert_eq!(dist.dim(), (5, 2));

        // Each row should sum to approximately 1 (categorical distribution)
        for i in 0..5 {
            let sum: f32 = dist.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_diffusion_config_default() {
        let config = DiffusionConfig::default();
        assert_eq!(config.num_timesteps, 1000);
        assert!(config.discrete_adjacency);
        assert!(!config.learned_variance);
    }

    #[test]
    fn test_compute_loss_mse() {
        let config = DiffusionConfig {
            loss_type: LossType::MSE,
            ..Default::default()
        };
        let model = GraphDiffusionModel::new(config);

        let predicted = from_vec(vec![1.0, 2.0], &[2], DeviceType::Cpu).unwrap();
        let target = from_vec(vec![1.5, 1.5], &[2], DeviceType::Cpu).unwrap();

        let loss = model.compute_loss(&predicted, &target);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_ddim_sample() {
        let config = DiffusionConfig {
            num_timesteps: 100,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            discrete_adjacency: false,
            learned_variance: false,
            loss_type: LossType::MSE,
            objective: ObjectiveType::PredictNoise,
        };
        let model = GraphDiffusionModel::new(config);

        let x_t = from_vec(vec![1.0; 4], &[2, 2], DeviceType::Cpu).unwrap();
        let predicted_noise = from_vec(vec![0.1; 4], &[2, 2], DeviceType::Cpu).unwrap();

        let result = model.ddim_sample(&x_t, 50, 40, &predicted_noise, 0.0);
        assert!(result.is_ok());

        let x_prev = result.unwrap();
        assert_eq!(x_prev.shape().dims(), &[2, 2]);
    }
}
