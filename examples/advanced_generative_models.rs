//! Advanced Generative Models Demo
//!
//! This example demonstrates sophisticated generative modeling techniques including:
//! - Variational Autoencoders (VAE) with β-VAE and β-TCVAE variants
//! - Generative Adversarial Networks (GAN) with progressive training
//! - Normalizing Flows for exact likelihood estimation
//! - Diffusion models for high-quality generation

use rand::{thread_rng, Rng};
use std::collections::HashMap;
use torsh::data::*;
use torsh::nn::*;
use torsh::optim::*;
use torsh::prelude::*;
use torsh_core::error::Result;

/// Variational Autoencoder with advanced variants
struct VAE {
    encoder: Encoder,
    decoder: Decoder,
    latent_dim: usize,
    beta: f64, // For β-VAE
    device: Device,
}

impl VAE {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        beta: f64,
        device: Device,
    ) -> Result<Self> {
        let encoder = Encoder::new(input_dim, hidden_dim, latent_dim)?;
        let decoder = Decoder::new(latent_dim, hidden_dim, input_dim)?;

        Ok(Self {
            encoder,
            decoder,
            latent_dim,
            beta,
            device,
        })
    }

    fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        let std = logvar.mul(&tensor![0.5])?.exp()?;
        let eps = randn(&mu.shape().dims())?;
        mu.add(&std.mul(&eps)?)
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Encode
        let (mu, logvar) = self.encoder.forward(x)?;

        // Reparameterize
        let z = self.reparameterize(&mu, &logvar)?;

        // Decode
        let x_recon = self.decoder.forward(&z)?;

        Ok((x_recon, mu, logvar, z))
    }

    fn loss(
        &self,
        x: &Tensor,
        x_recon: &Tensor,
        mu: &Tensor,
        logvar: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Reconstruction loss (binary cross entropy)
        let recon_loss = F::binary_cross_entropy_with_logits(x_recon, x, None, None, false)?;

        // KL divergence loss
        let kl_loss = logvar
            .exp()?
            .add(&mu.pow(2.0)?)?
            .sub(&logvar)?
            .sub(&tensor![1.0])?;
        let kl_loss = kl_loss.sum_dim(&[-1], false)?.mul(&tensor![0.5])?.mean()?;

        // Total loss with β-VAE weighting
        let total_loss = recon_loss.add(&kl_loss.mul(&tensor![self.beta as f32])?)?;

        Ok((total_loss, recon_loss, kl_loss))
    }

    fn sample(&self, num_samples: usize) -> Result<Tensor> {
        let z = randn(&[num_samples, self.latent_dim])?;
        self.decoder.forward(&z)
    }

    fn interpolate(&self, x1: &Tensor, x2: &Tensor, num_steps: usize) -> Result<Vec<Tensor>> {
        let (mu1, _) = self.encoder.forward(x1)?;
        let (mu2, _) = self.encoder.forward(x2)?;

        let mut interpolations = Vec::new();

        for i in 0..num_steps {
            let alpha = i as f32 / (num_steps - 1) as f32;
            let z_interp = mu1
                .mul(&tensor![1.0 - alpha])?
                .add(&mu2.mul(&tensor![alpha])?)?;
            let x_interp = self.decoder.forward(&z_interp)?;
            interpolations.push(x_interp);
        }

        Ok(interpolations)
    }
}

impl Module for VAE {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (x_recon, _, _, _) = self.forward(input)?;
        Ok(x_recon)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }
}

/// VAE Encoder
struct Encoder {
    layers: Sequential,
    mu_layer: Linear,
    logvar_layer: Linear,
}

impl Encoder {
    fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Result<Self> {
        let layers = Sequential::new()
            .add(Linear::new(input_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?;

        let mu_layer = Linear::new(hidden_dim, latent_dim)?;
        let logvar_layer = Linear::new(hidden_dim, latent_dim)?;

        Ok(Self {
            layers,
            mu_layer,
            logvar_layer,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.layers.forward(x)?;
        let mu = self.mu_layer.forward(&h)?;
        let logvar = self.logvar_layer.forward(&h)?;
        Ok((mu, logvar))
    }
}

impl Module for Encoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (mu, _) = self.forward(input)?;
        Ok(mu)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.layers.parameters();
        params.extend(self.mu_layer.parameters());
        params.extend(self.logvar_layer.parameters());
        params
    }
}

/// VAE Decoder
struct Decoder {
    layers: Sequential,
}

impl Decoder {
    fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        let layers = Sequential::new()
            .add(Linear::new(latent_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, output_dim)?)?;

        Ok(Self { layers })
    }
}

impl Module for Decoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layers.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.parameters()
    }
}

/// Generative Adversarial Network
struct GAN {
    generator: Generator,
    discriminator: Discriminator,
    latent_dim: usize,
    device: Device,
}

impl GAN {
    fn new(
        latent_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let generator = Generator::new(latent_dim, hidden_dim, output_dim)?;
        let discriminator = Discriminator::new(output_dim, hidden_dim)?;

        Ok(Self {
            generator,
            discriminator,
            latent_dim,
            device,
        })
    }

    fn generator_loss(&self, fake_scores: &Tensor) -> Result<Tensor> {
        // Generator wants discriminator to classify fake as real
        let real_labels = ones(&fake_scores.shape().dims())?;
        F::binary_cross_entropy_with_logits(fake_scores, &real_labels, None, None, false)
    }

    fn discriminator_loss(&self, real_scores: &Tensor, fake_scores: &Tensor) -> Result<Tensor> {
        let real_labels = ones(&real_scores.shape().dims())?;
        let fake_labels = zeros(&fake_scores.shape().dims())?;

        let real_loss =
            F::binary_cross_entropy_with_logits(real_scores, &real_labels, None, None, false)?;
        let fake_loss =
            F::binary_cross_entropy_with_logits(fake_scores, &fake_labels, None, None, false)?;

        real_loss.add(&fake_loss)?.mul(&tensor![0.5])
    }

    fn generate(&self, num_samples: usize) -> Result<Tensor> {
        let z = randn(&[num_samples, self.latent_dim])?;
        self.generator.forward(&z)
    }

    fn train_step(
        &self,
        real_data: &Tensor,
        g_optimizer: &mut Adam,
        d_optimizer: &mut Adam,
    ) -> Result<(f64, f64)> {
        let batch_size = real_data.shape().dims()[0];

        // Train Discriminator
        d_optimizer.zero_grad();

        // Real data
        let real_scores = self.discriminator.forward(real_data)?;

        // Fake data
        let z = randn(&[batch_size, self.latent_dim])?;
        let fake_data = self.generator.forward(&z)?;
        let fake_scores = self.discriminator.forward(&fake_data.detach())?;

        let d_loss = self.discriminator_loss(&real_scores, &fake_scores)?;
        d_loss.backward()?;
        d_optimizer.step()?;

        // Train Generator
        g_optimizer.zero_grad();

        let z = randn(&[batch_size, self.latent_dim])?;
        let fake_data = self.generator.forward(&z)?;
        let fake_scores = self.discriminator.forward(&fake_data)?;

        let g_loss = self.generator_loss(&fake_scores)?;
        g_loss.backward()?;
        g_optimizer.step()?;

        Ok((g_loss.item::<f32>() as f64, d_loss.item::<f32>() as f64))
    }
}

/// GAN Generator
struct Generator {
    layers: Sequential,
}

impl Generator {
    fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        let layers = Sequential::new()
            .add(Linear::new(latent_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, output_dim)?)?
            .add(Tanh::new())?; // Output in [-1, 1]

        Ok(Self { layers })
    }
}

impl Module for Generator {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layers.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.parameters()
    }
}

/// GAN Discriminator
struct Discriminator {
    layers: Sequential,
    output_layer: Linear,
}

impl Discriminator {
    fn new(input_dim: usize, hidden_dim: usize) -> Result<Self> {
        let layers = Sequential::new()
            .add(Linear::new(input_dim, hidden_dim)?)?
            .add(LeakyReLU::new(0.2)?)?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(LeakyReLU::new(0.2)?)?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(LeakyReLU::new(0.2)?)?;

        let output_layer = Linear::new(hidden_dim, 1)?;

        Ok(Self {
            layers,
            output_layer,
        })
    }
}

impl Module for Discriminator {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let h = self.layers.forward(input)?;
        self.output_layer.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.layers.parameters();
        params.extend(self.output_layer.parameters());
        params
    }
}

/// Normalizing Flow for exact likelihood estimation
struct NormalizingFlow {
    flows: Vec<CouplingLayer>,
    latent_dim: usize,
}

impl NormalizingFlow {
    fn new(latent_dim: usize, num_flows: usize, hidden_dim: usize) -> Result<Self> {
        let mut flows = Vec::new();

        for i in 0..num_flows {
            let mask = Self::create_alternating_mask(latent_dim, i % 2 == 0);
            flows.push(CouplingLayer::new(latent_dim, hidden_dim, mask)?);
        }

        Ok(Self { flows, latent_dim })
    }

    fn create_alternating_mask(dim: usize, first_half: bool) -> Tensor {
        let mut mask = vec![0.0f32; dim];

        if first_half {
            for i in 0..dim / 2 {
                mask[i] = 1.0;
            }
        } else {
            for i in dim / 2..dim {
                mask[i] = 1.0;
            }
        }

        tensor!(mask).unwrap()
    }

    fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = z.clone();
        let mut log_det_jac = zeros(&[z.shape().dims()[0]])?;

        for flow in &self.flows {
            let (x_new, ldj) = flow.forward(&x)?;
            x = x_new;
            log_det_jac = log_det_jac.add(&ldj)?;
        }

        Ok((x, log_det_jac))
    }

    fn inverse(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut z = x.clone();
        let mut log_det_jac = zeros(&[x.shape().dims()[0]])?;

        for flow in self.flows.iter().rev() {
            let (z_new, ldj) = flow.inverse(&z)?;
            z = z_new;
            log_det_jac = log_det_jac.add(&ldj)?;
        }

        Ok((z, log_det_jac))
    }

    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let (z, log_det_jac) = self.inverse(x)?;

        // Standard normal log probability
        let log_pz = z
            .pow(2.0)?
            .mul(&tensor![-0.5])?
            .sum_dim(&[-1], false)?
            .sub(&tensor![
                (self.latent_dim as f32 * (2.0 * std::f32::consts::PI).ln() / 2.0)
            ])?;

        log_pz.add(&log_det_jac)
    }
}

impl Module for NormalizingFlow {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (x, _) = self.forward(input)?;
        Ok(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for flow in &self.flows {
            params.extend(flow.parameters());
        }
        params
    }
}

/// Coupling layer for normalizing flows
struct CouplingLayer {
    scale_net: Sequential,
    translation_net: Sequential,
    mask: Tensor,
}

impl CouplingLayer {
    fn new(input_dim: usize, hidden_dim: usize, mask: Tensor) -> Result<Self> {
        let masked_dim = mask.sum()?.item::<f32>() as usize;
        let unmasked_dim = input_dim - masked_dim;

        let scale_net = Sequential::new()
            .add(Linear::new(masked_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, unmasked_dim)?)?
            .add(Tanh::new())?;

        let translation_net = Sequential::new()
            .add(Linear::new(masked_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, unmasked_dim)?)?;

        Ok(Self {
            scale_net,
            translation_net,
            mask,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x_masked = x.mul(&self.mask)?;
        let x_unmasked = x.mul(&tensor![1.0]?.sub(&self.mask)?)?;

        // Extract masked features for conditioning
        let mask_indices: Vec<usize> = (0..self.mask.numel())
            .filter(|&i| self.mask.data_ptr()[i] > 0.5)
            .collect();

        let x_cond = x_masked.index_select(
            1,
            &tensor!(mask_indices.iter().map(|&i| i as i64).collect::<Vec<_>>())?,
        )?;

        let scale = self.scale_net.forward(&x_cond)?;
        let translation = self.translation_net.forward(&x_cond)?;

        // Transform unmasked features
        let x_unmasked_transformed = x_unmasked.mul(&scale.exp())?.add(&translation)?;

        // Combine
        let x_out = x_masked.add(&x_unmasked_transformed)?;
        let log_det_jac = scale.sum_dim(&[-1], false)?;

        Ok((x_out, log_det_jac))
    }

    fn inverse(&self, y: &Tensor) -> Result<(Tensor, Tensor)> {
        let y_masked = y.mul(&self.mask)?;
        let y_unmasked = y.mul(&tensor![1.0]?.sub(&self.mask)?)?;

        // Extract masked features for conditioning
        let mask_indices: Vec<usize> = (0..self.mask.numel())
            .filter(|&i| self.mask.data_ptr()[i] > 0.5)
            .collect();

        let y_cond = y_masked.index_select(
            1,
            &tensor!(mask_indices.iter().map(|&i| i as i64).collect::<Vec<_>>())?,
        )?;

        let scale = self.scale_net.forward(&y_cond)?;
        let translation = self.translation_net.forward(&y_cond)?;

        // Inverse transform
        let x_unmasked = y_unmasked.sub(&translation)?.div(&scale.exp())?;

        // Combine
        let x_out = y_masked.add(&x_unmasked)?;
        let log_det_jac = scale.mul(&tensor![-1.0])?.sum_dim(&[-1], false)?;

        Ok((x_out, log_det_jac))
    }
}

impl Module for CouplingLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (x, _) = self.forward(input)?;
        Ok(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.scale_net.parameters();
        params.extend(self.translation_net.parameters());
        params
    }
}

/// Simplified Diffusion Model
struct DiffusionModel {
    denoiser: UNet,
    num_timesteps: usize,
    beta_schedule: Vec<f64>,
    alpha_schedule: Vec<f64>,
    alpha_cumprod: Vec<f64>,
}

impl DiffusionModel {
    fn new(input_dim: usize, hidden_dim: usize, num_timesteps: usize) -> Result<Self> {
        let denoiser = UNet::new(input_dim, hidden_dim)?;

        // Linear beta schedule
        let beta_start = 1e-4;
        let beta_end = 0.02;
        let mut beta_schedule = Vec::new();
        let mut alpha_schedule = Vec::new();
        let mut alpha_cumprod = Vec::new();

        let mut alpha_cumprod_prev = 1.0;

        for t in 0..num_timesteps {
            let beta = beta_start + (beta_end - beta_start) * t as f64 / (num_timesteps - 1) as f64;
            let alpha = 1.0 - beta;
            let alpha_cumprod_t = alpha_cumprod_prev * alpha;

            beta_schedule.push(beta);
            alpha_schedule.push(alpha);
            alpha_cumprod.push(alpha_cumprod_t);

            alpha_cumprod_prev = alpha_cumprod_t;
        }

        Ok(Self {
            denoiser,
            num_timesteps,
            beta_schedule,
            alpha_schedule,
            alpha_cumprod,
        })
    }

    fn q_sample(&self, x0: &Tensor, t: usize, noise: Option<&Tensor>) -> Result<Tensor> {
        let noise = noise
            .map(|n| n.clone())
            .unwrap_or_else(|| randn(&x0.shape().dims()).unwrap());

        let sqrt_alpha_cumprod = (self.alpha_cumprod[t].sqrt()) as f32;
        let sqrt_one_minus_alpha_cumprod = ((1.0 - self.alpha_cumprod[t]).sqrt()) as f32;

        x0.mul(&tensor![sqrt_alpha_cumprod])?
            .add(&noise.mul(&tensor![sqrt_one_minus_alpha_cumprod])?)
    }

    fn p_sample(&self, xt: &Tensor, t: usize) -> Result<Tensor> {
        let t_tensor = tensor![t as f32];
        let noise_pred = self.denoiser.forward(xt, &t_tensor)?;

        let alpha = self.alpha_schedule[t] as f32;
        let alpha_cumprod = self.alpha_cumprod[t] as f32;
        let beta = self.beta_schedule[t] as f32;

        let mean_coeff = 1.0 / alpha.sqrt();
        let noise_coeff = beta / (1.0 - alpha_cumprod).sqrt();

        let mean = xt
            .sub(&noise_pred.mul(&tensor![noise_coeff])?)?
            .mul(&tensor![mean_coeff])?;

        if t > 0 {
            let noise = randn(&xt.shape().dims())?;
            let std = (beta.sqrt()) as f32;
            mean.add(&noise.mul(&tensor![std])?)
        } else {
            Ok(mean)
        }
    }

    fn sample(&self, shape: &[usize]) -> Result<Tensor> {
        let mut x = randn(shape)?;

        for t in (0..self.num_timesteps).rev() {
            x = self.p_sample(&x, t)?;
        }

        Ok(x)
    }

    fn loss(&self, x0: &Tensor) -> Result<Tensor> {
        let batch_size = x0.shape().dims()[0];
        let mut rng = thread_rng();

        // Random timestep
        let t = rng.gen_range(0..self.num_timesteps);
        let noise = randn(&x0.shape().dims())?;

        // Forward diffusion
        let xt = self.q_sample(x0, t, Some(&noise))?;

        // Predict noise
        let t_tensor = tensor![t as f32];
        let noise_pred = self.denoiser.forward(&xt, &t_tensor)?;

        // MSE loss
        F::mse_loss(&noise_pred, &noise, false)
    }
}

impl Module for DiffusionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // This is a placeholder; diffusion models don't have a simple forward pass
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.denoiser.parameters()
    }
}

/// Simplified U-Net for diffusion model
struct UNet {
    time_embedding: Sequential,
    encoder: Sequential,
    decoder: Sequential,
    output_layer: Linear,
}

impl UNet {
    fn new(input_dim: usize, hidden_dim: usize) -> Result<Self> {
        let time_embedding = Sequential::new()
            .add(Linear::new(1, hidden_dim)?)?
            .add(ReLU::new())?;

        let encoder = Sequential::new()
            .add(Linear::new(input_dim + hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?;

        let decoder = Sequential::new()
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?;

        let output_layer = Linear::new(hidden_dim, input_dim)?;

        Ok(Self {
            time_embedding,
            encoder,
            decoder,
            output_layer,
        })
    }

    fn forward(&self, x: &Tensor, t: &Tensor) -> Result<Tensor> {
        let t_emb = self.time_embedding.forward(t)?;

        // Broadcast time embedding to match batch size
        let batch_size = x.shape().dims()[0];
        let t_emb_broadcast = t_emb
            .unsqueeze(0)?
            .expand(&[batch_size, t_emb.shape().dims()[0]])?;

        // Concatenate input with time embedding
        let x_with_time = torch::cat(&[x, &t_emb_broadcast], 1)?;

        // Encode
        let encoded = self.encoder.forward(&x_with_time)?;

        // Decode
        let decoded = self.decoder.forward(&encoded)?;

        // Output
        self.output_layer.forward(&decoded)
    }
}

impl Module for UNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // This requires a time input; this is a placeholder
        let t = zeros(&[1])?;
        self.forward(input, &t)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.time_embedding.parameters();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params.extend(self.output_layer.parameters());
        params
    }
}

/// Run comprehensive generative models demo
fn run_generative_models_demo() -> Result<()> {
    println!("=== Advanced Generative Models Demo ===\n");

    let device = Device::cpu();
    let data_dim = 784; // MNIST-like
    let latent_dim = 20;
    let hidden_dim = 512;
    let batch_size = 64;
    let epochs = 10; // Reduced for demo

    // Create synthetic dataset (normally you'd use real data)
    let dataset = create_synthetic_dataset(1000, data_dim)?;
    let dataloader = DataLoader::new(dataset, batch_size, true, 1, false);

    // VAE Training
    println!("--- Training Variational Autoencoder ---");
    let vae = VAE::new(data_dim, hidden_dim, latent_dim, 1.0, device.clone())?;
    let mut vae_optimizer = Adam::new(vae.parameters(), 1e-3)?;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            let (data, _) = batch;

            let (x_recon, mu, logvar, _) = vae.forward(&data)?;
            let (loss, recon_loss, kl_loss) = vae.loss(&data, &x_recon, &mu, &logvar)?;

            vae_optimizer.zero_grad();
            loss.backward()?;
            vae_optimizer.step()?;

            total_loss += loss.item::<f32>()? as f64;
            batch_count += 1;

            if batch_count % 5 == 0 {
                println!(
                    "  Batch {}: Loss = {:.6}, Recon = {:.6}, KL = {:.6}",
                    batch_count,
                    loss.item::<f32>(),
                    recon_loss.item::<f32>(),
                    kl_loss.item::<f32>()
                );
                break; // Limit batches for demo
            }
        }

        println!(
            "VAE Epoch {}: Avg Loss = {:.6}",
            epoch + 1,
            total_loss / batch_count as f64
        );
    }

    // GAN Training
    println!("\n--- Training Generative Adversarial Network ---");
    let gan = GAN::new(latent_dim, hidden_dim, data_dim, device.clone())?;
    let mut g_optimizer = Adam::new(gan.generator.parameters(), 2e-4)?;
    let mut d_optimizer = Adam::new(gan.discriminator.parameters(), 2e-4)?;

    for epoch in 0..epochs {
        let mut g_loss_total = 0.0;
        let mut d_loss_total = 0.0;
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            let (real_data, _) = batch;

            let (g_loss, d_loss) =
                gan.train_step(&real_data, &mut g_optimizer, &mut d_optimizer)?;

            g_loss_total += g_loss;
            d_loss_total += d_loss;
            batch_count += 1;

            if batch_count % 5 == 0 {
                println!(
                    "  Batch {}: G_Loss = {:.6}, D_Loss = {:.6}",
                    batch_count, g_loss, d_loss
                );
                break; // Limit batches for demo
            }
        }

        println!(
            "GAN Epoch {}: G_Loss = {:.6}, D_Loss = {:.6}",
            epoch + 1,
            g_loss_total / batch_count as f64,
            d_loss_total / batch_count as f64
        );
    }

    // Normalizing Flows Training
    println!("\n--- Training Normalizing Flows ---");
    let flow = NormalizingFlow::new(latent_dim, 8, hidden_dim)?;
    let mut flow_optimizer = Adam::new(flow.parameters(), 1e-3)?;

    for epoch in 0..epochs {
        let mut total_nll = 0.0;
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            let (data, _) = batch;

            // Project data to latent space (simplified)
            let latent_data = data.narrow(1, 0, latent_dim)?;

            let log_prob = flow.log_prob(&latent_data)?;
            let nll = log_prob.mul(&tensor![-1.0])?.mean()?;

            flow_optimizer.zero_grad();
            nll.backward()?;
            flow_optimizer.step()?;

            total_nll += nll.item::<f32>()? as f64;
            batch_count += 1;

            if batch_count % 5 == 0 {
                println!("  Batch {}: NLL = {:.6}", batch_count, nll.item::<f32>()?);
                break; // Limit batches for demo
            }
        }

        println!(
            "Flow Epoch {}: Avg NLL = {:.6}",
            epoch + 1,
            total_nll / batch_count as f64
        );
    }

    // Diffusion Model Training
    println!("\n--- Training Diffusion Model ---");
    let diffusion = DiffusionModel::new(latent_dim, hidden_dim, 1000)?;
    let mut diffusion_optimizer = Adam::new(diffusion.parameters(), 1e-4)?;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            let (data, _) = batch;

            // Project data to latent space (simplified)
            let latent_data = data.narrow(1, 0, latent_dim)?;

            let loss = diffusion.loss(&latent_data)?;

            diffusion_optimizer.zero_grad();
            loss.backward()?;
            diffusion_optimizer.step()?;

            total_loss += loss.item::<f32>()? as f64;
            batch_count += 1;

            if batch_count % 5 == 0 {
                println!("  Batch {}: Loss = {:.6}", batch_count, loss.item::<f32>()?);
                break; // Limit batches for demo
            }
        }

        println!(
            "Diffusion Epoch {}: Avg Loss = {:.6}",
            epoch + 1,
            total_loss / batch_count as f64
        );
    }

    // Generation and Evaluation
    println!("\n--- Model Evaluation and Generation ---");

    // VAE Generation
    println!("VAE Generation:");
    let vae_samples = vae.sample(8)?;
    println!(
        "  Generated samples shape: {:?}",
        vae_samples.shape().dims()
    );

    // GAN Generation
    println!("GAN Generation:");
    let gan_samples = gan.generate(8)?;
    println!(
        "  Generated samples shape: {:?}",
        gan_samples.shape().dims()
    );

    // Flow Generation
    println!("Flow Generation:");
    let z = randn(&[8, latent_dim])?;
    let (flow_samples, _) = flow.forward(&z)?;
    println!(
        "  Generated samples shape: {:?}",
        flow_samples.shape().dims()
    );

    // Diffusion Generation
    println!("Diffusion Generation:");
    let diffusion_samples = diffusion.sample(&[8, latent_dim])?;
    println!(
        "  Generated samples shape: {:?}",
        diffusion_samples.shape().dims()
    );

    // Interpolation demo (VAE)
    println!("\nVAE Interpolation:");
    let sample1 = randn(&[1, data_dim])?;
    let sample2 = randn(&[1, data_dim])?;
    let interpolations = vae.interpolate(&sample1, &sample2, 5)?;
    println!("  Generated {} interpolation steps", interpolations.len());

    println!("\n=== Generative Models Demo Complete ===");

    Ok(())
}

/// Create synthetic dataset for demonstration
fn create_synthetic_dataset(size: usize, dim: usize) -> Result<TensorDataset> {
    let mut data = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..size {
        // Generate synthetic data (mixture of Gaussians)
        let sample = if thread_rng().gen::<f32>() < 0.5 {
            randn(&[dim])?.add(&tensor![1.0]?)?
        } else {
            randn(&[dim])?.sub(&tensor![1.0]?)?
        };

        let target = tensor![0i64]?; // Dummy target

        data.push(sample);
        targets.push(target);
    }

    // Combine data and targets into a single vector for TensorDataset
    let mut combined_tensors = Vec::new();
    combined_tensors.extend(data);
    combined_tensors.extend(targets);
    Ok(TensorDataset::from_tensors(combined_tensors))
}

fn main() -> Result<()> {
    run_generative_models_demo()?;
    Ok(())
}
