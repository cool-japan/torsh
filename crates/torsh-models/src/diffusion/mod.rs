//! Diffusion Models and Components
//!
//! This module provides advanced diffusion model components for ToRSh,
//! including multi-view generation, identity-preserving conditioning,
//! and latent upsampling capabilities.
//!
//! # Components
//!
//! - [`latent_upsampler`]: U-Net for latent space upsampling (32×32 → 64×64)
//! - [`ip_adapter_projection`]: IP-Adapter image feature projection for identity preservation
//! - [`guidance`]: Classifier-Free Guidance utilities for quality improvement
//! - [`camera_embedding`]: Camera parameter encoding for multi-view synthesis
//! - [`multiview_attention`]: Cross-view attention for consistent multi-view generation
//! - [`multiview_unet`]: Multi-view aware diffusion UNet with camera conditioning
//!
//! # Feature Flag
//!
//! All components in this module require the `diffusion_extended` feature flag:
//!
//! ```toml
//! [dependencies]
//! torsh-models = { version = "0.1", features = ["diffusion_extended"] }
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_models::diffusion::{LatentUpsampler, MultiviewUNet};
//!
//! // Create latent upsampler
//! let upsampler = LatentUpsampler::new(channels, timestep_dim)?;
//!
//! // Upsample latents
//! let upsampled = upsampler.forward(&latents, timestep)?;
//! ```

pub mod camera_embedding;
pub mod guidance;
pub mod ip_adapter_projection;
pub mod latent_upsampler;
pub mod multiview_attention;
pub mod multiview_unet;

// Re-export main types for convenience
pub use camera_embedding::CameraEmbedding;
pub use guidance::{apply_classifier_free_guidance, prepare_cfg_batch, split_cfg_batch};
pub use ip_adapter_projection::IPAdapterProjection;
pub use latent_upsampler::LatentUpsampler;
pub use multiview_attention::CrossViewAttention;
pub use multiview_unet::MultiviewUNet;
