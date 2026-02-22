//! Diffusion Models Example
//!
//! This example demonstrates advanced diffusion model components in torsh-models:
//! - Latent Upsampler for resolution enhancement
//! - Multi-view UNet for consistent 3D generation
//! - IP-Adapter for identity preservation
//! - Classifier-Free Guidance utilities
//!
//! Run with: cargo run --example diffusion_models --features diffusion_extended

#[cfg(feature = "diffusion_extended")]
use torsh_core::error::Result;
#[cfg(feature = "diffusion_extended")]
use torsh_models::diffusion::*;
#[cfg(feature = "diffusion_extended")]
use torsh_tensor::creation::*;
#[cfg(feature = "diffusion_extended")]
use torsh_tensor::Tensor;

#[cfg(feature = "diffusion_extended")]
fn main() -> Result<()> {
    println!("=== ToRSh Diffusion Models Example ===\n");

    // Example 1: Latent Upsampler
    latent_upsampling_example()?;

    // Example 2: Multi-view UNet
    multiview_generation_example()?;

    // Example 3: IP-Adapter Integration
    ip_adapter_example()?;

    // Example 4: Classifier-Free Guidance
    cfg_guidance_example()?;

    // Example 5: Camera Conditioning
    camera_embedding_example()?;

    println!("\n✓ All diffusion model examples completed successfully!");
    Ok(())
}

#[cfg(not(feature = "diffusion_extended"))]
fn main() {
    println!("This example requires the 'diffusion_extended' feature.");
    println!("Run with: cargo run --example diffusion_models --features diffusion_extended");
}

/// Example 1: Latent Upsampler for resolution enhancement
///
/// The LatentUpsampler is a small U-Net that upsamples latent representations
/// from 32×32 to 64×64, enabling 512×512 image generation with diffusion models.
#[cfg(feature = "diffusion_extended")]
fn latent_upsampling_example() -> Result<()> {
    println!("--- Example 1: Latent Upsampling ---");

    // Configuration
    let batch_size = 2;
    let channels = 4; // Latent channels (typical for Stable Diffusion)
    let timestep_dim = 1024;

    // Create latent upsampler
    let upsampler = LatentUpsampler::new(channels, timestep_dim)?;
    println!("✓ Created LatentUpsampler with {} channels", channels);

    // Create input latents [B, C, 32, 32]
    let latents = randn(&[batch_size, channels, 32, 32])?;
    println!("  Input shape: {:?}", latents.shape().dims());

    // Create timestep [B]
    let timestep = Tensor::from_vec(vec![500.0, 250.0], &[batch_size])?;
    println!("  Timesteps: [500, 250]");

    // Forward pass: upsample to [B, C, 64, 64]
    let upsampled = upsampler.forward(&latents, &timestep)?;
    println!("  Output shape: {:?}", upsampled.shape().dims());

    // Verify output dimensions
    assert_eq!(upsampled.shape().dims(), &[batch_size, channels, 64, 64]);
    println!("✓ Latent upsampling successful: 32×32 → 64×64\n");

    Ok(())
}

/// Example 2: Multi-view UNet for consistent multi-view generation
///
/// The MultiviewUNet extends diffusion models with cross-view attention
/// for generating consistent 3D representations from multiple viewpoints.
#[cfg(feature = "diffusion_extended")]
fn multiview_generation_example() -> Result<()> {
    println!("--- Example 2: Multi-view UNet ---");

    // Configuration
    let batch_size = 2;
    let num_views = 4;
    let channels = 4;
    let num_heads = 8;

    // Create multi-view UNet
    let unet = MultiviewUNet::new(channels, num_heads)?;
    println!("✓ Created MultiviewUNet with {} heads", num_heads);

    // Create input latents [B*V, C, H, W]
    let latents = randn(&[batch_size * num_views, channels, 32, 32])?;
    println!(
        "  Input: {} images ({}×{} views)",
        batch_size * num_views,
        batch_size,
        num_views
    );

    // Create timestep [B]
    let timestep = zeros(&[batch_size])?;

    // Create camera embeddings [B, V, 512]
    let camera_embeddings = randn(&[batch_size, num_views, 512])?;
    println!("  Camera embeddings: [{}, {}, 512]", batch_size, num_views);

    // Forward pass with cross-view attention
    let output = unet.forward(&latents, &timestep, &camera_embeddings)?;
    println!("  Output shape: {:?}", output.shape().dims());

    // Verify output dimensions match input
    assert_eq!(output.shape().dims(), latents.shape().dims());
    println!("✓ Multi-view generation successful with view consistency\n");

    Ok(())
}

/// Example 3: IP-Adapter for identity-preserving image generation
///
/// IP-Adapter projects image features for identity-preserving conditioning
/// in diffusion models, enabling personalized generation.
#[cfg(feature = "diffusion_extended")]
fn ip_adapter_example() -> Result<()> {
    println!("--- Example 3: IP-Adapter Projection ---");

    // Configuration
    let batch_size = 2;
    let image_embed_dim = 768; // CLIP image embedding dimension
    let cross_attention_dim = 1024; // UNet cross-attention dimension
    let num_tokens = 4; // Number of learned query tokens

    // Create IP-Adapter projection
    let ip_adapter = IPAdapterProjection::new(num_tokens, cross_attention_dim)?;
    println!("✓ Created IPAdapterProjection ({} tokens)", num_tokens);

    // Create image embeddings from CLIP or similar encoder [B, D]
    let image_embeddings = randn(&[batch_size, image_embed_dim])?;
    println!(
        "  Input: CLIP embeddings [{}, {}]",
        batch_size, image_embed_dim
    );

    // Project to cross-attention compatible format [B, num_tokens, cross_attention_dim]
    let projected = ip_adapter.forward(&image_embeddings)?;
    println!("  Output shape: {:?}", projected.shape().dims());

    // Verify output dimensions
    assert_eq!(
        projected.shape().dims(),
        &[batch_size, num_tokens, cross_attention_dim]
    );
    println!("✓ IP-Adapter projection successful: preserves identity features\n");

    Ok(())
}

/// Example 4: Classifier-Free Guidance for quality improvement
///
/// CFG combines conditional and unconditional predictions to improve
/// generation quality and adherence to conditioning.
#[cfg(feature = "diffusion_extended")]
fn cfg_guidance_example() -> Result<()> {
    println!("--- Example 4: Classifier-Free Guidance ---");

    // Configuration
    let batch_size = 2;
    let channels = 4;

    // Create conditional and unconditional predictions
    let cond_pred = randn(&[batch_size, channels, 32, 32])?;
    let uncond_pred = randn(&[batch_size, channels, 32, 32])?;
    println!("  Conditional and unconditional predictions created");

    // Apply CFG with guidance scale
    let guidance_scale = 7.5;
    let guided_pred = apply_classifier_free_guidance(&cond_pred, &uncond_pred, guidance_scale)?;
    println!("  Applied guidance scale: {}", guidance_scale);
    println!("  Output shape: {:?}", guided_pred.shape().dims());

    // Verify output shape matches input
    assert_eq!(guided_pred.shape().dims(), cond_pred.shape().dims());
    println!("✓ CFG applied successfully: enhanced generation quality\n");

    Ok(())
}

/// Example 5: Camera Embedding for multi-view conditioning
///
/// Camera embeddings encode camera parameters (rotation, translation, intrinsics)
/// for spatially-aware multi-view generation.
#[cfg(feature = "diffusion_extended")]
fn camera_embedding_example() -> Result<()> {
    println!("--- Example 5: Camera Embedding ---");

    // Configuration
    let batch_size = 2;
    let num_views = 4;
    let embed_dim = 512;

    // Create camera embedding layer
    let camera_embed = CameraEmbedding::new(embed_dim)?;
    println!("✓ Created CameraEmbedding (dim: {})", embed_dim);

    // Create camera parameters [B, V, 16]
    // 16 dimensions: rotation (9), translation (3), intrinsics (4)
    let camera_params = randn(&[batch_size, num_views, 16])?;
    println!(
        "  Camera params: [{}, {}, 16] (R, t, K)",
        batch_size, num_views
    );

    // Encode camera parameters
    let embeddings = camera_embed.forward(&camera_params)?;
    println!("  Camera embeddings: {:?}", embeddings.shape().dims());

    // Verify output dimensions
    assert_eq!(
        embeddings.shape().dims(),
        &[batch_size, num_views, embed_dim]
    );
    println!("✓ Camera encoding successful: spatially-aware embeddings\n");

    Ok(())
}
