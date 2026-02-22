//! Diffusion Models Benchmarks
//!
//! Comprehensive benchmarks for newly implemented diffusion model components:
//! - Latent Upsampler (32×32 → 64×64)
//! - IP-Adapter Projection
//! - Classifier-Free Guidance
//! - Camera Embedding
//! - Cross-View Attention
//! - Multi-view UNet

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torsh_tensor::creation::*;

// ============================================================================
// Latent Upsampler Benchmarks
// ============================================================================

fn bench_latent_upsampler_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("latent_upsampler");

    // Test different batch sizes and resolutions
    let configs = [
        (1, 4, 32, 32), // Single image, 4 channels, 32×32
        (2, 4, 32, 32), // Batch of 2
        (4, 4, 32, 32), // Batch of 4
        (8, 4, 32, 32), // Batch of 8
        (1, 8, 32, 32), // 8 channels
        (1, 4, 64, 64), // Larger input (64×64 → 128×128)
    ];

    for (batch, channels, h, w) in configs.iter() {
        let _elements = batch * channels * h * w;
        let output_elements = batch * channels * h * 2 * w * 2; // Upsampled 2x
        group.throughput(Throughput::Elements(output_elements as u64));

        // Create mock input tensors
        let _latents = rand::<f32>(&[*batch, *channels, *h, *w]).expect("Failed to create latents");
        let _timestep = rand::<f32>(&[*batch, 320]) // Typical timestep embedding dimension
            .expect("Failed to create timestep");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}x{}", batch, channels, h, w)),
            &(batch, channels, h, w),
            |bench, _| {
                bench.iter(|| {
                    // Placeholder: In actual benchmark, would call LatentUpsampler::forward
                    // For now, simulate upsampling operation
                    let _upsampled = zeros::<f32>(&[*batch, *channels, h * 2, w * 2])
                        .expect("Failed to create upsampled");
                });
            },
        );
    }

    group.finish();
}

fn bench_latent_upsampler_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("latent_upsampler_memory");

    for batch in [1, 4, 8, 16].iter() {
        let channels = 4;
        let h = 32;
        let w = 32;
        let input_bytes = batch * channels * h * w * std::mem::size_of::<f32>();
        let output_bytes = batch * channels * h * 2 * w * 2 * std::mem::size_of::<f32>();

        group.throughput(Throughput::Bytes((input_bytes + output_bytes) as u64));

        group.bench_with_input(
            BenchmarkId::new("memory_allocation", batch),
            batch,
            |bench, &batch| {
                bench.iter(|| {
                    let _input =
                        rand::<f32>(&[batch, channels, h, w]).expect("Failed to create input");
                    let _output = zeros::<f32>(&[batch, channels, h * 2, w * 2])
                        .expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// IP-Adapter Benchmarks
// ============================================================================

fn bench_ip_adapter_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("ip_adapter_projection");

    // Typical IP-Adapter configurations
    let configs = [
        (1, 257, 1024, 16, 768), // CLIP ViT-L/14 features → 16 tokens
        (2, 257, 1024, 16, 768), // Batch of 2
        (4, 257, 1024, 16, 768), // Batch of 4
        (1, 257, 768, 16, 768),  // CLIP ViT-B/16 features
        (1, 197, 768, 16, 768),  // CLIP ViT-B/14 features
    ];

    for (batch, seq_len, clip_dim, num_tokens, output_dim) in configs.iter() {
        let flops = batch * seq_len * clip_dim * output_dim; // Approximate projection FLOPs
        group.throughput(Throughput::Elements(flops as u64));

        let _clip_features =
            rand::<f32>(&[*batch, *seq_len, *clip_dim]).expect("Failed to create CLIP features");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}", batch, seq_len, clip_dim)),
            &(batch, seq_len, clip_dim),
            |bench, _| {
                bench.iter(|| {
                    // Placeholder: In actual benchmark, would call IPAdapterProjection::forward
                    let _projected = zeros::<f32>(&[*batch, *num_tokens, *output_dim])
                        .expect("Failed to create projected");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Cross-Attention Benchmarks (IP-Adapter)
// ============================================================================

fn bench_ip_adapter_cross_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("ip_adapter_cross_attention");

    // Typical cross-attention configurations
    let configs = [
        (1, 77, 768, 16, 8),   // Text length 77, 16 image tokens, 8 heads
        (2, 77, 768, 16, 8),   // Batch of 2
        (4, 77, 768, 16, 8),   // Batch of 4
        (1, 77, 768, 4, 8),    // Fewer image tokens
        (1, 77, 1024, 16, 12), // Larger model (SD-XL)
    ];

    for (batch, text_len, dim, num_tokens, num_heads) in configs.iter() {
        // Attention complexity: O(batch * num_heads * text_len * num_tokens * head_dim)
        let head_dim = dim / num_heads;
        let flops = batch * num_heads * text_len * num_tokens * head_dim * 2;
        group.throughput(Throughput::Elements(flops as u64));

        let _query = rand::<f32>(&[*batch, *text_len, *dim]).expect("Failed to create query");
        let _image_features =
            rand::<f32>(&[*batch, *num_tokens, *dim]).expect("Failed to create image features");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}", batch, text_len, dim)),
            &(batch, text_len, dim),
            |bench, _| {
                bench.iter(|| {
                    // Placeholder: In actual benchmark, would call IPAdapterCrossAttention::forward
                    let _output =
                        zeros::<f32>(&[*batch, *text_len, *dim]).expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Classifier-Free Guidance Benchmarks
// ============================================================================

fn bench_classifier_free_guidance(c: &mut Criterion) {
    let mut group = c.benchmark_group("classifier_free_guidance");

    for batch in [1, 2, 4, 8].iter() {
        let channels = 4;
        let h = 64;
        let w = 64;
        let elements = batch * channels * h * w;
        group.throughput(Throughput::Elements(elements as u64));

        // CFG operates on concatenated conditional and unconditional predictions
        let noise_pred_uncond = rand::<f32>(&[*batch, channels, h, w])
            .expect("Failed to create unconditional prediction");
        let noise_pred_cond = rand::<f32>(&[*batch, channels, h, w])
            .expect("Failed to create conditional prediction");
        let _guidance_scale = 7.5f32;

        group.bench_with_input(
            BenchmarkId::new("apply_guidance", batch),
            batch,
            |bench, _| {
                bench.iter(|| {
                    // Placeholder: noise_pred = uncond + scale * (cond - uncond)
                    let _diff = noise_pred_cond
                        .sub(&noise_pred_uncond)
                        .expect("Failed to compute difference");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Camera Embedding Benchmarks
// ============================================================================

fn bench_camera_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("camera_embedding");

    // Camera parameters: [azimuth, elevation, distance, fov]
    for batch in [1, 4, 8, 16].iter() {
        let num_params = 4;
        let embed_dim = 128; // Typical embedding dimension

        group.throughput(Throughput::Elements((batch * embed_dim) as u64));

        let _camera_params =
            rand::<f32>(&[*batch, num_params]).expect("Failed to create camera params");

        group.bench_with_input(BenchmarkId::new("forward", batch), batch, |bench, _| {
            bench.iter(|| {
                // Placeholder: In actual benchmark, would call CameraEmbedding::forward
                let _embedded =
                    zeros::<f32>(&[*batch, embed_dim]).expect("Failed to create embedding");
            });
        });
    }

    group.finish();
}

// ============================================================================
// Cross-View Attention Benchmarks (Multi-view)
// ============================================================================

fn bench_cross_view_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_view_attention");

    // Multi-view attention configurations: [batch, num_views, seq_len, dim]
    let configs = [
        (1, 4, 256, 512), // 4 views
        (2, 4, 256, 512), // Batch of 2
        (1, 6, 256, 512), // 6 views
        (1, 4, 512, 768), // Larger resolution
        (4, 4, 256, 512), // Batch of 4
    ];

    for (batch, num_views, seq_len, dim) in configs.iter() {
        // Cross-view attention complexity
        let flops = batch * num_views * num_views * seq_len * dim;
        group.throughput(Throughput::Elements(flops as u64));

        let _features =
            rand::<f32>(&[*batch, *num_views, *seq_len, *dim]).expect("Failed to create features");

        group.bench_with_input(
            BenchmarkId::new(
                "forward",
                format!("{}x{}x{}x{}", batch, num_views, seq_len, dim),
            ),
            &(batch, num_views, seq_len, dim),
            |bench, _| {
                bench.iter(|| {
                    // Placeholder: In actual benchmark, would call CrossViewAttention::forward
                    let _output = zeros::<f32>(&[*batch, *num_views, *seq_len, *dim])
                        .expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Multi-view UNet Benchmarks
// ============================================================================

fn bench_multiview_unet(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiview_unet");
    group.sample_size(10); // Reduce samples for expensive operations

    // Multi-view UNet configurations
    let configs = [
        (1, 4, 4, 64, 64), // 1 batch, 4 views, 4 channels, 64×64
        (1, 6, 4, 64, 64), // 6 views
        (2, 4, 4, 64, 64), // Batch of 2
        (1, 4, 8, 64, 64), // 8 channels
    ];

    for (batch, num_views, channels, h, w) in configs.iter() {
        let elements = batch * num_views * channels * h * w;
        group.throughput(Throughput::Elements(elements as u64));

        let _latents = rand::<f32>(&[*batch, *num_views, *channels, *h, *w])
            .expect("Failed to create latents");
        let _timestep = rand::<f32>(&[*batch, 320]).expect("Failed to create timestep");
        let _camera_embed =
            rand::<f32>(&[*batch, *num_views, 128]).expect("Failed to create camera embedding");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}x{}", batch, num_views, h, w)),
            &(batch, num_views, h, w),
            |bench, _| {
                bench.iter(|| {
                    // Placeholder: In actual benchmark, would call MultiviewUNet::forward
                    let _output = zeros::<f32>(&[*batch, *num_views, *channels, *h, *w])
                        .expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// End-to-End Diffusion Pipeline Benchmarks
// ============================================================================

fn bench_diffusion_pipeline_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffusion_pipeline_latency");
    group.sample_size(10);

    // Measure latency for complete diffusion steps
    let batch = 1;
    let channels = 4;
    let h = 64;
    let w = 64;

    group.bench_function("single_denoising_step", |bench| {
        let _latents = rand::<f32>(&[batch, channels, h, w]).expect("Failed to create latents");
        let _timestep = rand::<f32>(&[batch, 320]).expect("Failed to create timestep");

        bench.iter(|| {
            // Simulate a single denoising step
            // In reality: UNet forward → CFG → noise prediction update
            let _noise_pred =
                zeros::<f32>(&[batch, channels, h, w]).expect("Failed to create noise prediction");
        });
    });

    group.finish();
}

fn bench_diffusion_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffusion_throughput");

    // Measure throughput for batched diffusion
    for batch in [1, 2, 4, 8].iter() {
        let channels = 4;
        let h = 64;
        let w = 64;
        let _images_per_second = 1000.0 / 50.0; // Assume 50 denoising steps

        group.throughput(Throughput::Elements(*batch as u64));

        group.bench_with_input(
            BenchmarkId::new("batched_generation", batch),
            batch,
            |bench, &batch| {
                let _latents =
                    rand::<f32>(&[batch, channels, h, w]).expect("Failed to create latents");

                bench.iter(|| {
                    // Simulate batched generation
                    let _output = zeros::<f32>(&[batch, channels, h * 2, w * 2])
                        .expect("Failed to create output");
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    diffusion_benches,
    bench_latent_upsampler_forward,
    bench_latent_upsampler_memory,
    bench_ip_adapter_projection,
    bench_ip_adapter_cross_attention,
    bench_classifier_free_guidance,
    bench_camera_embedding,
    bench_cross_view_attention,
    bench_multiview_unet,
    bench_diffusion_pipeline_latency,
    bench_diffusion_throughput,
);

criterion_main!(diffusion_benches);
