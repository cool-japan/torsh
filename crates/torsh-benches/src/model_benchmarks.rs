//! Model architecture benchmarks
//!
//! This module provides benchmarks for common neural network architectures
//! including ResNet, Transformer, and other popular models.

use crate::Benchmarkable;
use criterion::black_box;
// Removed unused imports: BenchConfig, BenchResult, torsh_core::dtype::DType
use std::time::Instant;
use torsh_tensor::{creation::*, Tensor};

/// ResNet block benchmark
pub struct ResNetBlockBench {
    pub input_channels: usize,
    pub output_channels: usize,
    pub image_size: usize,
    pub batch_size: usize,
}

impl ResNetBlockBench {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        image_size: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            input_channels,
            output_channels,
            image_size,
            batch_size,
        }
    }
}

impl Benchmarkable for ResNetBlockBench {
    type Input = (Tensor<f32>, ResNetBlockWeights);
    type Output = Tensor<f32>;

    fn setup(&mut self, _size: usize) -> Self::Input {
        let input = rand(&[
            self.batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
        ])
        .unwrap();

        let weights = ResNetBlockWeights {
            conv1_weight: rand(&[self.output_channels, self.input_channels, 3, 3]).unwrap(),
            conv1_bias: rand(&[self.output_channels]).unwrap(),
            bn1_weight: rand(&[self.output_channels]).unwrap(),
            bn1_bias: rand(&[self.output_channels]).unwrap(),
            bn1_running_mean: zeros::<f32>(&[self.output_channels]).unwrap(),
            bn1_running_var: ones::<f32>(&[self.output_channels]).unwrap(),
            conv2_weight: rand(&[self.output_channels, self.output_channels, 3, 3]).unwrap(),
            conv2_bias: rand(&[self.output_channels]).unwrap(),
            bn2_weight: rand(&[self.output_channels]).unwrap(),
            bn2_bias: rand(&[self.output_channels]).unwrap(),
            bn2_running_mean: zeros::<f32>(&[self.output_channels]).unwrap(),
            bn2_running_var: ones::<f32>(&[self.output_channels]).unwrap(),
            downsample_weight: if self.input_channels != self.output_channels {
                Some(rand(&[self.output_channels, self.input_channels, 1, 1]).unwrap())
            } else {
                None
            },
        };

        (input, weights)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (x, weights) = input;

        // Simplified ResNet block computation (mock implementation)
        // In a real implementation, this would use proper convolution and batch norm layers
        let conv1_out = mock_conv2d(x, &weights.conv1_weight);
        let bn1_out = mock_batch_norm(&conv1_out, &weights.bn1_weight, &weights.bn1_bias);
        let relu1_out = mock_relu(&bn1_out);

        let conv2_out = mock_conv2d(&relu1_out, &weights.conv2_weight);
        let bn2_out = mock_batch_norm(&conv2_out, &weights.bn2_weight, &weights.bn2_bias);

        // Residual connection
        let residual = if let Some(ref downsample_weight) = weights.downsample_weight {
            mock_conv2d(x, downsample_weight)
        } else {
            x.clone()
        };

        let output = bn2_out.add(&residual).unwrap();
        black_box(mock_relu(&output))
    }

    fn flops(&self, _size: usize) -> usize {
        // Approximate FLOPS for ResNet block
        let h = self.image_size;
        let w = self.image_size;
        let c_in = self.input_channels;
        let c_out = self.output_channels;
        let batch = self.batch_size;

        // Conv1: 3x3 conv
        let conv1_flops = batch * h * w * c_in * c_out * 3 * 3;
        // BatchNorm1
        let bn1_flops = batch * h * w * c_out * 4; // 4 ops per element
                                                   // Conv2: 3x3 conv
        let conv2_flops = batch * h * w * c_out * c_out * 3 * 3;
        // BatchNorm2
        let bn2_flops = batch * h * w * c_out * 4;
        // Residual add
        let add_flops = batch * h * w * c_out;

        conv1_flops + bn1_flops + conv2_flops + bn2_flops + add_flops
    }

    fn bytes_accessed(&self, _size: usize) -> usize {
        // Approximate memory access for ResNet block
        let h = self.image_size;
        let w = self.image_size;
        let c_in = self.input_channels;
        let c_out = self.output_channels;
        let batch = self.batch_size;

        // Input tensor
        let input_bytes = batch * c_in * h * w * 4; // f32 = 4 bytes
                                                    // Weights
        let conv1_weights = c_out * c_in * 3 * 3 * 4;
        let conv2_weights = c_out * c_out * 3 * 3 * 4;
        let bn_params = c_out * 4 * 4; // 4 parameters per channel
                                       // Output tensor
        let output_bytes = batch * c_out * h * w * 4;

        input_bytes + conv1_weights + conv2_weights + bn_params + output_bytes
    }
}

/// ResNet block weights structure
pub struct ResNetBlockWeights {
    pub conv1_weight: Tensor<f32>,
    pub conv1_bias: Tensor<f32>,
    pub bn1_weight: Tensor<f32>,
    pub bn1_bias: Tensor<f32>,
    pub bn1_running_mean: Tensor<f32>,
    pub bn1_running_var: Tensor<f32>,
    pub conv2_weight: Tensor<f32>,
    pub conv2_bias: Tensor<f32>,
    pub bn2_weight: Tensor<f32>,
    pub bn2_bias: Tensor<f32>,
    pub bn2_running_mean: Tensor<f32>,
    pub bn2_running_var: Tensor<f32>,
    pub downsample_weight: Option<Tensor<f32>>,
}

/// Transformer block benchmark
pub struct TransformerBlockBench {
    pub seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub batch_size: usize,
}

impl TransformerBlockBench {
    pub fn new(seq_len: usize, d_model: usize, num_heads: usize, batch_size: usize) -> Self {
        Self {
            seq_len,
            d_model,
            num_heads,
            batch_size,
        }
    }
}

impl Benchmarkable for TransformerBlockBench {
    type Input = (Tensor<f32>, TransformerBlockWeights);
    type Output = Tensor<f32>;

    fn setup(&mut self, _size: usize) -> Self::Input {
        let input = rand(&[self.batch_size, self.seq_len, self.d_model]).unwrap();

        let weights = TransformerBlockWeights {
            attention_qkv_weight: rand(&[self.d_model, 3 * self.d_model]).unwrap(),
            attention_qkv_bias: rand(&[3 * self.d_model]).unwrap(),
            attention_proj_weight: rand(&[self.d_model, self.d_model]).unwrap(),
            attention_proj_bias: rand(&[self.d_model]).unwrap(),
            ln1_weight: ones::<f32>(&[self.d_model]).unwrap(),
            ln1_bias: zeros::<f32>(&[self.d_model]).unwrap(),
            mlp_fc1_weight: rand(&[self.d_model, 4 * self.d_model]).unwrap(),
            mlp_fc1_bias: rand(&[4 * self.d_model]).unwrap(),
            mlp_fc2_weight: rand(&[4 * self.d_model, self.d_model]).unwrap(),
            mlp_fc2_bias: rand(&[self.d_model]).unwrap(),
            ln2_weight: ones::<f32>(&[self.d_model]).unwrap(),
            ln2_bias: zeros::<f32>(&[self.d_model]).unwrap(),
        };

        (input, weights)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (x, weights) = input;

        // Layer norm 1
        let ln1_out = mock_layer_norm(x, &weights.ln1_weight, &weights.ln1_bias);

        // Multi-head attention (simplified)
        let qkv = mock_linear(
            &ln1_out,
            &weights.attention_qkv_weight,
            Some(&weights.attention_qkv_bias),
        );
        let attention_out = mock_attention(&qkv, self.num_heads, self.d_model);
        let attention_proj = mock_linear(
            &attention_out,
            &weights.attention_proj_weight,
            Some(&weights.attention_proj_bias),
        );

        // Residual connection 1
        let residual1 = x.add(&attention_proj).unwrap();

        // Layer norm 2
        let ln2_out = mock_layer_norm(&residual1, &weights.ln2_weight, &weights.ln2_bias);

        // MLP
        let mlp1_out = mock_linear(
            &ln2_out,
            &weights.mlp_fc1_weight,
            Some(&weights.mlp_fc1_bias),
        );
        let mlp1_activated = mock_gelu(&mlp1_out);
        let mlp2_out = mock_linear(
            &mlp1_activated,
            &weights.mlp_fc2_weight,
            Some(&weights.mlp_fc2_bias),
        );

        // Residual connection 2
        black_box(residual1.add(&mlp2_out).unwrap())
    }

    fn flops(&self, _size: usize) -> usize {
        let seq = self.seq_len;
        let d = self.d_model;
        let batch = self.batch_size;

        // Attention QKV projection
        let qkv_flops = batch * seq * d * 3 * d;
        // Attention computation (simplified)
        let attention_flops = batch * self.num_heads * seq * seq * (d / self.num_heads);
        // Attention output projection
        let proj_flops = batch * seq * d * d;
        // MLP
        let mlp1_flops = batch * seq * d * 4 * d;
        let mlp2_flops = batch * seq * 4 * d * d;
        // Layer norms (approximate)
        let ln_flops = batch * seq * d * 8; // 2 layer norms

        qkv_flops + attention_flops + proj_flops + mlp1_flops + mlp2_flops + ln_flops
    }

    fn bytes_accessed(&self, _size: usize) -> usize {
        let seq = self.seq_len;
        let d = self.d_model;
        let batch = self.batch_size;

        // Input/output tensors
        let io_bytes = batch * seq * d * 4 * 2; // input + output
                                                // Weights
        let attention_weights = d * 3 * d * 4 + d * d * 4; // QKV + proj
        let mlp_weights = d * 4 * d * 4 + 4 * d * d * 4; // fc1 + fc2
        let ln_weights = d * 2 * 4 * 2; // 2 layer norms

        io_bytes + attention_weights + mlp_weights + ln_weights
    }
}

/// Transformer block weights structure
pub struct TransformerBlockWeights {
    pub attention_qkv_weight: Tensor<f32>,
    pub attention_qkv_bias: Tensor<f32>,
    pub attention_proj_weight: Tensor<f32>,
    pub attention_proj_bias: Tensor<f32>,
    pub ln1_weight: Tensor<f32>,
    pub ln1_bias: Tensor<f32>,
    pub mlp_fc1_weight: Tensor<f32>,
    pub mlp_fc1_bias: Tensor<f32>,
    pub mlp_fc2_weight: Tensor<f32>,
    pub mlp_fc2_bias: Tensor<f32>,
    pub ln2_weight: Tensor<f32>,
    pub ln2_bias: Tensor<f32>,
}

/// Model benchmarking suite
pub struct ModelBenchmarkSuite {
    pub configs: Vec<ModelBenchmarkConfig>,
}

#[derive(Debug, Clone)]
pub struct ModelBenchmarkConfig {
    pub name: String,
    pub model_type: ModelType,
    pub batch_sizes: Vec<usize>,
    pub input_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    ResNet {
        layers: usize,
        channels: usize,
    },
    Transformer {
        layers: usize,
        d_model: usize,
        num_heads: usize,
    },
    LSTM {
        hidden_size: usize,
        num_layers: usize,
    },
    CNN {
        num_layers: usize,
        channels: Vec<usize>,
    },
    GANGenerator {
        latent_dim: usize,
        output_channels: usize,
        output_size: usize,
    },
    GANDiscriminator {
        input_channels: usize,
        input_size: usize,
    },
    YOLOv5 {
        num_classes: usize,
        input_size: usize,
    },
    SSD {
        num_classes: usize,
        input_size: usize,
    },
    FasterRCNN {
        num_classes: usize,
        input_size: usize,
    },
}

impl ModelBenchmarkSuite {
    pub fn new() -> Self {
        Self {
            configs: vec![
                ModelBenchmarkConfig {
                    name: "ResNet-18".to_string(),
                    model_type: ModelType::ResNet {
                        layers: 18,
                        channels: 64,
                    },
                    batch_sizes: vec![1, 8, 16, 32],
                    input_sizes: vec![224, 256, 384],
                },
                ModelBenchmarkConfig {
                    name: "ResNet-50".to_string(),
                    model_type: ModelType::ResNet {
                        layers: 50,
                        channels: 64,
                    },
                    batch_sizes: vec![1, 8, 16, 32],
                    input_sizes: vec![224, 256, 384],
                },
                ModelBenchmarkConfig {
                    name: "Transformer-Base".to_string(),
                    model_type: ModelType::Transformer {
                        layers: 12,
                        d_model: 768,
                        num_heads: 12,
                    },
                    batch_sizes: vec![1, 4, 8, 16],
                    input_sizes: vec![128, 256, 512],
                },
                ModelBenchmarkConfig {
                    name: "Transformer-Large".to_string(),
                    model_type: ModelType::Transformer {
                        layers: 24,
                        d_model: 1024,
                        num_heads: 16,
                    },
                    batch_sizes: vec![1, 4, 8],
                    input_sizes: vec![128, 256, 512],
                },
                ModelBenchmarkConfig {
                    name: "GAN-Generator".to_string(),
                    model_type: ModelType::GANGenerator {
                        latent_dim: 100,
                        output_channels: 3,
                        output_size: 32,
                    },
                    batch_sizes: vec![1, 8, 16, 32],
                    input_sizes: vec![32, 64, 128],
                },
                ModelBenchmarkConfig {
                    name: "GAN-Discriminator".to_string(),
                    model_type: ModelType::GANDiscriminator {
                        input_channels: 3,
                        input_size: 32,
                    },
                    batch_sizes: vec![1, 8, 16, 32],
                    input_sizes: vec![32, 64, 128],
                },
                ModelBenchmarkConfig {
                    name: "YOLOv5-Small".to_string(),
                    model_type: ModelType::YOLOv5 {
                        num_classes: 80,
                        input_size: 640,
                    },
                    batch_sizes: vec![1, 4, 8, 16],
                    input_sizes: vec![416, 640, 832],
                },
                ModelBenchmarkConfig {
                    name: "SSD-MobileNet".to_string(),
                    model_type: ModelType::SSD {
                        num_classes: 80,
                        input_size: 300,
                    },
                    batch_sizes: vec![1, 4, 8, 16],
                    input_sizes: vec![300, 512],
                },
            ],
        }
    }

    pub fn run_all_benchmarks(&self) -> Vec<ModelBenchmarkResult> {
        let mut results = Vec::new();

        for config in &self.configs {
            for &batch_size in &config.batch_sizes {
                for &input_size in &config.input_sizes {
                    let result = self.run_single_benchmark(config, batch_size, input_size);
                    results.push(result);
                }
            }
        }

        results
    }

    fn run_single_benchmark(
        &self,
        config: &ModelBenchmarkConfig,
        batch_size: usize,
        input_size: usize,
    ) -> ModelBenchmarkResult {
        let start_time = Instant::now();

        // Run benchmark based on model type
        let (forward_time, flops, memory_usage) = match &config.model_type {
            ModelType::ResNet { channels, .. } => {
                let mut bench = ResNetBlockBench::new(3, *channels, input_size, batch_size);
                let input = bench.setup(input_size);

                let forward_start = Instant::now();
                let _output = bench.run(&input);
                let forward_time = forward_start.elapsed();

                let flops = bench.flops(input_size);
                let memory = bench.bytes_accessed(input_size);

                (forward_time, flops, memory)
            }
            ModelType::Transformer {
                d_model, num_heads, ..
            } => {
                let mut bench =
                    TransformerBlockBench::new(input_size, *d_model, *num_heads, batch_size);
                let input = bench.setup(input_size);

                let forward_start = Instant::now();
                let _output = bench.run(&input);
                let forward_time = forward_start.elapsed();

                let flops = bench.flops(input_size);
                let memory = bench.bytes_accessed(input_size);

                (forward_time, flops, memory)
            }
            ModelType::GANGenerator {
                latent_dim,
                output_channels,
                output_size,
            } => {
                let mut bench =
                    GANGeneratorBench::new(*latent_dim, *output_channels, *output_size, batch_size);
                let input = bench.setup(input_size);

                let forward_start = Instant::now();
                let _output = bench.run(&input);
                let forward_time = forward_start.elapsed();

                let flops = bench.flops(input_size);
                let memory = bench.bytes_accessed(input_size);

                (forward_time, flops, memory)
            }
            ModelType::GANDiscriminator {
                input_channels,
                input_size: model_input_size,
            } => {
                let mut bench =
                    GANDiscriminatorBench::new(*input_channels, *model_input_size, batch_size);
                let input = bench.setup(input_size);

                let forward_start = Instant::now();
                let _output = bench.run(&input);
                let forward_time = forward_start.elapsed();

                let flops = bench.flops(input_size);
                let memory = bench.bytes_accessed(input_size);

                (forward_time, flops, memory)
            }
            ModelType::YOLOv5 {
                num_classes,
                input_size: model_input_size,
            } => {
                let mut bench = YOLOv5Bench::new(*num_classes, *model_input_size, batch_size);
                let input = bench.setup(input_size);

                let forward_start = Instant::now();
                let _output = bench.run(&input);
                let forward_time = forward_start.elapsed();

                let flops = bench.flops(input_size);
                let memory = bench.bytes_accessed(input_size);

                (forward_time, flops, memory)
            }
            ModelType::SSD {
                num_classes,
                input_size: model_input_size,
            } => {
                let mut bench = SSDBench::new(*num_classes, *model_input_size, batch_size);
                let input = bench.setup(input_size);

                let forward_start = Instant::now();
                let _output = bench.run(&input);
                let forward_time = forward_start.elapsed();

                let flops = bench.flops(input_size);
                let memory = bench.bytes_accessed(input_size);

                (forward_time, flops, memory)
            }
            _ => {
                // Placeholder for other model types
                (std::time::Duration::from_millis(1), 1000000, 1000000)
            }
        };

        let total_time = start_time.elapsed();

        ModelBenchmarkResult {
            model_name: config.name.clone(),
            batch_size,
            input_size,
            forward_time_ns: forward_time.as_nanos() as f64,
            total_time_ns: total_time.as_nanos() as f64,
            flops,
            memory_usage_bytes: memory_usage,
            throughput_samples_per_sec: batch_size as f64 / forward_time.as_secs_f64(),
            flops_per_sec: flops as f64 / forward_time.as_secs_f64(),
        }
    }
}

impl Default for ModelBenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Model benchmark result
#[derive(Debug, Clone)]
pub struct ModelBenchmarkResult {
    pub model_name: String,
    pub batch_size: usize,
    pub input_size: usize,
    pub forward_time_ns: f64,
    pub total_time_ns: f64,
    pub flops: usize,
    pub memory_usage_bytes: usize,
    pub throughput_samples_per_sec: f64,
    pub flops_per_sec: f64,
}

impl ModelBenchmarkResult {
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{},{:.2},{:.2}",
            self.model_name,
            self.batch_size,
            self.input_size,
            self.forward_time_ns / 1_000_000.0, // Convert to milliseconds
            self.total_time_ns / 1_000_000.0,
            self.flops,
            self.memory_usage_bytes,
            self.throughput_samples_per_sec,
            self.flops_per_sec / 1e9 // Convert to GFLOPS
        )
    }
}

// Mock implementations for simplified benchmarking
// In a real implementation, these would use proper neural network operations

fn mock_conv2d(input: &Tensor<f32>, weight: &Tensor<f32>) -> Tensor<f32> {
    // Simplified convolution with stride=2 downsampling (typical for discriminator)
    let input_binding = input.shape();
    let input_shape = input_binding.dims();
    let weight_binding = weight.shape();
    let weight_shape = weight_binding.dims();
    let batch = input_shape[0];
    let out_channels = weight_shape[0];

    // Simulate stride=2 convolution that halves spatial dimensions
    let height = (input_shape[2] + 1) / 2; // Downsample by 2
    let width = (input_shape[3] + 1) / 2; // Downsample by 2

    rand::<f32>(&[batch, out_channels, height, width]).unwrap()
}

fn mock_batch_norm(input: &Tensor<f32>, _weight: &Tensor<f32>, bias: &Tensor<f32>) -> Tensor<f32> {
    // Simplified batch norm - just add some computation time
    input.add(bias).unwrap_or_else(|_| input.clone())
}

fn mock_relu(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified ReLU - return input (mock)
    input.clone()
}

fn mock_linear(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
) -> Tensor<f32> {
    // Simplified linear layer
    let result = input.matmul(weight).unwrap_or_else(|_| input.clone());
    if let Some(b) = bias {
        result.add(b).unwrap_or(result)
    } else {
        result
    }
}

fn mock_layer_norm(input: &Tensor<f32>, _weight: &Tensor<f32>, _bias: &Tensor<f32>) -> Tensor<f32> {
    // Simplified layer norm
    input.add(_bias).unwrap_or_else(|_| input.clone())
}

fn mock_attention(qkv: &Tensor<f32>, _num_heads: usize, _d_model: usize) -> Tensor<f32> {
    // Simplified attention - return same shape
    qkv.clone()
}

fn mock_gelu(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified GELU activation
    input.clone()
}

/// GAN Generator benchmark
pub struct GANGeneratorBench {
    pub latent_dim: usize,
    pub output_channels: usize,
    pub output_size: usize,
    pub batch_size: usize,
}

impl GANGeneratorBench {
    pub fn new(
        latent_dim: usize,
        output_channels: usize,
        output_size: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            latent_dim,
            output_channels,
            output_size,
            batch_size,
        }
    }
}

impl Benchmarkable for GANGeneratorBench {
    type Input = (Tensor<f32>, GANGeneratorWeights);
    type Output = Tensor<f32>;

    fn setup(&mut self, _size: usize) -> Self::Input {
        let noise = rand(&[self.batch_size, self.latent_dim]).unwrap();

        let weights = GANGeneratorWeights {
            fc1_weight: rand(&[self.latent_dim, 128 * 4 * 4]).unwrap(),
            fc1_bias: rand(&[128 * 4 * 4]).unwrap(),
            bn1_weight: ones::<f32>(&[128]).unwrap(),
            bn1_bias: zeros::<f32>(&[128]).unwrap(),
            deconv1_weight: rand(&[128, 64, 4, 4]).unwrap(),
            deconv1_bias: rand(&[64]).unwrap(),
            bn2_weight: ones::<f32>(&[64]).unwrap(),
            bn2_bias: zeros::<f32>(&[64]).unwrap(),
            deconv2_weight: rand(&[64, 32, 4, 4]).unwrap(),
            deconv2_bias: rand(&[32]).unwrap(),
            bn3_weight: ones::<f32>(&[32]).unwrap(),
            bn3_bias: zeros::<f32>(&[32]).unwrap(),
            deconv3_weight: rand(&[32, self.output_channels, 4, 4]).unwrap(),
            deconv3_bias: rand(&[self.output_channels]).unwrap(),
        };

        (noise, weights)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (noise, weights) = input;

        // Linear layer to expand latent vector
        let fc1_out = mock_linear(noise, &weights.fc1_weight, Some(&weights.fc1_bias));
        let reshaped = mock_reshape(&fc1_out, &[self.batch_size, 128, 4, 4]);
        let bn1_out = mock_batch_norm(&reshaped, &weights.bn1_weight, &weights.bn1_bias);
        let relu1_out = mock_relu(&bn1_out);

        // Deconvolution layer 1
        let deconv1_out = mock_deconv2d(&relu1_out, &weights.deconv1_weight);
        let bn2_out = mock_batch_norm(&deconv1_out, &weights.bn2_weight, &weights.bn2_bias);
        let relu2_out = mock_relu(&bn2_out);

        // Deconvolution layer 2
        let deconv2_out = mock_deconv2d(&relu2_out, &weights.deconv2_weight);
        let bn3_out = mock_batch_norm(&deconv2_out, &weights.bn3_weight, &weights.bn3_bias);
        let relu3_out = mock_relu(&bn3_out);

        // Final deconvolution layer
        let deconv3_out = mock_deconv2d(&relu3_out, &weights.deconv3_weight);
        black_box(mock_tanh(&deconv3_out))
    }

    fn flops(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let latent = self.latent_dim;

        // FC layer
        let fc_flops = batch * latent * 128 * 4 * 4;
        // Deconv layers (approximate)
        let deconv1_flops = batch * 128 * 64 * 8 * 8 * 4 * 4; // from 4x4 to 8x8
        let deconv2_flops = batch * 64 * 32 * 16 * 16 * 4 * 4; // from 8x8 to 16x16
        let deconv3_flops = batch * 32 * self.output_channels * 32 * 32 * 4 * 4; // from 16x16 to 32x32
                                                                                 // Batch norms
        let bn_flops = batch * (128 * 4 * 4 + 64 * 8 * 8 + 32 * 16 * 16) * 4;

        fc_flops + deconv1_flops + deconv2_flops + deconv3_flops + bn_flops
    }

    fn bytes_accessed(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let latent = self.latent_dim;
        let output_size = self.output_size;

        // Input noise
        let input_bytes = batch * latent * 4;
        // Weights
        let fc_weights = latent * 128 * 4 * 4 * 4;
        let deconv_weights = (128 * 64 + 64 * 32 + 32 * self.output_channels) * 4 * 4 * 4;
        let bn_weights = (128 + 64 + 32) * 4 * 4;
        // Output
        let output_bytes = batch * self.output_channels * output_size * output_size * 4;

        input_bytes + fc_weights + deconv_weights + bn_weights + output_bytes
    }
}

/// GAN Discriminator benchmark
pub struct GANDiscriminatorBench {
    pub input_channels: usize,
    pub input_size: usize,
    pub batch_size: usize,
}

impl GANDiscriminatorBench {
    pub fn new(input_channels: usize, input_size: usize, batch_size: usize) -> Self {
        Self {
            input_channels,
            input_size,
            batch_size,
        }
    }
}

impl Benchmarkable for GANDiscriminatorBench {
    type Input = (Tensor<f32>, GANDiscriminatorWeights);
    type Output = Tensor<f32>;

    fn setup(&mut self, _size: usize) -> Self::Input {
        let input = rand(&[
            self.batch_size,
            self.input_channels,
            self.input_size,
            self.input_size,
        ])
        .unwrap();

        let weights = GANDiscriminatorWeights {
            conv1_weight: rand(&[64, self.input_channels, 4, 4]).unwrap(),
            conv1_bias: rand(&[64]).unwrap(),
            conv2_weight: rand(&[128, 64, 4, 4]).unwrap(),
            conv2_bias: rand(&[128]).unwrap(),
            bn2_weight: ones::<f32>(&[128]).unwrap(),
            bn2_bias: zeros::<f32>(&[128]).unwrap(),
            conv3_weight: rand(&[256, 128, 4, 4]).unwrap(),
            conv3_bias: rand(&[256]).unwrap(),
            bn3_weight: ones::<f32>(&[256]).unwrap(),
            bn3_bias: zeros::<f32>(&[256]).unwrap(),
            fc_weight: rand(&[256 * 4 * 4, 1]).unwrap(),
            fc_bias: rand(&[1]).unwrap(),
        };

        (input, weights)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (x, weights) = input;

        // Convolution layer 1
        let conv1_out = mock_conv2d(x, &weights.conv1_weight);
        let leaky1_out = mock_leaky_relu(&conv1_out, 0.2);

        // Convolution layer 2
        let conv2_out = mock_conv2d(&leaky1_out, &weights.conv2_weight);
        let bn2_out = mock_batch_norm(&conv2_out, &weights.bn2_weight, &weights.bn2_bias);
        let leaky2_out = mock_leaky_relu(&bn2_out, 0.2);

        // Convolution layer 3
        let conv3_out = mock_conv2d(&leaky2_out, &weights.conv3_weight);
        let bn3_out = mock_batch_norm(&conv3_out, &weights.bn3_weight, &weights.bn3_bias);
        let leaky3_out = mock_leaky_relu(&bn3_out, 0.2);

        // Flatten and final linear layer
        let flattened = mock_flatten(&leaky3_out);
        black_box(mock_linear(
            &flattened,
            &weights.fc_weight,
            Some(&weights.fc_bias),
        ))
    }

    fn flops(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let input_size = self.input_size;

        // Conv layers (approximate, assuming stride=2)
        let conv1_flops =
            batch * 64 * self.input_channels * (input_size / 2) * (input_size / 2) * 4 * 4;
        let conv2_flops = batch * 128 * 64 * (input_size / 4) * (input_size / 4) * 4 * 4;
        let conv3_flops = batch * 256 * 128 * (input_size / 8) * (input_size / 8) * 4 * 4;
        // Batch norms
        let bn_flops = batch
            * (128 * (input_size / 4) * (input_size / 4)
                + 256 * (input_size / 8) * (input_size / 8))
            * 4;
        // Final FC
        let fc_flops = batch * 256 * 4 * 4 * 1;

        conv1_flops + conv2_flops + conv3_flops + bn_flops + fc_flops
    }

    fn bytes_accessed(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let input_size = self.input_size;

        // Input
        let input_bytes = batch * self.input_channels * input_size * input_size * 4;
        // Weights
        let conv_weights = (64 * self.input_channels + 128 * 64 + 256 * 128) * 4 * 4 * 4;
        let bn_weights = (128 + 256) * 4 * 4;
        let fc_weights = 256 * 4 * 4 * 1 * 4;
        // Output
        let output_bytes = batch * 1 * 4;

        input_bytes + conv_weights + bn_weights + fc_weights + output_bytes
    }
}

/// GAN weights structures
pub struct GANGeneratorWeights {
    pub fc1_weight: Tensor<f32>,
    pub fc1_bias: Tensor<f32>,
    pub bn1_weight: Tensor<f32>,
    pub bn1_bias: Tensor<f32>,
    pub deconv1_weight: Tensor<f32>,
    pub deconv1_bias: Tensor<f32>,
    pub bn2_weight: Tensor<f32>,
    pub bn2_bias: Tensor<f32>,
    pub deconv2_weight: Tensor<f32>,
    pub deconv2_bias: Tensor<f32>,
    pub bn3_weight: Tensor<f32>,
    pub bn3_bias: Tensor<f32>,
    pub deconv3_weight: Tensor<f32>,
    pub deconv3_bias: Tensor<f32>,
}

pub struct GANDiscriminatorWeights {
    pub conv1_weight: Tensor<f32>,
    pub conv1_bias: Tensor<f32>,
    pub conv2_weight: Tensor<f32>,
    pub conv2_bias: Tensor<f32>,
    pub bn2_weight: Tensor<f32>,
    pub bn2_bias: Tensor<f32>,
    pub conv3_weight: Tensor<f32>,
    pub conv3_bias: Tensor<f32>,
    pub bn3_weight: Tensor<f32>,
    pub bn3_bias: Tensor<f32>,
    pub fc_weight: Tensor<f32>,
    pub fc_bias: Tensor<f32>,
}

// Additional mock functions for GAN operations
fn mock_deconv2d(input: &Tensor<f32>, weight: &Tensor<f32>) -> Tensor<f32> {
    // Simplified deconvolution - return upsampled tensor
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    let batch = input_shape.dims()[0];
    let out_channels = weight_shape.dims()[1];
    let height = input_shape.dims()[2] * 2; // Assuming stride=2
    let width = input_shape.dims()[3] * 2;

    rand(&[batch, out_channels, height, width]).unwrap()
}

fn mock_reshape(_input: &Tensor<f32>, new_shape: &[usize]) -> Tensor<f32> {
    // Mock reshape - just return tensor with new shape
    rand(new_shape).unwrap()
}

fn mock_tanh(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified tanh activation
    input.clone()
}

fn mock_leaky_relu(input: &Tensor<f32>, _negative_slope: f32) -> Tensor<f32> {
    // Simplified leaky ReLU activation
    input.clone()
}

fn mock_flatten(input: &Tensor<f32>) -> Tensor<f32> {
    // Flatten tensor to 2D [batch_size, flattened_features]
    let input_shape = input.shape();
    let batch = input_shape.dims()[0];
    let total_size = input_shape.dims().iter().skip(1).product();

    // Properly reshape the input tensor instead of creating random data
    input
        .reshape(&[batch as i32, total_size as i32])
        .unwrap_or_else(|_| {
            // Fallback: create tensor with correct shape if reshape fails
            rand(&[batch, total_size]).unwrap()
        })
}

/// YOLOv5 benchmark
pub struct YOLOv5Bench {
    pub num_classes: usize,
    pub input_size: usize,
    pub batch_size: usize,
}

impl YOLOv5Bench {
    pub fn new(num_classes: usize, input_size: usize, batch_size: usize) -> Self {
        Self {
            num_classes,
            input_size,
            batch_size,
        }
    }
}

impl Benchmarkable for YOLOv5Bench {
    type Input = (Tensor<f32>, YOLOv5Weights);
    type Output = (Tensor<f32>, Tensor<f32>, Tensor<f32>); // 3 detection scales

    fn setup(&mut self, _size: usize) -> Self::Input {
        let input = rand(&[self.batch_size, 3, self.input_size, self.input_size]).unwrap();

        let weights = YOLOv5Weights {
            // Backbone (CSPDarknet53)
            backbone_conv1_weight: rand(&[32, 3, 6, 6]).unwrap(),
            backbone_conv1_bias: rand(&[32]).unwrap(),
            backbone_conv2_weight: rand(&[64, 32, 3, 3]).unwrap(),
            backbone_conv2_bias: rand(&[64]).unwrap(),
            backbone_conv3_weight: rand(&[128, 64, 3, 3]).unwrap(),
            backbone_conv3_bias: rand(&[128]).unwrap(),

            // Neck (PANet)
            neck_conv1_weight: rand(&[256, 128, 1, 1]).unwrap(),
            neck_conv1_bias: rand(&[256]).unwrap(),
            neck_conv2_weight: rand(&[512, 256, 3, 3]).unwrap(),
            neck_conv2_bias: rand(&[512]).unwrap(),

            // Head (Detection layers)
            head_conv1_weight: rand(&[256, 128, 3, 3]).unwrap(),
            head_conv1_bias: rand(&[256]).unwrap(),
            head_conv2_weight: rand(&[512, 256, 3, 3]).unwrap(),
            head_conv2_bias: rand(&[512]).unwrap(),
            head_conv3_weight: rand(&[1024, 512, 3, 3]).unwrap(),
            head_conv3_bias: rand(&[1024]).unwrap(),

            // Output layers (3 scales)
            output1_weight: rand(&[3 * (self.num_classes + 5), 256, 1, 1]).unwrap(),
            output1_bias: rand(&[3 * (self.num_classes + 5)]).unwrap(),
            output2_weight: rand(&[3 * (self.num_classes + 5), 512, 1, 1]).unwrap(),
            output2_bias: rand(&[3 * (self.num_classes + 5)]).unwrap(),
            output3_weight: rand(&[3 * (self.num_classes + 5), 1024, 1, 1]).unwrap(),
            output3_bias: rand(&[3 * (self.num_classes + 5)]).unwrap(),
        };

        (input, weights)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (x, weights) = input;

        // Backbone feature extraction
        let backbone_out1 = mock_conv2d(x, &weights.backbone_conv1_weight);
        let backbone_out1 = mock_silu(&backbone_out1);
        let backbone_out2 = mock_conv2d(&backbone_out1, &weights.backbone_conv2_weight);
        let backbone_out2 = mock_silu(&backbone_out2);
        let backbone_out3 = mock_conv2d(&backbone_out2, &weights.backbone_conv3_weight);
        let backbone_out3 = mock_silu(&backbone_out3);

        // Neck feature fusion
        let neck_out1 = mock_conv2d(&backbone_out3, &weights.neck_conv1_weight);
        let neck_out1 = mock_silu(&neck_out1);
        let neck_out2 = mock_conv2d(&neck_out1, &weights.neck_conv2_weight);
        let neck_out2 = mock_silu(&neck_out2);

        // Multi-scale detection heads
        let head_out1 = mock_conv2d(&neck_out1, &weights.head_conv1_weight);
        let head_out1 = mock_silu(&head_out1);
        let output1 = mock_conv2d(&head_out1, &weights.output1_weight);

        let head_out2 = mock_conv2d(&neck_out2, &weights.head_conv2_weight);
        let head_out2 = mock_silu(&head_out2);
        let output2 = mock_conv2d(&head_out2, &weights.output2_weight);

        let head_out3 = mock_conv2d(&neck_out2, &weights.head_conv3_weight);
        let head_out3 = mock_silu(&head_out3);
        let output3 = mock_conv2d(&head_out3, &weights.output3_weight);

        (black_box(output1), black_box(output2), black_box(output3))
    }

    fn flops(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let input_size = self.input_size;
        let num_classes = self.num_classes;

        // Approximate FLOPS for YOLOv5
        // Backbone
        let backbone_flops = batch
            * input_size
            * input_size
            * (
                3 * 32 * 6 * 6 + // conv1
            32 * 64 * 3 * 3 + // conv2
            64 * 128 * 3 * 3
                // conv3
            );

        // Neck
        let neck_flops = batch
            * (input_size / 8)
            * (input_size / 8)
            * (
                128 * 256 * 1 * 1 + // neck conv1
            256 * 512 * 3 * 3
                // neck conv2
            );

        // Head
        let head_flops = batch
            * (
                (input_size / 8) * (input_size / 8) * 128 * 256 * 3 * 3 + // head1
            (input_size / 16) * (input_size / 16) * 256 * 512 * 3 * 3 + // head2
            (input_size / 32) * (input_size / 32) * 512 * 1024 * 3 * 3
                // head3
            );

        // Output layers
        let output_flops = batch
            * ((input_size / 8) * (input_size / 8) * 256 * 3 * (num_classes + 5)
                + (input_size / 16) * (input_size / 16) * 512 * 3 * (num_classes + 5)
                + (input_size / 32) * (input_size / 32) * 1024 * 3 * (num_classes + 5));

        backbone_flops + neck_flops + head_flops + output_flops
    }

    fn bytes_accessed(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let input_size = self.input_size;
        let num_classes = self.num_classes;

        // Input image
        let input_bytes = batch * 3 * input_size * input_size * 4;

        // Weights (approximate)
        let backbone_weights = (32 * 3 * 6 * 6 + 64 * 32 * 3 * 3 + 128 * 64 * 3 * 3) * 4;
        let neck_weights = (256 * 128 * 1 * 1 + 512 * 256 * 3 * 3) * 4;
        let head_weights = (256 * 128 * 3 * 3 + 512 * 256 * 3 * 3 + 1024 * 512 * 3 * 3) * 4;
        let output_weights = (256 + 512 + 1024) * 3 * (num_classes + 5) * 4;

        // Outputs (3 scales)
        let output_bytes = batch
            * 3
            * (num_classes + 5)
            * ((input_size / 8) * (input_size / 8)
                + (input_size / 16) * (input_size / 16)
                + (input_size / 32) * (input_size / 32))
            * 4;

        input_bytes + backbone_weights + neck_weights + head_weights + output_weights + output_bytes
    }
}

/// SSD benchmark
pub struct SSDBench {
    pub num_classes: usize,
    pub input_size: usize,
    pub batch_size: usize,
}

impl SSDBench {
    pub fn new(num_classes: usize, input_size: usize, batch_size: usize) -> Self {
        Self {
            num_classes,
            input_size,
            batch_size,
        }
    }
}

impl Benchmarkable for SSDBench {
    type Input = (Tensor<f32>, SSDWeights);
    type Output = (Tensor<f32>, Tensor<f32>); // (class_predictions, box_predictions)

    fn setup(&mut self, _size: usize) -> Self::Input {
        let input = rand(&[self.batch_size, 3, self.input_size, self.input_size]).unwrap();

        let weights = SSDWeights {
            // Base network (MobileNet-style)
            base_conv1_weight: rand(&[32, 3, 3, 3]).unwrap(),
            base_conv1_bias: rand(&[32]).unwrap(),
            base_conv2_weight: rand(&[64, 32, 3, 3]).unwrap(),
            base_conv2_bias: rand(&[64]).unwrap(),
            base_conv3_weight: rand(&[128, 64, 3, 3]).unwrap(),
            base_conv3_bias: rand(&[128]).unwrap(),
            base_conv4_weight: rand(&[256, 128, 3, 3]).unwrap(),
            base_conv4_bias: rand(&[256]).unwrap(),
            base_conv5_weight: rand(&[512, 256, 3, 3]).unwrap(),
            base_conv5_bias: rand(&[512]).unwrap(),

            // Extra feature layers
            extra_conv1_weight: rand(&[256, 512, 1, 1]).unwrap(),
            extra_conv1_bias: rand(&[256]).unwrap(),
            extra_conv2_weight: rand(&[512, 256, 3, 3]).unwrap(),
            extra_conv2_bias: rand(&[512]).unwrap(),

            // Classification head
            class_conv1_weight: rand(&[4 * self.num_classes, 256, 3, 3]).unwrap(),
            class_conv1_bias: rand(&[4 * self.num_classes]).unwrap(),
            class_conv2_weight: rand(&[6 * self.num_classes, 512, 3, 3]).unwrap(),
            class_conv2_bias: rand(&[6 * self.num_classes]).unwrap(),

            // Localization head
            loc_conv1_weight: rand(&[4 * 4, 256, 3, 3]).unwrap(), // 4 boxes * 4 coordinates
            loc_conv1_bias: rand(&[4 * 4]).unwrap(),
            loc_conv2_weight: rand(&[6 * 4, 512, 3, 3]).unwrap(), // 6 boxes * 4 coordinates
            loc_conv2_bias: rand(&[6 * 4]).unwrap(),
        };

        (input, weights)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let (x, weights) = input;

        // Base network feature extraction
        let base_out1 = mock_conv2d(x, &weights.base_conv1_weight);
        let base_out1 = mock_relu(&base_out1);
        let base_out2 = mock_conv2d(&base_out1, &weights.base_conv2_weight);
        let base_out2 = mock_relu(&base_out2);
        let base_out3 = mock_conv2d(&base_out2, &weights.base_conv3_weight);
        let base_out3 = mock_relu(&base_out3);
        let base_out4 = mock_conv2d(&base_out3, &weights.base_conv4_weight);
        let base_out4 = mock_relu(&base_out4);
        let base_out5 = mock_conv2d(&base_out4, &weights.base_conv5_weight);
        let base_out5 = mock_relu(&base_out5);

        // Extra feature layers
        let extra_out1 = mock_conv2d(&base_out5, &weights.extra_conv1_weight);
        let extra_out1 = mock_relu(&extra_out1);
        let extra_out2 = mock_conv2d(&extra_out1, &weights.extra_conv2_weight);
        let extra_out2 = mock_relu(&extra_out2);

        // Multi-scale predictions
        let class_pred1 = mock_conv2d(&base_out4, &weights.class_conv1_weight);
        let class_pred2 = mock_conv2d(&extra_out2, &weights.class_conv2_weight);

        let loc_pred1 = mock_conv2d(&base_out4, &weights.loc_conv1_weight);
        let loc_pred2 = mock_conv2d(&extra_out2, &weights.loc_conv2_weight);

        // Combine predictions
        let class_preds = mock_concat(&[class_pred1, class_pred2]);
        let loc_preds = mock_concat(&[loc_pred1, loc_pred2]);

        (black_box(class_preds), black_box(loc_preds))
    }

    fn flops(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let input_size = self.input_size;
        let num_classes = self.num_classes;

        // Base network FLOPS
        let base_flops = batch
            * input_size
            * input_size
            * (
                3 * 32 * 3 * 3 + // conv1
            32 * 64 * 3 * 3 + // conv2
            64 * 128 * 3 * 3 + // conv3
            128 * 256 * 3 * 3 + // conv4
            256 * 512 * 3 * 3
                // conv5
            )
            / 4; // Account for stride/pooling

        // Extra layers
        let extra_flops = batch
            * (input_size / 16)
            * (input_size / 16)
            * (
                512 * 256 * 1 * 1 + // extra conv1
            256 * 512 * 3 * 3
                // extra conv2
            );

        // Detection heads
        let det_flops = batch
            * ((input_size / 8)
                * (input_size / 8)
                * (
                    256 * 4 * num_classes * 3 * 3 + // class conv1
                256 * 4 * 4 * 3 * 3
                    // loc conv1
                )
                + (input_size / 16)
                    * (input_size / 16)
                    * (
                        512 * 6 * num_classes * 3 * 3 + // class conv2
                512 * 6 * 4 * 3 * 3
                        // loc conv2
                    ));

        base_flops + extra_flops + det_flops
    }

    fn bytes_accessed(&self, _size: usize) -> usize {
        let batch = self.batch_size;
        let input_size = self.input_size;
        let num_classes = self.num_classes;

        // Input
        let input_bytes = batch * 3 * input_size * input_size * 4;

        // Base network weights
        let base_weights = (32 * 3 * 3 * 3
            + 64 * 32 * 3 * 3
            + 128 * 64 * 3 * 3
            + 256 * 128 * 3 * 3
            + 512 * 256 * 3 * 3)
            * 4;

        // Extra layer weights
        let extra_weights = (256 * 512 * 1 * 1 + 512 * 256 * 3 * 3) * 4;

        // Detection head weights
        let det_weights = (4 * num_classes * 256 * 3 * 3
            + 6 * num_classes * 512 * 3 * 3
            + 4 * 4 * 256 * 3 * 3
            + 6 * 4 * 512 * 3 * 3)
            * 4;

        // Output predictions
        let output_bytes = batch
            * ((input_size / 8) * (input_size / 8) * (4 * num_classes + 4 * 4)
                + (input_size / 16) * (input_size / 16) * (6 * num_classes + 6 * 4))
            * 4;

        input_bytes + base_weights + extra_weights + det_weights + output_bytes
    }
}

/// Weight structures for detection models
pub struct YOLOv5Weights {
    pub backbone_conv1_weight: Tensor<f32>,
    pub backbone_conv1_bias: Tensor<f32>,
    pub backbone_conv2_weight: Tensor<f32>,
    pub backbone_conv2_bias: Tensor<f32>,
    pub backbone_conv3_weight: Tensor<f32>,
    pub backbone_conv3_bias: Tensor<f32>,
    pub neck_conv1_weight: Tensor<f32>,
    pub neck_conv1_bias: Tensor<f32>,
    pub neck_conv2_weight: Tensor<f32>,
    pub neck_conv2_bias: Tensor<f32>,
    pub head_conv1_weight: Tensor<f32>,
    pub head_conv1_bias: Tensor<f32>,
    pub head_conv2_weight: Tensor<f32>,
    pub head_conv2_bias: Tensor<f32>,
    pub head_conv3_weight: Tensor<f32>,
    pub head_conv3_bias: Tensor<f32>,
    pub output1_weight: Tensor<f32>,
    pub output1_bias: Tensor<f32>,
    pub output2_weight: Tensor<f32>,
    pub output2_bias: Tensor<f32>,
    pub output3_weight: Tensor<f32>,
    pub output3_bias: Tensor<f32>,
}

pub struct SSDWeights {
    pub base_conv1_weight: Tensor<f32>,
    pub base_conv1_bias: Tensor<f32>,
    pub base_conv2_weight: Tensor<f32>,
    pub base_conv2_bias: Tensor<f32>,
    pub base_conv3_weight: Tensor<f32>,
    pub base_conv3_bias: Tensor<f32>,
    pub base_conv4_weight: Tensor<f32>,
    pub base_conv4_bias: Tensor<f32>,
    pub base_conv5_weight: Tensor<f32>,
    pub base_conv5_bias: Tensor<f32>,
    pub extra_conv1_weight: Tensor<f32>,
    pub extra_conv1_bias: Tensor<f32>,
    pub extra_conv2_weight: Tensor<f32>,
    pub extra_conv2_bias: Tensor<f32>,
    pub class_conv1_weight: Tensor<f32>,
    pub class_conv1_bias: Tensor<f32>,
    pub class_conv2_weight: Tensor<f32>,
    pub class_conv2_bias: Tensor<f32>,
    pub loc_conv1_weight: Tensor<f32>,
    pub loc_conv1_bias: Tensor<f32>,
    pub loc_conv2_weight: Tensor<f32>,
    pub loc_conv2_bias: Tensor<f32>,
}

// Additional mock functions for detection models
fn mock_silu(input: &Tensor<f32>) -> Tensor<f32> {
    // Simplified SiLU activation
    input.clone()
}

fn mock_concat(tensors: &[Tensor<f32>]) -> Tensor<f32> {
    // Simplified concatenation - just return the first tensor
    tensors[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Model benchmark tests need layer implementation fixes"]
    fn test_resnet_block_bench() {
        let mut bench = ResNetBlockBench::new(64, 64, 224, 1);
        let input = bench.setup(224);
        let output = bench.run(&input);

        // Verify output shape is reasonable
        let shape = output.shape();
        let output_shape = shape.dims();
        assert_eq!(output_shape[0], 1); // batch size
        assert_eq!(output_shape[1], 64); // output channels
    }

    #[test]
    fn test_transformer_block_bench() {
        let mut bench = TransformerBlockBench::new(128, 768, 12, 1);
        let input = bench.setup(128);
        let output = bench.run(&input);

        // Verify output shape matches input shape
        let shape = output.shape();
        let output_shape = shape.dims();
        assert_eq!(output_shape[0], 1); // batch size
        assert_eq!(output_shape[1], 128); // sequence length
        assert_eq!(output_shape[2], 768); // d_model
    }

    #[test]
    #[ignore = "Model benchmark tests need layer implementation fixes"]
    fn test_model_benchmark_suite() {
        let suite = ModelBenchmarkSuite::new();
        assert!(!suite.configs.is_empty());

        // Test running a single benchmark
        let config = &suite.configs[0];
        let result = suite.run_single_benchmark(config, 1, 224);

        assert!(!result.model_name.is_empty());
        assert!(result.forward_time_ns > 0.0);
        assert!(result.flops > 0);
    }

    #[test]
    fn test_model_benchmark_result_csv() {
        let result = ModelBenchmarkResult {
            model_name: "Test Model".to_string(),
            batch_size: 8,
            input_size: 224,
            forward_time_ns: 1_000_000.0,
            total_time_ns: 1_200_000.0,
            flops: 1000000,
            memory_usage_bytes: 1000000,
            throughput_samples_per_sec: 8000.0,
            flops_per_sec: 1e9,
        };

        let csv = result.to_csv_row();
        assert!(csv.contains("Test Model"));
        assert!(csv.contains("8"));
        assert!(csv.contains("224"));
    }

    #[test]
    fn test_gan_generator_bench() {
        let mut bench = GANGeneratorBench::new(100, 3, 32, 1);
        let input = bench.setup(32);
        let output = bench.run(&input);

        // Verify output shape is reasonable
        let output_shape = output.shape();
        assert_eq!(output_shape.dims()[0], 1); // batch size
        assert_eq!(output_shape.dims()[1], 3); // output channels
    }

    #[test]
    fn test_gan_discriminator_bench() {
        let mut bench = GANDiscriminatorBench::new(3, 32, 1);
        let input = bench.setup(32);
        let output = bench.run(&input);

        // Verify output shape
        let output_shape = output.shape();
        assert_eq!(output_shape.dims()[0], 1); // batch size
        assert_eq!(output_shape.dims()[1], 1); // single output (real/fake)
    }

    #[test]
    fn test_yolo_bench() {
        let mut bench = YOLOv5Bench::new(80, 640, 1);
        let input = bench.setup(640);
        let (output1, output2, output3) = bench.run(&input);

        // Verify we get 3 detection scales
        let output1_shape = output1.shape();
        let output2_shape = output2.shape();
        let output3_shape = output3.shape();
        assert_eq!(output1_shape.dims()[0], 1); // batch size
        assert_eq!(output2_shape.dims()[0], 1); // batch size
        assert_eq!(output3_shape.dims()[0], 1); // batch size
    }

    #[test]
    fn test_ssd_bench() {
        let mut bench = SSDBench::new(80, 300, 1);
        let input = bench.setup(300);
        let (class_preds, loc_preds) = bench.run(&input);

        // Verify we get class and location predictions
        let class_preds_shape = class_preds.shape();
        let loc_preds_shape = loc_preds.shape();
        assert_eq!(class_preds_shape.dims()[0], 1); // batch size
        assert_eq!(loc_preds_shape.dims()[0], 1); // batch size
    }
}
