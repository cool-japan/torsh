//! Domain-specific benchmarks for different machine learning domains
//!
//! This module provides specialized benchmarking suites tailored for specific
//! machine learning domains with realistic parameter distributions and gradient patterns.

use super::core::{BenchmarkConfig, BenchmarkResult};
use crate::{Optimizer, OptimizerResult};
use std::time::{Duration, Instant};
use torsh_tensor::{creation, Tensor};

/// Domain-specific benchmarks for Computer Vision
pub struct CVBenchmarks {
    config: BenchmarkConfig,
}

impl CVBenchmarks {
    /// Create new CV benchmarks with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create new CV benchmarks with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Benchmark optimizer on simulated ResNet-like architecture
    ///
    /// This benchmark simulates the parameter structure and gradient patterns
    /// typical in convolutional neural networks, particularly ResNet architectures.
    pub fn benchmark_resnet_training<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<BenchmarkResult> {
        // Simulate ResNet-18 parameter counts and shapes
        use torsh_tensor::creation;

        // Conv layers: [64, 3, 7, 7], [64, 64, 3, 3], [128, 64, 3, 3], etc.
        let conv_shapes = vec![
            vec![64, 3, 7, 7],    // First conv
            vec![64, 64, 3, 3],   // ResNet block conv1
            vec![64, 64, 3, 3],   // ResNet block conv2
            vec![128, 64, 3, 3],  // Downsample conv
            vec![128, 128, 3, 3], // ResNet block conv
            vec![256, 128, 3, 3], // Downsample conv
            vec![512, 256, 3, 3], // Downsample conv
        ];

        // Linear layer: [1000, 512] for ImageNet classification
        let fc_shape = vec![1000, 512];

        let mut params = Vec::new();
        for shape in &conv_shapes {
            params.push(creation::randn::<f32>(shape)?);
        }
        params.push(creation::randn::<f32>(&fc_shape)?);

        let mut iteration_times = Vec::new();
        let start_time = Instant::now();

        // Simulate CV training with batch size effects and typical gradient patterns
        for i in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // Simulate gradients with patterns typical in CV:
            // - Conv layers have smaller gradients than FC layers
            // - Early layers have smaller gradients (vanishing gradient)
            for (j, param) in params.iter_mut().enumerate() {
                let gradient_scale = if j < conv_shapes.len() {
                    // Convolutional layers - smaller gradients, vanishing toward input
                    0.01 * (1.0 / (j as f32 + 1.0))
                } else {
                    // Fully connected layer - larger gradients
                    0.1
                };

                let grads =
                    creation::randn::<f32>(param.shape().dims())?.mul_scalar(gradient_scale)?;
                param.set_grad(Some(grads));
            }

            let iter_start = Instant::now();
            optimizer.step()?;
            iteration_times.push(iter_start.elapsed());

            optimizer.zero_grad();
        }

        // Calculate statistics
        let total_time = start_time.elapsed();
        let iterations_completed = iteration_times.len();
        let avg_time = total_time / iterations_completed as u32;

        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        Ok(BenchmarkResult {
            name: "cv_resnet_training".to_string(),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: Duration::ZERO, // Simplified for now
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }

    /// Benchmark optimizer on simulated VGG-like architecture
    pub fn benchmark_vgg_training<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<BenchmarkResult> {
        // VGG-style layers with increasing depth
        let vgg_shapes = vec![
            vec![64, 3, 3, 3],    // Conv1_1
            vec![64, 64, 3, 3],   // Conv1_2
            vec![128, 64, 3, 3],  // Conv2_1
            vec![128, 128, 3, 3], // Conv2_2
            vec![256, 128, 3, 3], // Conv3_1
            vec![256, 256, 3, 3], // Conv3_2
            vec![256, 256, 3, 3], // Conv3_3
            vec![512, 256, 3, 3], // Conv4_1
            vec![512, 512, 3, 3], // Conv4_2
            vec![512, 512, 3, 3], // Conv4_3
            vec![4096, 25088],    // FC1 (7*7*512 = 25088)
            vec![4096, 4096],     // FC2
            vec![1000, 4096],     // FC3
        ];

        let mut params = Vec::new();
        for shape in &vgg_shapes {
            params.push(creation::randn::<f32>(shape)?);
        }

        let mut iteration_times = Vec::new();
        let start_time = Instant::now();

        for _ in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // VGG gradient patterns: deeper layers have progressively smaller gradients
            for (j, param) in params.iter_mut().enumerate() {
                let layer_depth = j as f32 / vgg_shapes.len() as f32;
                let gradient_scale = if j < 10 {
                    // Convolutional layers - exponentially decreasing gradients
                    0.01 * (0.8_f32).powf(layer_depth * 10.0)
                } else {
                    // Fully connected layers - larger gradients
                    0.05 * (1.0 - layer_depth * 0.5)
                };

                let grads =
                    creation::randn::<f32>(param.shape().dims())?.mul_scalar(gradient_scale)?;
                param.set_grad(Some(grads));
            }

            let iter_start = Instant::now();
            optimizer.step()?;
            iteration_times.push(iter_start.elapsed());

            optimizer.zero_grad();
        }

        let total_time = start_time.elapsed();
        let iterations_completed = iteration_times.len();
        let avg_time = total_time / iterations_completed as u32;

        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        Ok(BenchmarkResult {
            name: "cv_vgg_training".to_string(),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: Duration::ZERO,
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }
}

impl Default for CVBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Domain-specific benchmarks for Natural Language Processing
pub struct NLPBenchmarks {
    config: BenchmarkConfig,
}

impl NLPBenchmarks {
    /// Create new NLP benchmarks with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create new NLP benchmarks with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Benchmark optimizer on simulated transformer architecture
    ///
    /// This benchmark simulates BERT-base architecture with realistic
    /// parameter distributions and gradient patterns typical in transformers.
    pub fn benchmark_transformer_training<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<BenchmarkResult> {
        use torsh_tensor::creation;

        // Simulate BERT-base architecture
        let vocab_size = 30522;
        let hidden_size = 768;
        let intermediate_size = 3072;
        let num_attention_heads = 12;
        let head_size = hidden_size / num_attention_heads;

        // Transformer parameter shapes
        let param_shapes = vec![
            // Embedding layers
            vec![vocab_size, hidden_size], // Token embeddings
            vec![512, hidden_size],        // Position embeddings
            // Attention layers (simplified - one layer)
            vec![hidden_size, hidden_size], // Query projection
            vec![hidden_size, hidden_size], // Key projection
            vec![hidden_size, hidden_size], // Value projection
            vec![hidden_size, hidden_size], // Output projection
            // Feed-forward layers
            vec![hidden_size, intermediate_size], // Intermediate dense
            vec![intermediate_size, hidden_size], // Output dense
            // Layer norm parameters
            vec![hidden_size], // Attention layer norm
            vec![hidden_size], // FFN layer norm
        ];

        let mut params = Vec::new();
        for shape in param_shapes {
            params.push(creation::randn::<f32>(&shape)?);
        }

        let mut iteration_times = Vec::new();
        let start_time = Instant::now();

        for i in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // Simulate transformer gradients:
            // - Embedding layers have small gradients
            // - Attention layers have moderate gradients
            // - FFN layers have larger gradients
            // - Layer norms have very small gradients
            for (j, param) in params.iter_mut().enumerate() {
                let gradient_scale = match j {
                    0..=1 => 0.001, // Embeddings
                    2..=5 => 0.01,  // Attention
                    6..=7 => 0.05,  // FFN
                    _ => 0.0001,    // Layer norms
                };

                let grads =
                    creation::randn::<f32>(param.shape().dims())?.mul_scalar(gradient_scale)?;
                param.set_grad(Some(grads));
            }

            let iter_start = Instant::now();
            optimizer.step()?;
            iteration_times.push(iter_start.elapsed());

            optimizer.zero_grad();
        }

        // Calculate statistics
        let total_time = start_time.elapsed();
        let iterations_completed = iteration_times.len();
        let avg_time = total_time / iterations_completed as u32;

        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        Ok(BenchmarkResult {
            name: "nlp_transformer_training".to_string(),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: Duration::ZERO,
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }

    /// Benchmark optimizer on simulated LSTM architecture
    pub fn benchmark_lstm_training<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<BenchmarkResult> {
        // LSTM architecture parameters
        let vocab_size = 10000;
        let embedding_dim = 300;
        let hidden_size = 512;
        let num_layers = 2;

        let mut param_shapes = Vec::new();

        // Embedding layer
        param_shapes.push(vec![vocab_size, embedding_dim]);

        // LSTM layers
        for layer in 0..num_layers {
            let input_size = if layer == 0 {
                embedding_dim
            } else {
                hidden_size
            };

            // Input-to-hidden weights (Wi, Wf, Wg, Wo)
            for _ in 0..4 {
                param_shapes.push(vec![hidden_size, input_size]);
            }

            // Hidden-to-hidden weights (Ui, Uf, Ug, Uo)
            for _ in 0..4 {
                param_shapes.push(vec![hidden_size, hidden_size]);
            }

            // Biases (bi, bf, bg, bo)
            for _ in 0..4 {
                param_shapes.push(vec![hidden_size]);
            }
        }

        // Output layer
        param_shapes.push(vec![vocab_size, hidden_size]);

        let mut params = Vec::new();
        for shape in param_shapes {
            params.push(creation::randn::<f32>(&shape)?);
        }

        let mut iteration_times = Vec::new();
        let start_time = Instant::now();

        for _ in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // LSTM gradient patterns:
            // - Embedding layer has small gradients
            // - LSTM weights have moderate gradients that may vanish in deeper layers
            // - Output layer has larger gradients
            let params_len = params.len();
            for (j, param) in params.iter_mut().enumerate() {
                let gradient_scale = match j {
                    0 => 0.001, // Embedding
                    j if j < params_len - 1 => {
                        // LSTM parameters - gradients decrease with layer depth
                        let layer_approx = (j - 1) / 12; // Approximate layer index
                        0.01 * (0.8_f32).powf(layer_approx as f32)
                    }
                    _ => 0.05, // Output layer
                };

                let grads =
                    creation::randn::<f32>(param.shape().dims())?.mul_scalar(gradient_scale)?;
                param.set_grad(Some(grads));
            }

            let iter_start = Instant::now();
            optimizer.step()?;
            iteration_times.push(iter_start.elapsed());

            optimizer.zero_grad();
        }

        let total_time = start_time.elapsed();
        let iterations_completed = iteration_times.len();
        let avg_time = total_time / iterations_completed as u32;

        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        Ok(BenchmarkResult {
            name: "nlp_lstm_training".to_string(),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: Duration::ZERO,
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }
}

impl Default for NLPBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Domain-specific benchmarks for Reinforcement Learning
pub struct RLBenchmarks {
    config: BenchmarkConfig,
}

impl RLBenchmarks {
    /// Create new RL benchmarks with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create new RL benchmarks with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Benchmark optimizer on simulated policy gradient architecture
    pub fn benchmark_policy_gradient<O: Optimizer>(
        &self,
        mut optimizer: O,
    ) -> OptimizerResult<BenchmarkResult> {
        // Simple policy network: state -> hidden -> action probabilities
        let state_size = 84 * 84 * 4; // Atari-like state (84x84x4 frames)
        let hidden_sizes = vec![512, 256, 128];
        let action_size = 6; // Number of actions

        let mut param_shapes = Vec::new();

        // Input layer
        param_shapes.push(vec![hidden_sizes[0], state_size]);
        param_shapes.push(vec![hidden_sizes[0]]); // bias

        // Hidden layers
        for i in 0..hidden_sizes.len() - 1 {
            param_shapes.push(vec![hidden_sizes[i + 1], hidden_sizes[i]]);
            param_shapes.push(vec![hidden_sizes[i + 1]]); // bias
        }

        // Output layer
        param_shapes.push(vec![action_size, *hidden_sizes.last().unwrap()]);
        param_shapes.push(vec![action_size]); // bias

        let mut params = Vec::new();
        for shape in param_shapes {
            params.push(creation::randn::<f32>(&shape)?);
        }

        let mut iteration_times = Vec::new();
        let start_time = Instant::now();

        for _ in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // RL gradient patterns:
            // - High variance gradients typical in policy gradient methods
            // - Earlier layers may have vanishing gradients
            let params_len = params.len();
            for (j, param) in params.iter_mut().enumerate() {
                let layer_index = j / 2; // Each layer has weight + bias
                let is_output_layer = layer_index == (params_len / 2 - 1);

                let gradient_scale = if is_output_layer {
                    // Output layer gets larger gradients from policy gradient
                    0.1
                } else {
                    // Hidden layers have smaller gradients that vanish toward input
                    0.01 * (0.7_f32).powf(layer_index as f32)
                };

                // Add high variance typical of RL gradients
                let variance_multiplier = 1.0 + (j as f32 * 0.1).sin().abs() * 2.0;
                let final_scale = gradient_scale * variance_multiplier;

                let grads =
                    creation::randn::<f32>(param.shape().dims())?.mul_scalar(final_scale)?;
                param.set_grad(Some(grads));
            }

            let iter_start = Instant::now();
            optimizer.step()?;
            iteration_times.push(iter_start.elapsed());

            optimizer.zero_grad();
        }

        let total_time = start_time.elapsed();
        let iterations_completed = iteration_times.len();
        let avg_time = total_time / iterations_completed as u32;

        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        Ok(BenchmarkResult {
            name: "rl_policy_gradient".to_string(),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: Duration::ZERO,
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }
}

impl Default for RLBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_benchmarks_creation() {
        let cv_bench = CVBenchmarks::new();
        assert_eq!(cv_bench.config.num_iterations, 1000);
        assert_eq!(cv_bench.config.warmup_iterations, 100);
    }

    #[test]
    fn test_nlp_benchmarks_creation() {
        let nlp_bench = NLPBenchmarks::new();
        assert_eq!(nlp_bench.config.num_iterations, 1000);
        assert_eq!(nlp_bench.config.warmup_iterations, 100);
    }

    #[test]
    fn test_rl_benchmarks_creation() {
        let rl_bench = RLBenchmarks::new();
        assert_eq!(rl_bench.config.num_iterations, 1000);
        assert_eq!(rl_bench.config.warmup_iterations, 100);
    }

    #[test]
    fn test_custom_config() {
        let custom_config = BenchmarkConfig {
            num_iterations: 500,
            warmup_iterations: 50,
            max_time_seconds: 30.0,
            ..Default::default()
        };

        let cv_bench = CVBenchmarks::with_config(custom_config.clone());
        assert_eq!(cv_bench.config.num_iterations, 500);
        assert_eq!(cv_bench.config.warmup_iterations, 50);
        assert_eq!(cv_bench.config.max_time_seconds, 30.0);

        let nlp_bench = NLPBenchmarks::with_config(custom_config.clone());
        assert_eq!(nlp_bench.config.num_iterations, 500);

        let rl_bench = RLBenchmarks::with_config(custom_config);
        assert_eq!(rl_bench.config.num_iterations, 500);
    }
}
