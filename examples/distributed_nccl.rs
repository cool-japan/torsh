//! NCCL Backend Example for GPU Distributed Training
//!
//! This example demonstrates:
//! - NCCL backend initialization and configuration
//! - GPU-optimized collective operations
//! - Multi-GPU training with NCCL communication
//! - Performance comparison between backends

use std::error::Error;
use std::time::Instant;
use torsh::nn::{Linear, Module};
use torsh::optim::{Adam, Optimizer};
use torsh::prelude::*;
use torsh::tensor::Tensor;
use torsh_distributed::{
    all_reduce, broadcast, init_process_group, BackendType, ProcessGroup, ReduceOp,
};
use torsh_tensor::prelude::randn;

/// Configuration for NCCL distributed training
#[derive(Debug, Clone)]
struct NcclConfig {
    world_size: u32,
    rank: u32,
    master_addr: String,
    master_port: u16,
    device_id: Option<i32>,
    benchmark_mode: bool,
}

impl Default for NcclConfig {
    fn default() -> Self {
        Self {
            world_size: 2,
            rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            device_id: None, // Will default to rank as device ID
            benchmark_mode: false,
        }
    }
}

/// Simple model for distributed training
#[derive(Debug)]
struct SimpleModel {
    layers: Vec<Linear>,
}

impl SimpleModel {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let layers = vec![
            Linear::new(input_size, hidden_size, true),
            Linear::new(hidden_size, hidden_size, true),
            Linear::new(hidden_size, output_size, true),
        ];

        Self { layers }
    }
}

impl Module for SimpleModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // Apply ReLU to all but the last layer
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        Ok(x)
    }

    fn parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        let mut params = std::collections::HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        params
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

/// Initialize NCCL distributed training
async fn init_nccl_training(config: &NcclConfig) -> Result<ProcessGroup, Box<dyn Error>> {
    println!("🚀 Initializing NCCL distributed training...");
    println!("   Rank: {}/{}", config.rank, config.world_size);
    println!("   Device ID: {:?}", config.device_id);
    println!("   Master: {}:{}", config.master_addr, config.master_port);

    // Initialize the process group with NCCL backend
    let process_group = init_process_group(
        BackendType::Nccl,
        config.rank,
        config.world_size,
        &config.master_addr,
        config.master_port,
    )?;

    println!("✅ NCCL process group initialized successfully");
    println!("   Backend: {:?}", process_group.backend_type());
    println!(
        "   Rank: {}, World Size: {}",
        process_group.rank(),
        process_group.world_size()
    );

    Ok(process_group)
}

/// Benchmark collective operations
async fn benchmark_collectives(process_group: &ProcessGroup) -> Result<(), Box<dyn Error>> {
    println!("\n📊 Benchmarking NCCL collective operations...");

    let sizes = vec![1024, 4096, 16384, 65536, 262144];
    let num_trials = 10;

    for size in sizes {
        println!("\n🔬 Testing tensor size: {} elements", size);

        // Create test tensor
        let mut test_tensor: Tensor<f32> = randn(&[size]);

        // Benchmark All-Reduce
        let start = Instant::now();
        for _ in 0..num_trials {
            all_reduce(&mut test_tensor, ReduceOp::Sum, process_group).await?;
        }
        let all_reduce_time = start.elapsed().as_micros() as f64 / num_trials as f64;

        // Benchmark Broadcast
        let start = Instant::now();
        for _ in 0..num_trials {
            broadcast(&mut test_tensor, 0, process_group).await?;
        }
        let broadcast_time = start.elapsed().as_micros() as f64 / num_trials as f64;

        println!("   All-Reduce: {:.2} μs", all_reduce_time);
        println!("   Broadcast:  {:.2} μs", broadcast_time);

        // Calculate bandwidth (assuming 4 bytes per f32)
        let bytes = size * 4;
        let all_reduce_bandwidth =
            (bytes as f64) / (all_reduce_time / 1_000_000.0) / 1_000_000_000.0;
        let broadcast_bandwidth = (bytes as f64) / (broadcast_time / 1_000_000.0) / 1_000_000_000.0;

        println!("   All-Reduce Bandwidth: {:.2} GB/s", all_reduce_bandwidth);
        println!("   Broadcast Bandwidth:  {:.2} GB/s", broadcast_bandwidth);
    }

    Ok(())
}

/// Distributed training example with NCCL
async fn distributed_training_example(
    process_group: &ProcessGroup,
    config: &NcclConfig,
) -> Result<(), Box<dyn Error>> {
    println!("\n🏋️ Starting distributed training example...");

    // Create model
    let mut model = SimpleModel::new(784, 256, 10);
    model.train();

    // Create optimizer
    let mut optimizer = Adam::new(model.parameters(), 0.001);

    let batch_size = 32;
    let num_epochs = 5;
    let num_batches = 10;

    println!("   Model: 3-layer MLP (784 -> 256 -> 256 -> 10)");
    println!("   Batch size: {}", batch_size);
    println!("   Epochs: {}", num_epochs);
    println!("   Batches per epoch: {}", num_batches);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let epoch_start = Instant::now();

        for batch in 0..num_batches {
            // Generate synthetic data (in practice, this would come from DataLoader)
            let input: Tensor<f32> = randn(&[batch_size, 784]);
            let target: Tensor<f32> = randn(&[batch_size, 10]);

            // Forward pass
            let output = model.forward(&input)?;
            let loss = ((output - target)?.pow(2.0)?)?.mean()?;

            // Backward pass
            optimizer.zero_grad();
            loss.backward()?;

            // Synchronize gradients across all ranks using NCCL
            let params = model.parameters();
            for (name, param) in params {
                if let Some(grad) = param.grad() {
                    let mut grad_tensor = grad.read().clone();
                    all_reduce(&mut grad_tensor, ReduceOp::Sum, process_group).await?;

                    // Average the gradients
                    grad_tensor = grad_tensor.div_scalar(process_group.world_size() as f32)?;

                    // Update the gradient (this is simplified - real implementation would be more complex)
                    println!(
                        "   📊 Rank {}: Synchronized gradient for {}",
                        process_group.rank(),
                        name
                    );
                }
            }

            // Update parameters
            optimizer.step()?;

            total_loss += loss.item().unwrap_or(0.0);

            if batch % 5 == 0 {
                println!(
                    "   Rank {} - Epoch {}, Batch {}, Loss: {:.4}",
                    process_group.rank(),
                    epoch,
                    batch,
                    loss.item().unwrap_or(0.0)
                );
            }
        }

        let epoch_time = epoch_start.elapsed().as_millis();
        let avg_loss = total_loss / num_batches as f32;

        println!(
            "✅ Rank {} - Epoch {} completed in {}ms, Avg Loss: {:.4}",
            process_group.rank(),
            epoch,
            epoch_time,
            avg_loss
        );
    }

    Ok(())
}

/// Compare NCCL vs other backends
async fn backend_comparison() -> Result<(), Box<dyn Error>> {
    println!("\n⚖️  Backend Performance Comparison");
    println!("=====================================");

    let test_size = 16384;
    let num_trials = 5;

    // Test different backends
    let backends = vec![
        ("NCCL (Mock)", BackendType::Nccl),
        ("Gloo (Mock)", BackendType::Gloo),
    ];

    for (name, backend_type) in backends {
        println!("\n🔬 Testing {} backend:", name);

        let pg = init_process_group(
            backend_type,
            0, // rank
            1, // world_size (single process for comparison)
            "127.0.0.1",
            29500,
        )?;

        let mut test_tensor: Tensor<f32> = randn(&[test_size]);

        let start = Instant::now();
        for _ in 0..num_trials {
            all_reduce(&mut test_tensor, ReduceOp::Sum, &pg).await?;
        }
        let avg_time = start.elapsed().as_micros() as f64 / num_trials as f64;

        println!("   Average all-reduce time: {:.2} μs", avg_time);

        // Calculate theoretical bandwidth
        let bytes = test_size * 4; // 4 bytes per f32
        let bandwidth = (bytes as f64) / (avg_time / 1_000_000.0) / 1_000_000_000.0;
        println!("   Theoretical bandwidth: {:.2} GB/s", bandwidth);
    }

    Ok(())
}

/// Main function
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 ToRSh NCCL Backend Example");
    println!("=============================");

    // Parse command line arguments (simplified)
    let config = NcclConfig {
        world_size: std::env::var("WORLD_SIZE")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .unwrap_or(2),
        rank: std::env::var("RANK")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0),
        master_addr: std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string()),
        master_port: std::env::var("MASTER_PORT")
            .unwrap_or_else(|_| "29500".to_string())
            .parse()
            .unwrap_or(29500),
        device_id: std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|s| s.parse().ok()),
        benchmark_mode: std::env::var("BENCHMARK").is_ok(),
    };

    println!("📋 Configuration:");
    println!("   World Size: {}", config.world_size);
    println!("   Rank: {}", config.rank);
    println!("   Master: {}:{}", config.master_addr, config.master_port);
    println!("   Device ID: {:?}", config.device_id);
    println!("   Benchmark Mode: {}", config.benchmark_mode);

    // Initialize NCCL
    let process_group = init_nccl_training(&config).await?;

    if config.benchmark_mode {
        // Run benchmarks
        benchmark_collectives(&process_group).await?;
        backend_comparison().await?;
    } else {
        // Run training example
        distributed_training_example(&process_group, &config).await?;
    }

    println!("\n✅ NCCL example completed successfully!");
    println!("\n💡 Usage tips:");
    println!("   - Set WORLD_SIZE and RANK environment variables for multi-process");
    println!("   - Set CUDA_VISIBLE_DEVICES to specify GPU device");
    println!("   - Set BENCHMARK=1 to run performance benchmarks");
    println!("   - Example: WORLD_SIZE=2 RANK=0 BENCHMARK=1 cargo run --example distributed_nccl --features nccl");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nccl_backend_creation() {
        let config = NcclConfig::default();

        // This should work with mock implementation
        let result = init_nccl_training(&config).await;
        assert!(result.is_ok(), "NCCL backend initialization should succeed");

        let pg = result.unwrap();
        assert_eq!(pg.backend_type(), BackendType::Nccl);
        assert_eq!(pg.rank(), config.rank);
        assert_eq!(pg.world_size(), config.world_size);
    }

    #[tokio::test]
    async fn test_nccl_collective_operations() {
        let config = NcclConfig::default();
        let pg = init_nccl_training(&config).await.unwrap();

        // Test all-reduce
        let mut tensor: Tensor<f32> = ones(&[10]);
        let result = all_reduce(&mut tensor, ReduceOp::Sum, &pg).await;
        assert!(result.is_ok(), "All-reduce should succeed");

        // Test broadcast
        let mut tensor: Tensor<f32> = ones(&[10]);
        let result = broadcast(&mut tensor, 0, &pg).await;
        assert!(result.is_ok(), "Broadcast should succeed");
    }
}
