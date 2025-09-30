//! Advanced Fully Sharded Data Parallel (FSDP) Example
//!
//! This example demonstrates FSDP capabilities for training very large models:
//! - Parameter sharding across workers to reduce memory usage
//! - Automatic gather/scatter of parameters during computation
//! - Memory optimization through parameter re-sharding
//! - Different sharding strategies and configurations
//! - Memory usage monitoring and statistics

use parking_lot::RwLock;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use torsh_core::{DType, Device, Shape, Tensor};
use torsh_distributed::{
    auto_wrap_modules, fsdp_wrap, init_process_group, AutoWrapPolicy, BackendType,
    BackwardPrefetch, FsdpConfig, FullyShardedDataParallel, MemoryConfig, MixedPrecisionConfig,
    ProcessGroup, ShardingStrategy,
};
use torsh_nn::{Linear, Module, Sequential};

/// Large neural network for FSDP demonstration
struct LargeModel {
    layers: Vec<Linear>,
    num_layers: usize,
    hidden_size: usize,
}

impl LargeModel {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let mut layers = Vec::new();

        // Input layer
        layers.push(Linear::new(input_size, hidden_size, true));

        // Hidden layers
        for _ in 0..(num_layers - 2) {
            layers.push(Linear::new(hidden_size, hidden_size, true));
        }

        // Output layer
        layers.push(Linear::new(hidden_size, output_size, true));

        Ok(Self {
            layers,
            num_layers,
            hidden_size,
        })
    }

    fn count_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                let params = layer.parameters();
                params.values().map(|p| p.tensor().numel()).sum::<usize>()
            })
            .sum()
    }
}

impl Module for LargeModel {
    fn forward(&self, input: &Tensor) -> torsh_core::Result<Tensor> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // Apply ReLU activation (except for output layer)
            if i < self.num_layers - 1 {
                x = x.relu()?;
            }
        }

        Ok(x)
    }

    fn parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        let mut all_params = std::collections::HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                all_params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        all_params
    }

    fn train(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.train(mode);
        }
    }

    fn eval(&mut self) {
        self.train(false);
    }
}

/// Demonstrate basic FSDP functionality
async fn demo_basic_fsdp(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🔧 Basic FSDP Demo (Worker {})", rank);
    println!("==============================");

    // Initialize process group
    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        rank,
        world_size,
        "127.0.0.1",
        12350 + rank as u16,
    )?);

    // Create a large model
    let model = LargeModel::new(512, 1024, 8, 256)?;
    let total_params = model.count_parameters();
    println!("  📊 Model has {} parameters", total_params);

    // Wrap with FSDP using default configuration
    let config = FsdpConfig::default();
    let fsdp_model = fsdp_wrap(model, process_group.clone(), Some(config))?;

    println!("  ✅ FSDP model created successfully");
    println!(
        "  📈 Local sharding ratio: {:.2}%",
        fsdp_model.local_sharding_ratio() * 100.0
    );

    // Test forward pass
    let input = torsh_tensor::creation::randn(&[4, 512], Device::Cpu, DType::F32)?;
    let start_time = Instant::now();
    let output = fsdp_model.forward(&input)?;
    let forward_time = start_time.elapsed();

    println!("  🚀 Forward pass completed in {:?}", forward_time);
    println!("  📐 Output shape: {:?}", output.shape());

    // Check memory statistics
    let stats = fsdp_model.memory_stats();
    println!("  📊 All-gather operations: {}", stats.num_all_gathers);
    println!(
        "  📊 Reduce-scatter operations: {}",
        stats.num_reduce_scatters
    );

    Ok(())
}

/// Demonstrate different sharding strategies
async fn demo_sharding_strategies(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🎯 Sharding Strategies Demo (Worker {})", rank);
    println!("=====================================");

    let process_group = Arc::new(init_process_group(
        BackendType::Mock,
        rank,
        world_size,
        "127.0.0.1",
        12360 + rank as u16,
    )?);

    let strategies = vec![
        ("FullShard", ShardingStrategy::FullShard),
        ("ShardGradOp", ShardingStrategy::ShardGradOp),
        ("NoShard", ShardingStrategy::NoShard),
        ("HybridShard", ShardingStrategy::HybridShard),
    ];

    for (name, strategy) in strategies {
        println!("\n  🔄 Testing {} strategy", name);

        let model = LargeModel::new(256, 512, 4, 128)?;
        let config = FsdpConfig {
            sharding_strategy: strategy,
            min_num_params: 100, // Lower threshold for demonstration
            ..Default::default()
        };

        let fsdp_model = fsdp_wrap(model, process_group.clone(), Some(config))?;

        let input = torsh_tensor::creation::randn(&[2, 256], Device::Cpu, DType::F32)?;
        let start_time = Instant::now();
        let _output = fsdp_model.forward(&input)?;
        let time_taken = start_time.elapsed();

        println!("    ⏱️  Forward pass: {:?}", time_taken);
        println!(
            "    💾 Local ratio: {:.1}%",
            fsdp_model.local_sharding_ratio() * 100.0
        );

        let stats = fsdp_model.memory_stats();
        println!("    📈 All-gathers: {}", stats.num_all_gathers);
    }

    Ok(())
}

/// Demonstrate auto-wrapping policies
async fn demo_auto_wrapping(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🔀 Auto-Wrapping Policies Demo (Worker {})", rank);
    println!("=========================================");

    let process_group = Arc::new(init_process_group(
        BackendType::Mock,
        rank,
        world_size,
        "127.0.0.1",
        12370 + rank as u16,
    )?);

    let policies = vec![
        (
            "SizeBasedAutoWrap",
            AutoWrapPolicy::SizeBasedAutoWrap {
                min_num_params: 1000,
            },
        ),
        ("NoAutoWrap", AutoWrapPolicy::NoAutoWrap),
        ("CustomAutoWrap", AutoWrapPolicy::CustomAutoWrap),
    ];

    for (policy_name, policy) in policies {
        println!("\n  🎛️  Testing {} policy", policy_name);

        let model = LargeModel::new(128, 256, 6, 64)?;

        let fsdp_model = auto_wrap_modules(model, process_group.clone(), policy)?;

        let input = torsh_tensor::creation::randn(&[3, 128], Device::Cpu, DType::F32)?;
        let _output = fsdp_model.forward(&input)?;

        println!("    ✅ Auto-wrap successful");
        println!(
            "    📊 Sharding ratio: {:.1}%",
            fsdp_model.local_sharding_ratio() * 100.0
        );
    }

    Ok(())
}

/// Demonstrate memory optimization features
async fn demo_memory_optimization(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n💾 Memory Optimization Demo (Worker {})", rank);
    println!("====================================");

    let process_group = Arc::new(init_process_group(
        BackendType::Mock,
        rank,
        world_size,
        "127.0.0.1",
        12380 + rank as u16,
    )?);

    // Test different memory configurations
    let memory_configs = vec![
        ("Default", MemoryConfig::default()),
        (
            "LimitAllGathers",
            MemoryConfig {
                limit_all_gathers: true,
                use_orig_params: false,
                offload_to_cpu: false,
            },
        ),
        (
            "CPUOffload",
            MemoryConfig {
                limit_all_gathers: true,
                use_orig_params: false,
                offload_to_cpu: true,
            },
        ),
        (
            "UseOrigParams",
            MemoryConfig {
                limit_all_gathers: false,
                use_orig_params: true,
                offload_to_cpu: false,
            },
        ),
    ];

    for (config_name, memory_config) in memory_configs {
        println!("\n  ⚙️  Testing {} configuration", config_name);

        let model = LargeModel::new(256, 512, 5, 128)?;
        let config = FsdpConfig {
            memory_config,
            min_num_params: 500,
            ..Default::default()
        };

        let fsdp_model = fsdp_wrap(model, process_group.clone(), Some(config))?;

        // Simulate multiple forward passes
        for i in 0..3 {
            let input = torsh_tensor::creation::randn(&[2, 256], Device::Cpu, DType::F32)?;
            let start_time = Instant::now();
            let _output = fsdp_model.forward(&input)?;
            let time_taken = start_time.elapsed();

            println!("    🔄 Pass {}: {:?}", i + 1, time_taken);
        }

        let stats = fsdp_model.memory_stats();
        println!("    📊 Total all-gathers: {}", stats.num_all_gathers);
        println!("    💾 Memory saved: {:.1} MB", stats.memory_saved_mb);
    }

    Ok(())
}

/// Demonstrate mixed precision training
async fn demo_mixed_precision(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🎨 Mixed Precision Demo (Worker {})", rank);
    println!("===============================");

    let process_group = Arc::new(init_process_group(
        BackendType::Mock,
        rank,
        world_size,
        "127.0.0.1",
        12390 + rank as u16,
    )?);

    // Configure mixed precision
    let mixed_precision = MixedPrecisionConfig {
        param_dtype: DType::F16,
        reduce_dtype: DType::F32,
        buffer_dtype: DType::F32,
        keep_low_precision_grads: false,
    };

    let model = LargeModel::new(200, 400, 4, 100)?;
    let config = FsdpConfig {
        mixed_precision: Some(mixed_precision),
        min_num_params: 200,
        ..Default::default()
    };

    let fsdp_model = fsdp_wrap(model, process_group, Some(config))?;

    println!("  ✅ Mixed precision FSDP model created");
    println!("  🎯 Parameters in FP16, reductions in FP32");

    // Test forward pass with mixed precision
    let input = torsh_tensor::creation::randn(&[4, 200], Device::Cpu, DType::F32)?;
    let start_time = Instant::now();
    let _output = fsdp_model.forward(&input)?;
    let mixed_precision_time = start_time.elapsed();

    println!("  ⚡ Mixed precision forward: {:?}", mixed_precision_time);

    let stats = fsdp_model.memory_stats();
    println!(
        "  📊 Operations completed: {} all-gathers",
        stats.num_all_gathers
    );

    Ok(())
}

/// Demonstrate performance comparison
async fn demo_performance_comparison(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("\n🏁 Performance Comparison (Worker {})", rank);
    println!("=================================");

    let process_group = Arc::new(init_process_group(
        BackendType::Mock,
        rank,
        world_size,
        "127.0.0.1",
        12400 + rank as u16,
    )?);

    let model_sizes = vec![
        ("Small", 128, 256, 3),
        ("Medium", 256, 512, 5),
        ("Large", 512, 1024, 8),
    ];

    for (size_name, input_size, hidden_size, num_layers) in model_sizes {
        println!(
            "\n  📏 Testing {} model ({} -> {} x{})",
            size_name, input_size, hidden_size, num_layers
        );

        // Test regular model
        let regular_model = LargeModel::new(input_size, hidden_size, num_layers, input_size / 2)?;
        let input = torsh_tensor::creation::randn(&[4, input_size], Device::Cpu, DType::F32)?;

        let start_time = Instant::now();
        let _regular_output = regular_model.forward(&input)?;
        let regular_time = start_time.elapsed();

        // Test FSDP model
        let fsdp_regular_model =
            LargeModel::new(input_size, hidden_size, num_layers, input_size / 2)?;
        let fsdp_model = fsdp_wrap(fsdp_regular_model, process_group.clone(), None)?;

        let start_time = Instant::now();
        let _fsdp_output = fsdp_model.forward(&input)?;
        let fsdp_time = start_time.elapsed();

        println!("    🔄 Regular model: {:?}", regular_time);
        println!("    ⚡ FSDP model: {:?}", fsdp_time);
        println!(
            "    📊 FSDP ratio: {:.1}%",
            fsdp_model.local_sharding_ratio() * 100.0
        );

        let overhead = if fsdp_time > regular_time {
            fsdp_time.as_millis() as f64 / regular_time.as_millis() as f64 - 1.0
        } else {
            0.0
        };
        println!("    📈 FSDP overhead: {:.1}%", overhead * 100.0);
    }

    Ok(())
}

/// Initialize and run worker
async fn run_worker(rank: u32, world_size: u32) -> Result<(), Box<dyn Error>> {
    println!("[Worker {}] Starting FSDP demonstrations...", rank);

    // Run all demos
    demo_basic_fsdp(rank, world_size).await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    demo_sharding_strategies(rank, world_size).await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    demo_auto_wrapping(rank, world_size).await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    demo_memory_optimization(rank, world_size).await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    demo_mixed_precision(rank, world_size).await?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    demo_performance_comparison(rank, world_size).await?;

    println!("\n[Worker {}] All FSDP demonstrations completed! ✅", rank);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🌐 ToRSh Fully Sharded Data Parallel (FSDP) Demo");
    println!("================================================");

    // Simulate 2 workers for demonstration
    let world_size = 2;
    let mut worker_handles = Vec::new();

    // Start workers
    for rank in 0..world_size {
        let handle = tokio::spawn(async move {
            if let Err(e) = run_worker(rank, world_size).await {
                eprintln!("[Worker {}] Error: {}", rank, e);
            }
        });
        worker_handles.push(handle);
    }

    // Wait for workers to complete
    for handle in worker_handles {
        handle.await?;
    }

    println!("\n🏁 FSDP Demo Completed Successfully!");
    println!("\n📚 Key Features Demonstrated:");
    println!("  ✅ Parameter sharding across workers");
    println!("  ✅ Automatic gather/scatter during computation");
    println!("  ✅ Multiple sharding strategies");
    println!("  ✅ Auto-wrapping policies");
    println!("  ✅ Memory optimization configurations");
    println!("  ✅ Mixed precision training");
    println!("  ✅ Performance monitoring and statistics");

    println!("\n🚀 Production Benefits:");
    println!("  - Train models larger than single-GPU memory");
    println!(
        "  - Reduce memory usage by {}%",
        (1.0 - 1.0 / world_size as f64) * 100.0
    );
    println!("  - Maintain training throughput with optimized communication");
    println!("  - Support for hierarchical model parallelism");
    println!("  - Seamless integration with existing PyTorch-style code");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_model_creation() -> Result<(), Box<dyn Error>> {
        let model = LargeModel::new(64, 128, 3, 32)?;
        assert_eq!(model.num_layers, 3);
        assert_eq!(model.hidden_size, 128);
        assert!(model.count_parameters() > 0);
        Ok(())
    }

    #[test]
    fn test_large_model_forward() -> Result<(), Box<dyn Error>> {
        let model = LargeModel::new(32, 64, 2, 16)?;
        let input = torsh_tensor::creation::randn(&[2, 32]);
        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 16]);
        Ok(())
    }

    #[test]
    fn test_large_model_parameters() -> Result<(), Box<dyn Error>> {
        let model = LargeModel::new(10, 20, 2, 5)?;
        let params = model.parameters();

        // Should have weight and bias for each layer
        assert!(params.len() >= 4); // At least 2 layers × 2 params per layer

        // Check parameter naming
        assert!(params.contains_key("layer_0.weight"));
        assert!(params.contains_key("layer_0.bias"));
        assert!(params.contains_key("layer_1.weight"));
        assert!(params.contains_key("layer_1.bias"));

        Ok(())
    }

    #[test]
    fn test_parameter_counting() -> Result<(), Box<dyn Error>> {
        let model = LargeModel::new(10, 20, 2, 5)?;
        let param_count = model.count_parameters();

        // Manual calculation:
        // Layer 0: (10 * 20) + 20 = 220 (weight + bias)
        // Layer 1: (20 * 5) + 5 = 105 (weight + bias)
        // Total: 325
        assert_eq!(param_count, 325);

        Ok(())
    }
}
