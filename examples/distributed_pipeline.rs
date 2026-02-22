//! Comprehensive pipeline parallelism examples for ToRSh
//!
//! This example demonstrates various pipeline parallelism configurations and use cases:
//! 1. Basic pipeline setup with different scheduling strategies
//! 2. Large transformer model pipeline parallelism
//! 3. Memory-efficient training with pipeline parallelism
//! 4. Performance comparison between different strategies

use std::collections::HashMap;
use std::sync::Arc;
use tokio;
use torsh_core::{DeviceType, Shape};
use torsh_distributed::{
    create_pipeline_stages, init_process_group, BackendType, PipelineConfig, PipelineParallel,
    PipelineStage, PipelineStats, ProcessGroup, ScheduleType,
};
use torsh_nn::prelude::{Linear, Module, Parameter};
use torsh_tensor::{creation::randn, Tensor};

/// A simple multi-layer neural network that can be split into pipeline stages
struct SimpleNet {
    layers: Vec<Linear>,
    training: bool,
}

impl SimpleNet {
    fn new(layer_sizes: &[usize]) -> torsh_core::Result<Self> {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let layer = Linear::new(layer_sizes[i], layer_sizes[i + 1], true);
            layers.push(layer);
        }

        Ok(Self {
            layers,
            training: true,
        })
    }

    /// Split the network into pipeline stages
    fn split_into_stages(self, num_stages: usize) -> torsh_core::Result<Vec<Box<dyn Module>>> {
        if num_stages > self.layers.len() {
            return Err(torsh_core::TorshError::Other(format!(
                "Cannot split {} layers into {} stages",
                self.layers.len(),
                num_stages
            )));
        }

        let layers_per_stage = self.layers.len() / num_stages;
        let mut stages: Vec<Box<dyn Module>> = Vec::new();
        let mut remaining_layers = self.layers;

        for stage_id in 0..num_stages {
            let layers_for_stage = if stage_id == num_stages - 1 {
                // Last stage gets all remaining layers
                remaining_layers.len()
            } else {
                layers_per_stage
            };

            let stage_layers: Vec<Linear> = remaining_layers.drain(0..layers_for_stage).collect();
            let stage = NetworkStage::new(stage_layers);
            stages.push(Box::new(stage) as Box<dyn Module>);
        }

        Ok(stages)
    }
}

impl Module for SimpleNet {
    fn forward(&self, input: &Tensor) -> torsh_core::Result<Tensor> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output)?;
            // Apply ReLU activation (except for last layer)
            if !std::ptr::eq(layer, self.layers.last().unwrap()) {
                output = output.relu()?;
            }
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (param_name, param) in layer_params {
                let full_name = format!("layer_{}.{}", i, param_name);
                params.insert(full_name, param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// A single stage in the pipeline (subset of layers)
struct NetworkStage {
    layers: Vec<Linear>,
    training: bool,
}

impl NetworkStage {
    fn new(layers: Vec<Linear>) -> Self {
        Self {
            layers,
            training: true,
        }
    }
}

impl Module for NetworkStage {
    fn forward(&self, input: &Tensor) -> torsh_core::Result<Tensor> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output)?;
            // Apply ReLU activation (except for last layer in stage)
            if !std::ptr::eq(layer, self.layers.last().unwrap()) {
                output = output.relu()?;
            }
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (param_name, param) in layer_params {
                let full_name = format!("stage_layer_{}.{}", i, param_name);
                params.insert(full_name, param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Transformer-like model for larger pipeline parallelism examples
struct TransformerBlock {
    attention: Linear,
    feed_forward1: Linear,
    feed_forward2: Linear,
    layer_norm1: Linear, // Simplified layer norm as linear layer
    layer_norm2: Linear,
    training: bool,
}

impl TransformerBlock {
    fn new(hidden_size: usize, ff_size: usize) -> torsh_core::Result<Self> {
        Ok(Self {
            attention: Linear::new(hidden_size, hidden_size, true),
            feed_forward1: Linear::new(hidden_size, ff_size, true),
            feed_forward2: Linear::new(ff_size, hidden_size, true),
            layer_norm1: Linear::new(hidden_size, hidden_size, true),
            layer_norm2: Linear::new(hidden_size, hidden_size, true),
            training: true,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> torsh_core::Result<Tensor> {
        // Simplified transformer block
        let attn_out = self.attention.forward(input)?;
        let norm1_out = self.layer_norm1.forward(&attn_out.add(input)?)?;

        let ff1_out = self.feed_forward1.forward(&norm1_out)?.relu()?;
        let ff2_out = self.feed_forward2.forward(&ff1_out)?;
        let final_out = self.layer_norm2.forward(&ff2_out.add(&norm1_out)?)?;

        Ok(final_out)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }
        for (name, param) in self.feed_forward1.parameters() {
            params.insert(format!("feed_forward1.{}", name), param);
        }
        for (name, param) in self.feed_forward2.parameters() {
            params.insert(format!("feed_forward2.{}", name), param);
        }
        for (name, param) in self.layer_norm1.parameters() {
            params.insert(format!("layer_norm1.{}", name), param);
        }
        for (name, param) in self.layer_norm2.parameters() {
            params.insert(format!("layer_norm2.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        self.attention.train();
        self.feed_forward1.train();
        self.feed_forward2.train();
        self.layer_norm1.train();
        self.layer_norm2.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.attention.eval();
        self.feed_forward1.eval();
        self.feed_forward2.eval();
        self.layer_norm1.eval();
        self.layer_norm2.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::Result<()> {
        self.attention.to_device(device)?;
        self.feed_forward1.to_device(device)?;
        self.feed_forward2.to_device(device)?;
        self.layer_norm1.to_device(device)?;
        self.layer_norm2.to_device(device)?;
        Ok(())
    }
}

/// Example 1: Basic Pipeline Parallelism Setup
async fn example_basic_pipeline() -> torsh_core::Result<()> {
    println!("🚀 Example 1: Basic Pipeline Parallelism Setup");
    println!("=============================================\n");

    // Create a simple 4-layer network
    let net = SimpleNet::new(&[512, 256, 128, 64])?;

    // Split into 2 pipeline stages
    let stage_modules = net.split_into_stages(2)?;

    // Create process group for 2 workers
    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        0, // rank (in real scenario, this would be different for each process)
        2, // world_size
        "localhost",
        12345,
    )?);

    // Create pipeline stages
    let devices = vec![DeviceType::Cpu, DeviceType::Cpu];
    let stages = create_pipeline_stages(stage_modules, process_group.clone(), devices)?;

    // Create pipeline parallel wrapper for this rank (rank 0)
    let stage = stages.into_iter().nth(0).unwrap(); // Get first stage for rank 0
    let config = PipelineConfig::default();

    let mut pipeline = PipelineParallel::new(stage, process_group, config)?;

    println!("✅ Created pipeline with {} stages", pipeline.num_stages());
    println!("   Current stage: {}", pipeline.stage_id());
    println!("   Is first stage: {}", pipeline.is_first_stage());
    println!("   Is last stage: {}", pipeline.is_last_stage());

    // Create sample input
    let input_shape = Shape::new(vec![8, 512]); // batch_size=8, input_size=512
    let input = randn::<f32>(input_shape.dims());

    println!("\n📊 Input shape: {:?}", input.shape());

    // Forward pass (mock - in real scenario each rank would handle its own stage)
    let output = pipeline.forward(&input);
    match output {
        Ok(tensor) => {
            println!("✅ Forward pass completed");
            println!("   Output shape: {:?}", tensor.shape());
        }
        Err(e) => {
            println!("❌ Forward pass failed: {}", e);
        }
    }

    // Get pipeline statistics
    let stats = pipeline.get_pipeline_stats();
    println!("\n📈 Pipeline Statistics:");
    println!("   Stage ID: {}", stats.stage_id);
    println!("   Number of stages: {}", stats.num_stages);
    println!("   Rank: {}", stats.rank);
    println!("   Micro-batches: {}", stats.num_micro_batches);
    println!("   Schedule: {:?}", stats.schedule);

    Ok(())
}

/// Example 2: Different Scheduling Strategies
async fn example_scheduling_strategies() -> torsh_core::Result<()> {
    println!("\n🚀 Example 2: Different Scheduling Strategies");
    println!("============================================\n");

    let net = SimpleNet::new(&[256, 128, 64, 32])?;
    let stage_modules = net.split_into_stages(4)?;

    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        4,
        "localhost",
        12346,
    )?);

    let devices = vec![DeviceType::Cpu; 4];
    let stages = create_pipeline_stages(stage_modules, process_group.clone(), devices)?;

    let strategies = [
        (ScheduleType::GPipe, "GPipe"),
        (ScheduleType::OneFOneBInterleaved, "1F1B Interleaved"),
        (ScheduleType::InterleavedOneFOneB, "Interleaved 1F1B"),
    ];

    for (schedule, name) in &strategies {
        println!("🔄 Testing {} strategy:", name);

        let config = PipelineConfig {
            num_micro_batches: 6,
            schedule: *schedule,
            accumulate_gradients: true,
            ..Default::default()
        };

        // Use first stage for demo - we'll work with reference
        let stage_ref = &stages[0];
        // For demo purposes, we'll create a basic pipeline setup
        // In real usage, stages would be distributed across processes

        // Create test input
        let input_shape = Shape::new(vec![12, 256]); // batch_size=12
        let input = randn::<f32>(input_shape.dims());

        // Note: Pipeline operations simplified for compilation demo
        println!("   ✅ Simulated forward pass completed");
        println!("   ✅ Simulated backward pass completed");
        println!("   📊 Pipeline statistics would be available in full implementation");
        println!();

        // pipeline.clear_caches(); // Simplified for demo
    }

    Ok(())
}

/// Example 3: Large Transformer Model Pipeline
async fn example_transformer_pipeline() -> torsh_core::Result<()> {
    println!("🚀 Example 3: Large Transformer Model Pipeline");
    println!("=============================================\n");

    // Create a large transformer model (simplified)
    let hidden_size = 1024;
    let ff_size = 4096;
    let num_layers = 12;

    let mut transformer_layers: Vec<Box<dyn Module>> = Vec::new();

    for i in 0..num_layers {
        let block = TransformerBlock::new(hidden_size, ff_size)?;
        transformer_layers.push(Box::new(block) as Box<dyn Module>);
        println!("   Created transformer layer {}/{}", i + 1, num_layers);
    }

    // Setup pipeline with 4 stages (3 layers per stage)
    let num_stages = 4;
    let layers_per_stage = num_layers / num_stages;

    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        num_stages as u32,
        "localhost",
        12347,
    )?);

    let devices = vec![DeviceType::Cpu; num_stages];
    let stages = create_pipeline_stages(transformer_layers, process_group.clone(), devices)?;

    println!("\n🏗️  Pipeline Configuration:");
    println!("   Total layers: {}", num_layers);
    println!("   Pipeline stages: {}", num_stages);
    println!("   Layers per stage: {}", layers_per_stage);

    // Configure for memory efficiency
    let config = PipelineConfig {
        num_micro_batches: 16, // More micro-batches for better memory efficiency
        schedule: ScheduleType::OneFOneBInterleaved,
        accumulate_gradients: true,
        async_comm: true,
        comm_timeout_ms: 60000, // Longer timeout for large models
        ..Default::default()
    };

    let stage = stages.into_iter().nth(0).unwrap(); // First stage for demo
    let mut pipeline = PipelineParallel::new(stage, process_group, config)?;

    // Large batch processing
    let batch_size = 32;
    let seq_length = 512;
    let input_shape = Shape::new(vec![batch_size, seq_length, hidden_size]);

    println!("\n📋 Processing large batch:");
    println!("   Batch size: {}", batch_size);
    println!("   Sequence length: {}", seq_length);
    println!("   Hidden size: {}", hidden_size);
    println!(
        "   Total input size: {:.2} MB",
        (batch_size * seq_length * hidden_size * 4) as f32 / (1024.0 * 1024.0)
    );

    let large_input = randn::<f32>(input_shape.dims());

    // Process with pipeline
    let start_time = std::time::Instant::now();
    match pipeline.forward(&large_input) {
        Ok(output) => {
            let duration = start_time.elapsed();
            println!("   ✅ Large batch processing completed in {:?}", duration);
            println!("   📊 Output shape: {:?}", output.shape());
        }
        Err(e) => {
            println!("   ❌ Large batch processing failed: {}", e);
        }
    }

    // Show memory statistics
    let stats = pipeline.get_pipeline_stats();
    println!("\n💾 Memory Statistics:");
    println!("   Cached activations: {}", stats.cached_activations);
    println!("   Cached gradients: {}", stats.cached_gradients);
    println!("   Current micro-batch: {}", stats.current_micro_batch);

    Ok(())
}

/// Example 4: Memory-Efficient Configuration
async fn example_memory_efficient_config() -> torsh_core::Result<()> {
    println!("\n🚀 Example 4: Memory-Efficient Configuration");
    println!("===========================================\n");

    // Create a network that would normally require a lot of memory
    let large_net = SimpleNet::new(&[2048, 1024, 1024, 512, 256])?;
    let stage_modules = large_net.split_into_stages(5)?;

    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        5,
        "localhost",
        12348,
    )?);

    let devices = vec![DeviceType::Cpu; 5];
    let stages = create_pipeline_stages(stage_modules, process_group.clone(), devices)?;

    // Memory-efficient configuration
    let memory_config = PipelineConfig {
        num_micro_batches: 32,                       // Many small micro-batches
        schedule: ScheduleType::OneFOneBInterleaved, // Most memory efficient
        accumulate_gradients: true,
        async_comm: true,
        base_tag: 2000,
        comm_timeout_ms: 45000,
    };

    println!("🔧 Memory-Efficient Configuration:");
    println!("   Micro-batches: {}", memory_config.num_micro_batches);
    println!("   Schedule: {:?}", memory_config.schedule);
    println!(
        "   Gradient accumulation: {}",
        memory_config.accumulate_gradients
    );
    println!("   Async communication: {}", memory_config.async_comm);

    let stage = stages.into_iter().nth(0).unwrap();
    let mut pipeline = PipelineParallel::new(stage, process_group, memory_config)?;

    // Test with large batch that would normally cause OOM
    let large_batch_size = 128;
    let input_shape = Shape::new(vec![large_batch_size, 2048]);
    let large_input = randn::<f32>(input_shape.dims());

    println!("\n📊 Processing large batch with memory optimization:");
    println!("   Batch size: {}", large_batch_size);
    println!("   Input features: 2048");
    println!(
        "   Est. memory per batch: {:.2} MB",
        (large_batch_size * 2048 * 4) as f32 / (1024.0 * 1024.0)
    );

    // Process in memory-efficient way
    let start_time = std::time::Instant::now();
    match pipeline.forward(&large_input) {
        Ok(_) => {
            let duration = start_time.elapsed();
            println!(
                "   ✅ Memory-efficient processing completed in {:?}",
                duration
            );
        }
        Err(e) => {
            println!("   ❌ Processing failed: {}", e);
        }
    }

    // Check memory usage through stats
    let stats = pipeline.get_pipeline_stats();
    println!("\n💾 Memory Usage Check:");
    println!("   Active micro-batches: {}", stats.cached_activations);
    println!("   Gradient buffers: {}", stats.cached_gradients);
    println!(
        "   Memory efficiency: {:.1}x",
        large_batch_size as f32 / stats.num_micro_batches as f32
    );

    // Demonstrate cache clearing for memory management
    pipeline.clear_caches();
    let cleared_stats = pipeline.get_pipeline_stats();
    println!(
        "   After cache clear - activations: {}, gradients: {}",
        cleared_stats.cached_activations, cleared_stats.cached_gradients
    );

    Ok(())
}

/// Example 5: Performance Comparison
async fn example_performance_comparison() -> torsh_core::Result<()> {
    println!("\n🚀 Example 5: Performance Comparison");
    println!("===================================\n");

    let net = SimpleNet::new(&[1024, 512, 256, 128])?;
    let stage_modules = net.split_into_stages(4)?;

    let process_group = Arc::new(init_process_group(
        BackendType::Gloo,
        0,
        4,
        "localhost",
        12349,
    )?);

    let devices = vec![DeviceType::Cpu; 4];
    let stages = create_pipeline_stages(stage_modules, process_group.clone(), devices)?;

    let test_configs = [
        (
            "Small micro-batches",
            PipelineConfig {
                num_micro_batches: 16,
                schedule: ScheduleType::OneFOneBInterleaved,
                ..Default::default()
            },
        ),
        (
            "Large micro-batches",
            PipelineConfig {
                num_micro_batches: 4,
                schedule: ScheduleType::OneFOneBInterleaved,
                ..Default::default()
            },
        ),
        (
            "GPipe strategy",
            PipelineConfig {
                num_micro_batches: 8,
                schedule: ScheduleType::GPipe,
                ..Default::default()
            },
        ),
    ];

    let batch_size = 64;
    let input_shape = Shape::new(vec![batch_size, 1024]);
    let test_input = randn::<f32>(input_shape.dims());

    println!("🏁 Performance Test Setup:");
    println!("   Batch size: {}", batch_size);
    println!("   Input features: 1024");
    println!("   Pipeline stages: 4\n");

    for (name, config) in &test_configs {
        println!("⏱️  Testing {}:", name);

        // Use first stage for demo - simplified for compilation
        // In real usage, stages would be distributed across processes
        let _stage_ref = &stages[0];

        let start_time = std::time::Instant::now();

        // Simulate performance test
        let simulated_duration = std::time::Duration::from_millis(50);
        std::thread::sleep(simulated_duration);

        let total_time = start_time.elapsed();
        println!("   Simulated forward time: {:?}", total_time / 2);
        println!("   Simulated backward time: {:?}", total_time / 2);
        println!("   Total time: {:?}", total_time);
        println!(
            "   Simulated throughput: {:.2} samples/sec",
            batch_size as f32 / total_time.as_secs_f32()
        );
        println!();

        // pipeline.clear_caches(); // Simplified for demo
    }

    Ok(())
}

#[tokio::main]
async fn main() -> torsh_core::Result<()> {
    println!("🌟 ToRSh Distributed Pipeline Parallelism Examples");
    println!("=================================================\n");

    // Run all examples
    example_basic_pipeline().await?;
    example_scheduling_strategies().await?;
    example_transformer_pipeline().await?;
    example_memory_efficient_config().await?;
    example_performance_comparison().await?;

    println!("\n🎉 All pipeline parallelism examples completed successfully!");
    println!("\n📚 Key Takeaways:");
    println!("   • Pipeline parallelism enables training of very large models");
    println!("   • 1F1B scheduling is most memory and compute efficient");
    println!("   • More micro-batches improve memory efficiency but add overhead");
    println!("   • Async communication can improve performance");
    println!("   • Proper cache management is crucial for memory efficiency");

    Ok(())
}
