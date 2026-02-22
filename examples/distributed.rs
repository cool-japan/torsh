//! Distributed Training Example using ToRSh
//!
//! This example demonstrates:
//! - Data parallel training across multiple devices
//! - Model parallel training for large models
//! - Gradient synchronization and all-reduce operations
//! - Distributed data loading with sharding
//! - Mixed precision training
//! - Gradient accumulation for large batch sizes

use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use torsh::nn::{BatchNorm1d, Linear, Module, ReLU, Sequential};
use torsh::optim::{Adam, Optimizer};
use torsh::prelude::*;
use torsh::tensor::Tensor;

/// Distributed training configuration
#[derive(Debug, Clone)]
struct DistributedConfig {
    world_size: usize,
    rank: usize,
    backend: String,
    master_addr: String,
    master_port: u16,
    local_rank: usize,
    gradient_accumulation_steps: usize,
    mixed_precision: bool,
    gradient_clipping: Option<f32>,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 4,
            rank: 0,
            backend: "gloo".to_string(), // or "nccl" for GPUs
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            local_rank: 0,
            gradient_accumulation_steps: 1,
            mixed_precision: false,
            gradient_clipping: Some(1.0),
        }
    }
}

/// Process group for distributed communication
struct ProcessGroup {
    config: DistributedConfig,
    communicator: Arc<Mutex<Communicator>>,
}

impl ProcessGroup {
    fn new(config: DistributedConfig) -> Self {
        let communicator = Arc::new(Mutex::new(Communicator::new(&config)));
        Self {
            config,
            communicator,
        }
    }

    /// Initialize the process group
    fn init(&self) -> Result<(), Box<dyn Error>> {
        println!("🌐 Initializing process group...");
        println!("  Rank: {}/{}", self.config.rank, self.config.world_size);
        println!("  Backend: {}", self.config.backend);
        println!(
            "  Master: {}:{}",
            self.config.master_addr, self.config.master_port
        );

        // In a real implementation, we would initialize MPI/NCCL/Gloo here
        Ok(())
    }

    /// All-reduce operation for gradient synchronization
    fn all_reduce(&self, tensor: &mut Tensor) -> Result<(), Box<dyn Error>> {
        let comm = self.communicator.lock().unwrap();
        comm.all_reduce(tensor, "sum")?;

        // Average gradients across all processes
        *tensor = tensor.div_scalar(self.config.world_size as f32)?;
        Ok(())
    }

    /// Broadcast tensor from source rank to all others
    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<(), Box<dyn Error>> {
        let comm = self.communicator.lock().unwrap();
        comm.broadcast(tensor, src_rank)?;
        Ok(())
    }

    /// Barrier synchronization
    fn barrier(&self) -> Result<(), Box<dyn Error>> {
        let comm = self.communicator.lock().unwrap();
        comm.barrier()?;
        Ok(())
    }
}

/// Mock communicator for demonstration
struct Communicator {
    rank: usize,
    world_size: usize,
}

impl Communicator {
    fn new(config: &DistributedConfig) -> Self {
        Self {
            rank: config.rank,
            world_size: config.world_size,
        }
    }

    fn all_reduce(&self, tensor: &mut Tensor, op: &str) -> Result<(), Box<dyn Error>> {
        // Simulate all-reduce communication
        std::thread::sleep(std::time::Duration::from_millis(10));
        Ok(())
    }

    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<(), Box<dyn Error>> {
        // Simulate broadcast communication
        std::thread::sleep(std::time::Duration::from_millis(5));
        Ok(())
    }

    fn barrier(&self) -> Result<(), Box<dyn Error>> {
        // Simulate barrier synchronization
        std::thread::sleep(std::time::Duration::from_millis(2));
        Ok(())
    }
}

/// Distributed Data Parallel wrapper
struct DistributedDataParallel<M: Module> {
    module: M,
    process_group: Arc<ProcessGroup>,
    device_ids: Vec<usize>,
}

impl<M: Module> DistributedDataParallel<M> {
    fn new(module: M, process_group: Arc<ProcessGroup>, device_ids: Vec<usize>) -> Self {
        Self {
            module,
            process_group,
            device_ids,
        }
    }

    /// Synchronize gradients across all processes
    fn sync_gradients(&self) -> Result<(), Box<dyn Error>> {
        for param in self.module.parameters() {
            if let Some(grad) = param.grad() {
                let mut grad_clone = grad.clone();
                self.process_group.all_reduce(&mut grad_clone)?;
                // Update the gradient with synchronized version
                param.set_grad(Some(grad_clone))?;
            }
        }
        Ok(())
    }
}

impl<M: Module> Module for DistributedDataParallel<M> {
    type Error = M::Error;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        self.module.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.module.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        self.module.named_parameters()
    }
}

/// Model parallel example - split model across devices
struct ModelParallelNet {
    // First half of model on device 0
    layer1: Linear,
    layer2: Linear,

    // Second half of model on device 1
    layer3: Linear,
    layer4: Linear,

    device_map: Vec<usize>,
}

impl ModelParallelNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            layer1: Linear::new(input_size, hidden_size, true),
            layer2: Linear::new(hidden_size, hidden_size, true),
            layer3: Linear::new(hidden_size, hidden_size, true),
            layer4: Linear::new(hidden_size, output_size, true),
            device_map: vec![0, 0, 1, 1], // Which device each layer runs on
        }
    }
}

impl Module for ModelParallelNet {
    type Error = torsh_core::TorshError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        // First part on device 0
        let x = self.layer1.forward(input)?;
        let x = x.relu()?;
        let x = self.layer2.forward(&x)?;
        let x = x.relu()?;

        // Transfer to device 1
        let x = x.to_device(1)?; // Simulated device transfer

        // Second part on device 1
        let x = self.layer3.forward(&x)?;
        let x = x.relu()?;
        let x = self.layer4.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.layer1.weight,
            &self.layer1.bias.as_ref().unwrap(),
            &self.layer2.weight,
            &self.layer2.bias.as_ref().unwrap(),
            &self.layer3.weight,
            &self.layer3.bias.as_ref().unwrap(),
            &self.layer4.weight,
            &self.layer4.bias.as_ref().unwrap(),
        ]
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        vec![
            ("layer1.weight".to_string(), &self.layer1.weight),
            (
                "layer1.bias".to_string(),
                self.layer1.bias.as_ref().unwrap(),
            ),
            ("layer2.weight".to_string(), &self.layer2.weight),
            (
                "layer2.bias".to_string(),
                self.layer2.bias.as_ref().unwrap(),
            ),
            ("layer3.weight".to_string(), &self.layer3.weight),
            (
                "layer3.bias".to_string(),
                self.layer3.bias.as_ref().unwrap(),
            ),
            ("layer4.weight".to_string(), &self.layer4.weight),
            (
                "layer4.bias".to_string(),
                self.layer4.bias.as_ref().unwrap(),
            ),
        ]
    }
}

/// Distributed data loader that shards data across processes
struct DistributedDataLoader {
    data: Vec<(Tensor, usize)>,
    batch_size: usize,
    rank: usize,
    world_size: usize,
    shuffle: bool,
}

impl DistributedDataLoader {
    fn new(
        data: Vec<(Tensor, usize)>,
        batch_size: usize,
        rank: usize,
        world_size: usize,
        shuffle: bool,
    ) -> Self {
        Self {
            data,
            batch_size,
            rank,
            world_size,
            shuffle,
        }
    }

    /// Get the shard of data for this process
    fn get_shard(&self) -> Vec<(Tensor, usize)> {
        let total_size = self.data.len();
        let shard_size = total_size / self.world_size;
        let start_idx = self.rank * shard_size;
        let end_idx = if self.rank == self.world_size - 1 {
            total_size
        } else {
            (self.rank + 1) * shard_size
        };

        self.data[start_idx..end_idx].to_vec()
    }

    /// Iterator over batches
    fn iter(&self) -> impl Iterator<Item = Vec<(Tensor, usize)>> + '_ {
        let shard = self.get_shard();
        shard
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

/// Mixed precision training utilities
struct MixedPrecisionScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_update: usize,
}

impl MixedPrecisionScaler {
    fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_update: 0,
        }
    }

    /// Scale the loss for mixed precision training
    fn scale_loss(&self, loss: &Tensor) -> Result<Tensor, torsh_core::TorshError> {
        loss.mul_scalar(self.scale)
    }

    /// Unscale gradients before optimizer step
    fn unscale_gradients(&self, optimizer: &mut dyn Optimizer) -> Result<(), Box<dyn Error>> {
        for param_group in optimizer.param_groups() {
            for param in param_group {
                if let Some(grad) = param.grad() {
                    let unscaled = grad.div_scalar(self.scale)?;
                    param.set_grad(Some(unscaled))?;
                }
            }
        }
        Ok(())
    }

    /// Update scale factor based on gradient overflow
    fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.scale *= self.backoff_factor;
            self.steps_since_update = 0;
        } else {
            self.steps_since_update += 1;
            if self.steps_since_update >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }
    }
}

/// Training function for distributed data parallel
fn train_ddp(rank: usize, world_size: usize, backend: &str) -> Result<(), Box<dyn Error>> {
    println!("\n🚀 Starting DDP training on rank {}", rank);

    // Initialize process group
    let config = DistributedConfig {
        rank,
        world_size,
        backend: backend.to_string(),
        ..Default::default()
    };

    let process_group = Arc::new(ProcessGroup::new(config.clone()));
    process_group.init()?;

    // Create model
    let model = Sequential::new()
        .add_module("fc1", Linear::new(784, 256, true))
        .add_module("relu1", ReLU::new(false))
        .add_module("bn1", BatchNorm1d::new(256, 1e-5, 0.1, true, true))
        .add_module("fc2", Linear::new(256, 128, true))
        .add_module("relu2", ReLU::new(false))
        .add_module("bn2", BatchNorm1d::new(128, 1e-5, 0.1, true, true))
        .add_module("fc3", Linear::new(128, 10, true));

    // Wrap in DDP
    let ddp_model = DistributedDataParallel::new(model, process_group.clone(), vec![rank]);

    // Create optimizer
    let mut optimizer = Adam::builder().learning_rate(0.001).build();

    for param in ddp_model.parameters() {
        optimizer.add_param_group(param.clone());
    }

    // Create distributed data loader
    let mut train_data = vec![];
    for i in 0..1000 {
        let data = Tensor::randn(&[784])?;
        let label = (i % 10) as usize;
        train_data.push((data, label));
    }

    let dataloader = DistributedDataLoader::new(train_data, 32, rank, world_size, true);

    // Training loop
    println!("Starting training loop...");
    let start_time = Instant::now();

    for epoch in 1..=3 {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            // Stack batch data
            let batch_size = batch.len();
            let mut input_data = vec![];
            let mut labels = vec![];

            for (data, label) in batch {
                input_data.push(data);
                labels.push(label);
            }

            let input = Tensor::stack(&input_data, 0)?;
            let target = Tensor::from_vec(labels, &[batch_size])?;

            // Forward pass
            let output = ddp_model.forward(&input)?;
            let loss = cross_entropy_loss(&output, &target)?;

            // Backward pass
            optimizer.zero_grad()?;
            loss.backward()?;

            // Synchronize gradients
            ddp_model.sync_gradients()?;

            // Optimizer step
            optimizer.step()?;

            total_loss += loss.item::<f32>();
            batch_count += 1;
        }

        // Synchronize metrics across processes
        process_group.barrier()?;

        let avg_loss = total_loss / batch_count as f32;
        println!("Rank {} - Epoch {}: Loss = {:.4}", rank, epoch, avg_loss);
    }

    let elapsed = start_time.elapsed();
    println!(
        "Rank {} - Training completed in {:.2}s",
        rank,
        elapsed.as_secs_f32()
    );

    Ok(())
}

/// Simulate distributed training across multiple processes
fn simulate_distributed_training() -> Result<(), Box<dyn Error>> {
    println!("\n🌍 Simulating Distributed Training");
    println!("===================================");

    let world_size = 4;
    let backend = "gloo";

    // Spawn threads to simulate multiple processes
    let mut handles = vec![];

    for rank in 0..world_size {
        let handle = thread::spawn(move || {
            train_ddp(rank, world_size, backend).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all processes to complete
    for handle in handles {
        handle.join().unwrap();
    }

    println!("\n✅ All processes completed!");

    Ok(())
}

/// Pipeline parallel training example
fn pipeline_parallel_example() -> Result<(), Box<dyn Error>> {
    println!("\n🔧 Pipeline Parallel Training");
    println!("=============================");

    // Split model into stages
    println!("Model split into 4 pipeline stages:");
    println!("  Stage 0: Embedding + Transformer Blocks 0-2");
    println!("  Stage 1: Transformer Blocks 3-5");
    println!("  Stage 2: Transformer Blocks 6-8");
    println!("  Stage 3: Transformer Blocks 9-11 + Output");

    // Simulate micro-batching
    let num_micro_batches = 4;
    let micro_batch_size = 8;

    println!("\nMicro-batching configuration:");
    println!("  Number of micro-batches: {}", num_micro_batches);
    println!("  Micro-batch size: {}", micro_batch_size);

    // Forward pass schedule
    println!("\nPipeline schedule (1F1B):");
    for step in 0..8 {
        print!("Step {}: ", step);
        for stage in 0..4 {
            let activity = match (step + stage) % 8 {
                0..=3 => "F", // Forward
                4..=7 => "B", // Backward
                _ => " ",
            };
            print!("Stage{}: {} ", stage, activity);
        }
        println!();
    }

    Ok(())
}

/// Demonstrate gradient accumulation
fn gradient_accumulation_example() -> Result<(), Box<dyn Error>> {
    println!("\n📊 Gradient Accumulation");
    println!("========================");

    let true_batch_size = 256;
    let micro_batch_size = 32;
    let accumulation_steps = true_batch_size / micro_batch_size;

    println!("Configuration:");
    println!("  Effective batch size: {}", true_batch_size);
    println!("  Micro-batch size: {}", micro_batch_size);
    println!("  Accumulation steps: {}", accumulation_steps);

    // Create simple model
    let model = Linear::new(100, 10, true);
    let mut optimizer = Adam::new(0.001);
    optimizer.add_param_group(model.weight.clone());
    optimizer.add_param_group(model.bias.as_ref().unwrap().clone());

    // Training with gradient accumulation
    println!("\nTraining with gradient accumulation:");

    for step in 0..accumulation_steps {
        // Process micro-batch
        let input = Tensor::randn(&[micro_batch_size, 100])?;
        let target = Tensor::randint(0, 10, &[micro_batch_size])?;

        let output = model.forward(&input)?;
        let loss = cross_entropy_loss(&output, &target)?;

        // Scale loss by accumulation steps
        let scaled_loss = loss.div_scalar(accumulation_steps as f32)?;
        scaled_loss.backward()?;

        println!(
            "  Micro-batch {}: Loss = {:.4}",
            step + 1,
            loss.item::<f32>()
        );

        // Update weights only after all accumulation steps
        if (step + 1) % accumulation_steps == 0 {
            optimizer.step()?;
            optimizer.zero_grad()?;
            println!("  → Weights updated!");
        }
    }

    Ok(())
}

/// Cross-entropy loss function
fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor, torsh_core::TorshError> {
    let log_probs = logits.log_softmax(1)?;
    let batch_size = targets.shape().dims()[0];

    let mut loss_sum = 0.0;
    for i in 0..batch_size {
        let target_idx = targets.get(i)?.item::<i64>() as usize;
        let log_prob = log_probs.get([i, target_idx])?.item::<f32>();
        loss_sum -= log_prob;
    }

    Tensor::scalar(loss_sum / batch_size as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 ToRSh Distributed Training Example");
    println!("=====================================");

    // Display distributed training strategies
    println!("\n📚 Distributed Training Strategies:");
    println!("1. Data Parallel (DP/DDP)");
    println!("   - Replicate model on each device");
    println!("   - Split data across devices");
    println!("   - Synchronize gradients");

    println!("\n2. Model Parallel");
    println!("   - Split model layers across devices");
    println!("   - Forward activations between devices");
    println!("   - Useful for very large models");

    println!("\n3. Pipeline Parallel");
    println!("   - Split model into stages");
    println!("   - Process micro-batches in pipeline");
    println!("   - Hide communication latency");

    println!("\n4. Tensor/Operator Parallel");
    println!("   - Split individual operations");
    println!("   - Parallelize matrix multiplications");
    println!("   - Maximum parallelism");

    // Demonstrate gradient accumulation
    gradient_accumulation_example()?;

    // Demonstrate pipeline parallelism
    pipeline_parallel_example()?;

    // Mixed precision training demo
    println!("\n⚡ Mixed Precision Training");
    println!("===========================");

    let mut scaler = MixedPrecisionScaler::new();
    println!("Initial loss scale: {}", scaler.scale);

    // Simulate training steps
    for step in 0..5 {
        let found_inf = step == 2; // Simulate overflow on step 2
        scaler.update(found_inf);

        println!(
            "Step {}: Scale = {}, Overflow = {}",
            step, scaler.scale, found_inf
        );
    }

    // Performance comparison
    println!("\n📊 Performance Comparison");
    println!("========================");
    println!("Single GPU baseline: 100 samples/sec");
    println!("Data Parallel (4 GPUs): ~380 samples/sec (95% scaling)");
    println!("Model Parallel (4 GPUs): ~250 samples/sec (memory bound)");
    println!("Pipeline Parallel (4 GPUs): ~350 samples/sec (depends on model)");

    // Communication patterns
    println!("\n📡 Communication Patterns");
    println!("========================");
    println!("All-Reduce: O(n) communication for gradient sync");
    println!("Broadcast: O(1) from source to all others");
    println!("All-Gather: Collect tensors from all ranks");
    println!("Reduce-Scatter: Reduce and distribute chunks");

    // Best practices
    println!("\n✨ Best Practices");
    println!("=================");
    println!("1. Use DDP over DP for better performance");
    println!("2. Overlap communication with computation");
    println!("3. Use gradient accumulation for large batches");
    println!("4. Profile to find communication bottlenecks");
    println!("5. Consider mixed precision for speedup");
    println!("6. Use NCCL backend for NVIDIA GPUs");

    // Simulate distributed training
    simulate_distributed_training()?;

    println!("\n✅ Distributed training example completed!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::default();
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);
        assert_eq!(config.backend, "gloo");
    }

    #[test]
    fn test_distributed_data_loader() {
        let data: Vec<(Tensor, usize)> = (0..100)
            .map(|i| (Tensor::scalar(i as f32), i % 10))
            .collect();

        let loader = DistributedDataLoader::new(data, 10, 0, 4, false);
        let shard = loader.get_shard();

        assert_eq!(shard.len(), 25); // 100 / 4 = 25 per rank
    }

    #[test]
    fn test_mixed_precision_scaler() {
        let mut scaler = MixedPrecisionScaler::new();
        let initial_scale = scaler.scale;

        // Test backoff on overflow
        scaler.update(true);
        assert_eq!(scaler.scale, initial_scale * 0.5);

        // Test growth
        scaler.growth_interval = 1; // For testing
        scaler.update(false);
        scaler.update(false);
        assert_eq!(scaler.scale, initial_scale); // Back to original
    }
}
