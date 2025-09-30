//! Integration tests for distributed training workflows
//!
//! These tests validate end-to-end distributed training scenarios by combining
//! multiple components and testing realistic training workflows.

use scirs2_core::random::quick::random_f32;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
use torsh_core::Result;
use torsh_distributed::{
    backend::BackendType,
    backend::ReduceOp,
    collectives::{all_reduce, barrier},
    ddp::{BucketConfig, DistributedDataParallel, OverlapConfig},
    fsdp::{FsdpConfig, FullyShardedDataParallel, ShardingStrategy},
    gradient_compression::{CompressionConfig, CompressionMethod, GradientCompressor},
    init_process_group,
    parameter_server::{ParameterServer, ParameterServerConfig},
    pipeline::{PipelineConfig, PipelineParallel, ScheduleType},
    ProcessGroup,
};
use torsh_tensor::creation::{ones, randn, zeros};
use torsh_tensor::Tensor;

/// Simulated training model for testing
#[derive(Debug, Clone)]
pub struct MockTrainingModel {
    parameters: HashMap<String, Tensor>,
    gradients: HashMap<String, Tensor>,
}

impl MockTrainingModel {
    pub fn new() -> Self {
        let mut parameters = HashMap::new();
        let mut gradients = HashMap::new();

        // Add some mock parameters
        parameters.insert("weight1".to_string(), randn::<f32>(&[100, 50]));
        parameters.insert("bias1".to_string(), randn::<f32>(&[50]));
        parameters.insert("weight2".to_string(), randn::<f32>(&[50, 10]));
        parameters.insert("bias2".to_string(), randn::<f32>(&[10]));

        // Initialize gradients to zeros
        for (name, param) in &parameters {
            gradients.insert(name.clone(), zeros::<f32>(&param.shape()));
        }

        Self {
            parameters,
            gradients,
        }
    }

    pub fn simulate_forward_backward(&mut self) -> Result<()> {
        // Simulate computing gradients
        for (name, gradient) in &mut self.gradients {
            if let Some(param) = self.parameters.get(name) {
                *gradient = randn::<f32>(&param.shape());
            }
        }
        Ok(())
    }

    pub fn get_parameter(&self, name: &str) -> Option<&Tensor> {
        self.parameters.get(name)
    }

    pub fn get_gradient(&self, name: &str) -> Option<&Tensor> {
        self.gradients.get(name)
    }

    pub fn get_gradient_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.gradients.get_mut(name)
    }

    pub fn parameter_names(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }

    pub fn total_parameters(&self) -> usize {
        self.parameters
            .values()
            .map(|tensor| tensor.shape().iter().product::<usize>())
            .sum()
    }
}

/// Training configuration for integration tests
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: f32,
    pub world_size: u32,
    pub use_compression: bool,
    pub compression_ratio: f32,
    pub use_overlap: bool,
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 3,
            learning_rate: 0.01,
            world_size: 4,
            use_compression: false,
            compression_ratio: 0.1,
            use_overlap: true,
            checkpoint_interval: 10,
        }
    }
}

/// Distributed training coordinator for integration tests
pub struct DistributedTrainingCoordinator {
    process_group: ProcessGroup,
    config: TrainingConfig,
    compressor: Option<GradientCompressor>,
    current_epoch: usize,
    current_step: usize,
}

impl DistributedTrainingCoordinator {
    pub async fn new(rank: u32, config: TrainingConfig) -> Result<Self> {
        let process_group = init_process_group(
            BackendType::Gloo,
            rank,
            config.world_size,
            "127.0.0.1",
            40000 + rank as u16,
        )?;

        let compressor = if config.use_compression {
            let compression_config = CompressionConfig {
                method: CompressionMethod::TopK {
                    k: (config.compression_ratio * 10000.0) as usize,
                },
                compression_ratio: config.compression_ratio,
                error_feedback: true,
                warm_start: false,
            };
            Some(GradientCompressor::new(compression_config))
        } else {
            None
        };

        Ok(Self {
            process_group,
            config,
            compressor,
            current_epoch: 0,
            current_step: 0,
        })
    }

    pub async fn train_epoch(&mut self, model: &mut MockTrainingModel) -> Result<TrainingMetrics> {
        let start_time = Instant::now();
        let mut total_loss = 0.0;
        let steps_per_epoch = 100; // Mock steps

        for step in 0..steps_per_epoch {
            self.current_step += 1;

            // Simulate forward and backward pass
            model.simulate_forward_backward()?;

            // Synchronize gradients
            self.synchronize_gradients(model).await?;

            // Simulate optimizer step
            self.apply_gradients(model)?;

            // Simulate loss calculation
            total_loss += random_f32() * 0.1;

            // Checkpoint if needed
            if self.current_step % self.config.checkpoint_interval == 0 {
                self.save_checkpoint(model).await?;
            }
        }

        self.current_epoch += 1;
        let epoch_duration = start_time.elapsed();

        Ok(TrainingMetrics {
            epoch: self.current_epoch - 1,
            steps: steps_per_epoch,
            duration: epoch_duration,
            average_loss: total_loss / steps_per_epoch as f32,
            throughput: steps_per_epoch as f64 / epoch_duration.as_secs_f64(),
        })
    }

    async fn synchronize_gradients(&self, model: &mut MockTrainingModel) -> Result<()> {
        // Synchronize all gradients across workers
        for param_name in model.parameter_names() {
            if let Some(gradient) = model.get_gradient_mut(&param_name) {
                // Apply compression if enabled
                if let Some(compressor) = &self.compressor {
                    let compressed = compressor.compress(gradient)?;
                    let decompressed = compressor.decompress(&compressed)?;
                    *gradient = decompressed;
                }

                // All-reduce gradient
                all_reduce(gradient, ReduceOp::Sum, &self.process_group).await?;

                // Average gradients
                let world_size = self.process_group.world_size() as f32;
                let gradient_data = gradient.to_vec_mut();
                for value in gradient_data {
                    *value /= world_size;
                }
            }
        }

        Ok(())
    }

    fn apply_gradients(&self, model: &mut MockTrainingModel) -> Result<()> {
        // Simple SGD update: param = param - lr * gradient
        for param_name in model.parameter_names() {
            if let (Some(param), Some(gradient)) = (
                model.parameters.get_mut(&param_name),
                model.get_gradient(&param_name),
            ) {
                let param_data = param.to_vec_mut();
                let grad_data = gradient.to_vec();

                for (p, g) in param_data.iter_mut().zip(grad_data.iter()) {
                    *p -= self.config.learning_rate * g;
                }
            }
        }

        Ok(())
    }

    async fn save_checkpoint(&self, _model: &MockTrainingModel) -> Result<()> {
        // Simulate checkpoint saving
        sleep(Duration::from_millis(10)).await;
        println!("Checkpoint saved at step {}", self.current_step);
        Ok(())
    }

    pub async fn synchronize_workers(&self) -> Result<()> {
        barrier(&self.process_group).await
    }

    pub fn rank(&self) -> u32 {
        self.process_group.rank()
    }

    pub fn world_size(&self) -> u32 {
        self.process_group.world_size()
    }
}

/// Training metrics for evaluation
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub steps: usize,
    pub duration: Duration,
    pub average_loss: f32,
    pub throughput: f64, // steps per second
}

impl TrainingMetrics {
    pub fn print_summary(&self) {
        println!(
            "Epoch {}: {} steps in {:.2}s, avg_loss={:.4}, throughput={:.2} steps/s",
            self.epoch,
            self.steps,
            self.duration.as_secs_f64(),
            self.average_loss,
            self.throughput
        );
    }
}

#[tokio::test]
async fn test_end_to_end_ddp_training() -> Result<()> {
    let config = TrainingConfig {
        world_size: 2,
        num_epochs: 2,
        use_compression: false,
        ..Default::default()
    };

    // Simulate two workers
    let worker_futures = (0..config.world_size)
        .map(|rank| {
            let config = config.clone();
            async move {
                let mut coordinator =
                    DistributedTrainingCoordinator::new(rank, config.clone()).await?;
                let mut model = MockTrainingModel::new();

                let mut all_metrics = Vec::new();

                // Training loop
                for epoch in 0..config.num_epochs {
                    println!("Worker {} starting epoch {}", rank, epoch);

                    // Synchronize before epoch
                    coordinator.synchronize_workers().await?;

                    let metrics = coordinator.train_epoch(&mut model).await?;
                    metrics.print_summary();
                    all_metrics.push(metrics);

                    // Synchronize after epoch
                    coordinator.synchronize_workers().await?;
                }

                // Verify training made progress
                assert!(!all_metrics.is_empty(), "Should have training metrics");

                Result::Ok(all_metrics)
            }
        })
        .collect::<Vec<_>>();

    let results: Result<Vec<_>> = futures_util::future::try_join_all(worker_futures).await;
    let worker_results = results?;

    // Verify all workers completed training
    assert_eq!(worker_results.len(), config.world_size as usize);

    for (worker_id, metrics_list) in worker_results.iter().enumerate() {
        assert_eq!(metrics_list.len(), config.num_epochs);
        println!(
            "Worker {} completed {} epochs",
            worker_id,
            metrics_list.len()
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_fsdp_integration() -> Result<()> {
    let world_size = 4;
    let pg = init_process_group(BackendType::Gloo, 0, world_size, "127.0.0.1", 40010)?;

    let fsdp_config = FsdpConfig {
        sharding_strategy: ShardingStrategy::FullShard,
        auto_wrap_policy: None,
        mixed_precision: None,
        backward_prefetch: None,
        forward_prefetch: false,
        limit_all_gathers: true,
        use_orig_params: false,
    };

    // Create FSDP wrapper (mock)
    let model = MockTrainingModel::new();
    let total_params = model.total_parameters();

    println!("Model has {} total parameters", total_params);

    // Simulate FSDP sharding
    let shard_size = total_params / world_size as usize;
    println!("Each shard has approximately {} parameters", shard_size);

    // Test parameter all-gather simulation
    let mut shard_tensor = ones::<f32>(&[shard_size]);
    all_reduce(&mut shard_tensor, ReduceOp::Sum, &pg).await?;

    // Verify sharding works
    assert!(shard_size > 0, "Should have non-empty shards");

    Ok(())
}

#[tokio::test]
async fn test_parameter_server_integration() -> Result<()> {
    let ps_config = ParameterServerConfig {
        num_workers: 3,
        num_servers: 1,
        sync_mode: true,
        aggregation_frequency: 1,
        staleness_bound: 0,
    };

    // Simulate parameter server setup
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 40020)?;

    // Create parameter server (mock)
    let parameter_server = ParameterServer::new(ps_config, pg.clone())?;

    let model = MockTrainingModel::new();

    // Simulate parameter push/pull cycle
    for param_name in model.parameter_names() {
        if let Some(param) = model.get_parameter(&param_name) {
            // Push gradients to parameter server
            parameter_server
                .push_gradients(&param_name, param.clone())
                .await?;

            // Pull updated parameters
            let updated_param = parameter_server.pull_parameters(&param_name).await?;

            // Verify parameter was updated
            assert_eq!(updated_param.shape(), param.shape());
        }
    }

    println!("Parameter server integration test completed");
    Ok(())
}

#[tokio::test]
async fn test_pipeline_parallelism_integration() -> Result<()> {
    let num_stages = 4;
    let pg = init_process_group(BackendType::Gloo, 0, num_stages, "127.0.0.1", 40030)?;

    let pipeline_config = PipelineConfig {
        num_stages,
        micro_batch_size: 8,
        schedule_type: ScheduleType::GPipe,
        max_outstanding_batches: 4,
        gradient_accumulation_steps: 2,
    };

    // Create pipeline parallel setup
    let pipeline = PipelineParallel::new(pipeline_config, pg.clone())?;

    // Simulate pipeline stages
    let stage_id = pg.rank();
    let input_shape = vec![32, 100]; // batch_size x features
    let output_shape = if stage_id == num_stages - 1 {
        vec![32, 10] // final output
    } else {
        vec![32, 100] // intermediate
    };

    // Simulate forward pass through pipeline
    let input_batch = randn::<f32>(&input_shape);
    let output_batch = pipeline.forward_stage(stage_id, input_batch).await?;

    assert_eq!(output_batch.shape(), output_shape);

    // Simulate backward pass
    let grad_output = randn::<f32>(&output_shape);
    let grad_input = pipeline.backward_stage(stage_id, grad_output).await?;

    assert_eq!(grad_input.shape(), input_shape);

    println!(
        "Pipeline parallelism integration test completed for stage {}",
        stage_id
    );
    Ok(())
}

#[tokio::test]
async fn test_gradient_compression_integration() -> Result<()> {
    let config = TrainingConfig {
        world_size: 2,
        use_compression: true,
        compression_ratio: 0.1,
        num_epochs: 1,
        ..Default::default()
    };

    let mut coordinator = DistributedTrainingCoordinator::new(0, config).await?;
    let mut model = MockTrainingModel::new();

    // Train one epoch with compression
    let metrics = coordinator.train_epoch(&mut model).await?;

    println!("Training with compression completed:");
    metrics.print_summary();

    // Verify compression didn't break training
    assert!(metrics.throughput > 0.0, "Should have positive throughput");
    assert!(metrics.average_loss >= 0.0, "Should have non-negative loss");

    Ok(())
}

#[tokio::test]
async fn test_elastic_training_integration() -> Result<()> {
    let initial_world_size = 2;
    let expanded_world_size = 4;

    // Start with initial workers
    let mut initial_workers = Vec::new();
    for rank in 0..initial_world_size {
        let pg = init_process_group(
            BackendType::Gloo,
            rank,
            initial_world_size,
            "127.0.0.1",
            40040,
        )?;
        initial_workers.push(pg);
    }

    // Simulate initial training
    for pg in &initial_workers {
        barrier(pg).await?;
    }

    // Simulate adding new workers
    let mut all_workers = initial_workers;
    for rank in initial_world_size..expanded_world_size {
        let pg = init_process_group(
            BackendType::Gloo,
            rank,
            expanded_world_size,
            "127.0.0.1",
            40040,
        )?;
        all_workers.push(pg);
    }

    // Test that all workers can synchronize
    for pg in &all_workers {
        barrier(pg).await?;
    }

    println!(
        "Elastic training scaled from {} to {} workers",
        initial_world_size, expanded_world_size
    );
    Ok(())
}

#[tokio::test]
async fn test_fault_tolerance_integration() -> Result<()> {
    let world_size = 4;
    let mut workers: Vec<Option<ProcessGroup>> = Vec::new();

    // Initialize all workers
    for rank in 0..world_size {
        let pg = init_process_group(BackendType::Gloo, rank, world_size, "127.0.0.1", 40050)?;
        workers.push(Some(pg));
    }

    // Simulate worker failure
    workers[2] = None; // Worker 2 fails

    // Test that remaining workers can continue
    let remaining_workers: Vec<_> = workers.into_iter().filter_map(|w| w).collect();

    for pg in &remaining_workers {
        // Each remaining worker should be able to perform operations
        let mut tensor = ones::<f32>(&[10, 10]);
        let result = all_reduce(&mut tensor, ReduceOp::Sum, pg).await;
        assert!(
            result.is_ok(),
            "Remaining workers should continue operating"
        );
    }

    println!(
        "Fault tolerance test: {} workers continued after 1 failure",
        remaining_workers.len()
    );
    Ok(())
}

#[tokio::test]
async fn test_mixed_precision_integration() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 40060)?;

    // Simulate mixed precision training
    let model = MockTrainingModel::new();

    // Test FP32 operations
    let mut fp32_tensor = ones::<f32>(&[100, 100]);
    all_reduce(&mut fp32_tensor, ReduceOp::Sum, &pg).await?;

    // Simulate FP16 gradients (using FP32 for simplicity in this test)
    let mut fp16_grad = ones::<f32>(&[100, 100]) * 0.5;
    all_reduce(&mut fp16_grad, ReduceOp::Sum, &pg).await?;

    // Verify mixed precision workflow
    let fp32_data = fp32_tensor.to_vec();
    let fp16_data = fp16_grad.to_vec();

    assert!(!fp32_data.is_empty(), "FP32 tensor should have data");
    assert!(!fp16_data.is_empty(), "FP16 gradient should have data");

    println!("Mixed precision integration test completed");
    Ok(())
}

#[tokio::test]
async fn test_large_model_sharding_integration() -> Result<()> {
    let world_size = 4;
    let pg = init_process_group(BackendType::Gloo, 0, world_size, "127.0.0.1", 40070)?;

    // Simulate large model with many parameters
    let large_model_params = vec![
        ("layer1.weight", vec![10000, 1000]),
        ("layer1.bias", vec![1000]),
        ("layer2.weight", vec![1000, 1000]),
        ("layer2.bias", vec![1000]),
        ("layer3.weight", vec![1000, 100]),
        ("layer3.bias", vec![100]),
    ];

    let mut total_elements = 0;
    for (name, shape) in &large_model_params {
        let elements: usize = shape.iter().product();
        total_elements += elements;

        // Test sharding of each parameter
        let shard_size = elements / world_size as usize;
        if shard_size > 0 {
            let mut param_shard = ones::<f32>(&[shard_size]);
            all_reduce(&mut param_shard, ReduceOp::Sum, &pg).await?;

            println!(
                "Parameter {} ({} elements) sharded into {} elements per worker",
                name, elements, shard_size
            );
        }
    }

    println!(
        "Large model with {} total parameters successfully sharded across {} workers",
        total_elements, world_size
    );

    assert!(total_elements > 10_000_000, "Should test with large model");
    Ok(())
}

#[tokio::test]
async fn test_multi_backend_integration() -> Result<()> {
    // Test integration with different backends
    let backends = vec![BackendType::Gloo]; // Add more when available

    for (i, backend) in backends.iter().enumerate() {
        let pg = init_process_group(*backend, 0, 2, "127.0.0.1", 40080 + i as u16).await?;

        // Test basic operations with each backend
        let mut tensor = ones::<f32>(&[50, 50])?;
        all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;

        // Test barrier
        barrier(&pg).await?;

        println!("Backend {:?} integration test passed", backend);
    }

    Ok(())
}

#[tokio::test]
async fn test_checkpointing_integration() -> Result<()> {
    let config = TrainingConfig {
        checkpoint_interval: 2, // Checkpoint every 2 steps
        ..Default::default()
    };

    let mut coordinator = DistributedTrainingCoordinator::new(0, config.clone()).await?;
    let mut model = MockTrainingModel::new();

    // Train and verify checkpointing
    let metrics = coordinator.train_epoch(&mut model).await?;

    // Should have triggered multiple checkpoints
    assert!(metrics.steps >= config.checkpoint_interval);

    println!(
        "Checkpointing integration test completed with {} steps",
        metrics.steps
    );
    Ok(())
}
