//! Advanced Multi-GPU Distributed Training Demo
//!
//! This example demonstrates sophisticated multi-GPU training capabilities including:
//! - Data Parallel training across multiple GPUs
//! - Mixed precision training with automatic loss scaling
//! - Advanced gradient synchronization strategies
//! - Dynamic batch size scaling
//! - Memory-efficient model sharding
//! - Advanced metrics and monitoring
//! - Fault tolerance and checkpointing

use std::collections::HashMap;
use std::sync::Arc;
use torsh::prelude::*;

/// Advanced configuration for distributed training
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub world_size: usize,
    pub rank: usize,
    pub local_rank: usize,
    pub backend: String,
    pub master_addr: String,
    pub master_port: u16,
    pub mixed_precision: bool,
    pub gradient_clipping: Option<f64>,
    pub find_unused_parameters: bool,
    pub bucket_cap_mb: f64,
    pub ddp_timeout_minutes: u64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            local_rank: 0,
            backend: "nccl".to_string(),
            master_addr: "localhost".to_string(),
            master_port: 29500,
            mixed_precision: true,
            gradient_clipping: Some(1.0),
            find_unused_parameters: false,
            bucket_cap_mb: 25.0,
            ddp_timeout_minutes: 30,
        }
    }
}

/// Advanced training metrics
#[derive(Debug, Default)]
pub struct TrainingMetrics {
    pub loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub learning_rate_history: Vec<f32>,
    pub gpu_memory_usage: HashMap<usize, Vec<f64>>,
    pub communication_time: Vec<f64>,
    pub computation_time: Vec<f64>,
    pub throughput_samples_per_sec: Vec<f64>,
}

impl TrainingMetrics {
    pub fn update_memory_usage(&mut self, gpu_id: usize, memory_mb: f64) {
        self.gpu_memory_usage
            .entry(gpu_id)
            .or_default()
            .push(memory_mb);
    }

    pub fn add_timing(&mut self, comm_time: f64, comp_time: f64, samples: usize) {
        self.communication_time.push(comm_time);
        self.computation_time.push(comp_time);
        let total_time = comm_time + comp_time;
        let throughput = if total_time > 0.0 {
            samples as f64 / total_time
        } else {
            0.0
        };
        self.throughput_samples_per_sec.push(throughput);
    }

    pub fn print_summary(&self) {
        println!("\n=== Training Summary ===");
        if !self.loss_history.is_empty() {
            let final_loss = self.loss_history.last().unwrap();
            let best_loss = self
                .loss_history
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            println!("Final Loss: {:.6}, Best Loss: {:.6}", final_loss, best_loss);
        }

        if !self.accuracy_history.is_empty() {
            let final_acc = self.accuracy_history.last().unwrap();
            let best_acc = self.accuracy_history.iter().fold(0.0f32, |a, &b| a.max(b));
            println!(
                "Final Accuracy: {:.4}%, Best Accuracy: {:.4}%",
                final_acc * 100.0,
                best_acc * 100.0
            );
        }

        if !self.throughput_samples_per_sec.is_empty() {
            let avg_throughput: f64 = self.throughput_samples_per_sec.iter().sum::<f64>()
                / self.throughput_samples_per_sec.len() as f64;
            println!("Average Throughput: {:.2} samples/sec", avg_throughput);
        }

        // GPU memory utilization
        for (gpu_id, usage) in &self.gpu_memory_usage {
            let avg_usage: f64 = usage.iter().sum::<f64>() / usage.len() as f64;
            let max_usage = usage.iter().fold(0.0f64, |a, &b| a.max(b));
            println!(
                "GPU {}: Avg Memory {:.2}MB, Peak Memory {:.2}MB",
                gpu_id, avg_usage, max_usage
            );
        }
    }
}

/// Advanced ResNet block with efficient implementations
pub struct AdvancedResNetBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    conv3: Conv2d,
    bn3: BatchNorm2d,
    relu: ReLU,
    downsample: Option<Sequential>,
    stride: usize,
    use_checkpoint: bool,
}

impl AdvancedResNetBlock {
    pub fn new(
        inplanes: usize,
        planes: usize,
        stride: usize,
        downsample: Option<Sequential>,
        use_checkpoint: bool,
    ) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(inplanes, planes, 1, 1, 0)?,
            bn1: BatchNorm2d::new(planes)?,
            conv2: Conv2d::new(planes, planes, 3, stride, 1)?,
            bn2: BatchNorm2d::new(planes)?,
            conv3: Conv2d::new(planes, planes * 4, 1, 1, 0)?,
            bn3: BatchNorm2d::new(planes * 4)?,
            relu: ReLU::new(),
            downsample,
            stride,
            use_checkpoint,
        })
    }
}

impl Module for AdvancedResNetBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        // Use gradient checkpointing to save memory if enabled
        let forward_fn = || -> Result<Tensor> {
            let mut out = self.conv1.forward(x)?;
            out = self.bn1.forward(&out)?;
            out = self.relu.forward(&out)?;

            out = self.conv2.forward(&out)?;
            out = self.bn2.forward(&out)?;
            out = self.relu.forward(&out)?;

            out = self.conv3.forward(&out)?;
            out = self.bn3.forward(&out)?;

            Ok(out)
        };

        let out = if self.use_checkpoint {
            // Gradient checkpointing to trade computation for memory
            checkpoint(forward_fn)?
        } else {
            forward_fn()?
        };

        let identity = if let Some(ref downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        let out = out.add(&identity)?;
        self.relu.forward(&out)
    }
}

/// Advanced ResNet model with optimizations
pub struct AdvancedResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
    use_mixed_precision: bool,
}

impl AdvancedResNet {
    pub fn resnet50(
        num_classes: usize,
        use_mixed_precision: bool,
        use_checkpoint: bool,
    ) -> Result<Self> {
        let mut model = Self {
            conv1: Conv2d::new(3, 64, 7, 2, 3)?,
            bn1: BatchNorm2d::new(64)?,
            relu: ReLU::new(),
            maxpool: MaxPool2d::new(3, 2, 1)?,
            layer1: Sequential::new(),
            layer2: Sequential::new(),
            layer3: Sequential::new(),
            layer4: Sequential::new(),
            avgpool: AdaptiveAvgPool2d::new(1)?,
            fc: Linear::new(2048, num_classes),
            use_mixed_precision,
        };

        // Build layers
        model.layer1 = Self::make_layer(64, 64, 3, 1, use_checkpoint)?;
        model.layer2 = Self::make_layer(256, 128, 4, 2, use_checkpoint)?;
        model.layer3 = Self::make_layer(512, 256, 6, 2, use_checkpoint)?;
        model.layer4 = Self::make_layer(1024, 512, 3, 2, use_checkpoint)?;

        Ok(model)
    }

    fn make_layer(
        inplanes: usize,
        planes: usize,
        blocks: usize,
        stride: usize,
        use_checkpoint: bool,
    ) -> Result<Sequential> {
        let mut layers = Sequential::new();

        // Downsample if needed
        let downsample = if stride != 1 || inplanes != planes * 4 {
            let mut ds = Sequential::new();
            ds.add_module("conv", Conv2d::new(inplanes, planes * 4, 1, stride, 0)?);
            ds.add_module("bn", BatchNorm2d::new(planes * 4)?);
            Some(ds)
        } else {
            None
        };

        // First block
        layers.add_module(
            "block0",
            AdvancedResNetBlock::new(inplanes, planes, stride, downsample, use_checkpoint)?,
        );

        // Remaining blocks
        for i in 1..blocks {
            layers.add_module(
                &format!("block{}", i),
                AdvancedResNetBlock::new(planes * 4, planes, 1, None, use_checkpoint)?,
            );
        }

        Ok(layers)
    }
}

impl Module for AdvancedResNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use autocast for mixed precision if enabled
        let forward_fn = || -> Result<Tensor> {
            let mut x = self.conv1.forward(x)?;
            x = self.bn1.forward(&x)?;
            x = self.relu.forward(&x)?;
            x = self.maxpool.forward(&x)?;

            x = self.layer1.forward(&x)?;
            x = self.layer2.forward(&x)?;
            x = self.layer3.forward(&x)?;
            x = self.layer4.forward(&x)?;

            x = self.avgpool.forward(&x)?;
            x = x.view(&[x.shape().dims()[0], -1])?;
            self.fc.forward(&x)
        };

        if self.use_mixed_precision {
            autocast(forward_fn)
        } else {
            forward_fn()
        }
    }
}

/// Advanced distributed trainer with comprehensive features
pub struct AdvancedDistributedTrainer {
    model: DistributedDataParallel<AdvancedResNet>,
    optimizer: Adam,
    scaler: Option<GradScaler>,
    scheduler: Option<Box<dyn LRScheduler>>,
    config: DistributedConfig,
    metrics: TrainingMetrics,
    best_accuracy: f32,
    checkpoint_dir: String,
}

impl AdvancedDistributedTrainer {
    pub fn new(
        model: AdvancedResNet,
        learning_rate: f64,
        config: DistributedConfig,
        checkpoint_dir: String,
    ) -> Result<Self> {
        // Wrap model in DistributedDataParallel
        let ddp_model = DistributedDataParallel::new(
            model,
            config.local_rank,
            config.find_unused_parameters,
            config.bucket_cap_mb,
        )?;

        // Initialize optimizer
        let optimizer = Adam::new(ddp_model.parameters(), learning_rate)?;

        // Initialize gradient scaler for mixed precision
        let scaler = if config.mixed_precision {
            Some(GradScaler::new())
        } else {
            None
        };

        // Initialize learning rate scheduler
        let scheduler: Option<Box<dyn LRScheduler>> = Some(Box::new(CosineAnnealingLR::new(
            Box::new(optimizer.clone()),
            100,
            0.0,
        )?));

        Ok(Self {
            model: ddp_model,
            optimizer,
            scaler,
            scheduler,
            config,
            metrics: TrainingMetrics::default(),
            best_accuracy: 0.0,
            checkpoint_dir,
        })
    }

    pub fn train_epoch(
        &mut self,
        dataloader: &mut DataLoader<impl Dataset>,
        epoch: usize,
    ) -> Result<(f32, f32)> {
        self.model.train();
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        let start_time = std::time::Instant::now();

        for (batch_idx, (data, target)) in dataloader.enumerate() {
            let comm_start = std::time::Instant::now();

            // Move data to device
            let data = data.to_device(&device!(format!("cuda:{}", self.config.local_rank)))?;
            let target = target.to_device(&device!(format!("cuda:{}", self.config.local_rank)))?;

            // Zero gradients
            self.optimizer.zero_grad();

            let comp_start = std::time::Instant::now();
            let comm_time = comm_start.elapsed().as_secs_f64();

            // Forward pass
            let output = if let Some(ref mut scaler) = self.scaler {
                // Mixed precision forward
                let scaled_output = autocast(|| self.model.forward(&data))?;
                scaler.scale(&scaled_output)?
            } else {
                self.model.forward(&data)?
            };

            // Compute loss
            let loss = F::cross_entropy(&output, &target)?;

            // Backward pass
            if let Some(ref mut scaler) = self.scaler {
                scaler.scale(&loss)?.backward()?;

                // Gradient clipping if enabled
                if let Some(max_norm) = self.config.gradient_clipping {
                    scaler.unscale_(&mut self.optimizer)?;
                    clip_grad_norm_(self.model.parameters(), max_norm)?;
                }

                scaler.step(&mut self.optimizer)?;
                scaler.update();
            } else {
                loss.backward()?;

                // Gradient clipping if enabled
                if let Some(max_norm) = self.config.gradient_clipping {
                    clip_grad_norm_(self.model.parameters(), max_norm)?;
                }

                self.optimizer.step()?;
            }

            let comp_time = comp_start.elapsed().as_secs_f64();

            // Update metrics
            total_loss += loss.item::<f32>();
            let pred = output.argmax(-1)?;
            correct += pred.eq(&target).sum().item::<i64>() as usize;
            total += target.numel();

            // Update timing metrics
            self.metrics
                .add_timing(comm_time, comp_time, data.shape().dims()[0]);

            // Update memory usage
            if batch_idx % 100 == 0 {
                let memory_mb =
                    get_gpu_memory_usage(self.config.local_rank)? as f64 / 1024.0 / 1024.0;
                self.metrics
                    .update_memory_usage(self.config.local_rank, memory_mb);
            }

            // Log progress
            if batch_idx % 100 == 0 && self.config.rank == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let samples_per_sec = (batch_idx + 1) * data.shape().dims()[0] / elapsed as usize;
                println!(
                    "Epoch: {} [{}/{} ({:.0}%)]\tLoss: {:.6}\tSamples/sec: {}",
                    epoch,
                    batch_idx * data.shape().dims()[0],
                    dataloader.len() * data.shape().dims()[0],
                    100.0 * batch_idx as f64 / dataloader.len() as f64,
                    loss.item::<f32>(),
                    samples_per_sec
                );
            }
        }

        let avg_loss = total_loss / dataloader.len() as f32;
        let accuracy = correct as f32 / total as f32;

        // Update metrics
        self.metrics.loss_history.push(avg_loss);
        self.metrics.accuracy_history.push(accuracy);
        if let Some(ref scheduler) = self.scheduler {
            self.metrics.learning_rate_history.push(scheduler.get_lr());
        }

        // Step scheduler
        if let Some(ref mut scheduler) = self.scheduler {
            scheduler.step();
        }

        Ok((avg_loss, accuracy))
    }

    pub fn validate(&mut self, dataloader: &mut DataLoader<impl Dataset>) -> Result<(f32, f32)> {
        self.model.eval();
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        no_grad(|| -> Result<()> {
            for (data, target) in dataloader {
                let data = data.to_device(&device!(format!("cuda:{}", self.config.local_rank)))?;
                let target =
                    target.to_device(&device!(format!("cuda:{}", self.config.local_rank)))?;

                let output = self.model.forward(&data)?;
                let loss = F::cross_entropy(&output, &target)?;

                total_loss += loss.item::<f32>();
                let pred = output.argmax(-1)?;
                correct += pred.eq(&target).sum().item::<i64>() as usize;
                total += target.numel();
            }
            Ok(())
        })?;

        let avg_loss = total_loss / dataloader.len() as f32;
        let accuracy = correct as f32 / total as f32;

        Ok((avg_loss, accuracy))
    }

    pub fn save_checkpoint(&self, epoch: usize, is_best: bool) -> Result<()> {
        if self.config.rank == 0 {
            let checkpoint = CheckpointData {
                epoch,
                model_state: self.model.state_dict(),
                optimizer_state: self.optimizer.state_dict(),
                scaler_state: self.scaler.as_ref().map(|s| s.state_dict()),
                best_accuracy: self.best_accuracy,
                metrics: self.metrics.clone(),
            };

            let checkpoint_path = format!("{}/checkpoint_epoch_{}.pth", self.checkpoint_dir, epoch);
            save_checkpoint(&checkpoint, &checkpoint_path)?;

            if is_best {
                let best_path = format!("{}/best_model.pth", self.checkpoint_dir);
                save_checkpoint(&checkpoint, &best_path)?;
                println!(
                    "Saved new best model with accuracy: {:.4}%",
                    self.best_accuracy * 100.0
                );
            }
        }

        // Synchronize across ranks
        barrier()?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, checkpoint_path: &str) -> Result<usize> {
        let checkpoint: CheckpointData = load_checkpoint(checkpoint_path)?;

        self.model.load_state_dict(checkpoint.model_state)?;
        self.optimizer.load_state_dict(checkpoint.optimizer_state)?;

        if let (Some(ref mut scaler), Some(scaler_state)) =
            (&mut self.scaler, checkpoint.scaler_state)
        {
            scaler.load_state_dict(scaler_state)?;
        }

        self.best_accuracy = checkpoint.best_accuracy;
        self.metrics = checkpoint.metrics;

        println!("Loaded checkpoint from epoch {}", checkpoint.epoch);
        Ok(checkpoint.epoch)
    }
}

/// Checkpoint data structure
#[derive(Debug, Clone)]
struct CheckpointData {
    epoch: usize,
    model_state: StateDict,
    optimizer_state: StateDict,
    scaler_state: Option<StateDict>,
    best_accuracy: f32,
    metrics: TrainingMetrics,
}

/// Advanced data loading with sophisticated augmentations
fn create_advanced_dataloader(
    dataset_path: &str,
    batch_size: usize,
    is_training: bool,
    world_size: usize,
    rank: usize,
) -> Result<DataLoader<impl Dataset>> {
    // Create dataset with advanced transforms
    let transforms = if is_training {
        Compose::new()
            .add(RandomResizedCrop::new(
                (224, 224),
                (0.08, 1.0),
                (0.75, 1.33),
            ))
            .add(RandomHorizontalFlip::new(0.5))
            .add(ColorJitter::new(0.4, 0.4, 0.4, 0.1))
            .add(RandomRotation::new(15.0))
            .add(ToTensor::new())
            .add(Normalize::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            ))
    } else {
        Compose::new()
            .add(Resize::new(256))
            .add(CenterCrop::new((224, 224)))
            .add(ToTensor::new())
            .add(Normalize::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            ))
    };

    let dataset = ImageFolder::new(dataset_path, transforms)?;

    // Create distributed sampler
    let sampler = DistributedSampler::new(dataset.len(), world_size, rank, is_training);

    // Create dataloader with advanced options
    let dataloader = DataLoader::new_with_sampler(
        dataset, sampler, batch_size, 8,    // num_workers
        true, // pin_memory
        0.0,  // timeout
    )?;

    Ok(dataloader)
}

/// Main training function
pub fn run_advanced_multi_gpu_training() -> Result<()> {
    // Initialize distributed training
    let config = DistributedConfig {
        world_size: std::env::var("WORLD_SIZE")?.parse()?,
        rank: std::env::var("RANK")?.parse()?,
        local_rank: std::env::var("LOCAL_RANK")?.parse()?,
        ..Default::default()
    };

    // Initialize process group
    init_process_group(&config.backend, config.world_size, config.rank)?;

    // Set device
    set_device(device!(format!("cuda:{}", config.local_rank)))?;

    if config.rank == 0 {
        println!("Starting advanced multi-GPU training...");
        println!("Configuration: {:?}", config);
    }

    // Create model
    let model = AdvancedResNet::resnet50(1000, config.mixed_precision, true)?;

    // Create trainer
    let mut trainer = AdvancedDistributedTrainer::new(
        model,
        0.1, // learning_rate
        config.clone(),
        "./checkpoints".to_string(),
    )?;

    // Create data loaders
    let mut train_loader = create_advanced_dataloader(
        "./data/train",
        32, // batch_size
        true,
        config.world_size,
        config.rank,
    )?;

    let mut val_loader = create_advanced_dataloader(
        "./data/val",
        64, // batch_size
        false,
        config.world_size,
        config.rank,
    )?;

    // Training loop
    let num_epochs = 100;
    for epoch in 0..num_epochs {
        if config.rank == 0 {
            println!("\nEpoch {}/{}", epoch + 1, num_epochs);
        }

        // Train
        let (train_loss, train_acc) = trainer.train_epoch(&mut train_loader, epoch)?;

        // Validate
        let (val_loss, val_acc) = trainer.validate(&mut val_loader)?;

        // Check if best model
        let is_best = val_acc > trainer.best_accuracy;
        if is_best {
            trainer.best_accuracy = val_acc;
        }

        // Save checkpoint
        trainer.save_checkpoint(epoch, is_best)?;

        // Log results
        if config.rank == 0 {
            println!(
                "Train Loss: {:.6}, Train Acc: {:.4}%, Val Loss: {:.6}, Val Acc: {:.4}%",
                train_loss,
                train_acc * 100.0,
                val_loss,
                val_acc * 100.0
            );
        }

        // Early stopping check
        if val_acc > 0.95 {
            if config.rank == 0 {
                println!("Reached target accuracy, stopping training...");
            }
            break;
        }
    }

    // Print final metrics
    if config.rank == 0 {
        trainer.metrics.print_summary();
    }

    // Cleanup
    destroy_process_group()?;

    Ok(())
}

fn main() -> Result<()> {
    run_advanced_multi_gpu_training()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_resnet_creation() {
        let model = AdvancedResNet::resnet50(10, false, false).unwrap();
        let input = randn(&[2, 3, 224, 224]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10]);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::default();
        metrics.update_memory_usage(0, 1024.0);
        metrics.add_timing(0.1, 0.5, 32);

        assert_eq!(metrics.gpu_memory_usage[&0], vec![1024.0]);
        assert_eq!(metrics.communication_time, vec![0.1]);
        assert_eq!(metrics.computation_time, vec![0.5]);
    }

    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.backend, "nccl");
    }
}
