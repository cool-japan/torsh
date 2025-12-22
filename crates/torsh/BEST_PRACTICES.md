# ToRSh Best Practices Guide

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Code Organization](#code-organization)
4. [Memory Management](#memory-management)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling](#error-handling)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Device Management](#device-management)
10. [Model Development](#model-development)
11. [Training Best Practices](#training-best-practices)
12. [Deployment](#deployment)
13. [Debugging](#debugging)
14. [Security](#security)

## Overview

This guide provides best practices for developing with ToRSh, covering everything from basic usage patterns to advanced optimization techniques. Following these practices will help you write more efficient, maintainable, and robust deep learning applications.

## Project Structure

### Recommended Directory Layout

```
my_torsh_project/
├── Cargo.toml              # Project dependencies
├── src/
│   ├── main.rs             # Application entry point
│   ├── lib.rs              # Library root
│   ├── models/             # Model definitions
│   │   ├── mod.rs
│   │   ├── resnet.rs
│   │   └── transformer.rs
│   ├── data/               # Data loading and preprocessing
│   │   ├── mod.rs
│   │   ├── dataset.rs
│   │   └── transforms.rs
│   ├── training/           # Training utilities
│   │   ├── mod.rs
│   │   ├── trainer.rs
│   │   └── metrics.rs
│   └── utils/              # Utility functions
│       ├── mod.rs
│       └── checkpoint.rs
├── tests/                  # Integration tests
├── benches/                # Benchmarks
├── examples/               # Usage examples
├── data/                   # Dataset storage
├── models/                 # Trained model storage
└── configs/                # Configuration files
```

### Cargo.toml Configuration

```toml
[package]
name = "my_torsh_project"
version = "0.1.0"
edition = "2021"

[dependencies]
torsh = { version = "0.1.0-alpha.2", features = ["full"] }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
clap = { version = "4.0", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "model_benchmark"
harness = false

[features]
default = ["cuda"]
cuda = ["torsh/cuda"]
distributed = ["torsh/distributed"]
```

## Code Organization

### Module Structure

```rust
// lib.rs
pub mod models;
pub mod data;
pub mod training;
pub mod utils;

pub use torsh::prelude::*;

// Re-export commonly used types
pub use crate::models::*;
pub use crate::training::{Trainer, TrainerConfig};
```

### Model Definition Best Practices

```rust
use torsh::prelude::*;
use torsh::nn::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetConfig {
    pub num_classes: usize,
    pub layers: Vec<usize>,
    pub base_channels: usize,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            layers: vec![2, 2, 2, 2],
            base_channels: 64,
        }
    }
}

pub struct ResNet {
    config: ResNetConfig,
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    layers: Sequential,
    classifier: Linear,
}

impl ResNet {
    pub fn new(config: ResNetConfig) -> Result<Self> {
        let ResNetConfig { num_classes, layers, base_channels } = config.clone();
        
        Ok(Self {
            config,
            conv1: Conv2d::new(3, base_channels, 7, 2, 3)?,
            bn1: BatchNorm2d::new(base_channels)?,
            relu: ReLU::new(),
            layers: Self::make_layers(&layers, base_channels)?,
            classifier: Linear::new(base_channels * 8, num_classes)?,
        })
    }
    
    fn make_layers(layers: &[usize], base_channels: usize) -> Result<Sequential> {
        // Implementation details...
        todo!()
    }
    
    pub fn config(&self) -> &ResNetConfig {
        &self.config
    }
}

impl Module for ResNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(&x)?;
        let x = self.relu.forward(&x)?;
        let x = self.layers.forward(&x)?;
        let x = x.adaptive_avg_pool2d(&[1, 1])?;
        let x = x.flatten(1)?;
        self.classifier.forward(&x)
    }
}
```

## Memory Management

### Efficient Tensor Operations

```rust
// ✅ Good: Use references to avoid unnecessary clones
fn efficient_computation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let result = a + b;
    result?.relu()
}

// ❌ Bad: Unnecessary clones
fn inefficient_computation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_clone = a.clone();
    let b_clone = b.clone();
    let result = a_clone + b_clone;
    result?.relu()
}

// ✅ Good: Use in-place operations when possible
fn inplace_operations(mut tensor: Tensor) -> Result<Tensor> {
    tensor.add_(&other_tensor)?;
    tensor.relu_()?;
    Ok(tensor)
}

// ✅ Good: Proper scoping for temporary tensors
fn scoped_temporaries(input: &Tensor) -> Result<Tensor> {
    let intermediate = {
        let temp1 = input.conv2d(&weight1)?;
        let temp2 = temp1.relu()?;
        temp2.max_pool2d(2)?
    }; // temp1 and temp2 are dropped here
    
    intermediate.conv2d(&weight2)
}
```

### Memory Pool Usage

```rust
use torsh::memory::MemoryPool;

// Create a memory pool for efficient allocation
let pool = MemoryPool::new(1024 * 1024 * 1024); // 1GB pool

// Use pooled tensors for temporary allocations
fn efficient_batch_processing(inputs: &[Tensor], pool: &MemoryPool) -> Result<Vec<Tensor>> {
    let mut results = Vec::with_capacity(inputs.len());
    
    for input in inputs {
        let pooled_tensor = pool.allocate_tensor(input.shape(), input.dtype())?;
        // Use pooled_tensor for computations
        // It will be automatically returned to pool when dropped
        let result = process_tensor(input, &pooled_tensor)?;
        results.push(result);
    }
    
    Ok(results)
}
```

### RAII Patterns

```rust
// Use RAII for resource management
pub struct ModelContext {
    device: Device,
    memory_manager: MemoryManager,
}

impl ModelContext {
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            memory_manager: MemoryManager::new(&device)?,
        })
    }
    
    pub fn create_tensor(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        self.memory_manager.allocate_tensor(shape, dtype, &self.device)
    }
}

impl Drop for ModelContext {
    fn drop(&mut self) {
        // Automatic cleanup
        self.memory_manager.cleanup();
    }
}
```

## Performance Optimization

### SIMD Optimization

```rust
// Enable SIMD optimizations
#[cfg(target_arch = "x86_64")]
use torsh::simd::avx;

// Let ToRSh automatically choose the best SIMD implementation
fn optimized_operations(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // These operations will automatically use SIMD when available
    a.add(b)?.mul(&tensor![2.0])?.sqrt()
}

// For custom operations, use SIMD-aware implementations
fn custom_activation(input: &Tensor) -> Result<Tensor> {
    input.map(|x| {
        // This will be vectorized automatically
        if x > 0.0 { x } else { 0.01 * x }
    })
}
```

### Batch Processing

```rust
// ✅ Good: Process data in batches
fn efficient_inference(model: &dyn Module, inputs: &[Tensor], batch_size: usize) -> Result<Vec<Tensor>> {
    let mut results = Vec::new();
    
    for chunk in inputs.chunks(batch_size) {
        let batch = Tensor::stack(chunk, 0)?;
        let output = model.forward(&batch)?;
        
        // Split output back into individual results
        for i in 0..chunk.len() {
            results.push(output.slice(0, i, i + 1)?.squeeze(0)?);
        }
    }
    
    Ok(results)
}

// ❌ Bad: Process one at a time
fn inefficient_inference(model: &dyn Module, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    inputs.iter().map(|input| model.forward(input)).collect()
}
```

### Memory Layout Optimization

```rust
// Use contiguous memory layouts
fn optimize_memory_layout(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.is_contiguous()? {
        tensor.contiguous()
    } else {
        Ok(tensor.clone())
    }
}

// Use appropriate data types
fn mixed_precision_forward(input: &Tensor) -> Result<Tensor> {
    // Use half precision for forward pass
    let input_fp16 = input.to_dtype(DType::F16)?;
    let output_fp16 = model.forward(&input_fp16)?;
    
    // Convert back to fp32 for loss computation
    output_fp16.to_dtype(DType::F32)
}
```

## Error Handling

### Comprehensive Error Handling

```rust
use torsh::TorshError;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model configuration error: {0}")]
    Config(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Data loading error: {0}")]
    DataLoading(String),
    
    #[error("ToRSh error: {0}")]
    Torsh(#[from] TorshError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, ModelError>;

// Proper error propagation
fn train_model(config: &TrainingConfig) -> Result<TrainingResults> {
    let model = create_model(&config.model_config)
        .map_err(|e| ModelError::Config(format!("Failed to create model: {}", e)))?;
    
    let dataloader = create_dataloader(&config.data_config)
        .map_err(|e| ModelError::DataLoading(format!("Failed to create dataloader: {}", e)))?;
    
    let trainer = Trainer::new(model, config.clone())?;
    trainer.train(dataloader)
        .map_err(|e| ModelError::Training(format!("Training failed: {}", e)))
}
```

### Validation and Assertions

```rust
// Input validation
fn validate_tensor_shape(tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
    if tensor.shape().dims() != expected_shape {
        return Err(ModelError::Config(format!(
            "Expected shape {:?}, got {:?}", 
            expected_shape, 
            tensor.shape().dims()
        )));
    }
    Ok(())
}

// Runtime assertions
fn safe_division(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check for zeros in denominator
    let has_zeros = b.eq(&tensor![0.0])?.any()?.item::<bool>()?;
    if has_zeros {
        return Err(ModelError::Config("Division by zero detected".to_string()));
    }
    
    a.div(b).map_err(Into::into)
}
```

## Testing

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use torsh::testing::*;
    
    #[test]
    fn test_model_forward_pass() -> Result<()> {
        let model = ResNet::new(ResNetConfig::default())?;
        let input = randn(&[1, 3, 224, 224]);
        
        let output = model.forward(&input)?;
        
        assert_eq!(output.shape().dims(), &[1, 1000]);
        assert!(!output.has_nan()?);
        assert!(output.is_finite()?);
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_computation() -> Result<()> {
        let x = randn(&[2, 3]).requires_grad_(true);
        let y = x.pow(tensor![2.0])?.sum()?;
        
        y.backward()?;
        
        let grad = x.grad().expect("Gradient should be computed");
        assert_tensor_close!(grad, &(&x * &tensor![2.0]), 1e-6);
        
        Ok(())
    }
    
    #[test]
    fn test_model_serialization() -> Result<()> {
        let model = ResNet::new(ResNetConfig::default())?;
        let temp_path = "/tmp/test_model.torsh";
        
        // Save model
        model.save(temp_path)?;
        
        // Load model
        let loaded_model = ResNet::load(temp_path)?;
        
        // Test equivalence
        let input = randn(&[1, 3, 224, 224]);
        let output1 = model.forward(&input)?;
        let output2 = loaded_model.forward(&input)?;
        
        assert_tensor_close!(output1, output2, 1e-6);
        
        // Cleanup
        std::fs::remove_file(temp_path)?;
        
        Ok(())
    }
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_tensor_operations_properties(
        shape in prop::collection::vec(1usize..10, 1..4),
        values in prop::collection::vec(-100.0f32..100.0, 1..1000)
    ) {
        let tensor = Tensor::from_vec(values, &shape)?;
        
        // Test mathematical properties
        let zero = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?;
        let result = &tensor + &zero;
        
        // Addition with zero should be identity
        assert_tensor_close!(tensor, result, 1e-6)?;
        
        // Test commutativity
        let other = randn(tensor.shape());
        let result1 = &tensor + &other;
        let result2 = &other + &tensor;
        assert_tensor_close!(result1, result2, 1e-6)?;
    }
}
```

### Integration Testing

```rust
// tests/integration_test.rs
use my_torsh_project::*;

#[tokio::test]
async fn test_full_training_pipeline() -> Result<()> {
    let config = TrainingConfig {
        model: ModelConfig::resnet18(),
        data: DataConfig::cifar10(),
        optimizer: OptimizerConfig::adam(0.001),
        epochs: 1, // Short test
        batch_size: 32,
    };
    
    let trainer = Trainer::new(config)?;
    let results = trainer.train().await?;
    
    assert!(results.final_loss < results.initial_loss);
    assert!(results.final_accuracy > 0.1); // Basic sanity check
    
    Ok(())
}
```

## Documentation

### API Documentation

```rust
/// A ResNet model for image classification.
/// 
/// ResNet (Residual Network) uses skip connections to enable training
/// of very deep networks by mitigating the vanishing gradient problem.
/// 
/// # Examples
/// 
/// ```
/// use my_torsh_project::models::ResNet;
/// 
/// let config = ResNetConfig::default();
/// let model = ResNet::new(config)?;
/// 
/// let input = randn(&[1, 3, 224, 224]);
/// let output = model.forward(&input)?;
/// assert_eq!(output.shape().dims(), &[1, 1000]);
/// ```
/// 
/// # References
/// 
/// He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning 
/// for image recognition. In CVPR.
pub struct ResNet {
    // ... fields
}

impl ResNet {
    /// Creates a new ResNet model with the given configuration.
    /// 
    /// # Arguments
    /// 
    /// * `config` - The model configuration specifying architecture details
    /// 
    /// # Returns
    /// 
    /// Returns a Result containing the initialized model or an error if
    /// the configuration is invalid.
    /// 
    /// # Errors
    /// 
    /// This function will return an error if:
    /// - The number of classes is zero
    /// - The layer configuration is empty
    /// - Memory allocation fails
    pub fn new(config: ResNetConfig) -> Result<Self> {
        // Implementation...
    }
}
```

### Error Documentation

```rust
/// Errors that can occur during model operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// Invalid model configuration.
    /// 
    /// This error occurs when the model configuration contains
    /// invalid parameters, such as zero classes or incompatible
    /// layer dimensions.
    #[error("Model configuration error: {message}")]
    Config { message: String },
    
    /// Training process error.
    /// 
    /// This error occurs during the training process, such as
    /// when gradients explode or when the loss becomes NaN.
    #[error("Training error at epoch {epoch}: {message}")]
    Training { epoch: usize, message: String },
}
```

## Device Management

### Automatic Device Selection

```rust
use torsh::Device;

pub struct DeviceManager {
    primary_device: Device,
    fallback_device: Device,
}

impl DeviceManager {
    pub fn new() -> Self {
        let primary_device = if Device::cuda_is_available() {
            Device::cuda(0)
        } else if Device::metal_is_available() {
            Device::metal()
        } else {
            Device::cpu()
        };
        
        Self {
            primary_device,
            fallback_device: Device::cpu(),
        }
    }
    
    pub fn get_device(&self) -> &Device {
        &self.primary_device
    }
    
    pub fn move_to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.to_device(&self.primary_device)
            .or_else(|_| tensor.to_device(&self.fallback_device))
    }
}

// Use in models
impl ResNet {
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.layers.to_device(device)?;
        self.classifier.to_device(device)?;
        Ok(())
    }
}
```

### Multi-GPU Strategies

```rust
use torsh::distributed::DistributedDataParallel;

// Data parallelism
pub fn setup_data_parallel(model: ResNet, device_ids: Vec<usize>) -> Result<DistributedDataParallel<ResNet>> {
    DistributedDataParallel::new(model, &device_ids)
}

// Model parallelism (for large models)
pub struct ModelParallelResNet {
    feature_extractor: ResNet, // On GPU 0
    classifier: Linear,        // On GPU 1
}

impl Module for ModelParallelResNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let features = self.feature_extractor.forward(x)?;
        let features_gpu1 = features.to_device(&Device::cuda(1))?;
        self.classifier.forward(&features_gpu1)
    }
}
```

## Model Development

### Configuration Management

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_classes: usize,
    pub pretrained: bool,
    pub dropout: f64,
    pub batch_norm: bool,
    pub activation: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: "resnet18".to_string(),
            num_classes: 1000,
            pretrained: false,
            dropout: 0.1,
            batch_norm: true,
            activation: "relu".to_string(),
        }
    }
}

// Builder pattern for complex configurations
pub struct ModelBuilder {
    config: ModelConfig,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }
    
    pub fn architecture(mut self, arch: &str) -> Self {
        self.config.architecture = arch.to_string();
        self
    }
    
    pub fn num_classes(mut self, classes: usize) -> Self {
        self.config.num_classes = classes;
        self
    }
    
    pub fn build(self) -> Result<ResNet> {
        ResNet::new(self.config)
    }
}
```

### Model Registry

```rust
use std::collections::HashMap;
use std::sync::LazyLock;

type ModelConstructor = fn(ModelConfig) -> Result<Box<dyn Module>>;

static MODEL_REGISTRY: LazyLock<HashMap<String, ModelConstructor>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    registry.insert("resnet18".to_string(), |config| Ok(Box::new(ResNet::resnet18(config)?)));
    registry.insert("resnet50".to_string(), |config| Ok(Box::new(ResNet::resnet50(config)?)));
    registry.insert("vit_base".to_string(), |config| Ok(Box::new(VisionTransformer::base(config)?)));
    registry
});

pub fn create_model(name: &str, config: ModelConfig) -> Result<Box<dyn Module>> {
    MODEL_REGISTRY.get(name)
        .ok_or_else(|| ModelError::Config(format!("Unknown model: {}", name)))?
        (config)
}
```

## Training Best Practices

### Training Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
    pub data: DataConfig,
    pub training: TrainingParams,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub epochs: usize,
    pub batch_size: usize,
    pub accumulation_steps: usize,
    pub mixed_precision: bool,
    pub clip_grad_norm: Option<f64>,
    pub early_stopping: Option<EarlyStoppingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
    pub metric: String,
    pub mode: String, // "min" or "max"
}
```

### Training Loop Implementation

```rust
pub struct Trainer {
    model: Box<dyn Module>,
    optimizer: Box<dyn Optimizer>,
    scheduler: Option<Box<dyn LRScheduler>>,
    config: TrainingConfig,
    device: Device,
}

impl Trainer {
    pub async fn train(&mut self, train_loader: DataLoader, val_loader: DataLoader) -> Result<TrainingResults> {
        let mut best_metric = f64::NEG_INFINITY;
        let mut patience_counter = 0;
        let mut training_history = Vec::new();
        
        for epoch in 0..self.config.training.epochs {
            // Training phase
            self.model.train();
            let train_metrics = self.train_epoch(&train_loader).await?;
            
            // Validation phase
            self.model.eval();
            let val_metrics = self.validate_epoch(&val_loader).await?;
            
            // Learning rate scheduling
            if let Some(scheduler) = &mut self.scheduler {
                scheduler.step(val_metrics.get("loss").copied());
            }
            
            // Logging
            let epoch_results = EpochResults {
                epoch,
                train_metrics,
                val_metrics: val_metrics.clone(),
                learning_rate: self.optimizer.get_lr(),
            };
            
            training_history.push(epoch_results.clone());
            self.log_epoch_results(&epoch_results)?;
            
            // Early stopping check
            if let Some(early_stopping) = &self.config.training.early_stopping {
                let current_metric = val_metrics.get(&early_stopping.metric)
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY);
                
                if self.is_improvement(current_metric, best_metric, early_stopping) {
                    best_metric = current_metric;
                    patience_counter = 0;
                    self.save_checkpoint("best_model.pth")?;
                } else {
                    patience_counter += 1;
                    if patience_counter >= early_stopping.patience {
                        println!("Early stopping triggered at epoch {}", epoch);
                        break;
                    }
                }
            }
            
            // Periodic checkpointing
            if epoch % 10 == 0 {
                self.save_checkpoint(&format!("checkpoint_epoch_{}.pth", epoch))?;
            }
        }
        
        Ok(TrainingResults {
            history: training_history,
            best_metric,
            final_epoch: epoch,
        })
    }
    
    async fn train_epoch(&mut self, dataloader: &DataLoader) -> Result<HashMap<String, f64>> {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut batch_count = 0;
        
        for (batch_idx, (inputs, targets)) in dataloader.enumerate() {
            // Move to device
            let inputs = inputs.to_device(&self.device)?;
            let targets = targets.to_device(&self.device)?;
            
            // Forward pass
            let outputs = self.model.forward(&inputs)?;
            let loss = F::cross_entropy(&outputs, &targets)?;
            
            // Backward pass
            if batch_idx % self.config.training.accumulation_steps == 0 {
                self.optimizer.zero_grad();
            }
            
            let scaled_loss = loss / self.config.training.accumulation_steps as f32;
            scaled_loss.backward()?;
            
            if (batch_idx + 1) % self.config.training.accumulation_steps == 0 {
                // Gradient clipping
                if let Some(clip_norm) = self.config.training.clip_grad_norm {
                    clip_grad_norm_(self.model.parameters(), clip_norm)?;
                }
                
                self.optimizer.step()?;
            }
            
            // Metrics
            total_loss += loss.item::<f32>()? as f64;
            let accuracy = self.calculate_accuracy(&outputs, &targets)?;
            total_accuracy += accuracy;
            batch_count += 1;
            
            // Progress logging
            if batch_idx % 100 == 0 {
                println!("Batch {}: Loss = {:.6}, Accuracy = {:.4}", 
                        batch_idx, loss.item::<f32>()?, accuracy);
            }
        }
        
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), total_loss / batch_count as f64);
        metrics.insert("accuracy".to_string(), total_accuracy / batch_count as f64);
        
        Ok(metrics)
    }
}
```

### Checkpoint Management

```rust
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub epoch: usize,
    pub model_state: ModelState,
    pub optimizer_state: OptimizerState,
    pub scheduler_state: Option<SchedulerState>,
    pub metrics: HashMap<String, f64>,
    pub config: TrainingConfig,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Trainer {
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let checkpoint = Checkpoint {
            epoch: self.current_epoch,
            model_state: self.model.state_dict(),
            optimizer_state: self.optimizer.state_dict(),
            scheduler_state: self.scheduler.as_ref().map(|s| s.state_dict()),
            metrics: self.current_metrics.clone(),
            config: self.config.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        let serialized = bincode::serialize(&checkpoint)?;
        std::fs::write(path, serialized)?;
        
        Ok(())
    }
    
    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let data = std::fs::read(path)?;
        let checkpoint: Checkpoint = bincode::deserialize(&data)?;
        
        self.model.load_state_dict(&checkpoint.model_state)?;
        self.optimizer.load_state_dict(&checkpoint.optimizer_state)?;
        
        if let (Some(scheduler), Some(scheduler_state)) = (&mut self.scheduler, checkpoint.scheduler_state) {
            scheduler.load_state_dict(&scheduler_state)?;
        }
        
        self.current_epoch = checkpoint.epoch;
        self.current_metrics = checkpoint.metrics;
        
        Ok(())
    }
}
```

## Deployment

### Model Optimization for Deployment

```rust
use torsh::jit::trace;
use torsh::quantization::quantize_dynamic;

pub fn optimize_for_deployment(model: &dyn Module, example_input: &Tensor) -> Result<OptimizedModel> {
    // JIT compilation
    let traced_model = trace(model, example_input)?;
    
    // Quantization
    let quantized_model = quantize_dynamic(&traced_model, &QuantizationConfig::default())?;
    
    // Graph optimization
    let optimized_graph = optimize_graph(quantized_model.graph())?;
    
    Ok(OptimizedModel::new(optimized_graph))
}

// Model serving
use warp::Filter;

pub async fn serve_model(model: OptimizedModel, port: u16) -> Result<()> {
    let model = Arc::new(model);
    
    let predict = warp::path("predict")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_model(model.clone()))
        .and_then(handle_prediction);
    
    let routes = predict.with(warp::cors().allow_any_origin());
    
    warp::serve(routes)
        .run(([0, 0, 0, 0], port))
        .await;
    
    Ok(())
}

async fn handle_prediction(
    input: PredictionRequest,
    model: Arc<OptimizedModel>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let tensor = input.to_tensor()?;
    let output = model.forward(&tensor)?;
    let response = PredictionResponse::from_tensor(output)?;
    Ok(warp::reply::json(&response))
}
```

## Debugging

### Debug Utilities

```rust
// Debug tensor information
pub fn debug_tensor(tensor: &Tensor, name: &str) {
    println!("=== {} ===", name);
    println!("Shape: {:?}", tensor.shape());
    println!("DType: {:?}", tensor.dtype());
    println!("Device: {:?}", tensor.device());
    println!("Requires grad: {}", tensor.requires_grad());
    
    if tensor.numel() <= 10 {
        println!("Values: {:?}", tensor.to_vec::<f32>());
    } else {
        println!("Min: {:.6}", tensor.min().unwrap().item::<f32>().unwrap());
        println!("Max: {:.6}", tensor.max().unwrap().item::<f32>().unwrap());
        println!("Mean: {:.6}", tensor.mean().unwrap().item::<f32>().unwrap());
        println!("Std: {:.6}", tensor.std().unwrap().item::<f32>().unwrap());
    }
    
    if tensor.has_nan().unwrap() {
        println!("⚠️  Contains NaN values!");
    }
    
    if !tensor.is_finite().unwrap() {
        println!("⚠️  Contains infinite values!");
    }
    
    println!();
}

// Gradient debugging
pub fn debug_gradients(model: &dyn Module) {
    for (name, param) in model.named_parameters() {
        if let Some(grad) = param.grad() {
            let grad_norm = grad.norm().item::<f32>().unwrap();
            println!("{}: grad_norm = {:.6}", name, grad_norm);
            
            if grad_norm > 10.0 {
                println!("⚠️  Large gradient in {}", name);
            }
            
            if grad_norm < 1e-8 {
                println!("⚠️  Very small gradient in {}", name);
            }
        } else {
            println!("{}: No gradient", name);
        }
    }
}
```

### Profiling

```rust
use torsh::profiler::Profiler;

pub fn profile_training_step(
    model: &dyn Module,
    optimizer: &mut dyn Optimizer,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<()> {
    let mut profiler = Profiler::new();
    
    profiler.start("forward_pass");
    let outputs = model.forward(inputs)?;
    profiler.end("forward_pass");
    
    profiler.start("loss_computation");
    let loss = F::cross_entropy(&outputs, targets)?;
    profiler.end("loss_computation");
    
    profiler.start("backward_pass");
    optimizer.zero_grad();
    loss.backward()?;
    profiler.end("backward_pass");
    
    profiler.start("optimizer_step");
    optimizer.step()?;
    profiler.end("optimizer_step");
    
    // Print profiling results
    println!("{}", profiler.summary());
    
    Ok(())
}
```

## Security

### Input Validation

```rust
pub fn validate_input_tensor(tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
    // Shape validation
    if tensor.shape().dims() != expected_shape {
        return Err(ModelError::Config(format!(
            "Invalid input shape: expected {:?}, got {:?}",
            expected_shape,
            tensor.shape().dims()
        )));
    }
    
    // Value range validation
    let min_val = tensor.min()?.item::<f32>()?;
    let max_val = tensor.max()?.item::<f32>()?;
    
    if min_val < -10.0 || max_val > 10.0 {
        return Err(ModelError::Config(
            "Input values outside expected range [-10, 10]".to_string()
        ));
    }
    
    // NaN/Inf validation
    if tensor.has_nan()? {
        return Err(ModelError::Config("Input contains NaN values".to_string()));
    }
    
    if !tensor.is_finite()? {
        return Err(ModelError::Config("Input contains infinite values".to_string()));
    }
    
    Ok(())
}
```

### Safe Model Loading

```rust
use std::path::Path;

pub fn safe_load_model<P: AsRef<Path>>(path: P) -> Result<Box<dyn Module>> {
    let path = path.as_ref();
    
    // Validate file extension
    if !path.extension().map_or(false, |ext| ext == "torsh" || ext == "pth") {
        return Err(ModelError::Config("Invalid model file extension".to_string()));
    }
    
    // Check file size (prevent loading extremely large files)
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > 10 * 1024 * 1024 * 1024 { // 10GB limit
        return Err(ModelError::Config("Model file too large".to_string()));
    }
    
    // Load and validate model
    let model = Model::load(path)?;
    
    // Validate model structure
    validate_model_structure(&model)?;
    
    Ok(model)
}

fn validate_model_structure(model: &dyn Module) -> Result<()> {
    // Check parameter count
    let param_count = model.parameters().iter().map(|p| p.numel()).sum::<usize>();
    if param_count > 1_000_000_000 { // 1B parameter limit
        return Err(ModelError::Config("Model too large".to_string()));
    }
    
    // Validate parameter values
    for param in model.parameters() {
        if param.has_nan()? {
            return Err(ModelError::Config("Model contains NaN parameters".to_string()));
        }
        
        let param_norm = param.norm()?.item::<f32>()?;
        if param_norm > 1000.0 {
            return Err(ModelError::Config("Model contains abnormally large parameters".to_string()));
        }
    }
    
    Ok(())
}
```

## Conclusion

Following these best practices will help you build robust, efficient, and maintainable deep learning applications with ToRSh. Remember to:

1. **Start simple** - Begin with basic implementations and gradually add complexity
2. **Test thoroughly** - Write comprehensive tests for all components
3. **Profile regularly** - Monitor performance and optimize bottlenecks
4. **Document everything** - Maintain clear documentation for all APIs
5. **Handle errors gracefully** - Implement comprehensive error handling
6. **Plan for scale** - Design with production deployment in mind

For more specific guidance on particular topics, refer to the individual component documentation and examples in the ToRSh repository.