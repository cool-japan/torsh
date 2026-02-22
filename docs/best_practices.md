# ToRSh Best Practices Guide

## Table of Contents

1. [Overview](#overview)
2. [Memory Management](#memory-management)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling](#error-handling)
5. [Code Organization](#code-organization)
6. [Testing Strategies](#testing-strategies)
7. [Deployment Considerations](#deployment-considerations)
8. [Security Best Practices](#security-best-practices)
9. [Debugging and Profiling](#debugging-and-profiling)
10. [Cross-Platform Development](#cross-platform-development)

## Overview

This guide provides comprehensive best practices for developing with ToRSh, covering everything from memory management to deployment strategies. Following these practices will help you build robust, efficient, and maintainable machine learning applications.

### Key Principles

1. **Safety First**: Leverage Rust's type system and ToRSh's error handling
2. **Performance Matters**: Optimize for memory usage and computational efficiency
3. **Maintainability**: Write clear, documented, and testable code
4. **Resource Awareness**: Understand device capabilities and limitations
5. **Error Resilience**: Handle failures gracefully with proper recovery strategies

## Memory Management

### 1. Tensor Lifecycle Management

#### Use References When Possible

```rust
// ✅ Good: Pass by reference
fn process_batch(data: &Tensor, model: &impl Module) -> Result<Tensor> {
    model.forward(data)
}

// ❌ Avoid: Unnecessary ownership transfer
fn process_batch(data: Tensor, model: &impl Module) -> Result<Tensor> {
    model.forward(&data)
}
```

#### Clone Strategically

```rust
// ✅ Good: Clone only when necessary
fn split_processing(data: &Tensor) -> Result<(Tensor, Tensor)> {
    let normalized = data.div_scalar(255.0)?;
    let augmented = apply_augmentations(&normalized)?; // Uses reference
    Ok((normalized, augmented))
}

// ❌ Avoid: Unnecessary cloning
fn split_processing(data: &Tensor) -> Result<(Tensor, Tensor)> {
    let normalized = data.clone().div_scalar(255.0)?;
    let augmented = data.clone();
    Ok((normalized, augmented))
}
```

### 2. In-Place Operations

Use in-place operations when the original tensor is no longer needed:

```rust
// ✅ Good: In-place operations
fn preprocess_inplace(tensor: &mut Tensor) -> Result<()> {
    tensor.div_scalar_(255.0)?; // In-place division
    tensor.sub_(&tensor.mean()?)?; // In-place subtraction
    Ok(())
}

// ✅ Also good: When original is needed later
fn preprocess_copy(tensor: &Tensor) -> Result<Tensor> {
    let normalized = tensor.div_scalar(255.0)?;
    let centered = normalized.sub(&normalized.mean()?)?;
    Ok(centered)
}
```

### 3. Memory Pool Management

#### Pre-allocate Workspaces

```rust
struct TrainingContext {
    workspace: Tensor,
    gradient_buffer: Vec<Tensor>,
    loss_history: Vec<f32>,
}

impl TrainingContext {
    fn new(model: &impl Module, max_batch_size: usize) -> Result<Self> {
        Ok(Self {
            workspace: zeros(&[max_batch_size, 1024])?,
            gradient_buffer: model.parameters().iter()
                .map(|p| zeros_like(p))
                .collect::<Result<Vec<_>>>()?,
            loss_history: Vec::with_capacity(1000),
        })
    }
    
    fn reuse_workspace(&mut self, new_shape: &[usize]) -> Result<&mut Tensor> {
        if self.workspace.numel() < new_shape.iter().product() {
            self.workspace = zeros(new_shape)?;
        } else {
            self.workspace = self.workspace.reshape(new_shape)?;
        }
        Ok(&mut self.workspace)
    }
}
```

#### Manage Device Memory

```rust
fn training_step_with_memory_management(
    model: &impl Module,
    data: &Tensor,
    target: &Tensor,
    optimizer: &mut impl Optimizer,
) -> Result<f32> {
    // Forward pass
    let output = model.forward(data)?;
    let loss = F::cross_entropy(&output, target)?;
    let loss_value = loss.item::<f32>();
    
    // Backward pass
    optimizer.zero_grad();
    loss.backward()?;
    optimizer.step()?;
    
    // Explicit memory cleanup for large tensors
    if data.numel() > 1_000_000 {
        // Force garbage collection of intermediate tensors
        std::mem::drop(output);
        std::mem::drop(loss);
    }
    
    Ok(loss_value)
}
```

### 4. Memory Monitoring

```rust
use torsh::profiler::MemoryProfiler;

fn monitor_memory_usage() -> Result<()> {
    let mut profiler = MemoryProfiler::new();
    profiler.start_monitoring();
    
    // Your training code here
    let model = create_large_model()?;
    train_model(&model)?;
    
    let report = profiler.generate_report();
    println!("Peak memory usage: {} MB", report.peak_memory_mb);
    println!("Memory leaks detected: {}", report.potential_leaks.len());
    
    if report.peak_memory_mb > 8000.0 { // 8GB threshold
        println!("⚠️  High memory usage detected. Consider:");
        println!("   - Reducing batch size");
        println!("   - Using gradient checkpointing");
        println!("   - Enabling mixed precision training");
    }
    
    Ok(())
}
```

## Performance Optimization

### 1. Batch Processing

#### Optimal Batch Sizes

```rust
fn determine_optimal_batch_size(
    model: &impl Module,
    sample_input: &Tensor,
    max_memory_mb: f64,
) -> Result<usize> {
    let mut batch_size = 1;
    let mut max_tested = 1;
    
    // Binary search for optimal batch size
    while batch_size <= 512 {
        let test_batch = sample_input.repeat(&[batch_size, 1, 1, 1])?;
        
        match test_forward_pass(model, &test_batch) {
            Ok(memory_used) if memory_used < max_memory_mb => {
                max_tested = batch_size;
                batch_size *= 2;
            }
            _ => break,
        }
    }
    
    // Use 80% of maximum to account for gradient computation
    Ok((max_tested as f64 * 0.8) as usize)
}

fn test_forward_pass(model: &impl Module, input: &Tensor) -> Result<f64> {
    let start_memory = get_memory_usage().allocated_mb;
    let _output = model.forward(input)?;
    let peak_memory = get_memory_usage().allocated_mb;
    Ok(peak_memory - start_memory)
}
```

#### Efficient Data Loading

```rust
use torsh::data::{DataLoader, TensorDataset};
use std::sync::Arc;

fn create_efficient_dataloader(
    data: Tensor,
    targets: Tensor,
    batch_size: usize,
) -> Result<DataLoader<TensorDataset>> {
    // Pin memory for faster GPU transfers
    let pinned_data = data.pin_memory()?;
    let pinned_targets = targets.pin_memory()?;
    
    let dataset = TensorDataset::from_tensors(pinned_data, pinned_targets);
    
    let dataloader = DataLoader::new_with_options(
        dataset,
        batch_size,
        true,  // shuffle
        std::cmp::min(4, num_cpus::get()), // optimal worker count
        false, // drop_last
    );
    
    Ok(dataloader)
}
```

### 2. Device Optimization

#### Automatic Device Selection

```rust
fn select_optimal_device() -> Result<Device> {
    // Priority order: CUDA > Metal > WebGPU > CPU
    if Device::cuda(0).is_available() {
        let props = Device::cuda(0).properties();
        if props.memory_gb >= 4.0 && props.compute_capability >= 6.0 {
            return Ok(Device::cuda(0));
        }
    }
    
    if Device::metal().is_available() {
        let props = Device::metal().properties();
        if props.memory_gb >= 8.0 { // Metal typically needs more memory
            return Ok(Device::metal());
        }
    }
    
    if Device::webgpu().is_available() {
        return Ok(Device::webgpu());
    }
    
    Ok(Device::cpu())
}

fn optimize_for_device(model: &mut impl Module, device: &Device) -> Result<()> {
    model.to_device(device)?;
    
    match device.device_type() {
        DeviceType::CUDA => {
            // Enable tensor cores for supported operations
            model.enable_tensor_cores(true);
            // Use streams for overlapping computation
            model.set_stream_mode(true);
        }
        DeviceType::Metal => {
            // Use Metal Performance Shaders optimizations
            model.enable_mps_optimizations(true);
        }
        DeviceType::CPU => {
            // Optimize thread count for CPU
            set_num_threads(num_cpus::get());
            // Enable SIMD optimizations
            model.enable_simd(true);
        }
        _ => {}
    }
    
    Ok(())
}
```

#### Memory Transfer Optimization

```rust
async fn async_data_transfer(
    data: &Tensor,
    target_device: &Device,
) -> Result<Tensor> {
    // Use async transfer for large tensors
    if data.numel() > 1_000_000 {
        data.to_device_async(target_device).await
    } else {
        data.to_device(target_device)
    }
}

fn optimize_multi_gpu_training() -> Result<()> {
    let devices = (0..torch::cuda::device_count())
        .map(Device::cuda)
        .collect::<Vec<_>>();
    
    // Balance model across devices
    let model_parts = split_model_across_devices(&model, &devices)?;
    
    // Pipeline parallel processing
    for batch in dataloader {
        let futures: Vec<_> = model_parts.iter()
            .zip(devices.iter())
            .map(|(part, device)| {
                let batch_device = batch.to_device(device)?;
                part.forward_async(&batch_device)
            })
            .collect();
        
        let outputs = futures::future::join_all(futures).await;
        let combined = combine_outputs(outputs)?;
    }
    
    Ok(())
}
```

### 3. Mixed Precision Training

```rust
use torsh::optim::GradScaler;

fn mixed_precision_training_loop(
    model: &impl Module,
    dataloader: DataLoader<impl Dataset>,
    optimizer: &mut impl Optimizer,
) -> Result<()> {
    let mut scaler = GradScaler::new();
    
    for (batch_idx, (data, target)) in dataloader.enumerate() {
        // Forward pass with autocast
        let output = autocast(|| model.forward(&data))?;
        let loss = F::cross_entropy(&output, &target)?;
        
        // Scale loss to prevent underflow
        let scaled_loss = scaler.scale(&loss)?;
        
        // Backward pass
        optimizer.zero_grad();
        scaled_loss.backward()?;
        
        // Unscale gradients before clipping
        scaler.unscale_gradients(optimizer)?;
        
        // Gradient clipping (optional)
        clip_grad_norm_(model.parameters(), 1.0)?;
        
        // Optimizer step with scale checking
        scaler.step(optimizer)?;
        scaler.update();
        
        // Log training progress
        if batch_idx % 100 == 0 {
            println!("Batch {}: Loss = {:.4}, Scale = {:.0}",
                batch_idx, loss.item::<f32>(), scaler.get_scale());
        }
    }
    
    Ok(())
}
```

### 4. Gradient Checkpointing

```rust
fn memory_efficient_large_model() -> Result<impl Module> {
    struct CheckpointedResNet {
        layers: Vec<ResidualBlock>,
        checkpoint_every: usize,
    }
    
    impl Module for CheckpointedResNet {
        fn forward(&self, x: &Tensor) -> Result<Tensor> {
            let mut x = x.clone();
            
            for (i, layer) in self.layers.iter().enumerate() {
                if i % self.checkpoint_every == 0 {
                    // Use gradient checkpointing to save memory
                    x = gradient_checkpoint(|| layer.forward(&x))?;
                } else {
                    x = layer.forward(&x)?;
                }
            }
            
            Ok(x)
        }
    }
    
    Ok(CheckpointedResNet {
        layers: (0..50).map(|_| ResidualBlock::new(512)).collect(),
        checkpoint_every: 5, // Checkpoint every 5 layers
    })
}
```

## Error Handling

### 1. Comprehensive Error Management

#### Structured Error Types

```rust
use torsh::{Result, TorshError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    
    #[error("Model not initialized: {reason}")]
    NotInitialized { reason: String },
    
    #[error("Training failed at epoch {epoch}, batch {batch}: {source}")]
    TrainingFailed {
        epoch: usize,
        batch: usize,
        #[source]
        source: TorshError,
    },
    
    #[error("Resource exhausted: {resource} limit of {limit} exceeded")]
    ResourceExhausted { resource: String, limit: String },
    
    #[error("Device error: {0}")]
    Device(#[from] TorshError),
}

type ModelResult<T> = std::result::Result<T, ModelError>;
```

#### Error Recovery Strategies

```rust
fn robust_training_step(
    model: &impl Module,
    data: &Tensor,
    target: &Tensor,
    optimizer: &mut impl Optimizer,
    retry_count: usize,
) -> ModelResult<f32> {
    for attempt in 0..=retry_count {
        match training_step(model, data, target, optimizer) {
            Ok(loss) => return Ok(loss),
            Err(e) => {
                match &e {
                    ModelError::Device(TorshError::OutOfMemory) => {
                        // Try to recover from OOM
                        empty_cache();
                        
                        if attempt < retry_count {
                            println!("⚠️  OOM detected, reducing batch size and retrying...");
                            let smaller_data = data.narrow(0, 0, data.size(0) / 2)?;
                            let smaller_target = target.narrow(0, 0, target.size(0) / 2)?;
                            return robust_training_step(
                                model, &smaller_data, &smaller_target, 
                                optimizer, retry_count - 1
                            );
                        }
                    }
                    ModelError::Device(TorshError::DeviceNotAvailable) => {
                        // Fallback to CPU
                        println!("⚠️  GPU not available, falling back to CPU");
                        let cpu_data = data.to_device(&Device::cpu())?;
                        let cpu_target = target.to_device(&Device::cpu())?;
                        return training_step(model, &cpu_data, &cpu_target, optimizer);
                    }
                    _ => {
                        if attempt == retry_count {
                            return Err(e);
                        }
                        
                        println!("⚠️  Attempt {} failed: {}, retrying...", attempt + 1, e);
                        std::thread::sleep(std::time::Duration::from_millis(100 * (attempt + 1) as u64));
                    }
                }
            }
        }
    }
    
    Err(ModelError::TrainingFailed {
        epoch: 0,
        batch: 0,
        source: TorshError::Other("Max retries exceeded".to_string()),
    })
}
```

### 2. Validation and Assertions

#### Input Validation

```rust
fn validate_model_inputs(
    input: &Tensor,
    expected_shape: &[usize],
    expected_dtype: DType,
) -> ModelResult<()> {
    // Shape validation
    if input.shape().dims() != expected_shape {
        return Err(ModelError::InvalidInputShape {
            expected: expected_shape.to_vec(),
            actual: input.shape().dims().to_vec(),
        });
    }
    
    // Dtype validation
    if input.dtype() != expected_dtype {
        return Err(ModelError::Device(TorshError::Other(
            format!("Expected dtype {:?}, got {:?}", expected_dtype, input.dtype())
        )));
    }
    
    // Numerical validation
    if has_nan(input) {
        return Err(ModelError::Device(TorshError::Other(
            "Input contains NaN values".to_string()
        )));
    }
    
    if has_inf(input) {
        return Err(ModelError::Device(TorshError::Other(
            "Input contains infinite values".to_string()
        )));
    }
    
    // Range validation (example for image data)
    if input.min()?.item::<f32>() < 0.0 || input.max()?.item::<f32>() > 1.0 {
        println!("⚠️  Input values outside [0, 1] range, consider normalization");
    }
    
    Ok(())
}
```

#### Model State Validation

```rust
fn validate_model_state(model: &impl Module) -> ModelResult<()> {
    let parameters = model.parameters();
    
    for (i, param) in parameters.iter().enumerate() {
        // Check for NaN parameters
        if has_nan(param) {
            return Err(ModelError::NotInitialized {
                reason: format!("Parameter {} contains NaN values", i),
            });
        }
        
        // Check for exploding parameters
        let param_norm = param.norm(None)?.item::<f32>();
        if param_norm > 100.0 {
            println!("⚠️  Large parameter norm detected: {:.2} at parameter {}", param_norm, i);
        }
        
        // Check for vanishing parameters
        if param_norm < 1e-6 {
            println!("⚠️  Very small parameter norm detected: {:.2e} at parameter {}", param_norm, i);
        }
    }
    
    Ok(())
}
```

## Code Organization

### 1. Modular Architecture

#### Separate Concerns

```rust
// models/mod.rs
pub mod bert;
pub mod gpt;
pub mod vision_transformer;
pub mod resnet;

pub trait Model: Module {
    type Config;
    
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;
    fn load_pretrained(path: &str) -> Result<Self> where Self: Sized;
    fn save_pretrained(&self, path: &str) -> Result<()>;
}

// training/mod.rs
pub mod trainer;
pub mod callbacks;
pub mod metrics;

pub trait Trainer {
    type Model: Model;
    type DataLoader;
    
    fn train_epoch(&mut self, dataloader: Self::DataLoader) -> Result<TrainingMetrics>;
    fn validate(&mut self, dataloader: Self::DataLoader) -> Result<ValidationMetrics>;
    fn save_checkpoint(&self, path: &str) -> Result<()>;
}

// data/mod.rs
pub mod datasets;
pub mod transforms;
pub mod loaders;

pub trait DataProcessor {
    type Input;
    type Output;
    
    fn process(&self, input: Self::Input) -> Result<Self::Output>;
}
```

#### Configuration Management

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
    pub evaluation: EvaluationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub device: String,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub optimizer: OptimizerConfig,
    pub scheduler: Option<SchedulerConfig>,
    pub gradient_clipping: Option<f64>,
    pub mixed_precision: bool,
}

impl ExperimentConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::from_str(&contents)?
        } else {
            serde_json::from_str(&contents)?
        };
        
        config.validate()?;
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.training.batch_size == 0 {
            return Err(TorshError::Other("Batch size must be > 0".to_string()));
        }
        
        if self.training.learning_rate <= 0.0 {
            return Err(TorshError::Other("Learning rate must be > 0".to_string()));
        }
        
        if self.model.num_layers == 0 {
            return Err(TorshError::Other("Number of layers must be > 0".to_string()));
        }
        
        Ok(())
    }
}
```

### 2. Interface Design

#### Clean API Design

```rust
pub struct ModelBuilder<M> {
    config: Option<M::Config>,
    device: Option<Device>,
    dtype: Option<DType>,
    pretrained: Option<String>,
}

impl<M: Model> ModelBuilder<M> {
    pub fn new() -> Self {
        Self {
            config: None,
            device: None,
            dtype: None,
            pretrained: None,
        }
    }
    
    pub fn config(mut self, config: M::Config) -> Self {
        self.config = Some(config);
        self
    }
    
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
    
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    
    pub fn pretrained<S: Into<String>>(mut self, path: S) -> Self {
        self.pretrained = Some(path.into());
        self
    }
    
    pub fn build(self) -> Result<M> {
        let mut model = if let Some(pretrained_path) = self.pretrained {
            M::load_pretrained(&pretrained_path)?
        } else if let Some(config) = self.config {
            M::new(config)?
        } else {
            return Err(TorshError::Other("Either config or pretrained path must be provided".to_string()));
        };
        
        if let Some(device) = self.device {
            model.to_device(&device)?;
        }
        
        if let Some(dtype) = self.dtype {
            model.to_dtype(dtype)?;
        }
        
        Ok(model)
    }
}

// Usage example:
fn create_model() -> Result<BertModel> {
    let model = ModelBuilder::new()
        .config(BertConfig::default())
        .device(Device::cuda(0))
        .dtype(DType::F16)
        .build()?;
    
    Ok(model)
}
```

## Testing Strategies

### 1. Unit Testing

#### Tensor Operation Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use torsh::testing::*;
    
    #[test]
    fn test_tensor_operations() {
        let a = tensor![1.0, 2.0, 3.0];
        let b = tensor![4.0, 5.0, 6.0];
        
        let result = a.add(&b).unwrap();
        let expected = tensor![5.0, 7.0, 9.0];
        
        assert_tensors_close(&result, &expected, 1e-6);
    }
    
    #[test]
    fn test_gradient_computation() {
        let x = tensor![2.0].requires_grad_(true);
        let y = x.pow(2.0).unwrap();
        
        y.backward().unwrap();
        
        let grad = x.grad().unwrap();
        assert_relative_eq!(grad.item::<f32>(), 4.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_shape_broadcasting() {
        let a = randn(&[2, 1, 4]);
        let b = randn(&[1, 3, 1]);
        
        let result = a.add(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3, 4]);
    }
    
    #[test]
    fn test_device_transfer() {
        let tensor = randn(&[10, 10]);
        
        // Test CPU to CUDA transfer (if available)
        if Device::cuda(0).is_available() {
            let cuda_tensor = tensor.to_device(&Device::cuda(0)).unwrap();
            assert_eq!(cuda_tensor.device(), &Device::cuda(0));
            
            // Test back to CPU
            let cpu_tensor = cuda_tensor.to_device(&Device::cpu()).unwrap();
            assert_eq!(cpu_tensor.device(), &Device::cpu());
            
            // Verify data integrity
            assert_tensors_close(&tensor, &cpu_tensor, 1e-6);
        }
    }
}
```

#### Model Architecture Tests

```rust
#[test]
fn test_model_forward_pass() {
    let config = BertConfig {
        hidden_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        ..Default::default()
    };
    
    let model = BertModel::new(config).unwrap();
    let input_ids = randint(0, 1000, &[2, 10]); // batch_size=2, seq_len=10
    
    let output = model.forward(&input_ids).unwrap();
    
    // Check output shape
    assert_eq!(output.shape().dims(), &[2, 10, 128]);
    
    // Check output is finite
    assert!(!has_nan(&output));
    assert!(!has_inf(&output));
}

#[test]
fn test_model_gradient_flow() {
    let model = LinearModel::new(784, 10);
    let input = randn(&[32, 784]);
    let target = randint(0, 10, &[32]);
    
    // Forward pass
    let output = model.forward(&input).unwrap();
    let loss = F::cross_entropy(&output, &target).unwrap();
    
    // Backward pass
    loss.backward().unwrap();
    
    // Check all parameters have gradients
    for (i, param) in model.parameters().iter().enumerate() {
        assert!(param.has_grad(), "Parameter {} missing gradient", i);
        let grad = param.grad().unwrap();
        assert!(!has_nan(grad), "Parameter {} has NaN gradient", i);
    }
}
```

### 2. Integration Testing

#### End-to-End Training Tests

```rust
#[test]
fn test_complete_training_loop() {
    // Setup
    let model = create_test_model();
    let mut optimizer = Adam::new(model.parameters(), 0.001).unwrap();
    let dataloader = create_test_dataloader();
    
    let initial_loss = evaluate_model(&model, &dataloader).unwrap();
    
    // Training
    for epoch in 0..3 {
        for (data, target) in &dataloader {
            let output = model.forward(&data).unwrap();
            let loss = F::cross_entropy(&output, &target).unwrap();
            
            optimizer.zero_grad();
            loss.backward().unwrap();
            optimizer.step().unwrap();
        }
    }
    
    // Verify improvement
    let final_loss = evaluate_model(&model, &dataloader).unwrap();
    assert!(final_loss < initial_loss, 
           "Model should improve during training: {} -> {}", 
           initial_loss, final_loss);
}
```

### 3. Performance Testing

#### Benchmark Tests

```rust
use criterion::{criterion_group, criterion_main, Criterion, black_box};

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let sizes = vec![128, 256, 512, 1024];
    
    for size in sizes {
        let a = randn(&[size, size]);
        let b = randn(&[size, size]);
        
        c.bench_function(&format!("matmul_{}x{}", size, size), |bench| {
            bench.iter(|| {
                black_box(a.matmul(&b).unwrap())
            })
        });
    }
}

fn benchmark_model_inference(c: &mut Criterion) {
    let model = BertModel::new(BertConfig::default()).unwrap();
    let input = randint(0, 1000, &[1, 512]);
    
    c.bench_function("bert_inference", |bench| {
        bench.iter(|| {
            black_box(model.forward(&input).unwrap())
        })
    });
}

criterion_group!(benches, benchmark_matrix_multiplication, benchmark_model_inference);
criterion_main!(benches);
```

## Deployment Considerations

### 1. Model Optimization for Production

#### Model Quantization

```rust
use torsh::quantization::{quantize_model, QuantizationConfig};

fn optimize_model_for_deployment(model: impl Module) -> Result<impl Module> {
    // Post-training quantization
    let quantization_config = QuantizationConfig {
        dtype: DType::I8,
        calibration_dataset_size: 1000,
        symmetric: true,
        per_channel: true,
    };
    
    let quantized_model = quantize_model(model, quantization_config)?;
    
    // Verify accuracy after quantization
    let accuracy_drop = validate_quantized_model(&quantized_model)?;
    if accuracy_drop > 0.02 { // 2% threshold
        println!("⚠️  Quantization caused {:.1}% accuracy drop", accuracy_drop * 100.0);
    }
    
    Ok(quantized_model)
}
```

#### Model Compilation

```rust
use torsh::jit::{compile_model, OptimizationLevel};

fn compile_model_for_production(model: impl Module) -> Result<impl Module> {
    let compiled_model = compile_model(
        model,
        OptimizationLevel::Aggressive,
        &["fusion", "constant_folding", "dead_code_elimination"],
    )?;
    
    // Warm up compiled model
    let dummy_input = zeros(&[1, 784]);
    for _ in 0..10 {
        let _ = compiled_model.forward(&dummy_input)?;
    }
    
    Ok(compiled_model)
}
```

### 2. Inference Optimization

#### Batch Processing

```rust
pub struct BatchedInferenceEngine {
    model: Box<dyn Module>,
    max_batch_size: usize,
    timeout_ms: u64,
    pending_requests: Vec<InferenceRequest>,
}

impl BatchedInferenceEngine {
    pub fn new(model: Box<dyn Module>, max_batch_size: usize) -> Self {
        Self {
            model,
            max_batch_size,
            timeout_ms: 100, // 100ms timeout
            pending_requests: Vec::new(),
        }
    }
    
    pub async fn predict(&mut self, input: Tensor) -> Result<Tensor> {
        let request = InferenceRequest::new(input);
        self.pending_requests.push(request);
        
        if self.pending_requests.len() >= self.max_batch_size {
            self.process_batch().await
        } else {
            // Wait for more requests or timeout
            tokio::time::sleep(Duration::from_millis(self.timeout_ms)).await;
            self.process_batch().await
        }
    }
    
    async fn process_batch(&mut self) -> Result<Tensor> {
        if self.pending_requests.is_empty() {
            return Err(TorshError::Other("No pending requests".to_string()));
        }
        
        // Combine inputs into batch
        let batch_input = combine_tensors(
            &self.pending_requests.iter()
                .map(|req| &req.input)
                .collect::<Vec<_>>()
        )?;
        
        // Run inference
        let batch_output = self.model.forward(&batch_input)?;
        
        // Split outputs
        let outputs = split_tensor_batch(&batch_output, self.pending_requests.len())?;
        
        // Send results back to requesters
        for (request, output) in self.pending_requests.drain(..).zip(outputs) {
            request.respond(output);
        }
        
        Ok(batch_output)
    }
}
```

### 3. Monitoring and Logging

#### Production Monitoring

```rust
use torsh::profiler::{ModelProfiler, PerformanceMetrics};

pub struct ProductionMonitor {
    profiler: ModelProfiler,
    metrics_collector: MetricsCollector,
    alert_thresholds: AlertThresholds,
}

impl ProductionMonitor {
    pub fn monitor_inference(&mut self, 
        model: &impl Module,
        input: &Tensor,
    ) -> Result<(Tensor, InferenceMetrics)> {
        let start_time = std::time::Instant::now();
        let start_memory = get_memory_usage();
        
        // Run inference with profiling
        self.profiler.start();
        let output = model.forward(input)?;
        let profile = self.profiler.stop();
        
        let inference_time = start_time.elapsed();
        let memory_used = get_memory_usage() - start_memory;
        
        let metrics = InferenceMetrics {
            latency_ms: inference_time.as_millis() as f64,
            memory_mb: memory_used.allocated_mb,
            throughput: input.size(0) as f64 / inference_time.as_secs_f64(),
            gpu_utilization: profile.gpu_utilization,
        };
        
        // Check alerts
        self.check_performance_alerts(&metrics)?;
        
        // Log metrics
        self.metrics_collector.record(metrics.clone());
        
        Ok((output, metrics))
    }
    
    fn check_performance_alerts(&self, metrics: &InferenceMetrics) -> Result<()> {
        if metrics.latency_ms > self.alert_thresholds.max_latency_ms {
            println!("🚨 High latency alert: {:.1}ms", metrics.latency_ms);
        }
        
        if metrics.memory_mb > self.alert_thresholds.max_memory_mb {
            println!("🚨 High memory usage alert: {:.1}MB", metrics.memory_mb);
        }
        
        if metrics.gpu_utilization < self.alert_thresholds.min_gpu_utilization {
            println!("⚠️  Low GPU utilization: {:.1}%", metrics.gpu_utilization * 100.0);
        }
        
        Ok(())
    }
}
```

## Security Best Practices

### 1. Input Validation and Sanitization

```rust
pub fn sanitize_model_input(input: &Tensor) -> Result<Tensor> {
    // Check for malicious patterns
    if input.numel() > 10_000_000 {
        return Err(TorshError::Other("Input too large".to_string()));
    }
    
    // Clamp extreme values
    let clamped = input.clamp(-100.0, 100.0)?;
    
    // Replace NaN/Inf with safe values
    let sanitized = replace_non_finite(&clamped, 0.0)?;
    
    // Apply input normalization
    let normalized = sanitized.div_scalar(255.0)?;
    
    Ok(normalized)
}

pub fn validate_model_output(output: &Tensor) -> Result<()> {
    // Check for NaN/Inf in outputs
    if has_nan(output) || has_inf(output) {
        return Err(TorshError::Other("Model output contains invalid values".to_string()));
    }
    
    // Check output range for classification
    if output.ndim() == 2 { // Assume classification logits
        let softmax_output = F::softmax(output, -1)?;
        let max_prob = softmax_output.max()?.item::<f32>();
        if max_prob < 0.1 {
            println!("⚠️  Low confidence prediction: {:.3}", max_prob);
        }
    }
    
    Ok(())
}
```

### 2. Model Security

```rust
pub fn secure_model_loading(model_path: &str) -> Result<impl Module> {
    // Verify model file integrity
    verify_model_checksum(model_path)?;
    
    // Load with security constraints
    let model = load_model_with_constraints(
        model_path,
        ModelConstraints {
            max_parameters: 1_000_000_000, // 1B parameters max
            max_file_size_mb: 4000,        // 4GB max
            allowed_operations: vec!["linear", "conv", "attention"],
            disallowed_operations: vec!["exec", "file_io"],
        }
    )?;
    
    // Validate model architecture
    validate_model_architecture(&model)?;
    
    Ok(model)
}

fn verify_model_checksum(model_path: &str) -> Result<()> {
    let file_hash = compute_file_hash(model_path)?;
    let expected_hash = load_expected_hash(model_path)?;
    
    if file_hash != expected_hash {
        return Err(TorshError::Other("Model file integrity check failed".to_string()));
    }
    
    Ok(())
}
```

## Debugging and Profiling

### 1. Debugging Tools

#### Tensor Inspection

```rust
pub fn debug_tensor(tensor: &Tensor, name: &str) {
    println!("🔍 Debug: {}", name);
    println!("  Shape: {:?}", tensor.shape().dims());
    println!("  DType: {:?}", tensor.dtype());
    println!("  Device: {:?}", tensor.device());
    println!("  Requires grad: {}", tensor.requires_grad());
    
    if tensor.numel() > 0 {
        let stats = tensor_stats(tensor);
        println!("  Min: {:.6}", stats.min);
        println!("  Max: {:.6}", stats.max);
        println!("  Mean: {:.6}", stats.mean);
        println!("  Std: {:.6}", stats.std);
        
        if has_nan(tensor) {
            println!("  ⚠️  Contains NaN values!");
        }
        
        if has_inf(tensor) {
            println!("  ⚠️  Contains infinite values!");
        }
        
        // Show first few values for small tensors
        if tensor.numel() <= 20 {
            println!("  Values: {:?}", tensor.to_vec::<f32>());
        }
    }
    
    println!();
}
```

#### Model Architecture Visualization

```rust
pub fn visualize_model_architecture(model: &impl Module) -> Result<()> {
    let summary = ModelSummary::new(model);
    summary.print_detailed();
    
    // Generate architecture graph
    let graph = create_computation_graph(model)?;
    graph.save_as_dot("model_architecture.dot")?;
    
    // Memory analysis
    let memory_analysis = analyze_memory_usage(model)?;
    println!("Memory Analysis:");
    println!("  Parameters: {:.2} MB", memory_analysis.parameters_mb);
    println!("  Activations: {:.2} MB", memory_analysis.activations_mb);
    println!("  Gradients: {:.2} MB", memory_analysis.gradients_mb);
    println!("  Total: {:.2} MB", memory_analysis.total_mb);
    
    Ok(())
}
```

### 2. Performance Profiling

#### Training Performance Analysis

```rust
pub fn profile_training_performance(
    model: &impl Module,
    dataloader: &DataLoader<impl Dataset>,
) -> Result<TrainingProfile> {
    let mut profiler = TrainingProfiler::new();
    
    profiler.start();
    
    for (batch_idx, (data, target)) in dataloader.enumerate().take(10) {
        profiler.mark("data_loading");
        
        let output = model.forward(&data)?;
        profiler.mark("forward_pass");
        
        let loss = F::cross_entropy(&output, &target)?;
        profiler.mark("loss_computation");
        
        loss.backward()?;
        profiler.mark("backward_pass");
        
        profiler.step();
        
        if batch_idx >= 9 { break; } // Profile first 10 batches
    }
    
    let profile = profiler.finish();
    
    println!("Training Performance Profile:");
    println!("  Data loading: {:.2}ms", profile.avg_data_loading_ms);
    println!("  Forward pass: {:.2}ms", profile.avg_forward_ms);
    println!("  Loss computation: {:.2}ms", profile.avg_loss_ms);
    println!("  Backward pass: {:.2}ms", profile.avg_backward_ms);
    println!("  Total per batch: {:.2}ms", profile.avg_total_ms);
    println!("  Throughput: {:.1} samples/sec", profile.samples_per_second);
    
    // Identify bottlenecks
    if profile.avg_data_loading_ms > profile.avg_forward_ms {
        println!("💡 Data loading is the bottleneck. Consider:");
        println!("   - Increasing num_workers");
        println!("   - Pre-loading data to memory");
        println!("   - Using faster storage");
    }
    
    Ok(profile)
}
```

## Cross-Platform Development

### 1. Platform-Specific Optimizations

```rust
#[cfg(target_arch = "x86_64")]
fn optimize_for_x86_64() {
    // Enable AVX2/AVX-512 optimizations
    enable_avx_optimizations();
}

#[cfg(target_arch = "aarch64")]
fn optimize_for_arm64() {
    // Enable NEON optimizations
    enable_neon_optimizations();
}

#[cfg(target_os = "macos")]
fn optimize_for_macos() {
    // Use Metal Performance Shaders
    enable_mps_acceleration();
}

#[cfg(target_os = "linux")]
fn optimize_for_linux() {
    // Use CUDA optimizations if available
    if is_cuda_available() {
        enable_cuda_optimizations();
    }
}
```

### 2. WebAssembly Support

```rust
#[cfg(target_arch = "wasm32")]
pub mod wasm_support {
    use wasm_bindgen::prelude::*;
    
    #[wasm_bindgen]
    pub struct WasmModel {
        inner: Box<dyn Module>,
    }
    
    #[wasm_bindgen]
    impl WasmModel {
        #[wasm_bindgen(constructor)]
        pub fn new(model_bytes: &[u8]) -> Result<WasmModel, JsValue> {
            let model = load_model_from_bytes(model_bytes)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
            Ok(WasmModel {
                inner: Box::new(model),
            })
        }
        
        #[wasm_bindgen]
        pub fn predict(&self, input: &[f32], shape: &[usize]) -> Result<Vec<f32>, JsValue> {
            let tensor = Tensor::from_data(input, shape);
            let output = self.inner.forward(&tensor)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
            Ok(output.to_vec())
        }
    }
}
```

---

This comprehensive best practices guide covers all major aspects of ToRSh development, from memory management to deployment strategies. Following these practices will help you build robust, efficient, and maintainable machine learning applications with ToRSh.