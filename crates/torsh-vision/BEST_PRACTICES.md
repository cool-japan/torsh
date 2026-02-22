# ToRSh Vision - Best Practices Guide

This guide provides comprehensive best practices for using ToRSh Vision effectively in production computer vision applications, covering performance optimization, memory management, error handling, and development workflows.

## Table of Contents

1. [Performance Optimization](#performance-optimization)
2. [Memory Management](#memory-management)
3. [Hardware Utilization](#hardware-utilization)
4. [Error Handling and Debugging](#error-handling-and-debugging)
5. [Data Pipeline Design](#data-pipeline-design)
6. [Model Development](#model-development)
7. [Production Deployment](#production-deployment)
8. [Testing and Validation](#testing-and-validation)
9. [Code Organization](#code-organization)
10. [Monitoring and Profiling](#monitoring-and-profiling)

## Performance Optimization

### 1. Choose Optimal Data Types

```rust
// Use appropriate data types for your use case
let image_f32 = Tensor::zeros([3, 224, 224], DType::F32, Device::Cpu);  // Training
let image_f16 = Tensor::zeros([3, 224, 224], DType::F16, Device::Cpu);  // Inference (if supported)
let image_u8 = Tensor::zeros([3, 224, 224], DType::U8, Device::Cpu);    // Storage/I/O

// Convert when necessary
let training_tensor = image_u8.to(DType::F32)? / 255.0;  // Normalize to [0,1]
```

### 2. Optimize Transform Pipelines

```rust
// ✅ Good: Batch operations when possible
fn efficient_batch_processing(images: &[Tensor]) -> Result<Vec<Tensor>> {
    let transforms = TransformBuilder::new()
        .resize((224, 224))
        .imagenet_normalize()
        .build();
    
    // Stack into batch for efficient processing
    let batch = Tensor::stack(images, 0)?;
    let processed_batch = transforms.forward(&batch)?;
    
    // Unstack back to individual tensors
    (0..images.len()).map(|i| processed_batch.select(0, i))
        .collect::<Result<Vec<_>>>()
}

// ❌ Avoid: Processing images one by one when batch processing is possible
fn inefficient_processing(images: &[Tensor]) -> Result<Vec<Tensor>> {
    let transforms = TransformBuilder::new()
        .resize((224, 224))
        .imagenet_normalize()
        .build();
    
    images.iter()
        .map(|img| transforms.forward(img))
        .collect()
}
```

### 3. Use Hardware-Aware Transforms

```rust
// Automatically select optimal implementation
fn create_optimal_pipeline() -> Result<Box<dyn Transform>> {
    let hardware = HardwareContext::auto_detect()?;
    
    if hardware.cuda_available() {
        // Use GPU-accelerated transforms
        Ok(Box::new(create_gpu_pipeline()?))
    } else {
        // Use CPU-optimized transforms
        Ok(Box::new(create_cpu_pipeline()?))
    }
}

// Unified transform API for automatic optimization
fn use_unified_transforms() -> Result<()> {
    let context = TransformContext::auto_optimize()?;
    let resize = UnifiedResize::new((224, 224));
    
    // Automatically uses best available hardware
    let processed = resize.apply(&image, &context)?;
    Ok(())
}
```

### 4. Efficient Memory Access Patterns

```rust
// ✅ Good: Sequential memory access
fn efficient_image_processing(image: &Tensor) -> Result<Tensor> {
    // Process channels sequentially for better cache locality
    let mut result = image.clone();
    
    for channel in 0..3 {
        let channel_data = result.select(0, channel)?;
        // Process channel data...
    }
    
    Ok(result)
}

// ✅ Good: Reuse tensors when possible
struct ImageProcessor {
    temp_buffer: Tensor,
}

impl ImageProcessor {
    fn new() -> Result<Self> {
        Ok(Self {
            temp_buffer: Tensor::zeros([3, 1024, 1024], DType::F32, Device::Cpu),
        })
    }
    
    fn process(&mut self, image: &Tensor) -> Result<Tensor> {
        // Reuse temp_buffer instead of allocating new tensors
        self.temp_buffer.copy_from(image)?;
        // Process temp_buffer...
        Ok(self.temp_buffer.clone())
    }
}
```

## Memory Management

### 1. Configure Global Memory Settings

```rust
use torsh_vision::memory::*;

// Configure at application startup
fn setup_memory_management() -> Result<()> {
    let settings = MemorySettings {
        enable_pooling: true,
        max_pool_size: 1000,              // Pool up to 1000 tensors
        max_batch_memory_mb: 4096,        // 4GB max per batch
        enable_profiling: true,           // Monitor memory usage
        auto_optimization: true,          // Automatic optimizations
    };
    
    configure_global_memory(settings);
    
    // Monitor memory usage
    let profiler = MemoryProfiler::new();
    profiler.start_monitoring()?;
    
    Ok(())
}
```

### 2. Use Tensor Pooling for Training Loops

```rust
fn memory_efficient_training() -> Result<()> {
    let mut tensor_pool = TensorPool::new(200);  // Pool for 200 tensors
    
    for epoch in 0..num_epochs {
        for batch_idx in 0..num_batches {
            // Get tensors from pool
            let batch_tensor = tensor_pool.get_tensor(&[32, 3, 224, 224])?;
            let label_tensor = tensor_pool.get_tensor(&[32])?;
            
            // Use tensors for training
            // ... training code ...
            
            // Return to pool when done
            tensor_pool.return_tensor(batch_tensor)?;
            tensor_pool.return_tensor(label_tensor)?;
        }
        
        // Monitor pool efficiency
        let stats = tensor_pool.stats();
        println!("Pool reuse rate: {:.1}%", stats.reuse_rate * 100.0);
    }
    
    Ok(())
}
```

### 3. Optimize Batch Sizes

```rust
use torsh_vision::memory::MemoryOptimizer;

fn calculate_optimal_batch_size(image_shape: &[usize], available_memory_gb: f32) -> Result<usize> {
    let memory_optimizer = MemoryOptimizer::new();
    
    let optimal_batch = memory_optimizer.calculate_optimal_batch_size(
        image_shape,
        (available_memory_gb * 1024.0) as usize,  // Convert to MB
        0.8  // Use 80% of available memory
    )?;
    
    println!("Optimal batch size for {}GB memory: {}", available_memory_gb, optimal_batch);
    Ok(optimal_batch)
}

// Dynamic batch size adjustment
fn adaptive_batch_processing() -> Result<()> {
    let mut current_batch_size = 32;
    let max_batch_size = 128;
    
    loop {
        match process_batch_with_size(current_batch_size) {
            Ok(_) => {
                // Successful - try larger batch
                if current_batch_size < max_batch_size {
                    current_batch_size = (current_batch_size * 1.2) as usize;
                }
            }
            Err(VisionError::OutOfMemory(_)) => {
                // Out of memory - reduce batch size
                current_batch_size = (current_batch_size * 0.8) as usize;
                if current_batch_size < 1 {
                    break;
                }
            }
            Err(e) => return Err(e),
        }
    }
    
    Ok(())
}
```

### 4. Clean Up Resources

```rust
// Implement proper cleanup
struct VisionPipeline {
    tensor_pool: TensorPool,
    memory_profiler: MemoryProfiler,
}

impl Drop for VisionPipeline {
    fn drop(&mut self) {
        // Clean up resources
        self.tensor_pool.clear();
        let final_stats = self.memory_profiler.summary();
        println!("Peak memory usage: {:.2} GB", 
                 final_stats.peak_usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    }
}
```

## Hardware Utilization

### 1. Automatic Hardware Detection

```rust
fn setup_hardware_context() -> Result<TransformContext> {
    let hardware = HardwareContext::auto_detect()?;
    
    println!("Hardware capabilities:");
    println!("  CUDA available: {}", hardware.cuda_available());
    println!("  Mixed precision: {}", hardware.supports_mixed_precision());
    println!("  Tensor cores: {}", hardware.has_tensor_cores());
    println!("  Memory: {:.1} GB", hardware.memory_gb());
    
    let context = TransformContext::new(hardware)
        .with_auto_optimization(true)
        .with_mixed_precision_enabled(true);
    
    Ok(context)
}
```

### 2. GPU Memory Management

```rust
fn gpu_memory_best_practices() -> Result<()> {
    let hardware = HardwareContext::auto_detect()?;
    
    if hardware.cuda_available() {
        // Pre-allocate GPU memory to avoid fragmentation
        let gpu_pool = GpuTensorPool::new(hardware.memory_gb() * 0.8)?;  // Use 80% of GPU memory
        
        // Use pinned memory for faster CPU-GPU transfers
        let cpu_tensor = Tensor::zeros_pinned([32, 3, 224, 224], DType::F32)?;
        let gpu_tensor = cpu_tensor.to_device(Device::Cuda(0))?;
        
        // Batch GPU operations
        let batch_operations = vec![
            gpu_resize,
            gpu_normalize,
            gpu_augment,
        ];
        
        let result = batch_gpu_operations(&gpu_tensor, &batch_operations)?;
    }
    
    Ok(())
}
```

### 3. Mixed Precision Training

```rust
fn mixed_precision_best_practices() -> Result<()> {
    let hardware = HardwareContext::auto_detect()?;
    
    if hardware.supports_mixed_precision() {
        let mut mixed_precision = MixedPrecisionTraining::new()?;
        
        // Use f16 for forward pass to save memory
        let image_f16 = image.to(DType::F16)?;
        let features_f16 = model.forward(&image_f16)?;
        
        // Convert to f32 for loss computation (for numerical stability)
        let features_f32 = features_f16.to(DType::F32)?;
        let loss = compute_loss(&features_f32, &targets)?;
        
        // Scale loss for f16 gradient computation
        let scaled_loss = mixed_precision.scale_loss(&loss)?;
        
        // Backward pass and unscale gradients
        scaled_loss.backward()?;
        mixed_precision.unscale_gradients(&optimizer)?;
        mixed_precision.step(&optimizer)?;
    }
    
    Ok(())
}
```

## Error Handling and Debugging

### 1. Comprehensive Error Handling

```rust
use torsh_vision::error_handling::*;

// Use enhanced error types for better debugging
fn robust_image_processing(image_path: &str) -> Result<Tensor> {
    let image = VisionIO::load_image(image_path)
        .map_err(|e| EnhancedVisionError::IoError {
            path: image_path.to_string(),
            operation: "load_image".to_string(),
            message: e.to_string(),
            suggestions: vec![
                "Check file exists and is readable".to_string(),
                "Verify image format is supported".to_string(),
            ],
        })?;
    
    // Validate image before processing
    validate_image_tensor(&image)
        .map_err(|e| EnhancedVisionError::ValidationError {
            tensor_shape: image.shape().dims().to_vec(),
            expected_constraints: "CHW format with C=1 or C=3".to_string(),
            message: e.to_string(),
            suggestions: vec![
                "Convert to RGB if grayscale".to_string(),
                "Check image loading was successful".to_string(),
            ],
        })?;
    
    Ok(image)
}

// Error recovery strategies
fn process_with_fallbacks(image: &Tensor) -> Result<Tensor> {
    // Try GPU processing first
    if let Ok(result) = try_gpu_processing(image) {
        return Ok(result);
    }
    
    println!("GPU processing failed, falling back to CPU");
    
    // Fallback to CPU processing
    if let Ok(result) = try_cpu_processing(image) {
        return Ok(result);
    }
    
    println!("CPU processing failed, using minimal processing");
    
    // Minimal fallback
    Ok(image.clone())
}
```

### 2. Debugging Utilities

```rust
// Debug transform pipelines
fn debug_transform_pipeline() -> Result<()> {
    let mut debug_pipeline = DebugTransformPipeline::new();
    
    debug_pipeline
        .add(Box::new(Resize::new((224, 224))))
        .add(Box::new(RandomHorizontalFlip::new(0.5)))
        .add(Box::new(Normalize::new(imagenet_mean(), imagenet_std())));
    
    let input = Tensor::randn([3, 512, 512], DType::F32, Device::Cpu);
    
    // Process with debugging
    let result = debug_pipeline.forward_with_debug(&input)?;
    
    // Print debug information
    for (i, debug_info) in debug_pipeline.debug_info().iter().enumerate() {
        println!("Step {}: {} -> {}", 
                 i, 
                 debug_info.input_shape_str(), 
                 debug_info.output_shape_str());
        println!("  Time: {:.2}ms", debug_info.execution_time_ms());
        println!("  Memory: {:.1}MB", debug_info.memory_usage_mb());
    }
    
    Ok(())
}

// Tensor inspection utilities
fn inspect_tensor(tensor: &Tensor, name: &str) {
    println!("Tensor '{}' inspection:", name);
    println!("  Shape: {:?}", tensor.shape());
    println!("  DType: {:?}", tensor.dtype());
    println!("  Device: {:?}", tensor.device());
    
    if let Ok(stats) = calculate_tensor_stats(tensor) {
        println!("  Mean: {:.4}", stats.mean);
        println!("  Std: {:.4}", stats.std);
        println!("  Min: {:.4}", stats.min);
        println!("  Max: {:.4}", stats.max);
    }
}
```

### 3. Assertion and Validation

```rust
// Use assertions for debugging
fn validate_training_data(image: &Tensor, label: &Tensor) -> Result<()> {
    // Shape assertions
    debug_assert_eq!(image.ndim(), 3, "Image must be 3D (CHW)");
    debug_assert_eq!(image.shape()[0], 3, "Image must have 3 channels");
    debug_assert_eq!(label.ndim(), 0, "Label must be scalar");
    
    // Value range assertions
    if cfg!(debug_assertions) {
        let image_min = image.min()?;
        let image_max = image.max()?;
        assert!(image_min >= 0.0, "Image values must be non-negative");
        assert!(image_max <= 1.0, "Image values must be <= 1.0 for normalized data");
    }
    
    Ok(())
}

// Production validation
fn production_validate_batch(images: &Tensor, labels: &Tensor) -> Result<()> {
    // Validate shapes
    if images.ndim() != 4 {
        return Err(VisionError::InvalidShape(
            format!("Expected 4D batch tensor, got {}D", images.ndim())
        ));
    }
    
    let batch_size = images.shape()[0];
    if labels.shape()[0] != batch_size {
        return Err(VisionError::InvalidShape(
            format!("Batch size mismatch: images={}, labels={}", 
                    batch_size, labels.shape()[0])
        ));
    }
    
    // Validate data ranges
    let image_stats = calculate_tensor_stats(images)?;
    if image_stats.min < -3.0 || image_stats.max > 3.0 {
        return Err(VisionError::InvalidInput(
            "Image values outside expected range [-3, 3]".to_string()
        ));
    }
    
    Ok(())
}
```

## Data Pipeline Design

### 1. Efficient Data Loading

```rust
fn create_production_data_pipeline() -> Result<DataPipeline> {
    let config = DatasetConfig {
        cache_size_mb: 8192,           // 8GB cache
        prefetch_size: 256,            // Prefetch 256 samples
        max_workers: num_cpus::get(),  // Use all CPU cores
        enable_validation: true,       // Validate data integrity
        compression: true,             // Compress cached data
        memory_mapping: false,         // Use regular I/O for now
    };
    
    // Create optimized dataset
    let dataset = OptimizedImageDataset::new("data/train", config)?
        .with_transforms(create_training_transforms()?)
        .with_error_recovery(true);   // Handle corrupted images gracefully
    
    // Create parallel data loader
    let data_loader = ParallelDataLoader::new(dataset, 8)?  // 8 worker threads
        .with_batch_size(64)
        .with_shuffle(true)
        .with_pin_memory(true)        // For GPU training
        .with_drop_last(true);        // Consistent batch sizes
    
    Ok(DataPipeline::new(data_loader))
}
```

### 2. Separate Training and Validation Pipelines

```rust
struct TrainingPipeline {
    train_loader: ParallelDataLoader<OptimizedImageDataset>,
    val_loader: ParallelDataLoader<OptimizedImageDataset>,
}

impl TrainingPipeline {
    fn new(data_root: &str) -> Result<Self> {
        // Training pipeline with augmentation
        let train_transforms = TransformBuilder::new()
            .resize((256, 256))
            .random_resized_crop((224, 224))
            .random_horizontal_flip(0.5)
            .color_jitter(0.4, 0.4, 0.4, 0.1)
            .random_erasing(0.25)
            .imagenet_normalize()
            .build();
        
        // Validation pipeline without augmentation
        let val_transforms = TransformBuilder::new()
            .resize((256, 256))
            .center_crop((224, 224))
            .imagenet_normalize()
            .build();
        
        let train_dataset = OptimizedImageDataset::new(
            &format!("{}/train", data_root),
            DatasetConfig::default_training()
        )?.with_transforms(train_transforms);
        
        let val_dataset = OptimizedImageDataset::new(
            &format!("{}/val", data_root),
            DatasetConfig::default_validation()
        )?.with_transforms(val_transforms);
        
        Ok(Self {
            train_loader: ParallelDataLoader::new(train_dataset, 8)?,
            val_loader: ParallelDataLoader::new(val_dataset, 4)?,
        })
    }
}
```

### 3. Progressive Loading

```rust
// Load data progressively for large datasets
struct ProgressiveDataLoader {
    current_loader: ParallelDataLoader<OptimizedImageDataset>,
    next_batch_prepared: bool,
    background_loader: Option<thread::JoinHandle<()>>,
}

impl ProgressiveDataLoader {
    fn new(dataset: OptimizedImageDataset) -> Result<Self> {
        let loader = ParallelDataLoader::new(dataset, 4)?;
        
        Ok(Self {
            current_loader: loader,
            next_batch_prepared: false,
            background_loader: None,
        })
    }
    
    fn prepare_next_batch(&mut self) -> Result<()> {
        if !self.next_batch_prepared {
            // Start background loading for next batch
            // Implementation details...
            self.next_batch_prepared = true;
        }
        Ok(())
    }
}
```

## Model Development

### 1. Model Architecture Best Practices

```rust
// Use builder pattern for complex models
struct ResNetBuilder {
    layers: Vec<usize>,
    num_classes: usize,
    dropout_rate: Option<f32>,
    batch_norm: bool,
}

impl ResNetBuilder {
    fn new() -> Self {
        Self {
            layers: vec![2, 2, 2, 2],  // ResNet-18 default
            num_classes: 1000,
            dropout_rate: None,
            batch_norm: true,
        }
    }
    
    fn layers(mut self, layers: Vec<usize>) -> Self {
        self.layers = layers;
        self
    }
    
    fn num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        self
    }
    
    fn dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = Some(rate);
        self
    }
    
    fn build(self) -> Result<ResNet> {
        ResNet::new(self.layers, self.num_classes)
            .with_dropout(self.dropout_rate)
            .with_batch_norm(self.batch_norm)
    }
}

// Usage
let model = ResNetBuilder::new()
    .layers(vec![3, 4, 6, 3])  // ResNet-34
    .num_classes(100)          // CIFAR-100
    .dropout(0.5)
    .build()?;
```

### 2. Model Initialization

```rust
// Proper weight initialization
fn initialize_model(model: &mut dyn Module) -> Result<()> {
    for (name, param) in model.named_parameters() {
        match name.as_str() {
            name if name.contains("conv") && name.contains("weight") => {
                // Kaiming initialization for convolutional layers
                kaiming_normal_(param, 0.0, "fan_out", "relu")?;
            }
            name if name.contains("bn") && name.contains("weight") => {
                // BatchNorm weights to 1
                constant_(param, 1.0)?;
            }
            name if name.contains("bn") && name.contains("bias") => {
                // BatchNorm bias to 0
                constant_(param, 0.0)?;
            }
            name if name.contains("fc") && name.contains("weight") => {
                // Xavier initialization for fully connected layers
                xavier_normal_(param)?;
            }
            _ => {
                // Default initialization
                normal_(param, 0.0, 0.01)?;
            }
        }
    }
    Ok(())
}
```

### 3. Model Validation

```rust
fn validate_model_architecture(model: &dyn Module) -> Result<()> {
    // Check parameter count
    let total_params = model.parameters().iter()
        .map(|p| p.numel())
        .sum::<usize>();
    
    println!("Total parameters: {:.2}M", total_params as f64 / 1e6);
    
    // Check for gradient flow
    let sample_input = Tensor::randn([1, 3, 224, 224], DType::F32, Device::Cpu);
    let output = model.forward(&sample_input)?;
    
    println!("Model output shape: {:?}", output.shape());
    
    // Validate output shape matches expected
    let expected_classes = 1000;  // ImageNet
    if output.shape()[1] != expected_classes {
        return Err(VisionError::InvalidShape(
            format!("Expected {} classes, got {}", expected_classes, output.shape()[1])
        ));
    }
    
    Ok(())
}
```

## Production Deployment

### 1. Model Optimization for Inference

```rust
// Optimize model for production inference
fn optimize_for_inference(model: &mut dyn Module) -> Result<()> {
    // Set to evaluation mode
    model.eval();
    
    // Fuse batch normalization with convolution
    fuse_conv_bn(model)?;
    
    // Quantize model if supported
    if cfg!(feature = "quantization") {
        quantize_model(model, QuantizationConfig::default())?;
    }
    
    // JIT compile if available
    if cfg!(feature = "jit") {
        let jit_model = jit_compile(model)?;
        // Use jit_model for inference
    }
    
    Ok(())
}
```

### 2. Batch Processing for Throughput

```rust
struct InferenceEngine {
    model: Box<dyn Module>,
    batch_size: usize,
    preprocessing: Box<dyn Transform>,
    hardware_context: TransformContext,
}

impl InferenceEngine {
    fn new(model: Box<dyn Module>, batch_size: usize) -> Result<Self> {
        let preprocessing = TransformBuilder::new()
            .resize((224, 224))
            .imagenet_normalize()
            .build();
        
        let hardware_context = TransformContext::auto_optimize()?;
        
        Ok(Self {
            model,
            batch_size,
            preprocessing,
            hardware_context,
        })
    }
    
    fn predict_batch(&self, images: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        
        for chunk in images.chunks(self.batch_size) {
            // Preprocess batch
            let mut preprocessed = Vec::new();
            for image in chunk {
                let processed = self.preprocessing.forward(image)?;
                preprocessed.push(processed);
            }
            
            // Stack into batch tensor
            let batch = Tensor::stack(&preprocessed, 0)?;
            
            // Inference
            let batch_output = self.model.forward(&batch)?;
            
            // Split back to individual results
            for i in 0..chunk.len() {
                results.push(batch_output.select(0, i)?);
            }
        }
        
        Ok(results)
    }
    
    async fn predict_async(&self, image: Tensor) -> Result<Tensor> {
        // Async inference for web services
        let processed = self.preprocessing.forward(&image)?;
        let output = self.model.forward(&processed.unsqueeze(0)?)?;
        Ok(output.squeeze(0)?)
    }
}
```

### 3. Model Serving

```rust
// HTTP service for model inference
use warp::Filter;

struct ModelService {
    engine: Arc<InferenceEngine>,
}

impl ModelService {
    fn new(model_path: &str) -> Result<Self> {
        let model = load_model(model_path)?;
        let engine = Arc::new(InferenceEngine::new(model, 32)?);
        
        Ok(Self { engine })
    }
    
    fn routes(&self) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        let engine = Arc::clone(&self.engine);
        
        warp::path("predict")
            .and(warp::post())
            .and(warp::body::bytes())
            .and(warp::any().map(move || Arc::clone(&engine)))
            .and_then(Self::handle_prediction)
    }
    
    async fn handle_prediction(
        body: bytes::Bytes,
        engine: Arc<InferenceEngine>,
    ) -> Result<impl warp::Reply, warp::Rejection> {
        // Decode image from bytes
        let image = decode_image_bytes(&body)
            .map_err(|_| warp::reject::custom(InvalidImage))?;
        
        // Run inference
        let result = engine.predict_async(image).await
            .map_err(|_| warp::reject::custom(InferenceError))?;
        
        // Encode result as JSON
        let response = serde_json::json!({
            "predictions": tensor_to_predictions(&result)?,
        });
        
        Ok(warp::reply::json(&response))
    }
}
```

## Testing and Validation

### 1. Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_transform_output_shape() {
        let transform = Resize::new((224, 224));
        let input = Tensor::zeros([3, 512, 512], DType::F32, Device::Cpu);
        
        let output = transform.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 224, 224]);
    }
    
    #[test]
    fn test_normalization_statistics() {
        let mean = vec![0.485, 0.456, 0.406];
        let std = vec![0.229, 0.224, 0.225];
        let normalize = Normalize::new(mean.clone(), std.clone());
        
        // Create image with known statistics
        let mut image = Tensor::ones([3, 224, 224], DType::F32, Device::Cpu);
        
        let normalized = normalize.forward(&image).unwrap();
        
        // Check that normalization was applied correctly
        for c in 0..3 {
            let channel = normalized.select(0, c).unwrap();
            let channel_mean = channel.mean().unwrap().to_scalar().unwrap();
            let expected_mean = (1.0 - mean[c]) / std[c];
            assert_relative_eq!(channel_mean, expected_mean, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_dataset_consistency() {
        let dataset = create_test_dataset().unwrap();
        
        // Test that dataset length is consistent
        assert!(dataset.len() > 0);
        
        // Test that samples can be loaded
        for i in 0..std::cmp::min(10, dataset.len()) {
            let (image, label) = dataset.get(i).unwrap();
            assert_eq!(image.ndim(), 3);
            assert!(label >= 0);
        }
    }
}
```

### 2. Integration Testing

```rust
#[test]
fn test_end_to_end_pipeline() -> Result<()> {
    // Create test data
    let test_images = create_test_images()?;
    let test_dataset = TestDataset::new(test_images)?;
    
    // Create data loader
    let data_loader = ParallelDataLoader::new(test_dataset, 2)?
        .with_batch_size(4);
    
    // Create model
    let mut model = ResNet::resnet18(10)?;  // 10 classes for test
    
    // Test training step
    for batch in data_loader.iter()?.take(2) {  // Test 2 batches
        let (images, labels) = batch?;
        
        // Forward pass
        let outputs = model.forward(&images)?;
        assert_eq!(outputs.shape()[0], images.shape()[0]);  // Same batch size
        assert_eq!(outputs.shape()[1], 10);                 // 10 classes
        
        // Compute loss
        let loss = cross_entropy_loss(&outputs, &labels)?;
        assert!(loss.to_scalar::<f32>()? > 0.0);
        
        // Backward pass
        loss.backward()?;
        
        // Check gradients exist
        for param in model.parameters() {
            assert!(param.grad().is_some());
        }
    }
    
    Ok(())
}
```

### 3. Performance Testing

```rust
use std::time::Instant;

fn benchmark_transforms() -> Result<()> {
    let transforms = vec![
        ("Resize", Box::new(Resize::new((224, 224))) as Box<dyn Transform>),
        ("RandomHorizontalFlip", Box::new(RandomHorizontalFlip::new(0.5))),
        ("ColorJitter", Box::new(ColorJitter::new().brightness(0.2))),
        ("Normalize", Box::new(Normalize::new(vec![0.5; 3], vec![0.5; 3]))),
    ];
    
    let test_image = Tensor::randn([3, 512, 512], DType::F32, Device::Cpu);
    let iterations = 100;
    
    for (name, transform) in transforms {
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = transform.forward(&test_image)?;
        }
        
        let duration = start.elapsed();
        let avg_ms = duration.as_millis() as f64 / iterations as f64;
        
        println!("{}: {:.2} ms/image", name, avg_ms);
        
        // Performance regression test
        match name {
            "Resize" => assert!(avg_ms < 10.0, "Resize too slow: {:.2} ms", avg_ms),
            "RandomHorizontalFlip" => assert!(avg_ms < 1.0, "Flip too slow: {:.2} ms", avg_ms),
            _ => {}
        }
    }
    
    Ok(())
}
```

## Code Organization

### 1. Project Structure

```
src/
├── lib.rs                 # Main library entry point
├── prelude.rs            # Common imports
├── error_handling.rs     # Error types and utilities
├── transforms/           # Transform implementations
│   ├── mod.rs
│   ├── geometric.rs      # Resize, crop, flip, rotate
│   ├── color.rs          # Color transforms
│   ├── augmentation.rs   # Advanced augmentation
│   └── unified.rs        # Unified transform API
├── datasets/             # Dataset implementations
│   ├── mod.rs
│   ├── image_folder.rs
│   ├── cifar.rs
│   ├── mnist.rs
│   └── optimized/        # Optimized datasets
├── models/               # Model architectures
│   ├── mod.rs
│   ├── resnet.rs
│   ├── vgg.rs
│   └── detection/        # Detection models
├── hardware/             # Hardware acceleration
├── memory/               # Memory management
├── visualization/        # Interactive and 3D viz
└── examples/             # Usage examples
```

### 2. Module Organization

```rust
// lib.rs - Clean public API
#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(unused_imports)]

pub mod datasets;
pub mod transforms;
pub mod models;
pub mod hardware;
pub mod memory;
pub mod visualization;
pub mod error_handling;

// Re-export commonly used items
pub use error_handling::{VisionError, Result};
pub use transforms::{Transform, TransformBuilder};
pub use datasets::{Dataset, ImageDataset};

// Prelude for convenience
pub mod prelude {
    pub use crate::{
        VisionError, Result,
        Transform, TransformBuilder,
        Dataset, ImageDataset,
    };
    pub use torsh_tensor::{Tensor, Device, DType};
}
```

### 3. Feature Flags

```toml
# Cargo.toml
[features]
default = ["std", "cpu"]
std = []
cpu = []
cuda = ["torsh-tensor/cuda"]
mkl = ["torsh-tensor/mkl"]
quantization = ["torsh-tensor/quantization"]
visualization = ["image", "plotters"]
video = ["opencv"]
distributed = ["mpi"]
```

```rust
// Conditional compilation
#[cfg(feature = "cuda")]
pub mod cuda_transforms;

#[cfg(feature = "visualization")]
pub mod interactive;

#[cfg(feature = "quantization")]
impl Model {
    pub fn quantize(&mut self) -> Result<()> {
        // Quantization implementation
    }
}
```

## Monitoring and Profiling

### 1. Performance Monitoring

```rust
use std::time::Instant;
use std::collections::HashMap;

struct PerformanceMonitor {
    timings: HashMap<String, Vec<f64>>,
    memory_usage: HashMap<String, Vec<usize>>,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory_usage: HashMap::new(),
        }
    }
    
    fn time_operation<F, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed().as_secs_f64() * 1000.0;  // ms
        
        self.timings.entry(name.to_string())
            .or_default()
            .push(duration);
        
        result
    }
    
    fn record_memory_usage(&mut self, name: &str, bytes: usize) {
        self.memory_usage.entry(name.to_string())
            .or_default()
            .push(bytes);
    }
    
    fn report(&self) {
        println!("Performance Report:");
        
        for (name, times) in &self.timings {
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            println!("  {}: avg={:.2}ms, min={:.2}ms, max={:.2}ms", 
                     name, avg, min, max);
        }
    }
}
```

### 2. Resource Usage Tracking

```rust
struct ResourceTracker {
    gpu_memory_usage: Vec<f64>,
    cpu_utilization: Vec<f64>,
    start_time: Instant,
}

impl ResourceTracker {
    fn new() -> Self {
        Self {
            gpu_memory_usage: Vec::new(),
            cpu_utilization: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    fn sample_resources(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let gpu_memory = get_gpu_memory_usage()?;
            self.gpu_memory_usage.push(gpu_memory);
        }
        
        let cpu_usage = get_cpu_utilization()?;
        self.cpu_utilization.push(cpu_usage);
        
        Ok(())
    }
    
    fn get_summary(&self) -> ResourceSummary {
        ResourceSummary {
            total_time: self.start_time.elapsed().as_secs_f64(),
            avg_gpu_memory: self.gpu_memory_usage.iter().sum::<f64>() / self.gpu_memory_usage.len() as f64,
            peak_gpu_memory: self.gpu_memory_usage.iter().fold(0.0, |a, &b| a.max(b)),
            avg_cpu_utilization: self.cpu_utilization.iter().sum::<f64>() / self.cpu_utilization.len() as f64,
        }
    }
}
```

### 3. Logging and Telemetry

```rust
use log::{info, warn, error, debug};

// Structured logging
fn log_training_metrics(epoch: usize, loss: f32, accuracy: f32, lr: f32) {
    info!(
        "Training metrics - Epoch: {}, Loss: {:.4}, Accuracy: {:.2}%, LR: {:.6}",
        epoch, loss, accuracy * 100.0, lr
    );
}

// Performance logging
fn log_batch_processing_time(batch_size: usize, processing_time: f64) {
    let throughput = batch_size as f64 / processing_time;
    
    debug!(
        "Batch processing - Size: {}, Time: {:.2}ms, Throughput: {:.1} images/sec",
        batch_size, processing_time * 1000.0, throughput
    );
    
    if processing_time > 1.0 {  // More than 1 second per batch
        warn!("Slow batch processing detected: {:.2}s for {} images", 
              processing_time, batch_size);
    }
}

// Error logging with context
fn log_error_with_context(error: &VisionError, context: &str) {
    error!("Error in {}: {}", context, error);
    
    // Log additional debugging information
    match error {
        VisionError::InvalidShape(msg) => {
            debug!("Shape error details: {}", msg);
        }
        VisionError::IoError(io_err) => {
            debug!("IO error details: {}", io_err);
        }
        _ => {}
    }
}
```

## Conclusion

Following these best practices will help you:

- **Maximize Performance**: Efficient memory usage, hardware acceleration, and optimized data pipelines
- **Ensure Reliability**: Comprehensive error handling, validation, and testing
- **Maintain Code Quality**: Clean architecture, proper documentation, and organized code structure
- **Monitor Production Systems**: Performance tracking, resource monitoring, and proper logging
- **Scale Effectively**: Memory-aware processing, batch optimization, and hardware utilization

Remember to:
- Profile your specific use case to identify bottlenecks
- Test thoroughly before deploying to production
- Monitor resource usage in production environments
- Keep error handling comprehensive but not verbose
- Use appropriate hardware acceleration for your deployment target

For more specific examples and advanced techniques, refer to the [examples module](./src/examples.rs) and other documentation files.