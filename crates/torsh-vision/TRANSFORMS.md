# ToRSh Vision - Transform APIs and Usage Patterns

This document provides comprehensive documentation for the transform system in ToRSh Vision, covering basic transforms, advanced augmentation techniques, unified transforms, and usage patterns.

## Table of Contents

1. [Overview](#overview)
2. [Basic Transform API](#basic-transform-api)
3. [Transform Composition](#transform-composition)
4. [Built-in Transforms](#built-in-transforms)
5. [Advanced Augmentation](#advanced-augmentation)
6. [Unified Transform API](#unified-transform-api)
7. [Hardware Acceleration](#hardware-acceleration)
8. [Custom Transforms](#custom-transforms)
9. [Error Handling](#error-handling)
10. [Performance Considerations](#performance-considerations)
11. [Best Practices](#best-practices)

## Overview

ToRSh Vision provides a comprehensive transform system for image preprocessing and data augmentation. The system supports:

- **Basic transforms**: Resize, crop, flip, rotate, normalize
- **Advanced augmentation**: MixUp, CutMix, AutoAugment, RandAugment
- **Hardware acceleration**: GPU-accelerated transforms with automatic fallback
- **Memory optimization**: Efficient memory usage and tensor pooling
- **Type safety**: Compile-time guarantees and robust error handling

## Basic Transform API

### Transform Trait

All transforms implement the `Transform` trait:

```rust
pub trait Transform: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn describe(&self) -> String;
}
```

### Creating and Using Transforms

```rust
use torsh_vision::transforms::*;

// Create individual transforms
let resize = Resize::new((224, 224));
let normalize = Normalize::new(
    vec![0.485, 0.456, 0.406],  // ImageNet mean
    vec![0.229, 0.224, 0.225]   // ImageNet std
);

// Apply transforms
let resized = resize.forward(&input_tensor)?;
let normalized = normalize.forward(&resized)?;
```

### Transform Builder Pattern

The `TransformBuilder` provides a fluent API for creating transform pipelines:

```rust
let transforms = TransformBuilder::new()
    .resize((256, 256))
    .center_crop((224, 224))
    .imagenet_normalize()
    .build();

let processed = transforms.forward(&image)?;
```

## Transform Composition

### Compose Transform

Chain multiple transforms together:

```rust
let composed = Compose::new(vec![
    Box::new(Resize::new((256, 256))),
    Box::new(CenterCrop::new((224, 224))),
    Box::new(Normalize::new(mean, std)),
]);

let result = composed.forward(&image)?;
```

### Sequential Processing

```rust
let mut pipeline = vec![
    Box::new(Resize::new((256, 256))) as Box<dyn Transform>,
    Box::new(RandomHorizontalFlip::new(0.5)),
    Box::new(ColorJitter::new().brightness(0.2)),
];

let mut result = image.clone();
for transform in &pipeline {
    result = transform.forward(&result)?;
}
```

## Built-in Transforms

### Geometric Transforms

#### Resize
```rust
// Basic resize
let resize = Resize::new((224, 224));

// With interpolation mode
let resize_nearest = Resize::with_interpolation((224, 224), InterpolationMode::Nearest);
let resize_bilinear = Resize::with_interpolation((224, 224), InterpolationMode::Bilinear);
```

#### Crop Transforms
```rust
// Center crop
let center_crop = CenterCrop::new((224, 224));

// Random crop
let random_crop = RandomCrop::new((224, 224));

// Random crop with padding
let random_crop_padded = RandomCrop::with_padding((224, 224), 32);

// Random resized crop (like ImageNet training)
let random_resized_crop = RandomResizedCrop::new((224, 224))
    .scale((0.08, 1.0))
    .ratio((0.75, 1.33));
```

#### Flip Transforms
```rust
// Horizontal flip
let hflip = RandomHorizontalFlip::new(0.5);  // 50% probability

// Vertical flip
let vflip = RandomVerticalFlip::new(0.3);   // 30% probability

// Deterministic flips
let flip_h = HorizontalFlip::new();
let flip_v = VerticalFlip::new();
```

#### Rotation
```rust
// Fixed angle rotation
let rotate_90 = Rotation::new(90.0);

// Random rotation
let random_rotate = RandomRotation::new((-15.0, 15.0));

// With custom fill value
let rotate_fill = RandomRotation::new((-30.0, 30.0))
    .with_fill_value(128);  // Gray fill
```

### Color Transforms

#### Color Jitter
```rust
// Basic color jitter
let color_jitter = ColorJitter::new()
    .brightness(0.4)        // ±40% brightness
    .contrast(0.4)          // ±40% contrast
    .saturation(0.4)        // ±40% saturation
    .hue(0.1);              // ±10% hue

// Probability-based application
let color_jitter_prob = ColorJitter::new()
    .brightness(0.2)
    .with_probability(0.8); // Apply 80% of the time
```

#### Normalization
```rust
// ImageNet normalization
let imagenet_norm = Normalize::new(
    vec![0.485, 0.456, 0.406],
    vec![0.229, 0.224, 0.225]
);

// Custom normalization
let custom_norm = Normalize::new(
    vec![0.5, 0.5, 0.5],    // Mean
    vec![0.5, 0.5, 0.5]     // Std
);

// Per-channel normalization
let grayscale_norm = Normalize::new(vec![0.5], vec![0.5]);
```

#### Grayscale
```rust
// Convert to grayscale
let to_grayscale = ToGrayscale::new();

// Convert to grayscale with specified number of output channels
let to_grayscale_3ch = ToGrayscale::with_output_channels(3);
```

### Augmentation Transforms

#### Random Erasing
```rust
let random_erasing = RandomErasing::new(0.25)  // 25% probability
    .scale((0.02, 0.33))       // 2-33% of image area
    .ratio((0.3, 3.3))         // Aspect ratio range
    .value(0);                 // Fill value (0 = random)
```

#### Cutout
```rust
let cutout = Cutout::new(16);  // 16x16 square holes

// Multiple holes
let cutout_multiple = Cutout::new(16)
    .num_holes(3)
    .fill_value(128);
```

#### Padding
```rust
// Symmetric padding
let pad = Pad::new(32);  // 32 pixels on all sides

// Asymmetric padding
let pad_custom = Pad::with_padding((10, 20, 30, 40));  // left, top, right, bottom

// Reflection padding
let pad_reflect = Pad::new(16).with_mode(PaddingMode::Reflect);
```

## Advanced Augmentation

### MixUp
Blends two images and their labels:

```rust
let mut mixup = MixUp::new(1.0);  // Alpha parameter for Beta distribution

// Apply to image pair
let (mixed_image, mixed_labels) = mixup.apply_pair(
    &image1, &image2,
    label1, label2,
    num_classes
)?;

// The mixed_labels tensor contains the mixing ratio for both labels
```

### CutMix
Replaces a rectangular region of one image with another:

```rust
let mut cutmix = CutMix::new(1.0);  // Alpha parameter

let (cut_image, cut_labels) = cutmix.apply_pair(
    &image1, &image2,
    label1, label2,
    num_classes
)?;

// The mixing ratio is proportional to the area of the cut region
```

### AutoAugment
Policy-based augmentation with predefined strategies:

```rust
let auto_augment = AutoAugment::new();

// Apply random policy
let augmented = auto_augment.forward(&image)?;

// With custom policy
let custom_policy = AutoAugmentPolicy::ImageNet;
let auto_augment_custom = AutoAugment::with_policy(custom_policy);
```

### RandAugment
Randomly applies N operations with magnitude M:

```rust
let rand_augment = RandAugment::new(2, 5.0);  // 2 operations, magnitude 5

// With custom operation set
let operations = vec![
    RandAugmentOp::AutoContrast,
    RandAugmentOp::Brightness,
    RandAugmentOp::Color,
    RandAugmentOp::Contrast,
];
let rand_augment_custom = RandAugment::with_operations(2, 5.0, operations);
```

### AugMix
Multi-chain augmentation mixing:

```rust
let augmix = AugMix::new()
    .severity(3)           // Augmentation severity
    .width(3)              // Number of augmentation chains
    .depth(-1)             // Random depth (1-3)
    .alpha(1.0);           // Mixing parameter

let augmented = augmix.forward(&image)?;
```

### GridMask
Structured masking for improved robustness:

```rust
let gridmask = GridMask::new()
    .ratio(0.6)            // Grid hole ratio
    .rotate(true)          // Random rotation
    .probability(0.5);     // Application probability

let masked = gridmask.forward(&image)?;
```

### Mosaic
Multi-image composition for object detection:

```rust
let mosaic = Mosaic::new((640, 640));

// Apply to 4 images
let images = vec![image1, image2, image3, image4];
let mosaic_image = mosaic.apply(&images)?;
```

## Unified Transform API

The unified transform API provides hardware-aware execution and advanced features:

### UnifiedTransform Trait

```rust
pub trait UnifiedTransform: Send + Sync {
    fn apply(&self, input: &Tensor, context: &TransformContext) -> Result<Tensor>;
    fn apply_gpu(&self, input: &Tensor, context: &TransformContext) -> Result<Tensor>;
    fn apply_gpu_f16(&self, input: &Tensor, context: &TransformContext) -> Result<Tensor>;
    
    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>>;
    fn supports_gpu(&self) -> bool;
    fn supports_mixed_precision(&self) -> bool;
}
```

### Transform Context

```rust
use torsh_vision::unified_transforms::*;

// Auto-detect hardware
let hardware = HardwareContext::auto_detect()?;
let context = TransformContext::new(hardware);

// Create unified transforms
let resize = UnifiedResize::new((224, 224));
let normalize = UnifiedNormalize::new(mean, std);

// Automatic hardware selection
let processed = resize.apply(&image, &context)?;

// Explicit GPU usage (if available)
if context.hardware().cuda_available() {
    let gpu_processed = resize.apply_gpu(&image, &context)?;
}

// Mixed precision
if context.hardware().supports_mixed_precision() {
    let f16_processed = resize.apply_gpu_f16(&image, &context)?;
}
```

### Unified Composition

```rust
let unified_pipeline = UnifiedCompose::new(vec![
    Box::new(UnifiedResize::new((224, 224))),
    Box::new(UnifiedCenterCrop::new((224, 224))),
    Box::new(UnifiedNormalize::new(mean, std)),
]);

// Automatic optimization based on context
let result = unified_pipeline.apply(&image, &context)?;
```

### Bridge Between APIs

Convert between legacy and unified transforms:

```rust
// Legacy to unified
let legacy_resize = Resize::new((224, 224));
let unified_from_legacy = TransformBridge::new(Box::new(legacy_resize));

// Unified to legacy
let unified_resize = UnifiedResize::new((224, 224));
let legacy_from_unified = UnifiedTransformBridge::new(Box::new(unified_resize));
```

## Hardware Acceleration

### GPU Transforms

```rust
use torsh_vision::hardware::*;

// GPU-specific transforms
let gpu_resize = GpuResize::new((224, 224));
let gpu_normalize = GpuNormalize::new(mean, std);
let gpu_color_jitter = GpuColorJitter::new().brightness(0.2);

// GPU transform chain
let gpu_chain = GpuAugmentationChain::new(vec![
    Box::new(gpu_resize),
    Box::new(gpu_normalize),
    Box::new(gpu_color_jitter),
]);

// Apply with automatic fallback
let result = gpu_chain.forward(&image)?;
```

### Mixed Precision Support

```rust
// Mixed precision wrapper
let mixed_precision_transform = MixedPrecisionTransform::new(
    Box::new(gpu_resize)
);

// Automatic f16/f32 conversion
let result = mixed_precision_transform.apply(&image, &context)?;
```

### Batch Processing

```rust
let batch_processor = BatchProcessor::new(64)  // Batch size 64
    .with_hardware_acceleration(true)
    .with_mixed_precision(true);

// Process multiple images efficiently
let batch_results = batch_processor.process_batch(&images, &transforms)?;
```

## Custom Transforms

### Implementing Transform Trait

```rust
use torsh_vision::transforms::Transform;
use torsh_vision::{Result, VisionError};

pub struct CustomBlur {
    kernel_size: usize,
    sigma: f32,
}

impl CustomBlur {
    pub fn new(kernel_size: usize, sigma: f32) -> Self {
        Self { kernel_size, sigma }
    }
}

impl Transform for CustomBlur {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Validate input
        if input.ndim() < 2 {
            return Err(VisionError::InvalidShape(
                "Input must be at least 2D".to_string()
            ));
        }

        // Apply Gaussian blur
        gaussian_blur(input, self.kernel_size, self.sigma)
    }

    fn describe(&self) -> String {
        format!("CustomBlur(kernel_size={}, sigma={})", self.kernel_size, self.sigma)
    }
}

// Usage
let custom_blur = CustomBlur::new(5, 1.0);
let blurred = custom_blur.forward(&image)?;
```

### Implementing UnifiedTransform

```rust
use torsh_vision::unified_transforms::*;

pub struct UnifiedCustomBlur {
    kernel_size: usize,
    sigma: f32,
}

impl UnifiedTransform for UnifiedCustomBlur {
    fn apply(&self, input: &Tensor, context: &TransformContext) -> Result<Tensor> {
        if context.hardware().cuda_available() && self.supports_gpu() {
            self.apply_gpu(input, context)
        } else {
            // CPU implementation
            gaussian_blur(input, self.kernel_size, self.sigma)
        }
    }

    fn apply_gpu(&self, input: &Tensor, _context: &TransformContext) -> Result<Tensor> {
        // GPU implementation
        gpu_gaussian_blur(input, self.kernel_size, self.sigma)
    }

    fn apply_gpu_f16(&self, input: &Tensor, context: &TransformContext) -> Result<Tensor> {
        // Convert to f16, process, convert back
        let input_f16 = input.to(DType::F16)?;
        let result_f16 = self.apply_gpu(&input_f16, context)?;
        result_f16.to(DType::F32)
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        Ok(input_shape.to_vec())  // Shape unchanged
    }

    fn supports_gpu(&self) -> bool {
        true
    }

    fn supports_mixed_precision(&self) -> bool {
        true
    }
}
```

### Parameterized Transforms

```rust
pub struct ParameterizedTransform {
    params: HashMap<String, TransformParameter>,
}

impl ParameterizedTransform {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    pub fn set_param<T: Into<TransformParameter>>(&mut self, name: &str, value: T) {
        self.params.insert(name.to_string(), value.into());
    }

    pub fn get_param<T>(&self, name: &str) -> Option<T>
    where
        T: TryFrom<TransformParameter>,
    {
        self.params.get(name)?.clone().try_into().ok()
    }
}

#[derive(Clone, Debug)]
pub enum TransformParameter {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
    FloatArray(Vec<f32>),
}
```

## Error Handling

### Comprehensive Error Types

```rust
use torsh_vision::error_handling::*;

// Enhanced error handling with context
fn apply_transforms_with_context(image: &Tensor) -> Result<Tensor> {
    let resize = Resize::new((224, 224));
    
    resize.forward(image)
        .map_err(|e| EnhancedVisionError::TransformError {
            transform_name: "Resize".to_string(),
            input_shape: image.shape().dims().to_vec(),
            expected_shape: vec![3, 224, 224],
            message: e.to_string(),
            suggestions: vec![
                "Ensure input tensor has correct dimensions".to_string(),
                "Check that input is not empty".to_string(),
            ],
        })
}
```

### Error Recovery

```rust
// Automatic error recovery
fn robust_transform_pipeline(image: &Tensor) -> Result<Tensor> {
    let mut result = image.clone();
    
    // Try GPU acceleration first, fallback to CPU
    let transforms = if HardwareContext::auto_detect()?.cuda_available() {
        create_gpu_pipeline()
    } else {
        create_cpu_pipeline()
    };
    
    for transform in transforms {
        result = transform.forward(&result)
            .or_else(|_| {
                // Fallback to basic implementation
                println!("Transform failed, using fallback");
                Ok(result.clone())
            })?;
    }
    
    Ok(result)
}
```

### Validation Utilities

```rust
use torsh_vision::error_handling::validation::*;

// Validate transform inputs
fn validate_and_transform(image: &Tensor, transform: &dyn Transform) -> Result<Tensor> {
    // Validate tensor shape
    validate_tensor_shape(image, &[Some(3), None, None])?;  // 3 channels, any H/W
    
    // Validate tensor range
    validate_tensor_range(image, 0.0, 255.0)?;
    
    // Validate not empty
    validate_not_empty(image)?;
    
    // Apply transform
    transform.forward(image)
}
```

## Performance Considerations

### Memory Optimization

```rust
// Use in-place operations when possible
let mut image = load_image("input.jpg")?;

// In-place normalization
normalize_inplace(&mut image, &mean, &std)?;

// Tensor pooling for frequent operations
let mut pool = TensorPool::new(100);
let temp_tensor = pool.get_tensor(&image.shape().dims())?;
// Use temp_tensor for intermediate results
pool.return_tensor(temp_tensor)?;
```

### Batch Processing

```rust
// Process multiple images efficiently
fn batch_transform(images: &[Tensor], transform: &dyn Transform) -> Result<Vec<Tensor>> {
    // Stack into batch tensor
    let batch = Tensor::stack(images, 0)?;
    
    // Apply transform to entire batch
    let transformed_batch = transform.forward(&batch)?;
    
    // Unstack back to individual tensors
    let batch_size = images.len();
    let mut results = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let slice = transformed_batch.select(0, i)?;
        results.push(slice);
    }
    
    Ok(results)
}
```

### Asynchronous Processing

```rust
use tokio::task;

// Parallel transform processing
async fn parallel_transform(
    images: Vec<Tensor>,
    transform: Arc<dyn Transform + Send + Sync>
) -> Result<Vec<Tensor>> {
    let tasks: Vec<_> = images.into_iter().map(|img| {
        let transform = Arc::clone(&transform);
        task::spawn_blocking(move || transform.forward(&img))
    }).collect();
    
    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await??);
    }
    
    Ok(results)
}
```

## Best Practices

### Transform Pipeline Design

```rust
// Separate training and validation pipelines
fn create_training_pipeline() -> Result<Compose> {
    Ok(TransformBuilder::new()
        .resize((256, 256))
        .random_resized_crop((224, 224))
        .random_horizontal_flip(0.5)
        .add(ColorJitter::new().brightness(0.4).contrast(0.4))
        .add(RandomErasing::new(0.25))
        .imagenet_normalize()
        .build())
}

fn create_validation_pipeline() -> Result<Compose> {
    Ok(TransformBuilder::new()
        .resize((256, 256))
        .center_crop((224, 224))
        .imagenet_normalize()
        .build())
}
```

### Hardware-Aware Design

```rust
// Automatically select optimal transforms
fn create_optimal_pipeline() -> Result<Box<dyn Transform>> {
    let hardware = HardwareContext::auto_detect()?;
    
    if hardware.cuda_available() && hardware.supports_mixed_precision() {
        // GPU + mixed precision
        Ok(Box::new(create_gpu_f16_pipeline()?))
    } else if hardware.cuda_available() {
        // GPU only
        Ok(Box::new(create_gpu_pipeline()?))
    } else {
        // CPU optimized
        Ok(Box::new(create_cpu_pipeline()?))
    }
}
```

### Reproducibility

```rust
use torsh_vision::transforms::set_random_seed;

// Set seed for reproducible results
set_random_seed(42);

// Use deterministic transforms for validation
let deterministic_transforms = TransformBuilder::new()
    .resize((224, 224))
    .center_crop((224, 224))
    .imagenet_normalize()
    .build();
```

### Error Handling Strategy

```rust
// Comprehensive error handling with recovery
fn robust_image_processing(image_path: &str) -> Result<Tensor> {
    // Load with error recovery
    let image = VisionIO::load_image(image_path)
        .or_else(|_| {
            // Try different formats
            VisionIO::load_image_with_fallback(image_path)
        })?;
    
    // Validate input
    if image.shape().dims().len() < 2 {
        return Err(VisionError::InvalidShape(
            "Image must be at least 2D".to_string()
        ));
    }
    
    // Apply transforms with fallback
    let transforms = create_optimal_pipeline()
        .unwrap_or_else(|_| Box::new(create_basic_pipeline()));
    
    transforms.forward(&image)
        .or_else(|_| {
            // Fallback to minimal processing
            let basic = Resize::new((224, 224));
            basic.forward(&image)
        })
}
```

### Performance Monitoring

```rust
use std::time::Instant;

// Monitor transform performance
fn monitor_transform_performance(
    transform: &dyn Transform,
    images: &[Tensor]
) -> Result<(Vec<Tensor>, f64)> {
    let start = Instant::now();
    
    let mut results = Vec::new();
    for image in images {
        results.push(transform.forward(image)?);
    }
    
    let duration = start.elapsed().as_secs_f64();
    let avg_time = duration / images.len() as f64;
    
    println!("Processed {} images in {:.2}s (avg: {:.2}ms per image)", 
             images.len(), duration, avg_time * 1000.0);
    
    Ok((results, avg_time))
}
```

## Conclusion

The ToRSh Vision transform system provides:

- **Comprehensive functionality**: Complete set of transforms for computer vision
- **Hardware optimization**: Automatic GPU acceleration and mixed precision
- **Type safety**: Compile-time guarantees and robust error handling
- **Performance**: Optimized implementations with memory efficiency
- **Extensibility**: Easy custom transform development
- **Production readiness**: Advanced error handling and monitoring

For more examples and advanced usage, refer to the [examples module](./src/examples.rs) and the [vision guide](./VISION_GUIDE.md).