# ToRSh Vision - Dataset Documentation and Examples

This document provides comprehensive documentation for the dataset system in ToRSh Vision, covering built-in datasets, custom dataset creation, optimized data loading, and usage patterns.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Trait System](#dataset-trait-system)
3. [Built-in Datasets](#built-in-datasets)
4. [Optimized Data Loading](#optimized-data-loading)
5. [Custom Datasets](#custom-datasets)
6. [Memory Management](#memory-management)
7. [Performance Optimization](#performance-optimization)
8. [Data Augmentation Integration](#data-augmentation-integration)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Overview

ToRSh Vision provides a comprehensive dataset system that supports:

- **Built-in datasets**: ImageFolder, CIFAR-10/100, MNIST, Pascal VOC, MS COCO
- **Optimized loading**: Lazy loading, caching, prefetching, memory mapping
- **Memory efficiency**: Configurable memory limits and intelligent cache management
- **Parallel processing**: Multi-threaded data loading with worker threads
- **Type safety**: Compile-time guarantees and robust error handling

## Dataset Trait System

### Core Dataset Trait

All datasets implement the `Dataset` trait:

```rust
pub trait Dataset: Send + Sync {
    type Item;
    
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn get(&self, index: usize) -> Result<Self::Item>;
}
```

### Specialized Traits

```rust
// For image classification datasets
pub trait ImageDataset: Dataset<Item = (Tensor, i64)> {
    fn classes(&self) -> &[String];
    fn num_classes(&self) -> usize { self.classes().len() }
}

// For object detection datasets
pub trait DetectionDataset: Dataset<Item = (Tensor, Vec<Detection>)> {
    fn categories(&self) -> &[String];
    fn category_id_to_name(&self, id: usize) -> Option<&str>;
}

// For optimized datasets with caching
pub trait OptimizedDataset: Dataset {
    fn cache_stats(&self) -> CacheStats;
    fn memory_usage(&self) -> MemoryUsage;
}
```

## Built-in Datasets

### ImageFolder Dataset

For custom image classification datasets organized in folders:

```rust
use torsh_vision::datasets::*;

// Basic ImageFolder
let dataset = ImageFolder::new("path/to/imagenet/train")?;
println!("Classes: {:?}", dataset.classes());
println!("Number of samples: {}", dataset.len());

// With transforms
let transforms = TransformBuilder::new()
    .resize((224, 224))
    .imagenet_normalize()
    .build();

let dataset_with_transforms = ImageFolder::new("path/to/data")?
    .with_transforms(transforms);

// Access samples
let (image, label) = dataset.get(0)?;
println!("Image shape: {:?}, Label: {}", image.shape(), label);
```

**Directory Structure:**
```
dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
└── ...
```

### CIFAR-10/100 Datasets

```rust
// CIFAR-10
let cifar10_train = CIFAR10Dataset::new("path/to/cifar10", true)?;  // train=true
let cifar10_test = CIFAR10Dataset::new("path/to/cifar10", false)?; // train=false

println!("CIFAR-10 classes: {:?}", cifar10_train.classes());
println!("Training samples: {}", cifar10_train.len());

// CIFAR-100
let cifar100_train = CIFAR100Dataset::new("path/to/cifar100", true)?;
println!("CIFAR-100 fine classes: {}", cifar100_train.num_fine_classes());
println!("CIFAR-100 coarse classes: {}", cifar100_train.num_coarse_classes());

// Access sample with both fine and coarse labels
let (image, fine_label, coarse_label) = cifar100_train.get_with_coarse_label(0)?;
```

**CIFAR Data Structure:**
```
cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
├── batches.meta
└── readme.html
```

### MNIST Dataset

```rust
// MNIST
let mnist_train = MNISTDataset::new("path/to/mnist", true)?;   // training set
let mnist_test = MNISTDataset::new("path/to/mnist", false)?;  // test set

println!("Training samples: {}", mnist_train.len());
println!("Test samples: {}", mnist_test.len());

// Access sample
let (image, label) = mnist_train.get(0)?;
println!("Image shape: {:?} (28x28 grayscale)", image.shape());
```

**MNIST Data Structure:**
```
mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

### Pascal VOC Dataset (Object Detection)

```rust
// Pascal VOC 2012
let voc_train = VOCDataset::new("path/to/voc", "2012", "train")?;
let voc_val = VOCDataset::new("path/to/voc", "2012", "val")?;

println!("VOC categories: {:?}", voc_train.categories());

// Access detection sample
let (image, detections) = voc_train.get(0)?;
for detection in &detections {
    println!("Object: {} at bbox {:?} (confidence: {:.2})", 
             detection.category, detection.bbox, detection.confidence);
}
```

**VOC Data Structure:**
```
VOCdevkit/
└── VOC2012/
    ├── Annotations/
    │   ├── 2007_000001.xml
    │   └── ...
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt
    │       ├── val.txt
    │       └── ...
    └── JPEGImages/
        ├── 2007_000001.jpg
        └── ...
```

### MS COCO Dataset (Object Detection)

```rust
// MS COCO 2017
let coco_train = COCODataset::new("path/to/coco", "train2017")?;
let coco_val = COCODataset::new("path/to/coco", "val2017")?;

println!("COCO categories: {}", coco_train.categories().len());

// Access sample with multiple objects
let (image, detections) = coco_train.get(0)?;
println!("Found {} objects in image", detections.len());

// Get category info
for detection in &detections {
    let category_name = coco_train.category_id_to_name(detection.category_id)
        .unwrap_or("unknown");
    println!("Detected: {}", category_name);
}
```

**COCO Data Structure:**
```
coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017/
│   ├── 000000000009.jpg
│   └── ...
└── val2017/
    ├── 000000000139.jpg
    └── ...
```

## Optimized Data Loading

### OptimizedImageDataset

Memory-efficient ImageFolder replacement with caching:

```rust
use torsh_vision::datasets::optimized::*;

// Create optimized dataset configuration
let config = DatasetConfig {
    cache_size_mb: 2048,        // 2GB cache
    prefetch_size: 128,         // Prefetch 128 images
    max_workers: 8,             // 8 worker threads
    enable_validation: true,    // Validate images on load
    memory_mapping: false,      // Use regular loading
    compression: false,         // No compression
};

// Create optimized dataset
let optimized_dataset = OptimizedImageDataset::new("path/to/data", config)?;

// Monitor performance
let stats = optimized_dataset.cache_stats();
println!("Cache hit rate: {:.2}%", stats.hit_rate * 100.0);
println!("Memory usage: {:.1} MB", stats.memory_usage_mb);
```

### Memory-Mapped Loading

For very large datasets that exceed system RAM:

```rust
// Memory-mapped dataset loader
let memory_mapped = MemoryMappedLoader::new("large_dataset.bin")?;

// Configure with memory mapping
let config = DatasetConfig {
    cache_size_mb: 512,         // Smaller cache for memory-mapped data
    memory_mapping: true,       // Enable memory mapping
    ..Default::default()
};

let large_dataset = OptimizedImageDataset::with_memory_mapping("huge_dataset", config)?;
```

### Lazy Loading with LRU Cache

```rust
// Lazy dataset with intelligent caching
let lazy_dataset = LazyDataset::new(base_dataset, 1000)?;  // Cache 1000 images

// Configure cache behavior
let cache_config = CacheConfig {
    max_size: 2048,             // 2GB max cache size
    eviction_policy: EvictionPolicy::LRU,
    preload_factor: 0.1,        // Preload 10% of dataset
    background_loading: true,    // Load in background
};

lazy_dataset.configure_cache(cache_config)?;
```

### Asynchronous Prefetching

```rust
use torsh_vision::datasets::ImagePrefetcher;

// Create prefetcher for background loading
let prefetcher = ImagePrefetcher::new(4)?;  // 4 worker threads

// Start prefetching
for index in 0..100 {
    prefetcher.prefetch_image(&dataset, index)?;
}

// Images are loaded in background
// Access is fast when image is already cached
let (image, label) = dataset.get(50)?;  // Fast access if prefetched
```

## Custom Datasets

### Basic Custom Dataset

```rust
use torsh_vision::datasets::{Dataset, ImageDataset};
use std::path::PathBuf;

pub struct CustomImageDataset {
    image_paths: Vec<PathBuf>,
    labels: Vec<i64>,
    classes: Vec<String>,
    transforms: Option<Box<dyn Transform>>,
}

impl CustomImageDataset {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let mut image_paths = Vec::new();
        let mut labels = Vec::new();
        let mut classes = Vec::new();
        
        // Scan directory structure
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let class_name = entry.file_name().to_string_lossy().to_string();
                let class_id = classes.len() as i64;
                classes.push(class_name);
                
                // Scan class directory
                for img_entry in std::fs::read_dir(entry.path())? {
                    let img_path = img_entry?.path();
                    if Self::is_image_file(&img_path) {
                        image_paths.push(img_path);
                        labels.push(class_id);
                    }
                }
            }
        }
        
        Ok(Self {
            image_paths,
            labels,
            classes,
            transforms: None,
        })
    }
    
    pub fn with_transforms(mut self, transforms: Box<dyn Transform>) -> Self {
        self.transforms = Some(transforms);
        self
    }
    
    fn is_image_file(path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            matches!(ext.to_str().unwrap_or("").to_lowercase().as_str(),
                    "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp")
        } else {
            false
        }
    }
}

impl Dataset for CustomImageDataset {
    type Item = (Tensor, i64);
    
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(VisionError::InvalidArgument(
                format!("Index {} out of bounds", index)
            ));
        }
        
        // Load image
        let mut image = VisionIO::load_image(&self.image_paths[index])?;
        
        // Apply transforms
        if let Some(ref transforms) = self.transforms {
            image = transforms.forward(&image)?;
        }
        
        let label = self.labels[index];
        Ok((image, label))
    }
}

impl ImageDataset for CustomImageDataset {
    fn classes(&self) -> &[String] {
        &self.classes
    }
}
```

### Custom Detection Dataset

```rust
use torsh_vision::datasets::{Dataset, DetectionDataset, Detection};

pub struct CustomDetectionDataset {
    image_paths: Vec<PathBuf>,
    annotations: Vec<Vec<Detection>>,
    categories: Vec<String>,
}

impl CustomDetectionDataset {
    pub fn from_json<P: AsRef<Path>>(annotation_file: P, image_dir: P) -> Result<Self> {
        let file = std::fs::File::open(annotation_file)?;
        let json: serde_json::Value = serde_json::from_reader(file)?;
        
        let mut dataset = Self {
            image_paths: Vec::new(),
            annotations: Vec::new(),
            categories: Vec::new(),
        };
        
        // Parse categories
        if let Some(categories) = json["categories"].as_array() {
            for category in categories {
                if let Some(name) = category["name"].as_str() {
                    dataset.categories.push(name.to_string());
                }
            }
        }
        
        // Parse images and annotations
        if let Some(images) = json["images"].as_array() {
            for image in images {
                if let Some(filename) = image["file_name"].as_str() {
                    let image_path = Path::new(image_dir.as_ref()).join(filename);
                    dataset.image_paths.push(image_path);
                    
                    // Find annotations for this image
                    let image_id = image["id"].as_u64().unwrap_or(0);
                    let mut image_annotations = Vec::new();
                    
                    if let Some(annotations) = json["annotations"].as_array() {
                        for annotation in annotations {
                            if annotation["image_id"].as_u64().unwrap_or(0) == image_id {
                                // Parse bounding box
                                if let Some(bbox) = annotation["bbox"].as_array() {
                                    let detection = Detection {
                                        bbox: [
                                            bbox[0].as_f64().unwrap_or(0.0) as f32,
                                            bbox[1].as_f64().unwrap_or(0.0) as f32,
                                            bbox[2].as_f64().unwrap_or(0.0) as f32,
                                            bbox[3].as_f64().unwrap_or(0.0) as f32,
                                        ],
                                        category_id: annotation["category_id"].as_u64().unwrap_or(0) as usize,
                                        confidence: 1.0,  // Ground truth
                                        area: annotation["area"].as_f64().unwrap_or(0.0) as f32,
                                    };
                                    image_annotations.push(detection);
                                }
                            }
                        }
                    }
                    
                    dataset.annotations.push(image_annotations);
                }
            }
        }
        
        Ok(dataset)
    }
}

impl Dataset for CustomDetectionDataset {
    type Item = (Tensor, Vec<Detection>);
    
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let image = VisionIO::load_image(&self.image_paths[index])?;
        let annotations = self.annotations[index].clone();
        Ok((image, annotations))
    }
}

impl DetectionDataset for CustomDetectionDataset {
    fn categories(&self) -> &[String] {
        &self.categories
    }
    
    fn category_id_to_name(&self, id: usize) -> Option<&str> {
        self.categories.get(id).map(|s| s.as_str())
    }
}
```

### Streaming Dataset

For very large datasets or real-time data:

```rust
use std::sync::mpsc::{Receiver, channel};
use std::thread;

pub struct StreamingDataset {
    receiver: Receiver<(Tensor, i64)>,
    buffer_size: usize,
}

impl StreamingDataset {
    pub fn new<F>(data_generator: F, buffer_size: usize) -> Self 
    where 
        F: Fn() -> Result<(Tensor, i64)> + Send + 'static 
    {
        let (sender, receiver) = channel();
        
        // Background thread for data generation
        thread::spawn(move || {
            loop {
                match data_generator() {
                    Ok(sample) => {
                        if sender.send(sample).is_err() {
                            break;  // Receiver dropped
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        
        Self { receiver, buffer_size }
    }
    
    pub fn next(&self) -> Result<(Tensor, i64)> {
        self.receiver.recv()
            .map_err(|_| VisionError::Other("Stream ended".into()))
    }
}
```

## Memory Management

### Cache Configuration

```rust
use torsh_vision::datasets::CacheConfig;

// Configure LRU cache
let cache_config = CacheConfig {
    max_size_mb: 4096,          // 4GB cache
    eviction_policy: EvictionPolicy::LRU,
    ttl_seconds: 3600,          // 1 hour TTL
    compression: true,          // Compress cached images
    background_loading: true,   // Load in background
    prefetch_factor: 0.2,       // Prefetch 20% of dataset
};

// Apply to dataset
let cached_dataset = CachedDataset::new(base_dataset, cache_config)?;
```

### Memory Monitoring

```rust
// Monitor memory usage
let memory_usage = dataset.memory_usage();
println!("Cache memory: {:.1} MB", memory_usage.cache_mb);
println!("Buffer memory: {:.1} MB", memory_usage.buffer_mb);
println!("Total memory: {:.1} MB", memory_usage.total_mb);

// Set memory limits
if memory_usage.total_mb > 8192.0 {  // 8GB limit
    dataset.clear_cache()?;
    dataset.resize_cache(4096)?;  // Reduce to 4GB
}
```

### Batch Memory Optimization

```rust
use torsh_vision::memory::BatchMemoryOptimizer;

// Optimize batch size for available memory
let optimizer = BatchMemoryOptimizer::new(8192)?;  // 8GB available

let optimal_batch_size = optimizer.calculate_optimal_batch_size(
    &[3, 224, 224],    // Image shape
    0.8                // 80% memory utilization
)?;

println!("Optimal batch size: {}", optimal_batch_size);
```

## Performance Optimization

### Parallel Data Loading

```rust
use torsh_vision::datasets::ParallelDataLoader;

// Create parallel data loader
let data_loader = ParallelDataLoader::new(dataset, 8)?;  // 8 worker threads

// Configure prefetching
data_loader.set_prefetch_factor(2.0)?;  // Prefetch 2x batch size
data_loader.set_pin_memory(true)?;      // Pin memory for GPU transfer

// Load batches in parallel
for batch in data_loader.iter(32)? {    // Batch size 32
    // Process batch
    let (images, labels) = batch?;
    println!("Loaded batch: {} images", images.shape()[0]);
}
```

### GPU-Optimized Loading

```rust
// Pin memory for faster GPU transfer
let gpu_dataset = GpuOptimizedDataset::new(dataset)?
    .with_pin_memory(true)
    .with_prefetch_to_gpu(true);

// Automatic GPU transfer
let (gpu_image, label) = gpu_dataset.get(0)?;  // Already on GPU
assert_eq!(gpu_image.device(), Device::Cuda(0));
```

### Compression and Decompression

```rust
// Use compression for cached data
let compressed_config = DatasetConfig {
    compression: true,
    compression_level: 6,       // Balance speed/size
    compression_format: CompressionFormat::LZ4,
    ..Default::default()
};

let compressed_dataset = OptimizedImageDataset::new(path, compressed_config)?;

// Monitor compression ratio
let stats = compressed_dataset.compression_stats();
println!("Compression ratio: {:.2}x", stats.compression_ratio);
println!("Space saved: {:.1} MB", stats.space_saved_mb);
```

## Data Augmentation Integration

### Dataset with Transforms

```rust
// Apply transforms at dataset level
let augmented_dataset = dataset.with_transforms(
    TransformBuilder::new()
        .resize((256, 256))
        .random_horizontal_flip(0.5)
        .random_rotation((-15.0, 15.0))
        .color_jitter(0.4, 0.4, 0.4, 0.1)
        .imagenet_normalize()
        .build()
);

// Transforms are applied automatically on each get()
let (augmented_image, label) = augmented_dataset.get(0)?;
```

### Conditional Transforms

```rust
pub struct ConditionalDataset<D: Dataset> {
    dataset: D,
    train_transforms: Box<dyn Transform>,
    val_transforms: Box<dyn Transform>,
    is_training: bool,
}

impl<D: Dataset> ConditionalDataset<D> {
    pub fn new(
        dataset: D,
        train_transforms: Box<dyn Transform>,
        val_transforms: Box<dyn Transform>,
    ) -> Self {
        Self {
            dataset,
            train_transforms,
            val_transforms,
            is_training: true,
        }
    }
    
    pub fn train(&mut self) {
        self.is_training = true;
    }
    
    pub fn eval(&mut self) {
        self.is_training = false;
    }
}

impl<D: Dataset<Item = (Tensor, i64)>> Dataset for ConditionalDataset<D> {
    type Item = (Tensor, i64);
    
    fn len(&self) -> usize {
        self.dataset.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let (mut image, label) = self.dataset.get(index)?;
        
        let transforms = if self.is_training {
            &self.train_transforms
        } else {
            &self.val_transforms
        };
        
        image = transforms.forward(&image)?;
        Ok((image, label))
    }
}
```

### Mix-up at Dataset Level

```rust
pub struct MixUpDataset<D: ImageDataset> {
    dataset: D,
    mixup: MixUp,
    probability: f32,
}

impl<D: ImageDataset> MixUpDataset<D> {
    pub fn new(dataset: D, alpha: f32, probability: f32) -> Self {
        Self {
            dataset,
            mixup: MixUp::new(alpha),
            probability,
        }
    }
}

impl<D: ImageDataset> Dataset for MixUpDataset<D> {
    type Item = (Tensor, Tensor);  // Image, mixed labels
    
    fn len(&self) -> usize {
        self.dataset.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let (image1, label1) = self.dataset.get(index)?;
        
        if rand::random::<f32>() < self.probability {
            // Apply MixUp
            let index2 = rand::random::<usize>() % self.len();
            let (image2, label2) = self.dataset.get(index2)?;
            
            let (mixed_image, mixed_labels) = self.mixup.apply_pair(
                &image1, &image2,
                label1, label2,
                self.dataset.num_classes()
            )?;
            
            Ok((mixed_image, mixed_labels))
        } else {
            // One-hot encode label
            let mut labels = Tensor::zeros([self.dataset.num_classes()], DType::F32, Device::Cpu);
            labels.set(&[label1 as usize], Scalar::F32(1.0))?;
            
            Ok((image1, labels))
        }
    }
}
```

## Best Practices

### Dataset Organization

```rust
// Organize datasets by purpose
struct DatasetManager {
    train_dataset: Box<dyn ImageDataset>,
    val_dataset: Box<dyn ImageDataset>,
    test_dataset: Option<Box<dyn ImageDataset>>,
}

impl DatasetManager {
    pub fn new_imagenet(root: &str) -> Result<Self> {
        let train_transforms = TransformBuilder::new()
            .resize((256, 256))
            .random_resized_crop((224, 224))
            .random_horizontal_flip(0.5)
            .imagenet_normalize()
            .build();
            
        let val_transforms = TransformBuilder::new()
            .resize((256, 256))
            .center_crop((224, 224))
            .imagenet_normalize()
            .build();
        
        Ok(Self {
            train_dataset: Box::new(
                ImageFolder::new(&format!("{}/train", root))?
                    .with_transforms(train_transforms)
            ),
            val_dataset: Box::new(
                ImageFolder::new(&format!("{}/val", root))?
                    .with_transforms(val_transforms)
            ),
            test_dataset: None,
        })
    }
}
```

### Error Handling

```rust
// Robust dataset loading with error recovery
fn load_dataset_with_fallback(primary_path: &str, backup_path: &str) -> Result<Box<dyn ImageDataset>> {
    // Try primary dataset
    if let Ok(dataset) = ImageFolder::new(primary_path) {
        return Ok(Box::new(dataset));
    }
    
    // Fallback to backup
    if let Ok(dataset) = ImageFolder::new(backup_path) {
        println!("Warning: Using fallback dataset at {}", backup_path);
        return Ok(Box::new(dataset));
    }
    
    // Create empty dataset as last resort
    println!("Warning: Creating empty dataset");
    Ok(Box::new(EmptyDataset::new()))
}
```

### Memory-Aware Loading

```rust
// Automatically configure dataset based on available memory
fn create_memory_aware_dataset(path: &str, available_memory_gb: f32) -> Result<Box<dyn ImageDataset>> {
    let config = if available_memory_gb >= 16.0 {
        // High memory: aggressive caching
        DatasetConfig {
            cache_size_mb: 8192,
            prefetch_size: 256,
            max_workers: 16,
            compression: false,
            ..Default::default()
        }
    } else if available_memory_gb >= 8.0 {
        // Medium memory: moderate caching
        DatasetConfig {
            cache_size_mb: 2048,
            prefetch_size: 128,
            max_workers: 8,
            compression: true,
            ..Default::default()
        }
    } else {
        // Low memory: minimal caching
        DatasetConfig {
            cache_size_mb: 512,
            prefetch_size: 32,
            max_workers: 4,
            compression: true,
            memory_mapping: true,
            ..Default::default()
        }
    };
    
    Ok(Box::new(OptimizedImageDataset::new(path, config)?))
}
```

### Validation and Testing

```rust
// Validate dataset integrity
fn validate_dataset(dataset: &dyn Dataset<Item = (Tensor, i64)>) -> Result<()> {
    println!("Validating dataset with {} samples...", dataset.len());
    
    let mut errors = 0;
    let sample_size = (dataset.len() / 10).max(100);  // 10% or 100 samples
    
    for i in 0..sample_size {
        match dataset.get(i) {
            Ok((image, label)) => {
                // Validate image
                if image.shape().dims().len() < 2 {
                    println!("Warning: Invalid image shape at index {}: {:?}", i, image.shape());
                    errors += 1;
                }
                
                // Validate label
                if label < 0 {
                    println!("Warning: Invalid label at index {}: {}", i, label);
                    errors += 1;
                }
            }
            Err(e) => {
                println!("Error loading sample {}: {}", i, e);
                errors += 1;
            }
        }
        
        if i % (sample_size / 10) == 0 {
            println!("Validated {}/{} samples", i, sample_size);
        }
    }
    
    if errors > 0 {
        println!("Found {} errors in dataset validation", errors);
    } else {
        println!("Dataset validation passed");
    }
    
    Ok(())
}
```

## Examples

### Complete Training Pipeline

```rust
use torsh_vision::prelude::*;

fn create_training_pipeline() -> Result<()> {
    // Create optimized datasets
    let train_config = DatasetConfig {
        cache_size_mb: 4096,
        prefetch_size: 128,
        max_workers: 8,
        enable_validation: true,
        ..Default::default()
    };
    
    let train_dataset = OptimizedImageDataset::new("data/train", train_config)?;
    let val_dataset = ImageFolder::new("data/val")?;
    
    // Create data loaders
    let train_loader = ParallelDataLoader::new(train_dataset, 8)?
        .with_batch_size(64)
        .with_shuffle(true)
        .with_pin_memory(true);
    
    let val_loader = ParallelDataLoader::new(val_dataset, 4)?
        .with_batch_size(64)
        .with_shuffle(false);
    
    // Training loop
    for epoch in 0..100 {
        // Training
        for (batch_idx, batch) in train_loader.iter()?.enumerate() {
            let (images, labels) = batch?;
            
            // Forward pass, backward pass, optimizer step
            // ... training code ...
            
            if batch_idx % 100 == 0 {
                println!("Epoch {}, Batch {}", epoch, batch_idx);
            }
        }
        
        // Validation
        let mut total_correct = 0;
        let mut total_samples = 0;
        
        for batch in val_loader.iter()? {
            let (images, labels) = batch?;
            
            // Validation forward pass
            // ... validation code ...
            
            total_samples += images.shape()[0];
        }
        
        let accuracy = total_correct as f32 / total_samples as f32;
        println!("Epoch {} - Validation Accuracy: {:.2}%", epoch, accuracy * 100.0);
    }
    
    Ok(())
}
```

### Object Detection Training

```rust
fn create_detection_pipeline() -> Result<()> {
    // Load COCO dataset
    let train_dataset = COCODataset::new("data/coco", "train2017")?
        .with_transforms(
            TransformBuilder::new()
                .resize((640, 640))
                .random_horizontal_flip(0.5)
                .normalize(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0])
                .build()
        );
    
    let val_dataset = COCODataset::new("data/coco", "val2017")?
        .with_transforms(
            TransformBuilder::new()
                .resize((640, 640))
                .normalize(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0])
                .build()
        );
    
    // Apply Mosaic augmentation for detection
    let mosaic_dataset = MosaicDataset::new(train_dataset, 0.5)?;  // 50% probability
    
    // Training with detection-specific data loading
    let detection_loader = DetectionDataLoader::new(mosaic_dataset)?
        .with_batch_size(16)
        .with_collate_fn(detection_collate_fn);
    
    for batch in detection_loader.iter()? {
        let (images, targets) = batch?;
        
        // Detection training step
        // ... YOLO/RetinaNet training code ...
    }
    
    Ok(())
}

fn detection_collate_fn(batch: Vec<(Tensor, Vec<Detection>)>) -> Result<(Tensor, Vec<Vec<Detection>>)> {
    let mut images = Vec::new();
    let mut targets = Vec::new();
    
    for (image, detections) in batch {
        images.push(image);
        targets.push(detections);
    }
    
    let batched_images = Tensor::stack(&images, 0)?;
    Ok((batched_images, targets))
}
```

### Custom Dataset with Caching

```rust
fn create_custom_cached_dataset() -> Result<()> {
    // Custom dataset with advanced caching
    let custom_dataset = CustomImageDataset::new("custom_data")?
        .with_transforms(
            TransformBuilder::new()
                .resize((224, 224))
                .random_augmentation()
                .build()
        );
    
    // Wrap with optimized caching
    let cached_dataset = CachedDataset::new(
        custom_dataset,
        CacheConfig {
            max_size_mb: 2048,
            eviction_policy: EvictionPolicy::LRU,
            compression: true,
            background_loading: true,
            prefetch_factor: 0.3,
            ..Default::default()
        }
    )?;
    
    // Monitor cache performance
    let mut cache_monitor = CacheMonitor::new();
    
    for i in 0..1000 {
        let start = std::time::Instant::now();
        let (image, label) = cached_dataset.get(i % cached_dataset.len())?;
        let load_time = start.elapsed();
        
        cache_monitor.record_access(i, load_time);
        
        if i % 100 == 0 {
            let stats = cached_dataset.cache_stats();
            println!("Cache hit rate: {:.2}%, Avg load time: {:.2}ms",
                     stats.hit_rate * 100.0,
                     cache_monitor.average_load_time().as_millis());
        }
    }
    
    Ok(())
}
```

## Conclusion

The ToRSh Vision dataset system provides:

- **Comprehensive built-in datasets**: Ready-to-use implementations for common datasets
- **Memory optimization**: Advanced caching, prefetching, and memory mapping
- **Performance**: Parallel loading, GPU optimization, and intelligent batching
- **Flexibility**: Easy custom dataset creation and extension
- **Production readiness**: Robust error handling and monitoring

For more advanced usage patterns and examples, refer to the [examples module](./src/examples.rs) and the [vision guide](./VISION_GUIDE.md).