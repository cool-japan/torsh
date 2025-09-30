# ToRSh Vision - Comprehensive Vision Guide

ToRSh Vision is a production-ready computer vision framework built in pure Rust, providing PyTorch-compatible APIs with superior performance through Rust's zero-cost abstractions and memory safety.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Image Processing](#image-processing)
4. [Data Transforms](#data-transforms)
5. [Datasets and Data Loading](#datasets-and-data-loading)
6. [Models and Architectures](#models-and-architectures)
7. [Hardware Acceleration](#hardware-acceleration)
8. [Interactive Visualization](#interactive-visualization)
9. [3D Visualization](#3d-visualization)
10. [Video Processing](#video-processing)
11. [Memory Management](#memory-management)
12. [Advanced Features](#advanced-features)
13. [Performance Optimization](#performance-optimization)
14. [Best Practices](#best-practices)

## Quick Start

```rust
use torsh_vision::prelude::*;

// Create a sample image tensor
let image = Tensor::randn([3, 224, 224], DType::F32, Device::Cpu);

// Apply transforms
let transforms = TransformBuilder::new()
    .resize((256, 256))
    .center_crop((224, 224))
    .imagenet_normalize()
    .build();

let processed_image = transforms.forward(&image)?;
println!("Processed image shape: {:?}", processed_image.shape());
```

## Core Concepts

### Tensors
ToRSh Vision uses tensors from the torsh-tensor crate as the fundamental data structure for images and computations:

```rust
// Create image tensors
let rgb_image = Tensor::zeros([3, 512, 512], DType::F32, Device::Cpu);  // RGB
let grayscale = Tensor::zeros([1, 512, 512], DType::F32, Device::Cpu);  // Grayscale
let batch = Tensor::zeros([32, 3, 224, 224], DType::F32, Device::Cpu);  // Batch
```

### Error Handling
ToRSh Vision uses comprehensive error handling with the `VisionError` enum:

```rust
use torsh_vision::{Result, VisionError};

fn process_image() -> Result<Tensor> {
    let image = load_image("path/to/image.jpg")?;
    let processed = apply_transforms(&image)?;
    Ok(processed)
}
```

### Device Support
ToRSh Vision automatically detects and utilizes available hardware:

```rust
let hardware = HardwareContext::auto_detect()?;
println!("CUDA available: {}", hardware.cuda_available());
println!("Mixed precision: {}", hardware.supports_mixed_precision());
```

## Image Processing

### Basic I/O Operations

```rust
use torsh_vision::io::VisionIO;

// Load images
let image = VisionIO::load_image("image.jpg")?;

// Save images
VisionIO::save_image(&processed_image, "output.png")?;

// Batch loading
let images = VisionIO::load_images_from_directory("./images/")?;

// Format conversion
VisionIO::convert_image_format("input.jpg", "output.png")?;
```

### Advanced Image Operations

```rust
use torsh_vision::ops::*;

// Edge detection
let edges = sobel_edge_detection(&image)?;
let canny_edges = canny_edge_detection(&image, 50.0, 150.0)?;

// Filtering
let blurred = gaussian_blur(&image, 3, 1.0)?;
let sharpened = sharpen_filter(&image, 1.5)?;

// Morphological operations
let eroded = erosion(&image, 3)?;
let dilated = dilation(&image, 3)?;

// Feature detection
let corners = harris_corner_detection(&image, 0.04, 0.01)?;
let keypoints = fast_corner_detection(&image, 20)?;
```

## Data Transforms

### Basic Transforms

```rust
use torsh_vision::transforms::*;

// Individual transforms
let resize = Resize::new((224, 224));
let crop = CenterCrop::new((224, 224));
let flip = RandomHorizontalFlip::new(0.5);
let normalize = Normalize::new(
    vec![0.485, 0.456, 0.406],  // ImageNet mean
    vec![0.229, 0.224, 0.225]   // ImageNet std
);

// Apply transform
let transformed = resize.forward(&image)?;
```

### Transform Composition

```rust
// Using TransformBuilder
let train_transforms = TransformBuilder::new()
    .resize((256, 256))
    .random_resized_crop((224, 224))
    .random_horizontal_flip(0.5)
    .add(ColorJitter::new().brightness(0.4).contrast(0.4))
    .imagenet_normalize()
    .build();

// Using Compose
let composed = Compose::new(vec![
    Box::new(Resize::new((256, 256))),
    Box::new(CenterCrop::new((224, 224))),
    Box::new(ToTensor::new()),
]);
```

### Advanced Augmentation

```rust
// AutoAugment
let auto_augment = AutoAugment::new();
let augmented = auto_augment.forward(&image)?;

// RandAugment
let rand_augment = RandAugment::new(2, 5.0);  // 2 ops, magnitude 5
let augmented = rand_augment.forward(&image)?;

// MixUp
let mut mixup = MixUp::new(1.0);
let (mixed_image, mixed_labels) = mixup.apply_pair(&img1, &img2, 0, 1, 10)?;

// CutMix
let mut cutmix = CutMix::new(1.0);
let (cut_image, cut_labels) = cutmix.apply_pair(&img1, &img2, 0, 1, 10)?;
```

### Unified Transform API

```rust
use torsh_vision::unified_transforms::*;

// Hardware-aware transforms
let context = TransformContext::new(HardwareContext::auto_detect()?);
let unified_resize = UnifiedResize::new((224, 224));

// Automatic GPU/CPU selection
let processed = unified_resize.apply(&image, &context)?;

// Mixed precision support
if context.hardware().supports_mixed_precision() {
    let processed_f16 = unified_resize.apply_gpu_f16(&image, &context)?;
}
```

## Datasets and Data Loading

### Built-in Datasets

```rust
use torsh_vision::datasets::*;

// ImageFolder dataset
let dataset = ImageFolder::new("path/to/imagenet/train")?;
println!("Classes: {:?}", dataset.classes());

// CIFAR-10/100
let cifar10 = CIFAR10Dataset::new("path/to/cifar10", true)?;  // train=true
let cifar100 = CIFAR100Dataset::new("path/to/cifar100", false)?;  // train=false

// MNIST
let mnist = MNISTDataset::new("path/to/mnist", true)?;

// Pascal VOC (Object Detection)
let voc = VOCDataset::new("path/to/voc", "2012", "train")?;

// MS COCO (Object Detection)
let coco = COCODataset::new("path/to/coco", "train2017")?;
```

### Optimized Data Loading

```rust
use torsh_vision::datasets::optimized::*;

// Memory-efficient lazy loading
let config = DatasetConfig {
    cache_size_mb: 1024,  // 1GB cache
    prefetch_size: 64,
    max_workers: 8,
    enable_validation: true,
};

let optimized_dataset = OptimizedImageDataset::new("path/to/data", config)?;

// Memory-mapped loading for large datasets
let memory_mapped = MemoryMappedLoader::new("large_dataset.bin")?;
```

### Custom Datasets

```rust
use torsh_vision::datasets::Dataset;

struct CustomDataset {
    image_paths: Vec<String>,
    labels: Vec<i64>,
}

impl Dataset for CustomDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<(Tensor, i64)> {
        let image = VisionIO::load_image(&self.image_paths[index])?;
        let label = self.labels[index];
        Ok((image, label))
    }
}
```

## Models and Architectures

### Classification Models

```rust
use torsh_vision::models::*;

// ResNet family
let resnet18 = ResNet::resnet18(1000)?;
let resnet50 = ResNet::resnet50(1000)?;
let resnet152 = ResNet::resnet152(1000)?;

// VGG family
let vgg16 = VGG::vgg16(1000, false)?;  // without batch norm
let vgg16_bn = VGG::vgg16(1000, true)?;  // with batch norm

// EfficientNet
let efficientnet_b0 = EfficientNet::efficientnet_b0(1000)?;
let efficientnet_b7 = EfficientNet::efficientnet_b7(1000)?;

// Vision Transformer
let vit_base = ViT::vit_base_patch16_224(1000)?;
let vit_large = ViT::vit_large_patch16_224(1000)?;
```

### Object Detection Models

```rust
// YOLO v5
let yolo = YOLOv5::yolo_v5_medium(80)?;  // 80 COCO classes

// RetinaNet
let retinanet = RetinaNet::retina_net_resnet50(80)?;

// SSD
let ssd = SSD::ssd_300(80)?;
```

### Style Transfer and Super-Resolution

```rust
// Neural Style Transfer
let style_transfer = FastStyleTransfer::new()?;
let stylized = style_transfer.forward(&content_image, &style_image)?;

// Super-Resolution
let srcnn = SRCNN::new(3)?;  // 3x upscaling
let espcn = ESPCN::new(4)?;  // 4x upscaling
let edsr = EDSR::new(2, 64)?;  // 2x upscaling, 64 features
```

## Hardware Acceleration

### GPU Acceleration

```rust
use torsh_vision::hardware::*;

// Auto-detect hardware
let hardware = HardwareContext::auto_detect()?;

if hardware.cuda_available() {
    println!("Using GPU acceleration");
    
    // GPU-accelerated transforms
    let gpu_resize = GpuResize::new((224, 224));
    let gpu_normalize = GpuNormalize::new(mean, std);
    
    // Create GPU transform chain
    let gpu_chain = GpuAugmentationChain::new(vec![
        Box::new(gpu_resize),
        Box::new(gpu_normalize),
    ]);
}
```

### Mixed Precision Training

```rust
use torsh_vision::hardware::MixedPrecisionTraining;

if hardware.supports_mixed_precision() {
    let mut mixed_precision = MixedPrecisionTraining::new()?;
    
    // Convert to f16 for memory savings
    let image_f16 = mixed_precision.to_half(&image)?;
    
    // Process with automatic scaling
    let result = mixed_precision.forward_with_scaling(&image_f16, &model)?;
}
```

## Interactive Visualization

### Interactive Image Viewer

```rust
use torsh_vision::interactive::*;

// Create viewer
let mut viewer = InteractiveViewer::new();

// Load image
viewer.load_image(image)?;

// Add annotations
let bbox = Annotation::BoundingBox {
    x: 100.0, y: 150.0, width: 200.0, height: 150.0,
    label: "Object".to_string(),
    color: [255, 0, 0],
    confidence: Some(0.95),
};
viewer.add_annotation(bbox);

// Set up event handlers
viewer.on_event("mouse_click".to_string(), |event| {
    if let ViewerEvent::MouseClick { x, y, .. } = event {
        println!("Clicked at ({}, {})", x, y);
    }
});

// Export annotations
let json_annotations = viewer.export_annotations()?;
```

### Interactive Gallery

```rust
let mut gallery = InteractiveGallery::new();

// Load images from directory
gallery.load_from_directory("./images/")?;

// Navigate
gallery.next_image()?;
gallery.previous_image()?;
gallery.goto_image(5)?;

// Add annotations to current image
let annotation = Annotation::Point {
    x: 250.0, y: 200.0,
    label: "Landmark".to_string(),
    color: [0, 255, 0],
    radius: 5.0,
};
gallery.add_annotation_to_current(annotation)?;
```

### Live Visualization

```rust
let mut live_viz = LiveVisualization::new();

// Real-time processing loop
for frame in video_stream {
    live_viz.add_frame(frame)?;
    
    let fps = live_viz.current_fps();
    println!("Processing at {} FPS", fps);
}
```

## 3D Visualization

### Point Clouds

```rust
use torsh_vision::viz3d::*;

// Create point cloud
let points = vec![
    Point3D::with_color(1.0, 2.0, 3.0, [255, 0, 0]),
    Point3D::with_color(4.0, 5.0, 6.0, [0, 255, 0]),
];
let mut cloud = PointCloud3D::new(points);

// Voxel downsampling
let downsampled = cloud.voxel_downsample(0.1);

// Distance filtering
let center = Point3D::new(0.0, 0.0, 0.0);
let filtered = cloud.filter_by_distance(center, 10.0);

// Convert to/from tensors
let tensor = cloud.to_tensor()?;
let cloud_from_tensor = PointCloud3D::from_tensor(&tensor)?;
```

### 3D Meshes

```rust
// Create primitive meshes
let sphere = Mesh3D::create_sphere(Point3D::new(0.0, 0.0, 0.0), 5.0, 20, 20);
let cube = Mesh3D::create_cube(Point3D::new(0.0, 0.0, 0.0), 2.0);

// Compute normals
let mut custom_mesh = Mesh3D::new(vertices, faces);
custom_mesh.compute_face_normals();
custom_mesh.compute_vertex_normals();
```

### 3D Bounding Boxes

```rust
// 3D object detection
let bbox = BoundingBox3D::new(
    [5.0, 2.0, 1.0],      // center
    [4.0, 2.0, 6.0],      // dimensions
    [0.0, 0.2, 0.0],      // rotation (roll, pitch, yaw)
    "Car".to_string(),
    0.95                   // confidence
).with_color([255, 0, 0]);

// Test point containment
let point = Point3D::new(5.0, 2.0, 1.0);
let inside = bbox.contains_point(point);

// Get corner points
let corners = bbox.corners();
```

### 3D Scenes

```rust
let mut scene = Scene3D::new("Detection Scene".to_string());

// Add components
scene.add_point_cloud(lidar_cloud);
scene.add_mesh(ground_plane);
scene.add_bounding_box(car_detection);

// Scene statistics
println!("Objects: {}", scene.num_objects());
let summary = scene.export_summary();
```

## Video Processing

### Video I/O

```rust
use torsh_vision::video::*;

// Video reader
let mut reader = VideoReader::new("input.mp4")?;
let metadata = reader.metadata();
println!("FPS: {}, Duration: {}s", metadata.fps, metadata.duration);

// Read frames
while let Some(frame) = reader.next_frame()? {
    // Process frame
    let processed = apply_transforms(&frame.tensor)?;
}

// Video writer
let mut writer = VideoWriter::new("output.mp4", 30.0, (1920, 1080))?;
for frame_tensor in processed_frames {
    let frame = VideoFrame::new(frame_tensor, frame_index as f64 / 30.0);
    writer.write_frame(&frame)?;
}
```

### Video Datasets

```rust
let video_dataset = VideoDataset::new("path/to/videos")
    .sequence_length(16)    // 16 frames per sequence
    .overlap(8)             // 8 frames overlap
    .sampling_strategy(TemporalSampling::Uniform)?;

// Get video sequence
let (frames, label) = video_dataset.get(0)?;
println!("Frames shape: {:?}", frames.shape());
```

### Optical Flow

```rust
// Lucas-Kanade optical flow
let flow = lucas_kanade_optical_flow(&frame1, &frame2, 15, 3)?;
let flow_magnitude = optical_flow_magnitude(&flow)?;
```

### Video Models

```rust
// 3D convolution for video classification
let conv3d = Conv3d::new(3, 64, 3, 1, 1)?;
let video_features = conv3d.forward(&video_tensor)?;

// Temporal pooling
let pooled = temporal_pooling(&video_features, TemporalPooling::Average)?;
```

## Memory Management

### Memory Optimization

```rust
use torsh_vision::memory::*;

// Configure global memory settings
let settings = MemorySettings {
    enable_pooling: true,
    max_pool_size: 1000,
    max_batch_memory_mb: 4096,
    enable_profiling: true,
    auto_optimization: true,
};
configure_global_memory(settings);

// Use tensor pool
let mut pool = TensorPool::new(100);
let tensor = pool.get_tensor(&[3, 224, 224])?;
// ... use tensor
pool.return_tensor(tensor)?;
```

### Memory Profiling

```rust
let profiler = MemoryProfiler::new();

// Record allocations
profiler.record_allocation(1024 * 1024, "training", "batch_data");

// Get profiling summary
let summary = profiler.summary();
println!("Peak usage: {} MB", summary.peak_usage_bytes / (1024 * 1024));
```

### Batch Processing

```rust
let mut batch_processor = MemoryEfficientBatchProcessor::new(2048); // 2GB limit

// Add tensors
let tensor_id = batch_processor.add_tensor(image_tensor)?;

// Process when memory is full
let usage = batch_processor.current_memory_usage();
if usage.utilization > 0.8 {
    let results = batch_processor.process_batch()?;
}
```

## Advanced Features

### Feature Extraction

```rust
use torsh_vision::ops::*;

// HOG features
let hog_features = extract_hog_features(&image, 8, 8, 2, 9)?;

// SIFT-like features
let (keypoints, descriptors) = extract_sift_like_features(&image)?;

// ORB-like features
let orb_features = extract_orb_like_features(&image, 500)?;
```

### Similarity Search

```rust
// Image similarity
let similarity = calculate_similarity(&img1, &img2, SimilarityMetric::Euclidean)?;

// Content-based image retrieval
let mut retrieval = ContentBasedImageRetrieval::new();
retrieval.add_image("img1", &features1)?;
retrieval.add_image("img2", &features2)?;

let results = retrieval.query(&query_features, 5)?; // Top 5 similar images
```

### Quality Assessment

```rust
// Image quality metrics
let psnr = calculate_psnr(&original, &compressed, 255.0)?;
let ssim = calculate_ssim(&img1, &img2, None)?;
let mse = calculate_mse(&img1, &img2)?;
let mae = calculate_mae(&img1, &img2)?;

println!("PSNR: {:.2} dB, SSIM: {:.3}", psnr, ssim);
```

## Performance Optimization

### Benchmarking

```rust
use std::time::Instant;

// Transform performance
let start = Instant::now();
for _ in 0..100 {
    let _ = transform.forward(&image)?;
}
let avg_time = start.elapsed().as_millis() as f64 / 100.0;
println!("Average time: {:.2} ms", avg_time);

// Memory usage estimation
let estimate = MemoryOptimizer::estimate_batch_memory(&shapes);
println!("Estimated memory: {:.2} MB", estimate.total_mb);

// Optimal batch size calculation
let optimal_batch = MemoryOptimizer::calculate_optimal_batch_size(
    &shape, 8192, 0.8  // 8GB GPU, 80% utilization
);
```

### Hardware Optimization

```rust
// Automatic device selection
let context = TransformContext::auto_optimize()?;

// Tensor Core optimization
if hardware.has_tensor_cores() {
    let optimized_tensor = optimize_for_tensor_cores(&tensor)?;
}

// Batch size optimization
let optimal_batch = context.calculate_optimal_batch_size(&image_shape)?;
```

## Best Practices

### Error Handling

```rust
use torsh_vision::{Result, VisionError};

// Always use Result types
fn process_pipeline() -> Result<Tensor> {
    let image = load_image("input.jpg")
        .map_err(|e| VisionError::IoError(e))?;
    
    let processed = apply_transforms(&image)
        .map_err(|e| VisionError::TransformError(e.to_string()))?;
    
    Ok(processed)
}

// Handle specific error types
match process_pipeline() {
    Ok(result) => println!("Success: {:?}", result.shape()),
    Err(VisionError::IoError(e)) => eprintln!("I/O error: {}", e),
    Err(VisionError::TransformError(e)) => eprintln!("Transform error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Memory Management

```rust
// Use memory pooling for training loops
let mut pool = TensorPool::new(1000);

for epoch in 0..num_epochs {
    for batch in dataloader {
        // Get tensors from pool
        let tensor = pool.get_tensor(&batch_shape)?;
        
        // Process batch
        let result = model.forward(&tensor)?;
        
        // Return to pool when done
        pool.return_tensor(tensor)?;
    }
}
```

### Performance Optimization

```rust
// Use hardware acceleration when available
let hardware = HardwareContext::auto_detect()?;
let transforms = if hardware.cuda_available() {
    // GPU transforms
    create_gpu_transform_pipeline()
} else {
    // CPU transforms
    create_cpu_transform_pipeline()
};

// Batch processing for efficiency
let batch_size = MemoryOptimizer::calculate_optimal_batch_size(
    &image_shape, 
    hardware.memory_gb() * 1024,
    0.8
);
```

### Data Loading

```rust
// Use optimized datasets for large-scale training
let config = DatasetConfig {
    cache_size_mb: 2048,
    prefetch_size: batch_size * 4,
    max_workers: num_cpus::get(),
    enable_validation: true,
};

let dataset = OptimizedImageDataset::new(data_path, config)?;
```

### Transform Pipelines

```rust
// Separate training and validation transforms
let train_transforms = TransformBuilder::new()
    .resize((256, 256))
    .random_resized_crop((224, 224))
    .random_horizontal_flip(0.5)
    .add(ColorJitter::new().brightness(0.4))
    .imagenet_normalize()
    .build();

let val_transforms = TransformBuilder::new()
    .resize((256, 256))
    .center_crop((224, 224))
    .imagenet_normalize()
    .build();
```

## Conclusion

ToRSh Vision provides a comprehensive, high-performance computer vision framework with:

- **Zero-cost abstractions**: Rust's performance without compromising safety
- **Hardware acceleration**: Automatic GPU/CPU optimization with mixed precision support
- **Memory efficiency**: Advanced memory management and optimization
- **Comprehensive features**: Complete pipeline from data loading to model inference
- **Production ready**: Robust error handling and performance monitoring
- **Extensible**: Clean APIs for custom implementations

For more detailed examples and advanced usage patterns, see the [examples module](./src/examples.rs) and refer to the API documentation.