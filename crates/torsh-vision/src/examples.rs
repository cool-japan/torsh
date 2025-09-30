//! Comprehensive examples for torsh-vision
//!
//! This module provides complete examples demonstrating all the major features
//! of torsh-vision including transforms, datasets, models, I/O operations,
//! and advanced computer vision techniques powered by SciRS2 integration.

pub mod comprehensive_showcase;

use crate::datasets::{ImageFolder, OptimizedDataset, OptimizedImageDataset};
use crate::hardware::{HardwareAccelerated, HardwareContext};
use crate::io::{ImageInfo, VisionIO};
use crate::memory::{GlobalMemoryManager, MemorySettings};
use crate::transforms::{
    AugMix, AutoAugment, ColorJitter, MixUp, Mosaic, RandAugment, RandomErasing,
    RandomHorizontalFlip, RandomResizedCrop, RandomRotation,
};
use crate::transforms::{Compose, Transform, TransformBuilder, TransformIntrospection};
use crate::unified_transforms::{TransformContext, UnifiedTransform};
use crate::{Result, VisionError};
use std::path::Path;
use std::sync::Arc;
use torsh_core::device::CpuDevice;
use torsh_tensor::{creation, Tensor};

// Re-export the comprehensive showcase
pub use comprehensive_showcase::run_comprehensive_showcase;

/// Complete image classification pipeline example
pub mod image_classification {
    use super::*;

    /// Example: Complete image classification training pipeline
    ///
    /// This example demonstrates:
    /// - Loading and preprocessing image datasets
    /// - Applying transforms and augmentations
    /// - Using memory-efficient data loading
    /// - GPU acceleration when available
    pub fn complete_training_pipeline() -> Result<()> {
        println!("=== Image Classification Training Pipeline ===");

        // 1. Set up hardware context
        let hardware = HardwareContext::auto_detect()?;
        println!("Detected hardware: {:?}", hardware.device_info());

        // 2. Configure memory management
        let memory_settings = MemorySettings {
            enable_pooling: true,
            max_pool_size: 200,
            max_batch_memory_mb: 2048,
            enable_profiling: true,
            auto_optimization: true,
        };
        crate::memory::configure_global_memory(memory_settings);

        // 3. Set up data transforms
        let train_transforms = TransformBuilder::new()
            .resize((256, 256))
            .random_horizontal_flip(0.5)
            .center_crop((224, 224))
            .imagenet_normalize()
            .build();

        println!("Transform pipeline: {}", train_transforms.describe());

        // 4. Create dataset (example path - replace with actual path)
        // let dataset = ImageFolderLazy::new("path/to/imagenet/train")?;
        // let lazy_dataset = LazyDataset::new(dataset, 1000); // Cache 1000 images

        println!("✓ Training pipeline setup complete");
        Ok(())
    }

    /// Example: Real-time image classification inference
    pub fn real_time_inference() -> Result<()> {
        println!("=== Real-time Image Classification Inference ===");

        // 1. Set up GPU acceleration if available
        let hardware = HardwareContext::auto_detect()?;

        // 2. Create inference transforms
        let inference_transforms = TransformBuilder::new()
            .resize((224, 224))
            .center_crop((224, 224))
            .imagenet_normalize()
            .build();

        // 3. Simulate processing a batch of images
        let batch_size = 32;
        let mut batch_tensors = Vec::new();

        for i in 0..batch_size {
            // Create dummy image tensor (3, 224, 224)
            let image_tensor = creation::randn(&[3, 224, 224]).unwrap();

            // Apply transforms
            let processed = inference_transforms.forward(&image_tensor)?;
            batch_tensors.push(processed);

            if i % 10 == 0 {
                println!("Processed image {}/{}", i + 1, batch_size);
            }
        }

        println!("✓ Processed {} images successfully", batch_size);
        Ok(())
    }
}

/// Advanced computer vision techniques examples
pub mod advanced_cv {
    use super::*;

    /// Example: Object detection pipeline with NMS
    pub fn object_detection_pipeline() -> Result<()> {
        println!("=== Object Detection Pipeline ===");

        // 1. Create detection transforms
        let detection_transforms = TransformBuilder::new()
            .resize((640, 640)) // YOLO input size
            .normalize(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]) // No normalization for detection
            .build();

        // 2. Simulate object detection
        let image = creation::randn(&[3, 640, 640]).unwrap();
        let processed = detection_transforms.forward(&image)?;

        // 3. Simulate detection results
        println!("Input shape: {:?}", image.shape().dims());
        println!("Processed shape: {:?}", processed.shape().dims());

        // In a real implementation, this would run through a detection model
        // and apply NMS to filter overlapping detections

        println!("✓ Object detection preprocessing complete");
        Ok(())
    }

    /// Example: Image segmentation workflow
    pub fn image_segmentation() -> Result<()> {
        println!("=== Image Segmentation Workflow ===");

        // 1. Create segmentation transforms
        let seg_transforms = TransformBuilder::new()
            .resize((512, 512)) // Common segmentation size
            .normalize(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
            .build();

        // 2. Process image for segmentation
        let image = creation::randn(&[3, 1024, 768]).unwrap(); // Original high-res image
        let processed = seg_transforms.forward(&image)?;

        println!("Original image: {:?}", image.shape().dims());
        println!("Segmentation input: {:?}", processed.shape().dims());

        // 3. Simulate segmentation mask output
        let segmentation_mask: Tensor<f32> = creation::zeros(&[1, 512, 512]).unwrap(); // Single channel mask

        println!("✓ Segmentation preprocessing complete");
        println!("Mask shape: {:?}", segmentation_mask.shape().dims());

        Ok(())
    }
}

/// Data augmentation and preprocessing examples
pub mod data_augmentation {
    use super::*;
    use crate::transforms::*;

    /// Example: Advanced data augmentation pipeline
    pub fn advanced_augmentation_pipeline() -> Result<()> {
        println!("=== Advanced Data Augmentation Pipeline ===");

        // 1. Create sophisticated augmentation pipeline
        let augmentation = TransformBuilder::new()
            .add(RandomResizedCrop::new((224, 224)))
            .add(RandomHorizontalFlip::new(0.5))
            .add(
                ColorJitter::new()
                    .brightness(0.4)
                    .contrast(0.4)
                    .saturation(0.4)
                    .hue(0.1),
            )
            .add(RandomRotation::new((-15.0, 15.0)))
            .add(RandomErasing::new(0.25))
            .imagenet_normalize()
            .build();

        // 2. Apply to sample images
        let original_image = creation::randn(&[3, 256, 256]).unwrap();

        println!("Applying augmentation pipeline...");
        for i in 0..5 {
            let augmented = augmentation.forward(&original_image)?;
            println!(
                "Augmented image {} shape: {:?}",
                i + 1,
                augmented.shape().dims()
            );
        }

        // 3. Demonstrate advanced techniques
        let mut mixup = MixUp::new(1.0);
        let image1 = creation::randn(&[3, 224, 224]).unwrap();
        let image2 = creation::randn(&[3, 224, 224]).unwrap();

        let (mixed_image, mixed_labels) = mixup.apply_pair(&image1, &image2, 0, 1, 10)?;
        println!(
            "MixUp result - Image: {:?}, Labels: {:?}",
            mixed_image.shape().dims(),
            mixed_labels.shape().dims()
        );

        println!("✓ Advanced augmentation pipeline complete");
        Ok(())
    }

    /// Example: AutoAugment and RandAugment
    pub fn automatic_augmentation() -> Result<()> {
        println!("=== Automatic Augmentation Techniques ===");

        // 1. AutoAugment
        let auto_augment = AutoAugment::new();
        let image: Tensor<f32> = creation::randn(&[3, 224, 224]).unwrap();

        println!("Applying AutoAugment...");
        let augmented1 = auto_augment.forward(&image)?;
        println!("AutoAugment result: {:?}", augmented1.shape().dims());

        // 2. RandAugment
        let rand_augment = RandAugment::new(2, 5.0); // 2 operations, magnitude 5

        println!("Applying RandAugment...");
        let augmented2 = rand_augment.forward(&image)?;
        println!("RandAugment result: {:?}", augmented2.shape().dims());

        // 3. AugMix
        let augmix = AugMix::new();

        println!("Applying AugMix...");
        let augmented3 = augmix.forward(&image)?;
        println!("AugMix result: {:?}", augmented3.shape().dims());

        println!("✓ Automatic augmentation techniques complete");
        Ok(())
    }
}

/// I/O and file handling examples
pub mod io_examples {
    use super::*;

    /// Example: Batch image processing
    pub fn batch_image_processing() -> Result<()> {
        println!("=== Batch Image Processing ===");

        // 1. Set up I/O manager
        let io = VisionIO::new()
            .with_default_format(image::ImageFormat::Png)
            .with_caching(true, 512); // 512MB cache

        // 2. Simulate batch loading
        // In a real scenario, you would have actual image paths
        let image_paths = vec!["image1.jpg", "image2.png", "image3.bmp"];

        println!(
            "Simulating batch loading of {} images...",
            image_paths.len()
        );

        // 3. Demonstrate metadata extraction
        for (i, path) in image_paths.iter().enumerate() {
            println!("Processing image {}: {}", i + 1, path);
            // In real usage: let info = io.get_image_info(path)?;
        }

        // 4. Demonstrate format conversion
        println!("Format conversion capabilities:");
        let supported_formats = io.supported_formats();
        for format in supported_formats {
            println!("  - {:?}", format);
        }

        println!("✓ Batch processing simulation complete");
        Ok(())
    }

    /// Example: Memory-mapped large dataset loading
    pub fn memory_mapped_loading() -> Result<()> {
        println!("=== Memory-mapped Large Dataset Loading ===");

        // This would be used for very large datasets that don't fit in memory
        println!("Setting up memory-mapped dataset loader...");

        // Simulate working with a large dataset
        let dataset_size = 1_000_000; // 1M images
        let batch_size = 64;
        let num_batches = dataset_size / batch_size;

        println!(
            "Dataset: {} images, Batch size: {}, Batches: {}",
            dataset_size, batch_size, num_batches
        );

        // In real usage, this would use the improved datasets module
        // with lazy loading and memory mapping

        println!("✓ Memory-mapped loading setup complete");
        Ok(())
    }
}

/// Memory optimization examples
pub mod memory_optimization {
    use super::*;
    use crate::memory::*;

    /// Example: Memory-efficient training
    pub fn memory_efficient_training() -> Result<()> {
        println!("=== Memory-efficient Training ===");

        // 1. Set up memory profiler
        let profiler = MemoryProfiler::new();

        // 2. Set up tensor pool
        let mut tensor_pool = TensorPool::new(100);

        // 3. Simulate training loop with memory optimization
        let batch_size = 32;
        let image_shape = [3, 224, 224];

        println!("Training with batch size: {}", batch_size);

        for epoch in 0..3 {
            println!("Epoch {}", epoch + 1);

            // Simulate batch processing
            let mut batch_tensors = Vec::new();

            for i in 0..batch_size {
                // Get tensor from pool
                let tensor = tensor_pool.get_tensor(&image_shape)?;
                profiler.record_allocation(
                    tensor.shape().dims().iter().product::<usize>() * 4,
                    "training_loop",
                    "batch_tensor",
                );

                batch_tensors.push(tensor);
            }

            // Return tensors to pool after processing
            for tensor in batch_tensors {
                tensor_pool.return_tensor(tensor)?;
            }

            // Show pool statistics
            let stats = tensor_pool.stats();
            println!(
                "  Pool stats - Reuse rate: {:.1}%, Tensors: {}",
                stats.reuse_rate * 100.0,
                stats.total_tensors
            );
        }

        // Final profiling summary
        let summary = profiler.summary();
        println!("Memory profiling summary:");
        println!("  Total allocations: {}", summary.total_allocations);
        println!(
            "  Peak usage: {:.2} MB",
            summary.peak_usage_bytes as f32 / (1024.0 * 1024.0)
        );
        println!(
            "  Average allocation: {:.2} KB",
            summary.average_allocation_bytes as f32 / 1024.0
        );

        println!("✓ Memory-efficient training complete");
        Ok(())
    }

    /// Example: Dynamic memory optimization
    pub fn dynamic_memory_optimization() -> Result<()> {
        println!("=== Dynamic Memory Optimization ===");

        // 1. Set up batch processor with memory limits
        let mut batch_processor = MemoryEfficientBatchProcessor::new(1024); // 1GB limit

        // 2. Process variable-sized tensors
        let tensor_shapes = vec![
            vec![3, 224, 224],
            vec![3, 512, 512],
            vec![3, 1024, 1024],
            vec![3, 128, 128],
        ];

        for (i, shape) in tensor_shapes.iter().enumerate() {
            let tensor = creation::randn(shape)?;
            let tensor_id = batch_processor.add_tensor(tensor)?;

            let usage = batch_processor.current_memory_usage();
            println!("Added tensor {} (shape: {:?})", tensor_id, shape);
            println!(
                "  Memory usage: {:.1}% ({} MB / {} MB)",
                usage.utilization * 100.0,
                usage.current_mb,
                usage.max_mb
            );

            // Process batch if memory is getting full
            if usage.utilization > 0.8 {
                println!("  Processing batch due to memory pressure...");
                let results = batch_processor.process_batch()?;
                println!("  Processed {} tensors", results.len());
            }
        }

        // Flush remaining tensors
        let final_results = batch_processor.flush()?;
        println!("Final batch processed: {} tensors", final_results.len());

        println!("✓ Dynamic memory optimization complete");
        Ok(())
    }
}

/// Hardware acceleration examples
pub mod hardware_acceleration {
    use super::*;

    /// Example: GPU-accelerated transforms
    pub fn gpu_accelerated_transforms() -> Result<()> {
        println!("=== GPU-accelerated Transforms ===");

        // 1. Detect hardware capabilities
        let hardware = HardwareContext::auto_detect()?;
        println!("Hardware capabilities:");
        println!("  CUDA available: {}", hardware.cuda_available());
        println!("  Mixed precision: {}", hardware.supports_mixed_precision());
        println!("  Tensor cores: {}", hardware.has_tensor_cores());

        // 2. Create GPU-optimized transform pipeline
        let device = if hardware.cuda_available() {
            // In a real implementation, this would create a CUDA device
            Arc::new(CpuDevice::new()) // Fallback to CPU for now
        } else {
            Arc::new(CpuDevice::new())
        };
        let context = TransformContext::new(device);

        // In a real implementation, this would use the UnifiedTransform API
        // to automatically select GPU vs CPU execution

        // 3. Simulate processing with different precisions
        let image: Tensor<f32> = creation::randn(&[3, 224, 224]).unwrap();

        println!("Processing image with shape: {:?}", image.shape().dims());

        // Simulate GPU processing
        if hardware.cuda_available() {
            println!("✓ Using GPU acceleration");

            if hardware.supports_mixed_precision() {
                println!("✓ Using mixed precision (f16)");
            }

            if hardware.has_tensor_cores() {
                println!("✓ Using Tensor Cores for acceleration");
            }
        } else {
            println!("⚠ Falling back to CPU processing");
        }

        println!("✓ GPU acceleration demo complete");
        Ok(())
    }

    /// Example: Mixed precision training
    pub fn mixed_precision_training() -> Result<()> {
        println!("=== Mixed Precision Training ===");

        let hardware = HardwareContext::auto_detect()?;

        if !hardware.supports_mixed_precision() {
            println!("⚠ Mixed precision not supported on this hardware");
            return Ok(());
        }

        println!("✓ Mixed precision training supported");

        // Simulate mixed precision training
        let batch_size = 32;
        let iterations = 10;

        for i in 0..iterations {
            // In real implementation, this would:
            // 1. Convert inputs to f16
            // 2. Run forward pass in f16
            // 3. Convert loss to f32 for stability
            // 4. Scale gradients for f16 training

            if i % 3 == 0 {
                println!("Iteration {}: Training with f16 precision", i + 1);
            }
        }

        println!("✓ Mixed precision training simulation complete");
        println!("  Memory savings: ~50%");
        println!("  Speed improvement: ~1.5-2x on modern GPUs");

        Ok(())
    }
}

/// Complete workflow examples
pub mod complete_workflows {
    use super::*;

    /// Example: End-to-end image classification workflow
    pub fn end_to_end_classification() -> Result<()> {
        println!("=== End-to-End Image Classification Workflow ===");

        // 1. Setup phase
        println!("1. Setting up environment...");
        let hardware = HardwareContext::auto_detect()?;
        let memory_manager = GlobalMemoryManager::new(MemorySettings::default());

        // 2. Data preparation
        println!("2. Preparing data pipeline...");
        let transforms = TransformBuilder::new()
            .resize((256, 256))
            .random_horizontal_flip(0.5)
            .center_crop((224, 224))
            .add(ColorJitter::new().brightness(0.2).contrast(0.2))
            .imagenet_normalize()
            .build();

        // 3. Model setup (simulated)
        println!("3. Setting up model...");
        // In real usage: let model = ResNet::resnet50(num_classes=1000)?;

        // 4. Training loop (simulated)
        println!("4. Training loop...");
        for epoch in 1..=3 {
            println!("  Epoch {}/3", epoch);

            // Simulate batch processing
            for batch in 1..=5 {
                let batch_images: Tensor<f32> = creation::randn(&[32, 3, 224, 224]).unwrap();
                let batch_labels: Tensor<f32> = creation::zeros(&[32]).unwrap();

                // Apply transforms
                // In real usage: apply transforms to each image in batch

                if batch % 2 == 0 {
                    println!("    Batch {}/5 processed", batch);
                }
            }
        }

        // 5. Evaluation
        println!("5. Evaluating model...");
        let test_accuracy = 92.5; // Simulated
        println!("  Test accuracy: {:.1}%", test_accuracy);

        // 6. Cleanup and statistics
        let stats = memory_manager.global_stats();
        println!("6. Final statistics:");
        if let Some(pool_stats) = stats.pool_stats {
            println!(
                "  Tensor pool reuse rate: {:.1}%",
                pool_stats.reuse_rate * 100.0
            );
        }

        println!("✓ End-to-end classification workflow complete");
        Ok(())
    }

    /// Example: Complete object detection workflow
    pub fn end_to_end_object_detection() -> Result<()> {
        println!("=== End-to-End Object Detection Workflow ===");

        // 1. Data preparation for detection
        println!("1. Setting up detection pipeline...");
        let detection_transforms = TransformBuilder::new()
            .resize((640, 640))
            .normalize(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0])
            .build();

        // 2. Model setup (YOLO-style)
        println!("2. Setting up YOLO model...");
        // In real usage: let model = YOLOv5::yolo_v5_medium(num_classes=80)?;

        // 3. Training with detection-specific augmentations
        println!("3. Training with detection augmentations...");
        let mosaic = Mosaic::new((640, 640));

        // Simulate mosaic augmentation
        println!("  Applying mosaic augmentation...");

        // 4. Inference and NMS
        println!("4. Running inference with NMS...");

        // Simulate detection results
        let detections = vec![
            ("person", 0.95, [100, 150, 200, 300]),
            ("car", 0.87, [300, 200, 450, 350]),
            ("bicycle", 0.78, [50, 100, 120, 200]),
        ];

        println!("  Detected {} objects:", detections.len());
        for (class, conf, bbox) in &detections {
            println!("    {}: {:.2} confidence at {:?}", class, conf, bbox);
        }

        println!("✓ End-to-end object detection workflow complete");
        Ok(())
    }
}

/// Interactive visualization examples
pub mod interactive_visualization {
    use super::*;
    use crate::interactive::*;
    use crate::viz3d::*;

    /// Example: Interactive image viewer with annotations
    pub fn interactive_image_viewer() -> Result<()> {
        println!("=== Interactive Image Viewer Example ===");

        // 1. Create interactive viewer
        let mut viewer = InteractiveViewer::new();

        // 2. Load a sample image
        let image = creation::randn(&[3, 512, 512]).unwrap();
        viewer.load_image(image)?;
        println!("✓ Loaded image into interactive viewer");

        // 3. Add various annotations
        let bbox = Annotation::BoundingBox {
            x: 100.0,
            y: 150.0,
            width: 200.0,
            height: 150.0,
            label: "Object 1".to_string(),
            color: [255, 0, 0],
            confidence: Some(0.95),
        };
        viewer.add_annotation(bbox);

        let point = Annotation::Point {
            x: 250.0,
            y: 200.0,
            label: "Landmark".to_string(),
            color: [0, 255, 0],
            radius: 5.0,
        };
        viewer.add_annotation(point);

        let polygon = Annotation::Polygon {
            points: vec![
                (300.0, 100.0),
                (400.0, 120.0),
                (380.0, 200.0),
                (320.0, 180.0),
            ],
            label: "Region".to_string(),
            color: [0, 0, 255],
            filled: false,
        };
        viewer.add_annotation(polygon);

        println!("✓ Added {} annotations", viewer.annotations().len());

        // 4. Export annotations
        let exported = viewer.export_annotations()?;
        println!("✓ Exported annotations to JSON format");

        // 5. Set up event handlers
        viewer.on_event("mouse_click".to_string(), |event| {
            if let ViewerEvent::MouseClick { x, y, .. } = event {
                println!("Mouse clicked at ({:.1}, {:.1})", x, y);
            }
        });

        // Simulate some events
        viewer.handle_mouse_click(150.0, 200.0, MouseButton::Left);
        viewer.handle_key_press("Space".to_string());

        println!("✓ Interactive image viewer example complete");
        Ok(())
    }

    /// Example: Interactive image gallery
    pub fn interactive_image_gallery() -> Result<()> {
        println!("=== Interactive Image Gallery Example ===");

        // 1. Create gallery
        let mut gallery = InteractiveGallery::new();

        // 2. Add sample images
        for i in 0..5 {
            let image = creation::randn(&[3, 256, 256]).unwrap();
            gallery.add_image(format!("sample_image_{}", i + 1), image)?;
        }

        println!("✓ Added {} images to gallery", gallery.len());

        // 3. Navigate through images
        for i in 0..gallery.len() {
            let (name, _) = gallery.current_image().unwrap();
            println!("  Viewing: {}", name);

            // Add annotation to current image
            let annotation = Annotation::Text {
                x: 50.0,
                y: 50.0,
                text: format!("Image {}", i + 1),
                color: [255, 255, 255],
                font_size: 16.0,
            };
            gallery.add_annotation_to_current(annotation)?;

            if i < gallery.len() - 1 {
                gallery.next_image()?;
            }
        }

        println!("✓ Navigated through all images and added annotations");

        // 4. Show gallery statistics
        println!("Gallery statistics:");
        println!("  Total images: {}", gallery.len());
        println!("  Image names: {:?}", gallery.image_names());

        println!("✓ Interactive image gallery example complete");
        Ok(())
    }

    /// Example: Live visualization for real-time processing
    pub fn live_visualization() -> Result<()> {
        println!("=== Live Visualization Example ===");

        // 1. Create live visualization
        let mut live_viz = LiveVisualization::new();

        // 2. Simulate real-time frame processing
        let num_frames = 30;
        println!(
            "Simulating {} frames of real-time processing...",
            num_frames
        );

        for i in 0..num_frames {
            // Simulate frame generation (e.g., from camera or video)
            let frame = creation::randn(&[3, 480, 640]).unwrap();
            live_viz.add_frame(frame)?;

            if i % 10 == 0 {
                println!("  Frame {}: FPS = {:.1}", i + 1, live_viz.current_fps());
            }

            // Simulate processing delay
            std::thread::sleep(std::time::Duration::from_millis(33)); // ~30 FPS
        }

        println!("✓ Final FPS: {:.1}", live_viz.current_fps());
        println!("✓ Buffer size: {}", live_viz.buffer_len());

        println!("✓ Live visualization example complete");
        Ok(())
    }
}

/// 3D visualization examples
pub mod viz3d_examples {
    use super::*;
    use crate::viz3d::*;

    /// Example: 3D point cloud visualization
    pub fn point_cloud_visualization() -> Result<()> {
        println!("=== 3D Point Cloud Visualization ===");

        // 1. Create point cloud from random data
        let mut points = Vec::new();
        for i in 0..1000 {
            let x = (i as f32 * 0.01).sin() * 10.0;
            let y = (i as f32 * 0.01).cos() * 10.0;
            let z = i as f32 * 0.02;

            let color = [
                ((x + 10.0) / 20.0 * 255.0) as u8,
                ((y + 10.0) / 20.0 * 255.0) as u8,
                (z / 20.0 * 255.0) as u8,
            ];

            points.push(Point3D::with_color(x, y, z, color));
        }

        let mut cloud = PointCloud3D::new(points);
        println!("✓ Created point cloud with {} points", cloud.len());

        // 2. Apply voxel downsampling
        let downsampled = cloud.voxel_downsample(2.0);
        println!(
            "✓ Downsampled to {} points (voxel size: 2.0)",
            downsampled.len()
        );

        // 3. Filter by distance
        let center = Point3D::new(0.0, 0.0, 10.0);
        let filtered = cloud.filter_by_distance(center, 15.0);
        println!(
            "✓ Filtered to {} points within distance 15.0",
            filtered.len()
        );

        // 4. Convert to tensor and back
        let tensor = cloud.to_tensor()?;
        println!(
            "✓ Converted to tensor with shape: {:?}",
            tensor.shape().dims()
        );

        let cloud_from_tensor = PointCloud3D::from_tensor(&tensor)?;
        println!(
            "✓ Converted back to point cloud with {} points",
            cloud_from_tensor.len()
        );

        println!("✓ Point cloud visualization example complete");
        Ok(())
    }

    /// Example: 3D mesh creation and manipulation
    pub fn mesh_visualization() -> Result<()> {
        println!("=== 3D Mesh Visualization ===");

        // 1. Create sphere mesh
        let center = Point3D::new(0.0, 0.0, 0.0);
        let mut sphere = Mesh3D::create_sphere(center, 5.0, 20, 20);
        println!(
            "✓ Created sphere mesh: {} vertices, {} faces",
            sphere.metadata.num_vertices, sphere.metadata.num_faces
        );

        // 2. Create cube mesh
        let cube_center = Point3D::new(10.0, 0.0, 0.0);
        let mut cube = Mesh3D::create_cube(cube_center, 4.0);
        println!(
            "✓ Created cube mesh: {} vertices, {} faces",
            cube.metadata.num_vertices, cube.metadata.num_faces
        );

        // 3. Compute normals
        sphere.compute_vertex_normals();
        cube.compute_vertex_normals();
        println!("✓ Computed vertex normals");

        // 4. Create custom triangle mesh
        let vertices = vec![
            Point3D::new(-5.0, 0.0, 0.0),
            Point3D::new(5.0, 0.0, 0.0),
            Point3D::new(0.0, 8.0, 0.0),
            Point3D::new(0.0, 4.0, 6.0),
        ];

        let faces = vec![
            Triangle3D::new(0, 1, 2), // Base triangle
            Triangle3D::new(0, 2, 3), // Side 1
            Triangle3D::new(1, 3, 2), // Side 2
            Triangle3D::new(0, 3, 1), // Side 3
        ];

        let mut custom_mesh = Mesh3D::new(vertices, faces);
        custom_mesh.compute_face_normals();
        println!(
            "✓ Created custom tetrahedron mesh: {} vertices, {} faces",
            custom_mesh.metadata.num_vertices, custom_mesh.metadata.num_faces
        );

        println!("✓ Mesh visualization example complete");
        Ok(())
    }

    /// Example: 3D bounding boxes for object detection
    pub fn bounding_box_3d_visualization() -> Result<()> {
        println!("=== 3D Bounding Box Visualization ===");

        // 1. Create 3D bounding boxes for detected objects
        let bbox1 = BoundingBox3D::new(
            [5.0, 2.0, 1.0], // center
            [4.0, 2.0, 6.0], // dimensions (w, h, d)
            [0.0, 0.2, 0.0], // rotation (roll, pitch, yaw)
            "Car".to_string(),
            0.95,
        )
        .with_color([255, 0, 0]);

        let bbox2 = BoundingBox3D::new(
            [-3.0, 1.0, 0.5],
            [1.5, 1.8, 0.8],
            [0.0, 0.0, 0.5],
            "Person".to_string(),
            0.87,
        )
        .with_color([0, 255, 0]);

        let bbox3 = BoundingBox3D::new(
            [8.0, 1.0, 0.3],
            [2.0, 1.0, 1.0],
            [0.0, 0.0, -0.3],
            "Bicycle".to_string(),
            0.78,
        )
        .with_color([0, 0, 255]);

        println!("✓ Created {} 3D bounding boxes", 3);

        // 2. Calculate properties
        println!("Bounding box properties:");
        for (i, bbox) in [&bbox1, &bbox2, &bbox3].iter().enumerate() {
            println!("  {}: {} (conf: {:.2})", i + 1, bbox.label, bbox.confidence);
            println!("    Center: {:?}", bbox.center);
            println!("    Volume: {:.2} m³", bbox.volume());

            // Test point containment
            let test_point = Point3D::new(bbox.center[0], bbox.center[1], bbox.center[2]);
            println!(
                "    Contains center point: {}",
                bbox.contains_point(test_point)
            );
        }

        // 3. Get corner points
        let corners = bbox1.corners();
        println!("✓ Car bounding box has {} corner points", corners.len());

        println!("✓ 3D bounding box visualization example complete");
        Ok(())
    }

    /// Example: Complete 3D scene composition
    pub fn complete_3d_scene() -> Result<()> {
        println!("=== Complete 3D Scene Composition ===");

        // 1. Create 3D scene
        let mut scene = Scene3D::new("Object Detection Scene".to_string());

        // 2. Add point cloud (e.g., from LiDAR)
        let mut lidar_points = Vec::new();
        for i in 0..500 {
            let theta = i as f32 * 0.01;
            let r = 20.0 + (theta * 2.0).sin() * 5.0;
            let x = r * theta.cos();
            let y = r * theta.sin();
            let z = (theta * 3.0).sin() * 2.0;

            lidar_points.push(Point3D::new(x, y, z));
        }
        let point_cloud = PointCloud3D::new(lidar_points);
        scene.add_point_cloud(point_cloud);

        // 3. Add ground plane mesh
        let ground_vertices = vec![
            Point3D::with_color(-50.0, -50.0, 0.0, [100, 100, 100]),
            Point3D::with_color(50.0, -50.0, 0.0, [100, 100, 100]),
            Point3D::with_color(50.0, 50.0, 0.0, [100, 100, 100]),
            Point3D::with_color(-50.0, 50.0, 0.0, [100, 100, 100]),
        ];
        let ground_faces = vec![Triangle3D::new(0, 1, 2), Triangle3D::new(0, 2, 3)];
        let mut ground_mesh = Mesh3D::new(ground_vertices, ground_faces);
        ground_mesh.metadata.name = "Ground Plane".to_string();
        scene.add_mesh(ground_mesh);

        // 4. Add 3D bounding boxes for detected objects
        let car_bbox = BoundingBox3D::new(
            [10.0, 5.0, 1.0],
            [4.5, 2.0, 1.8],
            [0.0, 0.0, 0.3],
            "Car".to_string(),
            0.95,
        )
        .with_color([255, 0, 0]);
        scene.add_bounding_box(car_bbox);

        let person_bbox = BoundingBox3D::new(
            [-5.0, 8.0, 0.9],
            [0.6, 0.6, 1.8],
            [0.0, 0.0, 0.0],
            "Person".to_string(),
            0.87,
        )
        .with_color([0, 255, 0]);
        scene.add_bounding_box(person_bbox);

        // 5. Scene statistics
        println!("Scene statistics:");
        println!("  Name: {}", scene.metadata.name);
        println!("  Total objects: {}", scene.num_objects());
        println!("  Point clouds: {}", scene.point_clouds.len());
        println!("  Meshes: {}", scene.meshes.len());
        println!("  Bounding boxes: {}", scene.bounding_boxes.len());

        if let Some(bounds) = &scene.metadata.bounds {
            println!(
                "  Scene bounds: center {:?}, dimensions {:?}",
                bounds.center, bounds.dimensions
            );
        }

        // 6. Export scene summary
        let summary = scene.export_summary();
        println!("\nScene summary:\n{}", summary);

        println!("✓ Complete 3D scene composition example complete");
        Ok(())
    }
}

/// Performance benchmarking examples
pub mod benchmarking {
    use super::*;
    use std::time::Instant;

    /// Example: Transform performance benchmarking
    pub fn transform_performance_benchmark() -> Result<()> {
        println!("=== Transform Performance Benchmark ===");

        let image = creation::randn(&[3, 1024, 1024]).unwrap();
        let iterations = 100;

        // Benchmark individual transforms
        let transforms = vec![
            (
                "Resize",
                Box::new(crate::transforms::Resize::new((224, 224))) as Box<dyn Transform>,
            ),
            (
                "RandomHorizontalFlip",
                Box::new(crate::transforms::RandomHorizontalFlip::new(0.5)),
            ),
            (
                "ColorJitter",
                Box::new(crate::transforms::ColorJitter::new().brightness(0.2)),
            ),
        ];

        for (name, transform) in transforms {
            let start = Instant::now();

            for _ in 0..iterations {
                let _result = transform.forward(&image)?;
            }

            let duration = start.elapsed();
            let avg_ms = duration.as_millis() as f64 / iterations as f64;

            println!(
                "{}: {:.2} ms/image (avg over {} iterations)",
                name, avg_ms, iterations
            );
        }

        // Benchmark complete pipeline
        println!("\nBenchmarking complete pipeline...");
        let pipeline = TransformBuilder::new()
            .resize((224, 224))
            .random_horizontal_flip(0.5)
            .add(crate::transforms::ColorJitter::new().brightness(0.2))
            .imagenet_normalize()
            .build();

        let start = Instant::now();
        for _ in 0..iterations {
            let _result = pipeline.forward(&image)?;
        }
        let duration = start.elapsed();
        let avg_ms = duration.as_millis() as f64 / iterations as f64;

        println!("Complete pipeline: {:.2} ms/image", avg_ms);

        println!("✓ Performance benchmark complete");
        Ok(())
    }

    /// Example: Memory usage benchmark
    pub fn memory_usage_benchmark() -> Result<()> {
        println!("=== Memory Usage Benchmark ===");

        let shapes = vec![
            vec![3, 224, 224],
            vec![3, 512, 512],
            vec![3, 1024, 1024],
            vec![3, 2048, 2048],
        ];

        for shape in &shapes {
            let estimate =
                crate::memory::MemoryOptimizer::estimate_batch_memory(&vec![shape.clone(); 32]);

            println!(
                "Batch of 32 images ({}x{}x{}):",
                shape[1], shape[2], shape[0]
            );
            println!("  Memory usage: {:.2} MB", estimate.total_mb);
            println!(
                "  Per image: {:.2} KB",
                estimate.average_tensor_bytes as f32 / 1024.0
            );
        }

        // Test optimal batch size calculation
        println!("\nOptimal batch sizes for 8GB GPU:");
        for shape in &shapes {
            let optimal_batch = crate::memory::MemoryOptimizer::calculate_optimal_batch_size(
                shape, 8192, // 8GB in MB
                0.8,  // 80% safety factor
            );

            println!(
                "  {}x{}x{}: {} images/batch",
                shape[1], shape[2], shape[0], optimal_batch
            );
        }

        println!("✓ Memory usage benchmark complete");
        Ok(())
    }
}

/// Main example runner
pub fn run_all_examples() -> Result<()> {
    println!("🚀 ToRSh Vision - Comprehensive Examples\n");

    // Core functionality examples
    image_classification::complete_training_pipeline()?;
    println!();

    image_classification::real_time_inference()?;
    println!();

    // Advanced CV examples
    advanced_cv::object_detection_pipeline()?;
    println!();

    advanced_cv::image_segmentation()?;
    println!();

    // Data augmentation examples
    data_augmentation::advanced_augmentation_pipeline()?;
    println!();

    data_augmentation::automatic_augmentation()?;
    println!();

    // I/O examples
    io_examples::batch_image_processing()?;
    println!();

    io_examples::memory_mapped_loading()?;
    println!();

    // Memory optimization examples
    memory_optimization::memory_efficient_training()?;
    println!();

    memory_optimization::dynamic_memory_optimization()?;
    println!();

    // Hardware acceleration examples
    hardware_acceleration::gpu_accelerated_transforms()?;
    println!();

    hardware_acceleration::mixed_precision_training()?;
    println!();

    // Complete workflows
    complete_workflows::end_to_end_classification()?;
    println!();

    complete_workflows::end_to_end_object_detection()?;
    println!();

    // Interactive visualization examples
    interactive_visualization::interactive_image_viewer()?;
    println!();

    interactive_visualization::interactive_image_gallery()?;
    println!();

    interactive_visualization::live_visualization()?;
    println!();

    // 3D visualization examples
    viz3d_examples::point_cloud_visualization()?;
    println!();

    viz3d_examples::mesh_visualization()?;
    println!();

    viz3d_examples::bounding_box_3d_visualization()?;
    println!();

    viz3d_examples::complete_3d_scene()?;
    println!();

    // Performance benchmarks
    benchmarking::transform_performance_benchmark()?;
    println!();

    benchmarking::memory_usage_benchmark()?;

    println!("\n✅ All examples completed successfully!");
    println!("🎯 ToRSh Vision provides a comprehensive computer vision framework");
    println!("   with state-of-the-art performance and ease of use.");

    Ok(())
}

/// Quick start example for new users
pub fn quick_start_example() -> Result<()> {
    println!("🚀 ToRSh Vision - Quick Start Example\n");

    // 1. Basic image processing
    println!("1. Creating and processing an image tensor...");
    let image = creation::randn(&[3, 256, 256]).unwrap(); // RGB image
    println!("   Created image with shape: {:?}", image.shape().dims());

    // 2. Apply basic transforms
    println!("2. Applying transforms...");
    let transforms = TransformBuilder::new()
        .resize((224, 224))
        .center_crop((224, 224))
        .imagenet_normalize()
        .build();

    let processed = transforms.forward(&image)?;
    println!("   Processed image shape: {:?}", processed.shape().dims());

    // 3. Show hardware capabilities
    println!("3. Checking hardware capabilities...");
    let hardware = HardwareContext::auto_detect()?;
    println!("   CUDA available: {}", hardware.cuda_available());
    println!(
        "   Mixed precision: {}",
        hardware.supports_mixed_precision()
    );

    println!("\n✅ Quick start complete!");
    println!("📚 See the full examples for more advanced usage.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_start_example() {
        assert!(quick_start_example().is_ok());
    }

    #[test]
    fn test_image_classification_basic() {
        assert!(image_classification::real_time_inference().is_ok());
    }

    #[test]
    fn test_data_augmentation_basic() {
        assert!(data_augmentation::advanced_augmentation_pipeline().is_ok());
    }

    #[test]
    fn test_memory_optimization_basic() {
        assert!(memory_optimization::memory_efficient_training().is_ok());
    }
}
