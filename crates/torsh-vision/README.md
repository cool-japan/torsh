# torsh-vision

Computer vision utilities and models for ToRSh, leveraging scirs2-vision for optimized image operations.

## Overview

This crate provides comprehensive computer vision functionality:

- **Image I/O**: Loading, saving, and format conversion
- **Transforms**: Data augmentation and preprocessing
- **Datasets**: Common vision datasets (ImageNet, COCO, etc.)
- **Models**: Pre-trained vision models
- **Operations**: Image processing and computer vision algorithms
- **Visualization**: Image display and annotation utilities

Note: This crate integrates with scirs2-vision for optimized image processing operations.

## Usage

### Image I/O

```rust
use torsh_vision::prelude::*;

// Load images
let image = read_image("path/to/image.jpg")?;
let batch = read_image_batch(&["img1.jpg", "img2.jpg", "img3.jpg"])?;

// Save images
write_image(&tensor, "output.png")?;
write_video(&frames, "output.mp4", fps=30)?;

// Format conversion
let rgb = bgr_to_rgb(&bgr_image)?;
let gray = rgb_to_grayscale(&rgb_image)?;
let hsv = rgb_to_hsv(&rgb_image)?;
```

### Transforms

```rust
use torsh_vision::transforms::*;

// Create transform pipeline
let transform = Compose::new(vec![
    Box::new(Resize::new(256)),
    Box::new(CenterCrop::new(224)),
    Box::new(ToTensor::new()),
    Box::new(Normalize::new(
        vec![0.485, 0.456, 0.406],  // ImageNet mean
        vec![0.229, 0.224, 0.225],  // ImageNet std
    )),
]);

let transformed = transform.apply(&image)?;

// Data augmentation
let augment = Compose::new(vec![
    Box::new(RandomResizedCrop::new(224, scale=(0.08, 1.0))),
    Box::new(RandomHorizontalFlip::new(0.5)),
    Box::new(ColorJitter::new(0.4, 0.4, 0.4, 0.1)),
    Box::new(RandomErasing::new(0.2)),
    Box::new(ToTensor::new()),
    Box::new(Normalize::imagenet()),
]);

// Advanced augmentations
let mixup = MixUp::new(alpha=1.0);
let cutmix = CutMix::new(alpha=1.0);
let augmix = AugMix::new(severity=3, width=3, depth=-1, alpha=1.0);
```

### Datasets

```rust
use torsh_vision::datasets::*;

// ImageNet dataset
let imagenet = ImageNet::new(
    root="./data/imagenet",
    split="train",
    transform=Some(transform),
    download=false,
)?;

// COCO dataset
let coco = COCODetection::new(
    root="./data/coco",
    split="train2017",
    transform=Some(transform),
    target_transform=None,
)?;

// CIFAR datasets
let cifar10 = CIFAR10::new(
    root="./data",
    train=true,
    transform=Some(transform),
    download=true,
)?;

// Custom folder dataset
let dataset = ImageFolder::new(
    root="./data/custom",
    transform=Some(transform),
    extensions=Some(vec!["jpg", "jpeg", "png"]),
)?;

// Video dataset
let video_dataset = VideoFolder::new(
    root="./data/videos",
    clip_len=16,
    frame_interval=1,
    num_clips=1,
    transform=Some(video_transform),
)?;
```

### Pre-trained Models

```rust
use torsh_vision::models::*;

// Classification models
let resnet = resnet50(pretrained=true, num_classes=1000)?;
let efficientnet = efficientnet_b0(pretrained=true)?;
let vit = vit_base_patch16_224(pretrained=true)?;

// Object detection
let faster_rcnn = fasterrcnn_resnet50_fpn(
    pretrained=true,
    num_classes=91,
    pretrained_backbone=true,
)?;

// Segmentation
let deeplabv3 = deeplabv3_resnet101(
    pretrained=true,
    num_classes=21,
    aux_loss=true,
)?;

// Feature extraction
let features = resnet.features(&input)?;
let backbone = create_feature_extractor(
    &resnet,
    return_nodes=vec!["layer1", "layer2", "layer3", "layer4"],
)?;
```

### Image Operations

```rust
use torsh_vision::ops::*;

// Basic operations (leveraging scirs2-vision)
let resized = resize(&image, size=[224, 224], interpolation="bilinear")?;
let cropped = crop(&image, top=10, left=10, height=200, width=200)?;
let flipped = hflip(&image)?;
let rotated = rotate(&image, angle=45.0, fill=vec![0, 0, 0])?;

// Filtering
let blurred = gaussian_blur(&image, kernel_size=[5, 5], sigma=[1.0, 1.0])?;
let sharpened = adjust_sharpness(&image, sharpness_factor=2.0)?;
let edge = sobel_edge_detection(&image)?;

// Color adjustments
let bright = adjust_brightness(&image, brightness_factor=1.5)?;
let contrast = adjust_contrast(&image, contrast_factor=1.5)?;
let saturated = adjust_saturation(&image, saturation_factor=1.5)?;

// Advanced operations
let slic = slic_superpixels(&image, n_segments=100, compactness=10.0)?;
let optical_flow = dense_optical_flow(&frame1, &frame2, method="farneback")?;
```

### Object Detection Utilities

```rust
use torsh_vision::utils::*;

// Bounding box operations
let iou = box_iou(&boxes1, &boxes2)?;
let nms_keep = nms(&boxes, &scores, iou_threshold=0.5)?;
let converted = box_convert(&boxes, in_fmt="xyxy", out_fmt="cxcywh")?;

// Anchor generation
let anchors = AnchorGenerator::new(
    sizes=vec![vec![32], vec![64], vec![128], vec![256], vec![512]],
    aspect_ratios=vec![vec![0.5, 1.0, 2.0]; 5],
)?;

// ROI operations
let roi_pool = roi_pool(&features, &boxes, output_size=[7, 7], spatial_scale=0.25)?;
let roi_align = roi_align(&features, &boxes, output_size=[7, 7], spatial_scale=0.25)?;
```

### Visualization

```rust
use torsh_vision::utils::*;

// Draw bounding boxes
let annotated = draw_bounding_boxes(
    &image,
    &boxes,
    labels=Some(&labels),
    colors=None,
    width=2,
)?;

// Draw segmentation masks
let masked = draw_segmentation_masks(
    &image,
    &masks,
    alpha=0.7,
    colors=None,
)?;

// Draw keypoints
let keypoint_image = draw_keypoints(
    &image,
    &keypoints,
    connectivity=Some(&COCO_PERSON_SKELETON),
    colors=None,
    radius=3,
)?;

// Create image grid
let grid = make_grid(
    &tensor_list,
    nrow=8,
    padding=2,
    normalize=true,
    value_range=None,
)?;

// Save visualization
save_image(&grid, "visualization.png")?;
```

### Video Processing

```rust
use torsh_vision::video::*;

// Read video
let video = read_video("input.mp4", start_pts=0, end_pts=None)?;
let frames = video.frames;  // Tensor of shape [T, C, H, W]
let audio = video.audio;    // Optional audio tensor

// Write video
write_video(
    "output.mp4",
    &frames,
    fps=30.0,
    video_codec="h264",
    audio=audio,
    audio_codec="aac",
)?;

// Video transforms
let video_transform = VideoCompose::new(vec![
    Box::new(VideoResize::new(256)),
    Box::new(VideoCenterCrop::new(224)),
    Box::new(VideoNormalize::imagenet()),
]);
```

### Feature Extraction and Similarity

```rust
// Extract features
let feature_extractor = create_feature_extractor(
    &model,
    return_nodes=vec!["avgpool"],
)?;
let features = feature_extractor(&images)?;

// Image similarity
let similarity = cosine_similarity(&features1, &features2)?;

// Image retrieval
let retrieval_system = ImageRetrieval::new(feature_extractor);
retrieval_system.add_images(&database_images)?;
let similar_images = retrieval_system.search(&query_image, top_k=10)?;
```

## Integration with SciRS2

This crate leverages scirs2-vision for:
- Optimized image processing operations
- Efficient data augmentation
- Hardware-accelerated transforms
- Computer vision algorithms

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.