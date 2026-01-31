# ToRSh-Vision Roadmap

## Version 0.1.0-rc.1 (Current)

**Status:** Feature Complete for Initial Release
**Release Date:** 2026-01-28

### What's Included

- ✅ Core image operations (resize, crop, flip, rotate, pad)
- ✅ Basic filtering (Gaussian blur, median filter, edge detection)
- ✅ Color operations (brightness, contrast, hue, saturation, normalization)
- ✅ Transform pipeline with composition
- ✅ Pre-trained models (ResNet, VGG, AlexNet, DenseNet, MobileNet, EfficientNet)
- ✅ Detection utilities (NMS, IoU, anchor generation)
- ✅ Classification metrics
- ✅ Dataset interfaces
- ✅ Visualization utilities

### Known Issues

**TransformerBlock Tensor Slicing (ViT Models):**
- Vision Transformer models currently disabled due to complex tensor slicing issues
- Root cause: FlashMultiHeadAttention uses 5D tensor reshaping with narrow/squeeze operations
- Workaround: Use CNN-based models (ResNet, EfficientNet) for now
- Fix planned: v0.2.0 with attention mechanism refactor

**LayerNorm2d Edge Cases (ConvNeXt):**
- Fails with empty spatial dimensions (h*w=0)
- Requires minimum input size (32x32+)
- Fix planned: v0.2.0 with input validation

---

## Version 0.2.0 (Q2 2026)

**Theme:** Advanced Architectures & Training Support

### High Priority Features

#### 1. Vision Transformer Support
- [ ] Fix TransformerBlock tensor slicing issues
- [ ] Implement efficient attention mechanism
- [ ] Enable ViT-Tiny, ViT-Small, ViT-Base models
- [ ] Add DeiT (Data-efficient Image Transformers)
- [ ] Swin Transformer support

**Dependencies:**
- Improved tensor slicing API in torsh-tensor
- Better shape tracking in operations
- Benchmark against PyTorch implementation

#### 2. Detection Training Utilities
- [ ] `apply_bbox_deltas` - Apply deltas to bounding boxes
- [ ] `clip_bbox` - Clip boxes to image boundaries
- [ ] `compute_bbox_targets` - Compute training targets for R-CNN
- [ ] `filter_boxes_by_size` - Size-based box filtering
- [ ] `scale_bbox` - Scale bounding boxes for different resolutions
- [ ] `calculate_iou_matrix` - Batch IoU computation
- [ ] `convert_bbox_format` - Convert between [xyxy, xywh, cxcywh] formats

**Use Cases:**
- Faster R-CNN training
- YOLO-style detectors
- RetinaNet implementation

#### 3. Segmentation Support
- [ ] `dice_loss` - Dice loss for segmentation
- [ ] `focal_loss` - Focal loss for class imbalance
- [ ] `iou_loss` - IoU-based loss function
- [ ] `compute_segmentation_metrics` - IoU, Dice, pixel accuracy
- [ ] ROI pooling operations (`roi_pool`)

**Models:**
- U-Net architecture
- DeepLab v3+
- Mask R-CNN

#### 4. Input Validation & Error Handling
- [ ] Add minimum input size validation to all models
- [ ] Better error messages for tensor shape mismatches
- [ ] ConvNeXt input size enforcement (32x32 minimum)
- [ ] Automated shape inference for dynamic models

### Medium Priority Features

#### 5. Advanced Image Operations
- [ ] `bilateral_filter` - Edge-preserving smoothing
- [ ] `gamma_correction` - Gamma correction transform
- [ ] `rgb_to_yuv` - YUV color space conversion
- [ ] `compute_histogram` - Histogram computation
- [ ] `extract_channel` - Channel extraction utility
- [ ] Advanced interpolation modes (bicubic, Lanczos)

#### 6. Configuration-Based APIs
- [ ] `ResizeConfig` - Comprehensive resize configuration
- [ ] `CropConfig` - Advanced cropping options
- [ ] `FlipConfig` - Flip configuration
- [ ] `PaddingConfig` - Padding configuration
- [ ] `GaussianBlurConfig` - Blur parameters
- [ ] `NormalizationConfig` - Normalization options
- [ ] `HistogramConfig` - Histogram computation options

**Benefits:**
- More flexible operations
- Easier parameter tuning
- Better backward compatibility

#### 7. Batch Processing
- [ ] Batch resize support (batch_size > 1)
- [ ] Batch transform application
- [ ] Optimized batch operations with SIMD
- [ ] Parallel batch processing with Rayon

**Performance Target:**
- 3-5x speedup for batch operations
- Memory-efficient batch handling

### Low Priority Features

#### 8. Model Improvements
- [ ] Drop connect for EfficientNet training
- [ ] Stochastic depth for ResNet
- [ ] Label smoothing utilities
- [ ] Mixup/CutMix augmentation

#### 9. Advanced Metrics
- [ ] `compute_detection_metrics` - mAP, precision, recall
- [ ] Per-class metrics computation
- [ ] Confusion matrix generation
- [ ] ROC curve computation

---

## Version 0.3.0 (Q3 2026)

**Theme:** Production-Ready Training & Advanced Models

### Features

#### 1. Object Detection Models
- [ ] Faster R-CNN
- [ ] YOLO v5/v7
- [ ] RetinaNet
- [ ] DETR (DEtection TRansformer)

#### 2. Semantic Segmentation
- [ ] U-Net
- [ ] DeepLab v3+
- [ ] Mask R-CNN
- [ ] Panoptic segmentation support

#### 3. Advanced Data Augmentation
- [ ] AutoAugment
- [ ] RandAugment
- [ ] TrivialAugment
- [ ] Cutout variants (CutMix, MixUp, GridMask)

#### 4. Model Optimization
- [ ] Quantization support (int8, fp16)
- [ ] Pruning utilities
- [ ] Knowledge distillation helpers
- [ ] ONNX export

#### 5. Advanced Training Features
- [ ] Learning rate schedulers
- [ ] Gradient accumulation helpers
- [ ] Mixed precision training support
- [ ] Distributed training utilities

---

## Version 0.4.0 (Q4 2026)

**Theme:** State-of-the-Art Models & Deployment

### Features

#### 1. Modern Architectures
- [ ] ConvNeXt v2
- [ ] EfficientNet v3
- [ ] RegNet variants
- [ ] Vision Transformer variants (ViT-Large, ViT-Huge)
- [ ] CLIP-style models

#### 2. Multi-Modal Support
- [ ] Image-text models (CLIP)
- [ ] Video understanding
- [ ] 3D vision (NeRF basics)
- [ ] Point cloud processing

#### 3. Deployment Features
- [ ] TensorRT backend
- [ ] WebAssembly support
- [ ] Mobile optimization (ARM NEON)
- [ ] Real-time inference optimizations

#### 4. Advanced Operations
- [ ] Differentiable image warping
- [ ] Spatial transformer networks
- [ ] Deformable convolutions
- [ ] Attention mechanisms library

---

## Long-Term Vision (v1.0+)

### Research Features
- Neural architecture search (NAS)
- Self-supervised learning frameworks
- Few-shot learning support
- Meta-learning utilities

### Integration
- Hugging Face model hub integration
- ONNX model zoo compatibility
- PyTorch model conversion tools
- TensorFlow model import

### Performance
- GPU acceleration for all operations
- Custom CUDA kernels for critical operations
- Distributed training across multiple GPUs
- Mixed precision by default

### Ecosystem
- Comprehensive benchmark suite
- Pre-trained model zoo
- Transfer learning examples
- Production deployment guides

---

## Deferred Operations Reference

### Geometric Transformations
| Operation | Priority | Target Version |
|-----------|----------|----------------|
| `crop_with_config` | Low | v0.2.0 |
| `flip_with_config` | Low | v0.2.0 |
| `pad_with_config` | Low | v0.2.0 |
| `resize_bicubic` | Medium | v0.2.0 |
| `resize_bilinear` | Medium | v0.2.0 |
| `resize_nearest` | Low | v0.2.0 |
| `resize_with_config` | Medium | v0.2.0 |
| Batch resize (size>1) | High | v0.2.0 |

### Filtering Operations
| Operation | Priority | Target Version |
|-----------|----------|----------------|
| `bilateral_filter` | Medium | v0.2.0 |
| `conv2d` | High | v0.2.0 |
| `conv2d_with_config` | Medium | v0.2.0 |
| `gaussian_blur_with_config` | Low | v0.2.0 |

### Color Operations
| Operation | Priority | Target Version |
|-----------|----------|----------------|
| `compute_histogram` | Medium | v0.2.0 |
| `extract_channel` | Low | v0.2.0 |
| `gamma_correction` | Medium | v0.2.0 |
| `histogram_equalization_with_config` | Low | v0.2.0 |
| `normalize_imagenet` | Low | v0.2.0 |
| `normalize_with_config` | Low | v0.2.0 |
| `rgb_to_yuv` | Medium | v0.2.0 |

### Detection Operations
| Operation | Priority | Target Version |
|-----------|----------|----------------|
| `apply_bbox_deltas` | High | v0.2.0 |
| `calculate_iou_matrix` | High | v0.2.0 |
| `clip_bbox` | High | v0.2.0 |
| `compute_bbox_targets` | High | v0.2.0 |
| `convert_bbox_format` | High | v0.2.0 |
| `filter_boxes_by_size` | High | v0.2.0 |
| `roi_pool` | High | v0.2.0 |
| `scale_bbox` | High | v0.2.0 |

### Analysis & Loss
| Operation | Priority | Target Version |
|-----------|----------|----------------|
| `compute_detection_metrics` | Medium | v0.2.0 |
| `compute_segmentation_metrics` | High | v0.2.0 |
| `dice_loss` | High | v0.2.0 |
| `focal_loss` | High | v0.2.0 |
| `iou_loss` | High | v0.2.0 |

### Configuration Types
All config types (`CropConfig`, `FlipConfig`, `PaddingConfig`, `ResizeConfig`, `RotationConfig`, `EdgeDetectionConfig`, `FilteringConfig`, `GaussianBlurConfig`, `MorphologyConfig`, `ColorConfig`, `HistogramConfig`, `NormalizationConfig`, `AnchorConfig`, `BBoxFormat`, `NMSConfig`, `ROIPoolConfig`, `DiceLossConfig`, `FocalLossConfig`, `LossConfig`, `Reduction`) are targeted for v0.2.0.

---

## Contributing

See specific issues for each feature:
- Issues are tagged by version milestone
- High priority items are marked for v0.2.0
- Discussion welcome on feature design

## Feedback

Please open issues for:
- Feature requests not in this roadmap
- Priority adjustments based on use cases
- Implementation suggestions

---

## Notes

- This roadmap is subject to change based on community feedback
- Release dates are estimates and may shift
- Features may be added or removed based on technical feasibility
- Priority levels reflect ToRSh project goals, not individual use cases

**Last Updated:** 2026-01-28
**Current Version:** 0.1.0-rc.1
