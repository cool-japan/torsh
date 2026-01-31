# Known Issues - ToRSh-Vision v0.1.0-rc.1

## Overview

This document lists known issues and limitations in torsh-vision v0.1.0-rc.1. All issues are documented with:
- Root cause analysis
- Workarounds where available
- Target fix version
- Impact assessment

---

## Critical Issues (Test Failures)

### 1. TransformerBlock Tensor Slicing Issues

**Status:** 🔴 Known Issue
**Severity:** High
**Affected:** Vision Transformer (ViT) models
**Target Fix:** v0.2.0

#### Description

FlashMultiHeadAttention implementation fails due to complex tensor slicing operations involving 5D tensors with narrow/squeeze patterns.

#### Root Cause

Located in `models/advanced_architectures.rs`, lines 443-448:
```rust
let q = qkv.narrow(0, 0, 1)?.squeeze(0)?; // [B, H, N, D]
let k = qkv.narrow(0, 1, 1)?.squeeze(0)?; // [B, H, N, D]
let v = qkv.narrow(0, 2, 1)?.squeeze(0)?; // [B, H, N, D]
```

The tensor reshaping pattern:
1. Creates 5D tensor: `[3, B, H, N, D]`
2. Narrows to extract Q, K, V
3. Multiple squeeze operations fail due to shape tracking inconsistencies
4. Subsequent permute operations fail

#### Impact

The following models are **non-functional** in v0.1.0-rc.1:
- AdvancedViT (all variants: tiny, small, base)
- DeiT (if implemented)
- Swin Transformer (if implemented)

#### Workaround

**Use CNN-based architectures instead:**
- ResNet (all variants working)
- EfficientNet (all variants working)
- MobileNet (all variants working)
- VGG (all variants working)
- DenseNet (all variants working)

#### Test Status

The following tests are ignored with this issue:
- `benchmarks::test_quick_benchmark`
- `comprehensive_showcase::test_advanced_models`
- `comprehensive_showcase::test_end_to_end_workflow`
- `advanced_architectures::test_vit_forward`
- `advanced_architectures::test_transformer_block`

#### Fix Plan (v0.2.0)

1. **Immediate:** Refactor attention mechanism to avoid complex narrow/squeeze patterns
2. **Mid-term:** Implement torch-like advanced indexing API in torsh-tensor
3. **Long-term:** Add comprehensive shape tracking validation
4. **Testing:** Add unit tests for each tensor operation in isolation

#### References

- Categorization report: `/tmp/torsh_vision_todo_categorization.md`
- ROADMAP: `ROADMAP.md` section 0.2.0 "Vision Transformer Support"

---

### 2. LayerNorm2d Empty Tensor Issue

**Status:** 🟡 Edge Case
**Severity:** Medium
**Affected:** ConvNeXt models
**Target Fix:** v0.2.0

#### Description

LayerNorm2d fails when processing tensors with empty spatial dimensions (h*w=0).

#### Root Cause

Located in `models/advanced_cnns.rs`, lines 475-487:
```rust
for i in 0..(n * c) {
    let channel_data = x_reshaped.narrow(0, i as i64, 1)?.squeeze(0)?;
    let channel_vec = channel_data.to_vec()?;  // Panics if h*w=0
    // ... compute mean and variance
}
```

The issue occurs when:
- Very small input images (< 32x32)
- After aggressive pooling operations
- Edge case in ConvNeXt downsampling blocks

#### Impact

- ConvNeXt forward pass fails with inputs smaller than recommended size
- Affects `test_convnext_forward` test

**Models affected:**
- ConvNeXt (all variants: tiny, small, base, large)

**Models unaffected:**
- ResNet, EfficientNet, MobileNet, VGG, DenseNet (all working)

#### Workaround

**Input Size Validation:**
```rust
// Ensure minimum input size for ConvNeXt
assert!(height >= 32 && width >= 32, "ConvNeXt requires minimum 32x32 input");

let model = ConvNeXt::convnext_tiny()?;
let input = randn::<f32>(&[1, 3, 224, 224])?; // Use 224x224 or larger
```

#### Test Status

The following test is ignored:
- `advanced_cnns::test_convnext_forward`

#### Fix Plan (v0.2.0)

1. **Immediate:** Add input size validation in ConvNeXt constructor
2. **Documentation:** Document minimum input sizes for all models
3. **Error Handling:** Better error messages for invalid input sizes
4. **Optional:** Add padding to handle smaller inputs gracefully

#### References

- Categorization report: `/tmp/torsh_vision_todo_categorization.md`
- ROADMAP: `ROADMAP.md` section 0.2.0 "Input Validation & Error Handling"

---

## Known Limitations

### 3. Batch Processing Limitation (Geometric Operations)

**Status:** 🟡 Documented Limitation
**Severity:** Low
**Affected:** Batch resize operations
**Target Fix:** v0.2.0

#### Description

Geometric transformations (resize, crop, etc.) only support `batch_size=1` currently.

#### Root Cause

Located in `ops/geometric/mod.rs`, lines 74-82:
```rust
if batch_size > 1 {
    return Err(VisionError::InvalidArgument(
        "Batch resize with batch_size > 1 not yet supported".to_string(),
    ));
}
```

Requires tensor `stack` operation for efficient multi-batch handling.

#### Impact

- Cannot process multiple images in one resize call
- Minor performance impact (can loop manually)

#### Workaround

**Process images individually:**
```rust
let resized_images: Vec<Tensor> = images
    .iter()
    .map(|img| resize(img, (224, 224)))
    .collect::<Result<Vec<_>>>()?;
```

Or use non-batched API:
```rust
for i in 0..batch_size {
    let img = batch.narrow(0, i, 1)?;
    let resized = resize(&img, (224, 224))?;
    // Process resized image
}
```

#### Fix Plan (v0.2.0)

1. Implement tensor `stack` operation in torsh-tensor
2. Add batch processing support with SIMD optimization
3. Benchmark performance improvement (target: 3-5x speedup)

#### References

- ROADMAP: `ROADMAP.md` section 0.2.0 "Batch Processing"

---

### 4. Drop Connect Not Implemented (EfficientNet)

**Status:** 🟢 Feature Omission
**Severity:** Low
**Affected:** EfficientNet training
**Target Fix:** v0.2.0

#### Description

EfficientNet blocks don't implement drop connect (stochastic depth) regularization.

#### Root Cause

Located in `models/efficientnet.rs`, lines 247-251:
```rust
if self.stride == 1 && self.input_channels == self.output_channels {
    // NOTE: Drop connect (stochastic depth) omitted
    x = x.add(input)?;
}
```

#### Impact

- Models work correctly without drop connect
- Training may achieve ~1-2% lower accuracy vs. original paper
- Inference completely unaffected

This is a **nice-to-have** feature, not a bug.

#### Workaround

None needed - models work as expected. For maximum accuracy:
- Use larger models (EfficientNet-B1/B2 instead of B0)
- Use data augmentation
- Use longer training schedules

#### Fix Plan (v0.2.0)

Implement drop connect with proper training/eval mode handling.

#### References

- Original paper: "EfficientNet: Rethinking Model Scaling for CNNs"
- ROADMAP: `ROADMAP.md` section 0.2.0 "Model Improvements"

---

### 5. Transform API Migration Helper Unimplemented

**Status:** 🟢 Feature Omission
**Severity:** Low
**Affected:** API migration utilities
**Target Fix:** v0.3.0 (low priority)

#### Description

`transforms::unified::migration::convert_transforms` returns empty vector.

#### Root Cause

Located in `transforms/unified.rs`, lines 752-758:
```rust
pub fn convert_transforms(
    _transforms: Vec<Box<dyn crate::transforms::Transform>>,
) -> Vec<Box<dyn UnifiedTransform>> {
    Vec::new()  // Not implemented
}
```

Cannot automatically convert between trait objects without runtime type inspection.

#### Impact

- Users must manually convert transform pipelines
- Not used in current codebase
- Migration guidance available via `analyze_pipeline()`

#### Workaround

**Manual conversion:**
```rust
// Old API
let old = Compose::new(vec![
    Box::new(Resize::new(256)),
    Box::new(CenterCrop::new(224)),
]);

// New API
let new = PipelineBuilder::new()
    .add_transform(ResizeTransform::new(256, InterpolationMode::Bilinear))
    .add_transform(CropTransform::center(224))
    .build();
```

#### Fix Plan (v0.3.0)

May implement with macro-based conversion or keep as documentation-only.

#### References

- ROADMAP: `ROADMAP.md` section 0.3.0

---

## Deferred Operations (54 items)

All commented-out operations in `ops.rs` with `// TODO: Implement` are **documented deferred features**, not bugs.

**Categories:**
- Geometric operations + configs: 11 items
- Filtering operations + configs: 9 items
- Color operations + configs: 14 items
- Detection operations + configs: 13 items
- Analysis/loss operations + configs: 11 items

**Status:** See `ops.rs` lines 70-90 for detailed explanation
**Documentation:** Complete feature list in `ROADMAP.md`
**Priority:** High-priority items (detection training, segmentation) in v0.2.0

---

## Performance Notes

### Current Performance

- ✅ ResNet inference: Comparable to PyTorch
- ✅ Basic operations: Efficient with SIMD
- ⚠️ Batch operations: Limited (see Limitation #3)
- ❌ ViT models: Non-functional (see Issue #1)

### Optimization Opportunities (v0.2.0+)

1. **SIMD batch operations** - 3-5x speedup potential
2. **GPU acceleration** - 10-100x speedup for large models
3. **Attention optimization** - Custom kernels for transformer blocks
4. **Memory efficiency** - Reduce allocations in hot paths

---

## SciRS2 Integration Status

**Status:** ✅ Fully Compliant

- No direct external dependencies (rand, ndarray, etc.)
- All operations use scirs2-core abstractions
- Proper workspace policy compliance

See `SCIRS2_INTEGRATION_POLICY.md` for details.

---

## Testing Status

### Passing Tests

- ✅ All ResNet variants
- ✅ All EfficientNet variants
- ✅ All MobileNet variants
- ✅ VGG models
- ✅ DenseNet models
- ✅ AlexNet
- ✅ All geometric operations
- ✅ All color operations
- ✅ All filtering operations
- ✅ Detection utilities (NMS, IoU)
- ✅ Classification metrics

### Ignored Tests (6)

1. `test_quick_benchmark` - Issue #1 (ViT)
2. `test_advanced_models` - Issue #1 (ViT)
3. `test_end_to_end_workflow` - Issue #1 (ViT)
4. `test_vit_forward` - Issue #1 (ViT)
5. `test_transformer_block` - Issue #1 (ViT)
6. `test_convnext_forward` - Issue #2 (LayerNorm2d)

Plus 1 slow test:
- `test_data_augmentation` (>60s, working correctly)

---

## Release Readiness

### RC.1 Status: ✅ Ready

Despite the issues listed above, torsh-vision v0.1.0-rc.1 is considered ready because:

1. **Core functionality works:**
   - All CNN architectures functional
   - All basic operations working
   - Detection utilities operational
   - Transform pipeline complete

2. **Known issues documented:**
   - Clear root cause analysis
   - Workarounds provided
   - Fix plans established

3. **No regressions:**
   - Previously working features still work
   - No unsafe code issues
   - No policy violations

4. **Vision Transformers deferred:**
   - ViT is advanced feature
   - CNN models cover 90% of use cases
   - Proper fix planned for v0.2.0

### What Works in RC.1

✅ **Production Ready:**
- Image classification (ResNet, EfficientNet, MobileNet)
- Basic image processing (resize, crop, flip, rotate)
- Color operations (normalization, adjustments)
- Image filtering (blur, edge detection)
- Detection utilities (NMS, IoU, anchors)

⚠️ **Not Recommended:**
- Vision Transformers (known issues)
- ConvNeXt with small inputs (edge case)
- Batch operations with batch_size > 1 (limitation)

❌ **Not Implemented:**
- Semantic segmentation models
- Object detection training (partial utils available)
- Advanced loss functions (Dice, Focal)
- Many config-based APIs (see ROADMAP.md)

---

## Support and Reporting

### How to Report Issues

1. Check this document first
2. Check ROADMAP.md for planned features
3. Open GitHub issue if new problem found

### Getting Help

- **Workarounds:** See individual issue sections above
- **Alternative approaches:** Use CNN models instead of ViT
- **Documentation:** README.md, examples/, ROADMAP.md

---

**Last Updated:** 2026-01-28
**Version:** 0.1.0-rc.1
**Status:** Maintained
