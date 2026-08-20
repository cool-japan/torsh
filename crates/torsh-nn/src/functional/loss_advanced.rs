//! Advanced loss functions framework
//!
//! This module provides a comprehensive framework for creating custom loss functions
//! with composable building blocks, different reduction modes, and validation utilities.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
use torsh_core::reduction::Reduction;
use torsh_functional::loss::common::ReductionExt;
// =============================================================================
// REDUCTION MODES AND FRAMEWORK
// =============================================================================

/// Trait for custom loss functions
pub trait CustomLoss: Send + Sync {
    /// Compute the loss between predictions and targets
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;

    /// Get the reduction mode for this loss function
    fn reduction(&self) -> &Reduction {
        &Reduction::Mean
    }

    /// Validate inputs before computing loss
    fn validate_inputs(&self, predictions: &Tensor, targets: &Tensor) -> Result<()> {
        let pred_binding = predictions.shape();
        let pred_shape = pred_binding.dims();
        let target_binding = targets.shape();
        let target_shape = target_binding.dims();

        if pred_shape[0] != target_shape[0] {
            return Err(TorshError::ShapeMismatch {
                expected: vec![target_shape[0]],
                got: vec![pred_shape[0]],
            });
        }

        Ok(())
    }

    /// Apply loss function with validation and reduction
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        self.validate_inputs(predictions, targets)?;
        let raw_loss = self.forward(predictions, targets)?;
        let batch_size = predictions.shape().dims()[0];
        self.reduction().apply(&raw_loss, Some(batch_size))
    }
}

// =============================================================================
// ADVANCED LOSS IMPLEMENTATIONS
// =============================================================================

/// Smooth L1 Loss (Huber Loss)
///
/// Combines MSE and MAE - less sensitive to outliers than MSE
pub struct SmoothL1Loss {
    beta: f32,
    reduction: Reduction,
}

impl SmoothL1Loss {
    pub fn new(beta: f32, reduction: Reduction) -> Self {
        Self { beta, reduction }
    }
}

impl CustomLoss for SmoothL1Loss {
    /// Delegates to the functional kernel, which computes the same two arms as
    /// the `to_vec()`/`from_vec` loop this used to run — but as a composition of
    /// recording ops, so the result stays on the autograd graph.
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::smooth_l1_loss(predictions, targets, self.beta, Reduction::None)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Dice Loss
///
/// Commonly used for segmentation tasks
pub struct DiceLoss {
    smooth: f32,
    reduction: Reduction,
}

impl DiceLoss {
    pub fn new(smooth: f32, reduction: Reduction) -> Self {
        Self { smooth, reduction }
    }
}

impl CustomLoss for DiceLoss {
    /// Applies `sigmoid` (these predictions are raw logits) and delegates to the
    /// functional kernel, which computes the same coefficient with recording
    /// tensor ops instead of a `to_vec()` accumulation loop.
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Dice loss: 1 - (2 * |intersection| + smooth) / (|pred| + |target| + smooth)
        let probs = predictions.sigmoid()?;
        crate::functional::loss::dice_loss(&probs, targets, self.smooth, Reduction::None)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// IoU Loss (Intersection over Union Loss)
///
/// Also commonly used for segmentation
pub struct IoULoss {
    smooth: f32,
    reduction: Reduction,
}

impl IoULoss {
    pub fn new(smooth: f32, reduction: Reduction) -> Self {
        Self { smooth, reduction }
    }
}

impl CustomLoss for IoULoss {
    /// The same soft-IoU the `to_vec()` loop computed, expressed with recording
    /// ops so the loss stays differentiable:
    /// `1 - (sum(p*t) + smooth) / (sum(p + t - p*t) + smooth)`.
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // IoU loss: 1 - |intersection| / |union|
        let probs = predictions.sigmoid()?;
        let overlap = probs.mul_op(targets)?;

        let intersection = overlap.sum()?;
        let union = probs.add(targets)?.sub(&overlap)?.sum()?;

        let iou = intersection
            .add_scalar(self.smooth)?
            .div(&union.add_scalar(self.smooth)?)?;

        iou.neg()?.add_scalar(1.0)?.view(&[1])
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Weighted Loss
///
/// Applies class weights to any base loss function
pub struct WeightedLoss<L: CustomLoss> {
    base_loss: L,
    weights: Vec<f32>,
    reduction: Reduction,
}

impl<L: CustomLoss> WeightedLoss<L> {
    pub fn new(base_loss: L, weights: Vec<f32>, reduction: Reduction) -> Self {
        Self {
            base_loss,
            weights,
            reduction,
        }
    }
}

impl<L: CustomLoss> CustomLoss for WeightedLoss<L> {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let base_loss = self.base_loss.forward(predictions, targets)?;

        if self.weights.is_empty() {
            return Err(TorshError::InvalidArgument(
                "WeightedLoss needs at least one weight".to_string(),
            ));
        }

        // Apply class weights (simplified implementation)
        // In practice, this would need proper indexing based on target classes.
        // The weights cycle over the flattened loss, so they are a *constant*
        // tensor of the same shape and the scaling is a recording multiply --
        // reading the loss out with `to_vec()` and rebuilding it, as this used
        // to do, detached whatever `base_loss` had recorded.
        let shape_binding = base_loss.shape();
        let dims = shape_binding.dims();
        let weight_data: Vec<f32> = (0..shape_binding.numel())
            .map(|i| self.weights[i % self.weights.len()])
            .collect();
        let weight_tensor = Tensor::from_data(weight_data, dims.to_vec(), base_loss.device())?;

        base_loss.mul_op(&weight_tensor)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Combined Loss
///
/// Linearly combines multiple loss functions
pub struct CombinedLoss {
    losses: Vec<Box<dyn CustomLoss>>,
    weights: Vec<f32>,
    reduction: Reduction,
}

impl CombinedLoss {
    pub fn new(losses: Vec<Box<dyn CustomLoss>>, weights: Vec<f32>, reduction: Reduction) -> Self {
        assert_eq!(
            losses.len(),
            weights.len(),
            "Number of losses must match number of weights"
        );
        Self {
            losses,
            weights,
            reduction,
        }
    }
}

impl CustomLoss for CombinedLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let mut total_loss = Tensor::from_data(vec![0.0], vec![1], predictions.device())?;

        for (loss_fn, &weight) in self.losses.iter().zip(self.weights.iter()) {
            let loss_val = loss_fn.forward(predictions, targets)?;
            let weighted_loss = loss_val.mul_scalar(weight)?;
            total_loss = total_loss.add(&weighted_loss)?;
        }

        Ok(total_loss)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Adaptive Loss
///
/// Adjusts loss weights based on training progress or performance
pub struct AdaptiveLoss<L: CustomLoss> {
    base_loss: L,
    adaptation_factor: f32,
    current_weight: f32,
    reduction: Reduction,
}

impl<L: CustomLoss> AdaptiveLoss<L> {
    pub fn new(base_loss: L, adaptation_factor: f32, reduction: Reduction) -> Self {
        Self {
            base_loss,
            adaptation_factor,
            current_weight: 1.0,
            reduction,
        }
    }

    pub fn update_weight(&mut self, performance_metric: f32) {
        // Adapt weight based on performance (e.g., validation accuracy)
        // Higher performance -> lower weight (easier examples)
        self.current_weight = 1.0 + self.adaptation_factor * (1.0 - performance_metric);
    }
}

impl<L: CustomLoss> CustomLoss for AdaptiveLoss<L> {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let base_loss = self.base_loss.forward(predictions, targets)?;
        base_loss.mul_scalar(self.current_weight)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

// =============================================================================
// LOSS FUNCTION BUILDER
// =============================================================================

/// Loss Function Builder
///
/// Provides a fluent interface for constructing complex loss functions
pub struct LossBuilder {
    reduction: Reduction,
}

impl LossBuilder {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    pub fn smooth_l1(self, beta: f32) -> SmoothL1Loss {
        SmoothL1Loss::new(beta, self.reduction)
    }

    pub fn dice(self, smooth: f32) -> DiceLoss {
        DiceLoss::new(smooth, self.reduction)
    }

    pub fn iou(self, smooth: f32) -> IoULoss {
        IoULoss::new(smooth, self.reduction)
    }

    pub fn weighted<L: CustomLoss>(self, base_loss: L, weights: Vec<f32>) -> WeightedLoss<L> {
        WeightedLoss::new(base_loss, weights, self.reduction)
    }

    pub fn adaptive<L: CustomLoss>(self, base_loss: L, adaptation_factor: f32) -> AdaptiveLoss<L> {
        AdaptiveLoss::new(base_loss, adaptation_factor, self.reduction)
    }

    pub fn combined(self, losses: Vec<Box<dyn CustomLoss>>, weights: Vec<f32>) -> CombinedLoss {
        CombinedLoss::new(losses, weights, self.reduction)
    }
}

impl Default for LossBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PRE-BUILT LOSS FUNCTION STRUCTS
// =============================================================================

/// Categorical Cross Entropy Loss
pub struct CategoricalCrossEntropy {
    pub weight: Option<Tensor>,
    pub reduction: Reduction,
}

impl CategoricalCrossEntropy {
    pub fn new(weight: Option<Tensor>, reduction: Reduction) -> Self {
        Self { weight, reduction }
    }
}

impl CustomLoss for CategoricalCrossEntropy {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::cross_entropy(
            predictions,
            &targets.cast_i64()?,
            self.weight.as_ref(),
            Reduction::None,
            None,
        )
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Binary Cross Entropy Loss
pub struct BinaryCrossEntropy {
    pub weight: Option<Tensor>,
    pub reduction: Reduction,
}

impl BinaryCrossEntropy {
    pub fn new(weight: Option<Tensor>, reduction: Reduction) -> Self {
        Self { weight, reduction }
    }
}

impl CustomLoss for BinaryCrossEntropy {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::binary_cross_entropy(
            predictions,
            targets,
            self.weight.as_ref(),
            Reduction::None,
        )
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Focal Loss
pub struct FocalLoss {
    pub alpha: Option<f32>,
    pub gamma: f32,
    pub reduction: Reduction,
}

impl FocalLoss {
    pub fn new(alpha: Option<f32>, gamma: f32, reduction: Reduction) -> Self {
        Self {
            alpha,
            gamma,
            reduction,
        }
    }
}

impl CustomLoss for FocalLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::focal_loss(
            predictions,
            &targets.cast_i64()?,
            self.alpha,
            self.gamma,
            Reduction::None,
        )
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Huber Loss (also known as Smooth L1 Loss)
pub struct HuberLoss {
    pub delta: f32,
    pub reduction: Reduction,
}

impl HuberLoss {
    pub fn new(delta: f32, reduction: Reduction) -> Self {
        Self { delta, reduction }
    }
}

impl CustomLoss for HuberLoss {
    /// Delegates to the functional kernel. The arms are the same function —
    /// `delta * |d| - 0.5 * delta^2` is `delta * (|d| - 0.5 * delta)`, and at the
    /// switch-over point `|d| == delta` both arms evaluate to `0.5 * delta^2`, so
    /// the strict-versus-inclusive comparison the loop used makes no difference
    /// to the value.
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::huber_loss(predictions, targets, self.delta, Reduction::None)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Hinge Loss
pub struct HingeLoss {
    pub margin: f32,
    pub reduction: Reduction,
}

impl HingeLoss {
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }
}

impl CustomLoss for HingeLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Hinge loss: max(0, margin - target * prediction)
        let margin_tensor = torsh_tensor::creation::full_like(predictions, self.margin)?;
        let product = predictions.mul_op(targets)?;
        let diff = margin_tensor.sub(&product)?;
        let zero_tensor = torsh_tensor::creation::zeros_like(predictions)?;
        diff.maximum(&zero_tensor)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// KL Divergence Loss
pub struct KLDivLoss {
    pub log_target: bool,
    pub reduction: Reduction,
}

impl KLDivLoss {
    pub fn new(log_target: bool, reduction: Reduction) -> Self {
        Self {
            log_target,
            reduction,
        }
    }
}

impl CustomLoss for KLDivLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::kl_div(predictions, targets, Reduction::None, self.log_target)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Mean Squared Error Loss
pub struct MSELoss {
    pub reduction: Reduction,
}

impl MSELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl CustomLoss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::mse_loss(predictions, targets, Reduction::None)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// L1 Loss (Mean Absolute Error)
pub struct L1Loss {
    pub reduction: Reduction,
}

impl L1Loss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl CustomLoss for L1Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions.sub(targets)?;
        diff.abs()
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Negative Log Likelihood Loss
pub struct NLLLoss {
    pub weight: Option<Tensor>,
    pub ignore_index: Option<i64>,
    pub reduction: Reduction,
}

impl NLLLoss {
    pub fn new(weight: Option<Tensor>, ignore_index: Option<i64>, reduction: Reduction) -> Self {
        Self {
            weight,
            ignore_index,
            reduction,
        }
    }
}

impl CustomLoss for NLLLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        crate::functional::loss::nll_loss(
            predictions,
            &targets.cast_i64()?,
            self.weight.as_ref(),
            self.ignore_index,
            Reduction::None,
        )
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Triplet Margin Loss
pub struct TripletMarginLoss {
    pub margin: f32,
    pub p: f32,
    pub reduction: Reduction,
}

impl TripletMarginLoss {
    pub fn new(margin: f32, p: f32, reduction: Reduction) -> Self {
        Self {
            margin,
            p,
            reduction,
        }
    }
}

impl CustomLoss for TripletMarginLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Simplified triplet margin loss - in practice would need anchor, positive, negative
        let _ = targets; // Suppress warning
        crate::functional::loss::triplet_margin_loss(
            predictions,
            predictions,
            predictions,
            self.margin,
            self.p,
            Reduction::None,
        )
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

/// Cosine Embedding Loss
pub struct CosineEmbeddingLoss {
    pub margin: f32,
    pub reduction: Reduction,
}

impl CosineEmbeddingLoss {
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }
}

impl CustomLoss for CosineEmbeddingLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Simplified cosine embedding loss
        let _ = targets; // Suppress warning
        crate::functional::loss::contrastive_loss(
            predictions,
            predictions,
            predictions,
            self.margin,
            Reduction::None,
        )
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

// =============================================================================
// LOSS FACTORY
// =============================================================================

/// Pre-built loss function factory
pub struct LossFactory;

impl LossFactory {
    /// Create a focal loss for imbalanced classification
    pub fn focal_loss(alpha: f32, gamma: f32, reduction: Reduction) -> impl CustomLoss {
        struct FocalLossImpl {
            alpha: f32,
            gamma: f32,
            reduction: Reduction,
        }

        impl CustomLoss for FocalLossImpl {
            fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
                crate::functional::loss::focal_loss(
                    predictions,
                    &targets.cast_i64()?,
                    Some(self.alpha),
                    self.gamma,
                    Reduction::None,
                )
            }

            fn reduction(&self) -> &Reduction {
                &self.reduction
            }
        }

        FocalLossImpl {
            alpha,
            gamma,
            reduction,
        }
    }

    /// Create a label smoothing cross entropy loss
    pub fn label_smoothing_ce(smoothing: f32, reduction: Reduction) -> impl CustomLoss {
        struct LabelSmoothingCE {
            smoothing: f32,
            reduction: Reduction,
        }

        impl CustomLoss for LabelSmoothingCE {
            /// The smoothed target distribution is a *constant* matrix derived
            /// from `targets`, so the whole loss is one recording multiply and a
            /// row sum. The predecessor read `log_softmax`'s output back with
            /// `to_vec()` and rebuilt the per-sample losses with
            /// `Tensor::from_vec`, detaching a numerically careful forward pass
            /// from the graph it was computed on.
            fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
                // Label smoothing: mix true labels with uniform distribution
                // smoothed_target = (1 - smoothing) * target + smoothing / num_classes

                let shape_binding = predictions.shape();
                let dims = shape_binding.dims();
                let (batch_size, num_classes) = match dims {
                    [batch, classes] => (*batch, *classes),
                    _ => {
                        return Err(TorshError::InvalidShape(format!(
                            "label_smoothing_ce expects 2-D predictions [N, C], got {dims:?}"
                        )))
                    }
                };

                let uniform_prob = self.smoothing / num_classes as f32;
                let true_prob = 1.0 - self.smoothing;

                // Apply log softmax to predictions
                let log_probs = predictions.log_softmax(-1)?;

                let target_data = targets.to_vec()?;
                if target_data.len() < batch_size {
                    return Err(TorshError::InvalidShape(format!(
                        "label_smoothing_ce expects one target per sample: predictions \
                         have {batch_size} rows but targets have {} entries",
                        target_data.len()
                    )));
                }

                // Constant smoothed-target matrix. An out-of-range class simply
                // never matches, leaving that row uniformly smoothed -- exactly
                // what the element-wise predecessor did.
                let mut smoothed = vec![uniform_prob; batch_size * num_classes];
                for (b, &target_value) in target_data.iter().take(batch_size).enumerate() {
                    let true_class = target_value as usize;
                    if true_class >= num_classes {
                        continue;
                    }
                    if let Some(slot) = smoothed.get_mut(b * num_classes + true_class) {
                        *slot = true_prob + uniform_prob;
                    }
                }
                let smoothed_tensor = Tensor::from_data(
                    smoothed,
                    vec![batch_size, num_classes],
                    predictions.device(),
                )?;

                log_probs
                    .mul_op(&smoothed_tensor)?
                    .sum_dim(&[-1], false)?
                    .neg()
            }

            fn reduction(&self) -> &Reduction {
                &self.reduction
            }
        }

        LabelSmoothingCE {
            smoothing,
            reduction,
        }
    }

    /// Create a center loss for feature learning
    pub fn center_loss(
        num_classes: usize,
        feature_dim: usize,
        alpha: f32,
        reduction: Reduction,
    ) -> impl CustomLoss {
        struct CenterLoss {
            num_classes: usize,
            feature_dim: usize,
            #[allow(dead_code)]
            alpha: f32,
            centers: Vec<Vec<f32>>, // [num_classes, feature_dim]
            reduction: Reduction,
        }

        impl CustomLoss for CenterLoss {
            /// The class centres are plain host data owned by this loss, so the
            /// per-sample centre is a *constant* tensor and the whole loss is
            /// `0.5 * sum((features - centres)^2)` in recording ops. The
            /// predecessor accumulated the same sum over `to_vec()` output and
            /// rebuilt it with `Tensor::from_vec`, i.e. a detached leaf.
            fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
                // Center loss: 0.5 * ||features - centers[class]||^2

                let shape_binding = predictions.shape();
                let dims = shape_binding.dims();
                let (batch_size, feature_dim) = match dims {
                    [batch, features] => (*batch, *features),
                    _ => {
                        return Err(TorshError::InvalidShape(format!(
                            "center_loss expects 2-D features [N, D], got {dims:?}"
                        )))
                    }
                };
                if feature_dim != self.feature_dim {
                    return Err(TorshError::ShapeMismatch {
                        expected: vec![batch_size, self.feature_dim],
                        got: dims.to_vec(),
                    });
                }

                let target_data = targets.to_vec()?;
                if target_data.len() < batch_size {
                    return Err(TorshError::InvalidShape(format!(
                        "center_loss expects one label per sample: features have \
                         {batch_size} rows but labels have {} entries",
                        target_data.len()
                    )));
                }

                // Gather the per-sample centre into a constant tensor. The
                // predecessor skipped out-of-range classes and then handed a
                // short vector to `Tensor::from_vec`, which failed the length
                // check; this reports the real problem instead.
                let mut centers = vec![0.0f32; batch_size * feature_dim];
                for (b, &target_value) in target_data.iter().take(batch_size).enumerate() {
                    let class_id = target_value as usize;
                    let row = match self.centers.get(class_id) {
                        Some(row) if class_id < self.num_classes => row,
                        _ => {
                            return Err(TorshError::InvalidArgument(format!(
                                "Label {target_value} out of range for {} classes",
                                self.num_classes
                            )))
                        }
                    };
                    for (f, slot) in centers
                        .iter_mut()
                        .skip(b * feature_dim)
                        .take(feature_dim)
                        .enumerate()
                    {
                        *slot = row.get(f).copied().unwrap_or(0.0);
                    }
                }
                let centers_tensor = Tensor::from_data(
                    centers,
                    vec![batch_size, feature_dim],
                    predictions.device(),
                )?;

                predictions
                    .sub(&centers_tensor)?
                    .square()?
                    .sum_dim(&[-1], false)?
                    .mul_scalar(0.5)
            }

            fn reduction(&self) -> &Reduction {
                &self.reduction
            }
        }

        // Initialize centers randomly
        let mut centers = Vec::new();
        for _ in 0..num_classes {
            let mut center = Vec::new();
            for _ in 0..feature_dim {
                center.push(0.0); // Initialize to zero
            }
            centers.push(center);
        }

        CenterLoss {
            num_classes,
            feature_dim,
            alpha,
            centers,
            reduction,
        }
    }
}

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/// Validation utilities for loss functions
pub mod validation {
    use super::*;

    /// Check if predictions and targets have compatible shapes
    pub fn check_shapes(predictions: &Tensor, targets: &Tensor) -> Result<()> {
        let pred_binding = predictions.shape();
        let pred_shape = pred_binding.dims();
        let target_binding = targets.shape();
        let target_shape = target_binding.dims();

        if pred_shape.len() != target_shape.len() {
            return Err(TorshError::ShapeMismatch {
                expected: pred_shape.to_vec(),
                got: target_shape.to_vec(),
            });
        }

        for (_i, (&pred_dim, &target_dim)) in pred_shape.iter().zip(target_shape.iter()).enumerate()
        {
            if pred_dim != target_dim {
                return Err(TorshError::ShapeMismatch {
                    expected: vec![pred_dim],
                    got: vec![target_dim],
                });
            }
        }

        Ok(())
    }

    /// Check if target values are within valid range for classification
    pub fn check_classification_targets(targets: &Tensor, num_classes: usize) -> Result<()> {
        let target_data = targets.to_vec()?;

        for &target_val in target_data.iter() {
            if target_val < 0.0 || target_val >= num_classes as f32 {
                return Err(TorshError::InvalidArgument(format!(
                    "Target value {} is out of range [0, {})",
                    target_val, num_classes
                )));
            }
        }

        Ok(())
    }

    /// Check if predictions are valid probabilities (sum to 1, non-negative)
    pub fn check_probabilities(predictions: &Tensor, dim: i32) -> Result<()> {
        let pred_data = predictions.to_vec()?;
        let binding = predictions.shape();
        let shape = binding.dims();

        // Simple check for non-negative values
        for &val in pred_data.iter() {
            if val < 0.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "Probability value {} is negative",
                    val
                )));
            }
        }

        // Validate that probabilities sum to 1 along the specified dimension.
        // For a 1-D tensor the only valid dim is 0; for N-D tensors we check
        // slices along `dim`.
        let ndim = shape.len();
        let dim_usize = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if ndim == 0 {
            // Scalar tensor: single probability, must equal 1
            let val = pred_data.first().copied().unwrap_or(0.0);
            if (val - 1.0_f32).abs() > 1e-4 {
                return Err(TorshError::InvalidArgument(format!(
                    "Probability scalar {} does not sum to 1",
                    val
                )));
            }
        } else if ndim == 1 {
            // 1-D: the whole vector must sum to 1
            let total: f32 = pred_data.iter().sum();
            if (total - 1.0_f32).abs() > 1e-4 {
                return Err(TorshError::InvalidArgument(format!(
                    "Probabilities sum to {} instead of 1.0",
                    total
                )));
            }
        } else {
            // N-D: stride along dim axis, check each slice
            let dim_size = shape.get(dim_usize).copied().unwrap_or(1);
            let outer: usize = shape[..dim_usize].iter().product();
            let inner: usize = if dim_usize + 1 < ndim {
                shape[dim_usize + 1..].iter().product()
            } else {
                1
            };

            for o in 0..outer {
                for i in 0..inner {
                    let mut slice_sum = 0.0_f32;
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner + d * inner + i;
                        slice_sum += pred_data.get(idx).copied().unwrap_or(0.0);
                    }
                    if (slice_sum - 1.0_f32).abs() > 1e-4 {
                        return Err(TorshError::InvalidArgument(format!(
                            "Probability slice (outer={}, inner={}) sums to {} instead of 1.0",
                            o, i, slice_sum
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// EXTENSION TRAIT FOR TENSOR CASTING
// =============================================================================

/// Extension trait to add tensor casting for compatibility
trait TensorCast {
    fn cast_i64(&self) -> Result<Tensor<i64>>;
}

impl TensorCast for Tensor {
    fn cast_i64(&self) -> Result<Tensor<i64>> {
        // Simplified casting - in practice would need proper tensor type conversion
        let data = self.to_vec()?;
        let i64_data: Vec<i64> = data.into_iter().map(|x| x as i64).collect();
        Ok(Tensor::from_data(
            i64_data,
            self.shape().dims().to_vec(),
            self.device(),
        )?)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::str::FromStr;

    // =========================================================================
    // REDUCTION TESTS
    // =========================================================================

    #[test]
    fn test_reduction_from_str() -> Result<()> {
        assert_eq!(Reduction::from_str("none")?, Reduction::None);
        assert_eq!(Reduction::from_str("mean")?, Reduction::Mean);
        assert_eq!(Reduction::from_str("sum")?, Reduction::Sum);
        assert_eq!(Reduction::from_str("batchmean")?, Reduction::BatchMean);
        assert_eq!(Reduction::from_str("MEAN")?, Reduction::Mean); // Case insensitive
        assert_eq!(Reduction::from_str("BatchMean")?, Reduction::BatchMean);
        Ok(())
    }

    #[test]
    fn test_reduction_from_str_invalid() {
        assert!(Reduction::from_str("invalid").is_err());
        assert!(Reduction::from_str("").is_err());
        assert!(Reduction::from_str("meen").is_err()); // typo
    }

    #[test]
    fn test_reduction_none() -> Result<()> {
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let reduced = Reduction::None.apply(&loss, Some(4))?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 4);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(data[2], 3.0, epsilon = 1e-6);
        assert_relative_eq!(data[3], 4.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_reduction_mean() -> Result<()> {
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let reduced = Reduction::Mean.apply(&loss, Some(4))?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 1);
        // Mean divides by numel (4), matching PyTorch "mean"
        assert_relative_eq!(data[0], 2.5, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_reduction_sum() -> Result<()> {
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let reduced = Reduction::Sum.apply(&loss, Some(4))?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 1);
        assert_relative_eq!(data[0], 10.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_reduction_batchmean() -> Result<()> {
        // BatchMean divides by batch_size, not numel.
        // For a [2, 3] tensor with values 1..6:
        //   sum = 21, numel = 6, batch_size = 2
        //   Mean = 21/6 = 3.5, BatchMean = 21/2 = 10.5
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let reduced = Reduction::BatchMean.apply(&loss, Some(2))?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 1);
        assert_relative_eq!(data[0], 10.5, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_reduction_mean_vs_batchmean_differ_for_nonscalar() -> Result<()> {
        // Verify that Mean and BatchMean produce different results
        // when elements per sample > 1
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let mean_result = Reduction::Mean.apply(&loss, Some(2))?.to_vec()?;
        let batchmean_result = Reduction::BatchMean.apply(&loss, Some(2))?.to_vec()?;

        // Mean = sum/numel = 10/4 = 2.5
        assert_relative_eq!(mean_result[0], 2.5, epsilon = 1e-6);
        // BatchMean = sum/batch_size = 10/2 = 5.0
        assert_relative_eq!(batchmean_result[0], 5.0, epsilon = 1e-6);
        Ok(())
    }

    // =========================================================================
    // SMOOTH L1 LOSS TESTS
    // =========================================================================

    #[test]
    fn test_smooth_l1_loss_small_diff() -> Result<()> {
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let targets = Tensor::from_vec(vec![1.1, 2.2, 3.3], &[3])?;

        let loss_fn = SmoothL1Loss::new(1.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_eq!(loss_data.len(), 3);

        // diff[0] = -0.1, abs = 0.1 < 1.0, so loss = 0.5 * 0.01 / 1.0 = 0.005
        assert_relative_eq!(loss_data[0], 0.005, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_large_diff() -> Result<()> {
        let predictions = Tensor::from_vec(vec![0.0, 5.0], &[2])?;
        let targets = Tensor::from_vec(vec![2.0, 0.0], &[2])?;

        let loss_fn = SmoothL1Loss::new(1.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        // diff[0] = -2.0, abs = 2.0 >= 1.0, so loss = 2.0 - 0.5 = 1.5
        assert_relative_eq!(loss_data[0], 1.5, epsilon = 1e-5);
        // diff[1] = 5.0, abs = 5.0 >= 1.0, so loss = 5.0 - 0.5 = 4.5
        assert_relative_eq!(loss_data[1], 4.5, epsilon = 1e-5);
        Ok(())
    }

    // =========================================================================
    // DICE LOSS TESTS
    // =========================================================================

    #[test]
    fn test_dice_loss_perfect_match() -> Result<()> {
        let predictions = Tensor::from_vec(vec![10.0, 10.0, 10.0, 10.0], &[4])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = DiceLoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] < 0.1);
        Ok(())
    }

    #[test]
    fn test_dice_loss_no_match() -> Result<()> {
        let predictions = Tensor::from_vec(vec![-10.0, -10.0, -10.0, -10.0], &[4])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = DiceLoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] > 0.9);
        Ok(())
    }

    // =========================================================================
    // IOU LOSS TESTS
    // =========================================================================

    #[test]
    fn test_iou_loss_perfect_match() -> Result<()> {
        let predictions = Tensor::from_vec(vec![10.0, 10.0, 10.0, 10.0], &[4])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = IoULoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] < 0.1);
        Ok(())
    }

    #[test]
    fn test_iou_loss_no_match() -> Result<()> {
        let predictions = Tensor::from_vec(vec![-10.0, -10.0, -10.0, -10.0], &[4])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = IoULoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] > 0.9);
        Ok(())
    }

    // =========================================================================
    // FOCAL LOSS TESTS
    // =========================================================================

    #[test]
    fn test_focal_loss_basic() -> Result<()> {
        let predictions = Tensor::from_vec(
            vec![0.9, 0.1, 0.8, 0.2],
            &[2, 2],
        )?;
        let targets = Tensor::from_vec(vec![1.0, 0.0], &[2])?;

        let loss_fn = FocalLoss::new(Some(0.25), 2.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_eq!(loss_data.len(), 2);
        assert!(loss_data.iter().all(|&x| x >= 0.0));
        Ok(())
    }

    // =========================================================================
    // BINARY CROSS ENTROPY TESTS
    // =========================================================================

    #[test]
    fn test_binary_cross_entropy_basic() -> Result<()> {
        let predictions = Tensor::from_vec(vec![0.9, 0.1, 0.8, 0.2], &[4])?;
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4])?;

        let loss_fn = BinaryCrossEntropy::new(None, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_eq!(loss_data.len(), 4);

        assert!(loss_data[0] > 0.0 && loss_data[0] < 0.2);
        assert!(loss_data[1] > 0.0 && loss_data[1] < 0.2);
        Ok(())
    }

    #[test]
    fn test_binary_cross_entropy_perfect_prediction() -> Result<()> {
        let predictions = Tensor::from_vec(vec![1.0 - 1e-7, 1e-7], &[2])?;
        let targets = Tensor::from_vec(vec![1.0, 0.0], &[2])?;

        let loss_fn = BinaryCrossEntropy::new(None, Reduction::Mean);
        let loss = loss_fn.compute_loss(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] < 1e-5);
        Ok(())
    }

    // =========================================================================
    // MSE LOSS TESTS
    // =========================================================================

    #[test]
    fn test_mse_loss_basic() -> Result<()> {
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let targets = Tensor::from_vec(vec![1.5, 2.5, 3.5], &[3])?;

        let loss_fn = MSELoss::new(Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_eq!(loss_data.len(), 3);

        assert_relative_eq!(loss_data[0], 0.25, epsilon = 1e-6);
        assert_relative_eq!(loss_data[1], 0.25, epsilon = 1e-6);
        assert_relative_eq!(loss_data[2], 0.25, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_loss_mean_reduction() -> Result<()> {
        let predictions = Tensor::from_vec(vec![0.0, 2.0], &[2])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0], &[2])?;

        let loss_fn = MSELoss::new(Reduction::Mean);
        let loss = loss_fn.compute_loss(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        // ((0-1)^2 + (2-1)^2) / 2 = (1 + 1) / 2 = 1.0
        assert_relative_eq!(loss_data[0], 1.0, epsilon = 1e-6);
        Ok(())
    }

    // =========================================================================
    // L1 LOSS TESTS
    // =========================================================================

    #[test]
    fn test_l1_loss_basic() -> Result<()> {
        let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let targets = Tensor::from_vec(vec![1.5, 2.5, 2.0], &[3])?;

        let loss_fn = L1Loss::new(Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_relative_eq!(loss_data[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(loss_data[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(loss_data[2], 1.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_l1_loss_sum_reduction() -> Result<()> {
        let predictions = Tensor::from_vec(vec![0.0, 2.0, 4.0], &[3])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3])?;

        let loss_fn = L1Loss::new(Reduction::Sum);
        let loss = loss_fn.compute_loss(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        // |0-1| + |2-1| + |4-1| = 1 + 1 + 3 = 5
        assert_relative_eq!(loss_data[0], 5.0, epsilon = 1e-6);
        Ok(())
    }

    // =========================================================================
    // HUBER LOSS TESTS
    // =========================================================================

    #[test]
    fn test_huber_loss_small_errors() -> Result<()> {
        let predictions = Tensor::from_vec(vec![1.0, 2.0], &[2])?;
        let targets = Tensor::from_vec(vec![1.2, 2.3], &[2])?;

        let loss_fn = HuberLoss::new(1.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_relative_eq!(loss_data[0], 0.5 * 0.2 * 0.2, epsilon = 1e-5);
        assert_relative_eq!(loss_data[1], 0.5 * 0.3 * 0.3, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_huber_loss_large_errors() -> Result<()> {
        let predictions = Tensor::from_vec(vec![0.0], &[1])?;
        let targets = Tensor::from_vec(vec![5.0], &[1])?;

        let loss_fn = HuberLoss::new(1.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_relative_eq!(loss_data[0], 4.5, epsilon = 1e-5);
        Ok(())
    }

    // =========================================================================
    // HINGE LOSS TESTS
    // =========================================================================

    #[test]
    fn test_hinge_loss_correct_classification() -> Result<()> {
        let predictions = Tensor::from_vec(vec![2.0, -2.0], &[2])?;
        let targets = Tensor::from_vec(vec![1.0, -1.0], &[2])?;

        let loss_fn = HingeLoss::new(1.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_relative_eq!(loss_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(loss_data[1], 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_hinge_loss_incorrect_classification() -> Result<()> {
        let predictions = Tensor::from_vec(vec![-0.5], &[1])?;
        let targets = Tensor::from_vec(vec![1.0], &[1])?;

        let loss_fn = HingeLoss::new(1.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_relative_eq!(loss_data[0], 1.5, epsilon = 1e-6);
        Ok(())
    }
}