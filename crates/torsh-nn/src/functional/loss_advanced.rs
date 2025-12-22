//! Advanced loss functions framework
//!
//! This module provides a comprehensive framework for creating custom loss functions
//! with composable building blocks, different reduction modes, and validation utilities.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// =============================================================================
// REDUCTION MODES AND FRAMEWORK
// =============================================================================

/// Reduction modes for loss functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction - return loss for each sample
    None,
    /// Mean reduction - average over all samples
    Mean,
    /// Sum reduction - sum over all samples
    Sum,
}

impl Reduction {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown reduction mode: {}",
                s
            ))),
        }
    }

    pub fn apply(&self, loss: &Tensor, batch_size: usize) -> Result<Tensor> {
        match self {
            Self::None => Ok(loss.clone()),
            Self::Mean => {
                // Compute mean over batch dimension
                let total_sum = loss.sum()?;
                let mean_val = total_sum.to_vec()?[0] / batch_size as f32;
                Ok(Tensor::from_data(vec![mean_val], vec![1], loss.device())?)
            }
            Self::Sum => {
                // Compute sum over batch dimension
                let total_sum = loss.sum()?;
                Ok(Tensor::from_data(
                    vec![total_sum.to_vec()?[0]],
                    vec![1],
                    loss.device(),
                )?)
            }
        }
    }
}

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
        self.reduction().apply(&raw_loss, batch_size)
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
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Smooth L1 loss:
        // loss = 0.5 * (pred - target)^2 / beta  if |pred - target| < beta
        // loss = |pred - target| - 0.5 * beta    otherwise

        let diff = predictions.sub(&targets)?;
        let abs_diff = diff.abs()?;
        let abs_diff_data = abs_diff.to_vec()?;
        let diff_data = diff.to_vec()?;

        let mut loss_data = Vec::new();

        for i in 0..abs_diff_data.len() {
            let abs_val = abs_diff_data[i];
            let diff_val = diff_data[i];

            let loss_val = if abs_val < self.beta {
                0.5 * diff_val * diff_val / self.beta
            } else {
                abs_val - 0.5 * self.beta
            };

            loss_data.push(loss_val);
        }

        Ok(Tensor::from_vec(loss_data, predictions.shape().dims())?)
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
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Dice loss: 1 - (2 * |intersection| + smooth) / (|pred| + |target| + smooth)

        // Apply sigmoid to predictions if needed (assuming raw logits)
        let probs = predictions.sigmoid()?;

        // Flatten tensors for easier computation
        let pred_data = probs.to_vec()?;
        let target_data = targets.to_vec()?;

        let mut intersection = 0.0f32;
        let mut pred_sum = 0.0f32;
        let mut target_sum = 0.0f32;

        for i in 0..pred_data.len() {
            let pred_val = pred_data[i];
            let target_val = target_data[i];

            intersection += pred_val * target_val;
            pred_sum += pred_val;
            target_sum += target_val;
        }

        let dice_coeff = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth);
        let dice_loss = 1.0 - dice_coeff;

        Ok(Tensor::from_data(
            vec![dice_loss],
            vec![1],
            predictions.device(),
        )?)
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
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // IoU loss: 1 - |intersection| / |union|

        let probs = predictions.sigmoid()?;
        let pred_data = probs.to_vec()?;
        let target_data = targets.to_vec()?;

        let mut intersection = 0.0f32;
        let mut union = 0.0f32;

        for i in 0..pred_data.len() {
            let pred_val = pred_data[i];
            let target_val = target_data[i];

            intersection += pred_val * target_val;
            union += pred_val + target_val - pred_val * target_val;
        }

        let iou = (intersection + self.smooth) / (union + self.smooth);
        let iou_loss = 1.0 - iou;

        Ok(Tensor::from_data(
            vec![iou_loss],
            vec![1],
            predictions.device(),
        )?)
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

        // Apply class weights (simplified implementation)
        // In practice, this would need proper indexing based on target classes
        let loss_data = base_loss.to_vec()?;
        let mut weighted_data = Vec::new();

        for (i, &loss_val) in loss_data.iter().enumerate() {
            let weight_idx = i % self.weights.len();
            weighted_data.push(loss_val * self.weights[weight_idx]);
        }

        Ok(Tensor::from_vec(weighted_data, base_loss.shape().dims())?)
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
            "none",
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
            "none",
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
            "none",
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
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions.sub(targets)?;
        let abs_diff = diff.abs()?;
        let abs_diff_data = abs_diff.to_vec()?;
        let diff_data = diff.to_vec()?;

        let mut loss_data = Vec::new();
        for i in 0..abs_diff_data.len() {
            let abs_val = abs_diff_data[i];
            let diff_val = diff_data[i];

            let loss_val = if abs_val < self.delta {
                0.5 * diff_val * diff_val
            } else {
                self.delta * abs_val - 0.5 * self.delta * self.delta
            };

            loss_data.push(loss_val);
        }

        Ok(Tensor::from_vec(loss_data, predictions.shape().dims())?)
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
        crate::functional::loss::kl_div(predictions, targets, "none", self.log_target)
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
        crate::functional::loss::mse_loss(predictions, targets, "none")
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
            "none",
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
            "none",
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
            "none",
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
                    "none",
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
            fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
                // Label smoothing: mix true labels with uniform distribution
                // smoothed_target = (1 - smoothing) * target + smoothing / num_classes

                let num_classes = predictions.shape().dims()[1];
                let uniform_prob = self.smoothing / num_classes as f32;
                let true_prob = 1.0 - self.smoothing;

                // Apply log softmax to predictions
                let log_probs = predictions.log_softmax(-1)?;

                // Create smoothed targets (simplified implementation)
                let target_data = targets.to_vec()?;
                let log_prob_data = log_probs.to_vec()?;
                let batch_size = predictions.shape().dims()[0];

                let mut loss_data = Vec::new();

                for b in 0..batch_size {
                    let true_class = target_data[b] as usize;
                    let mut sample_loss = 0.0f32;

                    for c in 0..num_classes {
                        let log_prob = log_prob_data[b * num_classes + c];
                        let smooth_target = if c == true_class {
                            true_prob + uniform_prob
                        } else {
                            uniform_prob
                        };
                        sample_loss -= smooth_target * log_prob;
                    }

                    loss_data.push(sample_loss);
                }

                Ok(Tensor::from_vec(loss_data, &[batch_size])?)
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
            fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
                // Center loss: 0.5 * ||features - centers[class]||^2

                let batch_size = predictions.shape().dims()[0];
                let feature_data = predictions.to_vec()?;
                let target_data = targets.to_vec()?;

                let mut loss_data = Vec::new();

                for b in 0..batch_size {
                    let class_id = target_data[b] as usize;
                    if class_id >= self.num_classes {
                        continue;
                    }

                    let mut squared_distance = 0.0f32;
                    for f in 0..self.feature_dim {
                        let feature_val = feature_data[b * self.feature_dim + f];
                        let center_val = self.centers[class_id][f];
                        let diff = feature_val - center_val;
                        squared_distance += diff * diff;
                    }

                    loss_data.push(0.5 * squared_distance);
                }

                Ok(Tensor::from_vec(loss_data, &[batch_size])?)
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

        // TODO: Add sum-to-1 check when tensor operations support it
        let _ = (dim, shape); // Suppress warnings for now

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

    // =========================================================================
    // REDUCTION TESTS
    // =========================================================================

    #[test]
    fn test_reduction_from_str() -> Result<()> {
        assert_eq!(Reduction::from_str("none")?, Reduction::None);
        assert_eq!(Reduction::from_str("mean")?, Reduction::Mean);
        assert_eq!(Reduction::from_str("sum")?, Reduction::Sum);
        assert_eq!(Reduction::from_str("MEAN")?, Reduction::Mean); // Case insensitive
        Ok(())
    }

    #[test]
    fn test_reduction_from_str_invalid() {
        assert!(Reduction::from_str("invalid").is_err());
    }

    #[test]
    fn test_reduction_none() -> Result<()> {
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let reduced = Reduction::None.apply(&loss, 4)?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 4);
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data[1], 2.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_reduction_mean() -> Result<()> {
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let reduced = Reduction::Mean.apply(&loss, 4)?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 1);
        assert_relative_eq!(data[0], 2.5, epsilon = 1e-6); // (1+2+3+4)/4 = 2.5
        Ok(())
    }

    #[test]
    fn test_reduction_sum() -> Result<()> {
        let loss = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
        let reduced = Reduction::Sum.apply(&loss, 4)?;

        let data = reduced.to_vec()?;
        assert_eq!(data.len(), 1);
        assert_relative_eq!(data[0], 10.0, epsilon = 1e-6); // 1+2+3+4 = 10
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

        // For small differences (< beta), should use 0.5 * diff^2 / beta
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

        // For large differences (>= beta), should use |diff| - 0.5 * beta
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
        // Perfect match should give dice coefficient = 1, loss = 0
        let predictions = Tensor::from_vec(vec![10.0, 10.0, 10.0, 10.0], &[4])?; // Will sigmoid to ~1
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = DiceLoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] < 0.1); // Should be close to 0
        Ok(())
    }

    #[test]
    fn test_dice_loss_no_match() -> Result<()> {
        // No overlap should give dice coefficient close to 0, loss close to 1
        let predictions = Tensor::from_vec(vec![-10.0, -10.0, -10.0, -10.0], &[4])?; // Will sigmoid to ~0
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = DiceLoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] > 0.9); // Should be close to 1
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
        assert!(loss_data[0] < 0.1); // IoU close to 1, loss close to 0
        Ok(())
    }

    #[test]
    fn test_iou_loss_no_match() -> Result<()> {
        let predictions = Tensor::from_vec(vec![-10.0, -10.0, -10.0, -10.0], &[4])?;
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])?;

        let loss_fn = IoULoss::new(1e-5, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert!(loss_data[0] > 0.9); // IoU close to 0, loss close to 1
        Ok(())
    }

    // =========================================================================
    // FOCAL LOSS TESTS
    // =========================================================================

    #[test]
    fn test_focal_loss_basic() -> Result<()> {
        // Focal loss expects 2D input [batch_size, num_classes]
        let predictions = Tensor::from_vec(
            vec![0.9, 0.1, 0.8, 0.2],
            &[2, 2], // 2 samples, 2 classes
        )?;
        let targets = Tensor::from_vec(vec![1.0, 0.0], &[2])?; // Class indices

        let loss_fn = FocalLoss::new(Some(0.25), 2.0, Reduction::None);
        let loss = loss_fn.forward(&predictions, &targets)?;

        let loss_data = loss.to_vec()?;
        assert_eq!(loss_data.len(), 2); // One loss per sample
        assert!(loss_data.iter().all(|&x| x >= 0.0)); // All losses should be non-negative
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

        // For pred=0.9, target=1: -log(0.9) ≈ 0.105
        assert!(loss_data[0] > 0.0 && loss_data[0] < 0.2);
        // For pred=0.1, target=0: -log(0.9) ≈ 0.105
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
        assert!(loss_data[0] < 1e-5); // Should be very small
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

        // Each diff is 0.5, so (0.5)^2 = 0.25
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

        // Small errors use quadratic: 0.5 * error^2
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

        // Large errors use linear: delta * (|error| - 0.5 * delta)
        let loss_data = loss.to_vec()?;
        // delta=1.0, error=5.0: 1.0 * (5.0 - 0.5) = 4.5
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

        // For correct predictions with margin > 1, loss should be 0
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

        // max(0, 1 - (1.0 * -0.5)) = max(0, 1.5) = 1.5
        let loss_data = loss.to_vec()?;
        assert_relative_eq!(loss_data[0], 1.5, epsilon = 1e-6);
        Ok(())
    }
}
