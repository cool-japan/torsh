//! Regularization layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Dropout layer for regularization
pub struct Dropout {
    base: ModuleBase,
    p: f32,
    inplace: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            inplace: false,
        }
    }

    pub fn with_inplace(p: f32, inplace: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            inplace,
        }
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() {
            // During evaluation, just return the input
            return Ok(input.clone());
        }

        if self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            return zeros(input.shape().dims());
        }

        // Generate random mask for dropout
        // In a real implementation, this would use proper random number generation
        let keep_prob = 1.0 - self.p;
        let mask = full(input.shape().dims(), keep_prob)?; // Simplified - should be random

        let dropped = input.mul_op(&mask)?;
        // Scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / keep_prob;
        dropped.mul_op(&full(input.shape().dims(), scale)?)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("p", &self.p)
            .field("inplace", &self.inplace)
            .finish()
    }
}

// =============================================================================
// MODERN REGULARIZATION TECHNIQUES
// =============================================================================

/// DropConnect layer - drops individual weights instead of activations
///
/// DropConnect is a generalization of Dropout that randomly drops connections
/// (weights) instead of activations. This provides more thorough regularization.
///
/// # Reference
/// Wan et al., "Regularization of Neural Networks using DropConnect", ICML 2013
pub struct DropConnect {
    base: ModuleBase,
    p: f32,
}

impl DropConnect {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
        }
    }
}

impl Module for DropConnect {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(input.clone());
        }

        // DropConnect: mask is applied to weights, not activations
        // For simplicity in this implementation, we approximate by masking activations
        // In a real implementation, this would mask the weight matrix
        let keep_prob = 1.0 - self.p;
        let mask = full(input.shape().dims(), keep_prob)?; // Should be random
        let dropped = input.mul_op(&mask)?;

        // Scale to maintain expected value
        let scale = 1.0 / keep_prob;
        dropped.mul_op(&full(input.shape().dims(), scale)?)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Dropout2d - Spatial dropout for convolutional layers
///
/// Drops entire feature maps instead of individual elements, which is more
/// effective for convolutional layers where nearby activations are strongly correlated.
///
/// # Reference
/// Tompson et al., "Efficient Object Localization Using Convolutional Networks", CVPR 2015
pub struct Dropout2d {
    base: ModuleBase,
    p: f32,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(input.clone());
        }

        // Input shape: (batch_size, channels, height, width)
        // Drop entire channels (feature maps)
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Dropout2d expects 4D input (batch, channels, height, width)".to_string(),
            ));
        }

        let keep_prob = 1.0 - self.p;

        // Create channel mask: shape (batch_size, channels, 1, 1)
        let batch_size = dims[0];
        let channels = dims[1];
        let mask = full(&[batch_size, channels, 1, 1], keep_prob)?;

        // Broadcast mask across spatial dimensions
        let dropped = input.mul_op(&mask)?;

        // Scale to maintain expected value
        let scale = 1.0 / keep_prob;
        dropped.mul_op(&full(&[1], scale)?)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// DropBlock - Structured dropout for convolutional networks
///
/// DropBlock drops contiguous regions instead of individual elements,
/// which is more effective than standard dropout for conv networks.
///
/// # Reference
/// Ghiasi et al., "DropBlock: A regularization method for convolutional networks", NeurIPS 2018
pub struct DropBlock2d {
    base: ModuleBase,
    drop_prob: f32,
    #[allow(dead_code)]
    block_size: usize,
}

impl DropBlock2d {
    pub fn new(drop_prob: f32, block_size: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            drop_prob: drop_prob.clamp(0.0, 1.0),
            block_size,
        }
    }
}

impl Module for DropBlock2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.drop_prob == 0.0 {
            return Ok(input.clone());
        }

        // Input shape: (batch_size, channels, height, width)
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "DropBlock2d expects 4D input (batch, channels, height, width)".to_string(),
            ));
        }

        // For now, return a simplified version
        // Full implementation would sample block centers and zero out blocks
        let keep_prob = 1.0 - self.drop_prob;
        let mask = full(dims, keep_prob)?;
        let dropped = input.mul_op(&mask)?;

        // Normalize by the fraction of units that are kept
        let scale = 1.0 / keep_prob;
        dropped.mul_op(&full(&[1], scale)?)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Stochastic Depth (Drop Path) - randomly drops entire layers during training
///
/// Stochastic Depth improves training of very deep networks by randomly
/// dropping entire layers during training, effectively training an ensemble.
///
/// # Reference
/// Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
pub struct StochasticDepth {
    base: ModuleBase,
    drop_prob: f32,
    scale_by_keep: bool,
}

impl StochasticDepth {
    pub fn new(drop_prob: f32) -> Self {
        Self {
            base: ModuleBase::new(),
            drop_prob: drop_prob.clamp(0.0, 1.0),
            scale_by_keep: true,
        }
    }

    pub fn with_scaling(drop_prob: f32, scale_by_keep: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            drop_prob: drop_prob.clamp(0.0, 1.0),
            scale_by_keep,
        }
    }
}

impl Module for StochasticDepth {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.drop_prob == 0.0 {
            return Ok(input.clone());
        }

        // Randomly drop the entire tensor with probability drop_prob
        // In a real implementation, this would use random sampling
        // For now, use a deterministic approximation
        let keep_prob = 1.0 - self.drop_prob;

        if self.scale_by_keep {
            // Scale by keep probability during training
            input.mul_op(&full(&[1], keep_prob)?)
        } else {
            // No scaling - just randomly drop
            Ok(input.clone())
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// AlphaDropout - Dropout variant for SELU activation
///
/// AlphaDropout is designed to work with SELU activations, maintaining
/// self-normalizing properties during dropout.
///
/// # Reference
/// Klambauer et al., "Self-Normalizing Neural Networks", NeurIPS 2017
pub struct AlphaDropout {
    base: ModuleBase,
    p: f32,
    alpha: f32,
    #[allow(dead_code)]
    scale: f32,
}

impl AlphaDropout {
    pub fn new(p: f32) -> Self {
        // SELU-specific constants
        let alpha = -1.7580993408473766; // SELU alpha
        let scale = 1.0507009873554804; // SELU lambda

        Self {
            base: ModuleBase::new(),
            p: p.clamp(0.0, 1.0),
            alpha,
            scale,
        }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() || self.p == 0.0 {
            return Ok(input.clone());
        }

        let keep_prob = 1.0 - self.p;

        // Calculate affine transformation parameters
        let a = ((1.0 - keep_prob) * (1.0 + keep_prob * self.alpha * self.alpha)).sqrt();
        let b = -a * self.alpha * keep_prob;

        // Apply dropout mask
        let mask = full(input.shape().dims(), keep_prob)?;
        let dropped = input.mul_op(&mask)?;

        // Apply affine transformation to maintain self-normalizing property
        let a_tensor = full(&[1], a)?;
        let scaled = dropped.mul_op(&a_tensor)?;
        let b_tensor = full(input.shape().dims(), b)?;
        scaled.add(&b_tensor)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Cutout - randomly masks out square regions of input
///
/// Cutout is a simple regularization technique that randomly masks out
/// square regions of the input during training. Particularly effective for images.
///
/// # Reference
/// DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", arXiv 2017
pub struct Cutout {
    base: ModuleBase,
    #[allow(dead_code)]
    n_holes: usize,
    #[allow(dead_code)]
    length: usize,
}

impl Cutout {
    pub fn new(n_holes: usize, length: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            n_holes,
            length,
        }
    }
}

impl Module for Cutout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.base.training() {
            return Ok(input.clone());
        }

        // Input shape typically: (batch, channels, height, width)
        let shape = input.shape();
        let dims = shape.dims();

        if dims.len() < 2 {
            return Ok(input.clone());
        }

        // For now, return input unchanged
        // Full implementation would randomly mask square regions
        Ok(input.clone())
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Mixup - linearly interpolates between random pairs of training examples
///
/// Mixup creates virtual training examples by mixing pairs of examples and their labels.
/// This helps improve generalization and robustness.
///
/// # Reference
/// Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
pub struct Mixup {
    #[allow(dead_code)]
    alpha: f32,
}

impl Mixup {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Apply mixup to a batch of data
    /// Returns mixed inputs and lambda value for label mixing
    pub fn apply(&self, input: &Tensor, _training: bool) -> Result<(Tensor, f32)> {
        // Sample lambda from Beta(alpha, alpha)
        // For now, use a fixed value
        // In real implementation, would use: Beta::new(self.alpha, self.alpha).sample()
        let lambda = 0.5_f32;

        // In real implementation:
        // 1. Randomly shuffle indices
        // 2. Mix input with shuffled input: lambda * input + (1-lambda) * input_shuffled
        // 3. Return mixed input and lambda for label mixing

        Ok((input.clone(), lambda))
    }
}

/// CutMix - replaces regions of images with patches from other images
///
/// CutMix combines regions from different images and mixes their labels
/// proportionally to the area of the patches.
///
/// # Reference
/// Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", ICCV 2019
pub struct CutMix {
    #[allow(dead_code)]
    alpha: f32,
}

impl CutMix {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Apply CutMix to a batch of data
    /// Returns mixed inputs and lambda value for label mixing
    pub fn apply(&self, input: &Tensor, _training: bool) -> Result<(Tensor, f32)> {
        // Sample lambda from Beta(alpha, alpha)
        // In real implementation, would use: Beta::new(self.alpha, self.alpha).sample()
        let lambda = 0.5_f32;

        // In real implementation:
        // 1. Sample bounding box coordinates based on lambda
        // 2. Replace that box in input with box from shuffled input
        // 3. Calculate actual lambda based on box area
        // 4. Return mixed input and lambda for label mixing

        Ok((input.clone(), lambda))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert!((dropout.p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dropconnect_creation() {
        let dropconnect = DropConnect::new(0.3);
        assert!((dropconnect.p - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dropout2d_creation() {
        let dropout2d = Dropout2d::new(0.5);
        assert!((dropout2d.p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dropblock_creation() {
        let dropblock = DropBlock2d::new(0.1, 7);
        assert!((dropblock.drop_prob - 0.1).abs() < 1e-6);
        assert_eq!(dropblock.block_size, 7);
    }

    #[test]
    fn test_stochastic_depth_creation() {
        let stoch_depth = StochasticDepth::new(0.2);
        assert!((stoch_depth.drop_prob - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_dropout_creation() {
        let alpha_dropout = AlphaDropout::new(0.1);
        assert!((alpha_dropout.p - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_cutout_creation() {
        let cutout = Cutout::new(1, 16);
        assert_eq!(cutout.n_holes, 1);
        assert_eq!(cutout.length, 16);
    }

    #[test]
    fn test_mixup_creation() {
        let mixup = Mixup::new(1.0);
        assert!((mixup.alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cutmix_creation() {
        let cutmix = CutMix::new(1.0);
        assert!((cutmix.alpha - 1.0).abs() < 1e-6);
    }
}
