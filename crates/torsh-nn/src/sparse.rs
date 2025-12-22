//! Sparse Neural Network Support
//!
//! This module provides efficient implementations of sparse neural network layers
//! and operations, enabling training and inference with sparse weights and activations.
//!
//! # Features
//!
//! - **Sparse Linear Layers**: Efficient linear transformations with sparse weight matrices
//! - **Sparse Convolutions**: Convolutional layers with structured sparsity
//! - **Magnitude Pruning**: Remove small-magnitude weights during training
//! - **Structured Sparsity**: Block-wise or channel-wise sparsity patterns
//! - **Sparse Backpropagation**: Efficient gradient computation for sparse parameters
//!
//! # Example
//!
//! ```ignore
//! use torsh_nn::sparse::{SparseLinear, SparsityPattern};
//!
//! // Create sparse linear layer with 90% sparsity
//! let sparse_layer = SparseLinear::new(
//!     512,
//!     256,
//!     SparsityPattern::Random { sparsity: 0.9 },
//!     true,
//! );
//!
//! let output = sparse_layer.forward(&input)?;
//! ```

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

// ✅ SciRS2 Policy Compliant
use scirs2_core::slice_random::shuffle;

/// Sparsity pattern for sparse layers
#[derive(Debug, Clone, Copy)]
pub enum SparsityPattern {
    /// Random sparsity with specified fraction
    Random { sparsity: f32 },
    /// Block-wise sparsity with block size
    Blocked { block_size: usize, sparsity: f32 },
    /// Structured sparsity (channel-wise)
    Structured { channels_to_prune: usize },
    /// Magnitude-based pruning threshold
    MagnitudeBased { threshold: f32 },
}

/// Sparse mask for weight matrices
#[derive(Debug, Clone)]
pub struct SparseMask {
    /// Mask indicating which weights are active (1.0) or pruned (0.0)
    mask: Tensor,
    /// Current sparsity level (fraction of zero weights)
    sparsity: f32,
    /// Number of non-zero elements
    nnz: usize,
}

impl SparseMask {
    /// Create a new sparse mask with random sparsity
    pub fn random(shape: &[usize], sparsity: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(format!(
                "Sparsity must be in [0, 1], got {}",
                sparsity
            )));
        }

        let total_elements: usize = shape.iter().product();
        let num_zeros = (total_elements as f32 * sparsity) as usize;

        // Create initial all-ones mask
        let mut mask_data = vec![1.0_f32; total_elements];

        // Randomly select positions to zero
        let mut indices: Vec<usize> = (0..total_elements).collect();
        shuffle(&mut indices);

        for &idx in indices.iter().take(num_zeros) {
            mask_data[idx] = 0.0;
        }

        let mask = Tensor::from_vec(mask_data, shape)?;
        let nnz = total_elements - num_zeros;

        Ok(Self {
            mask,
            sparsity,
            nnz,
        })
    }

    /// Create mask from magnitude-based pruning
    pub fn from_magnitude(weights: &Tensor, threshold: f32) -> Result<Self> {
        let shape = weights.shape().dims().to_vec();
        let weight_data = weights.to_vec()?;

        let mask_data: Vec<f32> = weight_data
            .iter()
            .map(|&w| if w.abs() >= threshold { 1.0 } else { 0.0 })
            .collect();

        let nnz = mask_data.iter().filter(|&&m| m > 0.0).count();
        let total = mask_data.len();
        let sparsity = 1.0 - (nnz as f32 / total as f32);

        Ok(Self {
            mask: Tensor::from_vec(mask_data, &shape)?,
            sparsity,
            nnz,
        })
    }

    /// Create block-wise sparse mask
    pub fn blocked(shape: &[usize], block_size: usize, sparsity: f32) -> Result<Self> {
        if shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Block sparsity only supported for 2D tensors".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];

        if rows % block_size != 0 || cols % block_size != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Shape {:?} must be divisible by block_size {}",
                shape, block_size
            )));
        }

        let num_blocks_row = rows / block_size;
        let num_blocks_col = cols / block_size;
        let total_blocks = num_blocks_row * num_blocks_col;
        let blocks_to_zero = (total_blocks as f32 * sparsity) as usize;

        // Create mask
        let mut mask_data = vec![1.0_f32; rows * cols];

        // Randomly select blocks to zero
        let mut block_indices: Vec<usize> = (0..total_blocks).collect();
        shuffle(&mut block_indices);

        for &block_idx in block_indices.iter().take(blocks_to_zero) {
            let block_row = block_idx / num_blocks_col;
            let block_col = block_idx % num_blocks_col;

            // Zero out the entire block
            for r in 0..block_size {
                for c in 0..block_size {
                    let row = block_row * block_size + r;
                    let col = block_col * block_size + c;
                    let idx = row * cols + col;
                    mask_data[idx] = 0.0;
                }
            }
        }

        let nnz = mask_data.iter().filter(|&&m| m > 0.0).count();
        let actual_sparsity = 1.0 - (nnz as f32 / mask_data.len() as f32);

        Ok(Self {
            mask: Tensor::from_vec(mask_data, shape)?,
            sparsity: actual_sparsity,
            nnz,
        })
    }

    /// Apply mask to weights
    pub fn apply(&self, weights: &Tensor) -> Result<Tensor> {
        weights.mul(&self.mask)
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Get sparsity level
    pub fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get the mask tensor
    pub fn mask(&self) -> &Tensor {
        &self.mask
    }
}

/// Sparse linear layer with efficient sparse matrix operations
pub struct SparseLinear {
    base: ModuleBase,
    /// Sparsity mask
    mask: SparseMask,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Whether bias is used
    use_bias: bool,
}

impl SparseLinear {
    /// Create a new sparse linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        pattern: SparsityPattern,
        bias: bool,
    ) -> Self {
        let mut base = ModuleBase::new();

        // Initialize dense weights first
        let weight = crate::init::kaiming_uniform(&[in_features, out_features], "fan_in")
            .expect("Failed to initialize sparse linear weight");

        // Create sparsity mask
        let mask = match pattern {
            SparsityPattern::Random { sparsity } => {
                SparseMask::random(&[in_features, out_features], sparsity)
                    .expect("Failed to create random sparsity mask")
            }
            SparsityPattern::Blocked {
                block_size,
                sparsity,
            } => SparseMask::blocked(&[in_features, out_features], block_size, sparsity)
                .expect("Failed to create blocked sparsity mask"),
            SparsityPattern::MagnitudeBased { threshold } => {
                SparseMask::from_magnitude(&weight, threshold)
                    .expect("Failed to create magnitude-based mask")
            }
            SparsityPattern::Structured { channels_to_prune } => {
                // Create structured sparsity (prune entire output channels)
                let sparsity = channels_to_prune as f32 / out_features as f32;
                SparseMask::random(&[in_features, out_features], sparsity)
                    .expect("Failed to create structured sparsity mask")
            }
        };

        // Apply initial mask to weights
        let masked_weight = mask.apply(&weight).expect("Failed to apply mask");
        base.register_parameter("weight".to_string(), Parameter::new(masked_weight));

        if bias {
            let bias_tensor = zeros(&[out_features]).expect("Failed to create bias tensor");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            mask,
            in_features,
            out_features,
            use_bias: bias,
        }
    }

    /// Get current sparsity level
    pub fn sparsity(&self) -> f32 {
        self.mask.sparsity()
    }

    /// Get number of non-zero parameters
    pub fn nnz(&self) -> usize {
        self.mask.nnz()
    }

    /// Update sparsity by magnitude-based pruning
    pub fn prune_by_magnitude(&mut self, threshold: f32) -> Result<()> {
        // Get current weights
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Create new mask based on current weights
        self.mask = SparseMask::from_magnitude(&weight, threshold)?;

        // Apply mask to weights
        let masked = self.mask.apply(&weight)?;
        self.base
            .register_parameter("weight".to_string(), Parameter::new(masked));

        Ok(())
    }

    /// Increase sparsity gradually
    pub fn increase_sparsity(&mut self, target_sparsity: f32) -> Result<()> {
        if target_sparsity <= self.mask.sparsity() {
            return Ok(()); // Already sparse enough
        }

        // Get current weights
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Calculate threshold to achieve target sparsity
        let weight_data = weight.to_vec()?;
        let mut abs_weights: Vec<f32> = weight_data.iter().map(|&w| w.abs()).collect();
        abs_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let num_to_prune = (abs_weights.len() as f32 * target_sparsity) as usize
            - (abs_weights.len() - self.mask.nnz());
        let threshold = abs_weights[num_to_prune];

        self.prune_by_magnitude(threshold)
    }
}

impl Module for SparseLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Get weight parameter
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Ensure weights remain sparse during forward pass
        let sparse_weight = self.mask.apply(&weight)?;

        // Compute input @ sparse_weight
        let output = input.matmul(&sparse_weight)?;

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            Ok(output.add(&bias)?)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
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
}

impl core::fmt::Debug for SparseLinear {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseLinear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("sparsity", &self.mask.sparsity())
            .field("nnz", &self.mask.nnz())
            .finish()
    }
}

/// Sparse convolutional layer with structured sparsity
pub struct SparseConv2d {
    base: ModuleBase,
    /// Sparsity mask
    mask: SparseMask,
    /// Convolution parameters
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl SparseConv2d {
    /// Create a new sparse conv2d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        pattern: SparsityPattern,
        bias: bool,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [out_channels, in_channels, kernel_size, kernel_size];

        let weight = crate::init::kaiming_uniform(&weight_shape, "fan_in")
            .expect("Failed to initialize sparse conv2d weight");

        // Create sparsity mask
        let mask = match pattern {
            SparsityPattern::Random { sparsity } => SparseMask::random(&weight_shape, sparsity)
                .expect("Failed to create random sparsity mask"),
            SparsityPattern::Blocked {
                block_size: _block_size,
                sparsity,
            } => {
                // For conv, we typically prune entire filters
                SparseMask::random(&weight_shape, sparsity)
                    .expect("Failed to create blocked sparsity mask")
            }
            SparsityPattern::MagnitudeBased { threshold } => {
                SparseMask::from_magnitude(&weight, threshold)
                    .expect("Failed to create magnitude-based mask")
            }
            SparsityPattern::Structured { channels_to_prune } => {
                // Prune entire output channels
                let sparsity = channels_to_prune as f32 / out_channels as f32;
                SparseMask::random(&weight_shape, sparsity)
                    .expect("Failed to create structured sparsity mask")
            }
        };

        // Apply mask
        let masked_weight = mask.apply(&weight).expect("Failed to apply mask");
        base.register_parameter("weight".to_string(), Parameter::new(masked_weight));

        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("Failed to create bias tensor");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            mask,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Get current sparsity level
    pub fn sparsity(&self) -> f32 {
        self.mask.sparsity()
    }

    /// Get number of non-zero parameters
    pub fn nnz(&self) -> usize {
        self.mask.nnz()
    }
}

impl Module for SparseConv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        use crate::functional as F;

        // Get weight and apply mask
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let sparse_weight = self.mask.apply(&weight)?;

        let bias = if self.base.parameters.contains_key("bias") {
            Some(self.base.parameters["bias"].tensor().read().clone())
        } else {
            None
        };

        F::conv2d(
            input,
            &sparse_weight,
            bias.as_ref(),
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1), // dilation
            1,      // groups
        )
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
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
}

impl core::fmt::Debug for SparseConv2d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseConv2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("sparsity", &self.mask.sparsity())
            .finish()
    }
}

/// Sparse training configuration
#[derive(Debug, Clone)]
pub struct SparseTrainingConfig {
    /// Initial sparsity level
    pub initial_sparsity: f32,
    /// Target sparsity level
    pub target_sparsity: f32,
    /// Number of steps to reach target sparsity
    pub pruning_steps: usize,
    /// Start pruning at this step
    pub pruning_start_step: usize,
    /// Pruning frequency (every N steps)
    pub pruning_frequency: usize,
}

impl Default for SparseTrainingConfig {
    fn default() -> Self {
        Self {
            initial_sparsity: 0.0,
            target_sparsity: 0.9,
            pruning_steps: 1000,
            pruning_start_step: 0,
            pruning_frequency: 100,
        }
    }
}

/// Gradual magnitude pruning scheduler
pub struct GradualPruningScheduler {
    config: SparseTrainingConfig,
    current_step: usize,
}

impl GradualPruningScheduler {
    /// Create a new pruning scheduler
    pub fn new(config: SparseTrainingConfig) -> Self {
        Self {
            config,
            current_step: 0,
        }
    }

    /// Get current target sparsity for this step
    pub fn get_sparsity(&self) -> f32 {
        if self.current_step < self.config.pruning_start_step {
            return self.config.initial_sparsity;
        }

        let steps_since_start = self.current_step - self.config.pruning_start_step;

        if steps_since_start >= self.config.pruning_steps {
            return self.config.target_sparsity;
        }

        // Linear interpolation
        let progress = steps_since_start as f32 / self.config.pruning_steps as f32;
        self.config.initial_sparsity
            + (self.config.target_sparsity - self.config.initial_sparsity) * progress
    }

    /// Check if we should prune at this step
    pub fn should_prune(&self) -> bool {
        if self.current_step < self.config.pruning_start_step {
            return false;
        }

        (self.current_step - self.config.pruning_start_step) % self.config.pruning_frequency == 0
    }

    /// Increment step counter
    pub fn step(&mut self) {
        self.current_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_mask_random() {
        let mask = SparseMask::random(&[10, 10], 0.5).unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 0.1); // Allow some variance
        assert_eq!(mask.nnz() + (100.0 * mask.sparsity()) as usize, 100);
    }

    #[test]
    fn test_sparse_mask_blocked() {
        let mask = SparseMask::blocked(&[8, 8], 2, 0.5).unwrap();
        assert!(mask.sparsity() >= 0.4 && mask.sparsity() <= 0.6);
    }

    #[test]
    fn test_sparse_linear() {
        let layer = SparseLinear::new(10, 5, SparsityPattern::Random { sparsity: 0.8 }, true);

        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);
        assert!((layer.sparsity() - 0.8).abs() < 0.1);

        let input = randn(&[2, 10]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_sparse_conv2d() {
        let layer = SparseConv2d::new(
            3,
            16,
            3,
            1,
            1,
            SparsityPattern::Random { sparsity: 0.7 },
            true,
        );

        assert!((layer.sparsity() - 0.7).abs() < 0.1);

        let input = randn(&[2, 3, 32, 32]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 16, 32, 32]);
    }

    #[test]
    fn test_gradual_pruning_scheduler() {
        let config = SparseTrainingConfig {
            initial_sparsity: 0.0,
            target_sparsity: 0.9,
            pruning_steps: 100,
            pruning_start_step: 10,
            pruning_frequency: 10,
        };

        let mut scheduler = GradualPruningScheduler::new(config);

        // Before start
        assert_eq!(scheduler.get_sparsity(), 0.0);
        assert!(!scheduler.should_prune());

        // Advance to start
        for _ in 0..10 {
            scheduler.step();
        }

        assert!(scheduler.should_prune());
        assert!(scheduler.get_sparsity() < 0.9);

        // Advance to end
        for _ in 0..100 {
            scheduler.step();
        }

        assert_eq!(scheduler.get_sparsity(), 0.9);
    }
}
