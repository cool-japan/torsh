//! Stochastic Gradient Descent (SGD) optimizer
//!
//! This module provides implementation of SGD, one of the most fundamental and widely-used
//! optimization algorithms in deep learning. SGD updates parameters by taking steps proportional
//! to the negative of the gradient.
//!
//! ## Stochastic Gradient Descent (SGD)
//!
//! SGD is the foundational optimization algorithm for neural networks. Despite being simple,
//! it remains highly effective and is often preferred for training large-scale models due to
//! its simplicity, robustness, and excellent generalization properties.
//!
//! ### Key Features:
//! - Simple and memory-efficient
//! - Excellent generalization properties
//! - Optional momentum for faster convergence
//! - Nesterov acceleration for improved momentum updates
//! - Well-understood theoretical properties
//!
//! ### When to Use SGD:
//! - **Computer Vision** - ResNet, VGG, and other CNN architectures train well with SGD+momentum
//! - **Large-scale training** - Lower memory footprint than adaptive methods (Adam, RMSprop)
//! - **When generalization matters** - Often achieves better test accuracy than Adam
//! - **Transfer learning** - Fine-tuning pre-trained models on new tasks
//! - **When you can tune hyperparameters** - Requires more tuning than Adam but often worth it
//!
//! ## Mathematical Formulation
//!
//! ### Basic SGD (no momentum):
//! ```text
//! θ_t = θ_{t-1} - α * g_t
//! ```
//!
//! ### SGD with Momentum:
//! ```text
//! v_t = μ * v_{t-1} + g_t                    // Velocity update
//! θ_t = θ_{t-1} - α * v_t                    // Parameter update
//! ```
//!
//! ### SGD with Momentum and Dampening:
//! ```text
//! v_t = μ * v_{t-1} + (1 - d) * g_t         // Dampened velocity
//! θ_t = θ_{t-1} - α * v_t                    // Parameter update
//! ```
//!
//! ### SGD with Nesterov Momentum:
//! ```text
//! v_t = μ * v_{t-1} + g_t                    // Velocity update
//! θ_t = θ_{t-1} - α * (g_t + μ * v_t)       // Look-ahead update
//! ```
//!
//! Where:
//! - `g_t` is the gradient at step t (with optional weight decay: g_t + λ * θ_{t-1})
//! - `v_t` is the velocity (momentum buffer)
//! - `μ` is the momentum coefficient (typically 0.9)
//! - `α` is the learning rate
//! - `d` is the dampening factor (typically 0.0)
//! - `λ` is the weight decay coefficient
//!
//! ## Examples
//!
//! ### Basic SGD (no momentum)
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::{SGD, Optimizer};
//! use torsh_tensor::Tensor;
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! // Create parameters
//! let weight = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let bias = Arc::new(RwLock::new(randn::<f32>(&[20])?));
//! let params = vec![weight.clone(), bias.clone()];
//!
//! // Create basic SGD optimizer
//! let mut optimizer = SGD::new(
//!     params,
//!     0.01,           // learning rate
//!     None,           // no momentum
//!     None,           // no dampening
//!     None,           // no weight decay
//!     false           // no Nesterov
//! );
//!
//! // Training loop
//! for _epoch in 0..100 {
//!     // ... compute gradients via backward() ...
//!
//!     // Update parameters
//!     optimizer.step()?;
//!     optimizer.zero_grad();
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### SGD with Momentum (Recommended for CV)
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::sgd::SGDBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 100])?))];
//!
//! // ImageNet-style training (ResNet, VGG, etc.)
//! let optimizer = SGDBuilder::new(0.1)        // Initial LR (often use scheduler)
//!     .momentum(0.9)                          // Standard momentum for CV
//!     .weight_decay(1e-4)                     // L2 regularization
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ### Complete Training Loop Example
//! ```rust
//! # use torsh_tensor::creation::{randn, zeros};
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::{SGD, Optimizer};
//! use torsh_tensor::Tensor;
//!
//! // Model parameters (e.g., from a neural network)
//! let weight = Arc::new(RwLock::new(randn::<f32>(&[784, 10])?));
//! let bias = Arc::new(RwLock::new(zeros::<f32>(&[10])?));
//! let params = vec![weight.clone(), bias.clone()];
//!
//! // Create SGD optimizer with momentum
//! let mut optimizer = SGD::new(
//!     params,
//!     0.01,           // learning rate
//!     Some(0.9),      // momentum
//!     None,           // dampening
//!     Some(5e-4),     // weight decay
//!     false           // Nesterov
//! );
//!
//! // Training loop
//! let epochs = 10;
//! let batches_per_epoch = 100;
//!
//! for epoch in 0..epochs {
//!     let mut total_loss = 0.0;
//!
//!     for batch in 0..batches_per_epoch {
//!         // Forward pass (simplified - actual implementation would use nn::Module)
//!         // let output = model.forward(&input)?;
//!         // let loss = criterion(&output, &target)?;
//!         // let loss_value = loss.to_vec()?[0];
//!         // total_loss += loss_value;
//!
//!         // Backward pass
//!         // loss.backward()?;
//!
//!         // Optimizer step
//!         optimizer.step()?;
//!
//!         // Clear gradients for next iteration
//!         optimizer.zero_grad();
//!     }
//!
//!     // Log progress
//!     // println!("Epoch {}: Loss = {:.4}", epoch, total_loss / batches_per_epoch as f32);
//!
//!     // Optional: Learning rate scheduling
//!     // if epoch > 0 && epoch % 30 == 0 {
//!     //     let current_lr = optimizer.get_lr()[0];
//!     //     optimizer.set_lr(current_lr * 0.1);  // Decay by 10x
//!     // }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced: Nesterov Momentum
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::sgd::SGDBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 50])?))];
//!
//! // Nesterov accelerated gradient (better momentum for some problems)
//! let optimizer = SGDBuilder::new(0.01)
//!     .momentum(0.9)
//!     .nesterov(true)        // Enable Nesterov acceleration
//!     .weight_decay(1e-4)
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ## Hyperparameter Guidelines
//!
//! ### Learning Rate:
//! - **Critical hyperparameter** - requires careful tuning
//! - **Computer Vision**: Start with 0.1, use learning rate scheduling
//! - **Fine-tuning**: Use 0.001-0.01 for pre-trained models
//! - **Rule of thumb**: Increase LR proportionally with batch size
//! - **Too high**: Training diverges or oscillates
//! - **Too low**: Very slow convergence
//!
//! ### Momentum:
//! - **Standard value**: 0.9 works well for most problems
//! - **Range**: 0.8-0.99 depending on problem
//! - **Higher momentum**: Faster convergence but less stable
//! - **Lower momentum**: More stable but slower
//! - **No momentum**: Only for very small models or specific problems
//!
//! ### Weight Decay:
//! - **Computer Vision**: 1e-4 to 5e-4 (standard for ImageNet)
//! - **Smaller datasets**: Higher values (1e-3 to 1e-2)
//! - **Large datasets**: Lower values (1e-5 to 1e-4)
//! - **Purpose**: L2 regularization to prevent overfitting
//!
//! ### Dampening:
//! - **Usually**: Keep at 0.0 (default)
//! - **Non-zero**: Only for specific optimization problems
//! - **Cannot use with Nesterov**: Nesterov requires dampening = 0
//!
//! ## Performance Tips
//!
//! ### Learning Rate Scheduling:
//! SGD often requires learning rate scheduling for best results:
//! - **Step decay**: Reduce LR by 10x every N epochs
//! - **Cosine annealing**: Smooth decay following cosine curve
//! - **Warm restarts**: Periodically reset to higher learning rate
//!
//! ### Batch Size Scaling:
//! When increasing batch size, scale learning rate proportionally:
//! - Batch size 256 with LR 0.1 → Batch size 512 with LR 0.2
//!
//! ### Gradient Clipping:
//! For RNNs or transformers, combine with gradient clipping:
//! ```rust,ignore
//! clip_grad_norm_(&params, max_norm=1.0);
//! optimizer.step()?;
//! ```
//!
//! ## Common Configurations
//!
//! ### ResNet (ImageNet training)
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::sgd::SGDBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 100])?))];
//! let resnet_optimizer = SGDBuilder::new(0.1)
//!     .momentum(0.9)
//!     .weight_decay(1e-4)
//!     .build(params);
//! // Use with step LR scheduler: decay by 10x at epochs 30, 60, 90
//! # Ok(())
//! # }
//! ```
//!
//! ### Transfer Learning (Fine-tuning)
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::sgd::SGDBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 100])?))];
//! let finetune_optimizer = SGDBuilder::new(0.001)  // Lower LR
//!     .momentum(0.9)
//!     .weight_decay(5e-4)                          // Higher regularization
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ## Troubleshooting
//!
//! ### Loss not decreasing:
//! 1. Learning rate too low - increase by 10x
//! 2. Gradients vanishing - check gradient norms
//! 3. Wrong initialization - use proper weight initialization
//!
//! ### Loss diverging (NaN):
//! 1. Learning rate too high - decrease by 10x
//! 2. Gradient explosion - add gradient clipping
//! 3. Numerical instability - check for inf/nan in data
//!
//! ### Slow convergence:
//! 1. Add momentum (0.9) if not using
//! 2. Increase learning rate
//! 3. Try learning rate warmup for first few epochs
//!
//! ### Poor generalization:
//! 1. Increase weight decay
//! 2. Use learning rate scheduling
//! 3. Consider data augmentation
//!
//! ## Comparison with Other Optimizers
//!
//! - **vs Adam**: SGD often generalizes better but requires more tuning
//! - **vs AdamW**: Use SGD for CV, AdamW for transformers
//! - **vs RMSprop**: SGD more stable, RMSprop adapts per-parameter
//!
//! ## See Also
//!
//! - [`Adam`](crate::Adam) - Adaptive learning rate optimizer
//! - [`AdamW`](crate::AdamW) - Adam with decoupled weight decay
//! - [`RMSprop`](crate::RMSprop) - Root mean square propagation
//!
//! ## References
//! - [Original SGD paper](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)
//! - [On the importance of initialization and momentum](http://proceedings.mlr.press/v28/sutskever13.html)
//! - [Nesterov Accelerated Gradient](http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf)

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
// Temporarily disable scirs2 integration
// use scirs2::optim::sgd::SGD as SciSGD;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// SGD optimizer with momentum and Nesterov acceleration
///
/// Stochastic Gradient Descent (SGD) is the foundational optimization algorithm for training
/// neural networks. Despite its simplicity, SGD with momentum remains one of the most effective
/// optimizers, particularly for computer vision tasks.
///
/// # Algorithm Overview
///
/// SGD updates parameters by moving in the direction of the negative gradient. With momentum,
/// SGD accumulates a velocity vector in directions of persistent gradient reduction, which
/// helps accelerate convergence and dampen oscillations.
///
/// # Parameters
///
/// * `lr` - Learning rate (required). Typical range: [1e-4, 0.1]
///   - **Computer Vision**: Start with 0.1, use learning rate scheduling
///   - **Fine-tuning**: 0.001-0.01 for pre-trained models
///   - **Critical**: Requires more careful tuning than Adam
/// * `momentum` - Momentum factor (default: 0.0). Typical: 0.9
///   - Accelerates convergence by accumulating velocity
///   - 0.0 = no momentum, 0.9 = standard for CV, 0.99 = high momentum
/// * `dampening` - Dampening for momentum (default: 0.0)
///   - Usually kept at 0.0
///   - Must be 0.0 if using Nesterov
/// * `weight_decay` - L2 regularization coefficient (default: 0.0)
///   - Computer Vision: 1e-4 to 5e-4
///   - Helps prevent overfitting
/// * `nesterov` - Enable Nesterov momentum (default: false)
///   - Improved momentum with look-ahead
///   - Requires momentum > 0 and dampening = 0
///
/// # When to Use SGD
///
/// SGD is preferred when:
/// - Training convolutional neural networks (ResNet, VGG, EfficientNet)
/// - Generalization performance is critical
/// - You can afford hyperparameter tuning
/// - Training large-scale computer vision models
/// - Fine-tuning pre-trained models
///
/// # Performance Characteristics
///
/// - **Memory Usage**: Minimal (only momentum buffer if enabled)
/// - **Convergence Speed**: Moderate (requires good hyperparameters)
/// - **Hyperparameter Sensitivity**: High (learning rate critical)
/// - **Generalization**: Excellent (often better than adaptive methods)
///
/// # Example: Training a Simple Model
///
/// ```rust
/// # use torsh_tensor::creation::{randn, zeros};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_optim::{SGD, Optimizer};
/// use torsh_tensor::Tensor;
/// use parking_lot::RwLock;
/// use std::sync::Arc;
///
/// // Create model parameters
/// let weight = Arc::new(RwLock::new(randn::<f32>(&[784, 10])?));
/// let bias = Arc::new(RwLock::new(zeros::<f32>(&[10])?));
/// let params = vec![weight, bias];
///
/// // Create SGD optimizer with momentum
/// let mut optimizer = SGD::new(
///     params,
///     0.01,           // learning rate
///     Some(0.9),      // momentum
///     None,           // dampening
///     Some(1e-4),     // weight decay
///     false           // Nesterov
/// );
///
/// // Training step
/// // ... forward pass and loss computation ...
/// // loss.backward()?;
/// optimizer.step()?;
/// optimizer.zero_grad();
/// # Ok(())
/// # }
/// ```
///
/// # Example: Using the Builder Pattern
///
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # use parking_lot::RwLock;
/// # use std::sync::Arc;
/// # fn main() -> Result<()> {
/// use torsh_optim::sgd::SGDBuilder;
///
/// let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 50])?))];
///
/// let optimizer = SGDBuilder::new(0.01)
///     .momentum(0.9)
///     .weight_decay(1e-4)
///     .nesterov(true)
///     .build(params);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct SGD {
    base: BaseOptimizer,
    #[allow(dead_code)]
    momentum: f32,
    #[allow(dead_code)]
    dampening: f32,
    #[allow(dead_code)]
    weight_decay: f32,
    #[allow(dead_code)]
    nesterov: bool,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        momentum: Option<f32>,
        dampening: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: bool,
    ) -> Self {
        let momentum = momentum.unwrap_or(0.0);
        let dampening = dampening.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);

        if nesterov && (momentum <= 0.0 || dampening != 0.0) {
            panic!("Nesterov momentum requires a momentum and zero dampening");
        }

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("momentum".to_string(), momentum);
        defaults.insert("dampening".to_string(), dampening);
        defaults.insert("weight_decay".to_string(), weight_decay);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "SGD".to_string(),
            defaults,
        };

        Self {
            base,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> OptimizerResult<()> {
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param
                    .grad()
                    .expect("gradient should exist after has_grad check");
                let param_id = format!("{:p}", param_arc.as_ref());

                // Apply weight decay to gradient if specified
                let mut d_p = grad;
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param
                        .mul_scalar(self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    d_p = d_p
                        .add(&weight_decay_term)
                        .map_err(OptimizerError::TensorError)?;
                }

                if self.momentum != 0.0 {
                    // Get or initialize momentum buffer
                    let needs_init = !self.base.state.contains_key(&param_id);
                    let state = self
                        .base
                        .state
                        .entry(param_id.clone())
                        .or_insert_with(HashMap::new);

                    if needs_init {
                        state.insert(
                            "momentum_buffer".to_string(),
                            torsh_tensor::creation::zeros_like(&param)?,
                        );
                    }

                    let mut buf = state
                        .get("momentum_buffer")
                        .expect("momentum_buffer state should exist")
                        .clone();

                    // Update momentum buffer: buf = momentum * buf + (1 - dampening) * d_p
                    buf.mul_scalar_(self.momentum)
                        .map_err(OptimizerError::TensorError)?;
                    let grad_term = d_p
                        .mul_scalar(1.0 - self.dampening)
                        .map_err(OptimizerError::TensorError)?;
                    buf = buf.add(&grad_term).map_err(OptimizerError::TensorError)?;

                    if self.nesterov {
                        // Nesterov momentum: d_p = d_p + momentum * buf
                        let nesterov_term = buf
                            .mul_scalar(self.momentum)
                            .map_err(OptimizerError::TensorError)?;
                        d_p = d_p
                            .add(&nesterov_term)
                            .map_err(OptimizerError::TensorError)?;
                    } else {
                        // Standard momentum: d_p = buf
                        d_p = buf.clone();
                    }

                    // Update state
                    state.insert("momentum_buffer".to_string(), buf);
                }

                // Apply update: param = param - lr * d_p
                let update = d_p
                    .mul_scalar(group.lr)
                    .map_err(OptimizerError::TensorError)?;
                *param = param.sub(&update).map_err(OptimizerError::TensorError)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.base.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base.load_state_dict(state)
    }
}

/// Builder for SGD optimizer
pub struct SGDBuilder {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
}

impl SGDBuilder {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> SGD {
        SGD::new(
            params,
            self.lr,
            Some(self.momentum),
            Some(self.dampening),
            Some(self.weight_decay),
            self.nesterov,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_sgd_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let params = vec![param];

        let optimizer = SGD::new(params, 0.1, Some(0.9), None, None, false);
        assert_eq!(optimizer.base.get_lr(), vec![0.1]);
        Ok(())
    }

    #[test]
    fn test_sgd_builder() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let params = vec![param];

        let optimizer = SGDBuilder::new(0.01)
            .momentum(0.9)
            .weight_decay(1e-4)
            .nesterov(true)
            .build(params);

        assert_eq!(optimizer.momentum, 0.9);
        assert_eq!(optimizer.weight_decay, 1e-4);
        assert!(optimizer.nesterov);
        Ok(())
    }
}
