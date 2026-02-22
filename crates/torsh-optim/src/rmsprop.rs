//! RMSprop (Root Mean Square Propagation) optimizer
//!
//! This module provides implementation of RMSprop, an adaptive learning rate optimization
//! algorithm designed to address some of the shortcomings of AdaGrad by using a moving
//! average of squared gradients.
//!
//! ## RMSprop (Root Mean Square Propagation)
//!
//! RMSprop is an adaptive learning rate optimizer that divides the learning rate by an
//! exponentially decaying average of squared gradients. It was developed to address the
//! diminishing learning rates problem in AdaGrad and works well for non-stationary objectives.
//!
//! ### Key Features:
//! - Adaptive per-parameter learning rates
//! - Uses exponential moving average (doesn't accumulate all history like AdaGrad)
//! - Works well for RNNs and online/non-stationary problems
//! - Optional momentum and centered variants
//! - Memory efficient compared to Adam
//!
//! ### When to Use RMSprop:
//! - **Recurrent Neural Networks (RNNs/LSTMs)** - historically popular for RNN training
//! - **Non-stationary problems** - where data distribution changes over time
//! - **Online learning** - when training on streaming data
//! - **When Adam is unstable** - RMSprop can be more stable in some cases
//! - **Memory constraints** - slightly lower memory usage than Adam
//!
//! ## Mathematical Formulation
//!
//! ### Basic RMSprop:
//! ```text
//! E[g²]_t = α * E[g²]_{t-1} + (1 - α) * g_t²     // Update squared gradient average
//! θ_t = θ_{t-1} - lr * g_t / (√E[g²]_t + ε)      // Parameter update
//! ```
//!
//! ### RMSprop with Momentum:
//! ```text
//! E[g²]_t = α * E[g²]_{t-1} + (1 - α) * g_t²     // Squared gradient average
//! v_t = μ * v_{t-1} + g_t / (√E[g²]_t + ε)       // Momentum buffer
//! θ_t = θ_{t-1} - lr * v_t                        // Parameter update
//! ```
//!
//! ### Centered RMSprop:
//! ```text
//! E[g²]_t = α * E[g²]_{t-1} + (1 - α) * g_t²     // Squared gradient average
//! E[g]_t = α * E[g]_{t-1} + (1 - α) * g_t        // Gradient average
//! variance = E[g²]_t - (E[g]_t)²                  // Centered variance
//! θ_t = θ_{t-1} - lr * g_t / (√variance + ε)     // Parameter update
//! ```
//!
//! Where:
//! - `g_t` is the gradient at step t (with optional weight decay)
//! - `E[g²]_t` is the moving average of squared gradients
//! - `E[g]_t` is the moving average of gradients (centered variant)
//! - `α` is the smoothing constant (typically 0.99)
//! - `lr` is the learning rate (typically 1e-2 to 1e-3)
//! - `ε` is numerical stability constant (typically 1e-8)
//! - `μ` is momentum coefficient (typically 0.0 or 0.9)
//!
//! ## Examples
//!
//! ### Basic Usage
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::prelude::{RMSprop, Optimizer};
//! use torsh_tensor::Tensor;
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! // Create parameters
//! let weight = Arc::new(RwLock::new(randn::<f32>(&[128, 64])?));
//! let bias = Arc::new(RwLock::new(randn::<f32>(&[64])?));
//! let params = vec![weight, bias];
//!
//! // Create RMSprop optimizer with default settings
//! let mut optimizer = RMSprop::new(
//!     params,
//!     Some(1e-2),     // learning rate
//!     None,           // alpha (default: 0.99)
//!     None,           // eps (default: 1e-8)
//!     None,           // weight decay
//!     None,           // momentum
//!     false           // not centered
//! );
//!
//! // Training loop
//! for _epoch in 0..100 {
//!     // ... compute gradients via backward() ...
//!
//!     // Optimizer step
//!     // optimizer.step()?;
//!     // optimizer.zero_grad();
//! }
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
//! use torsh_optim::prelude::{RMSprop, Optimizer};
//! use torsh_tensor::Tensor;
//!
//! // Model parameters (e.g., LSTM layers)
//! let lstm_weight = Arc::new(RwLock::new(randn::<f32>(&[256, 512])?));
//! let lstm_bias = Arc::new(RwLock::new(zeros::<f32>(&[512])?));
//! let output_weight = Arc::new(RwLock::new(randn::<f32>(&[512, 10])?));
//! let params = vec![lstm_weight, lstm_bias, output_weight];
//!
//! // Create RMSprop optimizer (good for RNNs)
//! let mut optimizer = RMSprop::new(
//!     params,
//!     Some(1e-3),     // learning rate
//!     Some(0.99),     // alpha (smoothing)
//!     Some(1e-8),     // epsilon
//!     Some(1e-5),     // light weight decay
//!     Some(0.0),      // no momentum
//!     false           // standard RMSprop
//! );
//!
//! // Training loop
//! let epochs = 50;
//! let batches_per_epoch = 200;
//!
//! for epoch in 0..epochs {
//!     let mut epoch_loss = 0.0;
//!
//!     for batch in 0..batches_per_epoch {
//!         // Forward pass (simplified)
//!         // let output = model.forward(&input)?;
//!         // let loss = criterion(&output, &target)?;
//!         // epoch_loss += loss.to_vec()?[0];
//!
//!         // Backward pass
//!         // loss.backward()?;
//!
//!         // Gradient clipping for RNNs (recommended)
//!         // clip_grad_norm_(&params, 5.0);
//!
//!         // Optimizer step (when gradients are available)
//!         // optimizer.step()?;
//!         // optimizer.zero_grad();
//!     }
//!
//!     // Log progress
//!     // let avg_loss = epoch_loss / batches_per_epoch as f32;
//!     // println!("Epoch {}: Loss = {:.4}", epoch, avg_loss);
//!
//!     // Optional: Learning rate decay
//!     // if epoch > 0 && epoch % 10 == 0 {
//!     //     let current_lr = optimizer.get_lr()[0];
//!     //     optimizer.set_lr(current_lr * 0.5);  // Decay by half
//!     // }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Using the Builder Pattern
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::rmsprop::RMSpropBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 50])?))];
//!
//! // RMSprop with momentum and centering
//! let optimizer = RMSpropBuilder::new()
//!     .lr(1e-3)
//!     .alpha(0.95)           // Faster adaptation
//!     .momentum(0.9)         // Add momentum
//!     .centered(true)        // Use centered variant
//!     .weight_decay(1e-5)
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ### RNN/LSTM Training Configuration
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::rmsprop::RMSpropBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[256, 512])?))];
//!
//! // Classic RNN/LSTM setup
//! let rnn_optimizer = RMSpropBuilder::new()
//!     .lr(1e-3)              // Conservative LR for RNNs
//!     .alpha(0.99)           // Standard smoothing
//!     .eps(1e-8)
//!     .build(params);
//! // Combine with gradient clipping (max_norm=5.0)
//! # Ok(())
//! # }
//! ```
//!
//! ## Hyperparameter Guidelines
//!
//! ### Learning Rate:
//! - **Default**: 1e-2 (higher than Adam's default)
//! - **RNN/LSTM**: 1e-3 to 1e-4 (more conservative)
//! - **Computer Vision**: 1e-3 (if using instead of SGD/Adam)
//! - **Fine-tuning**: 1e-4 to 1e-5
//! - **Range**: [1e-5, 1e-1] depending on problem
//!
//! ### Alpha (Smoothing Constant):
//! - **Default**: 0.99 (strong smoothing)
//! - **Faster adaptation**: 0.9-0.95
//! - **More stable**: 0.99-0.999
//! - **Effect**: Higher α = smoother updates, lower α = more responsive
//!
//! ### Epsilon:
//! - **Default**: 1e-8 (numerical stability)
//! - **FP16 training**: Use 1e-4 or 1e-6
//! - **FP32 training**: 1e-8 is fine
//! - **Purpose**: Prevents division by zero
//!
//! ### Weight Decay:
//! - **Default**: 0.0 (no regularization)
//! - **Light regularization**: 1e-5 to 1e-4
//! - **Strong regularization**: 1e-3 to 1e-2
//! - **Note**: Applied to gradients (L2 penalty)
//!
//! ### Momentum:
//! - **Default**: 0.0 (no momentum)
//! - **With momentum**: 0.9 (standard value)
//! - **Effect**: Accelerates convergence, smooths updates
//! - **When to use**: Non-convex problems, noisy gradients
//!
//! ### Centered:
//! - **Default**: false (standard RMSprop)
//! - **Centered**: true (uses variance instead of second moment)
//! - **Effect**: Can improve convergence on some problems
//! - **Cost**: Slight increase in memory and computation
//!
//! ## Performance Tips
//!
//! ### For RNN/LSTM Training:
//! ```rust,ignore
//! // Recommended configuration
//! let optimizer = RMSpropBuilder::new()
//!     .lr(1e-3)
//!     .alpha(0.99)
//!     .build(params);
//!
//! // Always use gradient clipping with RNNs
//! clip_grad_norm_(&params, max_norm=5.0);
//! optimizer.step()?;
//! ```
//!
//! ### Learning Rate Scheduling:
//! Unlike SGD, RMSprop works reasonably well without aggressive scheduling:
//! - **Simple decay**: Reduce by 2-5x when validation loss plateaus
//! - **ReduceLROnPlateau**: Automatic reduction when metrics stagnate
//! - **Exponential decay**: Gentle continuous decay
//!
//! ### Gradient Clipping:
//! Essential for RNN/LSTM training with RMSprop:
//! - Clip by norm: typically 5.0 for RNNs
//! - Prevents gradient explosion
//! - Apply before optimizer.step()
//!
//! ## Common Configurations
//!
//! ### RNN/LSTM (Standard)
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::rmsprop::RMSpropBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 100])?))];
//! let rnn_opt = RMSpropBuilder::new()
//!     .lr(1e-3)
//!     .alpha(0.99)
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ### Deep Q-Networks (DQN) - Reinforcement Learning
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::rmsprop::RMSpropBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 100])?))];
//! let dqn_opt = RMSpropBuilder::new()
//!     .lr(2.5e-4)            // Lower LR for RL
//!     .alpha(0.95)           // Faster adaptation
//!     .eps(1e-5)             // Slightly higher epsilon
//!     .centered(true)        // Often helps in RL
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ### Online Learning / Streaming Data
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::rmsprop::RMSpropBuilder;
//!
//! let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 100])?))];
//! let online_opt = RMSpropBuilder::new()
//!     .lr(1e-2)              // Higher LR for online learning
//!     .alpha(0.9)            // More responsive to distribution changes
//!     .momentum(0.9)         // Add momentum for stability
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ## Troubleshooting
//!
//! ### Loss not decreasing:
//! 1. **Learning rate too low** - Increase to 1e-3 or 1e-2
//! 2. **Alpha too high** - Try lower alpha (0.9-0.95) for faster adaptation
//! 3. **Check gradients** - Ensure gradients are flowing (not zero/nan)
//!
//! ### Training unstable (loss oscillates):
//! 1. **Learning rate too high** - Reduce to 1e-4 or lower
//! 2. **Add gradient clipping** - Especially for RNNs (clip_norm=5.0)
//! 3. **Try centered variant** - Can improve stability
//! 4. **Add momentum** - Helps smooth updates (0.9)
//!
//! ### Slow convergence:
//! 1. **Increase learning rate** - Try 10x higher
//! 2. **Decrease alpha** - Faster adaptation (0.9 instead of 0.99)
//! 3. **Add momentum** - Accelerates convergence
//!
//! ### Gradient explosion (for RNNs):
//! 1. **Mandatory gradient clipping** - Use clip_grad_norm_(max_norm=5.0)
//! 2. **Lower learning rate** - Try 1e-4
//! 3. **Check initialization** - Use proper weight initialization
//!
//! ## Comparison with Other Optimizers
//!
//! - **vs Adam**: RMSprop is simpler, no bias correction, historically popular for RNNs
//! - **vs SGD**: RMSprop has adaptive learning rates, better for non-stationary problems
//! - **vs AdaGrad**: RMSprop doesn't accumulate all history, avoids diminishing LR
//! - **When to choose**: Use for RNNs, online learning, or when Adam is unstable
//!
//! ## See Also
//!
//! - [`Adam`](crate::Adam) - Combines RMSprop with momentum and bias correction
//! - [`SGD`](crate::SGD) - Simple gradient descent with optional momentum
//! - [`AdamW`](crate::AdamW) - Adam with decoupled weight decay
//!
//! ## References
//! - [Hinton's Coursera Lecture (Slide 29)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
//! - [Neural Network Tricks of the Trade](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_20)

use crate::{
    Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroup, ParamGroupState,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// RMSprop optimizer with optional momentum and centered variants
///
/// RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimizer that
/// uses a moving average of squared gradients to normalize updates. It was specifically
/// designed to work well with mini-batch learning and non-stationary objectives.
///
/// # Algorithm Overview
///
/// RMSprop maintains a moving average of squared gradients for each parameter and
/// divides the gradient by the square root of this average. This allows parameters
/// with large gradient magnitudes to have their learning rates automatically reduced,
/// while parameters with small gradients get effectively larger learning rates.
///
/// # Parameters
///
/// * `lr` - Learning rate (default: 1e-2). Typical range: [1e-5, 1e-1]
///   - **RNN/LSTM**: 1e-3 to 1e-4
///   - **Computer Vision**: 1e-3
///   - **Reinforcement Learning**: 2.5e-4 (DQN standard)
/// * `alpha` - Smoothing constant (default: 0.99). Range: [0.9, 0.999]
///   - Higher values = smoother updates (more history)
///   - Lower values = faster adaptation to recent gradients
/// * `eps` - Numerical stability constant (default: 1e-8)
///   - Use 1e-4 or 1e-6 for FP16 training
/// * `weight_decay` - L2 penalty coefficient (default: 0.0)
///   - Applied to gradients (not decoupled like AdamW)
///   - Typical range: [0.0, 1e-3]
/// * `momentum` - Momentum factor (default: 0.0)
///   - Add momentum for faster convergence
///   - Typical value: 0.9 when used
/// * `centered` - Use centered RMSprop (default: false)
///   - Computes variance instead of second moment
///   - Can improve convergence but increases memory usage
///
/// # When to Use RMSprop
///
/// RMSprop is well-suited for:
/// - Recurrent neural networks (RNNs, LSTMs, GRUs)
/// - Reinforcement learning (DQN and variants)
/// - Online learning scenarios
/// - Non-stationary optimization problems
/// - When Adam is unstable or not converging well
///
/// # Performance Characteristics
///
/// - **Memory Usage**: Moderate (stores squared gradient average, optional momentum buffer)
/// - **Convergence Speed**: Fast for RNNs, moderate for other tasks
/// - **Hyperparameter Sensitivity**: Moderate (less sensitive than SGD, more than Adam)
/// - **Generalization**: Good, especially for RNN tasks
///
/// # Example: Training an RNN
///
/// ```rust
/// # use torsh_tensor::creation::{randn, zeros};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_optim::prelude::{RMSprop, Optimizer};
/// use torsh_tensor::Tensor;
/// use parking_lot::RwLock;
/// use std::sync::Arc;
///
/// // LSTM parameters
/// let weight_ih = Arc::new(RwLock::new(randn::<f32>(&[256, 512])?));
/// let weight_hh = Arc::new(RwLock::new(randn::<f32>(&[512, 512])?));
/// let bias = Arc::new(RwLock::new(zeros::<f32>(&[512])?));
/// let params = vec![weight_ih, weight_hh, bias];
///
/// // Create RMSprop optimizer (good for RNNs)
/// let mut optimizer = RMSprop::new(
///     params,
///     Some(1e-3),     // learning rate
///     Some(0.99),     // alpha
///     Some(1e-8),     // eps
///     None,           // no weight decay
///     None,           // no momentum
///     false           // standard RMSprop
/// );
///
/// // Training step
/// // ... forward pass and loss computation ...
/// // loss.backward()?;
/// // Gradient clipping recommended for RNNs
/// // clip_grad_norm_(&params, 5.0);
/// // optimizer.step()?;
/// // optimizer.zero_grad();
/// # Ok(())
/// # }
/// ```
///
/// # Example: Using the Builder
///
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # use parking_lot::RwLock;
/// # use std::sync::Arc;
/// # fn main() -> Result<()> {
/// use torsh_optim::rmsprop::RMSpropBuilder;
///
/// let params = vec![Arc::new(RwLock::new(randn::<f32>(&[100, 50])?))];
///
/// let optimizer = RMSpropBuilder::new()
///     .lr(1e-3)
///     .alpha(0.99)
///     .momentum(0.9)
///     .centered(true)
///     .build(params);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RMSprop {
    param_groups: Vec<ParamGroup>,
    state: HashMap<String, HashMap<String, Tensor>>,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

impl RMSprop {
    /// Create a new RMSprop optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `lr` - Learning rate (default: 1e-2)
    /// * `alpha` - Smoothing constant (default: 0.99)
    /// * `eps` - Term added to the denominator to improve numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay (L2 penalty) (default: 0.0)
    /// * `momentum` - Momentum factor (default: 0.0)
    /// * `centered` - If True, compute the centered RMSprop
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        alpha: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-2);
        let alpha = alpha.unwrap_or(0.99);
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let momentum = momentum.unwrap_or(0.0);

        let param_group = ParamGroup::new(params, lr);

        Self {
            param_groups: vec![param_group],
            state: HashMap::new(),
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
        }
    }

    fn get_param_id(param: &Arc<RwLock<Tensor>>) -> String {
        format!("{:p}", Arc::as_ptr(param))
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> OptimizerResult<()> {
        for group in &self.param_groups {
            for param_arc in &group.params {
                let param_id = Self::get_param_id(param_arc);
                let param_read = param_arc.read();

                let grad = param_read.grad().ok_or_else(|| {
                    OptimizerError::TensorError(TorshError::invalid_argument_with_context(
                        "Parameter has no gradient",
                        "rmsprop_step",
                    ))
                })?;

                // Apply weight decay to gradient if specified
                let mut grad_to_use = grad.clone();
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param_read
                        .mul_scalar(self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    grad_to_use = grad_to_use
                        .add(&weight_decay_term)
                        .map_err(OptimizerError::TensorError)?;
                }

                // Get or initialize optimizer state
                let param_state = self.state.entry(param_id.clone()).or_default();

                let square_avg = if !param_state.contains_key("square_avg") {
                    let square_avg =
                        Tensor::zeros_like(&param_read).map_err(OptimizerError::TensorError)?;
                    param_state.insert("square_avg".to_string(), square_avg.clone());
                    square_avg
                } else {
                    param_state
                        .get("square_avg")
                        .expect("square_avg state should exist")
                        .clone()
                };

                // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
                let grad_squared = grad_to_use
                    .mul_op(&grad_to_use)
                    .map_err(OptimizerError::TensorError)?;
                let new_square_avg = square_avg
                    .mul_scalar(self.alpha)
                    .map_err(OptimizerError::TensorError)?
                    .add(
                        &grad_squared
                            .mul_scalar(1.0 - self.alpha)
                            .map_err(OptimizerError::TensorError)?,
                    )
                    .map_err(OptimizerError::TensorError)?;
                param_state.insert("square_avg".to_string(), new_square_avg.clone());

                let avg = if self.centered {
                    // Centered RMSprop: use variance instead of second moment
                    let grad_avg = if !param_state.contains_key("grad_avg") {
                        let grad_avg =
                            Tensor::zeros_like(&param_read).map_err(OptimizerError::TensorError)?;
                        param_state.insert("grad_avg".to_string(), grad_avg.clone());
                        grad_avg
                    } else {
                        param_state
                            .get("grad_avg")
                            .expect("grad_avg state should exist")
                            .clone()
                    };

                    // Update gradient average: grad_avg = alpha * grad_avg + (1 - alpha) * grad
                    let new_grad_avg = grad_avg
                        .mul_scalar(self.alpha)
                        .map_err(OptimizerError::TensorError)?
                        .add(
                            &grad_to_use
                                .mul_scalar(1.0 - self.alpha)
                                .map_err(OptimizerError::TensorError)?,
                        )
                        .map_err(OptimizerError::TensorError)?;
                    param_state.insert("grad_avg".to_string(), new_grad_avg.clone());

                    // Compute variance: square_avg - grad_avg^2
                    let grad_avg_squared = new_grad_avg
                        .mul_op(&new_grad_avg)
                        .map_err(OptimizerError::TensorError)?;
                    let variance = new_square_avg
                        .sub(&grad_avg_squared)
                        .map_err(OptimizerError::TensorError)?;
                    variance
                        .sqrt()
                        .map_err(OptimizerError::TensorError)?
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                } else {
                    // Standard RMSprop: avg = sqrt(square_avg) + eps
                    new_square_avg
                        .sqrt()
                        .map_err(OptimizerError::TensorError)?
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                };

                let mut update = grad_to_use.div(&avg).map_err(OptimizerError::TensorError)?;

                if self.momentum != 0.0 {
                    // Apply momentum to the update
                    let momentum_buffer = if !param_state.contains_key("momentum_buffer") {
                        let buf =
                            Tensor::zeros_like(&param_read).map_err(OptimizerError::TensorError)?;
                        param_state.insert("momentum_buffer".to_string(), buf.clone());
                        buf
                    } else {
                        param_state
                            .get("momentum_buffer")
                            .expect("momentum_buffer state should exist")
                            .clone()
                    };

                    // Update momentum buffer: buf = momentum * buf + update
                    let new_buf = momentum_buffer
                        .mul_scalar(self.momentum)
                        .map_err(OptimizerError::TensorError)?
                        .add(&update)
                        .map_err(OptimizerError::TensorError)?;
                    param_state.insert("momentum_buffer".to_string(), new_buf.clone());
                    update = new_buf;
                }

                // Apply update: param = param - lr * update
                drop(param_read);
                let mut param_write = param_arc.write();
                let step_update = update
                    .mul_scalar(group.lr)
                    .map_err(OptimizerError::TensorError)?;
                *param_write = param_write
                    .sub(&step_update)
                    .map_err(OptimizerError::TensorError)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &self.param_groups {
            for param in &group.params {
                param.write().zero_grad();
            }
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        self.param_groups.iter().map(|g| g.lr).collect()
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        let lr = options.get("lr").copied().unwrap_or(1e-2);
        let group = ParamGroup::new(params, lr).with_options(options);
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let param_groups = self
            .param_groups
            .iter()
            .map(|g| ParamGroupState {
                lr: g.lr,
                options: g.options.clone(),
                param_count: g.params.len(),
            })
            .collect();

        Ok(OptimizerState {
            optimizer_type: "RMSProp".to_string(),
            version: "0.1.0".to_string(),
            param_groups,
            state: self.state.clone(),
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        if state.param_groups.len() != self.param_groups.len() {
            return Err(
                TorshError::InvalidArgument("Parameter group count mismatch".to_string()).into(),
            );
        }

        for (i, group_state) in state.param_groups.iter().enumerate() {
            self.param_groups[i].lr = group_state.lr;
            self.param_groups[i].options = group_state.options.clone();
        }

        self.state = state.state;
        Ok(())
    }
}

/// Builder for RMSprop optimizer
pub struct RMSpropBuilder {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

impl RMSpropBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> RMSprop {
        RMSprop::new(
            params,
            Some(self.lr),
            Some(self.alpha),
            Some(self.eps),
            Some(self.weight_decay),
            Some(self.momentum),
            self.centered,
        )
    }
}

impl Default for RMSpropBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::error::TorshError;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_rmsprop_creation() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = RMSprop::new(params, None, None, None, None, None, false);
        assert_eq!(optimizer.get_lr()[0], 1e-2);
        Ok(())
    }

    #[test]
    fn test_rmsprop_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = RMSprop::new(params, Some(0.1), None, None, None, None, false);

        optimizer.step()?;
        Ok(())
    }

    #[test]
    fn test_rmsprop_centered() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = RMSprop::new(params, Some(0.1), None, None, None, None, true);

        optimizer.step()?;
        Ok(())
    }

    #[test]
    fn test_rmsprop_with_momentum() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));
        let mut param_write = param.write();
        param_write.set_grad(Some(randn::<f32>(&[2, 2])?));
        drop(param_write);

        let params = vec![param];
        let mut optimizer = RMSprop::new(params, Some(0.1), None, None, None, Some(0.9), false);

        optimizer.step()?;
        Ok(())
    }

    #[test]
    fn test_rmsprop_builder() -> OptimizerResult<()> {
        let params = vec![Arc::new(RwLock::new(randn::<f32>(&[2, 2])?))];
        let optimizer = RMSpropBuilder::new()
            .lr(0.01)
            .alpha(0.95)
            .eps(1e-7)
            .weight_decay(0.01)
            .momentum(0.8)
            .centered(true)
            .build(params);

        assert_eq!(optimizer.get_lr()[0], 0.01);
        assert_eq!(optimizer.alpha, 0.95);
        assert_eq!(optimizer.eps, 1e-7);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.momentum, 0.8);
        assert!(optimizer.centered);

        Ok(())
    }
}
