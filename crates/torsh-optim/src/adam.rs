//! Adam and AdamW optimizers
//!
//! This module provides implementations of the Adam and AdamW optimizers, two of the most
//! popular adaptive learning rate optimization algorithms for deep learning.
//!
//! ## Adam (Adaptive Moment Estimation)
//!
//! Adam combines the best properties of AdaGrad and RMSprop algorithms to provide an optimizer
//! that can handle sparse gradients on noisy problems. It computes adaptive learning rates for
//! each parameter by maintaining running averages of both the gradients and the second moments
//! of the gradients.
//!
//! ### Key Features:
//! - Adaptive learning rates per parameter
//! - Momentum-based updates with bias correction
//! - Robust to noisy gradients and sparse features
//! - Generally works well with default hyperparameters
//!
//! ### When to Use Adam:
//! - **General purpose optimization** - good starting point for most problems
//! - **Computer Vision** - works well for CNN training
//! - **Natural Language Processing** - excellent for transformer models
//! - **Noisy or sparse gradients** - handles these cases better than SGD
//!
//! ## AdamW (Adam with Decoupled Weight Decay)
//!
//! AdamW fixes a well-known issue with Adam's weight decay implementation. While Adam applies
//! weight decay to the gradients (L2 regularization), AdamW applies it directly to the parameters,
//! which provides better generalization and is theoretically more sound.
//!
//! ### Key Differences from Adam:
//! - **Decoupled weight decay** - applied directly to parameters, not gradients
//! - **Better generalization** - especially important for transformer models
//! - **Theoretical soundness** - proper implementation of weight decay
//!
//! ### When to Use AdamW:
//! - **Transformer models** - de facto standard for BERT, GPT, etc.
//! - **When weight decay is important** - better regularization properties
//! - **Modern deep learning** - generally preferred over Adam in recent research
//!
//! ## Mathematical Formulation
//!
//! ### Adam Algorithm:
//! ```text
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          // Update biased first moment estimate
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         // Update biased second moment estimate
//! m̂_t = m_t / (1 - β₁^t)                       // Compute bias-corrected first moment
//! v̂_t = v_t / (1 - β₂^t)                       // Compute bias-corrected second moment
//! θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        // Update parameters
//! ```
//!
//! ### AdamW Algorithm:
//! ```text
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          // Update first moment (same as Adam)
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         // Update second moment (same as Adam)
//! m̂_t = m_t / (1 - β₁^t)                       // Bias correction (same as Adam)
//! v̂_t = v_t / (1 - β₂^t)                       // Bias correction (same as Adam)
//! θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})  // Decoupled weight decay
//! ```
//!
//! Where:
//! - `g_t` is the gradient at step t
//! - `m_t, v_t` are the first and second moment estimates
//! - `β₁, β₂` are exponential decay rates (typically 0.9, 0.999)
//! - `α` is the learning rate
//! - `ε` is a small constant for numerical stability (typically 1e-8)
//! - `λ` is the weight decay coefficient (AdamW only)
//!
//! ## Examples
//!
//! ### Basic Usage
//! ```rust
//! # use torsh_tensor::creation::{randn, tensor_1d};
//! # use torsh_core::error::Result;
//! # fn main() -> Result<()> {
//! use torsh_optim::{Adam, AdamW, Optimizer};
//! use torsh_tensor::Tensor;
//! use parking_lot::RwLock;
//! use std::sync::Arc;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let param2 = Arc::new(RwLock::new(randn::<f32>(&[20, 1])?));
//! let params = vec![param1.clone(), param2.clone()];
//!
//! // Create Adam optimizer with default settings
//! let mut optimizer = Adam::new(params, None, None, None, None, false);
//!
//! // Training loop
//! for _epoch in 0..100 {
//!     // ... compute gradients ...
//!
//!     // Optimizer step
//!     optimizer.step()?;
//!     optimizer.zero_grad();
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Configuration
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::adam::AdamBuilder;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//!
//! // Using the builder pattern for better readability
//! let optimizer = AdamBuilder::new()
//!     .lr(1e-4)                    // Lower learning rate
//!     .betas(0.9, 0.98)           // More aggressive second moment decay
//!     .weight_decay(0.01)         // Add weight decay
//!     .eps(1e-6)                  // Tighter numerical stability
//!     .amsgrad(true)              // Use AMSGrad variant
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ### Domain-Specific Configurations
//! ```rust
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::adam::AdamBuilder;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1.clone()];
//!
//! // Computer Vision (ImageNet training)
//! let cv_optimizer = AdamBuilder::new()
//!     .lr(1e-3)
//!     .weight_decay(1e-4)
//!     .build(params.clone());
//!
//! // NLP Transformers (BERT/GPT fine-tuning)
//! let nlp_optimizer = AdamBuilder::new()
//!     .lr(5e-5)                   // Lower learning rate for fine-tuning
//!     .weight_decay(0.01)         // Strong regularization
//!     .eps(1e-6)                  // Tighter epsilon for stability
//!     .build_adamw(params.clone());       // Use AdamW for better weight decay
//!
//! // Research / Experimentation
//! let research_optimizer = AdamBuilder::new()
//!     .lr(3e-4)
//!     .betas(0.9, 0.999)
//!     .amsgrad(true)              // More stable for research
//!     .build(params);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Tips
//!
//! ### Learning Rate Guidelines:
//! - **Default**: 1e-3 works well for most problems
//! - **Fine-tuning**: Use lower rates (5e-5 to 1e-4)
//! - **Large batch sizes**: Scale learning rate proportionally
//! - **Unstable training**: Reduce to 1e-4 or lower
//!
//! ### Beta Parameters:
//! - **β₁ = 0.9**: Good for most cases, use 0.8-0.95 range
//! - **β₂ = 0.999**: Conservative default, use 0.98-0.999 range
//! - **Higher β₂**: More stable but slower adaptation
//! - **Lower β₂**: Faster adaptation but potentially unstable
//!
//! ### Weight Decay:
//! - **Adam**: Use L2 regularization instead or switch to AdamW
//! - **AdamW**: 0.01-0.1 typical range, higher for small datasets
//! - **No weight decay**: Set to 0.0 for some applications (e.g., some NLP tasks)
//!
//! ## Troubleshooting
//!
//! ### Common Issues:
//! 1. **Loss not decreasing**: Try lower learning rate (1e-4)
//! 2. **Training unstable**: Enable AMSGrad or switch to RAdam
//! 3. **Poor generalization**: Use AdamW with weight decay
//! 4. **Slow convergence**: Check if gradients are too small, adjust β₂
//!
//! ### Debugging Tips:
//! - Monitor gradient norms and parameter update norms
//! - Plot learning rate schedule if using schedulers
//! - Compare with SGD baseline for sanity check
//! - Use gradient clipping for very deep networks
//!
//! ## References
//! - [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
//! - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
//! - [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)

use crate::{
    optimizer::BaseOptimizer, Optimizer, OptimizerError, OptimizerResult, OptimizerState,
    ParamGroup,
};
// Temporarily disable scirs2 integration
// use scirs2::optim::adam::{Adam as SciAdam, AdamW as SciAdamW};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::{creation::zeros_like, Tensor};

/// Adam optimizer implementation
///
/// Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm
/// that computes individual learning rates for different parameters from estimates of
/// first and second moments of the gradients.
///
/// # Algorithm Overview
///
/// Adam maintains exponential moving averages of the gradient (`m_t`) and the squared
/// gradient (`v_t`), and uses these to adapt the learning rate for each parameter.
/// The key insight is that parameters that receive consistent gradients should have
/// their learning rates reduced, while parameters with sparse or inconsistent gradients
/// should maintain higher learning rates.
///
/// # Parameters
///
/// * `lr` - Learning rate (default: 1e-3). Controls the step size of parameter updates.
/// * `betas` - Coefficients for computing running averages (default: (0.9, 0.999))
///   - `beta1`: Coefficient for the running average of gradients
///   - `beta2`: Coefficient for the running average of squared gradients
/// * `eps` - Small constant for numerical stability (default: 1e-8)
/// * `weight_decay` - Weight decay (L2 penalty) coefficient (default: 0.0)
/// * `amsgrad` - Whether to use AMSGrad variant (default: false)
///
/// # AMSGrad Variant
///
/// When `amsgrad=true`, the optimizer uses the AMSGrad variant which maintains the
/// maximum of all `v_t` values and uses this for normalization. This can provide
/// better convergence properties in some cases but uses more memory.
///
/// # When to Use Adam
///
/// Adam is an excellent general-purpose optimizer that works well across a wide range
/// of problems. Consider Adam when:
///
/// - Starting a new project (good default choice)
/// - Working with noisy or sparse gradients
/// - Training deep neural networks
/// - You want an optimizer that works well with default hyperparameters
///
/// # Performance Characteristics
///
/// - **Memory Usage**: Higher than SGD (stores first and second moment estimates)
/// - **Convergence Speed**: Generally fast, especially early in training
/// - **Hyperparameter Sensitivity**: Low - works well with default settings
/// - **Generalization**: Good, but AdamW may be better for some applications
///
/// # Example Usage
///
/// ```rust
/// # use torsh_tensor::creation::{randn, tensor_1d};
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_optim::{Adam, Optimizer};
/// use torsh_tensor::Tensor;
/// use parking_lot::RwLock;
/// use std::sync::Arc;
///
/// // Create parameters
/// let param = Arc::new(RwLock::new(randn(&[100, 50])?));
/// let params = vec![param.clone()];
///
/// // Create Adam optimizer
/// let mut optimizer = Adam::new(
///     params,
///     Some(1e-3),      // learning rate
///     Some((0.9, 0.999)), // betas
///     Some(1e-8),      // eps
///     Some(0.01),      // weight decay
///     false            // amsgrad
/// );
///
/// // Training step
/// // ... compute gradients ...
/// optimizer.step()?;
/// optimizer.zero_grad();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Adam {
    base: BaseOptimizer,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

impl Adam {
    /// Creates a new Adam optimizer instance
    ///
    /// # Arguments
    ///
    /// * `params` - Vector of parameters to optimize. Each parameter should be wrapped
    ///              in `Arc<RwLock<Tensor>>` for thread-safe access.
    /// * `lr` - Learning rate (default: 1e-3). Recommended range: [1e-5, 1e-1]
    ///   - Start with 1e-3 for most problems
    ///   - Use 1e-4 or lower for fine-tuning pre-trained models
    ///   - Increase for large batch sizes or simple problems
    /// * `betas` - Exponential decay rates (default: (0.9, 0.999))
    ///   - `beta1` (first moment): Typically 0.8-0.95, controls momentum
    ///   - `beta2` (second moment): Typically 0.98-0.999, controls adaptation speed
    ///   - Lower `beta2` = faster adaptation but potentially less stable
    /// * `eps` - Small constant for numerical stability (default: 1e-8)
    ///   - Prevents division by zero in the denominator
    ///   - Use 1e-4 for FP16 training, 1e-8 for FP32
    /// * `weight_decay` - L2 regularization coefficient (default: 0.0)
    ///   - Note: Consider using AdamW for proper weight decay
    ///   - Typical range: [0.0, 0.1]
    /// * `amsgrad` - Whether to use AMSGrad variant (default: false)
    ///   - Use `true` for better long-term convergence guarantees
    ///   - Increases memory usage but may improve final performance
    ///
    /// # Returns
    ///
    /// A new `Adam` optimizer instance ready for training.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use torsh_tensor::creation::{randn, zeros};
    /// # use torsh_core::error::Result;
    /// # fn main() -> Result<()> {
    /// use torsh_optim::Adam;
    /// use torsh_tensor::Tensor;
    /// use parking_lot::RwLock;
    /// use std::sync::Arc;
    ///
    /// // Create parameters for a simple linear layer
    /// let weight = Arc::new(RwLock::new(randn::<f32>(&[10, 5])?));
    /// let bias = Arc::new(RwLock::new(zeros::<f32>(&[5])?));
    /// let params = vec![weight, bias];
    ///
    /// // Conservative settings for stable training
    /// let optimizer = Adam::new(
    ///     params,
    ///     Some(1e-4),        // Lower learning rate
    ///     Some((0.9, 0.999)), // Standard betas
    ///     Some(1e-8),        // Standard epsilon
    ///     Some(0.01),        // Light weight decay
    ///     false              // Standard Adam (not AMSGrad)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// This function does not panic but may return an optimizer that fails during
    /// training if invalid hyperparameters are provided (e.g., negative learning rate).
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-3);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("beta1".to_string(), betas.0);
        defaults.insert("beta2".to_string(), betas.1);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("weight_decay".to_string(), weight_decay);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "Adam".to_string(),
            defaults,
        };

        Self {
            base,
            betas,
            eps,
            weight_decay,
            amsgrad,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> OptimizerResult<()> {
        // Increment step count
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param.grad().unwrap();
                let param_id = format!("{:p}", param_arc.as_ref());

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    state.insert("step".to_string(), zeros_like(&param)?);
                    state.insert("exp_avg".to_string(), zeros_like(&param)?);
                    state.insert("exp_avg_sq".to_string(), zeros_like(&param)?);
                    if self.amsgrad {
                        state.insert("max_exp_avg_sq".to_string(), zeros_like(&param)?);
                    }
                }

                let mut step_tensor = state.get("step").unwrap().clone();
                let mut exp_avg = state.get("exp_avg").unwrap().clone();
                let mut exp_avg_sq = state.get("exp_avg_sq").unwrap().clone();

                // Increment step count
                step_tensor
                    .add_scalar_(1.0)
                    .map_err(OptimizerError::TensorError)?;
                let step = step_tensor.to_vec().map_err(OptimizerError::TensorError)?[0] as i32;

                // Apply weight decay
                let mut grad = grad;
                if self.weight_decay != 0.0 {
                    let weight_decay_term = param
                        .mul_scalar(self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    grad = grad
                        .add(&weight_decay_term)
                        .map_err(OptimizerError::TensorError)?;
                }

                // Update biased first moment estimate
                exp_avg
                    .mul_scalar_(self.betas.0)
                    .map_err(OptimizerError::TensorError)?;
                let grad_term = grad
                    .mul_scalar(1.0 - self.betas.0)
                    .map_err(OptimizerError::TensorError)?;
                exp_avg
                    .add(&grad_term)
                    .map_err(OptimizerError::TensorError)?;

                // Update biased second raw moment estimate
                exp_avg_sq
                    .mul_scalar_(self.betas.1)
                    .map_err(OptimizerError::TensorError)?;
                let grad_squared = grad.mul_op(&grad).map_err(OptimizerError::TensorError)?;
                let grad_sq_term = grad_squared
                    .mul_scalar(1.0 - self.betas.1)
                    .map_err(OptimizerError::TensorError)?;
                exp_avg_sq
                    .add(&grad_sq_term)
                    .map_err(OptimizerError::TensorError)?;

                let denom = if self.amsgrad {
                    // Update max of exp_avg_sq
                    let mut max_exp_avg_sq = state.get("max_exp_avg_sq").unwrap().clone();
                    max_exp_avg_sq = max_exp_avg_sq
                        .maximum(&exp_avg_sq)
                        .map_err(OptimizerError::TensorError)?;
                    state.insert("max_exp_avg_sq".to_string(), max_exp_avg_sq.clone());

                    // Use max for denominator
                    let sqrt_max = max_exp_avg_sq.sqrt().map_err(OptimizerError::TensorError)?;
                    sqrt_max
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                } else {
                    // Bias correction
                    let bias_correction1 = 1.0 - self.betas.0.powi(step);
                    let bias_correction2 = 1.0 - self.betas.1.powi(step);

                    let _corrected_exp_avg = exp_avg
                        .div_scalar(bias_correction1)
                        .map_err(OptimizerError::TensorError)?;
                    let corrected_exp_avg_sq = exp_avg_sq
                        .div_scalar(bias_correction2)
                        .map_err(OptimizerError::TensorError)?;

                    let sqrt_corrected = corrected_exp_avg_sq
                        .sqrt()
                        .map_err(OptimizerError::TensorError)?;
                    sqrt_corrected
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                };

                // Compute step
                let step_size = group.lr;
                let bias_correction1 = 1.0 - self.betas.0.powi(step);
                let corrected_exp_avg = exp_avg
                    .div_scalar(bias_correction1)
                    .map_err(OptimizerError::TensorError)?;

                let update = corrected_exp_avg
                    .div(&denom)
                    .map_err(OptimizerError::TensorError)?
                    .mul_scalar(step_size)
                    .map_err(OptimizerError::TensorError)?;
                param.sub(&update).map_err(OptimizerError::TensorError)?;

                // Update state
                state.insert("step".to_string(), step_tensor);
                state.insert("exp_avg".to_string(), exp_avg);
                state.insert("exp_avg_sq".to_string(), exp_avg_sq);
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

/// AdamW optimizer - Adam with decoupled weight decay
///
/// AdamW fixes a well-known issue with Adam's weight decay implementation. While Adam
/// applies weight decay to the gradients (which is equivalent to L2 regularization),
/// AdamW applies weight decay directly to the parameters. This approach provides
/// better generalization performance and is theoretically more sound.
///
/// # Key Differences from Adam
///
/// 1. **Decoupled Weight Decay**: Applied directly to parameters, not gradients
/// 2. **Better Regularization**: More effective at preventing overfitting
/// 3. **Improved Generalization**: Particularly important for transformer models
/// 4. **Theoretical Soundness**: Proper implementation of weight decay regularization
///
/// # Mathematical Formulation
///
/// The AdamW update rule differs from Adam in the weight decay step:
/// ```text
/// // Adam: weight decay applied to gradients
/// g_t = g_t + λ * θ_{t-1}
/// θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
///
/// // AdamW: weight decay applied to parameters
/// θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
/// ```
///
/// # When to Use AdamW
///
/// AdamW is preferred over Adam in most modern applications:
///
/// - **Transformer models** (BERT, GPT, T5, etc.) - industry standard
/// - **Computer vision** with proper weight decay
/// - **Transfer learning** and fine-tuning scenarios
/// - **When regularization is important** for generalization
/// - **Modern research** - generally preferred over Adam
///
/// # Performance Characteristics
///
/// - **Memory Usage**: Same as Adam (first and second moment estimates)
/// - **Convergence Speed**: Similar to Adam, sometimes faster
/// - **Generalization**: Significantly better than Adam with weight decay
/// - **Hyperparameter Sensitivity**: Low, robust to hyperparameter choices
///
/// # Hyperparameter Guidelines
///
/// ## Learning Rate
/// - **Transformers**: 5e-5 to 1e-4 for fine-tuning, 1e-4 to 3e-4 for pre-training
/// - **Computer Vision**: 1e-3 to 3e-4 depending on model size and dataset
/// - **Fine-tuning**: Start with 5e-5 and adjust based on validation performance
///
/// ## Weight Decay
/// - **Transformers**: 0.01 to 0.1 (higher than traditional CV models)
/// - **Computer Vision**: 1e-4 to 1e-2 depending on dataset size
/// - **Small datasets**: Higher weight decay (0.01-0.1)
/// - **Large datasets**: Lower weight decay (1e-4-1e-2)
///
/// # Example Usage
///
/// ```rust
/// # use torsh_tensor::creation::randn;
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_optim::{AdamW, Optimizer};
/// use torsh_tensor::Tensor;
/// use parking_lot::RwLock;
/// use std::sync::Arc;
///
/// // Transformer fine-tuning setup
/// let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
/// let params = vec![param1];
/// let mut optimizer = AdamW::new(
///     params,
///     Some(5e-5),         // Conservative LR for fine-tuning
///     Some((0.9, 0.999)), // Standard betas
///     Some(1e-8),         // Standard epsilon
///     Some(0.01),         // Moderate weight decay
///     false               // Standard AdamW
/// );
///
/// // Training loop (simplified example)
/// for _batch in 0..10 {
///     // Forward pass and loss computation
///     // ...
///
///     // Backward pass (in real code)
///     // loss.backward()?;
///
///     // Optional: gradient clipping for transformers
///     // clip_grad_norm_(&params, 1.0);
///
///     // Optimizer step
///     optimizer.step()?;
///     optimizer.zero_grad();
/// }
/// # Ok(())
/// # }
/// ```
pub struct AdamW {
    base: BaseOptimizer,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

impl AdamW {
    /// Creates a new AdamW optimizer instance with decoupled weight decay
    ///
    /// AdamW is the recommended optimizer for most modern deep learning applications,
    /// particularly transformer models and scenarios requiring strong regularization.
    ///
    /// # Arguments
    ///
    /// * `params` - Vector of parameters to optimize, wrapped in `Arc<RwLock<Tensor>>`
    /// * `lr` - Learning rate (default: 1e-3)
    ///   - **Transformers**: Use 5e-5 for fine-tuning, 1e-4 to 3e-4 for pre-training
    ///   - **Computer Vision**: Use 1e-3 as starting point, adjust based on model size
    ///   - **General rule**: Start conservative and increase if training is too slow
    /// * `betas` - Momentum coefficients (default: (0.9, 0.999))
    ///   - First value controls momentum, second controls adaptive learning rate
    ///   - (0.9, 0.999) works well for most applications
    ///   - For transformers, some use (0.9, 0.98) for slightly faster adaptation
    /// * `eps` - Numerical stability constant (default: 1e-8)
    ///   - Use 1e-6 or 1e-4 for mixed precision training
    ///   - Standard 1e-8 is fine for full precision training
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    ///   - **Critical parameter** for AdamW - significantly affects generalization
    ///   - Transformers: 0.01-0.1 (higher than traditional models)
    ///   - CV models: 1e-4 to 1e-2 depending on dataset size
    ///   - Set to 0.0 to disable (but you probably want some weight decay)
    /// * `amsgrad` - Use AMSGrad variant (default: false)
    ///   - Enable for problems requiring long-term convergence guarantees
    ///   - Increases memory usage but may improve final performance
    ///
    /// # Returns
    ///
    /// A configured AdamW optimizer ready for training.
    ///
    /// # Recommended Configurations
    ///
    /// ```rust
    /// # use torsh_tensor::creation::randn;
    /// # use torsh_core::error::Result;
    /// # use parking_lot::RwLock;
    /// # use std::sync::Arc;
    /// # fn main() -> Result<()> {
    /// use torsh_optim::AdamW;
    ///
    /// // Create some parameters
    /// let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
    /// let params = vec![param1.clone()];
    ///
    /// // BERT/RoBERTa fine-tuning (conservative)
    /// let bert_optimizer = AdamW::new(
    ///     params.clone(),
    ///     Some(5e-5),         // Low LR for fine-tuning
    ///     Some((0.9, 0.999)), // Standard betas
    ///     Some(1e-8),         // Standard eps
    ///     Some(0.01),         // Moderate weight decay
    ///     false
    /// );
    ///
    /// // GPT pre-training (more aggressive)
    /// let gpt_optimizer = AdamW::new(
    ///     params.clone(),
    ///     Some(3e-4),         // Higher LR for pre-training
    ///     Some((0.9, 0.98)),  // Faster second moment adaptation
    ///     Some(1e-6),         // Tighter epsilon
    ///     Some(0.1),          // Strong regularization
    ///     false
    /// );
    ///
    /// // Computer Vision (ResNet/EfficientNet)
    /// let cv_optimizer = AdamW::new(
    ///     params,
    ///     Some(1e-3),         // Standard LR
    ///     Some((0.9, 0.999)), // Standard betas
    ///     Some(1e-8),         // Standard eps
    ///     Some(1e-4),         // Light weight decay
    ///     false
    /// );
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - AdamW typically converges faster than Adam when weight decay > 0
    /// - Better final performance due to improved regularization
    /// - Memory usage identical to Adam
    /// - Computational overhead minimal compared to Adam
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: bool,
    ) -> Self {
        let lr = lr.unwrap_or(1e-3);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.01);

        let mut defaults = HashMap::new();
        defaults.insert("lr".to_string(), lr);
        defaults.insert("beta1".to_string(), betas.0);
        defaults.insert("beta2".to_string(), betas.1);
        defaults.insert("eps".to_string(), eps);
        defaults.insert("weight_decay".to_string(), weight_decay);

        let param_group = ParamGroup::new(params, lr);

        let base = BaseOptimizer {
            param_groups: vec![param_group],
            state: HashMap::new(),
            optimizer_type: "AdamW".to_string(),
            defaults,
        };

        Self {
            base,
            betas,
            eps,
            weight_decay,
            amsgrad,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> OptimizerResult<()> {
        // AdamW with decoupled weight decay
        for group in &mut self.base.param_groups {
            for param_arc in &group.params {
                let mut param = param_arc.write();

                // Check if parameter has gradients
                if !param.has_grad() {
                    continue;
                }

                let grad = param.grad().unwrap();
                let param_id = format!("{:p}", param_arc.as_ref());

                // Get or initialize optimizer state
                let needs_init = !self.base.state.contains_key(&param_id);
                let state = self
                    .base
                    .state
                    .entry(param_id.clone())
                    .or_insert_with(HashMap::new);

                if needs_init {
                    state.insert("step".to_string(), zeros_like(&param)?);
                    state.insert("exp_avg".to_string(), zeros_like(&param)?);
                    state.insert("exp_avg_sq".to_string(), zeros_like(&param)?);
                    if self.amsgrad {
                        state.insert("max_exp_avg_sq".to_string(), zeros_like(&param)?);
                    }
                }

                let mut step_tensor = state.get("step").unwrap().clone();
                let mut exp_avg = state.get("exp_avg").unwrap().clone();
                let mut exp_avg_sq = state.get("exp_avg_sq").unwrap().clone();

                // Increment step count
                step_tensor
                    .add_scalar_(1.0)
                    .map_err(OptimizerError::TensorError)?;
                let step = step_tensor.to_vec().map_err(OptimizerError::TensorError)?[0] as i32;

                // Apply weight decay directly to the parameter (decoupled)
                if self.weight_decay != 0.0 {
                    let weight_decay_update = param
                        .mul_scalar(group.lr * self.weight_decay)
                        .map_err(OptimizerError::TensorError)?;
                    param
                        .sub(&weight_decay_update)
                        .map_err(OptimizerError::TensorError)?;
                }

                // Update biased first moment estimate
                exp_avg
                    .mul_scalar_(self.betas.0)
                    .map_err(OptimizerError::TensorError)?;
                let grad_term = grad
                    .mul_scalar(1.0 - self.betas.0)
                    .map_err(OptimizerError::TensorError)?;
                exp_avg
                    .add(&grad_term)
                    .map_err(OptimizerError::TensorError)?;

                // Update biased second raw moment estimate
                exp_avg_sq
                    .mul_scalar_(self.betas.1)
                    .map_err(OptimizerError::TensorError)?;
                let grad_squared = grad.mul_op(&grad).map_err(OptimizerError::TensorError)?;
                let grad_sq_term = grad_squared
                    .mul_scalar(1.0 - self.betas.1)
                    .map_err(OptimizerError::TensorError)?;
                exp_avg_sq
                    .add(&grad_sq_term)
                    .map_err(OptimizerError::TensorError)?;

                // Bias correction
                let bias_correction1 = 1.0 - self.betas.0.powi(step);
                let bias_correction2 = 1.0 - self.betas.1.powi(step);

                let corrected_exp_avg = exp_avg
                    .div_scalar(bias_correction1)
                    .map_err(OptimizerError::TensorError)?;
                let corrected_exp_avg_sq = exp_avg_sq
                    .div_scalar(bias_correction2)
                    .map_err(OptimizerError::TensorError)?;

                let denom = if self.amsgrad {
                    // Update max of exp_avg_sq
                    let mut max_exp_avg_sq = state.get("max_exp_avg_sq").unwrap().clone();
                    max_exp_avg_sq = max_exp_avg_sq
                        .maximum(&corrected_exp_avg_sq)
                        .map_err(OptimizerError::TensorError)?;
                    state.insert("max_exp_avg_sq".to_string(), max_exp_avg_sq.clone());

                    // Use max for denominator
                    let sqrt_max = max_exp_avg_sq.sqrt().map_err(OptimizerError::TensorError)?;
                    sqrt_max
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                } else {
                    let sqrt_corrected = corrected_exp_avg_sq
                        .sqrt()
                        .map_err(OptimizerError::TensorError)?;
                    sqrt_corrected
                        .add_scalar(self.eps)
                        .map_err(OptimizerError::TensorError)?
                };

                // Compute step
                let step_size = group.lr;
                let update = corrected_exp_avg
                    .div(&denom)
                    .map_err(OptimizerError::TensorError)?
                    .mul_scalar(step_size)
                    .map_err(OptimizerError::TensorError)?;
                param.sub(&update).map_err(OptimizerError::TensorError)?;

                // Update state
                state.insert("step".to_string(), step_tensor);
                state.insert("exp_avg".to_string(), exp_avg);
                state.insert("exp_avg_sq".to_string(), exp_avg_sq);
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

/// Builder for Adam optimizer
pub struct AdamBuilder {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

impl AdamBuilder {
    pub fn new() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }

    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = (beta1, beta2);
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

    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    pub fn build(self, params: Vec<Arc<RwLock<Tensor>>>) -> Adam {
        Adam::new(
            params,
            Some(self.lr),
            Some(self.betas),
            Some(self.eps),
            Some(self.weight_decay),
            self.amsgrad,
        )
    }

    pub fn build_adamw(self, params: Vec<Arc<RwLock<Tensor>>>) -> AdamW {
        AdamW::new(
            params,
            Some(self.lr),
            Some(self.betas),
            Some(self.eps),
            Some(self.weight_decay),
            self.amsgrad,
        )
    }
}

impl Default for AdamBuilder {
    fn default() -> Self {
        Self::new()
    }
}
