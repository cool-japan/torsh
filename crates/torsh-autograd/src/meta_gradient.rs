use crate::context::AutogradContext;
use crate::error_handling::AutogradError;
use torsh_core::Result;
use torsh_tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MetaGradientConfig {
    pub learning_rate: f64,
    pub second_order: bool,
    pub allow_unused: bool,
    pub create_graph: bool,
    pub retain_graph: bool,
}

impl Default for MetaGradientConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            second_order: true,
            allow_unused: false,
            create_graph: true,
            retain_graph: false,
        }
    }
}

pub struct MetaGradientEngine {
    config: MetaGradientConfig,
    #[allow(dead_code)]
    context: AutogradContext,
}

impl MetaGradientEngine {
    pub fn new(config: MetaGradientConfig) -> Self {
        Self {
            config,
            context: AutogradContext::new(),
        }
    }

    pub fn compute_meta_gradient(
        &mut self,
        loss: &Tensor,
        parameters: &[Tensor],
        meta_parameters: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let first_order_grads = self.compute_first_order_gradients(loss, parameters)?;

        if !self.config.second_order {
            return Ok(first_order_grads);
        }

        let updated_parameters = self.apply_gradient_update(parameters, &first_order_grads)?;

        let meta_loss = self.compute_meta_loss(&updated_parameters, meta_parameters)?;

        self.compute_second_order_gradients(&meta_loss, meta_parameters, &first_order_grads)
    }

    /// Compute first-order gradients of `loss` with respect to `parameters`.
    ///
    /// This performs a real reverse-mode backward pass through the autograd tape
    /// carried by the `torsh_tensor::Tensor` operation graph. The caller MUST
    /// build `loss` from `parameters` using differentiable tensor operations and
    /// mark the parameters with [`Tensor::requires_grad_`] so that the tape links
    /// the scalar `loss` back to each parameter leaf.
    ///
    /// # Errors
    ///
    /// Returns an honest error (never a fabricated ones-/zeros-gradient) when:
    /// - `loss` is not a scalar (backward is only defined for scalar outputs),
    /// - `loss` does not require gradients (no tape was recorded), or
    /// - a parameter has no gradient after the backward pass, which means it was
    ///   not connected to `loss` in the autograd graph.
    fn compute_first_order_gradients(
        &mut self,
        loss: &Tensor,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        if loss.shape().numel() != 1 {
            return Err(AutogradError::gradient_computation(
                "meta_gradient::first_order",
                format!(
                    "first-order gradients require a scalar loss; got shape {:?}",
                    loss.shape().dims()
                ),
            )
            .into());
        }

        if !loss.requires_grad() {
            return Err(AutogradError::gradient_computation(
                "meta_gradient::first_order",
                "loss tensor does not require grad: no autograd tape was recorded, so \
                 real gradients cannot be computed. Build the loss from \
                 `requires_grad_(true)` parameters using differentiable tensor ops.",
            )
            .into());
        }

        // Clear any stale gradients on the parameters so we read only the
        // gradients produced by this backward pass. `set_grad` uses the
        // tensor's interior-mutable gradient slot, so a shared `&Tensor` is
        // sufficient (no `&mut` required).
        for param in parameters {
            param.set_grad(None);
        }

        // Real reverse-mode backward pass through the tensor autograd tape.
        loss.backward()?;

        let mut gradients = Vec::with_capacity(parameters.len());
        for (index, param) in parameters.iter().enumerate() {
            match param.grad() {
                Some(grad) => gradients.push(grad),
                None => {
                    if self.config.allow_unused {
                        // PyTorch `allow_unused=True` semantics: a parameter that
                        // did not participate in the loss yields a zero gradient.
                        gradients.push(Tensor::zeros(param.shape().dims(), param.device())?);
                    } else {
                        return Err(AutogradError::gradient_computation(
                            "meta_gradient::first_order",
                            format!(
                                "parameter {index} has no gradient after backward: it is not \
                                 connected to the loss in the autograd graph. Set \
                                 `allow_unused = true` to treat unconnected parameters as zero.",
                            ),
                        )
                        .into());
                    }
                }
            }
        }

        Ok(gradients)
    }

    fn apply_gradient_update(
        &self,
        parameters: &[Tensor],
        gradients: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let mut updated_params = Vec::new();

        for (param, grad) in parameters.iter().zip(gradients.iter()) {
            let update = grad.mul_scalar(self.config.learning_rate as f32)?;
            let updated_param = param.sub(&update)?;
            updated_params.push(updated_param);
        }

        Ok(updated_params)
    }

    fn compute_meta_loss(
        &self,
        updated_parameters: &[Tensor],
        meta_parameters: &[Tensor],
    ) -> Result<Tensor> {
        // Create initial meta_loss tensor as scalar
        let mut meta_loss = Tensor::scalar(0.0f32)?;

        for (updated_param, meta_param) in updated_parameters.iter().zip(meta_parameters.iter()) {
            let diff = updated_param.sub(meta_param)?;
            let squared_diff = diff.mul(&diff)?;
            let param_loss = squared_diff.sum()?;
            meta_loss = meta_loss.add(&param_loss)?;
        }

        Ok(meta_loss)
    }

    /// Compute second-order (meta) gradients.
    ///
    /// A genuine second-order meta-gradient is the gradient of the meta-loss with
    /// respect to the meta-parameters *through* the inner gradient-update step,
    /// i.e. it requires differentiating through the first-order gradients
    /// themselves (a Hessian-vector product / double backward).
    ///
    /// The underlying `torsh_tensor::Tensor` autograd tape performs a single
    /// reverse pass and stores gradients as plain data (it does not record the
    /// backward pass into the graph), so `create_graph`-style double
    /// backpropagation is not available. Returning the cloned first-order
    /// gradients here would be a silent fabrication: a first-order gradient is
    /// not a second-order gradient. We therefore return an honest error.
    ///
    /// Callers that only need first-order behaviour should set
    /// [`MetaGradientConfig::second_order`] to `false`, which short-circuits
    /// before this function is reached.
    fn compute_second_order_gradients(
        &mut self,
        _meta_loss: &Tensor,
        _meta_parameters: &[Tensor],
        _first_order_grads: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        Err(AutogradError::gradient_computation(
            "meta_gradient::second_order",
            "second-order meta-gradients require double backpropagation \
             (a Hessian-vector product through the inner update step), which the \
             tensor autograd tape does not support: it performs a single reverse \
             pass and stores gradients as plain data rather than graph-tracked \
             tensors. Set `MetaGradientConfig::second_order = false` to use \
             first-order meta-gradients (FOMAML/Reptile-style).",
        )
        .into())
    }

    pub fn maml_gradient_update(
        &mut self,
        task_losses: &[Tensor],
        parameters: &[Tensor],
        inner_lr: f64,
        _outer_lr: f64,
    ) -> Result<Vec<Tensor>> {
        let mut meta_gradients = Vec::new();

        for task_loss in task_losses {
            let task_grads = self.compute_first_order_gradients(task_loss, parameters)?;
            let updated_params =
                self.apply_gradient_update_with_lr(parameters, &task_grads, inner_lr)?;

            let meta_loss = self.compute_meta_loss(&updated_params, parameters)?;
            let meta_grads = self.compute_first_order_gradients(&meta_loss, parameters)?;

            meta_gradients.push(meta_grads);
        }

        self.average_gradients(&meta_gradients)
    }

    fn apply_gradient_update_with_lr(
        &self,
        parameters: &[Tensor],
        gradients: &[Tensor],
        learning_rate: f64,
    ) -> Result<Vec<Tensor>> {
        let mut updated_params = Vec::new();

        for (param, grad) in parameters.iter().zip(gradients.iter()) {
            let update = grad.mul_scalar(learning_rate as f32)?;
            let updated_param = param.sub(&update)?;
            updated_params.push(updated_param);
        }

        Ok(updated_params)
    }

    fn average_gradients(&self, gradient_sets: &[Vec<Tensor>]) -> Result<Vec<Tensor>> {
        if gradient_sets.is_empty() {
            return Ok(Vec::new());
        }

        let num_params = gradient_sets[0].len();
        let mut averaged_grads = Vec::new();

        for param_idx in 0..num_params {
            let mut sum_grad = gradient_sets[0][param_idx].clone();

            for grad_set in gradient_sets.iter().skip(1) {
                sum_grad = sum_grad.add(&grad_set[param_idx])?;
            }

            let count = gradient_sets.len() as f32;
            let avg_grad = sum_grad.div_scalar(count)?;
            averaged_grads.push(avg_grad);
        }

        Ok(averaged_grads)
    }

    pub fn reptile_gradient_update(
        &mut self,
        task_losses: &[Tensor],
        parameters: &[Tensor],
        inner_lr: f64,
        inner_steps: usize,
        _outer_lr: f64,
    ) -> Result<Vec<Tensor>> {
        let mut meta_gradients = Vec::new();

        for task_loss in task_losses {
            let mut adapted_params = parameters.to_vec();

            for _ in 0..inner_steps {
                let task_grads = self.compute_first_order_gradients(task_loss, &adapted_params)?;
                adapted_params =
                    self.apply_gradient_update_with_lr(&adapted_params, &task_grads, inner_lr)?;
            }

            let reptile_grads = self.compute_reptile_gradients(parameters, &adapted_params)?;
            meta_gradients.push(reptile_grads);
        }

        self.average_gradients(&meta_gradients)
    }

    fn compute_reptile_gradients(
        &self,
        original_params: &[Tensor],
        adapted_params: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        let mut reptile_grads = Vec::new();

        for (orig_param, adapted_param) in original_params.iter().zip(adapted_params.iter()) {
            let grad = orig_param.sub(adapted_param)?;
            reptile_grads.push(grad);
        }

        Ok(reptile_grads)
    }

    pub fn fomaml_gradient_update(
        &mut self,
        task_losses: &[Tensor],
        parameters: &[Tensor],
        inner_lr: f64,
        _outer_lr: f64,
    ) -> Result<Vec<Tensor>> {
        let mut meta_gradients = Vec::new();

        for task_loss in task_losses {
            let task_grads = self.compute_first_order_gradients(task_loss, parameters)?;
            let updated_params =
                self.apply_gradient_update_with_lr(parameters, &task_grads, inner_lr)?;

            let meta_loss = self.compute_meta_loss(&updated_params, parameters)?;
            let meta_grads = self.compute_first_order_gradients(&meta_loss, parameters)?;

            meta_gradients.push(meta_grads);
        }

        self.average_gradients(&meta_gradients)
    }
}

/// Gradient-based hyperparameter optimization engine
/// Supports automatic tuning of hyperparameters using gradient-based methods
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizationConfig {
    /// Learning rate for hyperparameter updates
    pub hyperparam_lr: f64,
    /// Number of optimization steps for each validation
    pub inner_steps: usize,
    /// Maximum number of hyperparameter optimization iterations
    pub max_iterations: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Whether to use second-order gradients for hyperparameter optimization
    pub second_order: bool,
    /// Whether to enable early stopping
    pub early_stopping: bool,
    /// Number of iterations to look back for early stopping
    pub patience: usize,
}

impl Default for HyperparameterOptimizationConfig {
    fn default() -> Self {
        Self {
            hyperparam_lr: 0.001,
            inner_steps: 10,
            max_iterations: 100,
            tolerance: 1e-6,
            second_order: true,
            early_stopping: true,
            patience: 10,
        }
    }
}

/// Represents different types of hyperparameters that can be optimized
#[derive(Debug, Clone)]
pub enum HyperparameterType {
    /// Learning rate optimization
    LearningRate,
    /// L2 regularization strength
    L2Regularization,
    /// L1 regularization strength
    L1Regularization,
    /// Dropout probability
    DropoutRate,
    /// Batch size (discrete optimization)
    BatchSize,
    /// Weight decay
    WeightDecay,
    /// Custom hyperparameter with name
    Custom(String),
}

/// Container for a hyperparameter with its current value and optimization metadata
#[derive(Debug, Clone)]
pub struct Hyperparameter {
    pub param_type: HyperparameterType,
    pub value: Tensor,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub is_log_space: bool,
    pub requires_grad: bool,
}

impl Hyperparameter {
    pub fn new(
        param_type: HyperparameterType,
        initial_value: f64,
        lower_bound: f64,
        upper_bound: f64,
        is_log_space: bool,
    ) -> Result<Self> {
        let value = if is_log_space {
            Tensor::scalar(initial_value.ln() as f32)?
        } else {
            Tensor::scalar(initial_value as f32)?
        };

        Ok(Self {
            param_type,
            value,
            lower_bound,
            upper_bound,
            is_log_space,
            requires_grad: true,
        })
    }

    pub fn get_actual_value(&self) -> Result<f64> {
        let raw_value = self.value.to_vec()?[0] as f64;
        let value = if self.is_log_space {
            raw_value.exp()
        } else {
            raw_value
        };
        Ok(value.clamp(self.lower_bound, self.upper_bound))
    }

    pub fn update_value(&mut self, gradient: &Tensor, learning_rate: f64) -> Result<()> {
        let update = gradient.mul_scalar(learning_rate as f32)?;
        self.value = self.value.sub(&update)?;
        Ok(())
    }
}

/// Result of hyperparameter optimization
#[derive(Debug)]
pub struct HyperparameterOptimizationResult {
    pub optimal_hyperparameters: Vec<Hyperparameter>,
    pub best_validation_loss: f64,
    pub optimization_history: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
}

/// Main engine for gradient-based hyperparameter optimization
pub struct HyperparameterOptimizer {
    config: HyperparameterOptimizationConfig,
    meta_gradient_engine: MetaGradientEngine,
}

impl HyperparameterOptimizer {
    pub fn new(config: HyperparameterOptimizationConfig) -> Self {
        Self {
            meta_gradient_engine: MetaGradientEngine::new(MetaGradientConfig {
                learning_rate: config.hyperparam_lr,
                second_order: config.second_order,
                allow_unused: true,
                create_graph: true,
                retain_graph: true,
            }),
            config,
        }
    }

    /// Optimize hyperparameters using bilevel optimization
    /// train_data: training dataset for inner optimization
    /// val_data: validation dataset for outer optimization
    /// model_parameters: parameters of the model to be optimized
    /// hyperparameters: hyperparameters to optimize
    pub fn optimize_hyperparameters(
        &mut self,
        train_losses: &[Tensor],
        val_losses: &[Tensor],
        model_parameters: &[Tensor],
        mut hyperparameters: Vec<Hyperparameter>,
    ) -> Result<HyperparameterOptimizationResult> {
        let mut optimization_history = Vec::new();
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for iteration in 0..self.config.max_iterations {
            // Step 1: Inner optimization - optimize model parameters with current hyperparameters
            let optimized_model_params =
                self.inner_optimization_step(train_losses, model_parameters, &hyperparameters)?;

            // Step 2: Compute validation loss with optimized parameters
            let val_loss = self.compute_validation_loss(val_losses, &optimized_model_params)?;
            let val_loss_value = val_loss.to_vec()?[0] as f64;

            optimization_history.push(val_loss_value);

            // Step 3: Compute gradients with respect to hyperparameters
            let hyperparam_gradients = self.compute_hyperparameter_gradients(
                &val_loss,
                &hyperparameters,
                &optimized_model_params,
                model_parameters,
            )?;

            // Step 4: Update hyperparameters
            for (hyperparam, gradient) in
                hyperparameters.iter_mut().zip(hyperparam_gradients.iter())
            {
                if hyperparam.requires_grad {
                    hyperparam.update_value(gradient, self.config.hyperparam_lr)?;
                }
            }

            // Step 5: Check for convergence and early stopping
            if val_loss_value < best_val_loss {
                best_val_loss = val_loss_value;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            if self.config.early_stopping && patience_counter >= self.config.patience {
                return Ok(HyperparameterOptimizationResult {
                    optimal_hyperparameters: hyperparameters,
                    best_validation_loss: best_val_loss,
                    optimization_history,
                    converged: true,
                    iterations: iteration + 1,
                });
            }

            // Check for tolerance-based convergence
            if optimization_history.len() > 1 {
                let improvement =
                    optimization_history[optimization_history.len() - 2] - val_loss_value;
                if improvement.abs() < self.config.tolerance {
                    return Ok(HyperparameterOptimizationResult {
                        optimal_hyperparameters: hyperparameters,
                        best_validation_loss: best_val_loss,
                        optimization_history,
                        converged: true,
                        iterations: iteration + 1,
                    });
                }
            }
        }

        Ok(HyperparameterOptimizationResult {
            optimal_hyperparameters: hyperparameters,
            best_validation_loss: best_val_loss,
            optimization_history,
            converged: false,
            iterations: self.config.max_iterations,
        })
    }

    /// Perform inner optimization step to optimize model parameters
    fn inner_optimization_step(
        &mut self,
        train_losses: &[Tensor],
        model_parameters: &[Tensor],
        hyperparameters: &[Hyperparameter],
    ) -> Result<Vec<Tensor>> {
        let mut current_params = model_parameters.to_vec();

        // Extract learning rate from hyperparameters
        let learning_rate = hyperparameters
            .iter()
            .find(|h| matches!(h.param_type, HyperparameterType::LearningRate))
            .map(|h| h.get_actual_value())
            .transpose()?
            .unwrap_or(0.01);

        // Perform multiple gradient descent steps
        for _ in 0..self.config.inner_steps {
            let mut total_gradients = current_params
                .iter()
                .map(|p| Tensor::zeros(p.shape().dims(), p.device()))
                .collect::<Result<Vec<_>>>()?;

            // Accumulate gradients from all training losses
            for train_loss in train_losses {
                let gradients = self
                    .meta_gradient_engine
                    .compute_first_order_gradients(train_loss, &current_params)?;
                for (total_grad, grad) in total_gradients.iter_mut().zip(gradients.iter()) {
                    *total_grad = total_grad.add(grad)?;
                }
            }

            // Apply regularization if specified
            self.apply_regularization(&mut total_gradients, &current_params, hyperparameters)?;

            // Update parameters
            current_params = self.meta_gradient_engine.apply_gradient_update_with_lr(
                &current_params,
                &total_gradients,
                learning_rate,
            )?;
        }

        Ok(current_params)
    }

    /// Apply L1 and L2 regularization based on hyperparameters
    fn apply_regularization(
        &self,
        gradients: &mut [Tensor],
        parameters: &[Tensor],
        hyperparameters: &[Hyperparameter],
    ) -> Result<()> {
        // L2 regularization
        if let Some(l2_hyperparam) = hyperparameters
            .iter()
            .find(|h| matches!(h.param_type, HyperparameterType::L2Regularization))
        {
            let l2_strength = l2_hyperparam.get_actual_value()?;
            if l2_strength > 0.0 {
                for (grad, param) in gradients.iter_mut().zip(parameters.iter()) {
                    let l2_term = param.mul_scalar(l2_strength as f32)?;
                    *grad = grad.add(&l2_term)?;
                }
            }
        }

        // L1 regularization (simplified - in practice would need subgradient)
        if let Some(l1_hyperparam) = hyperparameters
            .iter()
            .find(|h| matches!(h.param_type, HyperparameterType::L1Regularization))
        {
            let l1_strength = l1_hyperparam.get_actual_value()?;
            if l1_strength > 0.0 {
                for (grad, param) in gradients.iter_mut().zip(parameters.iter()) {
                    // Simplified L1 gradient (sign function approximation)
                    let l1_term = param.mul_scalar(l1_strength as f32)?;
                    *grad = grad.add(&l1_term)?;
                }
            }
        }

        Ok(())
    }

    /// Compute validation loss with given parameters
    fn compute_validation_loss(
        &self,
        val_losses: &[Tensor],
        _parameters: &[Tensor],
    ) -> Result<Tensor> {
        // For simplicity, average all validation losses
        if val_losses.is_empty() {
            return Tensor::scalar(0.0);
        }

        let mut total_loss = val_losses[0].clone();
        for loss in val_losses.iter().skip(1) {
            total_loss = total_loss.add(loss)?;
        }

        let num_losses = Tensor::scalar(val_losses.len() as f32)?;
        total_loss.div(&num_losses)
    }

    /// Compute gradients with respect to hyperparameters using the implicit
    /// function theorem (IFT).
    ///
    /// A correct hyperparameter gradient via the IFT is
    /// `dL_val/dλ = -∂L_val/∂θ* · (∂²L_train/∂θ²)⁻¹ · ∂²L_train/∂θ∂λ`,
    /// which needs a Hessian-inverse-vector product over the training loss plus
    /// mixed second derivatives with respect to the hyperparameters. None of
    /// those quantities are available from the single-pass tensor autograd tape
    /// (which does not support double backpropagation), so a real IFT gradient
    /// cannot be formed here.
    ///
    /// # Errors
    ///
    /// Returns an honest error rather than the previous fabricated gradient,
    /// whose sign was chosen from a hard-coded loss threshold and bore no
    /// mathematical relationship to the true hyperparameter gradient.
    fn compute_hyperparameter_gradients(
        &mut self,
        _val_loss: &Tensor,
        _hyperparameters: &[Hyperparameter],
        _optimized_params: &[Tensor],
        _original_params: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        Err(AutogradError::gradient_computation(
            "meta_gradient::hyperparameter_ift",
            "implicit-function-theorem hyperparameter gradients require a \
             Hessian-inverse-vector product over the training loss and mixed \
             second derivatives w.r.t. the hyperparameters. The tensor autograd \
             tape only supports single-pass first-order differentiation, so these \
             second-order quantities are unavailable and a real IFT gradient \
             cannot be computed.",
        )
        .into())
    }

    /// Create common hyperparameters for optimization
    pub fn create_learning_rate_hyperparam(initial_lr: f64) -> Result<Hyperparameter> {
        Hyperparameter::new(
            HyperparameterType::LearningRate,
            initial_lr,
            1e-6,
            1.0,
            true, // log space
        )
    }

    pub fn create_l2_regularization_hyperparam(initial_strength: f64) -> Result<Hyperparameter> {
        Hyperparameter::new(
            HyperparameterType::L2Regularization,
            initial_strength,
            0.0,
            1.0,
            true, // log space
        )
    }

    pub fn create_l1_regularization_hyperparam(initial_strength: f64) -> Result<Hyperparameter> {
        Hyperparameter::new(
            HyperparameterType::L1Regularization,
            initial_strength,
            0.0,
            1.0,
            true, // log space
        )
    }

    pub fn create_dropout_rate_hyperparam(initial_rate: f64) -> Result<Hyperparameter> {
        Hyperparameter::new(
            HyperparameterType::DropoutRate,
            initial_rate,
            0.0,
            0.9,
            false, // linear space
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;

    /// Extract the single scalar value from a tensor for assertions.
    fn scalar_value(t: &Tensor) -> Result<f32> {
        Ok(t.to_vec()?[0])
    }

    #[test]
    fn test_first_order_gradient_is_real_not_ones() -> Result<()> {
        // Build a genuinely connected graph: loss = param^2 for a scalar param.
        // The real derivative is dloss/dparam = 2 * param = 2 * 3 = 6.
        let mut engine = MetaGradientEngine::new(MetaGradientConfig::default());
        let param = Tensor::scalar(3.0f32)?.requires_grad_(true);
        let loss = param.pow(2.0f32)?;

        let grads = engine.compute_first_order_gradients(&loss, std::slice::from_ref(&param))?;

        assert_eq!(grads.len(), 1);
        let value = scalar_value(&grads[0])?;
        // A real gradient is 6.0, decisively NOT the old fabricated 1.0 (ones_like).
        assert!(
            (value - 6.0).abs() < 1e-4,
            "expected real gradient 6.0, got {value}"
        );
        assert!(
            (value - 1.0).abs() > 1e-3,
            "gradient must not be a fabricated ones-tensor"
        );
        Ok(())
    }

    #[test]
    fn test_first_order_gradient_errors_on_disconnected_loss() {
        // A loss that does not require grad has no tape linking it to params,
        // so an honest error must be returned (never a ones-tensor).
        let mut engine = MetaGradientEngine::new(MetaGradientConfig::default());
        let loss = Tensor::ones(&[1], DeviceType::Cpu).expect("tensor creation");
        let param = Tensor::ones(&[2, 2], DeviceType::Cpu).expect("tensor creation");

        let result = engine.compute_first_order_gradients(&loss, std::slice::from_ref(&param));
        assert!(
            result.is_err(),
            "disconnected loss must yield an honest error, not fabricated gradients"
        );
    }

    #[test]
    fn test_first_order_gradient_errors_on_non_scalar_loss() {
        // backward() is only defined for scalar outputs; a non-scalar loss must
        // produce an honest error rather than a fabricated per-element gradient.
        let mut engine = MetaGradientEngine::new(MetaGradientConfig::default());
        let loss = Tensor::ones(&[2, 2], DeviceType::Cpu)
            .expect("tensor creation")
            .requires_grad_(true);
        let param = Tensor::ones(&[2, 2], DeviceType::Cpu).expect("tensor creation");

        let result = engine.compute_first_order_gradients(&loss, std::slice::from_ref(&param));
        assert!(
            result.is_err(),
            "non-scalar loss must yield an honest error"
        );
    }

    #[test]
    fn test_allow_unused_yields_zero_not_ones() -> Result<()> {
        // With allow_unused=true a parameter that does not participate in the
        // loss yields a true zero gradient (PyTorch semantics), never a ones.
        let config = MetaGradientConfig {
            allow_unused: true,
            ..MetaGradientConfig::default()
        };
        let mut engine = MetaGradientEngine::new(config);

        let used = Tensor::scalar(2.0f32)?.requires_grad_(true);
        let loss = used.pow(2.0f32)?;
        let unused = Tensor::scalar(5.0f32)?.requires_grad_(true);

        let grads = engine.compute_first_order_gradients(&loss, &[used.clone(), unused])?;
        assert_eq!(grads.len(), 2);
        // Used param: real gradient 2*2 = 4.
        assert!((scalar_value(&grads[0])? - 4.0).abs() < 1e-4);
        // Unused param: honest zero, not a fabricated one.
        assert!((scalar_value(&grads[1])? - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_second_order_gradient_returns_honest_error() {
        // Second-order meta-gradients require double backprop which the tape
        // does not support; the engine must error rather than clone first-order
        // gradients and pass them off as second-order.
        let mut engine = MetaGradientEngine::new(MetaGradientConfig::default());
        let loss = Tensor::scalar(1.0f32).expect("tensor creation");
        let meta_param = Tensor::scalar(1.0f32).expect("tensor creation");
        let first_grad = Tensor::scalar(1.0f32).expect("tensor creation");

        let result = engine.compute_second_order_gradients(
            &loss,
            std::slice::from_ref(&meta_param),
            std::slice::from_ref(&first_grad),
        );
        assert!(
            result.is_err(),
            "second-order gradients are unsupported and must return an honest error"
        );
    }

    #[test]
    fn test_meta_gradient_first_order_path_is_real() -> Result<()> {
        // second_order=false routes through the real first-order path. Build a
        // connected scalar loss and verify the returned gradient is genuine.
        let config = MetaGradientConfig {
            second_order: false,
            ..MetaGradientConfig::default()
        };
        let mut engine = MetaGradientEngine::new(config);

        let param = Tensor::scalar(4.0f32)?.requires_grad_(true);
        let loss = param.pow(2.0f32)?;
        let meta_param = Tensor::scalar(0.0f32)?;

        let result = engine.compute_meta_gradient(
            &loss,
            std::slice::from_ref(&param),
            std::slice::from_ref(&meta_param),
        )?;
        assert_eq!(result.len(), 1);
        // Real derivative 2*4 = 8.
        assert!((scalar_value(&result[0])? - 8.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_meta_gradient_second_order_path_errors() {
        // Default config requests second-order gradients; with a connected loss
        // the first-order step succeeds but the second-order step must error.
        let mut engine = MetaGradientEngine::new(MetaGradientConfig::default());
        let param = Tensor::scalar(4.0f32)
            .expect("tensor creation")
            .requires_grad_(true);
        let loss = param.pow(2.0f32).expect("pow");
        let meta_param = Tensor::scalar(0.0f32).expect("tensor creation");

        let result = engine.compute_meta_gradient(
            &loss,
            std::slice::from_ref(&param),
            std::slice::from_ref(&meta_param),
        );
        assert!(
            result.is_err(),
            "second-order meta-gradient must surface an honest error"
        );
    }

    #[test]
    fn test_hyperparameter_creation() -> Result<()> {
        // Test learning rate hyperparameter
        let lr_hyperparam = HyperparameterOptimizer::create_learning_rate_hyperparam(0.01)?;
        assert!(matches!(
            lr_hyperparam.param_type,
            HyperparameterType::LearningRate
        ));
        assert!(lr_hyperparam.is_log_space);
        assert!(lr_hyperparam.requires_grad);
        assert!((lr_hyperparam.get_actual_value()? - 0.01).abs() < 1e-6);

        // Test L2 regularization hyperparameter
        let l2_hyperparam = HyperparameterOptimizer::create_l2_regularization_hyperparam(0.001)?;
        assert!(matches!(
            l2_hyperparam.param_type,
            HyperparameterType::L2Regularization
        ));
        assert!(l2_hyperparam.is_log_space);

        // Test dropout rate hyperparameter
        let dropout_hyperparam = HyperparameterOptimizer::create_dropout_rate_hyperparam(0.3)?;
        assert!(matches!(
            dropout_hyperparam.param_type,
            HyperparameterType::DropoutRate
        ));
        assert!(!dropout_hyperparam.is_log_space);
        assert!((dropout_hyperparam.get_actual_value()? - 0.3).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_hyperparameter_optimization_returns_honest_error() {
        // The inner optimization step relies on real first-order gradients of
        // disconnected scalar losses w.r.t. disconnected model parameters, and
        // the outer step relies on implicit-function-theorem hyperparameter
        // gradients. Neither is supported, so the optimizer must surface an
        // honest error rather than fabricate an optimization trajectory.
        let config = HyperparameterOptimizationConfig {
            hyperparam_lr: 0.01,
            inner_steps: 2,
            max_iterations: 3,
            tolerance: 1e-3,
            second_order: false,
            early_stopping: false,
            patience: 10,
        };
        let mut optimizer = HyperparameterOptimizer::new(config);

        // Create simple training and validation losses
        let train_loss1 = Tensor::scalar(2.0).expect("tensor creation");
        let train_loss2 = Tensor::scalar(1.5).expect("tensor creation");
        let train_losses = vec![train_loss1, train_loss2];

        let val_loss1 = Tensor::scalar(1.8).expect("tensor creation");
        let val_loss2 = Tensor::scalar(1.3).expect("tensor creation");
        let val_losses = vec![val_loss1, val_loss2];

        // Create simple model parameters
        let param1 = Tensor::ones(&[2, 2], DeviceType::Cpu).expect("tensor creation");
        let param2 = Tensor::ones(&[1, 2], DeviceType::Cpu).expect("tensor creation");
        let model_parameters = vec![param1, param2];

        // Create hyperparameters to optimize
        let lr_hyperparam =
            HyperparameterOptimizer::create_learning_rate_hyperparam(0.01).expect("hyperparam");
        let l2_hyperparam = HyperparameterOptimizer::create_l2_regularization_hyperparam(0.001)
            .expect("hyperparam");
        let hyperparameters = vec![lr_hyperparam, l2_hyperparam];

        // Run optimization
        let result = optimizer.optimize_hyperparameters(
            &train_losses,
            &val_losses,
            &model_parameters,
            hyperparameters,
        );

        assert!(
            result.is_err(),
            "optimization over disconnected tensors must return an honest error, \
             not a fabricated result"
        );
    }
}
