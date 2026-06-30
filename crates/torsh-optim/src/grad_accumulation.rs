//! Gradient accumulation utilities for optimizers
//!
//! Gradient accumulation allows you to simulate larger batch sizes by accumulating gradients
//! over multiple smaller batches before taking an optimizer step. This is especially useful
//! when memory constraints prevent you from using large batch sizes.

use crate::{Optimizer, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Wrapper optimizer that provides gradient accumulation functionality
pub struct GradientAccumulator<O: Optimizer> {
    optimizer: O,
    accumulation_steps: u32,
    current_step: u32,
    accumulated_grads: HashMap<String, Tensor>,
}

impl<O: Optimizer> GradientAccumulator<O> {
    /// Create a new gradient accumulator wrapper
    ///
    /// # Arguments
    /// * `optimizer` - The underlying optimizer to wrap
    /// * `accumulation_steps` - Number of gradient accumulation steps before taking an optimizer step
    pub fn new(optimizer: O, accumulation_steps: u32) -> Self {
        if accumulation_steps == 0 {
            panic!("Accumulation steps must be greater than 0");
        }

        Self {
            optimizer,
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
        }
    }

    /// Get the number of accumulation steps
    pub fn accumulation_steps(&self) -> u32 {
        self.accumulation_steps
    }

    /// Get the current accumulation step
    pub fn current_step(&self) -> u32 {
        self.current_step
    }

    /// Check if we should take an optimizer step (accumulation buffer is full)
    pub fn should_step(&self) -> bool {
        (self.current_step + 1) % self.accumulation_steps == 0
    }

    /// Stable per-parameter key derived from the tensor handle's identity.
    fn param_key(param: &Arc<RwLock<Tensor>>) -> String {
        format!("param_{:p}", Arc::as_ptr(param))
    }

    /// Accumulate gradients from the current batch.
    ///
    /// For each parameter exposed by the wrapped optimizer that currently holds a
    /// gradient, the gradient is summed into a per-parameter accumulation buffer.
    /// The buffer is initialised on first observation and added to on subsequent
    /// batches, so after `n` calls each buffer holds the sum of `n` gradients.
    pub fn accumulate_gradients(&mut self) -> Result<()> {
        let params = self.optimizer.parameters();

        // The wrapped optimizer must expose its parameters for accumulation to be
        // possible. An empty list means accumulation cannot be performed, so fail
        // loudly rather than silently dropping every gradient.
        if params.is_empty() {
            return Err(TorshError::RuntimeError(
                "GradientAccumulator requires the wrapped optimizer to expose its \
                 parameters via `Optimizer::parameters()`, but it returned an empty \
                 list. Cannot accumulate gradients."
                    .to_string(),
            ));
        }

        for param in &params {
            let key = Self::param_key(param);
            let grad = {
                let guard = param.read();
                match guard.grad() {
                    Some(g) => g,
                    None => continue,
                }
            };

            match self.accumulated_grads.get_mut(&key) {
                Some(buffer) => {
                    *buffer = buffer.add(&grad)?;
                }
                None => {
                    self.accumulated_grads.insert(key, grad);
                }
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Write the averaged accumulated gradients back onto the parameters so the
    /// wrapped optimizer's `step()` consumes the accumulated (mean) gradient.
    fn write_back_accumulated_gradients(&self) -> Result<()> {
        for param in self.optimizer.parameters() {
            let key = Self::param_key(&param);
            if let Some(buffer) = self.accumulated_grads.get(&key) {
                param.write().set_grad(Some(buffer.clone()));
            }
        }
        Ok(())
    }

    /// Step the optimizer if accumulation buffer is full, and reset accumulation
    pub fn step(&mut self) -> OptimizerResult<()> {
        // Determine whether this batch fills the accumulation buffer *before*
        // accumulating, so the boundary batch's gradients are included.
        let should_step = self.should_step();

        // Always fold the current batch's gradients into the accumulation buffers.
        self.accumulate_gradients()?;

        if should_step {
            // Average the accumulated gradients and place them back on the params.
            self.average_accumulated_gradients()?;
            self.write_back_accumulated_gradients()?;

            // Take optimizer step
            self.optimizer.step()?;

            // Reset accumulation
            self.reset_accumulation();
        }

        Ok(())
    }

    /// Manually trigger an optimizer step (regardless of accumulation buffer state)
    pub fn step_force(&mut self) -> Result<()> {
        if self.current_step > 0 {
            // Average the accumulated gradients and place them back on the params.
            self.average_accumulated_gradients()?;
            self.write_back_accumulated_gradients()?;

            // Take optimizer step
            self.optimizer.step()?;

            // Reset accumulation
            self.reset_accumulation();
        }

        Ok(())
    }

    /// Reset gradient accumulation state
    pub fn reset_accumulation(&mut self) {
        self.current_step = 0;
        self.accumulated_grads.clear();
    }

    /// Average the accumulated gradients by dividing by accumulation steps
    fn average_accumulated_gradients(&mut self) -> Result<()> {
        let divisor = if self.current_step == 0 {
            1.0
        } else {
            self.current_step as f32
        };

        for (_, accumulated_grad) in self.accumulated_grads.iter_mut() {
            accumulated_grad.div_scalar_(divisor)?;
        }

        Ok(())
    }

    /// Get a reference to the underlying optimizer
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get a mutable reference to the underlying optimizer
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

impl<O: Optimizer> Optimizer for GradientAccumulator<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step()
    }

    fn zero_grad(&mut self) {
        self.optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.optimizer.add_param_group(params, options);
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.optimizer.parameters()
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.optimizer.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.optimizer.load_state_dict(state)
    }
}

/// Helper function to wrap any optimizer with gradient accumulation
pub fn with_gradient_accumulation<O: Optimizer>(
    optimizer: O,
    accumulation_steps: u32,
) -> GradientAccumulator<O> {
    GradientAccumulator::new(optimizer, accumulation_steps)
}

/// Trait for optimizers that support efficient gradient accumulation
pub trait GradientAccumulationSupport {
    /// Accumulate gradients without taking an optimizer step
    fn accumulate_gradients(&mut self) -> Result<()>;

    /// Check if gradients should be averaged before the next step
    fn should_average_gradients(&self) -> bool;

    /// Set the number of accumulation steps
    fn set_accumulation_steps(&mut self, steps: u32);

    /// Get the number of accumulation steps
    fn get_accumulation_steps(&self) -> u32;
}

/// Enhanced base optimizer with built-in gradient accumulation support
pub struct AccumulatingOptimizer<O: Optimizer> {
    optimizer: O,
    accumulation_steps: u32,
    current_step: u32,
    auto_average: bool,
    accumulated_grads: HashMap<String, Tensor>,
}

impl<O: Optimizer> AccumulatingOptimizer<O> {
    /// Create a new accumulating optimizer
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            accumulation_steps: 1,
            current_step: 0,
            auto_average: true,
            accumulated_grads: HashMap::new(),
        }
    }

    /// Stable per-parameter key derived from the tensor handle's identity.
    fn param_key(param: &Arc<RwLock<Tensor>>) -> String {
        format!("param_{:p}", Arc::as_ptr(param))
    }

    /// Write the accumulated gradients (optionally averaged) back onto the
    /// parameters so the wrapped optimizer's `step()` consumes them.
    fn finalize_accumulated_gradients(&mut self) -> Result<()> {
        let divisor = if self.auto_average && self.current_step > 0 {
            self.current_step as f32
        } else {
            1.0
        };

        for param in self.optimizer.parameters() {
            let key = Self::param_key(&param);
            if let Some(buffer) = self.accumulated_grads.get(&key) {
                let grad = if divisor != 1.0 {
                    buffer.div_scalar(divisor)?
                } else {
                    buffer.clone()
                };
                param.write().set_grad(Some(grad));
            }
        }

        self.accumulated_grads.clear();
        Ok(())
    }

    /// Set whether to automatically average gradients
    pub fn set_auto_average(&mut self, auto_average: bool) {
        self.auto_average = auto_average;
    }

    /// Get the underlying optimizer
    pub fn inner(&self) -> &O {
        &self.optimizer
    }

    /// Get the underlying optimizer mutably
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Consume this optimizer and return the inner optimizer
    pub fn into_inner(self) -> O {
        self.optimizer
    }
}

impl<O: Optimizer> Optimizer for AccumulatingOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Fold the current batch's gradients into the accumulation buffers.
        self.accumulate_gradients()?;

        if self.current_step % self.accumulation_steps == 0 {
            // Place the (optionally averaged) accumulated gradients back on the
            // parameters, then let the wrapped optimizer consume them.
            self.finalize_accumulated_gradients()?;
            self.optimizer.step()?;
            self.current_step = 0;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.optimizer.zero_grad();
    }

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.optimizer.parameters()
    }

    fn get_lr(&self) -> Vec<f32> {
        self.optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.optimizer.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.optimizer.load_state_dict(state)
    }
}

impl<O: Optimizer> GradientAccumulationSupport for AccumulatingOptimizer<O> {
    fn accumulate_gradients(&mut self) -> Result<()> {
        let params = self.optimizer.parameters();

        // The wrapped optimizer must expose its parameters for accumulation to be
        // possible; an empty list means gradients cannot be accumulated.
        if params.is_empty() {
            return Err(TorshError::RuntimeError(
                "AccumulatingOptimizer requires the wrapped optimizer to expose its \
                 parameters via `Optimizer::parameters()`, but it returned an empty \
                 list. Cannot accumulate gradients."
                    .to_string(),
            ));
        }

        for param in &params {
            let key = Self::param_key(param);
            let grad = {
                let guard = param.read();
                match guard.grad() {
                    Some(g) => g,
                    None => continue,
                }
            };

            match self.accumulated_grads.get_mut(&key) {
                Some(buffer) => {
                    *buffer = buffer.add(&grad)?;
                }
                None => {
                    self.accumulated_grads.insert(key, grad);
                }
            }
        }

        self.current_step += 1;
        Ok(())
    }

    fn should_average_gradients(&self) -> bool {
        self.auto_average
            && self.current_step > 0
            && self.current_step % self.accumulation_steps == 0
    }

    fn set_accumulation_steps(&mut self, steps: u32) {
        if steps == 0 {
            panic!("Accumulation steps must be greater than 0");
        }
        self.accumulation_steps = steps;
    }

    fn get_accumulation_steps(&self) -> u32 {
        self.accumulation_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgd::SGD;
    use torsh_tensor::creation;

    #[test]
    fn test_gradient_accumulator_creation() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        let accumulator = GradientAccumulator::new(sgd, 4);
        assert_eq!(accumulator.accumulation_steps(), 4);
        assert_eq!(accumulator.current_step(), 0);
        assert!(!accumulator.should_step());
    }

    #[test]
    fn test_gradient_accumulator_step_logic() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        let mut accumulator = GradientAccumulator::new(sgd, 3);

        // First step - should accumulate
        assert!(!accumulator.should_step());
        accumulator.accumulate_gradients().unwrap();
        assert_eq!(accumulator.current_step(), 1);

        // Second step - should accumulate
        assert!(!accumulator.should_step());
        accumulator.accumulate_gradients().unwrap();
        assert_eq!(accumulator.current_step(), 2);

        // Third step - should trigger optimizer step
        assert!(accumulator.should_step());
    }

    #[test]
    fn test_accumulating_optimizer() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        let mut acc_optimizer = AccumulatingOptimizer::new(sgd);
        acc_optimizer.set_accumulation_steps(2);

        assert_eq!(acc_optimizer.get_accumulation_steps(), 2);
        assert!(!acc_optimizer.should_average_gradients());

        acc_optimizer.accumulate_gradients().unwrap();
        assert!(!acc_optimizer.should_average_gradients());

        acc_optimizer.accumulate_gradients().unwrap();
        assert!(acc_optimizer.should_average_gradients());
    }

    #[test]
    fn test_with_gradient_accumulation_helper() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        let accumulator = with_gradient_accumulation(sgd, 5);
        assert_eq!(accumulator.accumulation_steps(), 5);
    }

    #[test]
    #[should_panic(expected = "Accumulation steps must be greater than 0")]
    fn test_zero_accumulation_steps_panics() {
        let param = Arc::new(RwLock::new(creation::randn::<f32>(&[2, 3]).unwrap()));
        let sgd = SGD::new(vec![param], 0.01, None, None, None, false);

        GradientAccumulator::new(sgd, 0);
    }

    #[test]
    fn test_accumulate_then_average_produces_mean() {
        // Accumulating two gradients (values 2.0 then 4.0) and then averaging must
        // yield their mean (3.0). This guards against the previous fabrication where
        // `accumulate_gradients` did nothing and the buffers stayed empty.
        let param = Arc::new(RwLock::new(creation::ones::<f32>(&[2, 2]).unwrap()));
        let sgd = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut accumulator = GradientAccumulator::new(sgd, 4);

        // First batch: gradient of all 2.0
        {
            let grad = creation::ones::<f32>(&[2, 2])
                .unwrap()
                .mul_scalar(2.0)
                .unwrap();
            param.write().set_grad(Some(grad));
        }
        accumulator.accumulate_gradients().unwrap();
        assert_eq!(accumulator.current_step(), 1);

        // Second batch: gradient of all 4.0
        {
            let grad = creation::ones::<f32>(&[2, 2])
                .unwrap()
                .mul_scalar(4.0)
                .unwrap();
            param.write().set_grad(Some(grad));
        }
        accumulator.accumulate_gradients().unwrap();
        assert_eq!(accumulator.current_step(), 2);

        // After accumulation the buffer holds the sum (2.0 + 4.0 = 6.0 per element).
        let key = GradientAccumulator::<SGD>::param_key(&param);
        let summed = accumulator
            .accumulated_grads
            .get(&key)
            .expect("accumulated gradient buffer must exist")
            .to_vec()
            .unwrap();
        for v in &summed {
            assert!((v - 6.0).abs() < 1e-6, "sum should be 6.0, got {v}");
        }

        // Averaging by the two steps must produce the mean (6.0 / 2 = 3.0).
        accumulator.average_accumulated_gradients().unwrap();
        let averaged = accumulator
            .accumulated_grads
            .get(&key)
            .expect("accumulated gradient buffer must exist after averaging")
            .to_vec()
            .unwrap();
        for v in &averaged {
            assert!((v - 3.0).abs() < 1e-6, "mean should be 3.0, got {v}");
        }
    }

    #[test]
    fn test_accumulate_writes_mean_back_onto_param_grad() {
        // End-to-end: after two micro-batches, `step_force` must write the averaged
        // gradient back onto the parameter before the wrapped optimizer steps.
        let param = Arc::new(RwLock::new(creation::ones::<f32>(&[3]).unwrap()));
        let sgd = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut accumulator = GradientAccumulator::new(sgd, 4);

        {
            let grad = creation::ones::<f32>(&[3])
                .unwrap()
                .mul_scalar(2.0)
                .unwrap();
            param.write().set_grad(Some(grad));
        }
        accumulator.accumulate_gradients().unwrap();

        {
            let grad = creation::ones::<f32>(&[3])
                .unwrap()
                .mul_scalar(6.0)
                .unwrap();
            param.write().set_grad(Some(grad));
        }
        accumulator.accumulate_gradients().unwrap();

        // mean = (2 + 6) / 2 = 4.0; param starts at 1.0, lr = 0.01 -> 1.0 - 0.04 = 0.96
        accumulator.step_force().unwrap();

        let updated = param.read().to_vec().unwrap();
        for v in &updated {
            assert!(
                (v - 0.96).abs() < 1e-5,
                "expected 0.96 after mean-grad SGD step, got {v}"
            );
        }
        // Accumulation state is reset after a step.
        assert_eq!(accumulator.current_step(), 0);
    }

    #[test]
    fn test_accumulating_optimizer_real_accumulation() {
        // `AccumulatingOptimizer` must accumulate real gradients into its buffers,
        // not merely count steps. After two accumulations the buffer holds the sum.
        let param = Arc::new(RwLock::new(creation::ones::<f32>(&[2]).unwrap()));
        let sgd = SGD::new(vec![param.clone()], 0.01, None, None, None, false);
        let mut acc = AccumulatingOptimizer::new(sgd);
        acc.set_accumulation_steps(2);

        {
            let grad = creation::ones::<f32>(&[2])
                .unwrap()
                .mul_scalar(3.0)
                .unwrap();
            param.write().set_grad(Some(grad));
        }
        acc.accumulate_gradients().unwrap();

        {
            let grad = creation::ones::<f32>(&[2])
                .unwrap()
                .mul_scalar(5.0)
                .unwrap();
            param.write().set_grad(Some(grad));
        }
        acc.accumulate_gradients().unwrap();

        let key = AccumulatingOptimizer::<SGD>::param_key(&param);
        let summed = acc
            .accumulated_grads
            .get(&key)
            .expect("accumulated gradient buffer must exist")
            .to_vec()
            .unwrap();
        for v in &summed {
            assert!((v - 8.0).abs() < 1e-6, "sum should be 8.0, got {v}");
        }
    }
}
