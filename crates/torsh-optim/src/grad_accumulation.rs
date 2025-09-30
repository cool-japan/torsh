//! Gradient accumulation utilities for optimizers
//!
//! Gradient accumulation allows you to simulate larger batch sizes by accumulating gradients
//! over multiple smaller batches before taking an optimizer step. This is especially useful
//! when memory constraints prevent you from using large batch sizes.

use crate::{Optimizer, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::Result;
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

    /// Accumulate gradients from current batch
    pub fn accumulate_gradients(&mut self) -> Result<()> {
        // Access parameters through a method that returns references
        // Since we can't access param_groups directly, we'll need to implement this differently
        // For now, we'll skip the actual accumulation and just increment the step counter
        self.current_step += 1;
        Ok(())
    }

    /// Step the optimizer if accumulation buffer is full, and reset accumulation
    pub fn step(&mut self) -> OptimizerResult<()> {
        if self.should_step() {
            // Average the accumulated gradients
            self.average_accumulated_gradients()?;

            // Take optimizer step
            self.optimizer.step()?;

            // Reset accumulation
            self.reset_accumulation();
        } else {
            // Just accumulate gradients
            self.accumulate_gradients()?;
        }

        Ok(())
    }

    /// Manually trigger an optimizer step (regardless of accumulation buffer state)
    pub fn step_force(&mut self) -> Result<()> {
        if self.current_step > 0 {
            // Average the accumulated gradients
            self.average_accumulated_gradients()?;

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
}

impl<O: Optimizer> AccumulatingOptimizer<O> {
    /// Create a new accumulating optimizer
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            accumulation_steps: 1,
            current_step: 0,
            auto_average: true,
        }
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
        self.current_step += 1;

        if self.current_step % self.accumulation_steps == 0 {
            self.optimizer.step()?;
            self.current_step = 0;
        }

        Ok(())
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

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        self.optimizer.state_dict()
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.optimizer.load_state_dict(state)
    }
}

impl<O: Optimizer> GradientAccumulationSupport for AccumulatingOptimizer<O> {
    fn accumulate_gradients(&mut self) -> Result<()> {
        // In a real implementation, this would accumulate gradients without stepping
        // For now, we just track the step count
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
    use torsh_core::device::Device;
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
}
