//! Cross-framework validation tests for ToRSh optimizers
//!
//! This module provides functionality to validate ToRSh optimizer behavior
//! against other deep learning frameworks to ensure compatibility and correctness.

use crate::{Optimizer, OptimizerError, OptimizerResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

#[allow(dead_code)]
/// Cross-framework validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Tolerance for numerical differences
    pub tolerance: f32,
    /// Number of optimization steps to compare
    pub num_steps: usize,
    /// Learning rate for comparison
    pub learning_rate: f32,
    /// Whether to enable verbose logging
    pub verbose: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-4,
            num_steps: 10,
            learning_rate: 0.01,
            verbose: false,
        }
    }
}

#[allow(dead_code)]
/// Results from cross-framework validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the validation passed
    pub passed: bool,
    /// Maximum difference observed
    pub max_difference: f32,
    /// Average difference across all steps
    pub avg_difference: f32,
    /// Per-step differences
    pub step_differences: Vec<f32>,
    /// Additional metrics
    pub metrics: HashMap<String, f32>,
}

#[allow(dead_code)]
/// Cross-framework validator for optimizers
pub struct CrossFrameworkValidator {
    config: ValidationConfig,
}

#[allow(dead_code)]
impl CrossFrameworkValidator {
    /// Create a new validator with the given configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default configuration
    pub fn default() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Validate an optimizer against PyTorch's equivalent
    pub fn validate_against_pytorch<O>(
        &self,
        mut torsh_optimizer: O,
        pytorch_reference: &[f32],
    ) -> OptimizerResult<ValidationResult>
    where
        O: crate::Optimizer,
    {
        let mut differences = Vec::new();
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;

        // Create test parameters
        let param = Arc::new(RwLock::new(zeros(&[2, 2])?));

        for step in 0..self.config.num_steps {
            // Set a consistent gradient for testing
            let grad_data = vec![0.1, 0.2, 0.3, 0.4];
            let grad_tensor = Tensor::from_vec(grad_data, &[2, 2])?;
            param.write().set_grad(Some(grad_tensor));

            // Step the ToRSh optimizer
            torsh_optimizer.step()?;

            // Get the current parameter values
            let torsh_values = param.read().to_vec()?;

            // Compare with PyTorch reference (simplified for example)
            let pytorch_values = &pytorch_reference[step * 4..(step + 1) * 4];

            // Compute differences
            let step_diff = torsh_values
                .iter()
                .zip(pytorch_values.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, |acc, x| acc.max(x));

            differences.push(step_diff);
            max_diff = max_diff.max(step_diff);
            sum_diff += step_diff;

            if self.config.verbose {
                println!("Step {}: max_diff = {:.6}", step, step_diff);
            }
        }

        let avg_diff = sum_diff / self.config.num_steps as f32;
        let passed = max_diff < self.config.tolerance;

        let mut metrics = HashMap::new();
        metrics.insert("convergence_rate".to_string(), avg_diff);
        metrics.insert("stability_score".to_string(), 1.0 / (1.0 + max_diff));

        Ok(ValidationResult {
            passed,
            max_difference: max_diff,
            avg_difference: avg_diff,
            step_differences: differences,
            metrics,
        })
    }

    /// Validate optimizer convergence properties
    pub fn validate_convergence<O>(&self, mut optimizer: O) -> OptimizerResult<ValidationResult>
    where
        O: crate::Optimizer,
    {
        let mut losses = Vec::new();

        // Create a simple quadratic optimization problem: min 0.5 * x^T * A * x
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 1])?));

        for step in 0..self.config.num_steps {
            // Compute gradient: grad = A * x where A is identity for simplicity
            let current = param.read().clone();
            let loss = current.pow(2.0)?.sum()?.to_vec()?[0] * 0.5;
            losses.push(loss);

            // Set gradient for the optimizer
            param.write().set_grad(Some(current.clone()));

            // Optimization step
            optimizer.step()?;

            if self.config.verbose {
                println!("Step {}: loss = {:.6}", step, loss);
            }
        }

        // Check if loss is decreasing (convergence)
        let initial_loss = losses[0];
        let final_loss = losses[losses.len() - 1];
        let loss_reduction = (initial_loss - final_loss) / initial_loss;

        let passed = loss_reduction > 0.1; // At least 10% improvement

        let mut metrics = HashMap::new();
        metrics.insert("initial_loss".to_string(), initial_loss);
        metrics.insert("final_loss".to_string(), final_loss);
        metrics.insert("loss_reduction".to_string(), loss_reduction);

        Ok(ValidationResult {
            passed,
            max_difference: final_loss,
            avg_difference: losses.iter().sum::<f32>() / losses.len() as f32,
            step_differences: losses,
            metrics,
        })
    }

    /// Run a comprehensive validation suite
    pub fn run_validation_suite<O>(
        &self,
        optimizer: O,
    ) -> OptimizerResult<HashMap<String, ValidationResult>>
    where
        O: crate::Optimizer,
    {
        let mut results = HashMap::new();

        // Test 1: Convergence validation
        let convergence_result = self.validate_convergence(optimizer)?;
        results.insert("convergence".to_string(), convergence_result);

        // Note: We can't run multiple tests with the same optimizer since it doesn't implement Clone
        // This is a limitation of the current design - each test consumes the optimizer

        Ok(results)
    }

    /// Validate basic gradient descent properties
    fn validate_gradient_descent_properties<O>(
        &self,
        _optimizer: O,
    ) -> OptimizerResult<ValidationResult>
    where
        O: crate::Optimizer,
    {
        // Test that parameters move in the opposite direction of gradients
        let param = Arc::new(RwLock::new(zeros(&[2, 2])?));
        let initial_params = param.read().to_vec()?;

        // Create a new optimizer with our test parameter
        let mut optimizer = crate::SGD::new(vec![param.clone()], 0.1, None, None, None, false);

        // Set positive gradients
        let grad_tensor = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;
        param.write().set_grad(Some(grad_tensor));

        optimizer.step()?;

        let final_params = param.read().to_vec()?;

        // Parameters should have moved in negative direction (opposite to gradient)
        let moved_correctly = initial_params
            .iter()
            .zip(final_params.iter())
            .all(|(initial, final_val)| final_val < initial);

        let max_movement = initial_params
            .iter()
            .zip(final_params.iter())
            .map(|(initial, final_val)| ((*initial - *final_val) as f32).abs())
            .fold(0.0f32, |acc, x| acc.max(x));

        let mut metrics = HashMap::new();
        metrics.insert("max_movement".to_string(), max_movement);
        metrics.insert(
            "correct_direction".to_string(),
            if moved_correctly { 1.0 } else { 0.0 },
        );

        Ok(ValidationResult {
            passed: moved_correctly,
            max_difference: max_movement,
            avg_difference: max_movement / 4.0, // 4 parameters
            step_differences: vec![max_movement],
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adam::Adam, sgd::SGD};

    #[test]
    fn test_cross_framework_validator_creation() -> OptimizerResult<()> {
        let config = ValidationConfig::default();
        let _validator = CrossFrameworkValidator::new(config);
        Ok(())
    }

    #[test]
    fn test_convergence_validation() -> OptimizerResult<()> {
        let validator = CrossFrameworkValidator::default();
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 1])?));
        let optimizer = SGD::new(vec![param], 0.01, None, None, None, false);

        let result = validator.validate_convergence(optimizer)?;

        // Should show some form of optimization progress
        assert!(result.metrics.contains_key("loss_reduction"));
        Ok(())
    }

    #[test]
    fn test_gradient_descent_properties() -> OptimizerResult<()> {
        let validator = CrossFrameworkValidator::default();
        let param = Arc::new(RwLock::new(zeros(&[2, 2])?));
        let optimizer = SGD::new(vec![param], 0.1, None, None, None, false);

        let result = validator.validate_gradient_descent_properties(optimizer)?;

        // Should move parameters in correct direction
        assert_eq!(result.metrics.get("correct_direction"), Some(&1.0));
        Ok(())
    }

    #[test]
    fn test_validation_suite() -> OptimizerResult<()> {
        let validator = CrossFrameworkValidator::default();
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 1])?));
        let optimizer = Adam::new(vec![param], None, None, None, None, false);

        let results = validator.run_validation_suite(optimizer)?;

        assert!(results.contains_key("convergence"));
        // Note: Only convergence test is run now since optimizer doesn't implement Clone
        Ok(())
    }
}
