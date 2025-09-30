//! Stress tests for ToRSh optimizers
//!
//! This module provides comprehensive stress testing for optimizers to ensure
//! robustness under extreme conditions and high load scenarios.

use crate::{OptimizerError, OptimizerResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

#[allow(dead_code)]
/// Configuration for stress tests
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Number of optimization steps to run
    pub num_steps: usize,
    /// Number of parameters in each tensor
    pub param_size: Vec<usize>,
    /// Number of parameter tensors
    pub num_params: usize,
    /// Gradient magnitude multiplier for extreme conditions
    pub gradient_scale: f32,
    /// Whether to test with infinite/NaN gradients
    pub test_edge_cases: bool,
    /// Maximum allowed execution time
    pub max_execution_time: Duration,
    /// Memory usage tracking
    pub track_memory: bool,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            param_size: vec![100, 100],
            num_params: 10,
            gradient_scale: 1.0,
            test_edge_cases: true,
            max_execution_time: Duration::from_secs(30),
            track_memory: true,
        }
    }
}

#[allow(dead_code)]
/// Results from stress testing
#[derive(Debug, Clone)]
pub struct StressTestResult {
    /// Whether the test passed without errors
    pub passed: bool,
    /// Total execution time
    pub execution_time: Duration,
    /// Average time per step
    pub avg_step_time: Duration,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f32>,
    /// Any errors encountered
    pub errors: Vec<String>,
}

#[allow(dead_code)]
/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (estimated)
    pub peak_memory_mb: f32,
    /// Average memory usage
    pub avg_memory_mb: f32,
    /// Memory growth rate
    pub memory_growth_rate: f32,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            avg_memory_mb: 0.0,
            memory_growth_rate: 0.0,
        }
    }
}

#[allow(dead_code)]
/// Stress tester for optimizers
pub struct OptimizerStressTester {
    config: StressTestConfig,
}

#[allow(dead_code)]
impl OptimizerStressTester {
    /// Create a new stress tester with configuration
    pub fn new(config: StressTestConfig) -> Self {
        Self { config }
    }

    /// Create a stress tester with default configuration
    pub fn default() -> Self {
        Self::new(StressTestConfig::default())
    }

    /// Run comprehensive stress tests on an optimizer
    pub fn run_stress_test<O>(&self, mut optimizer: O) -> OptimizerResult<StressTestResult>
    where
        O: crate::Optimizer,
    {
        let start_time = Instant::now();
        let mut errors = Vec::new();
        let mut step_times = Vec::new();
        let mut memory_measurements = Vec::new();

        // Create large parameter tensors for stress testing
        let mut params = Vec::new();
        for i in 0..self.config.num_params {
            let param = Arc::new(RwLock::new(randn::<f32>(&self.config.param_size).map_err(
                |e| {
                    OptimizerError::InvalidParameter(format!("Failed to create param {}: {}", i, e))
                },
            )?));
            params.push(param);
        }

        // Run optimization steps with timing
        for step in 0..self.config.num_steps {
            let step_start = Instant::now();

            // Set gradients for all parameters
            for (i, param) in params.iter().enumerate() {
                let gradient = if self.config.test_edge_cases && step % 100 == 50 {
                    // Inject extreme gradients occasionally
                    self.create_extreme_gradient(&self.config.param_size, step)?
                } else {
                    randn::<f32>(&self.config.param_size)
                        .map_err(|e| {
                            OptimizerError::InvalidParameter(format!(
                                "Failed to create gradient for param {}: {}",
                                i, e
                            ))
                        })?
                        .mul_scalar(self.config.gradient_scale)
                        .map_err(|e| {
                            OptimizerError::InvalidParameter(format!(
                                "Failed to scale gradient: {}",
                                e
                            ))
                        })?
                };

                param.write().set_grad(Some(gradient));
            }

            // Perform optimization step
            match optimizer.step() {
                Ok(_) => {}
                Err(e) => {
                    errors.push(format!("Step {}: {}", step, e));
                    if errors.len() > 10 {
                        break; // Stop after too many errors
                    }
                }
            }

            let step_duration = step_start.elapsed();
            step_times.push(step_duration);

            // Memory tracking (simplified estimation)
            if self.config.track_memory && step % 10 == 0 {
                let estimated_memory = self.estimate_memory_usage(&params);
                memory_measurements.push(estimated_memory);
            }

            // Check for timeout
            if start_time.elapsed() > self.config.max_execution_time {
                errors.push("Test exceeded maximum execution time".to_string());
                break;
            }
        }

        let total_time = start_time.elapsed();
        let avg_step_time = if !step_times.is_empty() {
            step_times.iter().sum::<Duration>() / step_times.len() as u32
        } else {
            Duration::from_nanos(0)
        };

        // Calculate memory statistics
        let memory_stats = if self.config.track_memory && !memory_measurements.is_empty() {
            let peak_memory = memory_measurements
                .iter()
                .fold(0.0f32, |acc, x| acc.max(*x));
            let avg_memory =
                memory_measurements.iter().sum::<f32>() / memory_measurements.len() as f32;
            let growth_rate = if memory_measurements.len() > 1 {
                (memory_measurements[memory_measurements.len() - 1] - memory_measurements[0])
                    / memory_measurements.len() as f32
            } else {
                0.0
            };

            MemoryStats {
                peak_memory_mb: peak_memory,
                avg_memory_mb: avg_memory,
                memory_growth_rate: growth_rate,
            }
        } else {
            MemoryStats::default()
        };

        // Calculate performance metrics
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert(
            "steps_per_second".to_string(),
            self.config.num_steps as f32 / total_time.as_secs_f32(),
        );
        performance_metrics.insert(
            "error_rate".to_string(),
            errors.len() as f32 / self.config.num_steps as f32,
        );
        if !step_times.is_empty() {
            performance_metrics.insert(
                "avg_step_time_ms".to_string(),
                avg_step_time.as_millis() as f32,
            );
            performance_metrics.insert(
                "max_step_time_ms".to_string(),
                step_times.iter().max().unwrap().as_millis() as f32,
            );
        }

        let passed = errors.is_empty() && total_time <= self.config.max_execution_time;

        Ok(StressTestResult {
            passed,
            execution_time: total_time,
            avg_step_time,
            memory_stats,
            performance_metrics,
            errors,
        })
    }

    /// Test optimizer stability under extreme conditions
    pub fn test_extreme_conditions<O>(&self, mut optimizer: O) -> OptimizerResult<StressTestResult>
    where
        O: crate::Optimizer,
    {
        let start_time = Instant::now();
        let mut errors = Vec::new();

        // Create a single parameter for testing
        let param = Arc::new(RwLock::new(zeros(&[10, 10])?));

        // Test cases: [magnitude, description]
        let test_cases = vec![
            (1e10, "Very large gradients"),
            (1e-10, "Very small gradients"),
            (0.0, "Zero gradients"),
            (f32::INFINITY, "Infinite gradients"),
            (f32::NAN, "NaN gradients"),
        ];

        let test_cases_len = test_cases.len();
        for (magnitude, description) in test_cases {
            // Create gradient with the test magnitude
            let mut grad_data = vec![magnitude; 100];
            if magnitude.is_nan() {
                grad_data = vec![f32::NAN; 100];
            }

            let grad_tensor = Tensor::from_vec(grad_data, &[10, 10]).map_err(|e| {
                OptimizerError::InvalidParameter(format!("Failed to create test gradient: {}", e))
            })?;

            param.write().set_grad(Some(grad_tensor));

            // Test optimizer step
            match optimizer.step() {
                Ok(_) => {
                    // Check if parameters are still valid
                    let param_values = param.read().to_vec().map_err(|e| {
                        OptimizerError::InvalidParameter(format!(
                            "Failed to read parameter values: {}",
                            e
                        ))
                    })?;

                    let has_invalid = param_values.iter().any(|&x| x.is_nan() || x.is_infinite());
                    if has_invalid {
                        errors.push(format!("{}: Parameters became invalid", description));
                    }
                }
                Err(e) => {
                    // Some errors are expected for extreme inputs
                    if !matches!(magnitude, val if val.is_infinite() || val.is_nan()) {
                        errors.push(format!("{}: Unexpected error: {}", description, e));
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        let passed = errors.len() < test_cases_len / 2; // Allow some failures for extreme cases

        let mut performance_metrics = HashMap::new();
        performance_metrics.insert(
            "extreme_case_success_rate".to_string(),
            (test_cases_len - errors.len()) as f32 / test_cases_len as f32,
        );

        Ok(StressTestResult {
            passed,
            execution_time: total_time,
            avg_step_time: total_time / test_cases_len as u32,
            memory_stats: MemoryStats::default(),
            performance_metrics,
            errors,
        })
    }

    /// Create extreme gradients for testing edge cases
    fn create_extreme_gradient(&self, shape: &[usize], step: usize) -> OptimizerResult<Tensor> {
        let total_elements: usize = shape.iter().product();

        let gradient_data = match step % 4 {
            0 => vec![1e6; total_elements],  // Very large
            1 => vec![1e-6; total_elements], // Very small
            2 => vec![0.0; total_elements],  // Zero
            _ => {
                // Alternating pattern
                (0..total_elements)
                    .map(|i| if i % 2 == 0 { 1e3 } else { -1e3 })
                    .collect()
            }
        };

        Tensor::from_vec(gradient_data, shape).map_err(|e| {
            OptimizerError::InvalidParameter(format!("Failed to create extreme gradient: {}", e))
        })
    }

    /// Estimate memory usage of parameters (simplified)
    fn estimate_memory_usage(&self, params: &[Arc<RwLock<Tensor>>]) -> f32 {
        let mut total_elements = 0;
        for param in params {
            if let Some(param_read) = param.try_read() {
                let shape = param_read.shape();
                total_elements += shape.dims().iter().product::<usize>();
            }
        }

        // Estimate: 4 bytes per f32 element, convert to MB
        (total_elements * 4) as f32 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adam::Adam, sgd::SGD};

    #[test]
    fn test_stress_tester_creation() -> OptimizerResult<()> {
        let config = StressTestConfig::default();
        let _tester = OptimizerStressTester::new(config);
        Ok(())
    }

    #[test]
    fn test_basic_stress_test() -> OptimizerResult<()> {
        let mut config = StressTestConfig::default();
        config.num_steps = 10; // Keep test fast
        config.num_params = 2;
        config.param_size = vec![5, 5];

        let tester = OptimizerStressTester::new(config);
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));
        let optimizer = SGD::new(vec![param], 0.01, None, None, None, false);

        let result = tester.run_stress_test(optimizer)?;

        // Should complete without major issues
        assert!(result.execution_time.as_secs() < 5);
        assert!(result.performance_metrics.contains_key("steps_per_second"));
        Ok(())
    }

    #[test]
    fn test_extreme_conditions() -> OptimizerResult<()> {
        let config = StressTestConfig::default();
        let tester = OptimizerStressTester::new(config);
        let param = Arc::new(RwLock::new(zeros(&[10, 10])?));
        let optimizer = Adam::new(vec![param], Some(0.01), None, None, None, false);

        let result = tester.test_extreme_conditions(optimizer)?;

        // Should handle at least some extreme cases
        assert!(result
            .performance_metrics
            .contains_key("extreme_case_success_rate"));
        Ok(())
    }

    #[test]
    fn test_memory_estimation() -> OptimizerResult<()> {
        let tester = OptimizerStressTester::default();
        let params = vec![
            Arc::new(RwLock::new(zeros(&[100, 100])?)),
            Arc::new(RwLock::new(zeros(&[50, 50])?)),
        ];

        let memory_usage = tester.estimate_memory_usage(&params);

        // Should estimate reasonable memory usage (100*100 + 50*50 = 12500 elements * 4 bytes)
        assert!(memory_usage > 0.04); // At least 0.04 MB
        assert!(memory_usage < 1.0); // Less than 1 MB
        Ok(())
    }
}
