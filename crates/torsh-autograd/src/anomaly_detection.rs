//! Anomaly detection and automatic recovery for gradient computation
//!
//! This module provides comprehensive anomaly detection and automatic recovery
//! systems for robust gradient computation. It includes detection of numerical
//! anomalies (NaN, infinity, gradient explosion/vanishing) and automatic
//! recovery strategies to maintain training stability.
//!
//! # Features
//!
//! - **Complex anomaly detection**: Specialized detection for complex tensors
//! - **Automatic recovery**: Multiple recovery strategies for different anomaly types
//! - **Recovery statistics**: Tracking and analysis of recovery attempts
//! - **Configurable strategies**: Customizable recovery behavior
//! - **Gradient safety**: Specialized handling for gradient-related anomalies

use crate::autograd_traits::AutogradTensor;
use num_complex::Complex;
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use torsh_core::error::Result;
use tracing::{debug, error, info, warn};

/// Complex anomaly detection
///
/// Detects numerical anomalies in complex tensors including NaN, infinity,
/// and magnitude issues that could cause numerical instability.
pub fn detect_complex_anomalies<T>(tensor: &dyn AutogradTensor<Complex<T>>) -> Result<Vec<String>>
where
    T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
    Complex<T>: torsh_core::dtype::TensorElement,
    f32: From<T>,
{
    let mut anomalies = Vec::new();
    let data = tensor.data();

    for (i, complex_val) in data.iter().enumerate() {
        // Check for NaN in real or imaginary parts
        if complex_val.re.is_nan() || complex_val.im.is_nan() {
            anomalies.push(format!("NaN detected at index {} in complex tensor", i));
        }

        // Check for infinity in real or imaginary parts
        if complex_val.re.is_infinite() || complex_val.im.is_infinite() {
            anomalies.push(format!(
                "Infinity detected at index {} in complex tensor",
                i
            ));
        }

        // Check for very large magnitudes that might cause numerical issues
        let magnitude = complex_val.norm();
        let threshold = T::from(1e6).unwrap_or(<T as torsh_core::TensorElement>::one());
        if magnitude > threshold {
            anomalies.push(format!(
                "Large magnitude {:?} detected at index {} in complex tensor",
                magnitude, i
            ));
        }
    }

    if !anomalies.is_empty() {
        tracing::warn!("Detected {} complex anomalies", anomalies.len());
    }

    Ok(anomalies)
}

/// Automatic anomaly recovery strategies
pub mod recovery {
    use super::*;

    /// Configuration for automatic anomaly recovery
    #[derive(Debug, Clone)]
    pub struct RecoveryConfig {
        /// Enable automatic recovery
        pub enable_auto_recovery: bool,
        /// Maximum number of recovery attempts per anomaly
        pub max_recovery_attempts: usize,
        /// Timeout for recovery operations
        pub recovery_timeout: Duration,
        /// Whether to save checkpoints before recovery attempts
        pub save_recovery_checkpoints: bool,
        /// Learning rate reduction factor for gradient explosion
        pub lr_reduction_factor: f32,
        /// Gradient clipping threshold for recovery
        pub gradient_clip_threshold: f32,
        /// Number of recent losses to track for plateauing detection
        pub loss_history_size: usize,
    }

    impl Default for RecoveryConfig {
        fn default() -> Self {
            Self {
                enable_auto_recovery: true,
                max_recovery_attempts: 3,
                recovery_timeout: Duration::from_secs(60),
                save_recovery_checkpoints: true,
                lr_reduction_factor: 0.5,
                gradient_clip_threshold: 1.0,
                loss_history_size: 10,
            }
        }
    }

    /// Type of recovery strategy to apply
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum RecoveryStrategy {
        /// Reduce learning rate
        ReduceLearningRate,
        /// Apply gradient clipping
        GradientClipping,
        /// Reset to previous checkpoint
        ResetToCheckpoint,
        /// Reinitialize affected parameters
        ReinitializeParameters,
        /// Skip the current batch
        SkipBatch,
        /// Apply gradient scaling
        GradientScaling,
        /// Restart with different random seed
        RestartWithNewSeed,
        /// Apply regularization
        ApplyRegularization,
    }

    /// Result of a recovery attempt
    #[derive(Debug, Clone)]
    pub struct RecoveryResult {
        /// Whether recovery was successful
        pub success: bool,
        /// Strategy that was applied
        pub strategy: RecoveryStrategy,
        /// Time taken for recovery
        pub duration: Duration,
        /// Additional details about the recovery
        pub details: String,
        /// Whether a fallback strategy should be tried
        pub try_fallback: bool,
    }

    /// Automatic anomaly recovery system
    pub struct AnomalyRecoverySystem {
        /// Configuration
        config: RecoveryConfig,
        /// Recent recovery attempts
        recovery_history: VecDeque<(String, RecoveryStrategy, Instant, bool)>,
        /// Current recovery attempt count for each anomaly type
        recovery_attempts: HashMap<String, usize>,
        /// Cached parameter states for recovery
        #[allow(dead_code)]
        parameter_cache: HashMap<String, Vec<u8>>,
    }

    impl AnomalyRecoverySystem {
        /// Create a new recovery system
        pub fn new(config: RecoveryConfig) -> Self {
            Self {
                config,
                recovery_history: VecDeque::new(),
                recovery_attempts: HashMap::new(),
                parameter_cache: HashMap::new(),
            }
        }

        /// Attempt to recover from a detected anomaly
        pub fn attempt_recovery<T: num_traits::Float + num_traits::FromPrimitive>(
            &mut self,
            anomaly_type: &str,
            gradients: &mut HashMap<String, Vec<T>>,
            learning_rate: &mut f32,
        ) -> Result<RecoveryResult> {
            if !self.config.enable_auto_recovery {
                return Ok(RecoveryResult {
                    success: false,
                    strategy: RecoveryStrategy::SkipBatch,
                    duration: Duration::ZERO,
                    details: "Auto-recovery disabled".to_string(),
                    try_fallback: false,
                });
            }

            let start_time = Instant::now();

            // Check if we've exceeded maximum recovery attempts
            let attempt_count_value = {
                let attempt_count = self
                    .recovery_attempts
                    .entry(anomaly_type.to_string())
                    .or_insert(0);
                if *attempt_count >= self.config.max_recovery_attempts {
                    warn!(
                        "Maximum recovery attempts exceeded for anomaly: {}",
                        anomaly_type
                    );
                    return Ok(RecoveryResult {
                        success: false,
                        strategy: RecoveryStrategy::SkipBatch,
                        duration: start_time.elapsed(),
                        details: "Maximum recovery attempts exceeded".to_string(),
                        try_fallback: false,
                    });
                }
                *attempt_count += 1;
                *attempt_count
            };

            // Select recovery strategy based on anomaly type
            let strategy = self.select_recovery_strategy(anomaly_type, attempt_count_value);

            info!(
                "Attempting recovery for {} using strategy {:?} (attempt {})",
                anomaly_type, strategy, attempt_count_value
            );

            // Apply the selected recovery strategy
            let result = self.apply_recovery_strategy(strategy.clone(), gradients, learning_rate);

            // Record the recovery attempt
            self.recovery_history.push_back((
                anomaly_type.to_string(),
                strategy.clone(),
                start_time,
                result.is_ok(),
            ));

            // Limit history size
            if self.recovery_history.len() > 100 {
                self.recovery_history.pop_front();
            }

            match result {
                Ok(success) => {
                    if success {
                        info!(
                            "Recovery successful for {} using {:?}",
                            anomaly_type, strategy
                        );
                        // Reset attempt count on successful recovery
                        self.recovery_attempts.insert(anomaly_type.to_string(), 0);
                    } else {
                        warn!("Recovery failed for {} using {:?}", anomaly_type, strategy);
                    }

                    Ok(RecoveryResult {
                        success,
                        strategy,
                        duration: start_time.elapsed(),
                        details: if success {
                            "Recovery applied successfully".to_string()
                        } else {
                            "Recovery strategy failed".to_string()
                        },
                        try_fallback: !success
                            && attempt_count_value < self.config.max_recovery_attempts,
                    })
                }
                Err(e) => {
                    error!("Error during recovery for {}: {}", anomaly_type, e);
                    Ok(RecoveryResult {
                        success: false,
                        strategy,
                        duration: start_time.elapsed(),
                        details: format!("Recovery error: {}", e),
                        try_fallback: attempt_count_value < self.config.max_recovery_attempts,
                    })
                }
            }
        }

        /// Select the appropriate recovery strategy based on anomaly type and attempt number
        pub fn select_recovery_strategy(
            &self,
            anomaly_type: &str,
            attempt_count: usize,
        ) -> RecoveryStrategy {
            match anomaly_type.to_lowercase().as_str() {
                "nan" | "infinity" => match attempt_count {
                    1 => RecoveryStrategy::GradientClipping,
                    2 => RecoveryStrategy::ReduceLearningRate,
                    _ => RecoveryStrategy::ResetToCheckpoint,
                },
                "gradient_explosion" => match attempt_count {
                    1 => RecoveryStrategy::GradientClipping,
                    2 => RecoveryStrategy::ReduceLearningRate,
                    _ => RecoveryStrategy::GradientScaling,
                },
                "gradient_vanishing" => match attempt_count {
                    1 => RecoveryStrategy::ReduceLearningRate,
                    2 => RecoveryStrategy::ReinitializeParameters,
                    _ => RecoveryStrategy::RestartWithNewSeed,
                },
                "loss_plateauing" => match attempt_count {
                    1 => RecoveryStrategy::ReduceLearningRate,
                    2 => RecoveryStrategy::ApplyRegularization,
                    _ => RecoveryStrategy::RestartWithNewSeed,
                },
                "statistical_outlier" => match attempt_count {
                    1 => RecoveryStrategy::GradientClipping,
                    _ => RecoveryStrategy::SkipBatch,
                },
                _ => {
                    // Default strategy progression
                    match attempt_count {
                        1 => RecoveryStrategy::GradientClipping,
                        2 => RecoveryStrategy::ReduceLearningRate,
                        _ => RecoveryStrategy::SkipBatch,
                    }
                }
            }
        }

        /// Apply the selected recovery strategy
        fn apply_recovery_strategy<T: num_traits::Float + num_traits::FromPrimitive>(
            &mut self,
            strategy: RecoveryStrategy,
            gradients: &mut HashMap<String, Vec<T>>,
            learning_rate: &mut f32,
        ) -> Result<bool> {
            match strategy {
                RecoveryStrategy::ReduceLearningRate => {
                    let old_lr = *learning_rate;
                    *learning_rate *= self.config.lr_reduction_factor;
                    info!(
                        "Reduced learning rate from {} to {}",
                        old_lr, *learning_rate
                    );
                    Ok(true)
                }

                RecoveryStrategy::GradientClipping => {
                    let mut clipped_count = 0;
                    let threshold = T::from_f32(self.config.gradient_clip_threshold).unwrap();

                    for (param_name, grad_vec) in gradients.iter_mut() {
                        let mut max_norm = T::from_f32(0.0).unwrap();

                        // Calculate L2 norm
                        for &val in grad_vec.iter() {
                            let abs_val = val.abs();
                            if abs_val > max_norm {
                                max_norm = abs_val;
                            }
                        }

                        // Clip if necessary
                        if max_norm > threshold {
                            let scale_factor = threshold / max_norm;
                            for val in grad_vec.iter_mut() {
                                *val = *val * scale_factor;
                            }
                            clipped_count += 1;
                            debug!("Clipped gradients for parameter: {}", param_name);
                        }
                    }

                    info!("Gradient clipping applied to {} parameters", clipped_count);
                    Ok(clipped_count > 0)
                }

                RecoveryStrategy::ReinitializeParameters => {
                    // Reinitialize gradients to small random values
                    for grad_vec in gradients.values_mut() {
                        for val in grad_vec.iter_mut() {
                            // Use a simple random-like value based on current value
                            let random_factor = T::from_f32(0.01).unwrap();
                            *val = random_factor;
                        }
                    }
                    info!("Reinitialized {} parameter gradients", gradients.len());
                    Ok(true)
                }

                RecoveryStrategy::GradientScaling => {
                    let scale_factor = T::from_f32(0.5).unwrap();
                    for grad_vec in gradients.values_mut() {
                        for val in grad_vec.iter_mut() {
                            *val = *val * scale_factor;
                        }
                    }
                    info!("Applied gradient scaling with factor 0.5");
                    Ok(true)
                }

                RecoveryStrategy::SkipBatch => {
                    // Zero out gradients to effectively skip this batch
                    for grad_vec in gradients.values_mut() {
                        for val in grad_vec.iter_mut() {
                            *val = T::from_f32(0.0).unwrap();
                        }
                    }
                    info!("Skipped current batch by zeroing gradients");
                    Ok(true)
                }

                RecoveryStrategy::ApplyRegularization => {
                    // Add L2 regularization to gradients
                    let reg_factor = T::from_f32(0.01).unwrap();
                    for grad_vec in gradients.values_mut() {
                        for val in grad_vec.iter_mut() {
                            let reg_term = *val * reg_factor;
                            *val = *val + reg_term;
                        }
                    }
                    info!("Applied L2 regularization to gradients");
                    Ok(true)
                }

                RecoveryStrategy::ResetToCheckpoint => {
                    // This would require integration with checkpoint system
                    warn!("Reset to checkpoint strategy not yet implemented");
                    Ok(false)
                }

                RecoveryStrategy::RestartWithNewSeed => {
                    // This would require higher-level restart logic
                    warn!("Restart with new seed strategy requires higher-level coordination");
                    Ok(false)
                }
            }
        }

        /// Get recovery statistics
        pub fn get_recovery_stats(&self) -> RecoveryStats {
            let total_attempts = self.recovery_history.len();
            let successful_attempts = self
                .recovery_history
                .iter()
                .filter(|(_, _, _, success)| *success)
                .count();

            let mut strategy_counts = HashMap::new();
            for (_, strategy, _, _) in &self.recovery_history {
                *strategy_counts.entry(strategy.clone()).or_insert(0) += 1;
            }

            RecoveryStats {
                total_attempts,
                successful_attempts,
                success_rate: if total_attempts > 0 {
                    successful_attempts as f32 / total_attempts as f32
                } else {
                    0.0
                },
                strategy_usage: strategy_counts,
                current_anomaly_counts: self.recovery_attempts.clone(),
            }
        }

        /// Clear recovery history
        pub fn clear_history(&mut self) {
            self.recovery_history.clear();
            self.recovery_attempts.clear();
            info!("Cleared anomaly recovery history");
        }
    }

    /// Statistics about recovery attempts
    #[derive(Debug)]
    pub struct RecoveryStats {
        /// Total number of recovery attempts
        pub total_attempts: usize,
        /// Number of successful recovery attempts
        pub successful_attempts: usize,
        /// Overall success rate
        pub success_rate: f32,
        /// Usage count for each strategy
        pub strategy_usage: HashMap<RecoveryStrategy, usize>,
        /// Current attempt counts for each anomaly type
        pub current_anomaly_counts: HashMap<String, usize>,
    }

    impl RecoveryStats {
        /// Print a summary of recovery statistics
        pub fn print_summary(&self) {
            println!("\n=== Anomaly Recovery Statistics ===");
            println!("Total attempts: {}", self.total_attempts);
            println!("Successful attempts: {}", self.successful_attempts);
            println!("Success rate: {:.2}%", self.success_rate * 100.0);

            if !self.strategy_usage.is_empty() {
                println!("\nStrategy usage:");
                for (strategy, count) in &self.strategy_usage {
                    println!("  {:?}: {}", strategy, count);
                }
            }

            if !self.current_anomaly_counts.is_empty() {
                println!("\nCurrent anomaly attempt counts:");
                for (anomaly_type, count) in &self.current_anomaly_counts {
                    println!("  {}: {}", anomaly_type, count);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd_traits::AutogradTensor;

    use torsh_core::shape::Shape;

    // Mock implementation for testing
    struct MockComplexTensor {
        data: Vec<Complex<f32>>,
        shape: Shape,
    }

    impl AutogradTensor<Complex<f32>> for MockComplexTensor {
        fn shape(&self) -> Shape {
            self.shape.clone()
        }

        fn requires_grad(&self) -> bool {
            false
        }

        fn data(&self) -> Box<dyn std::ops::Deref<Target = [Complex<f32>]> + '_> {
            Box::new(self.data.as_slice())
        }

        fn clone_tensor(&self) -> Box<dyn AutogradTensor<Complex<f32>>> {
            Box::new(MockComplexTensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
            })
        }

        fn to_vec(&self) -> Vec<Complex<f32>> {
            self.data.clone()
        }

        fn device(&self) -> &dyn torsh_core::Device {
            use std::sync::LazyLock;
            static CPU_DEVICE: LazyLock<torsh_core::device::CpuDevice> =
                LazyLock::new(|| torsh_core::device::CpuDevice::new());
            &*CPU_DEVICE
        }

        fn ones_like(&self) -> Box<dyn AutogradTensor<Complex<f32>>> {
            let ones_data = vec![Complex::new(1.0f32, 0.0f32); self.data.len()];
            Box::new(MockComplexTensor {
                data: ones_data,
                shape: self.shape.clone(),
            })
        }

        fn zeros_like(&self) -> Box<dyn AutogradTensor<Complex<f32>>> {
            let zeros_data = vec![Complex::new(0.0f32, 0.0f32); self.data.len()];
            Box::new(MockComplexTensor {
                data: zeros_data,
                shape: self.shape.clone(),
            })
        }
    }

    #[test]
    fn test_complex_anomaly_detection_nan() {
        let data = vec![
            Complex::new(1.0f32, 2.0f32),
            Complex::new(f32::NAN, 0.0f32),
            Complex::new(0.0f32, f32::NAN),
        ];
        let tensor = MockComplexTensor {
            data,
            shape: Shape::new(vec![3]),
        };

        let anomalies = detect_complex_anomalies(&tensor).unwrap();
        assert_eq!(anomalies.len(), 2);
        assert!(anomalies[0].contains("NaN detected at index 1"));
        assert!(anomalies[1].contains("NaN detected at index 2"));
    }

    #[test]
    fn test_complex_anomaly_detection_infinity() {
        let data = vec![
            Complex::new(1.0f32, 2.0f32),
            Complex::new(f32::INFINITY, 0.0f32),
            Complex::new(0.0f32, f32::NEG_INFINITY),
        ];
        let tensor = MockComplexTensor {
            data,
            shape: Shape::new(vec![3]),
        };

        let anomalies = detect_complex_anomalies(&tensor).unwrap();
        assert_eq!(anomalies.len(), 4); // 2 infinity detections + 2 large magnitude detections
        assert!(anomalies
            .iter()
            .any(|a| a.contains("Infinity detected at index 1")));
        assert!(anomalies
            .iter()
            .any(|a| a.contains("Infinity detected at index 2")));
        assert!(anomalies
            .iter()
            .any(|a| a.contains("Large magnitude") && a.contains("index 1")));
        assert!(anomalies
            .iter()
            .any(|a| a.contains("Large magnitude") && a.contains("index 2")));
    }

    #[test]
    fn test_complex_anomaly_detection_large_magnitude() {
        let data = vec![
            Complex::new(1.0f32, 2.0f32),
            Complex::new(1e7f32, 0.0f32), // Very large magnitude
        ];
        let tensor = MockComplexTensor {
            data,
            shape: Shape::new(vec![2]),
        };

        let anomalies = detect_complex_anomalies(&tensor).unwrap();
        assert_eq!(anomalies.len(), 1);
        assert!(anomalies[0].contains("Large magnitude"));
        assert!(anomalies[0].contains("index 1"));
    }

    #[test]
    fn test_recovery_config_defaults() {
        let config = recovery::RecoveryConfig::default();
        assert!(config.enable_auto_recovery);
        assert_eq!(config.max_recovery_attempts, 3);
        assert_eq!(config.lr_reduction_factor, 0.5);
        assert_eq!(config.gradient_clip_threshold, 1.0);
        assert_eq!(config.loss_history_size, 10);
    }

    #[test]
    fn test_recovery_system_creation() {
        let config = recovery::RecoveryConfig::default();
        let system = recovery::AnomalyRecoverySystem::new(config);

        let stats = system.get_recovery_stats();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful_attempts, 0);
        assert_eq!(stats.success_rate, 0.0);
    }

    #[test]
    fn test_recovery_strategy_selection() {
        let config = recovery::RecoveryConfig::default();
        let system = recovery::AnomalyRecoverySystem::new(config);

        // Test NaN strategy selection
        let strategy = system.select_recovery_strategy("nan", 1);
        assert_eq!(strategy, recovery::RecoveryStrategy::GradientClipping);

        let strategy = system.select_recovery_strategy("nan", 2);
        assert_eq!(strategy, recovery::RecoveryStrategy::ReduceLearningRate);

        // Test gradient explosion strategy
        let strategy = system.select_recovery_strategy("gradient_explosion", 1);
        assert_eq!(strategy, recovery::RecoveryStrategy::GradientClipping);
    }
}
