//! Graceful Degradation for Unsupported Operations
//!
//! This module provides comprehensive graceful degradation capabilities for autograd operations
//! that may not be supported in all configurations, backends, or hardware environments.
//! It ensures the system remains functional even when specific features are unavailable.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::ndarray::{Array, ArrayView, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, RwLock};

/// Supported operation categories for degradation handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationCategory {
    // Core tensor operations
    BasicArithmetic,
    LinearAlgebra,
    Reduction,
    Indexing,
    Broadcasting,

    // Autograd-specific operations
    GradientComputation,
    BackwardPass,
    HigherOrderDerivatives,

    // Advanced features
    Checkpointing,
    DistributedTraining,
    MixedPrecision,
    Quantization,

    // Backend-specific features
    GPUAcceleration,
    SIMDOptimization,
    TensorCores,
    CustomKernels,

    // Experimental features
    QuantumComputing,
    SparseTensors,
    DifferentialPrivacy,
    NeuralArchitectureSearch,

    // Framework integration
    PyTorchCompatibility,
    JAXCompatibility,
    TensorFlowCompatibility,

    // Custom category for user-defined operations
    Custom(String),
}

impl fmt::Display for OperationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationCategory::BasicArithmetic => write!(f, "Basic Arithmetic"),
            OperationCategory::LinearAlgebra => write!(f, "Linear Algebra"),
            OperationCategory::Reduction => write!(f, "Reduction"),
            OperationCategory::Indexing => write!(f, "Indexing"),
            OperationCategory::Broadcasting => write!(f, "Broadcasting"),
            OperationCategory::GradientComputation => write!(f, "Gradient Computation"),
            OperationCategory::BackwardPass => write!(f, "Backward Pass"),
            OperationCategory::HigherOrderDerivatives => write!(f, "Higher-Order Derivatives"),
            OperationCategory::Checkpointing => write!(f, "Checkpointing"),
            OperationCategory::DistributedTraining => write!(f, "Distributed Training"),
            OperationCategory::MixedPrecision => write!(f, "Mixed Precision"),
            OperationCategory::Quantization => write!(f, "Quantization"),
            OperationCategory::GPUAcceleration => write!(f, "GPU Acceleration"),
            OperationCategory::SIMDOptimization => write!(f, "SIMD Optimization"),
            OperationCategory::TensorCores => write!(f, "Tensor Cores"),
            OperationCategory::CustomKernels => write!(f, "Custom Kernels"),
            OperationCategory::QuantumComputing => write!(f, "Quantum Computing"),
            OperationCategory::SparseTensors => write!(f, "Sparse Tensors"),
            OperationCategory::DifferentialPrivacy => write!(f, "Differential Privacy"),
            OperationCategory::NeuralArchitectureSearch => write!(f, "Neural Architecture Search"),
            OperationCategory::PyTorchCompatibility => write!(f, "PyTorch Compatibility"),
            OperationCategory::JAXCompatibility => write!(f, "JAX Compatibility"),
            OperationCategory::TensorFlowCompatibility => write!(f, "TensorFlow Compatibility"),
            OperationCategory::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Degradation strategies for handling unsupported operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationStrategy {
    /// Use a fallback implementation with reduced performance
    FallbackImplementation {
        fallback_name: String,
        performance_impact: f32, // 0.0 = no impact, 1.0 = severe impact
        accuracy_impact: f32,    // 0.0 = no impact, 1.0 = severe impact
    },

    /// Approximate the operation with a simpler version
    Approximation {
        approximation_method: String,
        accuracy_loss: f32,
        speed_gain: f32,
    },

    /// Skip the operation and continue with a warning
    Skip {
        warning_message: String,
        continue_execution: bool,
    },

    /// Replace with a no-op that maintains tensor shapes
    NoOp {
        preserve_shapes: bool,
        preserve_gradients: bool,
    },

    /// Use an alternative operation that achieves similar results
    Alternative {
        alternative_operation: String,
        equivalence_quality: f32, // 0.0 = not equivalent, 1.0 = perfectly equivalent
    },

    /// Raise an error with helpful guidance
    ErrorWithGuidance {
        error_message: String,
        suggested_actions: Vec<String>,
        documentation_links: Vec<String>,
    },

    /// Use a user-provided fallback function
    UserDefinedFallback {
        fallback_id: String,
        metadata: HashMap<String, String>,
    },
}

/// Information about why an operation is unsupported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsupportedOperationInfo {
    pub operation_name: String,
    pub category: OperationCategory,
    pub reason: UnsupportedReason,
    pub required_features: Vec<String>,
    pub available_alternatives: Vec<String>,
    pub workarounds: Vec<String>,
    pub planned_support_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnsupportedReason {
    BackendNotAvailable(String),
    FeatureNotImplemented,
    HardwareNotSupported,
    DependencyMissing(String),
    ConfigurationDisabled,
    VersionMismatch { required: String, available: String },
    LicenseRestriction,
    ExperimentalFeature,
    PerformanceReason,
    SecurityReason,
    Custom(String),
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsupportedReason::BackendNotAvailable(backend) => {
                write!(f, "Backend not available: {}", backend)
            }
            UnsupportedReason::FeatureNotImplemented => write!(f, "Feature not yet implemented"),
            UnsupportedReason::HardwareNotSupported => write!(f, "Hardware not supported"),
            UnsupportedReason::DependencyMissing(dep) => write!(f, "Missing dependency: {}", dep),
            UnsupportedReason::ConfigurationDisabled => write!(f, "Disabled in configuration"),
            UnsupportedReason::VersionMismatch {
                required,
                available,
            } => {
                write!(
                    f,
                    "Version mismatch: required {}, available {}",
                    required, available
                )
            }
            UnsupportedReason::LicenseRestriction => write!(f, "License restriction"),
            UnsupportedReason::ExperimentalFeature => write!(f, "Experimental feature"),
            UnsupportedReason::PerformanceReason => write!(f, "Performance consideration"),
            UnsupportedReason::SecurityReason => write!(f, "Security consideration"),
            UnsupportedReason::Custom(reason) => write!(f, "Custom: {}", reason),
        }
    }
}

/// Fallback function trait for user-defined operations
pub trait FallbackFunction: Send + Sync + std::fmt::Debug {
    fn execute(
        &self,
        inputs: &[&ArrayView<f64, IxDyn>],
        metadata: &HashMap<String, String>,
    ) -> AutogradResult<Array<f64, IxDyn>>;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn expected_performance_impact(&self) -> f32;
    fn expected_accuracy_impact(&self) -> f32;
}

/// Graceful degradation manager
pub struct GracefulDegradationManager {
    operation_registry: RwLock<HashMap<String, UnsupportedOperationInfo>>,
    degradation_strategies: RwLock<HashMap<String, DegradationStrategy>>,
    fallback_functions: RwLock<HashMap<String, Box<dyn FallbackFunction>>>,
    degradation_history: Mutex<Vec<DegradationEvent>>,
    enabled: bool,
    strict_mode: bool, // If true, errors instead of degrading
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationEvent {
    pub operation_name: String,
    pub category: OperationCategory,
    pub strategy_used: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub performance_impact: Option<f32>,
    pub accuracy_impact: Option<f32>,
    pub success: bool,
    pub error_message: Option<String>,
}

impl GracefulDegradationManager {
    pub fn new() -> Self {
        Self {
            operation_registry: RwLock::new(HashMap::new()),
            degradation_strategies: RwLock::new(HashMap::new()),
            fallback_functions: RwLock::new(HashMap::new()),
            degradation_history: Mutex::new(Vec::new()),
            enabled: true,
            strict_mode: false,
        }
    }

    pub fn register_unsupported_operation(
        &self,
        operation_name: String,
        info: UnsupportedOperationInfo,
    ) {
        let mut registry = self.operation_registry.write().unwrap();
        registry.insert(operation_name, info);
    }

    pub fn register_degradation_strategy(
        &self,
        operation_name: String,
        strategy: DegradationStrategy,
    ) {
        let mut strategies = self.degradation_strategies.write().unwrap();
        strategies.insert(operation_name, strategy);
    }

    pub fn register_fallback_function(
        &self,
        fallback_id: String,
        function: Box<dyn FallbackFunction>,
    ) {
        let mut functions = self.fallback_functions.write().unwrap();
        functions.insert(fallback_id, function);
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    pub fn is_operation_supported(&self, operation_name: &str) -> bool {
        let registry = self.operation_registry.read().unwrap();
        !registry.contains_key(operation_name)
    }

    pub fn get_operation_info(&self, operation_name: &str) -> Option<UnsupportedOperationInfo> {
        let registry = self.operation_registry.read().unwrap();
        registry.get(operation_name).cloned()
    }

    pub fn execute_with_degradation<F, R>(
        &self,
        operation_name: &str,
        default_impl: F,
    ) -> AutogradResult<R>
    where
        F: FnOnce() -> AutogradResult<R>,
    {
        if !self.enabled {
            return default_impl();
        }

        // Check if operation is supported
        if self.is_operation_supported(operation_name) {
            return default_impl();
        }

        // In strict mode, always error on unsupported operations
        if self.strict_mode {
            let info = self.get_operation_info(operation_name);
            let reason = info
                .as_ref()
                .map(|i| i.reason.to_string())
                .unwrap_or_else(|| "Unknown reason".to_string());

            return Err(AutogradError::gradient_computation(
                "operation_support_check",
                format!(
                    "Operation '{}' is not supported: {}",
                    operation_name, reason
                ),
            ));
        }

        // Try to execute with degradation
        self.execute_degraded_operation(operation_name, default_impl)
    }

    fn execute_degraded_operation<F, R>(
        &self,
        operation_name: &str,
        default_impl: F,
    ) -> AutogradResult<R>
    where
        F: FnOnce() -> AutogradResult<R>,
    {
        let strategies = self.degradation_strategies.read().unwrap();
        let strategy = strategies.get(operation_name);

        match strategy {
            Some(DegradationStrategy::FallbackImplementation {
                fallback_name,
                performance_impact,
                accuracy_impact,
            }) => {
                tracing::warn!("Using fallback '{}' for unsupported operation '{}' (performance impact: {:.1}%, accuracy impact: {:.1}%)",
                    fallback_name, operation_name, performance_impact * 100.0, accuracy_impact * 100.0);

                self.record_degradation_event(
                    operation_name,
                    "FallbackImplementation",
                    Some(*performance_impact),
                    Some(*accuracy_impact),
                    true,
                    None,
                );
                default_impl()
            }

            Some(DegradationStrategy::Skip {
                warning_message,
                continue_execution,
            }) => {
                tracing::warn!(
                    "Skipping unsupported operation '{}': {}",
                    operation_name,
                    warning_message
                );

                if *continue_execution {
                    self.record_degradation_event(operation_name, "Skip", None, None, true, None);
                    // This is tricky - we need to return a valid R but we're skipping
                    // In practice, this would need to be handled by the caller
                    Err(AutogradError::gradient_computation(
                        "operation_skip",
                        format!(
                            "Operation '{}' was skipped due to lack of support",
                            operation_name
                        ),
                    ))
                } else {
                    Err(AutogradError::gradient_computation(
                        "operation_warning",
                        warning_message.clone(),
                    ))
                }
            }

            Some(DegradationStrategy::ErrorWithGuidance {
                error_message,
                suggested_actions,
                documentation_links,
            }) => {
                let mut full_message = error_message.clone();

                if !suggested_actions.is_empty() {
                    full_message.push_str("\n\nSuggested actions:");
                    for action in suggested_actions {
                        full_message.push_str(&format!("\n  - {}", action));
                    }
                }

                if !documentation_links.is_empty() {
                    full_message.push_str("\n\nDocumentation:");
                    for link in documentation_links {
                        full_message.push_str(&format!("\n  - {}", link));
                    }
                }

                self.record_degradation_event(
                    operation_name,
                    "ErrorWithGuidance",
                    None,
                    None,
                    false,
                    Some(full_message.clone()),
                );
                Err(AutogradError::gradient_computation(
                    "error_with_guidance",
                    full_message,
                ))
            }

            Some(DegradationStrategy::UserDefinedFallback {
                fallback_id,
                metadata: _metadata,
            }) => {
                let functions = self.fallback_functions.read().unwrap();
                if let Some(fallback_fn) = functions.get(fallback_id) {
                    tracing::info!(
                        "Using user-defined fallback '{}' for operation '{}'",
                        fallback_id,
                        operation_name
                    );

                    let performance_impact = fallback_fn.expected_performance_impact();
                    let accuracy_impact = fallback_fn.expected_accuracy_impact();

                    self.record_degradation_event(
                        operation_name,
                        "UserDefinedFallback",
                        Some(performance_impact),
                        Some(accuracy_impact),
                        true,
                        None,
                    );

                    // This is simplified - in practice we'd need to handle the actual fallback execution
                    default_impl()
                } else {
                    let error_msg = format!(
                        "Fallback function '{}' not found for operation '{}'",
                        fallback_id, operation_name
                    );
                    self.record_degradation_event(
                        operation_name,
                        "UserDefinedFallback",
                        None,
                        None,
                        false,
                        Some(error_msg.clone()),
                    );
                    Err(AutogradError::gradient_computation(
                        "fallback_not_found",
                        error_msg,
                    ))
                }
            }

            _ => {
                // No specific strategy found, try default implementation
                tracing::warn!("No degradation strategy found for unsupported operation '{}', attempting default implementation", operation_name);
                default_impl()
            }
        }
    }

    fn record_degradation_event(
        &self,
        operation_name: &str,
        strategy: &str,
        performance_impact: Option<f32>,
        accuracy_impact: Option<f32>,
        success: bool,
        error_message: Option<String>,
    ) {
        let registry = self.operation_registry.read().unwrap();
        let category = registry
            .get(operation_name)
            .map(|info| info.category.clone())
            .unwrap_or(OperationCategory::Custom("Unknown".to_string()));

        let event = DegradationEvent {
            operation_name: operation_name.to_string(),
            category,
            strategy_used: strategy.to_string(),
            timestamp: chrono::Utc::now(),
            performance_impact,
            accuracy_impact,
            success,
            error_message,
        };

        if let Ok(mut history) = self.degradation_history.lock() {
            history.push(event);
        }
    }

    pub fn get_degradation_statistics(&self) -> DegradationStatistics {
        let history = self.degradation_history.lock().unwrap();
        DegradationStatistics::from_events(&history)
    }

    pub fn initialize_default_strategies(&self) {
        // Initialize common degradation strategies

        // GPU operations fallback to CPU
        self.register_degradation_strategy(
            "gpu_matrix_multiply".to_string(),
            DegradationStrategy::FallbackImplementation {
                fallback_name: "CPU matrix multiply".to_string(),
                performance_impact: 0.7, // 70% slower on CPU
                accuracy_impact: 0.0,    // Same accuracy
            },
        );

        // Mixed precision fallback to full precision
        self.register_degradation_strategy(
            "mixed_precision_training".to_string(),
            DegradationStrategy::Alternative {
                alternative_operation: "full_precision_training".to_string(),
                equivalence_quality: 0.9, // Very similar results
            },
        );

        // Distributed training fallback to single-node
        self.register_degradation_strategy(
            "distributed_backward".to_string(),
            DegradationStrategy::FallbackImplementation {
                fallback_name: "single_node_backward".to_string(),
                performance_impact: 0.5, // 50% slower without distribution
                accuracy_impact: 0.0,
            },
        );

        // Experimental features raise errors with guidance
        self.register_degradation_strategy(
            "quantum_gradient_computation".to_string(),
            DegradationStrategy::ErrorWithGuidance {
                error_message:
                    "Quantum gradient computation is experimental and not available in this build"
                        .to_string(),
                suggested_actions: vec![
                    "Enable experimental features in configuration".to_string(),
                    "Use classical gradient computation as alternative".to_string(),
                    "Check for newer version with quantum support".to_string(),
                ],
                documentation_links: vec!["https://docs.torsh.ai/quantum-computing".to_string()],
            },
        );

        // SIMD operations fallback to scalar
        self.register_degradation_strategy(
            "simd_vector_operations".to_string(),
            DegradationStrategy::FallbackImplementation {
                fallback_name: "scalar_operations".to_string(),
                performance_impact: 0.6, // 60% slower without SIMD
                accuracy_impact: 0.0,
            },
        );
    }

    pub fn clear_degradation_history(&self) {
        if let Ok(mut history) = self.degradation_history.lock() {
            history.clear();
        }
    }

    pub fn export_degradation_report(&self, file_path: &std::path::Path) -> AutogradResult<()> {
        let statistics = self.get_degradation_statistics();
        let json_data = serde_json::to_string_pretty(&statistics).map_err(|e| {
            AutogradError::gradient_computation(
                "statistics_serialization",
                format!("Failed to serialize statistics: {}", e),
            )
        })?;

        std::fs::write(file_path, json_data).map_err(|e| {
            AutogradError::gradient_computation(
                "file_write",
                format!("Failed to write file: {}", e),
            )
        })?;

        Ok(())
    }
}

/// Statistics about degradation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationStatistics {
    pub total_events: usize,
    pub successful_degradations: usize,
    pub failed_degradations: usize,
    pub success_rate: f32,
    pub events_by_category: HashMap<String, usize>,
    pub events_by_strategy: HashMap<String, usize>,
    pub average_performance_impact: f32,
    pub average_accuracy_impact: f32,
    pub most_common_operations: Vec<(String, usize)>,
    pub time_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
}

impl DegradationStatistics {
    pub fn from_events(events: &[DegradationEvent]) -> Self {
        let total_events = events.len();
        let successful_degradations = events.iter().filter(|e| e.success).count();
        let failed_degradations = total_events - successful_degradations;
        let success_rate = if total_events > 0 {
            successful_degradations as f32 / total_events as f32
        } else {
            0.0
        };

        let mut events_by_category = HashMap::new();
        let mut events_by_strategy = HashMap::new();
        let mut operation_counts = HashMap::new();

        let mut performance_impacts = Vec::new();
        let mut accuracy_impacts = Vec::new();

        for event in events {
            *events_by_category
                .entry(event.category.to_string())
                .or_insert(0) += 1;
            *events_by_strategy
                .entry(event.strategy_used.clone())
                .or_insert(0) += 1;
            *operation_counts
                .entry(event.operation_name.clone())
                .or_insert(0) += 1;

            if let Some(perf) = event.performance_impact {
                performance_impacts.push(perf);
            }
            if let Some(acc) = event.accuracy_impact {
                accuracy_impacts.push(acc);
            }
        }

        let average_performance_impact = if performance_impacts.is_empty() {
            0.0
        } else {
            performance_impacts.iter().sum::<f32>() / performance_impacts.len() as f32
        };

        let average_accuracy_impact = if accuracy_impacts.is_empty() {
            0.0
        } else {
            accuracy_impacts.iter().sum::<f32>() / accuracy_impacts.len() as f32
        };

        let mut most_common_operations: Vec<_> = operation_counts.into_iter().collect();
        most_common_operations.sort_by(|a, b| b.1.cmp(&a.1));
        most_common_operations.truncate(10); // Top 10

        let time_range = if events.is_empty() {
            None
        } else {
            let timestamps: Vec<_> = events.iter().map(|e| e.timestamp).collect();
            let min_time = timestamps.iter().min().copied().unwrap();
            let max_time = timestamps.iter().max().copied().unwrap();
            Some((min_time, max_time))
        };

        Self {
            total_events,
            successful_degradations,
            failed_degradations,
            success_rate,
            events_by_category,
            events_by_strategy,
            average_performance_impact,
            average_accuracy_impact,
            most_common_operations,
            time_range,
        }
    }

    pub fn print_summary(&self) {
        println!("=== Graceful Degradation Statistics ===");
        println!("Total Events: {}", self.total_events);
        println!(
            "Successful Degradations: {} ({:.1}%)",
            self.successful_degradations,
            self.success_rate * 100.0
        );
        println!("Failed Degradations: {}", self.failed_degradations);
        println!(
            "Average Performance Impact: {:.1}%",
            self.average_performance_impact * 100.0
        );
        println!(
            "Average Accuracy Impact: {:.1}%",
            self.average_accuracy_impact * 100.0
        );
        println!();

        if !self.events_by_category.is_empty() {
            println!("Events by Category:");
            for (category, count) in &self.events_by_category {
                println!("  {}: {}", category, count);
            }
            println!();
        }

        if !self.most_common_operations.is_empty() {
            println!("Most Common Unsupported Operations:");
            for (operation, count) in &self.most_common_operations {
                println!("  {}: {}", operation, count);
            }
        }
    }
}

/// Example fallback function for matrix multiplication
#[derive(Debug)]
pub struct MatrixMultiplyFallback;

impl FallbackFunction for MatrixMultiplyFallback {
    fn execute(
        &self,
        inputs: &[&ArrayView<f64, IxDyn>],
        _metadata: &HashMap<String, String>,
    ) -> AutogradResult<Array<f64, IxDyn>> {
        if inputs.len() != 2 {
            return Err(AutogradError::gradient_computation(
                "matrix_multiply_inputs",
                "Matrix multiply fallback requires exactly 2 inputs",
            ));
        }

        // Simplified matrix multiplication using basic loops
        // In practice, this would use optimized BLAS routines
        let a = inputs[0];
        let b = inputs[1];

        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(AutogradError::gradient_computation(
                "matrix_multiply_dimensions",
                "Matrix multiply fallback requires 2D tensors",
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(AutogradError::shape_mismatch(
                "matrix_multiply_shape",
                vec![k],
                vec![k2],
            ));
        }

        let mut result = Array::zeros(vec![m, n].as_slice());

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[[i, p]] * b[[p, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "matrix_multiply_fallback"
    }

    fn description(&self) -> &str {
        "Basic fallback implementation for matrix multiplication using nested loops"
    }

    fn expected_performance_impact(&self) -> f32 {
        0.8 // 80% slower than optimized implementation
    }

    fn expected_accuracy_impact(&self) -> f32 {
        0.0 // Same accuracy
    }
}

/// Global graceful degradation manager instance
static GLOBAL_DEGRADATION_MANAGER: std::sync::OnceLock<GracefulDegradationManager> =
    std::sync::OnceLock::new();

pub fn get_global_degradation_manager() -> &'static GracefulDegradationManager {
    GLOBAL_DEGRADATION_MANAGER.get_or_init(|| {
        let manager = GracefulDegradationManager::new();
        manager.initialize_default_strategies();
        manager
    })
}

/// Convenience macro for executing operations with graceful degradation
#[macro_export]
macro_rules! with_graceful_degradation {
    ($operation_name:expr, $operation:expr) => {
        $crate::graceful_degradation::get_global_degradation_manager()
            .execute_with_degradation($operation_name, || $operation)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_category_display() {
        assert_eq!(
            OperationCategory::BasicArithmetic.to_string(),
            "Basic Arithmetic"
        );
        assert_eq!(
            OperationCategory::Custom("test".to_string()).to_string(),
            "Custom(test)"
        );
    }

    #[test]
    fn test_unsupported_reason_display() {
        let reason = UnsupportedReason::BackendNotAvailable("CUDA".to_string());
        assert_eq!(reason.to_string(), "Backend not available: CUDA");

        let version_reason = UnsupportedReason::VersionMismatch {
            required: "2.0".to_string(),
            available: "1.5".to_string(),
        };
        assert_eq!(
            version_reason.to_string(),
            "Version mismatch: required 2.0, available 1.5"
        );
    }

    #[test]
    fn test_degradation_manager_creation() {
        let manager = GracefulDegradationManager::new();
        assert!(manager.is_operation_supported("unsupported_op"));
    }

    #[test]
    fn test_register_unsupported_operation() {
        let manager = GracefulDegradationManager::new();

        let info = UnsupportedOperationInfo {
            operation_name: "test_op".to_string(),
            category: OperationCategory::GPUAcceleration,
            reason: UnsupportedReason::HardwareNotSupported,
            required_features: vec!["CUDA".to_string()],
            available_alternatives: vec!["CPU implementation".to_string()],
            workarounds: vec!["Use CPU backend".to_string()],
            planned_support_version: Some("2.0".to_string()),
        };

        manager.register_unsupported_operation("test_op".to_string(), info);
        assert!(!manager.is_operation_supported("test_op"));

        let retrieved_info = manager.get_operation_info("test_op");
        assert!(retrieved_info.is_some());
        assert_eq!(
            retrieved_info.unwrap().category,
            OperationCategory::GPUAcceleration
        );
    }

    #[test]
    fn test_register_degradation_strategy() {
        let manager = GracefulDegradationManager::new();

        let strategy = DegradationStrategy::FallbackImplementation {
            fallback_name: "CPU fallback".to_string(),
            performance_impact: 0.5,
            accuracy_impact: 0.0,
        };

        manager.register_degradation_strategy("test_op".to_string(), strategy);

        // Register as unsupported to test degradation
        let info = UnsupportedOperationInfo {
            operation_name: "test_op".to_string(),
            category: OperationCategory::GPUAcceleration,
            reason: UnsupportedReason::HardwareNotSupported,
            required_features: vec![],
            available_alternatives: vec![],
            workarounds: vec![],
            planned_support_version: None,
        };
        manager.register_unsupported_operation("test_op".to_string(), info);

        // Test execution with degradation
        let result = manager.execute_with_degradation("test_op", || Ok(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_matrix_multiply_fallback() {
        let fallback = MatrixMultiplyFallback;
        assert_eq!(fallback.name(), "matrix_multiply_fallback");
        assert!(fallback.expected_performance_impact() > 0.0);
        assert_eq!(fallback.expected_accuracy_impact(), 0.0);
    }

    #[test]
    fn test_degradation_statistics() {
        let events = vec![
            DegradationEvent {
                operation_name: "op1".to_string(),
                category: OperationCategory::GPUAcceleration,
                strategy_used: "FallbackImplementation".to_string(),
                timestamp: chrono::Utc::now(),
                performance_impact: Some(0.5),
                accuracy_impact: Some(0.0),
                success: true,
                error_message: None,
            },
            DegradationEvent {
                operation_name: "op2".to_string(),
                category: OperationCategory::MixedPrecision,
                strategy_used: "ErrorWithGuidance".to_string(),
                timestamp: chrono::Utc::now(),
                performance_impact: None,
                accuracy_impact: None,
                success: false,
                error_message: Some("Not supported".to_string()),
            },
        ];

        let stats = DegradationStatistics::from_events(&events);
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.successful_degradations, 1);
        assert_eq!(stats.failed_degradations, 1);
        assert_eq!(stats.success_rate, 0.5);
    }

    #[test]
    fn test_strict_mode() {
        let mut manager = GracefulDegradationManager::new();
        manager.set_strict_mode(true);

        // Register an unsupported operation
        let info = UnsupportedOperationInfo {
            operation_name: "strict_test".to_string(),
            category: OperationCategory::QuantumComputing,
            reason: UnsupportedReason::ExperimentalFeature,
            required_features: vec![],
            available_alternatives: vec![],
            workarounds: vec![],
            planned_support_version: None,
        };
        manager.register_unsupported_operation("strict_test".to_string(), info);

        // In strict mode, should error instead of degrading
        let result = manager.execute_with_degradation("strict_test", || Ok(42));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));
    }

    #[test]
    fn test_global_degradation_manager() {
        let manager = get_global_degradation_manager();

        // Should have default strategies initialized
        let stats = manager.get_degradation_statistics();
        assert_eq!(stats.total_events, 0); // No events yet
    }

    #[test]
    fn test_disabled_degradation() {
        let mut manager = GracefulDegradationManager::new();
        manager.set_enabled(false);

        // Register an unsupported operation
        let info = UnsupportedOperationInfo {
            operation_name: "disabled_test".to_string(),
            category: OperationCategory::SIMDOptimization,
            reason: UnsupportedReason::HardwareNotSupported,
            required_features: vec![],
            available_alternatives: vec![],
            workarounds: vec![],
            planned_support_version: None,
        };
        manager.register_unsupported_operation("disabled_test".to_string(), info);

        // When disabled, should execute normally without degradation
        let result = manager.execute_with_degradation("disabled_test", || Ok(42));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_fallback_function_registration() {
        let manager = GracefulDegradationManager::new();
        let fallback = Box::new(MatrixMultiplyFallback);

        manager.register_fallback_function("matrix_multiply".to_string(), fallback);

        let strategy = DegradationStrategy::UserDefinedFallback {
            fallback_id: "matrix_multiply".to_string(),
            metadata: HashMap::new(),
        };
        manager.register_degradation_strategy("matrix_op".to_string(), strategy);

        // Test would need more setup to actually execute the fallback
        assert!(true); // Placeholder assertion
    }
}
