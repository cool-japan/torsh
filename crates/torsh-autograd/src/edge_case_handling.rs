//! Robust edge case handling for autograd operations
//!
//! This module provides comprehensive handling of edge cases that can occur
//! during automatic differentiation, including empty tensors, degenerate shapes,
//! extreme values, and other boundary conditions that could cause failures.

use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Types of edge cases that can occur in autograd operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeCaseType {
    /// Empty tensors (zero elements)
    EmptyTensor,
    /// Zero-dimensional tensors (scalars)
    ZeroDimensionalTensor,
    /// Degenerate shapes (dimensions with zero size)
    DegenerateShape,
    /// Single-element tensors
    SingleElementTensor,
    /// Extremely large tensors
    OversizedTensor,
    /// Mismatched tensor shapes
    ShapeMismatch,
    /// Invalid data types
    InvalidDataType,
    /// Extreme numerical values (very large/small)
    ExtremeValues,
    /// NaN or infinite values
    NonFiniteValues,
    /// Uninitialized or corrupt data
    CorruptData,
    /// Memory allocation failures
    AllocationFailure,
    /// Custom edge cases
    Custom(String),
}

impl std::fmt::Display for EdgeCaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTensor => write!(f, "empty_tensor"),
            Self::ZeroDimensionalTensor => write!(f, "zero_dimensional_tensor"),
            Self::DegenerateShape => write!(f, "degenerate_shape"),
            Self::SingleElementTensor => write!(f, "single_element_tensor"),
            Self::OversizedTensor => write!(f, "oversized_tensor"),
            Self::ShapeMismatch => write!(f, "shape_mismatch"),
            Self::InvalidDataType => write!(f, "invalid_data_type"),
            Self::ExtremeValues => write!(f, "extreme_values"),
            Self::NonFiniteValues => write!(f, "non_finite_values"),
            Self::CorruptData => write!(f, "corrupt_data"),
            Self::AllocationFailure => write!(f, "allocation_failure"),
            Self::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Strategies for handling edge cases
#[derive(Debug, Clone)]
pub enum EdgeCaseStrategy {
    /// Return an error when edge case is encountered
    Error,
    /// Return a warning but continue with default behavior
    Warn,
    /// Silently handle the edge case with fallback behavior
    Silent,
    /// Use specific fallback value or behavior
    Fallback {
        fallback_value: Option<f64>,
        fallback_shape: Option<Vec<usize>>,
        fallback_behavior: String,
    },
    /// Transform the input to a valid form
    Transform {
        transformation: EdgeCaseTransformation,
    },
    /// Skip the operation entirely
    Skip,
    /// Use alternative algorithm or implementation
    AlternativeImplementation { algorithm_name: String },
}

/// Transformations that can be applied to handle edge cases
#[derive(Debug, Clone)]
pub enum EdgeCaseTransformation {
    /// Pad empty tensors to minimum size
    PadToMinimumSize { min_size: usize, fill_value: f64 },
    /// Reshape degenerate tensors
    ReshapeDegenerate { target_shape: Vec<usize> },
    /// Clamp extreme values to acceptable range
    ClampValues { min_value: f64, max_value: f64 },
    /// Replace non-finite values with safe alternatives
    ReplaceNonFinite {
        nan_replacement: f64,
        inf_replacement: f64,
    },
    /// Normalize tensor to unit scale
    Normalize { target_norm: f64 },
    /// Add small noise to break degeneracies
    AddNoise { noise_level: f64 },
    /// Broadcast to compatible shapes
    Broadcast { target_shape: Vec<usize> },
    /// Truncate oversized tensors
    Truncate { max_elements: usize },
}

/// Configuration for edge case handling
#[derive(Debug, Clone)]
pub struct EdgeCaseHandlingConfig {
    pub enabled: bool,
    pub default_strategy: EdgeCaseStrategy,
    pub specific_strategies: HashMap<EdgeCaseType, EdgeCaseStrategy>,
    pub max_tensor_elements: usize,
    pub min_tensor_elements: usize,
    pub value_clamp_threshold: f64,
    pub enable_warnings: bool,
    pub enable_statistics: bool,
}

impl Default for EdgeCaseHandlingConfig {
    fn default() -> Self {
        let mut specific_strategies = HashMap::new();

        // Default strategies for common edge cases
        specific_strategies.insert(
            EdgeCaseType::EmptyTensor,
            EdgeCaseStrategy::Fallback {
                fallback_value: Some(0.0),
                fallback_shape: Some(vec![1]),
                fallback_behavior: "create_single_zero_tensor".to_string(),
            },
        );

        specific_strategies.insert(
            EdgeCaseType::NonFiniteValues,
            EdgeCaseStrategy::Transform {
                transformation: EdgeCaseTransformation::ReplaceNonFinite {
                    nan_replacement: 0.0,
                    inf_replacement: 1e6,
                },
            },
        );

        specific_strategies.insert(
            EdgeCaseType::ExtremeValues,
            EdgeCaseStrategy::Transform {
                transformation: EdgeCaseTransformation::ClampValues {
                    min_value: -1e6,
                    max_value: 1e6,
                },
            },
        );

        specific_strategies.insert(
            EdgeCaseType::DegenerateShape,
            EdgeCaseStrategy::Transform {
                transformation: EdgeCaseTransformation::ReshapeDegenerate {
                    target_shape: vec![1],
                },
            },
        );

        Self {
            enabled: true,
            default_strategy: EdgeCaseStrategy::Warn,
            specific_strategies,
            max_tensor_elements: 1_000_000_000, // 1 billion elements
            min_tensor_elements: 1,
            value_clamp_threshold: 1e10,
            enable_warnings: true,
            enable_statistics: true,
        }
    }
}

/// Statistics for edge case handling
#[derive(Debug, Clone)]
pub struct EdgeCaseStatistics {
    pub total_edge_cases: usize,
    pub edge_cases_by_type: HashMap<EdgeCaseType, usize>,
    pub successful_handles: usize,
    pub failed_handles: usize,
    pub transformations_applied: HashMap<String, usize>,
    pub average_handling_time_ms: f64,
}

impl EdgeCaseStatistics {
    pub fn new() -> Self {
        Self {
            total_edge_cases: 0,
            edge_cases_by_type: HashMap::new(),
            successful_handles: 0,
            failed_handles: 0,
            transformations_applied: HashMap::new(),
            average_handling_time_ms: 0.0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_edge_cases == 0 {
            return 1.0;
        }
        self.successful_handles as f64 / self.total_edge_cases as f64
    }
}

impl Default for EdgeCaseStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor information for edge case analysis
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
}

impl TensorInfo {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        Self {
            shape,
            data,
            dtype: "f64".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.element_count() == 0 || self.data.is_empty()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty() || (self.shape.len() == 1 && self.shape[0] == 1)
    }

    pub fn has_degenerate_dimensions(&self) -> bool {
        self.shape.iter().any(|&dim| dim == 0)
    }

    pub fn has_non_finite_values(&self) -> bool {
        self.data.iter().any(|&x| !x.is_finite())
    }

    pub fn has_extreme_values(&self, threshold: f64) -> bool {
        self.data.iter().any(|&x| x.abs() > threshold)
    }

    pub fn min_value(&self) -> Option<f64> {
        self.data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }

    pub fn max_value(&self) -> Option<f64> {
        self.data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Result of edge case handling
#[derive(Debug, Clone)]
pub struct EdgeCaseHandlingResult {
    pub handled: bool,
    pub edge_case_type: EdgeCaseType,
    pub strategy_used: EdgeCaseStrategy,
    pub transformation_applied: Option<EdgeCaseTransformation>,
    pub original_tensor: TensorInfo,
    pub processed_tensor: Option<TensorInfo>,
    pub warnings: Vec<String>,
    pub handling_time_ms: f64,
}

/// Main edge case handler
pub struct EdgeCaseHandler {
    config: EdgeCaseHandlingConfig,
    statistics: Arc<RwLock<EdgeCaseStatistics>>,
}

impl EdgeCaseHandler {
    /// Create a new edge case handler
    pub fn new(config: EdgeCaseHandlingConfig) -> Self {
        Self {
            config,
            statistics: Arc::new(RwLock::new(EdgeCaseStatistics::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(EdgeCaseHandlingConfig::default())
    }

    /// Analyze tensor for edge cases
    pub fn analyze_tensor(&self, tensor: &TensorInfo) -> Vec<EdgeCaseType> {
        let mut edge_cases = Vec::new();

        // Check for empty tensor
        if tensor.is_empty() {
            edge_cases.push(EdgeCaseType::EmptyTensor);
        }

        // Check for zero-dimensional tensor
        if tensor.is_scalar() {
            edge_cases.push(EdgeCaseType::ZeroDimensionalTensor);
        }

        // Check for degenerate shape
        if tensor.has_degenerate_dimensions() {
            edge_cases.push(EdgeCaseType::DegenerateShape);
        }

        // Check for single element tensor
        if tensor.element_count() == 1 && !tensor.is_empty() {
            edge_cases.push(EdgeCaseType::SingleElementTensor);
        }

        // Check for oversized tensor
        if tensor.element_count() > self.config.max_tensor_elements {
            edge_cases.push(EdgeCaseType::OversizedTensor);
        }

        // Check for extreme values
        if tensor.has_extreme_values(self.config.value_clamp_threshold) {
            edge_cases.push(EdgeCaseType::ExtremeValues);
        }

        // Check for non-finite values
        if tensor.has_non_finite_values() {
            edge_cases.push(EdgeCaseType::NonFiniteValues);
        }

        edge_cases
    }

    /// Handle a specific edge case
    pub fn handle_edge_case(
        &self,
        tensor: &TensorInfo,
        edge_case_type: EdgeCaseType,
    ) -> AutogradResult<EdgeCaseHandlingResult> {
        let start_time = std::time::Instant::now();

        if !self.config.enabled {
            return Ok(EdgeCaseHandlingResult {
                handled: false,
                edge_case_type,
                strategy_used: EdgeCaseStrategy::Silent,
                transformation_applied: None,
                original_tensor: tensor.clone(),
                processed_tensor: None,
                warnings: vec!["Edge case handling is disabled".to_string()],
                handling_time_ms: 0.0,
            });
        }

        // Get strategy for this edge case type
        let strategy = self.get_strategy_for_edge_case(&edge_case_type);
        let mut warnings = Vec::new();
        let mut processed_tensor = None;
        let mut transformation_applied = None;

        let handled = match &strategy {
            EdgeCaseStrategy::Error => {
                return Err(AutogradError::shape_mismatch(
                    "edge_case_handler",
                    vec![],
                    vec![],
                ));
            }
            EdgeCaseStrategy::Warn => {
                let warning = format!(
                    "Edge case detected: {} in tensor with shape {:?}",
                    edge_case_type, tensor.shape
                );
                warnings.push(warning);
                if self.config.enable_warnings {
                    eprintln!("WARNING: {}", warnings.last().unwrap());
                }
                false
            }
            EdgeCaseStrategy::Silent => false,
            EdgeCaseStrategy::Fallback {
                fallback_value,
                fallback_shape,
                fallback_behavior,
            } => {
                processed_tensor = Some(self.create_fallback_tensor(
                    tensor,
                    fallback_value,
                    fallback_shape,
                    fallback_behavior,
                )?);
                warnings.push(format!("Applied fallback behavior: {}", fallback_behavior));
                true
            }
            EdgeCaseStrategy::Transform { transformation } => {
                processed_tensor = Some(self.apply_transformation(tensor, transformation)?);
                transformation_applied = Some(transformation.clone());
                warnings.push(format!("Applied transformation: {:?}", transformation));
                true
            }
            EdgeCaseStrategy::Skip => {
                warnings.push("Skipping operation due to edge case".to_string());
                false
            }
            EdgeCaseStrategy::AlternativeImplementation { algorithm_name } => {
                warnings.push(format!(
                    "Using alternative implementation: {}",
                    algorithm_name
                ));
                false
            }
        };

        let handling_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        // Update statistics
        self.update_statistics(&edge_case_type, handled, handling_time_ms);

        Ok(EdgeCaseHandlingResult {
            handled,
            edge_case_type,
            strategy_used: strategy,
            transformation_applied,
            original_tensor: tensor.clone(),
            processed_tensor,
            warnings,
            handling_time_ms,
        })
    }

    /// Handle multiple edge cases in a tensor
    pub fn handle_tensor(
        &self,
        tensor: &TensorInfo,
    ) -> AutogradResult<Vec<EdgeCaseHandlingResult>> {
        let edge_cases = self.analyze_tensor(tensor);
        let mut results = Vec::new();

        let mut current_tensor = tensor.clone();

        for edge_case in edge_cases {
            let result = self.handle_edge_case(&current_tensor, edge_case)?;

            // Update current tensor if it was processed
            if let Some(ref processed) = result.processed_tensor {
                current_tensor = processed.clone();
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Validate tensor shapes for compatibility
    pub fn validate_shapes(&self, shapes: &[Vec<usize>], operation: &str) -> AutogradResult<()> {
        if shapes.is_empty() {
            return Err(AutogradError::gradient_computation(
                operation,
                "No tensor shapes provided for validation",
            ));
        }

        // Check for empty shapes
        for (i, shape) in shapes.iter().enumerate() {
            if shape.is_empty() {
                let edge_case = EdgeCaseType::EmptyTensor;
                let strategy = self.get_strategy_for_edge_case(&edge_case);

                match strategy {
                    EdgeCaseStrategy::Error => {
                        return Err(AutogradError::shape_mismatch(
                            operation,
                            shapes[0].clone(),
                            shape.clone(),
                        ));
                    }
                    EdgeCaseStrategy::Warn => {
                        if self.config.enable_warnings {
                            eprintln!(
                                "WARNING: Empty shape at index {} in operation {}",
                                i, operation
                            );
                        }
                    }
                    _ => {} // Other strategies handled elsewhere
                }
            }

            // Check for degenerate dimensions
            if shape.iter().any(|&dim| dim == 0) {
                let edge_case = EdgeCaseType::DegenerateShape;
                let strategy = self.get_strategy_for_edge_case(&edge_case);

                match strategy {
                    EdgeCaseStrategy::Error => {
                        return Err(AutogradError::shape_mismatch(
                            operation,
                            shapes[0].clone(),
                            shape.clone(),
                        ));
                    }
                    EdgeCaseStrategy::Warn => {
                        if self.config.enable_warnings {
                            eprintln!(
                                "WARNING: Degenerate shape {:?} at index {} in operation {}",
                                shape, i, operation
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        // Check for shape compatibility (basic broadcasting rules)
        if shapes.len() > 1 {
            let reference_shape = &shapes[0];
            for (i, shape) in shapes.iter().enumerate().skip(1) {
                if !self.are_shapes_broadcastable(reference_shape, shape) {
                    return Err(AutogradError::shape_mismatch(
                        operation,
                        reference_shape.clone(),
                        shape.clone(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check if two shapes are broadcastable
    fn are_shapes_broadcastable(&self, shape1: &[usize], shape2: &[usize]) -> bool {
        let max_len = shape1.len().max(shape2.len());

        for i in 0..max_len {
            let dim1 = shape1
                .get(shape1.len().saturating_sub(max_len - i))
                .unwrap_or(&1);
            let dim2 = shape2
                .get(shape2.len().saturating_sub(max_len - i))
                .unwrap_or(&1);

            if *dim1 != *dim2 && *dim1 != 1 && *dim2 != 1 {
                return false;
            }
        }

        true
    }

    /// Get strategy for specific edge case type
    fn get_strategy_for_edge_case(&self, edge_case_type: &EdgeCaseType) -> EdgeCaseStrategy {
        self.config
            .specific_strategies
            .get(edge_case_type)
            .cloned()
            .unwrap_or_else(|| self.config.default_strategy.clone())
    }

    /// Create fallback tensor
    fn create_fallback_tensor(
        &self,
        _original: &TensorInfo,
        fallback_value: &Option<f64>,
        fallback_shape: &Option<Vec<usize>>,
        _fallback_behavior: &str,
    ) -> AutogradResult<TensorInfo> {
        let shape = fallback_shape.clone().unwrap_or_else(|| vec![1]);
        let element_count = shape.iter().product();
        let value = fallback_value.unwrap_or(0.0);
        let data = vec![value; element_count];

        Ok(TensorInfo::new(shape, data))
    }

    /// Apply transformation to tensor
    fn apply_transformation(
        &self,
        tensor: &TensorInfo,
        transformation: &EdgeCaseTransformation,
    ) -> AutogradResult<TensorInfo> {
        match transformation {
            EdgeCaseTransformation::PadToMinimumSize {
                min_size,
                fill_value,
            } => {
                let mut new_data = tensor.data.clone();
                while new_data.len() < *min_size {
                    new_data.push(*fill_value);
                }
                let new_shape = if tensor.shape.is_empty() {
                    vec![new_data.len()]
                } else {
                    let mut shape = tensor.shape.clone();
                    let last_idx = shape.len() - 1;
                    shape[last_idx] = new_data.len();
                    shape
                };
                Ok(TensorInfo::new(new_shape, new_data))
            }
            EdgeCaseTransformation::ReshapeDegenerate { target_shape } => {
                let element_count = target_shape.iter().product();
                let mut new_data = tensor.data.clone();
                new_data.resize(element_count, 0.0);
                Ok(TensorInfo::new(target_shape.clone(), new_data))
            }
            EdgeCaseTransformation::ClampValues {
                min_value,
                max_value,
            } => {
                let new_data: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| x.clamp(*min_value, *max_value))
                    .collect();
                Ok(TensorInfo::new(tensor.shape.clone(), new_data))
            }
            EdgeCaseTransformation::ReplaceNonFinite {
                nan_replacement,
                inf_replacement,
            } => {
                let new_data: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| {
                        if x.is_nan() {
                            *nan_replacement
                        } else if x.is_infinite() {
                            if x.is_sign_positive() {
                                *inf_replacement
                            } else {
                                -*inf_replacement
                            }
                        } else {
                            x
                        }
                    })
                    .collect();
                Ok(TensorInfo::new(tensor.shape.clone(), new_data))
            }
            EdgeCaseTransformation::Normalize { target_norm } => {
                let current_norm = tensor.norm();
                if current_norm == 0.0 {
                    return Ok(tensor.clone());
                }
                let scale = target_norm / current_norm;
                let new_data: Vec<f64> = tensor.data.iter().map(|&x| x * scale).collect();
                Ok(TensorInfo::new(tensor.shape.clone(), new_data))
            }
            EdgeCaseTransformation::AddNoise { noise_level } => {
                let new_data: Vec<f64> = tensor
                    .data
                    .iter()
                    .map(|&x| x + (*noise_level * random_normal()))
                    .collect();
                Ok(TensorInfo::new(tensor.shape.clone(), new_data))
            }
            EdgeCaseTransformation::Broadcast { target_shape } => {
                // Simplified broadcasting - just repeat data
                let target_elements = target_shape.iter().product();
                let mut new_data = Vec::with_capacity(target_elements);
                for i in 0..target_elements {
                    let src_idx = i % tensor.data.len();
                    new_data.push(tensor.data[src_idx]);
                }
                Ok(TensorInfo::new(target_shape.clone(), new_data))
            }
            EdgeCaseTransformation::Truncate { max_elements } => {
                let new_data = if tensor.data.len() > *max_elements {
                    tensor.data[..*max_elements].to_vec()
                } else {
                    tensor.data.clone()
                };
                let new_shape = vec![new_data.len()];
                Ok(TensorInfo::new(new_shape, new_data))
            }
        }
    }

    /// Update statistics
    fn update_statistics(
        &self,
        edge_case_type: &EdgeCaseType,
        handled: bool,
        handling_time_ms: f64,
    ) {
        if !self.config.enable_statistics {
            return;
        }

        if let Ok(mut stats) = self.statistics.write() {
            stats.total_edge_cases += 1;
            *stats
                .edge_cases_by_type
                .entry(edge_case_type.clone())
                .or_insert(0) += 1;

            if handled {
                stats.successful_handles += 1;
            } else {
                stats.failed_handles += 1;
            }

            // Update average handling time
            let total_time = stats.average_handling_time_ms * (stats.total_edge_cases - 1) as f64;
            stats.average_handling_time_ms =
                (total_time + handling_time_ms) / stats.total_edge_cases as f64;
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> EdgeCaseStatistics {
        self.statistics.read().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        if let Ok(mut stats) = self.statistics.write() {
            *stats = EdgeCaseStatistics::new();
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: EdgeCaseHandlingConfig) {
        self.config = config;
    }

    /// Enable or disable edge case handling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

/// Simple random normal generator for noise injection
fn random_normal() -> f64 {
    // Box-Muller transform for normal distribution
    static mut HAVE_SPARE: bool = false;
    static mut SPARE: f64 = 0.0;

    unsafe {
        if HAVE_SPARE {
            HAVE_SPARE = false;
            return SPARE;
        }

        HAVE_SPARE = true;
        let u = rand_uniform();
        let v = rand_uniform();
        let mag = 0.1 * (-2.0 * u.ln()).sqrt();
        SPARE = mag * (2.0 * std::f64::consts::PI * v).cos();
        mag * (2.0 * std::f64::consts::PI * v).sin()
    }
}

/// Simple uniform random generator
fn rand_uniform() -> f64 {
    // Simple LCG for demo purposes
    static mut STATE: u64 = 1;
    unsafe {
        STATE = STATE.wrapping_mul(1103515245).wrapping_add(12345);
        (STATE as f64) / (u64::MAX as f64)
    }
}

/// Global edge case handler
static GLOBAL_EDGE_CASE_HANDLER: std::sync::OnceLock<std::sync::Mutex<EdgeCaseHandler>> =
    std::sync::OnceLock::new();

/// Get the global edge case handler
pub fn get_global_edge_case_handler() -> &'static std::sync::Mutex<EdgeCaseHandler> {
    GLOBAL_EDGE_CASE_HANDLER.get_or_init(|| std::sync::Mutex::new(EdgeCaseHandler::with_defaults()))
}

/// Convenience function to handle edge cases using the global handler
pub fn handle_tensor_edge_cases(
    tensor: &TensorInfo,
) -> AutogradResult<Vec<EdgeCaseHandlingResult>> {
    let handler = get_global_edge_case_handler().lock().unwrap();
    handler.handle_tensor(tensor)
}

/// Convenience function to validate shapes using the global handler
pub fn validate_tensor_shapes(shapes: &[Vec<usize>], operation: &str) -> AutogradResult<()> {
    let handler = get_global_edge_case_handler().lock().unwrap();
    handler.validate_shapes(shapes, operation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_case_handler_creation() {
        let handler = EdgeCaseHandler::with_defaults();
        assert!(handler.config.enabled);
    }

    #[test]
    fn test_empty_tensor_detection() {
        let handler = EdgeCaseHandler::with_defaults();
        let empty_tensor = TensorInfo::new(vec![], vec![]);

        let edge_cases = handler.analyze_tensor(&empty_tensor);
        assert!(edge_cases.contains(&EdgeCaseType::EmptyTensor));
    }

    #[test]
    fn test_non_finite_values_detection() {
        let handler = EdgeCaseHandler::with_defaults();
        let tensor = TensorInfo::new(vec![3], vec![1.0, f64::NAN, 3.0]);

        let edge_cases = handler.analyze_tensor(&tensor);
        assert!(edge_cases.contains(&EdgeCaseType::NonFiniteValues));
    }

    #[test]
    fn test_extreme_values_detection() {
        let handler = EdgeCaseHandler::with_defaults();
        let tensor = TensorInfo::new(vec![2], vec![1e20, 2.0]);

        let edge_cases = handler.analyze_tensor(&tensor);
        assert!(edge_cases.contains(&EdgeCaseType::ExtremeValues));
    }

    #[test]
    fn test_degenerate_shape_detection() {
        let handler = EdgeCaseHandler::with_defaults();
        let tensor = TensorInfo::new(vec![3, 0, 2], vec![]);

        let edge_cases = handler.analyze_tensor(&tensor);
        assert!(edge_cases.contains(&EdgeCaseType::DegenerateShape));
    }

    #[test]
    fn test_fallback_tensor_creation() {
        let handler = EdgeCaseHandler::with_defaults();
        let original = TensorInfo::new(vec![], vec![]);

        let fallback = handler
            .create_fallback_tensor(&original, &Some(42.0), &Some(vec![2, 2]), "test_fallback")
            .unwrap();

        assert_eq!(fallback.shape, vec![2, 2]);
        assert_eq!(fallback.data, vec![42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_clamp_transformation() {
        let handler = EdgeCaseHandler::with_defaults();
        let tensor = TensorInfo::new(vec![3], vec![-1000.0, 5.0, 1000.0]);

        let transform = EdgeCaseTransformation::ClampValues {
            min_value: -10.0,
            max_value: 10.0,
        };

        let result = handler.apply_transformation(&tensor, &transform).unwrap();
        assert_eq!(result.data, vec![-10.0, 5.0, 10.0]);
    }

    #[test]
    fn test_replace_non_finite_transformation() {
        let handler = EdgeCaseHandler::with_defaults();
        let tensor = TensorInfo::new(vec![4], vec![1.0, f64::NAN, f64::INFINITY, -f64::INFINITY]);

        let transform = EdgeCaseTransformation::ReplaceNonFinite {
            nan_replacement: 0.0,
            inf_replacement: 100.0,
        };

        let result = handler.apply_transformation(&tensor, &transform).unwrap();
        assert_eq!(result.data, vec![1.0, 0.0, 100.0, -100.0]);
    }

    #[test]
    fn test_shape_broadcasting_validation() {
        let handler = EdgeCaseHandler::with_defaults();

        // Compatible shapes
        assert!(handler.are_shapes_broadcastable(&[3, 1], &[1, 4]));
        assert!(handler.are_shapes_broadcastable(&[2, 3, 1], &[1, 4]));

        // Incompatible shapes
        assert!(!handler.are_shapes_broadcastable(&[3, 2], &[3, 4]));
    }

    #[test]
    fn test_tensor_info_methods() {
        let tensor = TensorInfo::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(tensor.element_count(), 6);
        assert!(!tensor.is_empty());
        assert!(!tensor.is_scalar());
        assert_eq!(tensor.min_value(), Some(1.0));
        assert_eq!(tensor.max_value(), Some(6.0));
        assert!((tensor.norm() - (1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_edge_case_statistics() {
        let handler = EdgeCaseHandler::with_defaults();

        let stats = handler.get_statistics();
        assert_eq!(stats.total_edge_cases, 0);
        assert_eq!(stats.success_rate(), 1.0);

        handler.update_statistics(&EdgeCaseType::EmptyTensor, true, 1.5);

        let updated_stats = handler.get_statistics();
        assert_eq!(updated_stats.total_edge_cases, 1);
        assert_eq!(updated_stats.successful_handles, 1);
        assert_eq!(updated_stats.success_rate(), 1.0);
    }

    #[test]
    fn test_edge_case_type_display() {
        assert_eq!(EdgeCaseType::EmptyTensor.to_string(), "empty_tensor");
        assert_eq!(
            EdgeCaseType::NonFiniteValues.to_string(),
            "non_finite_values"
        );
        assert_eq!(
            EdgeCaseType::Custom("test".to_string()).to_string(),
            "custom_test"
        );
    }

    #[test]
    fn test_global_handler_access() {
        let handler = get_global_edge_case_handler();
        assert!(handler.lock().unwrap().config.enabled);
    }

    #[test]
    fn test_convenience_functions() {
        let tensor = TensorInfo::new(vec![], vec![]);
        let results = handle_tensor_edge_cases(&tensor).unwrap();
        assert!(!results.is_empty());

        let shapes = vec![vec![2, 3], vec![2, 3]];
        assert!(validate_tensor_shapes(&shapes, "test_op").is_ok());
    }
}
