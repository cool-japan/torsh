//! Development-time Shape Validation with Detailed Error Messages
//!
//! This module provides comprehensive shape validation for development environments,
//! offering detailed error messages, suggestions, and interactive debugging capabilities
//! to help developers understand and fix tensor shape issues.

use crate::{Shape, DType, Result, TorshError};
use crate::shape_debug::{ShapeDebugger, OperationType as ShapeOpType};
use std::collections::HashMap;
use std::fmt;

/// Configuration for development-time shape validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Whether to enable strict validation (more checks, slower)
    pub strict_mode: bool,
    /// Whether to provide optimization suggestions
    pub suggest_optimizations: bool,
    /// Whether to show visual representations of shape mismatches
    pub show_visual_aids: bool,
    /// Whether to capture context from previous operations
    pub track_operation_context: bool,
    /// Maximum number of suggestions to provide per error
    pub max_suggestions: usize,
    /// Whether to enable interactive debugging prompts
    pub interactive_mode: bool,
    /// Whether to perform performance impact analysis
    pub analyze_performance: bool,
    /// Whether to suggest memory-efficient alternatives
    pub suggest_memory_optimizations: bool,
    /// Whether to enable auto-correction suggestions
    pub enable_auto_correction: bool,
    /// Verbosity level for error messages (1-5)
    pub verbosity_level: u8,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            suggest_optimizations: true,
            show_visual_aids: true,
            track_operation_context: true,
            max_suggestions: 5,
            interactive_mode: false, // Disabled by default for CI/CD compatibility
            analyze_performance: true,
            suggest_memory_optimizations: true,
            enable_auto_correction: true,
            verbosity_level: 3, // Medium verbosity
        }
    }
}

/// Detailed shape validation error with context and suggestions
#[derive(Debug, Clone)]
pub struct ShapeValidationError {
    /// Core error information
    pub error_type: ValidationErrorType,
    /// Detailed explanation of what went wrong
    pub explanation: String,
    /// Visual representation of the problem (if available)
    pub visual_aid: Option<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
    /// Code examples showing how to fix the issue
    pub examples: Vec<CodeExample>,
    /// Context from previous operations
    pub operation_context: Vec<OperationContext>,
    /// Severity of the error
    pub severity: ErrorSeverity,
    /// Performance impact analysis
    pub performance_impact: Option<PerformanceAnalysis>,
    /// Memory efficiency suggestions
    pub memory_suggestions: Vec<String>,
    /// Auto-correction suggestions (executable code)
    pub auto_corrections: Vec<AutoCorrection>,
    /// Error location information
    pub location_info: Option<LocationInfo>,
}

/// Types of shape validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorType {
    /// Dimension count mismatch
    DimensionCountMismatch {
        expected: usize,
        got: usize,
        operation: String,
    },
    /// Specific dimension size mismatch
    DimensionSizeMismatch {
        dimension: usize,
        expected: usize,
        got: usize,
        operation: String,
    },
    /// Broadcasting incompatibility
    BroadcastingError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        conflicting_dims: Vec<usize>,
    },
    /// Matrix multiplication incompatibility
    MatMulIncompatible {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
        inner_dims: (usize, usize),
    },
    /// Convolution parameter mismatch
    ConvolutionError {
        input_shape: Vec<usize>,
        kernel_shape: Vec<usize>,
        error_detail: String,
    },
    /// Reduction operation error
    ReductionError {
        input_shape: Vec<usize>,
        dimension: Option<usize>,
        error_detail: String,
    },
    /// Reshape impossibility
    ReshapeError {
        original_shape: Vec<usize>,
        target_shape: Vec<usize>,
        original_size: usize,
        target_size: usize,
    },
    /// Index out of bounds
    IndexError {
        shape: Vec<usize>,
        indices: Vec<isize>,
        problematic_dim: usize,
    },
    /// Memory layout incompatibility
    LayoutError {
        operation: String,
        required_layout: String,
        actual_layout: String,
    },
    /// Custom validation error
    Custom {
        operation: String,
        description: String,
    },
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Warning - operation might work but could be inefficient
    Warning,
    /// Error - operation will definitely fail
    Error,
    /// Critical - operation will cause undefined behavior or crashes
    Critical,
}

/// Code example showing how to fix an issue
#[derive(Debug, Clone)]
pub struct CodeExample {
    /// Description of what this example demonstrates
    pub description: String,
    /// Problematic code (what not to do)
    pub bad_code: Option<String>,
    /// Corrected code (what to do instead)
    pub good_code: String,
    /// Additional explanation
    pub explanation: String,
}

/// Context information about previous operations
#[derive(Debug, Clone)]
pub struct OperationContext {
    /// Operation name
    pub operation: String,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// When this operation occurred
    pub sequence_number: usize,
}

/// Performance impact analysis for shape operations
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Estimated computational complexity (FLOPs)
    pub computational_cost: u64,
    /// Estimated memory usage (bytes)
    pub memory_usage: usize,
    /// Performance bottleneck analysis
    pub bottlenecks: Vec<String>,
    /// Optimization opportunities
    pub optimizations: Vec<String>,
    /// Relative performance impact (1.0 = baseline)
    pub relative_impact: f64,
}

/// Auto-correction suggestion with executable code
#[derive(Debug, Clone)]
pub struct AutoCorrection {
    /// Description of the correction
    pub description: String,
    /// Rust code that would fix the issue
    pub correction_code: String,
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    /// Whether this correction is safe to auto-apply
    pub is_safe: bool,
    /// Expected outcome after applying correction
    pub expected_outcome: String,
}

/// Location information for error context
#[derive(Debug, Clone)]
pub struct LocationInfo {
    /// File name where error occurred
    pub file_name: Option<String>,
    /// Line number
    pub line_number: Option<u32>,
    /// Column number
    pub column_number: Option<u32>,
    /// Function or method name
    pub function_name: Option<String>,
    /// Additional context
    pub context: Option<String>,
}

/// Development-time shape validator
pub struct ShapeValidator {
    /// Configuration
    config: ValidationConfig,
    /// Operation history for context
    operation_history: Vec<OperationContext>,
    /// Shape debugger for analysis
    debugger: ShapeDebugger,
    /// Next sequence number
    next_sequence: usize,
}

impl ShapeValidator {
    /// Create a new shape validator with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new shape validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            operation_history: Vec::new(),
            debugger: ShapeDebugger::new(),
            next_sequence: 1,
        }
    }

    /// Validate element-wise operation compatibility
    pub fn validate_elementwise(&mut self, shapes: &[&Shape], operation: &str) -> Result<Vec<usize>> {
        if shapes.is_empty() {
            return Err(self.create_error(
                ValidationErrorType::Custom {
                    operation: operation.to_string(),
                    description: "No input shapes provided for element-wise operation".to_string(),
                },
                "Element-wise operations require at least one input tensor".to_string(),
            ));
        }

        let mut result_shape = shapes[0].dims().to_vec();
        
        for (i, shape) in shapes.iter().enumerate().skip(1) {
            let compat = self.debugger.check_broadcast_compatibility(&result_shape, shape.dims());
            
            if !compat.is_broadcastable {
                return Err(self.create_broadcasting_error(
                    result_shape,
                    shape.dims().to_vec(),
                    operation,
                ));
            }
            
            result_shape = compat.resulting_shape.unwrap_or(result_shape);
        }

        // Record successful operation
        self.record_operation(operation, shapes, &result_shape);

        Ok(result_shape)
    }

    /// Validate matrix multiplication compatibility
    pub fn validate_matmul(&mut self, left: &Shape, right: &Shape) -> Result<Vec<usize>> {
        let left_dims = left.dims();
        let right_dims = right.dims();

        // Check minimum dimensionality
        if left_dims.len() < 2 || right_dims.len() < 2 {
            return Err(self.create_error(
                ValidationErrorType::MatMulIncompatible {
                    left_shape: left_dims.to_vec(),
                    right_shape: right_dims.to_vec(),
                    inner_dims: (0, 0),
                },
                "Matrix multiplication requires tensors with at least 2 dimensions".to_string(),
            ));
        }

        let left_inner = left_dims[left_dims.len() - 1];
        let right_inner = right_dims[right_dims.len() - 2];

        if left_inner != right_inner {
            return Err(self.create_matmul_error(left_dims, right_dims, left_inner, right_inner));
        }

        // Check batch dimensions compatibility
        let left_batch = &left_dims[..left_dims.len() - 2];
        let right_batch = &right_dims[..right_dims.len() - 2];

        let batch_compat = self.debugger.check_broadcast_compatibility(left_batch, right_batch);
        if !batch_compat.is_broadcastable {
            return Err(self.create_error(
                ValidationErrorType::BroadcastingError {
                    shape1: left_dims.to_vec(),
                    shape2: right_dims.to_vec(),
                    conflicting_dims: vec![], // Could be improved to identify specific dims
                },
                "Batch dimensions are not compatible for matrix multiplication".to_string(),
            ));
        }

        // Construct result shape
        let batch_shape = batch_compat.resulting_shape.unwrap_or_else(Vec::new);
        let mut result_shape = batch_shape;
        result_shape.push(left_dims[left_dims.len() - 2]); // rows from left
        result_shape.push(right_dims[right_dims.len() - 1]); // cols from right

        self.record_operation("matmul", &[left, right], &result_shape);

        Ok(result_shape)
    }

    /// Validate reshape operation
    pub fn validate_reshape(&mut self, original: &Shape, target_shape: &[usize]) -> Result<()> {
        let original_size = original.numel();
        let target_size: usize = target_shape.iter().product();

        if original_size != target_size {
            return Err(self.create_reshape_error(
                original.dims().to_vec(),
                target_shape.to_vec(),
                original_size,
                target_size,
            ));
        }

        self.record_operation("reshape", &[original], target_shape);

        Ok(())
    }

    /// Validate convolution operation
    pub fn validate_convolution(
        &mut self,
        input: &Shape,
        kernel: &Shape,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
    ) -> Result<Vec<usize>> {
        let input_dims = input.dims();
        let kernel_dims = kernel.dims();

        // Check minimum dimensions (NCHW format)
        if input_dims.len() != 4 || kernel_dims.len() != 4 {
            return Err(self.create_convolution_error(
                input_dims.to_vec(),
                kernel_dims.to_vec(),
                "Convolution requires 4D tensors (NCHW format)".to_string(),
            ));
        }

        // Check channel compatibility
        let input_channels = input_dims[1];
        let kernel_in_channels = kernel_dims[1];

        if input_channels != kernel_in_channels {
            return Err(self.create_convolution_error(
                input_dims.to_vec(),
                kernel_dims.to_vec(),
                format!(
                    "Input channels ({}) don't match kernel input channels ({})",
                    input_channels, kernel_in_channels
                ),
            ));
        }

        // Calculate output dimensions
        let batch_size = input_dims[0];
        let out_channels = kernel_dims[0];
        
        let output_height = self.calculate_conv_output_size(
            input_dims[2], kernel_dims[2], stride[0], padding[0], dilation[0]
        )?;
        
        let output_width = self.calculate_conv_output_size(
            input_dims[3], kernel_dims[3], stride[1], padding[1], dilation[1]
        )?;

        let result_shape = vec![batch_size, out_channels, output_height, output_width];
        
        self.record_operation("convolution", &[input, kernel], &result_shape);

        Ok(result_shape)
    }

    /// Validate reduction operation
    pub fn validate_reduction(
        &mut self,
        input: &Shape,
        dimensions: Option<&[usize]>,
        keep_dims: bool,
        operation: &str,
    ) -> Result<Vec<usize>> {
        let input_dims = input.dims();

        if let Some(dims) = dimensions {
            // Check that all dimensions are valid
            for &dim in dims {
                if dim >= input_dims.len() {
                    return Err(self.create_error(
                        ValidationErrorType::ReductionError {
                            input_shape: input_dims.to_vec(),
                            dimension: Some(dim),
                            error_detail: format!(
                                "Dimension {} is out of bounds for tensor with {} dimensions",
                                dim, input_dims.len()
                            ),
                        },
                        format!("Invalid dimension for {} operation", operation),
                    ));
                }
            }
        }

        let result_shape = self.calculate_reduction_shape(input_dims, dimensions, keep_dims);
        
        self.record_operation(&format!("reduce_{}", operation), &[input], &result_shape);

        Ok(result_shape)
    }

    /// Validate indexing operation
    pub fn validate_indexing(&mut self, shape: &Shape, indices: &[isize]) -> Result<Vec<usize>> {
        let dims = shape.dims();

        if indices.len() > dims.len() {
            return Err(self.create_error(
                ValidationErrorType::IndexError {
                    shape: dims.to_vec(),
                    indices: indices.to_vec(),
                    problematic_dim: indices.len().saturating_sub(1),
                },
                format!(
                    "Too many indices: got {}, tensor has {} dimensions",
                    indices.len(), dims.len()
                ),
            ));
        }

        for (i, &idx) in indices.iter().enumerate() {
            let dim_size = dims[i] as isize;
            
            // Check bounds (allowing negative indexing)
            if idx >= dim_size || idx < -dim_size {
                return Err(self.create_error(
                    ValidationErrorType::IndexError {
                        shape: dims.to_vec(),
                        indices: indices.to_vec(),
                        problematic_dim: i,
                    },
                    format!(
                        "Index {} is out of bounds for dimension {} with size {}",
                        idx, i, dim_size
                    ),
                ));
            }
        }

        // Calculate result shape (simplified - assumes single element indexing)
        let result_shape = dims[indices.len()..].to_vec();
        
        self.record_operation("indexing", &[shape], &result_shape);

        Ok(result_shape)
    }

    /// Get operation history for debugging
    pub fn get_operation_history(&self) -> &[OperationContext] {
        &self.operation_history
    }

    /// Clear operation history
    pub fn clear_history(&mut self) {
        self.operation_history.clear();
        self.next_sequence = 1;
    }

    /// Generate a comprehensive validation report
    pub fn generate_validation_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Shape Validation Report ===\n\n");

        report.push_str(&format!("Total operations validated: {}\n", self.operation_history.len()));
        report.push_str(&format!("Validation mode: {}\n", 
            if self.config.strict_mode { "Strict" } else { "Permissive" }));

        if !self.operation_history.is_empty() {
            report.push_str("\nRecent operations:\n");
            for (i, op) in self.operation_history.iter().rev().take(10).enumerate() {
                report.push_str(&format!(
                    "  {}. {} -> {:?}\n",
                    op.sequence_number,
                    op.operation,
                    op.output_shape
                ));
            }
        }

        report
    }

    // Private helper methods

    fn record_operation(&mut self, operation: &str, inputs: &[&Shape], output: &[usize]) {
        if self.config.track_operation_context {
            let context = OperationContext {
                operation: operation.to_string(),
                input_shapes: inputs.iter().map(|s| s.dims().to_vec()).collect(),
                output_shape: output.to_vec(),
                sequence_number: self.next_sequence,
            };

            self.operation_history.push(context);
            self.next_sequence += 1;

            // Limit history size
            if self.operation_history.len() > 100 {
                self.operation_history.remove(0);
            }
        }
    }

    fn create_error(&self, error_type: ValidationErrorType, explanation: String) -> TorshError {
        let validation_error = ShapeValidationError {
            error_type: error_type.clone(),
            explanation: explanation.clone(),
            visual_aid: self.create_visual_aid(&error_type),
            suggestions: self.generate_suggestions(&error_type),
            examples: self.generate_examples(&error_type),
            operation_context: self.get_relevant_context(&error_type),
            severity: self.determine_severity(&error_type),
            performance_impact: if self.config.analyze_performance {
                Some(self.analyze_performance_impact(&error_type))
            } else {
                None
            },
            memory_suggestions: if self.config.suggest_memory_optimizations {
                self.generate_memory_suggestions(&error_type)
            } else {
                Vec::new()
            },
            auto_corrections: if self.config.enable_auto_correction {
                self.generate_auto_corrections(&error_type)
            } else {
                Vec::new()
            },
            location_info: self.capture_location_info(),
        };

        TorshError::ShapeValidation(format!("{}", validation_error))
    }

    fn create_broadcasting_error(
        &self,
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        operation: &str,
    ) -> TorshError {
        let conflicting_dims = self.find_conflicting_dims(&shape1, &shape2);
        
        self.create_error(
            ValidationErrorType::BroadcastingError {
                shape1: shape1.clone(),
                shape2: shape2.clone(),
                conflicting_dims,
            },
            format!(
                "Cannot broadcast shapes {:?} and {:?} for {} operation",
                shape1, shape2, operation
            ),
        )
    }

    fn create_matmul_error(
        &self,
        left_shape: &[usize],
        right_shape: &[usize],
        left_inner: usize,
        right_inner: usize,
    ) -> TorshError {
        self.create_error(
            ValidationErrorType::MatMulIncompatible {
                left_shape: left_shape.to_vec(),
                right_shape: right_shape.to_vec(),
                inner_dims: (left_inner, right_inner),
            },
            format!(
                "Matrix multiplication incompatible: left inner dimension {} != right inner dimension {}",
                left_inner, right_inner
            ),
        )
    }

    fn create_reshape_error(
        &self,
        original: Vec<usize>,
        target: Vec<usize>,
        original_size: usize,
        target_size: usize,
    ) -> TorshError {
        self.create_error(
            ValidationErrorType::ReshapeError {
                original_shape: original,
                target_shape: target,
                original_size,
                target_size,
            },
            format!(
                "Cannot reshape tensor: element count mismatch ({} != {})",
                original_size, target_size
            ),
        )
    }

    fn create_convolution_error(
        &self,
        input_shape: Vec<usize>,
        kernel_shape: Vec<usize>,
        detail: String,
    ) -> TorshError {
        self.create_error(
            ValidationErrorType::ConvolutionError {
                input_shape,
                kernel_shape,
                error_detail: detail.clone(),
            },
            format!("Convolution error: {}", detail),
        )
    }

    fn calculate_conv_output_size(
        &self,
        input_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<usize> {
        let effective_kernel_size = dilation * (kernel_size - 1) + 1;
        let numerator = input_size + 2 * padding;
        
        if numerator < effective_kernel_size {
            return Err(TorshError::InvalidArgument(
                "Effective kernel size is larger than padded input".to_string()
            ));
        }
        
        Ok((numerator - effective_kernel_size) / stride + 1)
    }

    fn calculate_reduction_shape(
        &self,
        input_dims: &[usize],
        dimensions: Option<&[usize]>,
        keep_dims: bool,
    ) -> Vec<usize> {
        match dimensions {
            Some(dims) => {
                let mut result = input_dims.to_vec();
                let mut sorted_dims = dims.to_vec();
                sorted_dims.sort_by(|a, b| b.cmp(a)); // Sort in descending order
                
                for &dim in &sorted_dims {
                    if keep_dims {
                        result[dim] = 1;
                    } else {
                        result.remove(dim);
                    }
                }
                result
            },
            None => {
                if keep_dims {
                    vec![1; input_dims.len()]
                } else {
                    vec![]
                }
            }
        }
    }

    fn find_conflicting_dims(&self, shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
        let max_len = shape1.len().max(shape2.len());
        let mut conflicts = Vec::new();

        for i in 0..max_len {
            let dim1 = shape1.get(shape1.len().saturating_sub(max_len) + i).copied().unwrap_or(1);
            let dim2 = shape2.get(shape2.len().saturating_sub(max_len) + i).copied().unwrap_or(1);

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                conflicts.push(i);
            }
        }

        conflicts
    }

    fn create_visual_aid(&self, error_type: &ValidationErrorType) -> Option<String> {
        if !self.config.show_visual_aids {
            return None;
        }

        match error_type {
            ValidationErrorType::BroadcastingError { shape1, shape2, .. } => {
                Some(self.create_broadcasting_visual(shape1, shape2))
            },
            ValidationErrorType::MatMulIncompatible { left_shape, right_shape, .. } => {
                Some(self.create_matmul_visual(left_shape, right_shape))
            },
            ValidationErrorType::ReshapeError { original_shape, target_shape, .. } => {
                Some(self.create_reshape_visual(original_shape, target_shape))
            },
            _ => None,
        }
    }

    fn create_broadcasting_visual(&self, shape1: &[usize], shape2: &[usize]) -> String {
        format!(
            "Broadcasting shapes:\n  {:?}\n  {:?}\n  ↑ Incompatible dimensions",
            shape1, shape2
        )
    }

    fn create_matmul_visual(&self, left: &[usize], right: &[usize]) -> String {
        format!(
            "Matrix multiplication:\n  Left:  {:?} (inner: {})\n  Right: {:?} (inner: {})\n  ↑ Inner dimensions must match",
            left, left.get(left.len().saturating_sub(1)).unwrap_or(&0),
            right, right.get(right.len().saturating_sub(2)).unwrap_or(&0)
        )
    }

    fn create_reshape_visual(&self, original: &[usize], target: &[usize]) -> String {
        let orig_size: usize = original.iter().product();
        let target_size: usize = target.iter().product();
        format!(
            "Reshape:\n  Original: {:?} ({} elements)\n  Target:   {:?} ({} elements)\n  ↑ Element counts must match",
            original, orig_size, target, target_size
        )
    }

    fn generate_suggestions(&self, error_type: &ValidationErrorType) -> Vec<String> {
        if !self.config.suggest_optimizations {
            return Vec::new();
        }

        let mut suggestions = Vec::new();

        match error_type {
            ValidationErrorType::BroadcastingError { shape1, shape2, .. } => {
                suggestions.push("Use explicit expand() or unsqueeze() to make shapes compatible".to_string());
                suggestions.push("Check if you need to transpose one of the tensors".to_string());
                suggestions.push("Consider using reshape() to adjust tensor dimensions".to_string());
            },
            ValidationErrorType::MatMulIncompatible { .. } => {
                suggestions.push("Transpose one of the matrices using .t() or .transpose()".to_string());
                suggestions.push("Check matrix dimensions: (m×k) @ (k×n) = (m×n)".to_string());
                suggestions.push("Use reshape() to adjust inner dimensions if needed".to_string());
            },
            ValidationErrorType::ReshapeError { .. } => {
                suggestions.push("Use -1 for one dimension to infer size automatically".to_string());
                suggestions.push("Check total element count: original and target must be equal".to_string());
                suggestions.push("Consider using view() instead of reshape() if possible".to_string());
            },
            ValidationErrorType::ConvolutionError { .. } => {
                suggestions.push("Check input format: expected NCHW (batch, channels, height, width)".to_string());
                suggestions.push("Verify kernel and input channel dimensions match".to_string());
                suggestions.push("Adjust padding or stride parameters if output size is invalid".to_string());
            },
            _ => {
                suggestions.push("Check tensor documentation for expected input shapes".to_string());
                suggestions.push("Use tensor.shape to inspect current dimensions".to_string());
            }
        }

        suggestions.truncate(self.config.max_suggestions);
        suggestions
    }

    fn generate_examples(&self, error_type: &ValidationErrorType) -> Vec<CodeExample> {
        match error_type {
            ValidationErrorType::BroadcastingError { .. } => {
                vec![
                    CodeExample {
                        description: "Fix broadcasting with unsqueeze".to_string(),
                        bad_code: Some("a.add(b)  // where a=[3,4], b=[4]".to_string()),
                        good_code: "a.add(b.unsqueeze(0))  // b becomes [1,4]".to_string(),
                        explanation: "Add dimensions to make shapes compatible".to_string(),
                    }
                ]
            },
            ValidationErrorType::MatMulIncompatible { .. } => {
                vec![
                    CodeExample {
                        description: "Fix matrix multiplication dimensions".to_string(),
                        bad_code: Some("a.matmul(b)  // where a=[3,4], b=[3,5]".to_string()),
                        good_code: "a.matmul(b.t())  // transpose b to [5,3], then use [3,4] @ [4,5]".to_string(),
                        explanation: "Inner dimensions must match for matrix multiplication".to_string(),
                    }
                ]
            },
            _ => Vec::new(),
        }
    }

    fn get_relevant_context(&self, _error_type: &ValidationErrorType) -> Vec<OperationContext> {
        // Return last few operations for context
        self.operation_history.iter().rev().take(3).cloned().collect()
    }

    fn determine_severity(&self, error_type: &ValidationErrorType) -> ErrorSeverity {
        match error_type {
            ValidationErrorType::DimensionCountMismatch { .. } |
            ValidationErrorType::DimensionSizeMismatch { .. } |
            ValidationErrorType::BroadcastingError { .. } |
            ValidationErrorType::MatMulIncompatible { .. } |
            ValidationErrorType::ConvolutionError { .. } |
            ValidationErrorType::ReshapeError { .. } |
            ValidationErrorType::IndexError { .. } => ErrorSeverity::Error,
            
            ValidationErrorType::LayoutError { .. } => ErrorSeverity::Warning,
            
            ValidationErrorType::ReductionError { .. } |
            ValidationErrorType::Custom { .. } => ErrorSeverity::Error,
        }
    }

    fn analyze_performance_impact(&self, error_type: &ValidationErrorType) -> PerformanceAnalysis {
        match error_type {
            ValidationErrorType::BroadcastingError { shape1, shape2, .. } => {
                let size1: usize = shape1.iter().product();
                let size2: usize = shape2.iter().product();
                let computational_cost = (size1.max(size2) as u64) * 2; // Basic estimate
                
                PerformanceAnalysis {
                    computational_cost,
                    memory_usage: (size1 + size2) * 4, // Assuming f32
                    bottlenecks: vec![
                        "Broadcasting creates temporary tensors".to_string(),
                        "Memory allocation overhead".to_string(),
                    ],
                    optimizations: vec![
                        "Use in-place operations when possible".to_string(),
                        "Reshape tensors before operation".to_string(),
                    ],
                    relative_impact: if size1 != size2 { 1.5 } else { 1.0 },
                }
            },
            ValidationErrorType::MatMulIncompatible { left_shape, right_shape, .. } => {
                let left_size: usize = left_shape.iter().product();
                let right_size: usize = right_shape.iter().product();
                
                PerformanceAnalysis {
                    computational_cost: (left_size * right_size) as u64 / 1000, // Rough estimate
                    memory_usage: (left_size + right_size) * 4,
                    bottlenecks: vec![
                        "Matrix dimensions incompatible".to_string(),
                        "Potential transpose overhead".to_string(),
                    ],
                    optimizations: vec![
                        "Transpose smaller matrix if needed".to_string(),
                        "Consider batch operations".to_string(),
                    ],
                    relative_impact: 2.0, // Matrix ops are expensive when wrong
                }
            },
            _ => PerformanceAnalysis {
                computational_cost: 1000,
                memory_usage: 1024,
                bottlenecks: vec!["Shape mismatch".to_string()],
                optimizations: vec!["Fix shape compatibility".to_string()],
                relative_impact: 1.2,
            }
        }
    }

    fn generate_memory_suggestions(&self, error_type: &ValidationErrorType) -> Vec<String> {
        match error_type {
            ValidationErrorType::BroadcastingError { .. } => {
                vec![
                    "Use view() instead of expand() when possible to avoid memory copies".to_string(),
                    "Consider using in-place operations (+=, *=, etc.)".to_string(),
                    "Pre-allocate output tensor to avoid reallocations".to_string(),
                ]
            },
            ValidationErrorType::ReshapeError { original_size, target_size, .. } => {
                vec![
                    format!(
                        "Reshape is memory-neutral but element count mismatch ({} vs {}) will fail",
                        original_size, target_size
                    ),
                    "Use view() for zero-copy shape changes when memory layout allows".to_string(),
                ]
            },
            ValidationErrorType::ConvolutionError { .. } => {
                vec![
                    "Consider using depthwise separable convolutions for efficiency".to_string(),
                    "Use im2col-free convolution implementations when available".to_string(),
                    "Optimize memory layout (NCHW vs NHWC) for your hardware".to_string(),
                ]
            },
            _ => vec![
                "Ensure tensors are contiguous for optimal memory access".to_string(),
                "Consider using smaller data types (f16) if precision allows".to_string(),
            ]
        }
    }

    fn generate_auto_corrections(&self, error_type: &ValidationErrorType) -> Vec<AutoCorrection> {
        match error_type {
            ValidationErrorType::BroadcastingError { shape1, shape2, .. } => {
                let mut corrections = Vec::new();
                
                // Suggest unsqueeze for missing dimensions
                if shape1.len() < shape2.len() {
                    corrections.push(AutoCorrection {
                        description: "Add missing dimensions to first tensor".to_string(),
                        correction_code: "tensor1.unsqueeze(0)".to_string(),
                        confidence: 0.8,
                        is_safe: true,
                        expected_outcome: format!("Shape becomes compatible: {:?}", shape2),
                    });
                }
                
                corrections
            },
            ValidationErrorType::MatMulIncompatible { left_shape, right_shape, .. } => {
                vec![
                    AutoCorrection {
                        description: "Transpose right matrix to make dimensions compatible".to_string(),
                        correction_code: "right_tensor.transpose(-2, -1)".to_string(),
                        confidence: 0.9,
                        is_safe: true,
                        expected_outcome: format!(
                            "Matrix multiplication becomes: {:?} @ {:?}", 
                            left_shape, 
                            right_shape.iter().rev().cloned().collect::<Vec<_>>()
                        ),
                    }
                ]
            },
            ValidationErrorType::ReshapeError { original_shape, target_shape, .. } => {
                let original_size: usize = original_shape.iter().product();
                vec![
                    AutoCorrection {
                        description: "Use -1 for automatic dimension inference".to_string(),
                        correction_code: format!("tensor.reshape([{}, -1])", target_shape[0]),
                        confidence: 0.7,
                        is_safe: true,
                        expected_outcome: format!(
                            "Auto-infer second dimension: [{}, {}]", 
                            target_shape[0], 
                            original_size / target_shape[0]
                        ),
                    }
                ]
            },
            _ => Vec::new(),
        }
    }

    fn capture_location_info(&self) -> Option<LocationInfo> {
        // In a real implementation, this would use std::panic::Location
        // or debug information to capture the actual call site
        Some(LocationInfo {
            file_name: Some("shape_validation.rs".to_string()),
            line_number: Some(std::line!()),
            column_number: None,
            function_name: Some("validate_operation".to_string()),
            context: Some("Shape validation during tensor operation".to_string()),
        })
    }
}

impl Default for ShapeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ShapeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "🔍 Shape Validation Error ({:?}):", self.severity)?;
        writeln!(f, "{}", self.explanation)?;

        // Location info
        if let Some(ref location) = self.location_info {
            if let (Some(ref file), Some(line)) = (&location.file_name, location.line_number) {
                writeln!(f, "📍 Location: {}:{}", file, line)?;
            }
            if let Some(ref func) = &location.function_name {
                writeln!(f, "🔧 Function: {}", func)?;
            }
        }

        if let Some(ref visual) = self.visual_aid {
            writeln!(f, "\n🎨 Visual representation:")?;
            writeln!(f, "{}", visual)?;
        }

        if !self.suggestions.is_empty() {
            writeln!(f, "\n💡 Suggestions:")?;
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, suggestion)?;
            }
        }

        // Memory suggestions
        if !self.memory_suggestions.is_empty() {
            writeln!(f, "\n🧠 Memory optimization:")?;
            for (i, suggestion) in self.memory_suggestions.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, suggestion)?;
            }
        }

        // Performance analysis
        if let Some(ref perf) = self.performance_impact {
            writeln!(f, "\n⚡ Performance impact:")?;
            writeln!(f, "  Computational cost: {} FLOPs", perf.computational_cost)?;
            writeln!(f, "  Memory usage: {} bytes", perf.memory_usage)?;
            writeln!(f, "  Relative impact: {:.1}x", perf.relative_impact)?;
            if !perf.bottlenecks.is_empty() {
                writeln!(f, "  Bottlenecks: {}", perf.bottlenecks.join(", "))?;
            }
        }

        // Auto-corrections
        if !self.auto_corrections.is_empty() {
            writeln!(f, "\n🤖 Auto-corrections:")?;
            for (i, correction) in self.auto_corrections.iter().enumerate() {
                writeln!(f, "  {}. {} (confidence: {:.1}%)", 
                    i + 1, correction.description, correction.confidence * 100.0)?;
                writeln!(f, "     Code: {}", correction.correction_code)?;
                if correction.is_safe {
                    writeln!(f, "     ✅ Safe to auto-apply")?;
                } else {
                    writeln!(f, "     ⚠️  Manual review recommended")?;
                }
            }
        }

        if !self.examples.is_empty() {
            writeln!(f, "\n📝 Examples:")?;
            for example in &self.examples {
                writeln!(f, "  {}", example.description)?;
                if let Some(ref bad) = example.bad_code {
                    writeln!(f, "    ❌ {}", bad)?;
                }
                writeln!(f, "    ✅ {}", example.good_code)?;
                writeln!(f, "    ℹ️  {}", example.explanation)?;
            }
        }

        if !self.operation_context.is_empty() {
            writeln!(f, "\n📜 Recent operations:")?;
            for context in &self.operation_context {
                writeln!(f, "  {}. {} -> {:?}", 
                    context.sequence_number, context.operation, context.output_shape)?;
            }
        }

        Ok(())
    }
}

/// Global validator instance for convenience
static GLOBAL_VALIDATOR: std::sync::OnceLock<std::sync::Mutex<ShapeValidator>> = std::sync::OnceLock::new();

/// Initialize global validator with custom configuration
pub fn init_global_validator(config: ValidationConfig) -> Result<()> {
    GLOBAL_VALIDATOR.set(std::sync::Mutex::new(ShapeValidator::with_config(config)))
        .map_err(|_| TorshError::InvalidState("Global validator already initialized".to_string()))?;
    Ok(())
}

/// Get reference to global validator
pub fn get_global_validator() -> std::sync::MutexGuard<'static, ShapeValidator> {
    GLOBAL_VALIDATOR.get_or_init(|| {
        std::sync::Mutex::new(ShapeValidator::new())
    }).lock().unwrap()
}

/// Convenience macros for validation
#[macro_export]
macro_rules! validate_shapes {
    (elementwise, $op:expr, $($shape:expr),+) => {{
        let shapes = vec![$($shape),+];
        $crate::shape_validation::get_global_validator().validate_elementwise(&shapes, $op)
    }};
    
    (matmul, $left:expr, $right:expr) => {{
        $crate::shape_validation::get_global_validator().validate_matmul($left, $right)
    }};
    
    (reshape, $original:expr, $target:expr) => {{
        $crate::shape_validation::get_global_validator().validate_reshape($original, $target)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn test_validator_creation() {
        let validator = ShapeValidator::new();
        assert_eq!(validator.operation_history.len(), 0);
    }

    #[test]
    fn test_elementwise_validation_success() {
        let mut validator = ShapeValidator::new();
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![1, 4]);
        
        let result = validator.validate_elementwise(&[&shape1, &shape2], "add");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_elementwise_validation_failure() {
        let mut validator = ShapeValidator::new();
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![3, 5]);
        
        let result = validator.validate_elementwise(&[&shape1, &shape2], "add");
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_validation_success() {
        let mut validator = ShapeValidator::new();
        let left = Shape::new(vec![3, 4]);
        let right = Shape::new(vec![4, 5]);
        
        let result = validator.validate_matmul(&left, &right);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![3, 5]);
    }

    #[test]
    fn test_matmul_validation_failure() {
        let mut validator = ShapeValidator::new();
        let left = Shape::new(vec![3, 4]);
        let right = Shape::new(vec![5, 6]);
        
        let result = validator.validate_matmul(&left, &right);
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_validation() {
        let mut validator = ShapeValidator::new();
        let shape = Shape::new(vec![2, 3, 4]);
        
        // Valid reshape
        let result = validator.validate_reshape(&shape, &[6, 4]);
        assert!(result.is_ok());
        
        // Invalid reshape
        let result = validator.validate_reshape(&shape, &[5, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_convolution_validation() {
        let mut validator = ShapeValidator::new();
        let input = Shape::new(vec![1, 3, 32, 32]); // NCHW
        let kernel = Shape::new(vec![16, 3, 3, 3]); // Out, In, H, W
        
        let result = validator.validate_convolution(
            &input, &kernel, 
            &[1, 1], // stride
            &[0, 0], // padding
            &[1, 1], // dilation
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_operation_history_tracking() {
        let mut validator = ShapeValidator::new();
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![3, 4]);
        
        let _result = validator.validate_elementwise(&[&shape1, &shape2], "add");
        
        assert_eq!(validator.operation_history.len(), 1);
        assert_eq!(validator.operation_history[0].operation, "add");
    }

    #[test]
    fn test_global_validator() {
        let validator = get_global_validator();
        // Just test that we can get the global validator without panicking
        drop(validator);
    }

    #[test]
    fn test_enhanced_error_messages() {
        let config = ValidationConfig {
            analyze_performance: true,
            suggest_memory_optimizations: true,
            enable_auto_correction: true,
            verbosity_level: 5,
            ..Default::default()
        };
        
        let mut validator = ShapeValidator::with_config(config);
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![3, 5]);
        
        let result = validator.validate_elementwise(&[&shape1, &shape2], "add");
        assert!(result.is_err());
        
        // Check that the error contains enhanced information
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Performance impact"));
        assert!(error_msg.contains("Memory optimization"));
    }

    #[test]
    fn test_performance_analysis() {
        let mut validator = ShapeValidator::new();
        let left = Shape::new(vec![1000, 1000]);
        let right = Shape::new(vec![1000, 1000]);
        
        let result = validator.validate_matmul(&left, &right);
        assert!(result.is_ok());
        
        // Test with incompatible shapes to trigger performance analysis
        let left_bad = Shape::new(vec![1000, 500]);
        let right_bad = Shape::new(vec![1000, 500]); // Wrong inner dimension
        
        let result = validator.validate_matmul(&left_bad, &right_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_corrections() {
        let config = ValidationConfig {
            enable_auto_correction: true,
            ..Default::default()
        };
        
        let mut validator = ShapeValidator::with_config(config);
        let left = Shape::new(vec![3, 4]);
        let right = Shape::new(vec![3, 5]); // Incompatible for matmul
        
        let result = validator.validate_matmul(&left, &right);
        assert!(result.is_err());
        
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Auto-corrections"));
        assert!(error_msg.contains("transpose"));
    }

    #[test]
    fn test_memory_suggestions() {
        let config = ValidationConfig {
            suggest_memory_optimizations: true,
            ..Default::default()
        };
        
        let mut validator = ShapeValidator::with_config(config);
        let original = Shape::new(vec![2, 3, 4]);
        
        // Invalid reshape to trigger memory suggestions
        let result = validator.validate_reshape(&original, &[5, 5]);
        assert!(result.is_err());
        
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Memory optimization"));
    }

    #[test]
    fn test_validation_config_customization() {
        let config = ValidationConfig {
            strict_mode: false,
            suggest_optimizations: false,
            show_visual_aids: false,
            analyze_performance: false,
            suggest_memory_optimizations: false,
            enable_auto_correction: false,
            verbosity_level: 1,
            ..Default::default()
        };
        
        let mut validator = ShapeValidator::with_config(config);
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![3, 5]);
        
        let result = validator.validate_elementwise(&[&shape1, &shape2], "add");
        assert!(result.is_err());
        
        // With minimal config, error should be less verbose
        let error_msg = format!("{}", result.unwrap_err());
        assert!(!error_msg.contains("Performance impact"));
        assert!(!error_msg.contains("Auto-corrections"));
    }

    #[test]
    fn test_operation_history_with_context() {
        let mut validator = ShapeValidator::new();
        let shape1 = Shape::new(vec![2, 3]);
        let shape2 = Shape::new(vec![3, 4]);
        let shape3 = Shape::new(vec![2, 4]);
        
        // Perform a series of operations
        let _result1 = validator.validate_matmul(&shape1, &shape2);
        let _result2 = validator.validate_elementwise(&[&shape3], "relu");
        
        // Now cause an error and check context
        let result = validator.validate_matmul(&shape1, &shape3);
        assert!(result.is_err());
        
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Recent operations"));
        assert!(error_msg.contains("matmul"));
        assert!(error_msg.contains("relu"));
    }

    #[test]
    fn test_convolution_detailed_validation() {
        let mut validator = ShapeValidator::new();
        let input = Shape::new(vec![1, 64, 224, 224]); // Batch, Channels, Height, Width
        let kernel = Shape::new(vec![128, 32, 3, 3]); // Out channels, Wrong in channels, H, W
        
        let result = validator.validate_convolution(
            &input, &kernel,
            &[1, 1], // stride
            &[1, 1], // padding
            &[1, 1], // dilation
        );
        
        assert!(result.is_err());
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Input channels"));
        assert!(error_msg.contains("kernel input channels"));
    }

    #[test]
    fn test_reduction_validation_edge_cases() {
        let mut validator = ShapeValidator::new();
        let input = Shape::new(vec![2, 3, 4]);
        
        // Valid reduction
        let result = validator.validate_reduction(&input, Some(&[1]), true, "sum");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![2, 1, 4]);
        
        // Invalid dimension
        let result = validator.validate_reduction(&input, Some(&[5]), false, "mean");
        assert!(result.is_err());
        
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("out of bounds"));
    }

    #[test]
    fn test_indexing_validation_comprehensive() {
        let mut validator = ShapeValidator::new();
        let shape = Shape::new(vec![10, 20, 30]);
        
        // Valid indexing
        let result = validator.validate_indexing(&shape, &[5, 10]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![30]);
        
        // Out of bounds positive index
        let result = validator.validate_indexing(&shape, &[15, 5]);
        assert!(result.is_err());
        
        // Out of bounds negative index
        let result = validator.validate_indexing(&shape, &[-15, 5]);
        assert!(result.is_err());
        
        // Too many indices
        let result = validator.validate_indexing(&shape, &[1, 2, 3, 4]);
        assert!(result.is_err());
    }
}