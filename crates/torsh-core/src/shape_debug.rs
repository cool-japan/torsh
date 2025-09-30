//! Shape Debugging Utilities
//!
//! This module provides comprehensive debugging and visualization tools for tensor shapes,
//! helping developers understand shape transformations, broadcasting behavior, and potential
//! shape-related errors in neural network operations.

use crate::error::{Result, TorshError};
use crate::{DType, Shape};
use std::collections::HashMap;
use std::fmt;

/// Comprehensive shape debugging and analysis tool
pub struct ShapeDebugger {
    /// Configuration for debugging behavior
    config: DebugConfig,
    /// History of shape transformations
    history: Vec<ShapeOperation>,
    /// Operation statistics
    stats: DebugStats,
}

/// Configuration for shape debugging
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Maximum number of operations to keep in history
    pub max_history: usize,
    /// Whether to show detailed operation breakdown
    pub show_detailed_ops: bool,
    /// Whether to show memory impact of operations
    pub show_memory_impact: bool,
    /// Whether to show broadcasting visualization
    pub show_broadcasting: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            max_history: 100,
            show_detailed_ops: true,
            show_memory_impact: true,
            show_broadcasting: true,
        }
    }
}

/// Record of a shape operation for debugging
#[derive(Debug, Clone)]
pub struct ShapeOperation {
    pub operation_type: OperationType,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub operation_name: String,
    pub timestamp: u64,
    pub memory_delta: Option<i64>,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Types of shape operations
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    /// Reshape operation (view change)
    Reshape,
    /// Transpose operation
    Transpose,
    /// Permute operation
    Permute,
    /// Broadcasting operation
    Broadcast,
    /// Concatenation along a dimension
    Concatenate,
    /// Element-wise operation
    ElementWise,
    /// Reduction operation (sum, mean, etc.)
    Reduction,
    /// Expansion operation
    Expand,
    /// Squeeze operation (remove size-1 dimensions)
    Squeeze,
    /// Unsqueeze operation (add size-1 dimensions)
    Unsqueeze,
    /// Slice operation
    Slice,
    /// Matrix multiplication
    MatMul,
    /// Convolution operation
    Convolution,
    /// Pooling operation
    Pooling,
    /// Custom operation
    Custom(String),
}

/// Statistics collected during shape debugging
#[derive(Debug, Default)]
pub struct DebugStats {
    pub total_operations: usize,
    pub failed_operations: usize,
    pub memory_allocated: i64,
    pub operation_counts: HashMap<String, usize>,
    pub max_tensor_size: usize,
    pub average_tensor_size: f64,
}

/// Result of shape analysis
#[derive(Debug)]
pub struct ShapeAnalysis {
    pub shape: Vec<usize>,
    pub total_elements: usize,
    pub memory_bytes: usize,
    pub dimensions: usize,
    pub contiguity: ContiguityInfo,
    pub broadcasting_compatibility: BroadcastCompatibility,
    pub common_issues: Vec<ShapeIssue>,
    pub optimization_suggestions: Vec<String>,
}

/// Information about tensor contiguity
#[derive(Debug)]
pub struct ContiguityInfo {
    pub is_contiguous: bool,
    pub memory_efficiency: f32,
    pub strides: Vec<usize>,
    pub layout_type: LayoutType,
}

/// Memory layout patterns
#[derive(Debug, PartialEq)]
pub enum LayoutType {
    /// C-style (row-major) layout
    CStyle,
    /// Fortran-style (column-major) layout
    FortranStyle,
    /// Strided layout
    Strided,
    /// Broadcast layout
    Broadcast,
}

/// Broadcasting compatibility information
#[derive(Debug)]
pub struct BroadcastCompatibility {
    pub is_broadcastable: bool,
    pub broadcast_dimensions: Vec<Option<usize>>,
    pub resulting_shape: Option<Vec<usize>>,
    pub memory_expansion_factor: f32,
}

/// Common shape-related issues
#[derive(Debug, Clone)]
pub enum ShapeIssue {
    /// Dimension mismatch in operations
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },
    /// Memory inefficient layout
    MemoryInefficient { efficiency: f32, suggestion: String },
    /// Broadcasting creates large memory expansion
    BroadcastExpansion {
        original_size: usize,
        expanded_size: usize,
        expansion_factor: f32,
    },
    /// Zero-sized dimensions
    ZeroDimensions { dimensions: Vec<usize> },
    /// Very large tensor
    LargeTensor { size: usize, recommendation: String },
    /// Potential numerical instability
    NumericalIssue {
        issue_type: String,
        suggestion: String,
    },
}

impl ShapeDebugger {
    /// Create a new shape debugger with default configuration
    pub fn new() -> Self {
        Self {
            config: DebugConfig::default(),
            history: Vec::new(),
            stats: DebugStats::default(),
        }
    }

    /// Create a new shape debugger with custom configuration
    pub fn with_config(config: DebugConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            stats: DebugStats::default(),
        }
    }

    /// Analyze a tensor shape comprehensively
    pub fn analyze_shape(&mut self, shape: &Shape, dtype: DType) -> Result<ShapeAnalysis> {
        let dims = shape.dims();
        let total_elements = shape.numel();
        let memory_bytes = total_elements * dtype.size_bytes();

        // Update statistics
        self.stats.total_operations += 1;
        self.stats.average_tensor_size = (self.stats.average_tensor_size
            * (self.stats.total_operations - 1) as f64
            + total_elements as f64)
            / self.stats.total_operations as f64;

        if total_elements > self.stats.max_tensor_size {
            self.stats.max_tensor_size = total_elements;
        }

        let contiguity = self.analyze_contiguity(dims);
        let broadcasting_compatibility = self.analyze_broadcasting_compatibility(dims);
        let common_issues = self.detect_common_issues(dims, total_elements, memory_bytes);
        let optimization_suggestions = self.generate_optimization_suggestions(dims, &common_issues);

        Ok(ShapeAnalysis {
            shape: dims.to_vec(),
            total_elements,
            memory_bytes,
            dimensions: dims.len(),
            contiguity,
            broadcasting_compatibility,
            common_issues,
            optimization_suggestions,
        })
    }

    /// Record a shape operation for debugging history
    pub fn record_operation(
        &mut self,
        op_type: OperationType,
        inputs: &[&Shape],
        output: &Shape,
        op_name: &str,
        success: bool,
        error: Option<&str>,
    ) {
        let input_shapes: Vec<Vec<usize>> = inputs.iter().map(|s| s.dims().to_vec()).collect();
        let memory_delta = self.calculate_memory_delta(inputs, output);

        let operation = ShapeOperation {
            operation_type: op_type.clone(),
            input_shapes,
            output_shape: output.dims().to_vec(),
            operation_name: op_name.to_string(),
            timestamp: self.get_timestamp(),
            memory_delta: Some(memory_delta),
            success,
            error_message: error.map(|s| s.to_string()),
        };

        // Update statistics
        self.stats.total_operations += 1;
        if !success {
            self.stats.failed_operations += 1;
        }
        self.stats.memory_allocated += memory_delta;

        let op_key = format!("{op_type:?}");
        *self.stats.operation_counts.entry(op_key).or_insert(0) += 1;

        // Add to history
        self.history.push(operation);

        // Maintain history size
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get operation history
    pub fn get_history(&self) -> &[ShapeOperation] {
        &self.history
    }

    /// Get debugging statistics
    pub fn get_stats(&self) -> &DebugStats {
        &self.stats
    }

    /// Clear history and reset statistics
    pub fn reset(&mut self) {
        self.history.clear();
        self.stats = DebugStats::default();
    }

    /// Visualize shape transformation
    pub fn visualize_transformation(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
        operation: &str,
    ) -> String {
        let mut visualization = String::new();

        visualization.push_str(&format!("=== Shape Transformation: {operation} ===\n"));

        for (i, input_shape) in input_shapes.iter().enumerate() {
            visualization.push_str(&format!(
                "Input {}: {:?} ({} elements)\n",
                i + 1,
                input_shape,
                input_shape.iter().product::<usize>()
            ));
        }

        visualization.push_str(&format!(
            "Output: {:?} ({} elements)\n",
            output_shape,
            output_shape.iter().product::<usize>()
        ));

        // Add ASCII art visualization for simple cases
        if input_shapes.len() == 1 && input_shapes[0].len() <= 3 && output_shape.len() <= 3 {
            visualization.push_str("\nVisualization:\n");
            visualization
                .push_str(&self.create_ascii_shape_diagram(&input_shapes[0], output_shape));
        }

        visualization
    }

    /// Check if two shapes are compatible for broadcasting
    pub fn check_broadcast_compatibility(
        &self,
        shape1: &[usize],
        shape2: &[usize],
    ) -> BroadcastCompatibility {
        let max_dims = shape1.len().max(shape2.len());
        let mut result_shape = Vec::new();
        let mut is_compatible = true;
        let mut broadcast_dims = Vec::new();

        // Pad shorter shape with 1s on the left
        let padded_shape1 = self.pad_shape_left(shape1, max_dims);
        let padded_shape2 = self.pad_shape_left(shape2, max_dims);

        for i in 0..max_dims {
            let dim1 = padded_shape1[i];
            let dim2 = padded_shape2[i];

            if dim1 == dim2 {
                result_shape.push(dim1);
                broadcast_dims.push(Some(dim1));
            } else if dim1 == 1 {
                result_shape.push(dim2);
                broadcast_dims.push(Some(dim2));
            } else if dim2 == 1 {
                result_shape.push(dim1);
                broadcast_dims.push(Some(dim1));
            } else {
                is_compatible = false;
                break;
            }
        }

        let original_size1: usize = shape1.iter().product();
        let original_size2: usize = shape2.iter().product();
        let result_size: usize = if is_compatible {
            result_shape.iter().product()
        } else {
            0
        };

        let expansion_factor = if original_size1.max(original_size2) > 0 {
            result_size as f32 / (original_size1.max(original_size2) as f32)
        } else {
            1.0
        };

        BroadcastCompatibility {
            is_broadcastable: is_compatible,
            broadcast_dimensions: if is_compatible {
                broadcast_dims
            } else {
                Vec::new()
            },
            resulting_shape: if is_compatible {
                Some(result_shape)
            } else {
                None
            },
            memory_expansion_factor: expansion_factor,
        }
    }

    /// Generate a detailed shape debugging report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Shape Debugging Report ===\n\n");

        // Statistics summary
        report.push_str("Statistics:\n");
        let total_ops = self.stats.total_operations;
        report.push_str(&format!("  Total Operations: {total_ops}\n"));
        let failed_ops = self.stats.failed_operations;
        report.push_str(&format!("  Failed Operations: {failed_ops}\n"));
        report.push_str(&format!(
            "  Success Rate: {:.2}%\n",
            if self.stats.total_operations > 0 {
                100.0 * (self.stats.total_operations - self.stats.failed_operations) as f32
                    / self.stats.total_operations as f32
            } else {
                100.0
            }
        ));
        let memory_allocated = self.stats.memory_allocated;
        report.push_str(&format!("  Memory Allocated: {memory_allocated} bytes\n"));
        let max_tensor_size = self.stats.max_tensor_size;
        report.push_str(&format!("  Max Tensor Size: {max_tensor_size} elements\n"));
        let avg_tensor_size = self.stats.average_tensor_size;
        report.push_str(&format!(
            "  Average Tensor Size: {avg_tensor_size:.2} elements\n"
        ));

        // Operation breakdown
        if !self.stats.operation_counts.is_empty() {
            report.push_str("\nOperation Breakdown:\n");
            for (op_type, count) in &self.stats.operation_counts {
                report.push_str(&format!("  {op_type}: {count} times\n"));
            }
        }

        // Recent failures
        let recent_failures: Vec<_> = self
            .history
            .iter()
            .filter(|op| !op.success)
            .rev()
            .take(5)
            .collect();

        if !recent_failures.is_empty() {
            report.push_str("\nRecent Failures:\n");
            for failure in recent_failures {
                let op_name = &failure.operation_name;
                let op_type = format!("{:?}", failure.operation_type);
                let error_msg = failure.error_message.as_deref().unwrap_or("Unknown error");
                report.push_str(&format!("  {op_name} - {op_type}: {error_msg}\n"));
            }
        }

        // Memory usage patterns
        let memory_operations: Vec<_> = self
            .history
            .iter()
            .filter(|op| op.memory_delta.is_some())
            .collect();

        if !memory_operations.is_empty() {
            let total_memory_delta: i64 = memory_operations
                .iter()
                .map(|op| op.memory_delta.unwrap_or(0))
                .sum();

            report.push_str("\nMemory Usage:\n");
            report.push_str(&format!(
                "  Net Memory Change: {total_memory_delta} bytes\n"
            ));

            let largest_allocation = memory_operations
                .iter()
                .map(|op| op.memory_delta.unwrap_or(0))
                .max()
                .unwrap_or(0);

            report.push_str(&format!(
                "  Largest Single Allocation: {largest_allocation} bytes\n"
            ));
        }

        report
    }

    fn analyze_contiguity(&self, dims: &[usize]) -> ContiguityInfo {
        let strides = self.compute_c_strides(dims);
        let is_contiguous = self.check_contiguity(dims, &strides);
        let efficiency = self.compute_memory_efficiency(dims, &strides);
        let layout_type = self.determine_layout_type(&strides);

        ContiguityInfo {
            is_contiguous,
            memory_efficiency: efficiency,
            strides,
            layout_type,
        }
    }

    fn compute_c_strides(&self, dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return Vec::new();
        }

        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    fn check_contiguity(&self, dims: &[usize], strides: &[usize]) -> bool {
        let expected_strides = self.compute_c_strides(dims);
        strides == expected_strides
    }

    fn compute_memory_efficiency(&self, dims: &[usize], strides: &[usize]) -> f32 {
        if dims.is_empty() || strides.is_empty() {
            return 1.0;
        }

        let total_elements: usize = dims.iter().product();
        let memory_span = if let Some(&max_stride) = strides.iter().max() {
            max_stride + 1
        } else {
            total_elements
        };

        if memory_span == 0 {
            1.0
        } else {
            total_elements as f32 / memory_span as f32
        }
    }

    fn determine_layout_type(&self, strides: &[usize]) -> LayoutType {
        if strides.is_empty() {
            return LayoutType::CStyle;
        }

        // Check for C-style (decreasing strides)
        let mut is_c_style = true;
        for i in 0..strides.len().saturating_sub(1) {
            if strides[i] < strides[i + 1] {
                is_c_style = false;
                break;
            }
        }

        if is_c_style {
            return LayoutType::CStyle;
        }

        // Check for Fortran-style (increasing strides)
        let mut is_fortran_style = true;
        for i in 0..strides.len().saturating_sub(1) {
            if strides[i] > strides[i + 1] {
                is_fortran_style = false;
                break;
            }
        }

        if is_fortran_style {
            LayoutType::FortranStyle
        } else {
            LayoutType::Strided
        }
    }

    fn analyze_broadcasting_compatibility(&self, dims: &[usize]) -> BroadcastCompatibility {
        // For single tensor analysis, check if it's broadcast-friendly
        let has_ones = dims.contains(&1);
        let max_dim = dims.iter().max().copied().unwrap_or(1);

        BroadcastCompatibility {
            is_broadcastable: true,
            broadcast_dimensions: dims.iter().map(|&d| Some(d)).collect(),
            resulting_shape: Some(dims.to_vec()),
            memory_expansion_factor: if has_ones { max_dim as f32 } else { 1.0 },
        }
    }

    fn detect_common_issues(
        &self,
        dims: &[usize],
        total_elements: usize,
        _memory_bytes: usize,
    ) -> Vec<ShapeIssue> {
        let mut issues = Vec::new();

        // Check for zero dimensions
        if dims.contains(&0) {
            issues.push(ShapeIssue::ZeroDimensions {
                dimensions: dims.to_vec(),
            });
        }

        // Check for very large tensors
        if total_elements > 100_000_000 {
            issues.push(ShapeIssue::LargeTensor {
                size: total_elements,
                recommendation: "Consider using tensor chunking or distributed computation"
                    .to_string(),
            });
        }

        // Check for memory efficiency
        let strides = self.compute_c_strides(dims);
        let efficiency = self.compute_memory_efficiency(dims, &strides);
        if efficiency < 0.5 {
            issues.push(ShapeIssue::MemoryInefficient {
                efficiency,
                suggestion: "Consider reshaping or using a more memory-efficient layout"
                    .to_string(),
            });
        }

        // Check for excessive broadcasting
        let has_many_ones = dims.iter().filter(|&&d| d == 1).count() > dims.len() / 2;
        if has_many_ones && dims.len() > 2 {
            let max_dim = dims.iter().max().copied().unwrap_or(1);
            issues.push(ShapeIssue::BroadcastExpansion {
                original_size: total_elements,
                expanded_size: max_dim.pow(dims.len() as u32),
                expansion_factor: max_dim as f32,
            });
        }

        issues
    }

    fn generate_optimization_suggestions(
        &self,
        dims: &[usize],
        issues: &[ShapeIssue],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        for issue in issues {
            match issue {
                ShapeIssue::MemoryInefficient { efficiency, .. } => {
                    if *efficiency < 0.3 {
                        suggestions.push(
                            "Consider using torch.contiguous() to improve memory layout"
                                .to_string(),
                        );
                    }
                }
                ShapeIssue::LargeTensor { .. } => {
                    suggestions.push(
                        "Consider using gradient checkpointing to reduce memory usage".to_string(),
                    );
                    suggestions.push(
                        "Use torch.chunk() or torch.split() for batch processing".to_string(),
                    );
                }
                ShapeIssue::BroadcastExpansion {
                    expansion_factor, ..
                } => {
                    if *expansion_factor > 10.0 {
                        suggestions.push(
                            "Avoid large broadcasting - consider explicit expansion".to_string(),
                        );
                    }
                }
                ShapeIssue::ZeroDimensions { .. } => {
                    suggestions.push(
                        "Check tensor creation - zero dimensions may cause runtime errors"
                            .to_string(),
                    );
                }
                _ => {}
            }
        }

        // General suggestions based on shape
        if dims.len() > 4 {
            suggestions.push(
                "High-dimensional tensor - consider dimension reduction techniques".to_string(),
            );
        }

        if dims.contains(&1) {
            suggestions
                .push("Consider using squeeze() to remove unnecessary dimensions".to_string());
        }

        suggestions
    }

    fn calculate_memory_delta(&self, inputs: &[&Shape], output: &Shape) -> i64 {
        let input_elements: usize = inputs.iter().map(|s| s.numel()).sum();
        let output_elements = output.numel();
        (output_elements as i64 - input_elements as i64) * 4 // Assuming 32-bit elements
    }

    fn get_timestamp(&self) -> u64 {
        // Simplified timestamp - in real implementation would use proper time
        self.stats.total_operations as u64
    }

    fn create_ascii_shape_diagram(&self, input_shape: &[usize], output_shape: &[usize]) -> String {
        let mut diagram = String::new();

        // Simple ASCII representation for 1D, 2D, 3D tensors
        match (input_shape.len(), output_shape.len()) {
            (1, 1) => {
                diagram.push_str(&format!("[{}] → [{}]\n", input_shape[0], output_shape[0]));
            }
            (2, 2) => {
                diagram.push_str(&format!(
                    "[{} × {}] → [{} × {}]\n",
                    input_shape[0], input_shape[1], output_shape[0], output_shape[1]
                ));
                diagram.push_str("┌─────┐    ┌─────┐\n");
                diagram.push_str("│     │ ➤  │     │\n");
                diagram.push_str("└─────┘    └─────┘\n");
            }
            (3, 3) => {
                diagram.push_str(&format!(
                    "[{} × {} × {}] → [{} × {} × {}]\n",
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    output_shape[0],
                    output_shape[1],
                    output_shape[2]
                ));
                diagram.push_str("┌─────┐    ┌─────┐\n");
                diagram.push_str("│ ┌─┐ │    │ ┌─┐ │\n");
                diagram.push_str("│ └─┘ │ ➤  │ └─┘ │\n");
                diagram.push_str("└─────┘    └─────┘\n");
            }
            _ => {
                diagram.push_str(&format!("{input_shape:?} → {output_shape:?}\n"));
            }
        }

        diagram
    }

    fn pad_shape_left(&self, shape: &[usize], target_len: usize) -> Vec<usize> {
        if shape.len() >= target_len {
            shape.to_vec()
        } else {
            let mut padded = vec![1; target_len - shape.len()];
            padded.extend_from_slice(shape);
            padded
        }
    }
}

impl Default for ShapeDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ShapeAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Shape Analysis ===")?;
        writeln!(f, "Shape: {:?}", self.shape)?;
        writeln!(f, "Elements: {}", self.total_elements)?;
        writeln!(
            f,
            "Memory: {} bytes ({:.2} MB)",
            self.memory_bytes,
            self.memory_bytes as f64 / 1_048_576.0
        )?;
        writeln!(f, "Dimensions: {}", self.dimensions)?;

        writeln!(f, "\nContiguity:")?;
        writeln!(f, "  Is Contiguous: {}", self.contiguity.is_contiguous)?;
        writeln!(
            f,
            "  Memory Efficiency: {:.2}%",
            self.contiguity.memory_efficiency * 100.0
        )?;
        writeln!(f, "  Layout Type: {:?}", self.contiguity.layout_type)?;
        writeln!(f, "  Strides: {:?}", self.contiguity.strides)?;

        writeln!(f, "\nBroadcasting:")?;
        writeln!(
            f,
            "  Is Broadcastable: {}",
            self.broadcasting_compatibility.is_broadcastable
        )?;
        writeln!(
            f,
            "  Memory Expansion: {:.2}x",
            self.broadcasting_compatibility.memory_expansion_factor
        )?;

        if !self.common_issues.is_empty() {
            writeln!(f, "\nIssues Found:")?;
            for issue in &self.common_issues {
                writeln!(f, "  - {issue:?}")?;
            }
        }

        if !self.optimization_suggestions.is_empty() {
            writeln!(f, "\nOptimization Suggestions:")?;
            for suggestion in &self.optimization_suggestions {
                writeln!(f, "  - {suggestion}")?;
            }
        }

        Ok(())
    }
}

/// Utility functions for shape debugging
pub mod shape_utils {
    use super::*;

    /// Quick shape analysis with default configuration
    pub fn quick_analyze(shape: &Shape, dtype: DType) -> String {
        let mut debugger = ShapeDebugger::new();
        match debugger.analyze_shape(shape, dtype) {
            Ok(analysis) => format!("{analysis}"),
            Err(e) => format!("Analysis failed: {e:?}"),
        }
    }

    /// Check if shapes are compatible for element-wise operations
    pub fn check_elementwise_compatibility(shapes: &[&Shape]) -> Result<Vec<usize>> {
        if shapes.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No shapes provided".to_string(),
            ));
        }

        let debugger = ShapeDebugger::new();
        let mut result_shape = shapes[0].dims().to_vec();

        for shape in shapes.iter().skip(1) {
            let compat = debugger.check_broadcast_compatibility(&result_shape, shape.dims());
            if !compat.is_broadcastable {
                return Err(TorshError::ShapeMismatch {
                    expected: result_shape,
                    got: shape.dims().to_vec(),
                });
            }
            result_shape = compat.resulting_shape.unwrap_or(result_shape);
        }

        Ok(result_shape)
    }

    /// Suggest optimal tensor layout for given operations
    pub fn suggest_layout_optimization(shape: &Shape, operations: &[&str]) -> Vec<String> {
        let mut suggestions = Vec::new();
        let dims = shape.dims();

        // Analyze operations to suggest optimizations
        let has_matmul = operations
            .iter()
            .any(|op| op.contains("matmul") || op.contains("linear"));
        let has_conv = operations.iter().any(|op| op.contains("conv"));
        let has_reduction = operations
            .iter()
            .any(|op| op.contains("sum") || op.contains("mean"));

        if has_matmul && dims.len() >= 2 {
            suggestions.push("Consider using contiguous layout for matrix operations".to_string());
            if dims[dims.len() - 1] < dims[dims.len() - 2] {
                suggestions
                    .push("Consider transposing for better cache locality in matmul".to_string());
            }
        }

        if has_conv && dims.len() == 4 && !dims[1].is_multiple_of(8) {
            suggestions.push(
                "Consider padding channels to multiple of 8 for SIMD optimization".to_string(),
            );
        }

        if has_reduction && dims.contains(&1) {
            suggestions
                .push("Consider squeezing singleton dimensions before reduction".to_string());
        }

        if suggestions.is_empty() {
            suggestions.push("Shape appears well-optimized for given operations".to_string());
        }

        suggestions
    }

    /// Estimate memory bandwidth requirements for shape operations
    pub fn estimate_memory_bandwidth(
        input_shapes: &[&Shape],
        output_shape: &Shape,
        operation: &str,
    ) -> (f64, String) {
        let input_bytes: usize = input_shapes.iter().map(|s| s.numel() * 4).sum(); // Assuming f32
        let output_bytes = output_shape.numel() * 4;
        let total_bytes = input_bytes + output_bytes;

        // Rough estimates based on operation type
        let (bandwidth_gb_s, description) = match operation {
            op if op.contains("matmul") => {
                let flops = if input_shapes.len() >= 2 {
                    let m = input_shapes[0].dims()[input_shapes[0].dims().len() - 2];
                    let n = input_shapes[1].dims()[input_shapes[1].dims().len() - 1];
                    let k = input_shapes[0].dims()[input_shapes[0].dims().len() - 1];
                    2 * m * n * k
                } else {
                    total_bytes / 4
                };
                let arithmetic_intensity = flops as f64 / total_bytes as f64;
                (
                    total_bytes as f64 / 1e9,
                    format!("Arithmetic intensity: {arithmetic_intensity:.2} FLOP/byte"),
                )
            }
            op if op.contains("conv") => (
                total_bytes as f64 / 1e9 * 1.5,
                "Convolution with spatial locality".to_string(),
            ),
            op if op.contains("elementwise") => (
                total_bytes as f64 / 1e9,
                "Memory-bound elementwise operation".to_string(),
            ),
            _ => (
                total_bytes as f64 / 1e9,
                "General memory access pattern".to_string(),
            ),
        };

        (bandwidth_gb_s, description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    #[test]
    fn test_shape_debugger_creation() {
        let debugger = ShapeDebugger::new();
        assert_eq!(debugger.stats.total_operations, 0);
    }

    #[test]
    fn test_shape_analysis() {
        let mut debugger = ShapeDebugger::new();
        let shape = Shape::new(vec![2, 3, 4]);

        let result = debugger.analyze_shape(&shape, DType::F32);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.shape, vec![2, 3, 4]);
        assert_eq!(analysis.total_elements, 24);
        assert_eq!(analysis.dimensions, 3);
    }

    #[test]
    fn test_broadcast_compatibility() {
        let debugger = ShapeDebugger::new();

        // Compatible shapes
        let compat = debugger.check_broadcast_compatibility(&[3, 1, 4], &[1, 2, 4]);
        assert!(compat.is_broadcastable);
        assert_eq!(compat.resulting_shape, Some(vec![3, 2, 4]));

        // Incompatible shapes
        let incompat = debugger.check_broadcast_compatibility(&[3, 4], &[2, 3]);
        assert!(!incompat.is_broadcastable);
    }

    #[test]
    fn test_operation_recording() {
        let mut debugger = ShapeDebugger::new();
        let input = Shape::new(vec![2, 3]);
        let output = Shape::new(vec![3, 2]);

        debugger.record_operation(
            OperationType::Transpose,
            &[&input],
            &output,
            "transpose",
            true,
            None,
        );

        assert_eq!(debugger.stats.total_operations, 1);
        assert_eq!(debugger.history.len(), 1);
        assert_eq!(debugger.history[0].operation_name, "transpose");
    }

    #[test]
    fn test_contiguity_analysis() {
        let debugger = ShapeDebugger::new();
        let dims = [2, 3, 4];
        let contiguity = debugger.analyze_contiguity(&dims);

        assert!(contiguity.is_contiguous);
        assert_eq!(contiguity.layout_type, LayoutType::CStyle);
        assert_eq!(contiguity.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_shape_visualization() {
        let debugger = ShapeDebugger::new();
        let input_shapes = vec![vec![2, 3]];
        let output_shape = vec![3, 2];

        let visualization =
            debugger.visualize_transformation(&input_shapes, &output_shape, "transpose");
        assert!(visualization.contains("transpose"));
        assert!(visualization.contains("[2, 3]"));
        assert!(visualization.contains("[3, 2]"));
    }

    #[test]
    fn test_quick_analyze_utility() {
        let shape = Shape::new(vec![5, 5]);
        let result = shape_utils::quick_analyze(&shape, DType::F32);
        assert!(result.contains("Shape: [5, 5]"));
        assert!(result.contains("Elements: 25"));
    }

    #[test]
    fn test_elementwise_compatibility() {
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![1, 2, 4]);
        let shapes = vec![&shape1, &shape2];

        let result = shape_utils::check_elementwise_compatibility(&shapes);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![3, 2, 4]);
    }
}
