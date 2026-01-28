//! Shape Inference Debugging with Detailed Traces
//!
//! This module provides comprehensive debugging capabilities for shape inference operations,
//! helping developers understand how tensor shapes are computed and why shape mismatches occur.
//!
//! # Features
//!
//! - **Detailed trace logging**: Record every step of shape inference with explanations
//! - **Shape compatibility checking**: Validate shape compatibility with detailed error messages
//! - **Broadcasting visualization**: Visualize how broadcasting affects shapes
//! - **Shape transformation tracking**: Track how shapes change through operations
//! - **Interactive debugging**: Step-by-step shape inference with intermediate results
//! - **Error diagnosis**: Detailed error messages explaining shape mismatches
//!
//! # Example
//!
//! ```rust
//! use torsh_tensor::shape_inference_debugger::*;
//! use torsh_core::shape::Shape;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut debugger = ShapeInferenceDebugger::new();
//!
//! // Enable tracing
//! debugger.enable_tracing(true);
//!
//! // Infer shape for matmul operation
//! let a_shape = Shape::new(vec![2, 3]);
//! let b_shape = Shape::new(vec![3, 4]);
//! let result_shape = debugger.infer_matmul_shape(&a_shape, &b_shape)?;
//!
//! // Get detailed trace
//! let trace = debugger.get_trace();
//! println!("Shape inference trace:\n{}", trace);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use torsh_core::{
    error::{Result, TorshError},
    shape::Shape,
};

/// Type of shape operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeOperation {
    /// Element-wise operation
    ElementWise,
    /// Matrix multiplication
    MatMul,
    /// Convolution operation
    Conv,
    /// Pooling operation
    Pool,
    /// Reshape operation
    Reshape,
    /// Transpose operation
    Transpose,
    /// Concatenation
    Concatenate,
    /// Stack operation
    Stack,
    /// Broadcast operation
    Broadcast,
    /// Reduction operation
    Reduce,
    /// Custom operation
    Custom(String),
}

impl fmt::Display for ShapeOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeOperation::ElementWise => write!(f, "ElementWise"),
            ShapeOperation::MatMul => write!(f, "MatMul"),
            ShapeOperation::Conv => write!(f, "Convolution"),
            ShapeOperation::Pool => write!(f, "Pooling"),
            ShapeOperation::Reshape => write!(f, "Reshape"),
            ShapeOperation::Transpose => write!(f, "Transpose"),
            ShapeOperation::Concatenate => write!(f, "Concatenate"),
            ShapeOperation::Stack => write!(f, "Stack"),
            ShapeOperation::Broadcast => write!(f, "Broadcast"),
            ShapeOperation::Reduce => write!(f, "Reduce"),
            ShapeOperation::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// A single step in shape inference trace
#[derive(Debug, Clone)]
pub struct ShapeTraceStep {
    /// Step number
    pub step: usize,
    /// Operation being performed
    pub operation: ShapeOperation,
    /// Input shapes
    pub input_shapes: Vec<Shape>,
    /// Output shape
    pub output_shape: Option<Shape>,
    /// Explanation of the inference
    pub explanation: String,
    /// Whether this step succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl fmt::Display for ShapeTraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Step {}: {}", self.step, self.operation)?;
        writeln!(f, "  Input shapes:")?;
        for (i, shape) in self.input_shapes.iter().enumerate() {
            writeln!(f, "    [{}] {:?}", i, shape.dims())?;
        }
        if let Some(output) = &self.output_shape {
            writeln!(f, "  Output shape: {:?}", output.dims())?;
        }
        writeln!(f, "  Explanation: {}", self.explanation)?;
        if let Some(error) = &self.error {
            writeln!(f, "  Error: {}", error)?;
        }
        writeln!(f, "  Status: {}", if self.success { "✓" } else { "✗" })
    }
}

/// Configuration for shape inference debugging
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Whether to enable detailed tracing
    pub tracing_enabled: bool,
    /// Whether to validate shapes automatically
    pub auto_validate: bool,
    /// Maximum number of trace steps to keep
    pub max_trace_steps: usize,
    /// Whether to include visual diagrams
    pub include_visuals: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            tracing_enabled: true,
            auto_validate: true,
            max_trace_steps: 1000,
            include_visuals: false,
        }
    }
}

/// Main shape inference debugger
pub struct ShapeInferenceDebugger {
    /// Configuration
    config: Arc<RwLock<DebugConfig>>,
    /// Trace of shape inference steps
    trace: Arc<RwLock<Vec<ShapeTraceStep>>>,
    /// Current step counter
    step_counter: Arc<RwLock<usize>>,
    /// Named shapes for reference
    named_shapes: Arc<RwLock<HashMap<String, Shape>>>,
}

impl ShapeInferenceDebugger {
    /// Create a new shape inference debugger
    pub fn new() -> Self {
        Self::with_config(DebugConfig::default())
    }

    /// Create a debugger with custom configuration
    pub fn with_config(config: DebugConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            trace: Arc::new(RwLock::new(Vec::new())),
            step_counter: Arc::new(RwLock::new(0)),
            named_shapes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enable or disable tracing
    pub fn enable_tracing(&mut self, enabled: bool) {
        self.config
            .write()
            .expect("lock should not be poisoned")
            .tracing_enabled = enabled;
    }

    /// Register a named shape for reference
    pub fn register_shape(&mut self, name: impl Into<String>, shape: Shape) {
        self.named_shapes
            .write()
            .unwrap()
            .insert(name.into(), shape);
    }

    /// Add a trace step
    fn add_trace_step(&self, step: ShapeTraceStep) {
        let config = self.config.read().expect("lock should not be poisoned");
        if !config.tracing_enabled {
            return;
        }
        drop(config);

        let mut trace = self.trace.write().expect("lock should not be poisoned");
        trace.push(step);

        // Trim if needed
        let config = self.config.read().expect("lock should not be poisoned");
        if trace.len() > config.max_trace_steps {
            trace.remove(0);
        }
    }

    /// Get the next step number
    fn next_step(&self) -> usize {
        let mut counter = self
            .step_counter
            .write()
            .expect("lock should not be poisoned");
        let step = *counter;
        *counter += 1;
        step
    }

    /// Infer shape for element-wise operation
    pub fn infer_elementwise_shape(&self, shapes: &[Shape]) -> Result<Shape> {
        let step = self.next_step();

        if shapes.is_empty() {
            let error = "Element-wise operation requires at least one input".to_string();
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::ElementWise,
                input_shapes: shapes.to_vec(),
                output_shape: None,
                explanation: "Checking input shapes".to_string(),
                success: false,
                error: Some(error.clone()),
            });
            return Err(TorshError::InvalidShape(error));
        }

        if shapes.len() == 1 {
            let output = shapes[0].clone();
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::ElementWise,
                input_shapes: shapes.to_vec(),
                output_shape: Some(output.clone()),
                explanation: "Single input - output shape matches input".to_string(),
                success: true,
                error: None,
            });
            return Ok(output);
        }

        // Check if all shapes are identical
        let first_shape = &shapes[0];
        let all_identical = shapes.iter().all(|s| s.dims() == first_shape.dims());

        if all_identical {
            let output = first_shape.clone();
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::ElementWise,
                input_shapes: shapes.to_vec(),
                output_shape: Some(output.clone()),
                explanation: "All input shapes are identical".to_string(),
                success: true,
                error: None,
            });
            return Ok(output);
        }

        // Try broadcasting
        let broadcast_result = self.infer_broadcast_shape(shapes);
        match broadcast_result {
            Ok(output) => {
                self.add_trace_step(ShapeTraceStep {
                    step,
                    operation: ShapeOperation::ElementWise,
                    input_shapes: shapes.to_vec(),
                    output_shape: Some(output.clone()),
                    explanation: "Shapes are compatible through broadcasting".to_string(),
                    success: true,
                    error: None,
                });
                Ok(output)
            }
            Err(e) => {
                self.add_trace_step(ShapeTraceStep {
                    step,
                    operation: ShapeOperation::ElementWise,
                    input_shapes: shapes.to_vec(),
                    output_shape: None,
                    explanation: "Shapes are not compatible".to_string(),
                    success: false,
                    error: Some(e.to_string()),
                });
                Err(e)
            }
        }
    }

    /// Infer shape for matrix multiplication
    pub fn infer_matmul_shape(&self, a: &Shape, b: &Shape) -> Result<Shape> {
        let step = self.next_step();
        let input_shapes = vec![a.clone(), b.clone()];

        let a_dims = a.dims();
        let b_dims = b.dims();

        // Check dimensions
        if a_dims.len() < 2 || b_dims.len() < 2 {
            let error = format!(
                "Matrix multiplication requires at least 2D tensors, got shapes {:?} and {:?}",
                a_dims, b_dims
            );
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::MatMul,
                input_shapes,
                output_shape: None,
                explanation: "Checking input dimensionality".to_string(),
                success: false,
                error: Some(error.clone()),
            });
            return Err(TorshError::InvalidShape(error));
        }

        // Get the matrix dimensions
        let a_rows = a_dims[a_dims.len() - 2];
        let a_cols = a_dims[a_dims.len() - 1];
        let b_rows = b_dims[b_dims.len() - 2];
        let b_cols = b_dims[b_dims.len() - 1];

        // Check compatibility
        if a_cols != b_rows {
            let error = format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {}) - inner dimensions {} and {} do not match",
                a_rows, a_cols, b_rows, b_cols, a_cols, b_rows
            );
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::MatMul,
                input_shapes,
                output_shape: None,
                explanation: format!(
                    "Checking inner dimension compatibility: {} vs {}",
                    a_cols, b_rows
                ),
                success: false,
                error: Some(error.clone()),
            });
            return Err(TorshError::InvalidShape(error));
        }

        // Build output shape
        let mut output_dims = Vec::new();

        // Handle batch dimensions
        if a_dims.len() > 2 || b_dims.len() > 2 {
            let max_batch_dims = std::cmp::max(a_dims.len() - 2, b_dims.len() - 2);
            for i in 0..max_batch_dims {
                let a_idx = a_dims.len().saturating_sub(3 + i);
                let b_idx = b_dims.len().saturating_sub(3 + i);

                let a_dim = if a_idx < a_dims.len() - 2 {
                    a_dims[a_idx]
                } else {
                    1
                };

                let b_dim = if b_idx < b_dims.len() - 2 {
                    b_dims[b_idx]
                } else {
                    1
                };

                if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
                    let error = format!(
                        "Batch dimension mismatch at position {}: {} vs {}",
                        i, a_dim, b_dim
                    );
                    self.add_trace_step(ShapeTraceStep {
                        step,
                        operation: ShapeOperation::MatMul,
                        input_shapes,
                        output_shape: None,
                        explanation: format!("Checking batch dimension {}", i),
                        success: false,
                        error: Some(error.clone()),
                    });
                    return Err(TorshError::InvalidShape(error));
                }

                output_dims.push(std::cmp::max(a_dim, b_dim));
            }
        }

        // Add matrix dimensions
        output_dims.push(a_rows);
        output_dims.push(b_cols);

        let output = Shape::new(output_dims);

        self.add_trace_step(ShapeTraceStep {
            step,
            operation: ShapeOperation::MatMul,
            input_shapes,
            output_shape: Some(output.clone()),
            explanation: format!(
                "Matrix multiplication: ({}, {}) @ ({}, {}) = ({}, {})",
                a_rows, a_cols, b_rows, b_cols, a_rows, b_cols
            ),
            success: true,
            error: None,
        });

        Ok(output)
    }

    /// Infer shape for broadcasting
    pub fn infer_broadcast_shape(&self, shapes: &[Shape]) -> Result<Shape> {
        let step = self.next_step();

        if shapes.is_empty() {
            let error = "Broadcasting requires at least one shape".to_string();
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::Broadcast,
                input_shapes: shapes.to_vec(),
                output_shape: None,
                explanation: "Checking number of inputs".to_string(),
                success: false,
                error: Some(error.clone()),
            });
            return Err(TorshError::InvalidShape(error));
        }

        if shapes.len() == 1 {
            let output = shapes[0].clone();
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::Broadcast,
                input_shapes: shapes.to_vec(),
                output_shape: Some(output.clone()),
                explanation: "Single shape - no broadcasting needed".to_string(),
                success: true,
                error: None,
            });
            return Ok(output);
        }

        // Find maximum number of dimensions
        let max_ndim = shapes
            .iter()
            .map(|s| s.dims().len())
            .max()
            .expect("reduction should succeed");

        // Build output shape by checking each dimension (from right to left for broadcasting)
        let mut output_dims = Vec::with_capacity(max_ndim);
        let mut explanations = Vec::new();

        for dim_idx in (0..max_ndim).rev() {
            let mut dim_size = 1;
            let mut conflict = false;
            let mut dim_sources = Vec::new();

            for (shape_idx, shape) in shapes.iter().enumerate() {
                let dims = shape.dims();
                // Access dimensions from the right
                let pos_from_right = max_ndim - 1 - dim_idx;

                let current_dim = if pos_from_right < dims.len() {
                    dims[dims.len() - 1 - pos_from_right]
                } else {
                    1
                };

                if current_dim != 1 {
                    if dim_size == 1 {
                        dim_size = current_dim;
                        dim_sources.push((shape_idx, current_dim));
                    } else if current_dim != dim_size {
                        conflict = true;
                        dim_sources.push((shape_idx, current_dim));
                    }
                }
            }

            if conflict {
                let error = format!(
                    "Broadcasting conflict at dimension {}: incompatible sizes {:?}",
                    dim_idx,
                    dim_sources.iter().map(|(_, d)| d).collect::<Vec<_>>()
                );
                self.add_trace_step(ShapeTraceStep {
                    step,
                    operation: ShapeOperation::Broadcast,
                    input_shapes: shapes.to_vec(),
                    output_shape: None,
                    explanation: format!("Checking dimension {}", dim_idx),
                    success: false,
                    error: Some(error.clone()),
                });
                return Err(TorshError::InvalidShape(error));
            }

            output_dims.insert(0, dim_size); // Insert at front since we're going right to left
            if dim_sources.len() > 1 {
                explanations.insert(
                    0,
                    format!(
                        "Dim {}: broadcast {} sources to size {}",
                        dim_idx,
                        dim_sources.len(),
                        dim_size
                    ),
                );
            } else if !dim_sources.is_empty() {
                explanations.insert(
                    0,
                    format!("Dim {}: size {} (no broadcast)", dim_idx, dim_size),
                );
            } else {
                explanations.insert(0, format!("Dim {}: size 1 (all implicit)", dim_idx));
            }
        }

        let output = Shape::new(output_dims);
        let explanation = if explanations.is_empty() {
            "All dimensions broadcast successfully".to_string()
        } else {
            explanations.join("; ")
        };

        self.add_trace_step(ShapeTraceStep {
            step,
            operation: ShapeOperation::Broadcast,
            input_shapes: shapes.to_vec(),
            output_shape: Some(output.clone()),
            explanation,
            success: true,
            error: None,
        });

        Ok(output)
    }

    /// Infer shape for concatenation
    pub fn infer_concat_shape(&self, shapes: &[Shape], dim: i32) -> Result<Shape> {
        let step = self.next_step();

        if shapes.is_empty() {
            let error = "Concatenation requires at least one tensor".to_string();
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::Concatenate,
                input_shapes: shapes.to_vec(),
                output_shape: None,
                explanation: format!("Checking inputs for concatenation along dim {}", dim),
                success: false,
                error: Some(error.clone()),
            });
            return Err(TorshError::InvalidShape(error));
        }

        let first_dims = shapes[0].dims();
        let ndim = first_dims.len() as i32;

        // Normalize dimension
        let concat_dim = if dim < 0 { ndim + dim } else { dim };

        if concat_dim < 0 || concat_dim >= ndim {
            let error = format!(
                "Concatenation dimension {} is out of range for {}-dimensional tensors",
                dim, ndim
            );
            self.add_trace_step(ShapeTraceStep {
                step,
                operation: ShapeOperation::Concatenate,
                input_shapes: shapes.to_vec(),
                output_shape: None,
                explanation: format!("Validating dimension {}", dim),
                success: false,
                error: Some(error.clone()),
            });
            return Err(TorshError::InvalidShape(error));
        }

        let concat_dim = concat_dim as usize;
        let mut total_concat_size = 0;

        // Check all shapes are compatible
        for (i, shape) in shapes.iter().enumerate() {
            let dims = shape.dims();

            if dims.len() != first_dims.len() {
                let error = format!(
                    "All tensors must have same number of dimensions for concatenation. \
                     Tensor 0 has {} dims, tensor {} has {} dims",
                    first_dims.len(),
                    i,
                    dims.len()
                );
                self.add_trace_step(ShapeTraceStep {
                    step,
                    operation: ShapeOperation::Concatenate,
                    input_shapes: shapes.to_vec(),
                    output_shape: None,
                    explanation: format!("Checking shape compatibility for tensor {}", i),
                    success: false,
                    error: Some(error.clone()),
                });
                return Err(TorshError::InvalidShape(error));
            }

            for (dim_idx, (&d1, &d2)) in first_dims.iter().zip(dims.iter()).enumerate() {
                if dim_idx != concat_dim && d1 != d2 {
                    let error = format!(
                        "All tensors must have same size in non-concat dimensions. \
                         Dimension {} differs: tensor 0 has size {}, tensor {} has size {}",
                        dim_idx, d1, i, d2
                    );
                    self.add_trace_step(ShapeTraceStep {
                        step,
                        operation: ShapeOperation::Concatenate,
                        input_shapes: shapes.to_vec(),
                        output_shape: None,
                        explanation: format!("Checking dimension {} for tensor {}", dim_idx, i),
                        success: false,
                        error: Some(error.clone()),
                    });
                    return Err(TorshError::InvalidShape(error));
                }
            }

            total_concat_size += dims[concat_dim];
        }

        // Build output shape
        let mut output_dims = first_dims.to_vec();
        output_dims[concat_dim] = total_concat_size;
        let output = Shape::new(output_dims);

        self.add_trace_step(ShapeTraceStep {
            step,
            operation: ShapeOperation::Concatenate,
            input_shapes: shapes.to_vec(),
            output_shape: Some(output.clone()),
            explanation: format!(
                "Concatenating {} tensors along dimension {}: total size {}",
                shapes.len(),
                concat_dim,
                total_concat_size
            ),
            success: true,
            error: None,
        });

        Ok(output)
    }

    /// Get the complete trace as a formatted string
    pub fn get_trace(&self) -> String {
        let trace = self.trace.read().expect("lock should not be poisoned");
        let mut output = String::new();
        output.push_str("=== Shape Inference Trace ===\n\n");

        for step in trace.iter() {
            output.push_str(&format!("{}\n", step));
        }

        output.push_str(&format!("Total steps: {}\n", trace.len()));
        let successful = trace.iter().filter(|s| s.success).count();
        output.push_str(&format!(
            "Successful: {} ({:.1}%)\n",
            successful,
            (successful as f64 / trace.len() as f64) * 100.0
        ));

        output
    }

    /// Clear the trace
    pub fn clear_trace(&mut self) {
        self.trace
            .write()
            .expect("lock should not be poisoned")
            .clear();
        *self
            .step_counter
            .write()
            .expect("lock should not be poisoned") = 0;
    }

    /// Get trace statistics
    pub fn get_statistics(&self) -> TraceStatistics {
        let trace = self.trace.read().expect("lock should not be poisoned");
        let total_steps = trace.len();
        let successful_steps = trace.iter().filter(|s| s.success).count();
        let failed_steps = total_steps - successful_steps;

        let mut operation_counts: HashMap<String, usize> = HashMap::new();
        for step in trace.iter() {
            *operation_counts
                .entry(step.operation.to_string())
                .or_insert(0) += 1;
        }

        TraceStatistics {
            total_steps,
            successful_steps,
            failed_steps,
            operation_counts,
        }
    }
}

impl Default for ShapeInferenceDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the shape inference trace
#[derive(Debug, Clone)]
pub struct TraceStatistics {
    /// Total number of trace steps
    pub total_steps: usize,
    /// Number of successful steps
    pub successful_steps: usize,
    /// Number of failed steps
    pub failed_steps: usize,
    /// Count of each operation type
    pub operation_counts: HashMap<String, usize>,
}

impl fmt::Display for TraceStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Shape Inference Statistics:")?;
        writeln!(f, "  Total steps: {}", self.total_steps)?;
        writeln!(f, "  Successful: {}", self.successful_steps)?;
        writeln!(f, "  Failed: {}", self.failed_steps)?;
        if self.total_steps > 0 {
            writeln!(
                f,
                "  Success rate: {:.1}%",
                (self.successful_steps as f64 / self.total_steps as f64) * 100.0
            )?;
        }
        writeln!(f, "  Operations:")?;
        for (op, count) in &self.operation_counts {
            writeln!(f, "    {}: {}", op, count)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_identical_shapes() {
        let debugger = ShapeInferenceDebugger::new();
        let shapes = vec![
            Shape::new(vec![2, 3]),
            Shape::new(vec![2, 3]),
            Shape::new(vec![2, 3]),
        ];

        let result = debugger.infer_elementwise_shape(&shapes).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_elementwise_broadcasting() {
        let debugger = ShapeInferenceDebugger::new();
        let shapes = vec![Shape::new(vec![2, 3]), Shape::new(vec![1, 3])];

        let result = debugger.infer_elementwise_shape(&shapes).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_matmul_simple() {
        let debugger = ShapeInferenceDebugger::new();
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![3, 4]);

        let result = debugger.infer_matmul_shape(&a, &b).unwrap();
        assert_eq!(result.dims(), &[2, 4]);
    }

    #[test]
    fn test_matmul_incompatible() {
        let debugger = ShapeInferenceDebugger::new();
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![4, 5]);

        let result = debugger.infer_matmul_shape(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_compatible() {
        let debugger = ShapeInferenceDebugger::new();
        let shapes = vec![
            Shape::new(vec![2, 3, 4]),
            Shape::new(vec![3, 4]),
            Shape::new(vec![4]),
        ];

        let result = debugger.infer_broadcast_shape(&shapes).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        let debugger = ShapeInferenceDebugger::new();
        let shapes = vec![Shape::new(vec![2, 3]), Shape::new(vec![2, 4])];

        let result = debugger.infer_broadcast_shape(&shapes);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_valid() {
        let debugger = ShapeInferenceDebugger::new();
        let shapes = vec![
            Shape::new(vec![2, 3]),
            Shape::new(vec![2, 5]),
            Shape::new(vec![2, 2]),
        ];

        let result = debugger.infer_concat_shape(&shapes, 1).unwrap();
        assert_eq!(result.dims(), &[2, 10]); // 3 + 5 + 2
    }

    #[test]
    fn test_concat_incompatible() {
        let debugger = ShapeInferenceDebugger::new();
        let shapes = vec![Shape::new(vec![2, 3]), Shape::new(vec![3, 3])];

        let result = debugger.infer_concat_shape(&shapes, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_collection() {
        let debugger = ShapeInferenceDebugger::new();

        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![3, 4]);
        let _ = debugger.infer_matmul_shape(&a, &b).unwrap();

        let trace = debugger.get_trace();
        assert!(trace.contains("MatMul"));
        assert!(trace.contains("Step 0"));
    }

    #[test]
    fn test_statistics() {
        let debugger = ShapeInferenceDebugger::new();

        // Successful operations
        let _ = debugger.infer_matmul_shape(&Shape::new(vec![2, 3]), &Shape::new(vec![3, 4]));
        let _ = debugger.infer_broadcast_shape(&[Shape::new(vec![2, 3]), Shape::new(vec![3])]);

        // Failed operation
        let _ = debugger.infer_matmul_shape(&Shape::new(vec![2, 3]), &Shape::new(vec![4, 5]));

        let stats = debugger.get_statistics();
        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.successful_steps, 2);
        assert_eq!(stats.failed_steps, 1);
    }
}
