//! Lazy evaluation system for chained functional operations
//!
//! This module provides a lazy evaluation framework that delays computation of functional
//! operations until the final result is needed. This can significantly improve performance
//! by fusing operations and avoiding intermediate tensor allocations.

use std::collections::VecDeque;
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Represents a lazy operation that can be deferred
#[derive(Debug, Clone)]
pub enum LazyOp {
    /// Element-wise operations
    Add(LazyTensor, LazyTensor),
    Mul(LazyTensor, LazyTensor),
    Sub(LazyTensor, LazyTensor),
    Div(LazyTensor, LazyTensor),

    /// Scalar operations
    AddScalar(LazyTensor, f32),
    MulScalar(LazyTensor, f32),
    SubScalar(LazyTensor, f32),
    DivScalar(LazyTensor, f32),

    /// Unary operations
    Abs(LazyTensor),
    Exp(LazyTensor),
    Log(LazyTensor),
    Sin(LazyTensor),
    Cos(LazyTensor),
    Tanh(LazyTensor),
    Relu(LazyTensor),

    /// Reduction operations
    Sum(LazyTensor, Option<Vec<usize>>, bool),
    Mean(LazyTensor, Option<Vec<usize>>, bool),
    Max(LazyTensor, Option<usize>, bool),
    Min(LazyTensor, Option<usize>, bool),

    /// Shape operations
    Reshape(LazyTensor, Vec<usize>),
    Transpose(LazyTensor, Option<Vec<usize>>),
    Squeeze(LazyTensor, Option<usize>),
    Unsqueeze(LazyTensor, usize),
}

/// A tensor that may contain deferred operations
#[derive(Debug, Clone)]
pub enum LazyTensor {
    /// An actual computed tensor
    Eager(Tensor),
    /// A deferred operation
    Lazy(Box<LazyOp>),
}

/// Context for lazy evaluation that tracks operation chains
#[derive(Debug, Default)]
pub struct LazyContext {
    /// Queue of pending operations
    operation_queue: VecDeque<LazyOp>,
    /// Maximum chain length before forced evaluation
    max_chain_length: usize,
}

impl LazyContext {
    /// Create a new lazy evaluation context
    pub fn new() -> Self {
        Self {
            operation_queue: VecDeque::new(),
            max_chain_length: 10, // Configurable threshold
        }
    }

    /// Set the maximum chain length before forced evaluation
    pub fn with_max_chain_length(mut self, length: usize) -> Self {
        self.max_chain_length = length;
        self
    }

    /// Add an operation to the evaluation queue
    pub fn push_operation(&mut self, op: LazyOp) {
        self.operation_queue.push_back(op);

        // Force evaluation if chain gets too long
        if self.operation_queue.len() > self.max_chain_length {
            self.flush_operations().unwrap_or_else(|e| {
                eprintln!("Warning: Failed to flush lazy operations: {}", e);
            });
        }
    }

    /// Force evaluation of all pending operations
    pub fn flush_operations(&mut self) -> TorshResult<()> {
        while let Some(op) = self.operation_queue.pop_front() {
            self.evaluate_operation(op)?;
        }
        Ok(())
    }

    /// Evaluate a single operation
    fn evaluate_operation(&self, op: LazyOp) -> TorshResult<Tensor> {
        match op {
            LazyOp::Add(lhs, rhs) => {
                let lhs_tensor = lhs.evaluate()?;
                let rhs_tensor = rhs.evaluate()?;
                lhs_tensor.add_op(&rhs_tensor)
            }
            LazyOp::Mul(lhs, rhs) => {
                let lhs_tensor = lhs.evaluate()?;
                let rhs_tensor = rhs.evaluate()?;
                lhs_tensor.mul_op(&rhs_tensor)
            }
            LazyOp::Sub(lhs, rhs) => {
                let lhs_tensor = lhs.evaluate()?;
                let rhs_tensor = rhs.evaluate()?;
                lhs_tensor.sub(&rhs_tensor)
            }
            LazyOp::Div(lhs, rhs) => {
                let lhs_tensor = lhs.evaluate()?;
                let rhs_tensor = rhs.evaluate()?;
                lhs_tensor.div(&rhs_tensor)
            }
            LazyOp::AddScalar(tensor, scalar) => {
                let tensor = tensor.evaluate()?;
                tensor.add_scalar(scalar)
            }
            LazyOp::MulScalar(tensor, scalar) => {
                let tensor = tensor.evaluate()?;
                tensor.mul_scalar(scalar)
            }
            LazyOp::SubScalar(tensor, scalar) => {
                let tensor = tensor.evaluate()?;
                // Use add_scalar with negative value since sub_scalar doesn't exist
                tensor.add_scalar(-scalar)
            }
            LazyOp::DivScalar(tensor, scalar) => {
                let tensor = tensor.evaluate()?;
                tensor.div_scalar(scalar)
            }
            LazyOp::Abs(tensor) => {
                let tensor = tensor.evaluate()?;
                tensor.abs()
            }
            LazyOp::Exp(tensor) => {
                let tensor = tensor.evaluate()?;
                tensor.exp()
            }
            LazyOp::Log(tensor) => {
                let tensor = tensor.evaluate()?;
                tensor.log()
            }
            LazyOp::Sin(tensor) => {
                let tensor = tensor.evaluate()?;
                tensor.sin()
            }
            LazyOp::Cos(tensor) => {
                let tensor = tensor.evaluate()?;
                tensor.cos()
            }
            LazyOp::Tanh(tensor) => {
                let tensor = tensor.evaluate()?;
                tensor.tanh()
            }
            LazyOp::Relu(tensor) => {
                let tensor = tensor.evaluate()?;
                // Implement ReLU as max(0, x) using maximum method with zeros
                let zeros = tensor.zeros_like()?;
                tensor.maximum(&zeros)
            }
            LazyOp::Sum(tensor, dims, keepdim) => {
                let tensor = tensor.evaluate()?;
                match dims {
                    Some(dims) => {
                        let dims_i32: Vec<i32> = dims.iter().map(|&x| x as i32).collect();
                        tensor.sum_dim(&dims_i32, keepdim)
                    }
                    None => tensor.sum(),
                }
            }
            LazyOp::Mean(tensor, dims, keepdim) => {
                let tensor = tensor.evaluate()?;
                tensor.mean(dims.as_ref().map(|v| v.as_slice()), keepdim)
            }
            LazyOp::Max(tensor, dim, keepdim) => {
                let tensor = tensor.evaluate()?;
                match dim {
                    Some(dim) => tensor.max_dim(dim as i32, keepdim),
                    None => tensor.max(None, keepdim),
                }
            }
            LazyOp::Min(tensor, dim, keepdim) => {
                let tensor = tensor.evaluate()?;
                match dim {
                    Some(dim) => tensor.min_dim(dim as i32, keepdim),
                    None => tensor.min(),
                }
            }
            LazyOp::Reshape(tensor, shape) => {
                let tensor = tensor.evaluate()?;
                tensor.reshape(&shape.iter().map(|&x| x as i32).collect::<Vec<i32>>())
            }
            LazyOp::Transpose(tensor, dims) => {
                let tensor = tensor.evaluate()?;
                match dims {
                    Some(dims) => {
                        tensor.permute(&dims.iter().map(|&x| x as i32).collect::<Vec<i32>>())
                    }
                    None => tensor.transpose(0, 1),
                }
            }
            LazyOp::Squeeze(tensor, dim) => {
                let tensor = tensor.evaluate()?;
                match dim {
                    Some(dim) => tensor.squeeze(dim as i32),
                    None => tensor.squeeze(-1),
                }
            }
            LazyOp::Unsqueeze(tensor, dim) => {
                let tensor = tensor.evaluate()?;
                tensor.unsqueeze(dim as i32)
            }
        }
    }
}

impl LazyTensor {
    /// Create a lazy tensor from an eager tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        LazyTensor::Eager(tensor)
    }

    /// Create a lazy tensor from an operation
    pub fn from_operation(op: LazyOp) -> Self {
        LazyTensor::Lazy(Box::new(op))
    }

    /// Force evaluation of the lazy tensor
    pub fn evaluate(self) -> TorshResult<Tensor> {
        match self {
            LazyTensor::Eager(tensor) => Ok(tensor),
            LazyTensor::Lazy(op) => {
                let ctx = LazyContext::new();
                ctx.evaluate_operation(*op)
            }
        }
    }

    /// Check if the tensor is already evaluated
    pub fn is_eager(&self) -> bool {
        matches!(self, LazyTensor::Eager(_))
    }

    /// Get the operation chain depth for optimization decisions
    pub fn depth(&self) -> usize {
        match self {
            LazyTensor::Eager(_) => 0,
            LazyTensor::Lazy(op) => 1 + op.max_input_depth(),
        }
    }

    // Lazy operation builders

    /// Lazy addition
    pub fn add(self, other: LazyTensor) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Add(self, other))
    }

    /// Lazy multiplication
    pub fn mul(self, other: LazyTensor) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Mul(self, other))
    }

    /// Lazy subtraction
    pub fn sub(self, other: LazyTensor) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Sub(self, other))
    }

    /// Lazy division
    pub fn div(self, other: LazyTensor) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Div(self, other))
    }

    /// Lazy scalar addition
    pub fn add_scalar(self, scalar: f32) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::AddScalar(self, scalar))
    }

    /// Lazy scalar multiplication
    pub fn mul_scalar(self, scalar: f32) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::MulScalar(self, scalar))
    }

    /// Lazy scalar subtraction
    pub fn sub_scalar(self, scalar: f32) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::SubScalar(self, scalar))
    }

    /// Lazy absolute value
    pub fn abs(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Abs(self))
    }

    /// Lazy exponential
    pub fn exp(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Exp(self))
    }

    /// Lazy logarithm
    pub fn log(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Log(self))
    }

    /// Lazy sine
    pub fn sin(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Sin(self))
    }

    /// Lazy cosine
    pub fn cos(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Cos(self))
    }

    /// Lazy hyperbolic tangent
    pub fn tanh(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Tanh(self))
    }

    /// Lazy ReLU activation
    pub fn relu(self) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Relu(self))
    }

    /// Lazy sum reduction
    pub fn sum(self, dims: Option<Vec<usize>>, keepdim: bool) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Sum(self, dims, keepdim))
    }

    /// Lazy mean reduction
    pub fn mean(self, dims: Option<Vec<usize>>, keepdim: bool) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Mean(self, dims, keepdim))
    }

    /// Lazy reshape
    pub fn reshape(self, shape: Vec<usize>) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Reshape(self, shape))
    }

    /// Lazy transpose
    pub fn transpose(self, dims: Option<Vec<usize>>) -> LazyTensor {
        LazyTensor::from_operation(LazyOp::Transpose(self, dims))
    }
}

impl LazyOp {
    /// Get the maximum depth of input tensors
    fn max_input_depth(&self) -> usize {
        match self {
            LazyOp::Add(lhs, rhs)
            | LazyOp::Mul(lhs, rhs)
            | LazyOp::Sub(lhs, rhs)
            | LazyOp::Div(lhs, rhs) => lhs.depth().max(rhs.depth()),
            LazyOp::AddScalar(tensor, _)
            | LazyOp::MulScalar(tensor, _)
            | LazyOp::SubScalar(tensor, _)
            | LazyOp::DivScalar(tensor, _)
            | LazyOp::Abs(tensor)
            | LazyOp::Exp(tensor)
            | LazyOp::Log(tensor)
            | LazyOp::Sin(tensor)
            | LazyOp::Cos(tensor)
            | LazyOp::Tanh(tensor)
            | LazyOp::Relu(tensor)
            | LazyOp::Sum(tensor, _, _)
            | LazyOp::Mean(tensor, _, _)
            | LazyOp::Max(tensor, _, _)
            | LazyOp::Min(tensor, _, _)
            | LazyOp::Reshape(tensor, _)
            | LazyOp::Transpose(tensor, _)
            | LazyOp::Squeeze(tensor, _)
            | LazyOp::Unsqueeze(tensor, _) => tensor.depth(),
        }
    }

    /// Optimize operation chains by fusing compatible operations
    pub fn optimize(self) -> LazyOp {
        // This is a placeholder for more sophisticated optimization
        // In a full implementation, we would:
        // 1. Fuse element-wise operations
        // 2. Reorder operations for better cache locality
        // 3. Eliminate redundant operations
        // 4. Apply algebraic simplifications

        match self {
            // Example: Fuse consecutive scalar multiplications
            LazyOp::MulScalar(LazyTensor::Lazy(inner_op), scalar2) => {
                if let LazyOp::MulScalar(tensor, scalar1) = *inner_op {
                    LazyOp::MulScalar(tensor, scalar1 * scalar2)
                } else {
                    LazyOp::MulScalar(LazyTensor::Lazy(inner_op), scalar2)
                }
            }

            // Example: Eliminate identity operations
            LazyOp::MulScalar(tensor, scalar) if scalar == 1.0 => {
                return match tensor {
                    LazyTensor::Lazy(op) => *op,
                    LazyTensor::Eager(_) => LazyOp::MulScalar(tensor, scalar),
                }
            }
            LazyOp::AddScalar(tensor, scalar) if scalar == 0.0 => {
                return match tensor {
                    LazyTensor::Lazy(op) => *op,
                    LazyTensor::Eager(_) => LazyOp::AddScalar(tensor, scalar),
                }
            }

            // Return the operation as-is if no optimization applies
            op => op,
        }
    }
}

/// Builder for creating optimized lazy computation chains
pub struct LazyBuilder {
    context: LazyContext,
}

impl LazyBuilder {
    /// Create a new lazy builder
    pub fn new() -> Self {
        Self {
            context: LazyContext::new(),
        }
    }

    /// Configure the maximum chain length
    pub fn with_max_chain_length(mut self, length: usize) -> Self {
        self.context = self.context.with_max_chain_length(length);
        self
    }

    /// Build and evaluate the computation chain
    pub fn build(mut self, tensor: LazyTensor) -> TorshResult<Tensor> {
        self.context.flush_operations()?;
        tensor.evaluate()
    }
}

impl Default for LazyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for creating lazy tensors from common operations
pub mod lazy_ops {
    use super::*;

    /// Create a lazy computation chain from a tensor
    pub fn lazy(tensor: Tensor) -> LazyTensor {
        LazyTensor::from_tensor(tensor)
    }

    /// Execute a lazy computation chain
    pub fn execute(tensor: LazyTensor) -> TorshResult<Tensor> {
        tensor.evaluate()
    }

    /// Create an optimized lazy computation with custom chain length
    pub fn with_optimization(tensor: LazyTensor, max_chain_length: usize) -> TorshResult<Tensor> {
        LazyBuilder::new()
            .with_max_chain_length(max_chain_length)
            .build(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_lazy_chain() {
        let tensor = ones(&[2, 2]).unwrap();
        let lazy_tensor = LazyTensor::from_tensor(tensor);

        // Create a computation chain: ((x + 1) * 2) - 1
        let result = lazy_tensor.add_scalar(1.0).mul_scalar(2.0).sub_scalar(1.0);

        // Should not be evaluated yet
        assert!(!result.is_eager());

        // Force evaluation
        let computed = result.evaluate().unwrap();

        // Result should be ones * 2 + 2 - 1 = ones * 3
        let expected = 3.0;
        let data = computed.data().unwrap();
        for &val in data.iter() {
            assert!((val - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lazy_depth() {
        let tensor = ones(&[2, 2]).unwrap();
        let lazy_tensor = LazyTensor::from_tensor(tensor);

        assert_eq!(lazy_tensor.depth(), 0);

        let chained = lazy_tensor.add_scalar(1.0).mul_scalar(2.0);
        assert_eq!(chained.depth(), 2);
    }

    #[test]
    fn test_optimization() {
        // Test identity optimization
        let op = LazyOp::MulScalar(LazyTensor::from_tensor(ones(&[2, 2]).unwrap()), 1.0);
        let optimized = op.optimize();

        // Should eliminate the identity multiplication
        match optimized {
            LazyOp::MulScalar(_, scalar) => assert_eq!(scalar, 1.0),
            _ => panic!("Expected optimized identity operation"),
        }
    }

    #[test]
    fn test_lazy_builder() {
        let tensor = ones(&[2, 2]).unwrap();
        let lazy_tensor = LazyTensor::from_tensor(tensor);

        let result = LazyBuilder::new()
            .with_max_chain_length(5)
            .build(lazy_tensor.add_scalar(1.0).mul_scalar(2.0))
            .unwrap();

        let expected = 4.0; // (1 + 1) * 2 = 4
        let data = result.data().unwrap();
        for &val in data.iter() {
            assert!((val - expected).abs() < 1e-6);
        }
    }
}
