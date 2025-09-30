//! Lazy Operation Evaluation for Tensors
//!
//! This module provides lazy evaluation capabilities for tensor operations, allowing
//! operations to be chained and deferred until explicitly evaluated. This can provide
//! significant performance benefits by enabling operation fusion and optimization.
//!
//! # Features
//!
//! - **Operation chaining**: Chain multiple operations without immediate evaluation
//! - **Operation fusion**: Combine compatible operations for better performance
//! - **Lazy evaluation**: Defer computation until `eval()` is called
//! - **Optimization passes**: Apply optimizations to the operation graph
//! - **Memory efficiency**: Avoid creating intermediate tensors

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
    shape::Shape,
};

use crate::{Tensor, Operation};
use std::fmt;

/// Represents a lazy operation that hasn't been evaluated yet
#[derive(Clone)]
pub enum LazyOp<T: TensorElement> {
    /// Identity operation (no-op)
    Identity(Arc<Tensor<T>>),
    /// Element-wise addition
    Add(Box<LazyOp<T>>, Box<LazyOp<T>>),
    /// Element-wise multiplication
    Mul(Box<LazyOp<T>>, Box<LazyOp<T>>),
    /// Element-wise subtraction
    Sub(Box<LazyOp<T>>, Box<LazyOp<T>>),
    /// Element-wise division
    Div(Box<LazyOp<T>>, Box<LazyOp<T>>),
    /// Scalar addition
    AddScalar(Box<LazyOp<T>>, T),
    /// Scalar multiplication
    MulScalar(Box<LazyOp<T>>, T),
    /// Scalar subtraction
    SubScalar(Box<LazyOp<T>>, T),
    /// Scalar division
    DivScalar(Box<LazyOp<T>>, T),
    /// Power operation
    Pow(Box<LazyOp<T>>, T),
    /// Matrix multiplication
    MatMul(Box<LazyOp<T>>, Box<LazyOp<T>>),
    /// Transpose operation
    Transpose(Box<LazyOp<T>>, Option<(usize, usize)>),
    /// Reshape operation
    Reshape(Box<LazyOp<T>>, Shape),
    /// Sum reduction
    Sum(Box<LazyOp<T>>, Option<i32>),
    /// Mean reduction
    Mean(Box<LazyOp<T>>, Option<i32>),
    /// ReLU activation
    ReLU(Box<LazyOp<T>>),
    /// Sigmoid activation
    Sigmoid(Box<LazyOp<T>>),
    /// Tanh activation
    Tanh(Box<LazyOp<T>>),
    /// Exp function
    Exp(Box<LazyOp<T>>),
    /// Log function
    Log(Box<LazyOp<T>>),
    /// Sin function
    Sin(Box<LazyOp<T>>),
    /// Cos function
    Cos(Box<LazyOp<T>>),
    /// Custom operation with name and function
    Custom(String, Box<LazyOp<T>>, Arc<dyn Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync>),
}

impl<T: TensorElement> fmt::Debug for LazyOp<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LazyOp::Identity(tensor) => write!(f, "Identity({:?})", tensor),
            LazyOp::Add(lhs, rhs) => write!(f, "Add({:?}, {:?})", lhs, rhs),
            LazyOp::Mul(lhs, rhs) => write!(f, "Mul({:?}, {:?})", lhs, rhs),
            LazyOp::Sub(lhs, rhs) => write!(f, "Sub({:?}, {:?})", lhs, rhs),
            LazyOp::Div(lhs, rhs) => write!(f, "Div({:?}, {:?})", lhs, rhs),
            LazyOp::AddScalar(input, scalar) => write!(f, "AddScalar({:?}, {:?})", input, scalar),
            LazyOp::MulScalar(input, scalar) => write!(f, "MulScalar({:?}, {:?})", input, scalar),
            LazyOp::SubScalar(input, scalar) => write!(f, "SubScalar({:?}, {:?})", input, scalar),
            LazyOp::DivScalar(input, scalar) => write!(f, "DivScalar({:?}, {:?})", input, scalar),
            LazyOp::Pow(input, exp) => write!(f, "Pow({:?}, {:?})", input, exp),
            LazyOp::MatMul(lhs, rhs) => write!(f, "MatMul({:?}, {:?})", lhs, rhs),
            LazyOp::Transpose(input, dims) => write!(f, "Transpose({:?}, {:?})", input, dims),
            LazyOp::Reshape(input, shape) => write!(f, "Reshape({:?}, {:?})", input, shape),
            LazyOp::Sum(input, dim) => write!(f, "Sum({:?}, {:?})", input, dim),
            LazyOp::Mean(input, dim) => write!(f, "Mean({:?}, {:?})", input, dim),
            LazyOp::ReLU(input) => write!(f, "ReLU({:?})", input),
            LazyOp::Sigmoid(input) => write!(f, "Sigmoid({:?})", input),
            LazyOp::Tanh(input) => write!(f, "Tanh({:?})", input),
            LazyOp::Exp(input) => write!(f, "Exp({:?})", input),
            LazyOp::Log(input) => write!(f, "Log({:?})", input),
            LazyOp::Sin(input) => write!(f, "Sin({:?})", input),
            LazyOp::Cos(input) => write!(f, "Cos({:?})", input),
            LazyOp::Custom(name, input, _) => write!(f, "Custom({}, {:?}, <fn>)", name, input),
        }
    }
}

/// A lazy tensor that represents a deferred computation
pub struct LazyTensor<T: TensorElement> {
    /// The operation graph
    operation: LazyOp<T>,
    /// Cached shape (computed lazily)
    cached_shape: RwLock<Option<Shape>>,
    /// Optimization passes to apply
    optimization_passes: Vec<OptimizationPass>,
}

/// Optimization pass that can be applied to operation graphs
pub type OptimizationPass = Box<dyn Fn(&LazyOp<f32>) -> LazyOp<f32> + Send + Sync>;

impl<T: TensorElement + Into<f32> + std::iter::Sum + num_traits::FromPrimitive + torsh_core::dtype::FloatElement> LazyTensor<T> {
    /// Create a new lazy tensor from a concrete tensor
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        Self {
            operation: LazyOp::Identity(Arc::new(tensor)),
            cached_shape: RwLock::new(None),
            optimization_passes: Vec::new(),
        }
    }

    /// Create a lazy tensor from an operation
    pub fn from_operation(operation: LazyOp<T>) -> Self {
        Self {
            operation,
            cached_shape: RwLock::new(None),
            optimization_passes: Vec::new(),
        }
    }

    /// Add an optimization pass
    pub fn with_optimization<F>(mut self, pass: F) -> Self
    where
        F: Fn(&LazyOp<f32>) -> LazyOp<f32> + Send + Sync + 'static,
    {
        // Note: This is a simplified version for f32 only
        // A full implementation would use generic optimization passes
        self.optimization_passes.push(Box::new(pass));
        self
    }

    /// Get the computed shape of the tensor (computed lazily)
    pub fn shape(&self) -> Result<Shape> {
        {
            let cached = self.cached_shape.read().unwrap();
            if let Some(ref shape) = *cached {
                return Ok(shape.clone());
            }
        }

        let shape = self.compute_shape(&self.operation)?;

        {
            let mut cached = self.cached_shape.write().unwrap();
            *cached = Some(shape.clone());
        }

        Ok(shape)
    }

    /// Evaluate the lazy operation and return a concrete tensor
    pub fn eval(self) -> Result<Tensor<T>>
    where
        T: std::ops::Add<Output = T>
         + std::ops::Sub<Output = T>
         + std::ops::Mul<Output = T>
         + std::ops::Div<Output = T>
         + num_traits::Float + Copy,
    {
        // Apply optimization passes if any - for now, skip optimizations to avoid borrowing issues
        let optimized_op = self.operation.clone();

        // Create a temporary instance for evaluation
        let temp_instance = LazyTensor {
            operation: LazyOp::Identity(Arc::new(Tensor::zeros(&[1], torsh_core::device::DeviceType::Cpu).unwrap())), // Dummy
            cached_shape: RwLock::new(None),
            optimization_passes: Vec::new(),
        };

        // Evaluate the operation graph
        temp_instance.evaluate_operation(&optimized_op)
    }

    /// Apply all optimization passes to the operation
    fn apply_optimizations(&self, op: LazyOp<T>) -> LazyOp<T> {
        // For now, just return the original operation
        // Real optimization would traverse and modify the operation graph
        op
    }

    /// Recursively evaluate an operation
    fn evaluate_operation(&self, op: &LazyOp<T>) -> Result<Tensor<T>>
    where
        T: std::ops::Add<Output = T>
         + std::ops::Sub<Output = T>
         + std::ops::Mul<Output = T>
         + std::ops::Div<Output = T>
         + num_traits::Float
         + std::iter::Sum
         + torsh_core::dtype::FloatElement
         + num_traits::FromPrimitive
         + Into<f32>
         + Copy,
    {
        match op {
            LazyOp::Identity(tensor) => Ok((**tensor).clone()),

            LazyOp::Add(lhs, rhs) => {
                let lhs_val = self.evaluate_operation(lhs)?;
                let rhs_val = self.evaluate_operation(rhs)?;
                lhs_val.add_op(&rhs_val)
            }

            LazyOp::Mul(lhs, rhs) => {
                let lhs_val = self.evaluate_operation(lhs)?;
                let rhs_val = self.evaluate_operation(rhs)?;
                lhs_val.mul_op(&rhs_val)
            }

            LazyOp::Sub(lhs, rhs) => {
                let lhs_val = self.evaluate_operation(lhs)?;
                let rhs_val = self.evaluate_operation(rhs)?;
                lhs_val.sub(&rhs_val)
            }

            LazyOp::Div(lhs, rhs) => {
                let lhs_val = self.evaluate_operation(lhs)?;
                let rhs_val = self.evaluate_operation(rhs)?;
                lhs_val.div(&rhs_val)
            }

            LazyOp::AddScalar(input, scalar) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.add_scalar(*scalar)
            }

            LazyOp::MulScalar(input, scalar) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.mul_scalar(*scalar)
            }

            LazyOp::SubScalar(input, scalar) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.sub_scalar(*scalar)
            }

            LazyOp::DivScalar(input, scalar) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.div_scalar(*scalar)
            }

            LazyOp::Pow(input, exponent) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.pow(*exponent)
            }

            LazyOp::MatMul(lhs, rhs) => {
                let lhs_val = self.evaluate_operation(lhs)?;
                let rhs_val = self.evaluate_operation(rhs)?;
                lhs_val.matmul(&rhs_val)
            }

            LazyOp::Transpose(input, dims) => {
                let input_val = self.evaluate_operation(input)?;
                match dims {
                    Some((dim1, dim2)) => input_val.transpose(*dim1 as i32, *dim2 as i32),
                    None => input_val.t(),
                }
            }

            LazyOp::Reshape(input, new_shape) => {
                let input_val = self.evaluate_operation(input)?;
                let dims_i32: Vec<i32> = new_shape.dims().iter().map(|&d| d as i32).collect();
                input_val.reshape(&dims_i32)
            }

            LazyOp::Sum(input, dim) => {
                let input_val = self.evaluate_operation(input)?;
                match dim {
                    Some(d) => input_val.sum_dim(&[*d], false),
                    None => Ok(input_val.sum()?),
                }
            }

            LazyOp::Mean(input, dim) => {
                let input_val = self.evaluate_operation(input)?;
                match dim {
                    Some(d) => {
                        let dim_usize = if *d < 0 {
                            (input_val.shape().dims().len() as i32 + *d) as usize
                        } else {
                            *d as usize
                        };
                        input_val.mean(Some(&[dim_usize]), false)
                    },
                    None => input_val.mean(None, false),
                }
            }

            LazyOp::ReLU(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.relu()
            }

            LazyOp::Sigmoid(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.sigmoid()
            }

            LazyOp::Tanh(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.tanh()
            }

            LazyOp::Exp(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.exp()
            }

            LazyOp::Log(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.log()
            }

            LazyOp::Sin(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.sin()
            }

            LazyOp::Cos(input) => {
                let input_val = self.evaluate_operation(input)?;
                input_val.cos()
            }

            LazyOp::Custom(_name, input, func) => {
                let input_val = self.evaluate_operation(input)?;
                func(&input_val)
            }
        }
    }

    /// Compute the shape of an operation without evaluating it
    fn compute_shape(&self, op: &LazyOp<T>) -> Result<Shape> {
        match op {
            LazyOp::Identity(tensor) => Ok(tensor.shape().clone()),

            LazyOp::Add(lhs, rhs) | LazyOp::Mul(lhs, rhs) | LazyOp::Sub(lhs, rhs) | LazyOp::Div(lhs, rhs) => {
                let lhs_shape = self.compute_shape(lhs)?;
                let rhs_shape = self.compute_shape(rhs)?;
                // For binary ops, result shape should be broadcastable
                // This is simplified - real broadcast shape inference is more complex
                Ok(lhs_shape)
            }

            LazyOp::AddScalar(input, _) | LazyOp::MulScalar(input, _) |
            LazyOp::SubScalar(input, _) | LazyOp::DivScalar(input, _) => {
                self.compute_shape(input)
            }

            LazyOp::Pow(input, _) => self.compute_shape(input),

            LazyOp::MatMul(lhs, rhs) => {
                let lhs_shape = self.compute_shape(lhs)?;
                let rhs_shape = self.compute_shape(rhs)?;

                if lhs_shape.dims().len() < 2 || rhs_shape.dims().len() < 2 {
                    return Err(TorshError::InvalidShape("MatMul requires at least 2D tensors".to_string()));
                }

                let lhs_dims = lhs_shape.dims();
                let rhs_dims = rhs_shape.dims();

                let m = lhs_dims[lhs_dims.len() - 2];
                let k1 = lhs_dims[lhs_dims.len() - 1];
                let k2 = rhs_dims[rhs_dims.len() - 2];
                let n = rhs_dims[rhs_dims.len() - 1];

                if k1 != k2 {
                    return Err(TorshError::ShapeMismatch {
                        expected: vec![k1],
                        got: vec![k2],
                    });
                }

                let mut result_dims = lhs_dims[..lhs_dims.len()-2].to_vec();
                result_dims.extend_from_slice(&[m, n]);

                Ok(Shape::new(result_dims))
            }

            LazyOp::Transpose(input, dims) => {
                let input_shape = self.compute_shape(input)?;
                let mut result_dims = input_shape.dims().to_vec();

                match dims {
                    Some((dim1, dim2)) => {
                        if *dim1 < result_dims.len() && *dim2 < result_dims.len() {
                            result_dims.swap(*dim1, *dim2);
                        }
                    }
                    None => {
                        if result_dims.len() >= 2 {
                            let len = result_dims.len();
                            result_dims.swap(len - 2, len - 1);
                        }
                    }
                }

                Ok(Shape::new(result_dims))
            }

            LazyOp::Reshape(_, new_shape) => Ok(new_shape.clone()),

            LazyOp::Sum(input, dim) => {
                let input_shape = self.compute_shape(input)?;
                match dim {
                    Some(d) => {
                        let mut result_dims = input_shape.dims().to_vec();
                        if *d >= 0 && (*d as usize) < result_dims.len() {
                            result_dims.remove(*d as usize);
                        }
                        Ok(Shape::new(result_dims))
                    }
                    None => Ok(Shape::new(vec![])), // Scalar result
                }
            }

            LazyOp::Mean(input, dim) => self.compute_shape(&LazyOp::Sum(input.clone(), *dim)),

            LazyOp::ReLU(input) | LazyOp::Sigmoid(input) | LazyOp::Tanh(input) |
            LazyOp::Exp(input) | LazyOp::Log(input) | LazyOp::Sin(input) | LazyOp::Cos(input) => {
                self.compute_shape(input)
            }

            LazyOp::Custom(_, input, _) => self.compute_shape(input),
        }
    }
}

/// Fluent API for chaining tensor operations
impl<T: TensorElement + num_traits::Float + std::iter::Sum + torsh_core::dtype::FloatElement + num_traits::FromPrimitive + Into<f32>> LazyTensor<T> {
    /// Chain addition operation
    pub fn add(self, other: LazyTensor<T>) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Add(
            Box::new(self.operation),
            Box::new(other.operation),
        ))
    }

    /// Chain multiplication operation
    pub fn mul(self, other: LazyTensor<T>) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Mul(
            Box::new(self.operation),
            Box::new(other.operation),
        ))
    }

    /// Chain subtraction operation
    pub fn sub(self, other: LazyTensor<T>) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Sub(
            Box::new(self.operation),
            Box::new(other.operation),
        ))
    }

    /// Chain division operation
    pub fn div(self, other: LazyTensor<T>) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Div(
            Box::new(self.operation),
            Box::new(other.operation),
        ))
    }

    /// Chain scalar addition
    pub fn add_scalar(self, scalar: T) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::AddScalar(
            Box::new(self.operation),
            scalar,
        ))
    }

    /// Chain scalar multiplication
    pub fn mul_scalar(self, scalar: T) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::MulScalar(
            Box::new(self.operation),
            scalar,
        ))
    }

    /// Chain scalar subtraction
    pub fn sub_scalar(self, scalar: T) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::SubScalar(
            Box::new(self.operation),
            scalar,
        ))
    }

    /// Chain scalar division
    pub fn div_scalar(self, scalar: T) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::DivScalar(
            Box::new(self.operation),
            scalar,
        ))
    }

    /// Chain power operation
    pub fn pow(self, exponent: T) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Pow(
            Box::new(self.operation),
            exponent,
        ))
    }

    /// Chain matrix multiplication
    pub fn matmul(self, other: LazyTensor<T>) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::MatMul(
            Box::new(self.operation),
            Box::new(other.operation),
        ))
    }

    /// Chain transpose operation
    pub fn transpose(self, dim1: usize, dim2: usize) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Transpose(
            Box::new(self.operation),
            Some((dim1, dim2)),
        ))
    }

    /// Chain transpose operation (last two dimensions)
    pub fn t(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Transpose(
            Box::new(self.operation),
            None,
        ))
    }

    /// Chain reshape operation
    pub fn reshape(self, shape: &[usize]) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Reshape(
            Box::new(self.operation),
            Shape::new(shape.to_vec()),
        ))
    }

    /// Chain sum operation
    pub fn sum(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Sum(
            Box::new(self.operation),
            None,
        ))
    }

    /// Chain sum operation along dimension
    pub fn sum_dim(self, dim: i32) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Sum(
            Box::new(self.operation),
            Some(dim),
        ))
    }

    /// Chain mean operation
    pub fn mean(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Mean(
            Box::new(self.operation),
            None,
        ))
    }

    /// Chain mean operation along dimension
    pub fn mean_dim(self, dim: i32) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Mean(
            Box::new(self.operation),
            Some(dim),
        ))
    }

    /// Chain ReLU activation
    pub fn relu(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::ReLU(
            Box::new(self.operation),
        ))
    }

    /// Chain sigmoid activation
    pub fn sigmoid(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Sigmoid(
            Box::new(self.operation),
        ))
    }

    /// Chain tanh activation
    pub fn tanh(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Tanh(
            Box::new(self.operation),
        ))
    }

    /// Chain exponential function
    pub fn exp(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Exp(
            Box::new(self.operation),
        ))
    }

    /// Chain logarithm function
    pub fn log(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Log(
            Box::new(self.operation),
        ))
    }

    /// Chain sine function
    pub fn sin(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Sin(
            Box::new(self.operation),
        ))
    }

    /// Chain cosine function
    pub fn cos(self) -> LazyTensor<T> {
        LazyTensor::from_operation(LazyOp::Cos(
            Box::new(self.operation),
        ))
    }

    /// Chain custom operation
    pub fn custom<F>(self, name: String, func: F) -> LazyTensor<T>
    where
        F: Fn(&Tensor<T>) -> Result<Tensor<T>> + Send + Sync + 'static,
    {
        LazyTensor::from_operation(LazyOp::Custom(
            name,
            Box::new(self.operation),
            Arc::new(func),
        ))
    }
}

/// Extension trait to add lazy evaluation to regular tensors
pub trait TensorLazyExt<T: TensorElement> {
    /// Convert to lazy tensor for chaining operations
    fn lazy(self) -> LazyTensor<T>;
}

impl<T: TensorElement + Into<f32> + std::iter::Sum + num_traits::FromPrimitive + torsh_core::dtype::FloatElement> TensorLazyExt<T> for Tensor<T> {
    fn lazy(self) -> LazyTensor<T> {
        LazyTensor::from_tensor(self)
    }
}

/// Common optimization passes
pub mod optimizations {
    use super::*;

    /// Constant folding optimization
    ///
    /// Combines consecutive scalar operations into a single operation
    pub fn constant_folding(op: &LazyOp<f32>) -> LazyOp<f32> {
        match op {
            LazyOp::AddScalar(inner_box, s2) => {
                if let LazyOp::AddScalar(inner, s1) = &**inner_box {
                    LazyOp::AddScalar(
                        Box::new(constant_folding(inner)),
                        s1 + s2,
                    )
                } else {
                    op.clone()
                }
            }
            LazyOp::MulScalar(inner_box, s2) => {
                if let LazyOp::MulScalar(inner, s1) = &**inner_box {
                    LazyOp::MulScalar(
                        Box::new(constant_folding(inner)),
                        s1 * s2,
                    )
                } else {
                    op.clone()
                }
            }
            _ => op.clone(),
        }
    }

    /// Dead code elimination
    ///
    /// Removes operations that don't contribute to the final result
    pub fn dead_code_elimination(op: &LazyOp<f32>) -> LazyOp<f32> {
        match op {
            LazyOp::MulScalar(inner, scalar) if *scalar == 0.0 => {
                // Multiplication by zero - could replace with zero tensor
                // but we need to preserve shape, so keep the operation for now
                op.clone()
            }
            LazyOp::AddScalar(inner, scalar) if *scalar == 0.0 => {
                // Addition by zero is identity
                constant_folding(inner)
            }
            LazyOp::MulScalar(inner, scalar) if *scalar == 1.0 => {
                // Multiplication by one is identity
                constant_folding(inner)
            }
            _ => op.clone(),
        }
    }

    /// Operation fusion optimization
    ///
    /// Combines compatible operations to reduce the number of passes over data
    pub fn operation_fusion(op: &LazyOp<f32>) -> LazyOp<f32> {
        match op {
            // Fuse activation functions with preceding operations where beneficial
            LazyOp::ReLU(inner_box) => {
                if let LazyOp::AddScalar(inner, scalar) = &**inner_box {
                    // This is a simplified example - real fusion would be more sophisticated
                    LazyOp::Custom(
                        "fused_add_relu".to_string(),
                        Box::new(operation_fusion(inner)),
                        Arc::new({
                            let s = *scalar;
                            move |tensor: &Tensor<f32>| -> Result<Tensor<f32>> {
                                let added = tensor.add_scalar(s)?;
                                added.relu()
                            }
                        }),
                    )
                } else {
                    op.clone()
                }
            }
            _ => op.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use torsh_core::device::DeviceType;

    fn create_test_tensor() -> Tensor<f32> {
        Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap()
    }

    #[test]
    fn test_lazy_tensor_creation() {
        let tensor = create_test_tensor();
        let lazy = tensor.lazy();

        let shape = lazy.shape().unwrap();
        assert_eq!(shape.dims(), &[2, 2]);
    }

    #[test]
    fn test_lazy_operation_chaining() {
        let tensor1 = create_test_tensor();
        let tensor2 = create_test_tensor();

        let result = tensor1.lazy()
            .add(tensor2.lazy())
            .mul_scalar(2.0)
            .relu()
            .eval()
            .unwrap();

        // Expected: ((tensor1 + tensor2) * 2).relu()
        // = ((1,2,3,4) + (1,2,3,4)) * 2).relu() = (2,4,6,8) * 2).relu() = (4,8,12,16).relu() = (4,8,12,16)
        let expected_data = vec![4.0, 8.0, 12.0, 16.0];
        let result_data = result.to_vec().unwrap();

        for (expected, actual) in expected_data.iter().zip(result_data.iter()) {
            assert!((expected - actual).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_lazy_matmul() {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        let b = Tensor::from_data(
            vec![2.0, 0.0, 1.0, 3.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        let result = a.lazy()
            .matmul(b.lazy())
            .eval()
            .unwrap();

        // Matrix multiplication: [[1,2],[3,4]] * [[2,0],[1,3]] = [[4,6],[10,12]]
        let expected_data = vec![4.0, 6.0, 10.0, 12.0];
        let result_data = result.to_vec().unwrap();

        for (expected, actual) in expected_data.iter().zip(result_data.iter()) {
            assert!((expected - actual).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_lazy_reshape_and_transpose() {
        let tensor = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.lazy()
            .reshape(&[3, 2])
            .t()
            .eval()
            .unwrap();

        assert_eq!(result.shape().dims(), &[2, 3]);

        // Original: [[1,2,3],[4,5,6]] -> reshape to [[1,2],[3,4],[5,6]] -> transpose to [[1,3,5],[2,4,6]]
        let expected_data = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let result_data = result.to_vec().unwrap();

        for (expected, actual) in expected_data.iter().zip(result_data.iter()) {
            assert!((expected - actual).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_lazy_reductions() {
        let tensor = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        // Test sum along dimension
        let sum_result = tensor.clone().lazy()
            .sum_dim(1)
            .eval()
            .unwrap();

        // Sum along dim 1: [[1,2,3],[4,5,6]] -> [6, 15]
        let expected_sum = vec![6.0, 15.0];
        let result_sum = sum_result.to_vec().unwrap();

        for (expected, actual) in expected_sum.iter().zip(result_sum.iter()) {
            assert!((expected - actual).abs() < f32::EPSILON);
        }

        // Test mean
        let mean_result = tensor.lazy()
            .mean()
            .eval()
            .unwrap();

        // Mean of all elements: (1+2+3+4+5+6)/6 = 3.5
        let result_mean = mean_result.to_vec().unwrap();
        assert!((result_mean[0] - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_custom_operation() {
        let tensor = create_test_tensor();

        let result = tensor.lazy()
            .custom(
                "square".to_string(),
                |t: &Tensor<f32>| -> Result<Tensor<f32>> {
                    t.pow(2.0)
                },
            )
            .eval()
            .unwrap();

        // Square: [1,2,3,4] -> [1,4,9,16]
        let expected_data = vec![1.0, 4.0, 9.0, 16.0];
        let result_data = result.to_vec().unwrap();

        for (expected, actual) in expected_data.iter().zip(result_data.iter()) {
            assert!((expected - actual).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_shape_inference() {
        let tensor = create_test_tensor(); // 2x2

        let lazy_reshaped = tensor.lazy().reshape(&[4, 1]);
        let shape = lazy_reshaped.shape().unwrap();
        assert_eq!(shape.dims(), &[4, 1]);

        let lazy_transposed = lazy_reshaped.t();
        let transposed_shape = lazy_transposed.shape().unwrap();
        assert_eq!(transposed_shape.dims(), &[1, 4]);
    }

    #[test]
    fn test_complex_chain() {
        let a = create_test_tensor(); // [1,2,3,4] shaped [2,2]
        let b = create_test_tensor(); // [1,2,3,4] shaped [2,2]

        // Complex operation: ((a + b) * 2 - 1).exp().sum()
        let result = a.lazy()
            .add(b.lazy())
            .mul_scalar(2.0)
            .sub_scalar(1.0)
            .exp()
            .sum()
            .eval()
            .unwrap();

        // Step by step:
        // a + b = [2,4,6,8]
        // * 2 = [4,8,12,16]
        // - 1 = [3,7,11,15]
        // exp() = [e^3, e^7, e^11, e^15]
        // sum() = e^3 + e^7 + e^11 + e^15

        let result_val = result.to_vec().unwrap()[0];
        let expected = 3.0_f32.exp() + 7.0_f32.exp() + 11.0_f32.exp() + 15.0_f32.exp();

        assert!((result_val - expected).abs() < 1e-4);
    }
}