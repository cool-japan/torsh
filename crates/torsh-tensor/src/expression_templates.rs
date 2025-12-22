//! Expression Templates for Compile-Time Tensor Operation Optimization
//!
//! This module provides expression templates that enable compile-time fusion of tensor operations.
//! Expression templates defer actual computation until evaluation, allowing the compiler to optimize
//! operation chains and eliminate intermediate allocations.
//!
//! # Features
//!
//! - **Compile-time optimization**: Operations are fused at compile time
//! - **Zero-cost abstractions**: No runtime overhead compared to hand-written fused loops
//! - **Lazy evaluation**: Computation is deferred until explicitly requested
//! - **Type-safe**: Full type checking at compile time
//! - **Intermediate elimination**: No temporary tensors created for operation chains
//!
//! # Example
//!
//! ```rust
//! use torsh_tensor::{Tensor, expression_templates::*};
//!
//! // Without expression templates:
//! // let temp1 = a.add(&b)?;  // Allocates temporary
//! // let temp2 = temp1.mul(&c)?;  // Allocates another temporary
//! // let result = temp2.add_scalar(1.0)?;
//!
//! // With expression templates:
//! // let expr = expr_add(expr_tensor(&a), expr_tensor(&b))
//! //     .mul(expr_tensor(&c))
//! //     .add_scalar(1.0);
//! // let result = expr.eval()?;  // Single allocation, fused computation
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use torsh_core::{dtype::TensorElement, error::Result};

use crate::Tensor;

/// Trait representing an expression that can be evaluated to produce a value
pub trait Expression<T: TensorElement> {
    /// Evaluate the expression at a specific index
    fn eval_at(&self, index: usize) -> T;

    /// Get the size (number of elements) of the expression
    fn size(&self) -> usize;

    /// Evaluate the entire expression into a Vec
    fn eval_vec(&self) -> Vec<T> {
        (0..self.size()).map(|i| self.eval_at(i)).collect()
    }

    /// Evaluate the expression into a tensor
    fn eval_tensor(
        &self,
        shape: Vec<usize>,
        device: torsh_core::device::DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Copy,
    {
        let data = self.eval_vec();
        Tensor::from_data(data, shape, device)
    }
}

/// Expression representing a tensor reference
pub struct TensorExpr<'a, T: TensorElement> {
    data: Vec<T>,
    size: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: TensorElement + Copy> TensorExpr<'a, T> {
    /// Create a new tensor expression from a tensor
    pub fn new(tensor: &'a Tensor<T>) -> Result<Self> {
        let data = tensor.to_vec()?;
        let size = data.len();

        Ok(Self {
            data,
            size,
            _phantom: PhantomData,
        })
    }
}

impl<'a, T: TensorElement> Expression<T> for TensorExpr<'a, T> {
    fn eval_at(&self, index: usize) -> T {
        self.data[index]
    }

    fn size(&self) -> usize {
        self.size
    }

    fn eval_vec(&self) -> Vec<T> {
        self.data.clone()
    }
}

/// Expression representing scalar addition
pub struct AddScalarExpr<T: TensorElement, E: Expression<T>> {
    expr: E,
    scalar: T,
}

impl<T: TensorElement + Add<Output = T>, E: Expression<T>> Expression<T> for AddScalarExpr<T, E> {
    fn eval_at(&self, index: usize) -> T {
        self.expr.eval_at(index) + self.scalar
    }

    fn size(&self) -> usize {
        self.expr.size()
    }
}

/// Expression representing scalar multiplication
pub struct MulScalarExpr<T: TensorElement, E: Expression<T>> {
    expr: E,
    scalar: T,
}

impl<T: TensorElement + Mul<Output = T>, E: Expression<T>> Expression<T> for MulScalarExpr<T, E> {
    fn eval_at(&self, index: usize) -> T {
        self.expr.eval_at(index) * self.scalar
    }

    fn size(&self) -> usize {
        self.expr.size()
    }
}

/// Expression representing scalar subtraction
pub struct SubScalarExpr<T: TensorElement, E: Expression<T>> {
    expr: E,
    scalar: T,
}

impl<T: TensorElement + Sub<Output = T>, E: Expression<T>> Expression<T> for SubScalarExpr<T, E> {
    fn eval_at(&self, index: usize) -> T {
        self.expr.eval_at(index) - self.scalar
    }

    fn size(&self) -> usize {
        self.expr.size()
    }
}

/// Expression representing scalar division
pub struct DivScalarExpr<T: TensorElement, E: Expression<T>> {
    expr: E,
    scalar: T,
}

impl<T: TensorElement + Div<Output = T>, E: Expression<T>> Expression<T> for DivScalarExpr<T, E> {
    fn eval_at(&self, index: usize) -> T {
        self.expr.eval_at(index) / self.scalar
    }

    fn size(&self) -> usize {
        self.expr.size()
    }
}

/// Expression representing element-wise addition
pub struct AddExpr<T: TensorElement, E1: Expression<T>, E2: Expression<T>> {
    left: E1,
    right: E2,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement + Add<Output = T>, E1: Expression<T>, E2: Expression<T>> Expression<T>
    for AddExpr<T, E1, E2>
{
    fn eval_at(&self, index: usize) -> T {
        self.left.eval_at(index) + self.right.eval_at(index)
    }

    fn size(&self) -> usize {
        self.left.size().min(self.right.size())
    }
}

/// Expression representing element-wise multiplication
pub struct MulExpr<T: TensorElement, E1: Expression<T>, E2: Expression<T>> {
    left: E1,
    right: E2,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement + Mul<Output = T>, E1: Expression<T>, E2: Expression<T>> Expression<T>
    for MulExpr<T, E1, E2>
{
    fn eval_at(&self, index: usize) -> T {
        self.left.eval_at(index) * self.right.eval_at(index)
    }

    fn size(&self) -> usize {
        self.left.size().min(self.right.size())
    }
}

/// Expression representing element-wise subtraction
pub struct SubExpr<T: TensorElement, E1: Expression<T>, E2: Expression<T>> {
    left: E1,
    right: E2,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement + Sub<Output = T>, E1: Expression<T>, E2: Expression<T>> Expression<T>
    for SubExpr<T, E1, E2>
{
    fn eval_at(&self, index: usize) -> T {
        self.left.eval_at(index) - self.right.eval_at(index)
    }

    fn size(&self) -> usize {
        self.left.size().min(self.right.size())
    }
}

/// Expression representing element-wise division
pub struct DivExpr<T: TensorElement, E1: Expression<T>, E2: Expression<T>> {
    left: E1,
    right: E2,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement + Div<Output = T>, E1: Expression<T>, E2: Expression<T>> Expression<T>
    for DivExpr<T, E1, E2>
{
    fn eval_at(&self, index: usize) -> T {
        self.left.eval_at(index) / self.right.eval_at(index)
    }

    fn size(&self) -> usize {
        self.left.size().min(self.right.size())
    }
}

/// Expression representing negation
pub struct NegExpr<T: TensorElement, E: Expression<T>> {
    expr: E,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement + std::ops::Neg<Output = T>, E: Expression<T>> Expression<T>
    for NegExpr<T, E>
{
    fn eval_at(&self, index: usize) -> T {
        -self.expr.eval_at(index)
    }

    fn size(&self) -> usize {
        self.expr.size()
    }
}

/// Expression builder for creating fused operation chains
pub struct ExprBuilder<T: TensorElement, E: Expression<T>> {
    expr: E,
    _phantom: PhantomData<T>,
}

impl<T: TensorElement, E: Expression<T>> ExprBuilder<T, E> {
    /// Create a new expression builder
    pub fn new(expr: E) -> Self {
        Self {
            expr,
            _phantom: PhantomData,
        }
    }

    /// Add a scalar to the expression
    pub fn add_scalar(self, scalar: T) -> ExprBuilder<T, AddScalarExpr<T, E>>
    where
        T: Add<Output = T>,
    {
        ExprBuilder::new(AddScalarExpr {
            expr: self.expr,
            scalar,
        })
    }

    /// Multiply the expression by a scalar
    pub fn mul_scalar(self, scalar: T) -> ExprBuilder<T, MulScalarExpr<T, E>>
    where
        T: Mul<Output = T>,
    {
        ExprBuilder::new(MulScalarExpr {
            expr: self.expr,
            scalar,
        })
    }

    /// Subtract a scalar from the expression
    pub fn sub_scalar(self, scalar: T) -> ExprBuilder<T, SubScalarExpr<T, E>>
    where
        T: Sub<Output = T>,
    {
        ExprBuilder::new(SubScalarExpr {
            expr: self.expr,
            scalar,
        })
    }

    /// Divide the expression by a scalar
    pub fn div_scalar(self, scalar: T) -> ExprBuilder<T, DivScalarExpr<T, E>>
    where
        T: Div<Output = T>,
    {
        ExprBuilder::new(DivScalarExpr {
            expr: self.expr,
            scalar,
        })
    }

    /// Add another expression element-wise
    pub fn add<E2: Expression<T>>(
        self,
        other: ExprBuilder<T, E2>,
    ) -> ExprBuilder<T, AddExpr<T, E, E2>>
    where
        T: Add<Output = T>,
    {
        ExprBuilder::new(AddExpr {
            left: self.expr,
            right: other.expr,
            _phantom: PhantomData,
        })
    }

    /// Multiply another expression element-wise
    pub fn mul<E2: Expression<T>>(
        self,
        other: ExprBuilder<T, E2>,
    ) -> ExprBuilder<T, MulExpr<T, E, E2>>
    where
        T: Mul<Output = T>,
    {
        ExprBuilder::new(MulExpr {
            left: self.expr,
            right: other.expr,
            _phantom: PhantomData,
        })
    }

    /// Subtract another expression element-wise
    pub fn sub<E2: Expression<T>>(
        self,
        other: ExprBuilder<T, E2>,
    ) -> ExprBuilder<T, SubExpr<T, E, E2>>
    where
        T: Sub<Output = T>,
    {
        ExprBuilder::new(SubExpr {
            left: self.expr,
            right: other.expr,
            _phantom: PhantomData,
        })
    }

    /// Divide by another expression element-wise
    pub fn div<E2: Expression<T>>(
        self,
        other: ExprBuilder<T, E2>,
    ) -> ExprBuilder<T, DivExpr<T, E, E2>>
    where
        T: Div<Output = T>,
    {
        ExprBuilder::new(DivExpr {
            left: self.expr,
            right: other.expr,
            _phantom: PhantomData,
        })
    }

    /// Negate the expression
    pub fn neg(self) -> ExprBuilder<T, NegExpr<T, E>>
    where
        T: std::ops::Neg<Output = T>,
    {
        ExprBuilder::new(NegExpr {
            expr: self.expr,
            _phantom: PhantomData,
        })
    }

    /// Evaluate the expression into a Vec
    pub fn eval_vec(&self) -> Vec<T> {
        self.expr.eval_vec()
    }

    /// Evaluate the expression into a Tensor
    pub fn eval_tensor(
        &self,
        shape: Vec<usize>,
        device: torsh_core::device::DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Copy,
    {
        self.expr.eval_tensor(shape, device)
    }
}

/// Create an expression from a tensor reference
pub fn expr<'a, T: TensorElement + Copy>(
    tensor: &'a Tensor<T>,
) -> Result<ExprBuilder<T, TensorExpr<'a, T>>> {
    let tensor_expr = TensorExpr::new(tensor)?;
    Ok(ExprBuilder::new(tensor_expr))
}

/// Trait for tensors that support expression templates
pub trait TensorExprExt<T: TensorElement> {
    /// Convert the tensor to an expression builder
    fn expr(&self) -> Result<ExprBuilder<T, TensorExpr<'_, T>>>
    where
        T: Copy;
}

impl<T: TensorElement + Copy> TensorExprExt<T> for Tensor<T> {
    fn expr(&self) -> Result<ExprBuilder<T, TensorExpr<'_, T>>> {
        expr(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_scalar_operations() {
        let tensor = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        let result = tensor
            .expr()
            .unwrap()
            .add_scalar(1.0)
            .mul_scalar(2.0)
            .eval_vec();

        assert_eq!(result, vec![4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_element_wise_operations() {
        let a = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = tensor_1d(&[2.0f32, 2.0, 2.0, 2.0]).unwrap();

        let result = a.expr().unwrap().add(b.expr().unwrap()).eval_vec();

        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_complex_expression() {
        let a = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = tensor_1d(&[2.0f32, 2.0, 2.0, 2.0]).unwrap();

        // (a + b) * 2 + 1
        let result = a
            .expr()
            .unwrap()
            .add(b.expr().unwrap())
            .mul_scalar(2.0)
            .add_scalar(1.0)
            .eval_vec();

        assert_eq!(result, vec![7.0, 9.0, 11.0, 13.0]);
    }

    #[test]
    fn test_negation() {
        let tensor = tensor_1d(&[1.0f32, 2.0, -3.0, 4.0]).unwrap();

        let result = tensor.expr().unwrap().neg().eval_vec();

        assert_eq!(result, vec![-1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_eval_tensor() {
        let tensor = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        let result = tensor
            .expr()
            .unwrap()
            .mul_scalar(2.0)
            .eval_tensor(vec![4], DeviceType::Cpu)
            .unwrap();

        let data = result.to_vec().unwrap();
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_division() {
        let a = tensor_1d(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let b = tensor_1d(&[2.0f32, 4.0, 5.0, 8.0]).unwrap();

        let result = a.expr().unwrap().div(b.expr().unwrap()).eval_vec();

        assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_subtraction() {
        let a = tensor_1d(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let b = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        let result = a.expr().unwrap().sub(b.expr().unwrap()).eval_vec();

        assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_multiple_operations_chain() {
        let a = tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b = tensor_1d(&[2.0f32, 2.0, 2.0, 2.0]).unwrap();
        let c = tensor_1d(&[3.0f32, 3.0, 3.0, 3.0]).unwrap();

        // ((a + b) * c) / 2 + 1
        let result = a
            .expr()
            .unwrap()
            .add(b.expr().unwrap())
            .mul(c.expr().unwrap())
            .div_scalar(2.0)
            .add_scalar(1.0)
            .eval_vec();

        assert_eq!(result, vec![5.5, 7.0, 8.5, 10.0]);
    }
}
