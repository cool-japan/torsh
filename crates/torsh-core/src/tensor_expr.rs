//! Tensor Expression Templates for Compile-Time Optimization
//!
//! This module provides expression template infrastructure for lazy evaluation
//! and compile-time optimization of tensor operations. Expression templates
//! allow chaining multiple operations without creating intermediate tensors,
//! resulting in significant performance improvements.
//!
//! # Key Features
//!
//! - **Lazy Evaluation**: Operations are not executed until needed
//! - **Zero Intermediate Allocations**: Entire expression trees optimized away
//! - **Compile-Time Fusion**: Multiple operations fused into single kernel
//! - **Type-Safe Operations**: All operations verified at compile time
//! - **Cache-Efficient**: Better memory locality through operation fusion
//!
//! # Example
//!
//! ```ignore
//! use torsh_core::tensor_expr::*;
//!
//! // These operations are fused at compile time
//! // No intermediate arrays are created
//! let result = (a + b) * c - d;
//! ```
//!
//! # Architecture
//!
//! Expression templates use Rust's type system to build computation graphs
//! at compile time. Each operation returns an expression object that encodes
//! the operation tree in its type. When the expression is finally evaluated,
//! the compiler can optimize the entire tree, potentially generating SIMD
//! code and eliminating redundant operations.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};
#[cfg(feature = "std")]
use std::vec::Vec;

/// Trait for evaluating expressions at a given index
///
/// This is the core trait for expression templates. All expression types
/// must implement this trait to support lazy evaluation.
pub trait TensorExpr: Sized {
    /// The scalar type of the expression result
    type Scalar: Copy;

    /// Evaluate the expression at the given linear index
    ///
    /// # Arguments
    ///
    /// * `index` - The linear index in the tensor
    ///
    /// # Returns
    ///
    /// The value of the expression at the given index
    fn eval(&self, index: usize) -> Self::Scalar;

    /// Get the total number of elements in the expression
    fn len(&self) -> usize;

    /// Check if the expression is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Map this expression to a new expression by applying a function
    #[inline]
    fn map<F, S>(self, f: F) -> MapExpr<Self, F, S>
    where
        F: Fn(Self::Scalar) -> S,
        S: Copy,
    {
        MapExpr {
            expr: self,
            func: f,
            _phantom: PhantomData,
        }
    }

    /// Reduce this expression to a single value
    fn reduce<F>(&self, init: Self::Scalar, f: F) -> Self::Scalar
    where
        F: Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    {
        let mut result = init;
        for i in 0..self.len() {
            result = f(result, self.eval(i));
        }
        result
    }

    /// Sum all elements in the expression
    fn sum(&self) -> Self::Scalar
    where
        Self::Scalar: core::ops::Add<Output = Self::Scalar> + Default,
    {
        self.reduce(Self::Scalar::default(), |a, b| a + b)
    }

    /// Find the maximum element in the expression
    fn max(&self) -> Option<Self::Scalar>
    where
        Self::Scalar: PartialOrd,
    {
        if self.is_empty() {
            return None;
        }
        let mut result = self.eval(0);
        for i in 1..self.len() {
            let val = self.eval(i);
            if val > result {
                result = val;
            }
        }
        Some(result)
    }

    /// Find the minimum element in the expression
    fn min(&self) -> Option<Self::Scalar>
    where
        Self::Scalar: PartialOrd,
    {
        if self.is_empty() {
            return None;
        }
        let mut result = self.eval(0);
        for i in 1..self.len() {
            let val = self.eval(i);
            if val < result {
                result = val;
            }
        }
        Some(result)
    }

    /// Materialize the expression into a vector
    ///
    /// This forces evaluation of the entire expression tree
    fn materialize(&self) -> Vec<Self::Scalar> {
        (0..self.len()).map(|i| self.eval(i)).collect()
    }

    /// Apply the expression to a mutable slice
    ///
    /// This is more efficient than materializing when you have
    /// pre-allocated storage
    fn apply_to_slice(&self, output: &mut [Self::Scalar]) {
        assert_eq!(output.len(), self.len(), "Output slice size mismatch");
        for (i, item) in output.iter_mut().enumerate() {
            *item = self.eval(i);
        }
    }
}

/// Scalar literal expression
///
/// Represents a constant scalar value broadcast to all indices
#[derive(Debug, Clone, Copy)]
pub struct ScalarExpr<T: Copy> {
    value: T,
    len: usize,
}

impl<T: Copy> ScalarExpr<T> {
    /// Create a new scalar expression
    #[inline]
    pub fn new(value: T, len: usize) -> Self {
        Self { value, len }
    }
}

impl<T: Copy> TensorExpr for ScalarExpr<T> {
    type Scalar = T;

    #[inline]
    fn eval(&self, _index: usize) -> Self::Scalar {
        self.value
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

/// Array reference expression
///
/// Wraps a slice for use in expression templates
#[derive(Debug, Clone, Copy)]
pub struct ArrayExpr<'a, T: Copy> {
    data: &'a [T],
}

impl<'a, T: Copy> ArrayExpr<'a, T> {
    /// Create a new array expression from a slice
    #[inline]
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }
}

impl<'a, T: Copy> TensorExpr for ArrayExpr<'a, T> {
    type Scalar = T;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        self.data[index]
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Binary operation expression
///
/// Represents a binary operation between two expressions
#[derive(Debug, Clone, Copy)]
pub struct BinaryExpr<L, R, Op> {
    left: L,
    right: R,
    _op: PhantomData<Op>,
}

impl<L, R, Op> BinaryExpr<L, R, Op> {
    /// Create a new binary expression
    #[inline]
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            _op: PhantomData,
        }
    }
}

/// Addition operation
#[derive(Debug, Clone, Copy)]
pub struct AddOp;

/// Subtraction operation
#[derive(Debug, Clone, Copy)]
pub struct SubOp;

/// Multiplication operation
#[derive(Debug, Clone, Copy)]
pub struct MulOp;

/// Division operation
#[derive(Debug, Clone, Copy)]
pub struct DivOp;

impl<L, R> TensorExpr for BinaryExpr<L, R, AddOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Add<Output = L::Scalar>,
{
    type Scalar = L::Scalar;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        self.left.eval(index) + self.right.eval(index)
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert_eq!(
            self.left.len(),
            self.right.len(),
            "Expression length mismatch"
        );
        self.left.len()
    }
}

impl<L, R> TensorExpr for BinaryExpr<L, R, SubOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Sub<Output = L::Scalar>,
{
    type Scalar = L::Scalar;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        self.left.eval(index) - self.right.eval(index)
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert_eq!(
            self.left.len(),
            self.right.len(),
            "Expression length mismatch"
        );
        self.left.len()
    }
}

impl<L, R> TensorExpr for BinaryExpr<L, R, MulOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Mul<Output = L::Scalar>,
{
    type Scalar = L::Scalar;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        self.left.eval(index) * self.right.eval(index)
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert_eq!(
            self.left.len(),
            self.right.len(),
            "Expression length mismatch"
        );
        self.left.len()
    }
}

impl<L, R> TensorExpr for BinaryExpr<L, R, DivOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Div<Output = L::Scalar>,
{
    type Scalar = L::Scalar;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        self.left.eval(index) / self.right.eval(index)
    }

    #[inline]
    fn len(&self) -> usize {
        debug_assert_eq!(
            self.left.len(),
            self.right.len(),
            "Expression length mismatch"
        );
        self.left.len()
    }
}

/// Unary negation expression
#[derive(Debug, Clone, Copy)]
pub struct NegExpr<E> {
    expr: E,
}

impl<E> NegExpr<E> {
    /// Create a new negation expression
    #[inline]
    pub fn new(expr: E) -> Self {
        Self { expr }
    }
}

impl<E> TensorExpr for NegExpr<E>
where
    E: TensorExpr,
    E::Scalar: Neg<Output = E::Scalar>,
{
    type Scalar = E::Scalar;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        -self.expr.eval(index)
    }

    #[inline]
    fn len(&self) -> usize {
        self.expr.len()
    }
}

/// Map expression - applies a function to each element
#[derive(Debug, Clone, Copy)]
pub struct MapExpr<E, F, S> {
    expr: E,
    func: F,
    _phantom: PhantomData<S>,
}

impl<E, F, S> TensorExpr for MapExpr<E, F, S>
where
    E: TensorExpr,
    F: Fn(E::Scalar) -> S,
    S: Copy,
{
    type Scalar = S;

    #[inline]
    fn eval(&self, index: usize) -> Self::Scalar {
        (self.func)(self.expr.eval(index))
    }

    #[inline]
    fn len(&self) -> usize {
        self.expr.len()
    }
}

// Operator overloading for expression composition

impl<L, R> Add<R> for BinaryExpr<L, R, AddOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Add<Output = L::Scalar>,
{
    type Output = BinaryExpr<Self, R, AddOp>;

    #[inline]
    fn add(self, rhs: R) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<L, R> Sub<R> for BinaryExpr<L, R, SubOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Sub<Output = L::Scalar>,
{
    type Output = BinaryExpr<Self, R, SubOp>;

    #[inline]
    fn sub(self, rhs: R) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<L, R> Mul<R> for BinaryExpr<L, R, MulOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Mul<Output = L::Scalar>,
{
    type Output = BinaryExpr<Self, R, MulOp>;

    #[inline]
    fn mul(self, rhs: R) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<L, R> Div<R> for BinaryExpr<L, R, DivOp>
where
    L: TensorExpr,
    R: TensorExpr<Scalar = L::Scalar>,
    L::Scalar: Div<Output = L::Scalar>,
{
    type Output = BinaryExpr<Self, R, DivOp>;

    #[inline]
    fn div(self, rhs: R) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

// ArrayExpr operator overloading

impl<'a, T> Add for ArrayExpr<'a, T>
where
    T: Copy + Add<Output = T>,
{
    type Output = BinaryExpr<Self, Self, AddOp>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<'a, T> Sub for ArrayExpr<'a, T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = BinaryExpr<Self, Self, SubOp>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<'a, T> Mul for ArrayExpr<'a, T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = BinaryExpr<Self, Self, MulOp>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<'a, T> Div for ArrayExpr<'a, T>
where
    T: Copy + Div<Output = T>,
{
    type Output = BinaryExpr<Self, Self, DivOp>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        BinaryExpr::new(self, rhs)
    }
}

impl<'a, T> Neg for ArrayExpr<'a, T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = NegExpr<Self>;

    #[inline]
    fn neg(self) -> Self::Output {
        NegExpr::new(self)
    }
}

/// Expression builder for convenient construction
pub struct ExprBuilder;

impl ExprBuilder {
    /// Create an array expression from a slice
    #[inline]
    pub fn array<T: Copy>(data: &[T]) -> ArrayExpr<'_, T> {
        ArrayExpr::new(data)
    }

    /// Create a scalar expression
    #[inline]
    pub fn scalar<T: Copy>(value: T, len: usize) -> ScalarExpr<T> {
        ScalarExpr::new(value, len)
    }

    /// Add two expressions
    #[inline]
    pub fn add<L, R>(left: L, right: R) -> BinaryExpr<L, R, AddOp>
    where
        L: TensorExpr,
        R: TensorExpr<Scalar = L::Scalar>,
    {
        BinaryExpr::new(left, right)
    }

    /// Subtract two expressions
    #[inline]
    pub fn sub<L, R>(left: L, right: R) -> BinaryExpr<L, R, SubOp>
    where
        L: TensorExpr,
        R: TensorExpr<Scalar = L::Scalar>,
    {
        BinaryExpr::new(left, right)
    }

    /// Multiply two expressions
    #[inline]
    pub fn mul<L, R>(left: L, right: R) -> BinaryExpr<L, R, MulOp>
    where
        L: TensorExpr,
        R: TensorExpr<Scalar = L::Scalar>,
    {
        BinaryExpr::new(left, right)
    }

    /// Divide two expressions
    #[inline]
    pub fn div<L, R>(left: L, right: R) -> BinaryExpr<L, R, DivOp>
    where
        L: TensorExpr,
        R: TensorExpr<Scalar = L::Scalar>,
    {
        BinaryExpr::new(left, right)
    }

    /// Negate an expression
    #[inline]
    pub fn neg<E>(expr: E) -> NegExpr<E>
    where
        E: TensorExpr,
    {
        NegExpr::new(expr)
    }
}

/// Specialized expressions for common mathematical operations
pub mod math {
    use super::*;

    /// Square expression
    #[derive(Debug, Clone, Copy)]
    pub struct SqrExpr<E> {
        expr: E,
    }

    impl<E> SqrExpr<E> {
        /// Create a new square expression
        #[inline]
        pub fn new(expr: E) -> Self {
            Self { expr }
        }
    }

    impl<E> TensorExpr for SqrExpr<E>
    where
        E: TensorExpr,
        E::Scalar: Mul<Output = E::Scalar>,
    {
        type Scalar = E::Scalar;

        #[inline]
        fn eval(&self, index: usize) -> Self::Scalar {
            let val = self.expr.eval(index);
            val * val
        }

        #[inline]
        fn len(&self) -> usize {
            self.expr.len()
        }
    }

    /// Absolute value expression
    #[derive(Debug, Clone, Copy)]
    pub struct AbsExpr<E> {
        expr: E,
    }

    impl<E> AbsExpr<E> {
        /// Create a new absolute value expression
        #[inline]
        pub fn new(expr: E) -> Self {
            Self { expr }
        }
    }

    impl<E> TensorExpr for AbsExpr<E>
    where
        E: TensorExpr,
        E::Scalar: PartialOrd + Neg<Output = E::Scalar> + Default,
    {
        type Scalar = E::Scalar;

        #[inline]
        fn eval(&self, index: usize) -> Self::Scalar {
            let val = self.expr.eval(index);
            if val < E::Scalar::default() {
                -val
            } else {
                val
            }
        }

        #[inline]
        fn len(&self) -> usize {
            self.expr.len()
        }
    }

    /// Extension trait for mathematical operations
    pub trait MathExpr: TensorExpr + Sized {
        /// Square each element
        fn sqr(self) -> SqrExpr<Self>
        where
            Self::Scalar: Mul<Output = Self::Scalar>,
        {
            SqrExpr::new(self)
        }

        /// Absolute value of each element
        fn abs(self) -> AbsExpr<Self>
        where
            Self::Scalar: PartialOrd + Neg<Output = Self::Scalar> + Default,
        {
            AbsExpr::new(self)
        }
    }

    // Implement MathExpr for all TensorExpr types
    impl<T: TensorExpr> MathExpr for T {}
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;

    #[test]
    fn test_array_expr_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expr = ArrayExpr::new(&data);

        assert_eq!(expr.len(), 4);
        assert_eq!(expr.eval(0), 1.0);
        assert_eq!(expr.eval(1), 2.0);
        assert_eq!(expr.eval(2), 3.0);
        assert_eq!(expr.eval(3), 4.0);
    }

    #[test]
    fn test_scalar_expr() {
        let expr = ScalarExpr::new(5.0, 4);

        assert_eq!(expr.len(), 4);
        assert_eq!(expr.eval(0), 5.0);
        assert_eq!(expr.eval(1), 5.0);
        assert_eq!(expr.eval(2), 5.0);
        assert_eq!(expr.eval(3), 5.0);
    }

    #[test]
    fn test_addition() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        let add_expr = ExprBuilder::add(expr_a, expr_b);

        assert_eq!(add_expr.eval(0), 6.0);
        assert_eq!(add_expr.eval(1), 8.0);
        assert_eq!(add_expr.eval(2), 10.0);
        assert_eq!(add_expr.eval(3), 12.0);
    }

    #[test]
    fn test_subtraction() {
        let a = vec![10.0, 20.0, 30.0, 40.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        let sub_expr = ExprBuilder::sub(expr_a, expr_b);

        assert_eq!(sub_expr.eval(0), 5.0);
        assert_eq!(sub_expr.eval(1), 14.0);
        assert_eq!(sub_expr.eval(2), 23.0);
        assert_eq!(sub_expr.eval(3), 32.0);
    }

    #[test]
    fn test_multiplication() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        let mul_expr = ExprBuilder::mul(expr_a, expr_b);

        assert_eq!(mul_expr.eval(0), 6.0);
        assert_eq!(mul_expr.eval(1), 12.0);
        assert_eq!(mul_expr.eval(2), 20.0);
        assert_eq!(mul_expr.eval(3), 30.0);
    }

    #[test]
    fn test_division() {
        let a = vec![12.0, 20.0, 30.0, 40.0];
        let b = vec![3.0, 4.0, 5.0, 8.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        let div_expr = ExprBuilder::div(expr_a, expr_b);

        assert_eq!(div_expr.eval(0), 4.0);
        assert_eq!(div_expr.eval(1), 5.0);
        assert_eq!(div_expr.eval(2), 6.0);
        assert_eq!(div_expr.eval(3), 5.0);
    }

    #[test]
    fn test_negation() {
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let expr = ArrayExpr::new(&a);
        let neg_expr = ExprBuilder::neg(expr);

        assert_eq!(neg_expr.eval(0), -1.0);
        assert_eq!(neg_expr.eval(1), 2.0);
        assert_eq!(neg_expr.eval(2), -3.0);
        assert_eq!(neg_expr.eval(3), 4.0);
    }

    #[test]
    fn test_complex_expression() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);
        let expr_c = ArrayExpr::new(&c);

        // (a + b) * c
        let complex_expr = ExprBuilder::mul(ExprBuilder::add(expr_a, expr_b), expr_c);

        assert_eq!(complex_expr.eval(0), 3.0); // (1 + 2) * 1
        assert_eq!(complex_expr.eval(1), 5.0); // (2 + 3) * 1
        assert_eq!(complex_expr.eval(2), 7.0); // (3 + 4) * 1
        assert_eq!(complex_expr.eval(3), 9.0); // (4 + 5) * 1
    }

    #[test]
    fn test_operator_overloading() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        // Using operator overloading
        let expr = expr_a + expr_b;

        assert_eq!(expr.eval(0), 3.0);
        assert_eq!(expr.eval(1), 5.0);
        assert_eq!(expr.eval(2), 7.0);
        assert_eq!(expr.eval(3), 9.0);
    }

    #[test]
    fn test_materialize() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        let expr = ExprBuilder::add(expr_a, expr_b);
        let result = expr.materialize();

        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_apply_to_slice() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        let expr = ExprBuilder::mul(expr_a, expr_b);

        let mut output = vec![0.0; 4];
        expr.apply_to_slice(&mut output);

        assert_eq!(output, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_sum_reduction() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let expr = ArrayExpr::new(&a);

        let sum = expr.sum();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_max_min() {
        let a = vec![3.0, 1.0, 4.0, 2.0];
        let expr = ArrayExpr::new(&a);

        assert_eq!(expr.max(), Some(4.0));
        assert_eq!(expr.min(), Some(1.0));
    }

    #[test]
    fn test_map() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let expr = ArrayExpr::new(&a);

        let mapped = expr.map(|x| x * 2.0);

        assert_eq!(mapped.eval(0), 2.0);
        assert_eq!(mapped.eval(1), 4.0);
        assert_eq!(mapped.eval(2), 6.0);
        assert_eq!(mapped.eval(3), 8.0);
    }

    #[test]
    fn test_math_square() {
        use math::MathExpr;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let expr = ArrayExpr::new(&a);

        let sqr_expr = expr.sqr();

        assert_eq!(sqr_expr.eval(0), 1.0);
        assert_eq!(sqr_expr.eval(1), 4.0);
        assert_eq!(sqr_expr.eval(2), 9.0);
        assert_eq!(sqr_expr.eval(3), 16.0);
    }

    #[test]
    fn test_math_abs() {
        use math::MathExpr;

        let a = vec![1.0, -2.0, 3.0, -4.0];
        let expr = ArrayExpr::new(&a);

        let abs_expr = expr.abs();

        assert_eq!(abs_expr.eval(0), 1.0);
        assert_eq!(abs_expr.eval(1), 2.0);
        assert_eq!(abs_expr.eval(2), 3.0);
        assert_eq!(abs_expr.eval(3), 4.0);
    }

    #[test]
    fn test_chained_operations() {
        use math::MathExpr;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);

        // (a + b).sqr()
        let expr = ExprBuilder::add(expr_a, expr_b).sqr();

        assert_eq!(expr.eval(0), 9.0); // (1 + 2)^2 = 9
        assert_eq!(expr.eval(1), 16.0); // (2 + 2)^2 = 16
        assert_eq!(expr.eval(2), 25.0); // (3 + 2)^2 = 25
        assert_eq!(expr.eval(3), 36.0); // (4 + 2)^2 = 36
    }

    #[test]
    fn test_expression_fusion() {
        // This test demonstrates that expressions are evaluated lazily
        // and can be fused at compile time
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let d = vec![0.5, 0.5, 0.5, 0.5];

        let expr_a = ArrayExpr::new(&a);
        let expr_b = ArrayExpr::new(&b);
        let expr_c = ArrayExpr::new(&c);
        let expr_d = ArrayExpr::new(&d);

        // Complex expression: ((a + b) * c) - d
        // Should be fused into single loop
        let expr = ExprBuilder::sub(
            ExprBuilder::mul(ExprBuilder::add(expr_a, expr_b), expr_c),
            expr_d,
        );

        assert_eq!(expr.eval(0), 2.5); // ((1+2)*1) - 0.5 = 2.5
        assert_eq!(expr.eval(1), 4.5); // ((2+3)*1) - 0.5 = 4.5
        assert_eq!(expr.eval(2), 6.5); // ((3+4)*1) - 0.5 = 6.5
        assert_eq!(expr.eval(3), 8.5); // ((4+5)*1) - 0.5 = 8.5

        // Materializing the expression should create the result in a single pass
        let result = expr.materialize();
        assert_eq!(result, vec![2.5, 4.5, 6.5, 8.5]);
    }

    #[test]
    fn test_empty_expression() {
        let data: Vec<f32> = vec![];
        let expr = ArrayExpr::new(&data);

        assert!(expr.is_empty());
        assert_eq!(expr.len(), 0);
        assert_eq!(expr.max(), None);
        assert_eq!(expr.min(), None);
    }
}
