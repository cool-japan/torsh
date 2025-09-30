//! Core transform framework for data transformations
//!
//! This module provides the fundamental building blocks for data transformations,
//! including core traits, combinators, and basic transform implementations.
//!
//! # Features
//!
//! - **Transform trait**: Core abstraction for data transformations
//! - **Transform combinators**: Chain, conditional, and composition operations
//! - **Builder pattern**: TransformBuilder trait for complex transform construction
//! - **Extension traits**: Convenient chainable API via TransformExt
//! - **Basic transforms**: Normalize, type conversion, and lambda transforms

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Trait for data transformations
///
/// This is the core abstraction for all data transformations in the ToRSh ecosystem.
/// Implementations should be stateless where possible and thread-safe.
pub trait Transform<T>: Send + Sync {
    /// Output type after transformation
    type Output;

    /// Apply the transformation to a single input
    fn transform(&self, input: T) -> Result<Self::Output>;

    /// Transform multiple items in batch
    ///
    /// Default implementation applies transform individually, but implementations
    /// can override this for more efficient batch processing.
    fn transform_batch(&self, inputs: Vec<T>) -> Result<Vec<Self::Output>> {
        inputs
            .into_iter()
            .map(|input| self.transform(input))
            .collect()
    }

    /// Check if the transform is deterministic
    ///
    /// A deterministic transform always produces the same output for the same input.
    /// Non-deterministic transforms include random augmentations.
    fn is_deterministic(&self) -> bool {
        true
    }
}

/// Builder trait for transforms with configuration options
pub trait TransformBuilder {
    /// The transform type this builder creates
    type Transform;

    /// Build the configured transform
    fn build(self) -> Self::Transform;
}

/// Macro to create simple stateless transforms
///
/// This macro generates a transform struct and implementation for simple cases
/// where the transform logic can be expressed as a function.
#[macro_export]
macro_rules! simple_transform {
    ($name:ident, $input:ty, $output:ty, $transform_fn:expr) => {
        /// Auto-generated simple transform
        #[derive(Clone, Debug, Default)]
        pub struct $name;

        impl $crate::core_framework::Transform<$input> for $name {
            type Output = $output;

            fn transform(&self, input: $input) -> $crate::core_framework::Result<Self::Output> {
                Ok($transform_fn(input))
            }
        }
    };

    ($name:ident, $input:ty, $output:ty, $transform_fn:expr, deterministic = $det:literal) => {
        /// Auto-generated simple transform with determinism setting
        #[derive(Clone, Debug, Default)]
        pub struct $name;

        impl $crate::core_framework::Transform<$input> for $name {
            type Output = $output;

            fn transform(&self, input: $input) -> $crate::core_framework::Result<Self::Output> {
                Ok($transform_fn(input))
            }

            fn is_deterministic(&self) -> bool {
                $det
            }
        }
    };
}

/// Extension trait for chainable transform operations
pub trait TransformExt<T>: Transform<T> + Sized + 'static {
    /// Chain this transform with another
    ///
    /// Creates a new transform that applies this transform first, then the next.
    fn then<U>(self, next: U) -> Chain<Self, U>
    where
        U: Transform<Self::Output>,
    {
        Chain::new(self, next)
    }

    /// Apply this transform conditionally based on a predicate
    ///
    /// The transform is only applied if the predicate returns true for the input.
    fn when<P>(self, predicate: P) -> Conditional<Self, P>
    where
        P: Fn(&T) -> bool + Send + Sync,
    {
        Conditional::new(self, predicate)
    }

    /// Convert to a boxed trait object for dynamic dispatch
    fn boxed(self) -> Box<dyn Transform<T, Output = Self::Output> + Send + Sync> {
        Box::new(self)
    }
}

// Blanket implementation for all transforms
impl<T, U: Transform<T> + 'static> TransformExt<T> for U {}

/// Chain two transforms together sequentially
#[derive(Debug, Clone)]
pub struct Chain<T1, T2> {
    first: T1,
    second: T2,
}

impl<T1, T2> Chain<T1, T2> {
    /// Create a new chain of transforms
    pub fn new(first: T1, second: T2) -> Self {
        Self { first, second }
    }
}

impl<T, T1, T2> Transform<T> for Chain<T1, T2>
where
    T1: Transform<T>,
    T2: Transform<T1::Output>,
{
    type Output = T2::Output;

    fn transform(&self, input: T) -> Result<Self::Output> {
        let intermediate = self.first.transform(input)?;
        self.second.transform(intermediate)
    }

    fn is_deterministic(&self) -> bool {
        self.first.is_deterministic() && self.second.is_deterministic()
    }
}

/// Conditionally apply a transform based on a predicate
#[derive(Debug, Clone)]
pub struct Conditional<T, P> {
    transform: T,
    predicate: P,
}

impl<T, P> Conditional<T, P> {
    /// Create a new conditional transform
    pub fn new(transform: T, predicate: P) -> Self {
        Self {
            transform,
            predicate,
        }
    }
}

impl<T, U, P> Transform<T> for Conditional<U, P>
where
    U: Transform<T, Output = T>,
    P: Fn(&T) -> bool + Send + Sync,
{
    type Output = T;

    fn transform(&self, input: T) -> Result<Self::Output> {
        if (self.predicate)(&input) {
            self.transform.transform(input)
        } else {
            Ok(input)
        }
    }

    fn is_deterministic(&self) -> bool {
        self.transform.is_deterministic()
    }
}

/// Compose multiple transforms that operate on the same type
#[derive(Debug)]
pub struct Compose<T> {
    transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>,
}

impl<T> Compose<T> {
    /// Create a new compose transform from a vector of transforms
    pub fn new(transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>) -> Self {
        Self { transforms }
    }

    /// Add a transform to the composition
    pub fn add<U>(&mut self, transform: U)
    where
        U: Transform<T, Output = T> + Send + Sync + 'static,
    {
        self.transforms.push(Box::new(transform));
    }

    /// Get the number of transforms in the composition
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the composition is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T> Transform<T> for Compose<T> {
    type Output = T;

    fn transform(&self, mut input: T) -> Result<Self::Output> {
        for transform in &self.transforms {
            input = transform.transform(input)?;
        }
        Ok(input)
    }

    fn is_deterministic(&self) -> bool {
        self.transforms.iter().all(|t| t.is_deterministic())
    }
}

/// Normalize tensor values using mean and standard deviation
#[derive(Debug, Clone)]
pub struct Normalize<T: TensorElement> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T: TensorElement> Normalize<T> {
    /// Create a new normalize transform
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Result<Self> {
        if mean.len() != std.len() {
            return Err(TorshError::InvalidArgument(
                "Mean and std vectors must have the same length".to_string(),
            ));
        }
        Ok(Self { mean, std })
    }
}

impl<T: TensorElement> Transform<Tensor<T>> for Normalize<T> {
    type Output = Tensor<T>;

    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        // Placeholder implementation - real normalization would require tensor operations
        // For now, just return the input tensor
        tracing::debug!(
            "Normalize transform applied with {} channels",
            self.mean.len()
        );
        Ok(input)
    }
}

/// Convert tensor from one type to another
#[derive(Debug, Clone)]
pub struct ToType<From, To> {
    _phantom: core::marker::PhantomData<(From, To)>,
}

impl<From, To> Default for ToType<From, To> {
    fn default() -> Self {
        Self::new()
    }
}

impl<From, To> ToType<From, To> {
    /// Create a new type conversion transform
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<From: TensorElement, To: TensorElement> Transform<Tensor<From>> for ToType<From, To> {
    type Output = Tensor<To>;

    fn transform(&self, input: Tensor<From>) -> Result<Self::Output> {
        // Placeholder implementation - real type conversion would require tensor operations
        // For now, create a new tensor with the target type (this is a simplification)
        tracing::debug!(
            "Type conversion from {} to {} requested",
            core::any::type_name::<From>(),
            core::any::type_name::<To>()
        );

        // In a real implementation, this would convert the tensor data
        // For now, we return an error as this requires complex tensor operations
        Err(TorshError::InvalidArgument(
            "Type conversion not yet implemented".to_string(),
        ))
    }
}

/// Apply a custom function as a transform
#[derive(Debug)]
pub struct Lambda<F> {
    func: F,
}

impl<F> Lambda<F> {
    /// Create a new lambda transform
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<T, O, F> Transform<T> for Lambda<F>
where
    F: Fn(T) -> Result<O> + Send + Sync,
{
    type Output = O;

    fn transform(&self, input: T) -> Result<Self::Output> {
        (self.func)(input)
    }

    fn is_deterministic(&self) -> bool {
        // Lambda functions are assumed to be deterministic unless specified otherwise
        true
    }
}

/// Convenience function to create a normalize transform
pub fn normalize<T: TensorElement>(mean: Vec<T>, std: Vec<T>) -> Result<Normalize<T>> {
    Normalize::new(mean, std)
}

/// Convenience function to create a type conversion transform
pub fn to_type<From: TensorElement, To: TensorElement>() -> ToType<From, To> {
    ToType::new()
}

/// Convenience function to create a lambda transform
pub fn lambda<F>(func: F) -> Lambda<F> {
    Lambda::new(func)
}

/// Convenience function to create a composition transform
pub fn compose<T>(transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>) -> Compose<T> {
    Compose::new(transforms)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock tensor for testing
    fn mock_tensor() -> Tensor<f32> {
        Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], torsh_core::device::DeviceType::Cpu).unwrap()
    }

    #[test]
    fn test_chain_transform() {
        let lambda1 = lambda(|x: i32| Ok(x * 2));
        let lambda2 = lambda(|x: i32| Ok(x + 1));

        let chained = lambda1.then(lambda2);
        let result = chained.transform(5).unwrap();
        assert_eq!(result, 11); // (5 * 2) + 1 = 11
    }

    #[test]
    fn test_conditional_transform() {
        let double = lambda(|x: i32| Ok(x * 2));
        let conditional = double.when(|&x| x > 5);

        assert_eq!(conditional.transform(3).unwrap(), 3); // Not applied
        assert_eq!(conditional.transform(7).unwrap(), 14); // Applied
    }

    #[test]
    fn test_compose_transform() {
        let lambda1 = lambda(|x: i32| Ok(x + 1));
        let lambda2 = lambda(|x: i32| Ok(x * 2));

        let mut composition = Compose::new(vec![]);
        composition.add(lambda1);
        composition.add(lambda2);

        let result = composition.transform(5).unwrap();
        assert_eq!(result, 12); // ((5 + 1) * 2) = 12
    }

    #[test]
    fn test_normalize_creation() {
        let mean = vec![0.485f32, 0.456, 0.406];
        let std = vec![0.229f32, 0.224, 0.225];

        let normalize_transform = normalize(mean, std);
        assert!(normalize_transform.is_ok());
    }

    #[test]
    fn test_normalize_invalid_dimensions() {
        let mean = vec![0.485f32, 0.456];
        let std = vec![0.229f32, 0.224, 0.225];

        let normalize_transform = normalize(mean, std);
        assert!(normalize_transform.is_err());
    }

    #[test]
    fn test_determinism() {
        let deterministic = lambda(|x: i32| Ok(x + 1));
        assert!(deterministic.is_deterministic());

        let chain = deterministic.then(lambda(|x: i32| Ok(x * 2)));
        assert!(chain.is_deterministic());
    }
}