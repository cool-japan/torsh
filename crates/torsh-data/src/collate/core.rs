//! Core collation trait and basic implementations

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Trait for collating a batch of samples
pub trait Collate<T> {
    /// Output type after collation
    type Output;

    /// Collate a batch of samples
    fn collate(&self, batch: Vec<T>) -> Result<Self::Output>;

    /// Get the expected batch size (returns None for variable batch sizes)
    fn expected_batch_size(&self) -> Option<usize> {
        None
    }

    /// Check if this collate function supports empty batches
    fn supports_empty_batch(&self) -> bool {
        false
    }

    /// Validate batch before collation (optional hook)
    fn validate_batch(&self, batch: &[T]) -> Result<()> {
        if batch.is_empty() && !self.supports_empty_batch() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }
        Ok(())
    }
}

/// Default collation function
#[derive(Debug, Clone, Copy)]
pub struct DefaultCollate;

impl<T: TensorElement + Copy> Collate<Tensor<T>> for DefaultCollate {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;
        super::stacking::TensorStacker::new().stack(&batch, 0)
    }
}

// Common implementations for tuple types used in datasets
impl<T: TensorElement + Copy> Collate<(Tensor<T>, usize)> for DefaultCollate {
    type Output = (Tensor<T>, Vec<usize>);

    fn collate(&self, batch: Vec<(Tensor<T>, usize)>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;

        let (tensors, labels): (Vec<Tensor<T>>, Vec<usize>) = batch.into_iter().unzip();
        let stacked_tensors = super::stacking::TensorStacker::new().stack(&tensors, 0)?;

        Ok((stacked_tensors, labels))
    }
}

impl<T: TensorElement + Copy> Collate<(Tensor<T>, String)> for DefaultCollate {
    type Output = (Tensor<T>, Vec<String>);

    fn collate(&self, batch: Vec<(Tensor<T>, String)>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;

        let (tensors, strings): (Vec<Tensor<T>>, Vec<String>) = batch.into_iter().unzip();
        let stacked_tensors = super::stacking::TensorStacker::new().stack(&tensors, 0)?;

        Ok((stacked_tensors, strings))
    }
}

// Implementations for common non-tensor types
impl Collate<usize> for DefaultCollate {
    type Output = Vec<usize>;

    fn collate(&self, batch: Vec<usize>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;
        Ok(batch)
    }
}

impl Collate<String> for DefaultCollate {
    type Output = Vec<String>;

    fn collate(&self, batch: Vec<String>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;
        Ok(batch)
    }
}

impl Collate<f32> for DefaultCollate {
    type Output = Vec<f32>;

    fn collate(&self, batch: Vec<f32>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;
        Ok(batch)
    }
}

impl Collate<i32> for DefaultCollate {
    type Output = Vec<i32>;

    fn collate(&self, batch: Vec<i32>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;
        Ok(batch)
    }
}

impl<T: TensorElement + Copy> Collate<Vec<Tensor<T>>> for DefaultCollate {
    type Output = Vec<Tensor<T>>;

    fn collate(&self, batch: Vec<Vec<Tensor<T>>>) -> Result<Self::Output> {
        self.validate_batch(&batch)?;

        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Check that all samples have the same number of tensors
        let num_tensors = batch[0].len();
        for sample in &batch {
            if sample.len() != num_tensors {
                return Err(TorshError::InvalidArgument(
                    "All samples must have the same number of tensors".to_string(),
                ));
            }
        }

        // Group tensors by position and stack them
        let mut result = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let tensors_to_stack: Vec<Tensor<T>> = batch
                .iter()
                .map(|sample| sample[tensor_idx].clone())
                .collect();

            // Stack tensors at this position
            let stacked = super::stacking::TensorStacker::new().stack(&tensors_to_stack, 0)?;
            result.push(stacked);
        }

        Ok(result)
    }
}

/// Generic collate function wrapper
pub struct CollateFn<F> {
    f: F,
}

impl<F> CollateFn<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<T, O, F> Collate<T> for CollateFn<F>
where
    F: Fn(Vec<T>) -> Result<O>,
{
    type Output = O;

    fn collate(&self, batch: Vec<T>) -> Result<Self::Output> {
        (self.f)(batch)
    }
}

/// Convenience function to create default collate function
pub fn collate_fn<T>() -> DefaultCollate {
    DefaultCollate
}
