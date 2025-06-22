//! Batch collation functions

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::{creation::ones, Tensor};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Trait for collating a batch of samples
pub trait Collate<T> {
    /// Output type after collation
    type Output;

    /// Collate a batch of samples
    fn collate(&self, batch: Vec<T>) -> Result<Self::Output>;
}

/// Default collation function
pub struct DefaultCollate;

impl<T: TensorElement> Collate<Tensor<T>> for DefaultCollate {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Stack tensors along a new dimension (batch dimension)
        stack_tensors(&batch, 0)
    }
}

impl<T: TensorElement> Collate<Vec<Tensor<T>>> for DefaultCollate {
    type Output = Vec<Tensor<T>>;

    fn collate(&self, batch: Vec<Vec<Tensor<T>>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        let num_tensors = batch[0].len();
        let mut collated = Vec::with_capacity(num_tensors);

        // Collate each tensor position across the batch
        for i in 0..num_tensors {
            let tensors: Vec<Tensor<T>> = batch.iter().map(|sample| sample[i].clone()).collect();

            collated.push(stack_tensors(&tensors, 0)?);
        }

        Ok(collated)
    }
}

/// Stack tensors along a new dimension
fn stack_tensors<T: TensorElement>(tensors: &[Tensor<T>], dim: usize) -> Result<Tensor<T>> {
    if tensors.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot stack empty tensor list".to_string(),
        ));
    }

    // Check that all tensors have the same shape
    let first_shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != first_shape {
            return Err(TorshError::ShapeMismatch {
                expected: first_shape.dims().to_vec(),
                got: tensor.shape().dims().to_vec(),
            });
        }
    }

    // TODO: Implement actual stacking when tensor operations are complete
    // For now, return the first tensor as a placeholder
    Ok(tensors[0].clone())
}

/// Collation function that can be customized
pub struct CollateFn<F> {
    func: F,
}

impl<F> CollateFn<F> {
    /// Create a new collation function
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<T, O, F> Collate<T> for CollateFn<F>
where
    F: Fn(Vec<T>) -> Result<O>,
{
    type Output = O;

    fn collate(&self, batch: Vec<T>) -> Result<Self::Output> {
        (self.func)(batch)
    }
}

/// Default collation function instance
pub fn collate_fn<T>() -> DefaultCollate {
    DefaultCollate
}

/// Padding collation for variable-length sequences
pub struct PadCollate<T: TensorElement> {
    padding_value: T,
}

impl<T: TensorElement> PadCollate<T> {
    /// Create a new padding collation function
    pub fn new(padding_value: T) -> Self {
        Self { padding_value }
    }
}

impl<T: TensorElement> Collate<Tensor<T>> for PadCollate<T> {
    type Output = Tensor<T>;

    fn collate(&self, batch: Vec<Tensor<T>>) -> Result<Self::Output> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Find maximum dimensions
        let ndim = batch[0].ndim();
        let mut max_dims = vec![0; ndim];

        for tensor in &batch {
            if tensor.ndim() != ndim {
                return Err(TorshError::InvalidArgument(
                    "All tensors must have the same number of dimensions".to_string(),
                ));
            }

            for (i, max_dim) in max_dims.iter_mut().enumerate().take(ndim) {
                let size = tensor.size(i as i32)?;
                if size > *max_dim {
                    *max_dim = size;
                }
            }
        }

        // TODO: Implement actual padding when tensor operations are complete
        // For now, return the first tensor as a placeholder
        Ok(batch[0].clone())
    }
}

/// Custom collation examples
pub mod examples {
    use super::*;

    /// Collate function for (data, label) tuples
    pub fn collate_data_label<T: TensorElement>(
        batch: Vec<(Tensor<T>, Tensor<i64>)>,
    ) -> Result<(Tensor<T>, Tensor<i64>)> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        let (data_batch, label_batch): (Vec<_>, Vec<_>) = batch.into_iter().unzip();

        let data = stack_tensors(&data_batch, 0)?;
        let labels = stack_tensors(&label_batch, 0)?;

        Ok((data, labels))
    }

    /// Collate function for dictionaries (key-value pairs)
    pub fn collate_dict<T: TensorElement>(
        batch: Vec<Vec<(&str, Tensor<T>)>>,
    ) -> Result<Vec<(&str, Tensor<T>)>> {
        if batch.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot collate empty batch".to_string(),
            ));
        }

        // Assuming all items have the same keys
        let keys: Vec<&str> = batch[0].iter().map(|(k, _)| *k).collect();
        let mut result = Vec::with_capacity(keys.len());

        for (i, key) in keys.iter().enumerate() {
            let tensors: Vec<Tensor<T>> = batch.iter().map(|sample| sample[i].1.clone()).collect();

            let stacked = stack_tensors(&tensors, 0)?;
            result.push((*key, stacked));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_collate() {
        let batch = vec![
            ones::<f32>(&[3, 4]),
            ones::<f32>(&[3, 4]),
            ones::<f32>(&[3, 4]),
        ];

        let collate = DefaultCollate;
        let result = collate.collate(batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_collate_fn() {
        let collate = CollateFn::new(|batch: Vec<i32>| Ok(batch.iter().sum::<i32>()));

        let result = collate.collate(vec![1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result, 15);
    }

    #[test]
    fn test_pad_collate() {
        let batch = vec![ones::<f32>(&[2, 3]), ones::<f32>(&[2, 3])];

        let collate = PadCollate::new(0.0f32);
        let result = collate.collate(batch);
        assert!(result.is_ok());
    }
}
