//! Custom collation examples

use super::optimized::stack_tensors;
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Collate function for (data, label) tuples
pub fn collate_data_label<T: TensorElement + Copy>(
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
pub fn collate_dict<T: TensorElement + Copy>(
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
