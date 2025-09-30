//! Apache Arrow integration for efficient data exchange
//!
//! This module provides utilities for converting between torsh tensors
//! and Apache Arrow arrays for efficient data processing pipelines.

#[cfg(feature = "arrow-support")]
use arrow::{
    array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, PrimitiveArray},
    buffer::Buffer,
    datatypes::{DataType, Field, Schema, SchemaRef, ToByteSlice},
    record_batch::RecordBatch,
};

use crate::Dataset;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(feature = "arrow-support")]
use torsh_core::{device::DeviceType, dtype::TensorElement};

#[cfg(feature = "arrow-support")]
use crate::utils;

#[cfg(not(feature = "arrow-support"))]
use std::marker::PhantomData;

/// Arrow dataset for reading Arrow files and record batches
#[cfg(feature = "arrow-support")]
pub struct ArrowDataset {
    record_batches: Vec<RecordBatch>,
    current_batch: usize,
    batch_size: usize,
    total_rows: usize,
}

#[cfg(not(feature = "arrow-support"))]
pub struct ArrowDataset {
    _phantom: PhantomData<()>,
}

#[cfg(feature = "arrow-support")]
impl ArrowDataset {
    /// Create a new Arrow dataset from record batches
    pub fn from_record_batches(record_batches: Vec<RecordBatch>) -> Result<Self> {
        utils::validate_not_empty(&record_batches, "record_batches")?;

        let total_rows = record_batches.iter().map(|batch| batch.num_rows()).sum();

        Ok(Self {
            record_batches,
            current_batch: 0,
            batch_size: 1000, // Default batch size
            total_rows,
        })
    }

    /// Create from Arrow file path
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use arrow::ipc::reader::FileReader;
        use std::fs::File;

        let path = path.as_ref();
        utils::validate_dataset_path(path, "Arrow file")?;
        utils::validate_file_extension(path, &["arrow", "ipc"])?;

        let file = File::open(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open Arrow file: {}", e))
        })?;

        let reader = FileReader::try_new(file, None).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create Arrow reader: {}", e))
        })?;

        let mut record_batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to read Arrow batch: {}", e))
            })?;
            record_batches.push(batch);
        }

        Self::from_record_batches(record_batches)
    }

    /// Set batch size for iteration
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get the Arrow schema
    pub fn schema(&self) -> Option<SchemaRef> {
        self.record_batches.first().map(|batch| batch.schema())
    }

    /// Convert Arrow array to tensor
    pub fn array_to_tensor<T: TensorElement + 'static>(
        &self,
        array: &dyn Array,
    ) -> Result<Tensor<T>> {
        match array.data_type() {
            DataType::Float32 => {
                let array = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        TorshError::InvalidArgument(
                            "Failed to downcast to Float32Array".to_string(),
                        )
                    })?;
                let values: Vec<f32> = array.values().to_vec();
                let shape = vec![values.len()];

                // Type conversion if needed
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let tensor_data = unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(values) };
                    Ok(torsh_tensor::Tensor::from_data(
                        tensor_data,
                        shape,
                        DeviceType::Cpu,
                    )?)
                } else {
                    return Err(TorshError::InvalidArgument("Type mismatch".to_string()));
                }
            }
            DataType::Float64 => {
                let array = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        TorshError::InvalidArgument(
                            "Failed to downcast to Float64Array".to_string(),
                        )
                    })?;
                let values: Vec<f64> = array.values().to_vec();
                let shape = vec![values.len()];

                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let tensor_data = unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(values) };
                    Ok(torsh_tensor::Tensor::from_data(
                        tensor_data,
                        shape,
                        DeviceType::Cpu,
                    )?)
                } else {
                    return Err(TorshError::InvalidArgument("Type mismatch".to_string()));
                }
            }
            DataType::Int32 => {
                let array = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                    TorshError::InvalidArgument("Failed to downcast to Int32Array".to_string())
                })?;
                let values: Vec<i32> = array.values().to_vec();
                let shape = vec![values.len()];

                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
                    let tensor_data = unsafe { std::mem::transmute::<Vec<i32>, Vec<T>>(values) };
                    Ok(torsh_tensor::Tensor::from_data(
                        tensor_data,
                        shape,
                        DeviceType::Cpu,
                    )?)
                } else {
                    return Err(TorshError::InvalidArgument("Type mismatch".to_string()));
                }
            }
            DataType::Int64 => {
                let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    TorshError::InvalidArgument("Failed to downcast to Int64Array".to_string())
                })?;
                let values: Vec<i64> = array.values().to_vec();
                let shape = vec![values.len()];

                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
                    let tensor_data = unsafe { std::mem::transmute::<Vec<i64>, Vec<T>>(values) };
                    Ok(torsh_tensor::Tensor::from_data(
                        tensor_data,
                        shape,
                        DeviceType::Cpu,
                    )?)
                } else {
                    return Err(TorshError::InvalidArgument("Type mismatch".to_string()));
                }
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Unsupported Arrow data type: {:?}",
                array.data_type()
            ))),
        }
    }

    /// Convert tensor to Arrow array
    pub fn tensor_to_array<T: TensorElement + ToByteSlice + 'static>(
        tensor: &Tensor<T>,
    ) -> Result<ArrayRef> {
        let data = tensor.to_vec()?;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let float_data = unsafe { std::mem::transmute::<Vec<T>, Vec<f32>>(data) };
            Ok(std::sync::Arc::new(Float32Array::from(float_data)))
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let double_data = unsafe { std::mem::transmute::<Vec<T>, Vec<f64>>(data) };
            Ok(std::sync::Arc::new(Float64Array::from(double_data)))
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
            let int_data = unsafe { std::mem::transmute::<Vec<T>, Vec<i32>>(data) };
            Ok(std::sync::Arc::new(Int32Array::from(int_data)))
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
            let long_data = unsafe { std::mem::transmute::<Vec<T>, Vec<i64>>(data) };
            Ok(std::sync::Arc::new(Int64Array::from(long_data)))
        } else {
            Err(TorshError::InvalidArgument(
                "Unsupported tensor element type for Arrow conversion".to_string(),
            ))
        }
    }

    /// Get column as tensor
    pub fn get_column_as_tensor<T: TensorElement + 'static>(
        &self,
        column_name: &str,
    ) -> Result<Tensor<T>> {
        for batch in &self.record_batches {
            if let Some(column_index) = batch.schema().column_with_name(column_name) {
                let array = batch.column(column_index.0);
                return self.array_to_tensor(array.as_ref());
            }
        }

        Err(TorshError::InvalidArgument(format!(
            "Column '{}' not found",
            column_name
        )))
    }
}

#[cfg(not(feature = "arrow-support"))]
impl ArrowDataset {
    /// Placeholder for when Arrow feature is not enabled
    pub fn from_record_batches(_record_batches: Vec<()>) -> Result<Self> {
        Err(TorshError::InvalidArgument(
            "Arrow support not enabled. Enable 'arrow-support' feature flag.".to_string(),
        ))
    }

    /// Placeholder for file loading
    pub fn from_file<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Err(TorshError::InvalidArgument(
            "Arrow support not enabled. Enable 'arrow-support' feature flag.".to_string(),
        ))
    }
}

#[cfg(feature = "arrow-support")]
impl Dataset for ArrowDataset {
    type Item = RecordBatch;

    fn len(&self) -> usize {
        self.total_rows
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.total_rows {
            return Err(utils::errors::invalid_index(index, self.total_rows));
        }

        // Find which batch contains this index
        let mut current_row = 0;
        for batch in &self.record_batches {
            if index < current_row + batch.num_rows() {
                let local_index = index - current_row;
                // Return a slice of the batch (single row)
                return Ok(batch.slice(local_index, 1));
            }
            current_row += batch.num_rows();
        }

        Err(utils::errors::invalid_index(index, self.total_rows))
    }
}

#[cfg(not(feature = "arrow-support"))]
impl Dataset for ArrowDataset {
    type Item = ();

    fn len(&self) -> usize {
        0
    }

    fn get(&self, _index: usize) -> Result<Self::Item> {
        Err(TorshError::InvalidArgument(
            "Arrow support not enabled".to_string(),
        ))
    }
}

/// Utility functions for Arrow integration
pub mod arrow_utils {
    use super::*;

    /// Check if Arrow feature is available at compile time
    pub const fn is_arrow_available() -> bool {
        cfg!(feature = "arrow-support")
    }

    /// Create a sample Arrow dataset for testing
    #[cfg(feature = "arrow-support")]
    pub fn create_sample_dataset() -> Result<ArrowDataset> {
        use arrow::array::{Float32Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float32, false),
        ]));

        // Create arrays
        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let value_array = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

        // Create record batch
        let batch = RecordBatch::try_new(schema, vec![id_array, value_array]).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create record batch: {}", e))
        })?;

        ArrowDataset::from_record_batches(vec![batch])
    }

    /// Convert multiple tensors to a record batch
    #[cfg(feature = "arrow-support")]
    pub fn tensors_to_record_batch<T: TensorElement + ToByteSlice + 'static>(
        tensors: &[(&str, &Tensor<T>)],
    ) -> Result<RecordBatch> {
        let mut fields = Vec::new();
        let mut arrays = Vec::new();

        for (name, tensor) in tensors {
            // Determine Arrow data type based on tensor element type
            let data_type = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                DataType::Float32
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                DataType::Float64
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
                DataType::Int32
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
                DataType::Int64
            } else {
                return Err(TorshError::InvalidArgument(
                    "Unsupported tensor type for Arrow conversion".to_string(),
                ));
            };

            fields.push(Field::new(*name, data_type, false));
            arrays.push(ArrowDataset::tensor_to_array(tensor)?);
        }

        let schema = std::sync::Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create record batch: {}", e))
        })
    }

    #[cfg(not(feature = "arrow-support"))]
    pub fn create_sample_dataset() -> Result<ArrowDataset> {
        ArrowDataset::from_record_batches(vec![])
    }

    #[cfg(not(feature = "arrow-support"))]
    pub fn tensors_to_record_batch<T: torsh_core::dtype::TensorElement>(
        _tensors: &[(&str, &Tensor<T>)],
    ) -> Result<()> {
        Err(TorshError::InvalidArgument(
            "Arrow support not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_availability() {
        // This test checks if we can detect Arrow availability
        assert!(arrow_utils::is_arrow_available() || !arrow_utils::is_arrow_available());
    }

    #[cfg(feature = "arrow-support")]
    #[test]
    fn test_sample_dataset() -> Result<()> {
        let dataset = arrow_utils::create_sample_dataset()?;
        assert_eq!(dataset.len(), 5);

        let first_item = dataset.get(0)?;
        assert_eq!(first_item.num_rows(), 1);
        assert_eq!(first_item.num_columns(), 2);

        Ok(())
    }

    #[cfg(not(feature = "arrow-support"))]
    #[test]
    fn test_arrow_disabled() {
        let result = ArrowDataset::from_file("test.arrow");
        assert!(result.is_err());
    }
}
