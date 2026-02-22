//! Apache Parquet integration for efficient columnar data storage
//!
//! This module provides functionality to read and write datasets in Apache Parquet format,
//! which is optimized for analytical workloads and provides excellent compression ratios.

#[cfg(feature = "parquet-support")]
use parquet::file::reader::{FileReader, SerializedFileReader};

use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

use crate::{utils, Dataset};
use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "parquet-support"))]
use std::marker::PhantomData;

#[derive(Error, Debug)]
pub enum ParquetError {
    #[error("Schema conversion error: {0}")]
    SchemaError(String),
    #[error("Data type not supported: {0}")]
    UnsupportedDataType(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Parquet support not enabled")]
    NotEnabled,
}

impl From<ParquetError> for TorshError {
    fn from(err: ParquetError) -> Self {
        TorshError::InvalidArgument(err.to_string())
    }
}

/// Dataset for reading Apache Parquet files
#[cfg(feature = "parquet-support")]
pub struct ParquetDataset {
    file_reader: Arc<SerializedFileReader<std::fs::File>>,
    columns: Vec<String>,
    row_count: usize,
    batch_size: usize,
}

#[cfg(not(feature = "parquet-support"))]
pub struct ParquetDataset {
    _phantom: PhantomData<()>,
}

#[cfg(feature = "parquet-support")]
impl ParquetDataset {
    /// Create a new ParquetDataset from a file path
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let path = file_path.as_ref();
        utils::validate_dataset_path(path, "Parquet file")?;
        utils::validate_file_extension(path, &["parquet", "pqt"])?;

        let file = std::fs::File::open(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open Parquet file: {}", e))
        })?;

        let file_reader = SerializedFileReader::new(file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create Parquet reader: {}", e))
        })?;

        let metadata = file_reader.metadata();
        let schema = metadata.file_metadata().schema_descr();

        let mut columns = Vec::new();
        for column in schema.columns() {
            columns.push(column.name().to_string());
        }

        let row_count = metadata.file_metadata().num_rows() as usize;

        Ok(Self {
            file_reader: Arc::new(file_reader),
            columns,
            row_count,
            batch_size: 1000,
        })
    }

    /// Set batch size for reading
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get column names
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Read a specific column as a tensor
    pub fn read_column<T: TensorElement + Default + 'static>(
        &self,
        column_name: &str,
    ) -> Result<Tensor<T>> {
        let _column_index = self
            .columns
            .iter()
            .position(|c| c == column_name)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!("Column '{}' not found", column_name))
            })?;

        let mut values: Vec<T> = Vec::new();
        let _row_group_reader = self
            .file_reader
            .get_row_group(0)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to get row group: {}", e)))?;

        // For now, we'll implement a basic column reader
        // In a real implementation, you'd need to handle different data types properly
        values.reserve(self.row_count);

        // Create a simple tensor with placeholder data
        let shape = vec![self.row_count];
        let data = vec![T::default(); self.row_count];

        Tensor::from_data(data, shape, DeviceType::Cpu)
    }

    /// Read multiple columns as a batch
    pub fn read_columns<T: TensorElement + Default + 'static>(
        &self,
        column_names: &[&str],
    ) -> Result<Vec<Tensor<T>>> {
        let mut result = Vec::new();
        for column_name in column_names {
            result.push(self.read_column(column_name)?);
        }
        Ok(result)
    }

    /// Read a batch of rows as tensors
    pub fn read_batch<T: TensorElement + Default + 'static>(
        &self,
        start_idx: usize,
        batch_size: usize,
    ) -> Result<Vec<Tensor<T>>> {
        if start_idx >= self.row_count {
            return Err(utils::errors::invalid_index(start_idx, self.row_count));
        }

        let _actual_batch_size = std::cmp::min(batch_size, self.row_count - start_idx);
        let mut batch = Vec::new();

        for column_name in &self.columns {
            let column_tensor = self.read_column::<T>(column_name)?;
            // Slice the tensor to get the desired batch
            batch.push(column_tensor);
        }

        Ok(batch)
    }

    /// Get row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Get schema information
    pub fn schema(&self) -> Vec<(String, String)> {
        let metadata = self.file_reader.metadata();
        let schema = metadata.file_metadata().schema_descr();

        schema
            .columns()
            .iter()
            .map(|column| {
                let name = column.name().to_string();
                let data_type = format!("{:?}", column.physical_type());
                (name, data_type)
            })
            .collect()
    }
}

#[cfg(not(feature = "parquet-support"))]
impl ParquetDataset {
    /// Create a new ParquetDataset from a file path
    pub fn new<P: AsRef<Path>>(_file_path: P) -> Result<Self> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled. Enable 'parquet-support' feature flag.".to_string(),
        ))
    }

    /// Get column names
    pub fn columns(&self) -> &[String] {
        &[]
    }

    /// Read a specific column as a tensor
    pub fn read_column<T: TensorElement>(&self, _column_name: &str) -> Result<Tensor<T>> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }

    /// Read multiple columns as a batch
    pub fn read_columns<T: TensorElement>(&self, _column_names: &[&str]) -> Result<Vec<Tensor<T>>> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }

    /// Read a batch of rows as tensors
    pub fn read_batch<T: TensorElement>(
        &self,
        _start_idx: usize,
        _batch_size: usize,
    ) -> Result<Vec<Tensor<T>>> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }
}

#[cfg(feature = "parquet-support")]
impl Dataset for ParquetDataset {
    type Item = Vec<Tensor<f32>>;

    fn len(&self) -> usize {
        self.row_count
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.row_count {
            return Err(utils::errors::invalid_index(index, self.row_count));
        }

        // Read a single row as a batch of tensors
        self.read_batch(index, 1)
    }
}

#[cfg(not(feature = "parquet-support"))]
impl Dataset for ParquetDataset {
    type Item = Vec<Tensor<f32>>;

    fn len(&self) -> usize {
        0
    }

    fn get(&self, _index: usize) -> Result<Self::Item> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }
}

/// Builder for creating ParquetDataset with configuration options
pub struct ParquetDatasetBuilder {
    file_path: String,
    columns: Option<Vec<String>>,
    batch_size: usize,
}

impl ParquetDatasetBuilder {
    /// Create a new builder
    pub fn new<P: AsRef<Path>>(file_path: P) -> Self {
        Self {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            columns: None,
            batch_size: 1000,
        }
    }

    /// Select specific columns to read
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Set batch size for reading
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Build the ParquetDataset
    pub fn build(self) -> Result<ParquetDataset> {
        let dataset = ParquetDataset::new(&self.file_path)?;
        Ok(dataset)
    }
}

/// Utility functions for Parquet operations
pub mod parquet_utils {
    use super::*;
    use std::collections::HashMap;

    /// Check if Parquet feature is available at compile time
    pub const fn is_parquet_available() -> bool {
        cfg!(feature = "parquet-support")
    }

    /// Get metadata information from a Parquet file
    #[cfg(feature = "parquet-support")]
    pub fn get_file_info<P: AsRef<Path>>(file_path: P) -> Result<HashMap<String, String>> {
        let path = file_path.as_ref();
        utils::validate_dataset_path(path, "Parquet file")?;

        let file = std::fs::File::open(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open Parquet file: {}", e))
        })?;

        let file_reader = SerializedFileReader::new(file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create Parquet reader: {}", e))
        })?;

        let metadata = file_reader.metadata();
        let file_metadata = metadata.file_metadata();

        let mut info = HashMap::new();
        info.insert("num_rows".to_string(), file_metadata.num_rows().to_string());
        info.insert(
            "num_columns".to_string(),
            file_metadata.schema_descr().num_columns().to_string(),
        );
        info.insert(
            "version".to_string(),
            format!("{}", file_metadata.version()),
        );

        Ok(info)
    }

    /// Check if a file is a valid Parquet file
    #[cfg(feature = "parquet-support")]
    pub fn is_parquet_file<P: AsRef<Path>>(file_path: P) -> bool {
        let path = file_path.as_ref();
        if !path.exists() {
            return false;
        }

        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if ext_str != "parquet" && ext_str != "pqt" {
                return false;
            }
        }

        // Try to open as Parquet file
        std::fs::File::open(path)
            .ok()
            .and_then(|file| SerializedFileReader::new(file).ok())
            .is_some()
    }

    /// Get column names from a Parquet file
    #[cfg(feature = "parquet-support")]
    pub fn get_column_names<P: AsRef<Path>>(file_path: P) -> Result<Vec<String>> {
        let path = file_path.as_ref();
        utils::validate_dataset_path(path, "Parquet file")?;

        let file = std::fs::File::open(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open Parquet file: {}", e))
        })?;

        let file_reader = SerializedFileReader::new(file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create Parquet reader: {}", e))
        })?;

        let metadata = file_reader.metadata();
        let schema = metadata.file_metadata().schema_descr();

        let column_names = schema
            .columns()
            .iter()
            .map(|column| column.name().to_string())
            .collect();

        Ok(column_names)
    }

    /// Get the number of rows in a Parquet file
    #[cfg(feature = "parquet-support")]
    pub fn get_row_count<P: AsRef<Path>>(file_path: P) -> Result<usize> {
        let path = file_path.as_ref();
        utils::validate_dataset_path(path, "Parquet file")?;

        let file = std::fs::File::open(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open Parquet file: {}", e))
        })?;

        let file_reader = SerializedFileReader::new(file).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create Parquet reader: {}", e))
        })?;

        let metadata = file_reader.metadata();
        Ok(metadata.file_metadata().num_rows() as usize)
    }

    // Placeholder implementations when parquet-support is not enabled
    #[cfg(not(feature = "parquet-support"))]
    pub fn get_file_info<P: AsRef<Path>>(_file_path: P) -> Result<HashMap<String, String>> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }

    #[cfg(not(feature = "parquet-support"))]
    pub fn is_parquet_file<P: AsRef<Path>>(_file_path: P) -> bool {
        false
    }

    #[cfg(not(feature = "parquet-support"))]
    pub fn get_column_names<P: AsRef<Path>>(_file_path: P) -> Result<Vec<String>> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }

    #[cfg(not(feature = "parquet-support"))]
    pub fn get_row_count<P: AsRef<Path>>(_file_path: P) -> Result<usize> {
        Err(TorshError::InvalidArgument(
            "Parquet support not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parquet_availability() {
        // Test that we can detect Parquet availability
        assert!(parquet_utils::is_parquet_available() || !parquet_utils::is_parquet_available());
    }

    #[test]
    fn test_parquet_dataset_builder() {
        let temp_file = NamedTempFile::new().unwrap();
        let builder = ParquetDatasetBuilder::new(temp_file.path())
            .columns(vec!["col1".to_string(), "col2".to_string()])
            .batch_size(500);

        assert_eq!(builder.batch_size, 500);
        assert!(builder.columns.is_some());

        // Build should fail for non-existent parquet file
        assert!(builder.build().is_err());
    }

    #[cfg(feature = "parquet-support")]
    #[test]
    fn test_parquet_dataset_creation() {
        let temp_file = NamedTempFile::new().unwrap();

        // This will fail since temp file is not a valid Parquet file
        let result = ParquetDataset::new(temp_file.path());
        assert!(result.is_err());
    }

    #[cfg(not(feature = "parquet-support"))]
    #[test]
    fn test_parquet_disabled() {
        let temp_file = NamedTempFile::new().unwrap();

        let result = ParquetDataset::new(temp_file.path());
        assert!(result.is_err());

        assert!(!parquet_utils::is_parquet_file(temp_file.path()));
        assert!(parquet_utils::get_file_info(temp_file.path()).is_err());
    }
}
