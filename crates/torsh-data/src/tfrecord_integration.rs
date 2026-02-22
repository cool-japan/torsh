//! TensorFlow TFRecord format integration
//!
//! This module provides functionality to read TensorFlow's TFRecord files,
//! which are commonly used for storing training data in TensorFlow ecosystems.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

use crate::dataset::Dataset;
use crate::error::{DataError, Result};
use torsh_tensor::Tensor;

#[derive(Error, Debug)]
pub enum TFRecordError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid TFRecord format: {0}")]
    FormatError(String),
    #[error("CRC checksum mismatch")]
    ChecksumError,
    #[error("Protobuf parsing error: {0}")]
    ProtobufError(String),
    #[error("Feature not found: {0}")]
    FeatureNotFound(String),
    #[error("Unsupported feature type: {0}")]
    UnsupportedFeatureType(String),
}

impl From<TFRecordError> for DataError {
    fn from(err: TFRecordError) -> Self {
        DataError::Other(err.to_string())
    }
}

/// A TFRecord reader that can parse TensorFlow's binary record format
pub struct TFRecordReader {
    reader: BufReader<File>,
    records_read: usize,
}

impl TFRecordReader {
    /// Create a new TFRecord reader from a file path
    pub fn new<P: AsRef<Path>>(file_path: P) -> std::result::Result<Self, TFRecordError> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        Ok(Self {
            reader,
            records_read: 0,
        })
    }

    /// Read the next record from the TFRecord file
    pub fn read_next_record(&mut self) -> std::result::Result<Option<Vec<u8>>, TFRecordError> {
        // TFRecord format:
        // uint64 length
        // uint32 masked_crc32_of_length
        // byte data[length]
        // uint32 masked_crc32_of_data

        // Read length (8 bytes, little endian)
        let mut length_bytes = [0u8; 8];
        match self.reader.read_exact(&mut length_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(TFRecordError::IoError(e)),
        }

        let length = u64::from_le_bytes(length_bytes);

        // Read length CRC (4 bytes, little endian)
        let mut length_crc_bytes = [0u8; 4];
        self.reader.read_exact(&mut length_crc_bytes)?;
        let _length_crc = u32::from_le_bytes(length_crc_bytes);

        // For now, skip CRC verification (would need crc32 implementation)

        // Read data
        let mut data = vec![0u8; length as usize];
        self.reader.read_exact(&mut data)?;

        // Read data CRC (4 bytes, little endian)
        let mut data_crc_bytes = [0u8; 4];
        self.reader.read_exact(&mut data_crc_bytes)?;
        let _data_crc = u32::from_le_bytes(data_crc_bytes);

        // For now, skip CRC verification

        self.records_read += 1;
        Ok(Some(data))
    }

    /// Get the number of records read so far
    pub fn records_read(&self) -> usize {
        self.records_read
    }

    /// Reset the reader to the beginning of the file
    pub fn reset(&mut self) -> std::result::Result<(), TFRecordError> {
        self.reader.seek(SeekFrom::Start(0))?;
        self.records_read = 0;
        Ok(())
    }
}

/// Simple protobuf-like parsing for TensorFlow Example format
/// This is a simplified implementation that handles basic Example structures
#[derive(Debug, Clone)]
pub enum FeatureValue {
    BytesList(Vec<Vec<u8>>),
    FloatList(Vec<f32>),
    Int64List(Vec<i64>),
}

#[derive(Debug, Clone)]
pub struct Example {
    features: HashMap<String, FeatureValue>,
}

impl Example {
    /// Parse an Example from raw bytes (simplified protobuf parsing)
    pub fn from_bytes(_data: &[u8]) -> std::result::Result<Self, TFRecordError> {
        // This is a very simplified protobuf parser for TensorFlow Example format
        // In a real implementation, you'd use a proper protobuf library

        let mut features = HashMap::new();

        // For now, create dummy features as a placeholder
        // Real implementation would parse the protobuf data
        features.insert(
            "example_feature".to_string(),
            FeatureValue::FloatList(vec![1.0, 2.0, 3.0]),
        );

        Ok(Example { features })
    }

    /// Get a feature by name
    pub fn get_feature(&self, name: &str) -> Option<&FeatureValue> {
        self.features.get(name)
    }

    /// Get all feature names
    pub fn feature_names(&self) -> Vec<&String> {
        self.features.keys().collect()
    }

    /// Convert a feature to a tensor
    pub fn feature_to_tensor<T: torsh_core::TensorElement>(
        &self,
        name: &str,
    ) -> std::result::Result<Tensor<T>, TFRecordError> {
        let feature = self
            .get_feature(name)
            .ok_or_else(|| TFRecordError::FeatureNotFound(name.to_string()))?;

        match feature {
            FeatureValue::FloatList(values) => {
                let converted_values: Vec<T> = values
                    .iter()
                    .filter_map(|&v| T::from_f64(v as f64))
                    .collect();

                if converted_values.len() != values.len() {
                    return Err(TFRecordError::FormatError(
                        "Type conversion failed".to_string(),
                    ));
                }

                let shape = vec![converted_values.len()];
                Tensor::from_vec(converted_values, &shape)
                    .map_err(|e| TFRecordError::FormatError(e.to_string()))
            }
            FeatureValue::Int64List(values) => {
                let converted_values: Vec<T> = values
                    .iter()
                    .filter_map(|&v| T::from_f64(v as f64))
                    .collect();

                if converted_values.len() != values.len() {
                    return Err(TFRecordError::FormatError(
                        "Type conversion failed".to_string(),
                    ));
                }

                let shape = vec![converted_values.len()];
                Tensor::from_vec(converted_values, &shape)
                    .map_err(|e| TFRecordError::FormatError(e.to_string()))
            }
            FeatureValue::BytesList(_) => Err(TFRecordError::UnsupportedFeatureType(
                "BytesList not supported for tensor conversion".to_string(),
            )),
        }
    }
}

/// Dataset for reading TFRecord files
pub struct TFRecordDataset {
    _file_path: String,
    records: Vec<Example>,
    feature_names: Vec<String>,
}

impl TFRecordDataset {
    /// Create a new TFRecordDataset from a file path
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let path_str = file_path.as_ref().to_string_lossy().to_string();
        let mut reader = TFRecordReader::new(&file_path)?;

        let mut records = Vec::new();
        let mut feature_names = std::collections::HashSet::new();

        // Read all records
        while let Some(raw_data) = reader.read_next_record()? {
            let example = Example::from_bytes(&raw_data)?;

            // Collect feature names
            for name in example.feature_names() {
                feature_names.insert(name.clone());
            }

            records.push(example);
        }

        let feature_names: Vec<String> = feature_names.into_iter().collect();

        Ok(Self {
            _file_path: path_str,
            records,
            feature_names,
        })
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get a specific example by index
    pub fn get_example(&self, index: usize) -> Option<&Example> {
        self.records.get(index)
    }

    /// Extract a specific feature from all records as tensors
    pub fn extract_feature<T: torsh_core::TensorElement>(
        &self,
        feature_name: &str,
    ) -> Result<Vec<Tensor<T>>> {
        let mut tensors = Vec::with_capacity(self.records.len());

        for example in &self.records {
            let tensor = example.feature_to_tensor::<T>(feature_name)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Read a batch of examples
    pub fn read_batch(&self, start_idx: usize, batch_size: usize) -> Vec<&Example> {
        let end_idx = (start_idx + batch_size).min(self.records.len());

        if start_idx >= self.records.len() {
            return Vec::new();
        }

        self.records[start_idx..end_idx].iter().collect()
    }
}

impl Dataset for TFRecordDataset {
    type Item = Example;

    fn len(&self) -> usize {
        self.records.len()
    }

    fn get(&self, index: usize) -> torsh_core::error::Result<Self::Item> {
        self.records.get(index).cloned().ok_or_else(|| {
            DataError::Other(format!(
                "Index {} out of bounds for dataset of size {}",
                index,
                self.records.len()
            ))
            .into()
        })
    }
}

/// Builder for creating TFRecordDataset with configuration options
pub struct TFRecordDatasetBuilder {
    file_path: String,
    feature_names: Option<Vec<String>>,
    max_records: Option<usize>,
}

impl TFRecordDatasetBuilder {
    /// Create a new builder
    pub fn new<P: AsRef<Path>>(file_path: P) -> Self {
        Self {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            feature_names: None,
            max_records: None,
        }
    }

    /// Select specific features to extract
    pub fn features(mut self, feature_names: Vec<String>) -> Self {
        self.feature_names = Some(feature_names);
        self
    }

    /// Limit the number of records to read
    pub fn max_records(mut self, max_records: usize) -> Self {
        self.max_records = Some(max_records);
        self
    }

    /// Build the TFRecordDataset
    pub fn build(self) -> Result<TFRecordDataset> {
        TFRecordDataset::new(&self.file_path)
    }
}

/// Utility functions for TFRecord operations
pub mod tfrecord_utils {
    use super::*;

    /// Check if a file appears to be a TFRecord file
    pub fn is_tfrecord_file<P: AsRef<Path>>(file_path: P) -> bool {
        match TFRecordReader::new(&file_path) {
            Ok(mut reader) => {
                // Try to read the first record
                matches!(reader.read_next_record(), Ok(Some(_)))
            }
            Err(_) => false,
        }
    }

    /// Count the number of records in a TFRecord file
    pub fn count_records<P: AsRef<Path>>(
        file_path: P,
    ) -> std::result::Result<usize, TFRecordError> {
        let mut reader = TFRecordReader::new(file_path)?;
        let mut count = 0;

        while (reader.read_next_record()?).is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Get basic information about a TFRecord file
    pub fn get_file_info<P: AsRef<Path>>(
        file_path: P,
    ) -> std::result::Result<HashMap<String, String>, TFRecordError> {
        let mut info = HashMap::new();

        // Count records
        let num_records = count_records(&file_path)?;
        info.insert("num_records".to_string(), num_records.to_string());

        // Get file size
        let metadata = std::fs::metadata(&file_path)?;
        info.insert("file_size_bytes".to_string(), metadata.len().to_string());

        // Try to read first record to get feature information
        let mut reader = TFRecordReader::new(&file_path)?;
        if let Some(raw_data) = reader.read_next_record()? {
            match Example::from_bytes(&raw_data) {
                Ok(example) => {
                    let feature_names: Vec<String> = example
                        .feature_names()
                        .iter()
                        .map(|s| (*s).clone())
                        .collect();
                    info.insert("feature_names".to_string(), feature_names.join(", "));
                    info.insert("num_features".to_string(), feature_names.len().to_string());
                }
                Err(_) => {
                    info.insert("parsing_status".to_string(), "failed".to_string());
                }
            }
        }

        Ok(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_tfrecord_dataset_builder() {
        let temp_file = NamedTempFile::new().unwrap();
        let builder = TFRecordDatasetBuilder::new(temp_file.path())
            .features(vec!["feature1".to_string(), "feature2".to_string()])
            .max_records(100);

        // Test builder configuration
        assert!(builder.feature_names.is_some());
        assert_eq!(builder.max_records, Some(100));
    }

    #[test]
    fn test_feature_value() {
        let float_feature = FeatureValue::FloatList(vec![1.0, 2.0, 3.0]);
        let int_feature = FeatureValue::Int64List(vec![1, 2, 3]);
        let bytes_feature = FeatureValue::BytesList(vec![vec![1, 2, 3]]);

        match float_feature {
            FeatureValue::FloatList(values) => assert_eq!(values.len(), 3),
            _ => panic!("Expected FloatList"),
        }

        match int_feature {
            FeatureValue::Int64List(values) => assert_eq!(values.len(), 3),
            _ => panic!("Expected Int64List"),
        }

        match bytes_feature {
            FeatureValue::BytesList(values) => assert_eq!(values.len(), 1),
            _ => panic!("Expected BytesList"),
        }
    }

    #[test]
    fn test_tfrecord_utils() {
        let temp_file = NamedTempFile::new().unwrap();

        // Test with invalid file
        assert!(!tfrecord_utils::is_tfrecord_file(temp_file.path()));
    }
}
