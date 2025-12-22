//! Core Tensor Serialization Implementation
//!
//! This module provides the main tensor serialization and deserialization
//! implementations that dispatch to format-specific modules based on the
//! requested serialization format.

use super::{
    binary,
    common::{SerializationFormat, SerializationOptions},
    data_science, ml_formats, scientific, text_formats,
};
use crate::{Tensor, TensorElement};
use std::path::Path;
use torsh_core::error::{Result, TorshError};

/// Main serialization implementation for Tensor (with serialize feature)
#[cfg(feature = "serialize")]
impl<T: TensorElement + serde::Serialize + for<'a> serde::Deserialize<'a> + hdf5::H5Type>
    Tensor<T>
{
    /// Serialize tensor to bytes using the specified format
    ///
    /// # Arguments
    /// * `format` - Serialization format to use
    /// * `options` - Serialization options
    ///
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized bytes or error
    pub fn serialize_to_bytes(
        &self,
        format: SerializationFormat,
        options: &SerializationOptions,
    ) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        match format {
            SerializationFormat::Binary => {
                binary::serialize_binary(self, &mut buffer, options)?;
            }
            SerializationFormat::Json => {
                text_formats::serialize_json(self, &mut buffer, options)?;
            }
            SerializationFormat::Numpy => {
                text_formats::numpy::serialize_numpy(self, &mut buffer)?;
            }
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => {
                return Err(TorshError::SerializationError(
                    "HDF5 format requires file path, use serialize_to_file instead".to_string(),
                ));
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow | SerializationFormat::Parquet => {
                return Err(TorshError::SerializationError(
                    "Arrow/Parquet format requires file path, use serialize_to_file instead"
                        .to_string(),
                ));
            }
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => {
                return Err(TorshError::SerializationError(
                    "ONNX format requires file path, use serialize_to_file instead".to_string(),
                ));
            }
        }

        Ok(buffer)
    }

    /// Serialize tensor to file using the specified format
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `format` - Serialization format to use
    /// * `options` - Serialization options
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn serialize_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        format: SerializationFormat,
        options: &SerializationOptions,
    ) -> Result<()> {
        let path = path.as_ref();

        match format {
            SerializationFormat::Binary
            | SerializationFormat::Json
            | SerializationFormat::Numpy => {
                // For formats that support byte serialization, write to file
                let bytes = self.serialize_to_bytes(format, options)?;
                std::fs::write(path, bytes).map_err(|e| {
                    TorshError::SerializationError(format!("Failed to write file: {}", e))
                })?;
            }
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => {
                let dataset_name = "tensor";
                scientific::hdf5::serialize_hdf5(self, path, dataset_name, options)?;
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow => {
                data_science::arrow::serialize_arrow(self, path, options)?;
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Parquet => {
                data_science::parquet::serialize_parquet(self, path, options)?;
            }
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => {
                ml_formats::onnx::serialize_onnx(self, path, options)?;
            }
        }

        Ok(())
    }

    /// Deserialize tensor from bytes using the specified format
    ///
    /// # Arguments
    /// * `data` - Serialized bytes
    /// * `format` - Serialization format used
    ///
    /// # Returns
    /// * `Result<Tensor<T>>` - Deserialized tensor or error
    pub fn deserialize_from_bytes(data: &[u8], format: SerializationFormat) -> Result<Tensor<T>> {
        let mut cursor = std::io::Cursor::new(data);

        match format {
            SerializationFormat::Binary => binary::deserialize_binary(&mut cursor),
            SerializationFormat::Json => text_formats::deserialize_json(&mut cursor),
            SerializationFormat::Numpy => text_formats::numpy::deserialize_numpy(&mut cursor),
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => Err(TorshError::SerializationError(
                "HDF5 format requires file path, use deserialize_from_file instead".to_string(),
            )),
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow | SerializationFormat::Parquet => {
                Err(TorshError::SerializationError(
                    "Arrow/Parquet format requires file path, use deserialize_from_file instead"
                        .to_string(),
                ))
            }
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => Err(TorshError::SerializationError(
                "ONNX format requires file path, use deserialize_from_file instead".to_string(),
            )),
        }
    }

    /// Deserialize tensor from file using the specified format
    ///
    /// # Arguments
    /// * `path` - Input file path
    /// * `format` - Serialization format used
    ///
    /// # Returns
    /// * `Result<Tensor<T>>` - Deserialized tensor or error
    pub fn deserialize_from_file<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Tensor<T>> {
        let path = path.as_ref();

        match format {
            SerializationFormat::Binary
            | SerializationFormat::Json
            | SerializationFormat::Numpy => {
                // For formats that support byte deserialization, read from file
                let bytes = std::fs::read(path).map_err(|e| {
                    TorshError::SerializationError(format!("Failed to read file: {}", e))
                })?;
                Self::deserialize_from_bytes(&bytes, format)
            }
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => {
                let dataset_name = "tensor";
                scientific::hdf5::deserialize_hdf5(path, dataset_name)
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow => data_science::arrow::deserialize_arrow(path),
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Parquet => data_science::parquet::deserialize_parquet(path),
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => ml_formats::onnx::deserialize_onnx(path),
        }
    }

    /// Auto-detect format from file extension and serialize
    ///
    /// # Arguments
    /// * `path` - Output file path (extension determines format)
    /// * `options` - Serialization options
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn save<P: AsRef<Path>>(&self, path: P, options: &SerializationOptions) -> Result<()> {
        let path = path.as_ref();
        let format = detect_format_from_path(path)?;
        self.serialize_to_file(path, format, options)
    }

    /// Auto-detect format from file extension and deserialize
    ///
    /// # Arguments
    /// * `path` - Input file path (extension determines format)
    ///
    /// # Returns
    /// * `Result<Tensor<T>>` - Deserialized tensor or error
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Tensor<T>> {
        let path = path.as_ref();
        let format = detect_format_from_path(path)?;
        Self::deserialize_from_file(path, format)
    }
}

/// Implementation for when serialize feature is not enabled
#[cfg(not(feature = "serialize"))]
impl<T: TensorElement> Tensor<T> {
    /// Serialize tensor to bytes using the specified format
    pub fn serialize_to_bytes(
        &self,
        format: SerializationFormat,
        options: &SerializationOptions,
    ) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        match format {
            SerializationFormat::Binary => {
                binary::serialize_binary(self, &mut buffer, options)?;
            }
            SerializationFormat::Json => {
                return Err(TorshError::SerializationError(
                    "JSON serialization requires the 'serialize' feature to be enabled".to_string(),
                ));
            }
            SerializationFormat::Numpy => {
                text_formats::numpy::serialize_numpy(self, &mut buffer)?;
            }
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => {
                return Err(TorshError::SerializationError(
                    "HDF5 format requires file path, use serialize_to_file instead".to_string(),
                ));
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow | SerializationFormat::Parquet => {
                return Err(TorshError::SerializationError(
                    "Arrow/Parquet format requires file path, use serialize_to_file instead"
                        .to_string(),
                ));
            }
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => {
                return Err(TorshError::SerializationError(
                    "ONNX format requires file path, use serialize_to_file instead".to_string(),
                ));
            }
        }

        Ok(buffer)
    }

    /// Serialize tensor to file using the specified format
    pub fn serialize_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        format: SerializationFormat,
        options: &SerializationOptions,
    ) -> Result<()> {
        let path = path.as_ref();

        match format {
            SerializationFormat::Binary | SerializationFormat::Numpy => {
                // These formats don't require the serialize feature
                let bytes = self.serialize_to_bytes(format, options)?;
                std::fs::write(path, bytes).map_err(|e| {
                    TorshError::SerializationError(format!("Failed to write file: {}", e))
                })?;
            }
            SerializationFormat::Json => {
                return Err(TorshError::SerializationError(
                    "JSON serialization requires the 'serialize' feature to be enabled".to_string(),
                ));
            }
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => {
                let dataset_name = "tensor";
                scientific::hdf5::serialize_hdf5(self, path, dataset_name, options)?;
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow => {
                data_science::arrow::serialize_arrow(self, path, options)?;
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Parquet => {
                data_science::parquet::serialize_parquet(self, path, options)?;
            }
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => {
                ml_formats::onnx::serialize_onnx(self, path, options)?;
            }
        }

        Ok(())
    }

    /// Deserialize tensor from bytes using the specified format
    pub fn deserialize_from_bytes(data: &[u8], format: SerializationFormat) -> Result<Tensor<T>> {
        let mut cursor = std::io::Cursor::new(data);

        match format {
            SerializationFormat::Binary => binary::deserialize_binary(&mut cursor),
            SerializationFormat::Json => Err(TorshError::SerializationError(
                "JSON deserialization requires the 'serialize' feature to be enabled".to_string(),
            )),
            SerializationFormat::Numpy => text_formats::numpy::deserialize_numpy(&mut cursor),
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => Err(TorshError::SerializationError(
                "HDF5 format requires file path, use deserialize_from_file instead".to_string(),
            )),
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow | SerializationFormat::Parquet => {
                Err(TorshError::SerializationError(
                    "Arrow/Parquet format requires file path, use deserialize_from_file instead"
                        .to_string(),
                ))
            }
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => Err(TorshError::SerializationError(
                "ONNX format requires file path, use deserialize_from_file instead".to_string(),
            )),
        }
    }

    /// Deserialize tensor from file using the specified format
    pub fn deserialize_from_file<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Tensor<T>> {
        let path = path.as_ref();

        match format {
            SerializationFormat::Binary | SerializationFormat::Numpy => {
                let bytes = std::fs::read(path).map_err(|e| {
                    TorshError::SerializationError(format!("Failed to read file: {}", e))
                })?;
                Self::deserialize_from_bytes(&bytes, format)
            }
            SerializationFormat::Json => Err(TorshError::SerializationError(
                "JSON deserialization requires the 'serialize' feature to be enabled".to_string(),
            )),
            #[cfg(feature = "serialize-hdf5")]
            SerializationFormat::Hdf5 => {
                let dataset_name = "tensor";
                scientific::hdf5::deserialize_hdf5(path, dataset_name)
            }
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Arrow => data_science::arrow::deserialize_arrow(path),
            #[cfg(feature = "serialize-arrow")]
            SerializationFormat::Parquet => data_science::parquet::deserialize_parquet(path),
            #[cfg(feature = "serialize-onnx")]
            SerializationFormat::Onnx => ml_formats::onnx::deserialize_onnx(path),
        }
    }

    /// Auto-detect format from file extension and serialize
    pub fn save<P: AsRef<Path>>(&self, path: P, options: &SerializationOptions) -> Result<()> {
        let path = path.as_ref();
        let format = detect_format_from_path(path)?;
        self.serialize_to_file(path, format, options)
    }

    /// Auto-detect format from file extension and deserialize
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Tensor<T>> {
        let path = path.as_ref();
        let format = detect_format_from_path(path)?;
        Self::deserialize_from_file(path, format)
    }
}

/// Detect serialization format from file path extension
///
/// # Arguments
/// * `path` - File path to analyze
///
/// # Returns
/// * `Result<SerializationFormat>` - Detected format or error
fn detect_format_from_path(path: &Path) -> Result<SerializationFormat> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| {
            TorshError::SerializationError(
                "Cannot detect format: file has no extension".to_string(),
            )
        })?;

    match extension.to_lowercase().as_str() {
        "trsh" | "bin" => Ok(SerializationFormat::Binary),
        "json" => Ok(SerializationFormat::Json),
        "npy" => Ok(SerializationFormat::Numpy),
        #[cfg(feature = "serialize-hdf5")]
        "h5" | "hdf5" => Ok(SerializationFormat::Hdf5),
        #[cfg(feature = "serialize-arrow")]
        "arrow" => Ok(SerializationFormat::Arrow),
        #[cfg(feature = "serialize-arrow")]
        "parquet" => Ok(SerializationFormat::Parquet),
        #[cfg(feature = "serialize-onnx")]
        "onnx" => Ok(SerializationFormat::Onnx),
        _ => Err(TorshError::SerializationError(format!(
            "Unsupported file extension: .{}",
            extension
        ))),
    }
}

/// Validate serialization format compatibility
///
/// # Arguments
/// * `format` - Format to validate
///
/// # Returns
/// * `Result<()>` - Ok if format is available, error otherwise
pub fn validate_format_support(format: SerializationFormat) -> Result<()> {
    match format {
        SerializationFormat::Binary | SerializationFormat::Numpy => {
            // Always supported
            Ok(())
        }
        SerializationFormat::Json => {
            #[cfg(feature = "serialize")]
            {
                Ok(())
            }
            #[cfg(not(feature = "serialize"))]
            {
                Err(TorshError::SerializationError(
                    "JSON format requires the 'serialize' feature".to_string(),
                ))
            }
        }
        #[cfg(feature = "serialize-hdf5")]
        SerializationFormat::Hdf5 => Ok(()),
        #[cfg(feature = "serialize-arrow")]
        SerializationFormat::Arrow | SerializationFormat::Parquet => Ok(()),
        #[cfg(feature = "serialize-onnx")]
        SerializationFormat::Onnx => Ok(()),
    }
}
