//! Common Types and Utilities for Tensor Serialization
//!
//! This module provides the foundational types, enumerations, and utilities
//! used across all serialization formats. It includes format specifications,
//! serialization options, and metadata structures.

use crate::{Tensor, TensorElement};
use std::collections::HashMap;
use torsh_core::{device::DeviceType, shape::Shape};

/// Serialization format types
///
/// Defines all supported serialization formats for tensors, including both
/// built-in formats and feature-gated external formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Custom binary format (fastest, most compact)
    ///
    /// A high-performance binary format optimized for speed and size.
    /// Includes CRC32 checksums for data integrity verification.
    Binary,

    /// JSON format (human-readable, portable)
    ///
    /// Text-based format that's easy to inspect and debug.
    /// Suitable for small tensors and configuration data.
    Json,

    /// NumPy .npy format (Python ecosystem compatibility)
    ///
    /// Compatible with NumPy's native binary format, enabling
    /// seamless interoperability with Python scientific stack.
    Numpy,

    /// HDF5 format (scientific computing standard)
    ///
    /// Hierarchical data format widely used in scientific computing.
    /// Supports compression, chunking, and metadata.
    #[cfg(feature = "serialize-hdf5")]
    Hdf5,

    /// Arrow format (data science standard)
    ///
    /// Columnar in-memory format optimized for analytical workloads.
    /// Provides zero-copy reads and efficient data processing.
    #[cfg(feature = "serialize-arrow")]
    Arrow,

    /// Parquet format (columnar storage)
    ///
    /// Compressed columnar storage format for big data analytics.
    /// Excellent compression ratios and query performance.
    #[cfg(feature = "serialize-arrow")]
    Parquet,

    /// ONNX format (machine learning models)
    ///
    /// Open Neural Network Exchange format for ML model interoperability.
    /// Enables model sharing between different ML frameworks.
    #[cfg(feature = "serialize-onnx")]
    Onnx,
}

impl SerializationFormat {
    /// Check if the format supports streaming I/O
    ///
    /// # Returns
    /// * `bool` - True if the format supports streaming operations
    pub fn supports_streaming(self) -> bool {
        match self {
            Self::Binary | Self::Numpy => true,
            Self::Json => false, // JSON needs complete structure
            #[cfg(feature = "serialize-hdf5")]
            Self::Hdf5 => true,
            #[cfg(feature = "serialize-arrow")]
            Self::Arrow | Self::Parquet => true,
            #[cfg(feature = "serialize-onnx")]
            Self::Onnx => false, // ONNX requires complete model
        }
    }

    /// Check if the format requires a file path (cannot serialize to memory)
    ///
    /// # Returns
    /// * `bool` - True if the format requires file-based operations
    pub fn requires_file_path(self) -> bool {
        match self {
            Self::Binary | Self::Json | Self::Numpy => false,
            #[cfg(feature = "serialize-hdf5")]
            Self::Hdf5 => true,
            #[cfg(feature = "serialize-arrow")]
            Self::Arrow | Self::Parquet => true,
            #[cfg(feature = "serialize-onnx")]
            Self::Onnx => true,
        }
    }

    /// Check if the format supports compression
    ///
    /// # Returns
    /// * `bool` - True if the format has built-in compression support
    pub fn supports_compression(self) -> bool {
        match self {
            Self::Binary => true, // Custom compression implementation
            Self::Json | Self::Numpy => false,
            #[cfg(feature = "serialize-hdf5")]
            Self::Hdf5 => true,
            #[cfg(feature = "serialize-arrow")]
            Self::Arrow | Self::Parquet => true,
            #[cfg(feature = "serialize-onnx")]
            Self::Onnx => false,
        }
    }

    /// Get typical file extension for the format
    ///
    /// # Returns
    /// * `&'static str` - File extension (without dot)
    pub fn file_extension(self) -> &'static str {
        match self {
            Self::Binary => "trsh",
            Self::Json => "json",
            Self::Numpy => "npy",
            #[cfg(feature = "serialize-hdf5")]
            Self::Hdf5 => "h5",
            #[cfg(feature = "serialize-arrow")]
            Self::Arrow => "arrow",
            #[cfg(feature = "serialize-arrow")]
            Self::Parquet => "parquet",
            #[cfg(feature = "serialize-onnx")]
            Self::Onnx => "onnx",
        }
    }

    /// Get MIME type for the format
    ///
    /// # Returns
    /// * `&'static str` - MIME type string
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Binary => "application/octet-stream",
            Self::Json => "application/json",
            Self::Numpy => "application/octet-stream",
            #[cfg(feature = "serialize-hdf5")]
            Self::Hdf5 => "application/x-hdf5",
            #[cfg(feature = "serialize-arrow")]
            Self::Arrow => "application/vnd.apache.arrow.file",
            #[cfg(feature = "serialize-arrow")]
            Self::Parquet => "application/vnd.apache.parquet",
            #[cfg(feature = "serialize-onnx")]
            Self::Onnx => "application/onnx",
        }
    }
}

/// Serialization options
///
/// Configuration options that control how tensors are serialized.
/// These options apply across different formats where supported.
#[derive(Debug, Clone)]
pub struct SerializationOptions {
    /// Include gradient information
    ///
    /// When true, gradient data is included in serialization.
    /// Only applies to formats that support gradient storage.
    pub include_gradients: bool,

    /// Include operation history
    ///
    /// When true, the computational graph history is preserved.
    /// Useful for debugging and analysis but increases file size.
    pub include_operations: bool,

    /// Compression level (0-9, 0 = no compression)
    ///
    /// Controls the trade-off between file size and serialization speed.
    /// - 0: No compression (fastest)
    /// - 1-3: Fast compression
    /// - 4-6: Balanced compression
    /// - 7-9: Maximum compression (slowest)
    pub compression_level: u8,

    /// Custom metadata
    ///
    /// User-defined key-value pairs to include with the tensor.
    /// Useful for storing application-specific information.
    pub metadata: HashMap<String, String>,

    /// Chunk size for streaming operations (bytes)
    ///
    /// Controls how data is divided during streaming I/O.
    /// Larger chunks improve throughput but use more memory.
    pub chunk_size: Option<usize>,

    /// Enable data validation during serialization
    ///
    /// When true, performs additional validation checks that can
    /// catch corruption but add overhead.
    pub validate_data: bool,

    /// Preserve precision for floating-point data
    ///
    /// When true, uses highest precision storage even if it increases size.
    /// When false, may use reduced precision for better compression.
    pub preserve_precision: bool,
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self {
            include_gradients: false,
            include_operations: false,
            compression_level: 0,
            metadata: HashMap::new(),
            chunk_size: None,
            validate_data: true,
            preserve_precision: true,
        }
    }
}

impl SerializationOptions {
    /// Create options optimized for speed
    ///
    /// # Returns
    /// * `Self` - Options configured for fastest serialization
    pub fn fast() -> Self {
        Self {
            include_gradients: false,
            include_operations: false,
            compression_level: 0,
            metadata: HashMap::new(),
            chunk_size: Some(64 * 1024 * 1024), // 64 MB chunks
            validate_data: false,
            preserve_precision: false,
        }
    }

    /// Create options optimized for size
    ///
    /// # Returns
    /// * `Self` - Options configured for smallest file size
    pub fn compact() -> Self {
        Self {
            include_gradients: false,
            include_operations: false,
            compression_level: 9,
            metadata: HashMap::new(),
            chunk_size: Some(1024 * 1024), // 1 MB chunks for better compression
            validate_data: true,
            preserve_precision: false,
        }
    }

    /// Create options optimized for debugging
    ///
    /// # Returns
    /// * `Self` - Options configured for maximum information preservation
    pub fn debug() -> Self {
        Self {
            include_gradients: true,
            include_operations: true,
            compression_level: 1, // Light compression for readability
            metadata: HashMap::new(),
            chunk_size: None,
            validate_data: true,
            preserve_precision: true,
        }
    }

    /// Add custom metadata entry
    ///
    /// # Arguments
    /// * `key` - Metadata key
    /// * `value` - Metadata value
    ///
    /// # Returns
    /// * `&mut Self` - Self for method chaining
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set compression level
    ///
    /// # Arguments
    /// * `level` - Compression level (0-9)
    ///
    /// # Returns
    /// * `Self` - Self for method chaining
    pub fn with_compression(mut self, level: u8) -> Self {
        self.compression_level = level.min(9);
        self
    }

    /// Enable or disable gradient inclusion
    ///
    /// # Arguments
    /// * `include` - Whether to include gradients
    ///
    /// # Returns
    /// * `Self` - Self for method chaining
    pub fn with_gradients(mut self, include: bool) -> Self {
        self.include_gradients = include;
        self
    }
}

/// Tensor metadata for serialization
///
/// Contains comprehensive metadata about a serialized tensor including
/// structural information, computational properties, and versioning data.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorMetadata {
    /// Tensor shape (dimensions)
    pub shape: Shape,

    /// Device where tensor is stored
    pub device: DeviceType,

    /// Whether tensor requires gradient computation
    pub requires_grad: bool,

    /// Data type name for deserialization
    pub dtype_name: String,

    /// ToRSh version used for serialization
    pub version: String,

    /// Unix timestamp when tensor was serialized
    pub timestamp: u64,

    /// Custom user-defined metadata
    pub custom_metadata: HashMap<String, String>,

    /// Serialization format used
    pub format: String,

    /// Size of data in bytes
    pub data_size: usize,

    /// Whether data is compressed
    pub compressed: bool,

    /// Checksum for data integrity (format-specific)
    pub checksum: Option<String>,
}

impl TensorMetadata {
    /// Create metadata from a tensor and serialization options
    ///
    /// # Arguments
    /// * `tensor` - Source tensor
    /// * `options` - Serialization options
    /// * `format` - Serialization format used
    /// * `data_size` - Size of serialized data in bytes
    ///
    /// # Returns
    /// * `Self` - Populated metadata structure
    pub fn from_tensor<T: TensorElement>(
        tensor: &Tensor<T>,
        options: &SerializationOptions,
        format: SerializationFormat,
        data_size: usize,
    ) -> Self {
        Self {
            shape: tensor.shape(),
            device: tensor.device(),
            requires_grad: tensor.requires_grad(),
            dtype_name: std::any::type_name::<T>().to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            custom_metadata: options.metadata.clone(),
            format: format!("{:?}", format),
            data_size,
            compressed: options.compression_level > 0,
            checksum: None, // Set by specific format implementations
        }
    }

    /// Validate metadata consistency
    ///
    /// # Returns
    /// * `Result<(), String>` - Ok if metadata is valid, error description otherwise
    pub fn validate(&self) -> Result<(), String> {
        if self.shape.numel() == 0 {
            return Err("Invalid shape: tensor cannot have zero size".to_string());
        }

        if self.dtype_name.is_empty() {
            return Err("Invalid dtype: type name cannot be empty".to_string());
        }

        if self.version.is_empty() {
            return Err("Invalid version: version string cannot be empty".to_string());
        }

        if self.data_size == 0 {
            return Err("Invalid data size: cannot be zero".to_string());
        }

        Ok(())
    }

    /// Get estimated memory requirements for deserialization
    ///
    /// # Returns
    /// * `usize` - Estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        // Base data size plus overhead for metadata and structure
        let base_size = self.data_size;
        let overhead = base_size / 10; // Estimate 10% overhead
        base_size + overhead
    }

    /// Check if tensor requires gradients
    ///
    /// # Returns
    /// * `bool` - True if gradients are required
    pub fn has_gradients(&self) -> bool {
        self.requires_grad
    }

    /// Get human-readable size description
    ///
    /// # Returns
    /// * `String` - Human-readable size (e.g., "1.5 MB")
    pub fn size_description(&self) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = self.data_size as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        if size.fract() == 0.0 {
            format!("{:.0} {}", size, UNITS[unit_idx])
        } else {
            format!("{:.1} {}", size, UNITS[unit_idx])
        }
    }
}
