//! Comprehensive Tensor Serialization Framework
//!
//! This module provides a unified interface for tensor serialization across multiple
//! formats including binary, text, scientific, and machine learning formats.
//! The implementation is organized into specialized modules for maintainability
//! and feature modularity.
//!
//! # Supported Formats
//!
//! ## Core Formats
//! - **Binary**: Custom high-performance binary format with CRC32 validation
//! - **JSON**: Human-readable JSON format with metadata
//! - **NumPy**: NumPy `.npy` format for Python ecosystem compatibility
//!
//! ## Scientific Computing (Feature: `serialize-hdf5`)
//! - **HDF5**: Hierarchical data format with compression and chunking
//!
//! ## Data Science (Feature: `serialize-arrow`)
//! - **Arrow**: Columnar format optimized for analytics
//! - **Parquet**: Compressed columnar storage for big data
//!
//! ## Machine Learning (Feature: `serialize-onnx`)
//! - **ONNX**: Open Neural Network Exchange format
//!
//! # Usage Examples
//!
//! ```rust
//! use torsh_tensor::serialize::{SerializationOptions, SerializationFormat};
//! use torsh_tensor::Tensor;
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! let tensor = Tensor::ones([2, 3], torsh_core::device::DeviceType::Cpu)?;
//!
//! // Save with auto-format detection
//! let options = SerializationOptions::default();
//! tensor.save("tensor.bin", &options)?;
//!
//! // Load with explicit format
//! let loaded = Tensor::<f32>::load("tensor.bin", Some(SerializationFormat::Binary))?;
//! # Ok(())
//! # }
//! ```

// Core serialization types and functionality
pub mod common;
pub mod core;

// Format-specific implementations
pub mod binary;
pub mod text_formats;

// Scientific computing formats
pub mod scientific;

// Data science formats
pub mod data_science;

// Machine learning formats
pub mod ml_formats;

// Advanced I/O capabilities
pub mod streaming;

// Re-export core types and functionality
pub use common::{SerializationFormat, SerializationOptions, TensorMetadata};

// Core serialization functionality is available as methods on Tensor<T>

// Re-export format-specific functions
pub use binary::{deserialize_binary, serialize_binary};

pub use text_formats::{
    deserialize_json,
    numpy::{deserialize_numpy, serialize_numpy, validate_numpy_format},
    serialize_json,
};

// Re-export scientific format functions
pub use scientific::hdf5::{deserialize_hdf5, get_dataset_metadata, list_datasets, serialize_hdf5};

// Re-export data science format functions
pub use data_science::{
    arrow::{deserialize_arrow, serialize_arrow},
    parquet::{deserialize_parquet, serialize_parquet},
};

// Re-export ML format functions
pub use ml_formats::onnx::{deserialize_onnx, serialize_onnx};

// Re-export streaming functionality
pub use streaming::{
    utils::{console_progress_callback, stream_serialize_to_file},
    ProgressCallback, StreamingConfig, StreamingTensorReader, StreamingTensorWriter,
};

// Re-export tensor implementation from core module
// Serialization methods are available directly on Tensor<T> when the serialize feature is enabled

/// Convenient type alias for serialization results
pub type SerializeResult<T> = torsh_core::error::Result<T>;

// Feature-gated module accessibility
#[cfg(feature = "serialize-hdf5")]
pub use scientific::hdf5;

#[cfg(feature = "serialize-arrow")]
pub use data_science::{arrow, parquet};

#[cfg(feature = "serialize-onnx")]
pub use ml_formats::onnx;

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::{SerializationFormat, SerializationOptions, TensorMetadata};

    // Serialization functions are available as methods on Tensor<T>

    // Streaming functionality available in streaming module
}
