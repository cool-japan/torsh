//! Text-Based and NumPy Format Implementations
//!
//! This module provides serialization support for human-readable JSON format
//! and NumPy's .npy binary format for Python ecosystem compatibility.

use super::common::{SerializationFormat, SerializationOptions, TensorMetadata};
use crate::{Tensor, TensorElement};
use std::io::{Read, Write};
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
};

/// Helper struct for JSON serialization
///
/// Combines tensor metadata and data into a single structure
/// that can be easily serialized to JSON.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct SerializableTensor<T> {
    metadata: TensorMetadata,
    data: Vec<T>,
}

/// Serialize tensor to JSON format
///
/// Creates a human-readable JSON representation of the tensor including
/// both metadata and data. Suitable for small tensors and debugging.
///
/// # Arguments
/// * `tensor` - Tensor to serialize
/// * `writer` - Output stream to write to
/// * `options` - Serialization options
///
/// # Returns
/// * `Result<()>` - Ok if successful, error otherwise
#[cfg(feature = "serialize")]
pub fn serialize_json<T: TensorElement + serde::Serialize, W: Write>(
    tensor: &Tensor<T>,
    writer: &mut W,
    options: &SerializationOptions,
) -> Result<()> {
    let data = tensor.data()?.clone();
    let metadata = TensorMetadata::from_tensor(
        tensor,
        options,
        SerializationFormat::Json,
        data.len() * std::mem::size_of::<T>(),
    );

    let serializable = SerializableTensor { metadata, data };

    // Serialize to JSON with optional pretty printing
    let json_data = if options.compression_level == 0 {
        // Pretty print for readability
        serde_json::to_vec_pretty(&serializable)
    } else {
        // Compact format for smaller size
        serde_json::to_vec(&serializable)
    }
    .map_err(|e| TorshError::SerializationError(format!("JSON serialization failed: {}", e)))?;

    writer
        .write_all(&json_data)
        .map_err(|e| TorshError::SerializationError(format!("Failed to write JSON data: {}", e)))?;

    Ok(())
}

/// Serialize tensor to JSON format (when serialize feature is disabled)
#[cfg(not(feature = "serialize"))]
pub fn serialize_json<T: TensorElement, W: Write>(
    _tensor: &Tensor<T>,
    _writer: &mut W,
    _options: &SerializationOptions,
) -> Result<()> {
    Err(TorshError::SerializationError(
        "JSON serialization requires the 'serialize' feature to be enabled".to_string(),
    ))
}

/// Deserialize tensor from JSON format
///
/// Reads a JSON representation of a tensor and reconstructs the tensor
/// with its metadata and data.
///
/// # Arguments
/// * `reader` - Input stream to read from
///
/// # Returns
/// * `Result<Tensor<T>>` - Deserialized tensor or error
#[cfg(feature = "serialize")]
pub fn deserialize_json<T: TensorElement + for<'a> serde::Deserialize<'a>, R: Read>(
    reader: &mut R,
) -> Result<Tensor<T>> {
    let mut json_data = Vec::new();
    reader
        .read_to_end(&mut json_data)
        .map_err(|e| TorshError::SerializationError(format!("Failed to read JSON data: {}", e)))?;

    let serializable: SerializableTensor<T> = serde_json::from_slice(&json_data).map_err(|e| {
        TorshError::SerializationError(format!("JSON deserialization failed: {}", e))
    })?;

    // Validate metadata
    serializable
        .metadata
        .validate()
        .map_err(|e| TorshError::SerializationError(format!("Invalid metadata in JSON: {}", e)))?;

    // Create tensor from data
    Tensor::from_data(
        serializable.data,
        serializable.metadata.shape.dims().to_vec(),
        serializable.metadata.device,
    )
}

/// Deserialize tensor from JSON format (when serialize feature is disabled)
#[cfg(not(feature = "serialize"))]
pub fn deserialize_json<T: TensorElement, R: Read>(_reader: &mut R) -> Result<Tensor<T>> {
    Err(TorshError::SerializationError(
        "JSON deserialization requires the 'serialize' feature to be enabled".to_string(),
    ))
}

/// NumPy .npy format implementation
pub mod numpy {
    use super::*;

    /// NumPy array header information
    ///
    /// Contains the metadata necessary to interpret a NumPy .npy file,
    /// including data type, memory layout, and array dimensions.
    #[derive(Debug, Clone)]
    struct NumpyHeader {
        dtype: String,
        fortran_order: bool,
        shape: Vec<usize>,
    }

    impl NumpyHeader {
        /// Create a NumPy header for the given tensor type and shape
        ///
        /// # Arguments
        /// * `shape` - Array dimensions
        ///
        /// # Returns
        /// * `Self` - NumPy header with appropriate dtype
        fn new<T: TensorElement>(shape: &[usize]) -> Self {
            let dtype = match std::any::TypeId::of::<T>() {
                id if id == std::any::TypeId::of::<f32>() => "<f4".to_string(),
                id if id == std::any::TypeId::of::<f64>() => "<f8".to_string(),
                id if id == std::any::TypeId::of::<i8>() => "<i1".to_string(),
                id if id == std::any::TypeId::of::<i16>() => "<i2".to_string(),
                id if id == std::any::TypeId::of::<i32>() => "<i4".to_string(),
                id if id == std::any::TypeId::of::<i64>() => "<i8".to_string(),
                id if id == std::any::TypeId::of::<u8>() => "<u1".to_string(),
                id if id == std::any::TypeId::of::<u16>() => "<u2".to_string(),
                id if id == std::any::TypeId::of::<u32>() => "<u4".to_string(),
                id if id == std::any::TypeId::of::<u64>() => "<u8".to_string(),
                _ => "<f4".to_string(), // Default to float32
            };

            Self {
                dtype,
                fortran_order: false, // Always use C order
                shape: shape.to_vec(),
            }
        }

        /// Convert header to NumPy header string format
        ///
        /// # Returns
        /// * `String` - NumPy header string
        fn to_string(&self) -> String {
            format!(
                "{{'descr': '{}', 'fortran_order': {}, 'shape': {:?}}}",
                self.dtype, self.fortran_order, self.shape
            )
        }

        /// Parse NumPy header from string
        ///
        /// # Arguments
        /// * `s` - Header string to parse
        ///
        /// # Returns
        /// * `Result<Self>` - Parsed header or error
        fn from_string(s: &str) -> Result<Self> {
            // Remove braces and parse key-value pairs
            let s = s.trim().trim_start_matches('{').trim_end_matches('}');

            let mut dtype = String::new();
            let mut fortran_order = false;
            let mut shape = Vec::new();

            // Simple parsing - in production, consider using a proper Python literal parser
            let parts: Vec<&str> = s.split(',').collect();

            for part in parts {
                let part = part.trim();

                if part.starts_with("'descr'") || part.starts_with("\"descr\"") {
                    // Extract dtype from 'descr': 'dtype'
                    if let Some(start) = part.find(':') {
                        let value_part = &part[start + 1..].trim();
                        if let Some(quote_start) = value_part.find(['\'', '"']) {
                            let quote_char = value_part.chars().nth(quote_start).unwrap();
                            if let Some(quote_end) = value_part[quote_start + 1..].find(quote_char)
                            {
                                dtype = value_part[quote_start + 1..quote_start + 1 + quote_end]
                                    .to_string();
                            }
                        }
                    }
                } else if part.starts_with("'fortran_order'")
                    || part.starts_with("\"fortran_order\"")
                {
                    fortran_order = part.contains("True");
                } else if part.starts_with("'shape'") || part.starts_with("\"shape\"") {
                    // Parse shape tuple
                    if let Some(tuple_start) = part.find('(') {
                        if let Some(tuple_end) = part.rfind(')') {
                            let tuple_content = &part[tuple_start + 1..tuple_end];
                            if !tuple_content.trim().is_empty() {
                                for dim in tuple_content.split(',') {
                                    let dim = dim.trim();
                                    if !dim.is_empty() {
                                        shape.push(dim.parse().map_err(|_| {
                                            TorshError::SerializationError(format!(
                                                "Invalid shape dimension: '{}'",
                                                dim
                                            ))
                                        })?);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if dtype.is_empty() {
                return Err(TorshError::SerializationError(
                    "Missing or invalid dtype in NumPy header".to_string(),
                ));
            }

            Ok(Self {
                dtype,
                fortran_order,
                shape,
            })
        }

        /// Get the size of each element in bytes
        ///
        /// # Returns
        /// * `Result<usize>` - Element size in bytes
        fn element_size(&self) -> Result<usize> {
            let size_str = &self.dtype[2..]; // Remove '<f', '<i', etc.
            size_str.parse().map_err(|_| {
                TorshError::SerializationError(format!("Invalid dtype format: {}", self.dtype))
            })
        }
    }

    /// Serialize tensor to NumPy .npy format
    ///
    /// Creates a .npy file compatible with NumPy's native binary format,
    /// enabling seamless interoperability with Python scientific stack.
    ///
    /// # Arguments
    /// * `tensor` - Tensor to serialize
    /// * `writer` - Output stream to write to
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn serialize_numpy<T: TensorElement, W: Write>(
        tensor: &Tensor<T>,
        writer: &mut W,
    ) -> Result<()> {
        // NumPy magic string (version 1.0)
        writer.write_all(b"\x93NUMPY").map_err(|e| {
            TorshError::SerializationError(format!("Failed to write NumPy magic: {}", e))
        })?;

        // Version (1.0)
        writer.write_all(&[0x01, 0x00]).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write NumPy version: {}", e))
        })?;

        // Create header
        let header = NumpyHeader::new::<T>(tensor.shape().dims());
        let header_str = header.to_string();

        // Calculate padding to align to 64-byte boundary
        let base_len = 6 + 2 + 2 + header_str.len() + 1; // magic + version + len + header + '\n'
        let padding = if base_len % 64 == 0 {
            0
        } else {
            64 - (base_len % 64)
        };
        let padded_header = format!("{}{}\n", header_str, " ".repeat(padding));

        // Write header length (little-endian u16)
        let header_len = padded_header.len() as u16;
        writer.write_all(&header_len.to_le_bytes()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write header length: {}", e))
        })?;

        // Write header
        writer.write_all(padded_header.as_bytes()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write header: {}", e))
        })?;

        // Write tensor data in little-endian format
        let data = tensor.data()?;
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        writer.write_all(data_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write tensor data: {}", e))
        })?;

        Ok(())
    }

    /// Deserialize tensor from NumPy .npy format
    ///
    /// Reads a NumPy .npy file and reconstructs the tensor with proper
    /// data type and shape validation.
    ///
    /// # Arguments
    /// * `reader` - Input stream to read from
    ///
    /// # Returns
    /// * `Result<Tensor<T>>` - Deserialized tensor or error
    pub fn deserialize_numpy<T: TensorElement, R: Read>(reader: &mut R) -> Result<Tensor<T>> {
        // Read and validate magic string
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read NumPy magic: {}", e))
        })?;

        if &magic != b"\x93NUMPY" {
            return Err(TorshError::SerializationError(format!(
                "Invalid NumPy magic string: expected b\"\\x93NUMPY\", got {:?}",
                magic
            )));
        }

        // Read and validate version
        let mut version = [0u8; 2];
        reader.read_exact(&mut version).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read NumPy version: {}", e))
        })?;

        if version[0] != 1 || version[1] != 0 {
            return Err(TorshError::SerializationError(format!(
                "Unsupported NumPy version: {}.{} (only 1.0 is supported)",
                version[0], version[1]
            )));
        }

        // Read header length
        let mut header_len_bytes = [0u8; 2];
        reader.read_exact(&mut header_len_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read header length: {}", e))
        })?;
        let header_len = u16::from_le_bytes(header_len_bytes) as usize;

        // Read header
        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| TorshError::SerializationError(format!("Failed to read header: {}", e)))?;

        let header_str = String::from_utf8(header_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Invalid UTF-8 in NumPy header: {}", e))
        })?;

        // Parse header
        let header = NumpyHeader::from_string(&header_str)?;

        // Validate dtype compatibility
        let expected_header = NumpyHeader::new::<T>(&header.shape);
        if header.dtype != expected_header.dtype {
            return Err(TorshError::SerializationError(format!(
                "NumPy dtype mismatch: file contains '{}', expected '{}' for type {}",
                header.dtype,
                expected_header.dtype,
                std::any::type_name::<T>()
            )));
        }

        if header.fortran_order {
            return Err(TorshError::SerializationError(
                "Fortran order arrays are not currently supported".to_string(),
            ));
        }

        // Calculate expected data size
        let numel = header.shape.iter().product::<usize>();
        let expected_data_size = numel * std::mem::size_of::<T>();

        if numel == 0 {
            return Err(TorshError::SerializationError(
                "Cannot deserialize array with zero elements".to_string(),
            ));
        }

        // Read tensor data
        let mut data_bytes = vec![0u8; expected_data_size];
        reader.read_exact(&mut data_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read tensor data: {}", e))
        })?;

        // Convert bytes to typed data safely
        let mut typed_data = Vec::with_capacity(numel);
        let byte_ptr = data_bytes.as_ptr();

        for i in 0..numel {
            unsafe {
                let element_ptr = byte_ptr.add(i * std::mem::size_of::<T>()) as *const T;
                typed_data.push(std::ptr::read(element_ptr));
            }
        }

        // Create tensor
        Tensor::from_data(typed_data, header.shape, DeviceType::Cpu)
    }

    /// Validate NumPy file format without full deserialization
    ///
    /// Performs quick validation of a NumPy file by checking the header
    /// without reading the full tensor data.
    ///
    /// # Arguments
    /// * `reader` - Input stream to validate
    ///
    /// # Returns
    /// * `Result<(Vec<usize>, String)>` - Shape and dtype if valid, error otherwise
    pub fn validate_numpy_format<R: Read>(reader: &mut R) -> Result<(Vec<usize>, String)> {
        // Read magic and version (reuse logic from deserialize_numpy)
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read NumPy magic: {}", e))
        })?;

        if &magic != b"\x93NUMPY" {
            return Err(TorshError::SerializationError(
                "Invalid NumPy magic string".to_string(),
            ));
        }

        let mut version = [0u8; 2];
        reader.read_exact(&mut version).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read NumPy version: {}", e))
        })?;

        if version[0] != 1 || version[1] != 0 {
            return Err(TorshError::SerializationError(format!(
                "Unsupported NumPy version: {}.{}",
                version[0], version[1]
            )));
        }

        // Read and parse header
        let mut header_len_bytes = [0u8; 2];
        reader.read_exact(&mut header_len_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read header length: {}", e))
        })?;
        let header_len = u16::from_le_bytes(header_len_bytes) as usize;

        let mut header_bytes = vec![0u8; header_len];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| TorshError::SerializationError(format!("Failed to read header: {}", e)))?;

        let header_str = String::from_utf8(header_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Invalid UTF-8 in header: {}", e))
        })?;

        let header = NumpyHeader::from_string(&header_str)?;

        Ok((header.shape, header.dtype))
    }
}
