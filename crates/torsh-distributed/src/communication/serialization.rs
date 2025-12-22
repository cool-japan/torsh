//! Unified serialization utilities for communication
//!
//! This module consolidates tensor and message serialization patterns
//! used across RPC, parameter server, and collective operations.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use torsh_core::device::DeviceType;
use torsh_core::dtype::TensorElement;
use torsh_tensor::Tensor;

/// Trait for messages that can be serialized for communication
pub trait CommunicationMessage: Serialize + for<'de> Deserialize<'de> + Send + Sync {}

// Implement for common message types
impl<T> CommunicationMessage for T where T: Serialize + for<'de> Deserialize<'de> + Send + Sync {}

/// Serialize a message for communication
pub fn serialize_message<T: CommunicationMessage>(msg: &T) -> TorshResult<Vec<u8>> {
    bincode::serde::encode_to_vec(msg, bincode::config::standard()).map_err(|e| {
        TorshDistributedError::SerializationError(format!("Message serialization failed: {}", e))
    })
}

/// Deserialize a message from communication
pub fn deserialize_message<T: CommunicationMessage>(data: &[u8]) -> TorshResult<T> {
    let (value, _): (T, usize) =
        bincode::serde::decode_from_slice(data, bincode::config::standard()).map_err(|e| {
            TorshDistributedError::SerializationError(format!(
                "Message deserialization failed: {}",
                e
            ))
        })?;
    Ok(value)
}

/// Serializable tensor representation for communication
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableTensor {
    /// Raw tensor data as bytes
    pub data: Vec<u8>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type identifier
    pub dtype: String,
    /// Device type
    pub device: DeviceType,
    /// Element size in bytes
    pub element_size: usize,
}

/// Serialize a tensor for communication
pub fn serialize_tensor<T>(tensor: &Tensor<T>) -> TorshResult<Vec<u8>>
where
    T: Clone + Send + Sync + 'static + TensorElement + Copy,
{
    // Get tensor data
    let data = tensor.to_vec().map_err(|e| {
        TorshDistributedError::SerializationError(format!("Failed to extract tensor data: {}", e))
    })?;

    // Convert to bytes
    let element_size = std::mem::size_of::<T>();
    let byte_data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * element_size).to_vec()
    };

    let serializable = SerializableTensor {
        data: byte_data,
        shape: tensor.shape().dims().to_vec(),
        dtype: std::any::type_name::<T>().to_string(),
        device: tensor.device(),
        element_size,
    };

    serialize_message(&serializable)
}

/// Deserialize a tensor from communication data
pub fn deserialize_tensor<T>(data: &[u8], expected_shape: &[usize]) -> TorshResult<Tensor<T>>
where
    T: Clone + Send + Sync + 'static + torsh_core::dtype::TensorElement,
{
    let serializable: SerializableTensor = deserialize_message(data)?;

    // Validate shape matches expectation
    if serializable.shape != expected_shape {
        return Err(TorshDistributedError::TensorShapeMismatch {
            expected: expected_shape.to_vec(),
            actual: serializable.shape,
        });
    }

    // Validate element size
    let expected_element_size = std::mem::size_of::<T>();
    if serializable.element_size != expected_element_size {
        return Err(TorshDistributedError::SerializationError(format!(
            "Element size mismatch: expected {}, got {}",
            expected_element_size, serializable.element_size
        )));
    }

    // Convert bytes back to typed data
    let element_count = serializable.data.len() / serializable.element_size;
    let typed_data = unsafe {
        std::slice::from_raw_parts(serializable.data.as_ptr() as *const T, element_count).to_vec()
    };

    // Create tensor
    Tensor::from_data(typed_data, serializable.shape, serializable.device).map_err(|e| {
        TorshDistributedError::SerializationError(format!(
            "Failed to create tensor from data: {}",
            e
        ))
    })
}

/// Estimate serialized size of a tensor without actually serializing
pub fn estimate_tensor_serialized_size<T>(tensor: &Tensor<T>) -> usize
where
    T: 'static + TensorElement + Copy,
{
    let element_size = std::mem::size_of::<T>();
    let data_size = tensor.numel() * element_size;
    let metadata_overhead = 256; // Rough estimate for shape, dtype, etc.
    data_size + metadata_overhead
}

/// Compress tensor data for communication (optional optimization)
#[cfg(feature = "compression")]
pub fn compress_tensor_data(data: Vec<u8>) -> TorshResult<Vec<u8>> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&data).map_err(|e| {
        TorshDistributedError::SerializationError(format!("Compression failed: {}", e))
    })?;

    encoder.finish().map_err(|e| {
        TorshDistributedError::SerializationError(format!("Compression finalization failed: {}", e))
            .into()
    })
}

/// Decompress tensor data from communication
#[cfg(feature = "compression")]
pub fn decompress_tensor_data(compressed_data: &[u8]) -> TorshResult<Vec<u8>> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(compressed_data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).map_err(|e| {
        TorshDistributedError::SerializationError(format!("Decompression failed: {}", e))
    })?;

    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use torsh_tensor::creation::zeros;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestMessage {
        id: u32,
        content: String,
    }

    #[test]
    fn test_message_serialization() {
        let msg = TestMessage {
            id: 42,
            content: "test message".to_string(),
        };

        let serialized = serialize_message(&msg).unwrap();
        let deserialized: TestMessage = deserialize_message(&serialized).unwrap();

        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_tensor_serialization() {
        let tensor = zeros::<f32>(&[2, 3]).unwrap();
        let binding = tensor.shape();
        let shape = binding.dims();

        let serialized = serialize_tensor(&tensor).unwrap();
        let deserialized: Tensor<f32> = deserialize_tensor(&serialized, shape).unwrap();

        assert_eq!(tensor.shape().dims(), deserialized.shape().dims());
        assert_eq!(tensor.device(), deserialized.device());
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let tensor = zeros::<f32>(&[2, 3]).unwrap();
        let serialized = serialize_tensor(&tensor).unwrap();

        // Try to deserialize with wrong shape
        let result: Result<Tensor<f32>, _> = deserialize_tensor(&serialized, &[3, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_tensor_size() {
        let tensor = zeros::<f32>(&[10, 10]).unwrap();
        let estimated_size = estimate_tensor_serialized_size(&tensor);

        // Should be at least the data size plus some overhead
        let expected_min_size = 100 * std::mem::size_of::<f32>();
        assert!(estimated_size >= expected_min_size);
    }
}
