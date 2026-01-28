//! Custom Binary Format Implementation for Tensor Serialization
//!
//! This module provides a high-performance binary format optimized for speed and size.
//! It includes CRC32 checksums for data integrity verification and supports optional
//! compression for space efficiency.

use super::common::{SerializationFormat, SerializationOptions, TensorMetadata};
use crate::{Tensor, TensorElement};
use std::io::{Read, Write};
use torsh_core::error::{Result, TorshError};

/// Magic bytes for ToRSh tensor binary format
const TORSH_MAGIC: &[u8] = b"TRSH";

/// Current format version
const FORMAT_VERSION: u32 = 1;

/// CRC32 implementation for data integrity verification
///
/// Provides fast CRC32 checksum calculation using a lookup table
/// for efficient data validation during serialization and deserialization.
struct Crc32 {
    table: [u32; 256],
}

impl Crc32 {
    /// Create a new CRC32 instance with precomputed lookup table
    ///
    /// # Returns
    /// * `Self` - New CRC32 instance ready for checksum calculation
    fn new() -> Self {
        let mut table = [0u32; 256];
        for i in 0..256 {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 == 1 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        Self { table }
    }

    /// Calculate CRC32 checksum for given data
    ///
    /// # Arguments
    /// * `data` - Data to calculate checksum for
    ///
    /// # Returns
    /// * `u32` - CRC32 checksum value
    fn checksum(&self, data: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFF_u32;
        for &byte in data {
            let index = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ self.table[index];
        }
        !crc
    }
}

/// Calculate CRC32 checksum for data
///
/// Convenience function that creates a CRC32 instance and calculates
/// the checksum in one call.
///
/// # Arguments
/// * `data` - Data to calculate checksum for
///
/// # Returns
/// * `u32` - CRC32 checksum value
fn calculate_crc32(data: &[u8]) -> u32 {
    let crc = Crc32::new();
    crc.checksum(data)
}

/// Binary format header structure
///
/// Contains format identification, version information, and data layout
/// details necessary for proper deserialization.
#[derive(Debug)]
struct BinaryHeader {
    /// Magic bytes for format identification
    magic: [u8; 4],
    /// Format version for compatibility checking
    version: u32,
    /// Size of metadata section in bytes
    metadata_size: u64,
    /// Size of tensor data section in bytes
    data_size: u64,
    /// CRC32 checksum of metadata + data
    checksum: u32,
}

impl BinaryHeader {
    /// Create a new header with calculated sizes
    ///
    /// # Arguments
    /// * `metadata_size` - Size of metadata section
    /// * `data_size` - Size of data section
    ///
    /// # Returns
    /// * `Self` - New header instance
    #[allow(dead_code)]
    fn new(metadata_size: u64, data_size: u64) -> Self {
        Self {
            magic: *b"TRSH",
            version: FORMAT_VERSION,
            metadata_size,
            data_size,
            checksum: 0, // Will be calculated later with actual data
        }
    }

    /// Create a header with all fields including checksum
    ///
    /// # Arguments
    /// * `metadata_size` - Size of metadata section
    /// * `data_size` - Size of data section
    /// * `checksum` - CRC32 checksum of data
    ///
    /// # Returns
    /// * `Self` - Complete header instance
    fn with_checksum(metadata_size: u64, data_size: u64, checksum: u32) -> Self {
        Self {
            magic: *b"TRSH",
            version: FORMAT_VERSION,
            metadata_size,
            data_size,
            checksum,
        }
    }

    /// Write header to output stream
    ///
    /// # Arguments
    /// * `writer` - Output stream to write to
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.magic).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write magic bytes: {}", e))
        })?;
        writer.write_all(&self.version.to_le_bytes()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write version: {}", e))
        })?;
        writer
            .write_all(&self.metadata_size.to_le_bytes())
            .map_err(|e| {
                TorshError::SerializationError(format!("Failed to write metadata size: {}", e))
            })?;
        writer
            .write_all(&self.data_size.to_le_bytes())
            .map_err(|e| {
                TorshError::SerializationError(format!("Failed to write data size: {}", e))
            })?;
        writer
            .write_all(&self.checksum.to_le_bytes())
            .map_err(|e| {
                TorshError::SerializationError(format!("Failed to write checksum: {}", e))
            })?;
        Ok(())
    }

    /// Read header from input stream
    ///
    /// # Arguments
    /// * `reader` - Input stream to read from
    ///
    /// # Returns
    /// * `Result<Self>` - Header instance or error
    fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read magic bytes: {}", e))
        })?;

        if &magic != TORSH_MAGIC {
            return Err(TorshError::SerializationError(format!(
                "Invalid magic bytes in binary format: expected {:?}, got {:?}",
                TORSH_MAGIC, magic
            )));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read version: {}", e))
        })?;
        let version = u32::from_le_bytes(version_bytes);

        if version != FORMAT_VERSION {
            return Err(TorshError::SerializationError(format!(
                "Unsupported format version: expected {}, got {}",
                FORMAT_VERSION, version
            )));
        }

        let mut metadata_size_bytes = [0u8; 8];
        reader.read_exact(&mut metadata_size_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read metadata size: {}", e))
        })?;
        let metadata_size = u64::from_le_bytes(metadata_size_bytes);

        let mut data_size_bytes = [0u8; 8];
        reader.read_exact(&mut data_size_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read data size: {}", e))
        })?;
        let data_size = u64::from_le_bytes(data_size_bytes);

        let mut checksum_bytes = [0u8; 4];
        reader.read_exact(&mut checksum_bytes).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read checksum: {}", e))
        })?;
        let checksum = u32::from_le_bytes(checksum_bytes);

        Ok(Self {
            magic,
            version,
            metadata_size,
            data_size,
            checksum,
        })
    }

    /// Get total header size in bytes
    ///
    /// # Returns
    /// * `usize` - Header size in bytes
    const fn size() -> usize {
        4 + 4 + 8 + 8 + 4 // magic + version + metadata_size + data_size + checksum
    }
}

/// Serialize tensor to binary format
///
/// Serializes a tensor using the custom ToRSh binary format with CRC32
/// integrity checking and optional compression.
///
/// # Arguments
/// * `tensor` - Tensor to serialize
/// * `writer` - Output stream to write to
/// * `options` - Serialization options
///
/// # Returns
/// * `Result<()>` - Ok if successful, error otherwise
pub fn serialize_binary<T: TensorElement, W: Write>(
    tensor: &Tensor<T>,
    writer: &mut W,
    options: &SerializationOptions,
) -> Result<()> {
    // Create metadata
    let data_size = tensor.numel() * std::mem::size_of::<T>();
    let mut metadata =
        TensorMetadata::from_tensor(tensor, options, SerializationFormat::Binary, data_size);

    // Serialize metadata
    #[cfg(feature = "serialize")]
    let _metadata_bytes = oxicode::serde::encode_to_vec(&metadata, oxicode::config::standard())
        .map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize metadata: {}", e))
        })?;

    #[cfg(not(feature = "serialize"))]
    let metadata_bytes = {
        return Err(TorshError::SerializationError(
            "Serialization feature not enabled".to_string(),
        ));
    };

    // Get tensor data
    let data = tensor.data()?;
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };

    // Apply compression if requested
    let (final_data_bytes, compressed) = if options.compression_level > 0 {
        // TODO: Implement compression using flate2 or zstd
        // For now, store uncompressed
        (data_bytes.to_vec(), false)
    } else {
        (data_bytes.to_vec(), false)
    };

    // Update metadata with actual compression status
    metadata.compressed = compressed;
    metadata.data_size = final_data_bytes.len();

    // Re-serialize metadata with updated compression info
    #[cfg(feature = "serialize")]
    let final_metadata_bytes =
        oxicode::serde::encode_to_vec(&metadata, oxicode::config::standard()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize updated metadata: {}", e))
        })?;

    #[cfg(not(feature = "serialize"))]
    let final_metadata_bytes = metadata_bytes;

    // Calculate CRC32 checksum of metadata + data
    let mut combined_data = Vec::new();
    combined_data.extend_from_slice(&final_metadata_bytes);
    combined_data.extend_from_slice(&final_data_bytes);
    let checksum = calculate_crc32(&combined_data);

    // Update metadata with checksum
    metadata.checksum = Some(format!("0x{:08X}", checksum));

    // Write header with checksum
    let header = BinaryHeader::with_checksum(
        final_metadata_bytes.len() as u64,
        final_data_bytes.len() as u64,
        checksum,
    );
    header.write_to(writer)?;

    // Write metadata
    writer
        .write_all(&final_metadata_bytes)
        .map_err(|e| TorshError::SerializationError(format!("Failed to write metadata: {}", e)))?;

    // Write tensor data
    writer.write_all(&final_data_bytes).map_err(|e| {
        TorshError::SerializationError(format!("Failed to write tensor data: {}", e))
    })?;

    Ok(())
}

/// Deserialize tensor from binary format
///
/// Deserializes a tensor from the custom ToRSh binary format with CRC32
/// integrity verification and automatic decompression.
///
/// # Arguments
/// * `reader` - Input stream to read from
///
/// # Returns
/// * `Result<Tensor<T>>` - Deserialized tensor or error
pub fn deserialize_binary<T: TensorElement, R: Read>(reader: &mut R) -> Result<Tensor<T>> {
    // Read header
    let header = BinaryHeader::read_from(reader)?;

    // Validate header
    if header.metadata_size == 0 {
        return Err(TorshError::SerializationError(
            "Invalid header: metadata size cannot be zero".to_string(),
        ));
    }

    if header.data_size == 0 {
        return Err(TorshError::SerializationError(
            "Invalid header: data size cannot be zero".to_string(),
        ));
    }

    // Read metadata
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader
        .read_exact(&mut metadata_bytes)
        .map_err(|e| TorshError::SerializationError(format!("Failed to read metadata: {}", e)))?;

    #[cfg(feature = "serialize")]
    let (metadata, _): (TensorMetadata, usize) =
        oxicode::serde::decode_from_slice(&metadata_bytes, oxicode::config::standard()).map_err(
            |e| TorshError::SerializationError(format!("Failed to deserialize metadata: {}", e)),
        )?;

    #[cfg(not(feature = "serialize"))]
    let metadata = {
        return Err(TorshError::SerializationError(
            "Serialization feature not enabled".to_string(),
        ));
    };

    // Validate metadata
    metadata
        .validate()
        .map_err(|e| TorshError::SerializationError(format!("Invalid metadata: {}", e)))?;

    // Read tensor data
    let mut data_bytes = vec![0u8; header.data_size as usize];
    reader.read_exact(&mut data_bytes).map_err(|e| {
        TorshError::SerializationError(format!("Failed to read tensor data: {}", e))
    })?;

    // Verify CRC32 checksum
    let mut combined_data = Vec::new();
    combined_data.extend_from_slice(&metadata_bytes);
    combined_data.extend_from_slice(&data_bytes);
    let calculated_checksum = calculate_crc32(&combined_data);

    if calculated_checksum != header.checksum {
        return Err(TorshError::SerializationError(format!(
            "Data corruption detected: checksum mismatch (expected 0x{:08X}, got 0x{:08X})",
            header.checksum, calculated_checksum
        )));
    }

    // Decompress data if needed
    let final_data_bytes = if metadata.compressed {
        // TODO: Implement decompression
        data_bytes
    } else {
        data_bytes
    };

    // Convert bytes back to tensor data
    let expected_len = metadata.shape.numel();
    let actual_len = final_data_bytes.len() / std::mem::size_of::<T>();

    if expected_len != actual_len {
        return Err(TorshError::SerializationError(format!(
            "Data size mismatch: expected {} elements, got {} (shape: {:?}, element size: {} bytes)",
            expected_len, actual_len, metadata.shape.dims(), std::mem::size_of::<T>()
        )));
    }

    // Safe conversion from bytes to typed data
    let mut typed_data = Vec::with_capacity(actual_len);
    let byte_ptr = final_data_bytes.as_ptr();

    for i in 0..actual_len {
        unsafe {
            let element_ptr = byte_ptr.add(i * std::mem::size_of::<T>()) as *const T;
            typed_data.push(std::ptr::read(element_ptr));
        }
    }

    // Create tensor from data
    Tensor::from_data(typed_data, metadata.shape.dims().to_vec(), metadata.device)
}

/// Get estimated file size for binary serialization
///
/// Provides an estimate of the file size that would result from
/// serializing the given tensor in binary format.
///
/// # Arguments
/// * `tensor` - Tensor to estimate size for
/// * `options` - Serialization options
///
/// # Returns
/// * `usize` - Estimated file size in bytes
pub fn estimate_binary_size<T: TensorElement>(
    tensor: &Tensor<T>,
    options: &SerializationOptions,
) -> usize {
    let header_size = BinaryHeader::size();
    let metadata_size = 200; // Rough estimate for metadata
    let data_size = tensor.numel() * std::mem::size_of::<T>();

    let compressed_data_size = if options.compression_level > 0 {
        // Rough compression estimate (actual ratio depends on data)
        let compression_ratio = match options.compression_level {
            1..=3 => 0.8,
            4..=6 => 0.6,
            7..=9 => 0.4,
            _ => 1.0,
        };
        (data_size as f64 * compression_ratio) as usize
    } else {
        data_size
    };

    header_size + metadata_size + compressed_data_size
}

/// Validate binary format stream without full deserialization
///
/// Performs quick validation of a binary format stream by checking
/// the header and metadata without reading the full tensor data.
///
/// # Arguments
/// * `reader` - Input stream to validate
///
/// # Returns
/// * `Result<TensorMetadata>` - Metadata if valid, error otherwise
pub fn validate_binary_format<R: Read>(reader: &mut R) -> Result<TensorMetadata> {
    // Read and validate header
    let header = BinaryHeader::read_from(reader)?;

    // Read and validate metadata
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes).map_err(|e| {
        TorshError::SerializationError(format!("Failed to read metadata for validation: {}", e))
    })?;

    #[cfg(feature = "serialize")]
    let (metadata, _): (TensorMetadata, usize) =
        oxicode::serde::decode_from_slice(&metadata_bytes, oxicode::config::standard()).map_err(
            |e| {
                TorshError::SerializationError(format!(
                    "Failed to deserialize metadata for validation: {}",
                    e
                ))
            },
        )?;

    #[cfg(not(feature = "serialize"))]
    return Err(TorshError::SerializationError(
        "Serialization feature not enabled".to_string(),
    ));

    // Validate metadata
    metadata.validate().map_err(|e| {
        TorshError::SerializationError(format!("Invalid metadata during validation: {}", e))
    })?;

    Ok(metadata)
}
