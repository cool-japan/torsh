//! Streaming I/O with Progress Reporting for Large Tensors
//!
//! This module provides streaming serialization capabilities for large tensors
//! with progress reporting, memory management, and efficient I/O operations.

use super::common::SerializationOptions;
use crate::{Tensor, TensorElement};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};
use torsh_core::error::{Result, TorshError};

/// Progress callback function type
///
/// Called periodically during streaming operations to report progress.
///
/// # Parameters
/// * `bytes_processed` - Number of bytes processed so far
/// * `total_bytes` - Total number of bytes to process
/// * `elapsed_time` - Time elapsed since operation started
pub type ProgressCallback = Arc<dyn Fn(u64, u64, Duration) + Send + Sync>;

/// Configuration for streaming operations
///
/// Controls various aspects of streaming I/O including chunk sizes,
/// buffering, compression, and memory limits.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Progress reporting interval in bytes
    pub progress_interval: u64,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
    /// Enable compression for chunks
    pub compress_chunks: bool,
    /// Maximum memory usage (bytes) before spilling to disk
    pub memory_limit: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024 * 1024,   // 64 MB chunks
            progress_interval: 1024 * 1024, // Report every 1 MB
            buffer_size: 8 * 1024,          // 8 KB buffer
            compress_chunks: false,
            memory_limit: 1024 * 1024 * 1024, // 1 GB memory limit
        }
    }
}

impl StreamingConfig {
    /// Create configuration optimized for speed
    ///
    /// # Returns
    /// * `Self` - Configuration for fastest streaming
    pub fn fast() -> Self {
        Self {
            chunk_size: 128 * 1024 * 1024,       // Larger chunks
            progress_interval: 16 * 1024 * 1024, // Less frequent reporting
            buffer_size: 64 * 1024,              // Larger buffer
            compress_chunks: false,
            memory_limit: 2 * 1024 * 1024 * 1024, // More memory
        }
    }

    /// Create configuration optimized for memory usage
    ///
    /// # Returns
    /// * `Self` - Configuration for minimal memory usage
    pub fn low_memory() -> Self {
        Self {
            chunk_size: 1024 * 1024,         // Smaller chunks
            progress_interval: 512 * 1024,   // More frequent reporting
            buffer_size: 4 * 1024,           // Smaller buffer
            compress_chunks: true,           // Enable compression
            memory_limit: 128 * 1024 * 1024, // Limited memory
        }
    }
}

/// Streaming tensor writer with progress reporting
///
/// Provides efficient streaming serialization of large tensors with
/// progress callbacks, compression, and memory management.
pub struct StreamingTensorWriter<W: Write + Seek> {
    writer: BufWriter<W>,
    config: StreamingConfig,
    bytes_written: u64,
    total_bytes: u64,
    last_progress_report: u64,
    start_time: Instant,
    progress_callback: Option<ProgressCallback>,
}

impl<W: Write + Seek> StreamingTensorWriter<W> {
    /// Create a new streaming writer
    ///
    /// # Arguments
    /// * `writer` - Output writer
    /// * `config` - Streaming configuration
    ///
    /// # Returns
    /// * `Self` - New streaming writer instance
    pub fn new(writer: W, config: StreamingConfig) -> Self {
        let buf_writer = BufWriter::with_capacity(config.buffer_size, writer);

        Self {
            writer: buf_writer,
            config,
            bytes_written: 0,
            total_bytes: 0,
            last_progress_report: 0,
            start_time: Instant::now(),
            progress_callback: None,
        }
    }

    /// Set progress callback
    ///
    /// # Arguments
    /// * `callback` - Progress callback function
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Start writing a tensor with known total size
    ///
    /// # Arguments
    /// * `total_bytes` - Total number of bytes that will be written
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn begin_tensor(&mut self, total_bytes: u64) -> Result<()> {
        self.total_bytes = total_bytes;
        self.bytes_written = 0;
        self.last_progress_report = 0;
        self.start_time = Instant::now();

        // Report initial progress
        self.report_progress();

        Ok(())
    }

    /// Write a chunk of tensor data
    ///
    /// # Arguments
    /// * `data` - Data chunk to write
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn write_chunk(&mut self, data: &[u8]) -> Result<()> {
        // Apply compression if enabled
        let chunk_data = if self.config.compress_chunks {
            // TODO: Implement compression (e.g., using flate2)
            data.to_vec()
        } else {
            data.to_vec()
        };

        // Write chunk
        self.writer
            .write_all(&chunk_data)
            .map_err(|e| TorshError::SerializationError(format!("Failed to write chunk: {}", e)))?;

        // Update progress
        self.bytes_written += data.len() as u64;

        // Report progress if needed
        if self.bytes_written - self.last_progress_report >= self.config.progress_interval {
            self.report_progress();
            self.last_progress_report = self.bytes_written;
        }

        Ok(())
    }

    /// Finish writing and flush all data
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn finish(mut self) -> Result<()> {
        self.writer.flush().map_err(|e| {
            TorshError::SerializationError(format!("Failed to flush writer: {}", e))
        })?;

        // Final progress report
        self.report_progress();

        Ok(())
    }

    /// Report current progress
    fn report_progress(&self) {
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self.start_time.elapsed();
            callback(self.bytes_written, self.total_bytes, elapsed);
        }
    }
}

/// Streaming tensor reader with progress reporting
///
/// Provides efficient streaming deserialization of large tensors with
/// progress callbacks and memory management.
pub struct StreamingTensorReader<R: Read> {
    reader: BufReader<R>,
    config: StreamingConfig,
    bytes_read: u64,
    total_bytes: u64,
    last_progress_report: u64,
    start_time: Instant,
    progress_callback: Option<ProgressCallback>,
}

impl<R: Read> StreamingTensorReader<R> {
    /// Create a new streaming reader
    ///
    /// # Arguments
    /// * `reader` - Input reader
    /// * `config` - Streaming configuration
    ///
    /// # Returns
    /// * `Self` - New streaming reader instance
    pub fn new(reader: R, config: StreamingConfig) -> Self {
        let buf_reader = BufReader::with_capacity(config.buffer_size, reader);

        Self {
            reader: buf_reader,
            config,
            bytes_read: 0,
            total_bytes: 0,
            last_progress_report: 0,
            start_time: Instant::now(),
            progress_callback: None,
        }
    }

    /// Set progress callback
    ///
    /// # Arguments
    /// * `callback` - Progress callback function
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Start reading a tensor with known total size
    ///
    /// # Arguments
    /// * `total_bytes` - Total number of bytes that will be read
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn begin_tensor(&mut self, total_bytes: u64) -> Result<()> {
        self.total_bytes = total_bytes;
        self.bytes_read = 0;
        self.last_progress_report = 0;
        self.start_time = Instant::now();

        // Report initial progress
        self.report_progress();

        Ok(())
    }

    /// Read a chunk of tensor data
    ///
    /// # Arguments
    /// * `buffer` - Buffer to read data into
    ///
    /// # Returns
    /// * `Result<usize>` - Number of bytes read or error
    pub fn read_chunk(&mut self, buffer: &mut [u8]) -> Result<usize> {
        let bytes_read = self
            .reader
            .read(buffer)
            .map_err(|e| TorshError::SerializationError(format!("Failed to read chunk: {}", e)))?;

        // Handle decompression if needed
        // TODO: Implement decompression if chunks were compressed

        // Update progress
        self.bytes_read += bytes_read as u64;

        // Report progress if needed
        if self.bytes_read - self.last_progress_report >= self.config.progress_interval {
            self.report_progress();
            self.last_progress_report = self.bytes_read;
        }

        Ok(bytes_read)
    }

    /// Read exact number of bytes
    ///
    /// # Arguments
    /// * `buffer` - Buffer to read data into
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn read_exact(&mut self, buffer: &mut [u8]) -> Result<()> {
        self.reader.read_exact(buffer).map_err(|e| {
            TorshError::SerializationError(format!("Failed to read exact bytes: {}", e))
        })?;

        // Update progress
        self.bytes_read += buffer.len() as u64;

        // Report progress if needed
        if self.bytes_read - self.last_progress_report >= self.config.progress_interval {
            self.report_progress();
            self.last_progress_report = self.bytes_read;
        }

        Ok(())
    }

    /// Finish reading and report final progress
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn finish(self) -> Result<()> {
        // Final progress report
        self.report_progress();
        Ok(())
    }

    /// Report current progress
    fn report_progress(&self) {
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self.start_time.elapsed();
            callback(self.bytes_read, self.total_bytes, elapsed);
        }
    }
}

/// Utility functions for streaming operations
pub mod utils {
    use super::*;

    /// Stream serialize a tensor to file with progress reporting
    ///
    /// # Arguments
    /// * `tensor` - Tensor to serialize
    /// * `path` - Output file path
    /// * `options` - Serialization options
    /// * `config` - Streaming configuration
    /// * `progress_callback` - Optional progress callback
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn stream_serialize_to_file<T: TensorElement>(
        tensor: &Tensor<T>,
        path: &std::path::Path,
        _options: &SerializationOptions,
        config: StreamingConfig,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| TorshError::SerializationError(format!("Failed to create file: {}", e)))?;

        let mut writer = StreamingTensorWriter::new(file, config);

        if let Some(callback) = progress_callback {
            writer = writer.with_progress_callback(callback);
        }

        // Calculate total size
        let data = tensor.data()?;
        let total_bytes = data.len() * std::mem::size_of::<T>();

        writer.begin_tensor(total_bytes as u64)?;

        // Write data in chunks
        let data_bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, total_bytes) };

        let chunk_size = writer.config.chunk_size;
        for chunk in data_bytes.chunks(chunk_size) {
            writer.write_chunk(chunk)?;
        }

        writer.finish()?;

        Ok(())
    }

    /// Create a simple progress callback that prints to console
    ///
    /// # Returns
    /// * `ProgressCallback` - Progress callback function
    pub fn console_progress_callback() -> ProgressCallback {
        Arc::new(|bytes_processed, total_bytes, elapsed| {
            let percentage = if total_bytes > 0 {
                (bytes_processed as f64 / total_bytes as f64) * 100.0
            } else {
                0.0
            };

            let rate = if elapsed.as_secs_f64() > 0.0 {
                bytes_processed as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0
            } else {
                0.0
            };

            println!(
                "Progress: {:.1}% ({}/{} bytes) - {:.1} MB/s - {:?}",
                percentage, bytes_processed, total_bytes, rate, elapsed
            );
        })
    }
}
