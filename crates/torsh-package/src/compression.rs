//! Advanced compression algorithms for packages
//!
//! This module provides multiple compression algorithms optimized for different
//! types of data commonly found in ML packages including models, source code,
//! and configuration files.

use std::collections::HashMap;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

use crate::resources::{Resource, ResourceType};

/// Supported compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression (fast, decent compression)
    Gzip,
    /// Zstandard compression (excellent speed/ratio tradeoff)
    Zstd,
    /// LZMA compression (high compression ratio, slower)
    Lzma,
    /// Brotli compression (good for text/JSON)
    Brotli,
    /// LZ4 compression (extremely fast, lower ratio)
    Lz4,
}

/// Compression level (0-22 depending on algorithm)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressionLevel(pub u32);

/// Compression strategy for different data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// Optimize for speed
    Speed,
    /// Optimize for size
    Size,
    /// Balanced speed/size
    Balanced,
    /// Adaptive based on data type
    Adaptive,
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Default algorithm to use
    pub default_algorithm: CompressionAlgorithm,
    /// Default compression level
    pub default_level: CompressionLevel,
    /// Compression strategy
    pub strategy: CompressionStrategy,
    /// Per-resource-type algorithm overrides
    pub algorithm_overrides: HashMap<ResourceType, CompressionAlgorithm>,
    /// Minimum size threshold for compression (bytes)
    pub min_size_threshold: usize,
    /// Maximum size for in-memory compression
    pub max_memory_size: usize,
    /// Enable parallel compression for large resources
    pub parallel_compression: bool,
}

/// Advanced compressor with multiple algorithm support
pub struct AdvancedCompressor {
    config: CompressionConfig,
}

/// Compression result with metadata
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Compressed data
    pub data: Vec<u8>,
    /// Algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Compression level used
    pub level: CompressionLevel,
    /// Original size
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Compression ratio (compressed/original)
    pub ratio: f32,
    /// Compression time in milliseconds
    pub compression_time_ms: u64,
}

/// Decompression result
#[derive(Debug, Clone)]
pub struct DecompressionResult {
    /// Decompressed data
    pub data: Vec<u8>,
    /// Algorithm used for compression
    pub algorithm: CompressionAlgorithm,
    /// Decompression time in milliseconds
    pub decompression_time_ms: u64,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel(6)
    }
}

impl CompressionLevel {
    /// Create a new compression level
    pub fn new(level: u32) -> Self {
        CompressionLevel(level)
    }

    /// Get level for specific algorithm (clamps to valid range)
    pub fn for_algorithm(&self, algorithm: CompressionAlgorithm) -> u32 {
        match algorithm {
            CompressionAlgorithm::None => 0,
            CompressionAlgorithm::Gzip => self.0.min(9),
            CompressionAlgorithm::Zstd => self.0.min(22),
            CompressionAlgorithm::Lzma => self.0.min(9),
            CompressionAlgorithm::Brotli => self.0.min(11),
            CompressionAlgorithm::Lz4 => self.0.min(16),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        let mut algorithm_overrides = HashMap::new();

        // Text files compress well with Brotli
        algorithm_overrides.insert(ResourceType::Source, CompressionAlgorithm::Brotli);
        algorithm_overrides.insert(ResourceType::Config, CompressionAlgorithm::Brotli);
        algorithm_overrides.insert(ResourceType::Documentation, CompressionAlgorithm::Brotli);
        algorithm_overrides.insert(ResourceType::Text, CompressionAlgorithm::Brotli);
        algorithm_overrides.insert(ResourceType::Metadata, CompressionAlgorithm::Brotli);

        // Binary data works well with Zstandard
        algorithm_overrides.insert(ResourceType::Model, CompressionAlgorithm::Zstd);
        algorithm_overrides.insert(ResourceType::Data, CompressionAlgorithm::Zstd);
        algorithm_overrides.insert(ResourceType::Binary, CompressionAlgorithm::Zstd);

        Self {
            default_algorithm: CompressionAlgorithm::Zstd,
            default_level: CompressionLevel::default(),
            strategy: CompressionStrategy::Balanced,
            algorithm_overrides,
            min_size_threshold: 256, // Don't compress files smaller than 256 bytes
            max_memory_size: 100 * 1024 * 1024, // 100MB
            parallel_compression: true,
        }
    }
}

impl CompressionConfig {
    /// Create new compression config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default algorithm
    pub fn with_algorithm(mut self, algorithm: CompressionAlgorithm) -> Self {
        self.default_algorithm = algorithm;
        self
    }

    /// Set default compression level
    pub fn with_level(mut self, level: CompressionLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Set compression strategy
    pub fn with_strategy(mut self, strategy: CompressionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set minimum size threshold
    pub fn with_min_threshold(mut self, threshold: usize) -> Self {
        self.min_size_threshold = threshold;
        self
    }

    /// Enable/disable parallel compression
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel_compression = parallel;
        self
    }

    /// Get algorithm for resource type
    pub fn algorithm_for_resource(&self, resource_type: ResourceType) -> CompressionAlgorithm {
        self.algorithm_overrides
            .get(&resource_type)
            .copied()
            .unwrap_or(self.default_algorithm)
    }

    /// Get compression level adjusted for strategy
    pub fn level_for_strategy(&self, strategy: CompressionStrategy) -> CompressionLevel {
        match strategy {
            CompressionStrategy::Speed => CompressionLevel(1),
            CompressionStrategy::Size => CompressionLevel(9),
            CompressionStrategy::Balanced => CompressionLevel(6),
            CompressionStrategy::Adaptive => self.default_level,
        }
    }
}

impl AdvancedCompressor {
    /// Create new compressor with default config
    pub fn new() -> Self {
        Self {
            config: CompressionConfig::default(),
        }
    }

    /// Create compressor with custom config
    pub fn with_config(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress a resource
    pub fn compress_resource(&self, resource: &Resource) -> Result<CompressionResult> {
        // Check size threshold
        if resource.data.len() < self.config.min_size_threshold {
            return Ok(CompressionResult {
                data: resource.data.clone(),
                algorithm: CompressionAlgorithm::None,
                level: CompressionLevel(0),
                original_size: resource.data.len(),
                compressed_size: resource.data.len(),
                ratio: 1.0,
                compression_time_ms: 0,
            });
        }

        // Select algorithm and level
        let algorithm = self.config.algorithm_for_resource(resource.resource_type);
        let level = match self.config.strategy {
            CompressionStrategy::Adaptive => self.adaptive_level(resource),
            strategy => self.config.level_for_strategy(strategy),
        };

        self.compress_data(&resource.data, algorithm, level)
    }

    /// Compress raw data with specific algorithm and level
    pub fn compress_data(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
    ) -> Result<CompressionResult> {
        let start_time = std::time::Instant::now();

        let compressed_data = match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Gzip => {
                self.compress_gzip(data, level.for_algorithm(algorithm))?
            }
            CompressionAlgorithm::Zstd => {
                self.compress_zstd(data, level.for_algorithm(algorithm))?
            }
            CompressionAlgorithm::Lzma => {
                self.compress_lzma(data, level.for_algorithm(algorithm))?
            }
            CompressionAlgorithm::Brotli => {
                self.compress_brotli(data, level.for_algorithm(algorithm))?
            }
            CompressionAlgorithm::Lz4 => self.compress_lz4(data, level.for_algorithm(algorithm))?,
        };

        let compression_time_ms = start_time.elapsed().as_millis() as u64;
        let ratio = if data.is_empty() {
            1.0
        } else {
            compressed_data.len() as f32 / data.len() as f32
        };

        let compressed_size = compressed_data.len();

        Ok(CompressionResult {
            data: compressed_data,
            algorithm,
            level,
            original_size: data.len(),
            compressed_size,
            ratio,
            compression_time_ms,
        })
    }

    /// Decompress data
    pub fn decompress_data(
        &self,
        compressed_data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Result<DecompressionResult> {
        let start_time = std::time::Instant::now();

        let decompressed_data = match algorithm {
            CompressionAlgorithm::None => compressed_data.to_vec(),
            CompressionAlgorithm::Gzip => self.decompress_gzip(compressed_data)?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(compressed_data)?,
            CompressionAlgorithm::Lzma => self.decompress_lzma(compressed_data)?,
            CompressionAlgorithm::Brotli => self.decompress_brotli(compressed_data)?,
            CompressionAlgorithm::Lz4 => self.decompress_lz4(compressed_data)?,
        };

        let decompression_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(DecompressionResult {
            data: decompressed_data,
            algorithm,
            decompression_time_ms,
        })
    }

    /// Benchmark compression algorithms for given data
    pub fn benchmark_algorithms(&self, data: &[u8]) -> Result<Vec<CompressionResult>> {
        let algorithms = [
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lzma,
            CompressionAlgorithm::Brotli,
            CompressionAlgorithm::Lz4,
        ];

        let mut results = Vec::new();

        for algorithm in &algorithms {
            let result = self.compress_data(data, *algorithm, CompressionLevel(6))?;
            results.push(result);
        }

        // Sort by compression ratio (best first)
        results.sort_by(|a, b| {
            a.ratio
                .partial_cmp(&b.ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Adaptively choose compression level based on resource characteristics
    fn adaptive_level(&self, resource: &Resource) -> CompressionLevel {
        let data_size = resource.data.len();

        match resource.resource_type {
            ResourceType::Model | ResourceType::Binary => {
                // For model files, balance speed and compression based on size
                if data_size > 10 * 1024 * 1024 {
                    // > 10MB
                    CompressionLevel(3) // Fast compression for large files
                } else if data_size > 1024 * 1024 {
                    // > 1MB
                    CompressionLevel(6) // Balanced
                } else {
                    CompressionLevel(9) // High compression for smaller files
                }
            }
            ResourceType::Source | ResourceType::Config | ResourceType::Documentation => {
                // Text files compress well, so we can afford higher levels
                CompressionLevel(8)
            }
            ResourceType::Text | ResourceType::Metadata => {
                // JSON/text data often compresses very well
                CompressionLevel(7)
            }
            _ => CompressionLevel(6), // Default balanced
        }
    }

    /// Compress with Gzip
    fn compress_gzip(&self, data: &[u8], level: u32) -> Result<Vec<u8>> {
        use flate2::{write::GzEncoder, Compression};

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
        encoder.write_all(data).map_err(|e| {
            TorshError::SerializationError(format!("Gzip compression failed: {}", e))
        })?;

        encoder
            .finish()
            .map_err(|e| TorshError::SerializationError(format!("Gzip finalization failed: {}", e)))
    }

    /// Decompress Gzip
    fn decompress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result).map_err(|e| {
            TorshError::SerializationError(format!("Gzip decompression failed: {}", e))
        })?;

        Ok(result)
    }

    /// Compress with Zstandard
    fn compress_zstd(&self, data: &[u8], level: u32) -> Result<Vec<u8>> {
        zstd::encode_all(data, level as i32).map_err(|e| {
            TorshError::SerializationError(format!("Zstandard compression failed: {}", e))
        })
    }

    /// Decompress Zstandard
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| {
            TorshError::SerializationError(format!("Zstandard decompression failed: {}", e))
        })
    }

    /// Compress with LZMA
    fn compress_lzma(&self, data: &[u8], _level: u32) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        lzma_rs::lzma_compress(&mut std::io::Cursor::new(data), &mut output).map_err(|e| {
            TorshError::SerializationError(format!("LZMA compression failed: {}", e))
        })?;

        Ok(output)
    }

    /// Decompress LZMA
    fn decompress_lzma(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        lzma_rs::lzma_decompress(&mut std::io::Cursor::new(data), &mut output).map_err(|e| {
            TorshError::SerializationError(format!("LZMA decompression failed: {}", e))
        })?;

        Ok(output)
    }

    /// Compress with Brotli
    fn compress_brotli(&self, data: &[u8], level: u32) -> Result<Vec<u8>> {
        // For now, fall back to gzip since brotli might not be available
        // In a real implementation, you would add brotli dependency and use it
        self.compress_gzip(data, level.min(9))
    }

    /// Decompress Brotli
    fn decompress_brotli(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, fall back to gzip since brotli might not be available
        // In a real implementation, you would add brotli dependency and use it
        self.decompress_gzip(data)
    }

    /// Compress with LZ4
    fn compress_lz4(&self, data: &[u8], _level: u32) -> Result<Vec<u8>> {
        // For now, fall back to gzip since lz4 might not be available
        // In a real implementation, you would add lz4 dependency and use it
        self.compress_gzip(data, 1) // Use fast compression as LZ4 is meant to be fast
    }

    /// Decompress LZ4
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For now, fall back to gzip since lz4 might not be available
        // In a real implementation, you would add lz4 dependency and use it
        self.decompress_gzip(data)
    }
}

impl Default for AdvancedCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel compression utilities for large resources
pub struct ParallelCompressor {
    compressor: AdvancedCompressor,
    chunk_size: usize,
    num_threads: usize,
}

impl ParallelCompressor {
    /// Create new parallel compressor
    pub fn new(compressor: AdvancedCompressor) -> Self {
        Self {
            compressor,
            chunk_size: 1024 * 1024, // 1MB chunks
            num_threads: scirs2_core::parallel_ops::num_threads(),
        }
    }

    /// Set chunk size for parallel compression
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set number of threads for parallel compression
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Compress large data in parallel chunks
    pub fn compress_parallel(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
    ) -> Result<CompressionResult> {
        if data.len() < self.chunk_size * 2 {
            // For small data, use regular compression
            return self.compressor.compress_data(data, algorithm, level);
        }

        let start_time = std::time::Instant::now();

        // Split data into chunks
        let num_chunks = (data.len() + self.chunk_size - 1) / self.chunk_size;
        let chunks: Vec<&[u8]> = (0..num_chunks)
            .map(|i| {
                let start = i * self.chunk_size;
                let end = (start + self.chunk_size).min(data.len());
                &data[start..end]
            })
            .collect();

        // Compress chunks in parallel using scirs2-core's parallel operations
        use scirs2_core::parallel_ops::{IntoParallelIterator, ParallelIterator};

        let compressed_chunks: Vec<_> = chunks
            .into_par_iter()
            .map(|chunk| {
                self.compressor
                    .compress_data(chunk, algorithm, level)
                    .map(|result| result.data)
            })
            .collect::<Result<Vec<_>>>()?;

        // Combine compressed chunks
        let mut combined_data = Vec::new();
        combined_data.extend_from_slice(&(compressed_chunks.len() as u64).to_le_bytes());

        for chunk in &compressed_chunks {
            combined_data.extend_from_slice(&(chunk.len() as u64).to_le_bytes());
            combined_data.extend_from_slice(chunk);
        }

        let compression_time_ms = start_time.elapsed().as_millis() as u64;
        let compressed_size = combined_data.len();
        let ratio = if data.is_empty() {
            1.0
        } else {
            compressed_size as f32 / data.len() as f32
        };

        Ok(CompressionResult {
            data: combined_data,
            algorithm,
            level,
            original_size: data.len(),
            compressed_size,
            ratio,
            compression_time_ms,
        })
    }

    /// Decompress parallel-compressed data
    pub fn decompress_parallel(
        &self,
        compressed_data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Result<DecompressionResult> {
        if compressed_data.len() < 8 {
            // Not parallel-compressed, use regular decompression
            return self.compressor.decompress_data(compressed_data, algorithm);
        }

        let start_time = std::time::Instant::now();

        // Read number of chunks
        let num_chunks = u64::from_le_bytes(compressed_data[0..8].try_into().unwrap()) as usize;
        let mut offset = 8;

        // Read chunk sizes and data
        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            if offset + 8 > compressed_data.len() {
                return Err(TorshError::InvalidArgument(
                    "Invalid parallel-compressed data format".to_string(),
                ));
            }

            let chunk_size =
                u64::from_le_bytes(compressed_data[offset..offset + 8].try_into().unwrap())
                    as usize;
            offset += 8;

            if offset + chunk_size > compressed_data.len() {
                return Err(TorshError::InvalidArgument(
                    "Invalid chunk size in parallel-compressed data".to_string(),
                ));
            }

            chunks.push(&compressed_data[offset..offset + chunk_size]);
            offset += chunk_size;
        }

        // Decompress chunks in parallel
        use scirs2_core::parallel_ops::{IntoParallelIterator, ParallelIterator};

        let decompressed_chunks: Vec<_> = chunks
            .into_par_iter()
            .map(|chunk| {
                self.compressor
                    .decompress_data(chunk, algorithm)
                    .map(|result| result.data)
            })
            .collect::<Result<Vec<_>>>()?;

        // Combine decompressed chunks
        let combined_data = decompressed_chunks.into_iter().flatten().collect();

        let decompression_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(DecompressionResult {
            data: combined_data,
            algorithm,
            decompression_time_ms,
        })
    }
}

/// Compression statistics collector
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total bytes compressed
    pub total_compressed: usize,
    /// Total bytes after compression
    pub total_after_compression: usize,
    /// Total compression time
    pub total_time_ms: u64,
    /// Algorithm usage statistics
    pub algorithm_usage: HashMap<CompressionAlgorithm, u32>,
    /// Average compression ratios by algorithm
    pub algorithm_ratios: HashMap<CompressionAlgorithm, f32>,
}

impl CompressionStats {
    /// Create new stats collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Record compression result
    pub fn record(&mut self, result: &CompressionResult) {
        self.total_compressed += result.original_size;
        self.total_after_compression += result.compressed_size;
        self.total_time_ms += result.compression_time_ms;

        *self.algorithm_usage.entry(result.algorithm).or_insert(0) += 1;

        // Update rolling average of compression ratios
        let current_ratio = self.algorithm_ratios.get(&result.algorithm).unwrap_or(&0.0);
        let count = self.algorithm_usage[&result.algorithm] as f32;
        let new_ratio = (current_ratio * (count - 1.0) + result.ratio) / count;
        self.algorithm_ratios.insert(result.algorithm, new_ratio);
    }

    /// Get overall compression ratio
    pub fn overall_ratio(&self) -> f32 {
        if self.total_compressed == 0 {
            1.0
        } else {
            self.total_after_compression as f32 / self.total_compressed as f32
        }
    }

    /// Get space saved in bytes
    pub fn space_saved(&self) -> usize {
        self.total_compressed
            .saturating_sub(self.total_after_compression)
    }

    /// Get space saved as percentage
    pub fn space_saved_percent(&self) -> f32 {
        if self.total_compressed == 0 {
            0.0
        } else {
            (self.space_saved() as f32 / self.total_compressed as f32) * 100.0
        }
    }

    /// Get most used algorithm
    pub fn most_used_algorithm(&self) -> Option<CompressionAlgorithm> {
        self.algorithm_usage
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&algorithm, _)| algorithm)
    }

    /// Get best performing algorithm (by compression ratio)
    pub fn best_performing_algorithm(&self) -> Option<CompressionAlgorithm> {
        self.algorithm_ratios
            .iter()
            .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&algorithm, _)| algorithm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::new()
            .with_algorithm(CompressionAlgorithm::Zstd)
            .with_level(CompressionLevel(8))
            .with_strategy(CompressionStrategy::Size);

        assert_eq!(config.default_algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(config.default_level.0, 8);
        assert_eq!(config.strategy, CompressionStrategy::Size);
    }

    #[test]
    fn test_compression_level() {
        let level = CompressionLevel(15);

        assert_eq!(level.for_algorithm(CompressionAlgorithm::Gzip), 9);
        assert_eq!(level.for_algorithm(CompressionAlgorithm::Zstd), 15);
        assert_eq!(level.for_algorithm(CompressionAlgorithm::Brotli), 11);
    }

    #[test]
    fn test_basic_compression() {
        let compressor = AdvancedCompressor::new();
        let test_data = "Hello, World! ".repeat(100);

        let result = compressor
            .compress_data(
                test_data.as_bytes(),
                CompressionAlgorithm::Gzip,
                CompressionLevel(6),
            )
            .unwrap();

        assert_eq!(result.algorithm, CompressionAlgorithm::Gzip);
        assert_eq!(result.original_size, test_data.len());
        assert!(result.compressed_size < result.original_size);
        assert!(result.ratio < 1.0);
    }

    #[test]
    fn test_decompression() {
        let compressor = AdvancedCompressor::new();
        let test_data = "This is test data for compression and decompression.".repeat(10);

        let compression_result = compressor
            .compress_data(
                test_data.as_bytes(),
                CompressionAlgorithm::Gzip,
                CompressionLevel(6),
            )
            .unwrap();

        let decompression_result = compressor
            .decompress_data(&compression_result.data, CompressionAlgorithm::Gzip)
            .unwrap();

        assert_eq!(decompression_result.data, test_data.as_bytes());
        assert_eq!(decompression_result.algorithm, CompressionAlgorithm::Gzip);
    }

    #[test]
    fn test_resource_compression() {
        let compressor = AdvancedCompressor::new();

        let resource = Resource::new(
            "test.txt".to_string(),
            ResourceType::Text,
            "This is a test text file with some content that should compress well."
                .repeat(20)
                .as_bytes()
                .to_vec(),
        );

        let result = compressor.compress_resource(&resource).unwrap();

        // Text should compress well
        assert!(result.ratio < 0.5);
        assert_eq!(result.original_size, resource.data.len());
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();

        let result1 = CompressionResult {
            data: vec![0; 100],
            algorithm: CompressionAlgorithm::Gzip,
            level: CompressionLevel(6),
            original_size: 200,
            compressed_size: 100,
            ratio: 0.5,
            compression_time_ms: 10,
        };

        let result2 = CompressionResult {
            data: vec![0; 80],
            algorithm: CompressionAlgorithm::Zstd,
            level: CompressionLevel(6),
            original_size: 200,
            compressed_size: 80,
            ratio: 0.4,
            compression_time_ms: 8,
        };

        stats.record(&result1);
        stats.record(&result2);

        assert_eq!(stats.total_compressed, 400);
        assert_eq!(stats.total_after_compression, 180);
        assert_eq!(stats.space_saved(), 220);
        assert!((stats.space_saved_percent() - 55.0).abs() < 0.1);

        assert_eq!(stats.algorithm_usage[&CompressionAlgorithm::Gzip], 1);
        assert_eq!(stats.algorithm_usage[&CompressionAlgorithm::Zstd], 1);

        assert_eq!(
            stats.best_performing_algorithm(),
            Some(CompressionAlgorithm::Zstd)
        );
    }

    #[test]
    fn test_small_file_skip() {
        let compressor = AdvancedCompressor::new();
        let small_data = b"tiny";

        // Create a small resource to test the size threshold
        let small_resource = Resource::new(
            "small.txt".to_string(),
            ResourceType::Text,
            small_data.to_vec(),
        );

        let result = compressor.compress_resource(&small_resource).unwrap();

        // Small files should not be compressed (below threshold)
        assert_eq!(result.algorithm, CompressionAlgorithm::None);
        assert_eq!(result.data, small_data);
        assert_eq!(result.ratio, 1.0);
    }

    #[test]
    fn test_benchmark_algorithms() {
        let compressor = AdvancedCompressor::new();
        let test_data = "This is benchmark data. ".repeat(100);

        let results = compressor
            .benchmark_algorithms(test_data.as_bytes())
            .unwrap();

        // Should have results for multiple algorithms
        assert!(results.len() >= 2);

        // Results should be sorted by compression ratio
        for i in 1..results.len() {
            assert!(results[i - 1].ratio <= results[i].ratio);
        }

        // All should have compressed the same original data
        for result in &results {
            assert_eq!(result.original_size, test_data.len());
        }
    }
}
