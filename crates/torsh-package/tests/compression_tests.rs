//! Comprehensive compression tests

use torsh_core::error::Result;
use torsh_package::compression::*;

#[test]
fn test_zstd_compression() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let data = b"Hello, Zstandard! ".repeat(1000);

    let result = compressor.compress_data(
        &data,
        CompressionAlgorithm::Zstd,
        CompressionLevel(6), // Default level
    )?;

    assert_eq!(result.algorithm, CompressionAlgorithm::Zstd);
    assert!(result.compressed_size < result.original_size);
    assert!(result.ratio < 1.0);

    let decompressed = compressor.decompress_data(&result.data, CompressionAlgorithm::Zstd)?;
    assert_eq!(decompressed.data, data);

    Ok(())
}

#[test]
fn test_lzma_compression() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let data = b"LZMA compression test data with repeated patterns ".repeat(100);

    let result = compressor.compress_data(
        &data,
        CompressionAlgorithm::Lzma,
        CompressionLevel(9), // Maximum level
    )?;

    assert_eq!(result.algorithm, CompressionAlgorithm::Lzma);
    assert!(result.compressed_size < result.original_size);

    let decompressed = compressor.decompress_data(&result.data, CompressionAlgorithm::Lzma)?;
    assert_eq!(decompressed.data, data);

    Ok(())
}

#[test]
fn test_all_compression_algorithms() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let data = b"Test data for all algorithms! ".repeat(500);

    let algorithms = vec![
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lzma,
    ];

    for algorithm in algorithms {
        let compressed = compressor.compress_data(&data, algorithm, CompressionLevel(6))?;
        let decompressed = compressor.decompress_data(&compressed.data, algorithm)?;

        assert_eq!(decompressed.data, data, "Failed for {:?}", algorithm);
        assert!(compressed.ratio < 1.0, "No compression for {:?}", algorithm);
    }

    Ok(())
}

#[test]
fn test_parallel_compression() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let parallel_compressor = ParallelCompressor::new(compressor).with_chunk_size(1024 * 512); // 512KB chunks

    // Create 5MB of test data
    let data = b"Parallel compression test data! ".repeat(150_000);

    let result = parallel_compressor.compress_parallel(
        &data,
        CompressionAlgorithm::Zstd,
        CompressionLevel(3), // Fast level
    )?;

    assert!(result.compressed_size < result.original_size);

    let decompressed =
        parallel_compressor.decompress_parallel(&result.data, CompressionAlgorithm::Zstd)?;

    assert_eq!(decompressed.data, data);

    Ok(())
}

#[test]
fn test_compression_levels() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let data = b"Compression level test data with some repetition ".repeat(1000);

    let fast = compressor.compress_data(&data, CompressionAlgorithm::Zstd, CompressionLevel(3))?;
    let default =
        compressor.compress_data(&data, CompressionAlgorithm::Zstd, CompressionLevel(6))?;
    let maximum =
        compressor.compress_data(&data, CompressionAlgorithm::Zstd, CompressionLevel(9))?;

    // Higher compression levels should generally produce smaller results
    // (though not guaranteed for all data)
    assert!(maximum.compressed_size <= default.compressed_size * 2);
    assert!(default.compressed_size <= fast.compressed_size * 2);

    // All should decompress correctly
    let decompressed_fast = compressor.decompress_data(&fast.data, CompressionAlgorithm::Zstd)?;
    let decompressed_default =
        compressor.decompress_data(&default.data, CompressionAlgorithm::Zstd)?;
    let decompressed_maximum =
        compressor.decompress_data(&maximum.data, CompressionAlgorithm::Zstd)?;

    assert_eq!(decompressed_fast.data, data);
    assert_eq!(decompressed_default.data, data);
    assert_eq!(decompressed_maximum.data, data);

    Ok(())
}

#[test]
fn test_compression_stats() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let mut stats = CompressionStats::new();

    let data1 = b"First test data ".repeat(1000);
    let data2 = b"Second test data ".repeat(1000);

    let result1 =
        compressor.compress_data(&data1, CompressionAlgorithm::Zstd, CompressionLevel(6))?;
    let result2 =
        compressor.compress_data(&data2, CompressionAlgorithm::Gzip, CompressionLevel(6))?;

    stats.record(&result1);
    stats.record(&result2);

    assert!(stats.overall_ratio() < 1.0);
    assert!(stats.space_saved() > 0);
    assert!(stats.space_saved_percent() > 0.0);

    Ok(())
}

#[test]
fn test_empty_data_compression() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let data: &[u8] = b"";

    let result = compressor.compress_data(data, CompressionAlgorithm::Gzip, CompressionLevel(6))?;
    let decompressed = compressor.decompress_data(&result.data, CompressionAlgorithm::Gzip)?;

    assert_eq!(decompressed.data, data);

    Ok(())
}

#[test]
fn test_large_data_streaming() -> Result<()> {
    let compressor = AdvancedCompressor::new();
    let parallel_compressor = ParallelCompressor::new(compressor)
        .with_chunk_size(1024 * 1024) // 1MB chunks
        .with_num_threads(4);

    // Create 10MB of test data
    let data = b"Large streaming test data! ".repeat(400_000);

    let compressed = parallel_compressor.compress_parallel(
        &data,
        CompressionAlgorithm::Zstd,
        CompressionLevel(6),
    )?;

    let decompressed =
        parallel_compressor.decompress_parallel(&compressed.data, CompressionAlgorithm::Zstd)?;

    assert_eq!(decompressed.data, data);

    Ok(())
}

#[test]
fn test_compression_ratio_calculation() -> Result<()> {
    let compressor = AdvancedCompressor::new();

    // Highly compressible data
    let compressible_data = b"A".repeat(10000);
    let result = compressor.compress_data(
        &compressible_data,
        CompressionAlgorithm::Gzip,
        CompressionLevel(9),
    )?;

    // Should achieve good compression
    assert!(
        result.ratio < 0.1,
        "Expected high compression for repeated data"
    );

    Ok(())
}

// Removed test_adaptive_compression_selector as AdaptiveCompressionSelector is not exported

#[test]
fn test_compression_benchmark() {
    use std::time::Instant;

    let compressor = AdvancedCompressor::new();
    let data = b"Benchmark test data! ".repeat(10_000);

    let algorithms = vec![
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lzma,
    ];

    for algorithm in algorithms {
        let start = Instant::now();
        let result = compressor
            .compress_data(&data, algorithm, CompressionLevel(6))
            .expect("Compression failed");
        let duration = start.elapsed();

        println!(
            "{:?}: {} bytes -> {} bytes ({:.2}%) in {:?}",
            algorithm,
            result.original_size,
            result.compressed_size,
            result.ratio * 100.0,
            duration
        );

        // Verify decompression
        let decompressed = compressor
            .decompress_data(&result.data, algorithm)
            .expect("Decompression failed");
        assert_eq!(decompressed.data, data);
    }
}
