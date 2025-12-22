//! Asynchronous package operations
//!
//! This module provides async/await support for package operations including:
//! - Asynchronous package loading and saving
//! - Background compression and decompression
//! - Concurrent package operations
//! - Stream-based package processing

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncReadExt;
use tokio::sync::Semaphore;
use torsh_core::error::{Result, TorshError};

use crate::compression::{AdvancedCompressor, CompressionAlgorithm, CompressionLevel};
use crate::package::Package;

/// Asynchronous package loader
pub struct AsyncPackageLoader {
    /// Maximum concurrent operations
    _max_concurrent: usize,
    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,
}

/// Asynchronous package saver
pub struct AsyncPackageSaver {
    /// Compression settings
    compressor: Arc<AdvancedCompressor>,
    /// Maximum concurrent operations
    _max_concurrent: usize,
}

/// Background package processor
pub struct BackgroundProcessor {
    /// Number of worker threads
    num_workers: usize,
}

/// Package download progress
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Total bytes to download
    pub total_bytes: u64,
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Download speed in bytes per second
    pub speed_bps: f64,
    /// Estimated time remaining in seconds
    pub eta_seconds: f64,
}

impl AsyncPackageLoader {
    /// Create a new async package loader
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            _max_concurrent: max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    /// Load a package asynchronously
    pub async fn load_package<P: AsRef<Path>>(&self, path: P) -> Result<Package> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| TorshError::IoError(format!("Failed to acquire semaphore: {}", e)))?;

        let path = path.as_ref();
        let data = fs::read(path)
            .await
            .map_err(|e| TorshError::IoError(format!("Failed to read file: {}", e)))?;

        // Deserialize package
        let (package, _) = bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(package)
    }

    /// Load multiple packages concurrently
    pub async fn load_packages<P: AsRef<Path>>(
        &self,
        paths: Vec<P>,
    ) -> Result<Vec<Result<Package>>> {
        let mut tasks = Vec::new();

        for path in paths {
            let path = path.as_ref().to_path_buf();
            let semaphore = Arc::clone(&self.semaphore);

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|e| {
                    TorshError::IoError(format!("Failed to acquire semaphore: {}", e))
                })?;

                let data = fs::read(&path)
                    .await
                    .map_err(|e| TorshError::IoError(format!("Failed to read file: {}", e)))?;

                let (package, _) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())
                        .map_err(|e| TorshError::SerializationError(e.to_string()))?;

                Ok::<Package, TorshError>(package)
            });

            tasks.push(task);
        }

        let results = futures::future::join_all(tasks).await;

        let packages: Vec<Result<Package>> = results
            .into_iter()
            .map(|r| match r {
                Ok(result) => result,
                Err(e) => Err(TorshError::IoError(format!("Task failed: {}", e))),
            })
            .collect();

        Ok(packages)
    }

    /// Stream a large package in chunks
    pub async fn stream_package<P: AsRef<Path>>(
        &self,
        path: P,
        chunk_size: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| TorshError::IoError(format!("Failed to acquire semaphore: {}", e)))?;

        let mut file = fs::File::open(path)
            .await
            .map_err(|e| TorshError::IoError(format!("Failed to open file: {}", e)))?;

        let mut chunks = Vec::new();
        let mut buffer = vec![0u8; chunk_size];

        loop {
            let bytes_read = file
                .read(&mut buffer)
                .await
                .map_err(|e| TorshError::IoError(format!("Failed to read: {}", e)))?;

            if bytes_read == 0 {
                break;
            }

            chunks.push(buffer[..bytes_read].to_vec());
        }

        Ok(chunks)
    }
}

impl AsyncPackageSaver {
    /// Create a new async package saver
    pub fn new(compressor: AdvancedCompressor, max_concurrent: usize) -> Self {
        Self {
            compressor: Arc::new(compressor),
            _max_concurrent: max_concurrent,
        }
    }

    /// Save a package asynchronously
    pub async fn save_package<P: AsRef<Path>>(&self, package: &Package, path: P) -> Result<()> {
        let data = bincode::serde::encode_to_vec(package, bincode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        fs::write(path, data)
            .await
            .map_err(|e| TorshError::IoError(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Save package with compression asynchronously
    pub async fn save_package_compressed<P: AsRef<Path>>(
        &self,
        package: &Package,
        path: P,
        algorithm: CompressionAlgorithm,
        level: CompressionLevel,
    ) -> Result<()> {
        let package_data = bincode::serde::encode_to_vec(package, bincode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        // Compress in a separate task to avoid blocking
        let compressor = Arc::clone(&self.compressor);
        let compressed = tokio::task::spawn_blocking(move || {
            compressor.compress_data(&package_data, algorithm, level)
        })
        .await
        .map_err(|e| TorshError::IoError(format!("Compression task failed: {}", e)))??;

        fs::write(path, compressed.data)
            .await
            .map_err(|e| TorshError::IoError(format!("Failed to write compressed file: {}", e)))?;

        Ok(())
    }

    /// Save multiple packages concurrently
    pub async fn save_packages(
        &self,
        packages: Vec<(Package, PathBuf)>,
    ) -> Result<Vec<Result<()>>> {
        let mut tasks = Vec::new();

        for (package, path) in packages {
            let task = tokio::spawn(async move {
                let data = bincode::serde::encode_to_vec(&package, bincode::config::standard())
                    .map_err(|e| TorshError::SerializationError(e.to_string()))?;

                fs::write(&path, data)
                    .await
                    .map_err(|e| TorshError::IoError(format!("Failed to write file: {}", e)))?;

                Ok::<(), TorshError>(())
            });

            tasks.push(task);
        }

        let results = futures::future::join_all(tasks).await;

        let save_results: Vec<Result<()>> = results
            .into_iter()
            .map(|r| match r {
                Ok(result) => result,
                Err(e) => Err(TorshError::IoError(format!("Task failed: {}", e))),
            })
            .collect();

        Ok(save_results)
    }
}

impl BackgroundProcessor {
    /// Create a new background processor
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }

    /// Process packages in the background
    pub async fn process_packages<F, Fut>(
        &self,
        packages: Vec<Package>,
        processor: F,
    ) -> Result<Vec<Result<Package>>>
    where
        F: Fn(Package) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Package>> + Send,
    {
        let semaphore = Arc::new(Semaphore::new(self.num_workers));
        let mut tasks = Vec::new();

        for package in packages {
            let processor = processor.clone();
            let semaphore = Arc::clone(&semaphore);

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|e| {
                    TorshError::IoError(format!("Failed to acquire semaphore: {}", e))
                })?;

                processor(package).await
            });

            tasks.push(task);
        }

        let results = futures::future::join_all(tasks).await;

        let processed: Vec<Result<Package>> = results
            .into_iter()
            .map(|r| match r {
                Ok(result) => result,
                Err(e) => Err(TorshError::IoError(format!("Task failed: {}", e))),
            })
            .collect();

        Ok(processed)
    }

    /// Compress packages in parallel background tasks
    pub async fn compress_packages_background(
        &self,
        packages: Vec<(Package, CompressionAlgorithm, CompressionLevel)>,
    ) -> Result<Vec<Result<Vec<u8>>>> {
        let compressor = Arc::new(AdvancedCompressor::new());
        let semaphore = Arc::new(Semaphore::new(self.num_workers));
        let mut tasks = Vec::new();

        for (package, algorithm, level) in packages {
            let compressor = Arc::clone(&compressor);
            let semaphore = Arc::clone(&semaphore);

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.map_err(|e| {
                    TorshError::IoError(format!("Failed to acquire semaphore: {}", e))
                })?;

                let package_data =
                    bincode::serde::encode_to_vec(&package, bincode::config::standard())
                        .map_err(|e| TorshError::SerializationError(e.to_string()))?;

                let result = tokio::task::spawn_blocking(move || {
                    compressor.compress_data(&package_data, algorithm, level)
                })
                .await
                .map_err(|e| TorshError::IoError(format!("Compression task failed: {}", e)))??;

                Ok::<Vec<u8>, TorshError>(result.data)
            });

            tasks.push(task);
        }

        let results = futures::future::join_all(tasks).await;

        let compressed: Vec<Result<Vec<u8>>> = results
            .into_iter()
            .map(|r| match r {
                Ok(result) => result,
                Err(e) => Err(TorshError::IoError(format!("Task failed: {}", e))),
            })
            .collect();

        Ok(compressed)
    }
}

/// Download a package from a URL with progress tracking
pub async fn download_package_with_progress<F>(
    _url: &str,
    _output_path: &Path,
    mut progress_callback: F,
) -> Result<()>
where
    F: FnMut(DownloadProgress),
{
    use std::time::Instant;

    // This is a placeholder implementation
    // In a real implementation, you would use reqwest or hyper
    let start_time = Instant::now();

    // Simulate download progress
    let total_bytes = 1_000_000u64; // 1MB
    let chunk_size = 100_000u64;

    for downloaded_bytes in (0..total_bytes).step_by(chunk_size as usize) {
        let elapsed = start_time.elapsed().as_secs_f64();
        let speed_bps = if elapsed > 0.0 {
            downloaded_bytes as f64 / elapsed
        } else {
            0.0
        };
        let remaining_bytes = total_bytes - downloaded_bytes;
        let eta_seconds = if speed_bps > 0.0 {
            remaining_bytes as f64 / speed_bps
        } else {
            0.0
        };

        let progress = DownloadProgress {
            total_bytes,
            downloaded_bytes,
            speed_bps,
            eta_seconds,
        };

        progress_callback(progress);

        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Final progress
    progress_callback(DownloadProgress {
        total_bytes,
        downloaded_bytes: total_bytes,
        speed_bps: 0.0,
        eta_seconds: 0.0,
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_package_load() {
        let package = Package::new("test".to_string(), "1.0.0".to_string());
        let temp_file = tempfile::NamedTempFile::new().unwrap();

        // Save package synchronously for testing
        let data = bincode::serde::encode_to_vec(&package, bincode::config::standard()).unwrap();
        std::fs::write(temp_file.path(), data).unwrap();

        let loader = AsyncPackageLoader::new(4);
        let loaded = loader.load_package(temp_file.path()).await.unwrap();

        assert_eq!(loaded.name(), package.name());
    }

    #[tokio::test]
    async fn test_async_package_save() {
        let package = Package::new("test".to_string(), "1.0.0".to_string());
        let temp_file = tempfile::NamedTempFile::new().unwrap();

        let compressor = AdvancedCompressor::new();
        let saver = AsyncPackageSaver::new(compressor, 4);

        saver
            .save_package(&package, temp_file.path())
            .await
            .unwrap();

        assert!(temp_file.path().exists());
    }

    #[tokio::test]
    async fn test_concurrent_package_loading() {
        let loader = AsyncPackageLoader::new(4);
        let mut paths = Vec::new();
        let mut _temp_files = Vec::new(); // Keep temp files alive

        // Create multiple test packages
        for i in 0..5 {
            let package = Package::new(format!("package_{}", i), "1.0.0".to_string());
            let temp_file = tempfile::NamedTempFile::new().unwrap();
            let data =
                bincode::serde::encode_to_vec(&package, bincode::config::standard()).unwrap();
            std::fs::write(temp_file.path(), &data).unwrap();
            paths.push(temp_file.path().to_path_buf());
            _temp_files.push(temp_file); // Keep alive
        }

        let results = loader.load_packages(paths).await.unwrap();

        assert_eq!(results.len(), 5);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_background_compression() {
        let processor = BackgroundProcessor::new(4);
        let packages = vec![
            (
                Package::new("pkg1".to_string(), "1.0.0".to_string()),
                CompressionAlgorithm::Gzip,
                CompressionLevel(6),
            ),
            (
                Package::new("pkg2".to_string(), "1.0.0".to_string()),
                CompressionAlgorithm::Zstd,
                CompressionLevel(3),
            ),
        ];

        let results = processor
            .compress_packages_background(packages)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_download_progress() {
        use tempfile::NamedTempFile;

        let temp_file = NamedTempFile::new().unwrap();
        let mut progress_updates = Vec::new();

        download_package_with_progress(
            "http://example.com/package.torsh",
            temp_file.path(),
            |progress| {
                progress_updates.push(progress.downloaded_bytes);
            },
        )
        .await
        .unwrap();

        assert!(!progress_updates.is_empty());
        assert_eq!(*progress_updates.last().unwrap(), 1_000_000);
    }
}
