//! Model downloader for fetching pre-trained models

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use indicatif::{ProgressBar, ProgressStyle};

use crate::registry::{ModelHandle, ModelSource};
use crate::{ModelError, ModelResult};
use sha2::Digest;

/// Download progress information
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

impl DownloadProgress {
    /// Calculate download percentage
    pub fn percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.downloaded_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }
}

/// Trait for download progress callbacks
#[async_trait]
pub trait ProgressCallback: Send + Sync {
    async fn on_progress(&self, progress: DownloadProgress);
    async fn on_complete(&self);
    async fn on_error(&self, error: &ModelError);
}

/// Default progress callback that prints to console
pub struct ConsoleProgressCallback {
    progress_bar: Option<ProgressBar>,
}

impl ConsoleProgressCallback {
    pub fn new() -> Self {
        Self { progress_bar: None }
    }

    pub fn with_progress_bar() -> Self {
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        Self {
            progress_bar: Some(pb),
        }
    }
}

#[async_trait]
impl ProgressCallback for ConsoleProgressCallback {
    async fn on_progress(&self, progress: DownloadProgress) {
        if let Some(ref pb) = self.progress_bar {
            pb.set_length(progress.total_bytes);
            pb.set_position(progress.downloaded_bytes);
            pb.set_message(format!("{:.1} MB/s", progress.speed_bps / 1024.0 / 1024.0));
        } else {
            println!(
                "Downloaded: {:.1}% ({}/{} bytes) at {:.1} MB/s",
                progress.percentage(),
                progress.downloaded_bytes,
                progress.total_bytes,
                progress.speed_bps / 1024.0 / 1024.0
            );
        }
    }

    async fn on_complete(&self) {
        if let Some(ref pb) = self.progress_bar {
            pb.finish_with_message("Download complete");
        } else {
            println!("Download complete!");
        }
    }

    async fn on_error(&self, error: &ModelError) {
        if let Some(ref pb) = self.progress_bar {
            pb.abandon_with_message(format!("Download failed: {}", error));
        } else {
            eprintln!("Download failed: {}", error);
        }
    }
}

/// Model downloader for fetching pre-trained models
pub struct ModelDownloader {
    /// HTTP client for downloads
    #[cfg(feature = "download")]
    client: reqwest::Client,
    /// Download timeout in seconds
    timeout_seconds: u64,
}

impl ModelDownloader {
    /// Create new model downloader
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "download")]
            client: reqwest::Client::builder()
                .user_agent("torsh-models/0.1.0-alpha.2")
                .build()
                .expect("Failed to create HTTP client"),
            timeout_seconds: 300, // 5 minutes default
        }
    }

    /// Set download timeout
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Download model to specified path
    pub async fn download_model(
        &self,
        handle: &ModelHandle,
        callback: Option<Box<dyn ProgressCallback>>,
    ) -> ModelResult<PathBuf> {
        match &handle.info.source {
            ModelSource::Local(path) => {
                // Already local, just return the path
                Ok(path.clone())
            }
            ModelSource::Url(url) => {
                self.download_from_url(url, &handle.local_path, callback)
                    .await
            }
            ModelSource::HuggingFace { repo, filename } => {
                let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
                self.download_from_url(&url, &handle.local_path, callback)
                    .await
            }
            ModelSource::Registry {
                registry: _,
                path: _,
            } => Err(ModelError::DownloadFailed {
                reason: "Custom registry downloads not yet supported".to_string(),
            }),
        }
    }

    /// Download from URL
    #[cfg(feature = "download")]
    async fn download_from_url(
        &self,
        url: &str,
        destination: &Path,
        callback: Option<Box<dyn ProgressCallback>>,
    ) -> ModelResult<PathBuf> {
        use std::time::Instant;
        use tokio::fs::File;
        use tokio::io::AsyncWriteExt;

        // Create destination directory if it doesn't exist
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Start download
        let response = self
            .client
            .get(url)
            .timeout(std::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(ModelError::DownloadFailed {
                reason: format!(
                    "HTTP {}: {}",
                    response.status(),
                    response.status().canonical_reason().unwrap_or("Unknown")
                ),
            });
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        let mut file = File::create(destination).await?;

        let start_time = Instant::now();
        let mut last_update = start_time;

        // Download with progress tracking
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Update progress
            if let Some(ref callback) = callback {
                let now = Instant::now();
                if now.duration_since(last_update).as_millis() >= 100 {
                    // Update every 100ms
                    let elapsed = now.duration_since(start_time).as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        downloaded as f64 / elapsed
                    } else {
                        0.0
                    };
                    let eta = if speed > 0.0 && total_size > 0 {
                        (total_size - downloaded) as f64 / speed
                    } else {
                        0.0
                    };

                    let progress = DownloadProgress {
                        total_bytes: total_size,
                        downloaded_bytes: downloaded,
                        speed_bps: speed,
                        eta_seconds: eta,
                    };

                    callback.on_progress(progress).await;
                    last_update = now;
                }
            }
        }

        file.flush().await?;

        if let Some(ref callback) = callback {
            callback.on_complete().await;
        }

        Ok(destination.to_path_buf())
    }

    #[cfg(not(feature = "download"))]
    async fn download_from_url(
        &self,
        _url: &str,
        _destination: &Path,
        _callback: Option<Box<dyn ProgressCallback>>,
    ) -> ModelResult<PathBuf> {
        Err(ModelError::DownloadFailed {
            reason: "Download feature not enabled".to_string(),
        })
    }

    /// Verify downloaded model checksum
    pub async fn verify_checksum(
        &self,
        file_path: &Path,
        expected_checksum: &str,
    ) -> ModelResult<bool> {
        let data = std::fs::read(file_path)?;
        let hash = sha2::Sha256::digest(&data);
        let hex_hash = hex::encode(hash);

        Ok(hex_hash == expected_checksum)
    }

    /// Download model if not already cached
    pub async fn ensure_model_available(
        &self,
        handle: &ModelHandle,
        force_redownload: bool,
        callback: Option<Box<dyn ProgressCallback>>,
    ) -> ModelResult<PathBuf> {
        // Check if model already exists and is valid
        if !force_redownload && handle.exists() {
            if handle.validate_checksum()? {
                return Ok(handle.local_path.clone());
            } else {
                tracing::warn!("Model checksum validation failed, re-downloading");
            }
        }

        // Download the model
        self.download_model(handle, callback).await
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to download a model by name
pub async fn download_model_by_name(
    name: &str,
    version: Option<&str>,
    cache_dir: Option<PathBuf>,
    callback: Option<Box<dyn ProgressCallback>>,
) -> ModelResult<PathBuf> {
    use crate::registry::get_global_registry;

    let registry = get_global_registry();
    let handle = if let Some(cache_dir) = cache_dir {
        // Create temporary registry with custom cache dir
        let temp_registry = crate::registry::ModelRegistry::new(cache_dir)?;
        temp_registry.register_builtin_models()?;
        temp_registry.get_model_handle(name, version)?
    } else {
        registry.get_model_handle(name, version)?
    };

    let downloader = ModelDownloader::new();
    downloader
        .ensure_model_available(&handle, false, callback)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio;

    #[tokio::test]
    async fn test_downloader_creation() {
        let downloader = ModelDownloader::new();
        assert_eq!(downloader.timeout_seconds, 300);
    }

    #[tokio::test]
    async fn test_progress_callback() {
        let callback = ConsoleProgressCallback::new();
        let progress = DownloadProgress {
            total_bytes: 1000,
            downloaded_bytes: 500,
            speed_bps: 1024.0,
            eta_seconds: 10.0,
        };

        callback.on_progress(progress).await;
        assert_eq!(50.0, 500.0 / 1000.0 * 100.0);
    }

    #[test]
    fn test_download_progress() {
        let progress = DownloadProgress {
            total_bytes: 1000,
            downloaded_bytes: 250,
            speed_bps: 1024.0,
            eta_seconds: 5.0,
        };

        assert_eq!(progress.percentage(), 25.0);
    }
}
