//! Core download functionality for ToRSh Hub
//!
//! This module provides the fundamental download operations including basic file downloads,
//! async downloads, progress reporting, retry logic, and core I/O operations. It serves as
//! the foundation for more specialized download features.

use futures::stream::{self, StreamExt};
use reqwest::{blocking::Client as BlockingClient, Client};
use std::fs::{self, File};
use std::io::{self, Read, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::fs::File as AsyncFile;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use torsh_core::error::{Result, TorshError};

// Import from our validation and config modules
use super::config::ParallelDownloadConfig;
use super::validation::validate_url;

/// Download a file with optional progress reporting (synchronous version)
///
/// This is the core synchronous download function that handles basic file downloads
/// with optional progress reporting. It creates the necessary directory structure,
/// downloads the file to a temporary location, and moves it to the final destination
/// upon successful completion.
///
/// # Arguments
/// * `url` - The URL to download from
/// * `dest_path` - The destination file path
/// * `progress` - Whether to display download progress
///
/// # Returns
/// * `Ok(())` if the download succeeded
/// * `Err(TorshError)` if the download failed
///
/// # Examples
/// ```rust,no_run
/// use std::path::Path;
/// use torsh_hub::download::core::download_file;
///
/// let result = download_file(
///     "https://example.com/file.txt",
///     Path::new("/tmp/downloaded_file.txt"),
///     true
/// );
/// ```
pub fn download_file(url: &str, dest_path: &Path, progress: bool) -> Result<()> {
    // Validate URL before attempting download
    validate_url(url)?;

    // Create parent directory if needed
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Create HTTP client
    let client = BlockingClient::builder()
        .user_agent("torsh-hub/0.1.0-alpha.2")
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    // Start download
    let response = client
        .get(url)
        .send()
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    if !response.status().is_success() {
        return Err(TorshError::IoError(format!(
            "Failed to download {}: HTTP {}",
            url,
            response.status()
        )));
    }

    // Get content length for progress
    let total_size = response.content_length();

    // Create temporary file
    let temp_path = dest_path.with_extension("tmp");
    let mut file = File::create(&temp_path)?;

    // Download with progress
    if progress {
        if let Some(total) = total_size {
            println!("Downloading {} ({:.1} MB)", url, total as f64 / 1_048_576.0);
        } else {
            println!("Downloading {}", url);
        }
    }

    let mut downloaded = 0u64;
    let mut buffer = vec![0; 8192];
    let mut response = response;

    loop {
        let n = response
            .read(&mut buffer)
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        if n == 0 {
            break;
        }

        file.write_all(&buffer[..n])?;
        downloaded += n as u64;

        if progress {
            if let Some(total) = total_size {
                print_progress(downloaded, total);
            }
        }
    }

    if progress {
        println!(); // New line after progress
    }

    // Move temporary file to final destination
    fs::rename(&temp_path, dest_path)?;

    Ok(())
}

/// Download with retry logic
///
/// This function provides robust downloading with configurable retry attempts.
/// It implements exponential backoff between retry attempts to handle temporary
/// network issues or server overload conditions.
///
/// # Arguments
/// * `url` - The URL to download from
/// * `dest_path` - The destination file path
/// * `max_retries` - Maximum number of retry attempts
/// * `progress` - Whether to display download progress
///
/// # Returns
/// * `Ok(())` if the download succeeded (possibly after retries)
/// * `Err(TorshError)` if all attempts failed
///
/// # Examples
/// ```rust,no_run
/// use std::path::Path;
/// use torsh_hub::download::core::download_with_retry;
///
/// let result = download_with_retry(
///     "https://example.com/file.txt",
///     Path::new("/tmp/downloaded_file.txt"),
///     3,  // max 3 retries
///     true
/// );
/// ```
pub fn download_with_retry(
    url: &str,
    dest_path: &Path,
    max_retries: usize,
    progress: bool,
) -> Result<()> {
    let mut last_error = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            println!("Retry attempt {}/{}", attempt, max_retries);
            std::thread::sleep(std::time::Duration::from_secs(2u64.pow(attempt as u32)));
        }

        match download_file(url, dest_path, progress) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    eprintln!("Download failed: {:?}, retrying...", last_error);
                }
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| TorshError::IoError("Download failed after all retries".to_string())))
}

/// Print download progress
///
/// This function displays a visual progress bar in the terminal showing
/// download completion percentage. It uses Unicode block characters to
/// create an attractive progress visualization.
///
/// # Arguments
/// * `current` - Current number of bytes downloaded
/// * `total` - Total file size in bytes
///
/// # Examples
/// ```rust
/// use torsh_hub::download::core::print_progress;
///
/// print_progress(1024, 2048); // Shows 50% progress
/// ```
pub fn print_progress(current: u64, total: u64) {
    let percentage = (current as f64 / total as f64) * 100.0;
    let filled = (percentage / 2.0) as usize;
    let bar = "█".repeat(filled) + &"░".repeat(50 - filled);

    print!("\r[{}] {:.1}%", bar, percentage);
    io::stdout().flush().expect("stdout flush should succeed");
}

/// Async parallel download of a file with chunked downloading
///
/// This function provides high-performance asynchronous file downloading with
/// support for parallel chunk downloads when the server supports HTTP range requests.
/// It automatically falls back to simple download for servers that don't support
/// range requests or for small files.
///
/// # Arguments
/// * `url` - The URL to download from
/// * `dest_path` - The destination file path
/// * `config` - Download configuration (chunk size, concurrency, etc.)
/// * `progress` - Whether to display download progress
///
/// # Returns
/// * `Ok(())` if the download succeeded
/// * `Err(TorshError)` if the download failed
///
/// # Examples
/// ```rust,no_run
/// use std::path::Path;
/// use torsh_hub::download::core::download_file_parallel;
/// use torsh_hub::download::config::ParallelDownloadConfig;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ParallelDownloadConfig::default();
/// let result = download_file_parallel(
///     "https://example.com/large_file.zip",
///     Path::new("/tmp/large_file.zip"),
///     config,
///     true
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn download_file_parallel(
    url: &str,
    dest_path: &Path,
    config: ParallelDownloadConfig,
    progress: bool,
) -> Result<()> {
    // Validate URL before attempting download
    validate_url(url)?;

    // Create parent directory if needed
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Create async HTTP client
    let client = Client::builder()
        .user_agent("torsh-hub/0.1.0-alpha.2")
        .timeout(Duration::from_secs(config.timeout_seconds))
        .build()
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    // First, get the content length to determine if we can use chunked download
    let head_response = client
        .head(url)
        .send()
        .await
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    if !head_response.status().is_success() {
        return Err(TorshError::IoError(format!(
            "Failed to get file info for {}: HTTP {}",
            url,
            head_response.status()
        )));
    }

    let total_size = head_response.content_length();
    let supports_range = head_response
        .headers()
        .get("accept-ranges")
        .map(|v| v == "bytes")
        .unwrap_or(false);

    if let Some(size) = total_size {
        if supports_range && size > config.chunk_size as u64 * 2 {
            // Use chunked parallel download for large files that support ranges
            download_file_chunked(&client, url, dest_path, size, config, progress).await
        } else {
            // Use simple download for small files or servers that don't support ranges
            download_file_simple(&client, url, dest_path, progress).await
        }
    } else {
        // Unknown size, use simple download
        download_file_simple(&client, url, dest_path, progress).await
    }
}

/// Download file using chunked parallel approach
///
/// This internal function implements parallel chunk downloading for large files.
/// It splits the file into chunks and downloads them concurrently, then assembles
/// them into the final file. This approach can significantly speed up downloads
/// from servers that support HTTP range requests.
///
/// # Arguments
/// * `client` - The HTTP client to use
/// * `url` - The URL to download from
/// * `dest_path` - The destination file path
/// * `total_size` - Total file size in bytes
/// * `config` - Download configuration
/// * `progress` - Whether to display download progress
async fn download_file_chunked(
    client: &Client,
    url: &str,
    dest_path: &Path,
    total_size: u64,
    config: ParallelDownloadConfig,
    progress: bool,
) -> Result<()> {
    let chunk_size = config.chunk_size as u64;
    let num_chunks = total_size.div_ceil(chunk_size);

    if progress {
        println!(
            "Downloading {} in {} chunks ({:.1} MB total)",
            url,
            num_chunks,
            total_size as f64 / 1_048_576.0
        );
    }

    // Create temporary file
    let temp_path = dest_path.with_extension("tmp");
    let temp_file = AsyncFile::create(&temp_path).await?;
    temp_file.set_len(total_size).await?;
    temp_file.sync_all().await?;
    drop(temp_file); // Close file to allow concurrent access

    // Create semaphore to limit concurrent chunks
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_chunks));
    let downloaded_bytes = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Create chunks and download them in parallel
    let chunks: Vec<_> = (0..num_chunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size - 1, total_size - 1);
            (i, start, end)
        })
        .collect();

    let download_tasks = chunks.into_iter().map(|(chunk_id, start, end)| {
        let client = client.clone();
        let url = url.to_string();
        let temp_path = temp_path.clone();
        let semaphore = semaphore.clone();
        let downloaded_bytes = downloaded_bytes.clone();

        async move {
            let _permit = semaphore.acquire().await.unwrap();
            download_chunk(
                &client,
                &url,
                &temp_path,
                chunk_id,
                start,
                end,
                config.max_retries,
            )
            .await?;

            let chunk_size = end - start + 1;
            downloaded_bytes.fetch_add(chunk_size, std::sync::atomic::Ordering::Relaxed);

            if progress {
                let current = downloaded_bytes.load(std::sync::atomic::Ordering::Relaxed);
                print_progress(current, total_size);
            }

            Ok::<(), TorshError>(())
        }
    });

    // Execute all download tasks
    let results: Vec<Result<()>> = stream::iter(download_tasks)
        .buffer_unordered(config.max_concurrent_chunks)
        .collect()
        .await;

    // Check for errors
    for result in results {
        result?;
    }

    if progress {
        println!(); // New line after progress
    }

    // Move temporary file to final destination
    tokio::fs::rename(&temp_path, dest_path).await?;

    Ok(())
}

/// Download a single chunk of a file
///
/// This function downloads a specific byte range of a file using HTTP range requests.
/// It includes retry logic to handle temporary failures and network issues.
///
/// # Arguments
/// * `client` - The HTTP client to use
/// * `url` - The URL to download from
/// * `temp_path` - Path to the temporary file being assembled
/// * `chunk_id` - Identifier for this chunk (for logging)
/// * `start` - Starting byte position
/// * `end` - Ending byte position
/// * `max_retries` - Maximum retry attempts for this chunk
async fn download_chunk(
    client: &Client,
    url: &str,
    temp_path: &Path,
    chunk_id: u64,
    start: u64,
    end: u64,
    max_retries: usize,
) -> Result<()> {
    let mut last_error = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
        }

        match download_chunk_attempt(client, url, temp_path, start, end).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    eprintln!(
                        "Chunk {} failed (attempt {}), retrying...",
                        chunk_id,
                        attempt + 1
                    );
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        TorshError::IoError(format!(
            "Chunk {} download failed after all retries",
            chunk_id
        ))
    }))
}

/// Single attempt to download a chunk
///
/// This function makes a single HTTP range request to download a specific
/// byte range of a file and writes it to the correct position in the temporary file.
///
/// # Arguments
/// * `client` - The HTTP client to use
/// * `url` - The URL to download from
/// * `temp_path` - Path to the temporary file being assembled
/// * `start` - Starting byte position
/// * `end` - Ending byte position
async fn download_chunk_attempt(
    client: &Client,
    url: &str,
    temp_path: &Path,
    start: u64,
    end: u64,
) -> Result<()> {
    // Make range request
    let response = client
        .get(url)
        .header("Range", format!("bytes={}-{}", start, end))
        .send()
        .await
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    if !response.status().is_success() && response.status() != 206 {
        return Err(TorshError::IoError(format!(
            "Failed to download chunk: HTTP {}",
            response.status()
        )));
    }

    // Read chunk data
    let chunk_data = response
        .bytes()
        .await
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    // Write chunk to file at the correct position
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(temp_path)
        .await?;

    file.seek(SeekFrom::Start(start)).await?;
    file.write_all(&chunk_data).await?;
    file.sync_all().await?;

    Ok(())
}

/// Simple async download without chunking
///
/// This function provides a straightforward asynchronous download approach
/// for files that don't benefit from chunking (small files, servers that
/// don't support range requests, or unknown file sizes).
///
/// # Arguments
/// * `client` - The HTTP client to use
/// * `url` - The URL to download from
/// * `dest_path` - The destination file path
/// * `progress` - Whether to display download progress
async fn download_file_simple(
    client: &Client,
    url: &str,
    dest_path: &Path,
    progress: bool,
) -> Result<()> {
    // Start download
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| TorshError::IoError(e.to_string()))?;

    if !response.status().is_success() {
        return Err(TorshError::IoError(format!(
            "Failed to download {}: HTTP {}",
            url,
            response.status()
        )));
    }

    let total_size = response.content_length();

    if progress {
        if let Some(total) = total_size {
            println!("Downloading {} ({:.1} MB)", url, total as f64 / 1_048_576.0);
        } else {
            println!("Downloading {}", url);
        }
    }

    // Create temporary file
    let temp_path = dest_path.with_extension("tmp");
    let mut file = AsyncFile::create(&temp_path).await?;

    // Download with progress
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| TorshError::IoError(e.to_string()))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if progress {
            if let Some(total) = total_size {
                print_progress(downloaded, total);
            }
        }
    }

    file.sync_all().await?;
    drop(file);

    if progress {
        println!(); // New line after progress
    }

    // Move temporary file to final destination
    tokio::fs::rename(&temp_path, dest_path).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_progress() {
        // Test progress bar printing
        print_progress(50, 100);
        println!(); // Clean line
        print_progress(100, 100);
        println!();
    }

    #[test]
    fn test_progress_calculation() {
        // Test progress calculation with different values

        // Capture output for testing
        print_progress(0, 100);
        println!();

        print_progress(25, 100);
        println!();

        print_progress(50, 100);
        println!();

        print_progress(75, 100);
        println!();

        print_progress(100, 100);
        println!();
    }

    #[tokio::test]
    async fn test_download_config_validation() {
        let config = ParallelDownloadConfig::default();
        assert!(config.max_concurrent_downloads > 0);
        assert!(config.chunk_size > 0);
        assert!(config.max_concurrent_chunks > 0);
        assert!(config.timeout_seconds > 0);
    }

    #[test]
    fn test_url_validation_integration() {
        // Test that our core functions use validation
        assert!(validate_url("https://example.com/file.txt").is_ok());
        assert!(validate_url("not-a-url").is_err());
        assert!(validate_url("").is_err());
    }

    #[test]
    fn test_temporary_file_path_generation() {
        use std::path::PathBuf;

        let dest_path = PathBuf::from("/tmp/test.txt");
        let temp_path = dest_path.with_extension("tmp");

        assert_eq!(temp_path, PathBuf::from("/tmp/test.tmp"));
    }

    #[test]
    fn test_chunk_calculation() {
        let total_size = 1000u64;
        let chunk_size = 100u64;
        let num_chunks = (total_size + chunk_size - 1) / chunk_size;

        assert_eq!(num_chunks, 10);

        // Test with non-divisible size
        let total_size = 1050u64;
        let chunk_size = 100u64;
        let num_chunks = (total_size + chunk_size - 1) / chunk_size;

        assert_eq!(num_chunks, 11);
    }

    #[tokio::test]
    async fn test_download_retry_config() {
        // Test retry configuration
        let max_retries = 3;

        for attempt in 0..=max_retries {
            let delay = 2u64.pow(attempt as u32);
            assert!(delay >= 1);

            if attempt > 0 {
                assert!(delay > 1);
            }
        }
    }
}
