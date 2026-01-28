//! Performance optimization utilities for ToRSh Vision
//!
//! This module provides caching, prefetching, and performance monitoring
//! infrastructure to optimize image loading and processing operations.
//!
//! ## Key Components
//!
//! - [`ImageCache`]: LRU cache for images with automatic memory management
//! - [`ImagePrefetcher`]: Asynchronous image prefetching for improved loading performance
//! - [`BatchImageLoader`]: Optimized batch loading with caching and prefetching
//! - [`MemoryMappedLoader`]: Memory-mapped loading for large datasets
//! - [`LoadingMetrics`]: Performance monitoring and metrics collection
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use torsh_vision::utils::performance::{ImageCache, BatchImageLoader, LoadingMetrics};
//! use std::sync::Arc;
//!
//! // Create a cache with 100MB limit
//! let cache = Arc::new(ImageCache::new(100));
//!
//! // Load an image with caching
//! let image = cache.get_or_load("path/to/image.jpg")?;
//!
//! // Create a batch loader with optimizations
//! let batch_loader = BatchImageLoader::new(256)
//!     .with_target_size(224, 224)
//!     .with_normalization(true);
//!
//! let paths = vec!["img1.jpg", "img2.jpg", "img3.jpg"];
//! let tensors = batch_loader.load_batch(&paths)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Result, VisionError};
use image::{DynamicImage, GenericImageView};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// Import required functions from sibling modules
use super::image_conversion::image_to_tensor;
use super::image_processing::resize_image;
use torsh_tensor::Tensor;

/// Image cache entry with metadata for LRU eviction and performance tracking
///
/// Each cache entry stores the image data along with access tracking information
/// used for implementing least-recently-used (LRU) cache eviction policy.
#[derive(Clone)]
pub struct CacheEntry {
    /// The cached image data
    pub image: DynamicImage,
    /// Timestamp of the last access to this entry
    pub access_time: Instant,
    /// Number of times this entry has been accessed
    pub access_count: usize,
    /// Estimated memory usage of this entry in bytes
    pub size_bytes: usize,
}

/// LRU cache for images with automatic memory management
///
/// Provides thread-safe caching of loaded images with automatic eviction
/// based on memory usage. Implements least-recently-used (LRU) policy
/// for cache eviction when the memory limit is exceeded.
///
/// ## Example
///
/// ```rust,no_run
/// use torsh_vision::utils::performance::ImageCache;
/// use std::sync::Arc;
///
/// // Create cache with 100MB limit
/// let cache = Arc::new(ImageCache::new(100));
///
/// // Load images with automatic caching
/// let image1 = cache.get_or_load("image1.jpg")?;
/// let image2 = cache.get_or_load("image2.jpg")?;
///
/// // Check cache performance
/// let stats = cache.stats();
/// println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ImageCache {
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    max_size_bytes: usize,
    current_size_bytes: Arc<Mutex<usize>>,
    hit_count: Arc<Mutex<usize>>,
    miss_count: Arc<Mutex<usize>>,
}

impl ImageCache {
    /// Create a new image cache with specified maximum size in megabytes
    ///
    /// # Arguments
    /// * `max_size_mb` - Maximum cache size in megabytes
    ///
    /// # Example
    /// ```rust
    /// use torsh_vision::utils::performance::ImageCache;
    ///
    /// let cache = ImageCache::new(256); // 256MB cache
    /// ```
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size_bytes: max_size_mb * 1024 * 1024, // Convert MB to bytes
            current_size_bytes: Arc::new(Mutex::new(0)),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get image from cache or load if not present
    ///
    /// This method first checks if the image is already cached. If found,
    /// it updates the access time and count, then returns the cached image.
    /// If not found, it loads the image from disk and caches it.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// The loaded image, either from cache or freshly loaded
    ///
    /// # Example
    /// ```rust,no_run
    /// use torsh_vision::utils::performance::ImageCache;
    ///
    /// let cache = ImageCache::new(100);
    /// let image = cache.get_or_load("path/to/image.jpg")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_or_load<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Try to get from cache first
        {
            let mut cache = self.cache.lock().expect("lock should not be poisoned");
            if let Some(entry) = cache.get_mut(&path_str) {
                entry.access_time = Instant::now();
                entry.access_count += 1;
                *self.hit_count.lock().expect("lock should not be poisoned") += 1;
                return Ok(entry.image.clone());
            }
        }

        // Cache miss - load image
        *self.miss_count.lock().expect("lock should not be poisoned") += 1;
        let image = crate::io::global::load_image(path)?;

        // Estimate image size in bytes (approximation)
        let estimated_size = (image.width() * image.height() * 4) as usize; // RGBA approximation

        self.insert(path_str, image.clone(), estimated_size);
        Ok(image)
    }

    /// Insert image into cache with LRU eviction
    ///
    /// This is an internal method that handles cache insertion with automatic
    /// eviction of least-recently-used entries when the cache size limit is exceeded.
    fn insert(&self, key: String, image: DynamicImage, size_bytes: usize) {
        let entry = CacheEntry {
            image: image.clone(),
            access_time: Instant::now(),
            access_count: 1,
            size_bytes,
        };

        let mut cache = self.cache.lock().expect("lock should not be poisoned");
        let mut current_size = self
            .current_size_bytes
            .lock()
            .expect("lock should not be poisoned");

        // Remove old entry if exists
        if let Some(old_entry) = cache.remove(&key) {
            *current_size -= old_entry.size_bytes;
        }

        // Evict LRU entries if necessary
        while *current_size + size_bytes > self.max_size_bytes && !cache.is_empty() {
            let lru_key = cache
                .iter()
                .min_by_key(|(_, entry)| entry.access_time)
                .map(|(k, _)| k.clone());

            if let Some(lru_key) = lru_key {
                if let Some(lru_entry) = cache.remove(&lru_key) {
                    *current_size -= lru_entry.size_bytes;
                }
            } else {
                break;
            }
        }

        // Insert new entry
        cache.insert(key, entry);
        *current_size += size_bytes;
    }

    /// Get cache statistics
    ///
    /// Returns detailed statistics about cache performance including
    /// hit/miss counts, hit rate, and memory usage.
    ///
    /// # Returns
    /// [`CacheStats`] containing comprehensive cache performance metrics
    pub fn stats(&self) -> CacheStats {
        let hit_count = *self.hit_count.lock().expect("lock should not be poisoned");
        let miss_count = *self.miss_count.lock().expect("lock should not be poisoned");
        let total_requests = hit_count + miss_count;
        let hit_rate = if total_requests > 0 {
            hit_count as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStats {
            hit_count,
            miss_count,
            hit_rate,
            current_size_bytes: *self
                .current_size_bytes
                .lock()
                .expect("lock should not be poisoned"),
            max_size_bytes: self.max_size_bytes,
            entry_count: self
                .cache
                .lock()
                .expect("lock should not be poisoned")
                .len(),
        }
    }

    /// Clear all cached entries
    ///
    /// Removes all cached images and resets all statistics.
    /// This is useful for freeing memory or resetting cache state.
    pub fn clear(&self) {
        self.cache
            .lock()
            .expect("lock should not be poisoned")
            .clear();
        *self
            .current_size_bytes
            .lock()
            .expect("lock should not be poisoned") = 0;
        *self.hit_count.lock().expect("lock should not be poisoned") = 0;
        *self.miss_count.lock().expect("lock should not be poisoned") = 0;
    }
}

/// Cache statistics for performance monitoring
///
/// Provides detailed metrics about cache performance that can be used
/// for monitoring and optimization of cache configuration.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hit_count: usize,
    /// Number of cache misses
    pub miss_count: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Current memory usage in bytes
    pub current_size_bytes: usize,
    /// Maximum allowed memory usage in bytes
    pub max_size_bytes: usize,
    /// Number of entries currently in cache
    pub entry_count: usize,
}

/// Asynchronous image prefetcher for improved loading performance
///
/// Runs a background worker thread that preloads images into cache
/// before they are explicitly requested, reducing latency for
/// sequential access patterns.
///
/// ## Example
///
/// ```rust,no_run
/// use torsh_vision::utils::performance::{ImageCache, ImagePrefetcher};
/// use std::sync::Arc;
///
/// let cache = Arc::new(ImageCache::new(100));
/// let mut prefetcher = ImagePrefetcher::new(cache);
///
/// // Queue images for prefetching
/// let paths = vec!["img1.jpg", "img2.jpg", "img3.jpg"];
/// prefetcher.prefetch_paths(&paths);
///
/// // Images will be loaded in background
/// let image = prefetcher.get_image("img1.jpg")?; // Likely cache hit
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ImagePrefetcher {
    cache: Arc<ImageCache>,
    prefetch_queue: Arc<Mutex<Vec<String>>>,
    worker_handle: Option<thread::JoinHandle<()>>,
    shutdown_signal: Arc<Mutex<bool>>,
}

impl ImagePrefetcher {
    /// Create a new image prefetcher with cache
    ///
    /// Starts a background worker thread that continuously processes
    /// the prefetch queue to load images into the cache.
    ///
    /// # Arguments
    /// * `cache` - Shared cache instance to use for prefetching
    pub fn new(cache: Arc<ImageCache>) -> Self {
        let prefetch_queue = Arc::new(Mutex::new(Vec::new()));
        let shutdown_signal = Arc::new(Mutex::new(false));

        let queue_clone = Arc::clone(&prefetch_queue);
        let cache_clone = Arc::clone(&cache);
        let shutdown_clone = Arc::clone(&shutdown_signal);

        let worker_handle = thread::spawn(move || {
            Self::worker_thread(queue_clone, cache_clone, shutdown_clone);
        });

        Self {
            cache,
            prefetch_queue,
            worker_handle: Some(worker_handle),
            shutdown_signal,
        }
    }

    /// Add paths to prefetch queue
    ///
    /// Queues the specified image paths for background loading.
    /// The worker thread will process these paths and load the
    /// images into cache.
    ///
    /// # Arguments
    /// * `paths` - Iterator of image file paths to prefetch
    pub fn prefetch_paths<P: AsRef<Path>>(&self, paths: &[P]) {
        let mut queue = self
            .prefetch_queue
            .lock()
            .expect("lock should not be poisoned");
        for path in paths {
            queue.push(path.as_ref().to_string_lossy().to_string());
        }
    }

    /// Worker thread for background prefetching
    ///
    /// Continuously processes the prefetch queue, loading images
    /// into cache in the background. Runs until shutdown signal
    /// is received.
    fn worker_thread(
        queue: Arc<Mutex<Vec<String>>>,
        cache: Arc<ImageCache>,
        shutdown: Arc<Mutex<bool>>,
    ) {
        loop {
            // Check for shutdown signal
            if *shutdown.lock().expect("lock should not be poisoned") {
                break;
            }

            // Get next path to prefetch
            let path = {
                let mut queue_guard = queue.lock().expect("lock should not be poisoned");
                queue_guard.pop()
            };

            if let Some(path) = path {
                // Prefetch image (ignore errors for background loading)
                if let Err(_) = cache.get_or_load(&path) {
                    // Log error in production, ignore for now
                }
            } else {
                // No work to do, sleep briefly
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    /// Get image with prefetching optimization
    ///
    /// Retrieves an image from cache, benefiting from any previous
    /// prefetching operations that may have loaded it in the background.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// The loaded image from cache or disk
    pub fn get_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        self.cache.get_or_load(path)
    }

    /// Shutdown the prefetcher and wait for worker thread
    ///
    /// Signals the worker thread to stop and waits for it to finish.
    /// This ensures clean shutdown of background operations.
    pub fn shutdown(&mut self) {
        *self
            .shutdown_signal
            .lock()
            .expect("lock should not be poisoned") = true;
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ImagePrefetcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Batch image loader with optimized memory usage
///
/// Provides efficient batch loading of images with automatic caching,
/// prefetching, resizing, and normalization. Designed for machine learning
/// workflows that process images in batches.
///
/// ## Example
///
/// ```rust,no_run
/// use torsh_vision::utils::performance::BatchImageLoader;
///
/// let loader = BatchImageLoader::new(256)  // 256MB cache
///     .with_target_size(224, 224)          // Auto-resize to 224x224
///     .with_normalization(true);           // Normalize to [0,1]
///
/// let paths = vec!["img1.jpg", "img2.jpg", "img3.jpg"];
/// let tensors = loader.load_batch(&paths)?;
///
/// println!("Loaded {} tensors", tensors.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BatchImageLoader {
    cache: Arc<ImageCache>,
    prefetcher: ImagePrefetcher,
    target_size: Option<(u32, u32)>,
    normalize: bool,
}

impl BatchImageLoader {
    /// Create new batch loader with caching and prefetching
    ///
    /// # Arguments
    /// * `cache_size_mb` - Size of image cache in megabytes
    pub fn new(cache_size_mb: usize) -> Self {
        let cache = Arc::new(ImageCache::new(cache_size_mb));
        let prefetcher = ImagePrefetcher::new(Arc::clone(&cache));

        Self {
            cache,
            prefetcher,
            target_size: None,
            normalize: false,
        }
    }

    /// Set target size for automatic resizing
    ///
    /// When set, all loaded images will be automatically resized
    /// to the specified dimensions using Lanczos3 interpolation.
    ///
    /// # Arguments
    /// * `width` - Target width in pixels
    /// * `height` - Target height in pixels
    pub fn with_target_size(mut self, width: u32, height: u32) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Enable automatic normalization
    ///
    /// When enabled, pixel values will be normalized from \[0,255\]
    /// to \[0,1\] range by dividing by 255.0.
    ///
    /// # Arguments
    /// * `normalize` - Whether to enable normalization
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Load a batch of images efficiently
    ///
    /// Loads multiple images with automatic caching, prefetching,
    /// resizing, and normalization as configured. Implements smart
    /// prefetching heuristics for improved performance.
    ///
    /// # Arguments
    /// * `paths` - Slice of image file paths to load
    ///
    /// # Returns
    /// Vector of loaded and processed image tensors
    ///
    /// # Example
    /// ```rust,no_run
    /// use torsh_vision::utils::performance::BatchImageLoader;
    ///
    /// let loader = BatchImageLoader::new(128);
    /// let paths = vec!["img1.jpg", "img2.jpg"];
    /// let tensors = loader.load_batch(&paths)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load_batch<P: AsRef<Path>>(&self, paths: &[P]) -> Result<Vec<Tensor<f32>>> {
        // Start prefetching for future batches (next batch heuristic)
        if paths.len() > 1 {
            let prefetch_paths: Vec<String> = paths
                .iter()
                .skip(1)
                .map(|p| p.as_ref().to_string_lossy().to_string())
                .collect();
            self.prefetcher.prefetch_paths(
                &prefetch_paths
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            );
        }

        let mut tensors = Vec::with_capacity(paths.len());

        for path in paths {
            let mut image = self.prefetcher.get_image(path)?;

            // Apply target size if specified
            if let Some((width, height)) = self.target_size {
                image = resize_image(&image, width, height, image::imageops::FilterType::Lanczos3);
            }

            // Convert to tensor
            let tensor = image_to_tensor(&image)?;

            // Apply normalization if requested
            let final_tensor = if self.normalize {
                let mut normalized = tensor.clone();
                normalized.div_scalar_(255.0)?;
                normalized
            } else {
                tensor
            };

            tensors.push(final_tensor);
        }

        Ok(tensors)
    }

    /// Get cache statistics
    ///
    /// Returns performance statistics for the underlying cache,
    /// useful for monitoring and optimization.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    ///
    /// Removes all cached images to free memory.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

/// Memory-mapped image loader for very large datasets
///
/// Provides memory-mapped file access for loading images from very large
/// datasets where memory usage needs to be carefully controlled. Uses
/// memory mapping to avoid loading entire files into memory at once.
///
/// ## Example
///
/// ```rust,no_run
/// use torsh_vision::utils::performance::MemoryMappedLoader;
///
/// let mut loader = MemoryMappedLoader::new();
/// let image = loader.load_image_mmap("large_image.jpg")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct MemoryMappedLoader {
    file_handles: HashMap<String, std::fs::File>,
    mmap_cache: HashMap<String, memmap2::Mmap>,
}

impl MemoryMappedLoader {
    /// Create a new memory-mapped loader
    pub fn new() -> Self {
        Self {
            file_handles: HashMap::new(),
            mmap_cache: HashMap::new(),
        }
    }

    /// Load image using memory mapping for large files
    ///
    /// Uses memory mapping to load images without reading the entire
    /// file into memory at once. Particularly useful for very large
    /// image files or when memory usage needs to be minimized.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// The loaded image decoded from memory-mapped data
    pub fn load_image_mmap<P: AsRef<Path>>(&mut self, path: P) -> Result<DynamicImage> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Check if already memory-mapped
        if let Some(mmap) = self.mmap_cache.get(&path_str) {
            return self.decode_from_mmap(mmap);
        }

        // Open file and create memory map
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let image = self.decode_from_mmap(&mmap)?;

        // Cache the memory map for future use
        self.file_handles.insert(path_str.clone(), file);
        self.mmap_cache.insert(path_str, mmap);

        Ok(image)
    }

    /// Decode image from memory-mapped data
    fn decode_from_mmap(&self, mmap: &memmap2::Mmap) -> Result<DynamicImage> {
        let cursor = std::io::Cursor::new(&mmap[..]);
        let image = image::load(
            cursor,
            image::ImageFormat::from_path("dummy.jpg").unwrap_or(image::ImageFormat::Jpeg),
        )?;
        Ok(image)
    }

    /// Clear all memory maps
    ///
    /// Releases all memory mappings and file handles to free resources.
    pub fn clear(&mut self) {
        self.mmap_cache.clear();
        self.file_handles.clear();
    }
}

impl Default for MemoryMappedLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance monitoring for data loading operations
///
/// Tracks detailed metrics about image loading performance including
/// timing, cache hit rates, and throughput statistics.
///
/// ## Example
///
/// ```rust
/// use torsh_vision::utils::performance::LoadingMetrics;
/// use std::time::{Duration, Instant};
///
/// let mut metrics = LoadingMetrics::default();
///
/// let start = Instant::now();
/// // ... load image ...
/// let duration = start.elapsed();
///
/// metrics.record_load(duration, true); // Cache hit
/// println!("Hit rate: {:.2}%", metrics.cache_hit_rate() * 100.0);
/// ```
#[derive(Debug, Default)]
pub struct LoadingMetrics {
    /// Total number of images loaded
    pub total_images_loaded: usize,
    /// Cumulative time spent loading images
    pub total_loading_time: Duration,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Average time per image load
    pub average_loading_time: Duration,
}

impl LoadingMetrics {
    /// Record a load operation with timing and cache hit information
    ///
    /// Updates all relevant metrics based on the load operation results.
    ///
    /// # Arguments
    /// * `duration` - Time taken for the load operation
    /// * `cache_hit` - Whether this was a cache hit or miss
    pub fn record_load(&mut self, duration: Duration, cache_hit: bool) {
        self.total_images_loaded += 1;
        self.total_loading_time += duration;

        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        self.average_loading_time = self.total_loading_time / self.total_images_loaded as u32;
    }

    /// Calculate cache hit rate as a percentage
    ///
    /// # Returns
    /// Cache hit rate as a value between 0.0 and 1.0
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_images_loaded == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_images_loaded as f64
        }
    }

    /// Get throughput in images per second
    ///
    /// # Returns
    /// Average throughput based on total images and time
    pub fn throughput_ips(&self) -> f64 {
        if self.total_loading_time.is_zero() {
            0.0
        } else {
            self.total_images_loaded as f64 / self.total_loading_time.as_secs_f64()
        }
    }

    /// Reset all metrics to initial state
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
