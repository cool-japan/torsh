//! Memory-mapped storage for large tensors with lazy loading
//!
//! This module provides efficient memory-mapped file storage for large tensors,
//! with support for lazy loading, page-based caching, and access pattern optimization.

use crate::dtype::TensorElement;
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Memory-mapped storage for large tensors with lazy loading
///
/// MappedStorage provides efficient access to large tensor data stored in files
/// through memory mapping and intelligent caching strategies. It supports both
/// full file mapping and page-based lazy loading depending on access patterns.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::MappedStorage;
///
/// // Create mapped storage for 1M f32 elements
/// let config = LazyLoadConfig::default();
/// let storage = MappedStorage::<f32>::new("data.bin", 1_000_000, config)?;
///
/// // Get a slice of data (will be loaded on demand)
/// let slice = storage.get_slice(1000, 100)?;
/// ```
#[derive(Debug)]
pub struct MappedStorage<T: TensorElement> {
    /// Memory-mapped file
    mmap: Arc<parking_lot::Mutex<Option<memmap2::Mmap>>>,
    /// File path for the backing storage
    file_path: PathBuf,
    /// Total number of elements
    total_elements: usize,
    /// Element size in bytes
    element_size: usize,
    /// Page size for lazy loading
    page_size: usize,
    /// Currently loaded pages (page_index -> data)
    loaded_pages: Arc<parking_lot::RwLock<HashMap<usize, Arc<Vec<T>>>>>,
    /// Page access pattern tracking for prefetching
    access_pattern: Arc<parking_lot::Mutex<AccessPatternTracker>>,
    /// Lazy loading configuration
    config: LazyLoadConfig,
    /// Phantom data for type safety
    _phantom: std::marker::PhantomData<T>,
}

impl<T: TensorElement> MappedStorage<T> {
    /// Create a new memory-mapped storage
    ///
    /// # Arguments
    /// * `file_path` - Path to the backing file
    /// * `total_elements` - Total number of elements in the storage
    /// * `config` - Configuration for lazy loading behavior
    ///
    /// # Returns
    /// A new MappedStorage instance or an error if file operations fail
    pub fn new<P: AsRef<std::path::Path>>(
        file_path: P,
        total_elements: usize,
        config: LazyLoadConfig,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let element_size = std::mem::size_of::<T>();
        let total_size = total_elements * element_size;

        // Create file if it doesn't exist
        if !file_path.exists() {
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    crate::error::TorshError::IoError(format!("Failed to create directory: {e}"))
                })?;
            }

            // Create file with the required size
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&file_path)
                .map_err(|e| {
                    crate::error::TorshError::IoError(format!("Failed to create file: {e}"))
                })?;

            file.set_len(total_size as u64).map_err(|e| {
                crate::error::TorshError::IoError(format!("Failed to set file size: {e}"))
            })?;
        }

        // Calculate optimal page size
        let page_size = config.page_size.unwrap_or_else(|| {
            let system_page_size = 4096; // 4KB default
            let elements_per_page = system_page_size / element_size;
            std::cmp::max(1, elements_per_page) * element_size
        });

        Ok(Self {
            mmap: Arc::new(parking_lot::Mutex::new(None)),
            file_path,
            total_elements,
            element_size,
            page_size,
            loaded_pages: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            access_pattern: Arc::new(parking_lot::Mutex::new(AccessPatternTracker::new())),
            config,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Load the entire file into memory (disable lazy loading)
    ///
    /// This method creates a full memory mapping of the file, which can be
    /// more efficient for workloads that access most of the data.
    pub fn load_all(&self) -> Result<()> {
        let mut mmap_guard = self.mmap.lock();
        if mmap_guard.is_none() {
            let file = std::fs::File::open(&self.file_path).map_err(|e| {
                crate::error::TorshError::IoError(format!("Failed to open file: {e}"))
            })?;

            let mmap = unsafe {
                memmap2::Mmap::map(&file).map_err(|e| {
                    crate::error::TorshError::IoError(format!("Failed to map file: {e}"))
                })?
            };

            *mmap_guard = Some(mmap);
        }
        Ok(())
    }

    /// Get a slice of data at the specified element range
    ///
    /// # Arguments
    /// * `start` - Starting element index
    /// * `len` - Number of elements to read
    ///
    /// # Returns
    /// A MappedSlice containing the requested data
    pub fn get_slice(&self, start: usize, len: usize) -> Result<MappedSlice<'_, T>> {
        if start + len > self.total_elements {
            return Err(crate::error::TorshError::IndexOutOfBounds {
                index: start + len,
                size: self.total_elements,
            });
        }

        // Update access pattern
        {
            let mut pattern = self.access_pattern.lock();
            pattern.record_access(start, len);
        }

        // Check if we should use full mapping or lazy loading
        // Use lazy loading when access size is small or threshold is 0 (force lazy loading)
        if self.config.force_full_load
            || (self.config.lazy_threshold > 0
                && (len * self.element_size) >= self.config.lazy_threshold)
        {
            self.load_all()?;
            let mmap_guard = self.mmap.lock();
            if let Some(ref mmap) = *mmap_guard {
                let ptr = mmap.as_ptr() as *const T;
                let slice = unsafe { std::slice::from_raw_parts(ptr.add(start), len) };
                return Ok(MappedSlice::FullMap {
                    slice,
                    _lifetime: std::marker::PhantomData,
                });
            }
        }

        // Use lazy loading
        let pages_needed = self.get_pages_for_range(start, len);
        let mut loaded_data = Vec::new();

        for page_idx in pages_needed {
            let page_data = self.load_page(page_idx)?;
            loaded_data.push(page_data);
        }

        Ok(MappedSlice::LazyLoaded {
            data: loaded_data,
            start_offset: start % (self.page_size / self.element_size),
            len,
        })
    }

    /// Get the pages needed for a specific element range
    fn get_pages_for_range(&self, start: usize, len: usize) -> Vec<usize> {
        let elements_per_page = self.page_size / self.element_size;
        let start_page = start / elements_per_page;
        let end_page = (start + len - 1) / elements_per_page;

        (start_page..=end_page).collect()
    }

    /// Load a specific page of data
    fn load_page(&self, page_idx: usize) -> Result<Arc<Vec<T>>> {
        // Check if page is already loaded
        {
            let loaded_pages = self.loaded_pages.read();
            if let Some(page_data) = loaded_pages.get(&page_idx) {
                return Ok(page_data.clone());
            }
        }

        // Load page from file
        let elements_per_page = self.page_size / self.element_size;
        let start_element = page_idx * elements_per_page;
        let page_elements = std::cmp::min(elements_per_page, self.total_elements - start_element);

        let file = std::fs::File::open(&self.file_path)
            .map_err(|e| crate::error::TorshError::IoError(format!("Failed to open file: {e}")))?;

        let offset = start_element * self.element_size;
        let size = page_elements * self.element_size;

        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset as u64)
                .len(size)
                .map(&file)
                .map_err(|e| {
                    crate::error::TorshError::IoError(format!("Failed to map page: {e}"))
                })?
        };

        // Convert mmap to owned data
        let ptr = mmap.as_ptr() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, page_elements) };
        let data = slice.to_vec();
        let arc_data = Arc::new(data);

        // Cache the page
        {
            let mut loaded_pages = self.loaded_pages.write();

            // Check cache size limit
            if loaded_pages.len() >= self.config.max_cached_pages {
                // Remove least recently used page
                // For simplicity, remove the first page
                if let Some(first_key) = loaded_pages.keys().next().copied() {
                    loaded_pages.remove(&first_key);
                }
            }

            loaded_pages.insert(page_idx, arc_data.clone());
        }

        // Trigger prefetching if enabled
        if self.config.enable_prefetch {
            self.prefetch_pages(page_idx);
        }

        Ok(arc_data)
    }

    /// Prefetch nearby pages based on access patterns
    fn prefetch_pages(&self, current_page: usize) {
        let pattern = self.access_pattern.lock();
        let prefetch_pages =
            pattern.predict_next_pages(current_page, self.config.prefetch_distance);
        drop(pattern);

        // Spawn background task for prefetching
        let storage_weak = Arc::downgrade(&self.loaded_pages);
        let file_path = self.file_path.clone();
        let page_size = self.page_size;
        let element_size = self.element_size;
        let total_elements = self.total_elements;

        std::thread::spawn(move || {
            for page_idx in prefetch_pages {
                // Check if storage still exists
                if storage_weak.upgrade().is_none() {
                    break;
                }

                // Simplified prefetch logic (full implementation would be more sophisticated)
                let _ = Self::load_page_static(
                    page_idx,
                    &file_path,
                    page_size,
                    element_size,
                    total_elements,
                );
            }
        });
    }

    /// Static method for loading pages (used in prefetching)
    fn load_page_static(
        page_idx: usize,
        file_path: &std::path::Path,
        page_size: usize,
        element_size: usize,
        total_elements: usize,
    ) -> Result<Vec<T>> {
        let elements_per_page = page_size / element_size;
        let start_element = page_idx * elements_per_page;
        let page_elements = std::cmp::min(elements_per_page, total_elements - start_element);

        let file = std::fs::File::open(file_path)
            .map_err(|e| crate::error::TorshError::IoError(format!("Failed to open file: {e}")))?;

        let offset = start_element * element_size;
        let size = page_elements * element_size;

        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset as u64)
                .len(size)
                .map(&file)
                .map_err(|e| {
                    crate::error::TorshError::IoError(format!("Failed to map page: {e}"))
                })?
        };

        let ptr = mmap.as_ptr() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, page_elements) };
        Ok(slice.to_vec())
    }

    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> MappedStorageStats {
        let loaded_pages = self.loaded_pages.read();
        let total_loaded_elements = loaded_pages.len() * (self.page_size / self.element_size);
        let memory_usage = total_loaded_elements * self.element_size;

        MappedStorageStats {
            total_elements: self.total_elements,
            loaded_elements: total_loaded_elements,
            cached_pages: loaded_pages.len(),
            memory_usage,
            file_size: self.total_elements * self.element_size,
            cache_hit_ratio: self.access_pattern.lock().cache_hit_ratio(),
        }
    }

    /// Clear cached pages
    pub fn clear_cache(&self) {
        let mut loaded_pages = self.loaded_pages.write();
        loaded_pages.clear();
    }

    /// Write data to a specific range
    ///
    /// # Arguments
    /// * `start` - Starting element index
    /// * `data` - Data to write
    ///
    /// # Returns
    /// Success or error from the write operation
    pub fn write_slice(&self, start: usize, data: &[T]) -> Result<()> {
        if start + data.len() > self.total_elements {
            return Err(crate::error::TorshError::IndexOutOfBounds {
                index: start + data.len(),
                size: self.total_elements,
            });
        }

        // Open file for writing
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.file_path)
            .map_err(|e| {
                crate::error::TorshError::IoError(format!("Failed to open file for writing: {e}"))
            })?;

        use std::io::{Seek, Write};

        let offset = start * self.element_size;
        file.seek(std::io::SeekFrom::Start(offset as u64))
            .map_err(|e| crate::error::TorshError::IoError(format!("Failed to seek: {e}")))?;

        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * self.element_size)
        };

        file.write_all(bytes)
            .map_err(|e| crate::error::TorshError::IoError(format!("Failed to write: {e}")))?;

        file.sync_all()
            .map_err(|e| crate::error::TorshError::IoError(format!("Failed to sync: {e}")))?;

        // Invalidate affected pages in cache
        let pages_affected = self.get_pages_for_range(start, data.len());
        {
            let mut loaded_pages = self.loaded_pages.write();
            for page_idx in pages_affected {
                loaded_pages.remove(&page_idx);
            }
        }

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &LazyLoadConfig {
        &self.config
    }

    /// Get file path
    pub fn file_path(&self) -> &std::path::Path {
        &self.file_path
    }

    /// Get total number of elements
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Get element size in bytes
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Get page size in bytes
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Check if the entire file is currently mapped
    pub fn is_fully_mapped(&self) -> bool {
        self.mmap.lock().is_some()
    }

    /// Get the number of currently cached pages
    pub fn cached_pages_count(&self) -> usize {
        self.loaded_pages.read().len()
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        // If fully mapped, sync the mapping
        if let Some(ref _mmap) = *self.mmap.lock() {
            // Note: memmap2::Mmap doesn't have a flush method
            // For memory-mapped files, the OS handles flushing automatically
            // If explicit flushing is needed, it would be done at the file level
        }
        Ok(())
    }
}

/// Configuration for lazy loading behavior
///
/// This structure controls how the memory-mapped storage behaves with respect
/// to caching, prefetching, and memory usage.
#[derive(Debug, Clone)]
pub struct LazyLoadConfig {
    /// Size of each page in bytes (None for auto-detection)
    pub page_size: Option<usize>,
    /// Maximum number of pages to keep in memory
    pub max_cached_pages: usize,
    /// Threshold size for switching to lazy loading (in bytes)
    pub lazy_threshold: usize,
    /// Force full file loading (disable lazy loading)
    pub force_full_load: bool,
    /// Enable prefetching of likely-to-be-accessed pages
    pub enable_prefetch: bool,
    /// Number of pages to prefetch ahead
    pub prefetch_distance: usize,
}

impl Default for LazyLoadConfig {
    fn default() -> Self {
        Self {
            page_size: None, // Auto-detect
            max_cached_pages: 100,
            lazy_threshold: 1024 * 1024, // 1MB
            force_full_load: false,
            enable_prefetch: true,
            prefetch_distance: 2,
        }
    }
}

impl LazyLoadConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set page size
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = Some(page_size);
        self
    }

    /// Set maximum cached pages
    pub fn with_max_cached_pages(mut self, max_pages: usize) -> Self {
        self.max_cached_pages = max_pages;
        self
    }

    /// Set lazy loading threshold
    pub fn with_lazy_threshold(mut self, threshold: usize) -> Self {
        self.lazy_threshold = threshold;
        self
    }

    /// Force full loading
    pub fn with_full_load(mut self, force: bool) -> Self {
        self.force_full_load = force;
        self
    }

    /// Enable or disable prefetching
    pub fn with_prefetch(mut self, enable: bool) -> Self {
        self.enable_prefetch = enable;
        self
    }

    /// Set prefetch distance
    pub fn with_prefetch_distance(mut self, distance: usize) -> Self {
        self.prefetch_distance = distance;
        self
    }
}

/// Access pattern tracker for predicting future memory accesses
///
/// This internal structure tracks access patterns to optimize prefetching
/// and caching decisions.
#[derive(Debug)]
struct AccessPatternTracker {
    /// Recent access history (element_start, length, timestamp)
    recent_accesses: VecDeque<(usize, usize, Instant)>,
    /// Total accesses
    total_accesses: usize,
    /// Cache hits
    cache_hits: usize,
    /// Sequential access detection
    last_access_end: Option<usize>,
    /// Stride pattern detection
    detected_stride: Option<usize>,
}

impl AccessPatternTracker {
    fn new() -> Self {
        Self {
            recent_accesses: VecDeque::new(),
            total_accesses: 0,
            cache_hits: 0,
            last_access_end: None,
            detected_stride: None,
        }
    }

    fn record_access(&mut self, start: usize, len: usize) {
        let now = Instant::now();

        // Add to recent accesses
        self.recent_accesses.push_back((start, len, now));

        // Keep only recent accesses (last 100)
        while self.recent_accesses.len() > 100 {
            self.recent_accesses.pop_front();
        }

        self.total_accesses += 1;

        // Detect sequential access pattern
        if let Some(last_end) = self.last_access_end {
            if start == last_end {
                // Sequential access detected
            } else if start > last_end {
                // Potential stride pattern
                let stride = start - last_end;
                if self.detected_stride == Some(stride) || self.detected_stride.is_none() {
                    self.detected_stride = Some(stride);
                }
            }
        }

        self.last_access_end = Some(start + len);
    }

    fn predict_next_pages(&self, current_page: usize, distance: usize) -> Vec<usize> {
        let mut predicted = Vec::new();

        // Simple prediction: next sequential pages
        for i in 1..=distance {
            predicted.push(current_page + i);
        }

        // If stride pattern detected, also predict based on stride
        if let Some(stride) = self.detected_stride {
            let elements_per_page = 1024; // Simplified
            let stride_pages = stride / elements_per_page;
            if stride_pages > 0 {
                for i in 1..=distance {
                    predicted.push(current_page + i * stride_pages);
                }
            }
        }

        predicted
    }

    fn cache_hit_ratio(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_accesses as f64
        }
    }

    /// Record a cache hit
    #[allow(dead_code)] // Cache hit recording - future implementation
    fn record_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Get access pattern statistics
    #[allow(dead_code)] // Access pattern statistics - future implementation
    fn statistics(&self) -> AccessPatternStats {
        AccessPatternStats {
            total_accesses: self.total_accesses,
            cache_hits: self.cache_hits,
            cache_hit_ratio: self.cache_hit_ratio(),
            detected_stride: self.detected_stride,
            recent_access_count: self.recent_accesses.len(),
            is_sequential: self.detected_stride == Some(0) || self.is_mostly_sequential(),
        }
    }

    /// Check if accesses are mostly sequential
    #[allow(dead_code)] // Sequential access detection - future implementation
    fn is_mostly_sequential(&self) -> bool {
        if self.recent_accesses.len() < 2 {
            return false;
        }

        let mut sequential_count = 0;
        let mut total_transitions = 0;

        for window in self.recent_accesses.iter().collect::<Vec<_>>().windows(2) {
            let (start1, len1, _) = *window[0];
            let (start2, _, _) = *window[1];

            total_transitions += 1;
            if start2 == start1 + len1 {
                sequential_count += 1;
            }
        }

        if total_transitions > 0 {
            sequential_count as f64 / total_transitions as f64 > 0.7
        } else {
            false
        }
    }
}

/// Slice of memory-mapped data
///
/// This enum represents different ways to access mapped data, either through
/// a full memory mapping or through lazy-loaded pages.
pub enum MappedSlice<'a, T: TensorElement> {
    /// Full memory mapping (entire file is mapped)
    FullMap {
        slice: &'a [T],
        _lifetime: std::marker::PhantomData<&'a ()>,
    },
    /// Lazy loaded data (specific pages loaded)
    LazyLoaded {
        data: Vec<Arc<Vec<T>>>,
        start_offset: usize,
        len: usize,
    },
}

impl<'a, T: TensorElement> MappedSlice<'a, T> {
    /// Get the length of the slice
    pub fn len(&self) -> usize {
        match self {
            MappedSlice::FullMap { slice, .. } => slice.len(),
            MappedSlice::LazyLoaded { len, .. } => *len,
        }
    }

    /// Check if the slice is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index (zero-copy when possible)
    pub fn get(&self, index: usize) -> Option<&T> {
        match self {
            MappedSlice::FullMap { slice, .. } => slice.get(index),
            MappedSlice::LazyLoaded {
                data,
                start_offset,
                len,
            } => {
                if index >= *len {
                    return None;
                }

                let global_index = start_offset + index;
                let elements_per_page = if !data.is_empty() {
                    data[0].len()
                } else {
                    return None;
                };

                let page_idx = global_index / elements_per_page;
                let element_idx = global_index % elements_per_page;

                data.get(page_idx)?.get(element_idx)
            }
        }
    }

    /// Convert to owned vector (copies data)
    pub fn to_vec(&self) -> Vec<T> {
        match self {
            MappedSlice::FullMap { slice, .. } => slice.to_vec(),
            MappedSlice::LazyLoaded {
                data: _,
                start_offset: _,
                len,
            } => {
                let mut result = Vec::with_capacity(*len);

                for i in 0..*len {
                    if let Some(element) = self.get(i) {
                        result.push(*element);
                    }
                }

                result
            }
        }
    }

    /// Check if this slice uses full mapping
    pub fn is_full_map(&self) -> bool {
        matches!(self, MappedSlice::FullMap { .. })
    }

    /// Check if this slice uses lazy loading
    pub fn is_lazy_loaded(&self) -> bool {
        matches!(self, MappedSlice::LazyLoaded { .. })
    }

    /// Get iterator over the elements
    pub fn iter(&self) -> MappedSliceIter<'_, T> {
        MappedSliceIter {
            slice: self,
            index: 0,
        }
    }
}

/// Iterator over MappedSlice elements
pub struct MappedSliceIter<'a, T: TensorElement> {
    slice: &'a MappedSlice<'a, T>,
    index: usize,
}

impl<'a, T: TensorElement> Iterator for MappedSliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.slice.get(self.index);
        if result.is_some() {
            self.index += 1;
        }
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T: TensorElement> ExactSizeIterator for MappedSliceIter<'a, T> {}

/// Statistics for memory-mapped storage
#[derive(Debug, Clone)]
pub struct MappedStorageStats {
    /// Total elements in the storage
    pub total_elements: usize,
    /// Currently loaded elements in memory
    pub loaded_elements: usize,
    /// Number of cached pages
    pub cached_pages: usize,
    /// Current memory usage in bytes
    pub memory_usage: usize,
    /// Total file size in bytes
    pub file_size: usize,
    /// Cache hit ratio (0.0 to 1.0)
    pub cache_hit_ratio: f64,
}

impl MappedStorageStats {
    /// Get memory efficiency ratio (loaded vs total)
    pub fn memory_efficiency(&self) -> f64 {
        if self.file_size > 0 {
            self.memory_usage as f64 / self.file_size as f64
        } else {
            0.0
        }
    }

    /// Get load ratio (loaded vs total elements)
    pub fn load_ratio(&self) -> f64 {
        if self.total_elements > 0 {
            self.loaded_elements as f64 / self.total_elements as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for MappedStorageStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MappedStats(loaded={}/{} elements, pages={}, memory={:.1}MB, hit_ratio={:.1}%)",
            self.loaded_elements,
            self.total_elements,
            self.cached_pages,
            self.memory_usage as f64 / (1024.0 * 1024.0),
            self.cache_hit_ratio * 100.0
        )
    }
}

/// Statistics for access patterns
#[derive(Debug, Clone)]
pub struct AccessPatternStats {
    /// Total number of accesses
    pub total_accesses: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Detected stride pattern
    pub detected_stride: Option<usize>,
    /// Number of recent accesses tracked
    pub recent_access_count: usize,
    /// Whether access pattern is mostly sequential
    pub is_sequential: bool,
}

impl std::fmt::Display for AccessPatternStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AccessPattern(accesses={}, hit_ratio={:.1}%, sequential={}, stride={:?})",
            self.total_accesses,
            self.cache_hit_ratio * 100.0,
            self.is_sequential,
            self.detected_stride
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file<T: TensorElement>(data: &[T]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        file.write_all(bytes).expect("Failed to write test data");
        file.flush().expect("Failed to flush");
        file
    }

    #[test]
    fn test_mapped_storage_creation() {
        let config = LazyLoadConfig::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        let storage = MappedStorage::<f32>::new(&file_path, 100, config).unwrap();

        assert_eq!(storage.total_elements(), 100);
        assert_eq!(storage.element_size(), 4);
        assert!(file_path.exists());
    }

    #[test]
    fn test_mapped_storage_read_write() {
        let config = LazyLoadConfig::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        let storage = MappedStorage::<f32>::new(&file_path, 10, config).unwrap();

        // Write some data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        storage.write_slice(0, &data).unwrap();

        // Read it back
        let slice = storage.get_slice(0, 5).unwrap();
        let read_data = slice.to_vec();

        assert_eq!(read_data, data);
    }

    #[test]
    fn test_lazy_load_config() {
        let config = LazyLoadConfig::new()
            .with_page_size(4096)
            .with_max_cached_pages(50)
            .with_lazy_threshold(512 * 1024)
            .with_prefetch(false);

        assert_eq!(config.page_size, Some(4096));
        assert_eq!(config.max_cached_pages, 50);
        assert_eq!(config.lazy_threshold, 512 * 1024);
        assert!(!config.enable_prefetch);
    }

    #[test]
    fn test_mapped_slice_iteration() {
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let _temp_file = create_test_file(&test_data);

        // Create storage and get slice
        let config = LazyLoadConfig::default().with_full_load(true);
        let storage = MappedStorage::<f32>::new(_temp_file.path(), 5, config).unwrap();
        let slice = storage.get_slice(0, 5).unwrap();

        // Test iteration
        let collected: Vec<f32> = slice.iter().copied().collect();
        assert_eq!(collected, test_data);

        // Test indexing
        assert_eq!(slice.get(0), Some(&1.0));
        assert_eq!(slice.get(4), Some(&5.0));
        assert_eq!(slice.get(5), None);
    }

    #[test]
    fn test_memory_stats() {
        let config = LazyLoadConfig::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        let storage = MappedStorage::<f32>::new(&file_path, 1000, config).unwrap();

        // Initially no pages loaded
        let stats = storage.memory_stats();
        assert_eq!(stats.cached_pages, 0);
        assert_eq!(stats.total_elements, 1000);

        // Load some data
        let _slice = storage.get_slice(0, 100).unwrap();

        // Should have some pages loaded now
        let stats = storage.memory_stats();
        assert!(stats.cached_pages > 0);
    }

    #[test]
    fn test_access_pattern_tracker() {
        let mut tracker = AccessPatternTracker::new();

        // Sequential accesses
        tracker.record_access(0, 10);
        tracker.record_access(10, 10);
        tracker.record_access(20, 10);

        let stats = tracker.statistics();
        assert_eq!(stats.total_accesses, 3);
        assert!(stats.is_sequential);
    }

    #[test]
    fn test_bounds_checking() {
        let config = LazyLoadConfig::default();
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        let storage = MappedStorage::<f32>::new(&file_path, 10, config).unwrap();

        // Out of bounds read
        let result = storage.get_slice(5, 10);
        assert!(result.is_err());

        // Out of bounds write
        let data = vec![1.0; 10];
        let result = storage.write_slice(5, &data);
        assert!(result.is_err());
    }
}
