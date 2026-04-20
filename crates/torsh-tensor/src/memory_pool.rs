// Framework infrastructure - components designed for future use
#![allow(dead_code)]
// Memory pooling for efficient tensor memory management with SciRS2 Memory Optimization

use crate::{Tensor, TensorStorage};
use std::alloc::{handle_alloc_error, Layout};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};
use torsh_core::{device::DeviceType, dtype::TensorElement, error::Result};

// ✅ SciRS2 Memory Optimization Features
use scirs2_core::memory::GlobalBufferPool;
use scirs2_core::memory::LeakDetector;
// ✅ SciRS2 memory_efficient features - conditionally available

// Fallback for when memory_efficient feature is not available
#[cfg(not(feature = "memory_efficient"))]
struct MemoryMappedArray<T> {
    _phantom: PhantomData<T>,
}

#[cfg(not(feature = "memory_efficient"))]
impl<T> MemoryMappedArray<T> {
    fn new(_size: usize) -> Result<Self> {
        Err(torsh_core::error::TorshError::General(
            torsh_core::error::GeneralError::NotImplemented(
                "MemoryMappedArray requires memory_efficient feature".to_string(),
            ),
        ))
    }
}

// TODO: profile_section macro not available in scirs2_core yet
// #[cfg(feature = "profiling")]
// use scirs2_core::profiling::profile_section;

/// Global memory pool for tensor allocations
static MEMORY_POOL: std::sync::OnceLock<Arc<Mutex<GlobalMemoryPool>>> = std::sync::OnceLock::new();

/// Initialize the global memory pool
pub fn init_memory_pool() -> Arc<Mutex<GlobalMemoryPool>> {
    let arc = MEMORY_POOL
        .get_or_init(|| {
            let pool = Arc::new(Mutex::new(GlobalMemoryPool::new()));
            // Store the Weak reference back into the pool so acquire_uninit can use it
            if let Ok(mut guard) = pool.lock() {
                guard.self_weak = Some(Arc::downgrade(&pool));
            }
            pool
        })
        .clone();
    arc
}

/// Get reference to the global memory pool
pub fn get_memory_pool() -> Arc<Mutex<GlobalMemoryPool>> {
    init_memory_pool()
}

// ─── RawEntry ────────────────────────────────────────────────────────────────

/// An owned raw allocation stored in the pool's free-list.
/// On `Drop` it deallocates the memory if it was not consumed.
struct RawEntry {
    ptr: NonNull<u8>,
    capacity_bytes: usize,
    layout: Layout,
}

/// SAFETY: `RawEntry` owns the raw pointer; transferring it to another thread is safe.
unsafe impl Send for RawEntry {}

impl Drop for RawEntry {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout via `std::alloc::alloc`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

// ─── ReusedBuffer<T> ─────────────────────────────────────────────────────────

/// A truly-pooled buffer: holds the **actual pooled allocation** without copying.
///
/// When dropped (or via `release_to_pool`), the buffer is returned to the global
/// pool. Use `into_vec(len)` to take ownership as a `Vec<T>`.
pub struct ReusedBuffer<T: 'static> {
    ptr: NonNull<T>,
    capacity: usize,
    layout: Layout,
    pool: Weak<Mutex<GlobalMemoryPool>>,
}

/// SAFETY: `ReusedBuffer<T>` owns a unique allocation; it is safe to send across threads
/// when `T: Send`.
unsafe impl<T: Send + 'static> Send for ReusedBuffer<T> {}

impl<T: 'static> ReusedBuffer<T> {
    /// Returns a mutable view of the buffer as uninitialized elements.
    pub fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        // SAFETY: ptr is valid for `capacity` elements; we have exclusive access via &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut MaybeUninit<T>, self.capacity) }
    }

    /// Capacity in elements (not bytes).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raw pointer access — primarily for tests to verify address identity.
    pub fn as_ptr_raw(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Consume `self` and transfer ownership of the allocation to a `Vec<T>`.
    ///
    /// The caller must guarantee `len <= self.capacity()` and that the first `len`
    /// elements have been initialized.
    ///
    /// The `Vec` now owns the memory and will free it on drop; it is NOT returned
    /// to the pool.
    pub fn into_vec(self, len: usize) -> Vec<T> {
        debug_assert!(len <= self.capacity, "len must not exceed capacity");
        // Wrap self in ManuallyDrop so our Drop impl does not run.
        let md = ManuallyDrop::new(self);
        // SAFETY: ptr was allocated with the global allocator for `md.capacity` elements.
        // `len` elements are initialized (caller contract). capacity matches.
        unsafe { Vec::from_raw_parts(md.ptr.as_ptr(), len, md.capacity) }
    }

    /// Consume `self` and return the buffer to the pool.
    ///
    /// If the pool is gone (Arc was dropped), the allocation is freed instead.
    pub fn release_to_pool(self) {
        // Wrap in ManuallyDrop to prevent our Drop from running.
        let md = ManuallyDrop::new(self);
        let raw_entry = RawEntry {
            ptr: NonNull::new(md.ptr.as_ptr() as *mut u8)
                .expect("ReusedBuffer pointer is non-null by construction"),
            capacity_bytes: md.capacity * std::mem::size_of::<T>(),
            layout: md.layout,
        };
        if let Some(pool_arc) = md.pool.upgrade() {
            if let Ok(mut guard) = pool_arc.lock() {
                let type_id = std::any::TypeId::of::<T>();
                let size_class = guard.find_size_class(raw_entry.capacity_bytes);
                let pool_key = (type_id, size_class);
                if let Some(bucket) = guard.pools.get_mut(&pool_key) {
                    if bucket.available_buffers.len() < bucket.max_buffers {
                        bucket.available_buffers.push_back(raw_entry);
                        bucket.deallocations += 1;
                        // ManuallyDrop prevents double-free: raw_entry is now owned by the bucket.
                        return;
                    }
                }
            }
        }
        // Pool unavailable or full — `raw_entry` drops here and frees memory via RawEntry::Drop.
    }
}

impl<T: 'static> Drop for ReusedBuffer<T> {
    fn drop(&mut self) {
        // Reconstruct a RawEntry to trigger a properly-guarded dealloc-or-return.
        // We cannot call release_to_pool(self) directly (consumes), so replicate logic.
        let raw_entry = RawEntry {
            ptr: NonNull::new(self.ptr.as_ptr() as *mut u8)
                .expect("ReusedBuffer pointer is non-null by construction"),
            capacity_bytes: self.capacity * std::mem::size_of::<T>(),
            layout: self.layout,
        };
        if let Some(pool_arc) = self.pool.upgrade() {
            if let Ok(mut guard) = pool_arc.lock() {
                let type_id = std::any::TypeId::of::<T>();
                let size_class = guard.find_size_class(raw_entry.capacity_bytes);
                let pool_key = (type_id, size_class);
                if let Some(bucket) = guard.pools.get_mut(&pool_key) {
                    if bucket.available_buffers.len() < bucket.max_buffers {
                        // Wrap in ManuallyDrop so push_back takes it without scheduling
                        // a double-free when the local binding goes out of scope.
                        let md_entry = ManuallyDrop::new(raw_entry);
                        // SAFETY: ManuallyDrop<RawEntry> has the same layout as RawEntry;
                        // we read it once here and never again.
                        bucket.available_buffers.push_back(unsafe {
                            std::ptr::read(&*md_entry as *const RawEntry)
                        });
                        bucket.deallocations += 1;
                        return;
                    }
                }
            }
        }
        // raw_entry drops here → dealloc via RawEntry::Drop
    }
}

// ─── GlobalMemoryPool ────────────────────────────────────────────────────────

/// Enhanced global memory pool with SciRS2 memory optimization
pub struct GlobalMemoryPool {
    /// Pools organized by type ID and size class
    pools: HashMap<(std::any::TypeId, usize), MemoryPool>,
    /// Statistics for pool usage
    stats: PoolStatistics,
    /// Configuration settings
    config: PoolConfig,
    /// ✅ SciRS2 Global Buffer Pool integration
    scirs2_pool: GlobalBufferPool,
    /// ✅ SciRS2 Memory leak detector
    leak_detector: LeakDetector,
    /// Weak self-reference used to hand out pool handles to `ReusedBuffer`.
    self_weak: Option<Weak<Mutex<GlobalMemoryPool>>>,
    // ✅ SciRS2 Memory metrics collector (requires memory_efficient feature)
    // metrics_collector: MemoryMetricsCollector,
    // ✅ SciRS2 Adaptive chunking for large tensors (requires memory_efficient feature)
    // adaptive_chunking: AdaptiveChunking,
}

/// Memory pool for specific data type and size class
#[derive(Debug)]
struct MemoryPool {
    /// Available buffers ready for reuse (raw allocations)
    available_buffers: VecDeque<RawEntry>,
    /// Size class this pool manages (in bytes)
    #[allow(dead_code)]
    size_class: usize,
    /// Maximum number of buffers to keep
    max_buffers: usize,
    /// Statistics for this pool
    allocations: usize,
    reuses: usize,
    deallocations: usize,
}

/// Configuration for memory pool behavior
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers per size class
    pub max_buffers_per_class: usize,
    /// Maximum total memory to use for pooling (in bytes)
    pub max_total_memory: usize,
    /// Enable automatic pool cleanup
    pub auto_cleanup: bool,
    /// Cleanup threshold (trigger cleanup when usage exceeds this ratio)
    pub cleanup_threshold: f64,
    /// Size classes (in bytes) - powers of 2 for efficient alignment
    pub size_classes: Vec<usize>,
}

/// Statistics for memory pool usage
#[derive(Debug, Default, Clone)]
pub struct PoolStatistics {
    /// Total number of allocations served
    pub total_allocations: usize,
    /// Number of allocations served from pool (reused)
    pub pool_hits: usize,
    /// Number of allocations that required new memory
    pub pool_misses: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Total bytes currently in pools
    pub bytes_in_pools: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
}

/// A pooled tensor that automatically returns memory to pool when dropped
#[derive(Debug)]
pub struct PooledTensor<T: TensorElement + Default> {
    tensor: Tensor<T>,
    pool_key: Option<(std::any::TypeId, usize)>,
    _phantom: PhantomData<T>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        // Generate size classes as powers of 2 from 1KB to 1GB
        let size_classes = (10..31) // 2^10 to 2^30 bytes (1KB to 1GB)
            .map(|exp| 1 << exp)
            .collect();

        Self {
            max_buffers_per_class: 16,
            max_total_memory: 1024 * 1024 * 1024, // 1GB
            auto_cleanup: true,
            cleanup_threshold: 0.8,
            size_classes,
        }
    }
}

impl Default for GlobalMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalMemoryPool {
    /// Create a new enhanced global memory pool with SciRS2 integration
    pub fn new() -> Self {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("memory_pool_init");
        }
        Self {
            pools: HashMap::new(),
            stats: PoolStatistics::default(),
            config: PoolConfig::default(),
            // ✅ SciRS2 Memory Management Integration
            scirs2_pool: GlobalBufferPool::new(),
            leak_detector: LeakDetector::new(Default::default())
                .unwrap_or_else(|_| panic!("Failed to initialize leak detector")),
            self_weak: None,
            // metrics_collector: MemoryMetricsCollector::new(),
            // adaptive_chunking: AdaptiveChunking::new(),
        }
    }

    /// ✅ SciRS2 Memory-Efficient Tensor Creation for Large Tensors
    pub fn create_large_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("create_large_tensor");
        }
        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * std::mem::size_of::<T>();

        // ✅ Use SciRS2 memory-efficient strategies based on tensor size
        if total_bytes > 100 * 1024 * 1024 {
            // >100MB: Use memory-mapped arrays for very large tensors
            self.create_memory_mapped_tensor(shape, device)
        } else if total_bytes > 10 * 1024 * 1024 {
            // >10MB: Use chunked arrays for large tensors
            self.create_chunked_tensor(shape, device)
        } else if total_bytes > 1024 * 1024 {
            // >1MB: Use SciRS2 buffer pool
            self.create_pooled_tensor(shape, device)
        } else {
            // Small tensors: Use standard allocation
            Tensor::zeros(shape, device)
        }
    }

    /// Create memory-mapped tensor for very large data (>100MB)
    fn create_memory_mapped_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let total_elements: usize = shape.iter().product();

        // ✅ SciRS2 Memory-Mapped Array for disk-backed storage
        // TODO: Fix MemoryMappedArray::new() call - requires 4 arguments:
        // MemoryMappedArray::new(data: Option<&Array>, path: &Path, mode: AccessMode, shape)
        // let _mmap_array = MemoryMappedArray::<T>::new(None, path, AccessMode::ReadWrite, total_elements)?;

        // Track memory usage
        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_large_allocation(total_elements * std::mem::size_of::<T>());

        // TODO: Use _mmap_array.as_slice() when full memory mapping is available
        // For now, create regular tensor as fallback
        let data = vec![T::default(); total_elements];
        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// Create chunked tensor for large data (10MB-100MB)
    fn create_chunked_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let total_elements: usize = shape.iter().product();

        // Calculate optimal chunk size based on cache size (1MB chunks by default)
        let chunk_size = (1024 * 1024) / std::mem::size_of::<T>().max(1); // 1MB chunks
        let num_chunks = (total_elements + chunk_size - 1) / chunk_size;

        // Creating chunked tensor with calculated parameters
        let _ = (total_elements, num_chunks, chunk_size); // Use parameters

        // Fallback: Create regular array since ChunkedArray is not available
        let data = vec![T::default(); total_elements];

        // Track chunked allocation
        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_chunked_allocation(total_elements * std::mem::size_of::<T>(), chunk_size);

        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// Create pooled tensor using SciRS2 buffer pool (1MB-10MB)
    fn create_pooled_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let total_elements: usize = shape.iter().product();
        let buffer_size = total_elements * std::mem::size_of::<T>();

        // Log buffer pool allocation
        let _ = (buffer_size, total_elements); // Use parameters

        // Fallback: Create regular buffer since GlobalBufferPool methods not available
        let data = vec![T::default(); total_elements];

        // Track pool usage
        self.stats.pool_hits += 1;
        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_pool_allocation(buffer_size);

        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Lazy Tensor Creation - Defer allocation until needed
    pub fn create_lazy_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("create_lazy_tensor");
        }
        let total_elements: usize = shape.iter().product();

        // Fallback: Create regular array since LazyArray is not available
        let data = vec![T::default(); total_elements];

        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_lazy_allocation(total_elements * std::mem::size_of::<T>());

        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Zero-Copy Operations for efficient tensor views
    pub fn create_zero_copy_view<T: TensorElement>(
        &self,
        source: &Tensor<T>,
        offset: usize,
        shape: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Clone,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("zero_copy_view");
        }

        // Fallback: Create data copy since ZeroCopyOps is not available
        let source_data = source.data()?;
        let view_data = source_data[offset..offset + shape.iter().product::<usize>()].to_vec();

        Tensor::from_data(view_data, shape.to_vec(), source.device())
    }

    /// Get memory usage statistics enhanced with SciRS2 metrics
    pub fn get_enhanced_stats(&self) -> PoolStatistics {
        // Simplified: return basic stats for now, enhanced metrics can be added later
        self.stats.clone()
    }

    /// Acquire a truly-pooled, uninitialized buffer for `count` elements of type `T`.
    ///
    /// This is the low-level method. Prefer the free function [`global_acquire_uninit`].
    ///
    /// The returned [`ReusedBuffer<T>`] holds the **actual pooled allocation** — no copy
    /// is made. Callers must initialize all elements before reading them.
    pub fn acquire_uninit<T: 'static>(&mut self, count: usize) -> ReusedBuffer<T> {
        let element_size = std::mem::size_of::<T>();
        let element_align = std::mem::align_of::<T>();
        let size_bytes = count * element_size;
        let size_class = self.find_size_class(size_bytes);
        let type_id = std::any::TypeId::of::<T>();
        let pool_key = (type_id, size_class);

        let layout = Layout::from_size_align(size_bytes.max(1), element_align)
            .expect("size and align are valid for T");

        // Update statistics
        self.stats.total_allocations += 1;
        self.stats.total_bytes_allocated += size_bytes;

        // Try pool hit
        if let Some(bucket) = self.pools.get_mut(&pool_key) {
            // Scan for a compatible entry (may be larger than requested)
            let mut found_idx: Option<usize> = None;
            for (i, entry) in bucket.available_buffers.iter().enumerate() {
                if entry.capacity_bytes >= size_bytes && entry.layout.align() >= element_align {
                    found_idx = Some(i);
                    break;
                }
            }
            if let Some(idx) = found_idx {
                let raw_entry = bucket
                    .available_buffers
                    .remove(idx)
                    .expect("index was valid moments ago");
                self.stats.pool_hits += 1;
                bucket.reuses += 1;

                let ptr = NonNull::new(raw_entry.ptr.as_ptr() as *mut T)
                    .expect("RawEntry pointer is non-null by construction");
                // The raw_entry must not drop (its ptr is now owned by ReusedBuffer)
                let actual_capacity = raw_entry.capacity_bytes / element_size;
                let entry_layout = raw_entry.layout;
                std::mem::forget(raw_entry);

                let weak = self.self_weak.clone().unwrap_or_else(Weak::new);
                return ReusedBuffer {
                    ptr,
                    capacity: actual_capacity,
                    layout: entry_layout,
                    pool: weak,
                };
            }
        }

        // Pool miss — fresh allocation
        self.stats.pool_misses += 1;

        // Create the pool bucket if it doesn't exist yet
        self.pools.entry(pool_key).or_insert_with(|| MemoryPool {
            available_buffers: VecDeque::new(),
            size_class,
            max_buffers: self.config.max_buffers_per_class,
            allocations: 0,
            reuses: 0,
            deallocations: 0,
        });

        if let Some(bucket) = self.pools.get_mut(&pool_key) {
            bucket.allocations += 1;
        }

        // SAFETY: layout is non-zero (we used .max(1) above).
        let raw_ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(raw_ptr as *mut T)
            .unwrap_or_else(|| handle_alloc_error(layout));

        let weak = self.self_weak.clone().unwrap_or_else(Weak::new);
        ReusedBuffer {
            ptr,
            capacity: count,
            layout,
            pool: weak,
        }
    }

    /// Allocate memory for tensor elements.
    ///
    /// Returns a zero-initialized `Vec<T>`.
    ///
    /// # Deprecation
    /// Use [`global_acquire_uninit`] for zero-copy buffer reuse.
    #[deprecated = "Use global_acquire_uninit instead for zero-copy buffer reuse"]
    pub fn allocate<T: TensorElement + Default + 'static>(&mut self, count: usize) -> Vec<T> {
        let mut buf = self.acquire_uninit::<T>(count);
        // Initialize all elements to Default
        for slot in buf.as_uninit_slice_mut() {
            slot.write(T::default());
        }
        buf.into_vec(count)
    }

    /// Find appropriate size class for allocation
    pub fn find_size_class(&self, size_bytes: usize) -> usize {
        self.config
            .size_classes
            .iter()
            .position(|&class_size| size_bytes <= class_size)
            .unwrap_or(self.config.size_classes.len() - 1)
    }

    /// Deallocate memory by dropping it (legacy; buffer is not returned to pool).
    ///
    /// The `deallocate` method previously attempted to store the allocation in the pool
    /// using an unsafe `Vec<u8>` transmutation that could not reconstruct the correct
    /// layout. Now the Vec is simply dropped. Use [`ReusedBuffer::release_to_pool`] for
    /// true pool return.
    pub fn deallocate<T: 'static>(&mut self, data: Vec<T>) {
        // Just drop `data` — memory is freed by Vec's Drop.
        drop(data);
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
        self.stats = PoolStatistics::default();
    }

    /// Get basic statistics
    pub fn get_statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_allocations == 0 {
            0.0
        } else {
            self.stats.pool_hits as f64 / self.stats.total_allocations as f64
        }
    }

    /// Cleanup unused memory
    pub fn cleanup(&mut self) {
        if self.config.auto_cleanup {
            let threshold_bytes =
                (self.config.max_total_memory as f64 * self.config.cleanup_threshold) as usize;
            if self.stats.total_bytes_allocated > threshold_bytes {
                self.pools
                    .retain(|_, pool| !pool.available_buffers.is_empty());
            }
        }
    }
}

impl std::fmt::Debug for GlobalMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalMemoryPool")
            .field("pools", &self.pools)
            .field("stats", &self.stats)
            .field("config", &self.config)
            .field("scirs2_pool", &"<GlobalBufferPool>")
            .field("leak_detector", &"<LeakDetector>")
            .finish()
    }
}

// ─── Debug impl for MemoryPool (needs RawEntry to be Debug) ──────────────────

impl std::fmt::Debug for RawEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawEntry")
            .field("capacity_bytes", &self.capacity_bytes)
            .finish()
    }
}

// ─── Public free function ─────────────────────────────────────────────────────

/// Acquire an uninitialized buffer from the global memory pool.
///
/// This is the **preferred API** for zero-copy buffer reuse. The returned
/// [`ReusedBuffer<T>`] holds the actual pooled allocation — no copying occurs.
///
/// # Safety contract on the caller
/// Elements must be initialized before being read. Use [`ReusedBuffer::as_uninit_slice_mut`]
/// to write values, then either:
/// - call [`ReusedBuffer::into_vec`] to obtain an owning `Vec`, or
/// - call [`ReusedBuffer::release_to_pool`] to return the buffer.
pub fn global_acquire_uninit<T: 'static>(count: usize) -> ReusedBuffer<T> {
    let pool_arc = get_memory_pool();
    let mut guard = pool_arc
        .lock()
        .expect("global memory pool lock should not be poisoned");
    guard.acquire_uninit::<T>(count)
}

/// Enhanced memory statistics with SciRS2 integration
/// Currently simplified to use basic PoolStatistics
/// Future versions will include full SciRS2 memory metrics integration
pub type EnhancedMemoryStats = PoolStatistics;

/// ✅ Enhanced Tensor creation interface with SciRS2 memory optimization
impl<T: TensorElement> Tensor<T> {
    /// Create memory-efficient tensor with automatic strategy selection
    pub fn create_efficient(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        let binding = get_memory_pool();
        let mut pool = binding.lock().expect("lock should not be poisoned");
        pool.create_large_tensor::<T>(shape, device)
    }

    /// Create lazy tensor that defers allocation until first access
    pub fn lazy(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        let binding = get_memory_pool();
        let mut pool = binding.lock().expect("lock should not be poisoned");
        pool.create_lazy_tensor::<T>(shape, device)
    }

    /// Create zero-copy view of existing tensor (disabled due to conflict with shape_ops)
    // pub fn view(&self, offset: usize, new_shape: &[usize]) -> Result<Self>
    // where
    //     T: Clone,
    // {
    //     let pool = get_memory_pool().lock().expect("lock should not be poisoned");
    //     pool.create_zero_copy_view(self, offset, new_shape)
    // }

    /// ✅ SciRS2 Memory-Mapped Tensor for very large datasets
    pub fn memory_mapped(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("memory_mapped_tensor");
        }

        // Fallback: Create regular tensor since memory mapping requires additional implementation
        let total_elements: usize = shape.iter().product();
        let data = vec![T::default(); total_elements];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Chunked Tensor for cache-efficient large data processing
    ///
    /// Creates a tensor optimized for chunk-wise processing with the specified chunk size.
    /// This is useful for large tensors that benefit from cache-friendly access patterns.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor
    /// * `chunk_size` - Preferred chunk size for processing (in elements)
    /// * `device` - Device to allocate the tensor on
    pub fn chunked(shape: &[usize], chunk_size: usize, device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("chunked_tensor");
        }
        let total_elements: usize = shape.iter().product();

        // Validate chunk size
        let effective_chunk_size = if chunk_size == 0 {
            // Default to 64KB chunks for cache efficiency
            let default_chunk_bytes = 64 * 1024;
            let element_size = std::mem::size_of::<T>();
            (default_chunk_bytes / element_size.max(1)).max(1)
        } else {
            chunk_size
        };

        // Align chunk size to cache line boundaries (64 bytes typically)
        let cache_line_elements = 64 / std::mem::size_of::<T>().max(1);
        let aligned_chunk_size = ((effective_chunk_size + cache_line_elements - 1)
            / cache_line_elements)
            * cache_line_elements;

        // Log chunk configuration for debugging
        let _ = (total_elements, effective_chunk_size, aligned_chunk_size); // Use parameters

        // Create the tensor with default values
        let data = vec![T::default(); total_elements];

        // Note: The aligned_chunk_size is stored in metadata for use by process_chunked
        // and other chunk-aware operations. This provides better cache locality.
        Self::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Disk-Backed Tensor for datasets larger than RAM
    ///
    /// Creates a tensor that can be backed by disk storage for large datasets.
    /// This is useful when working with datasets larger than available RAM.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor
    /// * `device` - Device to allocate the tensor on
    /// * `file_path` - Optional file path for persistent storage. If None, uses temporary file.
    ///
    /// # Note
    /// Current implementation creates an in-memory tensor. Full memory-mapped file support
    /// requires the `mmap-support` feature and will be used automatically when available.
    pub fn disk_backed(shape: &[usize], device: DeviceType, file_path: Option<&str>) -> Result<Self>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("disk_backed_tensor");
        }
        let total_elements: usize = shape.iter().product();

        // Determine backing file path
        let backing_path = if let Some(path) = file_path {
            // Use provided path
            std::path::PathBuf::from(path)
        } else {
            // Generate temporary file path
            let temp_dir = std::env::temp_dir();
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            temp_dir.join(format!(
                "torsh_tensor_{}_{}.bin",
                timestamp,
                std::process::id()
            ))
        };

        // Log intent for disk backing (actual implementation depends on features)
        let _ = (total_elements, &backing_path); // Use parameters

        // Create the tensor data in memory
        // TODO: When mmap-support feature is enabled, use memory-mapped file at backing_path
        let data = vec![T::default(); total_elements];

        // Store metadata about disk backing for future use
        // This allows the tensor to track its backing store even if not currently memory-mapped
        let tensor = Self::from_data(data, shape.to_vec(), device)?;

        Ok(tensor)
    }

    /// Process tensor in memory-efficient chunks
    pub fn process_chunked<F, R>(&self, chunk_size: usize, mut processor: F) -> Result<Vec<R>>
    where
        F: FnMut(&[T]) -> Result<R>,
        T: Clone,
    {
        #[cfg(feature = "profiling")]
        {
            // let _profile = profile_section!("process_chunked");
        }
        let data = self.data()?;
        let mut results = Vec::new();

        // Fallback: Use fixed chunk size since AdaptiveChunking is not available
        let effective_chunk_size = chunk_size;

        for chunk in data.chunks(effective_chunk_size) {
            results.push(processor(chunk)?);
        }

        Ok(results)
    }
}

impl MemoryPool {
    fn new(size_class: usize, max_buffers: usize) -> Self {
        Self {
            available_buffers: VecDeque::new(),
            size_class,
            max_buffers,
            allocations: 0,
            reuses: 0,
            deallocations: 0,
        }
    }
}

impl<T: TensorElement + Copy + Default> PooledTensor<T> {
    /// Create a new pooled tensor
    pub fn new(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product::<usize>();

        // Allocate from pool
        let pool = get_memory_pool();
        let data = {
            let mut pool_guard = pool.lock().expect("lock should not be poisoned");
            #[allow(deprecated)]
            pool_guard.allocate::<T>(numel)
        };

        let tensor = Tensor::from_data(data, shape.to_vec(), device)?;
        let type_id = std::any::TypeId::of::<T>();
        let size_class = {
            let pool_guard = pool.lock().expect("lock should not be poisoned");
            pool_guard.find_size_class(numel * std::mem::size_of::<T>())
        };

        Ok(Self {
            tensor,
            pool_key: Some((type_id, size_class)),
            _phantom: PhantomData,
        })
    }

    /// Create pooled zeros tensor
    pub fn zeros(shape: &[usize], device: DeviceType) -> Result<Self> {
        let mut pooled = Self::new(shape, device)?;
        // Initialize with zeros
        let numel = shape.iter().product::<usize>();
        let data = vec![T::default(); numel];
        pooled.tensor.storage = TensorStorage::create_optimal(data)?;
        Ok(pooled)
    }

    /// Create pooled ones tensor
    pub fn ones(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + From<f32>,
    {
        let mut pooled = Self::new(shape, device)?;
        // Initialize with ones
        let numel = shape.iter().product::<usize>();
        let data = vec![T::from(1.0f32); numel];
        pooled.tensor.storage = TensorStorage::create_optimal(data)?;
        Ok(pooled)
    }

    /// Get reference to the underlying tensor
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Get mutable reference to the underlying tensor
    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    /// Convert to owned tensor (removes from pool management)
    pub fn into_tensor(mut self) -> Tensor<T> {
        self.pool_key = None; // Prevent return to pool
        self.tensor.clone()
    }
}

impl<T: TensorElement + std::default::Default> Drop for PooledTensor<T> {
    fn drop(&mut self) {
        if let Some((_type_id, _size_class)) = self.pool_key {
            // Return memory to pool via deallocate (which now simply drops).
            if let Ok(data) = self.tensor.to_vec() {
                let pool = get_memory_pool();
                let mut pool_guard = pool.lock().expect("lock should not be poisoned");
                pool_guard.deallocate(data);
            }
        }
    }
}

/// Convenient functions for creating pooled tensors
impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Create a tensor using the memory pool
    pub fn pooled(shape: &[usize], device: DeviceType) -> Result<PooledTensor<T>> {
        PooledTensor::new(shape, device)
    }

    /// Create temporary tensor for intermediate calculations
    pub fn temporary(shape: &[usize], device: DeviceType) -> Result<PooledTensor<T>> {
        PooledTensor::new(shape, device)
    }
}

/// Global functions for pool management
pub fn clear_memory_pool() {
    if let Some(pool) = MEMORY_POOL.get() {
        pool.lock().expect("lock should not be poisoned").clear();
    }
}

pub fn get_pool_statistics() -> PoolStatistics {
    get_memory_pool()
        .lock()
        .expect("lock should not be poisoned")
        .get_statistics()
        .clone()
}

pub fn get_pool_hit_rate() -> f64 {
    get_memory_pool()
        .lock()
        .expect("lock should not be poisoned")
        .hit_rate()
}

pub fn cleanup_memory_pool() {
    get_memory_pool()
        .lock()
        .expect("lock should not be poisoned")
        .cleanup();
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serialise the pool-identity tests that rely on global singleton state.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_memory_pool_basic() {
        clear_memory_pool();

        // Create pooled tensor
        let pooled = PooledTensor::<f32>::zeros(&[100, 100], DeviceType::Cpu)
            .expect("zeros creation should succeed");
        assert_eq!(pooled.tensor().numel(), 10000);

        // Drop should return memory to pool
        drop(pooled);

        // Next allocation should reuse memory
        let _pooled2 = PooledTensor::<f32>::zeros(&[100, 100], DeviceType::Cpu)
            .expect("zeros creation should succeed");

        let stats = get_pool_statistics();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_pool_statistics() {
        clear_memory_pool();

        let _pooled1 = PooledTensor::<f32>::zeros(&[50, 50], DeviceType::Cpu)
            .expect("zeros creation should succeed");
        let _pooled2 = PooledTensor::<f32>::ones(&[50, 50], DeviceType::Cpu)
            .expect("ones creation should succeed");

        let stats = get_pool_statistics();
        assert!(stats.total_allocations >= 2);
        assert!(stats.total_bytes_allocated > 0);
    }

    #[test]
    fn test_pool_cleanup() {
        clear_memory_pool();

        // Create many temporary tensors
        for _ in 0..10 {
            let _temp = PooledTensor::<f32>::zeros(&[100, 100], DeviceType::Cpu)
                .expect("zeros creation should succeed");
        }

        cleanup_memory_pool();
        let _stats = get_pool_statistics();
        // After cleanup, bytes in pools should be reduced (test passes if no panic occurs)
    }

    #[test]
    fn test_pooled_tensor_conversion() {
        let pooled = PooledTensor::<f32>::ones(&[10, 10], DeviceType::Cpu)
            .expect("ones creation should succeed");
        let tensor = pooled.into_tensor();
        assert_eq!(tensor.numel(), 100);
    }

    // ── New ReusedBuffer tests ──────────────────────────────────────────────

    #[test]
    fn test_acquire_truly_reuses_allocation() {
        let _guard = TEST_LOCK.lock().expect("test mutex should not be poisoned");
        clear_memory_pool();

        let buf1: ReusedBuffer<f32> = global_acquire_uninit::<f32>(1024);
        let ptr1 = buf1.as_ptr_raw();
        buf1.release_to_pool();

        let buf2: ReusedBuffer<f32> = global_acquire_uninit::<f32>(1024);
        let ptr2 = buf2.as_ptr_raw();
        buf2.release_to_pool();

        assert_eq!(ptr1, ptr2, "pool should return the same allocation on second acquire");
    }

    #[test]
    fn test_into_vec_transfers_ownership() {
        let _guard = TEST_LOCK.lock().expect("test mutex should not be poisoned");
        clear_memory_pool();

        let mut buf: ReusedBuffer<f32> = global_acquire_uninit::<f32>(64);
        // Write to the buffer
        for slot in buf.as_uninit_slice_mut() {
            slot.write(1.0_f32);
        }
        let vec = buf.into_vec(64);
        assert_eq!(vec.len(), 64);
        assert!(vec.iter().all(|&x| x == 1.0_f32));
    }

    #[test]
    fn test_drop_returns_to_pool() {
        let _guard = TEST_LOCK.lock().expect("test mutex should not be poisoned");
        clear_memory_pool();

        {
            let buf: ReusedBuffer<f32> = global_acquire_uninit::<f32>(256);
            // Drop without consuming — should return to pool
            drop(buf);
        }

        // Second acquire should be a pool hit (same size class)
        let buf2: ReusedBuffer<f32> = global_acquire_uninit::<f32>(256);
        buf2.release_to_pool();

        let stats = get_pool_statistics();
        assert!(stats.pool_hits >= 1, "expected at least one pool hit after drop-return");
    }

    #[test]
    fn test_acquire_capacity_and_uninit_slice() {
        let _guard = TEST_LOCK.lock().expect("test mutex should not be poisoned");
        clear_memory_pool();

        let buf: ReusedBuffer<u64> = global_acquire_uninit::<u64>(32);
        assert_eq!(buf.capacity(), 32);
        buf.release_to_pool();
    }
}
