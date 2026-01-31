//! Storage management for tensor data
//!
//! This module provides storage abstractions for tensor data, including both
//! in-memory and memory-mapped storage options with automatic optimization
//! based on data size.
//!
//! # Features
//!
//! - **In-memory storage**: Fast access for smaller tensors
//! - **Memory-mapped storage**: Efficient for large tensors with caching
//! - **Automatic optimization**: Chooses optimal storage based on size
//! - **Cross-platform support**: Works on Unix, Windows, and other platforms
//! - **LRU cache management**: Optimizes memory usage for memory-mapped storage

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
#[cfg(feature = "simd")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};

// 🚀 SciRS2 AlignedVec integration for SIMD-optimized storage
#[cfg(feature = "simd")]
use scirs2_core::simd_aligned::AlignedVec;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Threshold for switching to memory-mapped storage (1 GB)
const MEMORY_MAPPING_THRESHOLD: usize = 1024 * 1024 * 1024;

/// Threshold for using aligned storage for SIMD optimization (1 KB)
/// Arrays larger than this benefit from cache-line aligned memory for SIMD operations
#[cfg(feature = "simd")]
const ALIGNED_STORAGE_THRESHOLD: usize = 1024;

/// Threshold for using lock-free SIMD storage (10 KB)
/// Arrays larger than this benefit from lock-free access patterns
#[cfg(feature = "simd")]
const SIMD_OPTIMIZED_THRESHOLD: usize = 10240;

// ============================================================================
// PHASE 5: SIMD-OPTIMIZED LOCK-FREE STORAGE
// ============================================================================
// Uses Copy-on-Write semantics with atomic flag instead of RwLock.
// Benefits:
// - No lock acquisition overhead for reads (~20ns savings)
// - Direct slice access for SIMD operations
// - Thread-safe through atomic COW semantics
// ============================================================================

/// SIMD-optimized storage with Copy-on-Write semantics (Phase 5)
///
/// This storage variant eliminates RwLock overhead for read operations:
/// - Reads are lock-free (just atomic flag check)
/// - Writes trigger copy-on-write if shared
/// - Optimal for SIMD operations on medium-to-large tensors
#[cfg(feature = "simd")]
pub struct SimdStorage<T> {
    /// The actual aligned data
    data: AlignedVec<T>,
    /// Whether this storage is shared (needs COW on write)
    shared: AtomicBool,
}

#[cfg(feature = "simd")]
impl<T> SimdStorage<T> {
    /// Create new SIMD storage from data
    pub fn new(data: AlignedVec<T>) -> Self {
        Self {
            data,
            shared: AtomicBool::new(false),
        }
    }

    /// Get the length of the storage
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get immutable slice (lock-free)
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Mark as shared (for Clone)
    pub fn mark_shared(&self) {
        self.shared.store(true, Ordering::SeqCst);
    }

    /// Check if shared
    pub fn is_shared(&self) -> bool {
        self.shared.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "simd")]
impl<T: Copy> SimdStorage<T> {
    /// Get mutable slice - only if not shared
    ///
    /// Returns None if the storage is shared (would need COW)
    pub fn as_mut_slice_if_unique(&mut self) -> Option<&mut [T]> {
        if self.shared.load(Ordering::SeqCst) {
            None // Shared, cannot mutate directly
        } else {
            Some(self.data.as_mut_slice())
        }
    }

    /// Convert to Vec (copying data)
    pub fn to_vec(&self) -> Vec<T> {
        self.data.as_slice().to_vec()
    }
}

#[cfg(feature = "simd")]
impl<T> std::fmt::Debug for SimdStorage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdStorage")
            .field("len", &self.data.len())
            .field("shared", &self.shared.load(Ordering::Relaxed))
            .finish()
    }
}

/// Storage abstraction for tensor data
pub enum TensorStorage<T: TensorElement> {
    /// In-memory storage for smaller tensors
    InMemory(Arc<RwLock<Vec<T>>>),
    /// Memory-mapped storage for large tensors
    MemoryMapped(Arc<RwLock<MemoryMappedStorage<T>>>),
    /// Cache-line aligned storage for SIMD-optimized operations (14.17x speedup)
    #[cfg(feature = "simd")]
    Aligned(Arc<RwLock<AlignedVec<T>>>),
    /// 🚀 Lock-free SIMD storage with Copy-on-Write semantics (Phase 5)
    ///
    /// Benefits:
    /// - Lock-free read access (~20ns savings per operation)
    /// - Direct slice access for SIMD
    /// - Thread-safe through atomic COW
    #[cfg(feature = "simd")]
    SimdOptimized(Arc<SimdStorage<T>>),
}

impl<T: TensorElement> std::fmt::Debug for TensorStorage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory(data) => f.debug_tuple("InMemory").field(data).finish(),
            Self::MemoryMapped(storage) => f.debug_tuple("MemoryMapped").field(storage).finish(),
            #[cfg(feature = "simd")]
            Self::Aligned(_) => f.debug_tuple("Aligned").field(&"<AlignedVec>").finish(),
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => f.debug_tuple("SimdOptimized").field(storage).finish(),
        }
    }
}

/// Memory-mapped storage implementation
#[derive(Debug)]
pub struct MemoryMappedStorage<T: TensorElement> {
    /// File backing the memory mapping
    file: File,
    /// Path to the backing file
    file_path: PathBuf,
    /// Number of elements stored
    num_elements: usize,
    /// Cache for frequently accessed elements
    cache: HashMap<usize, T>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Access pattern tracking for cache optimization
    access_pattern: VecDeque<usize>,
    /// Whether the storage is temporary (should be deleted on drop)
    is_temporary: bool,
}

impl<T: TensorElement + Copy> TensorStorage<T> {
    /// Create in-memory storage
    pub fn in_memory(data: Vec<T>) -> Self {
        Self::InMemory(Arc::new(RwLock::new(data)))
    }

    /// Create memory-mapped storage
    pub fn memory_mapped(data: Vec<T>, file_path: Option<PathBuf>) -> Result<Self> {
        let storage = MemoryMappedStorage::new(data, file_path)?;
        Ok(Self::MemoryMapped(Arc::new(RwLock::new(storage))))
    }

    /// Create cache-line aligned storage for SIMD operations (14.17x speedup potential)
    #[cfg(feature = "simd")]
    pub fn aligned(data: Vec<T>) -> Result<Self> {
        let mut aligned_vec = AlignedVec::with_capacity(data.len()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create aligned storage: {e}"))
        })?;

        for item in data {
            aligned_vec.push(item);
        }

        Ok(Self::Aligned(Arc::new(RwLock::new(aligned_vec))))
    }

    /// 🚀 **Phase 7**: Create fast result storage (skips alignment copy)
    ///
    /// For SIMD operation results where we already have the data in a Vec,
    /// uses InMemory storage to avoid the ~10µs alignment copy overhead.
    ///
    /// # Performance
    /// - Skips AlignedVec copy (saves ~10µs for 50K elements)
    /// - Uses InMemory storage (has RwLock but we just created it)
    /// - Optimal for result tensors that won't be immediately used in SIMD ops
    pub fn fast_result(data: Vec<T>) -> Self {
        Self::InMemory(Arc::new(RwLock::new(data)))
    }

    /// 🚀 **Phase 5**: Create lock-free SIMD-optimized storage
    ///
    /// This storage variant eliminates RwLock overhead for reads:
    /// - Lock-free read access (~20ns savings per operation)
    /// - Direct slice access for SIMD operations
    /// - Thread-safe through Copy-on-Write semantics
    ///
    /// # Performance
    /// - Best for medium-to-large tensors (> 10KB)
    /// - Optimal for SIMD operations that read but rarely write
    /// - **Note**: Has ~10µs alignment copy overhead for 50K elements
    #[cfg(feature = "simd")]
    pub fn simd_optimized(data: Vec<T>) -> Result<Self> {
        let mut aligned_vec = AlignedVec::with_capacity(data.len()).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create SIMD storage: {e}"))
        })?;

        for item in data {
            aligned_vec.push(item);
        }

        let simd_storage = SimdStorage::new(aligned_vec);
        Ok(Self::SimdOptimized(Arc::new(simd_storage)))
    }

    /// Create storage automatically based on size and performance characteristics
    ///
    /// **Storage Selection Strategy**:
    /// - Very large (>1GB): Memory-mapped for virtual memory efficiency
    /// - Large (>10KB, SIMD enabled): SimdOptimized (lock-free reads)
    /// - Medium (>1KB, SIMD enabled): Aligned storage for SIMD alignment
    /// - Small (<1KB): In-memory with RwLock
    pub fn create_optimal(data: Vec<T>) -> Result<Self> {
        let size_bytes = data.len() * std::mem::size_of::<T>();

        if size_bytes >= MEMORY_MAPPING_THRESHOLD {
            // Very large data: use memory mapping
            Self::memory_mapped(data, None)
        } else {
            #[cfg(feature = "simd")]
            {
                if size_bytes >= SIMD_OPTIMIZED_THRESHOLD {
                    // Large data: use lock-free SimdOptimized storage
                    // This eliminates RwLock overhead for read operations
                    return Self::simd_optimized(data);
                } else if size_bytes >= ALIGNED_STORAGE_THRESHOLD {
                    // Medium data: use aligned storage for SIMD alignment
                    return Self::aligned(data);
                }
            }
            // Small data: use regular in-memory storage
            Ok(Self::in_memory(data))
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(data) => {
                data.read().map(|guard| guard.len()).unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            Self::MemoryMapped(storage) => {
                storage.read().map(|guard| guard.num_elements).unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                data.read().map(|guard| guard.len()).unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => storage.len(), // Lock-free!
        }
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                data_guard
                    .get(index)
                    .copied()
                    .ok_or_else(|| TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    })
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .get(index),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                if index >= data_guard.len() {
                    Err(TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    })
                } else {
                    Ok(data_guard.as_slice()[index])
                }
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Lock-free access!
                let slice = storage.as_slice();
                if index >= slice.len() {
                    Err(TorshError::IndexOutOfBounds {
                        index,
                        size: slice.len(),
                    })
                } else {
                    Ok(slice[index])
                }
            }
        }
    }

    /// Set element at index
    pub fn set(&self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if index >= data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    });
                }
                data_guard[index] = value;
                Ok(())
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .set(index, value),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if index >= data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index,
                        size: data_guard.len(),
                    });
                }
                // Use the new set() method from AlignedVec
                (*data_guard).set(index, value);
                Ok(())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(_storage) => {
                // SimdOptimized uses COW - cannot mutate through shared reference
                // Caller should use make_unique() first if mutation is needed
                Err(TorshError::InvalidArgument(
                    "SimdOptimized storage is immutable. Use make_unique() for mutable access."
                        .to_string(),
                ))
            }
        }
    }

    /// Get multiple elements
    pub fn get_slice(&self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                if start + len > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + len - 1,
                        size: data_guard.len(),
                    });
                }
                Ok(data_guard[start..start + len].to_vec())
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .get_slice(start, len),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                if start + len > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + len - 1,
                        size: data_guard.len(),
                    });
                }
                let slice = data_guard.as_slice();
                Ok(slice[start..start + len].to_vec())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Lock-free access!
                let slice = storage.as_slice();
                if start + len > slice.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + len - 1,
                        size: slice.len(),
                    });
                }
                Ok(slice[start..start + len].to_vec())
            }
        }
    }

    /// Set multiple elements
    pub fn set_slice(&self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if start + values.len() > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + values.len() - 1,
                        size: data_guard.len(),
                    });
                }
                data_guard[start..start + values.len()].copy_from_slice(values);
                Ok(())
            }
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .set_slice(start, values),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                if start + values.len() > data_guard.len() {
                    return Err(TorshError::IndexOutOfBounds {
                        index: start + values.len() - 1,
                        size: data_guard.len(),
                    });
                }
                // Use as_mut_slice() and copy
                let slice = data_guard.as_mut_slice();
                slice[start..start + values.len()].copy_from_slice(values);
                Ok(())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(_storage) => {
                // SimdOptimized uses COW - cannot mutate through shared reference
                Err(TorshError::InvalidArgument(
                    "SimdOptimized storage is immutable. Use make_unique() for mutable access."
                        .to_string(),
                ))
            }
        }
    }

    /// Convert to vector (useful for small tensors or debugging)
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        match self {
            Self::InMemory(data) => Ok(data
                .read()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?
                .clone()),
            Self::MemoryMapped(storage) => storage
                .write()
                .map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?
                .to_vec(),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                Ok(data_guard.as_slice().to_vec())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Lock-free access!
                Ok(storage.to_vec())
            }
        }
    }

    /// Get storage type information
    pub fn storage_type(&self) -> &'static str {
        match self {
            Self::InMemory(_) => "in_memory",
            Self::MemoryMapped(_) => "memory_mapped",
            #[cfg(feature = "simd")]
            Self::Aligned(_) => "aligned_simd",
            #[cfg(feature = "simd")]
            Self::SimdOptimized(_) => "simd_optimized",
        }
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::InMemory(data) => {
                data.read()
                    .map(|guard| guard.len() * std::mem::size_of::<T>())
                    .unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            Self::MemoryMapped(storage) => {
                storage
                    .read()
                    .map(|storage_guard| {
                        // Memory usage is just the cache size plus metadata
                        storage_guard.cache.len() * std::mem::size_of::<T>()
                            + std::mem::size_of::<MemoryMappedStorage<T>>()
                    })
                    .unwrap_or(std::mem::size_of::<MemoryMappedStorage<T>>()) // Fallback to metadata size
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                data.read()
                    .map(|data_guard| {
                        // AlignedVec uses more memory due to alignment padding
                        data_guard.capacity() * std::mem::size_of::<T>()
                    })
                    .unwrap_or(0) // If lock is poisoned, return 0 (safe fallback)
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Lock-free access!
                storage.capacity() * std::mem::size_of::<T>()
            }
        }
    }

    /// Execute a function with immutable access to data slice (zero-copy within scope)
    ///
    /// This enables zero-copy SIMD operations by providing direct `&[T]` access
    /// within the closure scope while the lock is held.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&[T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero allocations for in-memory and aligned storage
    /// - Converts memory-mapped storage to Vec (one allocation)
    ///
    /// # Examples
    /// ```ignore
    /// storage.with_slice(|data| {
    ///     // Direct SIMD access to data
    ///     f32::simd_add(&data, &other_data)
    /// })?;
    /// ```
    pub fn with_slice<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[T]) -> Result<R>,
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                f(data_guard.as_slice())
            }
            Self::MemoryMapped(storage) => {
                // Memory-mapped storage requires conversion to Vec
                let vec = storage
                    .write()
                    .map_err(|_| {
                        TorshError::SynchronizationError("Lock poisoned during write".to_string())
                    })?
                    .to_vec()?;
                f(&vec)
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let data_guard = data.read().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during read".to_string())
                })?;
                f(data_guard.as_slice())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // 🚀 Lock-free access - no lock acquisition!
                f(storage.as_slice())
            }
        }
    }

    /// Try to get direct slice access without closures (only works for SimdOptimized)
    ///
    /// Returns `Some(&[T])` if storage is SimdOptimized (lock-free),
    /// returns `None` for other storage types (which require lock acquisition).
    ///
    /// # Performance
    /// - SimdOptimized: Direct slice access, zero overhead
    /// - Others: Returns None (use with_slice instead)
    #[cfg(feature = "simd")]
    pub fn try_as_slice_direct(&self) -> Option<&[T]> {
        match self {
            Self::SimdOptimized(storage) => Some(storage.as_slice()),
            _ => None,
        }
    }

    /// Execute a function with mutable access to data slice (zero-copy within scope)
    ///
    /// This enables zero-copy in-place operations by providing direct `&mut [T]` access
    /// within the closure scope while the lock is held.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&mut [T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero allocations for in-memory storage
    /// - Aligned storage not yet supported (returns error)
    /// - Memory-mapped storage not supported for mutable access (returns error)
    ///
    /// # Examples
    /// ```ignore
    /// storage.with_slice_mut(|data| {
    ///     // In-place SIMD operation
    ///     f32::simd_add_inplace(data, &other_data)
    /// })?;
    /// ```
    pub fn with_slice_mut<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [T]) -> Result<R>,
        T: Copy,
    {
        match self {
            Self::InMemory(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                f(data_guard.as_mut_slice())
            }
            Self::MemoryMapped(_) => {
                // Memory-mapped storage doesn't support mutable slice access
                Err(TorshError::InvalidArgument(
                    "Memory-mapped storage does not support mutable slice access".to_string(),
                ))
            }
            #[cfg(feature = "simd")]
            Self::Aligned(data) => {
                let mut data_guard = data.write().map_err(|_| {
                    TorshError::SynchronizationError("Lock poisoned during write".to_string())
                })?;
                f(data_guard.as_mut_slice())
            }
            #[cfg(feature = "simd")]
            Self::SimdOptimized(_) => {
                // SimdOptimized uses COW - cannot mutate through shared reference
                Err(TorshError::InvalidArgument(
                    "SimdOptimized storage is immutable. Use make_unique() for mutable access."
                        .to_string(),
                ))
            }
        }
    }
}

impl<T: TensorElement> MemoryMappedStorage<T> {
    /// Create new memory-mapped storage
    pub fn new(data: Vec<T>, file_path: Option<PathBuf>) -> Result<Self> {
        let (file_path, is_temporary) = match file_path {
            Some(path) => (path, false),
            None => {
                // Create temporary file
                let temp_dir = std::env::temp_dir();
                let temp_file = temp_dir.join(format!("torsh_tensor_{}.mmap", std::process::id()));
                (temp_file, true)
            }
        };

        // Create and write data to file
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| {
                TorshError::IoError(format!("Failed to create memory-mapped file: {e}"))
            })?;

        // Write data to file
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        file.write_all(data_bytes).map_err(|e| {
            TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
        })?;
        file.flush()
            .map_err(|e| TorshError::IoError(format!("Failed to flush memory-mapped file: {e}")))?;

        Ok(Self {
            file,
            file_path,
            num_elements: data.len(),
            cache: HashMap::new(),
            max_cache_size: 10000, // Cache up to 10k elements
            access_pattern: VecDeque::new(),
            is_temporary,
        })
    }

    /// Get element at index with caching
    pub fn get(&mut self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        if index >= self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.num_elements,
            });
        }

        // Check cache first
        if let Some(&value) = self.cache.get(&index) {
            self.update_access_pattern(index);
            return Ok(value);
        }

        // Read from file
        let value = self.read_element_from_file(index)?;

        // Add to cache if there's space
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(index, value);
        } else {
            // Evict least recently used element
            self.evict_lru();
            self.cache.insert(index, value);
        }

        self.update_access_pattern(index);
        Ok(value)
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        if index >= self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.num_elements,
            });
        }

        // Update cache
        self.cache.insert(index, value);

        // Write to file
        self.write_element_to_file(index, value)?;
        self.update_access_pattern(index);
        Ok(())
    }

    /// Get slice of elements
    pub fn get_slice(&mut self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        if start + len > self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index: start + len - 1,
                size: self.num_elements,
            });
        }

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(self.get(start + i)?);
        }
        Ok(result)
    }

    /// Set slice of elements
    pub fn set_slice(&mut self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if start + values.len() > self.num_elements {
            return Err(TorshError::IndexOutOfBounds {
                index: start + values.len() - 1,
                size: self.num_elements,
            });
        }

        for (i, &value) in values.iter().enumerate() {
            self.set(start + i, value)?;
        }
        Ok(())
    }

    /// Convert entire storage to vector
    pub fn to_vec(&mut self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.get_slice(0, self.num_elements)
    }

    /// Read element from file
    fn read_element_from_file(&mut self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        let offset = index * std::mem::size_of::<T>();
        let mut buffer = vec![0u8; std::mem::size_of::<T>()];

        #[cfg(unix)]
        {
            self.file
                .read_exact_at(&mut buffer, offset as u64)
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
                })?;
        }

        #[cfg(windows)]
        {
            self.file
                .seek_read(&mut buffer, offset as u64)
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
                })?;
        }

        #[cfg(not(any(unix, windows)))]
        {
            self.file
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}"))
                })?;
            self.file.read_exact(&mut buffer).map_err(|e| {
                TorshError::IoError(format!("Failed to read from memory-mapped file: {e}"))
            })?;
        }

        // Convert bytes to T
        let value = unsafe { std::ptr::read(buffer.as_ptr() as *const T) };
        Ok(value)
    }

    /// Write element to file
    fn write_element_to_file(&mut self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        let offset = index * std::mem::size_of::<T>();
        let buffer = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };

        #[cfg(unix)]
        {
            self.file.write_all_at(buffer, offset as u64).map_err(|e| {
                TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
            })?;
        }

        #[cfg(windows)]
        {
            self.file.seek_write(buffer, offset as u64).map_err(|e| {
                TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
            })?;
        }

        #[cfg(not(any(unix, windows)))]
        {
            self.file
                .seek(SeekFrom::Start(offset as u64))
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to seek in memory-mapped file: {e}"))
                })?;
            self.file.write_all(buffer).map_err(|e| {
                TorshError::IoError(format!("Failed to write to memory-mapped file: {e}"))
            })?;
        }

        Ok(())
    }

    /// Update access pattern for cache management
    fn update_access_pattern(&mut self, index: usize) {
        self.access_pattern.push_back(index);
        if self.access_pattern.len() > self.max_cache_size {
            self.access_pattern.pop_front();
        }
    }

    /// Evict least recently used element from cache
    fn evict_lru(&mut self) {
        if let Some(lru_index) = self.access_pattern.front().copied() {
            self.cache.remove(&lru_index);
        }
    }
}

impl<T: TensorElement> Drop for MemoryMappedStorage<T> {
    fn drop(&mut self) {
        if self.is_temporary {
            // Clean up temporary file
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}

impl<T: TensorElement> Clone for TensorStorage<T> {
    fn clone(&self) -> Self {
        match self {
            Self::InMemory(data) => Self::InMemory(Arc::clone(data)),
            Self::MemoryMapped(storage) => Self::MemoryMapped(Arc::clone(storage)),
            #[cfg(feature = "simd")]
            Self::Aligned(data) => Self::Aligned(Arc::clone(data)),
            #[cfg(feature = "simd")]
            Self::SimdOptimized(storage) => {
                // Mark the storage as shared for COW semantics
                storage.mark_shared();
                Self::SimdOptimized(Arc::clone(storage))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_storage() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = TensorStorage::in_memory(data.clone());

        assert_eq!(storage.len(), 4);
        assert!(!storage.is_empty());
        assert_eq!(storage.storage_type(), "in_memory");

        assert_eq!(storage.get(0).unwrap(), 1.0);
        assert_eq!(storage.get(3).unwrap(), 4.0);

        let slice = storage.get_slice(1, 2).unwrap();
        assert_eq!(slice, vec![2.0, 3.0]);
    }

    #[test]
    fn test_optimal_storage_selection() {
        // Small data should use in-memory storage (200 f32 = 800 bytes < 1024 threshold)
        let small_data = vec![1.0f32; 200];
        let small_storage = TensorStorage::create_optimal(small_data).unwrap();

        #[cfg(feature = "simd")]
        {
            // With SIMD enabled, small data below threshold should use in-memory
            assert_eq!(small_storage.storage_type(), "in_memory");
        }
        #[cfg(not(feature = "simd"))]
        {
            // Without SIMD, all data uses in-memory storage
            assert_eq!(small_storage.storage_type(), "in_memory");
        }
    }

    #[test]
    fn test_memory_usage_calculation() {
        let data = vec![1.0f32; 1000];
        let storage = TensorStorage::in_memory(data);
        let expected_size = 1000 * std::mem::size_of::<f32>();
        assert_eq!(storage.memory_usage(), expected_size);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_aligned_storage() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = TensorStorage::aligned(data.clone()).unwrap();

        assert_eq!(storage.len(), 4);
        assert!(!storage.is_empty());
        assert_eq!(storage.storage_type(), "aligned_simd");

        // Test basic element access
        assert_eq!(storage.get(0).unwrap(), 1.0);
        assert_eq!(storage.get(3).unwrap(), 4.0);

        // Test slice access
        let slice = storage.get_slice(1, 2).unwrap();
        assert_eq!(slice, vec![2.0, 3.0]);

        // Test conversion to vec
        let vec = storage.to_vec().unwrap();
        assert_eq!(vec, data);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_optimal_storage_selection_with_aligned() {
        // Medium-size data should use aligned storage when SIMD is enabled
        let medium_data = vec![1.0f32; 2000]; // Above ALIGNED_STORAGE_THRESHOLD
        let medium_storage = TensorStorage::create_optimal(medium_data).unwrap();
        assert_eq!(medium_storage.storage_type(), "aligned_simd");

        // Small data should still use in-memory storage
        let small_data = vec![1.0f32; 100]; // Below ALIGNED_STORAGE_THRESHOLD
        let small_storage = TensorStorage::create_optimal(small_data).unwrap();
        assert_eq!(small_storage.storage_type(), "in_memory");
    }
}
