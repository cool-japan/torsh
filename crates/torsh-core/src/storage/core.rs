//! Core storage interface for tensor data
//!
//! This module provides the fundamental storage abstraction and shared storage wrapper
//! that serves as the foundation for all tensor storage implementations.

use crate::device::Device;
use crate::dtype::TensorElement;
use crate::error::Result;
use std::sync::Arc;

/// Backend-agnostic storage for tensor data
///
/// This trait defines the interface for tensor storage across different backends.
/// It handles memory allocation, deallocation, and device management for tensors.
/// Implementations provide backend-specific storage for CPU, CUDA, Metal, etc.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::{Storage, TensorElement};
///
/// // Example of using storage (pseudocode)
/// fn use_storage<S, T>(device: &S::Device) -> torsh_core::Result<()>
/// where
///     S: Storage<Elem = T>,
///     T: TensorElement,
/// {
///     // Allocate storage for 1000 elements
///     let mut storage = S::allocate(device, 1000)?;
///
///     // Check storage properties
///     println!("Storage length: {}", storage.len());
///     println!("Is empty: {}", storage.is_empty());
///
///     // Clone the storage if needed
///     let cloned_storage = storage.clone_storage()?;
///
///     Ok(())
/// }
/// ```
pub trait Storage: Send + Sync + std::fmt::Debug + 'static {
    /// The element type stored
    type Elem: TensorElement;

    /// The device type
    type Device: Device;

    /// Allocate storage for the given number of elements
    fn allocate(device: &Self::Device, size: usize) -> Result<Self>
    where
        Self: Sized;

    /// Get the number of elements
    fn len(&self) -> usize;

    /// Check if the storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the device
    fn device(&self) -> &Self::Device;

    /// Clone the storage (deep copy)
    fn clone_storage(&self) -> Result<Self>
    where
        Self: Sized;
}

/// Shared storage with reference counting
///
/// This wrapper provides thread-safe shared access to storage with copy-on-write semantics.
/// It uses Arc internally for reference counting and provides methods for safe mutation.
#[derive(Debug)]
pub struct SharedStorage<S: Storage> {
    inner: Arc<S>,
}

impl<S: Storage> Clone for SharedStorage<S> {
    fn clone(&self) -> Self {
        SharedStorage {
            inner: self.inner.clone(),
        }
    }
}

impl<S: Storage> SharedStorage<S> {
    /// Create new shared storage
    pub fn new(storage: S) -> Self {
        SharedStorage {
            inner: Arc::new(storage),
        }
    }

    /// Get reference to the inner storage
    pub fn get(&self) -> &S {
        &self.inner
    }

    /// Get the inner Arc for pointer comparison
    pub fn inner_arc(&self) -> &Arc<S> {
        &self.inner
    }

    /// Get the reference count
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Try to get mutable access (only if refcount is 1)
    pub fn get_mut(&mut self) -> Option<&mut S> {
        Arc::get_mut(&mut self.inner)
    }

    /// Make the storage unique (clone if needed)
    ///
    /// This implements copy-on-write semantics. If the storage is shared (refcount > 1),
    /// it will be cloned to make it unique. Otherwise, returns mutable access to the
    /// existing storage.
    pub fn make_mut(&mut self) -> Result<&mut S>
    where
        S: Sized,
    {
        if Arc::strong_count(&self.inner) > 1 {
            let cloned = self.inner.clone_storage()?;
            self.inner = Arc::new(cloned);
        }
        Ok(Arc::get_mut(&mut self.inner)
            .expect("Arc::get_mut should succeed after ensuring unique reference"))
    }

    /// Check if this shared storage is uniquely owned
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Convert to the inner storage if uniquely owned
    pub fn try_unwrap(self) -> std::result::Result<S, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(storage) => Ok(storage),
            Err(arc) => Err(SharedStorage { inner: arc }),
        }
    }

    /// Get a weak reference to the storage
    pub fn downgrade(&self) -> std::sync::Weak<S> {
        Arc::downgrade(&self.inner)
    }

    /// Create from weak reference if still alive
    pub fn upgrade_from_weak(weak: &std::sync::Weak<S>) -> Option<Self> {
        weak.upgrade().map(|inner| SharedStorage { inner })
    }
}

/// Extension trait for Storage implementations to provide additional convenience methods
pub trait StorageExt: Storage {
    /// Allocate and initialize storage with a default value
    fn allocate_with_value(device: &Self::Device, size: usize, value: Self::Elem) -> Result<Self>
    where
        Self: Sized,
    {
        // Default implementation just allocates - override for efficiency
        let _ = value;
        Self::allocate(device, size)
    }

    /// Allocate storage with zero initialization
    fn allocate_zeros(device: &Self::Device, size: usize) -> Result<Self>
    where
        Self: Sized,
    {
        Self::allocate_with_value(device, size, Self::Elem::zero())
    }

    /// Allocate storage with one initialization
    fn allocate_ones(device: &Self::Device, size: usize) -> Result<Self>
    where
        Self: Sized,
    {
        Self::allocate_with_value(device, size, Self::Elem::one())
    }

    /// Get the memory usage in bytes
    fn memory_usage(&self) -> usize {
        self.len() * std::mem::size_of::<Self::Elem>()
    }

    /// Check if storage is compatible with another storage (same device)
    fn is_compatible_with<Other: Storage>(&self, other: &Other) -> bool
    where
        Self::Device: PartialEq<Other::Device>,
    {
        self.device() == other.device()
    }
}

/// Blanket implementation of StorageExt for all Storage types
impl<T: Storage> StorageExt for T {}

/// Storage factory trait for creating storage instances
pub trait StorageFactory<S: Storage> {
    /// Create storage with default configuration
    fn create_default(device: &S::Device, size: usize) -> Result<S>;

    /// Create storage with specific configuration
    fn create_with_config(device: &S::Device, size: usize, config: &StorageConfig) -> Result<S>;
}

/// Configuration for storage creation
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Initial value for storage elements
    pub initial_value: Option<f32>,
    /// Memory alignment requirements
    pub alignment: Option<usize>,
    /// Whether to use memory pooling
    pub use_pooling: bool,
    /// Whether to clear memory on deallocation (security)
    pub clear_on_dealloc: bool,
    /// Memory mapping hint for large allocations
    pub prefer_mmap: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            initial_value: None,
            alignment: None,
            use_pooling: true,
            clear_on_dealloc: false,
            prefer_mmap: false,
        }
    }
}

impl StorageConfig {
    /// Create new storage configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set initial value for storage elements
    pub fn with_initial_value(mut self, value: f32) -> Self {
        self.initial_value = Some(value);
        self
    }

    /// Set memory alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = Some(alignment);
        self
    }

    /// Enable or disable memory pooling
    pub fn with_pooling(mut self, use_pooling: bool) -> Self {
        self.use_pooling = use_pooling;
        self
    }

    /// Enable or disable memory clearing on deallocation
    pub fn with_clear_on_dealloc(mut self, clear: bool) -> Self {
        self.clear_on_dealloc = clear;
        self
    }

    /// Prefer memory mapping for large allocations
    pub fn with_mmap_preference(mut self, prefer: bool) -> Self {
        self.prefer_mmap = prefer;
        self
    }
}
