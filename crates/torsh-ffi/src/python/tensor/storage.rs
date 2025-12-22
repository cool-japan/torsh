use crate::error::{FfiError, FfiResult};
use crate::python::tensor::memory::MEMORY_POOL;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use torsh_core::DType;

/// Reference-counted tensor storage for memory management
#[derive(Debug, Clone)]
pub struct TensorStorage {
    data: Arc<Mutex<Vec<f32>>>,
    shape: Vec<usize>,
    dtype: DType,
    /// Track if this tensor was created from external memory (e.g., NumPy)
    is_external: bool,
    /// Gradient storage (None if no gradient computed yet)
    grad: Arc<Mutex<Option<Vec<f32>>>>,
    /// Version for gradient tracking (incremented on in-place operations)
    /// Using AtomicUsize for lock-free version tracking
    version: Arc<AtomicUsize>,
}

impl TensorStorage {
    /// Create new tensor storage with data
    pub fn new(data: Vec<f32>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            shape,
            dtype,
            is_external: false,
            grad: Arc::new(Mutex::new(None)),
            version: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Create new tensor storage with pooled allocation
    pub fn new_pooled(shape: Vec<usize>, dtype: DType) -> FfiResult<Self> {
        let size = shape.iter().product();
        let data = MEMORY_POOL.allocate(size)?;

        Ok(Self {
            data: Arc::new(Mutex::new(data)),
            shape,
            dtype,
            is_external: false,
            grad: Arc::new(Mutex::new(None)),
            version: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create tensor storage from external data (e.g., NumPy array)
    pub fn from_external(data: Vec<f32>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            shape,
            dtype,
            is_external: true,
            grad: Arc::new(Mutex::new(None)),
            version: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get a reference to the data (read-only)
    pub fn data(&self) -> parking_lot::MutexGuard<'_, Vec<f32>> {
        self.data.lock()
    }

    /// Get a mutable reference to the data
    pub fn data_mut(&self) -> parking_lot::MutexGuard<'_, Vec<f32>> {
        self.data.lock()
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Create a view with different shape (same data)
    pub fn view(&self, new_shape: Vec<usize>) -> FfiResult<Self> {
        let total_elements: usize = new_shape.iter().product();
        let current_elements: usize = self.shape.iter().product();

        if total_elements != current_elements {
            return Err(FfiError::ShapeMismatch {
                expected: vec![current_elements],
                actual: vec![total_elements],
            });
        }

        Ok(Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            dtype: self.dtype,
            is_external: self.is_external,
            grad: Arc::clone(&self.grad),
            version: Arc::clone(&self.version),
        })
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Check if this storage comes from external memory
    pub fn is_external(&self) -> bool {
        self.is_external
    }

    /// Get gradient data (None if no gradient computed)
    pub fn grad(&self) -> parking_lot::MutexGuard<'_, Option<Vec<f32>>> {
        self.grad.lock()
    }

    /// Set gradient data
    pub fn set_grad(&self, grad_data: Option<Vec<f32>>) {
        *self.grad.lock() = grad_data;
    }

    /// Clear gradient
    pub fn clear_grad(&self) {
        *self.grad.lock() = None;
    }

    /// Get current version (for gradient tracking)
    /// Uses atomic load for lock-free read access
    pub fn version(&self) -> usize {
        self.version.load(Ordering::Relaxed)
    }

    /// Increment version (called on in-place operations)
    /// Uses atomic increment for lock-free update
    pub fn increment_version(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for TensorStorage {
    fn drop(&mut self) {
        // Only return to pool if this is the last reference and not external memory
        if !self.is_external && Arc::strong_count(&self.data) == 1 {
            if let Ok(data) = Arc::try_unwrap(std::mem::replace(
                &mut self.data,
                Arc::new(Mutex::new(Vec::new())),
            )) {
                // parking_lot::Mutex::into_inner() returns the value directly (no Result)
                let data_vec = data.into_inner();
                // Attempt to return to pool (ignore errors during drop)
                let _ = MEMORY_POOL.deallocate(data_vec);
            }
        }
    }
}
