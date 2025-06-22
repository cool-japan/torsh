//! Storage abstraction for tensor data

use crate::device::Device;
use crate::dtype::TensorElement;
use crate::error::Result;
use std::sync::Arc;

/// Backend-agnostic storage for tensor data
pub trait Storage: Send + Sync + 'static {
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
#[derive(Clone)]
pub struct SharedStorage<S: Storage> {
    inner: Arc<S>,
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
    
    /// Get the reference count
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
    
    /// Try to get mutable access (only if refcount is 1)
    pub fn get_mut(&mut self) -> Option<&mut S> {
        Arc::get_mut(&mut self.inner)
    }
    
    /// Make the storage unique (clone if needed)
    pub fn make_mut(&mut self) -> Result<&mut S> 
    where 
        S: Sized 
    {
        if Arc::strong_count(&self.inner) > 1 {
            let cloned = self.inner.clone_storage()?;
            self.inner = Arc::new(cloned);
        }
        Ok(Arc::get_mut(&mut self.inner).unwrap())
    }
}

/// Memory format for tensor storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryFormat {
    /// Standard row-major (C-contiguous) format
    #[default]
    Contiguous,
    /// Channel-last format for 4D tensors (NHWC)
    ChannelsLast,
    /// Channel-last format for 5D tensors (NDHWC)
    ChannelsLast3d,
    /// Custom strided format
    Strided,
}

