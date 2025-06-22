//! Backend abstraction layer for ToRSh
//!
//! This crate provides the abstract interfaces and traits that all ToRSh
//! compute backends must implement. It defines the core backend functionality
//! for tensor operations, memory management, and kernel execution.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod backend;
pub mod buffer;
pub mod device;
pub mod kernel;
pub mod memory;
pub mod profiler;

// Re-exports
pub use backend::{Backend, BackendError, BackendResult};
pub use buffer::{Buffer, BufferDescriptor, BufferUsage};
pub use device::{Device, DeviceInfo, DeviceType};
pub use kernel::{Kernel, KernelDescriptor, KernelLaunchConfig};
pub use memory::{MemoryManager, MemoryPool, MemoryStats};
pub use profiler::{Profiler, ProfilerEvent, ProfilerStats};

use torsh_core::{device::DeviceType as CoreDeviceType, error::Result};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Global backend registry
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
    default_backend: Option<usize>,
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            default_backend: None,
        }
    }

    /// Register a backend
    pub fn register<B: Backend + 'static>(&mut self, backend: B) -> Result<()> {
        let is_available = backend.is_available()?;
        if is_available {
            self.backends.push(Box::new(backend));

            // Set as default if it's the first available backend
            if self.default_backend.is_none() {
                self.default_backend = Some(self.backends.len() - 1);
            }
        }
        Ok(())
    }

    /// Get the default backend
    pub fn default_backend(&self) -> Option<&dyn Backend> {
        self.default_backend
            .and_then(|idx| self.backends.get(idx))
            .map(|b| b.as_ref())
    }

    /// Get a backend by device type
    pub fn get_backend(&self, device_type: CoreDeviceType) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.device_type() == device_type)
            .map(|b| b.as_ref())
    }

    /// List all available backends
    pub fn available_backends(&self) -> Vec<&dyn Backend> {
        self.backends.iter().map(|b| b.as_ref()).collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global backend registry instance
static mut BACKEND_REGISTRY: Option<BackendRegistry> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global backend registry
pub fn init_backends() {
    INIT.call_once(|| unsafe {
        BACKEND_REGISTRY = Some(BackendRegistry::new());
    });
}

/// Get the global backend registry
#[allow(static_mut_refs)]
pub fn backend_registry() -> &'static BackendRegistry {
    init_backends();
    unsafe { BACKEND_REGISTRY.as_ref().unwrap() }
}

/// Get a mutable reference to the global backend registry
#[allow(static_mut_refs)]
pub fn backend_registry_mut() -> &'static mut BackendRegistry {
    init_backends();
    unsafe { BACKEND_REGISTRY.as_mut().unwrap() }
}

/// Register a backend globally
pub fn register_backend<B: Backend + 'static>(backend: B) -> Result<()> {
    backend_registry_mut().register(backend)
}

/// Get the default backend
pub fn default_backend() -> Option<&'static dyn Backend> {
    backend_registry().default_backend()
}

/// Get a backend by device type
pub fn get_backend(device_type: CoreDeviceType) -> Option<&'static dyn Backend> {
    backend_registry().get_backend(device_type)
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        default_backend, get_backend, register_backend, Backend, BackendResult, Buffer, Device,
        Kernel, MemoryManager,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_registry() {
        let mut registry = BackendRegistry::new();
        assert!(registry.default_backend().is_none());
        assert!(registry.available_backends().is_empty());
    }
}
