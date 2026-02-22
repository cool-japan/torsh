//! Gradient storage management for automatic differentiation
//!
//! This module provides unified gradient storage capabilities across different tensor types,
//! with thread-safe operations and efficient memory management for gradient computation.
//!
//! # Features
//!
//! - **Thread-safe storage**: Concurrent access to gradients with RwLock protection
//! - **Unified interface**: Single trait for all gradient storage implementations
//! - **Memory efficient**: Lazy storage allocation and cleanup capabilities
//! - **Type flexibility**: Support for different tensor element types

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::error::{Result, TorshError};

/// Unified trait for gradient storage across different tensor types
pub trait GradientStorage: Send + Sync {
    /// Store gradient for a tensor
    fn store_gradient(&self, tensor_id: usize, gradient: torsh_tensor::Tensor) -> Result<()>;

    /// Retrieve gradient for a tensor
    fn get_gradient(&self, tensor_id: usize) -> Result<Option<torsh_tensor::Tensor>>;

    /// Check if gradient exists for a tensor
    fn has_gradient(&self, tensor_id: usize) -> bool;

    /// Clear gradient for a tensor
    fn clear_gradient(&self, tensor_id: usize) -> Result<()>;

    /// Clear all gradients
    fn clear_all_gradients(&self) -> Result<()>;

    /// Get all tensor IDs that have gradients
    fn gradient_tensor_ids(&self) -> Vec<usize>;
}

/// Thread-safe unified gradient storage implementation
#[derive(Debug, Clone)]
pub struct UnifiedGradientStorage {
    gradients: Arc<RwLock<HashMap<usize, torsh_tensor::Tensor>>>,
}

impl UnifiedGradientStorage {
    /// Create a new unified gradient storage
    pub fn new() -> Self {
        Self {
            gradients: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for UnifiedGradientStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientStorage for UnifiedGradientStorage {
    fn store_gradient(&self, tensor_id: usize, gradient: torsh_tensor::Tensor) -> Result<()> {
        let mut gradients = self.gradients.write().map_err(|e| {
            TorshError::AutogradError(format!("Failed to acquire write lock for gradients: {}", e))
        })?;
        gradients.insert(tensor_id, gradient);
        Ok(())
    }

    fn get_gradient(&self, tensor_id: usize) -> Result<Option<torsh_tensor::Tensor>> {
        let gradients = self.gradients.read().map_err(|e| {
            TorshError::AutogradError(format!("Failed to acquire read lock for gradients: {}", e))
        })?;
        Ok(gradients.get(&tensor_id).cloned())
    }

    fn has_gradient(&self, tensor_id: usize) -> bool {
        self.gradients
            .read()
            .map(|gradients| gradients.contains_key(&tensor_id))
            .unwrap_or(false)
    }

    fn clear_gradient(&self, tensor_id: usize) -> Result<()> {
        let mut gradients = self.gradients.write().map_err(|e| {
            TorshError::AutogradError(format!("Failed to acquire write lock for gradients: {}", e))
        })?;
        gradients.remove(&tensor_id);
        Ok(())
    }

    fn clear_all_gradients(&self) -> Result<()> {
        let mut gradients = self.gradients.write().map_err(|e| {
            TorshError::AutogradError(format!("Failed to acquire write lock for gradients: {}", e))
        })?;
        gradients.clear();
        Ok(())
    }

    fn gradient_tensor_ids(&self) -> Vec<usize> {
        self.gradients
            .read()
            .map(|gradients| gradients.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Global gradient storage instance
static GLOBAL_GRADIENT_STORAGE: once_cell::sync::Lazy<UnifiedGradientStorage> =
    once_cell::sync::Lazy::new(|| UnifiedGradientStorage::new());

/// Get the global gradient storage instance
pub fn get_gradient_storage() -> &'static UnifiedGradientStorage {
    &GLOBAL_GRADIENT_STORAGE
}

/// Type alias for convenience
pub type GlobalGradientStorage = UnifiedGradientStorage;

/// Type alias for convenience
pub type HashMapGradientStorage = UnifiedGradientStorage;

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_gradient_storage_basic_operations() {
        let storage = UnifiedGradientStorage::new();

        // Test that no gradient exists initially
        assert!(!storage.has_gradient(1));
        assert!(storage.get_gradient(1).unwrap().is_none());

        // Create a test tensor and store its gradient
        let gradient =
            torsh_tensor::Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
                .unwrap();

        storage.store_gradient(1, gradient.clone()).unwrap();

        // Test that gradient now exists
        assert!(storage.has_gradient(1));
        let retrieved = storage.get_gradient(1).unwrap().unwrap();
        assert_eq!(retrieved.data().unwrap(), vec![1.0f32, 2.0, 3.0]);

        // Test gradient IDs
        let ids = storage.gradient_tensor_ids();
        assert_eq!(ids, vec![1]);

        // Test clearing specific gradient
        storage.clear_gradient(1).unwrap();
        assert!(!storage.has_gradient(1));

        // Test clear all gradients
        storage.store_gradient(1, gradient.clone()).unwrap();
        storage.store_gradient(2, gradient).unwrap();
        assert_eq!(storage.gradient_tensor_ids().len(), 2);

        storage.clear_all_gradients().unwrap();
        assert_eq!(storage.gradient_tensor_ids().len(), 0);
    }

    #[test]
    fn test_gradient_storage_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let storage = Arc::new(UnifiedGradientStorage::new());
        let storage_clone = storage.clone();

        // Test concurrent access
        let handle = thread::spawn(move || {
            let gradient =
                torsh_tensor::Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu).unwrap();
            storage_clone.store_gradient(1, gradient).unwrap();
        });

        handle.join().unwrap();
        assert!(storage.has_gradient(1));
    }
}
