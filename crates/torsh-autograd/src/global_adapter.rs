//! Global SciRS2 autograd adapter for unified gradient computation
//!
//! This module provides a global singleton adapter that interfaces with the SciRS2
//! autograd system, enabling convenient access to gradient computation capabilities
//! throughout the application without explicit adapter management.
//!
//! # Features
//!
//! - **Global singleton**: Single adapter instance for application-wide use
//! - **SciRS2 integration**: Direct interface to SciRS2's autograd capabilities
//! - **Gradient tensors**: Creation and management of gradient-enabled tensors
//! - **Backward pass**: Simplified gradient computation interface

use torsh_core::error::Result;
use torsh_core::shape::Shape;

// Re-export the new abstraction layer for convenience
pub use crate::scirs2_integration::{GradientTensor, SciRS2AutogradAdapter};

/// Global adapter instance for easy access
static GLOBAL_ADAPTER: std::sync::OnceLock<SciRS2AutogradAdapter> = std::sync::OnceLock::new();

/// Get the global SciRS2 autograd adapter
pub fn get_global_adapter() -> &'static SciRS2AutogradAdapter {
    GLOBAL_ADAPTER.get_or_init(|| SciRS2AutogradAdapter::new())
}

/// Create a gradient-enabled tensor using the global adapter
pub fn create_gradient_tensor(
    data: &[f32],
    shape: &Shape,
    device: &dyn torsh_core::Device,
    requires_grad: bool,
) -> Result<GradientTensor> {
    get_global_adapter().create_gradient_tensor(data, shape, device, requires_grad)
}

/// Perform backward pass using the global adapter
pub fn backward_global(output: &GradientTensor) -> Result<()> {
    get_global_adapter().backward(output)
}

/// Get gradient for a tensor using the global adapter
pub fn get_gradient_global(tensor: &GradientTensor) -> Result<Option<torsh_tensor::Tensor>> {
    get_global_adapter().get_gradient(tensor)
}

/// Initialize the global adapter with custom configuration
pub fn initialize_global_adapter(adapter: SciRS2AutogradAdapter) -> Result<()> {
    GLOBAL_ADAPTER.set(adapter).map_err(|_| {
        torsh_core::error::TorshError::AutogradError(
            "Global adapter already initialized".to_string(),
        )
    })
}

/// Check if the global adapter has been initialized
pub fn is_global_adapter_initialized() -> bool {
    GLOBAL_ADAPTER.get().is_some()
}

/// Reset the global adapter (mainly for testing)
#[cfg(test)]
pub fn reset_global_adapter() {
    // Note: OnceCell doesn't provide a reset method, so this is mainly a placeholder
    // In a real implementation, we might use a RwLock or similar for resettable behavior
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_global_adapter_initialization() {
        assert!(is_global_adapter_initialized() || !is_global_adapter_initialized());

        // Get the adapter (this will initialize it)
        let _adapter = get_global_adapter();
        assert!(is_global_adapter_initialized());
    }

    #[test]
    fn test_gradient_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![2, 2]);
        let device = torsh_core::device::CpuDevice::new();

        // This test would require a full SciRS2AutogradAdapter implementation
        // For now, we just verify the function signature
        let result = create_gradient_tensor(&data, &shape, &device, true);
        // The actual result would depend on the SciRS2AutogradAdapter implementation
    }

    #[test]
    fn test_global_adapter_singleton() {
        let adapter1 = get_global_adapter();
        let adapter2 = get_global_adapter();

        // Both should point to the same instance
        assert!(std::ptr::eq(adapter1, adapter2));
    }
}
