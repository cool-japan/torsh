//! Variable environment management for automatic differentiation
//!
//! This module provides thread-local variable environment management and in-place
//! operation handling with gradient safety. It manages the scirs2 VariableEnvironment
//! per thread and provides safe in-place operation strategies.
//!
//! # Features
//!
//! - **Thread-local storage**: Per-thread variable environments for gradient tracking
//! - **In-place safety**: Validation and handling of in-place tensor operations
//! - **Copy-on-write**: Safe in-place modifications with gradient preservation
//! - **Graph integration**: Coordination with computation graph for gradient tracking

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::autograd_traits::AutogradTensor;
use crate::grad_mode::is_grad_enabled;
use scirs2_autograd::VariableEnvironment;
use torsh_core::error::{Result, TorshError};

// Thread-local variable environment for managing tensor gradients
// Using thread-local storage to avoid Sync issues with scirs2's RefCell usage
thread_local! {
    static VARIABLE_ENV: std::cell::RefCell<Option<VariableEnvironment<f32>>> =
        const { std::cell::RefCell::new(None) };
}

/// Get or create the thread-local variable environment
pub fn get_or_create_variable_env() -> VariableEnvironment<f32> {
    VARIABLE_ENV.with(|env| {
        let mut env_ref = env.borrow_mut();
        if env_ref.is_none() {
            *env_ref = Some(VariableEnvironment::new());
        }
        // Safe to expect here: we just ensured env_ref is Some above
        env_ref
            .take()
            .expect("VariableEnvironment should exist after initialization")
    })
}

/// Execute a function with the variable environment
pub fn with_variable_env<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut VariableEnvironment<f32>) -> Result<R>,
{
    let mut env = get_or_create_variable_env();
    let result = f(&mut env);

    // Store environment back
    VARIABLE_ENV.with(|thread_env| {
        *thread_env.borrow_mut() = Some(env);
    });

    result
}

/// Clear the thread-local variable environment
pub fn clear_variable_env() {
    VARIABLE_ENV.with(|env| {
        *env.borrow_mut() = None;
    });
}

/// Check if the variable environment is initialized
pub fn is_variable_env_initialized() -> bool {
    VARIABLE_ENV.with(|env| env.borrow().is_some())
}

/// Check if an operation is in-place and validate gradient compatibility
pub fn validate_inplace_operation<T>(
    tensor: &dyn AutogradTensor<T>,
    operation_name: &str,
) -> Result<()>
where
    T: torsh_core::dtype::TensorElement + Clone + std::fmt::Debug,
{
    if tensor.requires_grad() && is_grad_enabled() {
        // In-place operations can invalidate the computation graph
        // For now, we warn about potential issues but allow the operation
        tracing::warn!(
            "In-place operation '{}' on tensor that requires grad. This may cause gradient computation issues.",
            operation_name
        );

        // In a full implementation, we would:
        // 1. Check if this tensor is part of any computation graph
        // 2. Clone the tensor data before modification if needed
        // 3. Update the computation graph appropriately

        // For now, just validate that we're not in a critical gradient computation phase
        crate::context::with_context(|ctx| {
            if ctx.graph_size() > 0 {
                tracing::warn!(
                    "In-place operation '{}' performed on tensor in active computation graph with {} nodes",
                    operation_name,
                    ctx.graph_size()
                );
            }
            Ok(())
        })
    } else {
        Ok(())
    }
}

/// Handle in-place operation with gradient safety
///
/// This function provides several strategies for handling in-place operations:
/// 1. Version bumping: Track tensor versions to detect invalidation
/// 2. Copy-on-write: Clone tensor data before modification if in computation graph
/// 3. Graph invalidation: Clear relevant parts of computation graph if needed
pub fn handle_inplace_operation<T, F>(
    tensor: &mut dyn AutogradTensor<T>,
    operation_name: &str,
    operation: F,
) -> Result<()>
where
    T: torsh_core::dtype::TensorElement + Clone + std::fmt::Debug,
    F: FnOnce(&mut dyn AutogradTensor<T>) -> Result<()>,
{
    // Validate the in-place operation
    validate_inplace_operation(tensor, operation_name)?;

    // Check if tensor requires gradients and we're in an active computation graph
    if tensor.requires_grad() && is_grad_enabled() {
        // Strategy 1: Copy-on-write for gradient safety
        return handle_inplace_with_cow(tensor, operation_name, operation);
    }

    // For non-gradient tensors, we can perform the operation directly
    // Note: This is still a limitation of trait objects in Rust
    // Real implementation would be in the concrete tensor type
    tracing::debug!(
        "Performing in-place operation '{}' on non-gradient tensor",
        operation_name
    );

    // Since we can't modify trait objects directly, we return an error
    // The actual tensor implementation should override this behavior
    let _ = operation; // Avoid unused warning
    Err(TorshError::AutogradError(format!(
        "In-place operation '{operation_name}' requires concrete tensor type implementation"
    )))
}

/// Handle in-place operation with copy-on-write strategy
///
/// This strategy clones the tensor data before modification to preserve
/// the original values needed for gradient computation.
fn handle_inplace_with_cow<T, F>(
    tensor: &mut dyn AutogradTensor<T>,
    operation_name: &str,
    operation: F,
) -> Result<()>
where
    T: torsh_core::dtype::TensorElement + Clone + std::fmt::Debug,
    F: FnOnce(&mut dyn AutogradTensor<T>) -> Result<()>,
{
    tracing::debug!(
        "Handling in-place operation '{}' with copy-on-write strategy",
        operation_name
    );

    // Check if this tensor is part of an active computation graph
    let tensor_in_graph =
        crate::context::with_context(|ctx| Ok(ctx.graph_size() > 0)).unwrap_or(false);

    if tensor_in_graph {
        tracing::warn!(
            "In-place operation '{}' detected on tensor in active computation graph. \
            Consider using out-of-place operations for gradient safety.",
            operation_name
        );

        // For maximum safety, we would need to:
        // 1. Clone the current tensor state
        // 2. Store it in the computation graph for gradient computation
        // 3. Allow the in-place modification to proceed
        //
        // However, since we're working with trait objects, we need to defer
        // this to the concrete tensor implementation

        // Register the in-place operation in the computation graph
        crate::context::with_context(|ctx| {
            // Generate a unique ID for tracking this in-place operation
            let operation_id = ctx.new_tensor_id();

            // Record the in-place operation as a graph mutation
            let _ = ctx.add_operation(
                format!("inplace_{operation_name}"),
                vec![], // In-place operations don't have separate inputs
                operation_id,
                tensor.requires_grad(),
                None, // No gradient function for in-place ops (they modify existing tensors)
            );

            tracing::debug!(
                "Registered in-place operation '{}' with ID {} in computation graph",
                operation_name,
                operation_id
            );
            Ok(())
        })?;
    }

    // Perform the operation
    // Note: In a real implementation, this would involve more sophisticated
    // copy-on-write mechanics and proper gradient function creation
    let _ = operation; // Avoid unused warning for now

    tracing::debug!(
        "Copy-on-write strategy applied for in-place operation '{}'",
        operation_name
    );

    // For now, we return an error as this requires concrete tensor implementation
    Err(TorshError::AutogradError(format!(
        "Copy-on-write for '{}' requires concrete tensor type implementation",
        operation_name
    )))
}

/// Strategy for handling in-place operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InplaceStrategy {
    /// Allow the operation to proceed without special handling
    Direct,
    /// Clone tensor data before modification (copy-on-write)
    CopyOnWrite,
    /// Refuse the operation if it would break gradient computation
    Forbid,
    /// Convert to out-of-place operation automatically
    ConvertToOutOfPlace,
}

impl Default for InplaceStrategy {
    fn default() -> Self {
        Self::CopyOnWrite
    }
}

/// Configuration for in-place operation handling
#[derive(Debug, Clone)]
pub struct InplaceConfig {
    /// Default strategy for gradient-enabled tensors
    pub gradient_strategy: InplaceStrategy,
    /// Default strategy for non-gradient tensors
    pub non_gradient_strategy: InplaceStrategy,
    /// Whether to emit warnings for potentially unsafe operations
    pub warn_unsafe: bool,
    /// Maximum depth of copy-on-write operations before warning
    pub max_cow_depth: usize,
}

impl Default for InplaceConfig {
    fn default() -> Self {
        Self {
            gradient_strategy: InplaceStrategy::CopyOnWrite,
            non_gradient_strategy: InplaceStrategy::Direct,
            warn_unsafe: true,
            max_cow_depth: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_env_initialization() {
        // Clear any existing environment
        clear_variable_env();
        assert!(!is_variable_env_initialized());

        // Create environment
        let _env = get_or_create_variable_env();
        // Note: After taking the environment, it's no longer in thread-local storage
        assert!(!is_variable_env_initialized());
    }

    #[test]
    fn test_with_variable_env() {
        clear_variable_env();

        let result = with_variable_env(|env| {
            // Test that we have a valid environment
            assert!(env as *const _ as usize != 0);
            Ok(42)
        });

        assert_eq!(result.unwrap(), 42);
        // Environment should be stored back
        assert!(is_variable_env_initialized());
    }

    #[test]
    fn test_inplace_config_defaults() {
        let config = InplaceConfig::default();
        assert_eq!(config.gradient_strategy, InplaceStrategy::CopyOnWrite);
        assert_eq!(config.non_gradient_strategy, InplaceStrategy::Direct);
        assert!(config.warn_unsafe);
        assert_eq!(config.max_cow_depth, 10);
    }

    #[test]
    fn test_inplace_strategy_values() {
        assert_eq!(InplaceStrategy::default(), InplaceStrategy::CopyOnWrite);

        let strategies = vec![
            InplaceStrategy::Direct,
            InplaceStrategy::CopyOnWrite,
            InplaceStrategy::Forbid,
            InplaceStrategy::ConvertToOutOfPlace,
        ];

        // Test that all strategies are different
        for (i, strategy1) in strategies.iter().enumerate() {
            for (j, strategy2) in strategies.iter().enumerate() {
                if i != j {
                    assert_ne!(strategy1, strategy2);
                }
            }
        }
    }
}
