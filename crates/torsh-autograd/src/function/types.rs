//! Type definitions for custom autograd functions
//!
//! This module provides the core type definitions, enums, and data structures
//! used by the custom autograd function framework.

use crate::AutogradTensor;
use serde::{Deserialize, Serialize};
use std::any::Any;
use torsh_core::error::{Result, TorshError};

/// Set of subgradients for non-differentiable functions
pub struct SubgradientSet<T: torsh_core::dtype::TensorElement> {
    /// Primary subgradient (commonly used one)
    pub primary: Box<dyn AutogradTensor<T>>,
    /// Alternative subgradients
    pub alternatives: Vec<Box<dyn AutogradTensor<T>>>,
    /// Selection strategy for choosing subgradient
    pub selection_strategy: SubgradientSelection,
}

impl<T: torsh_core::dtype::TensorElement> Clone for SubgradientSet<T> {
    fn clone(&self) -> Self {
        Self {
            primary: self.primary.clone_tensor(),
            alternatives: self.alternatives.iter().map(|t| t.clone_tensor()).collect(),
            selection_strategy: self.selection_strategy,
        }
    }
}

impl<T: torsh_core::dtype::TensorElement> std::fmt::Debug for SubgradientSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubgradientSet")
            .field(
                "primary",
                &format!("Box<dyn AutogradTensor<{}>>", std::any::type_name::<T>()),
            )
            .field(
                "alternatives",
                &format!(
                    "Vec<Box<dyn AutogradTensor<{}>>> (len: {})",
                    std::any::type_name::<T>(),
                    self.alternatives.len()
                ),
            )
            .field("selection_strategy", &self.selection_strategy)
            .finish()
    }
}

/// Strategy for selecting subgradients in non-differentiable operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgradientSelection {
    /// Always use the primary subgradient
    Primary,
    /// Choose randomly from available subgradients
    Random,
    /// Use the subgradient with minimum norm
    MinNorm,
    /// Use the subgradient with maximum norm
    MaxNorm,
    /// Use Clarke subgradient (convex hull)
    Clarke,
    /// Use generalized gradient (for locally Lipschitz functions)
    Generalized,
}

/// Memory complexity categories for function optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryComplexity {
    Constant,
    Linear,
    Quadratic,
    Exponential,
}

/// Computational complexity categories for function optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    Constant,
    Linear,
    LogLinear,
    Quadratic,
    Cubic,
    Exponential,
}

/// Function metadata for optimization and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    pub name: String,
    pub is_differentiable: bool,
    pub memory_complexity: MemoryComplexity,
    pub computational_complexity: ComputationalComplexity,
    pub is_fusable: bool,
    pub version: String,
    pub description: String,
    pub author: String,
    pub created_at: String,
    pub checksum: String,
    pub dependencies: Vec<String>,
}

/// Context for storing values between forward and backward passes
pub struct FunctionContext {
    /// Saved tensors for backward pass
    #[allow(dead_code)]
    saved_tensors: Vec<Box<dyn Any + Send + Sync>>,
    /// Saved scalar values for backward pass
    saved_values: Vec<Box<dyn Any + Send + Sync>>,
    /// Whether to materialize gradients for non-differentiable tensors
    materialize_grads: bool,
    /// Unique context ID for debugging
    context_id: usize,
    /// Function name for debugging
    function_name: String,
}

impl Default for FunctionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionContext {
    /// Create a new function context
    pub fn new() -> Self {
        static CONTEXT_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        Self {
            saved_tensors: Vec::new(),
            saved_values: Vec::new(),
            materialize_grads: true,
            context_id: CONTEXT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            function_name: "unknown".to_string(),
        }
    }

    /// Create a new function context with a specific name
    pub fn with_name(name: String) -> Self {
        let mut ctx = Self::new();
        ctx.function_name = name;
        ctx
    }

    /// Save arbitrary values for backward pass
    pub fn save_value<V: Any + Send + Sync + 'static>(&mut self, value: V) {
        self.saved_values.push(Box::new(value));
    }

    /// Get saved value by index
    pub fn get_saved_value<V: Any + 'static>(&self, index: usize) -> Result<&V> {
        self.saved_values
            .get(index)
            .and_then(|v| v.downcast_ref::<V>())
            .ok_or_else(|| {
                TorshError::AutogradError(format!(
                    "Saved value at index {} not found or type mismatch in context {}",
                    index, self.context_id
                ))
            })
    }

    /// Get the context ID
    pub fn context_id(&self) -> usize {
        self.context_id
    }

    /// Get the function name
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Set whether to materialize gradients
    pub fn set_materialize_grads(&mut self, materialize: bool) {
        self.materialize_grads = materialize;
    }

    /// Check if gradients should be materialized
    pub fn should_materialize_grads(&self) -> bool {
        self.materialize_grads
    }
}