//! Traits for custom autograd functions
//!
//! This module provides the core traits that define the interface for custom autograd functions,
//! including differentiable and non-differentiable functions with subgradients.

use crate::AutogradTensor;
use super::types::{FunctionContext, FunctionMetadata, MemoryComplexity, ComputationalComplexity, SubgradientSet, SubgradientSelection};
use torsh_core::error::Result;
use std::time::SystemTime;

/// Type-erased trait for custom autograd functions - dyn-compatible version
pub trait DynFunction: Send + Sync {
    /// Name of the function for debugging and profiling
    fn name(&self) -> &str;

    /// Whether this function is differentiable
    fn is_differentiable(&self) -> bool {
        true
    }

    /// Memory complexity hint for optimization
    fn memory_complexity(&self) -> MemoryComplexity {
        MemoryComplexity::Linear
    }

    /// Computational complexity hint for optimization
    fn computational_complexity(&self) -> ComputationalComplexity {
        ComputationalComplexity::Linear
    }

    /// Whether this function can be fused with others
    fn is_fusable(&self) -> bool {
        false
    }

    /// Get function metadata for optimization
    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            name: self.name().to_string(),
            is_differentiable: self.is_differentiable(),
            memory_complexity: self.memory_complexity(),
            computational_complexity: self.computational_complexity(),
            is_fusable: self.is_fusable(),
            version: "1.0.0".to_string(),
            description: "Custom autograd function".to_string(),
            author: "Unknown".to_string(),
            created_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            checksum: "".to_string(),
            dependencies: vec![],
        }
    }
}

/// Trait for custom autograd functions with generic methods
pub trait Function: Send + Sync + DynFunction {
    /// Forward pass computation
    fn forward<T>(
        &self,
        ctx: &mut FunctionContext,
        inputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
    where
        T: torsh_core::dtype::TensorElement;

    /// Backward pass computation
    fn backward<T>(
        &self,
        ctx: &mut FunctionContext,
        grad_outputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>>
    where
        T: torsh_core::dtype::TensorElement;
}

/// Trait for non-differentiable functions that have subgradients
pub trait SubgradientFunction: Send + Sync + DynFunction {
    /// Forward pass computation
    fn forward<T>(
        &self,
        ctx: &mut FunctionContext,
        inputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
    where
        T: torsh_core::dtype::TensorElement + num_traits::Float;

    /// Subgradient computation for non-differentiable operations
    /// Returns a set of possible subgradients
    fn subgradient<T>(
        &self,
        ctx: &mut FunctionContext,
        grad_outputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Option<SubgradientSet<T>>>>
    where
        T: torsh_core::dtype::TensorElement + num_traits::Float;
}