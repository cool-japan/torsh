//! Collate builder and strategy definitions

use super::{
    advanced::{CachedCollate, DynamicBatchCollateWrapper, PadCollate},
    core::DefaultCollate,
    optimized::OptimizedCollate,
};
use crate::collate::Collate;
use torsh_core::dtype::TensorElement;
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Different collation strategies
#[derive(Debug, Clone, Copy)]
pub enum CollateStrategy {
    /// Simple stacking (default)
    Stack,
    /// Optimized for performance
    Optimized,
    /// Variable-length sequences with padding
    Padding,
    /// Dynamic batching
    Dynamic,
    /// Cached collation for repeated use
    Cached,
}

/// Unified collate builder for creating collate functions with different strategies
pub struct CollateBuilder<T> {
    strategy: CollateStrategy,
    padding_value: Option<T>,
    max_length: Option<usize>,
    use_caching: bool,
    batch_size_hint: Option<usize>,
}

impl<T: TensorElement> Default for CollateBuilder<T> {
    fn default() -> Self {
        Self {
            strategy: CollateStrategy::Stack,
            padding_value: None,
            max_length: None,
            use_caching: false,
            batch_size_hint: None,
        }
    }
}

impl<
        T: TensorElement
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Default,
    > CollateBuilder<T>
{
    /// Create a new collate builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the collation strategy
    pub fn strategy(mut self, strategy: CollateStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set padding value for variable-length sequences
    pub fn with_padding(mut self, padding_value: T) -> Self {
        self.padding_value = Some(padding_value);
        self
    }

    /// Set maximum sequence length
    pub fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Enable caching for better performance
    pub fn with_caching(mut self) -> Self {
        self.use_caching = true;
        self
    }

    /// Provide batch size hint for optimization
    pub fn batch_size_hint(mut self, size: usize) -> Self {
        self.batch_size_hint = Some(size);
        self
    }

    /// Build the collate function
    pub fn build(self) -> Box<dyn Collate<Tensor<T>, Output = Tensor<T>> + Send + Sync>
    where
        T: Copy + 'static,
    {
        match self.strategy {
            CollateStrategy::Stack => Box::new(DefaultCollate),
            CollateStrategy::Optimized => Box::new(OptimizedCollate),
            CollateStrategy::Padding => {
                let padding_value = self.padding_value.unwrap_or_default();
                Box::new(PadCollate::new(padding_value))
            }
            CollateStrategy::Dynamic => {
                let padding_value = self.padding_value.unwrap_or_default();
                Box::new(DynamicBatchCollateWrapper::new(padding_value))
            }
            CollateStrategy::Cached => {
                if cfg!(feature = "std") {
                    Box::new(CachedCollate::new(1000))
                } else {
                    // Fallback to optimized for no_std
                    Box::new(OptimizedCollate)
                }
            }
        }
    }
}
