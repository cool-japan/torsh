//! Utility functions and types for collation

use super::{core::DefaultCollate, Collate};
use torsh_core::error::Result;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Collation function that can be customized
pub struct CollateFn<F> {
    func: F,
}

impl<F> CollateFn<F> {
    /// Create a new collation function
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<T, O, F> Collate<T> for CollateFn<F>
where
    F: Fn(Vec<T>) -> Result<O>,
{
    type Output = O;

    fn collate(&self, batch: Vec<T>) -> Result<Self::Output> {
        (self.func)(batch)
    }
}

/// Default collation function instance
pub fn collate_fn<T>() -> DefaultCollate {
    DefaultCollate
}
