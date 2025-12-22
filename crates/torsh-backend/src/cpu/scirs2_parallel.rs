//! SciRS2 Parallel Operations Integration (Phase 1)
//!
//! This module provides parallel operations using scirs2-core's parallel processing
//! capabilities when the `scirs2-parallel` feature is enabled. Falls back to rayon
//! when the feature is not enabled.
//!
//! ## SciRS2 POLICY Compliance
//! This module replaces direct rayon usage with scirs2-core::parallel_ops abstractions
//! where possible, while maintaining backward compatibility with rayon for a gradual migration.

// Re-export rayon functionality for now, with a path towards SciRS2
// Once scirs2-core::parallel_ops stabilizes, we'll replace these implementations
pub use rayon::current_num_threads;
pub use rayon::prelude::*;

/// Get the number of threads available for parallel operations
///
/// # SciRS2 POLICY
/// This function will be migrated to use scirs2_core::parallel_ops::get_num_threads()
/// once the API stabilizes. For now, it uses rayon as a proven parallel framework.
#[inline]
pub fn get_parallel_threads() -> usize {
    current_num_threads()
}

/// Execute a parallel for loop over a range
///
/// # SciRS2 POLICY
/// This function will be migrated to use scirs2_core::parallel_ops primitives
/// once the API stabilizes. For now, it uses rayon's proven parallel iterator trait.
#[inline]
pub fn parallel_for_range<F>(start: usize, end: usize, op: F)
where
    F: Fn(usize) + Send + Sync,
{
    (start..end).into_par_iter().for_each(op);
}

/// Execute a parallel map over a range
///
/// # SciRS2 POLICY
/// This function will be migrated to use scirs2_core::parallel_ops primitives
/// once the API stabilizes. For now, it uses rayon's proven parallel iterator trait.
#[inline]
pub fn parallel_map_range<F, R>(start: usize, end: usize, op: F) -> Vec<R>
where
    F: Fn(usize) -> R + Send + Sync,
    R: Send,
{
    (start..end).into_par_iter().map(op).collect()
}

/// Module for convenience imports
pub mod prelude {
    pub use super::{get_parallel_threads, parallel_for_range, parallel_map_range};
    pub use rayon::prelude::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_threads() {
        let num = get_parallel_threads();
        assert!(num > 0, "Should have at least 1 thread");
    }

    #[test]
    fn test_parallel_for_range() {
        use std::sync::{Arc, Mutex};
        let sum = Arc::new(Mutex::new(0));
        let sum_clone = Arc::clone(&sum);

        parallel_for_range(0, 100, move |i| {
            let mut s = sum_clone.lock().unwrap();
            *s += i;
        });

        let result = *sum.lock().unwrap();
        assert_eq!(result, (0..100).sum::<usize>());
    }

    #[test]
    fn test_parallel_map_range() {
        let results = parallel_map_range(0, 10, |i| i * i);
        assert_eq!(results, vec![0, 1, 4, 9, 16, 25, 36, 49, 64, 81]);
    }
}
