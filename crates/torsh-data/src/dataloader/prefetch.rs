//! Prefetch functionality for performance optimization
//!
//! This module provides prefetching capabilities to improve DataLoader performance
//! by loading data in a separate thread while the main thread processes batches.

use crossbeam::channel;
use std::thread;

/// Prefetch iterator for performance optimization
///
/// This iterator wraps another iterator and prefetches items in a background thread,
/// allowing for overlapping computation and data loading to improve overall throughput.
///
/// # Type Parameters
///
/// * `T` - The item type that the iterator yields
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::prefetch::{PrefetchIterator, PrefetchExt};
///
/// let data = vec![1, 2, 3, 4, 5];
/// let iter = data.into_iter();
/// let prefetch_iter = iter.prefetch(2); // Buffer size of 2
///
/// for item in prefetch_iter {
///     // Process item while next items are being prefetched
/// }
/// ```
pub struct PrefetchIterator<T> {
    receiver: channel::Receiver<Option<T>>,
    _handle: thread::JoinHandle<()>,
}

impl<T> PrefetchIterator<T>
where
    T: Send + 'static,
{
    /// Create a new prefetch iterator
    ///
    /// # Arguments
    ///
    /// * `inner` - The iterator to wrap with prefetching
    /// * `buffer_size` - The size of the prefetch buffer
    ///
    /// # Returns
    ///
    /// A new PrefetchIterator that will prefetch items from the inner iterator
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::dataloader::prefetch::PrefetchIterator;
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// let iter = data.into_iter();
    /// let prefetch_iter = PrefetchIterator::new(iter, 3);
    /// ```
    pub fn new<I>(inner: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        let (sender, receiver) = channel::bounded(buffer_size);

        let handle = thread::spawn(move || {
            for item in inner {
                if sender.send(Some(item)).is_err() {
                    // Receiver has been dropped, stop producing
                    break;
                }
            }
            // Send None to signal end of iteration
            let _ = sender.send(None);
        });

        Self {
            receiver,
            _handle: handle,
        }
    }

    /// Create a new prefetch iterator with unbounded buffer
    ///
    /// This creates a prefetch iterator with an unbounded channel, which can be useful
    /// when memory usage is not a concern and maximum throughput is desired.
    ///
    /// # Arguments
    ///
    /// * `inner` - The iterator to wrap with prefetching
    ///
    /// # Returns
    ///
    /// A new PrefetchIterator with unbounded buffering
    ///
    /// # Warning
    ///
    /// Using an unbounded buffer can lead to excessive memory usage if the consumer
    /// is significantly slower than the producer.
    pub fn new_unbounded<I>(inner: I) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        let (sender, receiver) = channel::unbounded();

        let handle = thread::spawn(move || {
            for item in inner {
                if sender.send(Some(item)).is_err() {
                    // Receiver has been dropped, stop producing
                    break;
                }
            }
            // Send None to signal end of iteration
            let _ = sender.send(None);
        });

        Self {
            receiver,
            _handle: handle,
        }
    }

    /// Get the number of items currently in the prefetch buffer
    ///
    /// This can be useful for monitoring the prefetch buffer utilization.
    ///
    /// # Returns
    ///
    /// The number of items currently buffered
    pub fn buffer_len(&self) -> usize {
        self.receiver.len()
    }

    /// Check if the prefetch buffer is empty
    ///
    /// # Returns
    ///
    /// True if the buffer is empty, false otherwise
    pub fn buffer_is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Try to get the next item without blocking
    ///
    /// This is useful when you want to check if an item is available without
    /// blocking the current thread.
    ///
    /// # Returns
    ///
    /// Some(item) if an item is available, None if no item is ready or the iterator is exhausted
    pub fn try_next(&mut self) -> Option<T> {
        match self.receiver.try_recv() {
            Ok(Some(item)) => Some(item),
            Ok(None) | Err(_) => None,
        }
    }
}

impl<T> Iterator for PrefetchIterator<T>
where
    T: Send + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(Some(item)) => Some(item),
            Ok(None) | Err(_) => None,
        }
    }
}

/// Extension trait for adding prefetching to iterators
///
/// This trait provides convenient methods for adding prefetching capabilities
/// to any iterator that meets the requirements (Send + 'static items).
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::prefetch::PrefetchExt;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let prefetch_iter = data.into_iter().prefetch(2);
///
/// for item in prefetch_iter {
///     // Process item while next items are being prefetched
/// }
/// ```
pub trait PrefetchExt<T>: Iterator<Item = T> + Sized + Send + 'static
where
    T: Send + 'static,
{
    /// Add prefetching to the iterator
    ///
    /// # Arguments
    ///
    /// * `buffer_size` - The size of the prefetch buffer
    ///
    /// # Returns
    ///
    /// A PrefetchIterator that will prefetch items from this iterator
    fn prefetch(self, buffer_size: usize) -> PrefetchIterator<T> {
        PrefetchIterator::new(self, buffer_size)
    }

    /// Add unbounded prefetching to the iterator
    ///
    /// # Returns
    ///
    /// A PrefetchIterator with unbounded buffering
    ///
    /// # Warning
    ///
    /// This can lead to excessive memory usage if the consumer is slower than the producer
    fn prefetch_unbounded(self) -> PrefetchIterator<T> {
        PrefetchIterator::new_unbounded(self)
    }
}

/// Blanket implementation of PrefetchExt for all compatible iterators
impl<I, T> PrefetchExt<T> for I
where
    I: Iterator<Item = T> + Send + 'static,
    T: Send + 'static,
{
}

/// Configuration for prefetch operations
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Size of the prefetch buffer
    pub buffer_size: usize,
    /// Whether to use unbounded buffering
    pub unbounded: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            buffer_size: 2,
            unbounded: false,
        }
    }
}

impl PrefetchConfig {
    /// Create a new prefetch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self.unbounded = false;
        self
    }

    /// Enable unbounded buffering
    pub fn unbounded(mut self) -> Self {
        self.unbounded = true;
        self
    }

    /// Apply this configuration to an iterator
    pub fn apply<I, T>(self, iter: I) -> PrefetchIterator<T>
    where
        I: Iterator<Item = T> + Send + 'static,
        T: Send + 'static,
    {
        if self.unbounded {
            PrefetchIterator::new_unbounded(iter)
        } else {
            PrefetchIterator::new(iter, self.buffer_size)
        }
    }
}

/// Utility functions for prefetch operations
pub mod utils {
    use super::*;

    /// Create a prefetch iterator with optimal buffer size
    ///
    /// This function automatically determines an appropriate buffer size based on
    /// heuristics and the expected workload characteristics.
    ///
    /// # Arguments
    ///
    /// * `iter` - The iterator to wrap
    /// * `expected_item_processing_time` - Expected time to process each item in milliseconds
    ///
    /// # Returns
    ///
    /// A PrefetchIterator with an optimized buffer size
    pub fn optimal_prefetch<I, T>(
        iter: I,
        expected_item_processing_time: u64,
    ) -> PrefetchIterator<T>
    where
        I: Iterator<Item = T> + Send + 'static,
        T: Send + 'static,
    {
        // Simple heuristic: buffer size inversely related to processing time
        let buffer_size = if expected_item_processing_time > 100 {
            2 // Small buffer for expensive operations
        } else if expected_item_processing_time > 10 {
            4 // Medium buffer for moderate operations
        } else {
            8 // Larger buffer for fast operations
        };

        PrefetchIterator::new(iter, buffer_size)
    }

    /// Create a prefetch iterator optimized for CPU-bound tasks
    ///
    /// # Arguments
    ///
    /// * `iter` - The iterator to wrap
    ///
    /// # Returns
    ///
    /// A PrefetchIterator configured for CPU-bound workloads
    pub fn cpu_bound_prefetch<I, T>(iter: I) -> PrefetchIterator<T>
    where
        I: Iterator<Item = T> + Send + 'static,
        T: Send + 'static,
    {
        // For CPU-bound tasks, use a smaller buffer to avoid excessive memory usage
        PrefetchIterator::new(iter, 2)
    }

    /// Create a prefetch iterator optimized for I/O-bound tasks
    ///
    /// # Arguments
    ///
    /// * `iter` - The iterator to wrap
    ///
    /// # Returns
    ///
    /// A PrefetchIterator configured for I/O-bound workloads
    pub fn io_bound_prefetch<I, T>(iter: I) -> PrefetchIterator<T>
    where
        I: Iterator<Item = T> + Send + 'static,
        T: Send + 'static,
    {
        // For I/O-bound tasks, use a larger buffer to hide I/O latency
        PrefetchIterator::new(iter, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_prefetch_iterator_basic() {
        let data = vec![1, 2, 3, 4, 5];
        let iter = data.into_iter();
        let mut prefetch_iter = PrefetchIterator::new(iter, 2);

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_prefetch_ext_trait() {
        let data = vec![1, 2, 3, 4, 5];
        let mut prefetch_iter = data.into_iter().prefetch(2);

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_prefetch_unbounded() {
        let data = vec![1, 2, 3, 4, 5];
        let mut prefetch_iter = data.into_iter().prefetch_unbounded();

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_prefetch_config() {
        let config = PrefetchConfig::new().buffer_size(4);
        assert_eq!(config.buffer_size, 4);
        assert!(!config.unbounded);

        let config = PrefetchConfig::new().unbounded();
        assert!(config.unbounded);
    }

    #[test]
    fn test_prefetch_config_apply() {
        let data = vec![1, 2, 3, 4, 5];
        let config = PrefetchConfig::new().buffer_size(3);
        let mut prefetch_iter = config.apply(data.into_iter());

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_try_next() {
        let data = vec![1, 2, 3];
        let mut prefetch_iter = PrefetchIterator::new(data.into_iter(), 2);

        // Give the prefetch thread a moment to work
        std::thread::sleep(Duration::from_millis(10));

        // Should have at least one item prefetched
        assert!(prefetch_iter.try_next().is_some());
    }

    #[test]
    fn test_buffer_status() {
        let data = vec![1, 2, 3, 4, 5];
        let prefetch_iter = PrefetchIterator::new(data.into_iter(), 3);

        // Give the prefetch thread a moment to work
        std::thread::sleep(Duration::from_millis(10));

        // Buffer should not be empty after prefetching starts
        assert!(!prefetch_iter.buffer_is_empty());
        assert!(prefetch_iter.buffer_len() > 0);
    }

    #[test]
    fn test_utils_optimal_prefetch() {
        let data = vec![1, 2, 3, 4, 5];
        let mut prefetch_iter = utils::optimal_prefetch(data.into_iter(), 50);

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_utils_cpu_bound_prefetch() {
        let data = vec![1, 2, 3, 4, 5];
        let mut prefetch_iter = utils::cpu_bound_prefetch(data.into_iter());

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_utils_io_bound_prefetch() {
        let data = vec![1, 2, 3, 4, 5];
        let mut prefetch_iter = utils::io_bound_prefetch(data.into_iter());

        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));
        assert_eq!(prefetch_iter.next(), Some(3));
        assert_eq!(prefetch_iter.next(), Some(4));
        assert_eq!(prefetch_iter.next(), Some(5));
        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_empty_iterator() {
        let data: Vec<i32> = vec![];
        let mut prefetch_iter = data.into_iter().prefetch(2);

        assert_eq!(prefetch_iter.next(), None);
    }

    #[test]
    fn test_prefetch_performance() {
        // Create a slow iterator that simulates expensive computation
        let slow_iter = (0..10).map(|x| {
            std::thread::sleep(Duration::from_millis(10));
            x
        });

        let start = Instant::now();
        let mut prefetch_iter = slow_iter.prefetch(3);

        // Consume the first few items
        assert_eq!(prefetch_iter.next(), Some(0));
        assert_eq!(prefetch_iter.next(), Some(1));
        assert_eq!(prefetch_iter.next(), Some(2));

        let elapsed = start.elapsed();

        // With prefetching, we should be able to get the first few items faster
        // than if we had to wait for all the computation sequentially
        assert!(elapsed < Duration::from_millis(100));
    }
}
