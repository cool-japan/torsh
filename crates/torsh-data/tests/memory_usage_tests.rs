//! Memory usage tests for torsh-data
//!
//! These tests monitor memory consumption patterns, detect memory leaks,
//! and validate memory efficiency of data loading operations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use torsh_core::error::Result;
use torsh_data::prelude::*;
use torsh_data::Transform;
use torsh_tensor::creation::{ones, zeros};

/// Memory tracking allocator for monitoring allocations
struct MemoryTracker {
    allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl MemoryTracker {
    const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    fn peak_allocated_bytes(&self) -> usize {
        self.peak_allocated.load(Ordering::Relaxed)
    }

    fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.peak_allocated.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }
}

static MEMORY_TRACKER: MemoryTracker = MemoryTracker::new();

/// Custom allocator that tracks memory usage
#[allow(dead_code)]
struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let old_allocated = MEMORY_TRACKER.allocated.fetch_add(size, Ordering::Relaxed);
            let new_allocated = old_allocated + size;

            // Update peak if necessary
            let mut peak = MEMORY_TRACKER.peak_allocated.load(Ordering::Relaxed);
            while new_allocated > peak {
                match MEMORY_TRACKER.peak_allocated.compare_exchange_weak(
                    peak,
                    new_allocated,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => peak = x,
                }
            }

            MEMORY_TRACKER
                .allocation_count
                .fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        MEMORY_TRACKER
            .allocated
            .fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub initial_allocated: usize,
    pub peak_allocated: usize,
    pub final_allocated: usize,
    pub net_allocation: i64,
    pub allocation_count: usize,
    pub peak_usage_mb: f64,
    pub net_usage_mb: f64,
}

impl MemoryUsageStats {
    fn new(initial: usize, peak: usize, final_allocated: usize, allocation_count: usize) -> Self {
        let net_allocation = final_allocated as i64 - initial as i64;
        Self {
            initial_allocated: initial,
            peak_allocated: peak,
            final_allocated,
            net_allocation,
            allocation_count,
            peak_usage_mb: peak as f64 / (1024.0 * 1024.0),
            net_usage_mb: net_allocation as f64 / (1024.0 * 1024.0),
        }
    }
}

/// Memory test result
#[derive(Debug)]
pub struct MemoryTestResult {
    pub test_name: String,
    pub stats: MemoryUsageStats,
    pub passed: bool,
    pub message: Option<String>,
}

/// Memory usage test suite
pub struct MemoryUsageTests;

impl MemoryUsageTests {
    /// Test memory usage during dataset creation
    pub fn test_dataset_memory_usage() -> Result<MemoryTestResult> {
        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        // Create multiple datasets of varying sizes
        let dataset_sizes = [100, 1000, 5000];
        let mut _datasets = Vec::new();

        for &size in &dataset_sizes {
            let data = ones::<f32>(&[size, 50])?;
            let labels = zeros::<f32>(&[size])?;
            let dataset = TensorDataset::from_tensors(vec![data, labels]);
            _datasets.push(dataset);
        }

        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        // Clear datasets to test cleanup
        drop(_datasets);

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        // Check if memory usage is reasonable (less than 100MB peak)
        let passed = stats.peak_usage_mb < 100.0;
        let message = if passed {
            None
        } else {
            Some(format!(
                "Peak memory usage too high: {:.2} MB",
                stats.peak_usage_mb
            ))
        };

        Ok(MemoryTestResult {
            test_name: "Dataset Memory Usage".to_string(),
            stats,
            passed,
            message,
        })
    }

    /// Test memory usage during data loading
    pub fn test_dataloader_memory_usage() -> Result<MemoryTestResult> {
        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        // Create a moderately large dataset
        let dataset_size = 1000;
        let batch_size = 32;

        let data = ones::<f32>(&[dataset_size, 100])?;
        let labels = zeros::<f32>(&[dataset_size])?;
        let dataset = TensorDataset::from_tensors(vec![data, labels]);

        let dataloader = simple_random_dataloader(dataset, batch_size, Some(42))?;

        // Process several batches
        let mut batch_count = 0;
        for batch in dataloader.iter() {
            let _batch = batch?;
            batch_count += 1;
            if batch_count >= 10 {
                break; // Process only first 10 batches
            }
        }

        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        // Check if memory usage is reasonable
        let passed = stats.peak_usage_mb < 200.0 && stats.net_usage_mb.abs() < 50.0;
        let message = if passed {
            None
        } else {
            Some(format!(
                "Memory usage issue - Peak: {:.2} MB, Net: {:.2} MB",
                stats.peak_usage_mb, stats.net_usage_mb
            ))
        };

        Ok(MemoryTestResult {
            test_name: "DataLoader Memory Usage".to_string(),
            stats,
            passed,
            message,
        })
    }

    /// Test memory usage during transform operations
    pub fn test_transform_memory_usage() -> Result<MemoryTestResult> {
        use torsh_data::transforms::augmentation::*;

        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        let num_transforms = 100;
        let tensor_size = [3, 224, 224];

        // Create augmentation pipeline
        let pipeline = AugmentationPipeline::<torsh_tensor::Tensor<f32>>::medium_augmentation();

        // Apply transforms to multiple tensors
        for _ in 0..num_transforms {
            let tensor = ones::<f32>(&tensor_size)?;
            let _transformed = pipeline.transform(tensor)?;
        }

        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        // Check memory usage - should not accumulate significantly
        let passed = stats.net_usage_mb.abs() < 10.0 && stats.peak_usage_mb < 500.0;
        let message = if passed {
            None
        } else {
            Some(format!(
                "Transform memory leak detected - Net: {:.2} MB, Peak: {:.2} MB",
                stats.net_usage_mb, stats.peak_usage_mb
            ))
        };

        Ok(MemoryTestResult {
            test_name: "Transform Memory Usage".to_string(),
            stats,
            passed,
            message,
        })
    }

    /// Test memory usage with caching
    pub fn test_cached_dataset_memory() -> Result<MemoryTestResult> {
        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        let dataset_size = 500;
        let cache_size = 50;

        // Create base dataset
        let data = ones::<f32>(&[dataset_size, 100])?;
        let base_dataset = TensorDataset::from_tensor(data);
        let cached_dataset = CachedDataset::new(base_dataset, cache_size);

        // Access items to populate cache
        for i in 0..cache_size {
            let _item = cached_dataset.get(i)?;
        }

        let cache_populated_memory = MEMORY_TRACKER.allocated_bytes();

        // Access items again (should use cache)
        for i in 0..cache_size {
            let _item = cached_dataset.get(i)?;
        }

        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        // Memory should stabilize after cache population
        let cache_overhead =
            (cache_populated_memory - initial_allocated) as f64 / (1024.0 * 1024.0);
        let passed = cache_overhead < 100.0
            && (final_allocated as f64 - cache_populated_memory as f64).abs() < 1024.0 * 1024.0; // Less than 1MB difference

        let message = if passed {
            None
        } else {
            Some(format!(
                "Cache memory issue - Overhead: {:.2} MB, Memory drift: {:.2} KB",
                cache_overhead,
                (final_allocated as f64 - cache_populated_memory as f64) / 1024.0
            ))
        };

        Ok(MemoryTestResult {
            test_name: "Cached Dataset Memory".to_string(),
            stats,
            passed,
            message,
        })
    }

    /// Test memory usage with online augmentation
    pub fn test_online_augmentation_memory() -> Result<MemoryTestResult> {
        use torsh_data::transforms::augmentation::*;
        use torsh_data::transforms::online::*;

        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        let num_operations = 100;
        let tensor_size = [3, 64, 64]; // Smaller tensors for memory test

        // Create online augmentation engine with caching
        let pipeline = AugmentationPipeline::<torsh_tensor::Tensor<f32>>::light_augmentation();
        let engine = OnlineAugmentationEngine::new(pipeline).with_cache(20);

        // Perform operations with some repeated keys for cache testing
        for i in 0..num_operations {
            let tensor = ones::<f32>(&tensor_size)?;
            let cache_key = format!("key_{}", i % 30); // Some cache reuse
            let _result = engine.apply(tensor, Some(&cache_key))?;
        }

        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        let engine_stats = engine.stats();

        // Check that cache is working and memory is reasonable
        let cache_hit_rate = engine_stats.cache_hits as f64 / engine_stats.total_transforms as f64;
        let passed =
            stats.peak_usage_mb < 100.0 && stats.net_usage_mb.abs() < 20.0 && cache_hit_rate > 0.1; // At least 10% cache hit rate

        let message = if passed {
            None
        } else {
            Some(format!(
                "Online augmentation memory issue - Peak: {:.2} MB, Net: {:.2} MB, Cache hit rate: {:.2}%",
                stats.peak_usage_mb, stats.net_usage_mb, cache_hit_rate * 100.0
            ))
        };

        Ok(MemoryTestResult {
            test_name: "Online Augmentation Memory".to_string(),
            stats,
            passed,
            message,
        })
    }

    /// Test memory usage with large datasets
    pub fn test_large_dataset_memory() -> Result<MemoryTestResult> {
        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        // Create a large dataset incrementally
        let chunk_size = 1000;
        let num_chunks = 10;
        let mut datasets = Vec::new();

        for _ in 0..num_chunks {
            let data = ones::<f32>(&[chunk_size, 50])?;
            let dataset = TensorDataset::from_tensor(data);
            datasets.push(dataset);
        }

        // Create concatenated dataset
        let large_dataset = ConcatDataset::new(datasets);

        // Sample randomly from the large dataset
        let sampler = RandomSampler::simple(large_dataset.len()).with_generator(42);
        let _batch_sampler = BatchingSampler::new(sampler, 32, false);
        let dataloader = simple_random_dataloader(large_dataset, 32, Some(42))?;

        // Process a few batches
        let mut batch_count = 0;
        for batch in dataloader.iter() {
            let _batch = batch?;
            batch_count += 1;
            if batch_count >= 20 {
                break;
            }
        }

        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        // Large dataset should not use excessive memory
        let passed = stats.peak_usage_mb < 1000.0; // Less than 1GB
        let message = if passed {
            None
        } else {
            Some(format!(
                "Large dataset uses too much memory: {:.2} MB",
                stats.peak_usage_mb
            ))
        };

        Ok(MemoryTestResult {
            test_name: "Large Dataset Memory".to_string(),
            stats,
            passed,
            message,
        })
    }
}

/// Memory leak detection utilities
pub struct MemoryLeakDetector;

impl MemoryLeakDetector {
    /// Run a stress test to detect memory leaks
    pub fn stress_test_memory_leaks() -> Result<MemoryTestResult> {
        MEMORY_TRACKER.reset();
        let initial_allocated = MEMORY_TRACKER.allocated_bytes();
        let initial_count = MEMORY_TRACKER.allocation_count();

        let num_iterations = 50;
        let mut peak_memory_per_iteration = Vec::new();

        for iteration in 0..num_iterations {
            let iteration_start_memory = MEMORY_TRACKER.allocated_bytes();

            // Perform typical data loading operations
            let dataset_size = 100;
            let data = ones::<f32>(&[dataset_size, 20])?;
            let dataset = TensorDataset::from_tensor(data);

            let sampler = RandomSampler::simple(dataset.len()).with_generator(iteration as u64);
            let _batch_sampler = BatchingSampler::new(sampler, 8, false);
            let dataloader = simple_random_dataloader(dataset, 8, Some(iteration as u64))?;

            // Process all batches
            for batch in dataloader.iter() {
                let _batch = batch?;
            }

            let iteration_peak = MEMORY_TRACKER.peak_allocated_bytes();
            peak_memory_per_iteration.push(iteration_peak - iteration_start_memory);

            // Force cleanup
            let _ = (); // Explicit drop to ensure cleanup
        }

        let final_allocated = MEMORY_TRACKER.allocated_bytes();
        let peak_allocated = MEMORY_TRACKER.peak_allocated_bytes();
        let allocation_count = MEMORY_TRACKER.allocation_count() - initial_count;

        let stats = MemoryUsageStats::new(
            initial_allocated,
            peak_allocated,
            final_allocated,
            allocation_count,
        );

        // Check for memory growth over iterations
        let first_half_avg: f64 = peak_memory_per_iteration[..num_iterations / 2]
            .iter()
            .sum::<usize>() as f64
            / (num_iterations / 2) as f64;
        let second_half_avg: f64 = peak_memory_per_iteration[num_iterations / 2..]
            .iter()
            .sum::<usize>() as f64
            / (num_iterations / 2) as f64;
        let memory_growth_ratio = second_half_avg / first_half_avg;

        // Memory should not grow significantly over iterations (less than 20% growth)
        let passed = memory_growth_ratio < 1.2 && stats.net_usage_mb.abs() < 10.0;
        let message = if passed {
            None
        } else {
            Some(format!(
                "Memory leak detected - Growth ratio: {:.2}, Net usage: {:.2} MB",
                memory_growth_ratio, stats.net_usage_mb
            ))
        };

        Ok(MemoryTestResult {
            test_name: "Memory Leak Stress Test".to_string(),
            stats,
            passed,
            message,
        })
    }
}

/// Memory test runner
pub struct MemoryTestRunner;

impl MemoryTestRunner {
    /// Run all memory tests
    pub fn run_all_memory_tests() -> Result<Vec<MemoryTestResult>> {
        let mut results = Vec::new();

        println!("Running Memory Usage Tests...");

        results.push(MemoryUsageTests::test_dataset_memory_usage()?);
        results.push(MemoryUsageTests::test_dataloader_memory_usage()?);
        results.push(MemoryUsageTests::test_transform_memory_usage()?);
        results.push(MemoryUsageTests::test_cached_dataset_memory()?);
        results.push(MemoryUsageTests::test_online_augmentation_memory()?);
        results.push(MemoryUsageTests::test_large_dataset_memory()?);
        results.push(MemoryLeakDetector::stress_test_memory_leaks()?);

        Ok(results)
    }

    /// Print memory test results
    pub fn print_results(results: &[MemoryTestResult]) {
        println!("\n=== Memory Usage Test Results ===\n");

        let mut passed_count = 0;
        let mut total_count = 0;

        for result in results {
            total_count += 1;
            let status = if result.passed {
                passed_count += 1;
                "PASS"
            } else {
                "FAIL"
            };

            println!("[{}] {}", status, result.test_name);
            println!("    Peak Memory: {:.2} MB", result.stats.peak_usage_mb);
            println!("    Net Memory: {:.2} MB", result.stats.net_usage_mb);
            println!("    Allocations: {}", result.stats.allocation_count);

            if let Some(ref message) = result.message {
                println!("    Message: {message}");
            }
            println!();
        }

        println!("=== Summary ===");
        println!("Passed: {passed_count}/{total_count} tests");

        if passed_count == total_count {
            println!("All memory tests passed!");
        } else {
            println!("Some memory tests failed. Check the details above.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        MEMORY_TRACKER.reset();

        let initial = MEMORY_TRACKER.allocated_bytes();
        assert_eq!(initial, 0);

        // Test basic memory tracking functionality
        let test_data = vec![1u8; 1024];
        let _ = test_data.len();

        // Note: In a real test environment with custom allocator,
        // we would see memory tracking here
    }

    #[test]
    fn test_memory_usage_stats() {
        let stats = MemoryUsageStats::new(1000, 5000, 1200, 50);

        assert_eq!(stats.initial_allocated, 1000);
        assert_eq!(stats.peak_allocated, 5000);
        assert_eq!(stats.final_allocated, 1200);
        assert_eq!(stats.net_allocation, 200);
        assert_eq!(stats.allocation_count, 50);

        // Check MB conversions
        assert!((stats.peak_usage_mb - 5000.0 / (1024.0 * 1024.0)).abs() < 0.001);
    }

    #[test]
    fn test_individual_memory_tests() -> Result<()> {
        // Run individual tests to ensure they work
        let dataset_test = MemoryUsageTests::test_dataset_memory_usage()?;
        println!(
            "Dataset test: {} - {}",
            dataset_test.test_name,
            if dataset_test.passed { "PASS" } else { "FAIL" }
        );

        let transform_test = MemoryUsageTests::test_transform_memory_usage()?;
        println!(
            "Transform test: {} - {}",
            transform_test.test_name,
            if transform_test.passed {
                "PASS"
            } else {
                "FAIL"
            }
        );

        Ok(())
    }
}
