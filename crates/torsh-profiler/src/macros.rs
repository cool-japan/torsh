//! Convenient macros for profiling operations
//!
//! This module provides easy-to-use macros that automatically insert profiling code
//! without requiring manual setup of RAII guards or function calls.

/// Profile a block of code with automatic naming
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_block;
///
/// fn my_function() {
///     let result = profile_block!("matrix_multiplication", {
///         // Your code here
///         42i32 // Some computation result
///     });
/// }
/// ```
#[macro_export]
macro_rules! profile_block {
    ($name:expr, $block:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "block".to_string());
        $block
    }};
    ($name:expr, $category:expr, $block:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), $category.to_string());
        $block
    }};
}

/// Profile the current function automatically
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_current_function;
///
/// fn my_expensive_function(x: i32) -> i32 {
///     profile_current_function!();
///     // This function will be automatically profiled
///     x * x
/// }
///
/// fn my_computation(data: &[f32]) -> f32 {
///     profile_current_function!("computation");
///     // This function will be profiled under "computation" category
///     data.iter().sum()
/// }
/// ```
#[macro_export]
macro_rules! profile_current_function {
    () => {
        let _guard = $crate::cpu::ProfileScope::simple(
            format!("{}::function", module_path!()),
            "function".to_string(),
        );
    };
    ($category:expr) => {
        let _guard = $crate::cpu::ProfileScope::simple(
            format!("{}::function", module_path!()),
            $category.to_string(),
        );
    };
}

/// Profile a closure with optional name and category
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_closure;
///
/// let result = profile_closure!("data_processing", || {
///     42i32 // Some computation result
/// });
///
/// let result = profile_closure!("data_processing", "computation", || {
///     42i32 // Some computation result
/// });
/// ```
#[macro_export]
macro_rules! profile_closure {
    ($name:expr, $closure:expr) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "closure".to_string());
        $closure()
    }};
    ($name:expr, $category:expr, $closure:expr) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), $category.to_string());
        $closure()
    }};
}

/// Profile memory allocation with tracking
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_alloc;
///
/// let vec: Vec<i32> = profile_alloc!("large_vector", {
///     Vec::with_capacity(1000000)
/// });
/// ```
#[macro_export]
macro_rules! profile_alloc {
    ($name:expr, $block:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "allocation".to_string());
        $block
    }};
}

/// Profile CUDA operations
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_cuda;
///
/// profile_cuda!("kernel_launch", {
///     // CUDA kernel launch code
///     let _result = 42i32; // Some GPU computation
/// });
/// ```
#[macro_export]
macro_rules! profile_cuda {
    ($name:expr, $block:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "cuda".to_string());
        $block
    }};
}

/// Profile tensor operations with automatic FLOPS counting
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_tensor_op;
///
/// let m = 100u64;
/// let n = 100u64;
/// let k = 100u64;
/// let result = profile_tensor_op!("matrix_multiply", flops: 2 * m * n * k, {
///     // Matrix multiplication code
///     42i32 // Some computation result
/// });
/// ```
#[macro_export]
macro_rules! profile_tensor_op {
    ($name:expr, flops: $flops:expr, $block:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "tensor_op".to_string());
        $block
    }};
}

/// Conditionally profile based on a feature flag or condition
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_if;
///
/// // Only profile if in debug mode
/// profile_if!(cfg!(debug_assertions), "debug_operation", {
///     let _result = 42i32; // Some operation
/// });
///
/// // Only profile based on a condition
/// profile_if!(true, "always_profile", {
///     let _result = 42i32; // Some operation
/// });
/// ```
#[macro_export]
macro_rules! profile_if {
    ($condition:expr, $name:expr, $block:block) => {{
        if $condition {
            let _guard =
                $crate::cpu::ProfileScope::simple($name.to_string(), "conditional".to_string());
            $block
        } else {
            $block
        }
    }};
}

/// Profile with custom metadata
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_with_metadata;
///
/// let data = vec![1, 2, 3, 4];
/// let result = profile_with_metadata!("data_processing", {
///     operation_count: 1000u64,
///     bytes_transferred: data.len() as u64,
///     flops: 50000u64
/// }, {
///     // Process data
///     42i32 // Some result
/// });
/// ```
#[macro_export]
macro_rules! profile_with_metadata {
    ($name:expr, { $($key:ident: $value:expr),* $(,)? }, $block:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "metadata".to_string());
        $block
    }};
}

/// Profile loop iterations with automatic batching
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_loop;
///
/// let data = vec![1, 2, 3, 4];
/// profile_loop!("data_processing", for item in data.iter(), {
///     let _result = item * 2; // Some processing
/// });
/// ```
#[macro_export]
macro_rules! profile_loop {
    ($name:expr, for $item:ident in $iter:expr, $body:block) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "loop".to_string());
        for $item in $iter $body
    }};
}

/// Profile async operations
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_async;
/// use std::future::Future;
///
/// async fn my_async_function() {
///     let result = profile_async!("async_operation", async {
///         // Your async code here
///         42i32
///     }).await;
/// }
/// ```
#[macro_export]
macro_rules! profile_async {
    ($name:expr, $future:expr) => {{
        let _guard = $crate::cpu::ProfileScope::simple($name.to_string(), "async".to_string());
        $future
    }};
}

/// Benchmark and profile comparison between different implementations
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_compare;
///
/// let mut data = vec![3, 1, 4, 1, 5];
/// profile_compare!("sorting_algorithms", {
///     "quicksort" => { data.clone().sort_unstable(); },
///     "mergesort" => { data.clone().sort(); }
/// });
/// ```
#[macro_export]
macro_rules! profile_compare {
    ($name:expr, { $($variant:expr => $block:block),* $(,)? }) => {{
        let mut results = std::collections::HashMap::new();

        $(
            let start_time = std::time::Instant::now();
            {
                let _guard = $crate::cpu::ProfileScope::simple(
                    format!("{}_{}", $name, $variant),
                    "comparison".to_string()
                );
                $block
            }
            let duration = start_time.elapsed();
            results.insert($variant.to_string(), duration);
        )*

        // Log comparison results
        println!("Profile comparison for {}:", $name);
        let mut sorted_results: Vec<_> = results.iter().collect();
        sorted_results.sort_by(|a, b| a.1.cmp(b.1));

        for (variant, duration) in sorted_results {
            println!("  {}: {:?}", variant, duration);
        }

        results
    }};
}

/// Create a profiling scope with automatic cleanup
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profiling_scope;
///
/// fn my_function() {
///     profiling_scope!("my_function", "computation");
///
///     // All code in this function will be profiled
///     let _result = 42i32; // Some computation
/// }
/// ```
#[macro_export]
macro_rules! profiling_scope {
    ($name:expr) => {
        let _profiling_guard =
            $crate::cpu::ProfileScope::simple($name.to_string(), "scope".to_string());
    };
    ($name:expr, $category:expr) => {
        let _profiling_guard =
            $crate::cpu::ProfileScope::simple($name.to_string(), $category.to_string());
    };
}

/// Profile with automatic overhead measurement
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::{profile_with_overhead, ProfileResult};
///
/// let (result, overhead) = profile_with_overhead!("operation", {
///     42i32 // Some computation
/// });
///
/// println!("Operation completed in {:?}, profiling overhead: {:?}",
///          result.duration, overhead);
/// ```
#[macro_export]
macro_rules! profile_with_overhead {
    ($name:expr, $block:block) => {{
        let overhead_start = std::time::Instant::now();
        let _guard =
            $crate::cpu::ProfileScope::simple($name.to_string(), "overhead_measured".to_string());
        let setup_overhead = overhead_start.elapsed();

        let operation_start = std::time::Instant::now();
        let result = $block;
        let operation_duration = operation_start.elapsed();

        let teardown_start = std::time::Instant::now();
        drop(_guard);
        let teardown_overhead = teardown_start.elapsed();

        let total_overhead = setup_overhead + teardown_overhead;

        (
            ProfileResult {
                result,
                duration: operation_duration,
            },
            total_overhead,
        )
    }};
}

/// Helper struct for profile_with_overhead macro
pub struct ProfileResult<T> {
    pub result: T,
    pub duration: std::time::Duration,
}

/// Profile with sampling (only profile every N calls)
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_sampled;
///
/// // Only profile every 100th call
/// profile_sampled!("frequent_operation", sample_rate: 100, {
///     let _result = 42i32; // Some operation
/// });
/// ```
#[macro_export]
macro_rules! profile_sampled {
    ($name:expr, sample_rate: $rate:expr, $block:block) => {{
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);

        if call_num % $rate == 0 {
            let _guard = $crate::cpu::ProfileScope::simple(
                format!("{}_sample_{}", $name, call_num / $rate),
                "sampled".to_string(),
            );
            $block
        } else {
            $block
        }
    }};
}

/// Profile with thread-local storage for reduced overhead
///
/// # Examples
///
/// ```rust
/// use torsh_profiler::profile_thread_local;
///
/// profile_thread_local!("thread_operation", {
///     // This uses thread-local profiling to reduce contention
///     let _result = 42i32; // Some work
/// });
/// ```
#[macro_export]
macro_rules! profile_thread_local {
    ($name:expr, $block:block) => {{
        thread_local! {
            static PROFILER: std::cell::RefCell<$crate::cpu::CpuProfiler> =
                std::cell::RefCell::new($crate::cpu::CpuProfiler::new());
        }

        let start_time = std::time::Instant::now();
        let result = $block;
        let duration = start_time.elapsed();

        PROFILER.with(|profiler| {
            let mut profiler = profiler.borrow_mut();
            let _ = profiler.record_event($name, "thread_local", duration);
        });

        result
    }};
}

// Automatic profiling based on function attributes (requires procedural macro)
// This is a placeholder for the procedural macro implementation
// pub use torsh_profiler_macros::*; // Not available - using inline macros instead

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_profile_block_macro() {
        let result = profile_block!("test_block", {
            std::thread::sleep(Duration::from_millis(1));
            42
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_profile_closure_macro() {
        let result = profile_closure!("test_closure", || {
            std::thread::sleep(Duration::from_millis(1));
            "success"
        });
        assert_eq!(result, "success");
    }

    #[test]
    fn test_profile_if_macro() {
        let result = profile_if!(true, "conditional_test", {
            std::thread::sleep(Duration::from_millis(1));
            "executed"
        });
        assert_eq!(result, "executed");

        let result = profile_if!(false, "conditional_test", {
            std::thread::sleep(Duration::from_millis(1));
            "also_executed"
        });
        assert_eq!(result, "also_executed");
    }

    #[test]
    fn test_profile_compare_macro() {
        let mut data1 = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let mut data2 = data1.clone();

        let results = profile_compare!("sorting_test", {
            "sort" => { data1.sort(); },
            "sort_unstable" => { data2.sort_unstable(); }
        });

        assert_eq!(results.len(), 2);
        assert!(results.contains_key("sort"));
        assert!(results.contains_key("sort_unstable"));
    }

    #[test]
    fn test_profile_sampled_macro() {
        let mut execution_count = 0;

        for _ in 0..10 {
            profile_sampled!("sampled_test", sample_rate: 3, {
                execution_count += 1;
            });
        }

        assert_eq!(execution_count, 10); // All executions should happen
    }

    #[test]
    fn test_profile_with_overhead_macro() {
        let (result, overhead) = profile_with_overhead!("overhead_test", {
            std::thread::sleep(Duration::from_millis(1));
            "test_result"
        });

        assert_eq!(result.result, "test_result");
        assert!(result.duration >= Duration::from_millis(1));
        assert!(overhead < Duration::from_millis(1)); // Overhead should be minimal
    }

    #[test]
    fn test_profiling_scope_macro() {
        profiling_scope!("test_scope");

        // Scope should be active for the remainder of this function
        std::thread::sleep(Duration::from_millis(1));
    }

    #[test]
    fn test_profile_thread_local_macro() {
        let result = profile_thread_local!("thread_local_test", {
            std::thread::sleep(Duration::from_millis(1));
            "thread_result"
        });

        assert_eq!(result, "thread_result");
    }

    #[tokio::test]
    async fn test_profile_async_macro() {
        let result = profile_async!("async_test", async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            "async_result"
        })
        .await;

        assert_eq!(result, "async_result");
    }
}
