//! Attribute-based profiling support
//!
//! This module provides function decorators and attribute-like functionality for automatic profiling.
//! While Rust doesn't have decorators like Python, we provide similar functionality through wrapper functions.

use crate::cpu::ProfileScope;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::any::type_name;
use std::time::Instant;

/// Trait for automatic profiling of methods
pub trait ProfiledMethod<Args, Return> {
    /// Execute the method with automatic profiling
    fn profiled(self, name: Option<&str>, category: Option<&str>) -> Return;
}

/// Wrapper for functions that enables automatic profiling
pub struct ProfiledFunction<F> {
    func: F,
    name: String,
    category: String,
    enabled: bool,
}

impl<F> ProfiledFunction<F> {
    /// Create a new profiled function wrapper
    pub fn new(func: F, name: String, category: String) -> Self {
        Self {
            func,
            name,
            category,
            enabled: true,
        }
    }

    /// Enable/disable profiling for this function
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if profiling is enabled for this function
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl<F, R> ProfiledFunction<F>
where
    F: FnOnce() -> R,
{
    /// Execute the function with profiling
    pub fn call(self) -> R {
        if self.enabled {
            let _guard = ProfileScope::simple(self.name, self.category);
            (self.func)()
        } else {
            (self.func)()
        }
    }
}

impl<F> ProfiledFunction<F> {
    /// Execute the function with one argument and profiling
    pub fn call_with_arg<A, R>(self, arg: A) -> R
    where
        F: FnOnce(A) -> R,
    {
        if self.enabled {
            let _guard = ProfileScope::simple(self.name, self.category);
            (self.func)(arg)
        } else {
            (self.func)(arg)
        }
    }

    /// Execute the function with two arguments and profiling
    pub fn call_with_args<A, B, R>(self, arg1: A, arg2: B) -> R
    where
        F: FnOnce(A, B) -> R,
    {
        if self.enabled {
            let _guard = ProfileScope::simple(self.name, self.category);
            (self.func)(arg1, arg2)
        } else {
            (self.func)(arg1, arg2)
        }
    }
}

/// Attribute configuration for profiling
#[derive(Debug, Clone)]
pub struct ProfileAttribute {
    /// Name of the profiling event
    pub name: Option<String>,
    /// Category of the profiling event
    pub category: Option<String>,
    /// Whether to include stack traces
    pub stack_trace: bool,
    /// Whether to track memory allocations
    pub track_memory: bool,
    /// Whether to count FLOPS (for tensor operations)
    pub count_flops: bool,
    /// Custom metadata to include
    pub metadata: std::collections::HashMap<String, String>,
    /// Sampling rate (1 = profile every call, 10 = profile every 10th call)
    pub sample_rate: usize,
    /// Minimum duration threshold to record (in microseconds)
    pub min_duration_us: u64,
}

impl Default for ProfileAttribute {
    fn default() -> Self {
        Self {
            name: None,
            category: Some("function".to_string()),
            stack_trace: false,
            track_memory: false,
            count_flops: false,
            metadata: std::collections::HashMap::new(),
            sample_rate: 1,
            min_duration_us: 0,
        }
    }
}

impl ProfileAttribute {
    /// Create a new profile attribute with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the profiling name
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the profiling category
    pub fn with_category<S: Into<String>>(mut self, category: S) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Enable stack trace collection
    pub fn with_stack_trace(mut self) -> Self {
        self.stack_trace = true;
        self
    }

    /// Enable memory tracking
    pub fn with_memory_tracking(mut self) -> Self {
        self.track_memory = true;
        self
    }

    /// Enable FLOPS counting
    pub fn with_flops_counting(mut self) -> Self {
        self.count_flops = true;
        self
    }

    /// Add custom metadata
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set sampling rate
    pub fn with_sample_rate(mut self, rate: usize) -> Self {
        self.sample_rate = rate.max(1);
        self
    }

    /// Set minimum duration threshold
    pub fn with_min_duration_us(mut self, min_us: u64) -> Self {
        self.min_duration_us = min_us;
        self
    }
}

/// Function attribute registry for managing profiling attributes
pub struct AttributeRegistry {
    attributes: std::collections::HashMap<String, ProfileAttribute>,
    global_enabled: bool,
}

impl Default for AttributeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AttributeRegistry {
    /// Create a new attribute registry
    pub fn new() -> Self {
        Self {
            attributes: std::collections::HashMap::new(),
            global_enabled: true,
        }
    }

    /// Register a function with profiling attributes
    pub fn register<S: Into<String>>(&mut self, function_name: S, attr: ProfileAttribute) {
        self.attributes.insert(function_name.into(), attr);
    }

    /// Get profiling attributes for a function
    pub fn get_attributes(&self, function_name: &str) -> Option<&ProfileAttribute> {
        self.attributes.get(function_name)
    }

    /// Enable/disable all profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.global_enabled = enabled;
    }

    /// Check if profiling is globally enabled
    pub fn is_enabled(&self) -> bool {
        self.global_enabled
    }

    /// Check if a specific function should be profiled
    pub fn should_profile(&self, function_name: &str, call_count: usize) -> bool {
        if !self.global_enabled {
            return false;
        }

        if let Some(attr) = self.attributes.get(function_name) {
            call_count % attr.sample_rate == 0
        } else {
            false
        }
    }
}

/// Global attribute registry
static mut GLOBAL_REGISTRY: Option<AttributeRegistry> = None;
static REGISTRY_INIT: std::sync::Once = std::sync::Once::new();

/// Get the global attribute registry
pub fn get_registry() -> &'static mut AttributeRegistry {
    unsafe {
        REGISTRY_INIT.call_once(|| {
            GLOBAL_REGISTRY = Some(AttributeRegistry::new());
        });
        GLOBAL_REGISTRY.as_mut().unwrap()
    }
}

/// Wrapper function that applies profiling attributes to any function
pub fn with_profiling<F, R>(function_name: &str, func: F) -> R
where
    F: FnOnce() -> R,
{
    let registry = get_registry();

    // Check if we should profile this call
    static CALL_COUNTS: Lazy<Mutex<std::collections::HashMap<String, usize>>> =
        Lazy::new(|| Mutex::new(std::collections::HashMap::new()));
    let call_count = {
        let mut counts = CALL_COUNTS.lock();
        let count = counts.entry(function_name.to_string()).or_insert(0);
        *count += 1;
        *count
    };

    if !registry.should_profile(function_name, call_count) {
        return func();
    }

    let attr = registry.get_attributes(function_name);

    // Determine profiling name and category
    let profile_name = attr
        .and_then(|a| a.name.as_ref())
        .cloned()
        .unwrap_or_else(|| function_name.to_string());

    let profile_category = attr
        .and_then(|a| a.category.as_ref())
        .cloned()
        .unwrap_or_else(|| "function".to_string());

    let start_time = Instant::now();

    // Set up profiling scope
    let _guard = ProfileScope::simple(profile_name.clone(), profile_category.clone());

    // Execute the function
    let result = func();

    let duration = start_time.elapsed();
    let duration_us = duration.as_micros() as u64;

    // Check minimum duration threshold
    if let Some(attr) = attr {
        if duration_us < attr.min_duration_us {
            return result;
        }
    }

    result
}

/// Helper macro for creating profiled function wrappers
#[macro_export]
macro_rules! profiled_fn {
    ($name:expr, $func:expr) => {
        $crate::attributes::ProfiledFunction::new($func, $name.to_string(), "function".to_string())
    };
    ($name:expr, $category:expr, $func:expr) => {
        $crate::attributes::ProfiledFunction::new($func, $name.to_string(), $category.to_string())
    };
}

/// Attribute-like macro for profiling functions
#[macro_export]
macro_rules! profile_attribute {
    // Basic profiling
    (#[profile]) => {
        let _attr_guard = $crate::cpu::ProfileScope::simple(
            format!("{}::{}", module_path!(), function_name!()),
            "function".to_string(),
        );
    };

    // Profiling with custom name
    (#[profile(name = $name:expr)]) => {
        let _attr_guard =
            $crate::cpu::ProfileScope::simple($name.to_string(), "function".to_string());
    };

    // Profiling with custom name and category
    (#[profile(name = $name:expr, category = $category:expr)]) => {
        let _attr_guard =
            $crate::cpu::ProfileScope::simple($name.to_string(), $category.to_string());
    };

    // Profiling with sampling
    (#[profile(sample_rate = $rate:expr)]) => {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

        let call_num = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let _attr_guard = if call_num % $rate == 0 {
            Some($crate::cpu::ProfileScope::simple(
                format!("{}::{}", module_path!(), function_name!()),
                "sampled_function".to_string(),
            ))
        } else {
            None
        };
    };
}

/// Method profiling wrapper for structs
pub trait ProfiledStruct {
    /// Execute a method with profiling
    fn profiled_method<F, R>(&self, method_name: &str, func: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        let type_name = type_name::<Self>();
        let full_name = format!("{type_name}::{method_name}");

        let _guard = ProfileScope::simple(full_name, "method".to_string());
        func(self)
    }

    /// Execute a mutable method with profiling
    fn profiled_method_mut<F, R>(&mut self, method_name: &str, func: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let type_name = type_name::<Self>();
        let full_name = format!("{type_name}::{method_name}");

        let _guard = ProfileScope::simple(full_name, "method".to_string());
        func(self)
    }
}

/// Blanket implementation for all types
impl<T> ProfiledStruct for T {}

/// Conditional profiling based on feature flags or runtime conditions
pub struct ConditionalProfiler {
    condition: Box<dyn Fn() -> bool + Send + Sync>,
    fallback_enabled: bool,
}

impl ConditionalProfiler {
    /// Create a new conditional profiler
    pub fn new<F>(condition: F) -> Self
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        Self {
            condition: Box::new(condition),
            fallback_enabled: true,
        }
    }

    /// Create a conditional profiler that only profiles in debug mode
    pub fn debug_only() -> Self {
        Self::new(|| cfg!(debug_assertions))
    }

    /// Create a conditional profiler based on an environment variable
    pub fn env_var(var_name: &str) -> Self {
        let var_name = var_name.to_string();
        Self::new(move || {
            std::env::var(&var_name)
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false)
        })
    }

    /// Create a conditional profiler based on a feature flag
    pub fn feature_flag(feature: &str) -> Self {
        let enabled = feature == "profiling";
        Self::new(move || enabled)
    }

    /// Execute a function with conditional profiling
    pub fn profile<F, R>(&self, name: &str, category: &str, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        if (self.condition)() {
            let _guard = ProfileScope::simple(name.to_string(), category.to_string());
            func()
        } else {
            func()
        }
    }
}

/// Helper for async function profiling
pub struct AsyncProfiler;

impl AsyncProfiler {
    /// Profile an async function
    pub async fn profile<F, Fut, R>(name: &str, category: &str, func: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let _guard = ProfileScope::simple(name.to_string(), category.to_string());
        func().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_profile_attribute_creation() {
        let attr = ProfileAttribute::new()
            .with_name("test_function")
            .with_category("test")
            .with_stack_trace()
            .with_memory_tracking()
            .with_sample_rate(5)
            .with_min_duration_us(1000);

        assert_eq!(attr.name, Some("test_function".to_string()));
        assert_eq!(attr.category, Some("test".to_string()));
        assert!(attr.stack_trace);
        assert!(attr.track_memory);
        assert_eq!(attr.sample_rate, 5);
        assert_eq!(attr.min_duration_us, 1000);
    }

    #[test]
    fn test_attribute_registry() {
        let mut registry = AttributeRegistry::new();

        let attr = ProfileAttribute::new()
            .with_name("test_func")
            .with_category("test");

        registry.register("my_function", attr);

        let retrieved = registry.get_attributes("my_function");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, Some("test_func".to_string()));
    }

    #[test]
    fn test_sampling() {
        let mut registry = AttributeRegistry::new();

        let attr = ProfileAttribute::new().with_sample_rate(3);
        registry.register("sampled_func", attr);

        // Should profile on calls 3, 6, 9, etc.
        assert!(!registry.should_profile("sampled_func", 1));
        assert!(!registry.should_profile("sampled_func", 2));
        assert!(registry.should_profile("sampled_func", 3));
        assert!(!registry.should_profile("sampled_func", 4));
        assert!(!registry.should_profile("sampled_func", 5));
        assert!(registry.should_profile("sampled_func", 6));
    }

    #[test]
    fn test_profiled_function() {
        let func = || {
            std::thread::sleep(Duration::from_millis(1));
            42
        };

        let profiled = ProfiledFunction::new(func, "test_func".to_string(), "test".to_string());
        let result = profiled.call();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_with_profiling() {
        let result = with_profiling("test_function", || {
            std::thread::sleep(Duration::from_millis(1));
            "success"
        });
        assert_eq!(result, "success");
    }

    #[test]
    fn test_profiled_struct() {
        struct TestStruct {
            value: i32,
        }

        let mut test_struct = TestStruct { value: 42 };

        let result = test_struct.profiled_method("get_value", |s| s.value);
        assert_eq!(result, 42);

        test_struct.profiled_method_mut("set_value", |s| {
            s.value = 100;
        });
        assert_eq!(test_struct.value, 100);
    }

    #[test]
    fn test_conditional_profiler() {
        let profiler = ConditionalProfiler::new(|| true);

        let result = profiler.profile("test_op", "test", || {
            std::thread::sleep(Duration::from_millis(1));
            "conditional_result"
        });
        assert_eq!(result, "conditional_result");

        // Test with false condition
        let profiler = ConditionalProfiler::new(|| false);
        let result = profiler.profile("test_op", "test", || {
            std::thread::sleep(Duration::from_millis(1));
            "not_profiled"
        });
        assert_eq!(result, "not_profiled");
    }

    #[test]
    fn test_debug_only_profiler() {
        let profiler = ConditionalProfiler::debug_only();

        let result = profiler.profile("debug_op", "debug", || "debug_result");
        assert_eq!(result, "debug_result");
    }

    #[tokio::test]
    async fn test_async_profiler() {
        let result = AsyncProfiler::profile("async_test", "async", || async {
            tokio::time::sleep(Duration::from_millis(1)).await;
            "async_success"
        })
        .await;

        assert_eq!(result, "async_success");
    }

    #[test]
    fn test_profiled_fn_macro() {
        let func = || {
            std::thread::sleep(Duration::from_millis(1));
            "macro_result"
        };

        let profiled = profiled_fn!("macro_test", func);
        let result = profiled.call();
        assert_eq!(result, "macro_result");
    }
}
