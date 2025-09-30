//! Common utilities to reduce code duplication across modules

use torsh_core::error::{Result, TorshError};

/// Utility macros for common patterns
///
/// This module provides macros and functions to eliminate repetitive code
/// patterns found throughout the torsh-data crate.
/// Validate probability value is in [0, 1] range
pub fn validate_probability(prob: f32, name: &str) -> Result<()> {
    if !(0.0..=1.0).contains(&prob) {
        return Err(TorshError::InvalidArgument(format!(
            "{name} must be between 0 and 1, got {prob}"
        )));
    }
    Ok(())
}

/// Validate that a range is valid (min <= max)
pub fn validate_range<T: PartialOrd + std::fmt::Debug>(range: (T, T), name: &str) -> Result<()> {
    if range.0 > range.1 {
        return Err(TorshError::InvalidArgument(format!(
            "Invalid {name} range: {range:?}"
        )));
    }
    Ok(())
}

/// Validate that a value is positive
pub fn validate_positive<T: PartialOrd + Default + std::fmt::Debug>(
    value: T,
    name: &str,
) -> Result<()> {
    if value <= T::default() {
        return Err(TorshError::InvalidArgument(format!(
            "{name} must be positive, got {value:?}"
        )));
    }
    Ok(())
}

/// Validate that vectors have the same length
pub fn validate_same_length<T, U>(vec1: &[T], vec2: &[U], name1: &str, name2: &str) -> Result<()> {
    if vec1.len() != vec2.len() {
        return Err(TorshError::InvalidArgument(format!(
            "{} and {} must have the same length, got {} and {}",
            name1,
            name2,
            vec1.len(),
            vec2.len()
        )));
    }
    Ok(())
}

/// Validate that a vector is not empty
pub fn validate_not_empty<T>(vec: &[T], name: &str) -> Result<()> {
    if vec.is_empty() {
        return Err(TorshError::InvalidArgument(format!(
            "{name} cannot be empty"
        )));
    }
    Ok(())
}

/// Error utilities for common error patterns
pub mod errors {
    use torsh_core::error::TorshError;

    /// Create an invalid index error
    pub fn invalid_index(index: usize, size: usize) -> TorshError {
        TorshError::IndexError { index, size }
    }

    /// Create an invalid argument error
    pub fn invalid_argument(msg: impl Into<String>) -> TorshError {
        TorshError::InvalidArgument(msg.into())
    }

    /// Create a configuration error
    pub fn config_error(msg: impl Into<String>) -> TorshError {
        TorshError::InvalidArgument(msg.into())
    }

    /// Create an empty batch error
    pub fn empty_batch() -> TorshError {
        TorshError::InvalidArgument("Cannot process empty batch".to_string())
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> TorshError {
        TorshError::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        }
    }

    /// Create a file not found error
    pub fn file_not_found(path: &std::path::Path) -> TorshError {
        TorshError::InvalidArgument(format!("File not found: {}", path.display()))
    }

    /// Create an invalid format error
    pub fn invalid_format(expected: &str, got: &str) -> TorshError {
        TorshError::InvalidArgument(format!("Expected {expected} format, got {got}"))
    }
}

/// Macro to create a simple constructor with validation
#[macro_export]
macro_rules! validated_constructor {
    // Constructor with single probability validation
    ($name:ident, $field:ident: f32, probability) => {
        impl $name {
            /// Create a new instance
            pub fn new($field: f32) -> Result<Self> {
                $crate::utils::validate_probability($field, stringify!($field))?;
                Ok(Self { $field })
            }
        }
    };

    // Constructor with range validation
    ($name:ident, $field:ident: ($t1:ty, $t2:ty), range) => {
        impl $name {
            /// Create a new instance
            pub fn new($field: ($t1, $t2)) -> Result<Self> {
                $crate::utils::validate_range($field, stringify!($field))?;
                Ok(Self { $field })
            }
        }
    };

    // Constructor with positive validation
    ($name:ident, $field:ident: $t:ty, positive) => {
        impl $name {
            /// Create a new instance
            pub fn new($field: $t) -> Result<Self> {
                $crate::utils::validate_positive($field, stringify!($field))?;
                Ok(Self { $field })
            }
        }
    };

    // Constructor with size tuple (for image operations)
    ($name:ident, size: ($t1:ty, $t2:ty)) => {
        impl $name {
            /// Create a new instance
            pub fn new(size: ($t1, $t2)) -> Self {
                Self { size }
            }
        }
    };

    // Constructor with custom validation
    ($name:ident, $($field:ident: $t:ty),+, validate = $validator:expr) => {
        impl $name {
            /// Create a new instance
            pub fn new($($field: $t),+) -> Result<Self> {
                $validator(&$($field),+)?;
                Ok(Self { $($field),+ })
            }
        }
    };
}

/// Macro to create builder patterns with fluent interface
#[macro_export]
macro_rules! builder_pattern {
    ($name:ident, $($field:ident: $t:ty),+) => {
        // Note: This macro would generate methods like with_field_name
        // For now, users should implement builders manually for better control
        // Example:
        // impl MyStruct {
        //     pub fn with_field(mut self, field: Type) -> Self {
        //         self.field = field;
        //         self
        //     }
        // }
    };
}

/// Common transform implementation macro
#[macro_export]
macro_rules! simple_random_transform {
    ($name:ident, $input:ty, $output:ty, $prob_field:ident, $transform_fn:expr) => {
        impl $crate::transforms::Transform<$input> for $name {
            type Output = $output;

            fn transform(&self, input: $input) -> Result<Self::Output> {
                // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
                use scirs2_core::random::{Random, Rng};
                let mut rng = Random::seed(42);

                if rng.gen::<f32>() < self.$prob_field {
                    $transform_fn(input, &mut rng)
                } else {
                    Ok(input)
                }
            }

            fn is_deterministic(&self) -> bool {
                false
            }
        }
    };
}

/// Common dataset path validation
pub fn validate_dataset_path(path: &std::path::Path, name: &str) -> Result<()> {
    if !path.exists() {
        return Err(TorshError::InvalidArgument(format!(
            "{} path does not exist: {}",
            name,
            path.display()
        )));
    }
    Ok(())
}

/// Common file extension validation
pub fn validate_file_extension(path: &std::path::Path, extensions: &[&str]) -> Result<()> {
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        if extensions.iter().any(|&e| e == ext_str) {
            return Ok(());
        }
    }

    Err(TorshError::InvalidArgument(format!(
        "File must have one of these extensions: {:?}, got: {}",
        extensions,
        path.display()
    )))
}

/// Common tensor shape validation
pub fn validate_tensor_shape(shape: &[usize], expected_dims: usize, name: &str) -> Result<()> {
    if shape.len() != expected_dims {
        return Err(TorshError::InvalidArgument(format!(
            "{} tensor must have {} dimensions, got {}",
            name,
            expected_dims,
            shape.len()
        )));
    }
    Ok(())
}

/// Helper for creating size tuples with validation
pub fn create_size_tuple(width: usize, height: usize) -> Result<(usize, usize)> {
    validate_positive(width, "width")?;
    validate_positive(height, "height")?;
    Ok((width, height))
}

/// Utility traits for common functionality
pub trait Resettable {
    /// Reset to initial state
    fn reset(&mut self);
}

pub trait Configurable<T> {
    /// Configure with settings
    fn configure(&mut self, config: T) -> Result<()>;
}

pub trait Cacheable {
    /// Clear any cached data
    fn clear_cache(&mut self);

    /// Get cache hit rate if applicable
    fn cache_hit_rate(&self) -> Option<f32> {
        None
    }
}

/// Helper for progress tracking
pub struct ProgressTracker {
    current: usize,
    total: usize,
    last_reported: f32,
    report_interval: f32,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        Self {
            current: 0,
            total,
            last_reported: 0.0,
            report_interval: 0.1, // Report every 10%
        }
    }

    /// Update progress and return true if should report
    pub fn update(&mut self) -> bool {
        self.current += 1;
        let progress = self.current as f32 / self.total as f32;

        if progress - self.last_reported >= self.report_interval {
            self.last_reported = progress;
            true
        } else {
            false
        }
    }

    /// Get current progress as percentage
    pub fn percentage(&self) -> f32 {
        (self.current as f32 / self.total as f32) * 100.0
    }

    /// Check if completed
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

/// Performance measurement utilities
pub mod performance {
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Simple performance timer
    pub struct Timer {
        start: Instant,
        measurements: VecDeque<Duration>,
        max_samples: usize,
    }

    impl Timer {
        /// Create a new timer
        pub fn new() -> Self {
            Self {
                start: Instant::now(),
                measurements: VecDeque::new(),
                max_samples: 100,
            }
        }

        /// Start timing
        pub fn start(&mut self) {
            self.start = Instant::now();
        }

        /// Stop timing and record measurement
        pub fn stop(&mut self) -> Duration {
            let duration = self.start.elapsed();

            if self.measurements.len() >= self.max_samples {
                self.measurements.pop_front();
            }
            self.measurements.push_back(duration);

            duration
        }

        /// Get average measurement
        pub fn average(&self) -> Option<Duration> {
            if self.measurements.is_empty() {
                None
            } else {
                let total: Duration = self.measurements.iter().sum();
                Some(total / self.measurements.len() as u32)
            }
        }

        /// Get throughput in items per second
        pub fn throughput(&self, items: usize) -> Option<f64> {
            self.average().map(|avg| items as f64 / avg.as_secs_f64())
        }
    }

    impl Default for Timer {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Memory management utilities
pub mod memory {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Simple memory pool for reusing allocations
    pub struct MemoryPool<T> {
        available: Vec<Vec<T>>,
        capacity: usize,
        default_size: usize,
    }

    impl<T: Clone + Default> MemoryPool<T> {
        /// Create a new memory pool
        pub fn new(capacity: usize, default_size: usize) -> Self {
            Self {
                available: Vec::with_capacity(capacity),
                capacity,
                default_size,
            }
        }

        /// Get a buffer from the pool
        pub fn get(&mut self) -> Vec<T> {
            self.available
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(self.default_size))
        }

        /// Return a buffer to the pool
        pub fn put(&mut self, mut buffer: Vec<T>) {
            if self.available.len() < self.capacity {
                buffer.clear();
                self.available.push(buffer);
            }
        }

        /// Get current pool size
        pub fn size(&self) -> usize {
            self.available.len()
        }
    }

    /// Thread-safe memory pool
    pub type SharedMemoryPool<T> = Arc<Mutex<MemoryPool<T>>>;

    /// Memory usage tracker
    pub struct MemoryTracker {
        allocations: HashMap<String, usize>,
        peak_memory: usize,
        current_memory: usize,
    }

    impl MemoryTracker {
        /// Create a new memory tracker
        pub fn new() -> Self {
            Self {
                allocations: HashMap::new(),
                peak_memory: 0,
                current_memory: 0,
            }
        }

        /// Record an allocation
        pub fn allocate(&mut self, name: &str, size: usize) {
            self.current_memory += size;
            self.peak_memory = self.peak_memory.max(self.current_memory);
            *self.allocations.entry(name.to_string()).or_insert(0) += size;
        }

        /// Record a deallocation
        pub fn deallocate(&mut self, name: &str, size: usize) {
            self.current_memory = self.current_memory.saturating_sub(size);
            if let Some(total) = self.allocations.get_mut(name) {
                *total = total.saturating_sub(size);
            }
        }

        /// Get current memory usage
        pub fn current_usage(&self) -> usize {
            self.current_memory
        }

        /// Get peak memory usage
        pub fn peak_usage(&self) -> usize {
            self.peak_memory
        }

        /// Get memory usage breakdown
        pub fn breakdown(&self) -> &HashMap<String, usize> {
            &self.allocations
        }
    }

    impl Default for MemoryTracker {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Batch processing utilities
pub mod batch {
    use rayon::prelude::*;
    use std::sync::mpsc;
    use std::thread;

    /// Process data in parallel batches
    pub fn parallel_batch_process<T, R, F>(data: Vec<T>, batch_size: usize, processor: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&[T]) -> R + Send + Sync,
    {
        data.par_chunks(batch_size).map(processor).collect()
    }

    /// Async batch processor with channels
    pub struct AsyncBatchProcessor<T, R> {
        sender: mpsc::Sender<Vec<T>>,
        receiver: mpsc::Receiver<R>,
        _handle: thread::JoinHandle<()>,
    }

    impl<T, R> AsyncBatchProcessor<T, R>
    where
        T: Send + 'static,
        R: Send + 'static,
    {
        /// Create a new async batch processor
        pub fn new<F>(batch_size: usize, processor: F) -> Self
        where
            F: Fn(Vec<T>) -> R + Send + 'static,
        {
            let (input_sender, input_receiver) = mpsc::channel();
            let (output_sender, output_receiver) = mpsc::channel();

            let handle = thread::spawn(move || {
                let mut buffer = Vec::with_capacity(batch_size);

                while let Ok(mut data) = input_receiver.recv() {
                    buffer.append(&mut data);

                    while buffer.len() >= batch_size {
                        let batch = buffer.drain(..batch_size).collect();
                        let result = processor(batch);
                        if output_sender.send(result).is_err() {
                            break;
                        }
                    }
                }

                // Process remaining data
                if !buffer.is_empty() {
                    let result = processor(buffer);
                    let _ = output_sender.send(result);
                }
            });

            Self {
                sender: input_sender,
                receiver: output_receiver,
                _handle: handle,
            }
        }

        /// Send data for processing
        pub fn send(&self, data: Vec<T>) -> Result<(), mpsc::SendError<Vec<T>>> {
            self.sender.send(data)
        }

        /// Receive processed results
        pub fn recv(&self) -> Result<R, mpsc::RecvError> {
            self.receiver.recv()
        }

        /// Try to receive without blocking
        pub fn try_recv(&self) -> Result<R, mpsc::TryRecvError> {
            self.receiver.try_recv()
        }
    }
}

/// Concurrent utilities for thread-safe operations
pub mod concurrent {
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Thread-safe cache with concurrent access
    pub struct ConcurrentCache<K, V>
    where
        K: Eq + std::hash::Hash,
    {
        data: Arc<RwLock<HashMap<K, V>>>,
        max_size: usize,
    }

    impl<K, V> ConcurrentCache<K, V>
    where
        K: Eq + std::hash::Hash + Clone,
        V: Clone,
    {
        /// Create a new concurrent cache
        pub fn new(max_size: usize) -> Self {
            Self {
                data: Arc::new(RwLock::new(HashMap::new())),
                max_size,
            }
        }

        /// Get a value from the cache
        pub fn get(&self, key: &K) -> Option<V> {
            self.data.read().get(key).cloned()
        }

        /// Insert a value into the cache
        pub fn insert(&self, key: K, value: V) {
            let mut data = self.data.write();

            // Simple eviction if at capacity
            if data.len() >= self.max_size && !data.contains_key(&key) {
                if let Some(first_key) = data.keys().next().cloned() {
                    data.remove(&first_key);
                }
            }

            data.insert(key, value);
        }

        /// Remove a value from the cache
        pub fn remove(&self, key: &K) -> Option<V> {
            self.data.write().remove(key)
        }

        /// Clear the cache
        pub fn clear(&self) {
            self.data.write().clear();
        }

        /// Get cache size
        pub fn len(&self) -> usize {
            self.data.read().len()
        }

        /// Check if cache is empty
        pub fn is_empty(&self) -> bool {
            self.data.read().is_empty()
        }
    }

    /// Thread-safe statistics collector
    pub struct StatisticsCollector {
        data: Arc<Mutex<Vec<f64>>>,
        max_samples: usize,
    }

    impl StatisticsCollector {
        /// Create a new statistics collector
        pub fn new(max_samples: usize) -> Self {
            Self {
                data: Arc::new(Mutex::new(Vec::new())),
                max_samples,
            }
        }

        /// Add a sample
        pub fn add_sample(&self, value: f64) {
            let mut data = self.data.lock();
            if data.len() >= self.max_samples {
                data.remove(0);
            }
            data.push(value);
        }

        /// Get mean
        pub fn mean(&self) -> Option<f64> {
            let data = self.data.lock();
            if data.is_empty() {
                None
            } else {
                Some(data.iter().sum::<f64>() / data.len() as f64)
            }
        }

        /// Get standard deviation
        pub fn std_dev(&self) -> Option<f64> {
            let data = self.data.lock();
            if data.len() < 2 {
                None
            } else {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let variance =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
                Some(variance.sqrt())
            }
        }

        /// Get min and max
        pub fn min_max(&self) -> Option<(f64, f64)> {
            let data = self.data.lock();
            if data.is_empty() {
                None
            } else {
                let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                Some((min, max))
            }
        }

        /// Get sample count
        pub fn count(&self) -> usize {
            self.data.lock().len()
        }
    }
}

/// Configuration management utilities
pub mod config {
    use std::collections::HashMap;
    use std::env;
    use std::path::Path;

    #[cfg(feature = "serialize")]
    use std::fs;
    use torsh_core::error::{Result, TorshError};

    #[cfg(feature = "serialize")]
    use serde::{Deserialize, Serialize};

    /// Configuration manager for handling settings from multiple sources
    pub struct ConfigManager {
        values: HashMap<String, ConfigValue>,
        env_prefix: String,
    }

    /// Represents different types of configuration values
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub enum ConfigValue {
        String(String),
        Integer(i64),
        Float(f64),
        Boolean(bool),
        Array(Vec<ConfigValue>),
        Object(HashMap<String, ConfigValue>),
    }

    impl ConfigValue {
        /// Convert to string if possible
        pub fn as_string(&self) -> Option<&str> {
            match self {
                ConfigValue::String(s) => Some(s),
                _ => None,
            }
        }

        /// Convert to integer if possible
        pub fn as_i64(&self) -> Option<i64> {
            match self {
                ConfigValue::Integer(i) => Some(*i),
                _ => None,
            }
        }

        /// Convert to float if possible
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                ConfigValue::Float(f) => Some(*f),
                ConfigValue::Integer(i) => Some(*i as f64),
                _ => None,
            }
        }

        /// Convert to boolean if possible
        pub fn as_bool(&self) -> Option<bool> {
            match self {
                ConfigValue::Boolean(b) => Some(*b),
                _ => None,
            }
        }

        /// Convert to array if possible
        pub fn as_array(&self) -> Option<&Vec<ConfigValue>> {
            match self {
                ConfigValue::Array(arr) => Some(arr),
                _ => None,
            }
        }

        /// Convert to object if possible
        pub fn as_object(&self) -> Option<&HashMap<String, ConfigValue>> {
            match self {
                ConfigValue::Object(obj) => Some(obj),
                _ => None,
            }
        }
    }

    impl ConfigManager {
        /// Create a new configuration manager
        pub fn new() -> Self {
            Self {
                values: HashMap::new(),
                env_prefix: "TORSH_".to_string(),
            }
        }

        /// Create a new configuration manager with custom environment prefix
        pub fn with_env_prefix(prefix: &str) -> Self {
            Self {
                values: HashMap::new(),
                env_prefix: prefix.to_string(),
            }
        }

        /// Load configuration from JSON file
        #[cfg(feature = "serialize")]
        pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
            let content = fs::read_to_string(path).map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to read config file: {}", e))
            })?;

            let json_value: serde_json::Value = serde_json::from_str(&content)
                .map_err(|e| TorshError::InvalidArgument(format!("Failed to parse JSON: {}", e)))?;

            self.load_from_json_value("", &json_value);
            Ok(())
        }

        /// Load configuration from JSON file (feature-gated fallback)
        #[cfg(not(feature = "serialize"))]
        pub fn load_from_file<P: AsRef<Path>>(&mut self, _path: P) -> Result<()> {
            Err(TorshError::InvalidArgument(
                "JSON loading requires 'serialize' feature. Enable with --features serialize"
                    .to_string(),
            ))
        }

        /// Load configuration from environment variables
        pub fn load_from_env(&mut self) {
            for (key, value) in env::vars() {
                if key.starts_with(&self.env_prefix) {
                    let config_key = key.strip_prefix(&self.env_prefix).unwrap().to_lowercase();
                    self.set_from_string(&config_key, &value);
                }
            }
        }

        /// Set a configuration value
        pub fn set(&mut self, key: &str, value: ConfigValue) {
            self.values.insert(key.to_string(), value);
        }

        /// Get a configuration value
        pub fn get(&self, key: &str) -> Option<&ConfigValue> {
            self.values.get(key)
        }

        /// Get a string value with default
        pub fn get_string(&self, key: &str, default: &str) -> String {
            self.get(key)
                .and_then(|v| v.as_string())
                .unwrap_or(default)
                .to_string()
        }

        /// Get an integer value with default
        pub fn get_i64(&self, key: &str, default: i64) -> i64 {
            self.get(key).and_then(|v| v.as_i64()).unwrap_or(default)
        }

        /// Get a float value with default
        pub fn get_f64(&self, key: &str, default: f64) -> f64 {
            self.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        }

        /// Get a boolean value with default
        pub fn get_bool(&self, key: &str, default: bool) -> bool {
            self.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
        }

        /// Check if a key exists
        pub fn contains_key(&self, key: &str) -> bool {
            self.values.contains_key(key)
        }

        /// Get all configuration keys
        pub fn keys(&self) -> Vec<&String> {
            self.values.keys().collect()
        }

        /// Clear all configuration
        pub fn clear(&mut self) {
            self.values.clear();
        }

        /// Merge configuration from another manager
        pub fn merge(&mut self, other: &ConfigManager) {
            for (key, value) in &other.values {
                self.values.insert(key.clone(), value.clone());
            }
        }

        /// Set value from string representation
        fn set_from_string(&mut self, key: &str, value: &str) {
            // Try to parse as different types
            if let Ok(b) = value.parse::<bool>() {
                self.set(key, ConfigValue::Boolean(b));
            } else if let Ok(i) = value.parse::<i64>() {
                self.set(key, ConfigValue::Integer(i));
            } else if let Ok(f) = value.parse::<f64>() {
                self.set(key, ConfigValue::Float(f));
            } else {
                self.set(key, ConfigValue::String(value.to_string()));
            }
        }

        /// Load from JSON value recursively
        #[cfg(feature = "serialize")]
        fn load_from_json_value(&mut self, prefix: &str, value: &serde_json::Value) {
            match value {
                serde_json::Value::String(s) => {
                    self.set(prefix, ConfigValue::String(s.clone()));
                }
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        self.set(prefix, ConfigValue::Integer(i));
                    } else if let Some(f) = n.as_f64() {
                        self.set(prefix, ConfigValue::Float(f));
                    }
                }
                serde_json::Value::Bool(b) => {
                    self.set(prefix, ConfigValue::Boolean(*b));
                }
                serde_json::Value::Array(arr) => {
                    let config_arr: Vec<ConfigValue> = arr
                        .iter()
                        .map(|v| self.json_value_to_config_value(v))
                        .collect();
                    self.set(prefix, ConfigValue::Array(config_arr));
                }
                serde_json::Value::Object(obj) => {
                    for (key, val) in obj {
                        let new_key = if prefix.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", prefix, key)
                        };
                        self.load_from_json_value(&new_key, val);
                    }
                }
                serde_json::Value::Null => {}
            }
        }

        /// Convert JSON value to ConfigValue
        #[cfg(feature = "serialize")]
        fn json_value_to_config_value(&self, value: &serde_json::Value) -> ConfigValue {
            match value {
                serde_json::Value::String(s) => ConfigValue::String(s.clone()),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        ConfigValue::Integer(i)
                    } else if let Some(f) = n.as_f64() {
                        ConfigValue::Float(f)
                    } else {
                        ConfigValue::String(n.to_string())
                    }
                }
                serde_json::Value::Bool(b) => ConfigValue::Boolean(*b),
                serde_json::Value::Array(arr) => {
                    let config_arr: Vec<ConfigValue> = arr
                        .iter()
                        .map(|v| self.json_value_to_config_value(v))
                        .collect();
                    ConfigValue::Array(config_arr)
                }
                serde_json::Value::Object(obj) => {
                    let mut config_obj = HashMap::new();
                    for (key, val) in obj {
                        config_obj.insert(key.clone(), self.json_value_to_config_value(val));
                    }
                    ConfigValue::Object(config_obj)
                }
                serde_json::Value::Null => ConfigValue::String("null".to_string()),
            }
        }
    }

    impl Default for ConfigManager {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Configuration builder for fluent API
    pub struct ConfigBuilder {
        manager: ConfigManager,
    }

    impl ConfigBuilder {
        /// Create a new config builder
        pub fn new() -> Self {
            Self {
                manager: ConfigManager::new(),
            }
        }

        /// Set environment prefix
        pub fn env_prefix(mut self, prefix: &str) -> Self {
            self.manager.env_prefix = prefix.to_string();
            self
        }

        /// Load from file
        #[cfg(feature = "serialize")]
        pub fn file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
            self.manager.load_from_file(path)?;
            Ok(self)
        }

        /// Load from file (feature-gated fallback)
        #[cfg(not(feature = "serialize"))]
        pub fn file<P: AsRef<Path>>(self, _path: P) -> Result<Self> {
            Err(TorshError::InvalidArgument(
                "JSON file loading requires 'serialize' feature. Enable with --features serialize"
                    .to_string(),
            ))
        }

        /// Load from environment
        pub fn env(mut self) -> Self {
            self.manager.load_from_env();
            self
        }

        /// Set a value
        pub fn set(mut self, key: &str, value: ConfigValue) -> Self {
            self.manager.set(key, value);
            self
        }

        /// Set a string value
        pub fn set_string(mut self, key: &str, value: &str) -> Self {
            self.manager
                .set(key, ConfigValue::String(value.to_string()));
            self
        }

        /// Set an integer value
        pub fn set_i64(mut self, key: &str, value: i64) -> Self {
            self.manager.set(key, ConfigValue::Integer(value));
            self
        }

        /// Set a float value
        pub fn set_f64(mut self, key: &str, value: f64) -> Self {
            self.manager.set(key, ConfigValue::Float(value));
            self
        }

        /// Set a boolean value
        pub fn set_bool(mut self, key: &str, value: bool) -> Self {
            self.manager.set(key, ConfigValue::Boolean(value));
            self
        }

        /// Build the configuration manager
        pub fn build(self) -> ConfigManager {
            self.manager
        }
    }

    impl Default for ConfigBuilder {
        fn default() -> Self {
            Self::new()
        }
    }
}

// Re-export commonly used macros
pub use builder_pattern;
pub use simple_random_transform;
pub use validated_constructor;

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_validate_probability() {
        assert!(validate_probability(0.5, "test").is_ok());
        assert!(validate_probability(0.0, "test").is_ok());
        assert!(validate_probability(1.0, "test").is_ok());
        assert!(validate_probability(-0.1, "test").is_err());
        assert!(validate_probability(1.1, "test").is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range((0.0, 1.0), "test").is_ok());
        assert!(validate_range((1.0, 1.0), "test").is_ok());
        assert!(validate_range((1.0, 0.0), "test").is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1, "test").is_ok());
        assert!(validate_positive(0, "test").is_err());
        assert!(validate_positive(-1, "test").is_err());
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new(10);
        assert!(!tracker.is_complete());

        // Should report at 10%, 20%, etc.
        assert!(tracker.update()); // 10% - should report
        assert!(tracker.update()); // 20% - should also report (10% interval)

        for _ in 0..8 {
            tracker.update();
        }
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_performance_timer() {
        let mut timer = performance::Timer::new();
        timer.start();
        thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();

        assert!(duration >= Duration::from_millis(10));
        assert!(timer.average().is_some());
        assert!(timer.throughput(100).is_some());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = memory::MemoryPool::<u8>::new(5, 1024);
        assert_eq!(pool.size(), 0);

        let buffer1 = pool.get();
        assert_eq!(buffer1.capacity(), 1024);

        pool.put(buffer1);
        assert_eq!(pool.size(), 1);

        let buffer2 = pool.get();
        assert_eq!(pool.size(), 0);
        assert_eq!(buffer2.len(), 0); // Should be cleared
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = memory::MemoryTracker::new();
        assert_eq!(tracker.current_usage(), 0);

        tracker.allocate("test", 1024);
        assert_eq!(tracker.current_usage(), 1024);
        assert_eq!(tracker.peak_usage(), 1024);

        tracker.deallocate("test", 512);
        assert_eq!(tracker.current_usage(), 512);
        assert_eq!(tracker.peak_usage(), 1024); // Peak should remain
    }

    #[test]
    fn test_parallel_batch_process() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let results = batch::parallel_batch_process(data, 3, |chunk| chunk.iter().sum::<i32>());

        assert_eq!(results.len(), 4); // 4 batches: [1,2,3], [4,5,6], [7,8,9], [10]
        assert_eq!(results[0], 6); // 1+2+3
        assert_eq!(results[1], 15); // 4+5+6
        assert_eq!(results[2], 24); // 7+8+9
        assert_eq!(results[3], 10); // 10
    }

    #[test]
    fn test_concurrent_cache() {
        let cache = concurrent::ConcurrentCache::new(2);

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key2"), Some("value2"));
        assert_eq!(cache.get(&"key3"), None);

        // Test eviction
        cache.insert("key3", "value3");
        assert_eq!(cache.len(), 2); // Should evict oldest
    }

    #[test]
    fn test_statistics_collector() {
        let collector = concurrent::StatisticsCollector::new(100);
        assert_eq!(collector.count(), 0);

        collector.add_sample(1.0);
        collector.add_sample(2.0);
        collector.add_sample(3.0);

        assert_eq!(collector.count(), 3);
        assert_eq!(collector.mean(), Some(2.0));
        assert_eq!(collector.min_max(), Some((1.0, 3.0)));
        assert!(collector.std_dev().is_some());
    }

    #[test]
    fn test_config_manager() {
        let mut config = config::ConfigManager::new();

        // Test setting and getting different types
        config.set(
            "string_key",
            config::ConfigValue::String("test_value".to_string()),
        );
        config.set("int_key", config::ConfigValue::Integer(42));
        config.set("float_key", config::ConfigValue::Float(3.14));
        config.set("bool_key", config::ConfigValue::Boolean(true));

        assert_eq!(config.get_string("string_key", "default"), "test_value");
        assert_eq!(config.get_i64("int_key", 0), 42);
        assert_eq!(config.get_f64("float_key", 0.0), 3.14);
        assert!(config.get_bool("bool_key", false));

        // Test defaults
        assert_eq!(config.get_string("missing_key", "default"), "default");
        assert_eq!(config.get_i64("missing_key", 123), 123);

        // Test contains_key
        assert!(config.contains_key("string_key"));
        assert!(!config.contains_key("missing_key"));

        // Test keys
        let keys = config.keys();
        assert!(keys.len() >= 4);

        // Test clear
        config.clear();
        assert_eq!(config.keys().len(), 0);
    }

    #[test]
    fn test_config_value_conversions() {
        let string_val = config::ConfigValue::String("test".to_string());
        let int_val = config::ConfigValue::Integer(42);
        let float_val = config::ConfigValue::Float(3.14);
        let bool_val = config::ConfigValue::Boolean(true);

        assert_eq!(string_val.as_string(), Some("test"));
        assert_eq!(int_val.as_i64(), Some(42));
        assert_eq!(float_val.as_f64(), Some(3.14));
        assert_eq!(bool_val.as_bool(), Some(true));

        // Test type mismatches
        assert_eq!(string_val.as_i64(), None);
        assert_eq!(int_val.as_string(), None);

        // Test integer to float conversion
        assert_eq!(int_val.as_f64(), Some(42.0));
    }

    #[test]
    fn test_config_builder() {
        let config = config::ConfigBuilder::new()
            .env_prefix("TEST_")
            .set_string("app_name", "torsh-data")
            .set_i64("version", 1)
            .set_f64("threshold", 0.5)
            .set_bool("debug", true)
            .build();

        assert_eq!(config.get_string("app_name", ""), "torsh-data");
        assert_eq!(config.get_i64("version", 0), 1);
        assert_eq!(config.get_f64("threshold", 0.0), 0.5);
        assert!(config.get_bool("debug", false));
    }

    #[test]
    fn test_config_merge() {
        let mut config1 = config::ConfigManager::new();
        config1.set("key1", config::ConfigValue::String("value1".to_string()));
        config1.set("key2", config::ConfigValue::Integer(42));

        let mut config2 = config::ConfigManager::new();
        config2.set("key2", config::ConfigValue::Integer(100)); // Override
        config2.set("key3", config::ConfigValue::Boolean(true)); // New key

        config1.merge(&config2);

        assert_eq!(config1.get_string("key1", ""), "value1");
        assert_eq!(config1.get_i64("key2", 0), 100); // Should be overridden
        assert!(config1.get_bool("key3", false));
    }
}
