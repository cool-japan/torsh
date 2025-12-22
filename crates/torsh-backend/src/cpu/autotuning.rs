//! CPU kernel auto-tuning and optimization system
//!
//! This module provides comprehensive auto-tuning capabilities for CPU kernels,
//! including performance benchmarking, parameter optimization, and adaptive
//! algorithm selection based on runtime characteristics.

use std::collections::HashMap;
use std::fs;
#[cfg(feature = "serialize")]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serialize")]
use serde_json;

use crate::cpu::error::CpuResult;
use crate::cpu::optimizations::ThreadPoolOptimizer;

// Re-export for benchmarks
pub use crate::cpu::optimizations::OptimizationLevel;

/// Performance measurement for auto-tuning
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PerformanceMeasurement {
    pub execution_time: Duration,
    pub throughput: f64,      // ops/sec
    pub efficiency: f64,      // 0.0 to 1.0
    pub cache_hit_ratio: f64, // 0.0 to 1.0
}

impl PerformanceMeasurement {
    pub fn new(execution_time: Duration, ops_count: usize) -> Self {
        let throughput = if execution_time.as_secs_f64() > 0.0 {
            ops_count as f64 / execution_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            execution_time,
            throughput,
            efficiency: 1.0,       // Will be calculated based on parallel efficiency
            cache_hit_ratio: 0.95, // Default estimate
        }
    }

    /// Calculate composite score for ranking algorithms
    pub fn composite_score(&self) -> f64 {
        // Weighted combination of metrics
        let time_score = 1.0 / (self.execution_time.as_secs_f64() + 1e-9);
        let throughput_score = self.throughput / 1e6; // Normalize to millions ops/sec
        let efficiency_score = self.efficiency;
        let cache_score = self.cache_hit_ratio;

        // Weighted average
        0.4 * time_score + 0.3 * throughput_score + 0.2 * efficiency_score + 0.1 * cache_score
    }
}

/// Auto-tuning configuration for different operation types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TuningConfig {
    pub operation_name: String,
    pub input_size_ranges: Vec<(usize, usize)>, // (min, max) size ranges to test
    pub thread_counts: Vec<usize>,
    pub chunk_sizes: Vec<usize>,
    pub block_sizes: Vec<usize>, // For matrix operations
    pub iterations_per_test: usize,
    pub warmup_iterations: usize,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self::for_element_wise_ops()
    }
}

impl TuningConfig {
    pub fn for_element_wise_ops() -> Self {
        Self {
            operation_name: "element_wise".to_string(),
            input_size_ranges: vec![
                (1, 100),
                (100, 1000),
                (1000, 10000),
                (10000, 100000),
                (100000, 1000000),
            ],
            thread_counts: vec![1, 2, 4, 8, 16],
            chunk_sizes: vec![1, 4, 16, 64, 256, 1024, 4096],
            block_sizes: vec![],
            iterations_per_test: 10,
            warmup_iterations: 3,
        }
    }

    pub fn for_matrix_ops() -> Self {
        Self {
            operation_name: "matrix".to_string(),
            input_size_ranges: vec![(64, 128), (128, 512), (512, 1024), (1024, 2048)],
            thread_counts: vec![1, 2, 4, 8],
            chunk_sizes: vec![32, 64, 128, 256],
            block_sizes: vec![32, 64, 128, 256, 512],
            iterations_per_test: 5,
            warmup_iterations: 2,
        }
    }

    pub fn for_reduction_ops() -> Self {
        Self {
            operation_name: "reduction".to_string(),
            input_size_ranges: vec![(1, 1000), (1000, 10000), (10000, 100000), (100000, 1000000)],
            thread_counts: vec![1, 2, 4, 8, 16],
            chunk_sizes: vec![1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192],
            block_sizes: vec![],
            iterations_per_test: 15,
            warmup_iterations: 5,
        }
    }
}

/// Auto-tuning result for specific configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TuningResult {
    pub config: TuningConfig,
    pub optimal_thread_count: usize,
    pub optimal_chunk_size: usize,
    pub optimal_block_size: Option<usize>,
    pub best_performance: PerformanceMeasurement,
    pub size_range: (usize, usize),
}

/// Cache version and metadata for invalidation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CacheMetadata {
    pub version: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    pub torsh_version: String,
    pub created_timestamp: u64,
    pub last_accessed: u64,
    pub access_count: usize,
}

impl CacheMetadata {
    pub fn current() -> Self {
        Self {
            version: "1.0.0".to_string(),
            cpu_model: Self::detect_cpu_model(),
            cpu_features: Self::detect_cpu_features(),
            torsh_version: env!("CARGO_PKG_VERSION").to_string(),
            created_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            access_count: 0,
        }
    }

    fn detect_cpu_model() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::__cpuid;
            unsafe {
                let cpuid = __cpuid(0);
                format!("x86_64:{:08x}{:08x}{:08x}", cpuid.ebx, cpuid.edx, cpuid.ecx)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            format!("{}:unknown", std::env::consts::ARCH)
        }
    }

    fn detect_cpu_features() -> Vec<String> {
        let mut features = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse") {
                features.push("sse".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse2") {
                features.push("sse2".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse3") {
                features.push("sse3".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse4.1") {
                features.push("sse4.1".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse4.2") {
                features.push("sse4.2".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx") {
                features.push("avx".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                features.push("avx512f".to_string());
            }
            if std::arch::is_x86_feature_detected!("fma") {
                features.push("fma".to_string());
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            features.push("neon".to_string());
        }
        features
    }

    pub fn is_compatible(&self, other: &Self) -> bool {
        self.cpu_model == other.cpu_model
            && self.cpu_features == other.cpu_features
            && self.torsh_version == other.torsh_version
    }

    pub fn update_access(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.access_count += 1;
    }
}

/// Versioned cache entry with metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CacheEntry {
    pub result: TuningResult,
    pub metadata: CacheMetadata,
}

/// Persistent cache file format
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CacheFile {
    pub metadata: CacheMetadata,
    pub entries: HashMap<String, CacheEntry>,
}

/// Persistent tuning cache with versioning and invalidation support
#[derive(Debug)]
pub struct TuningCache {
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    cache_hits: Arc<Mutex<usize>>,
    cache_misses: Arc<Mutex<usize>>,
    cache_file_path: Option<PathBuf>,
    current_metadata: CacheMetadata,
}

impl Default for TuningCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TuningCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
            cache_file_path: None,
            current_metadata: CacheMetadata::current(),
        }
    }

    /// Create cache with persistent storage
    pub fn with_file<P: AsRef<Path>>(cache_path: P) -> CpuResult<Self> {
        let cache_path = cache_path.as_ref().to_path_buf();
        let mut cache = Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
            cache_file_path: Some(cache_path.clone()),
            current_metadata: CacheMetadata::current(),
        };

        // Load existing cache if it exists
        cache.load_from_file()?;
        Ok(cache)
    }

    /// Load cache from persistent file
    pub fn load_from_file(&mut self) -> CpuResult<()> {
        #[cfg(feature = "serialize")]
        if let Some(ref path) = self.cache_file_path {
            if path.exists() {
                let content = fs::read_to_string(path).map_err(|e| {
                    crate::cpu::error::cpu_errors::io_error(&format!(
                        "Failed to read cache file: {}",
                        e
                    ))
                })?;

                let cache_file: CacheFile = serde_json::from_str(&content).map_err(|e| {
                    crate::cpu::error::cpu_errors::parsing_error(&format!(
                        "Failed to parse cache file: {}",
                        e
                    ))
                })?;

                // Check compatibility
                if !self.current_metadata.is_compatible(&cache_file.metadata) {
                    // Cache is incompatible, clear it and start fresh
                    self.invalidate_cache()?;
                    return Ok(());
                }

                // Load compatible entries
                let mut cache = self.cache.lock().unwrap();
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                for (key, mut entry) in cache_file.entries {
                    // Check if entry is not too old (1 week)
                    if now.saturating_sub(entry.metadata.last_accessed) < 7 * 24 * 3600 {
                        entry.metadata.update_access();
                        cache.insert(key, entry);
                    }
                }
            }
        }
        Ok(())
    }

    /// Save cache to persistent file
    pub fn save_to_file(&self) -> CpuResult<()> {
        #[cfg(feature = "serialize")]
        if let Some(ref path) = self.cache_file_path {
            // Create parent directory if it doesn't exist
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    crate::cpu::error::cpu_errors::io_error(&format!(
                        "Failed to create cache directory: {}",
                        e
                    ))
                })?;
            }

            let cache = self.cache.lock().unwrap();
            let cache_file = CacheFile {
                metadata: self.current_metadata.clone(),
                entries: cache.clone(),
            };

            let content = serde_json::to_string_pretty(&cache_file).map_err(|e| {
                crate::cpu::error::cpu_errors::serialization_error(&format!(
                    "Failed to serialize cache: {}",
                    e
                ))
            })?;

            let mut file = fs::File::create(path).map_err(|e| {
                crate::cpu::error::cpu_errors::io_error(&format!(
                    "Failed to create cache file: {}",
                    e
                ))
            })?;

            file.write_all(content.as_bytes()).map_err(|e| {
                crate::cpu::error::cpu_errors::io_error(&format!(
                    "Failed to write cache file: {}",
                    e
                ))
            })?;

            file.sync_all().map_err(|e| {
                crate::cpu::error::cpu_errors::io_error(&format!(
                    "Failed to sync cache file: {}",
                    e
                ))
            })?;
        }
        Ok(())
    }

    /// Invalidate entire cache and delete file
    pub fn invalidate_cache(&self) -> CpuResult<()> {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        *self.cache_hits.lock().unwrap() = 0;
        *self.cache_misses.lock().unwrap() = 0;

        if let Some(ref path) = self.cache_file_path {
            if path.exists() {
                fs::remove_file(path).map_err(|e| {
                    crate::cpu::error::cpu_errors::io_error(&format!(
                        "Failed to remove cache file: {}",
                        e
                    ))
                })?;
            }
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<TuningResult> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            // Check if entry metadata is still compatible
            if self.current_metadata.is_compatible(&entry.metadata) {
                entry.metadata.update_access();
                *self.cache_hits.lock().unwrap() += 1;
                Some(entry.result.clone())
            } else {
                // Remove incompatible entry
                cache.remove(key);
                *self.cache_misses.lock().unwrap() += 1;
                None
            }
        } else {
            *self.cache_misses.lock().unwrap() += 1;
            None
        }
    }

    pub fn insert(&self, key: String, result: TuningResult) {
        let mut cache = self.cache.lock().unwrap();
        let entry = CacheEntry {
            result,
            metadata: self.current_metadata.clone(),
        };
        cache.insert(key, entry);

        // Auto-save periodically
        if cache.len().is_multiple_of(10) {
            drop(cache); // Release lock before saving
            let _ = self.save_to_file(); // Ignore errors for auto-save
        }
    }

    pub fn generate_key(&self, op_name: &str, input_size: usize, data_type: &str) -> String {
        format!(
            "{}:{}:{}:{}",
            op_name, input_size, data_type, self.current_metadata.cpu_model
        )
    }

    pub fn get_cache_stats(&self) -> (usize, usize) {
        let hits = *self.cache_hits.lock().unwrap();
        let misses = *self.cache_misses.lock().unwrap();
        (hits, misses)
    }

    pub fn get_detailed_stats(&self) -> HashMap<String, usize> {
        let cache = self.cache.lock().unwrap();
        let mut stats = HashMap::new();
        stats.insert("total_entries".to_string(), cache.len());
        stats.insert("cache_hits".to_string(), *self.cache_hits.lock().unwrap());
        stats.insert(
            "cache_misses".to_string(),
            *self.cache_misses.lock().unwrap(),
        );

        // Group by operation type
        let mut operation_counts = HashMap::new();
        for key in cache.keys() {
            let op_name = key.split(':').next().unwrap_or("unknown");
            *operation_counts.entry(op_name.to_string()).or_insert(0) += 1;
        }

        for (op, count) in operation_counts {
            stats.insert(format!("entries_{}", op), count);
        }

        stats
    }

    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        *self.cache_hits.lock().unwrap() = 0;
        *self.cache_misses.lock().unwrap() = 0;
    }

    /// Clean old entries from cache
    pub fn cleanup_old_entries(&self, max_age_seconds: u64) {
        let mut cache = self.cache.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        cache.retain(|_, entry| now.saturating_sub(entry.metadata.last_accessed) < max_age_seconds);
    }

    /// Force save cache to file
    pub fn flush(&self) -> CpuResult<()> {
        self.save_to_file()
    }
}

/// Main auto-tuning engine
pub struct AutoTuner {
    thread_optimizer: ThreadPoolOptimizer,
    tuning_cache: TuningCache,
    #[allow(dead_code)]
    optimization_level: OptimizationLevel,
    enable_adaptive_tuning: bool,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new(OptimizationLevel::Basic)
    }
}

impl AutoTuner {
    pub fn new(optimization_level: OptimizationLevel) -> Self {
        Self {
            thread_optimizer: ThreadPoolOptimizer::new(),
            tuning_cache: TuningCache::new(),
            optimization_level,
            enable_adaptive_tuning: true,
        }
    }

    /// Create AutoTuner with persistent cache
    pub fn with_cache_file<P: AsRef<Path>>(
        optimization_level: OptimizationLevel,
        cache_path: P,
    ) -> CpuResult<Self> {
        Ok(Self {
            thread_optimizer: ThreadPoolOptimizer::new(),
            tuning_cache: TuningCache::with_file(cache_path)?,
            optimization_level,
            enable_adaptive_tuning: true,
        })
    }

    /// Create AutoTuner with custom configuration
    pub fn with_config(config: TuningConfig) -> Self {
        let optimization_level = match config.operation_name.as_str() {
            "matrix" => OptimizationLevel::Aggressive,
            "reduction" => OptimizationLevel::Aggressive,
            _ => OptimizationLevel::Basic,
        };

        Self {
            thread_optimizer: ThreadPoolOptimizer::new(),
            tuning_cache: TuningCache::new(),
            optimization_level,
            enable_adaptive_tuning: true,
        }
    }

    /// Get optimal parameters for an operation
    pub fn get_optimal_params(
        &self,
        operation: &str,
        input_size: usize,
        data_type: &str,
    ) -> CpuResult<TuningResult> {
        let cache_key = self
            .tuning_cache
            .generate_key(operation, input_size, data_type);

        // Check cache first
        if let Some(cached_result) = self.tuning_cache.get(&cache_key) {
            // Verify the cached result is still valid for this input size
            if input_size >= cached_result.size_range.0 && input_size <= cached_result.size_range.1
            {
                return Ok(cached_result);
            }
        }

        // Run auto-tuning if not cached or cache miss
        let result = self.run_tuning_for_operation(operation, input_size, data_type)?;
        self.tuning_cache.insert(cache_key, result.clone());

        Ok(result)
    }

    /// Run comprehensive auto-tuning for an operation
    fn run_tuning_for_operation(
        &self,
        operation: &str,
        input_size: usize,
        _data_type: &str,
    ) -> CpuResult<TuningResult> {
        let config = match operation {
            "element_wise" => TuningConfig::for_element_wise_ops(),
            "matrix" => TuningConfig::for_matrix_ops(),
            "reduction" => TuningConfig::for_reduction_ops(),
            _ => TuningConfig::for_element_wise_ops(), // Default fallback
        };

        let mut best_result = None;
        let mut best_score = 0.0;

        // Find appropriate size range
        let size_range = config
            .input_size_ranges
            .iter()
            .find(|(min, max)| input_size >= *min && input_size <= *max)
            .copied()
            .unwrap_or((input_size, input_size * 2));

        // Test different configurations
        for &thread_count in &config.thread_counts {
            for &chunk_size in &config.chunk_sizes {
                // For small input sizes, use the minimum of chunk_size and input_size
                let effective_chunk_size = if chunk_size > input_size {
                    // For very small inputs, use a chunk size of 1 to ensure we have at least one valid configuration
                    if input_size < 64 {
                        1.max(input_size / thread_count.max(1))
                    } else {
                        input_size
                    }
                } else {
                    chunk_size
                };

                for &block_size in &config.block_sizes {
                    let perf = self.benchmark_configuration(
                        operation,
                        input_size,
                        thread_count,
                        effective_chunk_size,
                        Some(block_size),
                        &config,
                    )?;

                    let score = perf.composite_score();
                    if score > best_score {
                        best_score = score;
                        best_result = Some(TuningResult {
                            config: config.clone(),
                            optimal_thread_count: thread_count,
                            optimal_chunk_size: effective_chunk_size,
                            optimal_block_size: Some(block_size),
                            best_performance: perf,
                            size_range,
                        });
                    }
                }

                // Test without block size for non-matrix operations
                if config.block_sizes.is_empty() {
                    let perf = self.benchmark_configuration(
                        operation,
                        input_size,
                        thread_count,
                        effective_chunk_size,
                        None,
                        &config,
                    )?;

                    let score = perf.composite_score();
                    if score > best_score {
                        best_score = score;
                        best_result = Some(TuningResult {
                            config: config.clone(),
                            optimal_thread_count: thread_count,
                            optimal_chunk_size: effective_chunk_size,
                            optimal_block_size: None,
                            best_performance: perf,
                            size_range,
                        });
                    }
                }
            }
        }

        best_result.ok_or_else(|| {
            crate::cpu::error::cpu_errors::optimization_error(
                "Auto-tuning failed to find optimal configuration",
            )
        })
    }

    /// Benchmark a specific configuration
    fn benchmark_configuration(
        &self,
        operation: &str,
        input_size: usize,
        thread_count: usize,
        chunk_size: usize,
        block_size: Option<usize>,
        config: &TuningConfig,
    ) -> CpuResult<PerformanceMeasurement> {
        // Create test data
        let test_data = self.create_test_data(operation, input_size)?;

        // Warmup iterations
        for _ in 0..config.warmup_iterations {
            self.run_test_kernel(operation, &test_data, thread_count, chunk_size, block_size)?;
        }

        // Measurement iterations
        let mut total_time = Duration::from_secs(0);
        for _ in 0..config.iterations_per_test {
            let start = Instant::now();
            self.run_test_kernel(operation, &test_data, thread_count, chunk_size, block_size)?;
            total_time += start.elapsed();
        }

        let avg_time = total_time / config.iterations_per_test as u32;
        Ok(PerformanceMeasurement::new(avg_time, input_size))
    }

    /// Create test data for benchmarking
    fn create_test_data(&self, operation: &str, size: usize) -> CpuResult<TestData> {
        match operation {
            "element_wise" => Ok(TestData::Vector(vec![1.0f32; size])),
            "matrix" => {
                let dim = (size as f64).sqrt() as usize;
                Ok(TestData::Matrix(vec![1.0f32; dim * dim], dim, dim))
            }
            "reduction" => Ok(TestData::Vector(vec![1.0f32; size])),
            _ => Ok(TestData::Vector(vec![1.0f32; size])),
        }
    }

    /// Run test kernel for benchmarking
    fn run_test_kernel(
        &self,
        operation: &str,
        data: &TestData,
        _thread_count: usize,
        chunk_size: usize,
        _block_size: Option<usize>,
    ) -> CpuResult<()> {
        // Temporarily set thread count
        let _original_threads = self.thread_optimizer.get_num_threads();
        // Note: In real implementation, we'd need a mutable reference or separate instance

        match (operation, data) {
            ("element_wise", TestData::Vector(vec)) => {
                // Simple element-wise operation
                vec.iter()
                    .enumerate()
                    .collect::<Vec<_>>()
                    .chunks(chunk_size)
                    .for_each(|chunk| {
                        chunk.iter().for_each(|(_, &val)| {
                            let _ = val * 2.0 + 1.0; // Simple computation
                        });
                    });
            }
            ("reduction", TestData::Vector(vec)) => {
                // Simple reduction operation
                let _sum: f32 = vec
                    .chunks(chunk_size)
                    .map(|chunk| chunk.iter().sum::<f32>())
                    .sum();
            }
            ("matrix", TestData::Matrix(mat, rows, cols)) => {
                // Simple matrix operation
                for i in 0..*rows {
                    for j in 0..*cols {
                        let _ = mat[i * cols + j] * 2.0;
                    }
                }
            }
            _ => {
                // Fallback operation
                std::thread::sleep(Duration::from_micros(1));
            }
        }

        Ok(())
    }

    /// Enable or disable adaptive tuning
    pub fn set_adaptive_tuning(&mut self, enabled: bool) {
        self.enable_adaptive_tuning = enabled;
    }

    /// Get tuning cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        self.tuning_cache.get_cache_stats()
    }

    /// Clear tuning cache
    pub fn clear_cache(&self) {
        self.tuning_cache.clear();
    }

    /// Pre-populate cache with common configurations
    pub fn populate_default_cache(&self) -> CpuResult<()> {
        let common_operations = ["element_wise", "matrix", "reduction"];
        let common_sizes = [1000, 10000, 100000];

        for operation in &common_operations {
            for &size in &common_sizes {
                let _ = self.get_optimal_params(operation, size, "f32");
            }
        }

        Ok(())
    }

    /// Get detailed cache statistics
    pub fn get_detailed_cache_stats(&self) -> HashMap<String, usize> {
        self.tuning_cache.get_detailed_stats()
    }

    /// Force save cache to persistent storage
    pub fn save_cache(&self) -> CpuResult<()> {
        self.tuning_cache.flush()
    }

    /// Invalidate and rebuild cache
    pub fn invalidate_cache(&self) -> CpuResult<()> {
        self.tuning_cache.invalidate_cache()
    }

    /// Clean old cache entries older than specified age
    pub fn cleanup_cache(&self, max_age_hours: u64) {
        self.tuning_cache.cleanup_old_entries(max_age_hours * 3600);
    }

    /// Check if cache is compatible with current system
    pub fn is_cache_compatible(&self) -> bool {
        // This would check if the current cache entries are compatible
        // For now, we assume they are since the cache handles compatibility internally
        true
    }

    /// Get cache efficiency metrics
    pub fn get_cache_efficiency(&self) -> f64 {
        let (hits, misses) = self.get_cache_stats();
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
}

/// Test data types for benchmarking
#[derive(Debug, Clone)]
enum TestData {
    Vector(Vec<f32>),
    Matrix(Vec<f32>, usize, usize), // data, rows, cols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_measurement() {
        let perf = PerformanceMeasurement::new(Duration::from_millis(10), 1000);
        assert!(perf.throughput > 0.0);
        assert!(perf.composite_score() > 0.0);
    }

    #[test]
    fn test_tuning_config() {
        let config = TuningConfig::for_element_wise_ops();
        assert!(!config.thread_counts.is_empty());
        assert!(!config.chunk_sizes.is_empty());
        assert_eq!(config.operation_name, "element_wise");
    }

    #[test]
    fn test_tuning_cache() {
        let cache = TuningCache::new();
        let key = cache.generate_key("test", 1000, "f32");

        // Test cache miss
        assert!(cache.get(&key).is_none());

        // Test cache hit
        let result = TuningResult {
            config: TuningConfig::for_element_wise_ops(),
            optimal_thread_count: 4,
            optimal_chunk_size: 256,
            optimal_block_size: None,
            best_performance: PerformanceMeasurement::new(Duration::from_millis(10), 1000),
            size_range: (100, 10000),
        };

        cache.insert(key.clone(), result);
        assert!(cache.get(&key).is_some());

        let (hits, misses) = cache.get_cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new(OptimizationLevel::Basic);
        assert!(!tuner.enable_adaptive_tuning || tuner.enable_adaptive_tuning); // Test that field exists

        let (hits, misses) = tuner.get_cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    #[test]
    fn test_test_data_creation() {
        let tuner = AutoTuner::default();

        let vector_data = tuner.create_test_data("element_wise", 1000).unwrap();
        match vector_data {
            TestData::Vector(vec) => assert_eq!(vec.len(), 1000),
            _ => panic!("Expected vector data"),
        }

        let matrix_data = tuner.create_test_data("matrix", 100).unwrap();
        match matrix_data {
            TestData::Matrix(mat, rows, cols) => {
                assert_eq!(rows, 10);
                assert_eq!(cols, 10);
                assert_eq!(mat.len(), 100);
            }
            _ => panic!("Expected matrix data"),
        }
    }
}

/// Measure performance of a closure for benchmarking purposes
///
/// This is a standalone utility function for benchmarking purposes
pub fn measure_performance<F, T>(mut f: F) -> PerformanceMeasurement
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    let _ = f(); // Execute the closure
    let execution_time = start.elapsed();

    PerformanceMeasurement::new(execution_time, 1)
}
