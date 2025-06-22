//! Utility functions for benchmarking

use std::time::{Duration, Instant};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Benchmark timing utilities
pub struct Timer {
    start: Option<Instant>,
    durations: Vec<Duration>,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: None,
            durations: Vec::new(),
        }
    }
    
    /// Start timing
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }
    
    /// Stop timing and record duration
    pub fn stop(&mut self) -> Duration {
        if let Some(start) = self.start.take() {
            let duration = start.elapsed();
            self.durations.push(duration);
            duration
        } else {
            Duration::ZERO
        }
    }
    
    /// Get all recorded durations
    pub fn durations(&self) -> &[Duration] {
        &self.durations
    }
    
    /// Get average duration
    pub fn average(&self) -> Duration {
        if self.durations.is_empty() {
            Duration::ZERO
        } else {
            let total_nanos: u64 = self.durations.iter()
                .map(|d| d.as_nanos() as u64)
                .sum();
            Duration::from_nanos(total_nanos / self.durations.len() as u64)
        }
    }
    
    /// Get minimum duration
    pub fn min(&self) -> Duration {
        self.durations.iter().min().copied().unwrap_or(Duration::ZERO)
    }
    
    /// Get maximum duration
    pub fn max(&self) -> Duration {
        self.durations.iter().max().copied().unwrap_or(Duration::ZERO)
    }
    
    /// Get standard deviation
    pub fn std_dev(&self) -> Duration {
        if self.durations.len() < 2 {
            return Duration::ZERO;
        }
        
        let mean = self.average().as_nanos() as f64;
        let variance: f64 = self.durations.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / (self.durations.len() - 1) as f64;
        
        Duration::from_nanos(variance.sqrt() as u64)
    }
    
    /// Clear all recorded durations
    pub fn clear(&mut self) {
        self.durations.clear();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Data generation utilities for benchmarks
pub struct DataGenerator {
    rng: StdRng,
}

impl DataGenerator {
    /// Create a new data generator with a fixed seed for reproducibility
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
    
    /// Generate random f32 data
    pub fn random_f32(&mut self, size: usize) -> Vec<f32> {
        (0..size).map(|_| self.rng.gen()).collect()
    }
    
    /// Generate random f64 data
    pub fn random_f64(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.rng.gen()).collect()
    }
    
    /// Generate random i32 data
    pub fn random_i32(&mut self, size: usize, min: i32, max: i32) -> Vec<i32> {
        (0..size).map(|_| self.rng.gen_range(min..=max)).collect()
    }
    
    /// Generate random matrix data (row-major)
    pub fn random_matrix_f32(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        self.random_f32(rows * cols)
    }
    
    /// Generate sparse matrix indices
    pub fn sparse_indices(&mut self, size: usize, sparsity: f32) -> Vec<usize> {
        let num_nonzero = ((1.0 - sparsity) * size as f32) as usize;
        let mut indices: Vec<usize> = (0..size).collect();
        
        // Shuffle and take first num_nonzero elements
        for i in 0..num_nonzero {
            let j = self.rng.gen_range(i..size);
            indices.swap(i, j);
        }
        
        indices.truncate(num_nonzero);
        indices.sort();
        indices
    }
    
    /// Generate data with specific distribution
    pub fn normal_f32(&mut self, size: usize, mean: f32, std_dev: f32) -> Vec<f32> {
        use rand_distr::{Normal, Distribution};
        let normal = Normal::new(mean, std_dev).unwrap();
        (0..size).map(|_| normal.sample(&mut self.rng)).collect()
    }
}

impl Default for DataGenerator {
    fn default() -> Self {
        Self::new(42) // Default seed
    }
}

/// System information utilities
pub struct SystemInfo;

impl SystemInfo {
    /// Get CPU information
    pub fn cpu_info() -> CpuInfo {
        CpuInfo {
            model_name: Self::get_cpu_model(),
            core_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            cache_sizes: Self::get_cache_sizes(),
        }
    }
    
    /// Get memory information
    pub fn memory_info() -> MemoryInfo {
        MemoryInfo {
            total_mb: Self::get_total_memory_mb(),
            available_mb: Self::get_available_memory_mb(),
            page_size: Self::get_page_size(),
        }
    }
    
    /// Get platform information
    pub fn platform_info() -> PlatformInfo {
        PlatformInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            family: std::env::consts::FAMILY.to_string(),
        }
    }
    
    fn get_cpu_model() -> String {
        // Platform-specific implementation would go here
        "Unknown CPU".to_string()
    }
    
    fn get_cache_sizes() -> CacheSizes {
        CacheSizes {
            l1_data_kb: 32,   // Typical values
            l1_instruction_kb: 32,
            l2_kb: 256,
            l3_kb: 8192,
        }
    }
    
    fn get_total_memory_mb() -> usize {
        // Would use platform-specific APIs
        8192 // 8GB default
    }
    
    fn get_available_memory_mb() -> usize {
        // Would use platform-specific APIs
        4096 // 4GB default
    }
    
    fn get_page_size() -> usize {
        4096 // 4KB default
    }
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model_name: String,
    pub core_count: usize,
    pub cache_sizes: CacheSizes,
}

#[derive(Debug, Clone)]
pub struct CacheSizes {
    pub l1_data_kb: usize,
    pub l1_instruction_kb: usize,
    pub l2_kb: usize,
    pub l3_kb: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_mb: usize,
    pub available_mb: usize,
    pub page_size: usize,
}

#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub family: String,
}

/// Benchmark validation utilities
pub struct Validator;

impl Validator {
    /// Validate that two f32 arrays are approximately equal
    pub fn arrays_approx_equal_f32(a: &[f32], b: &[f32], epsilon: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
    }
    
    /// Validate that two f64 arrays are approximately equal
    pub fn arrays_approx_equal_f64(a: &[f64], b: &[f64], epsilon: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
    }
    
    /// Check if a result is within expected bounds
    pub fn result_in_bounds<T: PartialOrd>(value: T, min: T, max: T) -> bool {
        value >= min && value <= max
    }
    
    /// Validate matrix multiplication result dimensions
    pub fn validate_matmul_dims(a_rows: usize, a_cols: usize, b_rows: usize, b_cols: usize) -> bool {
        a_cols == b_rows
    }
    
    /// Check for NaN or infinite values in f32 array
    pub fn check_finite_f32(arr: &[f32]) -> bool {
        arr.iter().all(|x| x.is_finite())
    }
    
    /// Check for NaN or infinite values in f64 array
    pub fn check_finite_f64(arr: &[f64]) -> bool {
        arr.iter().all(|x| x.is_finite())
    }
}

/// Formatting utilities for benchmark output
pub struct Formatter;

impl Formatter {
    /// Format duration in human-readable form
    pub fn format_duration(duration: Duration) -> String {
        let nanos = duration.as_nanos();
        
        if nanos < 1_000 {
            format!("{}ns", nanos)
        } else if nanos < 1_000_000 {
            format!("{:.2}μs", nanos as f64 / 1_000.0)
        } else if nanos < 1_000_000_000 {
            format!("{:.2}ms", nanos as f64 / 1_000_000.0)
        } else {
            format!("{:.2}s", nanos as f64 / 1_000_000_000.0)
        }
    }
    
    /// Format throughput in operations per second
    pub fn format_throughput(ops_per_sec: f64) -> String {
        if ops_per_sec < 1_000.0 {
            format!("{:.2} ops/s", ops_per_sec)
        } else if ops_per_sec < 1_000_000.0 {
            format!("{:.2}K ops/s", ops_per_sec / 1_000.0)
        } else if ops_per_sec < 1_000_000_000.0 {
            format!("{:.2}M ops/s", ops_per_sec / 1_000_000.0)
        } else {
            format!("{:.2}G ops/s", ops_per_sec / 1_000_000_000.0)
        }
    }
    
    /// Format memory size in human-readable form
    pub fn format_memory(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
    
    /// Format FLOPS in human-readable form
    pub fn format_flops(flops: f64) -> String {
        if flops < 1_000.0 {
            format!("{:.2} FLOPS", flops)
        } else if flops < 1_000_000.0 {
            format!("{:.2} KFLOPS", flops / 1_000.0)
        } else if flops < 1_000_000_000.0 {
            format!("{:.2} MFLOPS", flops / 1_000_000.0)
        } else if flops < 1_000_000_000_000.0 {
            format!("{:.2} GFLOPS", flops / 1_000_000_000.0)
        } else {
            format!("{:.2} TFLOPS", flops / 1_000_000_000_000.0)
        }
    }
    
    /// Format percentage
    pub fn format_percentage(value: f64) -> String {
        format!("{:.1}%", value * 100.0)
    }
}

/// Configuration management for benchmarks
pub struct BenchConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub statistical_significance: f64,
    pub output_format: OutputFormat,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            min_execution_time: Duration::from_millis(10),
            max_execution_time: Duration::from_secs(60),
            statistical_significance: 0.05, // 5%
            output_format: OutputFormat::Table,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OutputFormat {
    Table,
    Json,
    Csv,
    Markdown,
}

/// Environment setup utilities
pub struct Environment;

impl Environment {
    /// Set up optimal environment for benchmarking
    pub fn setup_for_benchmarking() {
        // Set high priority for current process
        Self::set_high_priority();
        
        // Disable frequency scaling if possible
        Self::disable_frequency_scaling();
        
        // Set thread affinity to specific cores
        Self::set_cpu_affinity();
    }
    
    /// Restore normal environment after benchmarking
    pub fn restore_environment() {
        Self::set_normal_priority();
        Self::enable_frequency_scaling();
        Self::clear_cpu_affinity();
    }
    
    fn set_high_priority() {
        // Platform-specific implementation would go here
    }
    
    fn set_normal_priority() {
        // Platform-specific implementation would go here
    }
    
    fn disable_frequency_scaling() {
        // Platform-specific implementation would go here
    }
    
    fn enable_frequency_scaling() {
        // Platform-specific implementation would go here
    }
    
    fn set_cpu_affinity() {
        // Platform-specific implementation would go here
    }
    
    fn clear_cpu_affinity() {
        // Platform-specific implementation would go here
    }
}