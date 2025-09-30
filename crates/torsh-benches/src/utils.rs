//! Utility functions for benchmarking

// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use std::time::{Duration, Instant};

/// Comprehensive timing statistics
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub count: usize,
    pub mean: Duration,
    pub median: Duration,
    pub std_dev: Duration,
    pub std_error: Duration,
    pub min: Duration,
    pub max: Duration,
    pub q25: Duration,
    pub q75: Duration,
    pub confidence_interval_95: (Duration, Duration),
    pub coefficient_of_variation: f64,
    pub outlier_count: usize,
}

impl TimingStats {
    /// Get the interquartile range (Q3 - Q1)
    pub fn interquartile_range(&self) -> Duration {
        Duration::from_nanos((self.q75.as_nanos() - self.q25.as_nanos()) as u64)
    }

    /// Check if timing is stable (CV < 10%)
    pub fn is_stable(&self) -> bool {
        self.coefficient_of_variation < 0.1
    }

    /// Get a summary string of the statistics
    pub fn summary(&self) -> String {
        format!(
            "Timing Stats: mean={:.2}ms ±{:.2}ms, median={:.2}ms, range=[{:.2}ms, {:.2}ms], CV={:.1}%, outliers={}",
            self.mean.as_secs_f64() * 1000.0,
            self.std_dev.as_secs_f64() * 1000.0,
            self.median.as_secs_f64() * 1000.0,
            self.min.as_secs_f64() * 1000.0,
            self.max.as_secs_f64() * 1000.0,
            self.coefficient_of_variation * 100.0,
            self.outlier_count
        )
    }
}

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
            let total_nanos: u64 = self.durations.iter().map(|d| d.as_nanos() as u64).sum();
            Duration::from_nanos(total_nanos / self.durations.len() as u64)
        }
    }

    /// Get minimum duration
    pub fn min(&self) -> Duration {
        self.durations
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO)
    }

    /// Get maximum duration
    pub fn max(&self) -> Duration {
        self.durations
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO)
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> Duration {
        if self.durations.len() < 2 {
            return Duration::ZERO;
        }

        let mean = self.average().as_nanos() as f64;
        let variance: f64 = self
            .durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.durations.len() - 1) as f64;

        Duration::from_nanos(variance.sqrt() as u64)
    }

    /// Get coefficient of variation (standard deviation / mean)
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.durations.is_empty() {
            return 0.0;
        }

        let mean = self.average().as_nanos() as f64;
        if mean == 0.0 {
            return 0.0;
        }

        let std_dev = self.std_dev().as_nanos() as f64;
        std_dev / mean
    }

    /// Get percentile value (p should be between 0.0 and 1.0)
    pub fn percentile(&self, p: f64) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = self.durations.clone();
        sorted.sort();

        let index = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Get median duration
    pub fn median(&self) -> Duration {
        self.percentile(0.5)
    }

    /// Get 95% confidence interval for the mean
    pub fn confidence_interval_95(&self) -> (Duration, Duration) {
        if self.durations.len() < 2 {
            let avg = self.average();
            return (avg, avg);
        }

        let mean = self.average().as_nanos() as f64;
        let std_err = self.standard_error().as_nanos() as f64;

        // t-value for 95% confidence with n-1 degrees of freedom (approximation for large n)
        let t_value = if self.durations.len() >= 30 {
            1.96 // Normal approximation
        } else {
            // Simplified t-table lookup for small samples
            match self.durations.len() {
                2 => 12.706,
                3 => 4.303,
                4 => 3.182,
                5 => 2.776,
                6 => 2.571,
                7 => 2.447,
                8 => 2.365,
                9 => 2.306,
                10 => 2.262,
                _ => 2.09, // Approximate for 11-29 samples
            }
        };

        let margin = t_value * std_err;
        (
            Duration::from_nanos((mean - margin).max(0.0) as u64),
            Duration::from_nanos((mean + margin) as u64),
        )
    }

    /// Get standard error of the mean
    pub fn standard_error(&self) -> Duration {
        if self.durations.len() < 2 {
            return Duration::ZERO;
        }

        let std_dev = self.std_dev().as_nanos() as f64;
        let n = self.durations.len() as f64;
        Duration::from_nanos((std_dev / n.sqrt()) as u64)
    }

    /// Detect outliers using the IQR method
    pub fn detect_outliers(&self) -> Vec<(usize, Duration)> {
        if self.durations.len() < 4 {
            return Vec::new();
        }

        let q1 = self.percentile(0.25);
        let q3 = self.percentile(0.75);
        let iqr = Duration::from_nanos((q3.as_nanos() - q1.as_nanos()) as u64);

        let lower_bound = Duration::from_nanos(
            (q1.as_nanos()
                .saturating_sub((1.5 * iqr.as_nanos() as f64) as u128)) as u64,
        );
        let upper_bound =
            Duration::from_nanos((q3.as_nanos() + (1.5 * iqr.as_nanos() as f64) as u128) as u64);

        self.durations
            .iter()
            .enumerate()
            .filter(|(_, &duration)| duration < lower_bound || duration > upper_bound)
            .map(|(idx, &duration)| (idx, duration))
            .collect()
    }

    /// Get comprehensive timing statistics
    pub fn comprehensive_stats(&self) -> TimingStats {
        TimingStats {
            count: self.durations.len(),
            mean: self.average(),
            median: self.median(),
            std_dev: self.std_dev(),
            std_error: self.standard_error(),
            min: self.min(),
            max: self.max(),
            q25: self.percentile(0.25),
            q75: self.percentile(0.75),
            confidence_interval_95: self.confidence_interval_95(),
            coefficient_of_variation: self.coefficient_of_variation(),
            outlier_count: self.detect_outliers().len(),
        }
    }

    /// Clear all recorded durations
    pub fn clear(&mut self) {
        self.durations.clear();
    }

    /// Adaptive timing: automatically determine optimal number of iterations
    /// based on timing stability
    pub fn adaptive_benchmark<F>(&mut self, mut operation: F, max_iterations: usize) -> Duration
    where
        F: FnMut(),
    {
        const MIN_ITERATIONS: usize = 5;
        const STABILITY_THRESHOLD: f64 = 0.05; // 5% CV
        const MIN_RUNTIME_MS: f64 = 100.0; // Minimum total runtime

        self.clear();

        // Initial warmup
        for _ in 0..3 {
            operation();
        }

        // Adaptive measurement
        let start_time = Instant::now();
        for i in 0..max_iterations {
            self.start();
            operation();
            self.stop();

            // Check stability after minimum iterations
            if i >= MIN_ITERATIONS {
                let stats = self.comprehensive_stats();
                let total_runtime = start_time.elapsed().as_secs_f64() * 1000.0;

                // Stop if we have stable results and minimum runtime
                if stats.coefficient_of_variation < STABILITY_THRESHOLD
                    && total_runtime >= MIN_RUNTIME_MS
                {
                    break;
                }
            }
        }

        self.average()
    }

    /// High-precision timing using multiple techniques
    pub fn precision_timing<F>(&mut self, operation: F, iterations: usize) -> TimingStats
    where
        F: Fn(),
    {
        self.clear();

        // Warmup phase
        for _ in 0..std::cmp::min(iterations / 10, 5) {
            operation();
        }

        // Memory barrier to ensure warmup completion
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        // Precise measurement phase
        for _ in 0..iterations {
            // Use CPU cycle counting for higher precision on x86
            #[cfg(target_arch = "x86_64")]
            {
                let start_cycles = unsafe { core::arch::x86_64::_rdtsc() };
                operation();
                let end_cycles = unsafe { core::arch::x86_64::_rdtsc() };

                // Convert cycles to duration (approximate)
                let cycle_duration =
                    Duration::from_nanos(((end_cycles - start_cycles) as f64 / 2.5e9 * 1e9) as u64);
                self.durations.push(cycle_duration);
            }

            // Fallback to Instant for other architectures
            #[cfg(not(target_arch = "x86_64"))]
            {
                let start = Instant::now();
                operation();
                let duration = start.elapsed();
                self.durations.push(duration);
            }
        }

        self.comprehensive_stats()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory monitoring utilities for benchmarks
#[derive(Debug, Clone)]
pub struct MemoryMonitor {
    initial_memory: usize,
    peak_memory: usize,
    current_memory: usize,
    allocations: Vec<(Instant, usize)>,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        let initial = Self::get_current_memory_usage();
        Self {
            initial_memory: initial,
            peak_memory: initial,
            current_memory: initial,
            allocations: Vec::new(),
        }
    }

    /// Start monitoring memory usage
    pub fn start(&mut self) {
        self.initial_memory = Self::get_current_memory_usage();
        self.peak_memory = self.initial_memory;
        self.current_memory = self.initial_memory;
        self.allocations.clear();
    }

    /// Record current memory usage
    pub fn record(&mut self) {
        let current = Self::get_current_memory_usage();
        self.current_memory = current;
        self.peak_memory = self.peak_memory.max(current);
        self.allocations.push((Instant::now(), current));
    }

    /// Get memory usage delta from start
    pub fn memory_delta(&self) -> isize {
        self.current_memory as isize - self.initial_memory as isize
    }

    /// Get peak memory usage during monitoring
    pub fn peak_memory_delta(&self) -> usize {
        self.peak_memory.saturating_sub(self.initial_memory)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            initial: self.initial_memory,
            peak: self.peak_memory,
            current: self.current_memory,
            delta: self.memory_delta(),
            peak_delta: self.peak_memory_delta(),
            allocation_count: self.allocations.len(),
        }
    }

    /// Get current memory usage (simplified implementation)
    fn get_current_memory_usage() -> usize {
        // This is a simplified implementation. In production, you might use:
        // - procfs on Linux
        // - task_info on macOS
        // - GetProcessMemoryInfo on Windows
        // For now, we'll use a mock implementation

        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback: estimate based on heap allocations (mock)
        std::mem::size_of::<usize>() * 1024 * 1024 // 1MB mock
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub initial: usize,
    pub peak: usize,
    pub current: usize,
    pub delta: isize,
    pub peak_delta: usize,
    pub allocation_count: usize,
}

impl MemoryStats {
    /// Format memory size in human-readable format
    pub fn format_size(bytes: usize) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Memory: initial={}, peak={} (+{}), current={} ({}{})",
            Self::format_size(self.initial),
            Self::format_size(self.peak),
            Self::format_size(self.peak_delta),
            Self::format_size(self.current),
            if self.delta >= 0 { "+" } else { "" },
            Self::format_size(self.delta.abs() as usize)
        )
    }
}

/// Parallel benchmark execution utilities
#[derive(Debug)]
pub struct ParallelBenchRunner {
    thread_count: usize,
    max_concurrent: usize,
}

impl ParallelBenchRunner {
    pub fn new() -> Self {
        Self {
            thread_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            max_concurrent: 4,
        }
    }

    pub fn with_threads(mut self, count: usize) -> Self {
        self.thread_count = count;
        self
    }

    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Run multiple benchmarks in parallel
    pub fn run_parallel_benchmarks<F, R>(&self, benchmarks: Vec<F>) -> Vec<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        use std::sync::mpsc;
        use std::thread;

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        // Create chunks of benchmarks for parallel execution
        let chunk_size = (benchmarks.len() + self.thread_count - 1) / self.thread_count;
        let mut benchmark_iter = benchmarks.into_iter();
        let mut chunks = Vec::new();

        loop {
            let chunk: Vec<_> = benchmark_iter.by_ref().take(chunk_size).collect();
            if chunk.is_empty() {
                break;
            }
            chunks.push(chunk);
        }

        // Spawn worker threads
        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let tx = tx.clone();
            let handle = thread::spawn(move || {
                let mut results = Vec::new();
                for (bench_idx, benchmark) in chunk.into_iter().enumerate() {
                    let result = benchmark();
                    results.push((chunk_idx, bench_idx, result));
                }
                for result in results {
                    tx.send(result).unwrap();
                }
            });
            handles.push(handle);
        }

        // Drop the original sender
        drop(tx);

        // Collect results
        let mut results: Vec<Option<R>> =
            (0..self.thread_count * chunk_size).map(|_| None).collect();
        for (chunk_idx, bench_idx, result) in rx {
            let global_idx = chunk_idx * chunk_size + bench_idx;
            if global_idx < results.len() {
                results[global_idx] = Some(result);
            }
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Return only the valid results
        results.into_iter().filter_map(|r| r).collect()
    }
}

impl Default for ParallelBenchRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced data generation utilities
pub struct DataGenerator;

impl DataGenerator {
    /// Generate random tensor data with specified characteristics
    pub fn random_tensor_data(
        size: usize,
        distribution: Distribution,
        seed: Option<u64>,
    ) -> Vec<f32> {
        let mut rng = if let Some(seed) = seed {
            Random::seed(seed)
        } else {
            Random::seed(42) // Use consistent type with default seed
        };

        match distribution {
            Distribution::Normal { mean, std } => {
                (0..size)
                    .map(|_| {
                        // Box-Muller transform for normal distribution
                        let u1: f32 = rng.random();
                        let u2: f32 = rng.random();
                        let z =
                            (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                        mean + std * z
                    })
                    .collect()
            }
            Distribution::Uniform { min, max } => (0..size)
                .map(|_| {
                    let r: f32 = rng.random();
                    min + r * (max - min)
                })
                .collect(),
            Distribution::Exponential { lambda } => (0..size)
                .map(|_| {
                    let u: f32 = rng.random();
                    -lambda.ln() * (1.0f32 - u).ln()
                })
                .collect(),
        }
    }

    /// Generate sparse data with specified sparsity
    pub fn sparse_data(size: usize, sparsity: f32, seed: Option<u64>) -> Vec<f32> {
        let mut rng = if let Some(seed) = seed {
            Random::seed(seed)
        } else {
            Random::seed(42) // Use consistent type with default seed
        };

        (0..size)
            .map(|_| {
                if rng.random::<f32>() < sparsity {
                    0.0
                } else {
                    rng.random::<f32>()
                }
            })
            .collect()
    }

    /// Generate correlated data for testing
    pub fn correlated_data(
        size: usize,
        correlation: f32,
        seed: Option<u64>,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut rng = if let Some(seed) = seed {
            Random::seed(seed)
        } else {
            Random::seed(42) // Use consistent type with default seed
        };

        let x: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let y: Vec<f32> = x
            .iter()
            .map(|&xi| {
                let noise = rng.random::<f32>();
                correlation * xi + (1.0 - correlation) * noise
            })
            .collect();

        (x, y)
    }
}

/// Distribution types for data generation
#[derive(Debug, Clone)]
pub enum Distribution {
    Normal { mean: f32, std: f32 },
    Uniform { min: f32, max: f32 },
    Exponential { lambda: f32 },
}

/// Environment detection and optimization
pub struct Environment;

impl Environment {
    /// Detect optimal number of threads for benchmarking
    pub fn optimal_thread_count() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(8) // Cap at 8 for benchmarking stability
    }

    /// Check if running in a suitable benchmarking environment
    pub fn is_suitable_for_benchmarking() -> bool {
        // Check CPU frequency scaling
        let cpu_stable = Self::is_cpu_frequency_stable();

        // Check system load
        let low_load = Self::is_system_load_low();

        // Check if running under debugger
        let not_debugging = !Self::is_under_debugger();

        cpu_stable && low_load && not_debugging
    }

    fn is_cpu_frequency_stable() -> bool {
        // Simplified check - in production, would check governor settings
        true
    }

    fn is_system_load_low() -> bool {
        // Simplified check - in production, would check load average
        true
    }

    fn is_under_debugger() -> bool {
        // Simplified check - in production, would check for debugger attachment
        false
    }

    /// Get environment information for benchmarking reports
    pub fn get_info() -> EnvironmentInfo {
        EnvironmentInfo {
            cpu_count: Self::optimal_thread_count(),
            is_suitable: Self::is_suitable_for_benchmarking(),
            architecture: std::env::consts::ARCH.to_string(),
            os: std::env::consts::OS.to_string(),
            debug_mode: cfg!(debug_assertions),
        }
    }
}

/// Environment information
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub cpu_count: usize,
    pub is_suitable: bool,
    pub architecture: String,
    pub os: String,
    pub debug_mode: bool,
}

impl EnvironmentInfo {
    pub fn summary(&self) -> String {
        format!(
            "Environment: {}/{}, {} cores, debug={}, suitable={}",
            self.os, self.architecture, self.cpu_count, self.debug_mode, self.is_suitable
        )
    }
}

/// Validator for benchmark results
pub struct Validator;

impl Validator {
    /// Validate timing results for consistency
    pub fn validate_timing(stats: &TimingStats) -> ValidationResult {
        let mut issues = Vec::new();

        // Check for unrealistic timings
        if stats.mean < Duration::from_nanos(1) {
            issues.push("Mean timing is unrealistically low".to_string());
        }

        // Check for high variance
        if stats.coefficient_of_variation > 0.5 {
            issues.push(format!(
                "High timing variance (CV={:.1}%)",
                stats.coefficient_of_variation * 100.0
            ));
        }

        // Check for outliers
        if stats.outlier_count > stats.count / 4 {
            issues.push(format!(
                "Too many outliers ({}/{})",
                stats.outlier_count, stats.count
            ));
        }

        if issues.is_empty() {
            ValidationResult::Valid
        } else {
            ValidationResult::Invalid(issues)
        }
    }

    /// Validate memory usage patterns
    pub fn validate_memory(stats: &MemoryStats) -> ValidationResult {
        let mut issues = Vec::new();

        // Check for memory leaks
        if stats.delta > 1024 * 1024 {
            // > 1MB increase
            issues.push("Potential memory leak detected".to_string());
        }

        // Check for excessive memory usage
        if stats.peak_delta > 1024 * 1024 * 1024 {
            // > 1GB peak increase
            issues.push("Excessive memory usage".to_string());
        }

        if issues.is_empty() {
            ValidationResult::Valid
        } else {
            ValidationResult::Invalid(issues)
        }
    }
}

/// Validation results
#[derive(Debug, Clone)]
pub enum ValidationResult {
    Valid,
    Invalid(Vec<String>),
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    pub fn issues(&self) -> &[String] {
        match self {
            ValidationResult::Valid => &[],
            ValidationResult::Invalid(issues) => issues,
        }
    }
}

/// Formatting utilities for benchmark output
pub struct Formatter;

impl Formatter {
    /// Format duration in appropriate units
    pub fn format_duration(duration: Duration) -> String {
        let nanos = duration.as_nanos();
        if nanos < 1_000 {
            format!("{}ns", nanos)
        } else if nanos < 1_000_000 {
            format!("{:.1}μs", nanos as f64 / 1_000.0)
        } else if nanos < 1_000_000_000 {
            format!("{:.1}ms", nanos as f64 / 1_000_000.0)
        } else {
            format!("{:.1}s", nanos as f64 / 1_000_000_000.0)
        }
    }

    /// Format throughput in appropriate units
    pub fn format_throughput(ops_per_sec: f64) -> String {
        if ops_per_sec < 1_000.0 {
            format!("{:.1} ops/s", ops_per_sec)
        } else if ops_per_sec < 1_000_000.0 {
            format!("{:.1} Kops/s", ops_per_sec / 1_000.0)
        } else if ops_per_sec < 1_000_000_000.0 {
            format!("{:.1} Mops/s", ops_per_sec / 1_000_000.0)
        } else {
            format!("{:.1} Gops/s", ops_per_sec / 1_000_000_000.0)
        }
    }

    /// Format bytes in appropriate units
    pub fn format_bytes(bytes: usize) -> String {
        MemoryStats::format_size(bytes)
    }
}

/// Enhanced benchmark suite integration utilities
pub struct EnhancedBenchSuite {
    pub timer: Timer,
    pub memory_monitor: MemoryMonitor,
    pub parallel_runner: ParallelBenchRunner,
    pub environment: EnvironmentInfo,
}

impl EnhancedBenchSuite {
    pub fn new() -> Self {
        Self {
            timer: Timer::new(),
            memory_monitor: MemoryMonitor::new(),
            parallel_runner: ParallelBenchRunner::new(),
            environment: Environment::get_info(),
        }
    }

    /// Run a comprehensive benchmark with all monitoring enabled
    pub fn comprehensive_benchmark<F, R>(
        &mut self,
        name: &str,
        operation: F,
    ) -> EnhancedBenchResult<R>
    where
        F: Fn() -> R + Clone,
        R: Clone,
    {
        // Start monitoring
        self.memory_monitor.start();

        // Run adaptive timing benchmark
        let timing_stats = self.timer.precision_timing(
            || {
                let _ = operation();
            },
            100,
        );

        // Get memory statistics
        let memory_stats = self.memory_monitor.memory_stats();

        // Validate results
        let timing_validation = Validator::validate_timing(&timing_stats);
        let memory_validation = Validator::validate_memory(&memory_stats);

        // Run the operation one more time to get the actual result
        let result = operation();

        EnhancedBenchResult {
            name: name.to_string(),
            result,
            timing_stats,
            memory_stats,
            timing_validation,
            memory_validation,
            environment: self.environment.clone(),
        }
    }

    /// Run multiple benchmarks in parallel with comprehensive monitoring
    pub fn parallel_comprehensive_benchmarks<F, R>(
        &mut self,
        benchmarks: Vec<(String, F)>,
    ) -> Vec<EnhancedBenchResult<R>>
    where
        F: Fn() -> R + Send + Clone + 'static,
        R: Send + Clone + 'static,
    {
        // Convert to closures that return EnhancedBenchResult
        let benchmark_closures: Vec<Box<dyn FnOnce() -> EnhancedBenchResult<R> + Send>> =
            benchmarks
                .into_iter()
                .map(|(name, operation)| {
                    let mut suite = EnhancedBenchSuite::new();
                    Box::new(move || suite.comprehensive_benchmark(&name, operation))
                        as Box<dyn FnOnce() -> EnhancedBenchResult<R> + Send>
                })
                .collect();

        self.parallel_runner
            .run_parallel_benchmarks(benchmark_closures)
    }

    /// Generate comprehensive performance report
    pub fn generate_comprehensive_report<R>(&self, results: &[EnhancedBenchResult<R>]) -> String {
        let mut report = String::new();

        report.push_str("# Enhanced ToRSh Benchmark Report\n\n");
        report.push_str(&format!(
            "Generated: {}\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!("{}\n\n", self.environment.summary()));

        // Summary statistics
        let valid_timing_results: Vec<_> = results
            .iter()
            .filter(|r| r.timing_validation.is_valid())
            .collect();

        if !valid_timing_results.is_empty() {
            let avg_throughput: f64 = valid_timing_results
                .iter()
                .map(|r| 1.0 / r.timing_stats.mean.as_secs_f64())
                .sum::<f64>()
                / valid_timing_results.len() as f64;

            report.push_str("## Summary\n\n");
            report.push_str(&format!("- Total benchmarks: {}\n", results.len()));
            report.push_str(&format!(
                "- Valid timing results: {}\n",
                valid_timing_results.len()
            ));
            report.push_str(&format!(
                "- Average throughput: {}\n",
                Formatter::format_throughput(avg_throughput)
            ));
            report.push_str("\n");
        }

        // Detailed results
        report.push_str("## Detailed Results\n\n");
        report.push_str("| Benchmark | Mean Time | Std Dev | CV% | Memory Delta | Validation |\n");
        report.push_str("|-----------|-----------|---------|-----|--------------|------------|\n");

        for result in results {
            let validation_status =
                if result.timing_validation.is_valid() && result.memory_validation.is_valid() {
                    "✅ Valid"
                } else {
                    "❌ Issues"
                };

            report.push_str(&format!(
                "| {} | {} | {} | {:.1}% | {} | {} |\n",
                result.name,
                Formatter::format_duration(result.timing_stats.mean),
                Formatter::format_duration(result.timing_stats.std_dev),
                result.timing_stats.coefficient_of_variation * 100.0,
                MemoryStats::format_size(result.memory_stats.delta.abs() as usize),
                validation_status
            ));
        }

        report.push_str("\n## Validation Issues\n\n");
        for result in results {
            if !result.timing_validation.is_valid() || !result.memory_validation.is_valid() {
                report.push_str(&format!("### {}\n", result.name));
                for issue in result.timing_validation.issues() {
                    report.push_str(&format!("- Timing: {}\n", issue));
                }
                for issue in result.memory_validation.issues() {
                    report.push_str(&format!("- Memory: {}\n", issue));
                }
                report.push_str("\n");
            }
        }

        report
    }
}

impl Default for EnhancedBenchSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced benchmark result with comprehensive monitoring
#[derive(Debug, Clone)]
pub struct EnhancedBenchResult<R> {
    pub name: String,
    pub result: R,
    pub timing_stats: TimingStats,
    pub memory_stats: MemoryStats,
    pub timing_validation: ValidationResult,
    pub memory_validation: ValidationResult,
    pub environment: EnvironmentInfo,
}

impl<R> EnhancedBenchResult<R> {
    /// Check if the benchmark result is completely valid
    pub fn is_valid(&self) -> bool {
        self.timing_validation.is_valid() && self.memory_validation.is_valid()
    }

    /// Get all validation issues
    pub fn all_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        issues.extend(
            self.timing_validation
                .issues()
                .iter()
                .map(|s| format!("Timing: {}", s)),
        );
        issues.extend(
            self.memory_validation
                .issues()
                .iter()
                .map(|s| format!("Memory: {}", s)),
        );
        issues
    }

    /// Get a summary of the benchmark result
    pub fn summary(&self) -> String {
        format!(
            "{}: {} ({})",
            self.name,
            self.timing_stats.summary(),
            if self.is_valid() { "Valid" } else { "Issues" }
        )
    }
}
