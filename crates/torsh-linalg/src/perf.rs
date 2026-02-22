//! Performance utilities and profiling helpers for linear algebra operations
//!
//! This module provides utilities for measuring and optimizing performance
//! of linear algebra operations.

use std::time::Instant;
use torsh_core::Result as TorshResult;

/// Simple performance timer for measuring operation duration
pub struct PerfTimer {
    start: Instant,
    label: String,
}

impl PerfTimer {
    /// Create a new performance timer with a label
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            label: label.into(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000_000.0
    }

    /// Stop the timer and return elapsed time in milliseconds
    pub fn stop(&self) -> f64 {
        self.elapsed_ms()
    }

    /// Stop the timer and print the result
    pub fn stop_and_print(&self) {
        println!("{}: {:.3} ms", self.label, self.elapsed_ms());
    }
}

/// Performance statistics for a series of measurements
#[derive(Debug, Clone)]
pub struct PerfStats {
    /// Number of measurements
    pub count: usize,
    /// Minimum time in milliseconds
    pub min_ms: f64,
    /// Maximum time in milliseconds
    pub max_ms: f64,
    /// Mean time in milliseconds
    pub mean_ms: f64,
    /// Median time in milliseconds
    pub median_ms: f64,
    /// Standard deviation in milliseconds
    pub std_dev_ms: f64,
}

impl PerfStats {
    /// Compute statistics from a series of timing measurements (in milliseconds)
    pub fn from_measurements(mut measurements: Vec<f64>) -> Self {
        if measurements.is_empty() {
            return Self {
                count: 0,
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                median_ms: 0.0,
                std_dev_ms: 0.0,
            };
        }

        measurements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = measurements.len();
        let min_ms = measurements[0];
        let max_ms = measurements[count - 1];
        let sum: f64 = measurements.iter().sum();
        let mean_ms = sum / count as f64;

        let median_ms = if count % 2 == 0 {
            (measurements[count / 2 - 1] + measurements[count / 2]) / 2.0
        } else {
            measurements[count / 2]
        };

        let variance: f64 = measurements
            .iter()
            .map(|&x| {
                let diff = x - mean_ms;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;

        let std_dev_ms = variance.sqrt();

        Self {
            count,
            min_ms,
            max_ms,
            mean_ms,
            median_ms,
            std_dev_ms,
        }
    }

    /// Print formatted statistics
    pub fn print(&self, label: &str) {
        println!("Performance Stats for {}", label);
        println!("  Count:      {}", self.count);
        println!("  Min:        {:.3} ms", self.min_ms);
        println!("  Max:        {:.3} ms", self.max_ms);
        println!("  Mean:       {:.3} ms", self.mean_ms);
        println!("  Median:     {:.3} ms", self.median_ms);
        println!("  Std Dev:    {:.3} ms", self.std_dev_ms);
    }
}

/// Benchmark a function multiple times and return statistics
///
/// # Arguments
///
/// * `label` - Description of the benchmark
/// * `iterations` - Number of times to run the function
/// * `warmup` - Number of warmup iterations (not counted in stats)
/// * `f` - Function to benchmark
///
/// # Returns
///
/// Performance statistics
pub fn benchmark<F, R>(
    label: impl Into<String>,
    iterations: usize,
    warmup: usize,
    mut f: F,
) -> TorshResult<PerfStats>
where
    F: FnMut() -> TorshResult<R>,
{
    let label = label.into();

    // Warmup iterations
    for _ in 0..warmup {
        f()?;
    }

    // Timed iterations
    let mut measurements = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        measurements.push(elapsed);
    }

    let stats = PerfStats::from_measurements(measurements);
    stats.print(&label);

    Ok(stats)
}

/// Macro for timing a block of code
///
/// # Example
///
/// ```ignore
/// use torsh_linalg::perf::time_block;
///
/// time_block!("Matrix multiplication", {
///     let result = matrix_a.matmul(&matrix_b)?;
/// });
/// ```
#[macro_export]
macro_rules! time_block {
    ($label:expr, $block:block) => {{
        let timer = $crate::perf::PerfTimer::new($label);
        let result = $block;
        timer.stop_and_print();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 9.0); // Allow for some variance
    }

    #[test]
    fn test_perf_stats() {
        let measurements = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = PerfStats::from_measurements(measurements);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, 5.0);
        assert_eq!(stats.mean_ms, 3.0);
        assert_eq!(stats.median_ms, 3.0);
    }

    #[test]
    fn test_benchmark() -> TorshResult<()> {
        let stats = benchmark("simple computation", 5, 2, || -> TorshResult<i32> {
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(42)
        })?;

        assert_eq!(stats.count, 5);
        assert!(stats.mean_ms >= 0.9); // At least 1ms per iteration

        Ok(())
    }
}
