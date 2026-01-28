//! Performance regression testing framework
//!
//! This module provides comprehensive performance regression testing with
//! baseline storage, statistical significance testing, and automated reporting.

use super::benchmarking::{benchmark, BenchmarkConfig, BenchmarkResults};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Performance regression testing framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Operation name
    pub operation: String,
    /// Timestamp when baseline was created
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Version information
    pub version: Option<String>,
    /// Baseline performance metrics
    pub baseline_summary: BaselineSummary,
    /// System information
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSummary {
    pub mean_duration: f64,
    pub std_duration: f64,
    pub min_duration: f64,
    pub max_duration: f64,
    pub mean_throughput: f64,
    pub mean_flops: Option<f64>,
    pub mean_memory_bandwidth: Option<f64>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    pub operation: String,
    pub current_performance: BaselineSummary,
    pub baseline_performance: BaselineSummary,
    pub regression_detected: bool,
    pub duration_regression_percent: f64,
    pub throughput_regression_percent: f64,
    pub significance_level: f64,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct RegressionTestConfig {
    /// Acceptable performance degradation threshold (as percentage)
    pub regression_threshold: f64,
    /// Statistical significance level for detecting regressions
    pub significance_level: f64,
    /// Minimum number of samples required for reliable testing
    pub min_samples: usize,
    /// Path to store baseline data
    pub baseline_path: String,
    /// Whether to update baselines automatically
    pub auto_update_baseline: bool,
}

impl Default for RegressionTestConfig {
    fn default() -> Self {
        Self {
            regression_threshold: 5.0, // 5% degradation threshold
            significance_level: 0.05,  // 95% confidence
            min_samples: 10,
            baseline_path: "/tmp/torsh_performance_baselines.json".to_string(),
            auto_update_baseline: false,
        }
    }
}

pub struct PerformanceRegressionTester {
    config: RegressionTestConfig,
    baselines: HashMap<String, PerformanceBaseline>,
}

impl PerformanceRegressionTester {
    /// Create a new regression tester
    pub fn new(config: RegressionTestConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
        }
    }

    /// Load baselines from file
    pub fn load_baselines(&mut self) -> TorshResult<()> {
        match std::fs::read_to_string(&self.config.baseline_path) {
            Ok(content) => {
                self.baselines = serde_json::from_str(&content)
                    .map_err(|e| TorshError::Other(format!("Failed to parse baselines: {}", e)))?;
                Ok(())
            }
            Err(_) => {
                // File doesn't exist, start with empty baselines
                self.baselines = HashMap::new();
                Ok(())
            }
        }
    }

    /// Save baselines to file
    pub fn save_baselines(&self) -> TorshResult<()> {
        let content = serde_json::to_string_pretty(&self.baselines)
            .map_err(|e| TorshError::Other(format!("Failed to serialize baselines: {}", e)))?;

        std::fs::write(&self.config.baseline_path, content)
            .map_err(|e| TorshError::Other(format!("Failed to write baselines file: {}", e)))?;

        Ok(())
    }

    /// Create or update baseline for an operation
    pub fn create_baseline(
        &mut self,
        operation: &str,
        benchmark_results: &BenchmarkResults,
        commit_hash: Option<String>,
        version: Option<String>,
    ) -> TorshResult<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_secs();

        let system_info = SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_count: 1, // Simplified - would use num_cpus crate in real implementation
            total_memory: None, // TODO: Implement memory detection
        };

        let mean_memory_bandwidth = if !benchmark_results.metrics.is_empty() {
            Some(
                benchmark_results
                    .metrics
                    .iter()
                    .filter_map(|m| m.memory_bandwidth)
                    .sum::<f64>()
                    / benchmark_results.metrics.len() as f64,
            )
        } else {
            None
        };

        let baseline_summary = BaselineSummary {
            mean_duration: benchmark_results.summary.mean_duration,
            std_duration: benchmark_results.summary.std_duration,
            min_duration: benchmark_results.summary.min_duration,
            max_duration: benchmark_results.summary.max_duration,
            mean_throughput: benchmark_results.summary.mean_throughput,
            mean_flops: benchmark_results
                .summary
                .total_flops
                .map(|f| f as f64 / benchmark_results.summary.count as f64),
            mean_memory_bandwidth,
            sample_count: benchmark_results.summary.count,
        };

        let baseline = PerformanceBaseline {
            operation: operation.to_string(),
            timestamp,
            commit_hash,
            version,
            baseline_summary,
            system_info,
        };

        self.baselines.insert(operation.to_string(), baseline);
        self.save_baselines()?;

        Ok(())
    }

    /// Test for performance regression
    pub fn test_regression(
        &self,
        operation: &str,
        current_results: &BenchmarkResults,
    ) -> TorshResult<RegressionTestResult> {
        let baseline = self.baselines.get(operation).ok_or_else(|| {
            TorshError::invalid_argument_with_context(
                &format!("No baseline found for operation: {}", operation),
                "test_regression",
            )
        })?;

        if current_results.summary.count < self.config.min_samples {
            return Err(TorshError::invalid_argument_with_context(
                &format!(
                    "Insufficient samples: {} < {}",
                    current_results.summary.count, self.config.min_samples
                ),
                "test_regression",
            ));
        }

        let current_memory_bandwidth = if !current_results.metrics.is_empty() {
            Some(
                current_results
                    .metrics
                    .iter()
                    .filter_map(|m| m.memory_bandwidth)
                    .sum::<f64>()
                    / current_results.metrics.len() as f64,
            )
        } else {
            None
        };

        let current_summary = BaselineSummary {
            mean_duration: current_results.summary.mean_duration,
            std_duration: current_results.summary.std_duration,
            min_duration: current_results.summary.min_duration,
            max_duration: current_results.summary.max_duration,
            mean_throughput: current_results.summary.mean_throughput,
            mean_flops: current_results
                .summary
                .total_flops
                .map(|f| f as f64 / current_results.summary.count as f64),
            mean_memory_bandwidth: current_memory_bandwidth,
            sample_count: current_results.summary.count,
        };

        // Calculate regression percentages
        let duration_regression_percent = ((current_summary.mean_duration
            - baseline.baseline_summary.mean_duration)
            / baseline.baseline_summary.mean_duration)
            * 100.0;

        let throughput_regression_percent = ((baseline.baseline_summary.mean_throughput
            - current_summary.mean_throughput)
            / baseline.baseline_summary.mean_throughput)
            * 100.0;

        // Perform statistical significance test (simplified t-test)
        let is_significant =
            self.is_statistically_significant(&baseline.baseline_summary, &current_summary);

        let regression_detected = is_significant
            && (duration_regression_percent > self.config.regression_threshold
                || throughput_regression_percent > self.config.regression_threshold);

        let details = format!(
            "Duration change: {:.2}%, Throughput change: {:.2}%, Significant: {}",
            duration_regression_percent,
            -throughput_regression_percent, // Negative because higher throughput is better
            is_significant
        );

        Ok(RegressionTestResult {
            operation: operation.to_string(),
            current_performance: current_summary,
            baseline_performance: baseline.baseline_summary.clone(),
            regression_detected,
            duration_regression_percent,
            throughput_regression_percent,
            significance_level: self.config.significance_level,
            details,
        })
    }

    /// Simplified statistical significance test
    fn is_statistically_significant(
        &self,
        baseline: &BaselineSummary,
        current: &BaselineSummary,
    ) -> bool {
        // Simplified two-sample t-test assumption
        let pooled_std = ((baseline.std_duration.powi(2) / baseline.sample_count as f64)
            + (current.std_duration.powi(2) / current.sample_count as f64))
            .sqrt();

        if pooled_std == 0.0 {
            return false;
        }

        let t_statistic = (current.mean_duration - baseline.mean_duration).abs() / pooled_std;

        // Simplified critical value for 95% confidence (approximately 1.96)
        let critical_value = 1.96;

        t_statistic > critical_value
    }

    /// Generate regression test report
    pub fn generate_report(&self, results: &[RegressionTestResult]) -> String {
        let mut report = String::from("Performance Regression Test Report\n");
        report.push_str("=====================================\n\n");

        let total_tests = results.len();
        let regressions = results.iter().filter(|r| r.regression_detected).count();
        let passed = total_tests - regressions;

        report.push_str(&format!(
            "Summary: {} tests, {} passed, {} regressions detected\n\n",
            total_tests, passed, regressions
        ));

        if regressions > 0 {
            report.push_str("REGRESSIONS DETECTED:\n");
            report.push_str("====================\n");

            for result in results.iter().filter(|r| r.regression_detected) {
                report.push_str(&format!("❌ {}\n", result.operation));
                report.push_str(&format!(
                    "   Duration regression: {:.2}%\n",
                    result.duration_regression_percent
                ));
                report.push_str(&format!(
                    "   Throughput regression: {:.2}%\n",
                    result.throughput_regression_percent
                ));
                report.push_str(&format!("   Details: {}\n\n", result.details));
            }
        }

        report.push_str("All Test Results:\n");
        report.push_str("================\n");

        for result in results {
            let status = if result.regression_detected {
                "❌ REGRESSION"
            } else {
                "✅ PASS"
            };
            report.push_str(&format!(
                "{} {}: {}\n",
                status, result.operation, result.details
            ));
        }

        report
    }

    /// List all available baselines
    pub fn list_baselines(&self) -> Vec<&PerformanceBaseline> {
        self.baselines.values().collect()
    }

    /// Remove a baseline
    pub fn remove_baseline(&mut self, operation: &str) -> bool {
        self.baselines.remove(operation).is_some()
    }

    /// Get baseline for an operation
    pub fn get_baseline(&self, operation: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(operation)
    }
}

/// Convenience function to create and run a regression test
pub fn run_performance_regression_test<F>(
    operation_name: &str,
    operation: F,
    inputs: &[&Tensor],
    config: Option<RegressionTestConfig>,
) -> TorshResult<RegressionTestResult>
where
    F: Fn(&[&Tensor]) -> TorshResult<Vec<Tensor>>,
{
    let config = config.unwrap_or_default();
    let mut tester = PerformanceRegressionTester::new(config);
    tester.load_baselines()?;

    let benchmark_config = BenchmarkConfig::default();
    let benchmark_results = benchmark(operation_name, operation, inputs, benchmark_config)?;

    match tester.test_regression(operation_name, &benchmark_results) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Create baseline if it doesn't exist
            tester.create_baseline(operation_name, &benchmark_results, None, None)?;
            Err(TorshError::invalid_argument_with_context(
                "Created new baseline for operation",
                "run_performance_regression_test",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_regression_tester_creation() {
        let config = RegressionTestConfig::default();
        let tester = PerformanceRegressionTester::new(config);
        assert_eq!(tester.baselines.len(), 0);
    }

    #[test]
    fn test_baseline_creation() -> TorshResult<()> {
        let input = randn(&[32, 32])?;
        let inputs = vec![&input];

        let config = BenchmarkConfig {
            warmup_iters: 1,
            bench_iters: 2,
            min_duration: 0.1,
            max_duration: 1.0,
            detailed_metrics: false,
        };

        let results = benchmark(
            "test_baseline_op",
            |inputs| -> TorshResult<Vec<Tensor>> { Ok(vec![inputs[0].clone()]) },
            &inputs,
            config,
        )?;

        let regression_config = RegressionTestConfig {
            baseline_path: "/tmp/test_baselines.json".to_string(),
            ..Default::default()
        };

        let mut tester = PerformanceRegressionTester::new(regression_config);
        tester.create_baseline("test_baseline_op", &results, None, None)?;

        assert!(tester.get_baseline("test_baseline_op").is_some());
        Ok(())
    }
}
