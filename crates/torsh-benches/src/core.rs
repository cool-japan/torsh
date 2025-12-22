//! Core infrastructure for ToRSh performance comparisons
//!
//! This module provides the foundational components for benchmarking ToRSh against
//! other tensor libraries, including comparison orchestration, result analysis,
//! and performance regression detection.

use crate::BenchResult;
use std::collections::HashMap;

/// Main comparison benchmark orchestrator
///
/// Manages and aggregates comparison results across different tensor libraries,
/// providing unified reporting and analysis capabilities.
#[derive(Debug)]
pub struct ComparisonRunner {
    results: Vec<ComparisonResult>,
}

impl ComparisonRunner {
    /// Create a new comparison runner
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add comparison results
    pub fn add_result(&mut self, result: ComparisonResult) {
        self.results.push(result);
    }

    /// Get all comparison results
    pub fn results(&self) -> &[ComparisonResult] {
        &self.results
    }

    /// Generate comparison report
    pub fn generate_report(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)?;

        writeln!(file, "# ToRSh Performance Comparison Report\n")?;

        // Group results by operation
        let mut grouped: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();

        for result in &self.results {
            grouped
                .entry(result.operation.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (operation, results) in grouped {
            writeln!(file, "## {}\n", operation)?;
            writeln!(file, "| Library | Size | Time (μs) | Speedup vs ToRSh |")?;
            writeln!(file, "|---------|------|-----------|------------------|")?;

            for result in &results {
                let speedup = if result.library == "torsh" {
                    1.0
                } else {
                    // Find corresponding ToRSh result
                    if let Some(torsh_result) = results
                        .iter()
                        .find(|r| r.library == "torsh" && r.size == result.size)
                    {
                        torsh_result.time_ns / result.time_ns
                    } else {
                        1.0
                    }
                };

                writeln!(
                    file,
                    "| {} | {} | {:.2} | {:.2}x |",
                    result.library,
                    result.size,
                    result.time_ns / 1000.0,
                    speedup
                )?;
            }
            writeln!(file)?;
        }

        Ok(())
    }
}

impl Default for ComparisonRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison benchmark result
///
/// Contains performance metrics for a specific operation across different libraries,
/// enabling cross-library performance analysis and optimization guidance.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub operation: String,
    pub library: String,
    pub size: usize,
    pub time_ns: f64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<usize>,
}

/// Performance analysis engine
///
/// Provides comprehensive analysis of benchmark results including statistical analysis,
/// performance recommendations, and cross-library comparison insights.
#[derive(Debug, Default)]
pub struct PerformanceAnalyzer {
    results: Vec<ComparisonResult>,
    analysis_cache: HashMap<String, AnalysisResult>,
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Add results from comparison
    pub fn add_results(&mut self, results: &[ComparisonResult]) {
        self.results.extend_from_slice(results);
    }

    /// Analyze performance characteristics
    pub fn analyze_operation(&mut self, operation: &str) -> AnalysisResult {
        if let Some(cached) = self.analysis_cache.get(operation) {
            return cached.clone();
        }

        let op_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.operation == operation)
            .collect();

        let analysis = self.compute_analysis(&op_results);
        self.analysis_cache
            .insert(operation.to_string(), analysis.clone());
        analysis
    }

    fn compute_analysis(&self, results: &[&ComparisonResult]) -> AnalysisResult {
        if results.is_empty() {
            return AnalysisResult::default();
        }

        // Group by library
        let mut by_library: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();

        for result in results {
            by_library
                .entry(result.library.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        let mut library_stats = HashMap::new();
        for (library, lib_results) in by_library {
            let times: Vec<f64> = lib_results.iter().map(|r| r.time_ns).collect();
            let mean_time = times.iter().sum::<f64>() / times.len() as f64;
            let throughputs: Vec<f64> = lib_results.iter().filter_map(|r| r.throughput).collect();
            let mean_throughput = if !throughputs.is_empty() {
                Some(throughputs.iter().sum::<f64>() / throughputs.len() as f64)
            } else {
                None
            };

            library_stats.insert(
                library,
                LibraryStats {
                    mean_time_ns: mean_time,
                    mean_throughput,
                    sample_count: lib_results.len(),
                },
            );
        }

        // Find best performing library
        let best_library = library_stats
            .iter()
            .min_by(|a, b| a.1.mean_time_ns.partial_cmp(&b.1.mean_time_ns).unwrap())
            .map(|(name, _)| name.clone());

        AnalysisResult {
            operation: results[0].operation.clone(),
            library_stats,
            best_library,
            recommendations: self.generate_recommendations(results),
        }
    }

    fn generate_recommendations(&self, results: &[&ComparisonResult]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Find ToRSh results
        let torsh_results: Vec<_> = results.iter().filter(|r| r.library == "torsh").collect();

        if !torsh_results.is_empty() {
            let avg_torsh_time =
                torsh_results.iter().map(|r| r.time_ns).sum::<f64>() / torsh_results.len() as f64;

            // Compare with other libraries
            for library in ["ndarray", "pytorch", "tensorflow", "jax", "numpy"] {
                let lib_results: Vec<_> = results.iter().filter(|r| r.library == library).collect();

                if !lib_results.is_empty() {
                    let avg_lib_time = lib_results.iter().map(|r| r.time_ns).sum::<f64>()
                        / lib_results.len() as f64;

                    let speedup = avg_lib_time / avg_torsh_time;

                    if speedup > 1.2 {
                        recommendations.push(format!(
                            "ToRSh is {:.2}x faster than {} for this operation",
                            speedup, library
                        ));
                    } else if speedup < 0.8 {
                        recommendations.push(format!(
                            "Consider optimizing ToRSh implementation - {} is {:.2}x faster",
                            library,
                            1.0 / speedup
                        ));
                    }
                }
            }
        }

        recommendations
    }
}

/// Performance analysis result
///
/// Contains comprehensive statistical analysis of benchmark results across multiple libraries,
/// including performance recommendations and best-performing library identification.
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {
    pub operation: String,
    pub library_stats: HashMap<String, LibraryStats>,
    pub best_library: Option<String>,
    pub recommendations: Vec<String>,
}

/// Library performance statistics
///
/// Statistical summary of a library's performance for a specific operation,
/// including timing metrics, throughput, and sample size.
#[derive(Debug, Clone)]
pub struct LibraryStats {
    pub mean_time_ns: f64,
    pub mean_throughput: Option<f64>,
    pub sample_count: usize,
}

/// Performance regression detection system
///
/// Monitors benchmark results over time to identify performance regressions,
/// supporting both JSON and CSV baseline formats for historical comparison.
pub struct RegressionDetector {
    baseline_results: Vec<BenchResult>,
    threshold: f64, // Performance degradation threshold (e.g., 0.1 = 10%)
}

impl RegressionDetector {
    /// Create a new regression detector with specified threshold
    ///
    /// # Arguments
    /// * `threshold` - Performance degradation threshold (0.1 = 10% slower triggers regression)
    pub fn new(threshold: f64) -> Self {
        Self {
            baseline_results: Vec::new(),
            threshold,
        }
    }

    /// Load baseline results from file
    ///
    /// Supports both JSON and CSV formats for flexibility in CI/CD integration.
    /// JSON format preserves all metadata, while CSV provides simple tabular format.
    pub fn load_baseline(&mut self, path: &str) -> std::io::Result<()> {
        use std::fs;
        use std::io::Read;

        let extension = std::path::Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "json" => {
                let mut file = fs::File::open(path)?;
                let mut contents = String::new();
                file.read_to_string(&mut contents)?;

                self.baseline_results = serde_json::from_str(&contents)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            }
            "csv" => {
                let contents = fs::read_to_string(path)?;
                let mut results = Vec::new();

                for (line_num, line) in contents.lines().enumerate() {
                    if line_num == 0 || line.trim().is_empty() {
                        continue; // Skip header and empty lines
                    }

                    let fields: Vec<&str> = line.split(',').collect();
                    if fields.len() >= 7 {
                        let metrics = HashMap::new();

                        let result = BenchResult {
                            name: fields[0].to_string(),
                            size: fields[1].parse().unwrap_or(0),
                            dtype: torsh_core::dtype::DType::F32, // Default to F32
                            mean_time_ns: fields[3].parse().unwrap_or(0.0),
                            std_dev_ns: fields[4].parse().unwrap_or(0.0),
                            throughput: if fields[5] == "None" {
                                None
                            } else {
                                fields[5].parse().ok()
                            },
                            memory_usage: if fields[6] == "None" {
                                None
                            } else {
                                fields[6].parse().ok()
                            },
                            peak_memory: None,
                            metrics,
                        };
                        results.push(result);
                    }
                }

                self.baseline_results = results;
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unsupported file format. Use .json or .csv",
                ));
            }
        }

        Ok(())
    }

    /// Save current results as new baseline
    ///
    /// Persists benchmark results for future regression detection,
    /// supporting both JSON and CSV output formats.
    pub fn save_baseline(&self, results: &[BenchResult], path: &str) -> std::io::Result<()> {
        use std::fs;
        use std::io::Write;

        let extension = std::path::Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "json" => {
                let json_data = serde_json::to_string_pretty(results)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                let mut file = fs::File::create(path)?;
                file.write_all(json_data.as_bytes())?;
            }
            "csv" => {
                let mut file = fs::File::create(path)?;

                // Write CSV header
                writeln!(
                    file,
                    "name,size,dtype,mean_time_ns,std_dev_ns,throughput,memory_usage,peak_memory"
                )?;

                // Write data rows
                for result in results {
                    writeln!(
                        file,
                        "{},{},{:?},{},{},{},{},{}",
                        result.name,
                        result.size,
                        result.dtype,
                        result.mean_time_ns,
                        result.std_dev_ns,
                        result
                            .throughput
                            .map_or("None".to_string(), |v| v.to_string()),
                        result
                            .memory_usage
                            .map_or("None".to_string(), |v| v.to_string()),
                        result
                            .peak_memory
                            .map_or("None".to_string(), |v| v.to_string())
                    )?;
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unsupported file format. Use .json or .csv",
                ));
            }
        }

        Ok(())
    }

    /// Check for performance regressions
    ///
    /// Compares current benchmark results against baseline to identify
    /// significant performance degradations exceeding the configured threshold.
    pub fn check_regression(&self, current_results: &[BenchResult]) -> Vec<RegressionResult> {
        let mut regressions = Vec::new();

        for current in current_results {
            if let Some(baseline) = self
                .baseline_results
                .iter()
                .find(|b| b.name == current.name && b.size == current.size)
            {
                let slowdown = current.mean_time_ns / baseline.mean_time_ns;
                if slowdown > (1.0 + self.threshold) {
                    regressions.push(RegressionResult {
                        benchmark: current.name.clone(),
                        size: current.size,
                        baseline_time: baseline.mean_time_ns,
                        current_time: current.mean_time_ns,
                        slowdown_factor: slowdown,
                        is_regression: true,
                    });
                }
            }
        }

        regressions
    }
}

/// Performance regression detection result
///
/// Contains detailed information about detected performance regressions,
/// including baseline comparison and slowdown quantification.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub benchmark: String,
    pub size: usize,
    pub baseline_time: f64,
    pub current_time: f64,
    pub slowdown_factor: f64,
    pub is_regression: bool,
}

/// Common benchmark utilities and helper functions
pub mod utils {
    use super::*;

    /// Convert ComparisonResult to BenchResult format
    pub fn comparison_to_bench_result(result: &ComparisonResult) -> BenchResult {
        BenchResult {
            name: format!("{}-{}", result.library, result.operation),
            size: result.size,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: result.time_ns,
            std_dev_ns: 0.0, // Not tracked in ComparisonResult
            throughput: result.throughput,
            memory_usage: result.memory_usage,
            peak_memory: None,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Generate summary statistics across all libraries for an operation
    pub fn generate_operation_summary(results: &[ComparisonResult], operation: &str) -> String {
        let op_results: Vec<_> = results
            .iter()
            .filter(|r| r.operation == operation)
            .collect();

        if op_results.is_empty() {
            return format!("No results found for operation: {}", operation);
        }

        let mut summary = format!("Summary for {}:\n", operation);

        for result in &op_results {
            summary.push_str(&format!(
                "  {} (size {}): {:.2} μs\n",
                result.library,
                result.size,
                result.time_ns / 1000.0
            ));
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_runner() {
        let mut runner = ComparisonRunner::new();
        assert_eq!(runner.results().len(), 0);

        let result = ComparisonResult {
            operation: "matmul".to_string(),
            library: "torsh".to_string(),
            size: 128,
            time_ns: 1000.0,
            throughput: Some(256.0),
            memory_usage: Some(65536),
        };

        runner.add_result(result);
        assert_eq!(runner.results().len(), 1);
    }

    #[test]
    fn test_performance_analyzer() {
        let mut analyzer = PerformanceAnalyzer::new();

        let results = vec![
            ComparisonResult {
                operation: "matmul".to_string(),
                library: "torsh".to_string(),
                size: 128,
                time_ns: 1000.0,
                throughput: Some(256.0),
                memory_usage: None,
            },
            ComparisonResult {
                operation: "matmul".to_string(),
                library: "pytorch".to_string(),
                size: 128,
                time_ns: 1200.0,
                throughput: Some(213.3),
                memory_usage: None,
            },
        ];

        analyzer.add_results(&results);
        let analysis = analyzer.analyze_operation("matmul");

        assert_eq!(analysis.operation, "matmul");
        assert_eq!(analysis.library_stats.len(), 2);
        assert!(analysis.best_library.is_some());
        assert_eq!(analysis.best_library.unwrap(), "torsh");
    }

    #[test]
    fn test_regression_detector() {
        let detector = RegressionDetector::new(0.1); // 10% threshold
        assert_eq!(detector.threshold, 0.1);
        assert_eq!(detector.baseline_results.len(), 0);
    }
}
