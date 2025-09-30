//! Performance analysis and regression detection utilities
//!
//! This module provides tools for analyzing benchmark results, detecting
//! performance regressions, and generating insights.

use super::core::ComparisonResult;
use crate::BenchResult;
use std::collections::HashMap;

/// Performance analysis and comparison utilities
#[derive(Default)]
pub struct PerformanceAnalyzer {
    results: Vec<ComparisonResult>,
    analysis_cache: HashMap<String, AnalysisResult>,
}

impl PerformanceAnalyzer {
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
            for library in ["ndarray", "pytorch", "tensorflow"] {
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

#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {
    pub operation: String,
    pub library_stats: HashMap<String, LibraryStats>,
    pub best_library: Option<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LibraryStats {
    pub mean_time_ns: f64,
    pub mean_throughput: Option<f64>,
    pub sample_count: usize,
}

/// Performance regression detection
pub struct RegressionDetector {
    baseline_results: Vec<BenchResult>,
    threshold: f64, // Performance degradation threshold (e.g., 0.1 = 10%)
}

impl RegressionDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            baseline_results: Vec::new(),
            threshold,
        }
    }

    /// Load baseline results from file
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
                        let mut metrics = HashMap::new();

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

/// Regression detection result
#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub benchmark: String,
    pub size: usize,
    pub baseline_time: f64,
    pub current_time: f64,
    pub slowdown_factor: f64,
    pub is_regression: bool,
}