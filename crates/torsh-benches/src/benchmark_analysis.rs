//! Enhanced benchmark analysis and reporting
//!
//! This module provides advanced analysis capabilities for benchmark results
//! including statistical analysis, performance trend detection, and detailed reporting.

use crate::BenchResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;

/// Statistical summary of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,                      // 25th percentile
    pub q75: f64,                      // 75th percentile
    pub coefficient_of_variation: f64, // std_dev / mean
    pub confidence_interval_95: (f64, f64),
}

/// Performance analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub benchmark_name: String,
    pub timestamp: DateTime<Utc>,
    pub total_runs: usize,
    pub statistics: BenchmarkStatistics,
    pub performance_rating: PerformanceRating,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub recommendations: Vec<String>,
}

/// Performance rating classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceRating {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
}

/// Bottleneck analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub memory_bound: bool,
    pub compute_bound: bool,
    pub cache_efficiency: f64,    // 0.0 to 1.0
    pub parallel_efficiency: f64, // 0.0 to 1.0
    pub estimated_peak_performance: f64,
    pub performance_gap: f64, // How far from theoretical peak
}

/// Enhanced benchmark analyzer
pub struct BenchmarkAnalyzer {
    results_history: Vec<PerformanceAnalysis>,
    baseline_results: Option<HashMap<String, BenchmarkStatistics>>,
}

impl BenchmarkAnalyzer {
    /// Create a new benchmark analyzer
    pub fn new() -> Self {
        Self {
            results_history: Vec::new(),
            baseline_results: None,
        }
    }

    /// Load baseline results from file
    pub fn load_baseline(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let baseline: HashMap<String, BenchmarkStatistics> = serde_json::from_str(&content)?;
        self.baseline_results = Some(baseline);
        Ok(())
    }

    /// Save baseline results to file
    pub fn save_baseline(
        &self,
        results: &[BenchResult],
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut baseline = HashMap::new();

        for result in results {
            let stats = self.calculate_statistics(&[result.mean_time_ns]);
            baseline.insert(format!("{}_{}", result.name, result.size), stats);
        }

        let content = serde_json::to_string_pretty(&baseline)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Analyze benchmark results with comprehensive analysis
    pub fn analyze_results(&mut self, results: &[BenchResult]) -> Vec<PerformanceAnalysis> {
        let mut analyses = Vec::new();

        // Group results by benchmark name
        let mut grouped_results: HashMap<String, Vec<&BenchResult>> = HashMap::new();
        for result in results {
            grouped_results
                .entry(result.name.clone())
                .or_default()
                .push(result);
        }

        for (benchmark_name, bench_results) in grouped_results {
            // Extract timing data
            let times: Vec<f64> = bench_results.iter().map(|r| r.mean_time_ns).collect();

            let statistics = self.calculate_statistics(&times);
            let performance_rating = self.classify_performance(&statistics, &benchmark_name);
            let bottleneck_analysis = self.analyze_bottlenecks(&bench_results);
            let recommendations =
                self.generate_recommendations(&statistics, &bottleneck_analysis, &benchmark_name);

            let analysis = PerformanceAnalysis {
                benchmark_name: benchmark_name.clone(),
                timestamp: Utc::now(),
                total_runs: bench_results.len(),
                statistics,
                performance_rating,
                bottleneck_analysis,
                recommendations,
            };

            analyses.push(analysis);
        }

        self.results_history.extend(analyses.clone());
        analyses
    }

    /// Calculate comprehensive statistics
    fn calculate_statistics(&self, data: &[f64]) -> BenchmarkStatistics {
        if data.is_empty() {
            return BenchmarkStatistics {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                q25: 0.0,
                q75: 0.0,
                coefficient_of_variation: 0.0,
                confidence_interval_95: (0.0, 0.0),
            };
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).expect("NaN values in benchmark data"));

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let median = self.calculate_percentile(&sorted_data, 0.5);
        let q25 = self.calculate_percentile(&sorted_data, 0.25);
        let q75 = self.calculate_percentile(&sorted_data, 0.75);

        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        let coefficient_of_variation = if mean != 0.0 { std_dev / mean } else { 0.0 };

        // 95% confidence interval (approximate using t-distribution)
        let standard_error = std_dev / (data.len() as f64).sqrt();
        let t_critical = 1.96; // Approximate for large samples
        let margin_of_error = t_critical * standard_error;
        let confidence_interval_95 = (mean - margin_of_error, mean + margin_of_error);

        BenchmarkStatistics {
            mean,
            median,
            std_dev,
            min: sorted_data[0],
            max: sorted_data[sorted_data.len() - 1],
            q25,
            q75,
            coefficient_of_variation,
            confidence_interval_95,
        }
    }

    fn calculate_percentile(&self, sorted_data: &[f64], percentile: f64) -> f64 {
        let index = percentile * (sorted_data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }

    /// Classify performance based on statistics and comparison with baseline
    fn classify_performance(
        &self,
        stats: &BenchmarkStatistics,
        benchmark_name: &str,
    ) -> PerformanceRating {
        // Check against baseline if available
        if let Some(ref baseline) = self.baseline_results {
            if let Some(baseline_stats) = baseline.get(benchmark_name) {
                let performance_ratio = stats.mean / baseline_stats.mean;
                return match performance_ratio {
                    ratio if ratio <= 0.9 => PerformanceRating::Excellent,
                    ratio if ratio <= 1.05 => PerformanceRating::Good,
                    ratio if ratio <= 1.2 => PerformanceRating::Acceptable,
                    ratio if ratio <= 1.5 => PerformanceRating::Poor,
                    _ => PerformanceRating::Critical,
                };
            }
        }

        // Fallback to coefficient of variation analysis
        match stats.coefficient_of_variation {
            cv if cv <= 0.05 => PerformanceRating::Excellent,
            cv if cv <= 0.1 => PerformanceRating::Good,
            cv if cv <= 0.2 => PerformanceRating::Acceptable,
            cv if cv <= 0.3 => PerformanceRating::Poor,
            _ => PerformanceRating::Critical,
        }
    }

    /// Analyze potential bottlenecks
    fn analyze_bottlenecks(&self, results: &[&BenchResult]) -> BottleneckAnalysis {
        let mut memory_bound = false;
        let mut compute_bound = false;
        let mut cache_efficiency = 0.8; // Default estimate
        let mut parallel_efficiency = 0.7; // Default estimate

        // Analyze memory bandwidth vs compute requirements
        if let Some(first_result) = results.first() {
            // Estimate if operation is memory bound
            if first_result.name.contains("elementwise")
                || first_result.name.contains("copy")
                || first_result.name.contains("reduction")
            {
                memory_bound = true;
                cache_efficiency = 0.6; // Lower for memory-bound operations
            }

            // Estimate if operation is compute bound
            if first_result.name.contains("matmul")
                || first_result.name.contains("conv")
                || first_result.name.contains("attention")
            {
                compute_bound = true;
                parallel_efficiency = 0.8; // Higher for compute-bound operations
            }
        }

        // Estimate theoretical peak performance (simplified)
        let estimated_peak_performance = if compute_bound {
            // Estimate based on typical FLOPS
            1e12 // 1 TFLOPS as example
        } else {
            // Estimate based on memory bandwidth
            100e9 // 100 GB/s as example
        };

        let actual_performance = results.first().and_then(|r| r.throughput).unwrap_or(1e6);

        let performance_gap =
            (estimated_peak_performance - actual_performance) / estimated_peak_performance;

        BottleneckAnalysis {
            memory_bound,
            compute_bound,
            cache_efficiency,
            parallel_efficiency,
            estimated_peak_performance,
            performance_gap: performance_gap.max(0.0).min(1.0),
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(
        &self,
        stats: &BenchmarkStatistics,
        bottleneck: &BottleneckAnalysis,
        benchmark_name: &str,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Variability recommendations
        if stats.coefficient_of_variation > 0.1 {
            recommendations.push("High performance variability detected. Consider:".to_string());
            recommendations.push("  - Running benchmarks on a quiet system".to_string());
            recommendations.push("  - Increasing warm-up iterations".to_string());
            recommendations.push("  - Using CPU isolation or real-time scheduling".to_string());
        }

        // Memory-bound recommendations
        if bottleneck.memory_bound {
            recommendations.push("Memory-bound operation detected. Optimize by:".to_string());
            recommendations.push("  - Improving data locality and cache utilization".to_string());
            recommendations.push("  - Using memory prefetching".to_string());
            recommendations
                .push("  - Considering data layout optimizations (AoS vs SoA)".to_string());

            if bottleneck.cache_efficiency < 0.7 {
                recommendations
                    .push("  - Cache efficiency is low - consider blocking algorithms".to_string());
            }
        }

        // Compute-bound recommendations
        if bottleneck.compute_bound {
            recommendations.push("Compute-bound operation detected. Optimize by:".to_string());
            recommendations
                .push("  - Utilizing SIMD instructions (AVX, AVX2, AVX-512)".to_string());
            recommendations.push("  - Improving parallel efficiency".to_string());
            recommendations.push("  - Using specialized libraries (BLAS, cuDNN)".to_string());

            if bottleneck.parallel_efficiency < 0.8 {
                recommendations.push(
                    "  - Parallel efficiency is suboptimal - check for synchronization overhead"
                        .to_string(),
                );
            }
        }

        // Size-specific recommendations
        if benchmark_name.contains("small") || benchmark_name.contains("64") {
            recommendations.push("Small tensor operations:".to_string());
            recommendations.push("  - Consider operation fusion to reduce overhead".to_string());
            recommendations.push("  - Use in-place operations when possible".to_string());
        }

        if benchmark_name.contains("large") || benchmark_name.contains("4096") {
            recommendations.push("Large tensor operations:".to_string());
            recommendations.push("  - Ensure efficient memory management".to_string());
            recommendations
                .push("  - Consider out-of-core algorithms for very large data".to_string());
        }

        // Performance gap recommendations
        if bottleneck.performance_gap > 0.5 {
            recommendations
                .push("Large performance gap detected (>50% from theoretical peak):".to_string());
            recommendations.push("  - Profile with detailed performance counters".to_string());
            recommendations.push("  - Check for algorithmic inefficiencies".to_string());
            recommendations.push("  - Consider alternative implementations".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance looks good! No specific recommendations.".to_string());
        }

        recommendations
    }

    /// Generate comprehensive analysis report
    pub fn generate_analysis_report(
        &self,
        analyses: &[PerformanceAnalysis],
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = fs::File::create(output_path)?;

        writeln!(file, "# ToRSh Benchmark Analysis Report")?;
        writeln!(
            file,
            "Generated: {}\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        writeln!(file, "## Executive Summary\n")?;

        let excellent_count = analyses
            .iter()
            .filter(|a| matches!(a.performance_rating, PerformanceRating::Excellent))
            .count();
        let good_count = analyses
            .iter()
            .filter(|a| matches!(a.performance_rating, PerformanceRating::Good))
            .count();
        let acceptable_count = analyses
            .iter()
            .filter(|a| matches!(a.performance_rating, PerformanceRating::Acceptable))
            .count();
        let poor_count = analyses
            .iter()
            .filter(|a| matches!(a.performance_rating, PerformanceRating::Poor))
            .count();
        let critical_count = analyses
            .iter()
            .filter(|a| matches!(a.performance_rating, PerformanceRating::Critical))
            .count();

        writeln!(file, "- **Total Benchmarks**: {}", analyses.len())?;
        writeln!(
            file,
            "- **Excellent Performance**: {} ({:.1}%)",
            excellent_count,
            (excellent_count as f64 / analyses.len() as f64) * 100.0
        )?;
        writeln!(
            file,
            "- **Good Performance**: {} ({:.1}%)",
            good_count,
            (good_count as f64 / analyses.len() as f64) * 100.0
        )?;
        writeln!(
            file,
            "- **Acceptable Performance**: {} ({:.1}%)",
            acceptable_count,
            (acceptable_count as f64 / analyses.len() as f64) * 100.0
        )?;
        writeln!(
            file,
            "- **Poor Performance**: {} ({:.1}%)",
            poor_count,
            (poor_count as f64 / analyses.len() as f64) * 100.0
        )?;
        writeln!(
            file,
            "- **Critical Performance**: {} ({:.1}%)",
            critical_count,
            (critical_count as f64 / analyses.len() as f64) * 100.0
        )?;

        writeln!(file, "\n## Detailed Analysis\n")?;

        for analysis in analyses {
            writeln!(file, "### {}\n", analysis.benchmark_name)?;

            writeln!(
                file,
                "**Performance Rating**: {:?}",
                analysis.performance_rating
            )?;
            writeln!(file, "**Total Runs**: {}", analysis.total_runs)?;

            writeln!(file, "\n**Statistics:**")?;
            writeln!(file, "- Mean: {:.2} ns", analysis.statistics.mean)?;
            writeln!(file, "- Median: {:.2} ns", analysis.statistics.median)?;
            writeln!(file, "- Std Dev: {:.2} ns", analysis.statistics.std_dev)?;
            writeln!(file, "- Min: {:.2} ns", analysis.statistics.min)?;
            writeln!(file, "- Max: {:.2} ns", analysis.statistics.max)?;
            writeln!(
                file,
                "- Coefficient of Variation: {:.3}",
                analysis.statistics.coefficient_of_variation
            )?;
            writeln!(
                file,
                "- 95% Confidence Interval: ({:.2}, {:.2}) ns",
                analysis.statistics.confidence_interval_95.0,
                analysis.statistics.confidence_interval_95.1
            )?;

            writeln!(file, "\n**Bottleneck Analysis:**")?;
            writeln!(
                file,
                "- Memory Bound: {}",
                analysis.bottleneck_analysis.memory_bound
            )?;
            writeln!(
                file,
                "- Compute Bound: {}",
                analysis.bottleneck_analysis.compute_bound
            )?;
            writeln!(
                file,
                "- Cache Efficiency: {:.1}%",
                analysis.bottleneck_analysis.cache_efficiency * 100.0
            )?;
            writeln!(
                file,
                "- Parallel Efficiency: {:.1}%",
                analysis.bottleneck_analysis.parallel_efficiency * 100.0
            )?;
            writeln!(
                file,
                "- Performance Gap: {:.1}%",
                analysis.bottleneck_analysis.performance_gap * 100.0
            )?;

            writeln!(file, "\n**Recommendations:**")?;
            for rec in &analysis.recommendations {
                writeln!(file, "{}", rec)?;
            }

            writeln!(file)?;
        }

        writeln!(file, "## Overall Recommendations\n")?;

        let memory_bound_count = analyses
            .iter()
            .filter(|a| a.bottleneck_analysis.memory_bound)
            .count();
        let compute_bound_count = analyses
            .iter()
            .filter(|a| a.bottleneck_analysis.compute_bound)
            .count();

        if memory_bound_count > compute_bound_count {
            writeln!(file, "**Primary Focus: Memory Optimization**")?;
            writeln!(file, "- Most operations are memory-bound")?;
            writeln!(file, "- Focus on cache optimization and data layout")?;
            writeln!(
                file,
                "- Consider memory prefetching and blocking algorithms"
            )?;
        } else {
            writeln!(file, "**Primary Focus: Compute Optimization**")?;
            writeln!(file, "- Most operations are compute-bound")?;
            writeln!(file, "- Focus on SIMD utilization and parallelization")?;
            writeln!(file, "- Consider specialized compute libraries")?;
        }

        let avg_variability: f64 = analyses
            .iter()
            .map(|a| a.statistics.coefficient_of_variation)
            .sum::<f64>()
            / analyses.len() as f64;

        if avg_variability > 0.15 {
            writeln!(file, "\n**Performance Stability Issues Detected**")?;
            writeln!(
                file,
                "- High average variability: {:.1}%",
                avg_variability * 100.0
            )?;
            writeln!(
                file,
                "- Consider system-level optimizations for consistent performance"
            )?;
        }

        writeln!(file, "\n---")?;
        writeln!(file, "*Report generated by ToRSh Benchmark Analysis Suite*")?;

        Ok(())
    }

    /// Generate CSV export with extended statistics
    pub fn export_detailed_csv(
        &self,
        analyses: &[PerformanceAnalysis],
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = fs::File::create(output_path)?;

        writeln!(file, "benchmark_name,timestamp,total_runs,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,q25_ns,q75_ns,coefficient_of_variation,ci_lower_ns,ci_upper_ns,performance_rating,memory_bound,compute_bound,cache_efficiency,parallel_efficiency,performance_gap")?;

        for analysis in analyses {
            writeln!(file, "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.2},{:.2},{:?},{},{},{:.3},{:.3},{:.3}",
                analysis.benchmark_name,
                analysis.timestamp.format("%Y-%m-%d %H:%M:%S"),
                analysis.total_runs,
                analysis.statistics.mean,
                analysis.statistics.median,
                analysis.statistics.std_dev,
                analysis.statistics.min,
                analysis.statistics.max,
                analysis.statistics.q25,
                analysis.statistics.q75,
                analysis.statistics.coefficient_of_variation,
                analysis.statistics.confidence_interval_95.0,
                analysis.statistics.confidence_interval_95.1,
                analysis.performance_rating,
                analysis.bottleneck_analysis.memory_bound,
                analysis.bottleneck_analysis.compute_bound,
                analysis.bottleneck_analysis.cache_efficiency,
                analysis.bottleneck_analysis.parallel_efficiency,
                analysis.bottleneck_analysis.performance_gap,
            )?;
        }

        Ok(())
    }
}

impl Default for BenchmarkAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to generate performance trend analysis
pub fn analyze_performance_trends(history: &[PerformanceAnalysis]) -> Vec<String> {
    let mut trends = Vec::new();

    if history.len() < 2 {
        trends
            .push("Insufficient data for trend analysis (need at least 2 data points)".to_string());
        return trends;
    }

    // Group by benchmark name and analyze trends
    let mut grouped: HashMap<String, Vec<&PerformanceAnalysis>> = HashMap::new();
    for analysis in history {
        grouped
            .entry(analysis.benchmark_name.clone())
            .or_default()
            .push(analysis);
    }

    for (benchmark_name, benchmark_history) in grouped {
        if benchmark_history.len() < 2 {
            continue;
        }

        // Sort by timestamp
        let mut sorted_history = benchmark_history;
        sorted_history.sort_by_key(|a| a.timestamp);

        let first = sorted_history
            .first()
            .expect("sorted_history should have at least 2 elements");
        let last = sorted_history
            .last()
            .expect("sorted_history should have at least 2 elements");

        let performance_change =
            (last.statistics.mean - first.statistics.mean) / first.statistics.mean;

        if performance_change.abs() > 0.05 {
            // 5% threshold
            let direction = if performance_change > 0.0 {
                "degraded"
            } else {
                "improved"
            };
            let percentage = (performance_change.abs() * 100.0) as i32;

            trends.push(format!(
                "{}: Performance {} by {}% over time",
                benchmark_name, direction, percentage
            ));
        }
    }

    if trends.is_empty() {
        trends.push("No significant performance trends detected".to_string());
    }

    trends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_calculation() {
        let analyzer = BenchmarkAnalyzer::new();
        let data = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let stats = analyzer.calculate_statistics(&data);

        assert_eq!(stats.mean, 300.0);
        assert_eq!(stats.median, 300.0);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 500.0);
    }

    #[test]
    fn test_performance_classification() {
        let analyzer = BenchmarkAnalyzer::new();
        let stats = BenchmarkStatistics {
            mean: 100.0,
            median: 100.0,
            std_dev: 5.0,
            min: 95.0,
            max: 105.0,
            q25: 97.5,
            q75: 102.5,
            coefficient_of_variation: 0.05,
            confidence_interval_95: (95.0, 105.0),
        };

        let rating = analyzer.classify_performance(&stats, "test_benchmark");
        assert!(matches!(rating, PerformanceRating::Excellent));
    }
}
