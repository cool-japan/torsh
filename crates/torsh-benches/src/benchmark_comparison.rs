//! Advanced benchmark comparison and result aggregation utilities
//!
//! This module provides tools for comparing benchmark results across different runs,
//! configurations, and implementations, with detailed statistical analysis.

use crate::BenchResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comparison between two benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub benchmark_name: String,
    pub baseline: ComparisonMetrics,
    pub candidate: ComparisonMetrics,
    pub speedup: f64, // candidate / baseline (> 1.0 means faster)
    pub improvement_percentage: f64,
    pub statistical_significance: SignificanceLevel,
    pub verdict: ComparisonVerdict,
}

/// Metrics for a single benchmark in a comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub mean_time_ns: f64,
    pub std_dev: f64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<usize>,
    pub label: String,
}

/// Statistical significance level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignificanceLevel {
    VeryHighlySignificant, // p < 0.001
    HighlySignificant,     // p < 0.01
    Significant,           // p < 0.05
    NotSignificant,        // p >= 0.05
}

/// Comparison verdict
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonVerdict {
    MajorImprovement, // > 20% faster
    Improvement,      // 5-20% faster
    NoChange,         // < 5% difference
    Regression,       // 5-20% slower
    MajorRegression,  // > 20% slower
}

/// Benchmark comparison tool
pub struct BenchmarkComparator {
    baseline_results: HashMap<String, BenchResult>,
    candidate_results: HashMap<String, BenchResult>,
    comparisons: Vec<BenchmarkComparison>,
}

impl BenchmarkComparator {
    /// Create a new comparator
    pub fn new() -> Self {
        Self {
            baseline_results: HashMap::new(),
            candidate_results: HashMap::new(),
            comparisons: Vec::new(),
        }
    }

    /// Add baseline results
    pub fn add_baseline(&mut self, results: &[BenchResult], _label: &str) {
        for result in results {
            let key = format!("{}_{}", result.name, result.size);
            self.baseline_results.insert(key, result.clone());
        }
    }

    /// Add candidate results to compare against baseline
    pub fn add_candidate(&mut self, results: &[BenchResult], _label: &str) {
        for result in results {
            let key = format!("{}_{}", result.name, result.size);
            self.candidate_results.insert(key, result.clone());
        }
    }

    /// Perform comparison analysis
    pub fn compare(&mut self) -> &[BenchmarkComparison] {
        self.comparisons.clear();

        for (key, baseline) in &self.baseline_results {
            if let Some(candidate) = self.candidate_results.get(key) {
                let comparison = self.compare_results(baseline, candidate);
                self.comparisons.push(comparison);
            }
        }

        &self.comparisons
    }

    /// Compare two individual benchmark results
    fn compare_results(
        &self,
        baseline: &BenchResult,
        candidate: &BenchResult,
    ) -> BenchmarkComparison {
        let baseline_label = baseline
            .metrics
            .get("label")
            .map(|_| "baseline_labeled")
            .unwrap_or("baseline");

        let candidate_label = candidate
            .metrics
            .get("label")
            .map(|_| "candidate_labeled")
            .unwrap_or("candidate");

        let baseline_metrics = ComparisonMetrics {
            mean_time_ns: baseline.mean_time_ns,
            std_dev: baseline.std_dev_ns,
            throughput: baseline.throughput,
            memory_usage: baseline.memory_usage,
            label: baseline_label.to_string(),
        };

        let candidate_metrics = ComparisonMetrics {
            mean_time_ns: candidate.mean_time_ns,
            std_dev: candidate.std_dev_ns,
            throughput: candidate.throughput,
            memory_usage: candidate.memory_usage,
            label: candidate_label.to_string(),
        };

        // Calculate speedup (baseline / candidate - higher is better for candidate)
        let speedup = baseline.mean_time_ns / candidate.mean_time_ns;
        let improvement_percentage = (speedup - 1.0) * 100.0;

        // Perform t-test for statistical significance
        let significance = self.calculate_significance(
            baseline.mean_time_ns,
            baseline.std_dev_ns,
            candidate.mean_time_ns,
            candidate.std_dev_ns,
        );

        // Determine verdict
        let verdict = if improvement_percentage > 20.0 {
            ComparisonVerdict::MajorImprovement
        } else if improvement_percentage > 5.0 {
            ComparisonVerdict::Improvement
        } else if improvement_percentage < -20.0 {
            ComparisonVerdict::MajorRegression
        } else if improvement_percentage < -5.0 {
            ComparisonVerdict::Regression
        } else {
            ComparisonVerdict::NoChange
        };

        BenchmarkComparison {
            benchmark_name: format!("{}_{}", baseline.name, baseline.size),
            baseline: baseline_metrics,
            candidate: candidate_metrics,
            speedup,
            improvement_percentage,
            statistical_significance: significance,
            verdict,
        }
    }

    /// Calculate statistical significance using approximate t-test
    fn calculate_significance(
        &self,
        mean1: f64,
        std1: f64,
        mean2: f64,
        std2: f64,
    ) -> SignificanceLevel {
        // Approximate t-statistic (assuming equal sample sizes n=30)
        let n = 30.0_f64;
        let pooled_std = ((std1.powi(2) + std2.powi(2)) / 2.0).sqrt();
        let t_stat = ((mean1 - mean2).abs() / (pooled_std * (2.0_f64 / n).sqrt())).abs();

        // Approximate p-value thresholds for two-tailed test
        if t_stat > 3.5 {
            SignificanceLevel::VeryHighlySignificant // p < 0.001
        } else if t_stat > 2.75 {
            SignificanceLevel::HighlySignificant // p < 0.01
        } else if t_stat > 2.0 {
            SignificanceLevel::Significant // p < 0.05
        } else {
            SignificanceLevel::NotSignificant
        }
    }

    /// Generate markdown comparison report
    pub fn generate_markdown_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Benchmark Comparison Report\n\n");
        report.push_str(&format!(
            "Generated: {}\n\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary section
        let improvements = self
            .comparisons
            .iter()
            .filter(|c| {
                matches!(
                    c.verdict,
                    ComparisonVerdict::Improvement | ComparisonVerdict::MajorImprovement
                )
            })
            .count();
        let regressions = self
            .comparisons
            .iter()
            .filter(|c| {
                matches!(
                    c.verdict,
                    ComparisonVerdict::Regression | ComparisonVerdict::MajorRegression
                )
            })
            .count();
        let no_change = self
            .comparisons
            .iter()
            .filter(|c| matches!(c.verdict, ComparisonVerdict::NoChange))
            .count();

        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- **Total Comparisons**: {}\n",
            self.comparisons.len()
        ));
        report.push_str(&format!(
            "- **Improvements**: {} ({:.1}%)\n",
            improvements,
            (improvements as f64 / self.comparisons.len() as f64) * 100.0
        ));
        report.push_str(&format!(
            "- **Regressions**: {} ({:.1}%)\n",
            regressions,
            (regressions as f64 / self.comparisons.len() as f64) * 100.0
        ));
        report.push_str(&format!(
            "- **No Change**: {} ({:.1}%)\n\n",
            no_change,
            (no_change as f64 / self.comparisons.len() as f64) * 100.0
        ));

        // Detailed results table
        report.push_str("## Detailed Comparison\n\n");
        report.push_str("| Benchmark | Baseline (ns) | Candidate (ns) | Speedup | Change | Significance | Verdict |\n");
        report.push_str("|-----------|---------------|----------------|---------|--------|--------------|----------|\n");

        for comparison in &self.comparisons {
            let significance = match comparison.statistical_significance {
                SignificanceLevel::VeryHighlySignificant => "***",
                SignificanceLevel::HighlySignificant => "**",
                SignificanceLevel::Significant => "*",
                SignificanceLevel::NotSignificant => "n.s.",
            };

            let verdict_emoji = match comparison.verdict {
                ComparisonVerdict::MajorImprovement => "🚀",
                ComparisonVerdict::Improvement => "✅",
                ComparisonVerdict::NoChange => "➖",
                ComparisonVerdict::Regression => "⚠️",
                ComparisonVerdict::MajorRegression => "🔴",
            };

            report.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:.2}x | {:+.1}% | {} | {} |\n",
                comparison.benchmark_name,
                comparison.baseline.mean_time_ns,
                comparison.candidate.mean_time_ns,
                comparison.speedup,
                comparison.improvement_percentage,
                significance,
                verdict_emoji
            ));
        }

        report.push_str("\n### Legend\n");
        report.push_str("- Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n");
        report.push_str("- Verdict: 🚀 Major improvement (>20%), ✅ Improvement (5-20%), ➖ No change (<5%), ⚠️ Regression (5-20%), 🔴 Major regression (>20%)\n");

        report
    }

    /// Export comparison results to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct ComparisonReport {
            timestamp: DateTime<Utc>,
            total_comparisons: usize,
            improvements: usize,
            regressions: usize,
            comparisons: Vec<BenchmarkComparison>,
        }

        let improvements = self
            .comparisons
            .iter()
            .filter(|c| {
                matches!(
                    c.verdict,
                    ComparisonVerdict::Improvement | ComparisonVerdict::MajorImprovement
                )
            })
            .count();
        let regressions = self
            .comparisons
            .iter()
            .filter(|c| {
                matches!(
                    c.verdict,
                    ComparisonVerdict::Regression | ComparisonVerdict::MajorRegression
                )
            })
            .count();

        let report = ComparisonReport {
            timestamp: Utc::now(),
            total_comparisons: self.comparisons.len(),
            improvements,
            regressions,
            comparisons: self.comparisons.clone(),
        };

        serde_json::to_string_pretty(&report)
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> ComparisonSummary {
        let mut speedups = Vec::new();
        let mut improvements = 0;
        let mut regressions = 0;
        let mut major_improvements = 0;
        let mut major_regressions = 0;

        for comparison in &self.comparisons {
            speedups.push(comparison.speedup);
            match comparison.verdict {
                ComparisonVerdict::MajorImprovement => major_improvements += 1,
                ComparisonVerdict::Improvement => improvements += 1,
                ComparisonVerdict::Regression => regressions += 1,
                ComparisonVerdict::MajorRegression => major_regressions += 1,
                ComparisonVerdict::NoChange => {}
            }
        }

        let avg_speedup = if !speedups.is_empty() {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        } else {
            1.0
        };

        let geomean_speedup = if !speedups.is_empty() {
            speedups
                .iter()
                .product::<f64>()
                .powf(1.0 / speedups.len() as f64)
        } else {
            1.0
        };

        ComparisonSummary {
            total_comparisons: self.comparisons.len(),
            improvements: improvements + major_improvements,
            regressions: regressions + major_regressions,
            major_improvements,
            major_regressions,
            average_speedup: avg_speedup,
            geometric_mean_speedup: geomean_speedup,
        }
    }
}

/// Summary of comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub total_comparisons: usize,
    pub improvements: usize,
    pub regressions: usize,
    pub major_improvements: usize,
    pub major_regressions: usize,
    pub average_speedup: f64,
    pub geometric_mean_speedup: f64,
}

impl Default for BenchmarkComparator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use torsh_core::dtype::DType;

    fn create_test_result(name: &str, size: usize, mean_ns: f64, std_ns: f64) -> BenchResult {
        BenchResult {
            name: name.to_string(),
            size,
            dtype: DType::F32,
            mean_time_ns: mean_ns,
            std_dev_ns: std_ns,
            throughput: Some(1_000_000_000.0 / mean_ns),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            metrics: HashMap::new(),
        }
    }

    #[test]
    fn test_comparator_creation() {
        let comparator = BenchmarkComparator::new();
        assert_eq!(comparator.baseline_results.len(), 0);
        assert_eq!(comparator.candidate_results.len(), 0);
    }

    #[test]
    fn test_add_baseline_and_candidate() {
        let mut comparator = BenchmarkComparator::new();

        let baseline = vec![create_test_result("matmul", 1024, 1000.0, 50.0)];
        let candidate = vec![create_test_result("matmul", 1024, 800.0, 40.0)];

        comparator.add_baseline(&baseline, "baseline_v1");
        comparator.add_candidate(&candidate, "optimized_v2");

        assert_eq!(comparator.baseline_results.len(), 1);
        assert_eq!(comparator.candidate_results.len(), 1);
    }

    #[test]
    fn test_comparison_improvement() {
        let mut comparator = BenchmarkComparator::new();

        // Baseline is slower (1000 ns)
        let baseline = vec![create_test_result("matmul", 1024, 1000.0, 50.0)];
        // Candidate is faster (700 ns) - 30% improvement
        let candidate = vec![create_test_result("matmul", 1024, 700.0, 35.0)];

        comparator.add_baseline(&baseline, "baseline");
        comparator.add_candidate(&candidate, "optimized");

        let comparisons = comparator.compare();
        assert_eq!(comparisons.len(), 1);

        let comparison = &comparisons[0];
        assert!(comparison.speedup > 1.0);
        assert!(comparison.improvement_percentage > 20.0);
        assert_eq!(comparison.verdict, ComparisonVerdict::MajorImprovement);
    }

    #[test]
    fn test_comparison_regression() {
        let mut comparator = BenchmarkComparator::new();

        // Baseline is faster (700 ns)
        let baseline = vec![create_test_result("matmul", 1024, 700.0, 35.0)];
        // Candidate is slower (1000 ns) - 30% regression
        let candidate = vec![create_test_result("matmul", 1024, 1000.0, 50.0)];

        comparator.add_baseline(&baseline, "baseline");
        comparator.add_candidate(&candidate, "changed");

        let comparisons = comparator.compare();
        assert_eq!(comparisons.len(), 1);

        let comparison = &comparisons[0];
        assert!(comparison.speedup < 1.0);
        assert!(comparison.improvement_percentage < -20.0);
        assert_eq!(comparison.verdict, ComparisonVerdict::MajorRegression);
    }

    #[test]
    fn test_comparison_no_change() {
        let mut comparator = BenchmarkComparator::new();

        let baseline = vec![create_test_result("matmul", 1024, 1000.0, 50.0)];
        let candidate = vec![create_test_result("matmul", 1024, 1020.0, 51.0)]; // 2% difference

        comparator.add_baseline(&baseline, "baseline");
        comparator.add_candidate(&candidate, "candidate");

        let comparisons = comparator.compare();
        assert_eq!(comparisons.len(), 1);

        let comparison = &comparisons[0];
        assert_eq!(comparison.verdict, ComparisonVerdict::NoChange);
    }

    #[test]
    fn test_summary_statistics() {
        let mut comparator = BenchmarkComparator::new();

        let baseline = vec![
            create_test_result("op1", 100, 1000.0, 50.0),
            create_test_result("op2", 200, 2000.0, 100.0),
            create_test_result("op3", 300, 1500.0, 75.0),
        ];

        let candidate = vec![
            create_test_result("op1", 100, 700.0, 35.0), // 30% improvement
            create_test_result("op2", 200, 1800.0, 90.0), // 10% improvement
            create_test_result("op3", 300, 1600.0, 80.0), // 6% regression
        ];

        comparator.add_baseline(&baseline, "baseline");
        comparator.add_candidate(&candidate, "optimized");
        comparator.compare();

        let summary = comparator.get_summary();
        assert_eq!(summary.total_comparisons, 3);
        assert_eq!(summary.improvements, 2);
        assert_eq!(summary.regressions, 1);
        assert!(summary.average_speedup > 1.0); // Overall improvement
    }

    #[test]
    fn test_markdown_report_generation() {
        let mut comparator = BenchmarkComparator::new();

        let baseline = vec![create_test_result("matmul", 1024, 1000.0, 50.0)];
        let candidate = vec![create_test_result("matmul", 1024, 800.0, 40.0)];

        comparator.add_baseline(&baseline, "baseline");
        comparator.add_candidate(&candidate, "optimized");
        comparator.compare();

        let report = comparator.generate_markdown_report();
        assert!(report.contains("# Benchmark Comparison Report"));
        assert!(report.contains("## Summary"));
        assert!(report.contains("## Detailed Comparison"));
        assert!(report.contains("matmul_1024"));
    }

    #[test]
    fn test_json_export() {
        let mut comparator = BenchmarkComparator::new();

        let baseline = vec![create_test_result("matmul", 1024, 1000.0, 50.0)];
        let candidate = vec![create_test_result("matmul", 1024, 800.0, 40.0)];

        comparator.add_baseline(&baseline, "baseline");
        comparator.add_candidate(&candidate, "optimized");
        comparator.compare();

        let json = comparator.export_json().unwrap();
        assert!(json.contains("\"benchmark_name\""));
        assert!(json.contains("\"speedup\""));
        assert!(json.contains("\"verdict\""));
    }
}
