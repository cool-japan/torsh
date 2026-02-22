//! CI/CD integration for performance profiling
//!
//! This module provides integration with continuous integration and deployment
//! systems for automated performance monitoring and regression detection.

use crate::{
    regression::{RegressionDetector, RegressionResult, RegressionSeverity},
    ProfileEvent,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// CI/CD platform types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CiCdPlatform {
    GitHub,
    GitLab,
    Jenkins,
    CircleCI,
    TravisCI,
    AppVeyor,
    TeamCity,
    Bamboo,
    Azure,
    Custom(String),
}

impl std::fmt::Display for CiCdPlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CiCdPlatform::GitHub => write!(f, "GitHub Actions"),
            CiCdPlatform::GitLab => write!(f, "GitLab CI"),
            CiCdPlatform::Jenkins => write!(f, "Jenkins"),
            CiCdPlatform::CircleCI => write!(f, "CircleCI"),
            CiCdPlatform::TravisCI => write!(f, "Travis CI"),
            CiCdPlatform::AppVeyor => write!(f, "AppVeyor"),
            CiCdPlatform::TeamCity => write!(f, "TeamCity"),
            CiCdPlatform::Bamboo => write!(f, "Bamboo"),
            CiCdPlatform::Azure => write!(f, "Azure DevOps"),
            CiCdPlatform::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// CI/CD build information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub build_id: String,
    pub commit_hash: String,
    pub branch: String,
    pub pr_number: Option<u32>,
    pub author: String,
    pub timestamp: SystemTime,
    pub platform: CiCdPlatform,
    pub build_url: Option<String>,
    pub artifacts_url: Option<String>,
}

/// Performance benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration_ns: u64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<u64>,
    pub cpu_usage: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// CI/CD performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub build_info: BuildInfo,
    pub benchmarks: Vec<BenchmarkResult>,
    pub regression_analysis: Option<RegressionAnalysis>,
    pub summary: ReportSummary,
    pub recommendations: Vec<String>,
}

/// Regression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub regressions_found: Vec<RegressionResult>,
    pub improvements_found: Vec<RegressionResult>,
    pub overall_status: RegressionStatus,
}

/// Overall regression status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RegressionStatus {
    Pass,
    Warning,
    Fail,
}

/// Report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_benchmarks: usize,
    pub passed_benchmarks: usize,
    pub failed_benchmarks: usize,
    pub regression_count: usize,
    pub improvement_count: usize,
    pub overall_status: RegressionStatus,
}

/// CI/CD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiCdConfig {
    pub platform: CiCdPlatform,
    pub baseline_path: String,
    pub report_path: String,
    pub fail_on_regression: bool,
    pub regression_threshold: f64,
    pub improvement_threshold: f64,
    pub enable_comments: bool,
    pub comment_template: Option<String>,
    pub artifact_retention_days: u32,
}

impl Default for CiCdConfig {
    fn default() -> Self {
        Self {
            platform: CiCdPlatform::GitHub,
            baseline_path: "performance_baseline.json".to_string(),
            report_path: "performance_report.json".to_string(),
            fail_on_regression: true,
            regression_threshold: 0.05,  // 5%
            improvement_threshold: 0.05, // 5%
            enable_comments: true,
            comment_template: None,
            artifact_retention_days: 30,
        }
    }
}

/// CI/CD integration manager
pub struct CiCdIntegration {
    config: CiCdConfig,
    regression_detector: RegressionDetector,
}

impl CiCdIntegration {
    pub fn new(config: CiCdConfig) -> Self {
        let regression_config = crate::regression::RegressionConfig {
            min_baseline_samples: 10,
            max_baseline_age_days: 1,
            regression_threshold_percent: config.regression_threshold * 100.0,
            improvement_threshold_percent: config.improvement_threshold * 100.0,
            significance_level: 0.05,
            adaptive_thresholds: true,
            rolling_window_size: 100,
            outlier_detection: true,
        };
        let regression_detector = RegressionDetector::new(regression_config);

        Self {
            config,
            regression_detector,
        }
    }

    /// Detect build environment and create BuildInfo
    pub fn detect_build_info(&self) -> Result<BuildInfo> {
        let platform = self.config.platform.clone();

        match platform {
            CiCdPlatform::GitHub => self.detect_github_info(),
            CiCdPlatform::GitLab => self.detect_gitlab_info(),
            CiCdPlatform::Jenkins => self.detect_jenkins_info(),
            CiCdPlatform::CircleCI => self.detect_circleci_info(),
            CiCdPlatform::TravisCI => self.detect_travis_info(),
            CiCdPlatform::Azure => self.detect_azure_info(),
            _ => self.detect_generic_info(),
        }
    }

    fn detect_github_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("GITHUB_SHA").unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("GITHUB_REF_NAME").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("GITHUB_EVENT_NUMBER")
                .ok()
                .and_then(|n| n.parse().ok()),
            author: std::env::var("GITHUB_ACTOR").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: CiCdPlatform::GitHub,
            build_url: Some(format!(
                "https://github.com/{}/actions/runs/{}",
                std::env::var("GITHUB_REPOSITORY").unwrap_or_else(|_| "unknown".to_string()),
                std::env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "unknown".to_string())
            )),
            artifacts_url: None,
        })
    }

    fn detect_gitlab_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("CI_JOB_ID").unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("CI_COMMIT_SHA").unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("CI_COMMIT_REF_NAME").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("CI_MERGE_REQUEST_IID")
                .ok()
                .and_then(|n| n.parse().ok()),
            author: std::env::var("CI_COMMIT_AUTHOR").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: CiCdPlatform::GitLab,
            build_url: std::env::var("CI_JOB_URL").ok(),
            artifacts_url: None,
        })
    }

    fn detect_jenkins_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("BUILD_NUMBER").unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("GIT_BRANCH").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("CHANGE_ID").ok().and_then(|n| n.parse().ok()),
            author: std::env::var("BUILD_USER").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: CiCdPlatform::Jenkins,
            build_url: std::env::var("BUILD_URL").ok(),
            artifacts_url: None,
        })
    }

    fn detect_circleci_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("CIRCLE_BUILD_NUM").unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("CIRCLE_SHA1").unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("CIRCLE_BRANCH").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("CIRCLE_PR_NUMBER")
                .ok()
                .and_then(|n| n.parse().ok()),
            author: std::env::var("CIRCLE_USERNAME").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: CiCdPlatform::CircleCI,
            build_url: std::env::var("CIRCLE_BUILD_URL").ok(),
            artifacts_url: None,
        })
    }

    fn detect_travis_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("TRAVIS_BUILD_NUMBER")
                .unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("TRAVIS_COMMIT").unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("TRAVIS_BRANCH").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("TRAVIS_PULL_REQUEST")
                .ok()
                .and_then(|n| n.parse().ok()),
            author: std::env::var("TRAVIS_COMMIT_AUTHOR").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: CiCdPlatform::TravisCI,
            build_url: std::env::var("TRAVIS_BUILD_WEB_URL").ok(),
            artifacts_url: None,
        })
    }

    fn detect_azure_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("BUILD_BUILDID").unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("BUILD_SOURCEVERSION")
                .unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("BUILD_SOURCEBRANCH").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER")
                .ok()
                .and_then(|n| n.parse().ok()),
            author: std::env::var("BUILD_REQUESTEDFOR").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: CiCdPlatform::Azure,
            build_url: std::env::var("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI").ok(),
            artifacts_url: None,
        })
    }

    fn detect_generic_info(&self) -> Result<BuildInfo> {
        Ok(BuildInfo {
            build_id: std::env::var("BUILD_ID").unwrap_or_else(|_| "unknown".to_string()),
            commit_hash: std::env::var("COMMIT_HASH").unwrap_or_else(|_| "unknown".to_string()),
            branch: std::env::var("BRANCH").unwrap_or_else(|_| "unknown".to_string()),
            pr_number: std::env::var("PR_NUMBER").ok().and_then(|n| n.parse().ok()),
            author: std::env::var("AUTHOR").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: SystemTime::now(),
            platform: self.config.platform.clone(),
            build_url: std::env::var("BUILD_URL").ok(),
            artifacts_url: None,
        })
    }

    /// Generate performance report from profile events
    pub fn generate_report(&mut self, events: &[ProfileEvent]) -> Result<PerformanceReport> {
        let build_info = self.detect_build_info()?;
        let benchmarks = self.extract_benchmarks(events)?;
        let regression_analysis = self.analyze_regressions(&benchmarks)?;
        let summary = self.generate_summary(&benchmarks, &regression_analysis);
        let recommendations = self.generate_recommendations(&regression_analysis);

        Ok(PerformanceReport {
            build_info,
            benchmarks,
            regression_analysis: Some(regression_analysis),
            summary,
            recommendations,
        })
    }

    fn extract_benchmarks(&self, events: &[ProfileEvent]) -> Result<Vec<BenchmarkResult>> {
        let mut benchmarks = Vec::new();
        let mut benchmark_map: HashMap<String, Vec<&ProfileEvent>> = HashMap::new();

        // Group events by name
        for event in events {
            benchmark_map
                .entry(event.name.clone())
                .or_default()
                .push(event);
        }

        // Convert to benchmark results
        for (name, events) in benchmark_map {
            let total_duration: u64 = events.iter().map(|e| e.duration_us * 1000).sum(); // Convert to ns
            let avg_duration = total_duration / events.len() as u64;

            let throughput = events
                .iter()
                .map(|e| e.operation_count)
                .sum::<Option<u64>>()
                .map(|total_ops| total_ops as f64 / (total_duration as f64 / 1_000_000_000.0));

            let memory_usage = events.iter().map(|e| e.bytes_transferred).sum();

            benchmarks.push(BenchmarkResult {
                name,
                duration_ns: avg_duration,
                throughput,
                memory_usage,
                cpu_usage: None,
                custom_metrics: HashMap::new(),
            });
        }

        Ok(benchmarks)
    }

    fn analyze_regressions(
        &mut self,
        benchmarks: &[BenchmarkResult],
    ) -> Result<RegressionAnalysis> {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Load baseline if exists
        if Path::new(&self.config.baseline_path).exists() {
            let baseline_data = fs::read_to_string(&self.config.baseline_path)?;
            let baseline_benchmarks: Vec<BenchmarkResult> = serde_json::from_str(&baseline_data)?;

            for current in benchmarks {
                if let Some(baseline) = baseline_benchmarks.iter().find(|b| b.name == current.name)
                {
                    // Update baseline for the regression detector
                    self.regression_detector.update_baseline(
                        &current.name,
                        "benchmark",
                        vec![baseline.duration_ns as f64],
                    )?;

                    // Create a dummy event for regression detection
                    let dummy_event = crate::ProfileEvent {
                        name: current.name.clone(),
                        category: "benchmark".to_string(),
                        start_us: 0,
                        duration_us: (current.duration_ns / 1000),
                        thread_id: 0,
                        operation_count: None,
                        flops: None,
                        bytes_transferred: None,
                        stack_trace: None,
                    };

                    // Detect regressions using the available method
                    let results = self
                        .regression_detector
                        .detect_regressions(&[dummy_event])?;
                    for result in results {
                        if result.is_regression {
                            regressions.push(result);
                        } else if result.is_improvement {
                            improvements.push(result);
                        }
                    }
                }
            }
        }

        let overall_status = if regressions
            .iter()
            .any(|r| matches!(r.severity, RegressionSeverity::Critical))
        {
            RegressionStatus::Fail
        } else if !regressions.is_empty() {
            RegressionStatus::Warning
        } else {
            RegressionStatus::Pass
        };

        Ok(RegressionAnalysis {
            regressions_found: regressions,
            improvements_found: improvements,
            overall_status,
        })
    }

    fn generate_summary(
        &self,
        benchmarks: &[BenchmarkResult],
        regression_analysis: &RegressionAnalysis,
    ) -> ReportSummary {
        let total_benchmarks = benchmarks.len();
        let regression_count = regression_analysis.regressions_found.len();
        let improvement_count = regression_analysis.improvements_found.len();
        let failed_benchmarks = regression_analysis
            .regressions_found
            .iter()
            .filter(|r| matches!(r.severity, RegressionSeverity::Critical))
            .count();
        let passed_benchmarks = total_benchmarks - failed_benchmarks;

        ReportSummary {
            total_benchmarks,
            passed_benchmarks,
            failed_benchmarks,
            regression_count,
            improvement_count,
            overall_status: regression_analysis.overall_status.clone(),
        }
    }

    fn generate_recommendations(&self, regression_analysis: &RegressionAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !regression_analysis.regressions_found.is_empty() {
            recommendations.push(
                "Performance regressions detected. Consider optimizing the affected code paths."
                    .to_string(),
            );

            for regression in &regression_analysis.regressions_found {
                if regression.change_percent > 20.0 {
                    // Consider >20% as critical
                    recommendations.push(format!(
                        "CRITICAL: {} has significant performance regression",
                        regression.metric_name
                    ));
                }
            }
        }

        if !regression_analysis.improvements_found.is_empty() {
            recommendations.push("Performance improvements detected. Good work!".to_string());
        }

        if regression_analysis.overall_status == RegressionStatus::Pass {
            recommendations
                .push("All performance benchmarks are within acceptable thresholds.".to_string());
        }

        recommendations
    }

    /// Save performance report to file
    pub fn save_report(&self, report: &PerformanceReport) -> Result<()> {
        let json = serde_json::to_string_pretty(report)?;
        fs::write(&self.config.report_path, json)?;
        Ok(())
    }

    /// Update performance baseline
    pub fn update_baseline(&self, benchmarks: &[BenchmarkResult]) -> Result<()> {
        let json = serde_json::to_string_pretty(benchmarks)?;
        fs::write(&self.config.baseline_path, json)?;
        Ok(())
    }

    /// Generate comment for pull request
    pub fn generate_pr_comment(&self, report: &PerformanceReport) -> String {
        if let Some(template) = &self.config.comment_template {
            // Use custom template (simplified template engine)
            template
                .replace("{status}", &format!("{:?}", report.summary.overall_status))
                .replace("{total}", &report.summary.total_benchmarks.to_string())
                .replace(
                    "{regressions}",
                    &report.summary.regression_count.to_string(),
                )
                .replace(
                    "{improvements}",
                    &report.summary.improvement_count.to_string(),
                )
        } else {
            // Default comment template
            let mut comment = String::new();
            comment.push_str("## Performance Report\n\n");
            comment.push_str(&format!(
                "**Status**: {:?}\n",
                report.summary.overall_status
            ));
            comment.push_str(&format!(
                "**Total Benchmarks**: {}\n",
                report.summary.total_benchmarks
            ));
            comment.push_str(&format!(
                "**Regressions**: {}\n",
                report.summary.regression_count
            ));
            comment.push_str(&format!(
                "**Improvements**: {}\n",
                report.summary.improvement_count
            ));

            if let Some(analysis) = &report.regression_analysis {
                if !analysis.regressions_found.is_empty() {
                    comment.push_str("\n### Regressions Found\n");
                    for regression in &analysis.regressions_found {
                        comment.push_str(&format!(
                            "- **{}**: Regression ({:.2}% change)\n",
                            regression.metric_name, regression.change_percent
                        ));
                    }
                }

                if !analysis.improvements_found.is_empty() {
                    comment.push_str("\n### Improvements Found\n");
                    for improvement in &analysis.improvements_found {
                        comment.push_str(&format!(
                            "- **{}**: {:.2}% improvement\n",
                            improvement.metric_name,
                            improvement.change_percent.abs()
                        ));
                    }
                }
            }

            if !report.recommendations.is_empty() {
                comment.push_str("\n### Recommendations\n");
                for recommendation in &report.recommendations {
                    comment.push_str(&format!("- {recommendation}\n"));
                }
            }

            comment
        }
    }

    /// Check if build should fail based on regression analysis
    pub fn should_fail_build(&self, report: &PerformanceReport) -> bool {
        if !self.config.fail_on_regression {
            return false;
        }

        matches!(report.summary.overall_status, RegressionStatus::Fail)
    }
}

/// Convenience functions for CI/CD integration
/// Create CI/CD integration with default configuration
pub fn create_ci_cd_integration() -> CiCdIntegration {
    CiCdIntegration::new(CiCdConfig::default())
}

/// Create CI/CD integration with custom configuration
pub fn create_ci_cd_integration_with_config(config: CiCdConfig) -> CiCdIntegration {
    CiCdIntegration::new(config)
}

/// Generate and save performance report
pub fn generate_performance_report(
    events: &[ProfileEvent],
    config: Option<CiCdConfig>,
) -> Result<PerformanceReport> {
    let mut integration = if let Some(config) = config {
        CiCdIntegration::new(config)
    } else {
        create_ci_cd_integration()
    };

    let report = integration.generate_report(events)?;
    integration.save_report(&report)?;

    Ok(report)
}

/// Update performance baseline from current benchmarks
pub fn update_performance_baseline(
    events: &[ProfileEvent],
    config: Option<CiCdConfig>,
) -> Result<()> {
    let integration = if let Some(config) = config {
        CiCdIntegration::new(config)
    } else {
        create_ci_cd_integration()
    };

    let benchmarks = integration.extract_benchmarks(events)?;
    integration.update_baseline(&benchmarks)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_build_info_detection() {
        let config = CiCdConfig::default();
        let integration = CiCdIntegration::new(config);

        let build_info = integration.detect_build_info().unwrap();
        assert!(!build_info.build_id.is_empty());
        assert!(!build_info.commit_hash.is_empty());
        assert!(!build_info.branch.is_empty());
        assert!(!build_info.author.is_empty());
    }

    #[test]
    fn test_benchmark_extraction() {
        let config = CiCdConfig::default();
        let integration = CiCdIntegration::new(config);

        let events = vec![
            ProfileEvent {
                name: "test_operation".to_string(),
                category: "test".to_string(),
                start_us: 0,
                duration_us: 1,
                thread_id: 0,
                operation_count: Some(10),
                flops: Some(100),
                bytes_transferred: Some(1024),
                stack_trace: Some("test trace".to_string()),
            },
            ProfileEvent {
                name: "test_operation".to_string(),
                category: "test".to_string(),
                start_us: 0,
                duration_us: 2,
                thread_id: 0,
                operation_count: Some(20),
                flops: Some(200),
                bytes_transferred: Some(2048),
                stack_trace: Some("test trace".to_string()),
            },
        ];

        let benchmarks = integration.extract_benchmarks(&events).unwrap();
        assert_eq!(benchmarks.len(), 1);
        assert_eq!(benchmarks[0].name, "test_operation");
        assert_eq!(benchmarks[0].duration_ns, 1500); // average
    }

    #[test]
    fn test_report_generation() {
        let mut integration = CiCdIntegration::new(CiCdConfig::default());

        let events = vec![ProfileEvent {
            name: "benchmark_1".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(10),
            bytes_transferred: Some(100),
            stack_trace: Some("test trace".to_string()),
        }];

        let report = integration.generate_report(&events).unwrap();
        assert_eq!(report.benchmarks.len(), 1);
        assert_eq!(report.summary.total_benchmarks, 1);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_pr_comment_generation() {
        let integration = CiCdIntegration::new(CiCdConfig::default());

        let report = PerformanceReport {
            build_info: BuildInfo {
                build_id: "123".to_string(),
                commit_hash: "abc123".to_string(),
                branch: "main".to_string(),
                pr_number: Some(456),
                author: "test_author".to_string(),
                timestamp: SystemTime::now(),
                platform: CiCdPlatform::GitHub,
                build_url: None,
                artifacts_url: None,
            },
            benchmarks: vec![],
            regression_analysis: None,
            summary: ReportSummary {
                total_benchmarks: 5,
                passed_benchmarks: 4,
                failed_benchmarks: 1,
                regression_count: 1,
                improvement_count: 2,
                overall_status: RegressionStatus::Warning,
            },
            recommendations: vec!["Test recommendation".to_string()],
        };

        let comment = integration.generate_pr_comment(&report);
        assert!(comment.contains("Performance Report"));
        assert!(comment.contains("Warning"));
        assert!(comment.contains("**Total Benchmarks**: 5"));
        assert!(comment.contains("Test recommendation"));
    }

    #[test]
    fn test_platform_display() {
        assert_eq!(CiCdPlatform::GitHub.to_string(), "GitHub Actions");
        assert_eq!(CiCdPlatform::GitLab.to_string(), "GitLab CI");
        assert_eq!(
            CiCdPlatform::Custom("Custom CI".to_string()).to_string(),
            "Custom CI"
        );
    }

    #[test]
    fn test_config_serialization() {
        let config = CiCdConfig {
            platform: CiCdPlatform::GitHub,
            baseline_path: "test_baseline.json".to_string(),
            report_path: "test_report.json".to_string(),
            fail_on_regression: true,
            regression_threshold: 0.1,
            improvement_threshold: 0.05,
            enable_comments: true,
            comment_template: Some("Custom template".to_string()),
            artifact_retention_days: 7,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CiCdConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.platform, deserialized.platform);
        assert_eq!(config.baseline_path, deserialized.baseline_path);
        assert_eq!(
            config.regression_threshold,
            deserialized.regression_threshold
        );
    }

    #[test]
    fn test_build_failure_conditions() {
        let config = CiCdConfig {
            fail_on_regression: true,
            ..CiCdConfig::default()
        };
        let integration = CiCdIntegration::new(config);

        let fail_report = PerformanceReport {
            build_info: BuildInfo {
                build_id: "123".to_string(),
                commit_hash: "abc123".to_string(),
                branch: "main".to_string(),
                pr_number: None,
                author: "test".to_string(),
                timestamp: SystemTime::now(),
                platform: CiCdPlatform::GitHub,
                build_url: None,
                artifacts_url: None,
            },
            benchmarks: vec![],
            regression_analysis: None,
            summary: ReportSummary {
                total_benchmarks: 1,
                passed_benchmarks: 0,
                failed_benchmarks: 1,
                regression_count: 1,
                improvement_count: 0,
                overall_status: RegressionStatus::Fail,
            },
            recommendations: vec![],
        };

        assert!(integration.should_fail_build(&fail_report));

        let pass_report = PerformanceReport {
            build_info: fail_report.build_info.clone(),
            benchmarks: vec![],
            regression_analysis: None,
            summary: ReportSummary {
                total_benchmarks: 1,
                passed_benchmarks: 1,
                failed_benchmarks: 0,
                regression_count: 0,
                improvement_count: 0,
                overall_status: RegressionStatus::Pass,
            },
            recommendations: vec![],
        };

        assert!(!integration.should_fail_build(&pass_report));
    }
}
