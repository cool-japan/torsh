//! Performance Analysis and Bottleneck Detection
//!
//! This module provides comprehensive performance analysis capabilities for autograd
//! operations, including bottleneck detection, performance pattern analysis, and
//! optimization recommendations.
//!
//! # Features
//!
//! - **Bottleneck Detection**: Identify performance limiting factors
//! - **Timing Analysis**: Analyze operation timing patterns and variances
//! - **Memory Analysis**: Detect memory-related performance issues
//! - **Hardware Utilization Analysis**: Monitor resource usage efficiency
//! - **Pipeline Analysis**: Analyze forward/backward pass balance
//! - **Synchronization Analysis**: Detect synchronization overhead
//! - **Data Movement Analysis**: Identify data transfer bottlenecks
//!
//! # Bottleneck Categories
//!
//! The analyzer can detect various types of performance bottlenecks:
//! - **CPU Compute**: CPU-bound operations
//! - **GPU Compute**: GPU-bound operations
//! - **Memory Bandwidth**: Memory bandwidth limitations
//! - **Memory Allocation**: Memory allocation overhead
//! - **Graph Construction**: Computation graph building overhead
//! - **Gradient Computation**: Gradient calculation bottlenecks
//! - **Data Movement**: Data transfer between devices
//! - **Synchronization**: Synchronization and coordination overhead
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::profiler::analysis::{PerformanceAnalyzer, BottleneckThresholds};
//! use torsh_autograd::profiler::types::AutogradProfile;
//!
//! // Create analyzer with custom thresholds
//! let thresholds = BottleneckThresholds {
//!     cpu_compute_threshold: 85.0,
//!     memory_threshold: 95.0,
//!     slow_operation_threshold: 5.0,
//!     memory_allocation_threshold: 50 * 1024 * 1024, // 50MB
//! };
//!
//! let analyzer = PerformanceAnalyzer::with_thresholds(thresholds);
//!
//! // Analyze profile for bottlenecks
//! let bottlenecks = analyzer.analyze_bottlenecks(&profile);
//! for bottleneck in &bottlenecks {
//!     println!("Bottleneck: {} (severity: {:.2})",
//!              bottleneck.description, bottleneck.severity);
//!     println!("Suggestion: {}", bottleneck.suggestion);
//! }
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use super::types::{AutogradProfile, BottleneckType, OperationProfile, PerformanceBottleneck};
use std::time::Duration;

/// Performance analyzer for detecting bottlenecks and optimization opportunities
///
/// The PerformanceAnalyzer examines autograd profiles to identify performance
/// bottlenecks, inefficiencies, and opportunities for optimization.
///
/// # Analysis Categories
///
/// The analyzer performs multiple types of analysis:
/// - Timing analysis for slow operations
/// - Memory analysis for allocation patterns
/// - Hardware utilization analysis
/// - Pipeline balance analysis
/// - Synchronization overhead detection
/// - Data movement bottleneck detection
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Bottleneck detection thresholds
    thresholds: BottleneckThresholds,
    /// Analysis configuration
    config: AnalysisConfig,
    /// Historical analysis results for trend detection
    analysis_history: Vec<AnalysisResult>,
    /// Custom analysis rules
    custom_rules: Vec<CustomAnalysisRule>,
}

/// Thresholds for bottleneck detection
///
/// These thresholds define the boundaries for considering various metrics
/// as indicative of performance bottlenecks.
#[derive(Debug, Clone)]
pub struct BottleneckThresholds {
    /// CPU utilization threshold for compute bottleneck (percentage)
    pub cpu_compute_threshold: f32,
    /// Memory utilization threshold for memory bottleneck (percentage)
    pub memory_threshold: f32,
    /// Time threshold for slow operations (milliseconds)
    pub slow_operation_threshold: f32,
    /// Memory allocation threshold (bytes)
    pub memory_allocation_threshold: usize,
    /// GPU utilization threshold for underutilization detection (percentage)
    pub gpu_underutilization_threshold: f32,
    /// Cache hit rate threshold for cache performance issues (percentage)
    pub cache_hit_rate_threshold: f32,
    /// Time variance threshold for inconsistent operations (percentage)
    pub time_variance_threshold: f32,
    /// Memory growth rate threshold (bytes per second)
    pub memory_growth_rate_threshold: f64,
}

/// Configuration for performance analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable advanced bottleneck detection
    pub enable_advanced_analysis: bool,
    /// Enable trend analysis across multiple profiles
    pub enable_trend_analysis: bool,
    /// Maximum number of bottlenecks to report
    pub max_bottlenecks: usize,
    /// Minimum severity threshold for reporting
    pub min_severity_threshold: f32,
    /// Enable custom analysis rules
    pub enable_custom_rules: bool,
}

/// Result of performance analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Timestamp of analysis
    pub timestamp: std::time::Instant,
    /// Detected bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Overall performance score (0.0 to 1.0)
    pub performance_score: f32,
    /// Analysis summary
    pub summary: AnalysisSummary,
}

/// Summary of performance analysis
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    /// Total number of operations analyzed
    pub total_operations: usize,
    /// Number of bottlenecks detected
    pub bottleneck_count: usize,
    /// Most critical bottleneck type
    pub primary_bottleneck: Option<BottleneckType>,
    /// Performance recommendations
    pub recommendations: Vec<String>,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Efficiency metrics computed during analysis
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// CPU efficiency (0.0 to 1.0)
    pub cpu_efficiency: f32,
    /// Memory efficiency (0.0 to 1.0)
    pub memory_efficiency: f32,
    /// GPU efficiency (0.0 to 1.0, if available)
    pub gpu_efficiency: Option<f32>,
    /// Overall efficiency score (0.0 to 1.0)
    pub overall_efficiency: f32,
    /// Time distribution efficiency
    pub time_distribution_efficiency: f32,
}

/// Custom analysis rule for domain-specific bottleneck detection
#[derive(Debug, Clone)]
pub struct CustomAnalysisRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule function (simplified for this implementation)
    pub operation_pattern: String,
    /// Threshold value
    pub threshold: f64,
    /// Bottleneck type to report
    pub bottleneck_type: BottleneckType,
}

impl PerformanceAnalyzer {
    /// Creates a new performance analyzer with default thresholds
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let analyzer = PerformanceAnalyzer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            thresholds: BottleneckThresholds::default(),
            config: AnalysisConfig::default(),
            analysis_history: Vec::new(),
            custom_rules: Vec::new(),
        }
    }

    /// Creates a new performance analyzer with custom thresholds
    ///
    /// # Arguments
    ///
    /// * `thresholds` - Custom bottleneck detection thresholds
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let thresholds = BottleneckThresholds {
    ///     cpu_compute_threshold: 90.0,
    ///     memory_threshold: 95.0,
    ///     ..Default::default()
    /// };
    /// let analyzer = PerformanceAnalyzer::with_thresholds(thresholds);
    /// ```
    pub fn with_thresholds(thresholds: BottleneckThresholds) -> Self {
        Self {
            thresholds,
            config: AnalysisConfig::default(),
            analysis_history: Vec::new(),
            custom_rules: Vec::new(),
        }
    }

    /// Creates a new performance analyzer with full configuration
    ///
    /// # Arguments
    ///
    /// * `thresholds` - Bottleneck detection thresholds
    /// * `config` - Analysis configuration
    pub fn with_config(thresholds: BottleneckThresholds, config: AnalysisConfig) -> Self {
        Self {
            thresholds,
            config,
            analysis_history: Vec::new(),
            custom_rules: Vec::new(),
        }
    }

    /// Analyzes an autograd profile for performance bottlenecks
    ///
    /// This is the main analysis method that examines all aspects of the profile
    /// to identify performance issues and optimization opportunities.
    ///
    /// # Arguments
    ///
    /// * `profile` - The autograd profile to analyze
    ///
    /// # Returns
    ///
    /// Vector of detected performance bottlenecks, sorted by severity
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let bottlenecks = analyzer.analyze_bottlenecks(&profile);
    /// for bottleneck in &bottlenecks {
    ///     println!("Found {} bottleneck: {}",
    ///              bottleneck.bottleneck_type, bottleneck.description);
    /// }
    /// ```
    pub fn analyze_bottlenecks(&mut self, profile: &AutogradProfile) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Core bottleneck detection
        bottlenecks.extend(self.detect_timing_bottlenecks(profile));
        bottlenecks.extend(self.detect_memory_bottlenecks(profile));
        bottlenecks.extend(self.detect_hardware_bottlenecks(profile));

        // Advanced analysis (if enabled)
        if self.config.enable_advanced_analysis {
            bottlenecks.extend(self.detect_pipeline_bottlenecks(profile));
            bottlenecks.extend(self.detect_synchronization_bottlenecks(profile));
            bottlenecks.extend(self.detect_data_movement_bottlenecks(profile));
            bottlenecks.extend(self.detect_graph_construction_bottlenecks(profile));
        }

        // Custom rules analysis (if enabled)
        if self.config.enable_custom_rules {
            bottlenecks.extend(self.apply_custom_rules(profile));
        }

        // Filter by severity threshold
        bottlenecks.retain(|b| b.severity >= self.config.min_severity_threshold);

        // Sort by severity (highest first)
        bottlenecks.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit number of reported bottlenecks
        bottlenecks.truncate(self.config.max_bottlenecks);

        // Store analysis result for trend tracking
        if self.config.enable_trend_analysis {
            let result = AnalysisResult {
                timestamp: std::time::Instant::now(),
                bottlenecks: bottlenecks.clone(),
                performance_score: self.compute_performance_score(profile, &bottlenecks),
                summary: self.create_analysis_summary(profile, &bottlenecks),
            };
            self.analysis_history.push(result);

            // Limit history size
            if self.analysis_history.len() > 100 {
                self.analysis_history.remove(0);
            }
        }

        bottlenecks
    }

    /// Performs comprehensive performance analysis with detailed results
    ///
    /// # Arguments
    ///
    /// * `profile` - The autograd profile to analyze
    ///
    /// # Returns
    ///
    /// Detailed analysis result including bottlenecks, scores, and recommendations
    pub fn analyze_comprehensive(&mut self, profile: &AutogradProfile) -> AnalysisResult {
        let bottlenecks = self.analyze_bottlenecks(profile);
        let performance_score = self.compute_performance_score(profile, &bottlenecks);
        let summary = self.create_analysis_summary(profile, &bottlenecks);

        AnalysisResult {
            timestamp: std::time::Instant::now(),
            bottlenecks,
            performance_score,
            summary,
        }
    }

    /// Adds a custom analysis rule
    ///
    /// # Arguments
    ///
    /// * `rule` - The custom analysis rule to add
    pub fn add_custom_rule(&mut self, rule: CustomAnalysisRule) {
        self.custom_rules.push(rule);
    }

    /// Gets analysis history for trend tracking
    ///
    /// # Returns
    ///
    /// Slice of historical analysis results
    pub fn get_analysis_history(&self) -> &[AnalysisResult] {
        &self.analysis_history
    }

    /// Updates analysis configuration
    ///
    /// # Arguments
    ///
    /// * `config` - New analysis configuration
    pub fn set_config(&mut self, config: AnalysisConfig) {
        self.config = config;
    }

    /// Updates bottleneck detection thresholds
    ///
    /// # Arguments
    ///
    /// * `thresholds` - New bottleneck detection thresholds
    pub fn set_thresholds(&mut self, thresholds: BottleneckThresholds) {
        self.thresholds = thresholds;
    }

    /// Detects timing-related bottlenecks
    fn detect_timing_bottlenecks(&self, profile: &AutogradProfile) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        for (op_name, op_profile) in &profile.operation_profiles {
            // Slow operations
            if op_profile.average_time.as_millis() as f32 > self.thresholds.slow_operation_threshold
            {
                let severity = (op_profile.average_time.as_millis() as f32
                    / self.thresholds.slow_operation_threshold)
                    .min(1.0);

                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::CpuCompute,
                    operation: op_name.clone(),
                    severity,
                    description: format!(
                        "Operation {} is taking {:.2}ms on average",
                        op_name,
                        op_profile.average_time.as_millis()
                    ),
                    suggestion: "Consider optimizing this operation or using hardware acceleration"
                        .to_string(),
                    time_impact: op_profile.total_time,
                    memory_impact: op_profile.memory_allocated,
                });
            }

            // High timing variance
            let time_variance = self.calculate_time_variance(op_profile);
            if time_variance > self.thresholds.time_variance_threshold {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::CpuCompute,
                    operation: op_name.clone(),
                    severity: (time_variance / 100.0).min(1.0) as f32,
                    description: format!(
                        "Operation {} has inconsistent timing (variance: {:.1}%)",
                        op_name, time_variance
                    ),
                    suggestion: "Operation may be affected by system load or memory pressure"
                        .to_string(),
                    time_impact: op_profile.total_time,
                    memory_impact: op_profile.memory_allocated,
                });
            }
        }

        bottlenecks
    }

    /// Detects memory-related bottlenecks
    fn detect_memory_bottlenecks(&self, profile: &AutogradProfile) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Large memory allocations
        for (op_name, op_profile) in &profile.operation_profiles {
            if op_profile.memory_allocated > self.thresholds.memory_allocation_threshold {
                let severity = (op_profile.memory_allocated as f32
                    / self.thresholds.memory_allocation_threshold as f32)
                    .min(1.0);

                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::MemoryAllocation,
                    operation: op_name.clone(),
                    severity,
                    description: format!(
                        "Operation {} allocates {} MB of memory",
                        op_name,
                        op_profile.memory_allocated / 1024 / 1024
                    ),
                    suggestion: "Consider memory pooling or reducing memory usage".to_string(),
                    time_impact: Duration::ZERO,
                    memory_impact: op_profile.memory_allocated,
                });
            }
        }

        // Memory growth patterns
        if let Some(memory_growth_rate) = self.calculate_memory_growth_rate(profile) {
            if memory_growth_rate > self.thresholds.memory_growth_rate_threshold {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::MemoryAllocation,
                    operation: "Memory growth".to_string(),
                    severity: (memory_growth_rate / (50.0 * 1024.0 * 1024.0)).min(1.0) as f32,
                    description: format!(
                        "Memory is growing at {:.1} MB/s",
                        memory_growth_rate / 1024.0 / 1024.0
                    ),
                    suggestion: "Check for memory leaks or excessive memory allocation".to_string(),
                    time_impact: Duration::ZERO,
                    memory_impact: 0,
                });
            }
        }

        bottlenecks
    }

    /// Detects hardware utilization bottlenecks
    fn detect_hardware_bottlenecks(&self, profile: &AutogradProfile) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // GPU underutilization
        if let Some(gpu_utilization) = profile.hardware_utilization.gpu_utilization {
            if gpu_utilization < self.thresholds.gpu_underutilization_threshold {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::GpuCompute,
                    operation: "GPU utilization".to_string(),
                    severity: (self.thresholds.gpu_underutilization_threshold - gpu_utilization)
                        / 100.0,
                    description: format!("GPU utilization is low at {:.1}%", gpu_utilization),
                    suggestion:
                        "Consider batching operations or using more GPU-intensive operations"
                            .to_string(),
                    time_impact: Duration::ZERO,
                    memory_impact: 0,
                });
            }
        }

        // High memory utilization
        if profile.hardware_utilization.memory_utilization > self.thresholds.memory_threshold {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::MemoryBandwidth,
                operation: "Memory utilization".to_string(),
                severity: profile.hardware_utilization.memory_utilization / 100.0,
                description: format!(
                    "Memory utilization is high at {:.1}%",
                    profile.hardware_utilization.memory_utilization
                ),
                suggestion: "Consider reducing memory usage or using memory-efficient algorithms"
                    .to_string(),
                time_impact: Duration::ZERO,
                memory_impact: 0,
            });
        }

        // Low cache hit rate
        if profile.hardware_utilization.cache_hit_rate < self.thresholds.cache_hit_rate_threshold {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::MemoryBandwidth,
                operation: "Cache performance".to_string(),
                severity: (self.thresholds.cache_hit_rate_threshold
                    - profile.hardware_utilization.cache_hit_rate)
                    / 100.0,
                description: format!(
                    "Cache hit rate is low at {:.1}%",
                    profile.hardware_utilization.cache_hit_rate
                ),
                suggestion:
                    "Consider optimizing memory access patterns or reducing working set size"
                        .to_string(),
                time_impact: Duration::ZERO,
                memory_impact: 0,
            });
        }

        bottlenecks
    }

    /// Detects pipeline-related bottlenecks
    fn detect_pipeline_bottlenecks(&self, profile: &AutogradProfile) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Analyze forward vs backward pass balance
        let total_time = profile.summary.total_time.as_secs_f64();
        if total_time > 0.0 {
            let forward_ratio = profile.summary.forward_time.as_secs_f64() / total_time;
            let backward_ratio = profile.summary.backward_time.as_secs_f64() / total_time;

            if forward_ratio > 0.7 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::CpuCompute,
                    operation: "Forward pass".to_string(),
                    severity: forward_ratio.min(1.0) as f32,
                    description: format!(
                        "Forward pass dominates execution time ({:.1}%)",
                        forward_ratio * 100.0
                    ),
                    suggestion: "Consider optimizing forward pass operations".to_string(),
                    time_impact: profile.summary.forward_time,
                    memory_impact: 0,
                });
            }

            if backward_ratio > 0.7 {
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::GradientComputation,
                    operation: "Backward pass".to_string(),
                    severity: backward_ratio.min(1.0) as f32,
                    description: format!(
                        "Backward pass dominates execution time ({:.1}%)",
                        backward_ratio * 100.0
                    ),
                    suggestion:
                        "Consider gradient checkpointing or optimizing gradient computation"
                            .to_string(),
                    time_impact: profile.summary.backward_time,
                    memory_impact: 0,
                });
            }
        }

        bottlenecks
    }

    /// Detects synchronization bottlenecks
    fn detect_synchronization_bottlenecks(
        &self,
        profile: &AutogradProfile,
    ) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        for (op_name, op_profile) in &profile.operation_profiles {
            if op_name.contains("sync") || op_name.contains("barrier") || op_name.contains("wait") {
                if op_profile.average_time.as_millis() > 5 {
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_type: BottleneckType::Synchronization,
                        operation: op_name.clone(),
                        severity: (op_profile.average_time.as_millis() as f32 / 100.0).min(1.0),
                        description: format!(
                            "Synchronization operation {} taking {:.2}ms",
                            op_name,
                            op_profile.average_time.as_millis()
                        ),
                        suggestion: "Consider reducing synchronization points or using asynchronous operations".to_string(),
                        time_impact: op_profile.total_time,
                        memory_impact: 0,
                    });
                }
            }
        }

        bottlenecks
    }

    /// Detects data movement bottlenecks
    fn detect_data_movement_bottlenecks(
        &self,
        profile: &AutogradProfile,
    ) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        for (op_name, op_profile) in &profile.operation_profiles {
            if op_name.contains("copy")
                || op_name.contains("transfer")
                || op_name.contains("device")
            {
                if op_profile.average_time.as_millis() > 2 {
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_type: BottleneckType::DataMovement,
                        operation: op_name.clone(),
                        severity: (op_profile.average_time.as_millis() as f32 / 50.0).min(1.0),
                        description: format!(
                            "Data movement operation {} taking {:.2}ms",
                            op_name,
                            op_profile.average_time.as_millis()
                        ),
                        suggestion: "Consider reducing data transfers or improving data locality"
                            .to_string(),
                        time_impact: op_profile.total_time,
                        memory_impact: 0,
                    });
                }
            }
        }

        bottlenecks
    }

    /// Detects graph construction bottlenecks
    fn detect_graph_construction_bottlenecks(
        &self,
        profile: &AutogradProfile,
    ) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        for (op_name, op_profile) in &profile.operation_profiles {
            if op_name.contains("graph")
                || op_name.contains("build")
                || op_name.contains("construct")
            {
                if op_profile.average_time.as_millis() > 1 {
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_type: BottleneckType::GraphConstruction,
                        operation: op_name.clone(),
                        severity: (op_profile.average_time.as_millis() as f32 / 20.0).min(1.0),
                        description: format!(
                            "Graph construction operation {} taking {:.2}ms",
                            op_name,
                            op_profile.average_time.as_millis()
                        ),
                        suggestion: "Consider graph caching or reducing graph complexity"
                            .to_string(),
                        time_impact: op_profile.total_time,
                        memory_impact: 0,
                    });
                }
            }
        }

        bottlenecks
    }

    /// Applies custom analysis rules
    fn apply_custom_rules(&self, profile: &AutogradProfile) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        for rule in &self.custom_rules {
            for (op_name, op_profile) in &profile.operation_profiles {
                if op_name.contains(&rule.operation_pattern) {
                    let metric_value = match rule.bottleneck_type {
                        BottleneckType::CpuCompute => op_profile.average_time.as_millis() as f64,
                        BottleneckType::MemoryAllocation => op_profile.memory_allocated as f64,
                        _ => op_profile.average_time.as_millis() as f64,
                    };

                    if metric_value > rule.threshold {
                        bottlenecks.push(PerformanceBottleneck {
                            bottleneck_type: rule.bottleneck_type.clone(),
                            operation: op_name.clone(),
                            severity: (metric_value / rule.threshold).min(1.0) as f32,
                            description: format!(
                                "{}: {} ({})",
                                rule.name, rule.description, op_name
                            ),
                            suggestion: format!("Custom rule triggered: {}", rule.description),
                            time_impact: op_profile.total_time,
                            memory_impact: op_profile.memory_allocated,
                        });
                    }
                }
            }
        }

        bottlenecks
    }

    /// Calculates time variance for an operation
    fn calculate_time_variance(&self, op_profile: &OperationProfile) -> f32 {
        if op_profile.execution_count < 2 {
            return 0.0;
        }

        let mean = op_profile.average_time.as_millis() as f32;
        let min = op_profile.min_time.as_millis() as f32;
        let max = op_profile.max_time.as_millis() as f32;

        if mean > 0.0 {
            ((max - min) / mean) * 100.0
        } else {
            0.0
        }
    }

    /// Calculates memory growth rate from profile
    fn calculate_memory_growth_rate(&self, profile: &AutogradProfile) -> Option<f64> {
        if profile.memory_timeline.len() < 2 {
            return None;
        }

        let first = &profile.memory_timeline[0];
        let last = &profile.memory_timeline[profile.memory_timeline.len() - 1];

        let time_diff = last
            .timestamp
            .duration_since(first.timestamp)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64();

        let memory_diff = last.total_memory as f64 - first.total_memory as f64;

        Some(memory_diff / time_diff)
    }

    /// Computes overall performance score
    fn compute_performance_score(
        &self,
        _profile: &AutogradProfile,
        bottlenecks: &[PerformanceBottleneck],
    ) -> f32 {
        if bottlenecks.is_empty() {
            return 1.0;
        }

        let total_severity: f32 = bottlenecks.iter().map(|b| b.severity).sum();
        let average_severity = total_severity / bottlenecks.len() as f32;

        (1.0 - average_severity).max(0.0)
    }

    /// Creates analysis summary
    fn create_analysis_summary(
        &self,
        profile: &AutogradProfile,
        bottlenecks: &[PerformanceBottleneck],
    ) -> AnalysisSummary {
        let primary_bottleneck = bottlenecks.first().map(|b| b.bottleneck_type.clone());

        let mut recommendations = Vec::new();
        if bottlenecks.len() > 5 {
            recommendations.push("Multiple performance issues detected. Prioritize fixing high-severity bottlenecks.".to_string());
        }
        if profile.hardware_utilization.gpu_utilization.is_some() {
            recommendations
                .push("Consider GPU acceleration for compute-intensive operations.".to_string());
        }

        let efficiency_metrics = self.compute_efficiency_metrics(profile, bottlenecks);

        AnalysisSummary {
            total_operations: profile.operation_profiles.len(),
            bottleneck_count: bottlenecks.len(),
            primary_bottleneck,
            recommendations,
            efficiency_metrics,
        }
    }

    /// Computes efficiency metrics
    fn compute_efficiency_metrics(
        &self,
        profile: &AutogradProfile,
        bottlenecks: &[PerformanceBottleneck],
    ) -> EfficiencyMetrics {
        let cpu_efficiency =
            (100.0 - profile.hardware_utilization.cpu_utilization).max(0.0) / 100.0;
        let memory_efficiency =
            (100.0 - profile.hardware_utilization.memory_utilization).max(0.0) / 100.0;
        let gpu_efficiency = profile
            .hardware_utilization
            .gpu_utilization
            .map(|gpu| (100.0 - gpu).max(0.0) / 100.0);

        let overall_efficiency = if bottlenecks.is_empty() {
            1.0
        } else {
            let avg_severity =
                bottlenecks.iter().map(|b| b.severity).sum::<f32>() / bottlenecks.len() as f32;
            (1.0 - avg_severity).max(0.0)
        };

        // Time distribution efficiency based on forward/backward balance
        let total_time = profile.summary.total_time.as_secs_f64();
        let time_distribution_efficiency = if total_time > 0.0 {
            let forward_ratio = profile.summary.forward_time.as_secs_f64() / total_time;
            let backward_ratio = profile.summary.backward_time.as_secs_f64() / total_time;
            let balance = 1.0 - (forward_ratio - backward_ratio).abs();
            balance as f32
        } else {
            1.0
        };

        EfficiencyMetrics {
            cpu_efficiency,
            memory_efficiency,
            gpu_efficiency,
            overall_efficiency,
            time_distribution_efficiency,
        }
    }
}

impl Default for BottleneckThresholds {
    fn default() -> Self {
        Self {
            cpu_compute_threshold: 80.0,
            memory_threshold: 90.0,
            slow_operation_threshold: 10.0,
            memory_allocation_threshold: 100 * 1024 * 1024, // 100MB
            gpu_underutilization_threshold: 30.0,
            cache_hit_rate_threshold: 80.0,
            time_variance_threshold: 50.0,
            memory_growth_rate_threshold: 10.0 * 1024.0 * 1024.0, // 10MB/s
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_advanced_analysis: true,
            enable_trend_analysis: true,
            max_bottlenecks: 20,
            min_severity_threshold: 0.1,
            enable_custom_rules: false,
        }
    }
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::types::{
        AutogradProfile, HardwareUtilization, OperationProfile, ProfileSummary,
    };
    use std::time::Duration;

    fn create_test_profile() -> AutogradProfile {
        let mut profile = AutogradProfile::new("test_session".to_string());

        // Add some operation profiles
        let mut op1 = OperationProfile::new("slow_op".to_string());
        op1.update_execution(Duration::from_millis(100), 1024 * 1024, 1000.0);
        profile.add_operation_profile(op1);

        let mut op2 = OperationProfile::new("fast_op".to_string());
        op2.update_execution(Duration::from_millis(5), 1024, 100.0);
        profile.add_operation_profile(op2);

        profile.hardware_utilization = HardwareUtilization {
            cpu_utilization: 85.0,
            gpu_utilization: Some(25.0),
            memory_utilization: 75.0,
            memory_bandwidth_utilization: 60.0,
            cache_hit_rate: 75.0,
        };

        profile.summary = ProfileSummary {
            forward_time: Duration::from_millis(80),
            backward_time: Duration::from_millis(20),
            total_time: Duration::from_millis(100),
            ..Default::default()
        };

        profile
    }

    #[test]
    fn test_performance_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert_eq!(analyzer.thresholds.cpu_compute_threshold, 80.0);
        assert_eq!(analyzer.config.max_bottlenecks, 20);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut analyzer = PerformanceAnalyzer::new();
        let profile = create_test_profile();

        let bottlenecks = analyzer.analyze_bottlenecks(&profile);
        assert!(!bottlenecks.is_empty());

        // Should detect slow operation
        assert!(bottlenecks.iter().any(|b| b.operation == "slow_op"));
    }

    #[test]
    fn test_custom_thresholds() {
        let thresholds = BottleneckThresholds {
            slow_operation_threshold: 50.0, // Lower threshold
            ..Default::default()
        };
        let mut analyzer = PerformanceAnalyzer::with_thresholds(thresholds);
        let profile = create_test_profile();

        let bottlenecks = analyzer.analyze_bottlenecks(&profile);

        // With lower threshold, should detect the 100ms operation
        assert!(bottlenecks
            .iter()
            .any(|b| b.operation == "slow_op" && b.bottleneck_type == BottleneckType::CpuCompute));
    }

    #[test]
    fn test_comprehensive_analysis() {
        let mut analyzer = PerformanceAnalyzer::new();
        let profile = create_test_profile();

        let result = analyzer.analyze_comprehensive(&profile);
        assert!(result.performance_score >= 0.0 && result.performance_score <= 1.0);
        assert_eq!(result.summary.total_operations, 2);
        assert!(!result.summary.recommendations.is_empty());
    }

    #[test]
    fn test_custom_rules() {
        let mut analyzer = PerformanceAnalyzer::new();
        analyzer.config.enable_custom_rules = true;

        let rule = CustomAnalysisRule {
            name: "Test Rule".to_string(),
            description: "Test custom rule".to_string(),
            operation_pattern: "slow".to_string(),
            threshold: 50.0,
            bottleneck_type: BottleneckType::CpuCompute,
        };
        analyzer.add_custom_rule(rule);

        let profile = create_test_profile();
        let bottlenecks = analyzer.analyze_bottlenecks(&profile);

        assert!(bottlenecks
            .iter()
            .any(|b| b.description.contains("Test Rule")));
    }

    #[test]
    fn test_time_variance_calculation() {
        let analyzer = PerformanceAnalyzer::new();
        let mut op_profile = OperationProfile::new("test_op".to_string());

        op_profile.min_time = Duration::from_millis(10);
        op_profile.max_time = Duration::from_millis(30);
        op_profile.average_time = Duration::from_millis(20);
        op_profile.execution_count = 5;

        let variance = analyzer.calculate_time_variance(&op_profile);
        assert!(variance > 0.0);
        assert!(variance <= 100.0);
    }

    #[test]
    fn test_efficiency_metrics() {
        let mut analyzer = PerformanceAnalyzer::new();
        let profile = create_test_profile();
        let bottlenecks = analyzer.analyze_bottlenecks(&profile);

        let metrics = analyzer.compute_efficiency_metrics(&profile, &bottlenecks);
        assert!(metrics.cpu_efficiency >= 0.0 && metrics.cpu_efficiency <= 1.0);
        assert!(metrics.memory_efficiency >= 0.0 && metrics.memory_efficiency <= 1.0);
        assert!(metrics.overall_efficiency >= 0.0 && metrics.overall_efficiency <= 1.0);
    }

    #[test]
    fn test_trend_analysis_storage() {
        let config = AnalysisConfig {
            enable_trend_analysis: true,
            ..Default::default()
        };
        let mut analyzer = PerformanceAnalyzer::with_config(Default::default(), config);
        let profile = create_test_profile();

        analyzer.analyze_bottlenecks(&profile);
        assert_eq!(analyzer.get_analysis_history().len(), 1);

        analyzer.analyze_bottlenecks(&profile);
        assert_eq!(analyzer.get_analysis_history().len(), 2);
    }

    #[test]
    fn test_severity_filtering() {
        let config = AnalysisConfig {
            min_severity_threshold: 0.8, // High threshold
            ..Default::default()
        };
        let mut analyzer = PerformanceAnalyzer::with_config(Default::default(), config);
        let profile = create_test_profile();

        let bottlenecks = analyzer.analyze_bottlenecks(&profile);

        // All returned bottlenecks should have high severity
        for bottleneck in &bottlenecks {
            assert!(bottleneck.severity >= 0.8);
        }
    }
}
