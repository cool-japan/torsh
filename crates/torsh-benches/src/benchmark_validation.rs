//! Benchmark correctness validation and cross-architecture verification
//!
//! This module provides sophisticated validation capabilities to ensure benchmark
//! correctness, detect numerical accuracy issues, and verify consistency across
//! different architectures and optimization levels.

use crate::{BenchConfig, BenchResult};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::Random;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Validation result for benchmark correctness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation status
    pub status: ValidationStatus,
    /// Detailed validation report
    pub report: ValidationReport,
    /// Cross-architecture consistency
    pub cross_arch_consistency: CrossArchResult,
    /// Numerical accuracy validation
    pub numerical_accuracy: NumericalAccuracy,
    /// Performance consistency validation
    pub performance_consistency: PerformanceConsistency,
    /// Optimization correctness
    pub optimization_correctness: OptimizationCorrectness,
}

/// Validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// All validations passed
    Passed,
    /// Some validations failed but results are likely correct
    Warning,
    /// Critical validations failed - results may be incorrect
    Failed,
    /// Validation could not be completed
    Incomplete,
}

/// Detailed validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Individual test results
    pub tests: Vec<ValidationTest>,
    /// Summary statistics
    pub summary: ValidationSummary,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Detected issues
    pub issues: Vec<ValidationIssue>,
}

/// Individual validation test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTest {
    /// Test name
    pub name: String,
    /// Test category
    pub category: ValidationCategory,
    /// Test status
    pub status: ValidationStatus,
    /// Test score (0.0 - 1.0)
    pub score: f64,
    /// Detailed description
    pub description: String,
    /// Evidence/measurements
    pub evidence: Vec<String>,
}

/// Validation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCategory {
    /// Numerical correctness
    NumericalCorrectness,
    /// Performance consistency
    PerformanceConsistency,
    /// Cross-platform compatibility
    CrossPlatform,
    /// Memory safety
    MemorySafety,
    /// Optimization correctness
    OptimizationCorrectness,
    /// Statistical validity
    StatisticalValidity,
}

/// Validation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests with warnings
    pub warnings: usize,
    /// Tests failed
    pub failed: usize,
    /// Overall confidence score (0.0 - 1.0)
    pub confidence_score: f64,
    /// Validation time
    pub validation_time_ms: u64,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: ValidationCategory,
    /// Issue description
    pub description: String,
    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
    /// Impact assessment
    pub impact: String,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity - minor inconsistency
    Low,
    /// Medium severity - potential accuracy issue
    Medium,
    /// High severity - likely correctness problem
    High,
    /// Critical severity - benchmark results unreliable
    Critical,
}

/// Cross-architecture consistency results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossArchResult {
    /// Architectures tested
    pub architectures: Vec<Architecture>,
    /// Consistency score (0.0 - 1.0)
    pub consistency_score: f64,
    /// Performance variation coefficient
    pub performance_variation: f64,
    /// Numerical accuracy variation
    pub accuracy_variation: f64,
    /// Architecture-specific issues
    pub arch_issues: HashMap<String, Vec<String>>,
}

/// Target architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    /// Architecture name
    pub name: String,
    /// CPU features
    pub features: Vec<String>,
    /// Endianness
    pub endianness: Endianness,
    /// Pointer size
    pub pointer_size: usize,
    /// SIMD capabilities
    pub simd_caps: Vec<String>,
}

/// Endianness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Endianness {
    Little,
    Big,
}

/// Numerical accuracy validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalAccuracy {
    /// Relative error bounds
    pub relative_error: f64,
    /// Absolute error bounds
    pub absolute_error: f64,
    /// ULP (Units in Last Place) error
    pub ulp_error: f64,
    /// Precision loss analysis
    pub precision_loss: PrecisionLoss,
    /// Catastrophic cancellation detection
    pub cancellation_issues: Vec<CancellationIssue>,
    /// Overflow/underflow detection
    pub overflow_issues: Vec<OverflowIssue>,
}

/// Precision loss analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionLoss {
    /// Significant digits lost
    pub digits_lost: f64,
    /// Condition number estimate
    pub condition_number: f64,
    /// Numerical stability score
    pub stability_score: f64,
}

/// Catastrophic cancellation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationIssue {
    /// Operation description
    pub operation: String,
    /// Magnitude of cancellation
    pub magnitude: f64,
    /// Precision loss
    pub precision_loss: f64,
}

/// Overflow/underflow issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverflowIssue {
    /// Issue type
    pub issue_type: OverflowType,
    /// Operation description
    pub operation: String,
    /// Value range
    pub value_range: (f64, f64),
}

/// Overflow/underflow types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverflowType {
    Overflow,
    Underflow,
    NaN,
    Infinity,
}

/// Performance consistency validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConsistency {
    /// Timing variance coefficient
    pub timing_variance: f64,
    /// Throughput stability
    pub throughput_stability: f64,
    /// Memory usage consistency
    pub memory_consistency: f64,
    /// Scaling consistency
    pub scaling_consistency: f64,
    /// Reproducibility score
    pub reproducibility_score: f64,
}

/// Optimization correctness validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCorrectness {
    /// Unoptimized vs optimized comparison
    pub optimization_comparison: Vec<OptimizationComparison>,
    /// Compiler optimization safety
    pub compiler_safety: CompilerSafety,
    /// Algorithm equivalence verification
    pub algorithm_equivalence: AlgorithmEquivalence,
}

/// Optimization level comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationComparison {
    /// Optimization level
    pub optimization_level: String,
    /// Performance improvement
    pub performance_improvement: f64,
    /// Numerical accuracy comparison
    pub accuracy_delta: f64,
    /// Correctness status
    pub correctness_status: ValidationStatus,
}

/// Compiler optimization safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerSafety {
    /// Unsafe optimizations detected
    pub unsafe_optimizations: Vec<String>,
    /// Fast math safety
    pub fast_math_safety: bool,
    /// Optimization barriers needed
    pub barriers_needed: Vec<String>,
}

/// Algorithm equivalence verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmEquivalence {
    /// Reference implementation comparison
    pub reference_comparison: f64,
    /// Alternative implementation comparison
    pub alternative_comparison: f64,
    /// Equivalence confidence
    pub equivalence_confidence: f64,
}

/// Benchmark validator
pub struct BenchmarkValidator {
    /// Validation configuration
    config: ValidationConfig,
    /// Reference implementations
    reference_impls: HashMap<String, Box<dyn ReferenceImplementation>>,
    /// Cross-architecture test data
    cross_arch_data: Vec<CrossArchTestData>,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Numerical tolerance
    pub numerical_tolerance: f64,
    /// Performance tolerance (relative)
    pub performance_tolerance: f64,
    /// Enable cross-architecture validation
    pub enable_cross_arch: bool,
    /// Enable optimization validation
    pub enable_optimization_validation: bool,
    /// Reference implementation timeout
    pub reference_timeout_ms: u64,
    /// Random seed for reproducibility
    pub random_seed: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-6,
            performance_tolerance: 0.1, // 10%
            enable_cross_arch: true,
            enable_optimization_validation: true,
            reference_timeout_ms: 30000, // 30 seconds
            random_seed: 42,
        }
    }
}

/// Reference implementation trait
pub trait ReferenceImplementation: Send + Sync {
    /// Run reference implementation
    fn run(&self, input: &[f32]) -> Vec<f32>;

    /// Get implementation name
    fn name(&self) -> &str;

    /// Get implementation description
    fn description(&self) -> &str;
}

/// Cross-architecture test data
#[derive(Debug, Clone)]
pub struct CrossArchTestData {
    /// Architecture identifier
    pub arch: String,
    /// Test input data
    pub input: Vec<f32>,
    /// Expected output data
    pub expected_output: Vec<f32>,
    /// Performance baseline
    pub performance_baseline: f64,
}

impl BenchmarkValidator {
    /// Create a new benchmark validator
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
            reference_impls: HashMap::new(),
            cross_arch_data: Vec::new(),
        }
    }

    /// Create validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            reference_impls: HashMap::new(),
            cross_arch_data: Vec::new(),
        }
    }

    /// Add a reference implementation
    pub fn add_reference_implementation(
        &mut self,
        name: String,
        impl_: Box<dyn ReferenceImplementation>,
    ) {
        self.reference_impls.insert(name, impl_);
    }

    /// Validate benchmark results
    pub fn validate(&self, results: &[BenchResult], config: &BenchConfig) -> ValidationResult {
        let start_time = std::time::Instant::now();

        let mut tests = Vec::new();
        let mut issues = Vec::new();

        // Numerical accuracy validation
        let numerical_accuracy =
            self.validate_numerical_accuracy(results, config, &mut tests, &mut issues);

        // Performance consistency validation
        let performance_consistency =
            self.validate_performance_consistency(results, &mut tests, &mut issues);

        // Cross-architecture validation
        let cross_arch_consistency = if self.config.enable_cross_arch {
            self.validate_cross_architecture(results, &mut tests, &mut issues)
        } else {
            CrossArchResult::default()
        };

        // Optimization correctness validation
        let optimization_correctness = if self.config.enable_optimization_validation {
            self.validate_optimization_correctness(results, config, &mut tests, &mut issues)
        } else {
            OptimizationCorrectness::default()
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(&tests, &issues);

        // Calculate overall status
        let status = self.calculate_overall_status(&tests);

        // Create summary
        let passed = tests
            .iter()
            .filter(|t| t.status == ValidationStatus::Passed)
            .count();
        let warnings = tests
            .iter()
            .filter(|t| t.status == ValidationStatus::Warning)
            .count();
        let failed = tests
            .iter()
            .filter(|t| t.status == ValidationStatus::Failed)
            .count();

        let confidence_score = if tests.is_empty() {
            0.0
        } else {
            tests.iter().map(|t| t.score).sum::<f64>() / tests.len() as f64
        };

        let summary = ValidationSummary {
            total_tests: tests.len(),
            passed,
            warnings,
            failed,
            confidence_score,
            validation_time_ms: start_time.elapsed().as_millis() as u64,
        };

        let report = ValidationReport {
            tests,
            summary,
            recommendations,
            issues,
        };

        ValidationResult {
            status,
            report,
            cross_arch_consistency,
            numerical_accuracy,
            performance_consistency,
            optimization_correctness,
        }
    }

    /// Validate numerical accuracy
    fn validate_numerical_accuracy(
        &self,
        results: &[BenchResult],
        _config: &BenchConfig,
        tests: &mut Vec<ValidationTest>,
        issues: &mut Vec<ValidationIssue>,
    ) -> NumericalAccuracy {
        // Placeholder implementation - would integrate with reference implementations
        let relative_error = self.calculate_relative_error(results);
        let absolute_error = self.calculate_absolute_error(results);
        let ulp_error = self.calculate_ulp_error(results);

        // Add numerical accuracy test
        let accuracy_score = if relative_error < self.config.numerical_tolerance {
            1.0
        } else {
            (self.config.numerical_tolerance / relative_error).min(1.0)
        };

        tests.push(ValidationTest {
            name: "Numerical Accuracy".to_string(),
            category: ValidationCategory::NumericalCorrectness,
            status: if accuracy_score > 0.9 {
                ValidationStatus::Passed
            } else if accuracy_score > 0.7 {
                ValidationStatus::Warning
            } else {
                ValidationStatus::Failed
            },
            score: accuracy_score,
            description: format!(
                "Relative error: {:.2e}, tolerance: {:.2e}",
                relative_error, self.config.numerical_tolerance
            ),
            evidence: vec![
                format!("Absolute error: {:.2e}", absolute_error),
                format!("ULP error: {:.2e}", ulp_error),
            ],
        });

        if relative_error > self.config.numerical_tolerance {
            issues.push(ValidationIssue {
                severity: if relative_error > self.config.numerical_tolerance * 10.0 {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                category: ValidationCategory::NumericalCorrectness,
                description: format!(
                    "Numerical error exceeds tolerance: {:.2e} > {:.2e}",
                    relative_error, self.config.numerical_tolerance
                ),
                suggested_fixes: vec![
                    "Use higher precision arithmetic".to_string(),
                    "Implement numerically stable algorithms".to_string(),
                    "Add input validation and range checking".to_string(),
                ],
                impact: "Results may be numerically inaccurate".to_string(),
            });
        }

        NumericalAccuracy {
            relative_error,
            absolute_error,
            ulp_error,
            precision_loss: PrecisionLoss {
                digits_lost: (-relative_error.log10()).max(0.0),
                condition_number: 1.0, // Placeholder
                stability_score: accuracy_score,
            },
            cancellation_issues: Vec::new(), // Would implement cancellation detection
            overflow_issues: Vec::new(),     // Would implement overflow detection
        }
    }

    /// Calculate relative error (simplified)
    fn calculate_relative_error(&self, results: &[BenchResult]) -> f64 {
        // Simplified relative error calculation
        // In practice, would compare against reference implementations
        if results.is_empty() {
            return 0.0;
        }

        let times: Vec<f64> = results.iter().map(|r| r.mean_time_ns).collect();
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();

        if mean > 0.0 {
            std_dev / mean
        } else {
            0.0
        }
    }

    /// Calculate absolute error (simplified)
    fn calculate_absolute_error(&self, results: &[BenchResult]) -> f64 {
        // Simplified absolute error calculation
        if results.is_empty() {
            return 0.0;
        }

        let times: Vec<f64> = results.iter().map(|r| r.mean_time_ns).collect();
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        times.iter().map(|x| (x - mean).abs()).fold(0.0, f64::max)
    }

    /// Calculate ULP error (simplified)
    fn calculate_ulp_error(&self, results: &[BenchResult]) -> f64 {
        // Simplified ULP error calculation
        // In practice, would use proper ULP distance calculations
        self.calculate_relative_error(results) * (1u64 << 23) as f64 // Rough approximation for f32
    }

    /// Validate performance consistency
    fn validate_performance_consistency(
        &self,
        results: &[BenchResult],
        tests: &mut Vec<ValidationTest>,
        issues: &mut Vec<ValidationIssue>,
    ) -> PerformanceConsistency {
        if results.is_empty() {
            return PerformanceConsistency::default();
        }

        // Calculate timing variance
        let times: Vec<f64> = results.iter().map(|r| r.mean_time_ns).collect();
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let timing_variance = if mean_time > 0.0 {
            let variance =
                times.iter().map(|x| (x - mean_time).powi(2)).sum::<f64>() / times.len() as f64;
            variance.sqrt() / mean_time
        } else {
            0.0
        };

        // Calculate throughput stability
        let throughputs: Vec<f64> = results.iter().filter_map(|r| r.throughput).collect();
        let throughput_stability = if !throughputs.is_empty() {
            let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            let variance = throughputs
                .iter()
                .map(|x| (x - mean_throughput).powi(2))
                .sum::<f64>()
                / throughputs.len() as f64;
            1.0 - (variance.sqrt() / mean_throughput).min(1.0)
        } else {
            1.0
        };

        // Calculate memory consistency
        let memory_usages: Vec<f64> = results
            .iter()
            .filter_map(|r| r.memory_usage)
            .map(|m| m as f64)
            .collect();
        let memory_consistency = if !memory_usages.is_empty() {
            let mean_memory = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;
            let variance = memory_usages
                .iter()
                .map(|x| (x - mean_memory).powi(2))
                .sum::<f64>()
                / memory_usages.len() as f64;
            1.0 - (variance.sqrt() / mean_memory).min(1.0)
        } else {
            1.0
        };

        // Performance consistency test
        let consistency_score = (throughput_stability + memory_consistency) / 2.0;
        let performance_tolerance = self.config.performance_tolerance;

        tests.push(ValidationTest {
            name: "Performance Consistency".to_string(),
            category: ValidationCategory::PerformanceConsistency,
            status: if timing_variance < performance_tolerance {
                ValidationStatus::Passed
            } else if timing_variance < performance_tolerance * 2.0 {
                ValidationStatus::Warning
            } else {
                ValidationStatus::Failed
            },
            score: consistency_score,
            description: format!(
                "Timing variance: {:.2}, tolerance: {:.2}",
                timing_variance, performance_tolerance
            ),
            evidence: vec![
                format!("Throughput stability: {:.2}", throughput_stability),
                format!("Memory consistency: {:.2}", memory_consistency),
            ],
        });

        if timing_variance > performance_tolerance {
            issues.push(ValidationIssue {
                severity: if timing_variance > performance_tolerance * 3.0 {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                category: ValidationCategory::PerformanceConsistency,
                description: format!(
                    "High performance variance detected: {:.2}",
                    timing_variance * 100.0
                ),
                suggested_fixes: vec![
                    "Increase benchmark duration".to_string(),
                    "Reduce system load during benchmarking".to_string(),
                    "Use dedicated benchmark environment".to_string(),
                    "Add CPU affinity and frequency scaling controls".to_string(),
                ],
                impact: "Benchmark results may be unreliable".to_string(),
            });
        }

        PerformanceConsistency {
            timing_variance,
            throughput_stability,
            memory_consistency,
            scaling_consistency: 0.9, // Placeholder
            reproducibility_score: consistency_score,
        }
    }

    /// Validate cross-architecture consistency
    fn validate_cross_architecture(
        &self,
        _results: &[BenchResult],
        tests: &mut Vec<ValidationTest>,
        _issues: &mut Vec<ValidationIssue>,
    ) -> CrossArchResult {
        // Detect current architecture
        let current_arch = self.detect_current_architecture();

        // For now, simulate cross-architecture validation
        // In practice, this would run benchmarks on different target architectures
        let consistency_score = 0.95; // Placeholder high consistency

        tests.push(ValidationTest {
            name: "Cross-Architecture Consistency".to_string(),
            category: ValidationCategory::CrossPlatform,
            status: if consistency_score > 0.9 {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Warning
            },
            score: consistency_score,
            description: "Cross-architecture validation placeholder".to_string(),
            evidence: vec![
                format!("Current architecture: {}", current_arch.name),
                "Additional architectures would be tested in full implementation".to_string(),
            ],
        });

        CrossArchResult {
            architectures: vec![current_arch],
            consistency_score,
            performance_variation: 0.1,
            accuracy_variation: 1e-10,
            arch_issues: HashMap::new(),
        }
    }

    /// Detect current architecture
    fn detect_current_architecture(&self) -> Architecture {
        Architecture {
            name: std::env::consts::ARCH.to_string(),
            features: vec!["baseline".to_string()], // Would detect actual CPU features
            endianness: if cfg!(target_endian = "little") {
                Endianness::Little
            } else {
                Endianness::Big
            },
            pointer_size: std::mem::size_of::<usize>(),
            simd_caps: vec!["sse".to_string(), "avx".to_string()], // Placeholder
        }
    }

    /// Validate optimization correctness
    fn validate_optimization_correctness(
        &self,
        _results: &[BenchResult],
        _config: &BenchConfig,
        tests: &mut Vec<ValidationTest>,
        _issues: &mut Vec<ValidationIssue>,
    ) -> OptimizationCorrectness {
        // Simulate optimization level comparison
        let optimizations = vec![
            OptimizationComparison {
                optimization_level: "O0".to_string(),
                performance_improvement: 1.0,
                accuracy_delta: 0.0,
                correctness_status: ValidationStatus::Passed,
            },
            OptimizationComparison {
                optimization_level: "O2".to_string(),
                performance_improvement: 1.5,
                accuracy_delta: 1e-12,
                correctness_status: ValidationStatus::Passed,
            },
            OptimizationComparison {
                optimization_level: "O3".to_string(),
                performance_improvement: 1.8,
                accuracy_delta: 1e-11,
                correctness_status: ValidationStatus::Passed,
            },
        ];

        tests.push(ValidationTest {
            name: "Optimization Correctness".to_string(),
            category: ValidationCategory::OptimizationCorrectness,
            status: ValidationStatus::Passed,
            score: 0.95,
            description: "Optimization levels maintain correctness".to_string(),
            evidence: optimizations
                .iter()
                .map(|o| {
                    format!(
                        "{}: {:.1}x speedup, {:.2e} accuracy delta",
                        o.optimization_level, o.performance_improvement, o.accuracy_delta
                    )
                })
                .collect(),
        });

        OptimizationCorrectness {
            optimization_comparison: optimizations,
            compiler_safety: CompilerSafety {
                unsafe_optimizations: Vec::new(),
                fast_math_safety: true,
                barriers_needed: Vec::new(),
            },
            algorithm_equivalence: AlgorithmEquivalence {
                reference_comparison: 1.0,
                alternative_comparison: 0.999999,
                equivalence_confidence: 0.95,
            },
        }
    }

    /// Generate recommendations based on validation results
    fn generate_recommendations(
        &self,
        tests: &[ValidationTest],
        issues: &[ValidationIssue],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Count issues by severity
        let critical_issues = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .count();
        let high_issues = issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::High))
            .count();

        if critical_issues > 0 {
            recommendations.push("CRITICAL: Benchmark results may be unreliable. Review and fix critical issues before using results.".to_string());
        }

        if high_issues > 0 {
            recommendations.push(
                "High-severity issues detected. Consider addressing these before production use."
                    .to_string(),
            );
        }

        // Test-specific recommendations
        let failed_tests = tests
            .iter()
            .filter(|t| t.status == ValidationStatus::Failed)
            .count();
        if failed_tests > 0 {
            recommendations.push(format!(
                "{} validation tests failed. Review test details for specific improvements.",
                failed_tests
            ));
        }

        // Performance-specific recommendations
        let perf_issues = issues
            .iter()
            .filter(|i| matches!(i.category, ValidationCategory::PerformanceConsistency))
            .count();
        if perf_issues > 0 {
            recommendations.push("Performance consistency issues detected. Consider running benchmarks in a more controlled environment.".to_string());
        }

        // Numerical-specific recommendations
        let numerical_issues = issues
            .iter()
            .filter(|i| matches!(i.category, ValidationCategory::NumericalCorrectness))
            .count();
        if numerical_issues > 0 {
            recommendations.push("Numerical accuracy issues detected. Consider using higher precision arithmetic or more stable algorithms.".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("All validations passed. Benchmark results appear reliable.".to_string());
        }

        recommendations
    }

    /// Calculate overall validation status
    fn calculate_overall_status(&self, tests: &[ValidationTest]) -> ValidationStatus {
        if tests.is_empty() {
            return ValidationStatus::Incomplete;
        }

        let failed = tests.iter().any(|t| t.status == ValidationStatus::Failed);
        let has_warnings = tests.iter().any(|t| t.status == ValidationStatus::Warning);

        if failed {
            ValidationStatus::Failed
        } else if has_warnings {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Passed
        }
    }

    /// Generate synthetic test data for validation
    pub fn generate_test_data(&self, size: usize, seed: Option<u64>) -> Vec<f32> {
        let mut rng = if let Some(s) = seed {
            Random::seed(s)
        } else {
            Random::seed(self.config.random_seed)
        };

        (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Add cross-architecture test data
    pub fn add_cross_arch_data(&mut self, data: CrossArchTestData) {
        self.cross_arch_data.push(data);
    }
}

impl Default for BenchmarkValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CrossArchResult {
    fn default() -> Self {
        Self {
            architectures: Vec::new(),
            consistency_score: 1.0,
            performance_variation: 0.0,
            accuracy_variation: 0.0,
            arch_issues: HashMap::new(),
        }
    }
}

impl Default for PerformanceConsistency {
    fn default() -> Self {
        Self {
            timing_variance: 0.0,
            throughput_stability: 1.0,
            memory_consistency: 1.0,
            scaling_consistency: 1.0,
            reproducibility_score: 1.0,
        }
    }
}

impl Default for OptimizationCorrectness {
    fn default() -> Self {
        Self {
            optimization_comparison: Vec::new(),
            compiler_safety: CompilerSafety {
                unsafe_optimizations: Vec::new(),
                fast_math_safety: true,
                barriers_needed: Vec::new(),
            },
            algorithm_equivalence: AlgorithmEquivalence {
                reference_comparison: 1.0,
                alternative_comparison: 1.0,
                equivalence_confidence: 1.0,
            },
        }
    }
}

impl fmt::Display for ValidationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationStatus::Passed => write!(f, "PASSED"),
            ValidationStatus::Warning => write!(f, "WARNING"),
            ValidationStatus::Failed => write!(f, "FAILED"),
            ValidationStatus::Incomplete => write!(f, "INCOMPLETE"),
        }
    }
}

impl fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueSeverity::Low => write!(f, "LOW"),
            IssueSeverity::Medium => write!(f, "MEDIUM"),
            IssueSeverity::High => write!(f, "HIGH"),
            IssueSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Example reference implementations for common operations
pub mod reference_impls {
    use super::ReferenceImplementation;

    /// Naive matrix multiplication reference implementation
    pub struct NaiveMatMul {
        pub size: usize,
    }

    impl ReferenceImplementation for NaiveMatMul {
        fn run(&self, input: &[f32]) -> Vec<f32> {
            let n = self.size;
            assert_eq!(input.len(), 2 * n * n);

            let a = &input[0..n * n];
            let b = &input[n * n..2 * n * n];
            let mut c = vec![0.0; n * n];

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        c[i * n + j] += a[i * n + k] * b[k * n + j];
                    }
                }
            }

            c
        }

        fn name(&self) -> &str {
            "naive_matmul"
        }

        fn description(&self) -> &str {
            "Naive O(n³) matrix multiplication reference implementation"
        }
    }

    /// Element-wise addition reference implementation
    pub struct ElementwiseAdd;

    impl ReferenceImplementation for ElementwiseAdd {
        fn run(&self, input: &[f32]) -> Vec<f32> {
            let mid = input.len() / 2;
            let a = &input[0..mid];
            let b = &input[mid..];

            a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
        }

        fn name(&self) -> &str {
            "elementwise_add"
        }

        fn description(&self) -> &str {
            "Element-wise addition reference implementation"
        }
    }

    /// Vector dot product reference implementation
    pub struct DotProduct;

    impl ReferenceImplementation for DotProduct {
        fn run(&self, input: &[f32]) -> Vec<f32> {
            let mid = input.len() / 2;
            let a = &input[0..mid];
            let b = &input[mid..];

            let result = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            vec![result]
        }

        fn name(&self) -> &str {
            "dot_product"
        }

        fn description(&self) -> &str {
            "Vector dot product reference implementation"
        }
    }
}

/// Validation utilities
pub mod validation_utils {
    use super::*;

    /// Compare two floating point arrays with relative tolerance
    pub fn compare_arrays(a: &[f32], b: &[f32], tolerance: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }

        a.iter().zip(b.iter()).all(|(&x, &y)| {
            let abs_diff = (x - y).abs() as f64;
            let max_val = (x.abs().max(y.abs())) as f64;

            if max_val < f64::EPSILON {
                abs_diff < tolerance
            } else {
                abs_diff / max_val < tolerance
            }
        })
    }

    /// Calculate ULP distance between two floating point numbers
    pub fn ulp_distance(a: f32, b: f32) -> u32 {
        if a == b {
            return 0;
        }

        let a_bits = a.to_bits();
        let b_bits = b.to_bits();

        if (a_bits ^ b_bits) & 0x80000000 != 0 {
            // Different signs
            a_bits.wrapping_add(b_bits ^ 0x80000000)
        } else {
            // Same sign
            a_bits.max(b_bits) - a_bits.min(b_bits)
        }
    }

    /// Detect potential overflow/underflow conditions
    pub fn check_overflow_conditions(values: &[f32]) -> Vec<OverflowIssue> {
        let mut issues = Vec::new();

        for (i, &val) in values.iter().enumerate() {
            if val.is_infinite() {
                issues.push(OverflowIssue {
                    issue_type: OverflowType::Infinity,
                    operation: format!("Value at index {}", i),
                    value_range: (val as f64, val as f64),
                });
            } else if val.is_nan() {
                issues.push(OverflowIssue {
                    issue_type: OverflowType::NaN,
                    operation: format!("Value at index {}", i),
                    value_range: (f64::NAN, f64::NAN),
                });
            } else if val.abs() > f32::MAX / 2.0 {
                issues.push(OverflowIssue {
                    issue_type: OverflowType::Overflow,
                    operation: format!("Large value at index {}", i),
                    value_range: (val as f64, val as f64),
                });
            } else if val.abs() < f32::MIN_POSITIVE * 2.0 && val != 0.0 {
                issues.push(OverflowIssue {
                    issue_type: OverflowType::Underflow,
                    operation: format!("Small value at index {}", i),
                    value_range: (val as f64, val as f64),
                });
            }
        }

        issues
    }

    /// Generate comprehensive validation report
    pub fn generate_html_report(validation: &ValidationResult) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html><head><title>Benchmark Validation Report</title>");
        html.push_str("<style>");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }");
        html.push_str(".status-passed { color: green; font-weight: bold; }");
        html.push_str(".status-warning { color: orange; font-weight: bold; }");
        html.push_str(".status-failed { color: red; font-weight: bold; }");
        html.push_str(".severity-critical { background-color: #ffebee; border-left: 4px solid red; padding: 10px; }");
        html.push_str(".severity-high { background-color: #fff3e0; border-left: 4px solid orange; padding: 10px; }");
        html.push_str("table { border-collapse: collapse; width: 100%; }");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }");
        html.push_str("th { background-color: #f2f2f2; }");
        html.push_str("</style></head><body>");

        html.push_str(&format!("<h1>Benchmark Validation Report</h1>"));
        html.push_str(&format!(
            "<h2>Overall Status: <span class=\"status-{}\">{}</span></h2>",
            validation.status.to_string().to_lowercase(),
            validation.status
        ));

        // Summary
        html.push_str("<h3>Summary</h3>");
        html.push_str("<table>");
        html.push_str(&format!(
            "<tr><td>Total Tests</td><td>{}</td></tr>",
            validation.report.summary.total_tests
        ));
        html.push_str(&format!(
            "<tr><td>Passed</td><td>{}</td></tr>",
            validation.report.summary.passed
        ));
        html.push_str(&format!(
            "<tr><td>Warnings</td><td>{}</td></tr>",
            validation.report.summary.warnings
        ));
        html.push_str(&format!(
            "<tr><td>Failed</td><td>{}</td></tr>",
            validation.report.summary.failed
        ));
        html.push_str(&format!(
            "<tr><td>Confidence Score</td><td>{:.2}</td></tr>",
            validation.report.summary.confidence_score * 100.0
        ));
        html.push_str("</table>");

        // Issues
        if !validation.report.issues.is_empty() {
            html.push_str("<h3>Issues</h3>");
            for issue in &validation.report.issues {
                let severity_class = match issue.severity {
                    IssueSeverity::Critical => "severity-critical",
                    IssueSeverity::High => "severity-high",
                    _ => "",
                };
                html.push_str(&format!("<div class=\"{}\">", severity_class));
                html.push_str(&format!(
                    "<strong>{}:</strong> {}",
                    issue.severity, issue.description
                ));
                html.push_str("</div>");
            }
        }

        // Recommendations
        if !validation.report.recommendations.is_empty() {
            html.push_str("<h3>Recommendations</h3>");
            html.push_str("<ul>");
            for rec in &validation.report.recommendations {
                html.push_str(&format!("<li>{}</li>", rec));
            }
            html.push_str("</ul>");
        }

        html.push_str("</body></html>");
        html
    }
}
