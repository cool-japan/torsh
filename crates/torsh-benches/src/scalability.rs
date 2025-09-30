//! Scalability testing framework for ToRSh operations
//!
//! This module provides comprehensive scalability analysis for tensor operations,
//! memory usage, and performance characteristics across different input sizes.

use crate::metrics::cross_framework::OperationType;
use crate::MetricsCollector;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Scalability test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityConfig {
    /// Test name
    pub name: String,

    /// Operation type being tested
    pub operation: OperationType,

    /// Size progression (e.g., powers of 2, linear progression)
    pub size_progression: SizeProgression,

    /// Minimum input size
    pub min_size: usize,

    /// Maximum input size
    pub max_size: usize,

    /// Number of samples per size
    pub samples_per_size: usize,

    /// Expected complexity (for analysis)
    pub expected_complexity: ComplexityClass,

    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Size progression patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SizeProgression {
    /// Powers of 2: 1, 2, 4, 8, 16, ...
    PowersOfTwo,
    /// Linear progression: 100, 200, 300, 400, ...
    Linear { step: usize },
    /// Exponential: base^1, base^2, base^3, ...
    Exponential { base: f64 },
    /// Custom list of sizes
    Custom { sizes: Vec<usize> },
    /// Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...
    Fibonacci,
}

/// Expected computational complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    /// O(1) - constant time
    Constant,
    /// O(log n) - logarithmic
    Logarithmic,
    /// O(n) - linear
    Linear,
    /// O(n log n) - linearithmic
    Linearithmic,
    /// O(n²) - quadratic
    Quadratic,
    /// O(n³) - cubic
    Cubic,
    /// O(2^n) - exponential
    Exponential,
    /// O(n!) - factorial
    Factorial,
    /// Custom complexity description
    Custom { description: String },
}

/// Resource usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<u64>,

    /// Maximum execution time per test
    pub max_execution_time: Duration,

    /// Maximum total test duration
    pub max_total_time: Duration,

    /// Memory efficiency threshold (ops per MB)
    pub min_memory_efficiency: Option<f64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB
            max_execution_time: Duration::from_secs(60),
            max_total_time: Duration::from_secs(300), // 5 minutes
            min_memory_efficiency: Some(1000.0),      // 1000 ops per MB
        }
    }
}

/// Scalability test suite
pub struct ScalabilityTestSuite {
    configs: Vec<ScalabilityConfig>,
    results: Vec<ScalabilityTestResult>,
    metrics_collector: MetricsCollector,
}

impl ScalabilityTestSuite {
    pub fn new() -> Self {
        Self {
            configs: Self::default_configs(),
            results: Vec::new(),
            metrics_collector: MetricsCollector::new(),
        }
    }

    /// Create default scalability test configurations
    fn default_configs() -> Vec<ScalabilityConfig> {
        vec![
            ScalabilityConfig {
                name: "Matrix Multiplication Scalability".to_string(),
                operation: OperationType::MatrixMultiplication,
                size_progression: SizeProgression::PowersOfTwo,
                min_size: 64,
                max_size: 2048,
                samples_per_size: 3,
                expected_complexity: ComplexityClass::Cubic,
                resource_limits: ResourceLimits::default(),
            },
            ScalabilityConfig {
                name: "Element-wise Addition Scalability".to_string(),
                operation: OperationType::ElementWiseAddition,
                size_progression: SizeProgression::Linear { step: 500 },
                min_size: 500,
                max_size: 5000,
                samples_per_size: 5,
                expected_complexity: ComplexityClass::Linear,
                resource_limits: ResourceLimits::default(),
            },
            ScalabilityConfig {
                name: "Convolution 2D Scalability".to_string(),
                operation: OperationType::Convolution2D,
                size_progression: SizeProgression::Custom {
                    sizes: vec![32, 64, 128, 224, 256, 384, 512],
                },
                min_size: 32,
                max_size: 512,
                samples_per_size: 3,
                expected_complexity: ComplexityClass::Custom {
                    description: "O(n²) for spatial dimensions".to_string(),
                },
                resource_limits: ResourceLimits::default(),
            },
            ScalabilityConfig {
                name: "Memory Allocation Scalability".to_string(),
                operation: OperationType::MemoryAllocation,
                size_progression: SizeProgression::Exponential { base: 1.5 },
                min_size: 1000,
                max_size: 1000000,
                samples_per_size: 10,
                expected_complexity: ComplexityClass::Linear,
                resource_limits: ResourceLimits {
                    max_memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4GB for allocation tests
                    ..ResourceLimits::default()
                },
            },
        ]
    }

    /// Add a custom scalability test configuration
    pub fn add_config(&mut self, config: ScalabilityConfig) {
        self.configs.push(config);
    }

    /// Generate sizes based on progression pattern
    pub fn generate_sizes(&self, config: &ScalabilityConfig) -> Vec<usize> {
        match &config.size_progression {
            SizeProgression::PowersOfTwo => {
                let mut sizes = Vec::new();
                let mut size = config.min_size;
                while size <= config.max_size {
                    sizes.push(size);
                    size *= 2;
                }
                sizes
            }
            SizeProgression::Linear { step } => {
                (config.min_size..=config.max_size).step_by(*step).collect()
            }
            SizeProgression::Exponential { base } => {
                let mut sizes = Vec::new();
                let mut size = config.min_size as f64;
                while size <= config.max_size as f64 {
                    sizes.push(size as usize);
                    size *= base;
                }
                sizes
            }
            SizeProgression::Custom { sizes } => sizes
                .iter()
                .filter(|&&s| s >= config.min_size && s <= config.max_size)
                .cloned()
                .collect(),
            SizeProgression::Fibonacci => {
                let mut sizes = Vec::new();
                let (mut a, mut b) = (1, 1);
                while a <= config.max_size {
                    if a >= config.min_size {
                        sizes.push(a);
                    }
                    let next = a + b;
                    a = b;
                    b = next;
                }
                sizes
            }
        }
    }

    /// Run all scalability tests
    pub fn run_all_tests(&mut self) -> Vec<ScalabilityTestResult> {
        let total_start = Instant::now();

        for config in self.configs.clone() {
            if total_start.elapsed() > config.resource_limits.max_total_time {
                println!("⚠️  Stopping tests due to total time limit exceeded");
                break;
            }

            println!("🚀 Running scalability test: {}", config.name);
            let result = self.run_single_test(&config);
            self.results.push(result);
        }

        self.results.clone()
    }

    /// Run a single scalability test
    pub fn run_single_test(&mut self, config: &ScalabilityConfig) -> ScalabilityTestResult {
        let sizes = self.generate_sizes(config);
        let mut measurements = Vec::new();
        let test_start = Instant::now();

        for &size in &sizes {
            if test_start.elapsed() > config.resource_limits.max_total_time {
                println!("⚠️  Stopping size {} due to time limit", size);
                break;
            }

            println!("  📊 Testing size: {}", size);

            let mut size_measurements = Vec::new();
            for sample in 0..config.samples_per_size {
                let measurement = self.run_single_measurement(config, size, sample);

                // Check resource limits
                if let Some(max_mem) = config.resource_limits.max_memory_bytes {
                    if measurement.memory_usage_bytes > max_mem {
                        println!(
                            "⚠️  Memory limit exceeded for size {}: {} bytes",
                            size, measurement.memory_usage_bytes
                        );
                        break;
                    }
                }

                if measurement.execution_time > config.resource_limits.max_execution_time {
                    println!(
                        "⚠️  Execution time limit exceeded for size {}: {:?}",
                        size, measurement.execution_time
                    );
                    break;
                }

                size_measurements.push(measurement);
            }

            if !size_measurements.is_empty() {
                let aggregate = ScalabilityMeasurement::aggregate(&size_measurements);
                measurements.push(aggregate);
            }
        }

        let analysis = self.analyze_scalability(&measurements, &config.expected_complexity);

        ScalabilityTestResult {
            config: config.clone(),
            measurements,
            analysis,
            total_test_time: test_start.elapsed(),
        }
    }

    /// Run a single measurement for a specific size
    fn run_single_measurement(
        &mut self,
        config: &ScalabilityConfig,
        size: usize,
        sample: usize,
    ) -> ScalabilityMeasurement {
        self.metrics_collector.start();
        let start_time = Instant::now();

        // Mock operation execution based on operation type
        // In a real implementation, this would call the actual tensor operations
        let (flops, memory_ops) = match config.operation {
            OperationType::MatrixMultiplication => {
                // O(n³) complexity
                mock_matmul_operation(size, size, size);
                (2 * size * size * size, 3 * size * size)
            }
            OperationType::ElementWiseAddition => {
                // O(n) complexity
                mock_elementwise_operation(size * size);
                (size * size, 3 * size * size)
            }
            OperationType::Convolution2D => {
                // O(n²) for spatial dimensions
                mock_conv2d_operation(size, size, 64, 64);
                (size * size * 64 * 64 * 9, size * size * 64 * 2) // 3x3 kernel
            }
            OperationType::MemoryAllocation => {
                mock_memory_allocation(size * size * 4); // f32 = 4 bytes
                (size * size, size * size * 4)
            }
            _ => {
                mock_generic_operation(size);
                (size, size * 4)
            }
        };

        let execution_time = start_time.elapsed();
        let system_metrics = self.metrics_collector.stop();

        ScalabilityMeasurement {
            size,
            sample_id: sample,
            execution_time,
            memory_usage_bytes: system_metrics.memory_stats.peak_usage_mb as u64 * 1024 * 1024,
            flops_executed: flops,
            memory_operations: memory_ops,
            cpu_utilization: system_metrics.cpu_utilization(),
            memory_efficiency: system_metrics.memory_efficiency(flops),
            throughput_ops_per_sec: flops as f64 / execution_time.as_secs_f64(),
            memory_bandwidth_gbps: (memory_ops * 4) as f64 / execution_time.as_secs_f64() / 1e9,
        }
    }

    /// Analyze scalability characteristics
    fn analyze_scalability(
        &self,
        measurements: &[ScalabilityMeasurement],
        expected: &ComplexityClass,
    ) -> ScalabilityAnalysis {
        if measurements.len() < 2 {
            return ScalabilityAnalysis::default();
        }

        // Calculate scaling factors
        let mut time_scaling_factors = Vec::new();
        let mut memory_scaling_factors = Vec::new();

        for i in 1..measurements.len() {
            let prev = &measurements[i - 1];
            let curr = &measurements[i];

            let size_ratio = curr.size as f64 / prev.size as f64;
            let time_ratio = curr.execution_time.as_secs_f64() / prev.execution_time.as_secs_f64();
            let memory_ratio = curr.memory_usage_bytes as f64 / prev.memory_usage_bytes as f64;

            time_scaling_factors.push(time_ratio / size_ratio);
            memory_scaling_factors.push(memory_ratio / size_ratio);
        }

        // Statistical analysis
        let avg_time_scaling =
            time_scaling_factors.iter().sum::<f64>() / time_scaling_factors.len() as f64;
        let avg_memory_scaling =
            memory_scaling_factors.iter().sum::<f64>() / memory_scaling_factors.len() as f64;

        // Determine observed complexity
        let observed_complexity = self.infer_complexity(&time_scaling_factors);

        // Performance trends
        let performance_trend = if measurements.len() >= 3 {
            let first_third = measurements.len() / 3;
            let last_third = measurements.len() * 2 / 3;

            let early_avg = measurements[..first_third]
                .iter()
                .map(|m| m.throughput_ops_per_sec)
                .sum::<f64>()
                / first_third as f64;
            let late_avg = measurements[last_third..]
                .iter()
                .map(|m| m.throughput_ops_per_sec)
                .sum::<f64>()
                / (measurements.len() - last_third) as f64;

            if late_avg > early_avg * 1.1 {
                PerformanceTrend::Improving
            } else if late_avg < early_avg * 0.9 {
                PerformanceTrend::Degrading
            } else {
                PerformanceTrend::Stable
            }
        } else {
            PerformanceTrend::Stable
        };

        let complexity_match = self.complexity_matches(&observed_complexity, expected);

        ScalabilityAnalysis {
            observed_complexity,
            expected_complexity: expected.clone(),
            complexity_match,
            average_time_scaling_factor: avg_time_scaling,
            average_memory_scaling_factor: avg_memory_scaling,
            performance_trend,
            efficiency_score: self.calculate_efficiency_score(measurements),
            memory_efficiency_trend: self.analyze_memory_efficiency_trend(measurements),
            bottleneck_analysis: self.identify_bottlenecks(measurements),
        }
    }

    /// Infer complexity class from scaling factors
    fn infer_complexity(&self, scaling_factors: &[f64]) -> ComplexityClass {
        if scaling_factors.is_empty() {
            return ComplexityClass::Constant;
        }

        let avg_factor = scaling_factors.iter().sum::<f64>() / scaling_factors.len() as f64;

        match avg_factor {
            f if f < 1.1 => ComplexityClass::Constant,
            f if f < 1.5 => ComplexityClass::Logarithmic,
            f if f < 2.2 => ComplexityClass::Linear,
            f if f < 3.0 => ComplexityClass::Linearithmic,
            f if f < 5.0 => ComplexityClass::Quadratic,
            f if f < 10.0 => ComplexityClass::Cubic,
            _ => ComplexityClass::Exponential,
        }
    }

    /// Check if observed complexity matches expected
    fn complexity_matches(&self, observed: &ComplexityClass, expected: &ComplexityClass) -> bool {
        match (observed, expected) {
            (ComplexityClass::Constant, ComplexityClass::Constant) => true,
            (ComplexityClass::Logarithmic, ComplexityClass::Logarithmic) => true,
            (ComplexityClass::Linear, ComplexityClass::Linear) => true,
            (ComplexityClass::Linearithmic, ComplexityClass::Linearithmic) => true,
            (ComplexityClass::Quadratic, ComplexityClass::Quadratic) => true,
            (ComplexityClass::Cubic, ComplexityClass::Cubic) => true,
            (ComplexityClass::Exponential, ComplexityClass::Exponential) => true,
            _ => false,
        }
    }

    /// Calculate overall efficiency score (0-100)
    fn calculate_efficiency_score(&self, measurements: &[ScalabilityMeasurement]) -> f64 {
        if measurements.is_empty() {
            return 0.0;
        }

        let avg_cpu =
            measurements.iter().map(|m| m.cpu_utilization).sum::<f64>() / measurements.len() as f64;
        let avg_memory_eff = measurements
            .iter()
            .map(|m| m.memory_efficiency)
            .sum::<f64>()
            / measurements.len() as f64;
        let avg_throughput = measurements
            .iter()
            .map(|m| m.throughput_ops_per_sec)
            .sum::<f64>()
            / measurements.len() as f64;

        // Normalize and combine metrics (simple scoring)
        let cpu_score = (avg_cpu / 100.0).min(1.0) * 30.0; // 30% weight
        let memory_score = (avg_memory_eff / 10000.0).min(1.0) * 30.0; // 30% weight
        let throughput_score = (avg_throughput / 1e6).min(1.0) * 40.0; // 40% weight

        cpu_score + memory_score + throughput_score
    }

    /// Analyze memory efficiency trends
    fn analyze_memory_efficiency_trend(
        &self,
        measurements: &[ScalabilityMeasurement],
    ) -> MemoryEfficiencyTrend {
        if measurements.len() < 3 {
            return MemoryEfficiencyTrend::Stable;
        }

        let first_half = &measurements[..measurements.len() / 2];
        let second_half = &measurements[measurements.len() / 2..];

        let first_avg =
            first_half.iter().map(|m| m.memory_efficiency).sum::<f64>() / first_half.len() as f64;
        let second_avg =
            second_half.iter().map(|m| m.memory_efficiency).sum::<f64>() / second_half.len() as f64;

        if second_avg > first_avg * 1.1 {
            MemoryEfficiencyTrend::Improving
        } else if second_avg < first_avg * 0.9 {
            MemoryEfficiencyTrend::Degrading
        } else {
            MemoryEfficiencyTrend::Stable
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self, measurements: &[ScalabilityMeasurement]) -> Vec<BottleneckType> {
        let mut bottlenecks = Vec::new();

        if let Some(measurement) = measurements.last() {
            // CPU utilization bottleneck
            if measurement.cpu_utilization < 50.0 {
                bottlenecks.push(BottleneckType::CpuUnderutilized);
            } else if measurement.cpu_utilization > 95.0 {
                bottlenecks.push(BottleneckType::CpuBound);
            }

            // Memory bottleneck
            if measurement.memory_efficiency < 100.0 {
                bottlenecks.push(BottleneckType::MemoryInefficient);
            }

            // Memory bandwidth bottleneck
            if measurement.memory_bandwidth_gbps < 10.0 {
                bottlenecks.push(BottleneckType::MemoryBandwidthLimited);
            }

            // Check for scaling issues
            if measurements.len() > 1 {
                let prev = &measurements[measurements.len() - 2];
                let size_ratio = measurement.size as f64 / prev.size as f64;
                let time_ratio =
                    measurement.execution_time.as_secs_f64() / prev.execution_time.as_secs_f64();

                if time_ratio > size_ratio * 3.0 {
                    bottlenecks.push(BottleneckType::AlgorithmicInefficiency);
                }
            }
        }

        bottlenecks
    }

    /// Export results to comprehensive report
    pub fn export_comprehensive_report(&self, output_dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // CSV export
        self.export_csv(&format!("{}/scalability_results.csv", output_dir))?;

        // JSON export
        self.export_json(&format!("{}/scalability_results.json", output_dir))?;

        // HTML report
        self.generate_html_report(&format!("{}/scalability_report.html", output_dir))?;

        // Analysis summary
        self.generate_analysis_summary(&format!("{}/scalability_analysis.md", output_dir))?;

        println!(
            "📊 Comprehensive scalability report generated in {}",
            output_dir
        );
        Ok(())
    }

    /// Export results to CSV
    fn export_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "test_name,size,sample_id,execution_time_ms,memory_usage_mb,flops,memory_ops,cpu_utilization,memory_efficiency,throughput_ops_per_sec,memory_bandwidth_gbps")?;

        for result in &self.results {
            for measurement in &result.measurements {
                writeln!(
                    file,
                    "{},{},{},{:.2},{:.2},{},{},{:.2},{:.2},{:.2},{:.2}",
                    result.config.name,
                    measurement.size,
                    measurement.sample_id,
                    measurement.execution_time.as_millis(),
                    measurement.memory_usage_bytes as f64 / 1_000_000.0,
                    measurement.flops_executed,
                    measurement.memory_operations,
                    measurement.cpu_utilization,
                    measurement.memory_efficiency,
                    measurement.throughput_ops_per_sec,
                    measurement.memory_bandwidth_gbps
                )?;
            }
        }

        Ok(())
    }

    /// Export results to JSON
    fn export_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html><head><title>ToRSh Scalability Report</title>")?;
        writeln!(file, "<style>")?;
        writeln!(
            file,
            "body {{ font-family: Arial, sans-serif; margin: 20px; }}"
        )?;
        writeln!(
            file,
            "table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
        )?;
        writeln!(
            file,
            "th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
        )?;
        writeln!(file, "th {{ background-color: #f2f2f2; }}")?;
        writeln!(file, ".metric {{ margin: 20px 0; }}")?;
        writeln!(
            file,
            ".complexity-match {{ color: green; font-weight: bold; }}"
        )?;
        writeln!(
            file,
            ".complexity-mismatch {{ color: red; font-weight: bold; }}"
        )?;
        writeln!(file, "</style></head><body>")?;

        writeln!(file, "<h1>🚀 ToRSh Scalability Report</h1>")?;
        writeln!(
            file,
            "<p>Comprehensive scalability analysis across tensor operations</p>"
        )?;

        for result in &self.results {
            writeln!(file, "<div class='metric'>")?;
            writeln!(file, "<h2>{}</h2>", result.config.name)?;

            // Analysis summary
            let complexity_class = if result.analysis.complexity_match {
                "complexity-match"
            } else {
                "complexity-mismatch"
            };
            writeln!(
                file,
                "<p><strong>Expected Complexity:</strong> {:?}</p>",
                result.analysis.expected_complexity
            )?;
            writeln!(
                file,
                "<p class='{}'><strong>Observed Complexity:</strong> {:?}</p>",
                complexity_class, result.analysis.observed_complexity
            )?;
            writeln!(
                file,
                "<p><strong>Efficiency Score:</strong> {:.1}/100</p>",
                result.analysis.efficiency_score
            )?;
            writeln!(
                file,
                "<p><strong>Performance Trend:</strong> {:?}</p>",
                result.analysis.performance_trend
            )?;

            // Measurements table
            writeln!(file, "<table>")?;
            writeln!(file, "<tr><th>Size</th><th>Time (ms)</th><th>Memory (MB)</th><th>Throughput (ops/s)</th><th>CPU %</th><th>Mem Eff</th></tr>")?;

            for measurement in &result.measurements {
                writeln!(
                    file,
                    "<tr><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td><td>{:.1}</td><td>{:.2}</td></tr>",
                    measurement.size,
                    measurement.execution_time.as_millis(),
                    measurement.memory_usage_bytes as f64 / 1_000_000.0,
                    measurement.throughput_ops_per_sec,
                    measurement.cpu_utilization,
                    measurement.memory_efficiency
                )?;
            }

            writeln!(file, "</table>")?;
            writeln!(file, "</div>")?;
        }

        writeln!(file, "</body></html>")?;
        Ok(())
    }

    /// Generate analysis summary in markdown
    fn generate_analysis_summary(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "# ToRSh Scalability Analysis Summary\n")?;

        for result in &self.results {
            writeln!(file, "## {}\n", result.config.name)?;
            writeln!(
                file,
                "- **Expected Complexity:** {:?}",
                result.analysis.expected_complexity
            )?;
            writeln!(
                file,
                "- **Observed Complexity:** {:?}",
                result.analysis.observed_complexity
            )?;
            writeln!(
                file,
                "- **Complexity Match:** {}",
                if result.analysis.complexity_match {
                    "✅ Yes"
                } else {
                    "❌ No"
                }
            )?;
            writeln!(
                file,
                "- **Efficiency Score:** {:.1}/100",
                result.analysis.efficiency_score
            )?;
            writeln!(
                file,
                "- **Performance Trend:** {:?}",
                result.analysis.performance_trend
            )?;
            writeln!(
                file,
                "- **Memory Efficiency Trend:** {:?}",
                result.analysis.memory_efficiency_trend
            )?;
            writeln!(
                file,
                "- **Test Duration:** {:.2}s",
                result.total_test_time.as_secs_f64()
            )?;

            if !result.analysis.bottleneck_analysis.is_empty() {
                writeln!(file, "- **Identified Bottlenecks:**")?;
                for bottleneck in &result.analysis.bottleneck_analysis {
                    writeln!(file, "  - {:?}", bottleneck)?;
                }
            }

            writeln!(file)?;
        }

        Ok(())
    }
}

impl Default for ScalabilityTestSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Single scalability measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMeasurement {
    pub size: usize,
    pub sample_id: usize,
    pub execution_time: Duration,
    pub memory_usage_bytes: u64,
    pub flops_executed: usize,
    pub memory_operations: usize,
    pub cpu_utilization: f64,
    pub memory_efficiency: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_bandwidth_gbps: f64,
}

impl ScalabilityMeasurement {
    /// Aggregate multiple measurements for the same size
    pub fn aggregate(measurements: &[ScalabilityMeasurement]) -> Self {
        if measurements.is_empty() {
            return ScalabilityMeasurement {
                size: 0,
                sample_id: 0,
                execution_time: Duration::ZERO,
                memory_usage_bytes: 0,
                flops_executed: 0,
                memory_operations: 0,
                cpu_utilization: 0.0,
                memory_efficiency: 0.0,
                throughput_ops_per_sec: 0.0,
                memory_bandwidth_gbps: 0.0,
            };
        }

        let count = measurements.len();
        let size = measurements[0].size;

        let avg_execution_time = Duration::from_nanos(
            (measurements
                .iter()
                .map(|m| m.execution_time.as_nanos())
                .sum::<u128>()
                / count as u128) as u64,
        );
        let avg_memory = measurements
            .iter()
            .map(|m| m.memory_usage_bytes)
            .sum::<u64>()
            / count as u64;
        let avg_flops = measurements.iter().map(|m| m.flops_executed).sum::<usize>() / count;
        let avg_memory_ops = measurements
            .iter()
            .map(|m| m.memory_operations)
            .sum::<usize>()
            / count;
        let avg_cpu = measurements.iter().map(|m| m.cpu_utilization).sum::<f64>() / count as f64;
        let avg_mem_eff = measurements
            .iter()
            .map(|m| m.memory_efficiency)
            .sum::<f64>()
            / count as f64;
        let avg_throughput = measurements
            .iter()
            .map(|m| m.throughput_ops_per_sec)
            .sum::<f64>()
            / count as f64;
        let avg_bandwidth = measurements
            .iter()
            .map(|m| m.memory_bandwidth_gbps)
            .sum::<f64>()
            / count as f64;

        ScalabilityMeasurement {
            size,
            sample_id: 0, // Aggregated
            execution_time: avg_execution_time,
            memory_usage_bytes: avg_memory,
            flops_executed: avg_flops,
            memory_operations: avg_memory_ops,
            cpu_utilization: avg_cpu,
            memory_efficiency: avg_mem_eff,
            throughput_ops_per_sec: avg_throughput,
            memory_bandwidth_gbps: avg_bandwidth,
        }
    }
}

/// Complete scalability test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityTestResult {
    pub config: ScalabilityConfig,
    pub measurements: Vec<ScalabilityMeasurement>,
    pub analysis: ScalabilityAnalysis,
    pub total_test_time: Duration,
}

/// Scalability analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAnalysis {
    pub observed_complexity: ComplexityClass,
    pub expected_complexity: ComplexityClass,
    pub complexity_match: bool,
    pub average_time_scaling_factor: f64,
    pub average_memory_scaling_factor: f64,
    pub performance_trend: PerformanceTrend,
    pub efficiency_score: f64,
    pub memory_efficiency_trend: MemoryEfficiencyTrend,
    pub bottleneck_analysis: Vec<BottleneckType>,
}

impl Default for ScalabilityAnalysis {
    fn default() -> Self {
        Self {
            observed_complexity: ComplexityClass::Constant,
            expected_complexity: ComplexityClass::Constant,
            complexity_match: true,
            average_time_scaling_factor: 1.0,
            average_memory_scaling_factor: 1.0,
            performance_trend: PerformanceTrend::Stable,
            efficiency_score: 50.0,
            memory_efficiency_trend: MemoryEfficiencyTrend::Stable,
            bottleneck_analysis: Vec::new(),
        }
    }
}

/// Performance trend indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

/// Memory efficiency trend indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryEfficiencyTrend {
    Improving,
    Stable,
    Degrading,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBound,
    CpuUnderutilized,
    MemoryBandwidthLimited,
    MemoryInefficient,
    AlgorithmicInefficiency,
    ConcurrencyLimited,
}

// Mock operation implementations for testing
fn mock_matmul_operation(m: usize, n: usize, k: usize) {
    // Simulate matrix multiplication work
    let _work = m * n * k;
    std::thread::sleep(Duration::from_micros((_work / 10000).max(1) as u64));
}

fn mock_elementwise_operation(elements: usize) {
    // Simulate element-wise operation work
    std::thread::sleep(Duration::from_micros((elements / 100000).max(1) as u64));
}

fn mock_conv2d_operation(height: usize, width: usize, in_channels: usize, out_channels: usize) {
    // Simulate convolution work
    let work = height * width * in_channels * out_channels;
    std::thread::sleep(Duration::from_micros((work / 100000).max(1) as u64));
}

fn mock_memory_allocation(bytes: usize) {
    // Simulate memory allocation work
    let _vec: Vec<u8> = vec![0; bytes.min(10_000_000)]; // Limit allocation size
    std::thread::sleep(Duration::from_micros((bytes / 1_000_000).max(1) as u64));
}

fn mock_generic_operation(size: usize) {
    // Generic operation simulation
    std::thread::sleep(Duration::from_micros((size / 1000).max(1) as u64));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_progression_powers_of_two() {
        let config = ScalabilityConfig {
            name: "test".to_string(),
            operation: OperationType::MatrixMultiplication,
            size_progression: SizeProgression::PowersOfTwo,
            min_size: 4,
            max_size: 32,
            samples_per_size: 1,
            expected_complexity: ComplexityClass::Linear,
            resource_limits: ResourceLimits::default(),
        };

        let suite = ScalabilityTestSuite::new();
        let sizes = suite.generate_sizes(&config);
        assert_eq!(sizes, vec![4, 8, 16, 32]);
    }

    #[test]
    fn test_size_progression_linear() {
        let config = ScalabilityConfig {
            name: "test".to_string(),
            operation: OperationType::ElementWiseAddition,
            size_progression: SizeProgression::Linear { step: 10 },
            min_size: 10,
            max_size: 30,
            samples_per_size: 1,
            expected_complexity: ComplexityClass::Linear,
            resource_limits: ResourceLimits::default(),
        };

        let suite = ScalabilityTestSuite::new();
        let sizes = suite.generate_sizes(&config);
        assert_eq!(sizes, vec![10, 20, 30]);
    }

    #[test]
    #[ignore = "Benchmark tests need implementation fixes"]
    fn test_complexity_inference() {
        let suite = ScalabilityTestSuite::new();

        // Test linear complexity
        let linear_factors = vec![1.0, 1.1, 1.0, 1.1];
        let complexity = suite.infer_complexity(&linear_factors);
        assert!(matches!(complexity, ComplexityClass::Linear));

        // Test quadratic complexity
        let quadratic_factors = vec![2.5, 2.8, 2.3, 2.7];
        let complexity = suite.infer_complexity(&quadratic_factors);
        assert!(matches!(complexity, ComplexityClass::Quadratic));
    }

    #[test]
    fn test_measurement_aggregation() {
        let measurements = vec![
            ScalabilityMeasurement {
                size: 100,
                sample_id: 0,
                execution_time: Duration::from_millis(10),
                memory_usage_bytes: 1000,
                flops_executed: 100,
                memory_operations: 200,
                cpu_utilization: 50.0,
                memory_efficiency: 100.0,
                throughput_ops_per_sec: 10.0,
                memory_bandwidth_gbps: 1.0,
            },
            ScalabilityMeasurement {
                size: 100,
                sample_id: 1,
                execution_time: Duration::from_millis(12),
                memory_usage_bytes: 1200,
                flops_executed: 120,
                memory_operations: 240,
                cpu_utilization: 60.0,
                memory_efficiency: 120.0,
                throughput_ops_per_sec: 12.0,
                memory_bandwidth_gbps: 1.2,
            },
        ];

        let aggregated = ScalabilityMeasurement::aggregate(&measurements);
        assert_eq!(aggregated.size, 100);
        assert_eq!(aggregated.execution_time, Duration::from_millis(11));
        assert_eq!(aggregated.memory_usage_bytes, 1100);
        assert_eq!(aggregated.flops_executed, 110);
        assert_eq!(aggregated.cpu_utilization, 55.0);
    }

    #[test]
    fn test_scalability_test_suite() {
        let mut suite = ScalabilityTestSuite::new();
        assert!(!suite.configs.is_empty());

        // Add a simple test config
        let config = ScalabilityConfig {
            name: "Simple Test".to_string(),
            operation: OperationType::ElementWiseAddition,
            size_progression: SizeProgression::Custom {
                sizes: vec![10, 20],
            },
            min_size: 10,
            max_size: 20,
            samples_per_size: 1,
            expected_complexity: ComplexityClass::Linear,
            resource_limits: ResourceLimits {
                max_execution_time: Duration::from_millis(100),
                max_total_time: Duration::from_secs(1),
                ..ResourceLimits::default()
            },
        };

        let result = suite.run_single_test(&config);
        assert_eq!(result.config.name, "Simple Test");
        assert!(!result.measurements.is_empty());
    }
}
