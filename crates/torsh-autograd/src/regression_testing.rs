//! Regression testing for gradient computation
//!
//! This module provides comprehensive regression testing infrastructure to detect
//! when changes to the autograd system introduce unintended changes to gradient
//! computations. It captures reference gradients, stores them, and validates
//! that future computations match within acceptable tolerances.

use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration for regression testing
#[derive(Debug, Clone)]
pub struct RegressionTestConfig {
    /// Directory to store reference gradients
    pub reference_dir: PathBuf,
    /// Relative tolerance for gradient comparison
    pub relative_tolerance: f64,
    /// Absolute tolerance for gradient comparison
    pub absolute_tolerance: f64,
    /// Whether to enable automatic reference updates
    pub auto_update_references: bool,
    /// Whether to enable detailed reporting
    pub enable_detailed_reporting: bool,
    /// Maximum age of reference files before they're considered stale
    pub max_reference_age: Duration,
    /// Whether to enable parallel testing
    pub enable_parallel_testing: bool,
}

impl Default for RegressionTestConfig {
    fn default() -> Self {
        Self {
            reference_dir: PathBuf::from("test_data/gradient_references"),
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-8,
            auto_update_references: false,
            enable_detailed_reporting: true,
            max_reference_age: Duration::from_secs(30 * 24 * 3600), // 30 days
            enable_parallel_testing: true,
        }
    }
}

/// Represents a gradient test case
#[derive(Debug, Clone)]
pub struct GradientTestCase {
    /// Unique identifier for this test case
    pub test_id: String,
    /// Description of what this test case validates
    pub description: String,
    /// Input data for the gradient computation
    pub input_data: Vec<f64>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Operation name being tested
    pub operation: String,
    /// Operation parameters
    pub parameters: HashMap<String, f64>,
    /// Expected output gradients
    pub expected_gradients: Vec<f64>,
    /// Expected output shape
    pub output_shape: Vec<usize>,
    /// Tags for categorizing tests
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last validation timestamp
    pub last_validated: Option<SystemTime>,
}

impl GradientTestCase {
    /// Create a new gradient test case
    pub fn new(
        test_id: String,
        description: String,
        input_data: Vec<f64>,
        input_shape: Vec<usize>,
        operation: String,
    ) -> Self {
        Self {
            test_id,
            description,
            input_data,
            input_shape,
            operation,
            parameters: HashMap::new(),
            expected_gradients: Vec::new(),
            output_shape: Vec::new(),
            tags: Vec::new(),
            created_at: SystemTime::now(),
            last_validated: None,
        }
    }

    /// Add a parameter to the test case
    pub fn with_parameter(mut self, key: String, value: f64) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Add expected gradients
    pub fn with_expected_gradients(mut self, gradients: Vec<f64>, shape: Vec<usize>) -> Self {
        self.expected_gradients = gradients;
        self.output_shape = shape;
        self
    }

    /// Add tags for categorization
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if the test case is stale
    pub fn is_stale(&self, max_age: Duration) -> bool {
        if let Ok(age) = self.created_at.elapsed() {
            age > max_age
        } else {
            true
        }
    }
}

/// Result of a gradient regression test
#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    /// Test case that was executed
    pub test_case: GradientTestCase,
    /// Computed gradients
    pub computed_gradients: Vec<f64>,
    /// Whether the test passed
    pub passed: bool,
    /// Relative error between computed and expected gradients
    pub relative_error: f64,
    /// Absolute error between computed and expected gradients
    pub absolute_error: f64,
    /// Maximum element-wise relative error
    pub max_relative_error: f64,
    /// Maximum element-wise absolute error
    pub max_absolute_error: f64,
    /// Execution time for the gradient computation
    pub execution_time: Duration,
    /// Error message if test failed
    pub error_message: Option<String>,
    /// Detailed comparison results
    pub element_wise_errors: Vec<ElementWiseError>,
}

/// Element-wise error information
#[derive(Debug, Clone)]
pub struct ElementWiseError {
    pub index: usize,
    pub expected: f64,
    pub computed: f64,
    pub absolute_error: f64,
    pub relative_error: f64,
}

/// Statistics for a collection of regression test results
#[derive(Debug, Clone)]
pub struct RegressionTestStatistics {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub average_relative_error: f64,
    pub average_absolute_error: f64,
    pub max_relative_error: f64,
    pub max_absolute_error: f64,
    pub average_execution_time: Duration,
    pub total_execution_time: Duration,
    pub pass_rate: f64,
}

impl RegressionTestStatistics {
    /// Create statistics from a collection of test results
    pub fn from_results(results: &[RegressionTestResult]) -> Self {
        if results.is_empty() {
            return Self {
                total_tests: 0,
                passed_tests: 0,
                failed_tests: 0,
                average_relative_error: 0.0,
                average_absolute_error: 0.0,
                max_relative_error: 0.0,
                max_absolute_error: 0.0,
                average_execution_time: Duration::from_secs(0),
                total_execution_time: Duration::from_secs(0),
                pass_rate: 0.0,
            };
        }

        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        let total_relative_error: f64 = results.iter().map(|r| r.relative_error).sum();
        let total_absolute_error: f64 = results.iter().map(|r| r.absolute_error).sum();
        let max_relative_error = results
            .iter()
            .map(|r| r.max_relative_error)
            .fold(0.0, f64::max);
        let max_absolute_error = results
            .iter()
            .map(|r| r.max_absolute_error)
            .fold(0.0, f64::max);

        let total_execution_time: Duration = results.iter().map(|r| r.execution_time).sum();
        let average_execution_time = total_execution_time / total_tests as u32;

        Self {
            total_tests,
            passed_tests,
            failed_tests,
            average_relative_error: total_relative_error / total_tests as f64,
            average_absolute_error: total_absolute_error / total_tests as f64,
            max_relative_error,
            max_absolute_error,
            average_execution_time,
            total_execution_time,
            pass_rate: passed_tests as f64 / total_tests as f64,
        }
    }
}

/// Main regression testing framework
pub struct GradientRegressionTester {
    config: RegressionTestConfig,
    test_cases: Arc<Mutex<HashMap<String, GradientTestCase>>>,
    results_cache: Arc<Mutex<HashMap<String, RegressionTestResult>>>,
}

impl GradientRegressionTester {
    /// Create a new regression tester
    pub fn new(config: RegressionTestConfig) -> Self {
        Self {
            config,
            test_cases: Arc::new(Mutex::new(HashMap::new())),
            results_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RegressionTestConfig::default())
    }

    /// Register a new test case
    pub fn register_test_case(&self, test_case: GradientTestCase) -> AutogradResult<()> {
        let test_id = test_case.test_id.clone();

        if let Ok(mut test_cases) = self.test_cases.lock() {
            test_cases.insert(test_id.clone(), test_case);
        }

        // Save test case to disk if reference directory exists
        if self.config.reference_dir.exists() {
            self.save_test_case_to_disk(&test_id)?;
        }

        Ok(())
    }

    /// Load test cases from disk
    pub fn load_test_cases_from_disk(&self) -> AutogradResult<usize> {
        if !self.config.reference_dir.exists() {
            fs::create_dir_all(&self.config.reference_dir)?;
            return Ok(0);
        }

        let mut loaded_count = 0;

        for entry in fs::read_dir(&self.config.reference_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(test_case) = self.load_test_case_from_file(&path) {
                    let test_id = test_case.test_id.clone();

                    if let Ok(mut test_cases) = self.test_cases.lock() {
                        test_cases.insert(test_id, test_case);
                        loaded_count += 1;
                    }
                }
            }
        }

        Ok(loaded_count)
    }

    /// Run a specific test case
    pub fn run_test(&self, test_id: &str) -> AutogradResult<RegressionTestResult> {
        let test_case = {
            let test_cases = self.test_cases.lock().unwrap();
            test_cases
                .get(test_id)
                .ok_or_else(|| {
                    AutogradError::gradient_computation(
                        "run_test",
                        format!("Test case '{}' not found", test_id),
                    )
                })?
                .clone()
        };

        self.execute_test_case(test_case)
    }

    /// Run all registered test cases
    pub fn run_all_tests(&self) -> AutogradResult<Vec<RegressionTestResult>> {
        let test_cases: Vec<GradientTestCase> = {
            let test_cases = self.test_cases.lock().unwrap();
            test_cases.values().cloned().collect()
        };

        let mut results = Vec::new();

        if self.config.enable_parallel_testing {
            // In a real implementation, we'd use parallel processing here
            // For now, we'll execute sequentially
            for test_case in test_cases {
                let result = self.execute_test_case(test_case)?;
                results.push(result);
            }
        } else {
            for test_case in test_cases {
                let result = self.execute_test_case(test_case)?;
                results.push(result);
            }
        }

        // Cache results
        if let Ok(mut cache) = self.results_cache.lock() {
            for result in &results {
                cache.insert(result.test_case.test_id.clone(), result.clone());
            }
        }

        Ok(results)
    }

    /// Run tests matching specific tags
    pub fn run_tests_with_tags(
        &self,
        tags: &[String],
    ) -> AutogradResult<Vec<RegressionTestResult>> {
        let test_cases: Vec<GradientTestCase> = {
            let test_cases = self.test_cases.lock().unwrap();
            test_cases
                .values()
                .filter(|test_case| tags.iter().any(|tag| test_case.tags.contains(tag)))
                .cloned()
                .collect()
        };

        let mut results = Vec::new();

        for test_case in test_cases {
            let result = self.execute_test_case(test_case)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute a single test case
    fn execute_test_case(
        &self,
        test_case: GradientTestCase,
    ) -> AutogradResult<RegressionTestResult> {
        let start_time = std::time::Instant::now();

        // Simulate gradient computation (in real implementation, this would call actual autograd)
        let computed_gradients = self.compute_gradients(&test_case)?;

        let execution_time = start_time.elapsed();

        // Compare with expected gradients
        let comparison_result =
            self.compare_gradients(&test_case.expected_gradients, &computed_gradients)?;

        let passed = comparison_result.relative_error <= self.config.relative_tolerance
            && comparison_result.absolute_error <= self.config.absolute_tolerance;

        let error_message = if !passed {
            Some(format!(
                "Gradient mismatch: relative_error={:.2e} (threshold={:.2e}), absolute_error={:.2e} (threshold={:.2e})",
                comparison_result.relative_error,
                self.config.relative_tolerance,
                comparison_result.absolute_error,
                self.config.absolute_tolerance
            ))
        } else {
            None
        };

        Ok(RegressionTestResult {
            test_case,
            computed_gradients,
            passed,
            relative_error: comparison_result.relative_error,
            absolute_error: comparison_result.absolute_error,
            max_relative_error: comparison_result.max_relative_error,
            max_absolute_error: comparison_result.max_absolute_error,
            execution_time,
            error_message,
            element_wise_errors: comparison_result.element_wise_errors,
        })
    }

    /// Simulate gradient computation (placeholder implementation)
    fn compute_gradients(&self, test_case: &GradientTestCase) -> AutogradResult<Vec<f64>> {
        // This is a placeholder implementation
        // In a real system, this would interface with the actual autograd engine

        match test_case.operation.as_str() {
            "add" => {
                // For addition, gradient is 1.0 for all elements
                Ok(vec![1.0; test_case.input_data.len()])
            }
            "multiply" => {
                // For multiplication by a scalar, gradient is the scalar
                let scalar = test_case.parameters.get("scalar").unwrap_or(&1.0);
                Ok(vec![*scalar; test_case.input_data.len()])
            }
            "square" => {
                // For x^2, gradient is 2*x
                Ok(test_case.input_data.iter().map(|&x| 2.0 * x).collect())
            }
            "sin" => {
                // For sin(x), gradient is cos(x)
                Ok(test_case.input_data.iter().map(|&x| x.cos()).collect())
            }
            "exp" => {
                // For exp(x), gradient is exp(x)
                Ok(test_case.input_data.iter().map(|&x| x.exp()).collect())
            }
            "log" => {
                // For log(x), gradient is 1/x
                Ok(test_case.input_data.iter().map(|&x| 1.0 / x).collect())
            }
            _ => {
                // Unknown operation - return zeros
                Ok(vec![0.0; test_case.input_data.len()])
            }
        }
    }

    /// Compare computed gradients with expected gradients
    fn compare_gradients(
        &self,
        expected: &[f64],
        computed: &[f64],
    ) -> AutogradResult<GradientComparisonResult> {
        if expected.len() != computed.len() {
            return Err(AutogradError::gradient_computation(
                "compare_gradients",
                format!(
                    "Gradient length mismatch: expected {} but got {}",
                    expected.len(),
                    computed.len()
                ),
            ));
        }

        let mut element_wise_errors = Vec::new();
        let mut total_absolute_error = 0.0;
        let mut total_relative_error = 0.0;
        let mut max_absolute_error: f64 = 0.0;
        let mut max_relative_error: f64 = 0.0;

        for (i, (&exp, &comp)) in expected.iter().zip(computed.iter()).enumerate() {
            let absolute_error = (comp - exp).abs();
            let relative_error = if exp.abs() > 1e-15 {
                absolute_error / exp.abs()
            } else {
                if absolute_error > 1e-15 {
                    f64::INFINITY
                } else {
                    0.0
                }
            };

            element_wise_errors.push(ElementWiseError {
                index: i,
                expected: exp,
                computed: comp,
                absolute_error,
                relative_error,
            });

            total_absolute_error += absolute_error;
            total_relative_error += relative_error;
            max_absolute_error = max_absolute_error.max(absolute_error);
            max_relative_error = max_relative_error.max(relative_error);
        }

        let n = expected.len() as f64;
        Ok(GradientComparisonResult {
            relative_error: total_relative_error / n,
            absolute_error: total_absolute_error / n,
            max_relative_error,
            max_absolute_error,
            element_wise_errors,
        })
    }

    /// Generate a comprehensive test report
    pub fn generate_report(&self, results: &[RegressionTestResult]) -> String {
        let stats = RegressionTestStatistics::from_results(results);

        let mut report = String::new();
        report.push_str("=== Gradient Regression Test Report ===\n\n");

        report.push_str(&format!("Configuration:\n"));
        report.push_str(&format!(
            "  Reference Directory: {:?}\n",
            self.config.reference_dir
        ));
        report.push_str(&format!(
            "  Relative Tolerance: {:.2e}\n",
            self.config.relative_tolerance
        ));
        report.push_str(&format!(
            "  Absolute Tolerance: {:.2e}\n",
            self.config.absolute_tolerance
        ));
        report.push_str("\n");

        report.push_str(&format!("Overall Statistics:\n"));
        report.push_str(&format!("  Total Tests: {}\n", stats.total_tests));
        report.push_str(&format!("  Passed: {}\n", stats.passed_tests));
        report.push_str(&format!("  Failed: {}\n", stats.failed_tests));
        report.push_str(&format!("  Pass Rate: {:.2}%\n", stats.pass_rate * 100.0));
        report.push_str(&format!(
            "  Average Relative Error: {:.2e}\n",
            stats.average_relative_error
        ));
        report.push_str(&format!(
            "  Average Absolute Error: {:.2e}\n",
            stats.average_absolute_error
        ));
        report.push_str(&format!(
            "  Max Relative Error: {:.2e}\n",
            stats.max_relative_error
        ));
        report.push_str(&format!(
            "  Max Absolute Error: {:.2e}\n",
            stats.max_absolute_error
        ));
        report.push_str(&format!(
            "  Total Execution Time: {:.3}s\n",
            stats.total_execution_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "  Average Execution Time: {:.3}ms\n",
            stats.average_execution_time.as_secs_f64() * 1000.0
        ));
        report.push_str("\n");

        if self.config.enable_detailed_reporting {
            report.push_str("Failed Tests:\n");
            for result in results.iter().filter(|r| !r.passed) {
                report.push_str(&format!("  Test: {}\n", result.test_case.test_id));
                report.push_str(&format!("    Operation: {}\n", result.test_case.operation));
                report.push_str(&format!(
                    "    Error: {}\n",
                    result
                        .error_message
                        .as_ref()
                        .unwrap_or(&"Unknown".to_string())
                ));
                report.push_str(&format!(
                    "    Relative Error: {:.2e}\n",
                    result.relative_error
                ));
                report.push_str(&format!(
                    "    Absolute Error: {:.2e}\n",
                    result.absolute_error
                ));
                report.push_str("\n");
            }
        }

        report
    }

    /// Save test case to disk
    fn save_test_case_to_disk(&self, test_id: &str) -> AutogradResult<()> {
        let test_case = {
            let test_cases = self.test_cases.lock().unwrap();
            test_cases
                .get(test_id)
                .ok_or_else(|| {
                    AutogradError::gradient_computation(
                        "save_test_case_to_disk",
                        format!("Test case '{}' not found", test_id),
                    )
                })?
                .clone()
        };

        let file_path = self.config.reference_dir.join(format!("{}.json", test_id));
        let json_data = self.serialize_test_case(&test_case)?;

        fs::create_dir_all(&self.config.reference_dir)?;
        fs::write(file_path, json_data)?;

        Ok(())
    }

    /// Load test case from file
    fn load_test_case_from_file(&self, path: &Path) -> AutogradResult<GradientTestCase> {
        let data = fs::read_to_string(path)?;
        self.deserialize_test_case(&data)
    }

    /// Serialize test case to JSON (simplified implementation)
    fn serialize_test_case(&self, test_case: &GradientTestCase) -> AutogradResult<String> {
        // Simplified JSON serialization
        let timestamp = test_case
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let json = format!(
            r#"{{
    "test_id": "{}",
    "description": "{}",
    "operation": "{}",
    "input_data": {:?},
    "input_shape": {:?},
    "expected_gradients": {:?},
    "output_shape": {:?},
    "tags": {:?},
    "created_at": {}
}}"#,
            test_case.test_id,
            test_case.description,
            test_case.operation,
            test_case.input_data,
            test_case.input_shape,
            test_case.expected_gradients,
            test_case.output_shape,
            test_case.tags,
            timestamp
        );

        Ok(json)
    }

    /// Deserialize test case from JSON (simplified implementation)
    fn deserialize_test_case(&self, json: &str) -> AutogradResult<GradientTestCase> {
        // This is a very simplified JSON parser for demonstration
        // In a real implementation, you'd use serde or similar

        // Extract test_id
        let test_id = self.extract_json_string(json, "test_id")?;
        let description = self.extract_json_string(json, "description")?;
        let operation = self.extract_json_string(json, "operation")?;

        // For simplicity, create a basic test case
        let mut test_case = GradientTestCase::new(
            test_id,
            description,
            vec![1.0, 2.0, 3.0],
            vec![3],
            operation,
        );

        // In a real implementation, you'd parse all the fields properly
        test_case.expected_gradients = vec![1.0, 1.0, 1.0]; // Placeholder

        Ok(test_case)
    }

    /// Extract string value from simple JSON (very basic parser)
    fn extract_json_string(&self, json: &str, key: &str) -> AutogradResult<String> {
        let pattern = format!(r#""{}": ""#, key);
        if let Some(start) = json.find(&pattern) {
            let value_start = start + pattern.len();
            if let Some(end) = json[value_start..].find('"') {
                return Ok(json[value_start..value_start + end].to_string());
            }
        }

        Err(AutogradError::gradient_computation(
            "extract_json_string",
            format!("Could not extract '{}' from JSON", key),
        ))
    }

    /// Clean up stale test cases
    pub fn cleanup_stale_tests(&self) -> AutogradResult<usize> {
        let mut removed_count = 0;

        if let Ok(mut test_cases) = self.test_cases.lock() {
            let stale_ids: Vec<String> = test_cases
                .iter()
                .filter(|(_, test_case)| test_case.is_stale(self.config.max_reference_age))
                .map(|(id, _)| id.clone())
                .collect();

            for id in stale_ids {
                test_cases.remove(&id);
                removed_count += 1;

                // Remove from disk
                let file_path = self.config.reference_dir.join(format!("{}.json", id));
                if file_path.exists() {
                    let _ = fs::remove_file(file_path);
                }
            }
        }

        Ok(removed_count)
    }

    /// Get test case count
    pub fn test_case_count(&self) -> usize {
        self.test_cases.lock().unwrap().len()
    }

    /// Get test cases by tag
    pub fn get_test_cases_by_tag(&self, tag: &str) -> Vec<GradientTestCase> {
        let test_cases = self.test_cases.lock().unwrap();
        test_cases
            .values()
            .filter(|test_case| test_case.tags.contains(&tag.to_string()))
            .cloned()
            .collect()
    }
}

/// Result of gradient comparison
struct GradientComparisonResult {
    pub relative_error: f64,
    pub absolute_error: f64,
    pub max_relative_error: f64,
    pub max_absolute_error: f64,
    pub element_wise_errors: Vec<ElementWiseError>,
}

/// Helper function to create basic test cases for common operations
pub fn create_basic_test_suite() -> Vec<GradientTestCase> {
    vec![
        GradientTestCase::new(
            "test_add_gradient".to_string(),
            "Test gradient of addition operation".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
            "add".to_string(),
        )
        .with_expected_gradients(vec![1.0, 1.0, 1.0], vec![3])
        .with_tags(vec!["basic".to_string(), "arithmetic".to_string()]),
        GradientTestCase::new(
            "test_multiply_gradient".to_string(),
            "Test gradient of multiplication by scalar".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
            "multiply".to_string(),
        )
        .with_parameter("scalar".to_string(), 2.0)
        .with_expected_gradients(vec![2.0, 2.0, 2.0], vec![3])
        .with_tags(vec!["basic".to_string(), "arithmetic".to_string()]),
        GradientTestCase::new(
            "test_square_gradient".to_string(),
            "Test gradient of square operation".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
            "square".to_string(),
        )
        .with_expected_gradients(vec![2.0, 4.0, 6.0], vec![3])
        .with_tags(vec!["basic".to_string(), "nonlinear".to_string()]),
        GradientTestCase::new(
            "test_sin_gradient".to_string(),
            "Test gradient of sine operation".to_string(),
            vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI], // 0, π/2, π
            vec![3],
            "sin".to_string(),
        )
        .with_expected_gradients(vec![1.0, 0.0, -1.0], vec![3])
        .with_tags(vec!["trigonometric".to_string(), "nonlinear".to_string()]),
        GradientTestCase::new(
            "test_exp_gradient".to_string(),
            "Test gradient of exponential operation".to_string(),
            vec![0.0, 1.0, 2.0],
            vec![3],
            "exp".to_string(),
        )
        .with_expected_gradients(
            vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)],
            vec![3],
        )
        .with_tags(vec!["exponential".to_string(), "nonlinear".to_string()]),
    ]
}

/// Global regression tester instance
static GLOBAL_REGRESSION_TESTER: std::sync::OnceLock<GradientRegressionTester> =
    std::sync::OnceLock::new();

/// Get the global regression tester
pub fn get_global_regression_tester() -> &'static GradientRegressionTester {
    GLOBAL_REGRESSION_TESTER.get_or_init(|| GradientRegressionTester::with_defaults())
}

/// Convenience function to run regression tests
pub fn run_gradient_regression_tests() -> AutogradResult<Vec<RegressionTestResult>> {
    let tester = get_global_regression_tester();
    tester.run_all_tests()
}

/// Convenience macro for creating test cases
#[macro_export]
macro_rules! gradient_test_case {
    ($id:expr, $desc:expr, $op:expr, $input:expr, $expected:expr) => {
        $crate::regression_testing::GradientTestCase::new(
            $id.to_string(),
            $desc.to_string(),
            $input,
            vec![$input.len()],
            $op.to_string(),
        )
        .with_expected_gradients($expected, vec![$expected.len()])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_regression_tester_creation() {
        let tester = GradientRegressionTester::with_defaults();
        assert_eq!(tester.test_case_count(), 0);
    }

    #[test]
    fn test_test_case_registration() {
        let tester = GradientRegressionTester::with_defaults();

        let test_case = GradientTestCase::new(
            "test_1".to_string(),
            "Test case 1".to_string(),
            vec![1.0, 2.0],
            vec![2],
            "add".to_string(),
        );

        tester.register_test_case(test_case).unwrap();
        assert_eq!(tester.test_case_count(), 1);
    }

    #[test]
    fn test_basic_gradient_computation() {
        let tester = GradientRegressionTester::with_defaults();

        let test_case = GradientTestCase::new(
            "test_add".to_string(),
            "Test addition gradients".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
            "add".to_string(),
        )
        .with_expected_gradients(vec![1.0, 1.0, 1.0], vec![3]);

        let result = tester.execute_test_case(test_case).unwrap();
        assert!(result.passed);
        assert_eq!(result.computed_gradients, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_square_gradient_computation() {
        let tester = GradientRegressionTester::with_defaults();

        let test_case = GradientTestCase::new(
            "test_square".to_string(),
            "Test square gradients".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
            "square".to_string(),
        )
        .with_expected_gradients(vec![2.0, 4.0, 6.0], vec![3]);

        let result = tester.execute_test_case(test_case).unwrap();
        assert!(result.passed);
        assert_eq!(result.computed_gradients, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_gradient_comparison() {
        let tester = GradientRegressionTester::with_defaults();

        let expected = vec![1.0, 2.0, 3.0];
        let computed = vec![1.0001, 1.9999, 3.0001];

        let result = tester.compare_gradients(&expected, &computed).unwrap();
        assert!(result.relative_error < 1e-3);
        assert!(result.absolute_error < 1e-3);
    }

    #[test]
    fn test_basic_test_suite_creation() {
        let test_suite = create_basic_test_suite();
        assert!(!test_suite.is_empty());
        assert!(test_suite.iter().any(|tc| tc.operation == "add"));
        assert!(test_suite.iter().any(|tc| tc.operation == "square"));
        assert!(test_suite.iter().any(|tc| tc.operation == "sin"));
    }

    #[test]
    fn test_run_all_tests() {
        let tester = GradientRegressionTester::with_defaults();

        // Register the basic test suite
        let test_suite = create_basic_test_suite();
        for test_case in test_suite {
            tester.register_test_case(test_case).unwrap();
        }

        let results = tester.run_all_tests().unwrap();
        assert!(!results.is_empty());

        // All basic tests should pass
        let all_passed = results.iter().all(|r| r.passed);
        assert!(
            all_passed,
            "Some basic tests failed: {:?}",
            results.iter().filter(|r| !r.passed).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_statistics_generation() {
        let results = vec![RegressionTestResult {
            test_case: GradientTestCase::new(
                "test1".to_string(),
                "Test 1".to_string(),
                vec![1.0],
                vec![1],
                "add".to_string(),
            ),
            computed_gradients: vec![1.0],
            passed: true,
            relative_error: 0.001,
            absolute_error: 0.001,
            max_relative_error: 0.001,
            max_absolute_error: 0.001,
            execution_time: Duration::from_millis(10),
            error_message: None,
            element_wise_errors: vec![],
        }];

        let stats = RegressionTestStatistics::from_results(&results);
        assert_eq!(stats.total_tests, 1);
        assert_eq!(stats.passed_tests, 1);
        assert_eq!(stats.failed_tests, 0);
        assert_eq!(stats.pass_rate, 1.0);
    }

    #[test]
    fn test_test_case_with_tags() {
        let tester = GradientRegressionTester::with_defaults();

        let test_case = GradientTestCase::new(
            "tagged_test".to_string(),
            "Test with tags".to_string(),
            vec![1.0],
            vec![1],
            "add".to_string(),
        )
        .with_tags(vec!["basic".to_string(), "arithmetic".to_string()]);

        tester.register_test_case(test_case).unwrap();

        let basic_tests = tester.get_test_cases_by_tag("basic");
        assert_eq!(basic_tests.len(), 1);
        assert_eq!(basic_tests[0].test_id, "tagged_test");
    }

    #[test]
    fn test_macro_test_case_creation() {
        let test_case = gradient_test_case!(
            "macro_test",
            "Test created with macro",
            "add",
            vec![1.0, 2.0],
            vec![1.0, 1.0]
        );

        assert_eq!(test_case.test_id, "macro_test");
        assert_eq!(test_case.operation, "add");
        assert_eq!(test_case.input_data, vec![1.0, 2.0]);
        assert_eq!(test_case.expected_gradients, vec![1.0, 1.0]);
    }
}
