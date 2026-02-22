//! Comprehensive benchmark validation script
//!
//! This script validates all benchmark implementations and ensures they work correctly.
//! It should be run after fixing any compilation issues to verify functionality.

use std::fs;
use std::path::Path;
use std::process::Command;

/// Validation configuration
#[derive(Debug)]
pub struct ValidationConfig {
    pub run_unit_tests: bool,
    pub run_integration_tests: bool,
    pub run_benchmarks: bool,
    pub check_cross_framework: bool,
    pub generate_reports: bool,
    pub output_dir: String,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            run_unit_tests: true,
            run_integration_tests: true,
            run_benchmarks: true,
            check_cross_framework: true,
            generate_reports: true,
            output_dir: "validation_output".to_string(),
        }
    }
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub tests_passed: bool,
    pub benchmarks_passed: bool,
    pub cross_framework_passed: bool,
    pub warnings_count: usize,
    pub errors_count: usize,
    pub recommendations: Vec<String>,
}

/// Main validation function
pub fn validate_all(config: &ValidationConfig) -> Result<ValidationResult, Box<dyn std::error::Error>> {
    println!("🚀 Starting comprehensive benchmark validation...");
    
    // Create output directory
    fs::create_dir_all(&config.output_dir)?;
    
    let mut result = ValidationResult {
        tests_passed: false,
        benchmarks_passed: false,
        cross_framework_passed: false,
        warnings_count: 0,
        errors_count: 0,
        recommendations: Vec::new(),
    };
    
    // Step 1: Check compilation
    println!("📋 Step 1: Checking compilation status...");
    let compile_result = check_compilation()?;
    result.errors_count = compile_result.errors;
    result.warnings_count = compile_result.warnings;
    
    if compile_result.errors > 0 {
        result.recommendations.push(format!("Fix {} compilation errors before proceeding", compile_result.errors));
        return Ok(result);
    }
    
    // Step 2: Run unit tests
    if config.run_unit_tests {
        println!("🧪 Step 2: Running unit tests...");
        result.tests_passed = run_unit_tests()?;
        if !result.tests_passed {
            result.recommendations.push("Fix failing unit tests before proceeding".to_string());
        }
    }
    
    // Step 3: Run integration tests
    if config.run_integration_tests {
        println!("🔗 Step 3: Running integration tests...");
        let integration_passed = run_integration_tests()?;
        result.tests_passed = result.tests_passed && integration_passed;
    }
    
    // Step 4: Run benchmark validation
    if config.run_benchmarks {
        println!("⚡ Step 4: Validating benchmark implementations...");
        result.benchmarks_passed = validate_benchmarks()?;
        if !result.benchmarks_passed {
            result.recommendations.push("Fix benchmark implementations".to_string());
        }
    }
    
    // Step 5: Cross-framework validation
    if config.check_cross_framework {
        println!("🔄 Step 5: Validating cross-framework comparisons...");
        result.cross_framework_passed = validate_cross_framework()?;
        if !result.cross_framework_passed {
            result.recommendations.push("Check cross-framework comparison implementations".to_string());
        }
    }
    
    // Step 6: Generate reports
    if config.generate_reports {
        println!("📊 Step 6: Generating validation reports...");
        generate_validation_report(&result, &config.output_dir)?;
    }
    
    println!("✅ Validation complete!");
    Ok(result)
}

/// Check compilation status
fn check_compilation() -> Result<CompileResult, Box<dyn std::error::Error>> {
    println!("  - Running cargo check...");
    let output = Command::new("cargo")
        .args(&["check", "--all-features"])
        .output()?;
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Parse errors and warnings
    let errors = stderr.matches("error:").count();
    let warnings = stderr.matches("warning:").count();
    
    println!("  - Found {} errors, {} warnings", errors, warnings);
    
    Ok(CompileResult { errors, warnings })
}

#[derive(Debug)]
struct CompileResult {
    errors: usize,
    warnings: usize,
}

/// Run unit tests
fn run_unit_tests() -> Result<bool, Box<dyn std::error::Error>> {
    println!("  - Running cargo nextest run...");
    let output = Command::new("cargo")
        .args(&["nextest", "run", "--lib"])
        .output()?;
    
    let success = output.status.success();
    if success {
        println!("  - ✅ Unit tests passed");
    } else {
        println!("  - ❌ Unit tests failed");
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("    Error: {}", stderr);
    }
    
    Ok(success)
}

/// Run integration tests
fn run_integration_tests() -> Result<bool, Box<dyn std::error::Error>> {
    println!("  - Running integration tests...");
    let output = Command::new("cargo")
        .args(&["nextest", "run", "--test", "*"])
        .output()?;
    
    let success = output.status.success();
    if success {
        println!("  - ✅ Integration tests passed");
    } else {
        println!("  - ❌ Integration tests failed");
    }
    
    Ok(success)
}

/// Validate benchmark implementations
fn validate_benchmarks() -> Result<bool, Box<dyn std::error::Error>> {
    println!("  - Validating individual benchmark implementations...");
    
    // List of benchmark modules to validate
    let benchmark_modules = [
        "tensor_creation",
        "tensor_arithmetic", 
        "matrix_multiplication",
        "model_benchmarks",
        "hardware_benchmarks",
        "precision_benchmarks",
        "distributed_training",
        "edge_deployment",
        "mobile_benchmarks",
        "wasm_benchmarks",
        "custom_ops",
    ];
    
    let mut all_passed = true;
    
    for module in &benchmark_modules {
        println!("    - Validating {} benchmarks...", module);
        
        // Run specific benchmark tests
        let output = Command::new("cargo")
            .args(&["test", &format!("test_{}", module), "--", "--nocapture"])
            .output()?;
            
        if !output.status.success() {
            println!("      ❌ {} benchmark validation failed", module);
            all_passed = false;
        } else {
            println!("      ✅ {} benchmark validation passed", module);
        }
    }
    
    if all_passed {
        println!("  - ✅ All benchmark implementations validated");
    } else {
        println!("  - ❌ Some benchmark implementations failed validation");
    }
    
    Ok(all_passed)
}

/// Validate cross-framework comparisons
fn validate_cross_framework() -> Result<bool, Box<dyn std::error::Error>> {
    println!("  - Validating cross-framework comparison functionality...");
    
    // Check if PyTorch comparison features work
    let pytorch_available = check_pytorch_feature()?;
    println!("    - PyTorch comparisons: {}", if pytorch_available { "✅" } else { "❌" });
    
    // Check NumPy baseline comparisons
    let numpy_available = check_numpy_feature()?;
    println!("    - NumPy comparisons: {}", if numpy_available { "✅" } else { "❌" });
    
    // Validate comparison framework
    let comparison_tests_pass = run_comparison_tests()?;
    println!("    - Comparison framework tests: {}", if comparison_tests_pass { "✅" } else { "❌" });
    
    let all_passed = pytorch_available && numpy_available && comparison_tests_pass;
    
    if all_passed {
        println!("  - ✅ Cross-framework validation passed");
    } else {
        println!("  - ❌ Cross-framework validation failed");
    }
    
    Ok(all_passed)
}

/// Check PyTorch feature availability
fn check_pytorch_feature() -> Result<bool, Box<dyn std::error::Error>> {
    let output = Command::new("cargo")
        .args(&["test", "--features", "pytorch", "test_pytorch_comparison", "--", "--nocapture"])
        .output()?;
    
    Ok(output.status.success())
}

/// Check NumPy feature availability  
fn check_numpy_feature() -> Result<bool, Box<dyn std::error::Error>> {
    let output = Command::new("cargo")
        .args(&["test", "--features", "numpy_baseline", "test_numpy_comparison", "--", "--nocapture"])
        .output()?;
    
    Ok(output.status.success())
}

/// Run comparison framework tests
fn run_comparison_tests() -> Result<bool, Box<dyn std::error::Error>> {
    let output = Command::new("cargo")
        .args(&["test", "comparison", "--", "--nocapture"])
        .output()?;
    
    Ok(output.status.success())
}

/// Generate validation report
fn generate_validation_report(result: &ValidationResult, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let report_path = format!("{}/validation_report.md", output_dir);
    let mut report = fs::File::create(&report_path)?;
    
    writeln!(report, "# ToRSh Benchmarks Validation Report")?;
    writeln!(report, "")?;
    writeln!(report, "Generated on: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))?;
    writeln!(report, "")?;
    
    writeln!(report, "## Summary")?;
    writeln!(report, "")?;
    writeln!(report, "- **Unit Tests**: {}", if result.tests_passed { "✅ PASS" } else { "❌ FAIL" })?;
    writeln!(report, "- **Benchmarks**: {}", if result.benchmarks_passed { "✅ PASS" } else { "❌ FAIL" })?;
    writeln!(report, "- **Cross-Framework**: {}", if result.cross_framework_passed { "✅ PASS" } else { "❌ FAIL" })?;
    writeln!(report, "- **Compilation Errors**: {}", result.errors_count)?;
    writeln!(report, "- **Warnings**: {}", result.warnings_count)?;
    writeln!(report, "")?;
    
    if !result.recommendations.is_empty() {
        writeln!(report, "## Recommendations")?;
        writeln!(report, "")?;
        for rec in &result.recommendations {
            writeln!(report, "- {}", rec)?;
        }
        writeln!(report, "")?;
    }
    
    writeln!(report, "## Next Steps")?;
    writeln!(report, "")?;
    
    if result.errors_count > 0 {
        writeln!(report, "1. Fix compilation errors")?;
        writeln!(report, "2. Re-run validation")?;
    } else if result.warnings_count > 0 {
        writeln!(report, "1. Clean up {} warnings", result.warnings_count)?;
        writeln!(report, "2. Optimize benchmark implementations")?;
    } else {
        writeln!(report, "1. Benchmarking suite is ready for production use")?;
        writeln!(report, "2. Consider running performance regression tests")?;
        writeln!(report, "3. Set up continuous benchmarking in CI")?;
    }
    
    println!("📄 Validation report saved to: {}", report_path);
    Ok(())
}

/// Main entry point
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ValidationConfig::default();
    let result = validate_all(&config)?;
    
    println!("\n🎯 Validation Summary:");
    println!("- Tests: {}", if result.tests_passed { "✅" } else { "❌" });
    println!("- Benchmarks: {}", if result.benchmarks_passed { "✅" } else { "❌" });
    println!("- Cross-Framework: {}", if result.cross_framework_passed { "✅" } else { "❌" });
    println!("- Errors: {}", result.errors_count);
    println!("- Warnings: {}", result.warnings_count);
    
    if !result.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for rec in &result.recommendations {
            println!("  - {}", rec);
        }
    }
    
    Ok(())
}