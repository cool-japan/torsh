//! Package Diagnostics and Optimization Example
//!
//! This example demonstrates how to use the diagnostics and optimization
//! features to analyze package health and identify optimization opportunities.

use torsh_package::{
    DiagnosticReport, HealthStatus, OptimizationReport, Package, PackageBuilder,
    PackageDiagnostics, PackageOptimizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Package Diagnostics and Optimization Example ===\n");

    // Create an example package
    println!("Step 1: Creating example package...");
    let package = create_test_package()?;
    println!(
        "✓ Package created: {} v{}\n",
        package.name(),
        package.get_version()
    );

    // Run diagnostics
    println!("Step 2: Running comprehensive diagnostics...");
    let diagnostic_report = run_diagnostics(&package)?;
    print_diagnostic_report(&diagnostic_report);

    // Run optimization analysis
    println!("\nStep 3: Analyzing optimization opportunities...");
    let optimization_report = analyze_optimizations(&package)?;
    print_optimization_report(&optimization_report);

    // Show recommendations
    println!("\nStep 4: Generating recommendations...");
    generate_recommendations(&diagnostic_report, &optimization_report);

    println!("\n=== Analysis completed successfully! ===");
    Ok(())
}

/// Create a test package for demonstration
fn create_test_package() -> Result<Package, Box<dyn std::error::Error>> {
    let package = PackageBuilder::new("example-model".to_string(), "2.0.0".to_string())
        .author("Demo Team".to_string())
        .description("A demonstration package for diagnostics".to_string())
        .license("MIT".to_string())
        .add_dependency("torsh-core", "0.1.0")
        .add_dependency("torsh-nn", "0.1.0")
        .package();

    Ok(package)
}

/// Run comprehensive diagnostics on the package
fn run_diagnostics(package: &Package) -> Result<DiagnosticReport, Box<dyn std::error::Error>> {
    let diagnostics = PackageDiagnostics::new();
    let report = diagnostics.diagnose(package)?;
    Ok(report)
}

/// Analyze optimization opportunities
fn analyze_optimizations(
    package: &Package,
) -> Result<OptimizationReport, Box<dyn std::error::Error>> {
    let optimizer = PackageOptimizer::new();
    let report = optimizer.analyze(package)?;
    Ok(report)
}

/// Print diagnostic report
fn print_diagnostic_report(report: &DiagnosticReport) {
    println!("\n=== Diagnostic Report ===");

    // Overall health
    println!("\nPackage Health:");
    println!("  Status: {:?}", report.status);
    println!("  Health Score: {}/100", report.health_score);

    let health_indicator = match report.status {
        HealthStatus::Healthy => "✓ HEALTHY",
        HealthStatus::Warning => "⚠ WARNING",
        HealthStatus::Degraded => "⚠ DEGRADED",
        HealthStatus::Critical => "✗ CRITICAL",
    };
    println!("  Indicator: {}", health_indicator);

    // Metadata validation
    println!("\nMetadata Validation:");
    if report.metadata_validation.passed {
        println!("  ✓ PASSED");
    } else {
        println!("  ✗ FAILED");
    }
    for message in &report.metadata_validation.messages {
        println!("    - {}", message);
    }

    // Issues
    if !report.issues.is_empty() {
        println!("\nDetected Issues ({} total):", report.issues.len());
        for (i, issue) in report.issues.iter().take(5).enumerate() {
            println!(
                "\n  Issue #{}: [{:?}] {:?}",
                i + 1,
                issue.severity,
                issue.category
            );
            println!("    Description: {}", issue.description);
            println!("    Recommendation: {}", issue.recommendation);
            if !issue.affected.is_empty() {
                println!("    Affected: {}", issue.affected.join(", "));
            }
        }
        if report.issues.len() > 5 {
            println!("\n  ... and {} more issues", report.issues.len() - 5);
        }
    } else {
        println!("\n  ✓ No issues detected");
    }

    // Security assessment
    println!("\nSecurity Assessment:");
    println!("  Security Score: {}/100", report.security.security_score);
    println!(
        "  Signed: {}",
        if report.security.is_signed {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "  Encrypted: {}",
        if report.security.is_encrypted {
            "Yes"
        } else {
            "No"
        }
    );

    if !report.security.issues.is_empty() {
        println!("  Security Issues:");
        for issue in &report.security.issues {
            println!("    - {}", issue);
        }
    }

    // Statistics
    println!("\nPackage Statistics:");
    println!("  Total Size: {} bytes", report.statistics.total_size);
    println!("  Resources: {}", report.statistics.resource_count);
    println!("  Dependencies: {}", report.statistics.dependency_count);
}

/// Print optimization report
fn print_optimization_report(report: &OptimizationReport) {
    println!("\n=== Optimization Report ===");

    // Size analysis
    println!("\nSize Analysis:");
    println!("  Original Size: {} bytes", report.original_size);
    println!("  Optimized Size: {} bytes", report.optimized_size);
    println!(
        "  Potential Savings: {} bytes ({:.1}%)",
        report.savings, report.savings_percent
    );

    // Deduplication analysis
    if report.deduplication.duplicate_count > 0 {
        println!("\nDeduplication Analysis:");
        println!(
            "  Total Resources: {}",
            report.deduplication.total_resources
        );
        println!(
            "  Unique Resources: {}",
            report.deduplication.unique_resources
        );
        println!(
            "  Duplicate Resources: {}",
            report.deduplication.duplicate_count
        );
        println!(
            "  Potential Savings: {} bytes",
            report.deduplication.potential_savings
        );

        if !report.deduplication.duplicate_groups.is_empty() {
            println!("\n  Duplicate Groups:");
            for (hash, resources) in report.deduplication.duplicate_groups.iter().take(3) {
                println!(
                    "    Hash {}... ({} duplicates):",
                    &hash[..8],
                    resources.len()
                );
                for resource in resources.iter().take(3) {
                    println!("      - {}", resource);
                }
            }
        }
    }

    // Compression analysis
    println!("\nCompression Analysis:");
    println!(
        "  Well Compressed: {} resources",
        report.compression.well_compressed_count
    );
    println!(
        "  Compressible: {} resources",
        report.compression.compressible_resources.len()
    );
    println!(
        "  Potential Savings: {} bytes",
        report.compression.potential_savings
    );

    if !report.compression.compressible_resources.is_empty() {
        println!("\n  Top Compressible Resources:");
        for (i, resource) in report
            .compression
            .compressible_resources
            .iter()
            .take(5)
            .enumerate()
        {
            println!(
                "    {}. {} ({} bytes → {} bytes, {:.1}% savings)",
                i + 1,
                resource.name,
                resource.current_size,
                resource.estimated_compressed_size,
                (1.0 - resource.compression_ratio) * 100.0
            );
        }
    }

    // Optimization opportunities
    if !report.opportunities.is_empty() {
        println!(
            "\nOptimization Opportunities ({} total):",
            report.opportunities.len()
        );
        for (i, opportunity) in report.opportunities.iter().take(5).enumerate() {
            println!(
                "\n  Opportunity #{} (Priority: {}/5):",
                i + 1,
                opportunity.priority
            );
            println!("    Type: {:?}", opportunity.optimization_type);
            println!("    Description: {}", opportunity.description);
            println!(
                "    Potential Savings: {} bytes",
                opportunity.potential_savings
            );
            if !opportunity.affected_resources.is_empty() {
                println!(
                    "    Affected: {} resources",
                    opportunity.affected_resources.len()
                );
            }
        }
        if report.opportunities.len() > 5 {
            println!(
                "\n  ... and {} more opportunities",
                report.opportunities.len() - 5
            );
        }
    } else {
        println!("\n  ✓ Package is already well-optimized");
    }
}

/// Generate recommendations based on reports
fn generate_recommendations(diagnostic: &DiagnosticReport, optimization: &OptimizationReport) {
    println!("\n=== Recommendations ===\n");

    let mut recommendations = Vec::new();

    // Health-based recommendations
    match diagnostic.status {
        HealthStatus::Critical | HealthStatus::Degraded => {
            recommendations.push((
                "Critical",
                "Address critical and high-severity issues immediately to improve package health",
            ));
        }
        HealthStatus::Warning => {
            recommendations.push((
                "Important",
                "Review and fix warning-level issues to maintain package quality",
            ));
        }
        HealthStatus::Healthy => {}
    }

    // Security recommendations
    if !diagnostic.security.is_signed {
        recommendations.push((
            "Security",
            "Sign the package to ensure authenticity and prevent tampering",
        ));
    }

    if diagnostic.security.security_score < 70 {
        recommendations.push((
            "Security",
            "Improve security measures to achieve at least a 70/100 security score",
        ));
    }

    // Optimization recommendations
    if optimization.savings_percent > 10.0 {
        let msg = format!(
            "Package can be optimized by {:.1}% - consider applying optimizations",
            optimization.savings_percent
        );
        recommendations.push(("Optimization", Box::leak(msg.into_boxed_str())));
    }

    if optimization.deduplication.duplicate_count > 0 {
        recommendations.push((
            "Deduplication",
            "Remove duplicate resources to reduce package size",
        ));
    }

    if !optimization.compression.compressible_resources.is_empty() {
        recommendations.push((
            "Compression",
            "Apply compression to large resources to reduce package size",
        ));
    }

    // Print recommendations
    if recommendations.is_empty() {
        println!("  ✓ No recommendations - package is in excellent condition!");
    } else {
        for (i, (category, recommendation)) in recommendations.iter().enumerate() {
            println!("  {}. [{}] {}", i + 1, category, recommendation);
        }
    }

    // Summary
    println!("\nSummary:");
    println!("  Health Status: {:?}", diagnostic.status);
    println!(
        "  Security Score: {}/100",
        diagnostic.security.security_score
    );
    println!(
        "  Optimization Potential: {:.1}%",
        optimization.savings_percent
    );
    println!("  Total Recommendations: {}", recommendations.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_package() {
        let package = create_test_package().unwrap();
        assert_eq!(package.name(), "example-model");
        assert_eq!(package.get_version(), "2.0.0");
    }
}
