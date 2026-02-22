//! Comprehensive Package Workflow Example
//!
//! This example demonstrates a complete workflow using multiple features:
//! - Package creation with profiling
//! - Validation utilities
//! - Performance monitoring
//! - Error handling

use torsh_package::{
    calculate_hash, format_file_size, global_profiler, normalize_path, parse_content_type,
    validate_package_metadata, validate_resource_path, MemoryStats, Package, PackageBuilder,
    PackageOperation, PerformanceTimer, ProfileGuard,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Package Comprehensive Workflow Example ===\n");

    // Initialize performance monitoring
    let workflow_timer = PerformanceTimer::new("complete_workflow");
    let mut memory_stats = MemoryStats::new();

    // Step 1: Create package with profiling
    println!("Step 1: Creating package with profiling...");
    let package = {
        let _guard = ProfileGuard::new(global_profiler(), PackageOperation::Create.as_str());
        create_example_package()?
    };
    println!("✓ Package created successfully\n");

    // Step 2: Add resources with validation
    println!("Step 2: Adding resources with validation...");
    let mut package = add_resources_with_validation(package, &mut memory_stats)?;
    println!("✓ Resources added and validated\n");

    // Step 3: Save package with compression
    println!("Step 3: Saving package with compression...");
    let temp_dir = tempfile::TempDir::new()?;
    let package_path = temp_dir.path().join("example.torshpkg");
    save_package_compressed(&mut package, &package_path)?;
    println!("✓ Package saved to {:?}\n", package_path);

    // Step 4: Load and verify package
    println!("Step 4: Loading and verifying package...");
    let _loaded_package = load_and_verify_package(&package_path)?;
    println!("✓ Package loaded and verified\n");

    // Step 5: Generate performance report
    println!("Step 5: Generating performance report...");
    generate_performance_report(&workflow_timer, &memory_stats)?;

    println!("\n=== Workflow completed successfully! ===");
    Ok(())
}

/// Create an example package with metadata validation
fn create_example_package() -> Result<Package, Box<dyn std::error::Error>> {
    let name = "ml-model-example";
    let version = "1.2.3";
    let description = Some("A comprehensive machine learning model package");

    // Validate metadata before creating package
    validate_package_metadata(name, version, description)?;

    let package = PackageBuilder::new(name.to_string(), version.to_string())
        .author("ToRSh Team".to_string())
        .description(description.unwrap().to_string())
        .license("MIT".to_string())
        .add_dependency("torsh-core", "0.1.0")
        .add_dependency("torsh-nn", "0.1.0")
        .package();

    Ok(package)
}

/// Add resources with path validation and memory tracking
fn add_resources_with_validation(
    mut package: Package,
    memory_stats: &mut MemoryStats,
) -> Result<Package, Box<dyn std::error::Error>> {
    let files = vec![
        ("model/weights.bin", b"model weights data".to_vec()),
        ("config/model.json", b"{\"layers\": 10}".to_vec()),
        ("src/inference.rs", b"fn predict() {}".to_vec()),
        ("data/vocab.txt", b"token1\ntoken2\ntoken3".to_vec()),
    ];

    for (path, data) in files {
        // Validate resource path
        let normalized_path = normalize_path(path);
        validate_resource_path(&normalized_path)?;

        // Calculate resource hash and content type
        let hash = calculate_hash(&data);
        let content_type = parse_content_type(path);

        println!(
            "  Validating: {} ({}, {}, hash: {}...)",
            path,
            format_file_size(data.len() as u64),
            content_type,
            &hash[..16]
        );

        // Track memory allocation
        memory_stats.record_allocation(data.len() as u64);

        // Add data file to package (using public API)
        let data_str = String::from_utf8_lossy(&data).to_string();
        package.add_data_file(path, &data_str)?;
    }

    Ok(package)
}

/// Save package with profiling
fn save_package_compressed(
    package: &mut Package,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let _guard = ProfileGuard::new(global_profiler(), PackageOperation::Save.as_str());

    println!("  Saving package...");

    // Save package
    package.save(path)?;

    // Get file size
    let file_size = std::fs::metadata(path)?.len();
    println!("  Package size: {}", format_file_size(file_size));

    Ok(())
}

/// Load and verify package integrity
fn load_and_verify_package(path: &std::path::Path) -> Result<Package, Box<dyn std::error::Error>> {
    let _guard = ProfileGuard::new(global_profiler(), PackageOperation::Load.as_str());

    let package = Package::load(path)?;

    println!("  Package: {} v{}", package.name(), package.get_version());

    // Verify package integrity
    let _verify_guard = ProfileGuard::new(global_profiler(), PackageOperation::Verify.as_str());
    package.verify()?;
    println!("  ✓ Package integrity verified");

    Ok(package)
}

/// Generate comprehensive performance report
fn generate_performance_report(
    timer: &PerformanceTimer,
    memory_stats: &MemoryStats,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Performance Report ===");

    // Overall workflow timing
    println!("\nWorkflow Timing:");
    println!("  Total time: {:.3}s", timer.elapsed_secs());

    // Memory statistics
    println!("\nMemory Usage:");
    println!("  {}", memory_stats.format());

    // Operation profiling statistics
    println!("\nOperation Statistics:");
    let stats = global_profiler().all_stats();

    if stats.is_empty() {
        println!("  No profiling data available");
        return Ok(());
    }

    // Create a simple table
    println!(
        "  {:<20} {:>8} {:>10} {:>10} {:>10}",
        "Operation", "Count", "Avg (ms)", "Min (ms)", "Max (ms)"
    );
    println!(
        "  {:-<20} {:->8} {:->10} {:->10} {:->10}",
        "", "", "", "", ""
    );

    for stat in &stats {
        println!(
            "  {:<20} {:>8} {:>10.2} {:>10} {:>10}",
            stat.name, stat.count, stat.avg_duration_ms, stat.min_duration_ms, stat.max_duration_ms
        );
    }

    // Additional statistics
    println!("\nPercentile Analysis:");
    for stat in &stats {
        if stat.count > 1 {
            println!("  {}", stat.name);
            println!(
                "    P50: {}ms, P95: {}ms, P99: {}ms",
                stat.p50_ms, stat.p95_ms, stat.p99_ms
            );
            println!("    StdDev: {:.2}ms", stat.std_dev_ms);
        }
    }

    // Export JSON report
    let json_report = global_profiler().export_json()?;
    println!("\nJSON Report (first 200 chars):");
    println!(
        "  {}...",
        &json_report.chars().take(200).collect::<String>()
    );

    Ok(())
}
