//! Comprehensive Rust example demonstrating ToRSh FFI integration capabilities
//!
//! This example showcases:
//! 1. FFI bindings usage from Rust
//! 2. Integration with external libraries
//! 3. Performance optimization techniques
//! 4. Error handling patterns

use std::collections::HashMap;
use torsh_ffi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh FFI Integration Examples (Rust) ===\n");

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Example 1: Basic FFI functionality
    basic_ffi_example()?;

    // Example 2: Performance optimization
    performance_example()?;

    // Example 3: Multi-language binding generation
    binding_generation_example()?;

    // Example 4: Benchmark suite usage
    benchmark_example()?;

    // Example 5: Migration tools usage
    migration_example()?;

    println!("=== All Examples Complete ===");
    Ok(())
}

fn basic_ffi_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic FFI Functionality");
    println!("-" * 40);

    // Example tensor operations through C API
    use torsh_ffi::c_api::*;

    println!("1.1 C API Tensor Operations");

    // Create tensors
    let shape = vec![3, 3];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // In a real implementation, these would call the C API functions
    println!("Created 3x3 tensor with sample data");
    println!("Shape: {:?}", shape);
    println!("Data: {:?}", &data[..3]); // Show first 3 elements

    // Demonstrate error handling
    println!("\n1.2 Error Handling");
    let error = FfiError::InvalidShape("Invalid tensor shape".to_string());
    println!("Example error: {}", error);

    println!();
    Ok(())
}

fn performance_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Performance Optimization");
    println!("-" * 40);

    // Example using performance utilities
    use torsh_ffi::performance::*;

    println!("2.1 Memory Pool Operations");

    // Create a memory pool for efficient allocation
    let mut memory_pool = MemoryPool::new(1024 * 1024)?; // 1MB pool

    // Allocate memory blocks
    for i in 0..10 {
        let size = (i + 1) * 1024; // Varying sizes
        let _block = memory_pool.allocate(size)?;
        println!("Allocated block {} of size {} bytes", i, size);
    }

    // Get pool statistics
    let stats = memory_pool.get_statistics();
    println!("Pool statistics:");
    println!("  Total allocations: {}", stats.total_allocations);
    println!("  Active blocks: {}", stats.active_blocks);
    println!("  Pool utilization: {:.2}%", stats.utilization_percentage);

    println!("\n2.2 Batched Operations");

    // Create batched operations for better performance
    let mut batched_ops = BatchedOperations::new()?;

    // Add operations to batch
    for i in 0..5 {
        let op = format!("operation_{}", i);
        batched_ops.add_operation(&op, vec![i as f32; 10])?;
    }

    // Execute batch
    let results = batched_ops.execute_batch()?;
    println!("Executed batch of {} operations", results.len());

    println!("\n2.3 Async Operations");

    // Demonstrate async operation queue
    let mut async_queue = AsyncOperationQueue::new(4)?; // 4 worker threads

    // Queue operations
    for i in 0..8 {
        let operation_id = format!("async_op_{}", i);
        async_queue.queue_operation(
            operation_id,
            Box::new(move || {
                // Simulate some work
                std::thread::sleep(std::time::Duration::from_millis(100));
                format!("Result from operation {}", i)
            }),
        )?;
    }

    // Wait for completion
    let completed_ops = async_queue.wait_for_completion()?;
    println!("Completed {} async operations", completed_ops);

    println!();
    Ok(())
}

fn binding_generation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Multi-Language Binding Generation");
    println!("-" * 40);

    use torsh_ffi::binding_generator::*;

    println!("3.1 Automatic Binding Generation");

    // Create binding generator
    let mut generator = BindingGenerator::new()?;

    // Configure for different languages
    let languages = vec!["python", "java", "csharp", "go", "swift", "ruby"];

    for lang in &languages {
        println!("Generating {} bindings...", lang);

        // Configure language-specific settings
        let mut config = LanguageConfig::new(lang);
        config.set_output_directory(&format!("generated/{}", lang));
        config.set_namespace(&format!("torsh_{}", lang));

        // Generate bindings
        let result = generator.generate_bindings(&config)?;
        println!("  Generated {} files", result.files_generated);
        println!("  Output directory: {}", result.output_path);
    }

    println!("\n3.2 Custom Template System");

    // Add custom templates
    let template_config = TemplateConfig {
        header_template: "// Auto-generated ToRSh bindings for {language}".to_string(),
        footer_template: "// End of ToRSh bindings".to_string(),
        function_template: "{return_type} {function_name}({parameters});".to_string(),
        class_template: "class {class_name} {{ {methods} }};".to_string(),
    };

    generator.add_template("custom", template_config)?;
    println!("Added custom template configuration");

    println!();
    Ok(())
}

fn benchmark_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Benchmark Suite Usage");
    println!("-" * 40);

    use torsh_ffi::benchmark_suite::*;

    println!("4.1 Performance Benchmarking");

    // Create benchmark suite
    let mut benchmark = BenchmarkSuite::new()?;

    // Configure benchmark
    let config = BenchmarkConfig {
        iterations: 100,
        warmup_iterations: 10,
        tensor_sizes: vec![(64, 64), (128, 128), (256, 256)],
        languages: vec!["rust".to_string(), "python".to_string(), "java".to_string()],
        operations: vec![
            "add".to_string(),
            "multiply".to_string(),
            "matmul".to_string(),
        ],
    };

    benchmark.configure(config)?;

    // Run benchmarks
    println!("Running benchmarks...");
    let results = benchmark.run_all_benchmarks()?;

    // Display results
    println!("\nBenchmark Results:");
    for (operation, metrics) in &results.operation_metrics {
        println!("Operation: {}", operation);
        println!("  Average time: {:.2}ms", metrics.average_time_ms);
        println!("  Throughput: {:.2} ops/sec", metrics.throughput);
        println!("  Memory usage: {:.2}MB", metrics.memory_usage_mb);
    }

    println!("\n4.2 Cross-Language Performance Comparison");

    // Compare performance across languages
    let comparison = results.compare_languages();
    for (lang, performance) in comparison {
        println!("{}: {:.2}x relative performance", lang, performance);
    }

    // Export results
    benchmark.export_results(&results, "benchmark_results.json", ExportFormat::Json)?;
    println!("Exported results to benchmark_results.json");

    println!();
    Ok(())
}

fn migration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Migration Tools Usage");
    println!("-" * 40);

    use torsh_ffi::migration_tools::*;

    println!("5.1 Framework Migration");

    // Create migration tool
    let mut migrator = MigrationTool::new()?;

    // Configure source and target frameworks
    let migration_config = MigrationConfig {
        source_framework: Framework::PyTorch,
        target_framework: Framework::ToRSh,
        preserve_api_compatibility: true,
        generate_compatibility_layer: true,
        output_directory: "migrated_code".to_string(),
    };

    // Example PyTorch code to migrate
    let pytorch_code = r#"
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
"#;

    // Perform migration
    println!("Migrating PyTorch code to ToRSh...");
    let migration_result = migrator.migrate_code(pytorch_code, &migration_config)?;

    println!("Migration completed:");
    println!(
        "  Success rate: {:.1}%",
        migration_result.success_rate * 100.0
    );
    println!("  Files processed: {}", migration_result.files_processed);
    println!(
        "  Patterns replaced: {}",
        migration_result.patterns_replaced
    );

    if !migration_result.warnings.is_empty() {
        println!("  Warnings:");
        for warning in &migration_result.warnings {
            println!("    - {}", warning);
        }
    }

    println!("\n5.2 Type System Mapping");

    // Demonstrate type mapping
    let type_mapper = TypeMapper::new()?;

    let pytorch_types = vec!["torch.float32", "torch.int64", "torch.bool"];

    println!("Type mappings PyTorch -> ToRSh:");
    for pt_type in &pytorch_types {
        let torsh_type = type_mapper.map_type(pt_type, Framework::PyTorch, Framework::ToRSh)?;
        println!("  {} -> {}", pt_type, torsh_type);
    }

    println!("\n5.3 Migration Report Generation");

    // Generate comprehensive migration report
    let report = migrator.generate_migration_report(&migration_result)?;

    println!("Generated migration report:");
    println!(
        "  API compatibility: {:.1}%",
        report.api_compatibility_score * 100.0
    );
    println!(
        "  Performance impact: {:+.1}%",
        report.performance_impact_percentage
    );
    println!(
        "  Manual review items: {}",
        report.manual_review_items.len()
    );

    // Export migration guide
    migrator.export_migration_guide(&report, "migration_guide.md")?;
    println!("Exported migration guide to migration_guide.md");

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_error_handling() {
        let error = FfiError::InvalidShape("Test error".to_string());
        assert!(error.to_string().contains("Test error"));
    }

    #[test]
    fn test_memory_pool_creation() {
        let pool = MemoryPool::new(1024);
        assert!(pool.is_ok());
    }

    #[test]
    fn test_binding_generator_creation() {
        let generator = BindingGenerator::new();
        assert!(generator.is_ok());
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let benchmark = BenchmarkSuite::new();
        assert!(benchmark.is_ok());
    }

    #[test]
    fn test_migration_tool_creation() {
        let migrator = MigrationTool::new();
        assert!(migrator.is_ok());
    }
}
