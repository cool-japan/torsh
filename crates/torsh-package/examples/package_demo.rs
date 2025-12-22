//! Comprehensive demonstration of torsh-package advanced features
//!
//! This example showcases all the advanced features implemented in torsh-package:
//! - Delta patching for incremental updates
//! - Dependency resolution with conflict handling
//! - Lazy resource loading with memory management
//! - Format compatibility with PyTorch and HuggingFace
//! - Advanced compression algorithms
//! - Comprehensive testing and validation
//!
//! Run with: cargo run --example comprehensive_demo

use std::fs;
use std::io::Write;
use tempfile::TempDir;

use torsh_package::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ToRSh Package - Comprehensive Feature Demonstration");
    println!("====================================================\n");

    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();

    // 1. Advanced Package Creation with Compression
    println!("📦 1. Advanced Package Creation with Compression");
    println!("------------------------------------------------");

    let advanced_package = create_advanced_package(base_path)?;
    println!("✅ Created advanced package with multiple resource types");

    // 2. Format Compatibility Demo
    println!("\n🔄 2. Format Compatibility Demonstration");
    println!("----------------------------------------");

    demonstrate_format_compatibility(base_path)?;
    println!("✅ Demonstrated PyTorch and HuggingFace format compatibility");

    // 3. Delta Patching Demo
    println!("\n📈 3. Delta Patching for Incremental Updates");
    println!("--------------------------------------------");

    demonstrate_delta_patching(base_path, &advanced_package)?;
    println!("✅ Demonstrated incremental package updates with delta patches");

    // 4. Dependency Resolution Demo
    println!("\n🔗 4. Advanced Dependency Resolution");
    println!("------------------------------------");

    demonstrate_dependency_resolution(base_path)?;
    println!("✅ Demonstrated dependency resolution with conflict handling");

    // 5. Lazy Resource Loading Demo
    println!("\n💾 5. Lazy Resource Loading & Memory Management");
    println!("-----------------------------------------------");

    demonstrate_lazy_loading(base_path)?;
    println!("✅ Demonstrated lazy loading with intelligent memory management");

    // 6. Advanced Compression Demo
    println!("\n🗜️ 6. Advanced Compression Algorithms");
    println!("--------------------------------------");

    demonstrate_compression_features()?;
    println!("✅ Demonstrated multiple compression algorithms with benchmarking");

    // 7. Performance & Scalability Demo
    println!("\n⚡ 7. Performance & Scalability Testing");
    println!("---------------------------------------");

    demonstrate_performance_features(base_path)?;
    println!("✅ Demonstrated performance optimizations for large packages");

    println!("\n🎉 All advanced features demonstrated successfully!");
    println!("The ToRSh Package system provides a comprehensive solution for:");
    println!("  • Model packaging and distribution");
    println!("  • Cross-format compatibility");
    println!("  • Efficient storage and compression");
    println!("  • Incremental updates and versioning");
    println!("  • Smart dependency management");
    println!("  • Memory-efficient resource loading");

    Ok(())
}

/// Create an advanced package with various features
fn create_advanced_package(
    base_path: &std::path::Path,
) -> Result<Package, Box<dyn std::error::Error>> {
    // Create package with advanced compression
    let compression_config = CompressionConfig::new()
        .with_algorithm(CompressionAlgorithm::Zstd)
        .with_strategy(CompressionStrategy::Adaptive)
        .with_min_threshold(100);

    let compressor = AdvancedCompressor::with_config(compression_config);

    let mut package = Package::new("advanced_ml_model".to_string(), "2.1.0".to_string());

    // Set comprehensive metadata
    package.manifest_mut().author = Some("AI Research Team".to_string());
    package.manifest_mut().description =
        Some("Advanced ML model with comprehensive packaging features".to_string());
    package.manifest_mut().license = Some("Apache-2.0".to_string());

    // Add various resource types

    // 1. Model weights (binary data)
    let model_weights = (0..10000).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
    let weights_resource = Resource::new(
        "model_weights.bin".to_string(),
        ResourceType::Model,
        model_weights,
    );
    let compressed_weights = compressor.compress_resource(&weights_resource)?;
    println!(
        "   Model weights: {} -> {} bytes (ratio: {:.2})",
        compressed_weights.original_size,
        compressed_weights.compressed_size,
        compressed_weights.ratio
    );

    package.add_resource(weights_resource);

    // 2. Source code
    let source_code = r#"
/// Advanced ML model implementation
use torsh::prelude::*;

pub struct AdvancedModel {
    layers: Vec<Layer>,
    optimizer: Optimizer,
}

impl AdvancedModel {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            layers: Self::build_layers(config),
            optimizer: Optimizer::Adam(config.learning_rate),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}
"#;
    package.add_source_file("model", source_code)?;

    // 3. Configuration
    let config_data = r#"{
    "model_type": "transformer",
    "hidden_size": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "dropout": 0.1,
    "attention_dropout": 0.1
}"#;
    let config_resource = Resource::new(
        "config.json".to_string(),
        ResourceType::Config,
        config_data.as_bytes().to_vec(),
    );
    package.add_resource(config_resource);

    // 4. Documentation
    let docs = r#"# Advanced ML Model

This package contains a state-of-the-art transformer model with the following features:

## Features
- Multi-head self-attention
- Layer normalization
- Residual connections
- Advanced optimization techniques

## Usage
```rust
use advanced_ml_model::AdvancedModel;

let model = AdvancedModel::new(&config);
let output = model.forward(&input)?;
```

## Performance
- Training speed: 1000 samples/sec
- Inference speed: 5000 samples/sec
- Memory usage: ~2GB
"#;
    let docs_resource = Resource::new(
        "README.md".to_string(),
        ResourceType::Documentation,
        docs.as_bytes().to_vec(),
    );
    package.add_resource(docs_resource);

    // 5. Training data sample
    let training_sample = serde_json::json!({
        "samples": [
            {"input": [1, 2, 3], "output": [0.1, 0.7, 0.2]},
            {"input": [4, 5, 6], "output": [0.3, 0.4, 0.3]},
            {"input": [7, 8, 9], "output": [0.6, 0.2, 0.2]}
        ],
        "metadata": {
            "dataset_version": "1.2",
            "preprocessing": "standard_normalization",
            "augmentation": true
        }
    });
    let sample_resource = Resource::new(
        "training_sample.json".to_string(),
        ResourceType::Data,
        training_sample.to_string().as_bytes().to_vec(),
    );
    package.add_resource(sample_resource);

    // Add dependencies
    package.add_dependency("torsh", "^2.0.0");
    package.add_dependency("serde", "^1.0");
    package.add_dependency("serde_json", "^1.0");

    // Save the package
    let package_path = base_path.join("advanced_model.torshpkg");
    package.save(&package_path)?;

    println!("   Package saved to: {:?}", package_path);
    println!("   Resources: {}", package.resources().len());
    println!("   Dependencies: {}", package.metadata().dependencies.len());

    Ok(package)
}

/// Demonstrate format compatibility features
fn demonstrate_format_compatibility(
    base_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let manager = FormatCompatibilityManager::new();

    // 1. Create a mock PyTorch package
    let pytorch_path = base_path.join("mock_pytorch_model.pt");
    create_mock_pytorch_package(&pytorch_path)?;

    // Import PyTorch package
    let (format, pytorch_package) = manager.import_package(&pytorch_path)?;
    println!(
        "   Imported {} package: {}",
        format!("{:?}", format),
        pytorch_package.metadata().name
    );

    // 2. Create a mock HuggingFace model
    let hf_dir = base_path.join("mock_hf_model");
    create_mock_huggingface_model(&hf_dir)?;

    // Import HuggingFace model
    let (format, hf_package) = manager.import_package(&hf_dir)?;
    println!(
        "   Imported {} package: {}",
        format!("{:?}", format),
        hf_package.metadata().name
    );

    // 3. Export to different formats
    let export_pytorch = base_path.join("exported_model.pt");
    manager.export_package(&hf_package, PackageFormat::PyTorch, &export_pytorch)?;
    println!("   Exported to PyTorch format: {:?}", export_pytorch);

    let export_hf = base_path.join("exported_hf_model");
    manager.export_package(&pytorch_package, PackageFormat::HuggingFace, &export_hf)?;
    println!("   Exported to HuggingFace format: {:?}", export_hf);

    Ok(())
}

/// Create a mock PyTorch package for demonstration
fn create_mock_pytorch_package(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);

    // Add version file
    zip.start_file::<_, ()>(".data/version", zip::write::FileOptions::default())?;
    zip.write_all(b"1.0.0")?;

    // Add Python code
    zip.start_file::<_, ()>("code/model.py", zip::write::FileOptions::default())?;
    zip.write_all(b"import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    pass")?;

    // Add model weights
    zip.start_file::<_, ()>("data/model.pkl", zip::write::FileOptions::default())?;
    zip.write_all(b"mock_pytorch_weights_data")?;

    zip.finish()?;
    Ok(())
}

/// Create a mock HuggingFace model for demonstration
fn create_mock_huggingface_model(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(path)?;

    // Create config.json
    let config = serde_json::json!({
        "model_type": "bert",
        "task": "text-classification",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12
    });
    fs::write(path.join("config.json"), config.to_string())?;

    // Create model weights
    fs::write(path.join("pytorch_model.bin"), b"mock_huggingface_weights")?;

    // Create tokenizer config
    let tokenizer_config = serde_json::json!({
        "do_lower_case": true,
        "model_max_length": 512,
        "tokenizer_class": "BertTokenizer"
    });
    fs::write(path.join("tokenizer.json"), tokenizer_config.to_string())?;

    Ok(())
}

/// Demonstrate delta patching for incremental updates
fn demonstrate_delta_patching(
    base_path: &std::path::Path,
    original: &Package,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create an updated version of the package
    let mut updated_package = Package::new("advanced_ml_model".to_string(), "2.2.0".to_string());
    updated_package.manifest_mut().author = original.metadata().author.clone();
    updated_package.manifest_mut().description =
        Some("Updated ML model with performance improvements".to_string());

    // Copy some original resources
    for (name, resource) in original.resources() {
        if name != "model_weights.bin" {
            // We'll update the weights
            updated_package.add_resource(resource.clone());
        }
    }

    // Add updated model weights
    let updated_weights = (0..12000)
        .map(|i| ((i * 2) % 256) as u8)
        .collect::<Vec<u8>>();
    let updated_weights_resource = Resource::new(
        "model_weights.bin".to_string(),
        ResourceType::Model,
        updated_weights,
    );
    updated_package.add_resource(updated_weights_resource);

    // Add new feature
    let new_feature = Resource::new(
        "new_feature.json".to_string(),
        ResourceType::Data,
        b"{\"feature\": \"advanced_attention\", \"enabled\": true}".to_vec(),
    );
    updated_package.add_resource(new_feature);

    // Create delta patch
    let patch_builder = DeltaPatchBuilder::new()
        .with_compression_level(6)
        .with_metadata_changes(true);

    let patch = patch_builder.create_patch(original, &updated_package)?;

    println!("   Delta patch created:");
    println!("     From version: {}", patch.from_version);
    println!("     To version: {}", patch.to_version);
    println!("     Operations: {}", patch.operations.len());
    println!("     Patch size: {} bytes", patch.patch_size);

    // Save patch
    let patch_path = base_path.join("model_v2.1.0_to_v2.2.0.patch");
    DeltaPatchApplier::save_patch(&patch, &patch_path)?;

    // Apply patch to original package
    let mut patched_package = Package::new(
        original.metadata().name.clone(),
        original.get_version().to_string(),
    );
    // Copy original resources first
    for (_name, resource) in original.resources() {
        patched_package.add_resource(resource.clone());
    }
    // Copy metadata
    patched_package.manifest_mut().author = original.metadata().author.clone();
    patched_package.manifest_mut().description = original.metadata().description.clone();

    let applier = DeltaPatchApplier::new()
        .with_checksum_verification(true)
        .with_backup(false);

    applier.apply_patch(&mut patched_package, &patch)?;

    println!("   Patch applied successfully:");
    println!("     Original resources: {}", original.resources().len());
    println!(
        "     Patched resources: {}",
        patched_package.resources().len()
    );
    println!("     Patched version: {}", patched_package.get_version());

    Ok(())
}

/// Demonstrate advanced dependency resolution
fn demonstrate_dependency_resolution(
    _base_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a local package registry
    let mut registry = LocalPackageRegistry::new();

    // Add packages to registry
    let packages = vec![
        PackageInfo {
            name: "torsh-core".to_string(),
            version: "2.0.0".to_string(),
            description: Some("Core tensor operations".to_string()),
            author: Some("ToRSh Team".to_string()),
            dependencies: Vec::new(),
            size: 1024 * 1024,
            checksum: "abc123".to_string(),
            registry_url: "https://registry.torsh.ai".to_string(),
        },
        PackageInfo {
            name: "torsh-nn".to_string(),
            version: "2.0.1".to_string(),
            description: Some("Neural network layers".to_string()),
            author: Some("ToRSh Team".to_string()),
            dependencies: vec![DependencySpec::new(
                "torsh-core".to_string(),
                "^2.0.0".to_string(),
            )],
            size: 2 * 1024 * 1024,
            checksum: "def456".to_string(),
            registry_url: "https://registry.torsh.ai".to_string(),
        },
        PackageInfo {
            name: "torsh-optim".to_string(),
            version: "1.5.0".to_string(),
            description: Some("Optimization algorithms".to_string()),
            author: Some("ToRSh Team".to_string()),
            dependencies: vec![
                DependencySpec::new("torsh-core".to_string(), "^2.0.0".to_string()),
                DependencySpec::new("torsh-nn".to_string(), "^2.0.0".to_string()),
            ],
            size: 512 * 1024,
            checksum: "ghi789".to_string(),
            registry_url: "https://registry.torsh.ai".to_string(),
        },
    ];

    for package in packages {
        registry.add_package(package);
    }

    // Create a package with dependencies
    let mut main_package = Package::new("ml_training_pipeline".to_string(), "1.0.0".to_string());
    main_package.add_dependency("torsh-nn", "^2.0.0");
    main_package.add_dependency("torsh-optim", "^1.0.0");

    // Resolve dependencies
    let resolver = DependencyResolver::new(Box::new(registry))
        .with_strategy(ResolutionStrategy::Highest)
        .with_max_depth(10);

    let resolved = resolver.resolve_dependencies(&main_package)?;

    println!("   Dependency resolution completed:");
    println!(
        "     Direct dependencies: {}",
        main_package.metadata().dependencies.len()
    );
    println!("     Total resolved: {}", resolved.len());

    for dep in &resolved {
        println!("     - {} v{}", dep.spec.name, dep.resolved_version);
    }

    // Build dependency graph
    let graph = resolver.build_dependency_graph(&main_package)?;
    let install_order = graph.topological_sort()?;

    println!("   Installation order:");
    for (i, package_name) in install_order.iter().enumerate() {
        println!("     {}. {}", i + 1, package_name);
    }

    Ok(())
}

/// Demonstrate lazy resource loading and memory management
fn demonstrate_lazy_loading(base_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut manager = LazyResourceManager::new()
        .with_memory_limit(5 * 1024 * 1024) // 5MB limit
        .with_eviction_strategy(EvictionStrategy::LargestFirst);

    // Create test files for lazy loading
    let test_files = vec![
        ("small_file.txt", vec![b'A'; 1024]),  // 1KB
        ("medium_file.bin", vec![b'B'; 2048]), // 2KB
        ("large_file.data", vec![b'C'; 4096]), // 4KB
    ];

    for (filename, data) in &test_files {
        let file_path = base_path.join(filename);
        fs::write(&file_path, data)?;

        let lazy_resource = LazyResource::new_lazy_file(
            filename.to_string(),
            ResourceType::Data,
            &file_path,
            0,
            data.len() as u64,
        );

        manager.add_resource(lazy_resource)?;
    }

    println!("   Created lazy resources:");
    println!("     Total resources: {}", test_files.len());
    println!(
        "     Initially loaded: {}",
        manager.loaded_resources().len()
    );
    println!("     Memory usage: {} bytes", manager.memory_usage());

    // Access resources (triggers loading)
    for (filename, _) in &test_files {
        let data = manager.load_resource_data(filename)?;
        println!("     Loaded {}: {} bytes", filename, data.len());
        println!("     Memory usage: {} bytes", manager.memory_usage());
        println!(
            "     Loaded resources: {}",
            manager.loaded_resources().len()
        );
    }

    // Create archive-based lazy resource
    let archive_path = base_path.join("test_archive.zip");
    create_test_archive(&archive_path)?;

    let archive_resource = LazyResource::new_lazy_archive(
        "archive_data".to_string(),
        ResourceType::Data,
        &archive_path,
        "test_entry.txt".to_string(),
    );

    manager.add_resource(archive_resource)?;

    // Load from archive
    let archive_data = manager.load_resource_data("archive_data")?;
    println!("   Loaded from archive: {} bytes", archive_data.len());

    // Demonstrate memory management
    println!("   Memory management:");
    println!("     Total memory limit: 5MB");
    println!("     Current usage: {} bytes", manager.memory_usage());

    // Evict all cache to free memory
    manager.evict_all_cache()?;
    println!("     After eviction: {} bytes", manager.memory_usage());

    Ok(())
}

/// Create a test ZIP archive for lazy loading demo
fn create_test_archive(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);

    zip.start_file::<_, ()>("test_entry.txt", zip::write::FileOptions::default())?;
    zip.write_all(b"This is test data stored in a ZIP archive for lazy loading demonstration.")?;

    zip.start_file::<_, ()>("another_entry.json", zip::write::FileOptions::default())?;
    zip.write_all(br#"{"message": "Hello from archive!", "size": 1024}"#)?;

    zip.finish()?;
    Ok(())
}

/// Demonstrate advanced compression features
fn demonstrate_compression_features() -> Result<(), Box<dyn std::error::Error>> {
    // Create test data with different characteristics
    let test_datasets: Vec<(&str, Vec<u8>)> = vec![
        ("highly_compressible", "A".repeat(10000).into_bytes()),
        (
            "moderately_compressible",
            "Hello World! ".repeat(500).into_bytes(),
        ),
        (
            "low_compressible",
            (0..5000).map(|i| (i % 256) as u8).collect::<Vec<u8>>(),
        ),
    ];

    let compressor = AdvancedCompressor::new();
    let mut stats = CompressionStats::new();

    for (name, data) in &test_datasets {
        let data_bytes = data.as_slice();

        println!(
            "   Testing compression on: {} ({} bytes)",
            name,
            data_bytes.len()
        );

        // Benchmark different algorithms
        let results = compressor.benchmark_algorithms(data_bytes)?;

        println!("     Algorithm performance:");
        for result in &results {
            println!(
                "       {:?}: {:.2} ratio, {} ms",
                result.algorithm, result.ratio, result.compression_time_ms
            );
            stats.record(result);
        }

        // Test adaptive compression
        let resource = Resource::new(
            format!("{}.data", name),
            if name.contains("compressible") {
                ResourceType::Text
            } else {
                ResourceType::Binary
            },
            data_bytes.to_vec(),
        );

        let adaptive_result = compressor.compress_resource(&resource)?;
        println!(
            "     Adaptive choice: {:?} (ratio: {:.2})",
            adaptive_result.algorithm, adaptive_result.ratio
        );
    }

    // Show overall statistics
    println!("   Overall compression statistics:");
    println!("     Total processed: {} bytes", stats.total_compressed);
    println!(
        "     After compression: {} bytes",
        stats.total_after_compression
    );
    println!(
        "     Space saved: {} bytes ({:.1}%)",
        stats.space_saved(),
        stats.space_saved_percent()
    );
    println!(
        "     Best algorithm: {:?}",
        stats.best_performing_algorithm()
    );

    Ok(())
}

/// Demonstrate performance and scalability features
fn demonstrate_performance_features(
    base_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating large package for performance testing...");

    let start_time = std::time::Instant::now();

    // Create package with many resources
    let mut large_package = Package::new("performance_test".to_string(), "1.0.0".to_string());

    // Add many small resources
    for i in 0..200 {
        let resource_data = format!(
            "Resource #{} with some content to make it realistic. {}",
            i,
            "x".repeat(i % 100)
        );

        let resource = Resource::new(
            format!("resource_{:03}.txt", i),
            ResourceType::Text,
            resource_data.as_bytes().to_vec(),
        );
        large_package.add_resource(resource);
    }

    let creation_time = start_time.elapsed();
    println!("     Package creation: {:?}", creation_time);

    // Save with compression
    let package_path = base_path.join("performance_test.torshpkg");
    let save_start = std::time::Instant::now();
    large_package.save(&package_path)?;
    let save_time = save_start.elapsed();

    let package_size = fs::metadata(&package_path)?.len();
    println!("     Package save: {:?}", save_time);
    println!(
        "     Package size: {} bytes ({} KB)",
        package_size,
        package_size / 1024
    );

    // Load package
    let load_start = std::time::Instant::now();
    let loaded_package = Package::load(&package_path)?;
    let load_time = load_start.elapsed();

    println!("     Package load: {:?}", load_time);
    println!(
        "     Resources loaded: {}",
        loaded_package.resources().len()
    );

    // Verify integrity
    let verify_start = std::time::Instant::now();
    let is_valid = loaded_package.verify()?;
    let verify_time = verify_start.elapsed();

    println!(
        "     Integrity check: {:?} (valid: {})",
        verify_time, is_valid
    );

    // Performance summary
    let total_time = creation_time + save_time + load_time + verify_time;
    println!("   Performance summary:");
    println!("     Total operation time: {:?}", total_time);
    println!(
        "     Resources per second: {:.0}",
        200.0 / total_time.as_secs_f64()
    );

    Ok(())
}
