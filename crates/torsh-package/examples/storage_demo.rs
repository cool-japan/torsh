//! Storage backend demonstration
//!
//! This example demonstrates the cloud storage abstraction layer for package distribution.
//! It shows how to use the LocalStorage backend and StorageManager for caching and
//! efficient package management.
//!
//! Run with: cargo run --example storage_demo

use std::time::Instant;
use tempfile::TempDir;
use torsh_package::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗄️  ToRSh Package - Storage Backend Demonstration");
    println!("=================================================\n");

    let temp_dir = TempDir::new()?;
    let storage_path = temp_dir.path().join("package_storage");

    // 1. Basic Local Storage Operations
    println!("📁 1. Basic Local Storage Operations");
    println!("------------------------------------");
    demonstrate_basic_storage(&storage_path)?;

    // 2. Storage Manager with Caching
    println!("\n💾 2. Storage Manager with Caching");
    println!("----------------------------------");
    demonstrate_storage_manager(&storage_path)?;

    // 3. Package Distribution Workflow
    println!("\n📦 3. Package Distribution Workflow");
    println!("------------------------------------");
    demonstrate_package_distribution(&storage_path)?;

    // 4. Performance Optimization
    println!("\n⚡ 4. Performance Optimization");
    println!("------------------------------");
    demonstrate_performance_features(&storage_path)?;

    // 5. Storage Statistics and Monitoring
    println!("\n📊 5. Storage Statistics and Monitoring");
    println!("---------------------------------------");
    demonstrate_storage_stats(&storage_path)?;

    println!("\n✅ All storage operations completed successfully!");
    println!("\nKey Features Demonstrated:");
    println!("  • Local file system storage backend");
    println!("  • In-memory caching with LRU eviction");
    println!("  • Automatic retry on failures");
    println!("  • Storage operation statistics");
    println!("  • Package organization and listing");

    Ok(())
}

/// Demonstrate basic storage operations
fn demonstrate_basic_storage(
    storage_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = LocalStorage::new(storage_path.to_path_buf())?;

    println!("   Storage backend type: {}", storage.backend_type());
    println!("   Storage location: {:?}", storage_path);

    // Store some packages
    let packages: Vec<(&str, &[u8])> = vec![
        (
            "models/bert/v1.0.0.torshpkg",
            b"BERT model package v1.0.0" as &[u8],
        ),
        (
            "models/bert/v1.1.0.torshpkg",
            b"BERT model package v1.1.0" as &[u8],
        ),
        (
            "models/gpt2/v1.0.0.torshpkg",
            b"GPT-2 model package v1.0.0" as &[u8],
        ),
        (
            "models/resnet/v1.0.0.torshpkg",
            b"ResNet model package v1.0.0" as &[u8],
        ),
    ];

    for (key, data) in &packages {
        storage.put(key, data)?;
        println!("   ✓ Stored: {}", key);
    }

    // Retrieve a package
    let retrieved = storage.get("models/bert/v1.0.0.torshpkg")?;
    println!(
        "   ✓ Retrieved: models/bert/v1.0.0.torshpkg ({} bytes)",
        retrieved.len()
    );

    // Check if packages exist
    assert!(storage.exists("models/bert/v1.0.0.torshpkg")?);
    assert!(!storage.exists("models/nonexistent.torshpkg")?);
    println!("   ✓ Existence checks working correctly");

    // List packages
    let bert_packages = storage.list("models/bert/")?;
    println!("   ✓ Found {} BERT packages:", bert_packages.len());
    for pkg in &bert_packages {
        println!("     - {} ({} bytes)", pkg.key, pkg.size);
    }

    // Get metadata
    let metadata = storage.get_metadata("models/bert/v1.0.0.torshpkg")?;
    println!("   ✓ Metadata for models/bert/v1.0.0.torshpkg:");
    println!("     - Size: {} bytes", metadata.size);
    println!("     - Last modified: {:?}", metadata.last_modified);

    // Copy a package
    storage.copy(
        "models/bert/v1.0.0.torshpkg",
        "models/bert/backup_v1.0.0.torshpkg",
    )?;
    println!("   ✓ Copied package to backup location");

    // Delete a package
    storage.delete("models/bert/backup_v1.0.0.torshpkg")?;
    println!("   ✓ Deleted backup package");

    Ok(())
}

/// Demonstrate storage manager with caching
fn demonstrate_storage_manager(
    storage_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage = LocalStorage::new(storage_path.to_path_buf())?;
    let mut manager = StorageManager::new(Box::new(storage))
        .with_cache_size(10 * 1024) // 10KB cache
        .with_retry_count(3);

    println!("   Cache size limit: 10KB");
    println!("   Retry count: 3");

    // Add test data
    let test_packages = vec![
        ("cache/pkg1.torshpkg", vec![1u8; 2048]), // 2KB
        ("cache/pkg2.torshpkg", vec![2u8; 2048]), // 2KB
        ("cache/pkg3.torshpkg", vec![3u8; 3072]), // 3KB
        ("cache/pkg4.torshpkg", vec![4u8; 4096]), // 4KB
    ];

    for (key, data) in &test_packages {
        manager.put(key, data)?;
        println!("   ✓ Stored: {} ({} bytes)", key, data.len());
    }

    // First access - cache miss
    let start = Instant::now();
    let _data1 = manager.get("cache/pkg1.torshpkg")?;
    let miss_time = start.elapsed();
    println!("   First access (cache miss): {:?}", miss_time);

    // Second access - cache hit
    let start = Instant::now();
    let _data2 = manager.get("cache/pkg1.torshpkg")?;
    let hit_time = start.elapsed();
    println!("   Second access (cache hit): {:?}", hit_time);
    println!(
        "   Cache speedup: {:.2}x",
        miss_time.as_nanos() as f64 / hit_time.as_nanos() as f64
    );

    // Access multiple packages to trigger eviction
    println!("\n   Accessing multiple packages to test cache eviction:");
    for (key, _) in &test_packages {
        manager.get(key)?;
        let stats = manager.stats();
        println!(
            "     - Accessed {}: {} hits, {} misses",
            key, stats.cache_hits, stats.cache_misses
        );
    }

    Ok(())
}

/// Demonstrate package distribution workflow
fn demonstrate_package_distribution(
    storage_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage = LocalStorage::new(storage_path.to_path_buf())?;
    let mut manager = StorageManager::new(Box::new(storage)).with_cache_size(100 * 1024 * 1024); // 100MB cache

    // Create a sample package
    let mut package = Package::new("distributed_model".to_string(), "1.0.0".to_string());
    package.manifest_mut().author = Some("ML Team".to_string());
    package.manifest_mut().description = Some("A model ready for distribution".to_string());

    // Add resources
    let model_weights = vec![0u8; 10000];
    let weights_resource = Resource::new(
        "model_weights.bin".to_string(),
        ResourceType::Model,
        model_weights,
    );
    package.add_resource(weights_resource);

    let config = br#"{"hidden_size": 768, "num_layers": 12}"#;
    let config_resource = Resource::new(
        "config.json".to_string(),
        ResourceType::Config,
        config.to_vec(),
    );
    package.add_resource(config_resource);

    println!("   Created package: {}", package.name());
    println!("   Version: {}", package.get_version());
    println!("   Resources: {}", package.resources().len());

    // Save package to local temp file
    let temp_pkg = storage_path.join("temp_package.torshpkg");
    package.save(&temp_pkg)?;
    println!("   ✓ Saved package to temporary location");

    // Read package data
    let package_data = std::fs::read(&temp_pkg)?;
    println!("   Package size: {} bytes", package_data.len());

    // Upload to storage with versioning
    let storage_key = format!(
        "packages/{}/v{}/{}.torshpkg",
        package.name(),
        package.get_version(),
        package.name()
    );
    manager.put(&storage_key, &package_data)?;
    println!("   ✓ Uploaded to storage: {}", storage_key);

    // Download and verify
    let downloaded = manager.get(&storage_key)?;
    assert_eq!(downloaded.len(), package_data.len());
    println!("   ✓ Downloaded and verified package");

    // List all versions of this package
    let package_prefix = format!("packages/{}/", package.name());
    let versions = manager.list(&package_prefix)?;
    println!("   Available versions:");
    for version in &versions {
        println!("     - {} ({} bytes)", version.key, version.size);
    }

    // Create a new version
    let mut package_v2 = Package::new("distributed_model".to_string(), "2.0.0".to_string());
    package_v2.manifest_mut().author = Some("ML Team".to_string());
    package_v2.manifest_mut().description = Some("Updated model with improvements".to_string());

    // Add updated resources
    let updated_weights = vec![1u8; 12000];
    let updated_resource = Resource::new(
        "model_weights.bin".to_string(),
        ResourceType::Model,
        updated_weights,
    );
    package_v2.add_resource(updated_resource);

    // Save and upload v2
    let temp_pkg_v2 = storage_path.join("temp_package_v2.torshpkg");
    package_v2.save(&temp_pkg_v2)?;
    let package_data_v2 = std::fs::read(&temp_pkg_v2)?;

    let storage_key_v2 = format!(
        "packages/{}/v{}/{}.torshpkg",
        package_v2.name(),
        package_v2.get_version(),
        package_v2.name()
    );
    manager.put(&storage_key_v2, &package_data_v2)?;
    println!("   ✓ Uploaded version 2.0.0");

    // List all versions again
    let versions_updated = manager.list(&package_prefix)?;
    println!("   Total versions available: {}", versions_updated.len());

    Ok(())
}

/// Demonstrate performance features
fn demonstrate_performance_features(
    storage_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage = LocalStorage::new(storage_path.to_path_buf())?;
    let mut manager = StorageManager::new(Box::new(storage))
        .with_cache_size(50 * 1024 * 1024) // 50MB cache
        .with_retry_count(3);

    // Create packages of different sizes
    let package_sizes = vec![
        ("small", 10 * 1024),        // 10KB
        ("medium", 100 * 1024),      // 100KB
        ("large", 1024 * 1024),      // 1MB
        ("xlarge", 5 * 1024 * 1024), // 5MB
    ];

    println!("   Testing performance with different package sizes:");

    for (name, size) in &package_sizes {
        let data = vec![0u8; *size];
        let key = format!("perf/{}.torshpkg", name);

        // Measure upload time
        let start = Instant::now();
        manager.put(&key, &data)?;
        let upload_time = start.elapsed();

        // Measure download time (first - cold)
        manager.clear_cache();
        let start = Instant::now();
        let _downloaded = manager.get(&key)?;
        let cold_download_time = start.elapsed();

        // Measure download time (second - cached)
        let start = Instant::now();
        let _downloaded = manager.get(&key)?;
        let cached_download_time = start.elapsed();

        println!("     {} ({} KB):", name, size / 1024);
        println!("       - Upload: {:?}", upload_time);
        println!("       - Download (cold): {:?}", cold_download_time);
        println!("       - Download (cached): {:?}", cached_download_time);
        println!(
            "       - Cache speedup: {:.2}x",
            cold_download_time.as_nanos() as f64 / cached_download_time.as_nanos() as f64
        );
    }

    // Test batch operations
    println!("\n   Testing batch operations:");
    let num_packages = 50;
    let package_size = 10 * 1024; // 10KB each

    let start = Instant::now();
    for i in 0..num_packages {
        let data = vec![i as u8; package_size];
        let key = format!("batch/pkg_{:03}.torshpkg", i);
        manager.put(&key, &data)?;
    }
    let batch_upload_time = start.elapsed();

    println!(
        "     Uploaded {} packages in {:?}",
        num_packages, batch_upload_time
    );
    println!(
        "     Average time per package: {:?}",
        batch_upload_time / num_packages
    );

    Ok(())
}

/// Demonstrate storage statistics
fn demonstrate_storage_stats(
    storage_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let storage = LocalStorage::new(storage_path.to_path_buf())?;
    let mut manager = StorageManager::new(Box::new(storage)).with_cache_size(20 * 1024 * 1024); // 20MB cache

    // Perform various operations
    let operations = vec![
        ("put", "stats/package1.torshpkg", vec![1u8; 5000]),
        ("put", "stats/package2.torshpkg", vec![2u8; 8000]),
        ("put", "stats/package3.torshpkg", vec![3u8; 12000]),
    ];

    for (op, key, data) in operations {
        match op {
            "put" => manager.put(key, &data)?,
            _ => {}
        }
    }

    // Read packages multiple times
    for _ in 0..5 {
        manager.get("stats/package1.torshpkg")?;
        manager.get("stats/package2.torshpkg")?;
        manager.get("stats/package3.torshpkg")?;
    }

    // Delete a package
    manager.delete("stats/package3.torshpkg")?;

    // Get statistics
    let stats = manager.stats();
    println!("   Storage Operation Statistics:");
    println!("   ----------------------------");
    println!("     Total puts: {}", stats.puts);
    println!("     Total gets: {}", stats.gets);
    println!("     Total deletes: {}", stats.deletes);
    println!("     Bytes written: {} KB", stats.bytes_written / 1024);
    println!("     Bytes read: {} KB", stats.bytes_read / 1024);
    println!("     Cache hits: {}", stats.cache_hits);
    println!("     Cache misses: {}", stats.cache_misses);
    println!(
        "     Cache hit rate: {:.1}%",
        (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0
    );

    // List all stored packages
    let all_packages = manager.list("stats/")?;
    println!("\n   Stored Packages:");
    let mut total_size = 0u64;
    for pkg in &all_packages {
        println!("     - {} ({} bytes)", pkg.key, pkg.size);
        total_size += pkg.size;
    }
    println!(
        "     Total: {} packages, {} bytes",
        all_packages.len(),
        total_size
    );

    Ok(())
}
