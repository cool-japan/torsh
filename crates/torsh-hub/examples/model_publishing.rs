//! Model Publishing Example
//!
//! This example demonstrates how to publish models to ToRSh Hub,
//! including validation, versioning, and metadata management.

#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::HashMap;
use std::path::PathBuf;
use torsh_core::error::Result;
use torsh_hub::metadata::{EnvironmentInfo, FileType, ProvenanceInfo, SystemInfo};
use torsh_hub::model_info::*;
use torsh_hub::security::*;
use torsh_hub::upload::*;
use torsh_hub::*;

fn main() -> Result<()> {
    println!("=== ToRSh Hub Model Publishing Example ===\n");

    // Example 1: Prepare model for publishing
    println!("1. Preparing model for publishing...");
    let model_dir = prepare_example_model()?;
    println!("✓ Example model prepared at: {}", model_dir.display());

    // Example 2: Create model metadata
    println!("\n2. Creating model metadata...");
    let metadata = create_model_metadata()?;
    println!("✓ Model metadata created");
    display_metadata(&metadata);

    // Example 3: Validate model before publishing
    println!("\n3. Validating model...");
    let validation_result = validate_model_for_publishing(&model_dir, &metadata)?;
    display_validation_result(&validation_result);

    // Example 4: Create model card
    println!("\n4. Creating model card...");
    let model_card = create_comprehensive_model_card()?;
    println!("✓ Comprehensive model card created");

    // Example 5: Sign model for security
    println!("\n5. Signing model for security...");
    let signature = sign_model_for_publishing(&model_dir)?;
    println!("✓ Model signed with digital signature");
    println!("  Signature algorithm: {:?}", signature.algorithm);
    println!("  Timestamp: {}", signature.timestamp);

    // Example 6: Upload model to hub
    println!("\n6. Uploading model to hub...");
    let upload_config = UploadConfig {
        endpoint: "https://torsh.rs/api/upload".to_string(),
        auth_token: None,
        validate: true,
        compress: true,
        progress: true,
    };

    let upload_result = upload_model(&model_dir, &metadata, &upload_config)?;
    display_upload_result(&upload_result);

    // Example 7: Publish different model versions
    println!("\n7. Publishing model versions...");
    demonstrate_version_management()?;

    // Example 8: Batch publishing
    println!("\n8. Batch publishing example...");
    demonstrate_batch_publishing()?;

    // Example 9: Private model publishing
    println!("\n9. Private model publishing...");
    demonstrate_private_publishing()?;

    // Example 10: Model updates and patches
    println!("\n10. Model updates and patches...");
    demonstrate_model_updates(&model_dir)?;

    println!("\n=== Publishing example completed successfully! ===");
    Ok(())
}

fn prepare_example_model() -> Result<PathBuf> {
    use torsh_nn::prelude::{Linear, Module};

    let model_dir = std::env::temp_dir().join("example_model_publish");
    std::fs::create_dir_all(&model_dir)?;

    // Create a simple model
    let model = Linear::new(784, 10, true); // MNIST-like classifier

    // Create model configuration
    let config_content = r#"
[model]
name = "mnist-classifier"
architecture = "LinearClassifier"
version = "1.0.0"
author = "example-author"
license = "MIT"
description = "Simple linear classifier for MNIST digits"

[architecture]
input_size = 784
output_size = 10
bias = true

[training]
dataset = "MNIST"
epochs = 10
batch_size = 32
learning_rate = 0.01
optimizer = "Adam"

[metrics]
test_accuracy = 0.923
test_loss = 0.245
train_accuracy = 0.956
train_loss = 0.189

[hardware]
min_ram_gb = 1.0
recommended_ram_gb = 2.0
supports_cpu = true
supports_gpu = true
"#;

    std::fs::write(model_dir.join("config.toml"), config_content)?;

    // Save model weights (simplified - in real scenario use proper serialization)
    let state_dict = model.state_dict();
    save_model_weights(&state_dict, &model_dir.join("model.safetensors"))?;

    // Create README
    let readme_content = r#"# MNIST Linear Classifier

A simple linear classifier for MNIST digit recognition.

## Usage

```rust
use torsh_hub::load;

let model = load("my-org/mnist-classifier", "", true, None)?;
```

## Performance

- Test Accuracy: 92.3%
- Inference Time: 0.5ms (CPU)
- Model Size: 31KB

## Training

Trained on the standard MNIST dataset with 60,000 training images.
"#;

    std::fs::write(model_dir.join("README.md"), readme_content)?;

    // Create sample input/output for testing
    let sample_input = torsh_tensor::creation::randn(&[1, 784])?;
    let sample_output = model.forward(&sample_input)?;

    save_tensor(&sample_input, &model_dir.join("sample_input.tensor"))?;
    save_tensor(&sample_output, &model_dir.join("sample_output.tensor"))?;

    Ok(model_dir)
}

fn create_model_metadata() -> Result<ExtendedMetadata> {
    let mut performance_metrics = HashMap::new();
    performance_metrics.insert("test_accuracy".to_string(), 0.923);
    performance_metrics.insert("test_loss".to_string(), 0.245);
    performance_metrics.insert("inference_time_ms".to_string(), 0.5);
    performance_metrics.insert("model_size_kb".to_string(), 31.0);

    let mut usage_stats = HashMap::new();
    usage_stats.insert("downloads".to_string(), 0.0);
    usage_stats.insert("stars".to_string(), 0.0);

    let mut quality_scores = HashMap::new();
    quality_scores.insert("code_quality".to_string(), 9.2);
    quality_scores.insert("documentation".to_string(), 8.8);
    quality_scores.insert("reproducibility".to_string(), 9.5);

    let metadata = ExtendedMetadata {
        model_info: ModelInfo {
            name: "mnist-classifier".to_string(),
            description: "Simple linear classifier for MNIST digits".to_string(),
            author: "example-author".to_string(),
            version: Version::new(1, 0, 0),
            license: "MIT".to_string(),
            tags: vec![
                "mnist".to_string(),
                "classification".to_string(),
                "beginner".to_string(),
            ],
            datasets: vec!["MNIST".to_string()],
            metrics: HashMap::new(),
            requirements: Requirements {
                torsh_version: "0.1.0".to_string(),
                dependencies: vec![
                    "torsh-tensor = \"0.1.0\"".to_string(),
                    "torsh-nn = \"0.1.0\"".to_string(),
                ],
                hardware: HardwareRequirements {
                    min_gpu_memory_gb: None,
                    recommended_gpu_memory_gb: Some(4.0),
                    min_ram_gb: Some(8.0),
                    recommended_ram_gb: Some(16.0),
                },
            },
            files: vec![],
            model_card: None,
            version_history: None,
        },
        model_card: None,
        registry_entry: None,
        file_metadata: vec![
            FileMetadata {
                file_path: "config.toml".to_string(),
                file_type: FileType::Configuration,
                size_bytes: 512,
                checksum: "abc123".to_string(),
                creation_date: chrono::Utc::now(),
                last_modified: chrono::Utc::now(),
                compression: None,
                encryption: None,
            },
            FileMetadata {
                file_path: "model.safetensors".to_string(),
                file_type: FileType::ModelWeights,
                size_bytes: 31744,
                checksum: "def456".to_string(),
                creation_date: chrono::Utc::now(),
                last_modified: chrono::Utc::now(),
                compression: None,
                encryption: None,
            },
        ],
        provenance: ProvenanceInfo {
            source_repository: Some("https://github.com/example/mnist-classifier".to_string()),
            source_commit: Some("abc123def456".to_string()),
            training_job_id: None,
            trained_by: "example-author".to_string(),
            training_start: Some(chrono::Utc::now() - chrono::Duration::hours(24)),
            training_end: Some(chrono::Utc::now() - chrono::Duration::hours(22)),
            parent_model: None,
            derived_models: vec![],
            training_script: Some("train.py".to_string()),
            environment_info: EnvironmentInfo {
                framework_version: "0.1.0".to_string(),
                python_version: "3.8.10".to_string(),
                gpu_info: vec![],
                system_info: SystemInfo {
                    os: "Linux".to_string(),
                    cpu: "Intel i7-10700K".to_string(),
                    memory_gb: 32.0,
                    storage_type: "SSD".to_string(),
                },
                dependencies: HashMap::new(),
            },
        },
        performance_metrics: PerformanceMetrics::default(),
        usage_statistics: UsageStatistics::default(),
        quality_scores: QualityScores {
            documentation_score: 0.88,
            code_quality_score: 0.92,
            reproducibility_score: 0.95,
            performance_score: 0.85,
            safety_score: 0.90,
            ethical_score: 0.95,
            overall_score: 0.91,
        },
        last_updated: chrono::Utc::now(),
    };

    Ok(metadata)
}

fn create_comprehensive_model_card() -> Result<ModelCard> {
    let card = ModelCardBuilder::new()
        .developed_by("example-author@university.edu".to_string())
        .model_type("Linear Classifier".to_string())
        .architecture("Linear Layer".to_string())
        .add_primary_use("Handwritten digit recognition".to_string())
        .add_primary_use("Educational purposes and tutorials".to_string())
        .add_training_dataset(
            "MNIST Training Set".to_string(),
            Some("http://yann.lecun.com/exdb/mnist/".to_string()),
            Some("60,000 handwritten digit images (28x28 grayscale)".to_string()),
            None,
        )
        .add_training_dataset(
            "MNIST Test Set".to_string(),
            None,
            Some("10,000 handwritten digit images for evaluation".to_string()),
            None,
        )
        .add_metric(
            "Test Accuracy".to_string(),
            MetricValue::Float(0.923),
            "Accuracy on test set".to_string(),
        )
        .add_metric(
            "Training Accuracy".to_string(),
            MetricValue::Float(0.956),
            "Accuracy on training set".to_string(),
        )
        .add_metric(
            "Model Parameters".to_string(),
            MetricValue::Integer(7850),
            "Total number of parameters".to_string(),
        )
        .ethical_considerations("MNIST dataset is synthetically balanced and may not represent real-world digit distribution. Model should not be used for automated decision-making without human oversight.".to_string())
        .build();

    Ok(card)
}

fn validate_model_for_publishing(
    model_dir: &PathBuf,
    metadata: &ExtendedMetadata,
) -> Result<ValidationResult> {
    println!("  Running model validation checks...");

    let mut validation = ValidationResult {
        is_valid: true,
        errors: Vec::new(),
        warnings: Vec::new(),
        info: Vec::new(),
    };

    // Check required files
    let required_files = ["config.toml", "model.safetensors", "README.md"];
    for file in &required_files {
        let file_path = model_dir.join(file);
        if !file_path.exists() {
            validation.is_valid = false;
            validation
                .errors
                .push(format!("Required file missing: {}", file));
        } else {
            validation
                .info
                .push(format!("✓ Found required file: {}", file));
        }
    }

    // Check file sizes
    let model_file = model_dir.join("model.safetensors");
    if model_file.exists() {
        let size = std::fs::metadata(&model_file)?.len();
        if size > 100 * 1024 * 1024 {
            // 100MB
            validation
                .warnings
                .push("Model file is quite large (>100MB), consider compression".to_string());
        } else {
            validation
                .info
                .push(format!("✓ Model size OK: {} bytes", size));
        }
    }

    // Validate metadata
    if metadata.model_info.name.is_empty() {
        validation.is_valid = false;
        validation.errors.push("Model name is required".to_string());
    }

    if metadata.model_info.description.len() < 10 {
        validation
            .warnings
            .push("Model description is very short, consider adding more details".to_string());
    }

    if metadata.model_info.tags.is_empty() {
        validation
            .warnings
            .push("No tags specified, tags help with discoverability".to_string());
    }

    // Check license
    let valid_licenses = ["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0", "CC0-1.0"];
    if !valid_licenses.contains(&metadata.model_info.license.as_str()) {
        validation.warnings.push(format!(
            "License '{}' not in common list, ensure it's valid",
            metadata.model_info.license
        ));
    }

    // Security checks
    let security_result = run_security_scan(model_dir)?;
    if !security_result.is_safe {
        validation.is_valid = false;
        validation.errors.extend(
            security_result
                .threats
                .into_iter()
                .map(|t| format!("Security: {}", t)),
        );
    }

    Ok(validation)
}

fn sign_model_for_publishing(model_dir: &PathBuf) -> Result<ModelSignature> {
    let mut security_manager = SecurityManager::new();

    // Generate or load signing key
    let key_pair =
        SecurityManager::generate_key_pair("example_key".to_string(), SignatureAlgorithm::Ed25519)?;
    security_manager.add_key(key_pair);

    // Sign the model directory
    let signature = security_manager.sign_model(model_dir, "example_key", None)?;

    Ok(signature)
}

fn upload_model(
    model_dir: &PathBuf,
    metadata: &ExtendedMetadata,
    config: &UploadConfig,
) -> Result<UploadResult> {
    let uploader = ModelUploader::new()?;

    // Start upload
    println!("  Starting model upload...");
    let upload_result = uploader.upload_model(model_dir, metadata, config)?;

    Ok(upload_result)
}

fn demonstrate_version_management() -> Result<()> {
    println!("  Managing model versions...");

    // Create version 1.0.0
    let v1 = Version::new(1, 0, 0);
    println!("  ✓ Created version: {}", v1);

    // Create version 1.1.0 (minor update)
    let v1_1 = Version::new(1, 1, 0);
    println!("  ✓ Created version: {} (minor update)", v1_1);

    // Create version 2.0.0 (major update)
    let v2 = Version::new(2, 0, 0);
    println!("  ✓ Created version: {} (major update)", v2);

    // Version comparison
    println!("  Version comparisons:");
    println!("    {} < {} = {}", v1, v1_1, v1 < v1_1);
    println!("    {} < {} = {}", v1_1, v2, v1_1 < v2);

    // Version history
    let mut history = VersionHistory::new(v1.clone(), "admin".to_string());
    history.add_version(
        v1_1,
        "Performance improvements".to_string(),
        "admin".to_string(),
        None,
    )?;
    history.add_version(
        v2,
        "Architecture overhaul".to_string(),
        "admin".to_string(),
        None,
    )?;

    println!(
        "  ✓ Version history created with {} versions",
        history.list_versions().len()
    );

    Ok(())
}

fn demonstrate_batch_publishing() -> Result<()> {
    println!("  Preparing batch publishing...");

    let batch_config = BatchPublishConfig {
        models: vec![
            ("model-1".to_string(), std::env::temp_dir().join("model1")),
            ("model-2".to_string(), std::env::temp_dir().join("model2")),
            ("model-3".to_string(), std::env::temp_dir().join("model3")),
        ],
        concurrent_uploads: 2,
        validate_all_before_upload: true,
        stop_on_first_error: false,
        progress_callback: Some(Box::new(|completed, total| {
            println!("    Batch progress: {}/{} models", completed, total);
        })),
    };

    // This would normally upload multiple models
    println!(
        "  ✓ Batch configuration prepared for {} models",
        batch_config.models.len()
    );
    println!(
        "  ✓ Concurrent uploads: {}",
        batch_config.concurrent_uploads
    );

    Ok(())
}

fn demonstrate_private_publishing() -> Result<()> {
    println!("  Setting up private model publishing...");

    let private_config = PrivateModelConfig {
        visibility: ModelVisibility::Private,
        access_control: AccessControl {
            allowed_users: vec!["user1".to_string(), "user2".to_string()],
            allowed_organizations: vec!["my-org".to_string()],
            require_api_key: true,
            ip_whitelist: vec!["192.168.1.0/24".to_string()],
        },
        encryption: EncryptionConfig {
            encrypt_model_weights: true,
            encrypt_metadata: false,
            algorithm: EncryptionAlgorithm::AES256,
        },
    };

    println!("  ✓ Private model configuration:");
    println!("    Visibility: {:?}", private_config.visibility);
    println!(
        "    Allowed users: {:?}",
        private_config.access_control.allowed_users
    );
    println!(
        "    Encryption enabled: {}",
        private_config.encryption.encrypt_model_weights
    );

    Ok(())
}

fn demonstrate_model_updates(model_dir: &PathBuf) -> Result<()> {
    println!("  Demonstrating model updates...");

    // Create update package
    let update_info = ModelUpdateInfo {
        current_version: Version::new(1, 0, 0),
        new_version: Version::new(1, 0, 1),
        update_type: UpdateType::Patch,
        changes: vec![
            "Fixed numerical stability issue".to_string(),
            "Improved inference speed by 5%".to_string(),
            "Updated documentation".to_string(),
        ],
        breaking_changes: vec![],
        migration_guide: None,
    };

    println!("  ✓ Update package created:");
    println!(
        "    {} → {}",
        update_info.current_version, update_info.new_version
    );
    println!("    Type: {:?}", update_info.update_type);
    println!("    Changes: {}", update_info.changes.len());

    // Create patch file (simplified)
    let patch_data = create_model_patch(&update_info)?;
    println!("  ✓ Patch created: {} bytes", patch_data.len());

    Ok(())
}

// Helper functions and types

#[derive(Debug)]
struct ValidationResult {
    is_valid: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
    info: Vec<String>,
}

#[derive(Debug)]
struct SecurityScanResult {
    is_safe: bool,
    threats: Vec<String>,
}

#[derive(Debug)]
struct UploadResult {
    success: bool,
    model_id: String,
    upload_size_bytes: u64,
    upload_time_seconds: f64,
    url: String,
}

struct BatchPublishConfig {
    models: Vec<(String, PathBuf)>,
    concurrent_uploads: usize,
    validate_all_before_upload: bool,
    stop_on_first_error: bool,
    progress_callback: Option<Box<dyn Fn(usize, usize)>>,
}

#[derive(Debug)]
#[allow(dead_code)]
enum ModelVisibility {
    Public,
    Private,
    Unlisted,
}

#[allow(dead_code)]
struct AccessControl {
    allowed_users: Vec<String>,
    allowed_organizations: Vec<String>,
    require_api_key: bool,
    ip_whitelist: Vec<String>,
}

#[allow(dead_code)]
struct EncryptionConfig {
    encrypt_model_weights: bool,
    encrypt_metadata: bool,
    algorithm: EncryptionAlgorithm,
}

#[derive(Debug)]
#[allow(dead_code)]
enum EncryptionAlgorithm {
    AES256,
    ChaCha20,
}

#[allow(dead_code)]
struct PrivateModelConfig {
    visibility: ModelVisibility,
    access_control: AccessControl,
    encryption: EncryptionConfig,
}

#[derive(Debug)]
#[allow(dead_code)]
enum UpdateType {
    Major,
    Minor,
    Patch,
}

#[allow(dead_code)]
struct ModelUpdateInfo {
    current_version: Version,
    new_version: Version,
    update_type: UpdateType,
    changes: Vec<String>,
    breaking_changes: Vec<String>,
    migration_guide: Option<String>,
}

struct ModelUploader;

impl ModelUploader {
    fn new() -> Result<Self> {
        Ok(Self)
    }

    fn upload_model(
        &self,
        _model_dir: &PathBuf,
        _metadata: &ExtendedMetadata,
        config: &UploadConfig,
    ) -> Result<UploadResult> {
        // Simulate upload
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(UploadResult {
            success: true,
            model_id: format!("example-model-{}", chrono::Utc::now().timestamp()),
            upload_size_bytes: 31744,
            upload_time_seconds: 2.5,
            url: format!("https://hub.torsh.dev/example-model"),
        })
    }
}

// Helper function implementations
fn display_metadata(metadata: &ExtendedMetadata) {
    println!("  Name: {}", metadata.model_info.name);
    println!("  Version: {}", metadata.model_info.version);
    println!("  Author: {}", metadata.model_info.author);
    println!("  License: {}", metadata.model_info.license);
    println!("  Tags: {:?}", metadata.model_info.tags);
    println!("  Files: {}", metadata.file_metadata.len());
}

fn display_validation_result(result: &ValidationResult) {
    println!(
        "  Validation result: {}",
        if result.is_valid {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    for error in &result.errors {
        println!("    ✗ Error: {}", error);
    }

    for warning in &result.warnings {
        println!("    ⚠ Warning: {}", warning);
    }

    for info in &result.info {
        println!("    ℹ {}", info);
    }
}

fn display_upload_result(result: &UploadResult) {
    println!(
        "  Upload result: {}",
        if result.success {
            "✓ SUCCESS"
        } else {
            "✗ FAILED"
        }
    );
    println!("    Model ID: {}", result.model_id);
    println!("    Size: {} bytes", result.upload_size_bytes);
    println!("    Time: {:.1}s", result.upload_time_seconds);
    println!("    URL: {}", result.url);
}

fn save_model_weights(
    state_dict: &HashMap<String, torsh_tensor::Tensor>,
    path: &PathBuf,
) -> Result<()> {
    // Simplified weight saving - in practice use SafeTensors or similar
    let data = format!("{{\"weights\": {}}}", state_dict.len());
    std::fs::write(path, data)?;
    Ok(())
}

fn save_tensor(tensor: &torsh_tensor::Tensor, path: &PathBuf) -> Result<()> {
    // Simplified tensor saving
    let data = format!("tensor shape: {:?}", tensor.shape());
    std::fs::write(path, data)?;
    Ok(())
}

fn run_security_scan(_model_dir: &PathBuf) -> Result<SecurityScanResult> {
    // Simplified security scan
    Ok(SecurityScanResult {
        is_safe: true,
        threats: vec![],
    })
}

fn create_model_patch(_update_info: &ModelUpdateInfo) -> Result<Vec<u8>> {
    // Simplified patch creation
    Ok(b"patch data".to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let version = Version::new(1, 2, 3);
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_upload_config() {
        let config = UploadConfig {
            endpoint: "https://test.example.com/upload".to_string(),
            auth_token: Some("test_token".to_string()),
            validate: true,
            compress: true,
            progress: false,
        };

        assert_eq!(config.endpoint, "https://test.example.com/upload");
        assert!(config.validate);
        assert!(config.compress);
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec!["Test warning".to_string()],
            info: vec!["Test info".to_string()],
        };

        assert!(result.is_valid);
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.info.len(), 1);

        result.errors.push("Test error".to_string());
        result.is_valid = false;

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
    }
}
