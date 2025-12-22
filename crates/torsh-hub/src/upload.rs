//! Model uploading functionality with versioning support

use crate::model_info::{ModelInfo, Version};
use crate::registry::{ModelRegistry, RegistryEntry};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

/// Upload configuration
#[derive(Debug, Clone)]
pub struct UploadConfig {
    pub endpoint: String,
    pub auth_token: Option<String>,
    pub validate: bool,
    pub compress: bool,
    pub progress: bool,
}

/// Version change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionChangeInfo {
    pub previous_version: Option<Version>,
    pub new_version: Version,
    pub is_breaking_change: bool,
    pub changelog: String,
    pub migration_notes: Option<String>,
}

/// Result of a publishing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResult {
    pub model_id: String,
    pub upload_url: String,
    pub version_info: VersionChangeInfo,
    pub is_new_model: bool,
    pub registry_updated: bool,
}

/// Publishing strategy
#[derive(Debug, Clone)]
pub enum PublishStrategy {
    /// Only allow patch version increments (1.0.0 -> 1.0.1)
    PatchOnly,
    /// Allow minor and patch increments (1.0.0 -> 1.1.0 or 1.0.1)
    MinorAndPatch,
    /// Allow all version increments including major (1.0.0 -> 2.0.0)
    AllowAll,
    /// Require manual approval for breaking changes
    RequireApprovalForBreaking,
}

/// Version validation rules
#[derive(Debug, Clone)]
pub struct VersionValidationRules {
    pub strategy: PublishStrategy,
    pub require_changelog: bool,
    pub require_migration_notes_for_breaking: bool,
    pub min_version_gap: Option<Version>,
    pub allowed_prerelease_patterns: Vec<String>,
}

impl Default for VersionValidationRules {
    fn default() -> Self {
        Self {
            strategy: PublishStrategy::AllowAll,
            require_changelog: true,
            require_migration_notes_for_breaking: true,
            min_version_gap: None,
            allowed_prerelease_patterns: vec![
                "alpha".to_string(),
                "beta".to_string(),
                "rc".to_string(),
            ],
        }
    }
}

impl Default for UploadConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://torsh.rs/api/upload".to_string(),
            auth_token: None,
            validate: true,
            compress: true,
            progress: true,
        }
    }
}

/// Package a model for upload
pub fn package_model(model_path: &Path, output_path: &Path, model_info: &ModelInfo) -> Result<()> {
    // Validate model info
    model_info.validate()?;

    // Create package directory
    let package_dir = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".torsh_package_tmp");

    fs::create_dir_all(&package_dir)?;

    // Copy model file
    let model_filename = model_path
        .file_name()
        .ok_or_else(|| TorshError::InvalidArgument("Invalid model path".to_string()))?;
    let packaged_model_path = package_dir.join(model_filename);
    fs::copy(model_path, &packaged_model_path)?;

    // Write model info
    let info_path = package_dir.join("model_info.json");
    model_info.to_file(&info_path)?;

    // Create archive
    create_package_archive(&package_dir, output_path)?;

    // Clean up temporary directory
    fs::remove_dir_all(&package_dir)?;

    Ok(())
}

/// Upload a model to the hub with versioning support
pub fn upload_model(
    model_path: &Path,
    repo: &str,
    model_name: &str,
    version: &Version,
    model_info: ModelInfo,
    config: UploadConfig,
) -> Result<String> {
    // Validate inputs
    if !model_path.exists() {
        return Err(TorshError::IoError(format!(
            "Model file not found: {:?}",
            model_path
        )));
    }

    // Package model
    let temp_package = std::env::temp_dir().join(format!("{}_package.torsh", model_name));
    package_model(model_path, &temp_package, &model_info)?;

    // Calculate checksums
    let file_hash = calculate_file_hash(&temp_package)?;
    let file_size = fs::metadata(&temp_package)?.len();

    // Prepare upload metadata
    let mut upload_metadata = HashMap::new();
    upload_metadata.insert("repo", repo.to_string());
    upload_metadata.insert("model", model_name.to_string());
    upload_metadata.insert("version", version.to_string());
    upload_metadata.insert("hash", file_hash.clone());
    upload_metadata.insert("size", file_size.to_string());

    // Perform upload
    if config.progress {
        println!("Uploading {} to {}/{}", model_name, repo, version);
    }

    let upload_url = perform_upload(&temp_package, &config, &upload_metadata)?;

    // Clean up
    fs::remove_file(&temp_package)?;

    if config.progress {
        println!("Upload complete: {}", upload_url);
    }

    Ok(upload_url)
}

/// Enhanced upload with version management and registry integration
pub fn upload_model_with_versioning(
    model_path: &Path,
    repo: &str,
    model_info: ModelInfo,
    upload_config: UploadConfig,
    registry_path: Option<&Path>,
    changelog: String,
    migration_notes: Option<String>,
) -> Result<PublishResult> {
    // Load or create registry
    let mut registry = if let Some(registry_path) = registry_path {
        ModelRegistry::new(registry_path)?
    } else {
        ModelRegistry::new(std::env::temp_dir().join("torsh_registry"))?
    };

    let model_id = format!("{}/{}", repo, model_info.name);

    // Check if model already exists
    let (is_new_model, version_info) = if let Some(existing_entry) = registry.get_model(&model_id) {
        // Existing model - validate version progression
        let new_version = &model_info.version;
        let current_version = &existing_entry.version;

        if new_version <= current_version {
            return Err(TorshError::InvalidArgument(format!(
                "New version {} must be greater than current version {}",
                new_version, current_version
            )));
        }

        let is_breaking = new_version.is_breaking_change(current_version);
        (
            false,
            VersionChangeInfo {
                previous_version: Some(current_version.clone()),
                new_version: new_version.clone(),
                is_breaking_change: is_breaking,
                changelog: changelog.clone(),
                migration_notes: migration_notes.clone(),
            },
        )
    } else {
        // New model
        (
            true,
            VersionChangeInfo {
                previous_version: None,
                new_version: model_info.version.clone(),
                is_breaking_change: false,
                changelog: "Initial release".to_string(),
                migration_notes: None,
            },
        )
    };

    // Validate model before upload
    validate_model_for_upload(model_path, &model_info)?;

    // Upload the model
    let upload_url = upload_model(
        model_path,
        repo,
        &model_info.name,
        &model_info.version,
        model_info.clone(),
        upload_config,
    )?;

    // Update version history in model info
    let mut updated_model_info = model_info;
    if let Some(ref mut history) = updated_model_info.version_history {
        history.add_version(
            version_info.new_version.clone(),
            changelog.clone(),
            updated_model_info.author.clone(),
            migration_notes,
        )?;
    }

    // Create or update registry entry
    let registry_entry =
        create_registry_entry_from_model_info(&updated_model_info, repo, &upload_url)?;

    if is_new_model {
        registry.register_model(registry_entry)?;
    } else {
        registry.update_model(registry_entry)?;
    }

    Ok(PublishResult {
        model_id: model_id.clone(),
        upload_url,
        version_info,
        is_new_model,
        registry_updated: true,
    })
}

/// Upload a directory of models
pub fn upload_model_directory(
    dir_path: &Path,
    repo: &str,
    config: UploadConfig,
) -> Result<Vec<String>> {
    if !dir_path.is_dir() {
        return Err(TorshError::InvalidArgument(
            "Path must be a directory".to_string(),
        ));
    }

    let mut uploaded_urls = Vec::new();

    // Find all model files
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("torsh") {
            // Look for accompanying model info
            let info_path = path.with_extension("json");
            let model_info = if info_path.exists() {
                ModelInfo::from_file(&info_path)?
            } else {
                // Create basic model info
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();

                ModelInfo::new_with_string_version(
                    name.clone(),
                    "Unknown".to_string(),
                    "1.0.0".to_string(),
                )?
            };

            let url = upload_model(
                &path,
                repo,
                &model_info.name,
                &model_info.version,
                model_info.clone(),
                config.clone(),
            )?;

            uploaded_urls.push(url);
        }
    }

    Ok(uploaded_urls)
}

/// Create a package archive
fn create_package_archive(src_dir: &Path, dest_path: &Path) -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use tar::Builder;

    let file = File::create(dest_path)?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut archive = Builder::new(encoder);

    // Add all files in directory
    for entry in fs::read_dir(src_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path
            .file_name()
            .ok_or_else(|| TorshError::InvalidArgument("Invalid filename".to_string()))?;

        if path.is_file() {
            archive.append_file(name, &mut File::open(&path)?)?;
        }
    }

    archive.finish()?;
    Ok(())
}

/// Calculate file hash
fn calculate_file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Perform the actual upload (placeholder)
fn perform_upload(
    file_path: &Path,
    config: &UploadConfig,
    metadata: &HashMap<&str, String>,
) -> Result<String> {
    // In a real implementation, this would:
    // 1. Authenticate with the hub using auth_token
    // 2. Request upload URL from the server
    // 3. Upload the file with progress tracking
    // 4. Verify the upload

    if config.validate {
        // Validate file before upload
        let size = fs::metadata(file_path)?.len();
        if size == 0 {
            return Err(TorshError::InvalidArgument(
                "Cannot upload empty file".to_string(),
            ));
        }
    }

    // Simulate upload
    if config.progress {
        println!(
            "Uploading {} bytes...",
            metadata.get("size").unwrap_or(&"0".to_string())
        );
    }

    // Return mock URL
    Ok(format!(
        "{}/{}/{}/{}",
        config.endpoint.trim_end_matches("/api/upload"),
        metadata.get("repo").unwrap_or(&"unknown".to_string()),
        metadata.get("model").unwrap_or(&"unknown".to_string()),
        metadata.get("version").unwrap_or(&"latest".to_string())
    ))
}

/// Validate a model before upload
pub fn validate_model_for_upload(model_path: &Path, model_info: &ModelInfo) -> Result<()> {
    // Check file exists
    if !model_path.exists() {
        return Err(TorshError::IoError(format!(
            "Model file not found: {:?}",
            model_path
        )));
    }

    // Validate model info
    model_info.validate()?;

    // Check file size
    let metadata = fs::metadata(model_path)?;
    if metadata.len() == 0 {
        return Err(TorshError::InvalidArgument(
            "Model file is empty".to_string(),
        ));
    }

    // Check file extension
    let ext = model_path.extension().and_then(|s| s.to_str());
    if !matches!(ext, Some("torsh") | Some("pt") | Some("pth")) {
        return Err(TorshError::InvalidArgument(
            "Model file must have .torsh, .pt, or .pth extension".to_string(),
        ));
    }

    Ok(())
}

/// Create a registry entry from model info
fn create_registry_entry_from_model_info(
    model_info: &ModelInfo,
    _repo: &str,
    upload_url: &str,
) -> Result<RegistryEntry> {
    use crate::registry::{create_registry_entry, HardwareSpec, ModelCategory};

    let mut entry = create_registry_entry(
        model_info.name.clone(),
        model_info.author.clone(),
        upload_url.to_string(),
        model_info.description.clone(),
    );

    // Update with proper version
    entry.version = model_info.version.clone();

    // Set other fields from model info
    entry.tags = model_info.tags.clone();

    // Try to infer category from tags
    entry.category = if model_info.tags.contains(&"vision".to_string()) {
        ModelCategory::Vision
    } else if model_info.tags.contains(&"nlp".to_string())
        || model_info.tags.contains(&"text".to_string())
    {
        ModelCategory::NLP
    } else if model_info.tags.contains(&"audio".to_string()) {
        ModelCategory::Audio
    } else if model_info.tags.contains(&"multimodal".to_string()) {
        ModelCategory::Multimodal
    } else if model_info.tags.contains(&"rl".to_string())
        || model_info.tags.contains(&"reinforcement".to_string())
    {
        ModelCategory::ReinforcementLearning
    } else if model_info.tags.contains(&"tabular".to_string()) {
        ModelCategory::TabularData
    } else if model_info.tags.contains(&"timeseries".to_string()) {
        ModelCategory::TimeSeriesForecasting
    } else if model_info.tags.contains(&"generative".to_string()) {
        ModelCategory::GenerativeAI
    } else {
        ModelCategory::Other("unspecified".to_string())
    };

    // Update hardware requirements from model_info if available
    entry.hardware_requirements = HardwareSpec {
        min_ram_gb: model_info.requirements.hardware.min_ram_gb,
        recommended_ram_gb: model_info.requirements.hardware.recommended_ram_gb,
        min_gpu_memory_gb: model_info.requirements.hardware.min_gpu_memory_gb,
        recommended_gpu_memory_gb: model_info.requirements.hardware.recommended_gpu_memory_gb,
        supports_cpu: true, // Default assumption
        supports_gpu: model_info.requirements.hardware.min_gpu_memory_gb.is_some(),
        supports_tpu: false, // Default assumption
    };

    // Extract metrics
    for (key, value) in &model_info.metrics {
        if let crate::model_info::MetricValue::Float(f) = value {
            entry.accuracy_metrics.insert(key.clone(), *f);
        }
    }

    Ok(entry)
}

/// Validate version change according to rules
pub fn validate_version_change(
    current_version: Option<&Version>,
    new_version: &Version,
    rules: &VersionValidationRules,
    changelog: &str,
    migration_notes: &Option<String>,
) -> Result<()> {
    // Check changelog requirement
    if rules.require_changelog && changelog.trim().is_empty() {
        return Err(TorshError::InvalidArgument(
            "Changelog is required for version updates".to_string(),
        ));
    }

    if let Some(current) = current_version {
        // Check version progression
        if new_version <= current {
            return Err(TorshError::InvalidArgument(format!(
                "New version {} must be greater than current version {}",
                new_version, current
            )));
        }

        // Check breaking change requirements
        let is_breaking = new_version.is_breaking_change(current);
        if is_breaking && rules.require_migration_notes_for_breaking && migration_notes.is_none() {
            return Err(TorshError::InvalidArgument(
                "Migration notes are required for breaking changes".to_string(),
            ));
        }

        // Check publishing strategy
        match rules.strategy {
            PublishStrategy::PatchOnly => {
                if new_version.major != current.major || new_version.minor != current.minor {
                    return Err(TorshError::InvalidArgument(
                        "Only patch version increments are allowed".to_string(),
                    ));
                }
            }
            PublishStrategy::MinorAndPatch => {
                if new_version.major != current.major {
                    return Err(TorshError::InvalidArgument(
                        "Major version increments are not allowed".to_string(),
                    ));
                }
            }
            PublishStrategy::RequireApprovalForBreaking => {
                if is_breaking {
                    return Err(TorshError::InvalidArgument(
                        "Breaking changes require manual approval".to_string(),
                    ));
                }
            }
            PublishStrategy::AllowAll => {
                // No restrictions
            }
        }

        // Check minimum version gap
        if let Some(ref min_gap) = rules.min_version_gap {
            let version_diff = Version::new(
                new_version.major - current.major,
                if new_version.major > current.major {
                    0
                } else {
                    new_version.minor - current.minor
                },
                if new_version.major > current.major || new_version.minor > current.minor {
                    0
                } else {
                    new_version.patch - current.patch
                },
            );

            if version_diff < *min_gap {
                return Err(TorshError::InvalidArgument(format!(
                    "Version increment {} is below minimum required gap {}",
                    version_diff, min_gap
                )));
            }
        }
    }

    // Validate prerelease pattern if present
    if let Some(ref prerelease) = new_version.pre_release {
        if !rules.allowed_prerelease_patterns.is_empty() {
            let is_valid = rules
                .allowed_prerelease_patterns
                .iter()
                .any(|pattern| prerelease.starts_with(pattern));

            if !is_valid {
                return Err(TorshError::InvalidArgument(format!(
                    "Prerelease pattern '{}' is not allowed. Allowed patterns: {:?}",
                    prerelease, rules.allowed_prerelease_patterns
                )));
            }
        }
    }

    Ok(())
}

/// Batch publish multiple models with version coordination
pub fn batch_publish_models(
    models: Vec<(PathBuf, ModelInfo, String, Option<String>)>, // (path, info, changelog, migration_notes)
    repo: &str,
    upload_config: UploadConfig,
    validation_rules: VersionValidationRules,
    registry_path: Option<&Path>,
) -> Result<Vec<PublishResult>> {
    let mut results = Vec::new();
    let registry = if let Some(registry_path) = registry_path {
        ModelRegistry::new(registry_path)?
    } else {
        ModelRegistry::new(std::env::temp_dir().join("torsh_registry"))?
    };

    // Sort models by dependency order if needed
    // For now, just process in order

    for (model_path, model_info, changelog, migration_notes) in models {
        // Validate version change
        let model_id = format!("{}/{}", repo, model_info.name);
        let current_version = registry.get_model(&model_id).map(|e| &e.version);

        validate_version_change(
            current_version,
            &model_info.version,
            &validation_rules,
            &changelog,
            &migration_notes,
        )?;

        // Upload model
        let result = upload_model_with_versioning(
            &model_path,
            repo,
            model_info,
            upload_config.clone(),
            registry_path,
            changelog,
            migration_notes,
        )?;

        results.push(result);
    }

    Ok(results)
}

/// Create upload manifest
pub fn create_upload_manifest(models: Vec<(PathBuf, ModelInfo)>) -> Result<UploadManifest> {
    let mut entries = Vec::new();

    for (path, info) in models {
        let hash = calculate_file_hash(&path)?;
        let size = fs::metadata(&path)?.len();

        entries.push(ManifestEntry {
            path: path.to_string_lossy().to_string(),
            model_info: info,
            file_hash: hash,
            file_size: size,
        });
    }

    Ok(UploadManifest {
        version: "1.0".to_string(),
        created_at: chrono::Utc::now(),
        entries,
    })
}

/// Upload manifest
#[derive(Debug, Serialize, Deserialize)]
pub struct UploadManifest {
    pub version: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub entries: Vec<ManifestEntry>,
}

/// Manifest entry
#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub path: String,
    pub model_info: ModelInfo,
    pub file_hash: String,
    pub file_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_calculate_file_hash() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, b"test content").unwrap();

        let hash = calculate_file_hash(&file_path).unwrap();
        assert_eq!(hash.len(), 64); // SHA256 produces 64 hex characters
    }

    #[test]
    fn test_validate_model() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.torsh");
        fs::write(&model_path, b"model data").unwrap();

        let model_info = ModelInfo::new_with_string_version(
            "test_model".to_string(),
            "test_author".to_string(),
            "1.0.0".to_string(),
        )
        .unwrap();

        assert!(validate_model_for_upload(&model_path, &model_info).is_ok());

        // Non-existent file should fail
        let missing_path = temp_dir.path().join("missing.torsh");
        assert!(validate_model_for_upload(&missing_path, &model_info).is_err());
    }

    #[test]
    fn test_version_validation() {
        let rules = VersionValidationRules::default();
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);

        // Valid version progression
        assert!(validate_version_change(Some(&v1), &v2, &rules, "Added features", &None).is_ok());

        // Invalid version progression (downgrade)
        assert!(validate_version_change(Some(&v2), &v1, &rules, "Downgrade", &None).is_err());

        // Breaking change requiring migration notes
        let strict_rules = VersionValidationRules {
            require_migration_notes_for_breaking: true,
            ..Default::default()
        };
        assert!(
            validate_version_change(Some(&v1), &v3, &strict_rules, "Breaking changes", &None)
                .is_err()
        );

        assert!(validate_version_change(
            Some(&v1),
            &v3,
            &strict_rules,
            "Breaking changes",
            &Some("Migration guide available".to_string())
        )
        .is_ok());
    }

    #[test]
    fn test_publish_strategy_validation() {
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);

        // Patch only strategy
        let patch_only_rules = VersionValidationRules {
            strategy: PublishStrategy::PatchOnly,
            ..Default::default()
        };
        let v1_1 = Version::new(1, 0, 1);
        assert!(
            validate_version_change(Some(&v1), &v1_1, &patch_only_rules, "Patch fix", &None)
                .is_ok()
        );
        assert!(
            validate_version_change(Some(&v1), &v2, &patch_only_rules, "Minor change", &None)
                .is_err()
        );

        // Minor and patch strategy
        let minor_patch_rules = VersionValidationRules {
            strategy: PublishStrategy::MinorAndPatch,
            ..Default::default()
        };
        assert!(
            validate_version_change(Some(&v1), &v2, &minor_patch_rules, "Minor change", &None)
                .is_ok()
        );
        assert!(
            validate_version_change(Some(&v1), &v3, &minor_patch_rules, "Major change", &None)
                .is_err()
        );
    }

    #[test]
    fn test_create_registry_entry_from_model_info() {
        let mut model_info = ModelInfo::new(
            "test_model".to_string(),
            "test_author".to_string(),
            Version::new(1, 2, 3),
        );
        model_info.description = "A test model".to_string();
        model_info.tags = vec!["vision".to_string(), "classification".to_string()];
        model_info.metrics.insert(
            "accuracy".to_string(),
            crate::model_info::MetricValue::Float(0.95),
        );

        let entry = create_registry_entry_from_model_info(
            &model_info,
            "test_repo",
            "https://example.com/model",
        )
        .unwrap();

        assert_eq!(entry.name, "test_model");
        assert_eq!(entry.author, "test_author");
        assert_eq!(entry.version, Version::new(1, 2, 3));
        assert_eq!(entry.category, crate::registry::ModelCategory::Vision);
        assert!(entry.tags.contains(&"vision".to_string()));
        assert_eq!(entry.accuracy_metrics.get("accuracy"), Some(&0.95));
    }
}
