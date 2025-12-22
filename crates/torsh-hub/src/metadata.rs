//! Metadata management system for model information
//!
//! This module provides a comprehensive system for managing model metadata,
//! including automatic metadata extraction, validation, and synchronization.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

use crate::model_info::{ModelCard, ModelInfo, Version};
use crate::registry::RegistryEntry;

/// Comprehensive metadata manager
pub struct MetadataManager {
    metadata_dir: PathBuf,
    cache_dir: PathBuf,
}

/// Extended metadata for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedMetadata {
    pub model_info: ModelInfo,
    pub model_card: Option<ModelCard>,
    pub registry_entry: Option<RegistryEntry>,
    pub file_metadata: Vec<FileMetadata>,
    pub provenance: ProvenanceInfo,
    pub performance_metrics: PerformanceMetrics,
    pub usage_statistics: UsageStatistics,
    pub quality_scores: QualityScores,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// File metadata for model files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub file_path: String,
    pub file_type: FileType,
    pub size_bytes: u64,
    pub checksum: String,
    pub creation_date: chrono::DateTime<chrono::Utc>,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub compression: Option<CompressionInfo>,
    pub encryption: Option<EncryptionInfo>,
}

/// File type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileType {
    ModelWeights,
    Configuration,
    Tokenizer,
    Documentation,
    Example,
    Test,
    Benchmark,
    Other(String),
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    pub algorithm: String,
    pub compression_ratio: f32,
    pub original_size_bytes: u64,
}

/// Encryption information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInfo {
    pub algorithm: String,
    pub key_id: String,
    pub encrypted_at: chrono::DateTime<chrono::Utc>,
}

/// Provenance information tracking the model's history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceInfo {
    pub source_repository: Option<String>,
    pub source_commit: Option<String>,
    pub training_job_id: Option<String>,
    pub trained_by: String,
    pub training_start: Option<chrono::DateTime<chrono::Utc>>,
    pub training_end: Option<chrono::DateTime<chrono::Utc>>,
    pub parent_model: Option<String>,
    pub derived_models: Vec<String>,
    pub training_script: Option<String>,
    pub environment_info: EnvironmentInfo,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub framework_version: String,
    pub python_version: String,
    pub gpu_info: Vec<GpuInfo>,
    pub system_info: SystemInfo,
    pub dependencies: HashMap<String, String>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: Option<String>,
    pub driver_version: String,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub memory_gb: f32,
    pub storage_type: String,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub inference_latency_ms: HashMap<String, f32>, // e.g., "batch_1": 50.0
    pub throughput_samples_per_second: HashMap<String, f32>,
    pub memory_usage_mb: HashMap<String, f32>,
    pub energy_consumption_j: Option<f32>,
    pub co2_emissions_g: Option<f32>,
    pub benchmark_scores: HashMap<String, f32>,
    pub accuracy_metrics: HashMap<String, f32>,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageStatistics {
    pub total_downloads: u64,
    pub monthly_downloads: HashMap<String, u64>, // "2024-01": 150
    pub user_ratings: Vec<UserRating>,
    pub usage_contexts: HashMap<String, u64>, // "research": 100, "production": 50
    pub geographic_distribution: HashMap<String, u64>, // "US": 200, "EU": 150
}

/// User rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRating {
    pub user_id: String,
    pub rating: f32, // 1.0 to 5.0
    pub comment: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub verified: bool,
}

/// Quality scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    pub documentation_score: f32, // 0.0 to 1.0
    pub code_quality_score: f32,
    pub reproducibility_score: f32,
    pub performance_score: f32,
    pub safety_score: f32,
    pub ethical_score: f32,
    pub overall_score: f32,
}

impl MetadataManager {
    /// Create a new metadata manager
    pub fn new<P: AsRef<Path>>(metadata_dir: P, cache_dir: P) -> Result<Self> {
        let metadata_dir = metadata_dir.as_ref().to_path_buf();
        let cache_dir = cache_dir.as_ref().to_path_buf();

        std::fs::create_dir_all(&metadata_dir)?;
        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            metadata_dir,
            cache_dir,
        })
    }

    /// Extract metadata from a model directory
    pub fn extract_metadata(&self, model_dir: &Path, _model_id: &str) -> Result<ExtendedMetadata> {
        // Load or create model info
        let model_info = self.load_or_create_model_info(model_dir)?;

        // Extract file metadata
        let file_metadata = self.extract_file_metadata(model_dir)?;

        // Create default metadata
        let mut metadata = ExtendedMetadata {
            model_info,
            model_card: None,
            registry_entry: None,
            file_metadata,
            provenance: self.extract_provenance_info(model_dir)?,
            performance_metrics: PerformanceMetrics::default(),
            usage_statistics: UsageStatistics::default(),
            quality_scores: self.calculate_quality_scores(model_dir)?,
            last_updated: chrono::Utc::now(),
        };

        // Try to load model card if it exists
        if let Ok(model_card) = self.load_model_card(model_dir) {
            metadata.model_card = Some(model_card);
        }

        Ok(metadata)
    }

    /// Save metadata to disk
    pub fn save_metadata(&self, model_id: &str, metadata: &ExtendedMetadata) -> Result<()> {
        let metadata_path = self.metadata_dir.join(format!("{}.json", model_id));
        let content = serde_json::to_string_pretty(metadata)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        std::fs::write(metadata_path, content)?;
        Ok(())
    }

    /// Load metadata from disk
    pub fn load_metadata(&self, model_id: &str) -> Result<ExtendedMetadata> {
        let metadata_path = self.metadata_dir.join(format!("{}.json", model_id));
        if !metadata_path.exists() {
            return Err(TorshError::IoError(format!(
                "Metadata not found for {}",
                model_id
            )));
        }

        let content = std::fs::read_to_string(metadata_path)?;
        serde_json::from_str(&content).map_err(|e| TorshError::SerializationError(e.to_string()))
    }

    /// Update metadata with new information
    pub fn update_metadata(
        &self,
        model_id: &str,
        update_fn: impl FnOnce(&mut ExtendedMetadata),
    ) -> Result<()> {
        let mut metadata = self.load_metadata(model_id)?;
        update_fn(&mut metadata);
        metadata.last_updated = chrono::Utc::now();
        self.save_metadata(model_id, &metadata)
    }

    /// Validate metadata consistency
    pub fn validate_metadata(&self, metadata: &ExtendedMetadata) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Check model info consistency
        metadata
            .model_info
            .validate()
            .map_err(|e| issues.push(format!("Model info validation failed: {}", e)))
            .ok();

        // Check file metadata consistency
        for file_meta in &metadata.file_metadata {
            if file_meta.size_bytes == 0 {
                issues.push(format!("File {} has zero size", file_meta.file_path));
            }
            if file_meta.checksum.len() != 64 {
                issues.push(format!("File {} has invalid checksum", file_meta.file_path));
            }
        }

        // Check quality scores
        let scores = &metadata.quality_scores;
        if scores.overall_score < 0.0 || scores.overall_score > 1.0 {
            issues.push("Overall quality score out of range".to_string());
        }

        // Check version consistency
        if let Some(ref registry_entry) = metadata.registry_entry {
            if registry_entry.version != metadata.model_info.version {
                issues.push("Version mismatch between model info and registry entry".to_string());
            }
        }

        Ok(issues)
    }

    /// Synchronize metadata across different storage systems
    pub fn sync_metadata(&self, model_id: &str) -> Result<()> {
        let metadata = self.load_metadata(model_id)?;

        // Sync with model info file
        let model_info_path = self.metadata_dir.join(format!("{}_info.json", model_id));
        metadata.model_info.to_file(&model_info_path)?;

        // Sync with model card if it exists
        if let Some(ref model_card) = metadata.model_card {
            let card_path = self.metadata_dir.join(format!("{}_card.json", model_id));
            let content = serde_json::to_string_pretty(model_card)
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;
            std::fs::write(card_path, content)?;
        }

        Ok(())
    }

    /// Search metadata by criteria
    pub fn search_metadata(&self, criteria: &MetadataSearchCriteria) -> Result<Vec<String>> {
        let mut matches = Vec::new();

        for entry in std::fs::read_dir(&self.metadata_dir)? {
            let entry = entry?;
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".json")
                    && !filename.contains("_info")
                    && !filename.contains("_card")
                {
                    let model_id = filename.strip_suffix(".json").unwrap();
                    if let Ok(metadata) = self.load_metadata(model_id) {
                        if self.matches_criteria(&metadata, criteria) {
                            matches.push(model_id.to_string());
                        }
                    }
                }
            }
        }

        Ok(matches)
    }

    /// Generate metadata report
    pub fn generate_report(&self, model_id: &str) -> Result<String> {
        let metadata = self.load_metadata(model_id)?;
        let issues = self.validate_metadata(&metadata)?;

        let mut report = String::new();
        report.push_str(&format!("# Metadata Report for {}\n\n", model_id));

        // Model info summary
        report.push_str("## Model Information\n");
        report.push_str(&format!("- Name: {}\n", metadata.model_info.name));
        report.push_str(&format!("- Author: {}\n", metadata.model_info.author));
        report.push_str(&format!("- Version: {}\n", metadata.model_info.version));
        report.push_str(&format!(
            "- Description: {}\n",
            metadata.model_info.description
        ));
        report.push('\n');

        // File information
        report.push_str("## Files\n");
        for file in &metadata.file_metadata {
            report.push_str(&format!(
                "- {} ({:.2} MB, {:?})\n",
                file.file_path,
                file.size_bytes as f64 / 1_000_000.0,
                file.file_type
            ));
        }
        report.push('\n');

        // Quality scores
        report.push_str("## Quality Scores\n");
        let scores = &metadata.quality_scores;
        report.push_str(&format!("- Overall: {:.2}\n", scores.overall_score));
        report.push_str(&format!(
            "- Documentation: {:.2}\n",
            scores.documentation_score
        ));
        report.push_str(&format!("- Performance: {:.2}\n", scores.performance_score));
        report.push_str(&format!("- Safety: {:.2}\n", scores.safety_score));
        report.push('\n');

        // Issues
        if !issues.is_empty() {
            report.push_str("## Issues\n");
            for issue in issues {
                report.push_str(&format!("- {}\n", issue));
            }
            report.push('\n');
        }

        // Last updated
        report.push_str(&format!(
            "Last updated: {}\n",
            metadata.last_updated.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        Ok(report)
    }

    // Private helper methods

    fn load_or_create_model_info(&self, model_dir: &Path) -> Result<ModelInfo> {
        // Try to load existing model info
        let info_path = model_dir.join("model_info.json");
        if info_path.exists() {
            return ModelInfo::from_file(&info_path);
        }

        // Create basic model info from directory
        let model_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ModelInfo::new(
            model_name,
            "unknown".to_string(),
            Version::new(1, 0, 0),
        ))
    }

    fn extract_file_metadata(&self, model_dir: &Path) -> Result<Vec<FileMetadata>> {
        let mut file_metadata = Vec::new();

        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let metadata = std::fs::metadata(&path)?;
                let checksum = self.calculate_checksum(&path)?;

                let file_type = self.determine_file_type(&path);

                file_metadata.push(FileMetadata {
                    file_path: path.to_string_lossy().to_string(),
                    file_type,
                    size_bytes: metadata.len(),
                    checksum,
                    creation_date: chrono::DateTime::from(metadata.created()?),
                    last_modified: chrono::DateTime::from(metadata.modified()?),
                    compression: None,
                    encryption: None,
                });
            }
        }

        Ok(file_metadata)
    }

    fn calculate_checksum(&self, file_path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};

        let mut file = std::fs::File::open(file_path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn determine_file_type(&self, file_path: &Path) -> FileType {
        if let Some(extension) = file_path.extension() {
            match extension.to_str() {
                Some("bin") | Some("pt") | Some("pth") | Some("ckpt") => FileType::ModelWeights,
                Some("json") | Some("yaml") | Some("toml") => FileType::Configuration,
                Some("md") | Some("txt") | Some("rst") => FileType::Documentation,
                Some("py") | Some("ipynb") => FileType::Example,
                Some("rs") | Some("cpp") | Some("c") => FileType::Other("source".to_string()),
                _ => FileType::Other("unknown".to_string()),
            }
        } else {
            FileType::Other("no_extension".to_string())
        }
    }

    fn extract_provenance_info(&self, _model_dir: &Path) -> Result<ProvenanceInfo> {
        // In a real implementation, this would extract git info, training logs, etc.
        Ok(ProvenanceInfo {
            source_repository: None,
            source_commit: None,
            training_job_id: None,
            trained_by: "unknown".to_string(),
            training_start: None,
            training_end: None,
            parent_model: None,
            derived_models: Vec::new(),
            training_script: None,
            environment_info: EnvironmentInfo {
                framework_version: "unknown".to_string(),
                python_version: "unknown".to_string(),
                gpu_info: Vec::new(),
                system_info: SystemInfo {
                    os: std::env::consts::OS.to_string(),
                    cpu: "unknown".to_string(),
                    memory_gb: 0.0,
                    storage_type: "unknown".to_string(),
                },
                dependencies: HashMap::new(),
            },
        })
    }

    fn load_model_card(&self, model_dir: &Path) -> Result<ModelCard> {
        let card_path = model_dir.join("model_card.json");
        if !card_path.exists() {
            return Err(TorshError::IoError("Model card not found".to_string()));
        }

        let content = std::fs::read_to_string(card_path)?;
        serde_json::from_str(&content).map_err(|e| TorshError::SerializationError(e.to_string()))
    }

    fn calculate_quality_scores(&self, _model_dir: &Path) -> Result<QualityScores> {
        // In a real implementation, this would analyze various aspects of the model
        Ok(QualityScores {
            documentation_score: 0.5,
            code_quality_score: 0.5,
            reproducibility_score: 0.5,
            performance_score: 0.5,
            safety_score: 0.5,
            ethical_score: 0.5,
            overall_score: 0.5,
        })
    }

    fn matches_criteria(
        &self,
        metadata: &ExtendedMetadata,
        criteria: &MetadataSearchCriteria,
    ) -> bool {
        if let Some(ref name_filter) = criteria.name_contains {
            if !metadata
                .model_info
                .name
                .to_lowercase()
                .contains(&name_filter.to_lowercase())
            {
                return false;
            }
        }

        if let Some(ref author_filter) = criteria.author {
            if metadata.model_info.author != *author_filter {
                return false;
            }
        }

        if let Some(min_score) = criteria.min_quality_score {
            if metadata.quality_scores.overall_score < min_score {
                return false;
            }
        }

        true
    }
}

/// Search criteria for metadata
#[derive(Debug, Clone)]
pub struct MetadataSearchCriteria {
    pub name_contains: Option<String>,
    pub author: Option<String>,
    pub min_quality_score: Option<f32>,
    pub file_type: Option<FileType>,
    pub has_model_card: Option<bool>,
    pub version_constraint: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_metadata_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let metadata_dir = temp_dir.path().join("metadata");
        let cache_dir = temp_dir.path().join("cache");

        let _manager = MetadataManager::new(&metadata_dir, &cache_dir).unwrap();

        assert!(metadata_dir.exists());
        assert!(cache_dir.exists());
    }

    #[test]
    fn test_metadata_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let manager = MetadataManager::new(temp_dir.path(), temp_dir.path()).unwrap();

        let model_info = ModelInfo::new(
            "test_model".to_string(),
            "test_author".to_string(),
            Version::new(1, 0, 0),
        );

        let metadata = ExtendedMetadata {
            model_info,
            model_card: None,
            registry_entry: None,
            file_metadata: Vec::new(),
            provenance: ProvenanceInfo {
                source_repository: None,
                source_commit: None,
                training_job_id: None,
                trained_by: "test_user".to_string(),
                training_start: None,
                training_end: None,
                parent_model: None,
                derived_models: Vec::new(),
                training_script: None,
                environment_info: EnvironmentInfo {
                    framework_version: "torsh-1.0".to_string(),
                    python_version: "3.8".to_string(),
                    gpu_info: Vec::new(),
                    system_info: SystemInfo {
                        os: "linux".to_string(),
                        cpu: "x86_64".to_string(),
                        memory_gb: 16.0,
                        storage_type: "ssd".to_string(),
                    },
                    dependencies: HashMap::new(),
                },
            },
            performance_metrics: PerformanceMetrics::default(),
            usage_statistics: UsageStatistics::default(),
            quality_scores: QualityScores {
                documentation_score: 0.8,
                code_quality_score: 0.9,
                reproducibility_score: 0.7,
                performance_score: 0.85,
                safety_score: 0.9,
                ethical_score: 0.8,
                overall_score: 0.83,
            },
            last_updated: chrono::Utc::now(),
        };

        manager.save_metadata("test_model", &metadata).unwrap();
        let loaded = manager.load_metadata("test_model").unwrap();

        assert_eq!(loaded.model_info.name, "test_model");
        assert_eq!(loaded.quality_scores.overall_score, 0.83);
    }

    #[test]
    fn test_metadata_validation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = MetadataManager::new(temp_dir.path(), temp_dir.path()).unwrap();

        let model_info = ModelInfo::new(
            "".to_string(), // Invalid empty name
            "test_author".to_string(),
            Version::new(1, 0, 0),
        );

        let metadata = ExtendedMetadata {
            model_info,
            model_card: None,
            registry_entry: None,
            file_metadata: vec![FileMetadata {
                file_path: "test.bin".to_string(),
                file_type: FileType::ModelWeights,
                size_bytes: 0,                   // Invalid zero size
                checksum: "invalid".to_string(), // Invalid checksum
                creation_date: chrono::Utc::now(),
                last_modified: chrono::Utc::now(),
                compression: None,
                encryption: None,
            }],
            provenance: ProvenanceInfo {
                source_repository: None,
                source_commit: None,
                training_job_id: None,
                trained_by: "test_user".to_string(),
                training_start: None,
                training_end: None,
                parent_model: None,
                derived_models: Vec::new(),
                training_script: None,
                environment_info: EnvironmentInfo {
                    framework_version: "torsh-1.0".to_string(),
                    python_version: "3.8".to_string(),
                    gpu_info: Vec::new(),
                    system_info: SystemInfo {
                        os: "linux".to_string(),
                        cpu: "x86_64".to_string(),
                        memory_gb: 16.0,
                        storage_type: "ssd".to_string(),
                    },
                    dependencies: HashMap::new(),
                },
            },
            performance_metrics: PerformanceMetrics::default(),
            usage_statistics: UsageStatistics::default(),
            quality_scores: QualityScores {
                documentation_score: 0.8,
                code_quality_score: 0.9,
                reproducibility_score: 0.7,
                performance_score: 0.85,
                safety_score: 0.9,
                ethical_score: 0.8,
                overall_score: 1.5, // Invalid score > 1.0
            },
            last_updated: chrono::Utc::now(),
        };

        let issues = manager.validate_metadata(&metadata).unwrap();
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|issue| issue.contains("zero size")));
        assert!(issues
            .iter()
            .any(|issue| issue.contains("invalid checksum")));
        assert!(issues.iter().any(|issue| issue.contains("out of range")));
    }
}
