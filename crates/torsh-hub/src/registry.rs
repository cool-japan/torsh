//! # Model Registry - Centralized Model Discovery and Management
//!
//! This module provides a comprehensive model registry system for tracking, discovering,
//! and managing machine learning models in the ToRSh ecosystem.
//!
//! ## Features
//!
//! - **Model Discovery**: Search and filter models by category, tags, metrics, and hardware
//! - **Version Management**: Track model versions with semantic versioning
//! - **Metadata Management**: Rich metadata including metrics, hardware requirements, and licenses
//! - **Popularity Tracking**: Track downloads and likes for community feedback
//! - **Hardware Filtering**: Filter models by hardware compatibility and requirements
//! - **Framework Compatibility**: Track which frameworks support each model
//!
//! ## Quick Start
//!
//! ```no_run
//! use torsh_hub::registry::{ModelRegistry, SearchQuery, ModelCategory, SortBy};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new registry
//! let mut registry = ModelRegistry::new("./models")?;
//!
//! // Search for vision models using default SearchQuery and modifying fields
//! let mut query = SearchQuery::default();
//! query.category = Some(ModelCategory::Vision);
//! query.tags = vec!["image-classification".to_string()];
//! query.limit = 10;
//! let results = registry.search(&query);
//!
//! // Get model details
//! if let Some(entry) = results.first() {
//!     println!("Model: {} by {}", entry.name, entry.author);
//!     println!("Downloads: {}", entry.downloads);
//!     println!("Category: {:?}", entry.category);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Registry Entry Structure
//!
//! Each model in the registry contains:
//! - Basic info: name, author, version, description
//! - Popularity: downloads, likes, timestamps
//! - Technical: architecture, framework compatibility, hardware requirements
//! - Performance: model size, inference time, accuracy metrics
//! - Legal: license, paper URL, demo URL
//! - Status: active, deprecated, experimental

use crate::model_info::Version;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use torsh_core::error::{Result, TorshError};

/// Model registry
pub struct ModelRegistry {
    registry_file: PathBuf,
    entries: HashMap<String, RegistryEntry>,
}

/// Registry entry for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub id: String,
    pub name: String,
    pub author: String,
    pub repository: String,
    pub version: Version,
    pub tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub description: String,
    pub metrics: HashMap<String, f32>,
    pub category: ModelCategory,
    pub architecture: String,
    pub framework_compatibility: Vec<String>,
    pub hardware_requirements: HardwareSpec,
    pub model_size_mb: Option<f32>,
    pub inference_time_ms: Option<f32>,
    pub accuracy_metrics: HashMap<String, f32>,
    pub license: String,
    pub paper_url: Option<String>,
    pub demo_url: Option<String>,
    pub status: ModelStatus,
}

impl RegistryEntry {
    /// Check if the model is compatible with a specific framework
    pub fn is_compatible_with_framework(&self, framework: &str) -> bool {
        self.framework_compatibility
            .iter()
            .any(|f| f.to_lowercase() == framework.to_lowercase())
    }

    /// Get the age of the model in days since last update
    pub fn age_in_days(&self) -> i64 {
        let now = chrono::Utc::now();
        (now - self.updated_at).num_days()
    }

    /// Check if the model is actively maintained (updated within last 365 days)
    pub fn is_actively_maintained(&self) -> bool {
        self.age_in_days() <= 365
    }

    /// Get a popularity score based on downloads and likes
    pub fn popularity_score(&self) -> f64 {
        // Simple scoring algorithm: weighted combination of downloads and likes
        let download_score = (self.downloads as f64).log10().max(0.0);
        let like_score = (self.likes as f64).log10().max(0.0);

        // Weight downloads more than likes (2:1 ratio)
        (download_score * 2.0 + like_score) / 3.0
    }

    /// Check if the model meets accuracy threshold for a specific metric
    pub fn meets_accuracy_threshold(&self, metric: &str, threshold: f32) -> bool {
        self.accuracy_metrics
            .get(metric)
            .map(|&value| value >= threshold)
            .unwrap_or(false)
    }

    /// Get a formatted string of all accuracy metrics
    pub fn accuracy_summary(&self) -> String {
        if self.accuracy_metrics.is_empty() {
            "No accuracy metrics available".to_string()
        } else {
            self.accuracy_metrics
                .iter()
                .map(|(metric, value)| format!("{}: {:.3}", metric, value))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

/// Model category for classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    Vision,
    NLP,
    Audio,
    Multimodal,
    ReinforcementLearning,
    TabularData,
    TimeSeriesForecasting,
    GenerativeAI,
    Other(String),
}

impl FromStr for ModelCategory {
    type Err = TorshError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "vision" => Ok(ModelCategory::Vision),
            "nlp" => Ok(ModelCategory::NLP),
            "audio" => Ok(ModelCategory::Audio),
            "multimodal" => Ok(ModelCategory::Multimodal),
            "reinforcementlearning" | "reinforcement_learning" | "rl" => {
                Ok(ModelCategory::ReinforcementLearning)
            }
            "tabulardata" | "tabular_data" | "tabular" => Ok(ModelCategory::TabularData),
            "timeseriesforecasting" | "time_series_forecasting" | "timeseries" => {
                Ok(ModelCategory::TimeSeriesForecasting)
            }
            "generativeai" | "generative_ai" | "generative" => Ok(ModelCategory::GenerativeAI),
            other => Ok(ModelCategory::Other(other.to_string())),
        }
    }
}

impl ModelCategory {
    /// Get the string representation of the model category
    pub fn as_str(&self) -> &str {
        match self {
            ModelCategory::Vision => "vision",
            ModelCategory::NLP => "nlp",
            ModelCategory::Audio => "audio",
            ModelCategory::Multimodal => "multimodal",
            ModelCategory::ReinforcementLearning => "reinforcement_learning",
            ModelCategory::TabularData => "tabular_data",
            ModelCategory::TimeSeriesForecasting => "time_series_forecasting",
            ModelCategory::GenerativeAI => "generative_ai",
            ModelCategory::Other(s) => s,
        }
    }

    /// Get a human-friendly display name for the model category
    pub fn display_name(&self) -> &str {
        match self {
            ModelCategory::Vision => "Computer Vision",
            ModelCategory::NLP => "Natural Language Processing",
            ModelCategory::Audio => "Audio Processing",
            ModelCategory::Multimodal => "Multimodal AI",
            ModelCategory::ReinforcementLearning => "Reinforcement Learning",
            ModelCategory::TabularData => "Tabular Data",
            ModelCategory::TimeSeriesForecasting => "Time Series Forecasting",
            ModelCategory::GenerativeAI => "Generative AI",
            ModelCategory::Other(s) => s,
        }
    }
}

/// Hardware specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub min_ram_gb: Option<f32>,
    pub recommended_ram_gb: Option<f32>,
    pub min_gpu_memory_gb: Option<f32>,
    pub recommended_gpu_memory_gb: Option<f32>,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub supports_tpu: bool,
}

impl HardwareSpec {
    /// Check if the current system meets the minimum hardware requirements
    pub fn meets_minimum_requirements(
        &self,
        available_ram_gb: f32,
        available_gpu_memory_gb: Option<f32>,
    ) -> bool {
        // Check RAM requirements
        if let Some(min_ram) = self.min_ram_gb {
            if available_ram_gb < min_ram {
                return false;
            }
        }

        // Check GPU memory requirements if GPU is required
        if !self.supports_cpu {
            // If GPU is required but not available
            if available_gpu_memory_gb.is_none() {
                return false;
            }

            if let Some(min_gpu_mem) = self.min_gpu_memory_gb {
                if let Some(available_gpu) = available_gpu_memory_gb {
                    if available_gpu < min_gpu_mem {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Get a human-readable description of the hardware requirements
    pub fn requirements_description(&self) -> String {
        let mut desc = Vec::new();

        if let Some(min_ram) = self.min_ram_gb {
            desc.push(format!("RAM: {:.1} GB minimum", min_ram));
            if let Some(rec_ram) = self.recommended_ram_gb {
                desc.push(format!(" ({:.1} GB recommended)", rec_ram));
            }
        }

        if let Some(min_gpu) = self.min_gpu_memory_gb {
            desc.push(format!("GPU Memory: {:.1} GB minimum", min_gpu));
            if let Some(rec_gpu) = self.recommended_gpu_memory_gb {
                desc.push(format!(" ({:.1} GB recommended)", rec_gpu));
            }
        }

        let mut platforms = Vec::new();
        if self.supports_cpu {
            platforms.push("CPU");
        }
        if self.supports_gpu {
            platforms.push("GPU");
        }
        if self.supports_tpu {
            platforms.push("TPU");
        }

        if !platforms.is_empty() {
            desc.push(format!("Platforms: {}", platforms.join(", ")));
        }

        if desc.is_empty() {
            "No specific requirements".to_string()
        } else {
            desc.join(", ")
        }
    }
}

/// Model status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Active,
    Deprecated,
    Experimental,
    UnderReview,
}

/// Search query for models
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: Option<String>,
    pub tags: Vec<String>,
    pub author: Option<String>,
    pub min_likes: Option<u64>,
    pub sort_by: SortBy,
    pub limit: usize,
    pub category: Option<ModelCategory>,
    pub architecture: Option<String>,
    pub min_accuracy: Option<f32>,
    pub max_model_size_mb: Option<f32>,
    pub max_inference_time_ms: Option<f32>,
    pub framework_compatibility: Vec<String>,
    pub hardware_filter: Option<HardwareFilter>,
    pub status_filter: Vec<ModelStatus>,
    pub license_filter: Vec<String>,
    pub has_demo: Option<bool>,
    pub has_paper: Option<bool>,
    pub version_constraint: Option<String>,
}

/// Hardware filtering options
#[derive(Debug, Clone)]
pub struct HardwareFilter {
    pub max_ram_gb: Option<f32>,
    pub max_gpu_memory_gb: Option<f32>,
    pub requires_gpu: Option<bool>,
    pub supports_cpu_only: Option<bool>,
}

/// Sort options
#[derive(Debug, Clone, Copy)]
pub enum SortBy {
    Downloads,
    Likes,
    Recent,
    Name,
    Accuracy,
    ModelSize,
    InferenceSpeed,
    Trending,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            text: None,
            tags: Vec::new(),
            author: None,
            min_likes: None,
            sort_by: SortBy::Downloads,
            limit: 50,
            category: None,
            architecture: None,
            min_accuracy: None,
            max_model_size_mb: None,
            max_inference_time_ms: None,
            framework_compatibility: Vec::new(),
            hardware_filter: None,
            status_filter: vec![ModelStatus::Active],
            license_filter: Vec::new(),
            has_demo: None,
            has_paper: None,
            version_constraint: None,
        }
    }
}

impl ModelRegistry {
    /// Create or load a model registry
    pub fn new<P: AsRef<Path>>(registry_dir: P) -> Result<Self> {
        let registry_dir = registry_dir.as_ref();
        fs::create_dir_all(registry_dir)?;

        let registry_file = registry_dir.join("registry.json");
        let entries = if registry_file.exists() {
            load_registry(&registry_file)?
        } else {
            HashMap::new()
        };

        Ok(Self {
            registry_file,
            entries,
        })
    }

    /// Register a new model
    pub fn register_model(&mut self, entry: RegistryEntry) -> Result<()> {
        if self.entries.contains_key(&entry.id) {
            return Err(TorshError::InvalidArgument(format!(
                "Model {} already exists",
                entry.id
            )));
        }

        self.entries.insert(entry.id.clone(), entry);
        self.save()
    }

    /// Update model entry
    pub fn update_model(&mut self, entry: RegistryEntry) -> Result<()> {
        if !self.entries.contains_key(&entry.id) {
            return Err(TorshError::InvalidArgument(format!(
                "Model {} not found",
                entry.id
            )));
        }

        self.entries.insert(entry.id.clone(), entry);
        self.save()
    }

    /// Get model by ID
    pub fn get_model(&self, id: &str) -> Option<&RegistryEntry> {
        self.entries.get(id)
    }

    /// Search models with advanced filtering
    pub fn search(&self, query: &SearchQuery) -> Vec<&RegistryEntry> {
        let mut results: Vec<&RegistryEntry> = self
            .entries
            .values()
            .filter(|entry| {
                // Status filter
                if !query.status_filter.is_empty() && !query.status_filter.contains(&entry.status) {
                    return false;
                }

                // Text search
                if let Some(text) = &query.text {
                    let text_lower = text.to_lowercase();
                    if !entry.name.to_lowercase().contains(&text_lower)
                        && !entry.description.to_lowercase().contains(&text_lower)
                        && !entry.architecture.to_lowercase().contains(&text_lower)
                        && !entry
                            .tags
                            .iter()
                            .any(|t| t.to_lowercase().contains(&text_lower))
                    {
                        return false;
                    }
                }

                // Category filter
                if let Some(ref category) = query.category {
                    if entry.category != *category {
                        return false;
                    }
                }

                // Architecture filter
                if let Some(ref architecture) = query.architecture {
                    if !entry
                        .architecture
                        .to_lowercase()
                        .contains(&architecture.to_lowercase())
                    {
                        return false;
                    }
                }

                // Tag filter
                if !query.tags.is_empty() && !query.tags.iter().all(|tag| entry.tags.contains(tag))
                {
                    return false;
                }

                // Author filter
                if let Some(author) = &query.author {
                    if entry.author != *author {
                        return false;
                    }
                }

                // Likes filter
                if let Some(min_likes) = query.min_likes {
                    if entry.likes < min_likes {
                        return false;
                    }
                }

                // Accuracy filter
                if let Some(min_accuracy) = query.min_accuracy {
                    let max_accuracy = entry.accuracy_metrics.values().cloned().fold(0.0, f32::max);
                    if max_accuracy < min_accuracy {
                        return false;
                    }
                }

                // Model size filter
                if let Some(max_size) = query.max_model_size_mb {
                    if let Some(size) = entry.model_size_mb {
                        if size > max_size {
                            return false;
                        }
                    }
                }

                // Inference time filter
                if let Some(max_time) = query.max_inference_time_ms {
                    if let Some(time) = entry.inference_time_ms {
                        if time > max_time {
                            return false;
                        }
                    }
                }

                // Framework compatibility filter
                if !query.framework_compatibility.is_empty()
                    && !query
                        .framework_compatibility
                        .iter()
                        .any(|framework| entry.framework_compatibility.contains(framework))
                {
                    return false;
                }

                // Hardware filter
                if let Some(ref hw_filter) = query.hardware_filter {
                    if let Some(max_ram) = hw_filter.max_ram_gb {
                        if let Some(min_ram) = entry.hardware_requirements.min_ram_gb {
                            if min_ram > max_ram {
                                return false;
                            }
                        }
                    }

                    if let Some(max_gpu) = hw_filter.max_gpu_memory_gb {
                        if let Some(min_gpu) = entry.hardware_requirements.min_gpu_memory_gb {
                            if min_gpu > max_gpu {
                                return false;
                            }
                        }
                    }

                    if let Some(requires_gpu) = hw_filter.requires_gpu {
                        if requires_gpu && !entry.hardware_requirements.supports_gpu {
                            return false;
                        }
                    }

                    if let Some(cpu_only) = hw_filter.supports_cpu_only {
                        if cpu_only && !entry.hardware_requirements.supports_cpu {
                            return false;
                        }
                    }
                }

                // License filter
                if !query.license_filter.is_empty()
                    && !query.license_filter.contains(&entry.license)
                {
                    return false;
                }

                // Demo filter
                if let Some(has_demo) = query.has_demo {
                    if has_demo != entry.demo_url.is_some() {
                        return false;
                    }
                }

                // Paper filter
                if let Some(has_paper) = query.has_paper {
                    if has_paper != entry.paper_url.is_some() {
                        return false;
                    }
                }

                // Version constraint filter
                if let Some(ref version_constraint) = query.version_constraint {
                    if !self.matches_version_constraint(&entry.version, version_constraint) {
                        return false;
                    }
                }

                true
            })
            .collect();

        // Sort results
        match query.sort_by {
            SortBy::Downloads => results.sort_by_key(|e| std::cmp::Reverse(e.downloads)),
            SortBy::Likes => results.sort_by_key(|e| std::cmp::Reverse(e.likes)),
            SortBy::Recent => results.sort_by_key(|e| std::cmp::Reverse(e.updated_at)),
            SortBy::Name => results.sort_by_key(|e| &e.name),
            SortBy::Accuracy => {
                results.sort_by(|a, b| {
                    let a_acc = a.accuracy_metrics.values().cloned().fold(0.0, f32::max);
                    let b_acc = b.accuracy_metrics.values().cloned().fold(0.0, f32::max);
                    b_acc
                        .partial_cmp(&a_acc)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortBy::ModelSize => {
                results.sort_by(|a, b| match (a.model_size_mb, b.model_size_mb) {
                    (Some(a_size), Some(b_size)) => a_size
                        .partial_cmp(&b_size)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
            }
            SortBy::InferenceSpeed => {
                results.sort_by(|a, b| match (a.inference_time_ms, b.inference_time_ms) {
                    (Some(a_time), Some(b_time)) => a_time
                        .partial_cmp(&b_time)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
            }
            SortBy::Trending => {
                let now = chrono::Utc::now();
                results.sort_by(|a, b| {
                    let a_score = self.calculate_trending_score(a, now);
                    let b_score = self.calculate_trending_score(b, now);
                    b_score
                        .partial_cmp(&a_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Limit results
        results.truncate(query.limit);

        results
    }

    /// Check if a version matches a constraint
    fn matches_version_constraint(&self, version: &Version, constraint: &str) -> bool {
        // Simple version constraint matching (could be more sophisticated)
        if let Some(stripped) = constraint.strip_prefix(">=") {
            if let Ok(min_version) = Version::from_str(stripped) {
                return *version >= min_version;
            }
        } else if let Some(stripped) = constraint.strip_prefix("<=") {
            if let Ok(max_version) = Version::from_str(stripped) {
                return *version <= max_version;
            }
        } else if let Some(stripped) = constraint.strip_prefix('>') {
            if let Ok(min_version) = Version::from_str(stripped) {
                return *version > min_version;
            }
        } else if let Some(stripped) = constraint.strip_prefix('<') {
            if let Ok(max_version) = Version::from_str(stripped) {
                return *version < max_version;
            }
        } else if let Some(stripped) = constraint.strip_prefix('=') {
            if let Ok(exact_version) = Version::from_str(stripped) {
                return *version == exact_version;
            }
        } else if let Ok(exact_version) = Version::from_str(constraint) {
            return *version == exact_version;
        }
        true // If constraint is invalid, don't filter
    }

    /// Calculate trending score for a model
    fn calculate_trending_score(
        &self,
        entry: &RegistryEntry,
        now: chrono::DateTime<chrono::Utc>,
    ) -> f64 {
        let days_since_update = (now - entry.updated_at).num_days().max(1) as f64;
        let download_velocity = entry.downloads as f64 / days_since_update;
        let like_velocity = entry.likes as f64 / days_since_update;

        // Weighted score: downloads weight more than likes, recent updates boost score
        let base_score = download_velocity * 1.0 + like_velocity * 2.0;
        let recency_boost = if days_since_update <= 7.0 {
            2.0
        } else if days_since_update <= 30.0 {
            1.5
        } else {
            1.0
        };

        base_score * recency_boost
    }

    /// List all models
    pub fn list_all(&self) -> Vec<&RegistryEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by_key(|e| &e.name);
        entries
    }

    /// List models by author
    pub fn list_by_author(&self, author: &str) -> Vec<&RegistryEntry> {
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.author == author)
            .collect();
        entries.sort_by_key(|e| &e.name);
        entries
    }

    /// List models by tag
    pub fn list_by_tag(&self, tag: &str) -> Vec<&RegistryEntry> {
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.tags.contains(&tag.to_string()))
            .collect();
        entries.sort_by_key(|e| &e.name);
        entries
    }

    /// Get popular models
    pub fn get_popular(&self, limit: usize) -> Vec<&RegistryEntry> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by_key(|e| std::cmp::Reverse((e.downloads, e.likes)));
        entries.truncate(limit);
        entries
    }

    /// Get trending models (recently updated with high activity)
    pub fn get_trending(&self, limit: usize) -> Vec<&RegistryEntry> {
        let now = chrono::Utc::now();
        let week_ago = now - chrono::Duration::weeks(1);

        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.updated_at > week_ago)
            .collect();

        // Sort by recent activity score (combination of downloads and likes)
        entries.sort_by_key(|e| {
            let days_old = (now - e.updated_at).num_days().max(1) as u64;
            let activity_score = (e.downloads + e.likes * 10) / days_old;
            std::cmp::Reverse(activity_score)
        });

        entries.truncate(limit);
        entries
    }

    /// Increment download count
    pub fn increment_downloads(&mut self, id: &str) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.downloads += 1;
            self.save()?;
        }
        Ok(())
    }

    /// Increment like count
    pub fn increment_likes(&mut self, id: &str) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.likes += 1;
            self.save()?;
        }
        Ok(())
    }

    /// List models by category
    pub fn list_by_category(&self, category: &ModelCategory) -> Vec<&RegistryEntry> {
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.category == *category && e.status == ModelStatus::Active)
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.downloads));
        entries
    }

    /// Get models compatible with specific hardware constraints
    pub fn list_by_hardware(&self, hw_filter: &HardwareFilter) -> Vec<&RegistryEntry> {
        let query = SearchQuery {
            hardware_filter: Some(hw_filter.clone()),
            limit: 100,
            ..Default::default()
        };
        self.search(&query)
    }

    /// Get similar models based on tags and category
    pub fn get_similar_models(&self, model_id: &str, limit: usize) -> Vec<&RegistryEntry> {
        if let Some(target_model) = self.get_model(model_id) {
            let mut similarities: Vec<_> = self
                .entries
                .values()
                .filter(|e| e.id != model_id && e.status == ModelStatus::Active)
                .map(|entry| {
                    let mut score = 0.0;

                    // Category match (high weight)
                    if entry.category == target_model.category {
                        score += 10.0;
                    }

                    // Architecture similarity
                    if entry.architecture == target_model.architecture {
                        score += 5.0;
                    }

                    // Tag overlap
                    let common_tags = entry
                        .tags
                        .iter()
                        .filter(|tag| target_model.tags.contains(tag))
                        .count() as f64;
                    score += common_tags * 2.0;

                    // Framework compatibility overlap
                    let common_frameworks = entry
                        .framework_compatibility
                        .iter()
                        .filter(|fw| target_model.framework_compatibility.contains(fw))
                        .count() as f64;
                    score += common_frameworks;

                    (entry, score)
                })
                .filter(|(_, score)| *score > 0.0)
                .collect();

            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            similarities.truncate(limit);
            similarities.into_iter().map(|(entry, _)| entry).collect()
        } else {
            Vec::new()
        }
    }

    /// Get recommended models for a user based on their usage history
    pub fn get_recommendations(
        &self,
        user_downloads: &[String],
        limit: usize,
    ) -> Vec<&RegistryEntry> {
        let mut category_counts: HashMap<ModelCategory, usize> = HashMap::new();
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        let mut architecture_counts: HashMap<String, usize> = HashMap::new();

        // Analyze user's download history
        for model_id in user_downloads {
            if let Some(model) = self.get_model(model_id) {
                *category_counts.entry(model.category.clone()).or_insert(0) += 1;
                *architecture_counts
                    .entry(model.architecture.clone())
                    .or_insert(0) += 1;
                for tag in &model.tags {
                    *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                }
            }
        }

        // Score models based on user preferences
        let mut recommendations: Vec<_> = self
            .entries
            .values()
            .filter(|e| !user_downloads.contains(&e.id) && e.status == ModelStatus::Active)
            .map(|entry| {
                let mut score = 0.0;

                // Category preference
                if let Some(&count) = category_counts.get(&entry.category) {
                    score += count as f64 * 3.0;
                }

                // Architecture preference
                if let Some(&count) = architecture_counts.get(&entry.architecture) {
                    score += count as f64 * 2.0;
                }

                // Tag preferences
                for tag in &entry.tags {
                    if let Some(&count) = tag_counts.get(tag) {
                        score += count as f64;
                    }
                }

                // Popularity boost
                score += (entry.downloads as f64).log10() * 0.5;
                score += (entry.likes as f64).log10() * 0.3;

                (entry, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        recommendations.truncate(limit);
        recommendations
            .into_iter()
            .map(|(entry, _)| entry)
            .collect()
    }

    /// Get model statistics
    pub fn get_statistics(&self) -> ModelStatistics {
        let entries: Vec<_> = self.entries.values().collect();
        let active_models = entries
            .iter()
            .filter(|e| e.status == ModelStatus::Active)
            .count();

        let mut category_distribution = HashMap::new();
        let mut total_downloads = 0;
        let mut total_likes = 0;

        for entry in &entries {
            *category_distribution
                .entry(entry.category.clone())
                .or_insert(0) += 1;
            total_downloads += entry.downloads;
            total_likes += entry.likes;
        }

        ModelStatistics {
            total_models: entries.len(),
            active_models,
            total_downloads,
            total_likes,
            category_distribution,
        }
    }

    /// Get featured models (curated high-quality models)
    pub fn get_featured(&self, limit: usize) -> Vec<&RegistryEntry> {
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|e| e.status == ModelStatus::Active)
            .map(|entry| {
                let mut score = 0.0;

                // High download count
                score += (entry.downloads as f64).log10() * 2.0;

                // High like ratio
                if entry.downloads > 0 {
                    let like_ratio = entry.likes as f64 / entry.downloads as f64;
                    score += like_ratio * 10.0;
                }

                // Has paper (indicates research quality)
                if entry.paper_url.is_some() {
                    score += 5.0;
                }

                // Has demo (indicates usability)
                if entry.demo_url.is_some() {
                    score += 3.0;
                }

                // Recent update (indicates maintenance)
                let days_old = (chrono::Utc::now() - entry.updated_at).num_days();
                if days_old <= 30 {
                    score += 2.0;
                } else if days_old <= 90 {
                    score += 1.0;
                }

                (entry, score)
            })
            .collect();

        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(limit);
        entries.into_iter().map(|(entry, _)| entry).collect()
    }

    /// Save registry to disk
    fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        let mut file = File::create(&self.registry_file)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatistics {
    pub total_models: usize,
    pub active_models: usize,
    pub total_downloads: u64,
    pub total_likes: u64,
    pub category_distribution: HashMap<ModelCategory, usize>,
}

/// Load registry from file
fn load_registry(path: &Path) -> Result<HashMap<String, RegistryEntry>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    serde_json::from_str(&content).map_err(|e| TorshError::SerializationError(e.to_string()))
}

/// Create a registry entry for a model
pub fn create_registry_entry(
    name: String,
    author: String,
    repository: String,
    description: String,
) -> RegistryEntry {
    let id = format!("{}/{}", author.to_lowercase(), name.to_lowercase());

    RegistryEntry {
        id,
        name,
        author,
        repository,
        version: Version::new(1, 0, 0),
        tags: Vec::new(),
        downloads: 0,
        likes: 0,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        description,
        metrics: HashMap::new(),
        category: ModelCategory::Other("unspecified".to_string()),
        architecture: "unknown".to_string(),
        framework_compatibility: vec!["torsh".to_string()],
        hardware_requirements: HardwareSpec {
            min_ram_gb: None,
            recommended_ram_gb: None,
            min_gpu_memory_gb: None,
            recommended_gpu_memory_gb: None,
            supports_cpu: true,
            supports_gpu: false,
            supports_tpu: false,
        },
        model_size_mb: None,
        inference_time_ms: None,
        accuracy_metrics: HashMap::new(),
        license: "Apache-2.0".to_string(),
        paper_url: None,
        demo_url: None,
        status: ModelStatus::Active,
    }
}

/// Create a complete registry entry for a model
pub fn create_complete_registry_entry(
    name: String,
    author: String,
    repository: String,
    description: String,
    category: ModelCategory,
    architecture: String,
    version: Version,
) -> RegistryEntry {
    let mut entry = create_registry_entry(name, author, repository, description);
    entry.category = category;
    entry.architecture = architecture;
    entry.version = version;
    entry
}

/// Registry API for easy querying
pub struct RegistryAPI {
    registry: ModelRegistry,
}

impl RegistryAPI {
    pub fn new<P: AsRef<Path>>(registry_dir: P) -> Result<Self> {
        let registry = ModelRegistry::new(registry_dir)?;
        Ok(Self { registry })
    }

    /// Quick search by text
    pub fn search_text(&self, text: &str, limit: usize) -> Vec<&RegistryEntry> {
        let query = SearchQuery {
            text: Some(text.to_string()),
            limit,
            ..Default::default()
        };
        self.registry.search(&query)
    }

    /// Quick search by category
    pub fn search_category(&self, category: ModelCategory, limit: usize) -> Vec<&RegistryEntry> {
        let query = SearchQuery {
            category: Some(category),
            limit,
            ..Default::default()
        };
        self.registry.search(&query)
    }

    /// Quick search by tags
    pub fn search_tags(&self, tags: Vec<String>, limit: usize) -> Vec<&RegistryEntry> {
        let query = SearchQuery {
            tags,
            limit,
            ..Default::default()
        };
        self.registry.search(&query)
    }

    /// Get popular models in a category
    pub fn get_popular_in_category(
        &self,
        category: ModelCategory,
        limit: usize,
    ) -> Vec<&RegistryEntry> {
        let query = SearchQuery {
            category: Some(category),
            sort_by: SortBy::Downloads,
            limit,
            ..Default::default()
        };
        self.registry.search(&query)
    }

    /// Get models suitable for specific hardware
    pub fn get_models_for_hardware(
        &self,
        max_ram_gb: f32,
        has_gpu: bool,
        limit: usize,
    ) -> Vec<&RegistryEntry> {
        let hw_filter = HardwareFilter {
            max_ram_gb: Some(max_ram_gb),
            max_gpu_memory_gb: if has_gpu { Some(16.0) } else { None },
            requires_gpu: None,
            supports_cpu_only: Some(!has_gpu),
        };

        let query = SearchQuery {
            hardware_filter: Some(hw_filter),
            limit,
            ..Default::default()
        };
        self.registry.search(&query)
    }

    /// Get trending models
    pub fn get_trending(&self, limit: usize) -> Vec<&RegistryEntry> {
        self.registry.get_trending(limit)
    }

    /// Get featured models
    pub fn get_featured(&self, limit: usize) -> Vec<&RegistryEntry> {
        self.registry.get_featured(limit)
    }

    /// Get model by ID
    pub fn get_model(&self, id: &str) -> Option<&RegistryEntry> {
        self.registry.get_model(id)
    }

    /// Get similar models
    pub fn get_similar(&self, model_id: &str, limit: usize) -> Vec<&RegistryEntry> {
        self.registry.get_similar_models(model_id, limit)
    }

    /// Get recommendations
    pub fn get_recommendations(
        &self,
        user_downloads: &[String],
        limit: usize,
    ) -> Vec<&RegistryEntry> {
        self.registry.get_recommendations(user_downloads, limit)
    }

    /// Get statistics
    pub fn get_stats(&self) -> ModelStatistics {
        self.registry.get_statistics()
    }
}

/// Get all available tags from registry
pub fn get_all_tags(registry: &ModelRegistry) -> Vec<(String, usize)> {
    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for entry in registry.entries.values() {
        for tag in &entry.tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let mut tags: Vec<_> = tag_counts.into_iter().collect();
    tags.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    tags
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_registry_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();

        // Create and register a model
        let mut entry = create_registry_entry(
            "TestModel".to_string(),
            "TestAuthor".to_string(),
            "github.com/test/model".to_string(),
            "A test model".to_string(),
        );
        entry.category = ModelCategory::Vision;
        entry.architecture = "CNN".to_string();
        entry.tags = vec!["test".to_string(), "vision".to_string()];

        registry.register_model(entry.clone()).unwrap();

        // Get model
        let retrieved = registry.get_model(&entry.id).unwrap();
        assert_eq!(retrieved.name, "TestModel");
        assert_eq!(retrieved.category, ModelCategory::Vision);

        // Search
        let query = SearchQuery {
            text: Some("test".to_string()),
            ..Default::default()
        };

        let results = registry.search(&query);
        assert_eq!(results.len(), 1);

        // Search by category
        let category_results = registry.list_by_category(&ModelCategory::Vision);
        assert_eq!(category_results.len(), 1);
    }

    #[test]
    fn test_registry_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ModelRegistry::new(temp_dir.path()).unwrap();

        // Add multiple models
        for i in 0..5 {
            let mut entry = create_registry_entry(
                format!("Model{}", i),
                "Author".to_string(),
                format!("repo{}", i),
                format!("Description {}", i),
            );
            entry.tags = vec!["vision".to_string()];
            entry.downloads = (i * 10) as u64;
            entry.likes = i as u64;
            entry.category = ModelCategory::Vision;
            entry.architecture = "CNN".to_string();

            registry.register_model(entry).unwrap();
        }

        // Search by tag
        let results = registry.list_by_tag("vision");
        assert_eq!(results.len(), 5);

        // Search by category
        let category_results = registry.list_by_category(&ModelCategory::Vision);
        assert_eq!(category_results.len(), 5);

        // Get popular
        let popular = registry.get_popular(3);
        assert_eq!(popular.len(), 3);
        assert!(popular[0].downloads >= popular[1].downloads);

        // Test advanced search
        let query = SearchQuery {
            category: Some(ModelCategory::Vision),
            architecture: Some("CNN".to_string()),
            limit: 2,
            ..Default::default()
        };
        let advanced_results = registry.search(&query);
        assert_eq!(advanced_results.len(), 2);
    }
}
