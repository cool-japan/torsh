//! Model information and metadata

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use torsh_core::error::{Result, TorshError};

/// Semantic version struct for proper version management
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
    pub build: Option<String>,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build: None,
        }
    }

    pub fn with_pre_release(mut self, pre_release: String) -> Self {
        self.pre_release = Some(pre_release);
        self
    }

    pub fn with_build(mut self, build: String) -> Self {
        self.build = Some(build);
        self
    }

    pub fn is_compatible_with(&self, other: &Version) -> bool {
        self.major == other.major
            && (self.minor > other.minor
                || (self.minor == other.minor && self.patch >= other.patch))
    }

    pub fn is_breaking_change(&self, other: &Version) -> bool {
        self.major != other.major
    }
}

impl FromStr for Version {
    type Err = TorshError;

    fn from_str(s: &str) -> Result<Self> {
        let version_str = s.trim_start_matches('v');
        let mut parts = version_str.splitn(2, '+');
        let version_part = parts
            .next()
            .expect("splitn always returns at least one element");
        let build = parts.next().map(|s| s.to_string());

        let mut parts = version_part.splitn(2, '-');
        let version_core = parts
            .next()
            .expect("splitn always returns at least one element");
        let pre_release = parts.next().map(|s| s.to_string());

        let version_parts: Vec<&str> = version_core.split('.').collect();
        if version_parts.len() != 3 {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid version format: {}",
                s
            )));
        }

        let major = version_parts[0].parse::<u32>().map_err(|_| {
            TorshError::InvalidArgument(format!("Invalid major version: {}", version_parts[0]))
        })?;
        let minor = version_parts[1].parse::<u32>().map_err(|_| {
            TorshError::InvalidArgument(format!("Invalid minor version: {}", version_parts[1]))
        })?;
        let patch = version_parts[2].parse::<u32>().map_err(|_| {
            TorshError::InvalidArgument(format!("Invalid patch version: {}", version_parts[2]))
        })?;

        Ok(Version {
            major,
            minor,
            patch,
            pre_release,
            build,
        })
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(ref build) = self.build {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.patch.cmp(&other.patch) {
            Ordering::Equal => {}
            ord => return ord,
        }

        match (&self.pre_release, &other.pre_release) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }
}

/// Version history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    pub version: Version,
    pub timestamp: String,
    pub changelog: String,
    pub author: String,
    pub breaking_changes: bool,
    pub migration_notes: Option<String>,
}

/// Version history for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    pub entries: Vec<VersionEntry>,
    pub current_version: Version,
}

impl VersionHistory {
    pub fn new(initial_version: Version, author: String) -> Self {
        let initial_entry = VersionEntry {
            version: initial_version.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            changelog: "Initial release".to_string(),
            author,
            breaking_changes: false,
            migration_notes: None,
        };

        Self {
            entries: vec![initial_entry],
            current_version: initial_version,
        }
    }

    pub fn add_version(
        &mut self,
        version: Version,
        changelog: String,
        author: String,
        migration_notes: Option<String>,
    ) -> Result<()> {
        if version <= self.current_version {
            return Err(TorshError::InvalidArgument(
                "New version must be greater than current version".to_string(),
            ));
        }

        let breaking_changes = version.is_breaking_change(&self.current_version);

        let entry = VersionEntry {
            version: version.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            changelog,
            author,
            breaking_changes,
            migration_notes,
        };

        self.entries.push(entry);
        self.current_version = version;
        Ok(())
    }

    pub fn get_version(&self, version: &Version) -> Option<&VersionEntry> {
        self.entries.iter().find(|entry| &entry.version == version)
    }

    pub fn get_latest_compatible(&self, min_version: &Version) -> Option<&VersionEntry> {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.version.is_compatible_with(min_version))
    }

    pub fn list_versions(&self) -> Vec<&Version> {
        self.entries.iter().map(|entry| &entry.version).collect()
    }

    pub fn get_breaking_changes_since(&self, version: &Version) -> Vec<&VersionEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.version > *version && entry.breaking_changes)
            .collect()
    }
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub author: String,
    pub version: Version,
    pub license: String,
    pub tags: Vec<String>,
    pub datasets: Vec<String>,
    pub metrics: HashMap<String, MetricValue>,
    pub requirements: Requirements,
    pub files: Vec<FileInfo>,
    pub model_card: Option<ModelCard>,
    pub version_history: Option<VersionHistory>,
}

/// Model metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricValue {
    Float(f32),
    String(String),
    Integer(i64),
}

/// Model requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirements {
    pub torsh_version: String,
    pub dependencies: Vec<String>,
    pub hardware: HardwareRequirements,
}

/// Hardware requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    pub min_gpu_memory_gb: Option<f32>,
    pub recommended_gpu_memory_gb: Option<f32>,
    pub min_ram_gb: Option<f32>,
    pub recommended_ram_gb: Option<f32>,
}

/// File information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub description: Option<String>,
}

/// Model card for documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub model_details: ModelDetails,
    pub intended_use: IntendedUse,
    pub training_data: TrainingData,
    pub evaluation: Evaluation,
    pub ethical_considerations: Option<String>,
    pub caveats_and_recommendations: Option<String>,
}

/// Model details section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    pub developed_by: String,
    pub model_date: String,
    pub model_type: String,
    pub architecture: String,
    pub paper_url: Option<String>,
    pub citation: Option<String>,
}

/// Intended use section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntendedUse {
    pub primary_uses: Vec<String>,
    pub primary_users: Vec<String>,
    pub out_of_scope_uses: Vec<String>,
}

/// Training data section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub datasets: Vec<DatasetInfo>,
    pub preprocessing: Vec<String>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub url: Option<String>,
    pub split: Option<String>,
    pub notes: Option<String>,
}

/// Evaluation section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub testing_data: Vec<DatasetInfo>,
    pub metrics: HashMap<String, MetricInfo>,
}

/// Metric information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricInfo {
    pub value: MetricValue,
    pub description: String,
    pub confidence_interval: Option<(f32, f32)>,
}

impl ModelInfo {
    /// Create a new model info
    pub fn new(name: String, author: String, version: Version) -> Self {
        let version_history = VersionHistory::new(version.clone(), author.clone());
        Self {
            name,
            author: author.clone(),
            version,
            description: String::new(),
            license: "Apache-2.0".to_string(),
            tags: Vec::new(),
            datasets: Vec::new(),
            metrics: HashMap::new(),
            requirements: Requirements {
                torsh_version: option_env!("CARGO_PKG_VERSION")
                    .unwrap_or("0.1.0")
                    .to_string(),
                dependencies: Vec::new(),
                hardware: HardwareRequirements {
                    min_gpu_memory_gb: None,
                    recommended_gpu_memory_gb: None,
                    min_ram_gb: None,
                    recommended_ram_gb: None,
                },
            },
            files: Vec::new(),
            model_card: None,
            version_history: Some(version_history),
        }
    }

    /// Create from legacy string version
    pub fn new_with_string_version(
        name: String,
        author: String,
        version_str: String,
    ) -> Result<Self> {
        let version = Version::from_str(&version_str)?;
        Ok(Self::new(name, author, version))
    }

    /// Update to a new version
    pub fn update_version(
        &mut self,
        new_version: Version,
        changelog: String,
        migration_notes: Option<String>,
    ) -> Result<()> {
        if let Some(ref mut history) = self.version_history {
            history.add_version(
                new_version.clone(),
                changelog,
                self.author.clone(),
                migration_notes,
            )?;
        } else {
            self.version_history = Some(VersionHistory::new(
                new_version.clone(),
                self.author.clone(),
            ));
        }
        self.version = new_version;
        Ok(())
    }

    /// Get version compatibility info
    pub fn is_compatible_with(&self, required_version: &Version) -> bool {
        self.version.is_compatible_with(required_version)
    }

    /// Get breaking changes since a version
    pub fn get_breaking_changes_since(&self, version: &Version) -> Vec<&VersionEntry> {
        self.version_history
            .as_ref()
            .map(|h| h.get_breaking_changes_since(version))
            .unwrap_or_default()
    }

    /// Get changelog for current version
    pub fn get_current_changelog(&self) -> Option<String> {
        self.version_history
            .as_ref()
            .and_then(|h| h.get_version(&self.version))
            .map(|entry| entry.changelog.clone())
    }

    /// Load model info from JSON file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| TorshError::SerializationError(e.to_string()))
    }

    /// Save model info to JSON file
    pub fn to_file(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate model info
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Model name cannot be empty".to_string(),
            ));
        }

        if self.author.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Model author cannot be empty".to_string(),
            ));
        }

        // Validate files
        for file in &self.files {
            if file.path.is_empty() {
                return Err(TorshError::InvalidArgument(
                    "File path cannot be empty".to_string(),
                ));
            }

            if file.sha256.len() != 64 {
                return Err(TorshError::InvalidArgument(format!(
                    "Invalid SHA256 hash for file: {}",
                    file.path
                )));
            }
        }

        Ok(())
    }
}

/// Load model info from a repository
pub fn load_model_info(repo_path: &Path) -> Result<ModelInfo> {
    let info_path = repo_path.join("model_info.json");

    if info_path.exists() {
        ModelInfo::from_file(&info_path)
    } else {
        // Try to load from README or other sources
        let readme_path = repo_path.join("README.md");
        if readme_path.exists() {
            parse_model_info_from_readme(&readme_path)
        } else {
            Err(TorshError::IoError(
                "No model_info.json or README.md found".to_string(),
            ))
        }
    }
}

/// Parse model info from README (basic implementation)
fn parse_model_info_from_readme(readme_path: &Path) -> Result<ModelInfo> {
    let content = std::fs::read_to_string(readme_path)?;

    // Extract basic info from README
    let name = extract_field(&content, "# ").unwrap_or_else(|| "Unknown Model".to_string());
    let author = extract_field(&content, "Author:").unwrap_or_else(|| "Unknown".to_string());
    let version_str = extract_field(&content, "Version:").unwrap_or_else(|| "0.0.0".to_string());
    let version = Version::from_str(&version_str)?;

    let mut info = ModelInfo::new(name, author, version);

    // Extract description
    if let Some(desc) = extract_field(&content, "## Description") {
        info.description = desc;
    }

    // Extract tags
    if let Some(tags_line) = extract_field(&content, "Tags:") {
        info.tags = tags_line.split(',').map(|s| s.trim().to_string()).collect();
    }

    Ok(info)
}

/// Extract field from markdown
fn extract_field(content: &str, prefix: &str) -> Option<String> {
    content
        .lines()
        .find(|line| line.starts_with(prefix))
        .map(|line| line[prefix.len()..].trim().to_string())
}

/// Create a model card template
pub fn create_model_card_template() -> ModelCard {
    ModelCard {
        model_details: ModelDetails {
            developed_by: "Your name or organization".to_string(),
            model_date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            model_type: "Type (e.g., Convolutional Neural Network)".to_string(),
            architecture: "Architecture details".to_string(),
            paper_url: None,
            citation: None,
        },
        intended_use: IntendedUse {
            primary_uses: vec!["Primary use case".to_string()],
            primary_users: vec!["Researchers".to_string(), "Practitioners".to_string()],
            out_of_scope_uses: vec!["Out of scope use case".to_string()],
        },
        training_data: TrainingData {
            datasets: vec![DatasetInfo {
                name: "Dataset name".to_string(),
                url: None,
                split: Some("train".to_string()),
                notes: None,
            }],
            preprocessing: vec!["Preprocessing step".to_string()],
        },
        evaluation: Evaluation {
            testing_data: vec![DatasetInfo {
                name: "Test dataset".to_string(),
                url: None,
                split: Some("test".to_string()),
                notes: None,
            }],
            metrics: HashMap::new(),
        },
        ethical_considerations: Some("Ethical considerations".to_string()),
        caveats_and_recommendations: Some("Caveats and recommendations".to_string()),
    }
}

/// Enhanced model card builder for easier creation
pub struct ModelCardBuilder {
    card: ModelCard,
}

impl ModelCardBuilder {
    pub fn new() -> Self {
        Self {
            card: create_model_card_template(),
        }
    }

    pub fn developed_by(mut self, developer: String) -> Self {
        self.card.model_details.developed_by = developer;
        self
    }

    pub fn model_type(mut self, model_type: String) -> Self {
        self.card.model_details.model_type = model_type;
        self
    }

    pub fn architecture(mut self, architecture: String) -> Self {
        self.card.model_details.architecture = architecture;
        self
    }

    pub fn paper_url(mut self, url: String) -> Self {
        self.card.model_details.paper_url = Some(url);
        self
    }

    pub fn citation(mut self, citation: String) -> Self {
        self.card.model_details.citation = Some(citation);
        self
    }

    pub fn add_primary_use(mut self, use_case: String) -> Self {
        self.card.intended_use.primary_uses.push(use_case);
        self
    }

    pub fn add_primary_user(mut self, user: String) -> Self {
        self.card.intended_use.primary_users.push(user);
        self
    }

    pub fn add_out_of_scope_use(mut self, use_case: String) -> Self {
        self.card.intended_use.out_of_scope_uses.push(use_case);
        self
    }

    pub fn add_training_dataset(
        mut self,
        name: String,
        url: Option<String>,
        split: Option<String>,
        notes: Option<String>,
    ) -> Self {
        self.card.training_data.datasets.push(DatasetInfo {
            name,
            url,
            split,
            notes,
        });
        self
    }

    pub fn add_preprocessing_step(mut self, step: String) -> Self {
        self.card.training_data.preprocessing.push(step);
        self
    }

    pub fn add_test_dataset(
        mut self,
        name: String,
        url: Option<String>,
        split: Option<String>,
        notes: Option<String>,
    ) -> Self {
        self.card.evaluation.testing_data.push(DatasetInfo {
            name,
            url,
            split,
            notes,
        });
        self
    }

    pub fn add_metric(mut self, name: String, value: MetricValue, description: String) -> Self {
        self.card.evaluation.metrics.insert(
            name,
            MetricInfo {
                value,
                description,
                confidence_interval: None,
            },
        );
        self
    }

    pub fn add_metric_with_confidence(
        mut self,
        name: String,
        value: MetricValue,
        description: String,
        confidence_interval: (f32, f32),
    ) -> Self {
        self.card.evaluation.metrics.insert(
            name,
            MetricInfo {
                value,
                description,
                confidence_interval: Some(confidence_interval),
            },
        );
        self
    }

    pub fn ethical_considerations(mut self, considerations: String) -> Self {
        self.card.ethical_considerations = Some(considerations);
        self
    }

    pub fn caveats_and_recommendations(mut self, caveats: String) -> Self {
        self.card.caveats_and_recommendations = Some(caveats);
        self
    }

    pub fn build(self) -> ModelCard {
        self.card
    }
}

impl Default for ModelCardBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Model card renderer for different formats
pub struct ModelCardRenderer;

impl ModelCardRenderer {
    /// Render model card as Markdown
    pub fn to_markdown(card: &ModelCard) -> String {
        let mut markdown = String::new();

        // Header
        markdown.push_str("# Model Card\n\n");

        // Model Details
        markdown.push_str("## Model Details\n\n");
        markdown.push_str(&format!(
            "**Developed by:** {}\n\n",
            card.model_details.developed_by
        ));
        markdown.push_str(&format!(
            "**Model date:** {}\n\n",
            card.model_details.model_date
        ));
        markdown.push_str(&format!(
            "**Model type:** {}\n\n",
            card.model_details.model_type
        ));
        markdown.push_str(&format!(
            "**Architecture:** {}\n\n",
            card.model_details.architecture
        ));

        if let Some(ref paper_url) = card.model_details.paper_url {
            markdown.push_str(&format!("**Paper:** [Link]({})\n\n", paper_url));
        }

        if let Some(ref citation) = card.model_details.citation {
            markdown.push_str(&format!("**Citation:**\n```\n{}\n```\n\n", citation));
        }

        // Intended Use
        markdown.push_str("## Intended Use\n\n");
        markdown.push_str("**Primary use cases:**\n");
        for use_case in &card.intended_use.primary_uses {
            markdown.push_str(&format!("- {}\n", use_case));
        }
        markdown.push('\n');

        markdown.push_str("**Primary users:**\n");
        for user in &card.intended_use.primary_users {
            markdown.push_str(&format!("- {}\n", user));
        }
        markdown.push('\n');

        markdown.push_str("**Out-of-scope uses:**\n");
        for use_case in &card.intended_use.out_of_scope_uses {
            markdown.push_str(&format!("- {}\n", use_case));
        }
        markdown.push('\n');

        // Training Data
        markdown.push_str("## Training Data\n\n");
        markdown.push_str("**Datasets:**\n");
        for dataset in &card.training_data.datasets {
            markdown.push_str(&format!("- **{}**", dataset.name));
            if let Some(ref url) = dataset.url {
                markdown.push_str(&format!(" ([Link]({}))", url));
            }
            if let Some(ref split) = dataset.split {
                markdown.push_str(&format!(" - Split: {}", split));
            }
            if let Some(ref notes) = dataset.notes {
                markdown.push_str(&format!(" - Notes: {}", notes));
            }
            markdown.push('\n');
        }
        markdown.push('\n');

        markdown.push_str("**Preprocessing:**\n");
        for step in &card.training_data.preprocessing {
            markdown.push_str(&format!("- {}\n", step));
        }
        markdown.push('\n');

        // Evaluation
        markdown.push_str("## Evaluation\n\n");
        markdown.push_str("**Testing data:**\n");
        for dataset in &card.evaluation.testing_data {
            markdown.push_str(&format!("- **{}**", dataset.name));
            if let Some(ref url) = dataset.url {
                markdown.push_str(&format!(" ([Link]({}))", url));
            }
            if let Some(ref split) = dataset.split {
                markdown.push_str(&format!(" - Split: {}", split));
            }
            if let Some(ref notes) = dataset.notes {
                markdown.push_str(&format!(" - Notes: {}", notes));
            }
            markdown.push('\n');
        }
        markdown.push('\n');

        markdown.push_str("**Metrics:**\n");
        for (metric_name, metric_info) in &card.evaluation.metrics {
            markdown.push_str(&format!(
                "- **{}:** {:?} - {}",
                metric_name, metric_info.value, metric_info.description
            ));
            if let Some((low, high)) = metric_info.confidence_interval {
                markdown.push_str(&format!(" (95% CI: {:.3}-{:.3})", low, high));
            }
            markdown.push('\n');
        }
        markdown.push('\n');

        // Ethical Considerations
        if let Some(ref ethical) = card.ethical_considerations {
            markdown.push_str("## Ethical Considerations\n\n");
            markdown.push_str(ethical);
            markdown.push_str("\n\n");
        }

        // Caveats and Recommendations
        if let Some(ref caveats) = card.caveats_and_recommendations {
            markdown.push_str("## Caveats and Recommendations\n\n");
            markdown.push_str(caveats);
            markdown.push_str("\n\n");
        }

        markdown
    }

    /// Render model card as HTML
    pub fn to_html(card: &ModelCard) -> String {
        let markdown = Self::to_markdown(card);
        // In a real implementation, you would use a markdown to HTML converter
        // For now, we'll return a basic HTML wrapper
        format!("<html><body><pre>{}</pre></body></html>", markdown)
    }

    /// Render model card as JSON
    pub fn to_json(card: &ModelCard) -> Result<String> {
        serde_json::to_string_pretty(card)
            .map_err(|e| TorshError::SerializationError(e.to_string()))
    }
}

/// Model card manager for handling model cards
pub struct ModelCardManager {
    cards_dir: PathBuf,
}

impl ModelCardManager {
    pub fn new<P: AsRef<Path>>(cards_dir: P) -> Result<Self> {
        let cards_dir = cards_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cards_dir)?;
        Ok(Self { cards_dir })
    }

    /// Save a model card
    pub fn save_card(&self, model_id: &str, card: &ModelCard) -> Result<()> {
        let card_path = self.cards_dir.join(format!("{}.json", model_id));
        let content = serde_json::to_string_pretty(card)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        std::fs::write(card_path, content)?;
        Ok(())
    }

    /// Load a model card
    pub fn load_card(&self, model_id: &str) -> Result<ModelCard> {
        let card_path = self.cards_dir.join(format!("{}.json", model_id));
        if !card_path.exists() {
            return Err(TorshError::IoError(format!(
                "Model card not found for {}",
                model_id
            )));
        }

        let content = std::fs::read_to_string(card_path)?;
        serde_json::from_str(&content).map_err(|e| TorshError::SerializationError(e.to_string()))
    }

    /// Check if a model card exists
    pub fn has_card(&self, model_id: &str) -> bool {
        self.cards_dir.join(format!("{}.json", model_id)).exists()
    }

    /// List all available model cards
    pub fn list_cards(&self) -> Result<Vec<String>> {
        let mut cards = Vec::new();
        for entry in std::fs::read_dir(&self.cards_dir)? {
            let entry = entry?;
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".json") {
                    let model_id = filename
                        .strip_suffix(".json")
                        .expect("filename should end with .json as checked");
                    cards.push(model_id.to_string());
                }
            }
        }
        cards.sort();
        Ok(cards)
    }

    /// Export model card to markdown file
    pub fn export_to_markdown(&self, model_id: &str, output_path: &Path) -> Result<()> {
        let card = self.load_card(model_id)?;
        let markdown = ModelCardRenderer::to_markdown(&card);
        std::fs::write(output_path, markdown)?;
        Ok(())
    }

    /// Export model card to HTML file
    pub fn export_to_html(&self, model_id: &str, output_path: &Path) -> Result<()> {
        let card = self.load_card(model_id)?;
        let html = ModelCardRenderer::to_html(&card);
        std::fs::write(output_path, html)?;
        Ok(())
    }

    /// Delete a model card
    pub fn delete_card(&self, model_id: &str) -> Result<()> {
        let card_path = self.cards_dir.join(format!("{}.json", model_id));
        if card_path.exists() {
            std::fs::remove_file(card_path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_info_creation() {
        let version = Version::new(1, 0, 0);
        let info = ModelInfo::new(
            "test_model".to_string(),
            "test_author".to_string(),
            version.clone(),
        );

        assert_eq!(info.name, "test_model");
        assert_eq!(info.author, "test_author");
        assert_eq!(info.version, version);
        assert!(info.version_history.is_some());
    }

    #[test]
    fn test_model_info_serialization() {
        let version = Version::new(1, 0, 0);
        let mut info = ModelInfo::new("test_model".to_string(), "test_author".to_string(), version);

        info.description = "Test model description".to_string();
        info.tags = vec!["vision".to_string(), "classification".to_string()];
        info.metrics
            .insert("accuracy".to_string(), MetricValue::Float(0.95));

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("model_info.json");

        info.to_file(&path).unwrap();
        let loaded = ModelInfo::from_file(&path).unwrap();

        assert_eq!(loaded.name, info.name);
        assert_eq!(loaded.description, info.description);
        assert_eq!(loaded.tags, info.tags);
        assert_eq!(loaded.version, info.version);
    }

    #[test]
    fn test_model_info_validation() {
        let version = Version::new(1, 0, 0);
        let mut info = ModelInfo::new("test_model".to_string(), "test_author".to_string(), version);

        assert!(info.validate().is_ok());

        // Empty name should fail
        info.name = String::new();
        assert!(info.validate().is_err());

        // Fix name, add invalid file
        info.name = "test_model".to_string();
        info.files.push(FileInfo {
            path: "model.bin".to_string(),
            size_bytes: 1000,
            sha256: "invalid".to_string(),
            description: None,
        });

        assert!(info.validate().is_err());
    }

    #[test]
    fn test_version_parsing() {
        let version = Version::from_str("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.pre_release, None);
        assert_eq!(version.build, None);

        let version = Version::from_str("2.0.0-alpha.1+build.123").unwrap();
        assert_eq!(version.major, 2);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);
        assert_eq!(version.pre_release, Some("alpha.1".to_string()));
        assert_eq!(version.build, Some("build.123".to_string()));
    }

    #[test]
    fn test_version_comparison() {
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 1, 0);
        let v3 = Version::new(2, 0, 0);

        assert!(v2 > v1);
        assert!(v3 > v2);
        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v2));
        assert!(v3.is_breaking_change(&v1));
        assert!(!v2.is_breaking_change(&v1));
    }

    #[test]
    fn test_version_history() {
        let v1 = Version::new(1, 0, 0);
        let mut history = VersionHistory::new(v1.clone(), "author".to_string());

        assert_eq!(history.current_version, v1);
        assert_eq!(history.entries.len(), 1);

        let v2 = Version::new(1, 1, 0);
        history
            .add_version(
                v2.clone(),
                "Added new features".to_string(),
                "author".to_string(),
                None,
            )
            .unwrap();

        assert_eq!(history.current_version, v2);
        assert_eq!(history.entries.len(), 2);
        assert!(!history.entries[1].breaking_changes);

        let v3 = Version::new(2, 0, 0);
        history
            .add_version(
                v3.clone(),
                "Breaking changes".to_string(),
                "author".to_string(),
                Some("Migration required".to_string()),
            )
            .unwrap();

        assert_eq!(history.current_version, v3);
        assert_eq!(history.entries.len(), 3);
        assert!(history.entries[2].breaking_changes);
    }

    #[test]
    fn test_model_card_builder() {
        let card = ModelCardBuilder::new()
            .developed_by("Test Developer".to_string())
            .model_type("CNN".to_string())
            .architecture("ResNet-50".to_string())
            .paper_url("https://example.com/paper".to_string())
            .add_primary_use("Image classification".to_string())
            .add_training_dataset(
                "ImageNet".to_string(),
                Some("https://imagenet.org".to_string()),
                Some("train".to_string()),
                None,
            )
            .add_metric(
                "accuracy".to_string(),
                MetricValue::Float(0.95),
                "Top-1 accuracy on test set".to_string(),
            )
            .ethical_considerations("No known ethical issues".to_string())
            .build();

        assert_eq!(card.model_details.developed_by, "Test Developer");
        assert_eq!(card.model_details.model_type, "CNN");
        assert_eq!(card.model_details.architecture, "ResNet-50");
        assert_eq!(
            card.model_details.paper_url,
            Some("https://example.com/paper".to_string())
        );
        assert_eq!(card.intended_use.primary_uses.len(), 2); // template + added
        assert!(card
            .intended_use
            .primary_uses
            .contains(&"Image classification".to_string()));
        assert_eq!(card.training_data.datasets.len(), 2); // template + added
        assert_eq!(card.evaluation.metrics.len(), 1);
        assert_eq!(
            card.ethical_considerations,
            Some("No known ethical issues".to_string())
        );
    }

    #[test]
    fn test_model_card_renderer() {
        let card = ModelCardBuilder::new()
            .developed_by("Test Developer".to_string())
            .model_type("CNN".to_string())
            .build();

        let markdown = ModelCardRenderer::to_markdown(&card);
        assert!(markdown.contains("# Model Card"));
        assert!(markdown.contains("**Developed by:** Test Developer"));
        assert!(markdown.contains("**Model type:** CNN"));

        let html = ModelCardRenderer::to_html(&card);
        assert!(html.contains("<html>"));
        assert!(html.contains("Test Developer"));

        let json = ModelCardRenderer::to_json(&card).unwrap();
        assert!(json.contains("Test Developer"));
        assert!(json.contains("CNN"));
    }

    #[test]
    fn test_model_card_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelCardManager::new(temp_dir.path()).unwrap();

        let card = ModelCardBuilder::new()
            .developed_by("Test Developer".to_string())
            .build();

        // Save card
        manager.save_card("test_model", &card).unwrap();
        assert!(manager.has_card("test_model"));

        // Load card
        let loaded_card = manager.load_card("test_model").unwrap();
        assert_eq!(loaded_card.model_details.developed_by, "Test Developer");

        // List cards
        let cards = manager.list_cards().unwrap();
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0], "test_model");

        // Export to markdown
        let md_path = temp_dir.path().join("test_model.md");
        manager.export_to_markdown("test_model", &md_path).unwrap();
        assert!(md_path.exists());

        // Delete card
        manager.delete_card("test_model").unwrap();
        assert!(!manager.has_card("test_model"));
    }
}
