//! Model Zoo Format and Management System
//!
//! This module provides a standardized format for saving, loading, and managing
//! pre-trained FX graph models in a model zoo repository. It includes:
//! - Model metadata and versioning
//! - Serialization/deserialization of FX graphs with weights
//! - Model registry and discovery
//! - Integrity verification and validation
//! - Remote model repository support

use crate::fx::{FxGraph, Node};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

/// Model zoo entry with complete metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelZooEntry {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Serialized FX graph
    pub graph: SerializedGraph,
    /// Model weights and parameters
    pub weights: ModelWeights,
    /// Training configuration used
    pub training_config: Option<TrainingConfig>,
    /// Performance metrics
    pub metrics: ModelMetrics,
    /// Model provenance and lineage
    pub provenance: ModelProvenance,
}

/// Comprehensive model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub id: String,
    /// Human-readable model name
    pub name: String,
    /// Model version (semver format)
    pub version: String,
    /// Model author/organization
    pub author: String,
    /// Model description
    pub description: String,
    /// License information
    pub license: String,
    /// Model tags for discovery
    pub tags: Vec<String>,
    /// Target task (classification, detection, etc.)
    pub task: String,
    /// Input shape specification
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape specification
    pub output_shapes: Vec<Vec<usize>>,
    /// Required framework version
    pub framework_version: String,
    /// Creation timestamp
    pub created_at: String,
    /// Last modified timestamp
    pub updated_at: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Serialized graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedGraph {
    /// Graph nodes in execution order
    pub nodes: Vec<SerializedNode>,
    /// Graph edges (connections)
    pub edges: Vec<(usize, usize)>,
    /// Input node indices
    pub inputs: Vec<usize>,
    /// Output node indices
    pub outputs: Vec<usize>,
    /// Graph-level metadata
    pub graph_metadata: HashMap<String, String>,
}

/// Serialized node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    /// Node unique ID
    pub id: usize,
    /// Node type/operation
    pub node_type: String,
    /// Node name
    pub name: String,
    /// Operation parameters
    pub params: HashMap<String, String>,
    /// Node metadata
    pub metadata: HashMap<String, String>,
}

/// Model weights and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Weight format (safetensors, numpy, pytorch, etc.)
    pub format: WeightFormat,
    /// Weight data (base64 encoded or file reference)
    pub data: WeightData,
    /// Weight shapes
    pub shapes: HashMap<String, Vec<usize>>,
    /// Weight dtypes
    pub dtypes: HashMap<String, String>,
    /// Total parameter count
    pub total_params: u64,
    /// Trainable parameter count
    pub trainable_params: u64,
}

/// Weight storage format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// NumPy format (.npy)
    Numpy,
    /// PyTorch format (.pt)
    PyTorch,
    /// ONNX format
    Onnx,
    /// Custom binary format
    Custom { format_name: String },
}

/// Weight data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightData {
    /// Embedded base64-encoded data
    Embedded { data: String },
    /// External file reference
    External { path: String },
    /// Remote URL
    Remote { url: String, checksum: String },
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Optimizer used
    pub optimizer: String,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Loss function
    pub loss_function: String,
    /// Dataset information
    pub dataset: DatasetInfo,
    /// Augmentation pipeline
    pub augmentations: Vec<String>,
    /// Training hyperparameters
    pub hyperparameters: HashMap<String, String>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Dataset split used (train, val, test)
    pub split: String,
    /// Number of samples
    pub num_samples: usize,
    /// Number of classes (if applicable)
    pub num_classes: Option<usize>,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy metrics
    pub accuracy: Option<f64>,
    /// Top-k accuracy
    pub top_k_accuracy: HashMap<usize, f64>,
    /// Loss value
    pub loss: Option<f64>,
    /// F1 score
    pub f1_score: Option<f64>,
    /// Precision
    pub precision: Option<f64>,
    /// Recall
    pub recall: Option<f64>,
    /// Inference latency (ms)
    pub latency_ms: Option<f64>,
    /// Throughput (samples/sec)
    pub throughput: Option<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Model provenance and lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenance {
    /// Base model (if fine-tuned)
    pub base_model: Option<String>,
    /// Training dataset
    pub training_dataset: String,
    /// Training framework
    pub training_framework: String,
    /// Hardware used for training
    pub training_hardware: String,
    /// Training duration
    pub training_duration: Option<String>,
    /// Reproducibility information
    pub random_seed: Option<u64>,
    /// Code repository
    pub code_repository: Option<String>,
    /// Paper/citation
    pub paper_citation: Option<String>,
}

/// Model zoo registry for managing collections of models
pub struct ModelZooRegistry {
    /// Registry base path
    base_path: PathBuf,
    /// Cached model index
    model_index: HashMap<String, ModelMetadata>,
    /// Remote repositories
    remote_repos: Vec<RemoteRepository>,
}

/// Remote model repository configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteRepository {
    /// Repository name
    pub name: String,
    /// Repository URL
    pub url: String,
    /// Authentication token (if required)
    pub auth_token: Option<String>,
    /// Mirror URLs
    pub mirrors: Vec<String>,
}

impl ModelZooEntry {
    /// Create a new model zoo entry
    pub fn new(metadata: ModelMetadata, graph: FxGraph, weights: ModelWeights) -> Result<Self> {
        let serialized_graph = Self::serialize_graph(&graph)?;

        Ok(Self {
            metadata,
            graph: serialized_graph,
            weights,
            training_config: None,
            metrics: ModelMetrics::default(),
            provenance: ModelProvenance::default(),
        })
    }

    /// Serialize an FX graph
    fn serialize_graph(graph: &FxGraph) -> Result<SerializedGraph> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Serialize nodes
        for (idx, node) in graph.nodes() {
            let serialized_node = SerializedNode {
                id: idx.index(),
                node_type: Self::node_type_string(node),
                name: format!("node_{}", idx.index()),
                params: HashMap::new(),
                metadata: HashMap::new(),
            };
            nodes.push(serialized_node);
        }

        // Serialize edges
        for edge in graph.edges() {
            edges.push((edge.source().index(), edge.target().index()));
        }

        Ok(SerializedGraph {
            nodes,
            edges,
            inputs: graph.inputs().iter().map(|idx| idx.index()).collect(),
            outputs: graph.outputs().iter().map(|idx| idx.index()).collect(),
            graph_metadata: HashMap::new(),
        })
    }

    /// Get node type as string
    fn node_type_string(node: &Node) -> String {
        match node {
            Node::Input(name) => format!("input:{}", name),
            Node::Output => "output".to_string(),
            Node::Call(name, _) => format!("call:{}", name),
            Node::GetAttr { target, attr } => format!("getattr:{}:{}", target, attr),
            Node::Conditional { .. } => "conditional".to_string(),
            Node::Loop { .. } => "loop".to_string(),
            Node::Merge { .. } => "merge".to_string(),
        }
    }

    /// Save model to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        fs::write(path, json).map_err(|e| TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load model from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path).map_err(|e| TorshError::IoError(e.to_string()))?;

        let entry: Self = serde_json::from_str(&json)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(entry)
    }

    /// Verify model integrity
    pub fn verify_integrity(&self) -> Result<bool> {
        // Verify checksum
        let computed_checksum = self.compute_checksum()?;
        if computed_checksum != self.metadata.checksum {
            return Ok(false);
        }

        // Verify graph structure
        if !self.verify_graph_structure()? {
            return Ok(false);
        }

        Ok(true)
    }

    /// Compute model checksum
    fn compute_checksum(&self) -> Result<String> {
        // Simple checksum based on serialized data
        let json = serde_json::to_string(self)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(format!("{:x}", md5::compute(json.as_bytes())))
    }

    /// Verify graph structure
    fn verify_graph_structure(&self) -> Result<bool> {
        // Check that all edges reference valid nodes
        for (src, dst) in &self.graph.edges {
            if *src >= self.graph.nodes.len() || *dst >= self.graph.nodes.len() {
                return Ok(false);
            }
        }

        // Check that inputs and outputs reference valid nodes
        for idx in &self.graph.inputs {
            if *idx >= self.graph.nodes.len() {
                return Ok(false);
            }
        }

        for idx in &self.graph.outputs {
            if *idx >= self.graph.nodes.len() {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

impl ModelZooRegistry {
    /// Create a new model zoo registry
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        fs::create_dir_all(&base_path).map_err(|e| TorshError::IoError(e.to_string()))?;

        let mut registry = Self {
            base_path,
            model_index: HashMap::new(),
            remote_repos: Vec::new(),
        };

        registry.refresh_index()?;
        Ok(registry)
    }

    /// Refresh model index
    pub fn refresh_index(&mut self) -> Result<()> {
        self.model_index.clear();

        // Scan base directory for models
        for entry in
            fs::read_dir(&self.base_path).map_err(|e| TorshError::IoError(e.to_string()))?
        {
            let entry = entry.map_err(|e| TorshError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(model) = ModelZooEntry::load_from_file(&path) {
                    self.model_index
                        .insert(model.metadata.id.clone(), model.metadata);
                }
            }
        }

        Ok(())
    }

    /// Register a model in the zoo
    pub fn register_model(&mut self, entry: ModelZooEntry) -> Result<()> {
        let filename = format!("{}.json", entry.metadata.id);
        let path = self.base_path.join(filename);

        entry.save_to_file(&path)?;
        self.model_index
            .insert(entry.metadata.id.clone(), entry.metadata);

        Ok(())
    }

    /// Load a model by ID
    pub fn load_model(&self, model_id: &str) -> Result<ModelZooEntry> {
        let filename = format!("{}.json", model_id);
        let path = self.base_path.join(filename);

        ModelZooEntry::load_from_file(path)
    }

    /// Search models by tags
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<&ModelMetadata> {
        self.model_index
            .values()
            .filter(|metadata| tags.iter().any(|tag| metadata.tags.contains(tag)))
            .collect()
    }

    /// Search models by task
    pub fn search_by_task(&self, task: &str) -> Vec<&ModelMetadata> {
        self.model_index
            .values()
            .filter(|metadata| metadata.task == task)
            .collect()
    }

    /// List all models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.model_index.values().collect()
    }

    /// Add remote repository
    pub fn add_remote_repository(&mut self, repo: RemoteRepository) {
        self.remote_repos.push(repo);
    }

    /// Download model from remote repository
    pub fn download_from_remote(&mut self, model_id: &str, repo_name: &str) -> Result<()> {
        let repository = self
            .remote_repos
            .iter()
            .find(|r| r.name == repo_name)
            .ok_or_else(|| {
                TorshError::RuntimeError(format!("Repository {} not found", repo_name))
            })?;

        // Implement model download logic
        // Build the download URL
        let download_url = if repository.url.ends_with('/') {
            format!("{}{}", repository.url, model_id)
        } else {
            format!("{}/{}", repository.url, model_id)
        };

        // Create a temporary download path
        let temp_dir = std::env::temp_dir();
        let model_file = temp_dir.join(format!("{}_{}.torsh", repo_name, model_id));

        // Note: Actual HTTP download would require adding an HTTP client dependency
        // For now, we provide a framework that users can extend
        // Example implementation would use reqwest or similar:
        //
        // let response = reqwest::blocking::get(&download_url)
        //     .map_err(|e| TorshError::RuntimeError(format!("Download failed: {}", e)))?;
        // let bytes = response.bytes()
        //     .map_err(|e| TorshError::RuntimeError(format!("Failed to read response: {}", e)))?;
        // std::fs::write(&model_file, &bytes)
        //     .map_err(|e| TorshError::RuntimeError(format!("Failed to write model: {}", e)))?;

        // For now, return an informative error with the download URL
        Err(TorshError::RuntimeError(format!(
            "Model download requires manual implementation. \
             Download URL: {} \
             Save to: {} \
             Then load using ModelZooRegistry::load_model()",
            download_url,
            model_file.display()
        )))
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            accuracy: None,
            top_k_accuracy: HashMap::new(),
            loss: None,
            f1_score: None,
            precision: None,
            recall: None,
            latency_ms: None,
            throughput: None,
            custom_metrics: HashMap::new(),
        }
    }
}

impl Default for ModelProvenance {
    fn default() -> Self {
        Self {
            base_model: None,
            training_dataset: "unknown".to_string(),
            training_framework: "torsh-fx".to_string(),
            training_hardware: "unknown".to_string(),
            training_duration: None,
            random_seed: None,
            code_repository: None,
            paper_citation: None,
        }
    }
}

/// Builder for creating model metadata
pub struct ModelMetadataBuilder {
    id: String,
    name: String,
    version: String,
    author: String,
    description: String,
    license: String,
    tags: Vec<String>,
    task: String,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl ModelMetadataBuilder {
    /// Create a new metadata builder
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            version: "1.0.0".to_string(),
            author: "unknown".to_string(),
            description: String::new(),
            license: "MIT".to_string(),
            tags: Vec::new(),
            task: "general".to_string(),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        }
    }

    /// Set version
    pub fn version(mut self, version: String) -> Self {
        self.version = version;
        self
    }

    /// Set author
    pub fn author(mut self, author: String) -> Self {
        self.author = author;
        self
    }

    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set license
    pub fn license(mut self, license: String) -> Self {
        self.license = license;
        self
    }

    /// Add tag
    pub fn add_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Set task
    pub fn task(mut self, task: String) -> Self {
        self.task = task;
        self
    }

    /// Add input shape
    pub fn add_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shapes.push(shape);
        self
    }

    /// Add output shape
    pub fn add_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shapes.push(shape);
        self
    }

    /// Build the metadata
    pub fn build(self) -> ModelMetadata {
        let now = chrono::Utc::now().to_rfc3339();

        ModelMetadata {
            id: self.id,
            name: self.name,
            version: self.version,
            author: self.author,
            description: self.description,
            license: self.license,
            tags: self.tags,
            task: self.task,
            input_shapes: self.input_shapes,
            output_shapes: self.output_shapes,
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: now.clone(),
            updated_at: now,
            size_bytes: 0,
            checksum: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder() {
        let metadata =
            ModelMetadataBuilder::new("test-model-001".to_string(), "Test Model".to_string())
                .version("1.0.0".to_string())
                .author("Test Author".to_string())
                .description("A test model".to_string())
                .add_tag("test".to_string())
                .add_tag("demo".to_string())
                .task("classification".to_string())
                .add_input_shape(vec![1, 3, 224, 224])
                .add_output_shape(vec![1, 1000])
                .build();

        assert_eq!(metadata.id, "test-model-001");
        assert_eq!(metadata.name, "Test Model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.tags.len(), 2);
    }

    #[test]
    fn test_model_zoo_entry_creation() {
        let metadata =
            ModelMetadataBuilder::new("test-001".to_string(), "Test".to_string()).build();

        let graph = FxGraph::new();

        let weights = ModelWeights {
            format: WeightFormat::SafeTensors,
            data: WeightData::Embedded {
                data: String::new(),
            },
            shapes: HashMap::new(),
            dtypes: HashMap::new(),
            total_params: 0,
            trainable_params: 0,
        };

        let result = ModelZooEntry::new(metadata, graph, weights);
        assert!(result.is_ok());
    }
}
