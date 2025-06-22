//! Model registry for managing pre-trained models

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::Digest;
use crate::{ModelError, ModelResult};

/// Information about a pre-trained model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
    /// Model architecture
    pub architecture: String,
    /// Model domain (vision, nlp, etc.)
    pub domain: String,
    /// Input shape/specifications
    pub input_spec: String,
    /// Output shape/specifications  
    pub output_spec: String,
    /// Model file path or URL
    pub source: ModelSource,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Model parameters count
    pub parameters: u64,
    /// Model accuracy metrics
    pub metrics: HashMap<String, f32>,
    /// Model tags for categorization
    pub tags: Vec<String>,
    /// License information
    pub license: String,
    /// Citation information
    pub citation: Option<String>,
    /// Checksum for validation
    pub checksum: String,
}

/// Model source specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    /// Local file path
    Local(PathBuf),
    /// HTTP/HTTPS URL
    Url(String),
    /// Hugging Face Hub model
    HuggingFace { repo: String, filename: String },
    /// Custom registry
    Registry { registry: String, path: String },
}

/// Handle to a loaded model
pub struct ModelHandle {
    /// Model information
    pub info: ModelInfo,
    /// Model file path (local)
    pub local_path: PathBuf,
    /// Whether model is loaded in memory
    pub loaded: bool,
}

impl ModelHandle {
    /// Create new model handle
    pub fn new(info: ModelInfo, local_path: PathBuf) -> Self {
        Self {
            info,
            local_path,
            loaded: false,
        }
    }
    
    /// Check if model file exists locally
    pub fn exists(&self) -> bool {
        self.local_path.exists()
    }
    
    /// Get model file size
    pub fn file_size(&self) -> ModelResult<u64> {
        let metadata = std::fs::metadata(&self.local_path)?;
        Ok(metadata.len())
    }
    
    /// Validate model checksum
    pub fn validate_checksum(&self) -> ModelResult<bool> {
        if !self.exists() {
            return Ok(false);
        }
        
        let data = std::fs::read(&self.local_path)?;
        let hash = sha2::Sha256::digest(&data);
        let hex_hash = hex::encode(hash);
        
        Ok(hex_hash == self.info.checksum)
    }
}

/// Model registry for managing pre-trained models
pub struct ModelRegistry {
    /// Registered models
    models: Arc<Mutex<HashMap<String, ModelInfo>>>,
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    /// Model handles cache
    handles: Arc<Mutex<HashMap<String, ModelHandle>>>,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> ModelResult<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        
        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }
        
        Ok(Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            cache_dir,
            handles: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    /// Create default registry (uses ~/.torsh/models)
    pub fn default() -> ModelResult<Self> {
        let home_dir = dirs::home_dir()
            .ok_or_else(|| ModelError::LoadingError {
                reason: "Could not find home directory".to_string(),
            })?;
        
        let cache_dir = home_dir.join(".torsh").join("models");
        Self::new(cache_dir)
    }
    
    /// Register a new model
    pub fn register_model(&self, info: ModelInfo) -> ModelResult<()> {
        let mut models = self.models.lock().unwrap();
        let key = format!("{}:{}", info.name, info.version);
        models.insert(key, info);
        Ok(())
    }
    
    /// Get model information by name and version
    pub fn get_model_info(&self, name: &str, version: Option<&str>) -> ModelResult<ModelInfo> {
        let models = self.models.lock().unwrap();
        
        if let Some(version) = version {
            let key = format!("{}:{}", name, version);
            models.get(&key)
                .cloned()
                .ok_or_else(|| ModelError::ModelNotFound { name: key })
        } else {
            // Find latest version
            let matching_models: Vec<_> = models
                .values()
                .filter(|info| info.name == name)
                .collect();
            
            if matching_models.is_empty() {
                return Err(ModelError::ModelNotFound { name: name.to_string() });
            }
            
            // Sort by version and return latest
            let mut sorted = matching_models;
            sorted.sort_by(|a, b| a.version.cmp(&b.version));
            
            Ok((*sorted.last().unwrap()).clone())
        }
    }
    
    /// List all registered models
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.lock().unwrap();
        models.values().cloned().collect()
    }
    
    /// Search models by domain
    pub fn search_by_domain(&self, domain: &str) -> Vec<ModelInfo> {
        let models = self.models.lock().unwrap();
        models
            .values()
            .filter(|info| info.domain == domain)
            .cloned()
            .collect()
    }
    
    /// Search models by tags
    pub fn search_by_tags(&self, tags: &[&str]) -> Vec<ModelInfo> {
        let models = self.models.lock().unwrap();
        models
            .values()
            .filter(|info| tags.iter().any(|tag| info.tags.contains(&tag.to_string())))
            .cloned()
            .collect()
    }
    
    /// Get model handle
    pub fn get_model_handle(&self, name: &str, version: Option<&str>) -> ModelResult<ModelHandle> {
        let info = self.get_model_info(name, version)?;
        let key = format!("{}:{}", info.name, info.version);
        
        // Check if handle already exists
        {
            let handles = self.handles.lock().unwrap();
            if let Some(handle) = handles.get(&key) {
                return Ok(ModelHandle {
                    info: handle.info.clone(),
                    local_path: handle.local_path.clone(),
                    loaded: handle.loaded,
                });
            }
        }
        
        // Create new handle
        let local_path = self.get_local_path(&info);
        let handle = ModelHandle::new(info, local_path);
        
        // Cache the handle
        {
            let mut handles = self.handles.lock().unwrap();
            handles.insert(key, ModelHandle {
                info: handle.info.clone(),
                local_path: handle.local_path.clone(),
                loaded: handle.loaded,
            });
        }
        
        Ok(handle)
    }
    
    /// Get local file path for a model
    fn get_local_path(&self, info: &ModelInfo) -> PathBuf {
        let filename = format!("{}-{}.safetensors", info.name, info.version);
        self.cache_dir.join(filename)
    }
    
    /// Load models from registry file
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        let content = std::fs::read_to_string(path)?;
        let model_infos: Vec<ModelInfo> = serde_json::from_str(&content)?;
        
        for info in model_infos {
            self.register_model(info)?;
        }
        
        Ok(())
    }
    
    /// Save models to registry file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        let models = self.list_models();
        let content = serde_json::to_string_pretty(&models)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Register built-in models
    pub fn register_builtin_models(&self) -> ModelResult<()> {
        // Vision models
        #[cfg(feature = "vision")]
        {
            self.register_vision_models()?;
        }
        
        // NLP models
        #[cfg(feature = "nlp")]
        {
            self.register_nlp_models()?;
        }
        
        Ok(())
    }
    
    #[cfg(feature = "vision")]
    fn register_vision_models(&self) -> ModelResult<()> {
        // ResNet-18
        let resnet18 = ModelInfo {
            name: "resnet18".to_string(),
            version: "1.0.0".to_string(),
            description: "ResNet-18 model pre-trained on ImageNet".to_string(),
            architecture: "ResNet".to_string(),
            domain: "vision".to_string(),
            input_spec: "RGB image [3, 224, 224]".to_string(),
            output_spec: "1000 class probabilities".to_string(),
            source: ModelSource::Url("https://github.com/pytorch/vision/releases/download/v0.1.9/resnet18-5c106cde.pth".to_string()),
            size_bytes: 46827520,
            parameters: 11689512,
            metrics: {
                let mut m = HashMap::new();
                m.insert("top1_accuracy".to_string(), 69.758);
                m.insert("top5_accuracy".to_string(), 89.078);
                m
            },
            tags: vec!["classification".to_string(), "imagenet".to_string(), "cnn".to_string()],
            license: "BSD".to_string(),
            citation: Some("He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.".to_string()),
            checksum: "5c106cde0abbf5e61f9b0e5d5c51b2a9e17896b7".to_string(),
        };
        self.register_model(resnet18)?;
        
        // ResNet-50
        let resnet50 = ModelInfo {
            name: "resnet50".to_string(),
            version: "1.0.0".to_string(),
            description: "ResNet-50 model pre-trained on ImageNet".to_string(),
            architecture: "ResNet".to_string(),
            domain: "vision".to_string(),
            input_spec: "RGB image [3, 224, 224]".to_string(),
            output_spec: "1000 class probabilities".to_string(),
            source: ModelSource::Url("https://github.com/pytorch/vision/releases/download/v0.1.9/resnet50-19c8e357.pth".to_string()),
            size_bytes: 102502400,
            parameters: 25557032,
            metrics: {
                let mut m = HashMap::new();
                m.insert("top1_accuracy".to_string(), 76.130);
                m.insert("top5_accuracy".to_string(), 92.862);
                m
            },
            tags: vec!["classification".to_string(), "imagenet".to_string(), "cnn".to_string()],
            license: "BSD".to_string(),
            citation: Some("He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.".to_string()),
            checksum: "19c8e357f2b6c76a2a39b97e94f5e71e8bbde6b7".to_string(),
        };
        self.register_model(resnet50)?;
        
        Ok(())
    }
    
    #[cfg(feature = "nlp")]
    fn register_nlp_models(&self) -> ModelResult<()> {
        // BERT-base
        let bert_base = ModelInfo {
            name: "bert-base-uncased".to_string(),
            version: "1.0.0".to_string(),
            description: "BERT base model (uncased) pre-trained on English corpus".to_string(),
            architecture: "BERT".to_string(),
            domain: "nlp".to_string(),
            input_spec: "Tokenized text [seq_len]".to_string(),
            output_spec: "Hidden states [seq_len, 768]".to_string(),
            source: ModelSource::HuggingFace { 
                repo: "bert-base-uncased".to_string(), 
                filename: "pytorch_model.bin".to_string() 
            },
            size_bytes: 440473133,
            parameters: 110000000,
            metrics: {
                let mut m = HashMap::new();
                m.insert("glue_avg".to_string(), 79.6);
                m
            },
            tags: vec!["transformer".to_string(), "encoder".to_string(), "english".to_string()],
            license: "Apache-2.0".to_string(),
            citation: Some("Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding.".to_string()),
            checksum: "abc123def456789".to_string(),
        };
        self.register_model(bert_base)?;
        
        Ok(())
    }
}

/// Create a global model registry instance
lazy_static::lazy_static! {
    static ref GLOBAL_REGISTRY: ModelRegistry = {
        let registry = ModelRegistry::default().expect("Failed to create model registry");
        registry.register_builtin_models().expect("Failed to register builtin models");
        registry
    };
}

/// Get the global model registry
pub fn get_global_registry() -> &'static ModelRegistry {
    &GLOBAL_REGISTRY
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_model_registry_creation() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).unwrap();
        assert!(temp_dir.path().exists());
    }
    
    #[test]
    fn test_model_registration() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let info = ModelInfo {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            description: "Test model".to_string(),
            architecture: "TestNet".to_string(),
            domain: "test".to_string(),
            input_spec: "test input".to_string(),
            output_spec: "test output".to_string(),
            source: ModelSource::Local(PathBuf::from("test.safetensors")),
            size_bytes: 1024,
            parameters: 100,
            metrics: HashMap::new(),
            tags: vec!["test".to_string()],
            license: "MIT".to_string(),
            citation: None,
            checksum: "test_checksum".to_string(),
        };
        
        registry.register_model(info.clone()).unwrap();
        
        let retrieved = registry.get_model_info("test_model", Some("1.0.0")).unwrap();
        assert_eq!(retrieved.name, "test_model");
        assert_eq!(retrieved.version, "1.0.0");
    }
    
    #[test]
    fn test_model_search() {
        let temp_dir = tempdir().unwrap();
        let registry = ModelRegistry::new(temp_dir.path()).unwrap();
        
        let info1 = ModelInfo {
            name: "model1".to_string(),
            version: "1.0.0".to_string(),
            description: "Model 1".to_string(),
            architecture: "Net1".to_string(),
            domain: "vision".to_string(),
            input_spec: "image".to_string(),
            output_spec: "class".to_string(),
            source: ModelSource::Local(PathBuf::from("model1.safetensors")),
            size_bytes: 1024,
            parameters: 100,
            metrics: HashMap::new(),
            tags: vec!["cnn".to_string(), "classification".to_string()],
            license: "MIT".to_string(),
            citation: None,
            checksum: "checksum1".to_string(),
        };
        
        let info2 = ModelInfo {
            name: "model2".to_string(),
            version: "1.0.0".to_string(),
            description: "Model 2".to_string(),
            architecture: "Net2".to_string(),
            domain: "nlp".to_string(),
            input_spec: "text".to_string(),
            output_spec: "embedding".to_string(),
            source: ModelSource::Local(PathBuf::from("model2.safetensors")),
            size_bytes: 2048,
            parameters: 200,
            metrics: HashMap::new(),
            tags: vec!["transformer".to_string(), "embedding".to_string()],
            license: "Apache-2.0".to_string(),
            citation: None,
            checksum: "checksum2".to_string(),
        };
        
        registry.register_model(info1).unwrap();
        registry.register_model(info2).unwrap();
        
        let vision_models = registry.search_by_domain("vision");
        assert_eq!(vision_models.len(), 1);
        assert_eq!(vision_models[0].name, "model1");
        
        let cnn_models = registry.search_by_tags(&["cnn"]);
        assert_eq!(cnn_models.len(), 1);
        assert_eq!(cnn_models[0].name, "model1");
    }
}