use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::device::DeviceType;
use torsh_core::Result;

// Temporarily commented out until torsh_nn is available
// use super::bert::BertModel;
// use super::gpt::GPTModel;
// use super::lstm::LSTMTextModel;
use super::{TextModel, TextModelConfig};

/// Type alias for shared model storage
type ModelStorage = Arc<Mutex<HashMap<String, Box<dyn TextModel + Send + Sync>>>>;

/// Type alias for shared config storage
type ConfigStorage = Arc<Mutex<HashMap<String, TextModelConfig>>>;

// Placeholder model implementation
#[derive(Debug, Clone)]
struct PlaceholderModel {
    config: TextModelConfig,
    model_type: String,
}

impl PlaceholderModel {
    fn new(config: TextModelConfig, model_type: String) -> Self {
        Self { config, model_type }
    }
}

impl TextModel for PlaceholderModel {
    fn name(&self) -> &str {
        &self.model_type
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn max_seq_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

/// Model registry for text models
pub struct ModelRegistry {
    models: ModelStorage,
    configs: ConfigStorage,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            configs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a pre-configured model
    pub fn register_config(&self, name: &str, config: TextModelConfig) {
        let mut configs = self.configs.lock().expect("lock should not be poisoned");
        configs.insert(name.to_string(), config);
    }

    /// Get a registered configuration
    pub fn get_config(&self, name: &str) -> Option<TextModelConfig> {
        let configs = self.configs.lock().expect("lock should not be poisoned");
        configs.get(name).cloned()
    }

    /// List all registered configurations
    pub fn list_configs(&self) -> Vec<String> {
        let configs = self.configs.lock().expect("lock should not be poisoned");
        configs.keys().cloned().collect()
    }

    /// Create a model from registered configuration
    pub fn create_model(
        &self,
        name: &str,
        model_type: &str,
        _device: DeviceType,
    ) -> Result<Box<dyn TextModel + Send + Sync>> {
        let config = self.get_config(name).ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Configuration '{}' not found",
                name
            ))
        })?;

        match model_type.to_lowercase().as_str() {
            "bert" | "gpt" | "lstm" => {
                // Placeholder implementation until torsh-nn is available
                let model = PlaceholderModel::new(config.clone(), model_type.to_string());
                Ok(Box::new(model))
            }
            _ => Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Unsupported model type: {}",
                model_type
            ))),
        }
    }

    /// Register common model configurations
    pub fn register_common_configs(&self) {
        // BERT configurations
        self.register_config("bert-base-uncased", TextModelConfig::bert_base());
        self.register_config("bert-large-uncased", TextModelConfig::bert_large());

        // GPT configurations
        self.register_config("gpt2", TextModelConfig::gpt2_small());
        self.register_config("gpt2-medium", TextModelConfig::gpt2_medium());
        self.register_config("gpt2-large", TextModelConfig::gpt2_large());

        // T5 configurations
        self.register_config("t5-small", TextModelConfig::t5_small());

        // Custom LSTM configurations
        let mut lstm_config = TextModelConfig::default();
        lstm_config.num_layers = 2;
        lstm_config.hidden_dim = 256;
        self.register_config("lstm-small", lstm_config);

        let mut lstm_large_config = TextModelConfig::default();
        lstm_large_config.num_layers = 4;
        lstm_large_config.hidden_dim = 512;
        self.register_config("lstm-large", lstm_large_config);
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let registry = Self::new();
        registry.register_common_configs();
        registry
    }
}

lazy_static::lazy_static! {
    static ref GLOBAL_REGISTRY: ModelRegistry = ModelRegistry::default();
}

/// Get the global model registry
pub fn get_global_registry() -> &'static ModelRegistry {
    &GLOBAL_REGISTRY
}

/// Convenience function to create a model from the global registry
pub fn create_model(
    config_name: &str,
    model_type: &str,
    device: DeviceType,
) -> Result<Box<dyn TextModel + Send + Sync>> {
    get_global_registry().create_model(config_name, model_type, device)
}

/// Convenience function to get a configuration from the global registry
pub fn get_config(name: &str) -> Option<TextModelConfig> {
    get_global_registry().get_config(name)
}

/// Convenience function to list all configurations in the global registry
pub fn list_configs() -> Vec<String> {
    get_global_registry().list_configs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry() {
        let registry = ModelRegistry::new();

        // Test config registration
        let config = TextModelConfig::bert_base();
        registry.register_config("test-bert", config.clone());

        let retrieved = registry.get_config("test-bert").unwrap();
        assert_eq!(retrieved.vocab_size, config.vocab_size);
        assert_eq!(retrieved.hidden_dim, config.hidden_dim);

        // Test listing configs
        let configs = registry.list_configs();
        assert!(configs.contains(&"test-bert".to_string()));
    }

    #[test]
    fn test_global_registry() {
        let configs = list_configs();
        assert!(configs.contains(&"bert-base-uncased".to_string()));
        assert!(configs.contains(&"gpt2".to_string()));
        assert!(configs.contains(&"lstm-small".to_string()));
    }
}
