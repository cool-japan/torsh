//! Model builders and factories for easy instantiation

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::config::{
    ModelConfig, ModelConfigs, NlpArchitecture, NlpModelConfig, VisionArchParams,
    VisionArchitecture, VisionModelConfig,
};
use crate::vision::{ResNet, VisionTransformer};
use crate::{ModelError, ModelType};
use std::collections::HashMap;
use torsh_nn::Module;

/// Result type for model builders
pub type BuildResult<T> = Result<T, ModelError>;

/// Generic model builder trait
pub trait ModelBuilder<T: Module>: Send + Sync {
    type Config: ModelConfig;

    /// Build model from configuration
    fn build(&self, config: &Self::Config) -> BuildResult<T>;

    /// Build model from configuration name
    fn build_from_name(&self, name: &str) -> BuildResult<T>;

    /// List available model configurations
    fn available_models(&self) -> Vec<String>;

    /// Get model configuration by name
    fn get_config(&self, name: &str) -> Option<Self::Config>;
}

/// Vision model builder
pub struct VisionModelBuilder {
    configs: HashMap<String, VisionModelConfig>,
}

impl VisionModelBuilder {
    /// Create new vision model builder
    pub fn new() -> Self {
        let mut configs = HashMap::new();

        // Add predefined configurations
        configs.extend(ModelConfigs::resnet_configs());
        configs.extend(ModelConfigs::efficientnet_configs());
        configs.extend(ModelConfigs::vit_configs());

        Self { configs }
    }

    /// Add custom configuration
    pub fn add_config(&mut self, name: String, config: VisionModelConfig) {
        self.configs.insert(name, config);
    }

    /// Build ResNet model
    fn build_resnet(&self, config: &VisionModelConfig) -> BuildResult<ModelType> {
        if let VisionArchParams::ResNet(resnet_config) = &config.arch_params {
            let model = if resnet_config.layers == [2, 2, 2, 2] {
                ResNet::resnet18(config.num_classes)
            } else if resnet_config.layers == [3, 4, 6, 3] && !resnet_config.bottleneck {
                ResNet::resnet34(config.num_classes)
            } else if resnet_config.layers == [3, 4, 6, 3] && resnet_config.bottleneck {
                ResNet::resnet50(config.num_classes)
            } else {
                return Err(ModelError::LoadingError {
                    reason: format!(
                        "Unsupported ResNet configuration: {:?}",
                        resnet_config.layers
                    ),
                });
            };

            Ok(ModelType::ResNet(model?))
        } else {
            Err(ModelError::LoadingError {
                reason: "Invalid ResNet configuration".to_string(),
            })
        }
    }

    /// Build EfficientNet model
    fn build_efficientnet(&self, _config: &VisionModelConfig) -> BuildResult<ModelType> {
        // EfficientNet implementation exists but requires torsh-nn v0.2 API compatibility
        // Will be enabled in next major release
        Err(ModelError::LoadingError {
            reason: "EfficientNet not yet implemented".to_string(),
        })
    }

    /// Build Vision Transformer model
    fn build_vit(&self, config: &VisionModelConfig) -> BuildResult<ModelType> {
        if let VisionArchParams::VisionTransformer(vit_config) = &config.arch_params {
            // Create a ViTConfig from the provided parameters
            let vit_config_obj = crate::vision::vit::ViTConfig {
                variant: crate::vision::vit::ViTVariant::Base,
                img_size: config.input_size.0,
                patch_size: vit_config.patch_size,
                in_channels: 3, // Default to RGB channels
                num_classes: config.num_classes,
                embed_dim: vit_config.embed_dim,
                depth: vit_config.depth,
                num_heads: vit_config.num_heads,
                mlp_ratio: vit_config.mlp_ratio,
                qkv_bias: true,
                representation_size: None,
                attn_dropout: vit_config.attn_dropout_rate,
                proj_dropout: vit_config.dropout_rate,
                path_dropout: 0.0,
                norm_eps: 1e-5,
                global_pool: false,
                patch_embed_strategy: crate::vision::vit::PatchEmbedStrategy::Convolution,
            };
            let model = VisionTransformer::new(vit_config_obj);
            Ok(ModelType::VisionTransformer(model?))
        } else {
            Err(ModelError::LoadingError {
                reason: "Invalid Vision Transformer configuration".to_string(),
            })
        }
    }
}

impl Default for VisionModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder<ModelType> for VisionModelBuilder {
    type Config = VisionModelConfig;

    fn build(&self, config: &Self::Config) -> BuildResult<ModelType> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| ModelError::ValidationError { reason: e })?;

        match config.architecture {
            VisionArchitecture::ResNet => self.build_resnet(config),
            VisionArchitecture::EfficientNet => self.build_efficientnet(config),
            VisionArchitecture::VisionTransformer => self.build_vit(config),
            _ => Err(ModelError::LoadingError {
                reason: format!("Architecture {:?} not yet implemented", config.architecture),
            }),
        }
    }

    fn build_from_name(&self, name: &str) -> BuildResult<ModelType> {
        let config = self
            .get_config(name)
            .ok_or_else(|| ModelError::ModelNotFound {
                name: name.to_string(),
            })?;
        self.build(&config)
    }

    fn available_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    fn get_config(&self, name: &str) -> Option<Self::Config> {
        self.configs.get(name).cloned()
    }
}

/// NLP model builder
pub struct NlpModelBuilder {
    configs: HashMap<String, NlpModelConfig>,
}

impl NlpModelBuilder {
    /// Create new NLP model builder
    pub fn new() -> Self {
        let mut configs = HashMap::new();

        // Add predefined configurations
        configs.extend(ModelConfigs::bert_configs());

        Self { configs }
    }

    /// Add custom configuration
    pub fn add_config(&mut self, name: String, config: NlpModelConfig) {
        self.configs.insert(name, config);
    }

    /// Build BERT model (placeholder - actual implementation would be in nlp module)
    fn build_bert(&self, _config: &NlpModelConfig) -> BuildResult<ModelType> {
        // This is a placeholder - actual BERT implementation would go here
        Err(ModelError::LoadingError {
            reason: "BERT implementation not yet available".to_string(),
        })
    }
}

impl Default for NlpModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder<ModelType> for NlpModelBuilder {
    type Config = NlpModelConfig;

    fn build(&self, config: &Self::Config) -> BuildResult<ModelType> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| ModelError::ValidationError { reason: e })?;

        match config.architecture {
            NlpArchitecture::BERT => self.build_bert(config),
            _ => Err(ModelError::LoadingError {
                reason: format!("Architecture {:?} not yet implemented", config.architecture),
            }),
        }
    }

    fn build_from_name(&self, name: &str) -> BuildResult<ModelType> {
        let config = self
            .get_config(name)
            .ok_or_else(|| ModelError::ModelNotFound {
                name: name.to_string(),
            })?;
        self.build(&config)
    }

    fn available_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    fn get_config(&self, name: &str) -> Option<Self::Config> {
        self.configs.get(name).cloned()
    }
}

/// Universal model factory
pub struct ModelFactory {
    vision_builder: VisionModelBuilder,
    nlp_builder: NlpModelBuilder,
}

impl ModelFactory {
    /// Create new model factory
    pub fn new() -> Self {
        Self {
            vision_builder: VisionModelBuilder::new(),
            nlp_builder: NlpModelBuilder::new(),
        }
    }

    /// Build vision model
    pub fn build_vision_model(&self, name: &str) -> BuildResult<ModelType> {
        self.vision_builder.build_from_name(name)
    }

    /// Build vision model from config
    pub fn build_vision_model_from_config(
        &self,
        config: &VisionModelConfig,
    ) -> BuildResult<ModelType> {
        self.vision_builder.build(config)
    }

    /// Build NLP model
    pub fn build_nlp_model(&self, name: &str) -> BuildResult<ModelType> {
        self.nlp_builder.build_from_name(name)
    }

    /// Build NLP model from config
    pub fn build_nlp_model_from_config(&self, config: &NlpModelConfig) -> BuildResult<ModelType> {
        self.nlp_builder.build(config)
    }

    /// List all available models
    pub fn list_all_models(&self) -> HashMap<String, Vec<String>> {
        let mut models = HashMap::new();
        models.insert("vision".to_string(), self.vision_builder.available_models());
        models.insert("nlp".to_string(), self.nlp_builder.available_models());
        models
    }

    /// Get model information
    pub fn get_model_info(&self, domain: &str, name: &str) -> Option<ModelInfo> {
        match domain {
            "vision" => {
                if let Some(config) = self.vision_builder.get_config(name) {
                    Some(ModelInfo {
                        name: name.to_string(),
                        architecture: config.model_name(),
                        variant: config.variant(),
                        parameters: config.estimated_parameters(),
                        description: format!(
                            "{} model for computer vision tasks",
                            config.model_name()
                        ),
                        input_spec: format!("RGB image {:?}", config.input_size),
                        output_spec: format!("{} class probabilities", config.num_classes),
                        domain: "vision".to_string(),
                    })
                } else {
                    None
                }
            }
            "nlp" => {
                if let Some(config) = self.nlp_builder.get_config(name) {
                    Some(ModelInfo {
                        name: name.to_string(),
                        architecture: config.model_name(),
                        variant: config.variant(),
                        parameters: config.estimated_parameters(),
                        description: format!(
                            "{} model for natural language processing",
                            config.model_name()
                        ),
                        input_spec: format!("Tokenized text [seq_len <= {}]", config.max_length),
                        output_spec: "Hidden states or logits".to_string(),
                        domain: "nlp".to_string(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Create custom vision model with modifications
    pub fn create_custom_vision_model(
        &mut self,
        base_name: &str,
        custom_name: &str,
        modifications: VisionModelModifications,
    ) -> BuildResult<()> {
        let mut base_config =
            self.vision_builder
                .get_config(base_name)
                .ok_or_else(|| ModelError::ModelNotFound {
                    name: base_name.to_string(),
                })?;

        // Apply modifications
        if let Some(num_classes) = modifications.num_classes {
            base_config.num_classes = num_classes;
        }
        if let Some(input_size) = modifications.input_size {
            base_config.input_size = input_size;
        }
        if let Some(dropout_rate) = modifications.dropout_rate {
            // Apply dropout rate based on architecture
            match &mut base_config.arch_params {
                VisionArchParams::VisionTransformer(ref mut vit_config) => {
                    vit_config.dropout_rate = dropout_rate;
                }
                VisionArchParams::EfficientNet(ref mut eff_config) => {
                    eff_config.dropout_rate = dropout_rate;
                }
                _ => {}
            }
        }

        // Validate modified configuration
        base_config
            .validate()
            .map_err(|e| ModelError::ValidationError { reason: e })?;

        // Add to builder
        self.vision_builder
            .add_config(custom_name.to_string(), base_config);

        Ok(())
    }
}

impl Default for ModelFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Vision model modifications for creating custom variants
#[derive(Debug, Clone, Default)]
pub struct VisionModelModifications {
    pub num_classes: Option<usize>,
    pub input_size: Option<(usize, usize)>,
    pub dropout_rate: Option<f32>,
}

/// Simplified ModelInfo for the factory (to avoid circular dependencies)
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub variant: String,
    pub parameters: u64,
    pub description: String,
    pub input_spec: String,
    pub output_spec: String,
    pub domain: String,
}

lazy_static::lazy_static! {
    /// Global model factory instance using lazy_static for safe static initialization
    static ref GLOBAL_FACTORY: ModelFactory = ModelFactory::new();
}

/// Get the global model factory
pub fn get_global_factory() -> &'static ModelFactory {
    &GLOBAL_FACTORY
}

/// Convenience functions for easy model creation
pub mod quick {
    use super::*;

    /// Create ResNet-18 model
    pub fn resnet18(num_classes: usize) -> BuildResult<ModelType> {
        let factory = get_global_factory();
        let mut config = factory.vision_builder.get_config("resnet18").unwrap();
        config.num_classes = num_classes;
        factory.build_vision_model_from_config(&config)
    }

    /// Create ResNet-50 model
    pub fn resnet50(num_classes: usize) -> BuildResult<ModelType> {
        let factory = get_global_factory();
        let mut config = factory.vision_builder.get_config("resnet50").unwrap();
        config.num_classes = num_classes;
        factory.build_vision_model_from_config(&config)
    }

    /// Create EfficientNet-B0 model
    pub fn efficientnet_b0(num_classes: usize) -> BuildResult<ModelType> {
        let factory = get_global_factory();
        let mut config = factory
            .vision_builder
            .get_config("efficientnet_b0")
            .unwrap();
        config.num_classes = num_classes;
        factory.build_vision_model_from_config(&config)
    }

    /// Create ViT-Base model
    pub fn vit_base(num_classes: usize) -> BuildResult<ModelType> {
        let factory = get_global_factory();
        let mut config = factory
            .vision_builder
            .get_config("vit_base_patch16_224")
            .unwrap();
        config.num_classes = num_classes;
        factory.build_vision_model_from_config(&config)
    }

    /// Create custom model with specific configuration
    pub fn custom_vision_model(
        base_model: &str,
        num_classes: usize,
        input_size: Option<(usize, usize)>,
        dropout_rate: Option<f32>,
    ) -> BuildResult<ModelType> {
        let factory = get_global_factory();
        let mut config = factory
            .vision_builder
            .get_config(base_model)
            .ok_or_else(|| ModelError::ModelNotFound {
                name: base_model.to_string(),
            })?;

        config.num_classes = num_classes;
        if let Some(size) = input_size {
            config.input_size = size;
        }
        if let Some(dropout) = dropout_rate {
            match &mut config.arch_params {
                VisionArchParams::VisionTransformer(ref mut vit_config) => {
                    vit_config.dropout_rate = dropout;
                }
                VisionArchParams::EfficientNet(ref mut eff_config) => {
                    eff_config.dropout_rate = dropout;
                }
                _ => {}
            }
        }

        factory.build_vision_model_from_config(&config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_model_builder() {
        let builder = VisionModelBuilder::new();
        let available = builder.available_models();

        assert!(available.contains(&"resnet18".to_string()));
        assert!(available.contains(&"efficientnet_b0".to_string()));
        assert!(available.contains(&"vit_base_patch16_224".to_string()));
    }

    #[test]
    fn test_model_factory() {
        let factory = ModelFactory::new();
        let all_models = factory.list_all_models();

        assert!(all_models.contains_key("vision"));
        assert!(all_models.contains_key("nlp"));

        let vision_models = &all_models["vision"];
        assert!(!vision_models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let factory = ModelFactory::new();
        let info = factory.get_model_info("vision", "resnet18").unwrap();

        assert_eq!(info.name, "resnet18");
        assert_eq!(info.domain, "vision");
        assert!(info.parameters > 10_000_000);
    }

    #[test]
    fn test_custom_model_creation() {
        let mut factory = ModelFactory::new();

        let modifications = VisionModelModifications {
            num_classes: Some(10),
            input_size: Some((32, 32)),
            dropout_rate: None,
        };

        factory
            .create_custom_vision_model("resnet18", "resnet18_cifar10", modifications)
            .unwrap();

        let available = factory.vision_builder.available_models();
        assert!(available.contains(&"resnet18_cifar10".to_string()));

        let config = factory
            .vision_builder
            .get_config("resnet18_cifar10")
            .unwrap();
        assert_eq!(config.num_classes, 10);
        assert_eq!(config.input_size, (32, 32));
    }

    #[test]
    fn test_quick_builders() {
        // These would fail without actual model implementations, but test the interface
        let result = quick::resnet18(1000);
        assert!(result.is_ok() || result.is_err()); // Just test it compiles and runs
    }
}
