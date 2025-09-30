use crate::{ModelConfig, Result, VisionError, VisionModel};
use std::collections::HashMap;

/// Model registry for vision models
pub struct ModelRegistry {
    models: HashMap<String, String>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        registry.register_default_models();
        registry
    }

    /// Register default models
    fn register_default_models(&mut self) {
        self.models
            .insert("resnet18".to_string(), "ResNet-18".to_string());
        self.models
            .insert("resnet34".to_string(), "ResNet-34".to_string());
        self.models
            .insert("resnet50".to_string(), "ResNet-50".to_string());
        self.models
            .insert("alexnet".to_string(), "AlexNet".to_string());
        self.models
            .insert("vgg16".to_string(), "VGG-16".to_string());
    }

    /// List available models
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Check if model exists
    pub fn has_model(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
