use super::basic::{CenterCrop, Normalize, Resize, ToTensor};
use super::core::{Compose, Transform};
use super::random::{RandomHorizontalFlip, RandomVerticalFlip};
use std::collections::HashMap;

/// Transform registry for dynamic transform creation and management
pub struct TransformRegistry {
    transforms: HashMap<String, Box<dyn Fn() -> Box<dyn Transform>>>,
}

impl TransformRegistry {
    /// Create a new transform registry
    pub fn new() -> Self {
        let mut registry = Self {
            transforms: HashMap::new(),
        };
        registry.register_default_transforms();
        registry
    }

    /// Register default transforms
    fn register_default_transforms(&mut self) {
        self.register(
            "resize",
            Box::new(|| Box::new(Resize::new((224, 224))) as Box<dyn Transform>),
        );
        self.register(
            "center_crop",
            Box::new(|| Box::new(CenterCrop::new((224, 224))) as Box<dyn Transform>),
        );
        self.register(
            "horizontal_flip",
            Box::new(|| Box::new(RandomHorizontalFlip::new(0.5)) as Box<dyn Transform>),
        );
        self.register(
            "vertical_flip",
            Box::new(|| Box::new(RandomVerticalFlip::new(0.5)) as Box<dyn Transform>),
        );
        self.register(
            "normalize",
            Box::new(|| {
                Box::new(Normalize::new(
                    vec![0.485, 0.456, 0.406],
                    vec![0.229, 0.224, 0.225],
                )) as Box<dyn Transform>
            }),
        );
        self.register(
            "to_tensor",
            Box::new(|| Box::new(ToTensor::new()) as Box<dyn Transform>),
        );
    }

    /// Register a new transform
    pub fn register<F>(&mut self, name: &str, factory: Box<F>)
    where
        F: Fn() -> Box<dyn Transform> + 'static,
    {
        self.transforms.insert(name.to_string(), Box::new(factory));
    }

    /// Create a transform by name
    pub fn create(&self, name: &str) -> Option<Box<dyn Transform>> {
        self.transforms.get(name).map(|factory| factory())
    }

    /// List all available transforms
    pub fn list_transforms(&self) -> Vec<String> {
        self.transforms.keys().cloned().collect()
    }
}

impl Default for TransformRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder pattern for creating transform pipelines
pub struct TransformBuilder {
    transforms: Vec<Box<dyn Transform>>,
}

impl TransformBuilder {
    /// Create a new transform builder
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the pipeline
    pub fn add<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Add resize transform
    pub fn resize(self, size: (usize, usize)) -> Self {
        self.add(Resize::new(size))
    }

    /// Add center crop transform
    pub fn center_crop(self, size: (usize, usize)) -> Self {
        self.add(CenterCrop::new(size))
    }

    /// Add random horizontal flip
    pub fn random_horizontal_flip(self, p: f32) -> Self {
        self.add(RandomHorizontalFlip::new(p))
    }

    /// Add random vertical flip
    pub fn random_vertical_flip(self, p: f32) -> Self {
        self.add(RandomVerticalFlip::new(p))
    }

    /// Add normalization
    pub fn normalize(self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.add(Normalize::new(mean, std))
    }

    /// Add ImageNet normalization (common default)
    pub fn imagenet_normalize(self) -> Self {
        self.normalize(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }

    /// Build the final transform pipeline
    pub fn build(self) -> Compose {
        Compose::new(self.transforms)
    }
}

impl Default for TransformBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Transform introspection utilities
pub trait TransformIntrospection {
    /// Get a human-readable description of the transform pipeline
    fn describe(&self) -> String;

    /// Get detailed statistics about the pipeline
    fn statistics(&self) -> TransformStats;

    /// Validate the transform pipeline for common issues
    fn validate(&self) -> crate::Result<()>;
}

#[derive(Debug, Clone)]
pub struct TransformStats {
    pub total_transforms: usize,
    pub random_transforms: usize,
    pub deterministic_transforms: usize,
    pub resize_operations: usize,
    pub augmentation_operations: usize,
}

impl TransformIntrospection for Compose {
    fn describe(&self) -> String {
        let descriptions: Vec<String> = self
            .transforms()
            .iter()
            .map(|t| {
                let params = t.parameters();
                if params.is_empty() {
                    t.name().to_string()
                } else {
                    let param_str = params
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}({})", t.name(), param_str)
                }
            })
            .collect();

        format!("Compose([{}])", descriptions.join(" -> "))
    }

    fn statistics(&self) -> TransformStats {
        let mut stats = TransformStats {
            total_transforms: self.transforms().len(),
            random_transforms: 0,
            deterministic_transforms: 0,
            resize_operations: 0,
            augmentation_operations: 0,
        };

        for transform in self.transforms() {
            let name = transform.name();

            // Count random vs deterministic transforms
            if name.contains("Random") || name.contains("Cutout") || name.contains("Erasing") {
                stats.random_transforms += 1;
            } else {
                stats.deterministic_transforms += 1;
            }

            // Count resize operations
            if name.contains("Resize") || name.contains("Crop") {
                stats.resize_operations += 1;
            }

            // Count augmentation operations (all transforms that modify image appearance)
            if name.contains("Flip")
                || name.contains("Rotation")
                || name.contains("Jitter")
                || name.contains("Mix")
                || name.contains("Augment")
                || name.contains("Cutout")
                || name.contains("Erasing")
            {
                stats.augmentation_operations += 1;
            }
        }

        stats
    }

    fn validate(&self) -> crate::Result<()> {
        let stats = self.statistics();

        // Check for common issues
        if stats.total_transforms == 0 {
            return Err(crate::VisionError::TransformError(
                "Empty transform pipeline".to_string(),
            ));
        }

        // Warn about excessive augmentation
        if stats.augmentation_operations > stats.total_transforms / 2 {
            eprintln!(
                "Warning: High ratio of augmentation operations ({}/{})",
                stats.augmentation_operations, stats.total_transforms
            );
        }

        // Check for normalize at the end (common practice)
        if let Some(last_transform) = self.transforms().last() {
            if last_transform.name() != "Normalize" && last_transform.name() != "ToTensor" {
                eprintln!("Warning: Consider adding normalization as the last transform");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_registry_creation() {
        let registry = TransformRegistry::new();
        let transforms = registry.list_transforms();
        assert!(!transforms.is_empty());
        assert!(transforms.contains(&"resize".to_string()));
        assert!(transforms.contains(&"normalize".to_string()));
    }

    #[test]
    fn test_transform_registry_create() {
        let registry = TransformRegistry::new();

        let resize = registry.create("resize");
        assert!(resize.is_some());
        assert_eq!(resize.unwrap().name(), "Resize");

        let unknown = registry.create("unknown");
        assert!(unknown.is_none());
    }

    #[test]
    fn test_transform_registry_register() {
        let mut registry = TransformRegistry::new();
        registry.register(
            "custom_resize",
            Box::new(|| Box::new(Resize::new((128, 128))) as Box<dyn Transform>),
        );

        let custom = registry.create("custom_resize");
        assert!(custom.is_some());
        assert_eq!(custom.unwrap().name(), "Resize");
    }

    #[test]
    fn test_transform_builder() {
        let pipeline = TransformBuilder::new()
            .resize((224, 224))
            .random_horizontal_flip(0.5)
            .imagenet_normalize()
            .build();

        assert_eq!(pipeline.len(), 3);
        assert_eq!(pipeline.transforms()[0].name(), "Resize");
        assert_eq!(pipeline.transforms()[1].name(), "RandomHorizontalFlip");
        assert_eq!(pipeline.transforms()[2].name(), "Normalize");
    }

    #[test]
    fn test_transform_builder_default() {
        let builder = TransformBuilder::default();
        let pipeline = builder.build();
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_transform_introspection_describe() {
        let pipeline = TransformBuilder::new()
            .resize((224, 224))
            .random_horizontal_flip(0.5)
            .build();

        let description = pipeline.describe();
        assert!(description.contains("Resize"));
        assert!(description.contains("RandomHorizontalFlip"));
        assert!(description.contains("size=(224, 224)"));
        assert!(description.contains("probability=0.50"));
    }

    #[test]
    fn test_transform_introspection_statistics() {
        let pipeline = TransformBuilder::new()
            .resize((224, 224))
            .random_horizontal_flip(0.5)
            .center_crop((200, 200))
            .build();

        let stats = pipeline.statistics();
        assert_eq!(stats.total_transforms, 3);
        assert_eq!(stats.random_transforms, 1);
        assert_eq!(stats.deterministic_transforms, 2);
        assert_eq!(stats.resize_operations, 2); // resize and crop
        assert_eq!(stats.augmentation_operations, 1); // flip
    }

    #[test]
    fn test_transform_introspection_validate() {
        let pipeline = TransformBuilder::new()
            .resize((224, 224))
            .imagenet_normalize()
            .build();

        let result = pipeline.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_transform_introspection_validate_empty() {
        let pipeline = TransformBuilder::new().build();
        let result = pipeline.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_builder_methods() {
        let pipeline = TransformBuilder::new()
            .center_crop((256, 256))
            .random_vertical_flip(0.3)
            .normalize(vec![0.5], vec![0.5])
            .build();

        assert_eq!(pipeline.len(), 3);
        assert_eq!(pipeline.transforms()[0].name(), "CenterCrop");
        assert_eq!(pipeline.transforms()[1].name(), "RandomVerticalFlip");
        assert_eq!(pipeline.transforms()[2].name(), "Normalize");
    }
}
