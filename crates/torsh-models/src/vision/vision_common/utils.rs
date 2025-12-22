//! Utility functions for vision models

use super::preprocessing::ImagePreprocessor;
use super::types::{VisionArchitecture, VisionModelVariant};

/// Vision model utilities
pub struct VisionModelUtils;

impl VisionModelUtils {
    /// Get recommended preprocessor for a model
    pub fn get_preprocessor(model_name: &str) -> ImagePreprocessor {
        // Default to ImageNet preprocessing for most models
        match model_name {
            name if name.contains("cifar") => ImagePreprocessor::cifar10(),
            name if name.contains("yolo") => ImagePreprocessor::yolo(),
            name if name.contains("detr") || name.contains("mask") => {
                ImagePreprocessor::large_image()
            }
            _ => ImagePreprocessor::imagenet(),
        }
    }

    /// Get model variant by name
    pub fn get_model_variant(name: &str) -> Option<VisionModelVariant> {
        let models = get_common_vision_models();
        models.into_iter().find(|m| m.variant == name)
    }

    /// List models by architecture
    pub fn get_models_by_architecture(architecture: VisionArchitecture) -> Vec<VisionModelVariant> {
        let models = get_common_vision_models();
        models
            .into_iter()
            .filter(|m| m.architecture == architecture)
            .collect()
    }

    /// Get all available model names
    pub fn get_available_models() -> Vec<String> {
        get_common_vision_models()
            .into_iter()
            .map(|m| m.variant)
            .collect()
    }

    /// Softmax function for probability computation
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }

    /// Get top-k predictions with class names
    pub fn get_top_k_predictions(
        logits: &[f32],
        k: usize,
        class_names: Option<&[String]>,
    ) -> Vec<(usize, f32, Option<String>)> {
        let probabilities = Self::softmax(logits);

        let mut indexed_probs: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();

        // Sort by probability in descending order
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top k and add class names if available
        indexed_probs
            .into_iter()
            .take(k)
            .map(|(idx, prob)| {
                let class_name = class_names.and_then(|names| names.get(idx)).cloned();
                (idx, prob, class_name)
            })
            .collect()
    }

    /// Calculate model efficiency score (accuracy per parameter)
    pub fn calculate_efficiency_score(variant: &VisionModelVariant) -> Option<f32> {
        variant.imagenet_top1_accuracy.map(|accuracy| {
            accuracy / (variant.parameters as f32 / 1_000_000.0) // accuracy per million parameters
        })
    }

    /// Get models sorted by efficiency
    pub fn get_most_efficient_models(
        architecture: Option<VisionArchitecture>,
    ) -> Vec<VisionModelVariant> {
        let mut models = if let Some(arch) = architecture {
            Self::get_models_by_architecture(arch)
        } else {
            get_common_vision_models()
        };

        // Sort by efficiency score (descending)
        models.sort_by(|a, b| {
            let score_a = Self::calculate_efficiency_score(a).unwrap_or(0.0);
            let score_b = Self::calculate_efficiency_score(b).unwrap_or(0.0);
            score_b.partial_cmp(&score_a).unwrap()
        });

        models
    }

    /// Get recommended model for target accuracy and parameter budget
    pub fn recommend_model(
        min_accuracy: f32,
        max_parameters: u64,
        architecture: Option<VisionArchitecture>,
    ) -> Option<VisionModelVariant> {
        let models = if let Some(arch) = architecture {
            Self::get_models_by_architecture(arch)
        } else {
            get_common_vision_models()
        };

        models
            .into_iter()
            .filter(|m| {
                m.parameters <= max_parameters
                    && m.imagenet_top1_accuracy
                        .map_or(false, |acc| acc >= min_accuracy)
            })
            .max_by(|a, b| {
                let score_a = Self::calculate_efficiency_score(a).unwrap_or(0.0);
                let score_b = Self::calculate_efficiency_score(b).unwrap_or(0.0);
                score_a.partial_cmp(&score_b).unwrap()
            })
    }
}

/// Common vision model variants with benchmarks
pub fn get_common_vision_models() -> Vec<VisionModelVariant> {
    vec![
        // ResNet variants
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet18".to_string(),
            parameters: 11_689_512,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(69.758),
            imagenet_top5_accuracy: Some(89.078),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet34".to_string(),
            parameters: 21_797_672,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(73.314),
            imagenet_top5_accuracy: Some(91.420),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet50".to_string(),
            parameters: 25_557_032,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(76.130),
            imagenet_top5_accuracy: Some(92.862),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet101".to_string(),
            parameters: 44_549_160,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(77.374),
            imagenet_top5_accuracy: Some(93.546),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet152".to_string(),
            parameters: 60_192_808,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(78.312),
            imagenet_top5_accuracy: Some(94.046),
        },
        // EfficientNet variants
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b0".to_string(),
            parameters: 5_288_548,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(77.692),
            imagenet_top5_accuracy: Some(93.532),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b1".to_string(),
            parameters: 7_794_184,
            input_size: (3, 240, 240),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(78.642),
            imagenet_top5_accuracy: Some(94.186),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b2".to_string(),
            parameters: 9_109_994,
            input_size: (3, 260, 260),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(79.688),
            imagenet_top5_accuracy: Some(94.876),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b3".to_string(),
            parameters: 12_233_232,
            input_size: (3, 300, 300),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(81.178),
            imagenet_top5_accuracy: Some(95.718),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b4".to_string(),
            parameters: 19_341_616,
            input_size: (3, 380, 380),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(82.932),
            imagenet_top5_accuracy: Some(96.448),
        },
        // MobileNet variants
        VisionModelVariant {
            architecture: VisionArchitecture::MobileNet,
            variant: "mobilenet_v2".to_string(),
            parameters: 3_504_872,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(71.878),
            imagenet_top5_accuracy: Some(90.286),
        },
        // Vision Transformer variants
        VisionModelVariant {
            architecture: VisionArchitecture::VisionTransformer,
            variant: "vit_base_patch16_224".to_string(),
            parameters: 86_567_656,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(81.072),
            imagenet_top5_accuracy: Some(95.318),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::VisionTransformer,
            variant: "vit_large_patch16_224".to_string(),
            parameters: 304_326_632,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(82.632),
            imagenet_top5_accuracy: Some(96.176),
        },
        // Swin Transformer variants
        VisionModelVariant {
            architecture: VisionArchitecture::SwinTransformer,
            variant: "swin_tiny_patch4_window7_224".to_string(),
            parameters: 28_288_354,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(81.474),
            imagenet_top5_accuracy: Some(95.776),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::SwinTransformer,
            variant: "swin_small_patch4_window7_224".to_string(),
            parameters: 49_606_258,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(83.196),
            imagenet_top5_accuracy: Some(96.360),
        },
        // ConvNeXt variants
        VisionModelVariant {
            architecture: VisionArchitecture::ConvNeXt,
            variant: "convnext_tiny".to_string(),
            parameters: 28_589_128,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(82.520),
            imagenet_top5_accuracy: Some(96.146),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ConvNeXt,
            variant: "convnext_small".to_string(),
            parameters: 50_223_688,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(83.608),
            imagenet_top5_accuracy: Some(96.640),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_model_variant() {
        let variant = VisionModelUtils::get_model_variant("resnet18");
        assert!(variant.is_some());
        let variant = variant.unwrap();
        assert_eq!(variant.architecture, VisionArchitecture::ResNet);
        assert_eq!(variant.variant, "resnet18");
    }

    #[test]
    fn test_get_models_by_architecture() {
        let resnet_models =
            VisionModelUtils::get_models_by_architecture(VisionArchitecture::ResNet);
        assert!(!resnet_models.is_empty());
        assert!(resnet_models
            .iter()
            .all(|m| m.architecture == VisionArchitecture::ResNet));
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = VisionModelUtils::softmax(&logits);
        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Check that probabilities are in ascending order for ascending logits
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_get_top_k_predictions() {
        let logits = vec![0.1, 2.0, 1.0, 3.0];
        let class_names = vec![
            "class0".to_string(),
            "class1".to_string(),
            "class2".to_string(),
            "class3".to_string(),
        ];
        let top_3 = VisionModelUtils::get_top_k_predictions(&logits, 3, Some(&class_names));
        assert_eq!(top_3.len(), 3);
        assert_eq!(top_3[0].0, 3); // Index of highest score
        assert_eq!(top_3[0].2, Some("class3".to_string()));
        assert_eq!(top_3[1].0, 1); // Index of second highest
        assert_eq!(top_3[2].0, 2); // Index of third highest
    }

    #[test]
    fn test_calculate_efficiency_score() {
        let variant = VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "test".to_string(),
            parameters: 10_000_000, // 10M parameters
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(80.0),
            imagenet_top5_accuracy: Some(95.0),
        };

        let score = VisionModelUtils::calculate_efficiency_score(&variant);
        assert!(score.is_some());
        assert!((score.unwrap() - 8.0).abs() < 1e-6); // 80.0 / 10 = 8.0
    }

    #[test]
    fn test_recommend_model() {
        let recommendation = VisionModelUtils::recommend_model(
            75.0,       // min accuracy
            30_000_000, // max parameters (30M)
            Some(VisionArchitecture::ResNet),
        );

        assert!(recommendation.is_some());
        let model = recommendation.unwrap();
        assert!(model.imagenet_top1_accuracy.unwrap() >= 75.0);
        assert!(model.parameters <= 30_000_000);
        assert_eq!(model.architecture, VisionArchitecture::ResNet);
    }

    #[test]
    fn test_get_preprocessor() {
        let imagenet_preprocessor = VisionModelUtils::get_preprocessor("resnet18");
        assert_eq!(imagenet_preprocessor.output_size(), (224, 224));

        let cifar_preprocessor = VisionModelUtils::get_preprocessor("resnet_cifar10");
        assert_eq!(cifar_preprocessor.output_size(), (32, 32));

        let yolo_preprocessor = VisionModelUtils::get_preprocessor("yolo_v5");
        assert_eq!(yolo_preprocessor.output_size(), (640, 640));
    }
}
