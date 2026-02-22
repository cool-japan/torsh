//! Common utilities and types for multimodal models
//!
//! This module provides shared types, enums, and utility functions
//! used across different multimodal architectures in ToRSh.

use torsh_core::error::{Result, TorshError};

/// Multimodal model architectures supported by ToRSh
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MultimodalArchitecture {
    /// CLIP: Contrastive Language-Image Pre-Training
    CLIP,
    /// ALIGN: Large-scale Alignment of Vision and Language
    ALIGN,
    /// Flamingo: Few-Shot Learning with Frozen Vision Models
    Flamingo,
    /// DALL-E: Text-to-Image Generation
    DallE,
    /// BLIP: Bootstrapping Language-Image Pre-training
    BLIP,
    /// LLaVA: Large Language and Vision Assistant
    LLaVA,
    /// InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
    InstructBLIP,
}

impl MultimodalArchitecture {
    /// Get string representation of the architecture
    pub fn as_str(&self) -> &'static str {
        match self {
            MultimodalArchitecture::CLIP => "CLIP",
            MultimodalArchitecture::ALIGN => "ALIGN",
            MultimodalArchitecture::Flamingo => "Flamingo",
            MultimodalArchitecture::DallE => "DALL-E",
            MultimodalArchitecture::BLIP => "BLIP",
            MultimodalArchitecture::LLaVA => "LLaVA",
            MultimodalArchitecture::InstructBLIP => "InstructBLIP",
        }
    }

    /// Get detailed description of the architecture
    pub fn description(&self) -> &'static str {
        match self {
            MultimodalArchitecture::CLIP => {
                "Contrastive Language-Image Pre-Training for learning visual representations from natural language supervision"
            }
            MultimodalArchitecture::ALIGN => {
                "Large-scale Alignment of vision and language representations learned from noisy web data"
            }
            MultimodalArchitecture::Flamingo => {
                "Few-Shot Learning with frozen vision models and cross-attention mechanisms"
            }
            MultimodalArchitecture::DallE => {
                "Text-to-Image generation using transformer architectures"
            }
            MultimodalArchitecture::BLIP => {
                "Bootstrapping Language-Image Pre-training for unified vision-language understanding and generation"
            }
            MultimodalArchitecture::LLaVA => {
                "Large Language and Vision Assistant for general-purpose visual and language understanding"
            }
            MultimodalArchitecture::InstructBLIP => {
                "Instruction-tuned BLIP models for improved vision-language instruction following"
            }
        }
    }

    /// Get the primary use case for this architecture
    pub fn primary_use_case(&self) -> &'static str {
        match self {
            MultimodalArchitecture::CLIP => "Zero-shot image classification and retrieval",
            MultimodalArchitecture::ALIGN => "Large-scale contrastive learning",
            MultimodalArchitecture::Flamingo => "Few-shot multimodal learning",
            MultimodalArchitecture::DallE => "Text-to-image generation",
            MultimodalArchitecture::BLIP => "Vision-language understanding and captioning",
            MultimodalArchitecture::LLaVA => "Visual question answering and conversation",
            MultimodalArchitecture::InstructBLIP => "Instruction-following for vision-language tasks",
        }
    }

    /// Check if the architecture supports text-to-image generation
    pub fn supports_text_to_image(&self) -> bool {
        matches!(self, MultimodalArchitecture::DallE)
    }

    /// Check if the architecture supports image-to-text generation
    pub fn supports_image_to_text(&self) -> bool {
        matches!(
            self,
            MultimodalArchitecture::BLIP
                | MultimodalArchitecture::LLaVA
                | MultimodalArchitecture::InstructBLIP
                | MultimodalArchitecture::Flamingo
        )
    }

    /// Check if the architecture supports contrastive learning
    pub fn supports_contrastive_learning(&self) -> bool {
        matches!(
            self,
            MultimodalArchitecture::CLIP | MultimodalArchitecture::ALIGN
        )
    }

    /// Check if the architecture supports few-shot learning
    pub fn supports_few_shot_learning(&self) -> bool {
        matches!(self, MultimodalArchitecture::Flamingo)
    }

    /// Get all available architectures
    pub fn all() -> Vec<Self> {
        vec![
            Self::CLIP,
            Self::ALIGN,
            Self::Flamingo,
            Self::DallE,
            Self::BLIP,
            Self::LLaVA,
            Self::InstructBLIP,
        ]
    }

    /// Parse architecture from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "clip" => Ok(Self::CLIP),
            "align" => Ok(Self::ALIGN),
            "flamingo" => Ok(Self::Flamingo),
            "dall-e" | "dalle" => Ok(Self::DallE),
            "blip" => Ok(Self::BLIP),
            "llava" => Ok(Self::LLaVA),
            "instructblip" | "instruct-blip" => Ok(Self::InstructBLIP),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown multimodal architecture: {}. Available: {}",
                s,
                Self::all()
                    .iter()
                    .map(|a| a.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

impl std::fmt::Display for MultimodalArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for MultimodalArchitecture {
    type Err = TorshError;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_str(s)
    }
}

/// Common activation functions used in multimodal models
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    ReLU,
    GELU,
    QuickGELU,
    Swish,
    Tanh,
    Sigmoid,
}

impl ActivationType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ActivationType::ReLU => "relu",
            ActivationType::GELU => "gelu",
            ActivationType::QuickGELU => "quick_gelu",
            ActivationType::Swish => "swish",
            ActivationType::Tanh => "tanh",
            ActivationType::Sigmoid => "sigmoid",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "relu" => Ok(Self::ReLU),
            "gelu" => Ok(Self::GELU),
            "quick_gelu" | "quickgelu" => Ok(Self::QuickGELU),
            "swish" | "silu" => Ok(Self::Swish),
            "tanh" => Ok(Self::Tanh),
            "sigmoid" => Ok(Self::Sigmoid),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown activation type: {}",
                s
            ))),
        }
    }
}

/// Vision encoder architectures used in multimodal models
#[derive(Debug, Clone, PartialEq)]
pub enum VisionEncoderType {
    /// Vision Transformer (ViT)
    ViT,
    /// Convolutional Neural Network (CNN)
    CNN,
    /// ResNet architecture
    ResNet,
    /// EfficientNet architecture
    EfficientNet,
    /// Swin Transformer
    SwinTransformer,
}

impl VisionEncoderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VisionEncoderType::ViT => "vit",
            VisionEncoderType::CNN => "cnn",
            VisionEncoderType::ResNet => "resnet",
            VisionEncoderType::EfficientNet => "efficientnet",
            VisionEncoderType::SwinTransformer => "swin",
        }
    }
}

/// Text encoder architectures used in multimodal models
#[derive(Debug, Clone, PartialEq)]
pub enum TextEncoderType {
    /// BERT-style bidirectional encoder
    BERT,
    /// GPT-style autoregressive decoder
    GPT,
    /// T5-style encoder-decoder
    T5,
    /// RoBERTa encoder
    RoBERTa,
    /// Plain transformer encoder
    Transformer,
}

impl TextEncoderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TextEncoderType::BERT => "bert",
            TextEncoderType::GPT => "gpt",
            TextEncoderType::T5 => "t5",
            TextEncoderType::RoBERTa => "roberta",
            TextEncoderType::Transformer => "transformer",
        }
    }
}

/// Common model tasks supported by multimodal architectures
#[derive(Debug, Clone, PartialEq)]
pub enum MultimodalTask {
    /// Image-text retrieval
    ImageTextRetrieval,
    /// Visual question answering
    VisualQuestionAnswering,
    /// Image captioning
    ImageCaptioning,
    /// Text-to-image generation
    TextToImageGeneration,
    /// Zero-shot image classification
    ZeroShotClassification,
    /// Visual reasoning
    VisualReasoning,
    /// Multimodal conversation
    MultimodalConversation,
}

impl MultimodalTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            MultimodalTask::ImageTextRetrieval => "image_text_retrieval",
            MultimodalTask::VisualQuestionAnswering => "visual_question_answering",
            MultimodalTask::ImageCaptioning => "image_captioning",
            MultimodalTask::TextToImageGeneration => "text_to_image_generation",
            MultimodalTask::ZeroShotClassification => "zero_shot_classification",
            MultimodalTask::VisualReasoning => "visual_reasoning",
            MultimodalTask::MultimodalConversation => "multimodal_conversation",
        }
    }

    /// Get architectures that support this task
    pub fn supported_by(&self) -> Vec<MultimodalArchitecture> {
        match self {
            MultimodalTask::ImageTextRetrieval => {
                vec![MultimodalArchitecture::CLIP, MultimodalArchitecture::ALIGN]
            }
            MultimodalTask::VisualQuestionAnswering => vec![
                MultimodalArchitecture::BLIP,
                MultimodalArchitecture::LLaVA,
                MultimodalArchitecture::InstructBLIP,
                MultimodalArchitecture::Flamingo,
            ],
            MultimodalTask::ImageCaptioning => vec![
                MultimodalArchitecture::BLIP,
                MultimodalArchitecture::Flamingo,
            ],
            MultimodalTask::TextToImageGeneration => vec![MultimodalArchitecture::DallE],
            MultimodalTask::ZeroShotClassification => {
                vec![MultimodalArchitecture::CLIP, MultimodalArchitecture::ALIGN]
            }
            MultimodalTask::VisualReasoning => vec![
                MultimodalArchitecture::LLaVA,
                MultimodalArchitecture::InstructBLIP,
            ],
            MultimodalTask::MultimodalConversation => {
                vec![MultimodalArchitecture::LLaVA, MultimodalArchitecture::Flamingo]
            }
        }
    }
}

/// Common utility functions for multimodal models
pub struct MultimodalUtils;

impl MultimodalUtils {
    /// Get recommended architecture for a specific task
    pub fn recommended_architecture_for_task(task: &MultimodalTask) -> MultimodalArchitecture {
        match task {
            MultimodalTask::ImageTextRetrieval => MultimodalArchitecture::CLIP,
            MultimodalTask::VisualQuestionAnswering => MultimodalArchitecture::LLaVA,
            MultimodalTask::ImageCaptioning => MultimodalArchitecture::BLIP,
            MultimodalTask::TextToImageGeneration => MultimodalArchitecture::DallE,
            MultimodalTask::ZeroShotClassification => MultimodalArchitecture::CLIP,
            MultimodalTask::VisualReasoning => MultimodalArchitecture::LLaVA,
            MultimodalTask::MultimodalConversation => MultimodalArchitecture::LLaVA,
        }
    }

    /// Compare architectures and their capabilities
    pub fn compare_architectures(
        arch1: &MultimodalArchitecture,
        arch2: &MultimodalArchitecture,
    ) -> String {
        format!(
            "Comparison:\n{} ({}): {}\n{} ({}): {}",
            arch1.as_str(),
            arch1.primary_use_case(),
            arch1.description(),
            arch2.as_str(),
            arch2.primary_use_case(),
            arch2.description()
        )
    }

    /// Get architecture family tree
    pub fn architecture_family_info() -> String {
        format!(
            "Multimodal Architecture Families:\n\
            \n• Contrastive Learning:\n  - CLIP: Dual encoders with contrastive loss\n  - ALIGN: Large-scale noisy web data\n\
            \n• Unified Models:\n  - BLIP: Encoder-decoder with bootstrapping\n  - InstructBLIP: Instruction-tuned BLIP\n\
            \n• Generative Models:\n  - DALL-E: Text-to-image generation\n  - LLaVA: Conversational AI\n\
            \n• Few-Shot Learning:\n  - Flamingo: Frozen vision + cross-attention"
        )
    }

    /// Get task capability matrix
    pub fn task_capability_matrix() -> String {
        let architectures = MultimodalArchitecture::all();
        let tasks = vec![
            MultimodalTask::ImageTextRetrieval,
            MultimodalTask::VisualQuestionAnswering,
            MultimodalTask::ImageCaptioning,
            MultimodalTask::TextToImageGeneration,
            MultimodalTask::ZeroShotClassification,
        ];

        let mut matrix = String::from("Task Capability Matrix:\n");
        matrix.push_str(&format!("{:<15}", "Architecture"));
        for task in &tasks {
            matrix.push_str(&format!("{:<20}", task.as_str()));
        }
        matrix.push('\n');

        for arch in &architectures {
            matrix.push_str(&format!("{:<15}", arch.as_str()));
            for task in &tasks {
                let supported = task.supported_by().contains(arch);
                matrix.push_str(&format!("{:<20}", if supported { "✓" } else { "✗" }));
            }
            matrix.push('\n');
        }

        matrix
    }

    /// Validate model configuration compatibility
    pub fn validate_config_compatibility(
        architecture: &MultimodalArchitecture,
        vision_type: &VisionEncoderType,
        text_type: &TextEncoderType,
    ) -> Result<()> {
        // Define valid combinations
        let valid = match architecture {
            MultimodalArchitecture::CLIP => {
                matches!(vision_type, VisionEncoderType::ViT | VisionEncoderType::ResNet)
                    && matches!(text_type, TextEncoderType::Transformer)
            }
            MultimodalArchitecture::ALIGN => {
                matches!(vision_type, VisionEncoderType::EfficientNet)
                    && matches!(text_type, TextEncoderType::BERT)
            }
            MultimodalArchitecture::BLIP => {
                matches!(vision_type, VisionEncoderType::ViT)
                    && matches!(text_type, TextEncoderType::BERT)
            }
            MultimodalArchitecture::Flamingo => {
                matches!(vision_type, VisionEncoderType::ViT)
                    && matches!(text_type, TextEncoderType::GPT)
            }
            _ => true, // Other architectures are flexible
        };

        if !valid {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid combination: {} with vision={} and text={}",
                architecture.as_str(),
                vision_type.as_str(),
                text_type.as_str()
            )));
        }

        Ok(())
    }
}

/// Factory trait for creating multimodal models
pub trait MultimodalModelFactory {
    type Model;
    type Config;

    /// Create model with default configuration
    fn create_default() -> Result<Self::Model>;

    /// Create model with custom configuration
    fn create_with_config(config: Self::Config) -> Result<Self::Model>;

    /// Get model information
    fn model_info() -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_architecture_string_conversion() {
        let arch = MultimodalArchitecture::CLIP;
        assert_eq!(arch.as_str(), "CLIP");
        assert_eq!(arch.to_string(), "CLIP");
    }

    #[test]
    fn test_multimodal_architecture_from_str() {
        assert_eq!(
            MultimodalArchitecture::from_str("clip").unwrap(),
            MultimodalArchitecture::CLIP
        );
        assert_eq!(
            MultimodalArchitecture::from_str("ALIGN").unwrap(),
            MultimodalArchitecture::ALIGN
        );
        assert!(MultimodalArchitecture::from_str("invalid").is_err());
    }

    #[test]
    fn test_multimodal_architecture_capabilities() {
        let clip = MultimodalArchitecture::CLIP;
        assert!(clip.supports_contrastive_learning());
        assert!(!clip.supports_text_to_image());

        let dalle = MultimodalArchitecture::DallE;
        assert!(dalle.supports_text_to_image());
        assert!(!dalle.supports_contrastive_learning());

        let flamingo = MultimodalArchitecture::Flamingo;
        assert!(flamingo.supports_few_shot_learning());
        assert!(flamingo.supports_image_to_text());
    }

    #[test]
    fn test_activation_type_conversion() {
        let gelu = ActivationType::GELU;
        assert_eq!(gelu.as_str(), "gelu");

        assert_eq!(
            ActivationType::from_str("quick_gelu").unwrap(),
            ActivationType::QuickGELU
        );
        assert!(ActivationType::from_str("invalid").is_err());
    }

    #[test]
    fn test_task_supported_architectures() {
        let vqa_task = MultimodalTask::VisualQuestionAnswering;
        let supported = vqa_task.supported_by();
        assert!(supported.contains(&MultimodalArchitecture::BLIP));
        assert!(supported.contains(&MultimodalArchitecture::LLaVA));
        assert!(!supported.contains(&MultimodalArchitecture::CLIP));
    }

    #[test]
    fn test_multimodal_utils_recommendations() {
        let task = MultimodalTask::ImageCaptioning;
        let recommended = MultimodalUtils::recommended_architecture_for_task(&task);
        assert_eq!(recommended, MultimodalArchitecture::BLIP);
    }

    #[test]
    fn test_config_validation() {
        // Valid combination
        assert!(MultimodalUtils::validate_config_compatibility(
            &MultimodalArchitecture::CLIP,
            &VisionEncoderType::ViT,
            &TextEncoderType::Transformer
        )
        .is_ok());

        // Invalid combination
        assert!(MultimodalUtils::validate_config_compatibility(
            &MultimodalArchitecture::ALIGN,
            &VisionEncoderType::ViT, // Should be EfficientNet
            &TextEncoderType::BERT
        )
        .is_err());
    }

    #[test]
    fn test_architecture_comparison() {
        let comparison = MultimodalUtils::compare_architectures(
            &MultimodalArchitecture::CLIP,
            &MultimodalArchitecture::BLIP,
        );
        assert!(comparison.contains("CLIP"));
        assert!(comparison.contains("BLIP"));
        assert!(comparison.contains("Zero-shot"));
        assert!(comparison.contains("vision-language understanding"));
    }

    #[test]
    fn test_family_info() {
        let info = MultimodalUtils::architecture_family_info();
        assert!(info.contains("Contrastive Learning"));
        assert!(info.contains("Unified Models"));
        assert!(info.contains("Generative Models"));
        assert!(info.contains("Few-Shot Learning"));
    }

    #[test]
    fn test_capability_matrix() {
        let matrix = MultimodalUtils::task_capability_matrix();
        assert!(matrix.contains("Task Capability Matrix"));
        assert!(matrix.contains("CLIP"));
        assert!(matrix.contains("✓"));
        assert!(matrix.contains("✗"));
    }
}