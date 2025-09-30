//! Prelude module for convenient imports
//!
//! This module re-exports the most commonly used types and functions from torsh-models
//! for easy importing with a single `use torsh_models::prelude::*;` statement.

// Core model registry and utilities
pub use crate::builder::ModelBuilder;
pub use crate::config::{ModelConfig, TrainingConfig, UnifiedModelRegistry};
pub use crate::downloader::{DownloadProgress, ModelDownloader};
pub use crate::registry::{ModelHandle, ModelInfo, ModelRegistry};

// Error types
pub use crate::ModelError;
pub use crate::ModelResult;

// Architecture-specific models
#[cfg(feature = "vision")]
pub use crate::vision::{
    ConvNeXt, ConvNeXtConfig, DETRConfig, EfficientNet, EfficientNetConfig, EfficientNetVariant,
    MaskRCNN, MaskRCNNConfig, ResNet, ResNetConfig, ResNetVariant, SwinTransformer,
    SwinTransformerConfig, VisionTransformer, VisionTransformerConfig, YOLOConfig, DETR, YOLO,
};

#[cfg(feature = "nlp")]
pub use crate::nlp::{
    BartConfig, BartForConditionalGeneration, BigBirdConfig, BigBirdForSequenceClassification,
    DebertaConfig, DebertaForSequenceClassification, ElectraConfig,
    ElectraForSequenceClassification, LongformerConfig, LongformerForSequenceClassification,
    RobertaConfig, RobertaForSequenceClassification, XLNetConfig, XLNetForSequenceClassification,
};

#[cfg(feature = "audio")]
pub use crate::audio::{
    AudioSceneClassifier, EmotionRecognitionClassifier, HuBERTConfig,
    HuBERTForSequenceClassification, MusicGenreClassifier, UrbanSoundClassifier, Wav2Vec2Config,
    Wav2Vec2ForCTC, WavLMConfig, WavLMForSequenceClassification, WhisperConfig,
    WhisperForConditionalGeneration,
};

#[cfg(feature = "multimodal")]
pub use crate::multimodal::{
    ALIGNConfig, ALIGNModel, BLIPConfig, BLIPModel, CLIPConfig, CLIPModel, CLIPTextConfig,
    CLIPVisionConfig, DallEConfig, DallEModel, FlamingoConfig, FlamingoModel, InstructBLIPConfig,
    InstructBLIPModel, LLaVAConfig, LLaVAModel,
};

#[cfg(feature = "gnn")]
pub use crate::gnn::{GATConfig, GCNConfig, GINConfig, GraphSAGE, GraphSAGEConfig, GAT, GCN, GIN};

#[cfg(feature = "vision_3d")]
pub use crate::vision_3d::{
    CNN3DConfig, PointNet, PointNetConfig, PointNetPlusPlus, PointNetPlusPlusConfig, CNN3D,
};

#[cfg(feature = "video")]
pub use crate::video::{
    ResNet3D, ResNet3DConfig, SlowFast, SlowFastConfig, VideoTransformer, VideoTransformerConfig,
};

#[cfg(feature = "generative")]
pub use crate::generative::{DiffusionConfig, DiffusionUNet, GANConfig, VAEConfig, GAN, VAE};

#[cfg(feature = "rl")]
pub use crate::rl::{A3CConfig, DQNConfig, PPOConfig, A3C, DQN, PPO};

#[cfg(feature = "domain")]
pub use crate::domain::{FNOConfig, PINNConfig, UNet, UNet3D, UNet3DConfig, UNetConfig, FNO, PINN};

// Advanced architectures and components
pub use crate::architectures::{
    ALiBiPositionEncoder, AdvancedAttentionConfig, AdvancedMultiHeadAttention,
    AttentionMaskProcessor, AttentionType, LearnedPositionEncoder, PositionEncodingType,
    RelativePositionEncoder, RotaryPositionEncoder, SinusoidalPositionEncoder, SparsePattern,
};

// Model utilities
pub use crate::benchmark::{BenchmarkConfig, BenchmarkResults, ModelBenchmark};
pub use crate::comparison::{
    ComparisonConfig, ComparisonDimension, ComparisonResults, ModelComparator, ModelSpec,
};
pub use crate::validation::{
    ModelValidator, ToleranceConfig, ValidationConfig, ValidationMetric, ValidationResults,
    ValidationStrategy,
};

// Optimization and performance
pub use crate::optimization::{
    CompressionConfig, ComputeOptimizationConfig, MemoryOptimizationConfig, ModelOptimizer,
    OptimizationConfig, OptimizationRecommendation, OptimizationResults, PerformanceTargets,
    PrecisionOptimizationConfig,
};

// Model compression and fine-tuning
pub use crate::distillation::{
    DistillationConfig, DistillationLoss, DistillationStrategy, KnowledgeDistiller,
};
pub use crate::fine_tuning::{
    AdapterConfig, FineTuningConfig, FreezingStrategy, LoRAConfig, ModelFineTuner,
    UnfreezingSchedule,
};
pub use crate::pruning::{
    ModelPruner, PruningConfig, PruningStrategy, PruningType, StructuredPruningType,
};
pub use crate::quantization::{
    ModelQuantizer, ObserverType, QuantizationConfig, QuantizationGranularity,
    QuantizationStrategy, QuantizationType,
};

// Ensemble and meta-learning
pub use crate::ensembling::{EnsembleMethod, EnsembleResults, EnsemblingConfig, ModelEnsemble};
pub use crate::few_shot::{FewShotConfig, FewShotLearner, FewShotStrategy, MetaLearningAlgorithm};

// Model surgery and modification
pub use crate::surgery::{
    AdapterLayer, LayerReplacement, LoRALayer, ModelComposition, ModelSurgery, SurgeryConfig,
};

// Common types and utilities
pub use crate::utils::{
    convert_model_format, load_model_from_file, load_model_weights, map_parameter_names,
    save_model_to_file, save_model_weights, ModelFormat, ModelMetadata,
};

// Device and tensor utilities (re-exported from core crates)
pub use torsh_core::{DType, DeviceType, Shape};
pub use torsh_nn::{Module, Parameter};
pub use torsh_tensor::Tensor;

// Common standard library types for convenience
pub use std::collections::HashMap;
pub use std::path::{Path, PathBuf};

/// Prelude trait for common model operations
pub trait ModelPrelude: Module {
    /// Get the model's parameter count
    fn parameter_count(&self) -> usize {
        self.parameters().len()
    }

    /// Get the model's memory usage estimate (simplified)
    fn memory_usage_bytes(&self) -> usize {
        self.parameter_count() * 4 // Assuming f32 parameters
    }

    /// Check if the model is trainable (has parameters)
    fn is_trainable(&self) -> bool {
        !self.parameters().is_empty()
    }

    /// Get model summary information
    fn summary(&self) -> ModelSummary {
        ModelSummary {
            parameter_count: self.parameter_count(),
            memory_usage_bytes: self.memory_usage_bytes(),
            is_trainable: self.is_trainable(),
            training_mode: self.training(),
        }
    }
}

// Implement the prelude trait for all modules
impl<T: Module> ModelPrelude for T {}

/// Model summary information
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub parameter_count: usize,
    pub memory_usage_bytes: usize,
    pub is_trainable: bool,
    pub training_mode: bool,
}

impl std::fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model Summary:\n\
             - Parameters: {}\n\
             - Memory Usage: {} bytes ({:.2} MB)\n\
             - Trainable: {}\n\
             - Training Mode: {}",
            self.parameter_count,
            self.memory_usage_bytes,
            self.memory_usage_bytes as f64 / 1024.0 / 1024.0,
            self.is_trainable,
            self.training_mode
        )
    }
}

/// Helper macros for common operations
#[macro_export]
macro_rules! model_forward {
    ($model:expr, $input:expr) => {
        $model
            .forward($input)
            .map_err(|e| $crate::ModelError::LoadingError {
                reason: format!("Forward pass failed: {}", e),
            })
    };
}

#[macro_export]
macro_rules! load_pretrained {
    ($model_name:expr) => {{
        let registry = $crate::prelude::ModelRegistry::new();
        registry.get_model($model_name)
    }};
}

#[macro_export]
macro_rules! benchmark_model {
    ($model:expr, $config:expr) => {{
        let mut benchmarker = $crate::prelude::ModelBenchmark::new($config);
        benchmarker.benchmark($model)
    }};
}

pub use benchmark_model;
pub use load_pretrained;
/// Re-export common macros
pub use model_forward;
