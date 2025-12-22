//! Pre-trained models and model zoo for ToRSh deep learning framework
//!
//! This crate provides a comprehensive collection of pre-trained models and utilities
//! for loading, using, and managing deep learning models in ToRSh.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]

pub mod architectures;
pub mod audio;
pub mod benchmark;
pub mod builder;
pub mod common;
pub mod comparison;
pub mod config;
pub mod distillation;
pub mod domain;
pub mod downloader;
pub mod ensembling;
pub mod few_shot;
pub mod fine_tuning;
pub mod generative;
pub mod gnn;
pub mod lazy_loading;
pub mod model_merging;
pub mod model_sharding;
pub mod multimodal;
pub mod nlp;
pub mod optimization;
// pub mod prelude; // Defined inline below
pub mod pruning;
pub mod quantization;
pub mod registry;
pub mod rl;
pub mod surgery;
pub mod utils;
pub mod validation;
pub mod video;
pub mod vision;
pub mod vision_3d;

// Re-exports
pub use downloader::{DownloadProgress, ModelDownloader};
pub use lazy_loading::{CacheStats, LazyModelLoader, LazyTensor, StreamingModelLoader};
pub use model_merging::{LoRAMerger, MergeStrategy, ModelMerger, ModelSoup};
pub use model_sharding::{DevicePlacement, ModelSharder, ShardingStats, ShardingStrategy};
pub use registry::{ModelHandle, ModelInfo, ModelRegistry};
pub use utils::{
    convert_model_format, convert_pytorch_state_dict, convert_to_pytorch_state_dict,
    load_model_from_file, load_model_weights, load_pytorch_checkpoint, load_safetensors_weights,
    load_state_dict, map_parameter_names, save_model_to_file, save_pytorch_checkpoint,
    save_tensors_to_safetensors, ModelFormat, ModelMetadata,
};

/// Common error types
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {name}")]
    ModelNotFound { name: String },

    #[error("Download failed: {reason}")]
    DownloadFailed { reason: String },

    #[error("Invalid model format: {format}")]
    InvalidFormat { format: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] safetensors::SafeTensorError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    #[cfg(feature = "download")]
    Network(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Model loading error: {reason}")]
    LoadingError { reason: String },

    #[error("Model validation error: {reason}")]
    ValidationError { reason: String },

    #[error("ToRSh error: {0}")]
    TorshError(#[from] torsh_core::error::TorshError),
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Macro to define model types and generate implementations
macro_rules! define_model_type {
    (
        $(
            $(#[cfg(feature = $feature:literal)])?
            $variant:ident($type:ty),
        )*
    ) => {
        /// Concrete model enum to avoid trait object issues
        pub enum ModelType {
            $(
                $(#[cfg(feature = $feature)])?
                $variant($type),
            )*
        }

        impl torsh_nn::Module for ModelType {
            fn forward(&self, input: &torsh_tensor::Tensor) -> torsh_core::error::Result<torsh_tensor::Tensor> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.forward(input),
                    )*
                }
            }

            fn parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.parameters(),
                    )*
                }
            }

            fn named_parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.named_parameters(),
                    )*
                }
            }

            fn training(&self) -> bool {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.training(),
                    )*
                }
            }

            fn train(&mut self) {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.train(),
                    )*
                }
            }

            fn eval(&mut self) {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.eval(),
                    )*
                }
            }

            fn to_device(&mut self, device: torsh_core::DeviceType) -> torsh_core::error::Result<()> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        ModelType::$variant(model) => model.to_device(device),
                    )*
                }
            }
        }
    };
}

// Define all model types using the macro
define_model_type! {
    #[cfg(feature = "vision")]
    ResNet(crate::vision::ResNet),
    #[cfg(feature = "vision")]
    VisionTransformer(crate::vision::VisionTransformer),
    // NOTE: Additional vision models exist but require API updates for torsh-nn compatibility:
    // - EfficientNet, SwinTransformer, ConvNeXt (implemented, needs torsh-nn v0.2 API)
    // - DETR, MaskRCNN, YOLO (implemented, needs torsh-nn v0.2 API)
    // - MobileNetV2, DenseNet (implemented, needs torsh-nn v0.2 API)
    // These will be enabled in a future release once API compatibility is resolved
    // NOTE: Additional model types planned for v0.2.0:
    // - NLP: RoBERTa, BART, T5, GPT-2, XLNet, ELECTRA, DeBERTa, Longformer, BigBird
    // - Audio: Wav2Vec2, Whisper, HuBERT, WavLM (base implementations exist)
    // - Multimodal: CLIP, ALIGN (base implementations exist), Flamingo, DALL-E, BLIP, LLaVA, InstructBLIP
    // - GNN: GCN, GraphSAGE, GAT, GIN
    // These require module completion and/or API compatibility updates
    // #[cfg(feature = "vision_3d")]
    // CNN3D(crate::vision_3d::CNN3D),
    // #[cfg(feature = "vision_3d")]
    // PointNet(crate::vision_3d::PointNet),
    // #[cfg(feature = "vision_3d")]
    // PointNetPlusPlus(crate::vision_3d::PointNetPlusPlus),
    // #[cfg(feature = "video")]
    // ResNet3D(crate::video::ResNet3D),
    // #[cfg(feature = "video")]
    // SlowFast(crate::video::SlowFast),
    // #[cfg(feature = "video")]
    // VideoTransformer(crate::video::VideoTransformer),
    // #[cfg(feature = "generative")]
    // VAE(crate::generative::VAE),
    // #[cfg(feature = "generative")]
    // GAN(crate::generative::GAN),
    // #[cfg(feature = "generative")]
    // DiffusionModel(crate::generative::DiffusionUNet),
    // #[cfg(feature = "rl")]
    // DQN(crate::rl::DQN),
    // #[cfg(feature = "rl")]
    // PPO(crate::rl::PPO),
    // #[cfg(feature = "rl")]
    // A3C(crate::rl::A3C),
    // #[cfg(feature = "domain")]
    // UNet(crate::domain::UNet),
    // #[cfg(feature = "domain")]
    // UNet3D(crate::domain::UNet3D),
    // #[cfg(feature = "domain")]
    // PINN(crate::domain::PINN),
    // #[cfg(feature = "domain")]
    // FNO(crate::domain::FNO),
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        convert_model_format, convert_pytorch_state_dict, convert_to_pytorch_state_dict,
        load_model_from_file, load_model_weights, load_pytorch_checkpoint,
        load_safetensors_weights, load_state_dict, map_parameter_names, save_model_to_file,
        save_pytorch_checkpoint, save_tensors_to_safetensors, DownloadProgress, ModelDownloader,
        ModelError, ModelFormat, ModelHandle, ModelInfo, ModelMetadata, ModelRegistry, ModelResult,
        ModelType,
    };

    pub use crate::comparison::*;
    pub use crate::distillation::*;
    pub use crate::ensembling::*;
    pub use crate::few_shot::*;
    pub use crate::fine_tuning::*;
    pub use crate::pruning::*;
    pub use crate::quantization::*;
    pub use crate::surgery::*;
    pub use crate::validation::*;

    #[cfg(feature = "vision")]
    pub use crate::vision::*;

    #[cfg(feature = "nlp")]
    pub use crate::nlp::*;

    #[cfg(feature = "audio")]
    pub use crate::audio::*;

    #[cfg(feature = "multimodal")]
    pub use crate::multimodal::*;

    #[cfg(feature = "gnn")]
    pub use crate::gnn::*;

    #[cfg(feature = "vision_3d")]
    pub use crate::vision_3d::*;

    #[cfg(feature = "video")]
    pub use crate::video::*;

    #[cfg(feature = "generative")]
    pub use crate::generative::*;

    #[cfg(feature = "rl")]
    pub use crate::rl::*;

    #[cfg(feature = "domain")]
    pub use crate::domain::*;
}
