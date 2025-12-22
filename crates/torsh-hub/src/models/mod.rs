//! Model architectures for torsh-hub
//!
//! This module provides pre-built model architectures across different domains:
//! - **NLP**: Transformer-based models (BERT, GPT) for natural language processing
//! - **Vision**: CNN and Vision Transformer models (ResNet, ViT) for computer vision
//! - **Multimodal**: Cross-modal models (CLIP, Vision-Language) combining vision and language
//! - **Audio**: Models for audio processing and speech recognition (Wav2Vec2, WaveNet)
//! - **RL**: Reinforcement learning models (DQN, Actor-Critic, PPO)
//!
//! All models implement the `Module` trait from `torsh-nn`, providing consistent
//! interfaces for forward passes, parameter management, and state serialization.
//!
//! # SciRS2 POLICY Compliance
//!
//! All models in this module strictly follow the SciRS2 POLICY:
//! - Array operations: `scirs2_core::ndarray::*`
//! - Random generation: `scirs2_core::random::*`
//! - Numerical traits: `scirs2_core::numeric::*`
//! - NO direct imports of external dependencies (ndarray, rand, num-traits, etc.)
//!
//! # Examples
//!
//! ```no_run
//! use torsh_hub::models::nlp::{BertEncoder, MultiHeadAttention};
//! use torsh_hub::models::vision::ResNet;
//! use torsh_hub::models::rl::DQN;
//! use torsh_core::Device;
//!
//! // Use BERT encoder components
//! // let encoder = BertEncoder::new(...);
//!
//! // Use vision models
//! // let resnet = ResNet::new(...);
//!
//! // Use reinforcement learning models
//! // let dqn = DQN::new(...);
//! ```

pub mod audio;
pub mod multimodal;
pub mod nlp;
pub mod rl;
pub mod vision;

// Re-export commonly used types for convenience
pub use audio::*;
pub use multimodal::*;
pub use rl::*;

// Re-export main NLP types and keep pretrained module accessible
pub use nlp::pretrained as nlp_pretrained;
pub use nlp::{
    BertEmbeddings, BertEncoder, GPTDecoder, GPTEmbeddings, MultiHeadAttention, TransformerBlock,
};

// Re-export main vision types and keep pretrained module accessible
pub use vision::pretrained as vision_pretrained;
pub use vision::{BasicBlock, EfficientNet, ResNet, VisionTransformer};
