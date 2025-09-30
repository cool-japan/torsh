//! Pre-built model zoo for ToRSh Hub
//!
//! This module contains implementations of popular model architectures
//! that can be easily loaded and used within the ToRSh ecosystem.

pub mod audio;
pub mod multimodal;
pub mod nlp;
pub mod rl;
pub mod vision;

// Re-exports from sub-modules
pub use audio::*;
pub use multimodal::*;
// Re-export main NLP types and keep pretrained module accessible
pub use nlp::pretrained as nlp_pretrained;
pub use nlp::{
    BertEmbeddings, BertEncoder, GPTDecoder, GPTEmbeddings, MultiHeadAttention, TransformerBlock,
};
pub use rl::*;
// Re-export main vision types and keep pretrained module accessible
pub use vision::pretrained as vision_pretrained;
pub use vision::{BasicBlock, EfficientNet, ResNet, VisionTransformer};
