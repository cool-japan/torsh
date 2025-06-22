//! Pre-trained models and model zoo for ToRSh deep learning framework
//!
//! This crate provides a comprehensive collection of pre-trained models and utilities
//! for loading, using, and managing deep learning models in ToRSh.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod registry;
pub mod downloader;
pub mod vision;
pub mod nlp;
pub mod utils;

// Re-exports
pub use registry::{ModelRegistry, ModelInfo, ModelHandle};
pub use downloader::{ModelDownloader, DownloadProgress};
pub use utils::{load_model_from_file, save_model_to_file, ModelFormat};

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
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        ModelRegistry, ModelInfo, ModelHandle, ModelDownloader, DownloadProgress,
        load_model_from_file, save_model_to_file, ModelFormat, ModelError, ModelResult
    };
    
    #[cfg(feature = "vision")]
    pub use crate::vision::*;
    
    #[cfg(feature = "nlp")]
    pub use crate::nlp::*;
}