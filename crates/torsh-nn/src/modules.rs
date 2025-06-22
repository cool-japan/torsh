//! Common neural network modules
//!
//! This module re-exports all neural network layers from the organized `layers` module.
//! The layers have been split into separate files for better maintainability and to comply
//! with the 2000-line file size limit.

// Re-export all layers from the layers module
pub use crate::layers::*;

// For backward compatibility, also re-export specific layer types
pub use crate::layers::{
    AdaptiveAvgPool2d,
    AvgPool2d,
    // Normalization layers
    BatchNorm2d,
    // Convolutional layers
    Conv1d,
    Conv2d,
    // Regularization layers
    Dropout,
    // Embedding layers
    Embedding,
    LayerNorm,
    LeakyReLU,
    // Linear layers
    Linear,
    LogSoftmax,
    // Pooling layers
    MaxPool2d,
    // Attention layers
    MultiheadAttention,
    // Activation functions
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    GELU,
    GRU,
    LSTM,
    // Recurrent layers
    RNN,
};
