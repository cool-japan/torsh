//! Common neural network modules
//!
//! This module re-exports all neural network layers from the organized `layers` module.
//! The layers have been split into separate files for better maintainability and to comply
//! with the 2000-line file size limit.

// Re-export all layers from the layers module
pub use crate::layers::*;

// For backward compatibility, also re-export specific layer types
pub use crate::layers::{
    // Linear layers
    Linear,
    // Convolutional layers
    Conv1d, Conv2d,
    // Activation functions
    ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax, LogSoftmax,
    // Pooling layers
    MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
    // Normalization layers
    BatchNorm2d, LayerNorm,
    // Regularization layers
    Dropout,
    // Recurrent layers
    RNN, LSTM, GRU,
    // Attention layers
    MultiheadAttention,
    // Embedding layers
    Embedding,
};