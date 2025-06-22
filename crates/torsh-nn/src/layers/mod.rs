//! Neural network layer modules
//!
//! This module contains all the neural network layers organized by functionality:
//! - `activation`: Activation function layers (ReLU, Sigmoid, Tanh, GELU, etc.)
//! - `attention`: Attention mechanism layers (MultiheadAttention)
//! - `conv`: Convolutional layers (Conv1d, Conv2d)
//! - `embedding`: Embedding layers
//! - `linear`: Linear/fully connected layers
//! - `normalization`: Normalization layers (BatchNorm2d, LayerNorm)
//! - `pooling`: Pooling layers (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d)
//! - `recurrent`: Recurrent layers (RNN, LSTM, GRU)
//! - `regularization`: Regularization layers (Dropout)

pub mod activation;
pub mod attention;
pub mod conv;
pub mod embedding;
pub mod linear;
pub mod normalization;
pub mod pooling;
pub mod recurrent;
pub mod regularization;

// Re-export all layer types for convenience
pub use activation::*;
pub use attention::*;
pub use conv::*;
pub use embedding::*;
pub use linear::*;
pub use normalization::*;
pub use pooling::*;
pub use recurrent::*;
pub use regularization::*;
