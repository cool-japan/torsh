//! Neural network layer modules
//!
//! This module contains all the neural network layers organized by functionality:
//! - `activation`: Activation function layers (ReLU, Sigmoid, Tanh, GELU, etc.)
//! - `attention`: Attention mechanism layers (MultiheadAttention)
//! - `blocks`: Pre-built blocks (ResNet blocks, DenseNet blocks, SE blocks, MBConv blocks)
//! - `conv`: Convolutional layers (Conv1d, Conv2d, Conv3d)
//! - `embedding`: Embedding layers
//! - `lazy`: Lazy initialization layers (LazyLinear, LazyConv1d, LazyConv2d)
//! - `linear`: Linear/fully connected layers
//! - `normalization`: Normalization layers (BatchNorm2d, BatchRenorm2d, SwitchableNorm2d, LayerNorm, GroupNorm, InstanceNorm, SpectralNorm)
//! - `pooling`: Pooling layers (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d)
//! - `recurrent`: Recurrent layers (RNN, LSTM, GRU)
//! - `regularization`: Regularization layers (Dropout)
//! - `transformer`: Transformer layers (TransformerEncoder, TransformerEncoderLayer, Transformer)
//! - `upsampling`: Upsampling layers (PixelShuffle, PixelUnshuffle)

pub mod activation;
pub mod advanced;
pub mod attention;
pub mod blocks;
pub mod conv;
pub mod efficientnet;
pub mod embedding;
pub mod lazy;
pub mod linear;
pub mod mobilenet;
pub mod normalization;
pub mod pooling;
pub mod recurrent;
pub mod regularization;
pub mod transformer;
pub mod upsampling;

// Re-export all layer types for convenience
pub use activation::*;
pub use advanced::*;
pub use attention::*;
pub use blocks::*;
pub use conv::*;
pub use efficientnet::*;
pub use embedding::*;
pub use lazy::*;
pub use linear::*;
pub use mobilenet::*;
pub use normalization::*;
pub use pooling::*;
pub use recurrent::*;
pub use regularization::*;
pub use transformer::*;
pub use upsampling::*;
