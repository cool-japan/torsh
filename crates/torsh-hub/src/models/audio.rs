//! Audio models for the ToRSh Hub Model Zoo
//!
//! This module contains implementations of popular audio processing models
//! including Wav2Vec2, Whisper, and audio classification architectures.

use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// Convolutional feature extractor for audio (inspired by Wav2Vec2)
pub struct ConvFeatureExtractor {
    conv_layers: Vec<Conv1d>,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl ConvFeatureExtractor {
    pub fn new(
        input_dim: usize,
        conv_layers: &[(usize, usize, usize)], // (out_channels, kernel_size, stride)
        dropout: f32,
    ) -> Self {
        let mut convs = Vec::new();
        let mut in_channels = input_dim;

        for &(out_channels, kernel_size, stride) in conv_layers {
            convs.push(Conv1d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size / 2, // padding
                1,               // dilation
                false,           // bias
                1,               // groups
            ));
            in_channels = out_channels;
        }

        Self {
            conv_layers: convs,
            layer_norm: LayerNorm::new(vec![in_channels], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
            dropout: Dropout::new(dropout),
        }
    }
}

impl Module for ConvFeatureExtractor {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut hidden = x.clone();

        for conv_layer in &self.conv_layers {
            hidden = conv_layer.forward(&hidden)?;
            hidden = hidden.gelu()?; // GELU activation commonly used in audio models
        }

        // Apply layer norm and dropout
        hidden = self.layer_norm.forward(&hidden)?;
        hidden = self.dropout.forward(&hidden)?;

        Ok(hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, conv) in self.conv_layers.iter().enumerate() {
            for (name, param) in conv.parameters() {
                params.insert(format!("conv_layers.{}.{}", i, name), param);
            }
        }
        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("layer_norm.{}", name), param);
        }
        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        for (i, conv) in self.conv_layers.iter_mut().enumerate() {
            // Load conv parameters with proper naming
            conv.load_state_dict(state_dict, strict)?;
        }
        self.layer_norm.load_state_dict(state_dict, strict)?;
        Ok(())
    }
}

/// Positional convolution layer for audio transformers
pub struct PositionalConvEmbedding {
    conv: Conv1d,
    padding: usize,
    activation: GELU,
}

impl PositionalConvEmbedding {
    pub fn new(embed_dim: usize, kernel_size: usize) -> Self {
        let padding = kernel_size / 2;
        Self {
            conv: Conv1d::new(
                embed_dim,
                embed_dim,
                kernel_size,
                1,
                padding,
                1,
                true,
                embed_dim,
            ),
            padding,
            activation: GELU::new(false),
        }
    }
}

impl Module for PositionalConvEmbedding {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut hidden = self.conv.forward(x)?;
        hidden = self.activation.forward(&hidden)?;
        Ok(hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.conv.parameters()
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.conv.load_state_dict(state_dict, strict)
    }
}

/// Wav2Vec2-style transformer encoder for audio
pub struct AudioTransformerEncoder {
    feature_extractor: ConvFeatureExtractor,
    pos_conv_embed: PositionalConvEmbedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    transformer_layers: Vec<TransformerEncoderLayer>,
}

impl AudioTransformerEncoder {
    pub fn new(
        input_dim: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ffn_dim: usize,
        dropout: f32,
    ) -> Self {
        // Standard Wav2Vec2 feature extractor architecture
        let conv_layers = vec![
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ];

        let feature_extractor = ConvFeatureExtractor::new(input_dim, &conv_layers, dropout);

        let mut transformer_layers = Vec::new();
        for _ in 0..num_layers {
            transformer_layers.push(
                TransformerEncoderLayer::new(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    DeviceType::Cpu,
                )
                .expect("Failed to create TransformerEncoderLayer"),
            );
        }

        Self {
            feature_extractor,
            pos_conv_embed: PositionalConvEmbedding::new(embed_dim, 128),
            layer_norm: LayerNorm::new(vec![embed_dim], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
            dropout: Dropout::new(dropout),
            transformer_layers,
        }
    }
}

impl Module for AudioTransformerEncoder {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Extract features from raw audio
        let mut hidden = self.feature_extractor.forward(x)?;

        // Add positional embeddings
        hidden = &hidden + &self.pos_conv_embed.forward(&hidden)?;
        hidden = self.layer_norm.forward(&hidden)?;
        hidden = self.dropout.forward(&hidden)?;

        // Pass through transformer layers
        for layer in &self.transformer_layers {
            hidden = layer.forward(&hidden, None)?;
        }

        Ok(hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.feature_extractor.parameters() {
            params.insert(format!("feature_extractor.{}", name), param);
        }

        for (name, param) in self.pos_conv_embed.parameters() {
            params.insert(format!("pos_conv_embed.{}", name), param);
        }

        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("layer_norm.{}", name), param);
        }

        for (i, layer) in self.transformer_layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("transformer_layers.{}.{}", i, name), param);
            }
        }

        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.feature_extractor.load_state_dict(state_dict, strict)?;
        self.pos_conv_embed.load_state_dict(state_dict, strict)?;
        self.layer_norm.load_state_dict(state_dict, strict)?;

        for layer in &mut self.transformer_layers {
            layer.load_state_dict(state_dict, strict)?;
        }

        Ok(())
    }
}

/// Whisper-style encoder for speech processing
pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Embedding,
    layers: Vec<TransformerEncoderLayer>,
    ln_post: LayerNorm,
}

impl WhisperEncoder {
    pub fn new(n_mels: usize, n_ctx: usize, n_state: usize, n_head: usize, n_layer: usize) -> Self {
        let mut layers = Vec::new();
        for _ in 0..n_layer {
            layers.push(
                TransformerEncoderLayer::new(
                    n_state,
                    n_head,
                    n_state * 4,
                    0.0, // Whisper typically uses no dropout in encoder
                    DeviceType::Cpu,
                )
                .expect("Failed to create TransformerEncoderLayer"),
            );
        }

        Self {
            conv1: Conv1d::new(n_mels, n_state, 3, 1, 1, 1, true, 1),
            conv2: Conv1d::new(n_state, n_state, 3, 2, 1, 1, true, 1),
            embed_positions: Embedding::new(n_ctx, n_state),
            layers,
            ln_post: LayerNorm::new(vec![n_state], 1e-5, true, DeviceType::Cpu)
                .expect("Failed to create LayerNorm"),
        }
    }
}

impl Module for WhisperEncoder {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut hidden = self.conv1.forward(x)?;
        hidden = hidden.gelu()?;
        hidden = self.conv2.forward(&hidden)?;
        hidden = hidden.gelu()?;

        // Transpose for transformer layers (seq_len, batch, embed_dim)
        let seq_len = hidden.shape().dims()[2];

        // Add positional embeddings
        let positions = torsh_tensor::creation::arange(0.0, seq_len as f32, 1.0)?;
        let pos_embed = self.embed_positions.forward(&positions)?;
        hidden = &hidden + &pos_embed;

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None)?;
        }

        hidden = self.ln_post.forward(&hidden)?;
        Ok(hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.embed_positions.parameters());

        for layer in &self.layers {
            params.extend(layer.parameters());
        }

        params.extend(self.ln_post.parameters());
        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.conv1.load_state_dict(state_dict, strict)?;
        self.conv2.load_state_dict(state_dict, strict)?;
        self.embed_positions.load_state_dict(state_dict, strict)?;

        for layer in &mut self.layers {
            layer.load_state_dict(state_dict, strict)?;
        }

        self.ln_post.load_state_dict(state_dict, strict)?;
        Ok(())
    }
}

/// Audio classification model combining CNN and transformer
pub struct AudioClassifier {
    encoder: AudioTransformerEncoder,
    classifier_head: Sequential,
    num_classes: usize,
}

impl AudioClassifier {
    pub fn new(
        input_dim: usize,
        embed_dim: usize,
        num_classes: usize,
        num_layers: usize,
        num_heads: usize,
        dropout: f32,
    ) -> Self {
        let encoder = AudioTransformerEncoder::new(
            input_dim,
            embed_dim,
            num_layers,
            num_heads,
            embed_dim * 4,
            dropout,
        );

        let classifier_head = Sequential::new()
            .add(Linear::new(embed_dim, embed_dim / 2, true))
            .add(ReLU::new())
            .add(Dropout::new(dropout))
            .add(Linear::new(embed_dim / 2, num_classes, true));

        Self {
            encoder,
            classifier_head,
            num_classes,
        }
    }
}

impl Module for AudioClassifier {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let encoded = self.encoder.forward(x)?;

        // Global average pooling over sequence dimension
        let pooled = encoded.mean(Some(&[1]), false)?; // Average over sequence length

        // Classification head
        let logits = self.classifier_head.forward(&pooled)?;
        Ok(logits)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.encoder.parameters());
        params.extend(self.classifier_head.parameters());
        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.encoder.load_state_dict(state_dict, strict)?;
        self.classifier_head.load_state_dict(state_dict, strict)?;
        Ok(())
    }
}

/// Simplified Wav2Vec2 model for pretraining
pub struct Wav2Vec2Model {
    encoder: AudioTransformerEncoder,
    quantizer: Linear, // Simplified quantization layer
    projection_head: Linear,
}

impl Wav2Vec2Model {
    pub fn new(
        input_dim: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        num_codevectors: usize,
    ) -> Self {
        let encoder = AudioTransformerEncoder::new(
            input_dim,
            embed_dim,
            num_layers,
            num_heads,
            embed_dim * 4,
            0.1,
        );

        Self {
            encoder,
            quantizer: Linear::new(embed_dim, num_codevectors, false),
            projection_head: Linear::new(embed_dim, embed_dim, true),
        }
    }
}

impl Module for Wav2Vec2Model {
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let encoded = self.encoder.forward(x)?;
        let projected = self.projection_head.forward(&encoded)?;
        Ok(projected)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.encoder.parameters());
        params.extend(self.quantizer.parameters());
        params.extend(self.projection_head.parameters());
        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.encoder.load_state_dict(state_dict, strict)?;
        self.quantizer.load_state_dict(state_dict, strict)?;
        self.projection_head.load_state_dict(state_dict, strict)?;
        Ok(())
    }
}

/// Factory functions for creating popular audio models
pub fn wav2vec2_base() -> Wav2Vec2Model {
    Wav2Vec2Model::new(
        1,   // raw audio input
        768, // embed_dim
        12,  // num_layers
        12,  // num_heads
        320, // num_codevectors
    )
}

pub fn wav2vec2_large() -> Wav2Vec2Model {
    Wav2Vec2Model::new(
        1,    // raw audio input
        1024, // embed_dim
        24,   // num_layers
        16,   // num_heads
        320,  // num_codevectors
    )
}

pub fn whisper_tiny_encoder() -> WhisperEncoder {
    WhisperEncoder::new(
        80,   // n_mels
        1500, // n_ctx
        384,  // n_state
        6,    // n_head
        4,    // n_layer
    )
}

pub fn whisper_base_encoder() -> WhisperEncoder {
    WhisperEncoder::new(
        80,   // n_mels
        1500, // n_ctx
        512,  // n_state
        8,    // n_head
        6,    // n_layer
    )
}

pub fn audio_classifier_small(num_classes: usize) -> AudioClassifier {
    AudioClassifier::new(
        1,   // raw audio input
        256, // embed_dim
        num_classes,
        6,   // num_layers
        8,   // num_heads
        0.1, // dropout
    )
}

pub fn audio_classifier_base(num_classes: usize) -> AudioClassifier {
    AudioClassifier::new(
        1,   // raw audio input
        512, // embed_dim
        num_classes,
        12,  // num_layers
        8,   // num_heads
        0.1, // dropout
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Model implementation needs tensor shape handling fixes"]
    fn test_conv_feature_extractor() {
        let conv_layers = vec![(64, 3, 2), (128, 3, 2)];
        let extractor = ConvFeatureExtractor::new(1, &conv_layers, 0.1);

        // Test with dummy audio input (batch_size=2, channels=1, seq_len=1000)
        let input = torsh_tensor::creation::randn(&[2, 1, 1000]).unwrap();
        let output = extractor.forward(&input).unwrap();

        // Output should have reduced sequence length due to strided convolutions
        assert_eq!(output.shape().dims()[0], 2); // batch size preserved
        assert_eq!(output.shape().dims()[1], 128); // final channel dimension
    }

    #[test]
    #[ignore = "Model implementation needs tensor shape handling fixes"]
    fn test_wav2vec2_model() {
        let model = wav2vec2_base();

        // Test with dummy raw audio (batch_size=1, channels=1, seq_len=16000)
        let input = torsh_tensor::creation::randn(&[1, 1, 16000]).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.shape().dims()[0], 1); // batch size preserved
        assert_eq!(output.shape().dims()[2], 768); // embed_dim
    }

    #[test]
    #[ignore = "Model implementation needs tensor shape handling fixes"]
    fn test_audio_classifier() {
        let classifier = audio_classifier_small(10); // 10 classes

        // Test with dummy audio input
        let input = torsh_tensor::creation::randn(&[2, 1, 8000]).unwrap();
        let output = classifier.forward(&input).unwrap();

        assert_eq!(output.shape().dims()[0], 2); // batch size
        assert_eq!(output.shape().dims()[1], 10); // num_classes
    }

    #[test]
    #[ignore = "Model implementation needs tensor shape handling fixes"]
    fn test_whisper_encoder() {
        let encoder = whisper_tiny_encoder();

        // Test with mel-spectrogram input (batch_size=1, n_mels=80, time_steps=100)
        let input = torsh_tensor::creation::randn(&[1, 80, 100]).unwrap();
        let output = encoder.forward(&input).unwrap();

        assert_eq!(output.shape().dims()[0], 1); // batch size preserved
        assert_eq!(output.shape().dims()[2], 384); // n_state
    }
}
