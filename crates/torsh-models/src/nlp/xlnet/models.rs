//! XLNet Models
//!
//! This module implements complete XLNet models for various tasks.

use crate::nlp::xlnet::config::XLNetConfig;
use crate::nlp::xlnet::embeddings::XLNetEmbeddings;
use crate::nlp::xlnet::layers::XLNetEncoder;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// XLNet base model
///
/// This is the base XLNet model that outputs hidden states.
pub struct XLNetModel {
    /// Embeddings
    embeddings: XLNetEmbeddings,
    /// Encoder
    encoder: XLNetEncoder,
    /// Configuration
    config: XLNetConfig,
}

impl XLNetModel {
    /// Create new XLNet model
    pub fn new(config: XLNetConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))?;

        Ok(Self {
            embeddings: XLNetEmbeddings::new(config.clone())?,
            encoder: XLNetEncoder::new(config.clone())?,
            config,
        })
    }

    /// Create XLNet-base model
    pub fn xlnet_base() -> Result<Self> {
        Self::new(XLNetConfig::xlnet_base())
    }

    /// Create XLNet-large model
    pub fn xlnet_large() -> Result<Self> {
        Self::new(XLNetConfig::xlnet_large())
    }

    /// Get configuration
    pub fn config(&self) -> &XLNetConfig {
        &self.config
    }
}

impl Module for XLNetModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Get embeddings
        let embeddings = self.embeddings.forward(input)?;

        // Pass through encoder
        self.encoder.forward(&embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.embeddings.training()
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        Ok(())
    }
}

/// XLNet for sequence classification
///
/// This model adds a classification head on top of the base XLNet model.
pub struct XLNetForSequenceClassification {
    /// Base model
    xlnet: XLNetModel,
    /// Classification head
    classifier: Linear,
    /// Dropout
    dropout: Dropout,
    /// Configuration
    config: XLNetConfig,
}

impl XLNetForSequenceClassification {
    /// Create new XLNet for sequence classification
    pub fn new(config: XLNetConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))?;

        let num_labels = config.num_labels.ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument(
                "num_labels must be specified for sequence classification".to_string(),
            )
        })?;

        Ok(Self {
            xlnet: XLNetModel::new(config.clone())?,
            classifier: Linear::new(config.hidden_size, num_labels, true),
            dropout: Dropout::new(config.hidden_dropout_prob),
            config,
        })
    }

    /// Create XLNet-base for sequence classification
    pub fn xlnet_base_for_classification(num_labels: usize) -> Result<Self> {
        Self::new(XLNetConfig::xlnet_base_for_classification(num_labels))
    }

    /// Create XLNet-large for sequence classification
    pub fn xlnet_large_for_classification(num_labels: usize) -> Result<Self> {
        let mut config = XLNetConfig::xlnet_large();
        config.num_labels = Some(num_labels);
        Self::new(config)
    }

    /// Get configuration
    pub fn config(&self) -> &XLNetConfig {
        &self.config
    }
}

impl Module for XLNetForSequenceClassification {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Get base model output
        let hidden_states = self.xlnet.forward(input)?;

        // Take the first token's hidden state (similar to BERT's [CLS] token)
        let pooled_output = hidden_states.select(1, 0)?;

        // Apply dropout
        let pooled_output = self.dropout.forward(&pooled_output)?;

        // Classification
        self.classifier.forward(&pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.xlnet.parameters() {
            params.insert(format!("xlnet.{}", name), param);
        }
        for (name, param) in self.classifier.parameters() {
            params.insert(format!("classifier.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.xlnet.training()
    }

    fn train(&mut self) {
        self.xlnet.train();
        self.classifier.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.xlnet.eval();
        self.classifier.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.xlnet.to_device(device)?;
        self.classifier.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xlnet_model_creation() {
        let config = XLNetConfig::xlnet_base();
        let model = XLNetModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_xlnet_base_factory() {
        let model = XLNetModel::xlnet_base();
        assert!(model.is_ok());
    }

    #[test]
    fn test_xlnet_for_classification_creation() {
        let config = XLNetConfig::xlnet_base_for_classification(2);
        let model = XLNetForSequenceClassification::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_xlnet_classification_factory() {
        let model = XLNetForSequenceClassification::xlnet_base_for_classification(2);
        assert!(model.is_ok());
    }

    #[test]
    fn test_xlnet_parameters() {
        let model = XLNetModel::xlnet_base().unwrap();
        let params = model.parameters();
        assert!(!params.is_empty());
        assert!(params.contains_key("embeddings.word_embeddings.weight"));
    }
}
