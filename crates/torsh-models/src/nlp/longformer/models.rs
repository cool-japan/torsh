//! Longformer Models

use crate::nlp::longformer::config::LongformerConfig;
use crate::nlp::longformer::embeddings::LongformerEmbeddings;
use crate::nlp::longformer::layers::LongformerEncoder;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// Longformer base model
pub struct LongformerModel {
    embeddings: LongformerEmbeddings,
    encoder: LongformerEncoder,
    config: LongformerConfig,
}

impl LongformerModel {
    pub fn new(config: LongformerConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))?;
        Ok(Self {
            embeddings: LongformerEmbeddings::new(config.clone())?,
            encoder: LongformerEncoder::new(config.clone())?,
            config,
        })
    }

    pub fn longformer_base() -> Result<Self> {
        Self::new(LongformerConfig::longformer_base())
    }

    pub fn longformer_large() -> Result<Self> {
        Self::new(LongformerConfig::longformer_large())
    }

    pub fn config(&self) -> &LongformerConfig {
        &self.config
    }
}

impl Module for LongformerModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(input)?;
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

/// Longformer for sequence classification
pub struct LongformerForSequenceClassification {
    longformer: LongformerModel,
    classifier: Linear,
    dropout: Dropout,
    config: LongformerConfig,
}

impl LongformerForSequenceClassification {
    pub fn new(config: LongformerConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))?;
        let num_labels = config.num_labels.ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument(
                "num_labels must be specified for sequence classification".to_string(),
            )
        })?;

        Ok(Self {
            longformer: LongformerModel::new(config.clone())?,
            classifier: Linear::new(config.hidden_size, num_labels, true),
            dropout: Dropout::new(config.hidden_dropout_prob),
            config,
        })
    }

    pub fn longformer_base_for_classification(num_labels: usize) -> Result<Self> {
        Self::new(LongformerConfig::longformer_base_for_classification(
            num_labels,
        ))
    }

    pub fn config(&self) -> &LongformerConfig {
        &self.config
    }
}

impl Module for LongformerForSequenceClassification {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_states = self.longformer.forward(input)?;
        let pooled_output = hidden_states.select(1, 0)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.longformer.parameters() {
            params.insert(format!("longformer.{}", name), param);
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
        self.longformer.training()
    }

    fn train(&mut self) {
        self.longformer.train();
        self.classifier.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.longformer.eval();
        self.classifier.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.longformer.to_device(device)?;
        self.classifier.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longformer_model_creation() {
        let model = LongformerModel::longformer_base();
        assert!(model.is_ok());
    }

    #[test]
    fn test_longformer_classification_creation() {
        let model = LongformerForSequenceClassification::longformer_base_for_classification(2);
        assert!(model.is_ok());
    }
}
