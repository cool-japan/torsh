//! BigBird Models

use crate::nlp::bigbird::config::BigBirdConfig;
use crate::nlp::bigbird::embeddings::BigBirdEmbeddings;
use crate::nlp::bigbird::layers::BigBirdEncoder;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

pub struct BigBirdModel {
    embeddings: BigBirdEmbeddings,
    encoder: BigBirdEncoder,
    config: BigBirdConfig,
}

impl BigBirdModel {
    pub fn new(config: BigBirdConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))?;
        Ok(Self {
            embeddings: BigBirdEmbeddings::new(config.clone())?,
            encoder: BigBirdEncoder::new(config.clone())?,
            config,
        })
    }

    pub fn bigbird_base() -> Result<Self> {
        Self::new(BigBirdConfig::bigbird_base())
    }

    pub fn bigbird_large() -> Result<Self> {
        Self::new(BigBirdConfig::bigbird_large())
    }

    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

impl Module for BigBirdModel {
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

pub struct BigBirdForSequenceClassification {
    bigbird: BigBirdModel,
    classifier: Linear,
    dropout: Dropout,
    config: BigBirdConfig,
}

impl BigBirdForSequenceClassification {
    pub fn new(config: BigBirdConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))?;
        let num_labels = config.num_labels.ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument(
                "num_labels must be specified for sequence classification".to_string(),
            )
        })?;

        Ok(Self {
            bigbird: BigBirdModel::new(config.clone())?,
            classifier: Linear::new(config.hidden_size, num_labels, true),
            dropout: Dropout::new(config.hidden_dropout_prob),
            config,
        })
    }

    pub fn bigbird_base_for_classification(num_labels: usize) -> Result<Self> {
        Self::new(BigBirdConfig::bigbird_base_for_classification(num_labels))
    }

    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

impl Module for BigBirdForSequenceClassification {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_states = self.bigbird.forward(input)?;
        let pooled_output = hidden_states.select(1, 0)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.bigbird.parameters() {
            params.insert(format!("bigbird.{}", name), param);
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
        self.bigbird.training()
    }

    fn train(&mut self) {
        self.bigbird.train();
        self.classifier.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.bigbird.eval();
        self.classifier.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.bigbird.to_device(device)?;
        self.classifier.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigbird_model_creation() {
        let model = BigBirdModel::bigbird_base();
        assert!(model.is_ok());
    }

    #[test]
    fn test_bigbird_classification_creation() {
        let model = BigBirdForSequenceClassification::bigbird_base_for_classification(2);
        assert!(model.is_ok());
    }
}
