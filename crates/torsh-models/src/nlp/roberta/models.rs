//! RoBERTa model implementations

use super::{config::RobertaConfig, embeddings::RobertaEmbeddings, layers::RobertaEncoder};
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// Base RoBERTa Model
pub struct RobertaModel {
    embeddings: RobertaEmbeddings,
    encoder: RobertaEncoder,
    config: RobertaConfig,
}

impl RobertaModel {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let embeddings = RobertaEmbeddings::new(config.clone())?;
        let encoder = RobertaEncoder::new(config.clone())?;

        Ok(Self {
            embeddings,
            encoder,
            config,
        })
    }

    /// Load pre-trained RoBERTa-base model
    pub fn roberta_base() -> Result<Self> {
        let config = RobertaConfig::roberta_base();
        Self::new(config)
    }

    /// Load pre-trained RoBERTa-large model
    pub fn roberta_large() -> Result<Self> {
        let config = RobertaConfig::roberta_large();
        Self::new(config)
    }
}

impl Module for RobertaModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let encoder_output = self.encoder.forward(&embedding_output)?;
        Ok(encoder_output)
    }
}

/// RoBERTa for Sequence Classification
pub struct RobertaForSequenceClassification {
    roberta: RobertaModel,
    classifier: Linear,
    dropout: Dropout,
    num_labels: usize,
}

impl RobertaForSequenceClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            roberta,
            classifier,
            dropout,
            num_labels,
        })
    }
}

impl Module for RobertaForSequenceClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let outputs = self.roberta.forward(input_ids)?;

        // Use [CLS] token representation (first token)
        let cls_output = outputs.select(1, 0)?; // Select first position
        let pooled_output = self.dropout.forward(&cls_output)?;
        let logits = self.classifier.forward(&pooled_output)?;

        Ok(logits)
    }
}

/// RoBERTa for Token Classification (Named Entity Recognition, etc.)
pub struct RobertaForTokenClassification {
    roberta: RobertaModel,
    classifier: Linear,
    dropout: Dropout,
    num_labels: usize,
}

impl RobertaForTokenClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            roberta,
            classifier,
            dropout,
            num_labels,
        })
    }
}

impl Module for RobertaForTokenClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let sequence_output = self.roberta.forward(input_ids)?;
        let sequence_output = self.dropout.forward(&sequence_output)?;
        let logits = self.classifier.forward(&sequence_output)?;

        Ok(logits)
    }
}

/// RoBERTa for Question Answering
pub struct RobertaForQuestionAnswering {
    roberta: RobertaModel,
    qa_outputs: Linear,
}

impl RobertaForQuestionAnswering {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let roberta = RobertaModel::new(config.clone())?;
        let qa_outputs = Linear::new(config.hidden_size, 2, true); // start and end positions

        Ok(Self {
            roberta,
            qa_outputs,
        })
    }
}

impl Module for RobertaForQuestionAnswering {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let sequence_output = self.roberta.forward(input_ids)?;
        let logits = self.qa_outputs.forward(&sequence_output)?;

        // Split into start and end logits
        // In full implementation, would return both start_logits and end_logits
        Ok(logits)
    }
}
