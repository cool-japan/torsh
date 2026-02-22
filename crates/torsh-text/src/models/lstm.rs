use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_core::Result;
use torsh_nn::{prelude::*, Module, Parameter};
use torsh_tensor::Tensor;

use super::{TextModel, TextModelConfig};

#[derive(Debug)]
pub struct LSTMTextModel {
    pub name: String,
    pub config: TextModelConfig,
    pub embedding: Embedding,
    pub lstm: LSTM,
    pub dropout: Dropout,
    pub classifier: Linear,
    pub device: DeviceType,
}

impl LSTMTextModel {
    pub fn new(config: TextModelConfig, device: DeviceType) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim);
        let lstm = LSTM::new(config.hidden_dim, config.hidden_dim, config.num_layers);
        let dropout = Dropout::new(config.dropout);
        let classifier = Linear::new(config.hidden_dim, config.vocab_size, true);

        Ok(Self {
            name: "LSTM".to_string(),
            config,
            embedding,
            lstm,
            dropout,
            classifier,
            device,
        })
    }

    pub fn from_config(config: TextModelConfig) -> Result<Self> {
        Self::new(config, DeviceType::Cpu)
    }

    pub fn for_classification(
        config: TextModelConfig,
        num_classes: usize,
        device: DeviceType,
    ) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim);
        let lstm = LSTM::new(config.hidden_dim, config.hidden_dim, config.num_layers);
        let dropout = Dropout::new(config.dropout);
        let classifier = Linear::new(config.hidden_dim, num_classes, true);

        Ok(Self {
            name: "LSTM-Classifier".to_string(),
            config,
            embedding,
            lstm,
            dropout,
            classifier,
            device,
        })
    }
}

impl Module for LSTMTextModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input shape: [batch_size, seq_len]
        let embedded = self.embedding.forward(input)?; // [batch_size, seq_len, hidden_dim]
        let lstm_out = self.lstm.forward(&embedded)?; // [batch_size, seq_len, hidden_dim]
        let dropped = self.dropout.forward(&lstm_out)?;
        let output = self.classifier.forward(&dropped)?; // [batch_size, seq_len, vocab_size]
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embedding.parameters() {
            params.insert(format!("embedding.{}", name), param);
        }
        for (name, param) in self.lstm.parameters() {
            params.insert(format!("lstm.{}", name), param);
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
        true // Simplified implementation
    }

    fn train(&mut self) {
        // Simplified implementation - individual layers are not mutable here
    }

    fn eval(&mut self) {
        // Simplified implementation - individual layers are not mutable here
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.device = device;
        Ok(())
    }
}

impl TextModel for LSTMTextModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn max_seq_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

/// Bidirectional LSTM for text modeling
#[derive(Debug)]
pub struct BiLSTMTextModel {
    pub name: String,
    pub config: TextModelConfig,
    pub embedding: Embedding,
    pub lstm: LSTM,
    pub dropout: Dropout,
    pub classifier: Linear,
    pub device: DeviceType,
}

impl BiLSTMTextModel {
    pub fn new(config: TextModelConfig, device: DeviceType) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim);
        // Bidirectional LSTM has 2x hidden dim output
        let lstm = LSTM::new(config.hidden_dim, config.hidden_dim, config.num_layers);
        let dropout = Dropout::new(config.dropout);
        let classifier = Linear::new(config.hidden_dim * 2, config.vocab_size, true);

        Ok(Self {
            name: "BiLSTM".to_string(),
            config,
            embedding,
            lstm,
            dropout,
            classifier,
            device,
        })
    }

    pub fn for_classification(
        config: TextModelConfig,
        num_classes: usize,
        device: DeviceType,
    ) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim);
        let lstm = LSTM::new(config.hidden_dim, config.hidden_dim, config.num_layers);
        let dropout = Dropout::new(config.dropout);
        let classifier = Linear::new(config.hidden_dim * 2, num_classes, true);

        Ok(Self {
            name: "BiLSTM-Classifier".to_string(),
            config,
            embedding,
            lstm,
            dropout,
            classifier,
            device,
        })
    }
}

impl Module for BiLSTMTextModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input shape: [batch_size, seq_len]
        let embedded = self.embedding.forward(input)?; // [batch_size, seq_len, hidden_dim]

        // Forward pass: process sequence from left to right
        let forward_out = self.lstm.forward(&embedded)?;

        // Backward pass: reverse the sequence, process, then reverse back
        let seq_len = embedded.size(1)? as i64;
        let mut reversed_embedded_data = Vec::new();

        // Reverse the sequence dimension (dim=1)
        for i in (0..seq_len).rev() {
            let slice = embedded.narrow(1, i, 1)?;
            reversed_embedded_data.push(slice);
        }

        // Concatenate reversed slices back together
        let mut reversed_embedded = reversed_embedded_data[0].clone();
        for slice in &reversed_embedded_data[1..] {
            reversed_embedded = reversed_embedded.cat(slice, 1)?;
        }

        // Process reversed sequence
        let backward_out_reversed = self.lstm.forward(&reversed_embedded)?;

        // Reverse the output back to original order
        let mut backward_out_data = Vec::new();
        for i in (0..seq_len).rev() {
            let slice = backward_out_reversed.narrow(1, i, 1)?;
            backward_out_data.push(slice);
        }

        let mut backward_out = backward_out_data[0].clone();
        for slice in &backward_out_data[1..] {
            backward_out = backward_out.cat(slice, 1)?;
        }

        // Concatenate forward and backward outputs along the hidden dimension
        let concat_out = forward_out.cat(&backward_out, -1)?;
        let dropped = self.dropout.forward(&concat_out)?;
        let output = self.classifier.forward(&dropped)?;
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embedding.parameters() {
            params.insert(format!("embedding.{}", name), param);
        }
        for (name, param) in self.lstm.parameters() {
            params.insert(format!("lstm.{}", name), param);
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
        true // Simplified implementation
    }

    fn train(&mut self) {
        // Simplified implementation - individual layers are not mutable here
    }

    fn eval(&mut self) {
        // Simplified implementation - individual layers are not mutable here
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.device = device;
        Ok(())
    }
}

impl TextModel for BiLSTMTextModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim * 2 // Bidirectional doubles the hidden dim
    }

    fn max_seq_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}
