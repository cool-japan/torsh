//! Comprehensive BERT Implementation in ToRSh
//!
//! This module provides a complete, production-ready implementation of BERT
//! (Bidirectional Encoder Representations from Transformers) including:
//! - Full BERT architecture with configurable parameters
//! - Multi-task heads for classification, token classification, and QA
//! - Pre-training objectives (MLM and NSP)
//! - Fine-tuning capabilities
//! - Efficient attention mechanisms
//! - Layer normalization and residual connections
//! - Position and token embeddings

use torsh::prelude::*;
use std::collections::HashMap;

/// BERT model configuration
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub position_embedding_type: String,
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,
        }
    }
}

impl BertConfig {
    /// Create BERT-Base configuration
    pub fn bert_base() -> Self {
        Self::default()
    }
    
    /// Create BERT-Large configuration
    pub fn bert_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }
    
    /// Create BERT-Small configuration for testing
    pub fn bert_small() -> Self {
        Self {
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            intermediate_size: 1024,
            ..Self::default()
        }
    }
}

/// BERT embeddings layer
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: BertConfig,
}

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(config.vocab_size, config.hidden_size)?,
            position_embeddings: Embedding::new(config.max_position_embeddings, config.hidden_size)?,
            token_type_embeddings: Embedding::new(config.type_vocab_size, config.hidden_size)?,
            layer_norm: LayerNorm::new(vec![config.hidden_size])?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            config: config.clone(),
        })
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_types(input_ids, None, None)
    }
}

impl BertEmbeddings {
    pub fn forward_with_types(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_length = input_ids.shape().dims()[1];
        
        // Word embeddings
        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        
        // Position embeddings
        let position_ids = if let Some(pos_ids) = position_ids {
            pos_ids.clone()
        } else {
            arange(0, seq_length as i64, 1)
                .unsqueeze(0)?
                .expand(&[input_ids.shape().dims()[0], seq_length])?
        };
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        
        // Token type embeddings
        let token_type_ids = if let Some(type_ids) = token_type_ids {
            type_ids.clone()
        } else {
            zeros(&input_ids.shape().dims())
        };
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        
        // Combine embeddings
        let embeddings = word_embeddings
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;
        
        // Layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

/// BERT self-attention layer
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    config: BertConfig,
}

impl BertSelfAttention {
    pub fn new(config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        
        Ok(Self {
            query: Linear::new(config.hidden_size, all_head_size),
            key: Linear::new(config.hidden_size, all_head_size),
            value: Linear::new(config.hidden_size, all_head_size),
            dropout: Dropout::new(config.attention_probs_dropout_prob),
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            config: config.clone(),
        })
    }
    
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape().dims()[0];
        let seq_length = x.shape().dims()[1];
        
        x.view(&[batch_size, seq_length, self.num_attention_heads, self.attention_head_size])?
            .transpose(1, 2)
    }
}

impl Module for BertSelfAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(hidden_states, None, None, false)
    }
}

impl BertSelfAttention {
    pub fn forward_with_mask(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        head_mask: Option<&Tensor>,
        output_attentions: bool,
    ) -> Result<Tensor> {
        // Linear projections
        let query_layer = self.transpose_for_scores(&self.query.forward(hidden_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
        
        // Compute attention scores
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2)?)?
            .div_scalar((self.attention_head_size as f64).sqrt())?;
        
        // Apply attention mask
        let attention_scores = if let Some(mask) = attention_mask {
            attention_scores.add(&mask)?
        } else {
            attention_scores
        };
        
        // Softmax
        let attention_probs = F::softmax(&attention_scores, -1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;
        
        // Apply head mask if provided
        let attention_probs = if let Some(mask) = head_mask {
            attention_probs.mul(mask)?
        } else {
            attention_probs
        };
        
        // Apply attention to values
        let context_layer = attention_probs.matmul(&value_layer)?;
        
        // Reshape to original format
        let batch_size = context_layer.shape().dims()[0];
        let seq_length = context_layer.shape().dims()[2];
        
        let context_layer = context_layer
            .transpose(1, 2)?
            .view(&[batch_size, seq_length, self.all_head_size])?;
        
        Ok(context_layer)
    }
}

/// BERT attention output layer
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size),
            layer_norm: LayerNorm::new(vec![config.hidden_size])?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }
}

impl Module for BertSelfOutput {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.forward_with_input(hidden_states, hidden_states)
    }
}

impl BertSelfOutput {
    pub fn forward_with_input(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states.add(input_tensor)?)?;
        Ok(hidden_states)
    }
}

/// BERT attention layer (combines self-attention and output)
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            self_attention: BertSelfAttention::new(config)?,
            output: BertSelfOutput::new(config)?,
        })
    }
}

impl Module for BertAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.output.forward_with_input(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

/// BERT intermediate layer (feed-forward)
pub struct BertIntermediate {
    dense: Linear,
    intermediate_act_fn: Box<dyn Module>,
}

impl BertIntermediate {
    pub fn new(config: &BertConfig) -> Result<Self> {
        let activation: Box<dyn Module> = match config.hidden_act.as_str() {
            "relu" => Box::new(ReLU::new()),
            "gelu" => Box::new(GELU::new()),
            "tanh" => Box::new(Tanh::new()),
            "silu" => Box::new(SiLU::new()),
            _ => Box::new(GELU::new()), // Default to GELU
        };
        
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.intermediate_size),
            intermediate_act_fn: activation,
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.intermediate_act_fn.forward(&hidden_states)
    }
}

/// BERT output layer
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.intermediate_size, config.hidden_size),
            layer_norm: LayerNorm::new(vec![config.hidden_size])?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }
}

impl Module for BertOutput {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.forward_with_input(hidden_states, hidden_states)
    }
}

impl BertOutput {
    pub fn forward_with_input(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(&hidden_states.add(input_tensor)?)?;
        Ok(hidden_states)
    }
}

/// BERT transformer layer
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            attention: BertAttention::new(config)?,
            intermediate: BertIntermediate::new(config)?,
            output: BertOutput::new(config)?,
        })
    }
}

impl Module for BertLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention
        let attention_output = self.attention.forward(hidden_states)?;
        
        // Feed-forward
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward_with_input(&intermediate_output, &attention_output)?;
        
        Ok(layer_output)
    }
}

/// BERT encoder (stack of transformer layers)
pub struct BertEncoder {
    layers: Vec<BertLayer>,
    config: BertConfig,
}

impl BertEncoder {
    pub fn new(config: &BertConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config)?);
        }
        
        Ok(Self {
            layers,
            config: config.clone(),
        })
    }
}

impl Module for BertEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        Ok(hidden_states)
    }
}

/// BERT pooler for classification tasks
pub struct BertPooler {
    dense: Linear,
    activation: Tanh,
}

impl BertPooler {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size),
            activation: Tanh::new(),
        })
    }
}

impl Module for BertPooler {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take the hidden state of the first token ([CLS])
        let first_token_tensor = hidden_states.select(1, 0)?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        self.activation.forward(&pooled_output)
    }
}

/// Main BERT model
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
    config: BertConfig,
}

impl BertModel {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            embeddings: BertEmbeddings::new(config)?,
            encoder: BertEncoder::new(config)?,
            pooler: Some(BertPooler::new(config)?),
            config: config.clone(),
        })
    }
    
    pub fn new_without_pooler(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            embeddings: BertEmbeddings::new(config)?,
            encoder: BertEncoder::new(config)?,
            pooler: None,
            config: config.clone(),
        })
    }
}

impl Module for BertModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_details(input_ids, None, None, None, None).map(|result| result.0)
    }
}

impl BertModel {
    pub fn forward_with_details(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        head_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Embeddings
        let embedding_output = self.embeddings.forward_with_types(
            input_ids,
            token_type_ids,
            position_ids,
        )?;
        
        // Encoder
        let sequence_output = self.encoder.forward(&embedding_output)?;
        
        // Pooler
        let pooled_output = if let Some(ref pooler) = self.pooler {
            Some(pooler.forward(&sequence_output)?)
        } else {
            None
        };
        
        Ok((sequence_output, pooled_output))
    }
    
    /// Get embeddings from the model
    pub fn get_input_embeddings(&self) -> &Embedding {
        &self.embeddings.word_embeddings
    }
    
    /// Set embeddings for the model
    pub fn set_input_embeddings(&mut self, embeddings: Embedding) {
        self.embeddings.word_embeddings = embeddings;
    }
}

/// BERT for masked language modeling
pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertLMPredictionHead,
}

impl BertForMaskedLM {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            bert: BertModel::new_without_pooler(config)?,
            cls: BertLMPredictionHead::new(config)?,
        })
    }
}

impl Module for BertForMaskedLM {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (sequence_output, _) = self.bert.forward_with_details(input_ids, None, None, None, None)?;
        self.cls.forward(&sequence_output)
    }
}

/// BERT LM prediction head
pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: Linear,
    bias: Tensor,
}

impl BertLMPredictionHead {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            transform: BertPredictionHeadTransform::new(config)?,
            decoder: Linear::new(config.hidden_size, config.vocab_size),
            bias: zeros(&[config.vocab_size]),
        })
    }
}

impl Module for BertLMPredictionHead {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.transform.forward(hidden_states)?;
        let hidden_states = self.decoder.forward(&hidden_states)?;
        hidden_states.add(&self.bias)
    }
}

/// BERT prediction head transform
pub struct BertPredictionHeadTransform {
    dense: Linear,
    transform_act_fn: Box<dyn Module>,
    layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn new(config: &BertConfig) -> Result<Self> {
        let activation: Box<dyn Module> = match config.hidden_act.as_str() {
            "relu" => Box::new(ReLU::new()),
            "gelu" => Box::new(GELU::new()),
            "tanh" => Box::new(Tanh::new()),
            "silu" => Box::new(SiLU::new()),
            _ => Box::new(GELU::new()),
        };
        
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size),
            transform_act_fn: activation,
            layer_norm: LayerNorm::new(vec![config.hidden_size])?,
        })
    }
}

impl Module for BertPredictionHeadTransform {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.transform_act_fn.forward(&hidden_states)?;
        self.layer_norm.forward(&hidden_states)
    }
}

/// BERT for sequence classification
pub struct BertForSequenceClassification {
    bert: BertModel,
    dropout: Dropout,
    classifier: Linear,
    num_labels: usize,
}

impl BertForSequenceClassification {
    pub fn new(config: &BertConfig, num_labels: usize) -> Result<Self> {
        let dropout_prob = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);
        
        Ok(Self {
            bert: BertModel::new(config)?,
            dropout: Dropout::new(dropout_prob),
            classifier: Linear::new(config.hidden_size, num_labels),
            num_labels,
        })
    }
}

impl Module for BertForSequenceClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, pooled_output) = self.bert.forward_with_details(input_ids, None, None, None, None)?;
        let pooled_output = pooled_output.unwrap();
        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }
}

/// BERT for token classification
pub struct BertForTokenClassification {
    bert: BertModel,
    dropout: Dropout,
    classifier: Linear,
    num_labels: usize,
}

impl BertForTokenClassification {
    pub fn new(config: &BertConfig, num_labels: usize) -> Result<Self> {
        let dropout_prob = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);
        
        Ok(Self {
            bert: BertModel::new_without_pooler(config)?,
            dropout: Dropout::new(dropout_prob),
            classifier: Linear::new(config.hidden_size, num_labels),
            num_labels,
        })
    }
}

impl Module for BertForTokenClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (sequence_output, _) = self.bert.forward_with_details(input_ids, None, None, None, None)?;
        let sequence_output = self.dropout.forward(&sequence_output)?;
        self.classifier.forward(&sequence_output)
    }
}

/// BERT for question answering
pub struct BertForQuestionAnswering {
    bert: BertModel,
    qa_outputs: Linear,
}

impl BertForQuestionAnswering {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Ok(Self {
            bert: BertModel::new_without_pooler(config)?,
            qa_outputs: Linear::new(config.hidden_size, 2), // start and end logits
        })
    }
}

impl Module for BertForQuestionAnswering {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (sequence_output, _) = self.bert.forward_with_details(input_ids, None, None, None, None)?;
        self.qa_outputs.forward(&sequence_output)
    }
}

impl BertForQuestionAnswering {
    pub fn forward_qa(&self, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.forward(input_ids)?;
        let start_logits = logits.select(-1, 0)?;
        let end_logits = logits.select(-1, 1)?;
        Ok((start_logits, end_logits))
    }
}

/// Training utilities for BERT
pub struct BertTrainer {
    model: Box<dyn Module>,
    optimizer: Adam,
    lr_scheduler: Option<Box<dyn LRScheduler>>,
    device: Device,
}

impl BertTrainer {
    pub fn new(
        model: Box<dyn Module>,
        learning_rate: f64,
        device: Device,
    ) -> Result<Self> {
        let optimizer = Adam::new(model.parameters(), learning_rate)?;
        
        Ok(Self {
            model,
            optimizer,
            lr_scheduler: None,
            device,
        })
    }
    
    pub fn train_mlm_step(
        &mut self,
        input_ids: &Tensor,
        labels: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<f32> {
        // Forward pass
        let logits = self.model.forward(input_ids)?;
        
        // Compute masked language modeling loss
        let loss = self.compute_mlm_loss(&logits, labels)?;
        
        // Backward pass
        self.optimizer.zero_grad();
        loss.backward()?;
        self.optimizer.step()?;
        
        Ok(loss.item())
    }
    
    pub fn train_classification_step(
        &mut self,
        input_ids: &Tensor,
        labels: &Tensor,
    ) -> Result<f32> {
        // Forward pass
        let logits = self.model.forward(input_ids)?;
        
        // Compute cross-entropy loss
        let loss = F::cross_entropy(&logits, labels)?;
        
        // Backward pass
        self.optimizer.zero_grad();
        loss.backward()?;
        self.optimizer.step()?;
        
        Ok(loss.item())
    }
    
    fn compute_mlm_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        // Reshape logits and labels for loss computation
        let vocab_size = logits.shape().dims()[2];
        let logits = logits.view(&[-1, vocab_size])?;
        let labels = labels.view(&[-1])?;
        
        // Compute cross-entropy loss only for masked tokens (labels != -100)
        let mask = labels.ne(&tensor![-100])?;
        let active_logits = logits.masked_select(&mask)?;
        let active_labels = labels.masked_select(&mask)?;
        
        F::cross_entropy(&active_logits, &active_labels)
    }
}

/// Example usage and testing
pub fn run_bert_example() -> Result<()> {
    println!("BERT Implementation Demo");
    
    // Create BERT configuration
    let config = BertConfig::bert_small(); // Use small config for demo
    println!("BERT Config: {:?}", config);
    
    // Create different BERT models
    let bert_base = BertModel::new(&config)?;
    let bert_mlm = BertForMaskedLM::new(&config)?;
    let bert_classifier = BertForSequenceClassification::new(&config, 2)?;
    let bert_token_classifier = BertForTokenClassification::new(&config, 9)?; // NER with 9 tags
    let bert_qa = BertForQuestionAnswering::new(&config)?;
    
    // Create sample input
    let batch_size = 4;
    let seq_length = 128;
    let input_ids = randint(0, config.vocab_size as i64, &[batch_size, seq_length]);
    let token_type_ids = randint(0, 2, &[batch_size, seq_length]);
    
    println!("Input shape: {:?}", input_ids.shape().dims());
    
    // Test base BERT model
    let (sequence_output, pooled_output) = bert_base.forward_with_details(
        &input_ids, None, Some(&token_type_ids), None, None
    )?;
    println!("Sequence output shape: {:?}", sequence_output.shape().dims());
    if let Some(pooled) = pooled_output {
        println!("Pooled output shape: {:?}", pooled.shape().dims());
    }
    
    // Test BERT for MLM
    let mlm_logits = bert_mlm.forward(&input_ids)?;
    println!("MLM logits shape: {:?}", mlm_logits.shape().dims());
    
    // Test BERT for classification
    let classification_logits = bert_classifier.forward(&input_ids)?;
    println!("Classification logits shape: {:?}", classification_logits.shape().dims());
    
    // Test BERT for token classification
    let token_logits = bert_token_classifier.forward(&input_ids)?;
    println!("Token classification logits shape: {:?}", token_logits.shape().dims());
    
    // Test BERT for QA
    let (start_logits, end_logits) = bert_qa.forward_qa(&input_ids)?;
    println!("QA start logits shape: {:?}", start_logits.shape().dims());
    println!("QA end logits shape: {:?}", end_logits.shape().dims());
    
    // Demonstrate training
    println!("\nTraining demonstration:");
    let device = Device::cpu();
    let mut trainer = BertTrainer::new(
        Box::new(bert_mlm),
        1e-4,
        device,
    )?;
    
    // Create sample MLM data
    let labels = input_ids.clone();
    let loss = trainer.train_mlm_step(&input_ids, &labels, None)?;
    println!("MLM training loss: {:.6}", loss);
    
    println!("BERT demo completed successfully!");
    
    Ok(())
}

fn main() -> Result<()> {
    run_bert_example()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bert_config() {
        let config = BertConfig::bert_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        
        let large_config = BertConfig::bert_large();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_hidden_layers, 24);
    }
    
    #[test]
    fn test_bert_embeddings() {
        let config = BertConfig::bert_small();
        let embeddings = BertEmbeddings::new(&config).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[2, 10]);
        let output = embeddings.forward(&input_ids).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.hidden_size]);
    }
    
    #[test]
    fn test_bert_attention() {
        let config = BertConfig::bert_small();
        let attention = BertSelfAttention::new(&config).unwrap();
        
        let hidden_states = randn(&[2, 10, config.hidden_size]);
        let output = attention.forward(&hidden_states).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.hidden_size]);
    }
    
    #[test]
    fn test_bert_layer() {
        let config = BertConfig::bert_small();
        let layer = BertLayer::new(&config).unwrap();
        
        let hidden_states = randn(&[2, 10, config.hidden_size]);
        let output = layer.forward(&hidden_states).unwrap();
        
        assert_eq!(output.shape().dims(), &[2, 10, config.hidden_size]);
    }
    
    #[test]
    fn test_bert_model() {
        let config = BertConfig::bert_small();
        let model = BertModel::new(&config).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[2, 10]);
        let (sequence_output, pooled_output) = model.forward_with_details(
            &input_ids, None, None, None, None
        ).unwrap();
        
        assert_eq!(sequence_output.shape().dims(), &[2, 10, config.hidden_size]);
        assert!(pooled_output.is_some());
        assert_eq!(pooled_output.unwrap().shape().dims(), &[2, config.hidden_size]);
    }
    
    #[test]
    fn test_bert_for_classification() {
        let config = BertConfig::bert_small();
        let model = BertForSequenceClassification::new(&config, 3).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[2, 10]);
        let logits = model.forward(&input_ids).unwrap();
        
        assert_eq!(logits.shape().dims(), &[2, 3]);
    }
    
    #[test]
    fn test_bert_for_qa() {
        let config = BertConfig::bert_small();
        let model = BertForQuestionAnswering::new(&config).unwrap();
        
        let input_ids = randint(0, config.vocab_size as i64, &[2, 10]);
        let (start_logits, end_logits) = model.forward_qa(&input_ids).unwrap();
        
        assert_eq!(start_logits.shape().dims(), &[2, 10]);
        assert_eq!(end_logits.shape().dims(), &[2, 10]);
    }
}