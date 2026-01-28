// Models are temporarily commented out until torsh_nn is available
// pub mod bert;
// pub mod gpt;
// pub mod lstm;
pub mod registry;
// pub mod t5;
// pub mod transformer;

// pub use bert::*;
// pub use gpt::*;
// pub use lstm::*;
pub use registry::*;
// pub use t5::*;
// pub use transformer::*;

use torsh_core::{device::DeviceType, Result};
use torsh_tensor::creation::randn;
use torsh_tensor::Tensor;

/// Base trait for text models (temporarily simplified)
pub trait TextModel {
    /// Get the model name
    fn name(&self) -> &str;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get hidden dimension
    fn hidden_dim(&self) -> usize;

    /// Get maximum sequence length
    fn max_seq_length(&self) -> usize;
}

/// Text encoder trait for encoding sequences
pub trait TextEncoder {
    /// Encode input tokens to hidden representations
    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;

    /// Get the output dimension of encoded representations
    fn output_dim(&self) -> usize;

    /// Get the maximum input length
    fn max_input_length(&self) -> usize;
}

/// Text decoder trait for generating sequences
pub trait TextDecoder {
    /// Decode from hidden states to output tokens
    fn decode(
        &self,
        hidden_states: &Tensor,
        past_key_values: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)>;

    /// Generate sequences autoregressively
    fn generate(
        &self,
        input_ids: &Tensor,
        max_length: usize,
        generation_config: &GenerationConfig,
    ) -> Result<Tensor>;
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub early_stopping: bool,
    pub num_beams: usize,
    pub do_sample: bool,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            early_stopping: false,
            num_beams: 1,
            do_sample: false,
            pad_token_id: None,
            eos_token_id: None,
            bos_token_id: None,
        }
    }
}

impl GenerationConfig {
    /// Create a configuration for greedy decoding
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            num_beams: 1,
            temperature: 1.0,
            ..Default::default()
        }
    }

    /// Create a configuration for sampling-based generation
    pub fn sampling(temperature: f32) -> Self {
        Self {
            do_sample: true,
            temperature,
            num_beams: 1,
            ..Default::default()
        }
    }

    /// Create a configuration for beam search
    pub fn beam_search(num_beams: usize) -> Self {
        Self {
            do_sample: false,
            num_beams,
            temperature: 1.0,
            ..Default::default()
        }
    }

    /// Create a configuration for nucleus (top-p) sampling
    pub fn nucleus_sampling(top_p: f32, temperature: f32) -> Self {
        Self {
            do_sample: true,
            top_p: Some(top_p),
            temperature,
            num_beams: 1,
            ..Default::default()
        }
    }

    /// Validate the configuration and return any issues
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(crate::TextError::ValidationError(
                "Temperature must be positive".to_string(),
            )
            .into());
        }

        if self.num_beams == 0 {
            return Err(crate::TextError::ValidationError(
                "Number of beams must be at least 1".to_string(),
            )
            .into());
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(crate::TextError::ValidationError(
                    "top_p must be between 0.0 and 1.0".to_string(),
                )
                .into());
            }
        }

        if self.repetition_penalty <= 0.0 {
            return Err(crate::TextError::ValidationError(
                "Repetition penalty must be positive".to_string(),
            )
            .into());
        }

        if self.length_penalty <= 0.0 {
            return Err(crate::TextError::ValidationError(
                "Length penalty must be positive".to_string(),
            )
            .into());
        }

        Ok(())
    }
}

/// Common text model configurations
#[derive(Debug, Clone)]
pub struct TextModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_dim: usize,
    pub max_position_embeddings: usize,
    pub dropout: f32,
    pub attention_dropout: f32,
    pub layer_norm_eps: f32,
    pub initializer_range: f32,
}

impl Default for TextModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522, // BERT default
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_dim: 3072,
            max_position_embeddings: 512,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
        }
    }
}

impl TextModelConfig {
    /// Create a custom configuration with validation
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let config = Self {
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            intermediate_dim: hidden_dim * 4, // Common default: 4x hidden size
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }

    /// Builder method to set intermediate dimension
    pub fn with_intermediate_dim(mut self, intermediate_dim: usize) -> Self {
        self.intermediate_dim = intermediate_dim;
        self
    }

    /// Builder method to set max position embeddings
    pub fn with_max_position_embeddings(mut self, max_position_embeddings: usize) -> Self {
        self.max_position_embeddings = max_position_embeddings;
        self
    }

    /// Builder method to set dropout rates
    pub fn with_dropout(mut self, dropout: f32, attention_dropout: f32) -> Self {
        self.dropout = dropout;
        self.attention_dropout = attention_dropout;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(crate::TextError::ValidationError(
                "Vocabulary size must be positive".to_string(),
            )
            .into());
        }

        if self.hidden_dim == 0 {
            return Err(crate::TextError::ValidationError(
                "Hidden dimension must be positive".to_string(),
            )
            .into());
        }

        if self.num_layers == 0 {
            return Err(crate::TextError::ValidationError(
                "Number of layers must be positive".to_string(),
            )
            .into());
        }

        if self.num_heads == 0 {
            return Err(crate::TextError::ValidationError(
                "Number of heads must be positive".to_string(),
            )
            .into());
        }

        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(crate::TextError::ValidationError(
                "Hidden dimension must be divisible by number of heads".to_string(),
            )
            .into());
        }

        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(crate::TextError::ValidationError(
                "Dropout must be between 0.0 and 1.0".to_string(),
            )
            .into());
        }

        if !(0.0..=1.0).contains(&self.attention_dropout) {
            return Err(crate::TextError::ValidationError(
                "Attention dropout must be between 0.0 and 1.0".to_string(),
            )
            .into());
        }

        Ok(())
    }

    /// BERT base configuration
    pub fn bert_base() -> Self {
        Self::default()
    }

    /// BERT large configuration
    pub fn bert_large() -> Self {
        Self {
            vocab_size: 30522,
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_dim: 4096,
            max_position_embeddings: 512,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
        }
    }

    /// GPT-2 small configuration
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_dim: 3072,
            max_position_embeddings: 1024,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// GPT-2 medium configuration
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_dim: 4096,
            max_position_embeddings: 1024,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// GPT-2 large configuration
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1280,
            num_layers: 36,
            num_heads: 20,
            intermediate_dim: 5120,
            max_position_embeddings: 1024,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// T5 small configuration
    pub fn t5_small() -> Self {
        Self {
            vocab_size: 32128,
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            intermediate_dim: 2048,
            max_position_embeddings: 512,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-6,
            initializer_range: 1.0,
        }
    }
}

/// Model integration utilities and wrappers
pub mod integration {
    use super::*;
    use crate::tokenization::Tokenizer;
    use std::sync::Arc;

    /// Universal text encoder wrapper that can work with different model types
    pub struct UniversalTextEncoder {
        model: Box<dyn TextModel + Send + Sync>,
        tokenizer: Arc<dyn Tokenizer>,
        device: DeviceType,
        pooling_strategy: PoolingStrategy,
    }

    #[derive(Debug, Clone)]
    pub enum PoolingStrategy {
        Mean,
        Max,
        CLS,
        Last,
    }

    impl UniversalTextEncoder {
        pub fn new(
            model: Box<dyn TextModel + Send + Sync>,
            tokenizer: Arc<dyn Tokenizer>,
            device: DeviceType,
        ) -> Self {
            Self {
                model,
                tokenizer,
                device,
                pooling_strategy: PoolingStrategy::Mean,
            }
        }

        pub fn with_pooling(mut self, strategy: PoolingStrategy) -> Self {
            self.pooling_strategy = strategy;
            self
        }

        /// Encode text to dense representations
        pub fn encode_text(&self, texts: &[String]) -> Result<Vec<Tensor>> {
            let mut results = Vec::new();

            for text in texts {
                let tokens = self.tokenizer.encode(text)?;
                let tokens_len = tokens.len();
                let tokens_f32: Vec<f32> = tokens.into_iter().map(|x| x as f32).collect();
                let input_tensor = Tensor::from_vec(tokens_f32, &[1, tokens_len])?;

                // Get hidden states from model
                let hidden_states = self.forward_model(&input_tensor)?;

                // Apply pooling
                let pooled = self.apply_pooling(&hidden_states)?;
                results.push(pooled);
            }

            Ok(results)
        }

        /// Batch encode multiple texts efficiently
        pub fn batch_encode(&self, texts: &[String], max_length: Option<usize>) -> Result<Tensor> {
            let max_len = max_length.unwrap_or(self.model.max_seq_length());
            let mut batch_tokens = Vec::new();
            let mut attention_masks = Vec::new();

            for text in texts {
                let mut tokens = self.tokenizer.encode(text)?;
                let mut attention_mask = vec![1i32; tokens.len()];

                // Truncate or pad to max_length
                if tokens.len() > max_len {
                    tokens.truncate(max_len);
                    attention_mask.truncate(max_len);
                } else {
                    while tokens.len() < max_len {
                        tokens.push(0); // Assume 0 is pad token
                        attention_mask.push(0);
                    }
                }

                batch_tokens.push(tokens);
                attention_masks.push(attention_mask);
            }

            // Convert to tensors
            let batch_size = texts.len();
            let input_ids: Vec<f32> = batch_tokens
                .into_iter()
                .flatten()
                .map(|x| x as f32)
                .collect();
            let input_tensor = Tensor::from_vec(input_ids, &[batch_size, max_len])?;

            let attention_mask: Vec<f32> = attention_masks
                .into_iter()
                .flatten()
                .map(|x| x as f32)
                .collect();
            let mask_tensor = Tensor::from_vec(attention_mask, &[batch_size, max_len])?;

            // Forward pass
            let hidden_states = self.forward_model(&input_tensor)?;

            // Apply pooling with attention mask
            self.apply_pooling_with_mask(&hidden_states, &mask_tensor)
        }

        fn forward_model(&self, input: &Tensor) -> Result<Tensor> {
            // This is a placeholder - in real implementation would call the actual model
            // For now, just return dummy hidden states
            let batch_size = input.size(0)?;
            let seq_len = input.size(1)?;
            let hidden_dim = self.model.hidden_dim();

            Tensor::zeros(&[batch_size, seq_len, hidden_dim], DeviceType::Cpu)
        }

        fn apply_pooling(&self, hidden_states: &Tensor) -> Result<Tensor> {
            match self.pooling_strategy {
                PoolingStrategy::Mean => hidden_states.mean(Some(&[1]), false),
                PoolingStrategy::Max => {
                    let max_values = hidden_states.max(Some(1), false)?;
                    Ok(max_values)
                }
                PoolingStrategy::CLS => {
                    // Take first token (CLS token)
                    hidden_states.narrow(1, 0, 1)?.squeeze(1)
                }
                PoolingStrategy::Last => {
                    // Take last token
                    let seq_len = hidden_states.size(1)? as i64;
                    hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)
                }
            }
        }

        fn apply_pooling_with_mask(
            &self,
            hidden_states: &Tensor,
            _attention_mask: &Tensor,
        ) -> Result<Tensor> {
            match self.pooling_strategy {
                PoolingStrategy::Mean => {
                    // Simplified masked mean pooling
                    hidden_states.mean(Some(&[1]), false)
                }
                _ => {
                    // For other strategies, use simple pooling for now
                    self.apply_pooling(hidden_states)
                }
            }
        }
    }

    impl TextEncoder for UniversalTextEncoder {
        fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
            let hidden_states = self.forward_model(input_ids)?;

            if let Some(mask) = attention_mask {
                self.apply_pooling_with_mask(&hidden_states, mask)
            } else {
                self.apply_pooling(&hidden_states)
            }
        }

        fn output_dim(&self) -> usize {
            self.model.hidden_dim()
        }

        fn max_input_length(&self) -> usize {
            self.model.max_seq_length()
        }
    }

    /// Advanced text decoder with multiple generation strategies
    pub struct AdvancedTextDecoder {
        model: Box<dyn TextModel + Send + Sync>,
        tokenizer: Arc<dyn Tokenizer>,
        device: DeviceType,
    }

    impl AdvancedTextDecoder {
        pub fn new(
            model: Box<dyn TextModel + Send + Sync>,
            tokenizer: Arc<dyn Tokenizer>,
            device: DeviceType,
        ) -> Self {
            Self {
                model,
                tokenizer,
                device,
            }
        }

        /// Generate text with beam search
        pub fn beam_search_generate(
            &self,
            input_ids: &Tensor,
            config: &GenerationConfig,
        ) -> Result<Vec<Tensor>> {
            let num_beams = config.num_beams;
            let max_length = config.max_length;
            let batch_size = input_ids.size(0)?;

            // Initialize beam search state
            let mut beam_scores = vec![0.0f32; batch_size * num_beams];
            let mut beam_tokens = vec![input_ids.clone(); num_beams];
            let mut _beam_indices = (0..num_beams).collect::<Vec<_>>();

            for _step in 0..max_length {
                let mut all_candidates = Vec::new();

                // For each beam, generate candidates
                for (beam_idx, beam_tokens) in beam_tokens.iter().enumerate() {
                    let logits = self.forward_model(beam_tokens)?;
                    let seq_len = logits.size(1)? as i64;
                    let next_token_logits = logits.narrow(1, seq_len - 1, 1)?;

                    // Apply temperature if needed
                    let next_token_logits = if config.temperature != 1.0 {
                        next_token_logits.div_scalar(config.temperature)?
                    } else {
                        next_token_logits
                    };

                    let probs = next_token_logits.softmax(-1)?;

                    // Get top-k candidates for this beam
                    let vocab_size = self.model.vocab_size();
                    for token_id in 0..vocab_size.min(config.top_k.unwrap_or(vocab_size)) {
                        let prob = probs.narrow(-1, token_id as i64, 1)?;
                        let prob_scalar = prob.item()?;
                        let score = beam_scores[beam_idx] + prob_scalar.ln();

                        all_candidates.push((score, beam_idx, token_id as u32));
                    }
                }

                // Select top beams
                all_candidates
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                all_candidates.truncate(num_beams);

                // Update beams
                let mut new_beam_tokens = Vec::new();
                let mut new_beam_scores = Vec::new();

                for (score, beam_idx, _token_id) in all_candidates {
                    let new_tokens = beam_tokens[beam_idx].clone();
                    // Append token (simplified - actual implementation would be more complex)
                    new_beam_tokens.push(new_tokens);
                    new_beam_scores.push(score);
                }

                beam_tokens = new_beam_tokens;
                beam_scores = new_beam_scores;

                // Check for early stopping
                if config.early_stopping {
                    if let Some(_eos_id) = config.eos_token_id {
                        // Check if all beams have generated EOS
                        // Simplified implementation
                        break;
                    }
                }
            }

            Ok(beam_tokens)
        }

        /// Generate with nucleus (top-p) sampling
        pub fn nucleus_sampling_generate(
            &self,
            input_ids: &Tensor,
            config: &GenerationConfig,
        ) -> Result<Tensor> {
            let batch_size = input_ids.size(0)?;
            let initial_seq_len = input_ids.size(1)?;
            let _max_new_tokens = config.max_length;

            // Collect all tokens in a vector for each batch
            let mut all_tokens: Vec<Vec<i64>> = Vec::with_capacity(batch_size);

            // Initialize with input tokens
            let input_vec = input_ids.to_vec()?;
            for batch_idx in 0..batch_size {
                let start = batch_idx * initial_seq_len;
                let end = start + initial_seq_len;
                let batch_tokens: Vec<i64> =
                    input_vec[start..end].iter().map(|&x| x as i64).collect();
                all_tokens.push(batch_tokens);
            }

            // Generation loop - simplified single-step for now due to method availability
            // Full iterative implementation would require tensor concatenation support
            let logits = self.forward_model(input_ids)?;
            let seq_len = logits.size(1)? as i64;
            let next_token_logits = logits.narrow(1, seq_len - 1, 1)?;

            // Apply temperature
            let next_token_logits = if config.temperature != 1.0 {
                next_token_logits.div_scalar(config.temperature)?
            } else {
                next_token_logits
            };

            let probs = next_token_logits.softmax(-1)?;

            // Sample next tokens
            let next_tokens = if let Some(top_p) = config.top_p {
                self.sample_nucleus(&probs, top_p)?
            } else {
                // Argmax sampling - convert to f32 to match sample_nucleus return type
                let argmax_result = probs.argmax(Some(-1))?;
                let argmax_vec = argmax_result.to_vec()?;
                let argmax_f32: Vec<f32> = argmax_vec.iter().map(|&x| x as f32).collect();
                Tensor::from_vec(argmax_f32, &[batch_size])?
            };

            // Append next tokens to sequences
            let next_tokens_vec = next_tokens.to_vec()?;
            for (batch_idx, &token) in next_tokens_vec.iter().enumerate() {
                all_tokens[batch_idx].push(token as i64);
            }

            // Convert back to tensor
            let total_seq_len = initial_seq_len + 1;
            let mut flat_tokens = Vec::with_capacity(batch_size * total_seq_len);
            for tokens in &all_tokens {
                flat_tokens.extend(tokens.iter().map(|&x| x as f32));
            }

            let result = Tensor::from_vec(flat_tokens, &[batch_size, total_seq_len])?;
            result.to_dtype(torsh_core::DType::I64)
        }

        fn forward_model(&self, input: &Tensor) -> Result<Tensor> {
            // Placeholder implementation - would call actual model
            let batch_size = input.size(0)?;
            let seq_len = input.size(1)?;
            let vocab_size = self.model.vocab_size();

            Ok(randn::<f32>(&[batch_size, seq_len, vocab_size])?)
        }

        fn sample_nucleus(&self, probs: &Tensor, top_p: f32) -> Result<Tensor> {
            // Improved nucleus sampling with proper probability cutoff
            let batch_size = probs.size(0)?;
            let vocab_size = probs.size(-1)?;

            // Get probabilities as vector for processing
            let probs_vec = probs.to_vec()?;
            let mut result_tokens: Vec<i64> = Vec::with_capacity(batch_size);

            // Initialize random number generator
            let mut rng = scirs2_core::random::thread_rng();

            for batch_idx in 0..batch_size {
                // Extract probabilities for this batch item
                let start_idx = batch_idx * vocab_size;
                let end_idx = start_idx + vocab_size;
                let batch_probs = &probs_vec[start_idx..end_idx];

                // Create (index, probability) pairs and sort by probability descending
                let mut indexed_probs: Vec<(usize, f32)> = batch_probs
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                indexed_probs
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Find nucleus - accumulate probabilities until reaching top_p
                let mut cumsum = 0.0;
                let mut nucleus_size = 0;
                for (_, prob) in &indexed_probs {
                    cumsum += prob;
                    nucleus_size += 1;
                    if cumsum >= top_p {
                        break;
                    }
                }

                // Sample from nucleus
                let nucleus = &indexed_probs[..nucleus_size];
                let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();

                // Renormalize probabilities in nucleus
                let rand_val: f32 = rng.random_range(0.0..nucleus_sum);
                let target = rand_val;

                let mut cumsum = 0.0;
                let mut selected_token = nucleus[0].0;
                for &(idx, prob) in nucleus {
                    cumsum += prob;
                    if cumsum >= target {
                        selected_token = idx;
                        break;
                    }
                }

                result_tokens.push(selected_token as i64);
            }

            // Convert i64 token indices to tensor
            // Use from_vec with f32 and then cast to i64
            let result_tokens_f32: Vec<f32> = result_tokens.iter().map(|&x| x as f32).collect();
            let result_tensor = Tensor::from_vec(result_tokens_f32, &[batch_size])?;
            result_tensor.to_dtype(torsh_core::DType::I64)
        }
    }

    impl TextDecoder for AdvancedTextDecoder {
        fn decode(
            &self,
            _hidden_states: &Tensor,
            _past_key_values: Option<&Tensor>,
        ) -> Result<(Tensor, Option<Tensor>)> {
            // Convert hidden states to logits over vocabulary
            let vocab_size = self.model.vocab_size();
            let batch_size = _hidden_states.size(0)?;
            let seq_len = _hidden_states.size(1)?;

            // Placeholder: create random logits
            let logits = randn::<f32>(&[batch_size, seq_len, vocab_size])?;

            Ok((logits, None))
        }

        fn generate(
            &self,
            input_ids: &Tensor,
            _max_length: usize,
            config: &GenerationConfig,
        ) -> Result<Tensor> {
            if config.num_beams > 1 {
                // Use beam search
                let beams = self.beam_search_generate(input_ids, config)?;
                Ok(beams
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| input_ids.clone()))
            } else {
                // Use sampling
                self.nucleus_sampling_generate(input_ids, config)
            }
        }
    }

    /// High-level model wrapper that combines encoder and decoder functionality
    pub struct TextModelWrapper {
        encoder: Option<Box<dyn TextEncoder>>,
        decoder: Option<Box<dyn TextDecoder>>,
        tokenizer: Arc<dyn Tokenizer>,
        device: DeviceType,
    }

    impl TextModelWrapper {
        pub fn new(tokenizer: Arc<dyn Tokenizer>, device: DeviceType) -> Self {
            Self {
                encoder: None,
                decoder: None,
                tokenizer,
                device,
            }
        }

        pub fn with_encoder(mut self, encoder: Box<dyn TextEncoder>) -> Self {
            self.encoder = Some(encoder);
            self
        }

        pub fn with_decoder(mut self, decoder: Box<dyn TextDecoder>) -> Self {
            self.decoder = Some(decoder);
            self
        }

        /// Encode text to embeddings
        pub fn encode(&self, text: &str) -> Result<Tensor> {
            if let Some(encoder) = &self.encoder {
                let tokens = self.tokenizer.encode(text)?;
                let tokens_f32: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();
                let input_tensor = Tensor::from_vec(tokens_f32, &[1, tokens.len()])?;
                encoder.encode(&input_tensor, None)
            } else {
                Err(torsh_core::error::TorshError::InvalidArgument(
                    "No encoder available".to_string(),
                ))
            }
        }

        /// Generate text from prompt
        pub fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
            if let Some(decoder) = &self.decoder {
                let tokens = self.tokenizer.encode(prompt)?;
                let tokens_f32: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();
                let input_tensor = Tensor::from_vec(tokens_f32, &[1, tokens.len()])?;
                let generated_tokens =
                    decoder.generate(&input_tensor, config.max_length, config)?;

                // Convert tokens back to text
                let generated_ids: Vec<u32> = generated_tokens
                    .to_vec()?
                    .into_iter()
                    .map(|x| x as u32)
                    .collect();
                Ok(self.tokenizer.decode(&generated_ids)?)
            } else {
                Err(torsh_core::error::TorshError::InvalidArgument(
                    "No decoder available".to_string(),
                ))
            }
        }

        /// Similarity search between texts
        pub fn text_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
            if let Some(_encoder) = &self.encoder {
                let _emb1 = self.encode(text1)?;
                let _emb2 = self.encode(text2)?;

                // Compute cosine similarity
                // let dot_product = emb1.mul(&emb2)?.sum(None, false)?;
                // let norm1 = emb1.norm(None, false, false)?.unwrap();
                // let norm2 = emb2.norm(None, false, false)?.unwrap();
                // let similarity = dot_product.div(&norm1.mul(&norm2)?)?;

                // Extract scalar value (simplified)
                Ok(0.5) // Placeholder
            } else {
                Err(torsh_core::error::TorshError::InvalidArgument(
                    "No encoder available for similarity computation".to_string(),
                ))
            }
        }

        /// Question answering functionality
        pub fn question_answering(&self, context: &str, question: &str) -> Result<String> {
            // Combine context and question
            let input = format!("{} [SEP] {}", context, question);

            if let Some(_decoder) = &self.decoder {
                let config = GenerationConfig {
                    max_length: 50,
                    temperature: 0.7,
                    ..Default::default()
                };
                self.generate(&input, &config)
            } else {
                Err(torsh_core::error::TorshError::InvalidArgument(
                    "No decoder available for question answering".to_string(),
                ))
            }
        }

        /// Text classification
        pub fn classify(&self, text: &str, labels: &[String]) -> Result<(String, f32)> {
            if let Some(_encoder) = &self.encoder {
                let _text_embedding = self.encode(text)?;

                // For each label, compute similarity (simplified approach)
                let mut best_label = String::new();
                let mut best_score = 0.0f32;

                for label in labels {
                    let _label_embedding = self.encode(label)?;

                    // Compute similarity (placeholder)
                    let score = 1.0 / (1.0 + labels.len() as f32); // Dummy score

                    if score > best_score {
                        best_score = score;
                        best_label = label.clone();
                    }
                }

                Ok((best_label, best_score))
            } else {
                Err(torsh_core::error::TorshError::InvalidArgument(
                    "No encoder available for classification".to_string(),
                ))
            }
        }
    }

    /// Training utilities for text models
    pub struct TextModelTrainer {
        model: Box<dyn TextModel + Send + Sync>,
        tokenizer: Arc<dyn Tokenizer>,
        device: DeviceType,
        learning_rate: f32,
        batch_size: usize,
    }

    impl TextModelTrainer {
        pub fn new(
            model: Box<dyn TextModel + Send + Sync>,
            tokenizer: Arc<dyn Tokenizer>,
            device: DeviceType,
        ) -> Self {
            Self {
                model,
                tokenizer,
                device,
                learning_rate: 1e-4,
                batch_size: 32,
            }
        }

        pub fn with_learning_rate(mut self, lr: f32) -> Self {
            self.learning_rate = lr;
            self
        }

        pub fn with_batch_size(mut self, batch_size: usize) -> Self {
            self.batch_size = batch_size;
            self
        }

        /// Train on a dataset of text pairs (simplified)
        pub fn train_on_texts(&mut self, texts: &[String], labels: &[String]) -> Result<()> {
            // Simplified training loop
            for batch_start in (0..texts.len()).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(texts.len());
                let batch_texts = &texts[batch_start..batch_end];
                let batch_labels = &labels[batch_start..batch_end];

                // Tokenize batch
                let mut batch_tokens = Vec::new();
                for text in batch_texts {
                    let tokens = self.tokenizer.encode(text)?;
                    batch_tokens.push(tokens);
                }

                // Forward pass (placeholder)
                self.forward_backward_step(&batch_tokens, batch_labels)?;
            }

            Ok(())
        }

        fn forward_backward_step(&self, _tokens: &[Vec<u32>], _labels: &[String]) -> Result<()> {
            // Placeholder for actual training step
            // Would involve forward pass, loss computation, and backward pass
            Ok(())
        }

        /// Fine-tune on a specific task
        pub fn fine_tune(
            &mut self,
            task_data: &[(String, String)],
            num_epochs: usize,
        ) -> Result<()> {
            for epoch in 0..num_epochs {
                println!("Epoch {}/{}", epoch + 1, num_epochs);

                for (input_text, target_text) in task_data {
                    // Simplified fine-tuning step
                    let _input_tokens = self.tokenizer.encode(input_text)?;
                    let _target_tokens = self.tokenizer.encode(target_text)?;

                    // Placeholder for actual fine-tuning logic
                }
            }

            Ok(())
        }

        /// Evaluate model performance
        pub fn evaluate(&self, test_data: &[(String, String)]) -> Result<f32> {
            let mut correct = 0;
            let total = test_data.len();

            for (input_text, _expected_output) in test_data {
                // Simplified evaluation
                let _predicted = self.predict(input_text)?;
                // Compare with expected_output
                correct += 1; // Placeholder
            }

            Ok(correct as f32 / total as f32)
        }

        fn predict(&self, input_text: &str) -> Result<String> {
            // Placeholder prediction
            let _tokens = self.tokenizer.encode(input_text)?;
            Ok("placeholder_prediction".to_string())
        }
    }

    /// Attention visualization utilities for understanding model behavior
    pub struct AttentionVisualizer {
        tokenizer: Arc<dyn Tokenizer>,
        max_display_length: usize,
    }

    impl AttentionVisualizer {
        pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
            Self {
                tokenizer,
                max_display_length: 64,
            }
        }

        pub fn with_max_display_length(mut self, max_length: usize) -> Self {
            self.max_display_length = max_length;
            self
        }

        /// Extract and visualize attention patterns from transformer model
        pub fn visualize_attention(
            &self,
            input_text: &str,
            attention_weights: &Tensor,
            layer_idx: Option<usize>,
            head_idx: Option<usize>,
        ) -> Result<AttentionVisualization> {
            let tokens = self.tokenizer.encode(input_text)?;
            let token_strings = self.tokens_to_strings(&tokens)?;

            // Extract attention weights for specified layer and head
            let attention_matrix =
                self.extract_attention_matrix(attention_weights, layer_idx, head_idx)?;

            // Create visualization
            let visualization = AttentionVisualization {
                tokens: token_strings,
                attention_matrix: {
                    let flat_data = attention_matrix.to_vec()?;
                    let shape_dims = attention_matrix.shape().dims().to_vec();
                    let rows = shape_dims[0];
                    let cols = shape_dims[1];
                    let mut matrix = Vec::with_capacity(rows);
                    for i in 0..rows {
                        let mut row = Vec::with_capacity(cols);
                        for j in 0..cols {
                            row.push(flat_data[i * cols + j]);
                        }
                        matrix.push(row);
                    }
                    matrix
                },
                layer_idx,
                head_idx,
                max_attention_value: self.find_max_attention(&attention_matrix)?,
                min_attention_value: self.find_min_attention(&attention_matrix)?,
            };

            Ok(visualization)
        }

        /// Create a text-based attention heatmap
        pub fn create_text_heatmap(&self, visualization: &AttentionVisualization) -> String {
            let mut output = String::new();

            // Header
            if let (Some(layer), Some(head)) = (visualization.layer_idx, visualization.head_idx) {
                output.push_str(&format!(
                    "Attention Visualization - Layer {}, Head {}\n",
                    layer, head
                ));
            } else {
                output.push_str("Attention Visualization\n");
            }
            output.push_str(&"=".repeat(50));
            output.push('\n');

            // Token header
            output.push_str("     ");
            for (i, token) in visualization.tokens.iter().enumerate() {
                if i < self.max_display_length {
                    output.push_str(&format!("{:>8}", token));
                }
            }
            output.push('\n');

            // Attention matrix
            let max_val = visualization.max_attention_value;
            let min_val = visualization.min_attention_value;
            let range = max_val - min_val;

            for (i, row) in visualization.attention_matrix.iter().enumerate() {
                if i >= self.max_display_length {
                    break;
                }

                // Row token
                output.push_str(&format!("{:>5}", visualization.tokens[i]));

                // Attention values
                for (j, &value) in row.iter().enumerate() {
                    if j >= self.max_display_length {
                        break;
                    }

                    // Normalize to 0-9 scale for text display
                    let normalized = if range > 0.0 {
                        ((value - min_val) / range * 9.0) as u8
                    } else {
                        5u8
                    };

                    let char = match normalized {
                        0..=1 => " ",
                        2..=3 => ".",
                        4..=5 => "-",
                        6..=7 => "=",
                        8..=9 => "#",
                        _ => "#",
                    };

                    output.push_str(&format!("{:>8}", char));
                }
                output.push('\n');
            }

            // Legend
            output.push('\n');
            output.push_str("Legend: [ ] = low attention, [#] = high attention\n");
            output.push_str(&format!("Range: {:.4} - {:.4}\n", min_val, max_val));

            output
        }

        /// Generate attention statistics for analysis
        pub fn compute_attention_stats(
            &self,
            visualization: &AttentionVisualization,
        ) -> AttentionStats {
            let matrix = &visualization.attention_matrix;
            let total_elements = matrix.len() * matrix[0].len();

            let mut sum = 0.0;
            let mut max_val = f32::NEG_INFINITY;
            let mut min_val = f32::INFINITY;
            let mut entropy_sum = 0.0;

            for row in matrix {
                for &value in row {
                    sum += value;
                    max_val = max_val.max(value);
                    min_val = min_val.min(value);

                    // Entropy calculation (for attention distribution)
                    if value > 0.0 {
                        entropy_sum -= value * value.ln();
                    }
                }
            }

            let mean = sum / total_elements as f32;

            // Variance calculation
            let mut variance_sum = 0.0;
            for row in matrix {
                for &value in row {
                    variance_sum += (value - mean).powi(2);
                }
            }
            let variance = variance_sum / total_elements as f32;
            let std_dev = variance.sqrt();

            AttentionStats {
                mean_attention: mean,
                std_attention: std_dev,
                max_attention: max_val,
                min_attention: min_val,
                entropy: entropy_sum,
                sparsity: self.compute_sparsity(matrix),
            }
        }

        /// Create attention flow visualization showing token-to-token attention
        pub fn create_attention_flow(
            &self,
            visualization: &AttentionVisualization,
            threshold: f32,
        ) -> Vec<AttentionFlow> {
            let mut flows = Vec::new();

            for (i, row) in visualization.attention_matrix.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    if weight >= threshold
                        && i < visualization.tokens.len()
                        && j < visualization.tokens.len()
                    {
                        flows.push(AttentionFlow {
                            from_token: visualization.tokens[j].clone(),
                            to_token: visualization.tokens[i].clone(),
                            from_position: j,
                            to_position: i,
                            weight,
                        });
                    }
                }
            }

            // Sort by weight descending
            flows.sort_by(|a, b| {
                b.weight
                    .partial_cmp(&a.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            flows
        }

        /// Compare attention patterns between different layers or heads
        pub fn compare_attention_patterns(
            &self,
            pattern1: &AttentionVisualization,
            pattern2: &AttentionVisualization,
        ) -> Result<AttentionComparison> {
            if pattern1.tokens.len() != pattern2.tokens.len() {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "Cannot compare attention patterns with different sequence lengths".to_string(),
                ));
            }

            let seq_len = pattern1.tokens.len();
            let mut correlation_sum = 0.0;
            let mut diff_sum = 0.0;
            let mut max_diff: f32 = 0.0;

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let val1 = pattern1.attention_matrix[i][j];
                    let val2 = pattern2.attention_matrix[i][j];

                    correlation_sum += val1 * val2;
                    let diff = (val1 - val2).abs();
                    diff_sum += diff;
                    max_diff = max_diff.max(diff);
                }
            }

            let total_elements = (seq_len * seq_len) as f32;
            let mean_absolute_difference = diff_sum / total_elements;
            let correlation = correlation_sum / total_elements;

            Ok(AttentionComparison {
                correlation,
                mean_absolute_difference,
                max_difference: max_diff,
                pattern1_info: format!(
                    "Layer {:?}, Head {:?}",
                    pattern1.layer_idx, pattern1.head_idx
                ),
                pattern2_info: format!(
                    "Layer {:?}, Head {:?}",
                    pattern2.layer_idx, pattern2.head_idx
                ),
            })
        }

        fn extract_attention_matrix(
            &self,
            attention_weights: &Tensor,
            layer_idx: Option<usize>,
            head_idx: Option<usize>,
        ) -> Result<Tensor> {
            let shape = attention_weights.shape();

            match shape.ndim() {
                4 => {
                    // Shape: [batch, num_heads, seq_len, seq_len]
                    let batch_idx = 0; // Take first batch
                    let head_idx = head_idx.unwrap_or(0);

                    let attention_slice = attention_weights
                        .narrow(0, batch_idx as i64, 1)?
                        .narrow(1, head_idx as i64, 1)?
                        .squeeze(0)?
                        .squeeze(0)?; // Note: after squeezing dim 0, the old dim 1 becomes dim 0

                    Ok(attention_slice)
                }
                5 => {
                    // Shape: [batch, num_layers, num_heads, seq_len, seq_len]
                    let batch_idx = 0; // Take first batch
                    let layer_idx = layer_idx.unwrap_or(0);
                    let head_idx = head_idx.unwrap_or(0);

                    let attention_slice = attention_weights
                        .narrow(0, batch_idx as i64, 1)?
                        .narrow(1, layer_idx as i64, 1)?
                        .narrow(2, head_idx as i64, 1)?
                        .squeeze(0)?
                        .squeeze(0)?
                        .squeeze(0)?; // Squeeze three times as we have three size-1 dimensions

                    Ok(attention_slice)
                }
                _ => Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Unsupported attention tensor shape: {:?}",
                    shape
                ))),
            }
        }

        fn tokens_to_strings(&self, tokens: &[u32]) -> Result<Vec<String>> {
            // For now, just convert token IDs to strings
            // In a real implementation, would use proper token-to-string conversion
            let token_strings: Vec<String> = tokens
                .iter()
                .map(|&token_id| format!("tok_{}", token_id))
                .collect();

            Ok(token_strings)
        }

        fn find_max_attention(&self, attention_matrix: &Tensor) -> Result<f32> {
            let values = attention_matrix.to_vec()?;
            Ok(values.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        }

        fn find_min_attention(&self, attention_matrix: &Tensor) -> Result<f32> {
            let values = attention_matrix.to_vec()?;
            Ok(values.iter().copied().fold(f32::INFINITY, f32::min))
        }

        fn compute_sparsity(&self, matrix: &[Vec<f32>]) -> f32 {
            let total_elements = matrix.len() * matrix[0].len();
            let mut zero_count = 0;

            for row in matrix {
                for &value in row {
                    if value.abs() < 1e-6 {
                        zero_count += 1;
                    }
                }
            }

            zero_count as f32 / total_elements as f32
        }
    }

    /// Visualization data structure for attention patterns
    #[derive(Debug, Clone)]
    pub struct AttentionVisualization {
        pub tokens: Vec<String>,
        pub attention_matrix: Vec<Vec<f32>>,
        pub layer_idx: Option<usize>,
        pub head_idx: Option<usize>,
        pub max_attention_value: f32,
        pub min_attention_value: f32,
    }

    /// Statistical analysis of attention patterns
    #[derive(Debug, Clone)]
    pub struct AttentionStats {
        pub mean_attention: f32,
        pub std_attention: f32,
        pub max_attention: f32,
        pub min_attention: f32,
        pub entropy: f32,
        pub sparsity: f32,
    }

    /// Individual attention flow between tokens
    #[derive(Debug, Clone)]
    pub struct AttentionFlow {
        pub from_token: String,
        pub to_token: String,
        pub from_position: usize,
        pub to_position: usize,
        pub weight: f32,
    }

    /// Comparison between two attention patterns
    #[derive(Debug, Clone)]
    pub struct AttentionComparison {
        pub correlation: f32,
        pub mean_absolute_difference: f32,
        pub max_difference: f32,
        pub pattern1_info: String,
        pub pattern2_info: String,
    }
}
