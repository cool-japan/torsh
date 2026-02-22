//! NLP model utilities and helper functions

use super::{preprocessing::*, types::*};
use crate::ModelResult;

/// NLP model utilities
pub struct NlpModelUtils;

impl NlpModelUtils {
    /// Get recommended preprocessor for a model
    pub fn get_preprocessor(model_name: &str) -> TextPreprocessor {
        match model_name {
            name if name.contains("bert") || name.contains("roberta") => TextPreprocessor::bert(),
            name if name.contains("gpt") => TextPreprocessor::gpt(),
            name if name.contains("t5") => TextPreprocessor::t5(),
            name if name.contains("longformer") => TextPreprocessor::custom(4096, 0),
            name if name.contains("bigbird") => TextPreprocessor::custom(4096, 0),
            _ => TextPreprocessor::bert(), // Default to BERT-style
        }
    }

    /// Get model variant by name
    pub fn get_model_variant(name: &str) -> Option<NlpModelVariant> {
        let models = get_common_nlp_models();
        models.into_iter().find(|m| m.variant == name)
    }

    /// List models by architecture
    pub fn list_models_by_architecture(arch: NlpArchitecture) -> Vec<NlpModelVariant> {
        let models = get_common_nlp_models();
        models
            .into_iter()
            .filter(|m| m.architecture == arch)
            .collect()
    }

    /// Get models by parameter count range
    pub fn get_models_by_size(min_params: u64, max_params: u64) -> Vec<NlpModelVariant> {
        let models = get_common_nlp_models();
        models
            .into_iter()
            .filter(|m| m.parameters >= min_params && m.parameters <= max_params)
            .collect()
    }

    /// Get suitable models for a specific task
    pub fn get_models_for_task(task: NlpTask) -> Vec<NlpModelVariant> {
        let suitable_archs = task.suitable_architectures();
        let models = get_common_nlp_models();

        models
            .into_iter()
            .filter(|m| suitable_archs.contains(&m.architecture))
            .collect()
    }

    /// Apply softmax to logits
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum).collect()
    }

    /// Apply temperature scaling to logits
    pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        logits.iter().map(|&x| x / temperature).collect()
    }

    /// Apply top-p (nucleus) sampling
    pub fn apply_top_p_sampling(logits: &[f32], p: f32) -> Vec<f32> {
        let mut probs = Self::softmax(logits);
        let mut indexed_probs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();

        // Sort by probability descending
        indexed_probs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("probabilities should be comparable")
        });

        let mut cumulative_prob = 0.0;
        let mut cutoff_index = indexed_probs.len();

        for (idx, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= p {
                cutoff_index = idx + 1;
                break;
            }
        }

        // Zero out probabilities beyond cutoff
        for (original_idx, _) in indexed_probs.iter().skip(cutoff_index) {
            probs[*original_idx] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            probs.iter().map(|&x| x / sum).collect()
        } else {
            probs
        }
    }

    /// Get top-k tokens from logits
    pub fn get_top_k_tokens(
        logits: &[f32],
        k: usize,
        tokenizer: Option<&dyn Tokenizer>,
    ) -> Vec<(u32, f32, Option<String>)> {
        let mut predictions: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &score)| (i as u32, score))
            .collect();

        // Sort by score descending
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("scores should be comparable"));

        // Take top-k
        predictions
            .into_iter()
            .take(k)
            .map(|(token_id, score)| {
                let token_text = tokenizer.and_then(|t| t.decode(&[token_id]).ok());
                (token_id, score, token_text)
            })
            .collect()
    }

    /// Simple text generation (greedy decoding)
    pub fn generate_text_greedy(
        initial_tokens: &[u32],
        model_fn: &dyn Fn(&[u32]) -> ModelResult<Vec<f32>>,
        max_length: usize,
        eos_token_id: Option<u32>,
    ) -> ModelResult<Vec<u32>> {
        let mut tokens = initial_tokens.to_vec();

        for _ in 0..(max_length - initial_tokens.len()) {
            // Get model predictions
            let logits = model_fn(&tokens)?;

            // Get most likely next token (greedy)
            let next_token_id = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits should be comparable"))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            tokens.push(next_token_id);

            // Check for end-of-sequence
            if let Some(eos_id) = eos_token_id {
                if next_token_id == eos_id {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Text generation with sampling (multinomial)
    pub fn generate_text_sampling(
        initial_tokens: &[u32],
        model_fn: &dyn Fn(&[u32]) -> ModelResult<Vec<f32>>,
        max_length: usize,
        temperature: f32,
        top_p: Option<f32>,
        eos_token_id: Option<u32>,
    ) -> ModelResult<Vec<u32>> {
        let mut tokens = initial_tokens.to_vec();

        for _ in 0..(max_length - initial_tokens.len()) {
            // Get model predictions
            let mut logits = model_fn(&tokens)?;

            // Apply temperature scaling
            if temperature != 1.0 {
                logits = Self::apply_temperature(&logits, temperature);
            }

            // Apply top-p sampling if specified
            if let Some(p) = top_p {
                logits = Self::apply_top_p_sampling(&logits, p);
            }

            // Convert to probabilities
            let probs = Self::softmax(&logits);

            // Sample from the distribution (simplified - would use proper sampling in real implementation)
            let next_token_id = probs
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .expect("probabilities should be comparable")
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            tokens.push(next_token_id);

            // Check for end-of-sequence
            if let Some(eos_id) = eos_token_id {
                if next_token_id == eos_id {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Calculate perplexity from cross-entropy loss
    pub fn calculate_perplexity(cross_entropy_loss: f32) -> f32 {
        cross_entropy_loss.exp()
    }

    /// Calculate BLEU score approximation (simplified)
    pub fn calculate_bleu_score_simple(
        reference: &[u32],
        hypothesis: &[u32],
        n_gram: usize,
    ) -> f32 {
        if reference.is_empty() || hypothesis.is_empty() {
            return 0.0;
        }

        // Simple n-gram overlap calculation
        let ref_ngrams = Self::extract_ngrams(reference, n_gram);
        let hyp_ngrams = Self::extract_ngrams(hypothesis, n_gram);

        let mut matches = 0;
        for ngram in &hyp_ngrams {
            if ref_ngrams.contains(ngram) {
                matches += 1;
            }
        }

        if hyp_ngrams.is_empty() {
            0.0
        } else {
            matches as f32 / hyp_ngrams.len() as f32
        }
    }

    /// Extract n-grams from token sequence
    fn extract_ngrams(tokens: &[u32], n: usize) -> Vec<Vec<u32>> {
        if tokens.len() < n {
            return vec![];
        }

        (0..=tokens.len() - n)
            .map(|i| tokens[i..i + n].to_vec())
            .collect()
    }
}
