use crate::{Result, TextError};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use scirs2_core::rngs::StdRng;
use torsh_tensor::Tensor;

// ============================================================================
// Text Generation Utilities
// ============================================================================

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub min_length: usize,
    pub do_sample: bool,
    pub early_stopping: bool,
    pub num_beams: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub encoder_no_repeat_ngram_size: usize,
    pub bad_words_ids: Vec<Vec<u32>>,
    pub force_words_ids: Vec<Vec<u32>>,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub decoder_start_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 50,
            min_length: 0,
            do_sample: false,
            early_stopping: false,
            num_beams: 1,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            encoder_no_repeat_ngram_size: 0,
            bad_words_ids: Vec::new(),
            force_words_ids: Vec::new(),
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            decoder_start_token_id: None,
        }
    }
}

// ============================================================================
// Sampling Methods
// ============================================================================

pub struct TextSampler {
    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
    rng: Random<StdRng>,
}

impl Default for TextSampler {
    fn default() -> Self {
        Self {
            rng: Random::seed(42),
        }
    }
}

impl TextSampler {
    /// Greedy sampling - always select the token with highest probability
    pub fn greedy_sample(&self, logits: &Tensor) -> Result<u32> {
        let vocab_size = logits.shape().dims()[logits.shape().ndim() - 1];
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        // Simple implementation - could be optimized with tensor operations
        for i in 0..vocab_size {
            let val = logits.select(0, i as i64)?.item()?;
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx as u32)
    }

    /// Temperature sampling
    pub fn temperature_sample(&mut self, logits: &Tensor, temperature: f32) -> Result<u32> {
        if temperature <= 0.0 {
            return self.greedy_sample(logits);
        }

        // Apply temperature scaling
        let scaled_logits = logits.div_scalar(temperature)?;

        // Apply softmax to get probabilities
        let probs = scaled_logits.softmax(-1)?;

        self.multinomial_sample(&probs)
    }

    /// Top-k sampling
    pub fn top_k_sample(&mut self, logits: &Tensor, k: usize, temperature: f32) -> Result<u32> {
        let vocab_size = logits.shape().dims()[logits.shape().ndim() - 1];
        let k = k.min(vocab_size);

        // Get top-k indices and values
        let (top_values, top_indices) = self.get_top_k(logits, k)?;

        // Apply temperature scaling
        let scaled_values = if temperature > 0.0 {
            top_values.div_scalar(temperature)?
        } else {
            top_values
        };

        // Apply softmax
        let probs = scaled_values.softmax(-1)?;

        // Sample from the distribution
        let local_idx = self.multinomial_sample(&probs)?;

        // Convert back to original vocabulary index
        let original_idx = top_indices.select(0, local_idx as i64)?.item()?;
        Ok(original_idx as u32)
    }

    /// Top-p (nucleus) sampling
    pub fn top_p_sample(&mut self, logits: &Tensor, p: f32, temperature: f32) -> Result<u32> {
        // Apply temperature scaling
        let scaled_logits = if temperature > 0.0 {
            logits.div_scalar(temperature)?
        } else {
            logits.clone()
        };

        // Apply softmax to get probabilities
        let probs = scaled_logits.softmax(-1)?;

        // Get sorted probabilities and indices
        let (sorted_probs, sorted_indices) = self.sort_descending(&probs)?;

        // Calculate cumulative probabilities
        let cumsum = self.cumulative_sum(&sorted_probs)?;

        // Find the cutoff point where cumulative probability exceeds p
        let vocab_size = probs.shape().dims()[probs.shape().ndim() - 1];
        let mut cutoff = vocab_size;

        for i in 0..vocab_size {
            let cum_prob = cumsum.select(0, i as i64)?.item()?;
            if cum_prob > p {
                cutoff = i + 1;
                break;
            }
        }

        // Keep only the top-p tokens
        let nucleus_probs = sorted_probs.narrow(0, 0, cutoff)?;
        let nucleus_indices = sorted_indices.narrow(0, 0, cutoff)?;

        // Renormalize probabilities
        let sum_tensor = nucleus_probs.sum()?;
        let renormalized_probs = nucleus_probs.div(&sum_tensor)?;

        // Sample from the nucleus
        let local_idx = self.multinomial_sample(&renormalized_probs)?;

        // Convert back to original vocabulary index
        let original_idx = nucleus_indices.select(0, local_idx as i64)?.item()?;
        Ok(original_idx as u32)
    }

    /// Combined top-k and top-p sampling
    pub fn top_k_top_p_sample(
        &mut self,
        logits: &Tensor,
        k: Option<usize>,
        p: Option<f32>,
        temperature: f32,
    ) -> Result<u32> {
        let mut working_logits = logits.clone();

        // Apply top-k filtering first if specified
        if let Some(k_val) = k {
            let vocab_size = working_logits.shape().dims()[working_logits.shape().ndim() - 1];
            if k_val < vocab_size {
                let (top_values, top_indices) = self.get_top_k(&working_logits, k_val)?;

                // Create new logits tensor filled with negative infinity
                let mut new_logits_data = vec![f32::NEG_INFINITY; vocab_size];

                // Set top-k values
                for i in 0..k_val {
                    let idx = top_indices.select(0, i as i64)?.item()? as usize;
                    let val = top_values.select(0, i as i64)?.item()?;
                    if idx < vocab_size {
                        new_logits_data[idx] = val;
                    }
                }

                working_logits = Tensor::from_data(
                    new_logits_data,
                    working_logits.shape().dims().to_vec(),
                    torsh_core::device::DeviceType::Cpu,
                )?;
            }
        }

        // Apply top-p filtering if specified
        if let Some(p_val) = p {
            return self.top_p_sample(&working_logits, p_val, temperature);
        }

        // Otherwise use temperature sampling
        self.temperature_sample(&working_logits, temperature)
    }

    // Helper methods
    fn multinomial_sample(&mut self, probs: &Tensor) -> Result<u32> {
        let vocab_size = probs.shape().dims()[probs.shape().ndim() - 1];
        let random_val: f32 = self.rng.random();

        let mut cumulative = 0.0;
        for i in 0..vocab_size {
            let prob = probs.select(0, i as i64)?.item()?;
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(i as u32);
            }
        }

        // Fallback to last token if rounding errors occur
        Ok((vocab_size - 1) as u32)
    }

    fn get_top_k(&self, tensor: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        // Simplified implementation - in practice would use more efficient sorting
        let vocab_size = tensor.shape().dims()[tensor.shape().ndim() - 1];
        let mut values_and_indices: Vec<(f32, usize)> = Vec::new();

        for i in 0..vocab_size {
            let val = tensor.select(0, i as i64)?.item()?;
            values_and_indices.push((val, i));
        }

        values_and_indices
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        values_and_indices.truncate(k);

        let values: Vec<f32> = values_and_indices.iter().map(|(v, _)| *v).collect();
        let indices: Vec<f32> = values_and_indices.iter().map(|(_, i)| *i as f32).collect();

        let values_tensor = Tensor::from_vec(values, &[k])?.to_dtype(tensor.dtype())?;
        let indices_tensor = Tensor::from_vec(indices, &[k])?.to_dtype(tensor.dtype())?;

        Ok((values_tensor, indices_tensor))
    }

    fn sort_descending(&self, tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        let vocab_size = tensor.shape().dims()[tensor.shape().ndim() - 1];
        let mut values_and_indices: Vec<(f32, usize)> = Vec::new();

        for i in 0..vocab_size {
            let val = tensor.select(0, i as i64)?.item()?;
            values_and_indices.push((val, i));
        }

        values_and_indices
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let values: Vec<f32> = values_and_indices.iter().map(|(v, _)| *v).collect();
        let indices: Vec<f32> = values_and_indices.iter().map(|(_, i)| *i as f32).collect();

        let values_tensor = Tensor::from_vec(values, &[vocab_size])?.to_dtype(tensor.dtype())?;
        let indices_tensor = Tensor::from_vec(indices, &[vocab_size])?.to_dtype(tensor.dtype())?;

        Ok((values_tensor, indices_tensor))
    }

    fn cumulative_sum(&self, tensor: &Tensor) -> Result<Tensor> {
        let size = tensor.shape().dims()[tensor.shape().ndim() - 1];
        let mut cumsum = Vec::new();
        let mut running_sum = 0.0;

        for i in 0..size {
            let val = tensor.select(0, i as i64)?.item()?;
            running_sum += val;
            cumsum.push(running_sum);
        }

        Ok(Tensor::from_vec(cumsum, &[size])?.to_dtype(tensor.dtype())?)
    }
}

// ============================================================================
// Beam Search
// ============================================================================

#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub length: usize,
}

impl BeamHypothesis {
    pub fn new(tokens: Vec<u32>, score: f32) -> Self {
        let length = tokens.len();
        Self {
            tokens,
            score,
            length,
        }
    }

    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        self.score / (self.length as f32).powf(length_penalty)
    }
}

pub struct BeamSearchDecoder {
    num_beams: usize,
    max_length: usize,
    length_penalty: f32,
    early_stopping: bool,
    eos_token_id: Option<u32>,
}

impl BeamSearchDecoder {
    pub fn new(
        num_beams: usize,
        max_length: usize,
        length_penalty: f32,
        early_stopping: bool,
        eos_token_id: Option<u32>,
    ) -> Self {
        Self {
            num_beams,
            max_length,
            length_penalty,
            early_stopping,
            eos_token_id,
        }
    }

    pub fn search(
        &self,
        initial_tokens: Vec<u32>,
        vocab_size: usize,
        get_logits: impl Fn(&[u32]) -> Result<Tensor>,
    ) -> Result<Vec<BeamHypothesis>> {
        let mut beam_hypotheses = BeamHypothesesPool::new(
            self.num_beams,
            self.max_length,
            self.length_penalty,
            self.early_stopping,
        );

        // Initialize beams
        let mut beams: Vec<BeamHypothesis> = vec![BeamHypothesis::new(initial_tokens.clone(), 0.0)];

        for _step in 0..self.max_length {
            let mut all_candidates = Vec::new();

            for beam in &beams {
                if let Some(eos_id) = self.eos_token_id {
                    if beam.tokens.last() == Some(&eos_id) {
                        // This beam has ended, add to final hypotheses
                        beam_hypotheses.add(beam.clone());
                        continue;
                    }
                }

                // Get logits for current beam
                let logits = get_logits(&beam.tokens)?;
                let log_probs = logits.log_softmax(-1)?;

                // Get top-k candidates for this beam
                for token_id in 0..vocab_size.min(self.num_beams * 2) {
                    let token_log_prob = log_probs.select(0, token_id as i64)?.item()?;
                    let new_score = beam.score + token_log_prob;

                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token_id as u32);

                    all_candidates.push(BeamHypothesis::new(new_tokens, new_score));
                }
            }

            // Sort candidates by score and keep top num_beams
            all_candidates.sort_by(|a, b| {
                b.normalized_score(self.length_penalty)
                    .partial_cmp(&a.normalized_score(self.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            beams = all_candidates.into_iter().take(self.num_beams).collect();

            // Check early stopping
            if self.early_stopping
                && beam_hypotheses.is_done(
                    beams
                        .iter()
                        .map(|b| b.normalized_score(self.length_penalty))
                        .fold(f32::NEG_INFINITY, f32::max),
                )
            {
                break;
            }

            // Remove finished beams
            beams.retain(|beam| {
                if let Some(eos_id) = self.eos_token_id {
                    beam.tokens.last() != Some(&eos_id)
                } else {
                    true
                }
            });

            if beams.is_empty() {
                break;
            }
        }

        // Add remaining beams to hypotheses
        for beam in beams {
            beam_hypotheses.add(beam);
        }

        Ok(beam_hypotheses.finalize())
    }
}

struct BeamHypothesesPool {
    hypotheses: Vec<BeamHypothesis>,
    max_hypotheses: usize,
    max_length: usize,
    length_penalty: f32,
    early_stopping: bool,
}

impl BeamHypothesesPool {
    fn new(
        max_hypotheses: usize,
        max_length: usize,
        length_penalty: f32,
        early_stopping: bool,
    ) -> Self {
        Self {
            hypotheses: Vec::new(),
            max_hypotheses,
            max_length,
            length_penalty,
            early_stopping,
        }
    }

    fn add(&mut self, hypothesis: BeamHypothesis) {
        let score = hypothesis.normalized_score(self.length_penalty);

        // Insert in sorted order
        let insert_pos = self
            .hypotheses
            .binary_search_by(|h| {
                score
                    .partial_cmp(&h.normalized_score(self.length_penalty))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|e| e);

        self.hypotheses.insert(insert_pos, hypothesis);

        // Keep only top hypotheses
        if self.hypotheses.len() > self.max_hypotheses {
            self.hypotheses.truncate(self.max_hypotheses);
        }
    }

    fn is_done(&self, best_sum_logprobs: f32) -> bool {
        if !self.early_stopping {
            return false;
        }

        if self.hypotheses.len() < self.max_hypotheses {
            return false;
        }

        let worst_score = self
            .hypotheses
            .last()
            .map(|h| h.normalized_score(self.length_penalty))
            .unwrap_or(f32::NEG_INFINITY);
        let best_possible_score =
            best_sum_logprobs / (self.max_length as f32).powf(self.length_penalty);

        worst_score >= best_possible_score
    }

    fn finalize(mut self) -> Vec<BeamHypothesis> {
        self.hypotheses.sort_by(|a, b| {
            b.normalized_score(self.length_penalty)
                .partial_cmp(&a.normalized_score(self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.hypotheses
    }
}

// ============================================================================
// Repetition and Constraint Handling
// ============================================================================

pub struct RepetitionPenalty;

impl RepetitionPenalty {
    pub fn apply(logits: &Tensor, generated_tokens: &[u32], penalty: f32) -> Result<Tensor> {
        if penalty == 1.0 {
            return Ok(logits.clone());
        }

        let mut penalized_logits = logits.clone();

        // Apply penalty to repeated tokens
        for &token in generated_tokens {
            let current_logit = penalized_logits.select(0, token as i64)?.item()?;
            let penalized_value = if current_logit > 0.0 {
                current_logit / penalty
            } else {
                current_logit * penalty
            };

            // Use index_select and scatter to simulate index_put
            let _token_tensor = Tensor::from_vec(vec![token as i64], &[1])?;
            let _penalty_tensor = Tensor::scalar(penalized_value)?;

            // Simplified approach: directly modify the logit at the token position
            let vocab_size = penalized_logits.shape().dims()[0];
            let mut logits_vec = penalized_logits.to_vec()?;
            logits_vec[token as usize] = penalized_value;
            penalized_logits = Tensor::from_vec(logits_vec, &[vocab_size])?;
        }

        Ok(penalized_logits)
    }
}

pub struct NGramRepetitionFilter {
    no_repeat_ngram_size: usize,
}

impl NGramRepetitionFilter {
    pub fn new(no_repeat_ngram_size: usize) -> Self {
        Self {
            no_repeat_ngram_size,
        }
    }

    pub fn filter_logits(&self, logits: &Tensor, generated_tokens: &[u32]) -> Result<Tensor> {
        if self.no_repeat_ngram_size == 0 || generated_tokens.len() < self.no_repeat_ngram_size {
            return Ok(logits.clone());
        }

        let mut filtered_logits = logits.clone();
        let _vocab_size = logits.shape().dims()[logits.shape().ndim() - 1];

        // Extract n-grams from generated sequence
        let mut banned_tokens = std::collections::HashSet::new();

        for i in 0..generated_tokens.len() - self.no_repeat_ngram_size + 1 {
            let ngram = &generated_tokens[i..i + self.no_repeat_ngram_size - 1];

            // Check if current context matches this n-gram prefix
            let current_context =
                &generated_tokens[generated_tokens.len() - self.no_repeat_ngram_size + 1..];

            if ngram == current_context {
                // Ban the token that would complete this n-gram
                let banned_token = generated_tokens[i + self.no_repeat_ngram_size - 1];
                banned_tokens.insert(banned_token);
            }
        }

        // Set banned tokens to negative infinity
        let vocab_size = filtered_logits.shape().dims()[0];
        let mut logits_vec = filtered_logits.to_vec()?;
        for banned_token in banned_tokens {
            if (banned_token as usize) < logits_vec.len() {
                logits_vec[banned_token as usize] = f32::NEG_INFINITY;
            }
        }
        filtered_logits = Tensor::from_vec(logits_vec, &[vocab_size])?;

        Ok(filtered_logits)
    }
}

// ============================================================================
// Text Generation Pipeline
// ============================================================================

pub struct TextGenerator {
    sampler: TextSampler,
    beam_decoder: Option<BeamSearchDecoder>,
    repetition_penalty: RepetitionPenalty,
    ngram_filter: Option<NGramRepetitionFilter>,
}

impl TextGenerator {
    pub fn new(config: &GenerationConfig) -> Self {
        let beam_decoder = if config.num_beams > 1 {
            Some(BeamSearchDecoder::new(
                config.num_beams,
                config.max_length,
                config.length_penalty,
                config.early_stopping,
                config.eos_token_id,
            ))
        } else {
            None
        };

        let ngram_filter = if config.no_repeat_ngram_size > 0 {
            Some(NGramRepetitionFilter::new(config.no_repeat_ngram_size))
        } else {
            None
        };

        Self {
            sampler: TextSampler::default(),
            beam_decoder,
            repetition_penalty: RepetitionPenalty,
            ngram_filter,
        }
    }

    pub fn generate(
        &mut self,
        initial_tokens: Vec<u32>,
        vocab_size: usize,
        config: &GenerationConfig,
        get_logits: impl Fn(&[u32]) -> Result<Tensor> + Clone,
    ) -> Result<Vec<Vec<u32>>> {
        if config.num_beams > 1 {
            // Beam search
            if let Some(ref decoder) = self.beam_decoder {
                let hypotheses = decoder.search(initial_tokens, vocab_size, get_logits)?;
                Ok(hypotheses.into_iter().map(|h| h.tokens).collect())
            } else {
                Err(TextError::ModelError(
                    "Beam decoder not initialized".to_string(),
                ))
            }
        } else {
            // Sampling-based generation
            let result =
                self.generate_with_sampling(initial_tokens, vocab_size, config, get_logits)?;
            Ok(vec![result])
        }
    }

    fn generate_with_sampling(
        &mut self,
        mut tokens: Vec<u32>,
        _vocab_size: usize,
        config: &GenerationConfig,
        get_logits: impl Fn(&[u32]) -> Result<Tensor>,
    ) -> Result<Vec<u32>> {
        for _ in 0..config.max_length {
            // Get logits for current sequence
            let mut logits = get_logits(&tokens)?;

            // Apply repetition penalty
            if config.repetition_penalty != 1.0 {
                logits = RepetitionPenalty::apply(&logits, &tokens, config.repetition_penalty)?;
            }

            // Apply n-gram repetition filter
            if let Some(ref filter) = self.ngram_filter {
                logits = filter.filter_logits(&logits, &tokens)?;
            }

            // Sample next token
            let next_token = if config.do_sample {
                self.sampler.top_k_top_p_sample(
                    &logits,
                    config.top_k,
                    config.top_p,
                    config.temperature,
                )?
            } else {
                self.sampler.greedy_sample(&logits)?
            };

            tokens.push(next_token);

            // Check for EOS token
            if let Some(eos_id) = config.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }

            // Check minimum length
            if tokens.len() >= config.min_length {
                if let Some(eos_id) = config.eos_token_id {
                    if next_token == eos_id {
                        break;
                    }
                }
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::{device::DeviceType as Device, dtype::DType};

    #[test]
    fn test_text_sampler_creation() {
        let _sampler = TextSampler::default();
        // Just test that it doesn't panic
    }

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 50);
        assert_eq!(config.num_beams, 1);
        assert!(!config.do_sample);
    }

    #[test]
    fn test_beam_hypothesis() {
        let tokens = vec![1, 2, 3];
        let score = -1.5;
        let hypothesis = BeamHypothesis::new(tokens.clone(), score);

        assert_eq!(hypothesis.tokens, tokens);
        assert_eq!(hypothesis.score, score);
        assert_eq!(hypothesis.length, 3);
    }

    #[test]
    fn test_greedy_sampling() {
        let _device = Device::Cpu;
        let dtype = DType::F32;

        // Create a simple logits tensor where token 2 has highest probability
        let logits = Tensor::from_vec(vec![0.1, 0.2, 0.9, 0.3], &[4])
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let sampler = TextSampler::default();
        let result = sampler.greedy_sample(&logits).unwrap();

        assert_eq!(result, 2); // Should select the token with highest logit
    }
}
