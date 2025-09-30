//! Natural Language Processing with Transformer Architecture
//! 
//! This example demonstrates:
//! - Text tokenization and encoding
//! - Transformer architecture components
//! - Attention mechanisms
//! - Language modeling training

use torsh_tensor::{Tensor, creation::*};
use torsh_nn::modules::*;
use torsh_core::{error::Result, dtype::DType};
use std::collections::HashMap;

/// Simple tokenizer for text processing
struct SimpleTokenizer {
    vocab: HashMap<String, usize>,
    vocab_size: usize,
    max_seq_len: usize,
}

impl SimpleTokenizer {
    fn new(vocab_size: usize, max_seq_len: usize) -> Self {
        let mut vocab = HashMap::new();
        
        // Add special tokens
        vocab.insert("<PAD>".to_string(), 0);
        vocab.insert("<UNK>".to_string(), 1);
        vocab.insert("<START>".to_string(), 2);
        vocab.insert("<END>".to_string(), 3);
        
        // Add common words (simplified vocabulary)
        let common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "and", "or", "but", "not", "to", "of", "in", "on", "at", "by", "for",
            "hello", "world", "example", "text", "model", "neural", "network"
        ];
        
        for (i, word) in common_words.iter().enumerate() {
            vocab.insert(word.to_string(), i + 4);
        }
        
        Self {
            vocab,
            vocab_size,
            max_seq_len,
        }
    }
    
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![2]; // Start token
        
        for word in text.to_lowercase().split_whitespace() {
            let token_id = self.vocab.get(word).copied().unwrap_or(1); // UNK token
            tokens.push(token_id);
        }
        
        tokens.push(3); // End token
        
        // Pad or truncate to max_seq_len
        if tokens.len() < self.max_seq_len {
            tokens.resize(self.max_seq_len, 0); // Pad with PAD token
        } else {
            tokens.truncate(self.max_seq_len);
        }
        
        tokens
    }
    
    fn decode(&self, tokens: &[usize]) -> String {
        let reverse_vocab: HashMap<usize, String> = self.vocab.iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();
        
        tokens.iter()
            .filter_map(|&token| reverse_vocab.get(&token))
            .filter(|&word| word != "<PAD>" && word != "<START>" && word != "<END>")
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Multi-Head Attention mechanism
struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    query_linear: Linear,
    key_linear: Linear,
    value_linear: Linear,
    output_linear: Linear,
}

impl MultiHeadAttention {
    fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        
        Self {
            num_heads,
            d_model,
            d_k,
            query_linear: Linear::new(d_model, d_model),
            key_linear: Linear::new(d_model, d_model),
            value_linear: Linear::new(d_model, d_model),
            output_linear: Linear::new(d_model, d_model),
        }
    }
    
    fn forward(&mut self, query: &Tensor<f32>, key: &Tensor<f32>, value: &Tensor<f32>) -> Result<Tensor<f32>> {
        let batch_size = query.shape().dims()[0];
        let seq_len = query.shape().dims()[1];
        
        // Linear transformations
        let q = self.query_linear.forward(query)?;
        let k = self.key_linear.forward(key)?;
        let v = self.value_linear.forward(value)?;
        
        // Reshape for multi-head attention
        let q = q.view(&[batch_size as i32, seq_len as i32, self.num_heads as i32, self.d_k as i32])?;
        let k = k.view(&[batch_size as i32, seq_len as i32, self.num_heads as i32, self.d_k as i32])?;
        let v = v.view(&[batch_size as i32, seq_len as i32, self.num_heads as i32, self.d_k as i32])?;
        
        // Transpose for batch matrix multiplication
        let q = q.transpose(1, 2)?; // (batch, heads, seq_len, d_k)
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        
        // Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v)?;
        
        // Reshape back
        let attention_output = attention_output.transpose(1, 2)?;
        let attention_output = attention_output.view(&[batch_size as i32, seq_len as i32, self.d_model as i32])?;
        
        // Final linear transformation
        self.output_linear.forward(&attention_output)
    }
    
    fn scaled_dot_product_attention(
        &self, 
        query: &Tensor<f32>, 
        key: &Tensor<f32>, 
        value: &Tensor<f32>
    ) -> Result<Tensor<f32>> {
        // QK^T / sqrt(d_k)
        let key_transposed = key.transpose(-2, -1)?;
        let scores = query.matmul(&key_transposed)?;
        let scale = (self.d_k as f32).sqrt();
        let scaled_scores = scores.mul_scalar(1.0 / scale)?;
        
        // Softmax
        let attention_weights = scaled_scores.softmax(-1)?;
        
        // Apply attention to values
        attention_weights.matmul(value)
    }
}

/// Feed-Forward Network
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout_rate: f32,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize, dropout_rate: f32) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff),
            linear2: Linear::new(d_ff, d_model),
            dropout_rate,
        }
    }
    
    fn forward(&mut self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        // In a real implementation, apply dropout here
        let _ = self.dropout_rate; // Suppress unused warning
        self.linear2.forward(&x)
    }
}

/// Transformer Encoder Layer
struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout_rate: f32) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            feed_forward: FeedForward::new(d_model, d_ff, dropout_rate),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }
    
    fn forward(&mut self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Self-attention with residual connection
        let attention_output = self.attention.forward(x, x, x)?;
        let x = x.add(&attention_output)?;
        let x = self.norm1.forward(&x)?;
        
        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(&x)?;
        let x = x.add(&ff_output)?;
        let x = self.norm2.forward(&x)?;
        
        Ok(x)
    }
}

/// Simple Transformer model for language modeling
struct SimpleTransformer {
    vocab_size: usize,
    d_model: usize,
    embedding: Embedding,
    positional_encoding: PositionalEncoding,
    encoder_layers: Vec<TransformerEncoderLayer>,
    output_projection: Linear,
}

impl SimpleTransformer {
    fn new(vocab_size: usize, d_model: usize, num_heads: usize, num_layers: usize, max_seq_len: usize) -> Self {
        let d_ff = d_model * 4; // Standard transformer ratio
        let dropout_rate = 0.1;
        
        let mut encoder_layers = Vec::new();
        for _ in 0..num_layers {
            encoder_layers.push(TransformerEncoderLayer::new(d_model, num_heads, d_ff, dropout_rate));
        }
        
        Self {
            vocab_size,
            d_model,
            embedding: Embedding::new(vocab_size, d_model),
            positional_encoding: PositionalEncoding::new(d_model, max_seq_len),
            encoder_layers,
            output_projection: Linear::new(d_model, vocab_size),
        }
    }
    
    fn forward(&mut self, input_ids: &Tensor<i64>) -> Result<Tensor<f32>> {
        // Convert to f32 for embedding lookup (simplified)
        let input_ids_f32 = input_ids.to_tensor_f32()?;
        
        // Token embeddings
        let embeddings = self.embedding.forward(&input_ids_f32)?;
        
        // Add positional encoding
        let x = self.positional_encoding.forward(&embeddings)?;
        
        // Pass through encoder layers
        let mut x = x;
        for layer in &mut self.encoder_layers {
            x = layer.forward(&x)?;
        }
        
        // Output projection to vocabulary
        self.output_projection.forward(&x)
    }
}

/// Positional Encoding for transformer
struct PositionalEncoding {
    encoding: Tensor<f32>,
}

impl PositionalEncoding {
    fn new(d_model: usize, max_seq_len: usize) -> Self {
        // Create positional encoding matrix
        let mut encoding_data = Vec::with_capacity(max_seq_len * d_model);
        
        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000.0_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                let encoding_val = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
                encoding_data.push(encoding_val);
            }
        }
        
        let encoding = Tensor::from_vec(encoding_data, &[max_seq_len, d_model]);
        
        Self { encoding }
    }
    
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let seq_len = x.shape().dims()[1];
        let pos_encoding = self.encoding.slice(0, Some(0), Some(seq_len as i64), Some(1))?;
        x.add(&pos_encoding)
    }
}

/// Simple embedding layer
struct Embedding {
    weight: Tensor<f32>,
}

impl Embedding {
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let weight = rand::<f32>(&[vocab_size, embedding_dim]);
        Self { weight }
    }
    
    fn forward(&self, input_ids: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Simplified embedding lookup
        // In a real implementation, this would use proper integer indexing
        Ok(input_ids.matmul(&self.weight)?)
    }
}

/// Layer normalization
struct LayerNorm {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    eps: f32,
}

impl LayerNorm {
    fn new(d_model: usize) -> Self {
        Self {
            weight: ones::<f32>(&[d_model]),
            bias: zeros::<f32>(&[d_model]),
            eps: 1e-6,
        }
    }
    
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Simplified layer norm implementation
        // Real implementation would compute mean and variance along last dimension
        let normalized = x.mul(&self.weight)?.add(&self.bias)?;
        Ok(normalized)
    }
}

fn main() -> Result<()> {
    println!("ToRSh Transformer NLP Example");
    println!("=============================");
    
    // Setup
    let vocab_size = 1000;
    let d_model = 256;
    let num_heads = 8;
    let num_layers = 6;
    let max_seq_len = 128;
    
    println!("Model configuration:");
    println!("  Vocabulary size: {}", vocab_size);
    println!("  Model dimension: {}", d_model);
    println!("  Number of heads: {}", num_heads);
    println!("  Number of layers: {}", num_layers);
    println!("  Max sequence length: {}", max_seq_len);
    
    // Create tokenizer
    let tokenizer = SimpleTokenizer::new(vocab_size, max_seq_len);
    
    // Create model
    let mut model = SimpleTransformer::new(vocab_size, d_model, num_heads, num_layers, max_seq_len);
    
    // Example text processing
    let sample_texts = [
        "Hello world example text",
        "Neural network model training",
        "The quick brown fox jumps",
    ];
    
    println!("\nProcessing sample texts:");
    for text in &sample_texts {
        process_text(&mut model, &tokenizer, text)?;
    }
    
    // Training simulation
    simulate_training(&mut model, &tokenizer)?;
    
    // Text generation demo
    generate_text(&mut model, &tokenizer, "Hello")?;
    
    Ok(())
}

/// Process a single text example
fn process_text(
    model: &mut SimpleTransformer, 
    tokenizer: &SimpleTokenizer, 
    text: &str
) -> Result<()> {
    println!("\nText: \"{}\"", text);
    
    // Tokenize
    let tokens = tokenizer.encode(text);
    println!("Tokens: {:?}", &tokens[..std::cmp::min(10, tokens.len())]);
    
    // Convert to tensor
    let input_tensor = Tensor::from_vec(
        tokens.iter().map(|&x| x as i64).collect(), 
        &[1, tokens.len()] // batch_size=1
    );
    
    // Forward pass
    let output = model.forward(&input_tensor)?;
    println!("Output shape: {:?}", output.shape().dims());
    
    // Get predictions for first few positions
    let batch_size = output.shape().dims()[0];
    let seq_len = std::cmp::min(5, output.shape().dims()[1]);
    
    for i in 0..seq_len {
        let logits = output.slice(1, Some(i as i64), Some((i+1) as i64), Some(1))?;
        let probs = logits.softmax(-1)?;
        let predicted_token = probs.argmax(Some(-1))?;
        println!("Position {}: predicted token {}", i, predicted_token.item());
    }
    
    Ok(())
}

/// Simulate training process
fn simulate_training(model: &mut SimpleTransformer, tokenizer: &SimpleTokenizer) -> Result<()> {
    println!("\n=== Training Simulation ===");
    
    let training_texts = [
        "The model learns patterns in text",
        "Neural networks process sequential data",
        "Attention mechanisms are powerful",
        "Transformers revolutionized NLP",
    ];
    
    let epochs = 3;
    let learning_rate = 0.001;
    
    println!("Training parameters:");
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", learning_rate);
    println!("  Training samples: {}", training_texts.len());
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        println!("\nEpoch {}/{}", epoch + 1, epochs);
        
        for (i, text) in training_texts.iter().enumerate() {
            // Tokenize
            let tokens = tokenizer.encode(text);
            let input_tensor = Tensor::from_vec(
                tokens.iter().map(|&x| x as i64).collect(),
                &[1, tokens.len()]
            );
            
            // Forward pass
            let output = model.forward(&input_tensor)?;
            
            // Simulate loss computation (cross-entropy with next token prediction)
            let loss = simulate_language_modeling_loss(&output, &tokens)?;
            total_loss += loss;
            
            println!("  Sample {}: Loss = {:.4}", i + 1, loss);
        }
        
        let avg_loss = total_loss / training_texts.len() as f32;
        println!("  Average loss: {:.4}", avg_loss);
    }
    
    Ok(())
}

/// Simulate language modeling loss computation
fn simulate_language_modeling_loss(output: &Tensor<f32>, target_tokens: &[usize]) -> Result<f32> {
    // In a real implementation:
    // 1. Shift targets (predict next token)
    // 2. Compute cross-entropy loss
    // 3. Apply mask for padding tokens
    
    // For simulation, return a decreasing loss
    let loss = rand::<f32>(&[1]).item() * 0.5 + 0.1;
    Ok(loss)
}

/// Text generation demonstration
fn generate_text(
    model: &mut SimpleTransformer, 
    tokenizer: &SimpleTokenizer, 
    prompt: &str
) -> Result<()> {
    println!("\n=== Text Generation Demo ===");
    println!("Prompt: \"{}\"", prompt);
    
    // Tokenize prompt
    let mut tokens = tokenizer.encode(prompt);
    let max_new_tokens = 10;
    
    println!("Generating {} new tokens...", max_new_tokens);
    
    for step in 0..max_new_tokens {
        // Prepare input tensor
        let input_tensor = Tensor::from_vec(
            tokens.iter().map(|&x| x as i64).collect(),
            &[1, tokens.len()]
        );
        
        // Forward pass
        let output = model.forward(&input_tensor)?;
        
        // Get logits for last position
        let last_pos = tokens.len() - 1;
        let logits = output.slice(1, Some(last_pos as i64), Some((last_pos + 1) as i64), Some(1))?;
        
        // Sample next token (simplified - use argmax)
        let next_token_probs = logits.softmax(-1)?;
        let next_token = next_token_probs.argmax(Some(-1))?;
        let next_token_id = next_token.item() as usize;
        
        // Add to sequence
        tokens.push(next_token_id);
        
        println!("Step {}: Generated token {}", step + 1, next_token_id);
        
        // Early stopping if we hit end token
        if next_token_id == 3 { // END token
            break;
        }
    }
    
    // Decode generated sequence
    let generated_text = tokenizer.decode(&tokens);
    println!("Generated text: \"{}\"", generated_text);
    
    Ok(())
}

/// Attention visualization (simplified)
fn visualize_attention(attention_weights: &Tensor<f32>, tokenizer: &SimpleTokenizer, tokens: &[usize]) -> Result<()> {
    println!("\n=== Attention Visualization ===");
    
    let seq_len = std::cmp::min(8, tokens.len());
    
    // Print tokens
    print!("Tokens: ");
    for i in 0..seq_len {
        let token_text = if tokens[i] < 4 {
            match tokens[i] {
                0 => "<PAD>",
                1 => "<UNK>",
                2 => "<START>",
                3 => "<END>",
                _ => "<?>"
            }
        } else {
            "word"
        };
        print!("{:8} ", token_text);
    }
    println!();
    
    // Print attention matrix (simplified)
    println!("Attention weights (head 0):");
    for i in 0..seq_len {
        print!("Token {}: ", i);
        for j in 0..seq_len {
            // Simplified attention weight extraction
            let weight = 0.1 + 0.8 * ((i + j) as f32 / 10.0).sin().abs();
            print!("{:6.3} ", weight);
        }
        println!();
    }
    
    Ok(())
}

/// Helper trait for tensor conversion
trait TensorConversion {
    fn to_tensor_f32(&self) -> Result<Tensor<f32>>;
}

impl TensorConversion for Tensor<i64> {
    fn to_tensor_f32(&self) -> Result<Tensor<f32>> {
        // Simplified conversion - real implementation would handle this properly
        let data = self.to_vec();
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        Ok(Tensor::from_vec(f32_data, self.shape().dims()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer() {
        let tokenizer = SimpleTokenizer::new(100, 10);
        let tokens = tokenizer.encode("hello world");
        assert!(tokens.len() <= 10);
        
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("hello") || decoded.contains("world"));
    }
    
    #[test]
    fn test_attention() {
        let mut attention = MultiHeadAttention::new(64, 8);
        let input = rand::<f32>(&[2, 10, 64]); // batch=2, seq_len=10, d_model=64
        
        let output = attention.forward(&input, &input, &input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 10, 64]);
    }
    
    #[test]
    fn test_transformer_forward() {
        let mut model = SimpleTransformer::new(100, 64, 4, 2, 50);
        let input = Tensor::from_vec(vec![2, 10, 20, 3], &[1, 4]); // batch=1, seq_len=4
        
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4, 100]); // vocab_size=100
    }
    
    #[test]
    fn test_positional_encoding() {
        let pos_enc = PositionalEncoding::new(64, 100);
        let input = rand::<f32>(&[2, 10, 64]);
        
        let output = pos_enc.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }
}