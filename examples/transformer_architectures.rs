//! Comprehensive examples of advanced transformer architectures in ToRSh
//!
//! This example demonstrates usage of GPT, BERT, and T5 models for various tasks
//! including text generation, classification, and sequence-to-sequence modeling.

use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_nn::{Module, Parameter};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;
use torsh_text::{
    BertForSequenceClassification, BertModel, GPTForCausalLM, GPTModel, T5ForConditionalGeneration,
    T5Model, TextModelConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Advanced Transformer Architectures in ToRSh");
    println!("================================================\n");

    // Set device (CPU for this example)
    let device = DeviceType::Cpu;

    // Run all transformer demonstrations
    demonstrate_gpt_models(device)?;
    demonstrate_bert_models(device)?;
    demonstrate_t5_models(device)?;
    demonstrate_model_comparison(device)?;

    println!("✅ All transformer architecture examples completed successfully!");
    Ok(())
}

/// Demonstrate GPT models for text generation
fn demonstrate_gpt_models(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPT Models Demonstration");
    println!("============================\n");

    // Test different GPT configurations
    let configs = vec![
        ("GPT-2 Small", TextModelConfig::gpt2_small()),
        ("GPT-2 Medium", TextModelConfig::gpt2_medium()),
        ("GPT-2 Large", TextModelConfig::gpt2_large()),
    ];

    for (name, config) in configs {
        println!("📝 Testing {} Configuration:", name);
        println!("   Vocabulary size: {}", config.vocab_size);
        println!("   Hidden dimension: {}", config.hidden_dim);
        println!("   Number of layers: {}", config.num_layers);
        println!("   Number of heads: {}", config.num_heads);
        println!("   Max sequence length: {}", config.max_position_embeddings);

        // Create GPT model
        let mut model = GPTModel::new(config.clone());
        let mut causal_lm = GPTForCausalLM::new(config.clone());

        // Create sample input (batch_size=2, seq_len=10)
        let input_ids: Tensor<f32> = rand(&[2, 10]);
        let input_ids = input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;

        println!("   Input shape: {:?}", input_ids.shape().dims());

        // Test forward pass
        let base_output = model.forward(&input_ids)?;
        println!("   Base model output shape: {:?}", base_output.shape().dims());

        let lm_output = causal_lm.forward(&input_ids)?;
        println!("   Language model output shape: {:?}", lm_output.shape().dims());

        // Test parameter counting
        let params = model.parameters();
        let total_params: usize = params
            .values()
            .map(|p| p.data().numel())
            .sum();
        println!("   Total parameters: {:.2}M", total_params as f32 / 1_000_000.0);

        // Test generation capability (simplified)
        println!("   Testing text generation...");
        let generated = causal_lm.generate(
            &input_ids.narrow(0, 0, 1).unwrap(), // Take first sample
            50257, // EOS token for GPT-2
            20,    // max_length
            1.0,   // temperature
            None,  // top_k
            None,  // top_p
        )?;
        println!("   Generated sequence shape: {:?}", generated.shape().dims());

        println!("   ✅ {} test completed\n", name);
    }

    Ok(())
}

/// Demonstrate BERT models for text understanding
fn demonstrate_bert_models(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 BERT Models Demonstration");
    println!("=============================\n");

    // Test different BERT configurations
    let configs = vec![
        ("BERT Base", TextModelConfig::bert_base()),
        ("BERT Large", TextModelConfig::bert_large()),
    ];

    for (name, config) in configs {
        println!("📚 Testing {} Configuration:", name);
        println!("   Vocabulary size: {}", config.vocab_size);
        println!("   Hidden dimension: {}", config.hidden_dim);
        println!("   Number of layers: {}", config.num_layers);
        println!("   Number of heads: {}", config.num_heads);
        println!("   Max sequence length: {}", config.max_position_embeddings);

        // Create BERT models
        let mut bert_model = BertModel::new(config.clone(), device)?;
        let mut bert_classifier = BertForSequenceClassification::new(config.clone(), 3, device)?; // 3 classes

        // Create sample input
        let input_ids: Tensor<f32> = rand(&[2, 128]); // batch_size=2, seq_len=128
        let input_ids = input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;

        // Create token type ids (for sentence pair tasks)
        let token_type_ids: Tensor<f32> = zeros(&[2, 128]);

        println!("   Input shape: {:?}", input_ids.shape().dims());

        // Test BERT base model
        let base_output = bert_model.forward(&input_ids)?;
        println!("   Base model output shape: {:?}", base_output.shape().dims());

        // Test BERT with token types
        let (sequence_output, pooled_output) = bert_model.forward_with_type_ids(
            &input_ids,
            Some(&token_type_ids),
            None, // attention_mask
        )?;
        println!("   Sequence output shape: {:?}", sequence_output.shape().dims());
        if let Some(pooled) = pooled_output {
            println!("   Pooled output shape: {:?}", pooled.shape().dims());
        }

        // Test classification model
        let classification_output = bert_classifier.forward(&input_ids)?;
        println!("   Classification output shape: {:?}", classification_output.shape().dims());

        // Test parameter counting
        let params = bert_model.parameters();
        let total_params: usize = params
            .values()
            .map(|p| p.data().numel())
            .sum();
        println!("   Total parameters: {:.2}M", total_params as f32 / 1_000_000.0);

        println!("   ✅ {} test completed\n", name);
    }

    Ok(())
}

/// Demonstrate T5 models for text-to-text generation
fn demonstrate_t5_models(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 T5 Models Demonstration");
    println!("===========================\n");

    let config = TextModelConfig::t5_small();
    println!("📋 Testing T5 Small Configuration:");
    println!("   Vocabulary size: {}", config.vocab_size);
    println!("   Hidden dimension: {}", config.hidden_dim);
    println!("   Number of layers: {}", config.num_layers);
    println!("   Number of heads: {}", config.num_heads);
    println!("   Max sequence length: {}", config.max_position_embeddings);

    // Create T5 models
    let mut t5_model = T5Model::new(config.clone(), device)?;
    let mut t5_generation = T5ForConditionalGeneration::new(config.clone(), device)?;

    // Create sample inputs
    let encoder_input_ids: Tensor<f32> = rand(&[2, 64]); // batch_size=2, encoder_seq_len=64
    let encoder_input_ids = encoder_input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;
    
    let decoder_input_ids: Tensor<f32> = rand(&[2, 32]); // batch_size=2, decoder_seq_len=32
    let decoder_input_ids = decoder_input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;

    println!("   Encoder input shape: {:?}", encoder_input_ids.shape().dims());
    println!("   Decoder input shape: {:?}", decoder_input_ids.shape().dims());

    // Test encoder-only mode
    let encoder_output = t5_model.encode(&encoder_input_ids, None)?;
    println!("   Encoder output shape: {:?}", encoder_output.shape().dims());

    // Test encoder-decoder mode
    let (enc_out, dec_out) = t5_model.forward_encoder_decoder(
        &encoder_input_ids,
        &decoder_input_ids,
        None, // encoder attention mask
        None, // decoder attention mask
    )?;
    println!("   Encoder-decoder encoder output shape: {:?}", enc_out.shape().dims());
    println!("   Encoder-decoder decoder output shape: {:?}", dec_out.shape().dims());

    // Test conditional generation model
    let generation_output = t5_generation.forward(&encoder_input_ids)?;
    println!("   Generation model output shape: {:?}", generation_output.shape().dims());

    // Test generation capability
    println!("   Testing conditional generation...");
    let generated = t5_generation.generate(
        &encoder_input_ids.narrow(0, 0, 1).unwrap(), // Take first sample
        0,   // decoder_start_token_id (usually 0 for T5)
        50,  // max_length
        1,   // num_beams (greedy search)
        1.0, // temperature
    )?;
    println!("   Generated sequence shape: {:?}", generated.shape().dims());

    // Test parameter counting
    let params = t5_model.parameters();
    let total_params: usize = params
        .values()
        .map(|p| p.data().numel())
        .sum();
    println!("   Total parameters: {:.2}M", total_params as f32 / 1_000_000.0);

    println!("   ✅ T5 test completed\n");

    Ok(())
}

/// Compare different transformer architectures
fn demonstrate_model_comparison(device: DeviceType) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️  Model Architecture Comparison");
    println!("==================================\n");

    // Create models with similar sizes for comparison
    let gpt_config = TextModelConfig::gpt2_small();
    let bert_config = TextModelConfig::bert_base();
    let t5_config = TextModelConfig::t5_small();

    let gpt_model = GPTModel::new(gpt_config.clone());
    let bert_model = BertModel::new(bert_config.clone(), device)?;
    let t5_model = T5Model::new(t5_config.clone(), device)?;

    // Count parameters for each model
    fn count_parameters(params: &HashMap<String, Parameter>) -> usize {
        params.values().map(|p| p.data().numel()).sum()
    }

    let gpt_params = count_parameters(&gpt_model.parameters());
    let bert_params = count_parameters(&bert_model.parameters());
    let t5_params = count_parameters(&t5_model.parameters());

    println!("📊 Parameter Comparison:");
    println!("   GPT-2 Small:  {:.2}M parameters", gpt_params as f32 / 1_000_000.0);
    println!("   BERT Base:    {:.2}M parameters", bert_params as f32 / 1_000_000.0);
    println!("   T5 Small:     {:.2}M parameters", t5_params as f32 / 1_000_000.0);

    println!("\n🏗️  Architecture Comparison:");
    println!("┌─────────────┬──────────┬────────────┬──────────────┬──────────────┐");
    println!("│ Model       │ Layers   │ Hidden Dim │ Vocab Size   │ Max Seq Len  │");
    println!("├─────────────┼──────────┼────────────┼──────────────┼──────────────┤");
    println!("│ GPT-2 Small │ {:8} │ {:10} │ {:12} │ {:12} │", 
             gpt_config.num_layers, gpt_config.hidden_dim, 
             gpt_config.vocab_size, gpt_config.max_position_embeddings);
    println!("│ BERT Base   │ {:8} │ {:10} │ {:12} │ {:12} │", 
             bert_config.num_layers, bert_config.hidden_dim, 
             bert_config.vocab_size, bert_config.max_position_embeddings);
    println!("│ T5 Small    │ {:8} │ {:10} │ {:12} │ {:12} │", 
             t5_config.num_layers, t5_config.hidden_dim, 
             t5_config.vocab_size, t5_config.max_position_embeddings);
    println!("└─────────────┴──────────┴────────────┴──────────────┴──────────────┘");

    println!("\n💡 Use Case Recommendations:");
    println!("   🚀 GPT Models:");
    println!("      - Text generation and completion");
    println!("      - Creative writing and storytelling");
    println!("      - Code generation and completion");
    println!("      - Conversational AI");
    
    println!("\n   🔍 BERT Models:");
    println!("      - Text classification and sentiment analysis");
    println!("      - Named entity recognition (NER)");
    println!("      - Question answering systems");
    println!("      - Text similarity and semantic search");
    
    println!("\n   🔄 T5 Models:");
    println!("      - Machine translation");
    println!("      - Text summarization");
    println!("      - Question answering with generation");
    println!("      - Text-to-text transformation tasks");

    println!("\n⚡ Performance Tips:");
    println!("   1. Use appropriate batch sizes for your hardware");
    println!("   2. Consider mixed precision training for larger models");
    println!("   3. Implement gradient checkpointing for memory efficiency");
    println!("   4. Use attention masking for variable-length sequences");

    Ok(())
}

/// Demonstration of training setup for transformer models
fn demonstrate_training_setup() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Training Setup Demonstration");
    println!("================================\n");

    let device = DeviceType::Cpu;
    let config = TextModelConfig::bert_base();
    
    // Create model
    let mut model = BertForSequenceClassification::new(config, 2, device)?;
    
    println!("📚 Training Configuration:");
    println!("   Model: BERT for Binary Classification");
    println!("   Classes: 2 (positive/negative sentiment)");
    
    // Set model to training mode
    model.train();
    println!("   Training mode: {}", model.training());
    
    // Create dummy training data
    let batch_size = 8;
    let seq_length = 128;
    let input_ids: Tensor<f32> = rand(&[batch_size, seq_length]);
    let labels: Tensor<f32> = rand(&[batch_size]);
    
    println!("   Batch size: {}", batch_size);
    println!("   Sequence length: {}", seq_length);
    
    // Forward pass
    let logits = model.forward(&input_ids)?;
    println!("   Logits shape: {:?}", logits.shape().dims());
    
    // Demonstrate parameter access for optimizer setup
    let parameters = model.named_parameters();
    println!("   Total parameter tensors: {}", parameters.len());
    
    // Show some parameter names (useful for setting up optimizers)
    let param_names: Vec<&String> = parameters.keys().take(5).collect();
    println!("   Sample parameter names: {:?}", param_names);
    
    // Switch to evaluation mode
    model.eval();
    println!("   Evaluation mode: {}", !model.training());
    
    println!("   ✅ Training setup demonstration completed\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let device = DeviceType::Cpu;
        
        // Test that all models can be created without panicking
        let gpt_config = TextModelConfig::gpt2_small();
        let _gpt_model = GPTModel::new(gpt_config);
        
        let bert_config = TextModelConfig::bert_base();
        let _bert_model = BertModel::new(bert_config, device).unwrap();
        
        let t5_config = TextModelConfig::t5_small();
        let _t5_model = T5Model::new(t5_config, device).unwrap();
    }

    #[test]
    fn test_forward_pass_shapes() {
        let device = DeviceType::Cpu;
        let batch_size = 2;
        let seq_len = 10;
        
        // Test GPT
        let gpt_config = TextModelConfig::gpt2_small();
        let mut gpt_model = GPTModel::new(gpt_config.clone());
        let input: Tensor<f32> = rand(&[batch_size, seq_len]);
        let gpt_output = gpt_model.forward(&input).unwrap();
        assert_eq!(gpt_output.shape().dims(), &[batch_size, seq_len, gpt_config.hidden_dim]);
        
        // Test BERT
        let bert_config = TextModelConfig::bert_base();
        let mut bert_model = BertModel::new(bert_config.clone(), device).unwrap();
        let bert_output = bert_model.forward(&input).unwrap();
        assert_eq!(bert_output.shape().dims(), &[batch_size, seq_len, bert_config.hidden_dim]);
        
        // Test T5
        let t5_config = TextModelConfig::t5_small();
        let mut t5_model = T5Model::new(t5_config.clone(), device).unwrap();
        let t5_output = t5_model.forward(&input).unwrap();
        assert_eq!(t5_output.shape().dims(), &[batch_size, seq_len, t5_config.hidden_dim]);
    }

    #[test]
    fn test_parameter_counting() {
        let device = DeviceType::Cpu;
        
        let gpt_config = TextModelConfig::gpt2_small();
        let gpt_model = GPTModel::new(gpt_config);
        let gpt_params: usize = gpt_model.parameters().values().map(|p| p.data().numel()).sum();
        assert!(gpt_params > 100_000_000); // Should be > 100M parameters
        
        let bert_config = TextModelConfig::bert_base();
        let bert_model = BertModel::new(bert_config, device).unwrap();
        let bert_params: usize = bert_model.parameters().values().map(|p| p.data().numel()).sum();
        assert!(bert_params > 100_000_000); // Should be > 100M parameters
    }
}