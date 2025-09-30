//! Transformer architectures performance benchmarking
//!
//! This example provides comprehensive benchmarks for GPT, BERT, and T5 models,
//! measuring throughput, memory usage, and scaling characteristics.

use std::time::Instant;
use torsh_core::device::DeviceType;
use torsh_nn::Module;
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;
use torsh_text::{
    BertForSequenceClassification, BertModel, GPTForCausalLM, GPTModel, T5ForConditionalGeneration,
    T5Model, TextModelConfig,
};

#[derive(Debug, Clone)]
struct BenchmarkResult {
    model_name: String,
    batch_size: usize,
    sequence_length: usize,
    forward_time_ms: f64,
    throughput_tokens_per_sec: f64,
    memory_usage_mb: f64,
    parameters_millions: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Transformer Architecture Benchmarks");
    println!("========================================\n");

    let device = DeviceType::Cpu;
    let mut results = Vec::new();

    // Benchmark configurations
    let batch_sizes = vec![1, 4, 8, 16];
    let sequence_lengths = vec![128, 256, 512];

    println!("📊 Running comprehensive benchmarks...\n");

    // Benchmark GPT models
    results.extend(benchmark_gpt_models(device, &batch_sizes, &sequence_lengths)?);

    // Benchmark BERT models
    results.extend(benchmark_bert_models(device, &batch_sizes, &sequence_lengths)?);

    // Benchmark T5 models
    results.extend(benchmark_t5_models(device, &batch_sizes, &sequence_lengths)?);

    // Print results
    print_benchmark_results(&results);

    // Generate performance analysis
    analyze_performance(&results);

    println!("\n✅ Benchmark suite completed!");
    Ok(())
}

fn benchmark_gpt_models(
    device: DeviceType,
    batch_sizes: &[usize],
    sequence_lengths: &[usize],
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("🤖 Benchmarking GPT Models");
    println!("===========================");

    let mut results = Vec::new();
    let configs = vec![
        ("GPT-2 Small", TextModelConfig::gpt2_small()),
        ("GPT-2 Medium", TextModelConfig::gpt2_medium()),
    ];

    for (model_name, config) in configs {
        println!("Testing {}...", model_name);
        
        let mut model = GPTModel::new(config.clone());
        let mut causal_lm = GPTForCausalLM::new(config.clone());
        
        // Calculate parameter count
        let params_count: usize = model.parameters().values().map(|p| p.data().numel()).sum();
        let params_millions = params_count as f64 / 1_000_000.0;

        for &batch_size in batch_sizes {
            for &seq_len in sequence_lengths {
                // Skip large configurations to avoid memory issues
                if batch_size * seq_len > 4096 && model_name.contains("Medium") {
                    continue;
                }

                println!("  Batch size: {}, Sequence length: {}", batch_size, seq_len);

                // Create input
                let input_ids: Tensor<f32> = rand(&[batch_size, seq_len]);
                let input_ids = input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;

                // Benchmark base model
                let start = Instant::now();
                let _output = model.forward(&input_ids)?;
                let forward_time = start.elapsed();
                let forward_time_ms = forward_time.as_secs_f64() * 1000.0;

                let total_tokens = batch_size * seq_len;
                let throughput = total_tokens as f64 / forward_time.as_secs_f64();

                // Estimate memory usage (simplified)
                let memory_mb = estimate_memory_usage(params_count, batch_size, seq_len, config.hidden_dim);

                results.push(BenchmarkResult {
                    model_name: format!("{} (Base)", model_name),
                    batch_size,
                    sequence_length: seq_len,
                    forward_time_ms,
                    throughput_tokens_per_sec: throughput,
                    memory_usage_mb: memory_mb,
                    parameters_millions: params_millions,
                });

                // Benchmark causal LM head
                let start = Instant::now();
                let _logits = causal_lm.forward(&input_ids)?;
                let lm_time = start.elapsed();
                let lm_time_ms = lm_time.as_secs_f64() * 1000.0;

                let lm_throughput = total_tokens as f64 / lm_time.as_secs_f64();

                results.push(BenchmarkResult {
                    model_name: format!("{} (LM Head)", model_name),
                    batch_size,
                    sequence_length: seq_len,
                    forward_time_ms: lm_time_ms,
                    throughput_tokens_per_sec: lm_throughput,
                    memory_usage_mb: memory_mb * 1.1, // Slightly more for LM head
                    parameters_millions: params_millions + (config.vocab_size * config.hidden_dim) as f64 / 1_000_000.0,
                });
            }
        }
    }

    Ok(results)
}

fn benchmark_bert_models(
    device: DeviceType,
    batch_sizes: &[usize],
    sequence_lengths: &[usize],
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("\n🔍 Benchmarking BERT Models");
    println!("============================");

    let mut results = Vec::new();
    let configs = vec![
        ("BERT Base", TextModelConfig::bert_base()),
        ("BERT Large", TextModelConfig::bert_large()),
    ];

    for (model_name, config) in configs {
        println!("Testing {}...", model_name);
        
        let mut model = BertModel::new(config.clone(), device)?;
        let mut classifier = BertForSequenceClassification::new(config.clone(), 2, device)?;
        
        // Calculate parameter count
        let params_count: usize = model.parameters().values().map(|p| p.data().numel()).sum();
        let params_millions = params_count as f64 / 1_000_000.0;

        for &batch_size in batch_sizes {
            for &seq_len in sequence_lengths {
                // Skip large configurations for BERT Large
                if batch_size * seq_len > 2048 && model_name.contains("Large") {
                    continue;
                }

                println!("  Batch size: {}, Sequence length: {}", batch_size, seq_len);

                // Create input
                let input_ids: Tensor<f32> = rand(&[batch_size, seq_len]);
                let input_ids = input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;

                // Benchmark base model
                let start = Instant::now();
                let _output = model.forward(&input_ids)?;
                let forward_time = start.elapsed();
                let forward_time_ms = forward_time.as_secs_f64() * 1000.0;

                let total_tokens = batch_size * seq_len;
                let throughput = total_tokens as f64 / forward_time.as_secs_f64();

                // Estimate memory usage
                let memory_mb = estimate_memory_usage(params_count, batch_size, seq_len, config.hidden_dim);

                results.push(BenchmarkResult {
                    model_name: format!("{} (Base)", model_name),
                    batch_size,
                    sequence_length: seq_len,
                    forward_time_ms,
                    throughput_tokens_per_sec: throughput,
                    memory_usage_mb: memory_mb,
                    parameters_millions: params_millions,
                });

                // Benchmark classification head
                let start = Instant::now();
                let _logits = classifier.forward(&input_ids)?;
                let cls_time = start.elapsed();
                let cls_time_ms = cls_time.as_secs_f64() * 1000.0;

                let cls_throughput = total_tokens as f64 / cls_time.as_secs_f64();

                results.push(BenchmarkResult {
                    model_name: format!("{} (Classification)", model_name),
                    batch_size,
                    sequence_length: seq_len,
                    forward_time_ms: cls_time_ms,
                    throughput_tokens_per_sec: cls_throughput,
                    memory_usage_mb: memory_mb * 1.05, // Slightly more for classification head
                    parameters_millions: params_millions + (config.hidden_dim * 2) as f64 / 1_000_000.0,
                });
            }
        }
    }

    Ok(results)
}

fn benchmark_t5_models(
    device: DeviceType,
    batch_sizes: &[usize],
    sequence_lengths: &[usize],
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    println!("\n🔄 Benchmarking T5 Models");
    println!("==========================");

    let mut results = Vec::new();
    let config = TextModelConfig::t5_small(); // Only test small for performance reasons

    println!("Testing T5 Small...");
    
    let mut model = T5Model::new(config.clone(), device)?;
    let mut generator = T5ForConditionalGeneration::new(config.clone(), device)?;
    
    // Calculate parameter count
    let params_count: usize = model.parameters().values().map(|p| p.data().numel()).sum();
    let params_millions = params_count as f64 / 1_000_000.0;

    for &batch_size in batch_sizes {
        for &seq_len in sequence_lengths {
            // T5 has both encoder and decoder, so limit configurations more aggressively
            if batch_size * seq_len > 1024 {
                continue;
            }

            println!("  Batch size: {}, Sequence length: {}", batch_size, seq_len);

            // Create inputs
            let encoder_input_ids: Tensor<f32> = rand(&[batch_size, seq_len]);
            let encoder_input_ids = encoder_input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;
            
            let decoder_seq_len = seq_len / 2; // Decoder typically shorter
            let decoder_input_ids: Tensor<f32> = rand(&[batch_size, decoder_seq_len]);
            let decoder_input_ids = decoder_input_ids.mul_scalar(config.vocab_size as f32)?.floor()?.abs()?;

            // Benchmark encoder-only
            let start = Instant::now();
            let _encoder_output = model.encode(&encoder_input_ids, None)?;
            let encoder_time = start.elapsed();
            let encoder_time_ms = encoder_time.as_secs_f64() * 1000.0;

            let encoder_throughput = (batch_size * seq_len) as f64 / encoder_time.as_secs_f64();

            results.push(BenchmarkResult {
                model_name: "T5 Small (Encoder Only)".to_string(),
                batch_size,
                sequence_length: seq_len,
                forward_time_ms: encoder_time_ms,
                throughput_tokens_per_sec: encoder_throughput,
                memory_usage_mb: estimate_memory_usage(params_count / 2, batch_size, seq_len, config.hidden_dim),
                parameters_millions: params_millions / 2.0, // Rough estimate for encoder only
            });

            // Benchmark full encoder-decoder
            let start = Instant::now();
            let _outputs = model.forward_encoder_decoder(
                &encoder_input_ids,
                &decoder_input_ids,
                None,
                None,
            )?;
            let full_time = start.elapsed();
            let full_time_ms = full_time.as_secs_f64() * 1000.0;

            let total_tokens = batch_size * (seq_len + decoder_seq_len);
            let full_throughput = total_tokens as f64 / full_time.as_secs_f64();

            results.push(BenchmarkResult {
                model_name: "T5 Small (Encoder-Decoder)".to_string(),
                batch_size,
                sequence_length: seq_len + decoder_seq_len,
                forward_time_ms: full_time_ms,
                throughput_tokens_per_sec: full_throughput,
                memory_usage_mb: estimate_memory_usage(params_count, batch_size, seq_len + decoder_seq_len, config.hidden_dim),
                parameters_millions: params_millions,
            });

            // Benchmark generation model
            let start = Instant::now();
            let _gen_output = generator.forward(&encoder_input_ids)?;
            let gen_time = start.elapsed();
            let gen_time_ms = gen_time.as_secs_f64() * 1000.0;

            let gen_throughput = (batch_size * seq_len) as f64 / gen_time.as_secs_f64();

            results.push(BenchmarkResult {
                model_name: "T5 Small (Generation)".to_string(),
                batch_size,
                sequence_length: seq_len,
                forward_time_ms: gen_time_ms,
                throughput_tokens_per_sec: gen_throughput,
                memory_usage_mb: estimate_memory_usage(params_count, batch_size, seq_len, config.hidden_dim) * 1.1,
                parameters_millions: params_millions + (config.vocab_size * config.hidden_dim) as f64 / 1_000_000.0,
            });
        }
    }

    Ok(results)
}

fn estimate_memory_usage(params_count: usize, batch_size: usize, seq_len: usize, hidden_dim: usize) -> f64 {
    // Rough estimation of memory usage in MB
    let param_memory = params_count * 4; // 4 bytes per float32 parameter
    let activation_memory = batch_size * seq_len * hidden_dim * 4 * 10; // Rough estimate for activations
    (param_memory + activation_memory) as f64 / (1024.0 * 1024.0) // Convert to MB
}

fn print_benchmark_results(results: &[BenchmarkResult]) {
    println!("\n📈 Benchmark Results");
    println!("=====================\n");

    // Group results by model
    let mut model_groups: std::collections::HashMap<String, Vec<&BenchmarkResult>> = std::collections::HashMap::new();
    for result in results {
        let base_name = result.model_name.split(" (").next().unwrap_or(&result.model_name).to_string();
        model_groups.entry(base_name).or_insert_with(Vec::new).push(result);
    }

    for (model_name, model_results) in model_groups {
        println!("🔹 {}", model_name);
        println!("   Parameters: {:.1}M", model_results[0].parameters_millions);
        
        // Print table header
        println!("   ┌─────────┬─────────┬──────────┬─────────────┬─────────────┐");
        println!("   │ Batch   │ Seq Len │ Time(ms) │ Tokens/sec  │ Memory(MB)  │");
        println!("   ├─────────┼─────────┼──────────┼─────────────┼─────────────┤");
        
        for result in &model_results {
            if result.model_name.contains(&model_name) {
                println!("   │ {:7} │ {:7} │ {:8.1} │ {:11.0} │ {:11.1} │",
                    result.batch_size,
                    result.sequence_length,
                    result.forward_time_ms,
                    result.throughput_tokens_per_sec,
                    result.memory_usage_mb
                );
            }
        }
        
        println!("   └─────────┴─────────┴──────────┴─────────────┴─────────────┘\n");
    }
}

fn analyze_performance(results: &[BenchmarkResult]) {
    println!("🔬 Performance Analysis");
    println!("========================\n");

    // Find best and worst performers
    let fastest = results.iter().max_by(|a, b| a.throughput_tokens_per_sec.partial_cmp(&b.throughput_tokens_per_sec).unwrap()).unwrap();
    let slowest = results.iter().min_by(|a, b| a.throughput_tokens_per_sec.partial_cmp(&b.throughput_tokens_per_sec).unwrap()).unwrap();

    println!("⚡ Throughput Analysis:");
    println!("   Fastest: {} ({:.0} tokens/sec)", fastest.model_name, fastest.throughput_tokens_per_sec);
    println!("   Slowest: {} ({:.0} tokens/sec)", slowest.model_name, slowest.throughput_tokens_per_sec);
    println!("   Speedup: {:.1}x", fastest.throughput_tokens_per_sec / slowest.throughput_tokens_per_sec);

    // Memory efficiency analysis
    let most_efficient = results.iter().min_by(|a, b| {
        let a_ratio = a.memory_usage_mb / a.parameters_millions;
        let b_ratio = b.memory_usage_mb / b.parameters_millions;
        a_ratio.partial_cmp(&b_ratio).unwrap()
    }).unwrap();
    
    println!("\n💾 Memory Efficiency:");
    println!("   Most efficient: {} ({:.1} MB per million params)", 
             most_efficient.model_name, 
             most_efficient.memory_usage_mb / most_efficient.parameters_millions);

    // Scaling analysis
    println!("\n📊 Scaling Characteristics:");
    
    // Group by model and analyze batch size scaling
    let mut model_groups: std::collections::HashMap<String, Vec<&BenchmarkResult>> = std::collections::HashMap::new();
    for result in results {
        let base_name = result.model_name.split(" (").next().unwrap_or(&result.model_name).to_string();
        model_groups.entry(base_name).or_insert_with(Vec::new).push(result);
    }

    for (model_name, model_results) in model_groups {
        if model_results.len() >= 2 {
            // Find results with different batch sizes but same sequence length
            let seq_len_512_results: Vec<_> = model_results.iter()
                .filter(|r| r.sequence_length == 512)
                .collect();
                
            if seq_len_512_results.len() >= 2 {
                let min_batch = seq_len_512_results.iter().min_by_key(|r| r.batch_size).unwrap();
                let max_batch = seq_len_512_results.iter().max_by_key(|r| r.batch_size).unwrap();
                
                let batch_scaling_efficiency = max_batch.throughput_tokens_per_sec / 
                    (min_batch.throughput_tokens_per_sec * (max_batch.batch_size as f64 / min_batch.batch_size as f64));
                
                println!("   {}: Batch scaling efficiency: {:.1}% (batch {} → {})",
                    model_name,
                    batch_scaling_efficiency * 100.0,
                    min_batch.batch_size,
                    max_batch.batch_size
                );
            }
        }
    }

    println!("\n💡 Optimization Recommendations:");
    println!("   1. For inference: Use larger batch sizes when possible");
    println!("   2. For memory-constrained environments: Consider smaller models or gradient checkpointing");
    println!("   3. For production: Monitor batch size vs. throughput trade-offs");
    println!("   4. For training: Use mixed precision and gradient accumulation");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_execution() {
        let device = DeviceType::Cpu;
        let batch_sizes = vec![1, 2];
        let sequence_lengths = vec![64, 128];

        // Test that benchmarks can run without panicking
        let gpt_results = benchmark_gpt_models(device, &batch_sizes, &sequence_lengths).unwrap();
        assert!(!gpt_results.is_empty());

        let bert_results = benchmark_bert_models(device, &batch_sizes, &sequence_lengths).unwrap();
        assert!(!bert_results.is_empty());

        let t5_results = benchmark_t5_models(device, &batch_sizes, &sequence_lengths).unwrap();
        assert!(!t5_results.is_empty());
    }

    #[test]
    fn test_memory_estimation() {
        let memory = estimate_memory_usage(1_000_000, 4, 128, 768);
        assert!(memory > 0.0);
        assert!(memory < 10000.0); // Should be reasonable
    }
}