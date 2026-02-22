//! Cross-Attention Layer Usage Example
//!
//! This example demonstrates how to use the CrossAttention layer
//! in encoder-decoder architectures, such as in Transformers.

use torsh_core::error::Result;
use torsh_nn::layers::CrossAttention;
use torsh_nn::Module;
use torsh_tensor::creation;

fn main() -> Result<()> {
    println!("CrossAttention Layer Usage Example");
    println!("===================================\n");

    // Configuration
    let embed_dim = 512;
    let num_heads = 8;
    let batch_size = 4;
    let encoder_seq_len = 50;
    let decoder_seq_len = 20;

    println!("Configuration:");
    println!("  Embedding dimension: {}", embed_dim);
    println!("  Number of heads: {}", num_heads);
    println!("  Batch size: {}", batch_size);
    println!("  Encoder sequence length: {}", encoder_seq_len);
    println!("  Decoder sequence length: {}", decoder_seq_len);
    println!();

    // Create cross-attention layer
    println!("Creating CrossAttention layer...");
    let cross_attn = CrossAttention::new(embed_dim, embed_dim, embed_dim, num_heads)?;
    println!("  Parameters: {:?}", cross_attn.parameters().len());
    println!();

    // Simulate encoder output (from encoder)
    println!("Creating encoder output...");
    let encoder_output = creation::randn(&[batch_size, encoder_seq_len, embed_dim])?;
    println!(
        "  Shape: [{}, {}, {}]",
        batch_size, encoder_seq_len, embed_dim
    );
    println!();

    // Simulate decoder query (from decoder self-attention)
    println!("Creating decoder query...");
    let decoder_query = creation::randn(&[batch_size, decoder_seq_len, embed_dim])?;
    println!(
        "  Shape: [{}, {}, {}]",
        batch_size, decoder_seq_len, embed_dim
    );
    println!();

    // Apply cross-attention
    println!("Applying cross-attention...");
    let output = cross_attn.forward_cross(&decoder_query, &encoder_output, &encoder_output, None)?;
    println!("  Output shape: {:?}", output.shape().dims());
    println!();

    // Verify output shape
    assert_eq!(
        output.shape().dims(),
        &[batch_size, decoder_seq_len, embed_dim]
    );
    println!("✓ Output shape matches expected: [{}, {}, {}]", batch_size, decoder_seq_len, embed_dim);
    println!();

    // Example 2: With attention mask
    println!("Example 2: Cross-attention with mask");
    println!("=====================================\n");

    // Create attention mask (e.g., for padding)
    println!("Creating attention mask...");
    let mask = creation::zeros(&[batch_size, decoder_seq_len, encoder_seq_len])?;
    println!(
        "  Mask shape: [{}, {}, {}]",
        batch_size, decoder_seq_len, encoder_seq_len
    );
    println!();

    println!("Applying masked cross-attention...");
    let masked_output =
        cross_attn.forward_cross(&decoder_query, &encoder_output, &encoder_output, Some(&mask))?;
    println!("  Output shape: {:?}", masked_output.shape().dims());
    println!();

    assert_eq!(
        masked_output.shape().dims(),
        &[batch_size, decoder_seq_len, embed_dim]
    );
    println!("✓ Masked output shape matches expected");
    println!();

    // Example 3: Different dimensions
    println!("Example 3: Different encoder/decoder dimensions");
    println!("================================================\n");

    let decoder_dim = 256;
    let encoder_dim = 512;
    let output_dim = 512;

    println!("Configuration:");
    println!("  Decoder dimension: {}", decoder_dim);
    println!("  Encoder dimension: {}", encoder_dim);
    println!("  Output dimension: {}", output_dim);
    println!();

    let cross_attn_diff =
        CrossAttention::new(decoder_dim, encoder_dim, output_dim, num_heads)?;

    let decoder_query_diff = creation::randn(&[batch_size, decoder_seq_len, decoder_dim])?;
    let encoder_output_diff = creation::randn(&[batch_size, encoder_seq_len, encoder_dim])?;

    println!("Applying cross-attention with different dimensions...");
    let output_diff = cross_attn_diff.forward_cross(
        &decoder_query_diff,
        &encoder_output_diff,
        &encoder_output_diff,
        None,
    )?;
    println!("  Output shape: {:?}", output_diff.shape().dims());
    println!();

    assert_eq!(
        output_diff.shape().dims(),
        &[batch_size, decoder_seq_len, output_dim]
    );
    println!("✓ Output dimension matches expected: {}", output_dim);
    println!();

    // Example 4: Training vs Evaluation mode
    println!("Example 4: Training vs Evaluation modes");
    println!("========================================\n");

    let mut cross_attn_dropout =
        CrossAttention::with_config(embed_dim, embed_dim, embed_dim, num_heads, 0.1, true)?;

    println!("Training mode: {}", cross_attn_dropout.training());
    let train_output =
        cross_attn_dropout.forward_cross(&decoder_query, &encoder_output, &encoder_output, None)?;
    println!("  Training output shape: {:?}", train_output.shape().dims());
    println!();

    cross_attn_dropout.eval();
    println!("Evaluation mode: {}", cross_attn_dropout.training());
    let eval_output =
        cross_attn_dropout.forward_cross(&decoder_query, &encoder_output, &encoder_output, None)?;
    println!("  Evaluation output shape: {:?}", eval_output.shape().dims());
    println!();

    println!("✓ All examples completed successfully!");
    println!();

    // Summary
    println!("Summary");
    println!("=======");
    println!("CrossAttention is used in encoder-decoder architectures to allow");
    println!("the decoder to attend to encoder outputs. Key features:");
    println!("  - Separate projections for queries, keys, and values");
    println!("  - Multi-head attention for parallel processing");
    println!("  - Support for attention masking (e.g., padding masks)");
    println!("  - Configurable dimensions for encoder and decoder");
    println!("  - Optional dropout for regularization");

    Ok(())
}
