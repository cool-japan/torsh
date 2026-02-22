use anyhow::Result;
use torsh_text::tokenization::{BPETokenizer, Tokenizer, WhitespaceTokenizer};
use torsh_text::vocab::{SpecialTokens, Vocabulary};

fn main() -> Result<()> {
    println!("=== Basic Tokenization Examples ===\n");

    // Example 1: Whitespace tokenization
    println!("1. Whitespace Tokenization:");
    let tokenizer = WhitespaceTokenizer::new();
    let text = "Hello world! This is a test.";
    let tokens = tokenizer.tokenize(text)?;
    println!("Input: {}", text);
    println!("Tokens: {:?}\n", tokens);

    // Example 2: BPE tokenization
    println!("2. BPE Tokenization:");
    let training_texts = vec![
        "hello world".to_string(),
        "world hello".to_string(),
        "hello there".to_string(),
        "there world".to_string(),
        "hello hello world world".to_string(),
    ];

    let bpe_tokenizer = BPETokenizer::from_texts(&training_texts, 50)?;

    let test_text = "hello there world";
    let bpe_tokens = bpe_tokenizer.tokenize(test_text)?;
    let token_ids = bpe_tokenizer.encode(test_text)?;

    println!("Input: {}", test_text);
    println!("BPE Tokens: {:?}", bpe_tokens);
    println!("Token IDs: {:?}", token_ids);

    // Decode back to text
    let decoded = bpe_tokenizer.decode(&token_ids)?;
    println!("Decoded: {}\n", decoded);

    // Example 3: Vocabulary management
    println!("3. Vocabulary Management:");
    let mut vocab = Vocabulary::new(Some(SpecialTokens::default()));
    // Special tokens are already added by default

    // Add some words
    let words = vec!["hello", "world", "test", "example"];
    for word in &words {
        vocab.add_token(word);
    }

    println!("Vocabulary size: {}", vocab.len());
    // Token to ID conversion
    for word in &words {
        if let Some(id) = vocab.get_token_id(word) {
            println!("{} -> {}", word, id);
        }
    }

    Ok(())
}
