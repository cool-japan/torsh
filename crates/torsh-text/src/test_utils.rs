// Simple test module for our utils functionality
use crate::utils::*;

#[test]
fn test_text_normalizer() {
    let normalizer = TextNormalizer::new()
        .lowercase(true)
        .remove_extra_spaces(true);

    let text = "  Hello    WORLD  ";
    let normalized = normalizer.normalize(text);
    assert_eq!(normalized, "hello world");
}

#[test]
fn test_text_cleaner() {
    let cleaner = TextCleaner::new().remove_urls(true).remove_emails(true);

    let text = "Check out https://example.com and email me at test@example.com";
    let cleaned = cleaner.clean(text);
    assert!(!cleaned.contains("https://example.com"));
    assert!(!cleaned.contains("test@example.com"));
}

#[test]
#[allow(unused_mut)]
fn test_text_augmenter() {
    let mut augmenter = TextAugmenter::new();
    let text = "This is a good example";

    // Test synonym replacement (should not panic)
    let augmented = augmenter.synonym_replacement(text, 0.1);
    assert!(!augmented.is_empty());

    // Test random deletion
    let deleted = augmenter.random_deletion(text, 0.1);
    assert!(!deleted.is_empty());
}

#[test]
fn test_padding_and_truncation() {
    let tokens = vec![1, 2, 3, 4, 5];

    // Test padding
    let padded = pad_sequence(&tokens, 10, 0, PaddingStrategy::Right);
    assert_eq!(padded.len(), 10);
    assert_eq!(padded[..5], tokens);
    assert_eq!(padded[5..], vec![0; 5]);

    // Test truncation
    let truncated = truncate_sequence(&tokens, 3, TruncationStrategy::Right);
    assert_eq!(truncated, vec![1, 2, 3]);
}

#[test]
fn test_encoding_schemes() {
    let token_ids = vec![0, 1, 2];
    let vocab_size = 4;

    let one_hot = one_hot_encode(&token_ids, vocab_size);
    assert_eq!(one_hot.len(), 3);
    assert_eq!(one_hot[0].len(), vocab_size);

    // Check first encoding
    assert_eq!(one_hot[0][0], 1.0);
    assert_eq!(one_hot[0][1], 0.0);

    // Test label encoding
    let labels = vec!["cat".to_string(), "dog".to_string(), "cat".to_string()];
    let (encoded, mapping) = label_encode(&labels);
    assert_eq!(encoded.len(), 3);
    assert_eq!(mapping.len(), 2); // Two unique labels
}

#[test]
fn test_legacy_functions() {
    // Use modern TextPreprocessingPipeline instead of deprecated functions
    let pipeline = PreprocessingUtils::classification_pipeline();

    let text = "  Hello    WORLD  ";
    let normalized = pipeline.process_text(text).unwrap();
    assert!(!normalized.is_empty());

    // Use sentence splitter utility for sentence counting
    let sentences: Vec<&str> = "Hello. World! How are you?"
        .split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();
    assert_eq!(sentences.len(), 3);

    let word_count = count_words("Hello world test");
    assert_eq!(word_count, 3);

    // Use TextCleaner directly instead of deprecated clean_text
    let cleaner = TextCleaner::new();
    let cleaned = cleaner.clean("Hello @#$% world!");
    assert!(cleaned.contains("Hello"));
    assert!(cleaned.contains("world"));
}
