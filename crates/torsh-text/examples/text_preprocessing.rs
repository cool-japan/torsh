use anyhow::Result;
use torsh_text::utils::{CustomStep, TextCleaner, TextNormalizer, TextPreprocessingPipeline};

fn main() -> Result<()> {
    println!("=== Text Preprocessing Examples ===\n");

    // Example 1: Text normalization
    println!("1. Text Normalization:");
    let normalizer = TextNormalizer::new()
        .normalize_unicode(true)
        .remove_accents(true)
        .remove_punctuation(true)
        .remove_digits(true)
        .remove_extra_spaces(true);

    let messy_text = "Héllo    Wörld123!!!   This  is   â  tëst.";
    let normalized = normalizer.normalize(messy_text);
    println!("Original: {}", messy_text);
    println!("Normalized: {}\n", normalized);

    // Example 2: Text cleaning
    println!("2. Text Cleaning:");
    let cleaner = TextCleaner::new()
        .remove_urls(true)
        .remove_emails(true)
        .remove_html(true)
        .remove_mentions(true)
        .remove_hashtags(true)
        .remove_special_chars(false);

    let dirty_text = "Check out https://example.com for more info! Contact us at test@email.com. Follow @user #hashtag <script>alert('test')</script>";
    let cleaned = cleaner.clean(dirty_text);
    println!("Original: {}", dirty_text);
    println!("Cleaned: {}\n", cleaned);

    // Example 3: Combined preprocessing pipeline
    println!("3. Preprocessing Pipeline:");
    let mut pipeline = TextPreprocessingPipeline::new();

    // Add normalization step
    let normalizer = TextNormalizer::new()
        .normalize_unicode(true)
        .remove_accents(true)
        .remove_punctuation(true);
    pipeline = pipeline.with_normalization(normalizer);

    // Add cleaning step
    let cleaner = TextCleaner::new()
        .remove_urls(true)
        .remove_emails(true)
        .remove_html(true);
    pipeline = pipeline.with_cleaning(cleaner);

    // Add custom lowercase step
    pipeline = pipeline.add_custom_step(Box::new(CustomStep::new(
        |text: &str| text.to_lowercase(),
        "Lowercase".to_string(),
    )));

    let raw_text = "Visit HTTPS://Example.COM for MORE info! Email: Test@Email.COM #AI #MachineLearning <b>Bold text</b>";
    let processed = pipeline.process_text(raw_text)?;
    println!("Original: {}", raw_text);
    println!("Processed: {}\n", processed);

    // Example 4: Batch processing
    println!("4. Batch Processing:");
    let texts = vec![
        "First example text with URLs: https://test1.com",
        "Second example with @mentions and #hashtags",
        "Third example with HTML <tags> and émojis 😀",
        "Fourth example with     extra    spaces",
    ];

    println!("Processing {} texts:", texts.len());
    for (i, text) in texts.iter().enumerate() {
        let processed = pipeline.process_text(text)?;
        println!("  {}: {} -> {}", i + 1, text, processed);
    }

    // Example 5: Task-specific preprocessing
    println!("\n5. Task-Specific Preprocessing:");

    // Sentiment analysis preprocessing (preserve emoticons)
    let sentiment_text = "I love this! :) Amazing product 😍 https://buy.now.com";
    let mut sentiment_pipeline = TextPreprocessingPipeline::new();
    let sentiment_cleaner = TextCleaner::new()
        .remove_urls(true)
        .remove_special_chars(false); // Keep emoticons
    let sentiment_normalizer = TextNormalizer::new()
        .normalize_unicode(true)
        .remove_accents(false)
        .remove_punctuation(false);
    sentiment_pipeline = sentiment_pipeline
        .with_cleaning(sentiment_cleaner)
        .with_normalization(sentiment_normalizer);

    let sentiment_processed = sentiment_pipeline.process_text(sentiment_text)?;
    println!(
        "Sentiment text: {} -> {}",
        sentiment_text, sentiment_processed
    );

    // NER preprocessing (preserve casing and punctuation)
    let ner_text = "John Smith works at Apple Inc. in New York.";
    let mut ner_pipeline = TextPreprocessingPipeline::new();
    let ner_cleaner = TextCleaner::new()
        .remove_urls(false)
        .remove_emails(false)
        .remove_special_chars(false);
    let ner_normalizer = TextNormalizer::new()
        .normalize_unicode(true)
        .remove_accents(false)
        .remove_punctuation(false);
    ner_pipeline = ner_pipeline
        .with_cleaning(ner_cleaner)
        .with_normalization(ner_normalizer);

    let ner_processed = ner_pipeline.process_text(ner_text)?;
    println!("NER text: {} -> {}", ner_text, ner_processed);

    Ok(())
}
