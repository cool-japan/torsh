//! Comprehensive SciRS2 Text Integration Showcase
//!
//! This example demonstrates the powerful NLP capabilities provided by
//! torsh-text's integration with scirs2-text, including:
//! - Text embeddings generation
//! - Sentiment analysis
//! - Text classification
//! - Named entity recognition
//! - Language detection
//! - Question answering
//!
//! Run with: cargo run --example scirs2_integration_showcase

use torsh_text::scirs2_text_integration::{
    advanced_ops::{cluster_documents, extract_topics, paraphrase_text},
    DeviceType as TextDeviceType, PrecisionLevel, SciRS2TextProcessor, TextConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ToRSh-Text SciRS2 Integration Showcase");
    println!("==========================================\n");

    // Configure the SciRS2 text processor
    let config = TextConfig {
        model_name: "bert-base-uncased".to_string(),
        max_length: 512,
        device: TextDeviceType::Cpu,
        batch_size: 32,
        precision: PrecisionLevel::Float32,
    };

    let processor = SciRS2TextProcessor::new(config);

    // Example 1: Text Embeddings
    println!("📊 Example 1: Generating Text Embeddings");
    println!("------------------------------------------");
    let texts = vec![
        "Machine learning is transforming technology.".to_string(),
        "Deep learning models are powerful tools.".to_string(),
        "Natural language processing enables AI systems.".to_string(),
    ];

    match processor.generate_embeddings(&texts) {
        Ok(embeddings) => {
            println!("✅ Generated embeddings:");
            println!(
                "   Shape: {} texts × {} dimensions",
                texts.len(),
                embeddings.embedding_dim
            );
            println!("   Model: {}", embeddings.model_name);
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    println!();

    // Example 2: Sentiment Analysis
    println!("😊 Example 2: Sentiment Analysis");
    println!("------------------------------------------");
    let sentiment_texts = vec![
        "This product is absolutely amazing! I love it!".to_string(),
        "The service was terrible and disappointing.".to_string(),
        "It's okay, nothing special but not bad either.".to_string(),
    ];

    for text in &sentiment_texts {
        match processor.analyze_sentiment(text) {
            Ok(result) => {
                println!("   Text: \"{}\"", text);
                println!("   Sentiment: {:?}", result.sentiment);
                println!("   Confidence: {:.2}%", result.confidence * 100.0);
                println!();
            }
            Err(e) => println!("❌ Error: {}", e),
        }
    }

    // Example 3: Text Classification
    println!("🏷️  Example 3: Text Classification");
    println!("------------------------------------------");
    let classification_text = "Scientists discover new exoplanet in habitable zone";
    let categories: Vec<String> = vec!["science", "sports", "politics", "entertainment"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    match processor.classify_text(classification_text, &categories) {
        Ok(result) => {
            println!("   Text: \"{}\"", classification_text);
            println!("   Category: {}", result.predicted_category);
            println!("   Confidence: {:.2}%", result.confidence * 100.0);
            println!("   All scores:");
            for (cat, score) in result.all_scores.iter() {
                println!("     - {}: {:.2}%", cat, score * 100.0);
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    println!();

    // Example 4: Named Entity Recognition
    println!("🏢 Example 4: Named Entity Recognition");
    println!("------------------------------------------");
    let ner_text = "Elon Musk founded SpaceX in California in 2002.";

    match processor.extract_entities(ner_text) {
        Ok(entities) => {
            println!("   Text: \"{}\"", ner_text);
            println!("   Entities found: {}", entities.len());
            for entity in entities {
                println!(
                    "     - \"{}\" ({:?}): {:.2}% confidence",
                    entity.text,
                    entity.entity_type,
                    entity.confidence * 100.0
                );
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    println!();

    // Example 5: Language Detection
    println!("🌍 Example 5: Language Detection");
    println!("------------------------------------------");
    let multilingual_texts = vec![
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "Guten Tag, wie geht es Ihnen?",
    ];

    for text in &multilingual_texts {
        match processor.detect_language(text) {
            Ok(result) => {
                println!("   Text: \"{}\"", text);
                println!(
                    "   Language: {} ({:.2}%)",
                    result.language,
                    result.confidence * 100.0
                );
            }
            Err(e) => println!("❌ Error: {}", e),
        }
    }
    println!();

    // Example 6: Semantic Similarity
    println!("🔗 Example 6: Semantic Similarity");
    println!("------------------------------------------");
    let text1 = "The cat sat on the mat.";
    let text2 = "A feline rested on the rug.";

    match processor.semantic_similarity(text1, text2) {
        Ok(similarity) => {
            println!("   Text 1: \"{}\"", text1);
            println!("   Text 2: \"{}\"", text2);
            println!("   Similarity: {:.2}%", similarity * 100.0);
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    println!();

    // Example 7: Advanced Operations - Topic Extraction
    println!("🔍 Example 7: Topic Extraction");
    println!("------------------------------------------");
    let topic_texts = vec![
        "Climate change is affecting global temperatures.".to_string(),
        "Renewable energy sources are becoming more affordable.".to_string(),
        "Electric vehicles are gaining market share.".to_string(),
        "Machine learning improves prediction accuracy.".to_string(),
        "Neural networks are used in image recognition.".to_string(),
    ];

    match extract_topics(&topic_texts, 2) {
        Ok(topics) => {
            println!("   Extracted {} topics:", topics.len());
            for (i, topic) in topics.iter().enumerate() {
                println!("   Topic {}: ID={}", i + 1, topic.id);
                println!("     Keywords: {}", topic.keywords.join(", "));
                println!("     Weight: {:.3}", topic.weight);
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    println!();

    // Example 8: Advanced Operations - Document Clustering
    println!("📁 Example 8: Document Clustering");
    println!("------------------------------------------");
    let cluster_texts = vec![
        "Python is a popular programming language.".to_string(),
        "JavaScript runs in web browsers.".to_string(),
        "Basketball is a team sport.".to_string(),
        "Soccer is played worldwide.".to_string(),
    ];

    // First generate embeddings for clustering
    match processor.generate_embeddings(&cluster_texts) {
        Ok(embeddings) => match cluster_documents(&embeddings, 2) {
            Ok(clusters) => {
                println!("   Created {} clusters:", clusters.len());
                for (i, cluster) in clusters.iter().enumerate() {
                    println!(
                        "   Cluster {}: {} documents",
                        i + 1,
                        cluster.documents.len()
                    );
                    println!("     Document indices: {:?}", cluster.documents);
                    println!("     Coherence: {:.3}", cluster.coherence_score);
                }
            }
            Err(e) => println!("❌ Error: {}", e),
        },
        Err(e) => println!("❌ Error generating embeddings: {}", e),
    }
    println!();

    // Example 9: Advanced Operations - Text Paraphrasing
    println!("🔄 Example 9: Text Paraphrasing");
    println!("------------------------------------------");
    let original_text = "The quick brown fox jumps over the lazy dog.";

    match paraphrase_text(original_text, 3) {
        Ok(paraphrases) => {
            println!("   Original: \"{}\"", original_text);
            println!("   Paraphrases:");
            for (i, paraphrase) in paraphrases.iter().enumerate() {
                println!("   {}. \"{}\"", i + 1, paraphrase);
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    println!();

    println!("✨ Showcase completed successfully!");
    println!("\nNote: These examples use placeholder implementations.");
    println!("In production, they would use actual scirs2-text models.");

    Ok(())
}
