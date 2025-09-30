//! Model Registry and Search Example
//!
//! This example demonstrates how to use the ToRSh Hub model registry
//! to discover, search, and manage models.

use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_hub::model_info::*;
use torsh_hub::registry::*;

fn main() -> Result<()> {
    println!("=== ToRSh Hub Model Registry and Search Example ===\n");

    // Example 1: Initialize model registry
    println!("1. Initializing model registry...");
    let registry_path = std::env::temp_dir().join("torsh_hub_registry.json");
    let mut registry = ModelRegistry::new(&registry_path)?;

    // Populate with example models for demonstration
    populate_example_models(&mut registry)?;
    println!("✓ Registry initialized with example models");

    // Example 2: Basic model search
    println!("\n2. Basic model search...");
    let search_query = SearchQuery {
        text: Some("image classification".to_string()),
        limit: 5,
        ..Default::default()
    };

    let results = registry.search(&search_query);
    println!(
        "✓ Found {} models for 'image classification':",
        results.len()
    );
    for (i, model) in results.iter().enumerate() {
        println!(
            "  {}. {} - {} downloads",
            i + 1,
            model.name,
            model.downloads
        );
    }

    // Example 3: Category-based search
    println!("\n3. Searching by category...");
    let vision_query = SearchQuery {
        category: Some(ModelCategory::Vision),
        sort_by: SortBy::Accuracy,
        limit: 3,
        ..Default::default()
    };

    let vision_models = registry.search(&vision_query);
    println!("✓ Top vision models by accuracy:");
    for model in vision_models {
        let accuracy = model.accuracy_metrics.get("accuracy").unwrap_or(&0.0);
        println!("  - {}: {:.3} accuracy", model.name, accuracy);
    }

    // Example 4: Hardware-filtered search
    println!("\n4. Hardware-filtered search...");
    let hardware_filter = HardwareFilter {
        max_ram_gb: Some(8.0),
        max_gpu_memory_gb: Some(4.0),
        requires_gpu: Some(false), // CPU-friendly models
        supports_cpu_only: Some(true),
    };

    let hardware_query = SearchQuery {
        hardware_filter: Some(hardware_filter),
        sort_by: SortBy::InferenceSpeed,
        limit: 5,
        ..Default::default()
    };

    let efficient_models = registry.search(&hardware_query);
    println!("✓ CPU-friendly models with low memory requirements:");
    for model in efficient_models {
        let inference_time = model.inference_time_ms.unwrap_or(0.0);
        let ram_req = model
            .hardware_requirements
            .recommended_ram_gb
            .unwrap_or(0.0);
        println!(
            "  - {}: {:.1}ms inference, {:.1}GB RAM",
            model.name, inference_time, ram_req
        );
    }

    // Example 5: Advanced filtering
    println!("\n5. Advanced filtering...");
    let advanced_query = SearchQuery {
        min_accuracy: Some(0.85),
        max_model_size_mb: Some(100.0),
        framework_compatibility: vec!["torsh".to_string(), "onnx".to_string()],
        license_filter: vec!["MIT".to_string(), "Apache-2.0".to_string()],
        status_filter: vec![ModelStatus::Active],
        has_demo: Some(true),
        sort_by: SortBy::Downloads,
        limit: 10,
        ..Default::default()
    };

    let filtered_models = registry.search(&advanced_query);
    println!("✓ High-quality models (>85% accuracy, <100MB, with demos):");
    for model in filtered_models {
        println!(
            "  - {} ({}): {} downloads, {} license",
            model.name, model.architecture, model.downloads, model.license
        );
    }

    // Example 6: Get trending models
    println!("\n6. Getting trending models...");
    let trending = registry.get_trending(7); // Last 7 days
    println!("✓ Trending models this week:");
    for (i, model) in trending.iter().enumerate() {
        println!(
            "  {}. {} - {} recent downloads",
            i + 1,
            model.name,
            model.downloads
        );
    }

    // Example 7: Get featured models
    println!("\n7. Getting featured models...");
    let featured = registry.get_featured(5);
    println!("✓ Featured models:");
    for model in featured {
        println!(
            "  - {} by {} - {}",
            model.name, model.author, model.description
        );
    }

    // Example 8: Model recommendations
    println!("\n8. Getting model recommendations...");
    let user_history = vec!["resnet50".to_string(), "efficientnet-b0".to_string()];
    let recommendations = registry.get_recommendations(&user_history, 5);
    println!("✓ Recommendations based on your history:");
    for (i, model) in recommendations.iter().enumerate() {
        println!("  {}. {} - Similar to your interests", i + 1, model.name);
    }

    // Example 9: Model statistics and analytics
    println!("\n9. Model statistics...");
    display_registry_statistics(&registry)?;

    // Example 10: Model card operations
    println!("\n10. Working with model cards...");
    demonstrate_model_cards()?;

    // Example 11: Registry management
    println!("\n11. Registry management...");
    demonstrate_registry_management(&mut registry)?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

fn populate_example_models(registry: &mut ModelRegistry) -> Result<()> {
    // Vision models
    let resnet50 = RegistryEntry {
        id: "resnet50".to_string(),
        name: "ResNet-50".to_string(),
        author: "torsh-vision".to_string(),
        repository: "torsh-models/resnet50".to_string(),
        version: Version::new(1, 0, 0),
        tags: vec![
            "vision".to_string(),
            "classification".to_string(),
            "imagenet".to_string(),
        ],
        downloads: 15420,
        likes: 892,
        created_at: chrono::Utc::now() - chrono::Duration::days(90),
        updated_at: chrono::Utc::now() - chrono::Duration::days(10),
        description: "Deep residual network with 50 layers for image classification".to_string(),
        metrics: {
            let mut m = HashMap::new();
            m.insert("top1_accuracy".to_string(), 0.761);
            m.insert("top5_accuracy".to_string(), 0.928);
            m
        },
        category: ModelCategory::Vision,
        architecture: "ResNet".to_string(),
        framework_compatibility: vec!["torsh".to_string(), "onnx".to_string()],
        hardware_requirements: HardwareSpec {
            min_ram_gb: Some(4.0),
            recommended_ram_gb: Some(8.0),
            min_gpu_memory_gb: Some(2.0),
            recommended_gpu_memory_gb: Some(4.0),
            supports_cpu: true,
            supports_gpu: true,
            supports_tpu: false,
        },
        model_size_mb: Some(98.5),
        inference_time_ms: Some(12.3),
        accuracy_metrics: {
            let mut m = HashMap::new();
            m.insert("accuracy".to_string(), 0.761);
            m
        },
        license: "MIT".to_string(),
        paper_url: Some("https://arxiv.org/abs/1512.03385".to_string()),
        demo_url: Some("https://example.com/resnet50-demo".to_string()),
        status: ModelStatus::Active,
    };

    let efficientnet = RegistryEntry {
        id: "efficientnet-b0".to_string(),
        name: "EfficientNet-B0".to_string(),
        author: "torsh-vision".to_string(),
        repository: "torsh-models/efficientnet-b0".to_string(),
        version: Version::new(2, 1, 0),
        tags: vec![
            "vision".to_string(),
            "classification".to_string(),
            "efficient".to_string(),
        ],
        downloads: 23100,
        likes: 1247,
        created_at: chrono::Utc::now() - chrono::Duration::days(60),
        updated_at: chrono::Utc::now() - chrono::Duration::days(5),
        description: "Efficient convolutional neural network for image classification".to_string(),
        metrics: {
            let mut m = HashMap::new();
            m.insert("top1_accuracy".to_string(), 0.772);
            m.insert("params".to_string(), 5.3);
            m
        },
        category: ModelCategory::Vision,
        architecture: "EfficientNet".to_string(),
        framework_compatibility: vec![
            "torsh".to_string(),
            "onnx".to_string(),
            "tensorflow".to_string(),
        ],
        hardware_requirements: HardwareSpec {
            min_ram_gb: Some(2.0),
            recommended_ram_gb: Some(4.0),
            min_gpu_memory_gb: Some(1.0),
            recommended_gpu_memory_gb: Some(2.0),
            supports_cpu: true,
            supports_gpu: true,
            supports_tpu: true,
        },
        model_size_mb: Some(21.4),
        inference_time_ms: Some(8.7),
        accuracy_metrics: {
            let mut m = HashMap::new();
            m.insert("accuracy".to_string(), 0.772);
            m
        },
        license: "Apache-2.0".to_string(),
        paper_url: Some("https://arxiv.org/abs/1905.11946".to_string()),
        demo_url: Some("https://example.com/efficientnet-demo".to_string()),
        status: ModelStatus::Active,
    };

    // NLP models
    let bert_base = RegistryEntry {
        id: "bert-base-uncased".to_string(),
        name: "BERT Base Uncased".to_string(),
        author: "torsh-nlp".to_string(),
        repository: "torsh-models/bert-base-uncased".to_string(),
        version: Version::new(1, 2, 0),
        tags: vec![
            "nlp".to_string(),
            "transformer".to_string(),
            "language-model".to_string(),
        ],
        downloads: 45670,
        likes: 2104,
        created_at: chrono::Utc::now() - chrono::Duration::days(120),
        updated_at: chrono::Utc::now() - chrono::Duration::days(15),
        description: "Bidirectional encoder representations from transformers".to_string(),
        metrics: {
            let mut m = HashMap::new();
            m.insert("glue_score".to_string(), 82.1);
            m.insert("params".to_string(), 110.0);
            m
        },
        category: ModelCategory::NLP,
        architecture: "BERT".to_string(),
        framework_compatibility: vec![
            "torsh".to_string(),
            "onnx".to_string(),
            "huggingface".to_string(),
        ],
        hardware_requirements: HardwareSpec {
            min_ram_gb: Some(8.0),
            recommended_ram_gb: Some(16.0),
            min_gpu_memory_gb: Some(4.0),
            recommended_gpu_memory_gb: Some(8.0),
            supports_cpu: true,
            supports_gpu: true,
            supports_tpu: true,
        },
        model_size_mb: Some(440.0),
        inference_time_ms: Some(45.2),
        accuracy_metrics: {
            let mut m = HashMap::new();
            m.insert("accuracy".to_string(), 0.856);
            m
        },
        license: "Apache-2.0".to_string(),
        paper_url: Some("https://arxiv.org/abs/1810.04805".to_string()),
        demo_url: None,
        status: ModelStatus::Active,
    };

    // Audio model
    let wav2vec2 = RegistryEntry {
        id: "wav2vec2-base".to_string(),
        name: "Wav2Vec2 Base".to_string(),
        author: "torsh-audio".to_string(),
        repository: "torsh-models/wav2vec2-base".to_string(),
        version: Version::new(1, 0, 0),
        tags: vec![
            "audio".to_string(),
            "speech".to_string(),
            "self-supervised".to_string(),
        ],
        downloads: 8340,
        likes: 421,
        created_at: chrono::Utc::now() - chrono::Duration::days(45),
        updated_at: chrono::Utc::now() - chrono::Duration::days(12),
        description: "Self-supervised speech representation learning".to_string(),
        metrics: {
            let mut m = HashMap::new();
            m.insert("wer".to_string(), 6.1); // Word Error Rate
            m.insert("params".to_string(), 95.0);
            m
        },
        category: ModelCategory::Audio,
        architecture: "Wav2Vec2".to_string(),
        framework_compatibility: vec!["torsh".to_string(), "huggingface".to_string()],
        hardware_requirements: HardwareSpec {
            min_ram_gb: Some(6.0),
            recommended_ram_gb: Some(12.0),
            min_gpu_memory_gb: Some(3.0),
            recommended_gpu_memory_gb: Some(6.0),
            supports_cpu: true,
            supports_gpu: true,
            supports_tpu: false,
        },
        model_size_mb: Some(378.0),
        inference_time_ms: Some(89.5),
        accuracy_metrics: {
            let mut m = HashMap::new();
            m.insert("accuracy".to_string(), 0.939); // 1 - WER/100
            m
        },
        license: "MIT".to_string(),
        paper_url: Some("https://arxiv.org/abs/2006.11477".to_string()),
        demo_url: Some("https://example.com/wav2vec2-demo".to_string()),
        status: ModelStatus::Active,
    };

    // Register all models
    registry.register_model(resnet50)?;
    registry.register_model(efficientnet)?;
    registry.register_model(bert_base)?;
    registry.register_model(wav2vec2)?;

    Ok(())
}

fn display_registry_statistics(registry: &ModelRegistry) -> Result<()> {
    let stats = registry.get_statistics();

    println!("Registry Statistics:");
    println!("  Total models: {}", stats.total_models);
    println!("  Total downloads: {}", stats.total_downloads);
    println!("  Total likes: {}", stats.total_likes);

    println!("  Models by category:");
    for (category, count) in &stats.category_distribution {
        println!("    {:?}: {}", category, count);
    }

    Ok(())
}

fn demonstrate_model_cards() -> Result<()> {
    use torsh_hub::model_info::{ModelCardBuilder, ModelCardRenderer};

    println!("Creating example model card...");

    let card = ModelCardBuilder::new()
        .developed_by("researcher@university.edu".to_string())
        .model_type("Vision Model".to_string())
        .architecture("ResNet-50".to_string())
        .add_primary_use("Image classification".to_string())
        .add_primary_use("Feature extraction".to_string())
        .add_training_dataset(
            "ImageNet".to_string(),
            Some("https://imagenet.org".to_string()),
            Some("1.4M images for training".to_string()),
            None,
        )
        .add_metric(
            "Top-1 Accuracy".to_string(),
            MetricValue::Float(0.785),
            "Accuracy on test set".to_string(),
        )
        .add_metric(
            "Top-5 Accuracy".to_string(),
            MetricValue::Float(0.945),
            "Top-5 accuracy on test set".to_string(),
        )
        .ethical_considerations("Trained on web-scraped data which may contain biases".to_string())
        .build();

    println!("✓ Model card created");

    // Render to different formats
    let markdown = ModelCardRenderer::to_markdown(&card);
    println!("✓ Rendered to Markdown ({} characters)", markdown.len());

    let html = ModelCardRenderer::to_html(&card);
    println!("✓ Rendered to HTML ({} characters)", html.len());

    let json = ModelCardRenderer::to_json(&card)?;
    println!("✓ Rendered to JSON ({} characters)", json.len());

    // Save to files
    let temp_dir = std::env::temp_dir().join("torsh_hub_examples");
    std::fs::create_dir_all(&temp_dir)?;

    std::fs::write(temp_dir.join("model_card.md"), &markdown)?;
    std::fs::write(temp_dir.join("model_card.html"), &html)?;
    std::fs::write(temp_dir.join("model_card.json"), &json)?;

    println!("✓ Model card files saved to: {}", temp_dir.display());

    Ok(())
}

fn demonstrate_registry_management(registry: &mut ModelRegistry) -> Result<()> {
    println!("Registry management operations...");

    // Increment download count (this method exists)
    let model_id = "resnet50";
    registry.increment_downloads(model_id)?;
    println!("✓ Incremented download count for {}", model_id);

    // Increment likes (this method exists)
    registry.increment_likes(model_id)?;
    println!("✓ Incremented like count for {}", model_id);

    // Note: Methods like update_model_metrics, add_model_tags, save, and load
    // are not available in the current ModelRegistry implementation

    // Verify the model changes in current registry
    if let Some(model) = registry.get_model(model_id) {
        println!("✓ Verified model updates in registry");
        println!("  Downloads: {}", model.downloads);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_query_default() {
        let query = SearchQuery::default();
        assert_eq!(query.limit, 50);
        assert_eq!(query.sort_by as u8, SortBy::Downloads as u8);
        assert_eq!(query.status_filter, vec![ModelStatus::Active]);
    }

    #[test]
    fn test_hardware_filter() {
        let filter = HardwareFilter {
            max_ram_gb: Some(8.0),
            max_gpu_memory_gb: None,
            requires_gpu: Some(false),
            supports_cpu_only: Some(true),
        };

        assert_eq!(filter.max_ram_gb, Some(8.0));
        assert_eq!(filter.requires_gpu, Some(false));
    }

    #[test]
    fn test_model_category() {
        assert_eq!(ModelCategory::Vision, ModelCategory::Vision);
        assert_ne!(ModelCategory::Vision, ModelCategory::NLP);

        // Test custom category
        let custom = ModelCategory::Other("Custom".to_string());
        if let ModelCategory::Other(name) = custom {
            assert_eq!(name, "Custom");
        } else {
            panic!("Expected Other variant");
        }
    }

    #[test]
    fn test_registry_operations() {
        let temp_file = std::env::temp_dir().join("test_registry.json");
        let mut registry = ModelRegistry::new(&temp_file);

        // Create test model
        let test_model = RegistryEntry {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            author: "test".to_string(),
            repository: "test/test".to_string(),
            version: Version::new(1, 0, 0),
            tags: vec!["test".to_string()],
            downloads: 0,
            likes: 0,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            description: "Test model".to_string(),
            metrics: HashMap::new(),
            category: ModelCategory::Other("Test".to_string()),
            architecture: "Test".to_string(),
            framework_compatibility: vec!["torsh".to_string()],
            hardware_requirements: HardwareSpec {
                min_ram_gb: None,
                recommended_ram_gb: None,
                min_gpu_memory_gb: None,
                recommended_gpu_memory_gb: None,
                supports_cpu: true,
                supports_gpu: false,
                supports_tpu: false,
            },
            model_size_mb: None,
            inference_time_ms: None,
            accuracy_metrics: HashMap::new(),
            license: "MIT".to_string(),
            paper_url: None,
            demo_url: None,
            status: ModelStatus::Active,
        };

        // Test registration
        assert!(registry.register_model(test_model).is_ok());

        // Test retrieval
        let retrieved = registry.get_model("test-model");
        assert!(retrieved.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }
}
