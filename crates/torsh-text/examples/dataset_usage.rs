use anyhow::Result;
use torsh_text::datasets::{ClassificationDataset, Dataset};
use torsh_text::utils::TextPreprocessingPipeline;

fn main() -> Result<()> {
    println!("=== Dataset Usage Examples ===\n");

    // Example 1: Creating a simple classification dataset
    println!("1. Creating Classification Dataset:");

    let texts = vec![
        "I love this movie! It's amazing.".to_string(),
        "This film is terrible and boring.".to_string(),
        "Great acting and wonderful story.".to_string(),
        "Worst movie I've ever seen.".to_string(),
        "Absolutely fantastic! Highly recommend.".to_string(),
    ];

    let labels = vec![
        "positive".to_string(),
        "negative".to_string(),
        "positive".to_string(),
        "negative".to_string(),
        "positive".to_string(),
    ];

    let mut dataset = ClassificationDataset::new();
    for (text, label) in texts.into_iter().zip(labels.into_iter()) {
        dataset.texts.push(text);
        dataset.labels.push(label.clone());
        let id = dataset.labels.len() - 1;
        dataset.label_to_id.entry(label.clone()).or_insert(id);
        dataset.id_to_label.insert(id, label);
    }

    println!("Dataset size: {}", dataset.len());
    println!("Number of classes: {}", dataset.num_classes());
    println!();

    // Example 2: Iterating through dataset
    println!("2. Dataset Iteration:");
    for i in 0..3.min(dataset.len()) {
        let (text, label) = dataset.get_item(i)?;
        println!("Sample {}: \"{}\" -> {}", i, text, label);
    }
    println!();

    // Example 3: Applying preprocessing to dataset
    println!("3. Dataset Preprocessing:");

    let pipeline = TextPreprocessingPipeline::new();

    println!("Before preprocessing:");
    let (first_text, _) = dataset.get_item(0)?;
    println!("  \"{}\"", first_text);

    let processed_text = pipeline.process_text(&first_text)?;
    println!("After preprocessing:");
    println!("  \"{}\"", processed_text);
    println!();

    // Example 4: Dataset splitting
    println!("4. Dataset Splitting:");

    // Simple split example
    let split_idx = (dataset.len() as f64 * 0.8) as usize;
    println!("Original dataset size: {}", dataset.len());
    println!("Split at index: {}", split_idx);
    println!();

    // Example 5: Label encoding
    println!("5. Label Encoding:");

    println!("Label mapping:");
    for (label, id) in &dataset.label_to_id {
        println!("  {} -> {}", label, id);
    }

    // Convert labels to IDs
    println!("\nSamples with encoded labels:");
    for i in 0..3.min(dataset.len()) {
        let (text, label) = dataset.get_item(i)?;
        if let Some(&label_id) = dataset.label_to_id.get(&label) {
            println!(
                "  Sample {}: \"{}\" -> {} (id: {})",
                i,
                text.chars().take(30).collect::<String>() + "...",
                label,
                label_id
            );
        }
    }
    println!();

    // Example 6: Batch processing
    println!("6. Batch Processing:");

    let batch_size = 3;
    let mut batch_texts = Vec::new();
    let mut batch_labels = Vec::new();

    for i in 0..dataset.len() {
        let (text, label) = dataset.get_item(i)?;
        batch_texts.push(text);
        batch_labels.push(label);

        if batch_texts.len() == batch_size || i == dataset.len() - 1 {
            println!(
                "Batch {} (size: {}):",
                (i / batch_size) + 1,
                batch_texts.len()
            );
            for j in 0..batch_texts.len() {
                println!(
                    "  {}: {} -> {}",
                    j,
                    batch_texts[j].chars().take(20).collect::<String>() + "...",
                    batch_labels[j]
                );
            }
            batch_texts.clear();
            batch_labels.clear();
            println!();
        }
    }

    // Example 7: Dataset statistics
    println!("7. Dataset Statistics:");

    println!("Total samples: {}", dataset.len());
    let avg_len: f64 =
        dataset.texts.iter().map(|t| t.len()).sum::<usize>() as f64 / dataset.len() as f64;
    let min_len = dataset.texts.iter().map(|t| t.len()).min().unwrap_or(0);
    let max_len = dataset.texts.iter().map(|t| t.len()).max().unwrap_or(0);
    println!("Average text length: {:.2} characters", avg_len);
    println!("Min text length: {} characters", min_len);
    println!("Max text length: {} characters", max_len);
    println!("Number of unique labels: {}", dataset.num_classes());

    Ok(())
}
