//! Integration tests for torsh-data
//!
//! These tests verify that the complete data loading pipeline works correctly,
//! including the interaction between datasets, samplers, data loaders, and collation.

use torsh_core::error::Result;
use torsh_data::dataloader::{simple_dataloader, simple_random_dataloader};
use torsh_data::prelude::*;
use torsh_tensor::creation::{ones, zeros};

/// Test the complete data loading pipeline with tensor dataset
#[test]
fn test_tensor_dataset_pipeline() -> Result<()> {
    // Create test data
    let data = ones::<f32>(&[10, 3])?;
    let labels = zeros::<f32>(&[10])?;

    // Create dataset
    let dataset = TensorDataset::from_tensors(vec![data, labels]);
    assert_eq!(dataset.len(), 10);

    // Test sequential sampler
    let sampler = SequentialSampler::new(dataset.len());
    let mut indices: Vec<usize> = sampler.iter().collect();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Test random sampler with seed
    let random_sampler = RandomSampler::simple(dataset.len()).with_generator(42);
    indices = random_sampler.iter().collect();
    assert_eq!(indices.len(), 10);
    // Verify all indices are unique
    indices.sort();
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Test batch sampler
    let batch_sampler = BatchingSampler::new(SequentialSampler::new(dataset.len()), 3, false);
    let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
    assert_eq!(batches.len(), 4); // 10 / 3 = 3 full batches + 1 partial
    assert_eq!(batches[0], vec![0, 1, 2]);
    assert_eq!(batches[3], vec![9]); // Last partial batch

    Ok(())
}

/// Test data loader with collation
#[test]
fn test_dataloader_with_collation() -> Result<()> {
    // Create test dataset
    let data = ones::<f32>(&[8, 3])?;
    let dataset = TensorDataset::from_tensor(data);

    // Create data loader with batch size 2
    let dataloader = DataLoaderBuilder::new(dataset).batch_size(2).build()?;

    // Test iteration
    let mut batch_count = 0;
    for batch in dataloader.iter() {
        let batch = batch?;
        assert_eq!(batch.len(), 1); // One tensor per sample
        assert_eq!(batch[0].shape().dims()[0], 2); // Batch size of 2
        assert_eq!(batch[0].shape().dims()[1], 3); // Feature dimension
        batch_count += 1;
    }

    assert_eq!(batch_count, 4); // 8 samples / 2 batch_size = 4 batches

    Ok(())
}

/// Test weighted random sampling
#[test]
fn test_weighted_sampling() -> Result<()> {
    // Create dataset
    let data = ones::<f32>(&[5, 2])?;
    let _dataset = TensorDataset::from_tensor(data);

    // Create weighted sampler - give much higher weight to last element
    let weights = vec![0.1, 0.1, 0.1, 0.1, 0.6];
    let sampler = WeightedRandomSampler::new(weights, 100, true).with_generator(42);

    let indices: Vec<usize> = sampler.iter().collect();
    assert_eq!(indices.len(), 100);

    // Count occurrences
    let mut counts = [0; 5];
    for &idx in &indices {
        counts[idx] += 1;
    }

    // Last element should appear much more frequently
    assert!(counts[4] > counts[0]);
    assert!(counts[4] > counts[1]);
    assert!(counts[4] > counts[2]);
    assert!(counts[4] > counts[3]);

    Ok(())
}

/// Test distributed sampling
#[test]
fn test_distributed_sampling() -> Result<()> {
    let dataset_len = 100;
    let num_replicas = 4;

    // Create samplers for different ranks
    let sampler0 = DistributedSampler::new(dataset_len, num_replicas, 0, false).with_generator(42);
    let sampler1 = DistributedSampler::new(dataset_len, num_replicas, 1, false).with_generator(42);
    let sampler2 = DistributedSampler::new(dataset_len, num_replicas, 2, false).with_generator(42);
    let sampler3 = DistributedSampler::new(dataset_len, num_replicas, 3, false).with_generator(42);

    let indices0: Vec<usize> = sampler0.iter().collect();
    let indices1: Vec<usize> = sampler1.iter().collect();
    let indices2: Vec<usize> = sampler2.iter().collect();
    let indices3: Vec<usize> = sampler3.iter().collect();

    // Each rank should have 25 samples
    assert_eq!(indices0.len(), 25);
    assert_eq!(indices1.len(), 25);
    assert_eq!(indices2.len(), 25);
    assert_eq!(indices3.len(), 25);

    // Indices should not overlap
    let mut all_indices = indices0.clone();
    all_indices.extend(indices1);
    all_indices.extend(indices2);
    all_indices.extend(indices3);
    all_indices.sort();

    // Should have all indices from 0 to 99
    let expected: Vec<usize> = (0..100).collect();
    assert_eq!(all_indices, expected);

    Ok(())
}

/// Test stratified sampling
#[test]
fn test_stratified_sampling() -> Result<()> {
    // Create imbalanced dataset: 5 samples of class 0, 2 samples of class 1, 1 sample of class 2
    let labels = vec![0, 0, 0, 0, 0, 1, 1, 2];
    let sampler = StratifiedSampler::new(&labels, 8, false).with_generator(42);

    assert_eq!(sampler.len(), 8);
    assert_eq!(sampler.num_strata(), 3);

    let indices: Vec<usize> = sampler.iter().collect();
    assert_eq!(indices.len(), 8);

    // Count samples per class
    let mut class_counts = [0; 3];
    for &idx in &indices {
        class_counts[labels[idx]] += 1;
    }

    // Should maintain proportions: class 0 (5/8), class 1 (2/8), class 2 (1/8)
    assert_eq!(class_counts[0], 5);
    assert_eq!(class_counts[1], 2);
    assert_eq!(class_counts[2], 1);

    Ok(())
}

/// Test concatenated datasets
#[test]
fn test_concat_dataset() -> Result<()> {
    let ds1 = TensorDataset::from_tensor(ones::<f32>(&[5, 3])?);
    let ds2 = TensorDataset::from_tensor(zeros::<f32>(&[3, 3])?);

    let concat = ConcatDataset::new(vec![ds1, ds2]);
    assert_eq!(concat.len(), 8);

    // Test accessing different parts
    let item0 = concat.get(0)?;
    let item5 = concat.get(5)?;

    // First 5 should be ones, last 3 should be zeros
    assert_eq!(item0.len(), 1);
    assert_eq!(item5.len(), 1);

    Ok(())
}

/// Test subset dataset
#[test]
fn test_subset_dataset() -> Result<()> {
    let dataset = TensorDataset::from_tensor(ones::<f32>(&[10, 3])?);
    let subset = Subset::new(dataset, vec![0, 2, 4, 6, 8]);

    assert_eq!(subset.len(), 5);

    // Test accessing subset
    for i in 0..5 {
        let item = subset.get(i)?;
        assert_eq!(item.len(), 1);
    }

    // Test out of bounds
    assert!(subset.get(5).is_err());

    Ok(())
}

/// Test text dataset and tokenization
#[test]
fn test_text_dataset() -> Result<()> {
    let texts = vec![
        "This is a positive example".to_string(),
        "This is a negative example".to_string(),
        "Another positive text".to_string(),
        "Another negative text".to_string(),
    ];
    let labels = vec![1, 0, 1, 0];

    let dataset = TextClassificationDataset::new(texts, labels)?;
    assert_eq!(dataset.len(), 4);
    assert_eq!(dataset.num_classes(), 2);

    // Test accessing items
    let (tensor, label) = dataset.get(0)?;
    assert_eq!(label, 1);
    assert!(tensor.ndim() > 0);

    // Test vocabulary
    let vocab = dataset.vocabulary();
    assert!(!vocab.is_empty());

    // Test encoding/decoding
    let text = "test text";
    let ids = vocab.encode(text);
    let decoded = vocab.decode(&ids);
    assert!(!ids.is_empty());
    assert!(!decoded.is_empty());

    Ok(())
}

/// Test dynamic batch collation for variable-length sequences
#[test]
fn test_dynamic_batching() -> Result<()> {
    // Create tensors of different lengths
    let t1 = ones::<f32>(&[3, 4])?; // length 3
    let t2 = ones::<f32>(&[5, 4])?; // length 5
    let t3 = ones::<f32>(&[2, 4])?; // length 2

    let batch = vec![t1, t2, t3];

    // Test dynamic collation with padding
    let collator = DynamicBatchCollate::new(0.0f32).with_max_length(6);
    let (padded_tensor, lengths) = collator.collate(batch)?;

    // Should be padded to max length (5)
    assert_eq!(padded_tensor.shape().dims(), &[3, 5, 4]); // batch_size, max_seq_len, features

    // Check lengths tensor
    assert_eq!(lengths.shape().dims(), &[3]);
    let length_data = lengths.to_vec()?;
    assert_eq!(length_data, vec![3, 5, 2]);

    Ok(())
}

/// Test cached dataset
#[test]
fn test_cached_dataset() -> Result<()> {
    let base_dataset = TensorDataset::from_tensor(ones::<f32>(&[10, 3])?);
    let cached = CachedDataset::new(base_dataset, 5); // Cache up to 5 items

    assert_eq!(cached.len(), 10);

    // Access some items multiple times
    for _ in 0..3 {
        let _ = cached.get(0)?;
        let _ = cached.get(1)?;
        let _ = cached.get(2)?;
    }

    // Cache hit rate should be > 0 after repeated access
    let hit_rate = cached.cache_hit_rate();
    assert!(hit_rate > 0.0);

    Ok(())
}

/// Test curriculum learning sampler
#[test]
fn test_curriculum_sampler() -> Result<()> {
    // Create difficulty function (index-based for simplicity)
    let difficulty_fn = |idx: usize| -> f64 { idx as f64 / 10.0 };

    let mut sampler = CurriculumSampler::new(
        10,
        difficulty_fn,
        5, // total epochs
        CurriculumStrategy::Linear,
    )
    .with_generator(42);

    // At epoch 0, should only include easy samples
    sampler.set_epoch(0);
    let indices_epoch0: Vec<usize> = sampler.iter().collect();

    // At epoch 4 (last), should include all samples
    sampler.set_epoch(4);
    let indices_epoch4: Vec<usize> = sampler.iter().collect();

    // Later epoch should have more samples
    assert!(indices_epoch4.len() >= indices_epoch0.len());

    Ok(())
}

/// Test active learning sampler
#[test]
fn test_active_learning_sampler() -> Result<()> {
    let mut sampler = ActiveLearningSampler::new(
        100, // dataset size
        AcquisitionStrategy::UncertaintySampling,
        10, // budget per round
    )
    .with_generator(42);

    // Set some uncertainty scores
    let uncertainties: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
    sampler.update_uncertainties(uncertainties);

    // First round should select most uncertain samples
    let selected: Vec<usize> = sampler.iter().collect();
    assert_eq!(selected.len(), 10);

    // Mark these as labeled
    sampler.add_labeled_samples(&selected);

    // Second round should select different samples
    let selected2: Vec<usize> = sampler.iter().collect();
    assert_eq!(selected2.len(), 10);

    // Should not overlap with first selection
    for &idx in &selected2 {
        assert!(!selected.contains(&idx));
    }

    Ok(())
}

/// Test importance sampling
#[test]
fn test_importance_sampling() -> Result<()> {
    // Create importance weights - exponentially decreasing
    let weights: Vec<f64> = (0..10).map(|i| 2.0_f64.powi(-i)).collect();

    let sampler = ImportanceSampler::new(weights, 100, true)
        .with_temperature(1.0)
        .with_generator(42);

    let indices: Vec<usize> = sampler.iter().collect();
    assert_eq!(indices.len(), 100);

    // Count occurrences
    let mut counts = [0; 10];
    for &idx in &indices {
        counts[idx] += 1;
    }

    // Earlier indices should appear more frequently
    assert!(counts[0] > counts[9]);
    assert!(counts[1] > counts[8]);

    Ok(())
}

/// Test complete data loading workflow
#[test]
fn test_complete_workflow() -> Result<()> {
    // Create dataset
    let data = ones::<f32>(&[20, 5])?;
    let labels = zeros::<f32>(&[20])?;
    let dataset = TensorDataset::from_tensors(vec![data, labels]);

    // Create train/val split
    let splits = random_split(dataset, &[16, 4], Some(42))?;
    let train_dataset = splits[0].clone();
    let val_dataset = splits[1].clone();

    assert_eq!(train_dataset.len(), 16);
    assert_eq!(val_dataset.len(), 4);

    // Create data loaders
    let train_loader = simple_random_dataloader(train_dataset, 4, Some(42))?;
    let val_loader = simple_dataloader(val_dataset, 2, false)?;

    // Test training loop simulation
    let mut train_batches = 0;
    for batch in train_loader.iter() {
        let batch = batch?;
        assert_eq!(batch.len(), 2); // data and labels
        assert!(batch[0].shape().dims()[0] <= 4); // batch size <= 4
        train_batches += 1;
    }
    assert_eq!(train_batches, 4); // 16 / 4 = 4 batches

    // Test validation loop simulation
    let mut val_batches = 0;
    for batch in val_loader.iter() {
        let batch = batch?;
        assert_eq!(batch.len(), 2); // data and labels
        assert!(batch[0].shape().dims()[0] <= 2); // batch size <= 2
        val_batches += 1;
    }
    assert_eq!(val_batches, 2); // 4 / 2 = 2 batches

    Ok(())
}
