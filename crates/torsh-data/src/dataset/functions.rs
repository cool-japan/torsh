//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::types::{FeatureStats, Subset, TensorDataset};

/// A map-style dataset
///
/// Represents a dataset that supports random access with a known length.
pub trait Dataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;
    /// Returns the number of items in the dataset
    fn len(&self) -> usize;
    /// Returns true if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get a single item from the dataset
    fn get(&self, index: usize) -> Result<Self::Item>;
}
/// An iterable-style dataset
///
/// Represents a dataset that can be iterated over but may not support
/// random access or have a known length.
pub trait IterableDataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;
    /// The iterator type
    type Iter: Iterator<Item = Result<Self::Item>> + Send;
    /// Create an iterator over the dataset
    fn iter(&self) -> Self::Iter;
}
/// Split a dataset into train and validation sets
pub fn random_split<D>(
    dataset: D,
    lengths: &[usize],
    generator: Option<u64>,
) -> Result<Vec<Subset<D>>>
where
    D: Dataset + Clone,
{
    let total_length: usize = lengths.iter().sum();
    if total_length != dataset.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Sum of lengths {} does not equal dataset length {}",
            total_length,
            dataset.len()
        )));
    }
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    if let Some(_seed) = generator {
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::seq::ScientificSliceRandom;
        let mut rng = thread_rng();
        indices.scientific_shuffle(&mut rng);
    }
    let mut subsets = Vec::with_capacity(lengths.len());
    let mut offset = 0;
    for &length in lengths {
        let subset_indices = indices[offset..offset + length].to_vec();
        subsets.push(Subset::new(dataset.clone(), subset_indices));
        offset += length;
    }
    Ok(subsets)
}
/// Streaming dataset interface for real-time data processing
///
/// This trait represents datasets that can continuously produce data,
/// potentially from real-time sources or infinite data generators.
pub trait StreamingDataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;
    /// The streaming iterator type
    type Stream: Iterator<Item = Result<Self::Item>> + Send;
    /// Create a stream over the dataset
    fn stream(&self) -> Self::Stream;
    /// Check if the stream has more data available
    fn has_more(&self) -> bool {
        true
    }
    /// Reset the stream to the beginning (if supported)
    fn reset(&self) -> Result<()> {
        Ok(())
    }
}
/// Compute statistics for a tensor dataset
///
/// Returns feature statistics for each feature dimension in the dataset.
/// Only works with `TensorDataset<f32>` where the first tensor contains the features.
pub fn dataset_statistics(dataset: &TensorDataset<f32>) -> Result<Vec<FeatureStats>> {
    if dataset.len() == 0 {
        return Ok(Vec::new());
    }
    let first_item = dataset.get(0)?;
    if first_item.is_empty() {
        return Ok(Vec::new());
    }
    let features_tensor = &first_item[0];
    let n_features = features_tensor.numel();
    let mut feature_data: Vec<Vec<f32>> = vec![Vec::with_capacity(dataset.len()); n_features];
    for i in 0..dataset.len() {
        let item = dataset.get(i)?;
        if item.is_empty() {
            continue;
        }
        let features = &item[0];
        for feat_idx in 0..n_features.min(features.numel()) {
            if let Ok(indices) = torsh_tensor::Tensor::from_vec(vec![feat_idx as i64], &[1]) {
                if let Ok(value_tensor) = features.index_select(0, &indices) {
                    if let Ok(value) = value_tensor.item() {
                        feature_data[feat_idx].push(value);
                    }
                }
            }
        }
    }
    Ok(feature_data
        .iter()
        .map(|data| FeatureStats::from_data(data))
        .collect())
}
/// Stratified split that preserves class distribution
///
/// Splits data into train/val/test sets while maintaining the same class distribution
/// in each split as in the original dataset.
pub fn stratified_split<D>(
    dataset: D,
    labels: &[usize],
    train_ratio: f32,
    val_ratio: Option<f32>,
    random_seed: Option<u64>,
) -> Result<(Subset<D>, Subset<D>, Option<Subset<D>>)>
where
    D: Dataset + Clone,
{
    if train_ratio <= 0.0 || train_ratio >= 1.0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "train_ratio must be between 0 and 1".to_string(),
        ));
    }
    let has_val = val_ratio.is_some();
    let val_r = val_ratio.unwrap_or(0.0);
    if has_val && (train_ratio + val_r >= 1.0) {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "train_ratio + val_ratio must be less than 1".to_string(),
        ));
    }
    if labels.len() != dataset.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "labels length must equal dataset length".to_string(),
        ));
    }
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        class_indices.entry(label).or_default().push(idx);
    }
    use scirs2_core::random::prelude::*;
    use scirs2_core::random::seq::ScientificSliceRandom;
    use scirs2_core::random::SeedableRng;
    let mut rng = if let Some(seed) = random_seed {
        StdRng::seed_from_u64(seed)
    } else {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("time should be after UNIX_EPOCH")
            .as_secs();
        StdRng::seed_from_u64(seed)
    };
    let mut train_indices = Vec::new();
    let mut val_indices = Vec::new();
    let mut test_indices = Vec::new();
    for (_class, mut indices) in class_indices {
        indices.scientific_shuffle(&mut rng);
        let n_train = (indices.len() as f32 * train_ratio).round() as usize;
        let n_val = if has_val {
            (indices.len() as f32 * val_r).round() as usize
        } else {
            0
        };
        train_indices.extend_from_slice(&indices[0..n_train]);
        if has_val {
            val_indices.extend_from_slice(&indices[n_train..n_train + n_val]);
            test_indices.extend_from_slice(&indices[n_train + n_val..]);
        } else {
            test_indices.extend_from_slice(&indices[n_train..]);
        }
    }
    let train_subset = Subset::new(dataset.clone(), train_indices);
    let test_subset = Subset::new(dataset.clone(), test_indices);
    let val_subset = if has_val {
        Some(Subset::new(dataset, val_indices))
    } else {
        None
    };
    Ok((train_subset, test_subset, val_subset))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::types::*;
    use torsh_tensor::creation::*;
    #[test]
    fn test_tensor_dataset() {
        let data = ones::<f32>(&[10, 3]).unwrap();
        let labels = zeros::<f32>(&[10]).unwrap();
        let dataset = TensorDataset::from_tensors(vec![data, labels]);
        assert_eq!(dataset.len(), 10);
        let item = dataset.get(0).unwrap();
        assert_eq!(item.len(), 2);
    }
    #[test]
    fn test_concat_dataset() {
        let ds1 = TensorDataset::from_tensor(ones::<f32>(&[5, 3]).unwrap());
        let ds2 = TensorDataset::from_tensor(zeros::<f32>(&[3, 3]).unwrap());
        let concat = ConcatDataset::new(vec![ds1, ds2]);
        assert_eq!(concat.len(), 8);
        assert_eq!(concat.dataset_idx(0), Some((0, 0)));
        assert_eq!(concat.dataset_idx(4), Some((0, 4)));
        assert_eq!(concat.dataset_idx(5), Some((1, 0)));
        assert_eq!(concat.dataset_idx(7), Some((1, 2)));
        assert_eq!(concat.dataset_idx(8), None);
    }
    #[test]
    fn test_subset() {
        let dataset = TensorDataset::from_tensor(ones::<f32>(&[10, 3]).unwrap());
        let subset = Subset::new(dataset, vec![0, 2, 4, 6, 8]);
        assert_eq!(subset.len(), 5);
        assert!(subset.get(0).is_ok());
        assert!(subset.get(5).is_err());
    }
    #[derive(Clone)]
    struct SimpleIterableDataset {
        data: Vec<i32>,
    }
    impl IterableDataset for SimpleIterableDataset {
        type Item = i32;
        type Iter = std::iter::Map<std::vec::IntoIter<i32>, fn(i32) -> Result<i32>>;
        fn iter(&self) -> Self::Iter {
            self.data.clone().into_iter().map(|x| Ok(x) as Result<i32>)
        }
    }
    #[test]
    fn test_chain_dataset() {
        let ds1 = SimpleIterableDataset {
            data: vec![1, 2, 3],
        };
        let ds2 = SimpleIterableDataset {
            data: vec![4, 5, 6],
        };
        let ds3 = SimpleIterableDataset {
            data: vec![7, 8, 9],
        };
        let chain = ChainDataset::new(vec![ds1, ds2, ds3]);
        let collected: Result<Vec<_>> = chain.iter().collect();
        assert!(collected.is_ok());
        let values = collected.unwrap();
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
    #[test]
    fn test_chain_dataset_empty() {
        let chain: ChainDataset<SimpleIterableDataset> = ChainDataset::new(vec![]);
        let collected: Result<Vec<_>> = chain.iter().collect();
        assert!(collected.is_ok());
        let values = collected.unwrap();
        assert_eq!(values, Vec::<i32>::new());
    }
    #[test]
    fn test_chain_dataset_with_empty_datasets() {
        let ds1 = SimpleIterableDataset { data: vec![] };
        let ds2 = SimpleIterableDataset {
            data: vec![1, 2, 3],
        };
        let ds3 = SimpleIterableDataset { data: vec![] };
        let ds4 = SimpleIterableDataset { data: vec![4, 5] };
        let chain = ChainDataset::new(vec![ds1, ds2, ds3, ds4]);
        let collected: Result<Vec<_>> = chain.iter().collect();
        assert!(collected.is_ok());
        let values = collected.unwrap();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }
    #[test]
    fn test_infinite_dataset() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let dataset = InfiniteDataset::new(move || {
            let val = counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(val)
        });
        assert!(dataset.has_more());
        let mut stream = dataset.stream();
        assert_eq!(stream.next().unwrap().unwrap(), 0);
        assert_eq!(stream.next().unwrap().unwrap(), 1);
        assert_eq!(stream.next().unwrap().unwrap(), 2);
    }
    #[test]
    fn test_buffered_streaming_dataset() {
        let dataset = InfiniteDataset::new(|| Ok(42i32));
        let buffered = BufferedStreamingDataset::new(dataset, 5).with_prefetch(true);
        assert!(buffered.has_more());
        let mut stream = buffered.stream();
        for _ in 0..10 {
            assert_eq!(stream.next().unwrap().unwrap(), 42);
        }
    }
    #[test]
    fn test_data_pipeline() {
        let pipeline = DataPipeline::new()
            .add_transform(|x: i32| Ok(x * 2))
            .add_transform(|x: i32| Ok(x + 1));
        let result = pipeline.apply(5).unwrap();
        assert_eq!(result, 11);
    }
    #[test]
    fn test_pipeline_streaming_dataset() {
        let dataset = InfiniteDataset::new(|| Ok(5i32));
        let pipeline = DataPipeline::new()
            .add_transform(|x: i32| Ok(x * 2))
            .add_transform(|x: i32| Ok(x + 1));
        let pipeline_dataset = PipelineStreamingDataset::new(dataset, pipeline);
        assert!(pipeline_dataset.has_more());
        let mut stream = pipeline_dataset.stream();
        for _ in 0..5 {
            assert_eq!(stream.next().unwrap().unwrap(), 11);
        }
    }
    #[test]
    fn test_real_time_dataset() {
        let (dataset, _receiver) = RealTimeDataset::<i32>::new();
        let sender = dataset.sender();
        {
            let sender_lock = sender.lock().expect("lock should not be poisoned");
            sender_lock.send(1).unwrap();
            sender_lock.send(2).unwrap();
            sender_lock.send(3).unwrap();
        }
        assert!(dataset.has_more());
        let _stream = dataset.stream();
    }
    #[test]
    fn test_dataset_to_streaming() {
        let tensor = ones::<f32>(&[5, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let streaming = DatasetToStreaming::new(dataset);
        assert!(streaming.has_more());
        let stream = streaming.stream();
        let mut count = 0;
        for result in stream {
            assert!(result.is_ok());
            count += 1;
            if count >= 5 {
                break;
            }
        }
        assert_eq!(count, 5);
    }
    #[test]
    fn test_dataset_to_streaming_repeat() {
        let tensor = ones::<f32>(&[3, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let streaming = DatasetToStreaming::new(dataset).repeat();
        assert!(streaming.has_more());
        let stream = streaming.stream();
        let mut count = 0;
        for result in stream {
            assert!(result.is_ok());
            count += 1;
            if count >= 10 {
                break;
            }
        }
        assert_eq!(count, 10);
    }
    #[test]
    fn test_streaming_dataset_reset() {
        let dataset = InfiniteDataset::new(|| Ok(42i32));
        let buffered = BufferedStreamingDataset::new(dataset, 3);
        assert!(buffered.reset().is_ok());
    }
    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_sequential_access() {
        use std::thread;
        use std::time::Duration;
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);
        for i in 0..10 {
            let _ = profiled.get(i).unwrap();
            thread::sleep(Duration::from_micros(100));
        }
        let stats = profiled.stats();
        assert_eq!(stats.total_accesses, 10);
        assert_eq!(stats.sequential_accesses, 9);
        assert!(stats.sequential_ratio > 0.8);
        assert!(stats.avg_access_time_us > 0.0);
        assert!(stats.throughput_accesses_per_sec > 0.0);
    }
    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_random_access() {
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);
        let indices = [0, 5, 2, 8, 1];
        for &i in &indices {
            let _ = profiled.get(i).unwrap();
        }
        let stats = profiled.stats();
        assert_eq!(stats.total_accesses, 5);
        assert_eq!(stats.sequential_accesses, 0);
        assert_eq!(stats.sequential_ratio, 0.0);
    }
    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_hints() {
        let tensor = ones::<f32>(&[100, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);
        for i in 0..20 {
            let _ = profiled.get(i).unwrap();
        }
        let hints = profiled.hints();
        assert!(!hints.is_empty());
        assert!(hints
            .iter()
            .any(|h| h.contains("sequential") || h.contains("good")));
    }
    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_reset() {
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);
        for i in 0..5 {
            let _ = profiled.get(i).unwrap();
        }
        assert_eq!(profiled.stats().total_accesses, 5);
        profiled.profiler().reset();
        assert_eq!(profiled.stats().total_accesses, 0);
    }
    #[test]
    #[cfg(feature = "std")]
    fn test_dataset_profiler_display() {
        let tensor = ones::<f32>(&[10, 2]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let profiled = ProfiledDataset::new(dataset);
        for i in 0..5 {
            let _ = profiled.get(i).unwrap();
        }
        let stats_string = format!("{}", profiled.stats());
        assert!(stats_string.contains("Dataset Profile Statistics"));
        assert!(stats_string.contains("Total Accesses: 5"));
    }
    #[test]
    fn test_feature_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FeatureStats::from_data(&data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.std - 1.4142).abs() < 0.01);
    }
    #[test]
    fn test_feature_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = FeatureStats::from_data(&data);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }
    #[test]
    fn test_dataset_statistics() {
        let data = torsh_tensor::creation::randn::<f32>(&[10, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let stats = dataset_statistics(&dataset).unwrap();
        assert_eq!(stats.len(), 3);
        for stat in &stats {
            assert_eq!(stat.count, 10);
            assert!(stat.min <= stat.mean);
            assert!(stat.mean <= stat.max);
            assert!(stat.std >= 0.0);
        }
    }
    #[test]
    fn test_dataset_statistics_empty() {
        let data = torsh_tensor::creation::zeros::<f32>(&[0, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let stats = dataset_statistics(&dataset).unwrap();
        assert_eq!(stats.len(), 0);
    }
    #[test]
    fn test_kfold_basic() {
        let kfold = KFold::new(5, false, Some(42));
        let folds = kfold.split(100);
        assert_eq!(folds.len(), 5);
        for (fold_idx, (train_indices, val_indices)) in folds.iter().enumerate() {
            assert_eq!(val_indices.len(), 20);
            assert_eq!(train_indices.len(), 80);
            for &val_idx in val_indices {
                assert!(!train_indices.contains(&val_idx));
            }
            for &idx in train_indices.iter().chain(val_indices.iter()) {
                assert!(idx < 100);
            }
            println!(
                "Fold {}: train={}, val={}",
                fold_idx,
                train_indices.len(),
                val_indices.len()
            );
        }
    }
    #[test]
    fn test_kfold_shuffle() {
        let kfold_shuffled = KFold::new(3, true, Some(42));
        let kfold_unshuffled = KFold::new(3, false, None);
        let folds_shuffled = kfold_shuffled.split(30);
        let folds_unshuffled = kfold_unshuffled.split(30);
        assert_eq!(folds_shuffled.len(), folds_unshuffled.len());
        let shuffled_val = &folds_shuffled[0].1;
        let unshuffled_val = &folds_unshuffled[0].1;
        assert_eq!(unshuffled_val, &(0..10).collect::<Vec<_>>());
        assert_ne!(shuffled_val, unshuffled_val);
    }
    #[test]
    fn test_kfold_uneven_split() {
        let kfold = KFold::new(3, false, None);
        let folds = kfold.split(10);
        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0].1.len(), 3);
        assert_eq!(folds[1].1.len(), 3);
        assert_eq!(folds[2].1.len(), 4);
        let all_val_samples: usize = folds.iter().map(|(_, val)| val.len()).sum();
        assert_eq!(all_val_samples, 10);
    }
    #[test]
    #[should_panic(expected = "n_splits must be at least 2")]
    fn test_kfold_invalid_splits() {
        KFold::new(1, false, None);
    }
    #[test]
    fn test_stratified_split_binary() {
        let data = ones::<f32>(&[100, 5]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = (0..100).map(|i| if i < 50 { 0 } else { 1 }).collect();
        let (train, test, val) =
            stratified_split(dataset, &labels, 0.6, Some(0.2), Some(42)).unwrap();
        assert_eq!(train.len(), 60);
        assert!(val.is_some());
        assert_eq!(val.as_ref().unwrap().len(), 20);
        assert_eq!(test.len(), 20);
        println!(
            "Stratified split: train={}, val={}, test={}",
            train.len(),
            val.as_ref().unwrap().len(),
            test.len()
        );
    }
    #[test]
    fn test_stratified_split_multi_class() {
        let data = ones::<f32>(&[90, 5]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = (0..90).map(|i| i / 30).collect();
        let (train, test, _val) = stratified_split(dataset, &labels, 0.7, None, Some(42)).unwrap();
        assert_eq!(train.len(), 63);
        assert_eq!(test.len(), 27);
        println!(
            "Multi-class split: train={}, test={}",
            train.len(),
            test.len()
        );
    }
    #[test]
    fn test_stratified_split_no_val() {
        let data = ones::<f32>(&[50, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = (0..50).map(|i| i % 2).collect();
        let (train, test, val) = stratified_split(dataset, &labels, 0.8, None, Some(42)).unwrap();
        assert_eq!(train.len(), 40);
        assert_eq!(test.len(), 10);
        assert!(val.is_none());
    }
    #[test]
    fn test_stratified_split_invalid_ratio() {
        let data = ones::<f32>(&[50, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = (0..50).map(|i| i % 2).collect();
        let result = stratified_split(dataset.clone(), &labels, 1.0, None, None);
        assert!(result.is_err());
        let result = stratified_split(dataset, &labels, 0.7, Some(0.4), None);
        assert!(result.is_err());
    }
    #[test]
    fn test_stratified_split_mismatched_labels() {
        let data = ones::<f32>(&[50, 3]).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        let labels: Vec<usize> = vec![0, 1];
        let result = stratified_split(dataset, &labels, 0.8, None, None);
        assert!(result.is_err());
    }
    #[test]
    fn test_kfold_reproducibility() {
        let kfold1 = KFold::new(5, true, Some(42));
        let kfold2 = KFold::new(5, true, Some(42));
        let folds1 = kfold1.split(50);
        let folds2 = kfold2.split(50);
        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.0, f2.0);
            assert_eq!(f1.1, f2.1);
        }
    }
    #[test]
    fn test_stratified_split_reproducibility() {
        let data = ones::<f32>(&[100, 5]).unwrap();
        let labels: Vec<usize> = (0..100).map(|i| i % 3).collect();
        let (train1, test1, _) = stratified_split(
            TensorDataset::from_tensor(data.clone()),
            &labels,
            0.7,
            None,
            Some(42),
        )
        .unwrap();
        let (train2, test2, _) = stratified_split(
            TensorDataset::from_tensor(data),
            &labels,
            0.7,
            None,
            Some(42),
        )
        .unwrap();
        assert_eq!(train1.len(), train2.len());
        assert_eq!(test1.len(), test2.len());
    }
}
