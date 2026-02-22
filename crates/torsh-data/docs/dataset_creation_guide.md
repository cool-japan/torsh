# Dataset Creation Guide

This guide provides comprehensive instructions for creating custom datasets in the ToRSh data loading framework.

## Overview

The ToRSh data loading framework provides two main dataset types:
- **Map-style datasets**: Support random access with known length (implements `Dataset` trait)
- **Iterable-style datasets**: Support sequential iteration, may not have known length (implements `IterableDataset` trait)

## Core Traits

### Dataset Trait

```rust
pub trait Dataset: Send + Sync {
    type Item;
    
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn get(&self, index: usize) -> Result<Self::Item>;
}
```

### IterableDataset Trait

```rust
pub trait IterableDataset: Send + Sync {
    type Item;
    type Iter: Iterator<Item = Result<Self::Item>> + Send;
    
    fn iter(&self) -> Self::Iter;
}
```

## Built-in Dataset Types

### TensorDataset

A simple dataset wrapping tensors, treating the first dimension as the dataset size.

```rust
use torsh_data::dataset::TensorDataset;
use torsh_tensor::Tensor;

// Create from a single tensor
let data = Tensor::randn(&[100, 784], None)?; // 100 samples, 784 features
let dataset = TensorDataset::from_tensor(data);

// Create from multiple tensors (e.g., features and labels)
let features = Tensor::randn(&[100, 784], None)?;
let labels = Tensor::zeros(&[100], None)?;
let dataset = TensorDataset::from_tensors(vec![features, labels]);
```

### ConcatDataset

Concatenates multiple datasets into a single dataset.

```rust
use torsh_data::dataset::ConcatDataset;

let dataset1 = TensorDataset::from_tensor(tensor1);
let dataset2 = TensorDataset::from_tensor(tensor2);
let concat_dataset = ConcatDataset::new(vec![dataset1, dataset2]);
```

### Subset

Creates a subset of a dataset using specified indices.

```rust
use torsh_data::dataset::Subset;

let indices = vec![0, 2, 4, 6, 8]; // Select even indices
let subset = Subset::new(dataset, indices);
```

### ChainDataset

Chains multiple iterable datasets sequentially.

```rust
use torsh_data::dataset::ChainDataset;

let chain_dataset = ChainDataset::new(vec![iterable_dataset1, iterable_dataset2]);
```

## Creating Custom Map-Style Datasets

### Basic Custom Dataset

```rust
use torsh_data::dataset::Dataset;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

#[derive(Clone)]
pub struct CustomDataset {
    data: Vec<f32>,
    labels: Vec<i32>,
}

impl CustomDataset {
    pub fn new(data: Vec<f32>, labels: Vec<i32>) -> Self {
        assert_eq!(data.len(), labels.len(), "Data and labels must have same length");
        Self { data, labels }
    }
}

impl Dataset for CustomDataset {
    type Item = (Tensor<f32>, Tensor<i32>);
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        
        let data_tensor = Tensor::from_slice(&[self.data[index]], &[1])?;
        let label_tensor = Tensor::from_slice(&[self.labels[index]], &[1])?;
        
        Ok((data_tensor, label_tensor))
    }
}
```

### File-based Dataset

```rust
use std::path::Path;
use std::fs;

#[derive(Clone)]
pub struct FileDataset {
    file_paths: Vec<String>,
}

impl FileDataset {
    pub fn new<P: AsRef<Path>>(directory: P) -> Result<Self> {
        let mut file_paths = Vec::new();
        
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                file_paths.push(path.to_string_lossy().to_string());
            }
        }
        
        Ok(Self { file_paths })
    }
}

impl Dataset for FileDataset {
    type Item = Vec<u8>;
    
    fn len(&self) -> usize {
        self.file_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        
        let file_content = fs::read(&self.file_paths[index])?;
        Ok(file_content)
    }
}
```

## Creating Custom Iterable-Style Datasets

### Basic Iterable Dataset

```rust
use torsh_data::dataset::IterableDataset;

#[derive(Clone)]
pub struct InfiniteDataset {
    seed: u64,
}

impl InfiniteDataset {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

pub struct InfiniteDatasetIter {
    rng: rand::rngs::StdRng,
}

impl Iterator for InfiniteDatasetIter {
    type Item = Result<Tensor<f32>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        use rand::RngCore;
        
        // Generate random tensor
        let data: Vec<f32> = (0..784)
            .map(|_| (self.rng.next_u32() as f32) / (u32::MAX as f32))
            .collect();
        
        match Tensor::from_slice(&data, &[784]) {
            Ok(tensor) => Some(Ok(tensor)),
            Err(e) => Some(Err(e)),
        }
    }
}

impl IterableDataset for InfiniteDataset {
    type Item = Tensor<f32>;
    type Iter = InfiniteDatasetIter;
    
    fn iter(&self) -> Self::Iter {
        use rand::SeedableRng;
        
        InfiniteDatasetIter {
            rng: rand::rngs::StdRng::seed_from_u64(self.seed),
        }
    }
}
```

### Streaming Dataset

```rust
use std::io::{BufRead, BufReader};
use std::fs::File;

#[derive(Clone)]
pub struct StreamingDataset {
    file_path: String,
}

impl StreamingDataset {
    pub fn new<P: AsRef<Path>>(file_path: P) -> Self {
        Self {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
        }
    }
}

pub struct StreamingDatasetIter {
    reader: BufReader<File>,
}

impl Iterator for StreamingDatasetIter {
    type Item = Result<String>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                line.pop(); // Remove newline
                Some(Ok(line))
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

impl IterableDataset for StreamingDataset {
    type Item = String;
    type Iter = StreamingDatasetIter;
    
    fn iter(&self) -> Self::Iter {
        let file = File::open(&self.file_path).unwrap();
        StreamingDatasetIter {
            reader: BufReader::new(file),
        }
    }
}
```

## Advanced Dataset Patterns

### Cached Dataset

```rust
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

#[derive(Clone)]
pub struct CachedDataset<D: Dataset> {
    inner: D,
    cache: Arc<RwLock<HashMap<usize, D::Item>>>,
}

impl<D: Dataset> CachedDataset<D> {
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl<D: Dataset> Dataset for CachedDataset<D>
where
    D::Item: Clone,
{
    type Item = D::Item;
    
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        // Try to get from cache first
        if let Ok(cache) = self.cache.read() {
            if let Some(item) = cache.get(&index) {
                return Ok(item.clone());
            }
        }
        
        // Not in cache, fetch from inner dataset
        let item = self.inner.get(index)?;
        
        // Store in cache
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(index, item.clone());
        }
        
        Ok(item)
    }
}
```

### Lazy Dataset

```rust
use std::sync::Arc;

#[derive(Clone)]
pub struct LazyDataset<T> {
    generator: Arc<dyn Fn(usize) -> Result<T> + Send + Sync>,
    length: usize,
}

impl<T> LazyDataset<T> {
    pub fn new<F>(generator: F, length: usize) -> Self
    where
        F: Fn(usize) -> Result<T> + Send + Sync + 'static,
    {
        Self {
            generator: Arc::new(generator),
            length,
        }
    }
}

impl<T> Dataset for LazyDataset<T> {
    type Item = T;
    
    fn len(&self) -> usize {
        self.length
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        
        (self.generator)(index)
    }
}
```

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```rust
impl Dataset for MyDataset {
    type Item = MyItem;
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        // Check bounds
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        
        // Handle potential errors in data loading
        match self.load_data(index) {
            Ok(data) => Ok(data),
            Err(e) => Err(torsh_core::error::TorshError::DataError(e.to_string())),
        }
    }
}
```

### 2. Thread Safety

Ensure your dataset is thread-safe by implementing `Send + Sync`:

```rust
// Use Arc for shared immutable data
use std::sync::Arc;

#[derive(Clone)]
pub struct ThreadSafeDataset {
    data: Arc<Vec<f32>>,
}

// Use RwLock for shared mutable data
use std::sync::RwLock;

#[derive(Clone)]
pub struct MutableDataset {
    data: Arc<RwLock<Vec<f32>>>,
}
```

### 3. Memory Efficiency

Consider memory usage for large datasets:

```rust
// Use memory mapping for large files
use memmap2::MmapOptions;

#[derive(Clone)]
pub struct MmapDataset {
    mmap: Arc<memmap2::Mmap>,
    item_size: usize,
}

impl Dataset for MmapDataset {
    type Item = &[u8];
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let start = index * self.item_size;
        let end = start + self.item_size;
        
        if end > self.mmap.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        
        Ok(&self.mmap[start..end])
    }
}
```

### 4. Deterministic Behavior

Ensure reproducible results when needed:

```rust
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone)]
pub struct RandomDataset {
    seed: u64,
    length: usize,
}

impl Dataset for RandomDataset {
    type Item = f32;
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        // Use index as additional seed for deterministic per-item generation
        let mut rng = StdRng::seed_from_u64(self.seed + index as u64);
        Ok(rng.gen())
    }
}
```

## Testing Custom Datasets

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_custom_dataset_length() {
        let dataset = CustomDataset::new(vec![1.0, 2.0, 3.0], vec![0, 1, 2]);
        assert_eq!(dataset.len(), 3);
    }
    
    #[test]
    fn test_custom_dataset_get() {
        let dataset = CustomDataset::new(vec![1.0, 2.0, 3.0], vec![0, 1, 2]);
        let item = dataset.get(1).unwrap();
        // Add assertions for item content
    }
    
    #[test]
    fn test_custom_dataset_bounds() {
        let dataset = CustomDataset::new(vec![1.0], vec![0]);
        assert!(dataset.get(1).is_err());
    }
}
```

### Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_data::dataloader::DataLoader;
    
    #[test]
    fn test_dataset_with_dataloader() {
        let dataset = CustomDataset::new(vec![1.0, 2.0, 3.0], vec![0, 1, 2]);
        let dataloader = DataLoader::new(dataset)
            .batch_size(2)
            .shuffle(true)
            .build();
        
        let mut batch_count = 0;
        for batch in dataloader {
            batch_count += 1;
            assert!(batch.is_ok());
        }
        
        assert_eq!(batch_count, 2); // 3 items, batch size 2 -> 2 batches
    }
}
```

## Common Patterns

### Data Preprocessing

```rust
use torsh_data::transforms::Transform;

#[derive(Clone)]
pub struct PreprocessedDataset<D: Dataset, T: Transform<D::Item>> {
    inner: D,
    transform: T,
}

impl<D: Dataset, T: Transform<D::Item>> Dataset for PreprocessedDataset<D, T> {
    type Item = T::Output;
    
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let item = self.inner.get(index)?;
        Ok(self.transform.transform(item))
    }
}
```

### Multi-Modal Datasets

```rust
pub struct MultiModalDataset {
    text_data: Vec<String>,
    image_data: Vec<Vec<u8>>,
    audio_data: Vec<Vec<f32>>,
}

impl Dataset for MultiModalDataset {
    type Item = (String, Vec<u8>, Vec<f32>);
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        Ok((
            self.text_data[index].clone(),
            self.image_data[index].clone(),
            self.audio_data[index].clone(),
        ))
    }
}
```

This guide provides a comprehensive foundation for creating custom datasets in the ToRSh framework. Adapt these patterns to your specific use case and data types.