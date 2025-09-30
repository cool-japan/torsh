# torsh-data

Data loading and preprocessing framework for ToRSh with PyTorch-compatible API.

## Overview

This crate provides comprehensive data handling utilities including:

- **Datasets**: Abstract interfaces and common implementations
- **DataLoader**: Efficient multi-threaded data loading with batching
- **Samplers**: Various sampling strategies for data iteration
- **Transformations**: Common data preprocessing operations
- **Domain-specific support**: Vision, audio, and tabular data

Note: This crate can leverage scirs2-datasets for additional dataset utilities and sample datasets.

## Features

- `std` (default): Standard library support
- `image-support`: Enable image loading and vision datasets
- `audio-support`: Enable audio processing capabilities
- `dataframe`: Enable tabular data support with Polars integration

## Usage

### Basic Dataset and DataLoader

```rust
use torsh_data::prelude::*;
use torsh_tensor::prelude::*;

// Create a simple tensor dataset
let data = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let labels = tensor![0, 1, 0];
let dataset = TensorDataset::new(vec![data, labels]);

// Create a dataloader
let dataloader = DataLoader::builder(dataset)
    .batch_size(2)
    .shuffle(true)
    .num_workers(4)
    .build()?;

// Iterate through batches
for batch in dataloader {
    let inputs = &batch[0];
    let targets = &batch[1];
    // Process batch...
}
```

### Custom Dataset Implementation

```rust
use torsh_data::dataset::Dataset;
use torsh_tensor::Tensor;
use std::path::PathBuf;

struct ImageDataset {
    image_paths: Vec<PathBuf>,
    labels: Vec<i64>,
    transform: Option<Box<dyn Transform>>,
}

impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.image_paths.len()
    }
    
    fn get(&self, index: usize) -> Result<Vec<Tensor>> {
        // Load image from path
        let image = load_image(&self.image_paths[index])?;
        
        // Apply transformations if any
        let image = if let Some(transform) = &self.transform {
            transform.apply(image)?
        } else {
            image
        };
        
        // Return image and label
        Ok(vec![image, tensor![self.labels[index]]])
    }
}
```

### Samplers

```rust
// Sequential sampling
let sampler = SequentialSampler::new(dataset.len());

// Random sampling with replacement
let sampler = RandomSampler::new(dataset.len())
    .with_replacement(true)
    .num_samples(10000);

// Batch sampling
let batch_sampler = BatchSampler::new(sampler, 32, false);

// Use with DataLoader
let dataloader = DataLoader::builder(dataset)
    .batch_sampler(batch_sampler)
    .build()?;
```

### Data Transformations

```rust
use torsh_data::transforms::{Compose, Normalize, RandomCrop, ToTensor};

// Create a transformation pipeline
let transform = Compose::new(vec![
    Box::new(RandomCrop::new(224)),
    Box::new(ToTensor::new()),
    Box::new(Normalize::new(
        vec![0.485, 0.456, 0.406],
        vec![0.229, 0.224, 0.225],
    )),
]);

// Apply to data
let transformed = transform.apply(image)?;
```

### Collate Functions

```rust
use torsh_data::collate::{collate_fn, PadSequence};

// Default collate function (stacks tensors)
let dataloader = DataLoader::builder(dataset)
    .collate_fn(collate_fn)
    .build()?;

// Custom collate for variable-length sequences
let pad_collate = PadSequence::new(0.0, true, -100);
let dataloader = DataLoader::builder(dataset)
    .collate_fn(move |batch| pad_collate.collate(batch))
    .build()?;
```

### Vision Datasets (with `image-support` feature)

```rust
#[cfg(feature = "image-support")]
use torsh_data::vision::{ImageFolder, CIFAR10, MNIST};

// Load from directory structure
let dataset = ImageFolder::new("path/to/images")?
    .with_transform(transform);

// Built-in datasets
let mnist = MNIST::new("./data", true, Some(transform), true)?;
let cifar = CIFAR10::new("./data", true, Some(transform), true)?;
```

### Tabular Data (with `dataframe` feature)

```rust
#[cfg(feature = "dataframe")]
use torsh_data::tabular::CSVDataset;

let dataset = CSVDataset::new("data.csv")?
    .with_target_column("label")
    .with_features(vec!["feature1", "feature2", "feature3"])
    .with_dtype(DType::F32);
```

### Audio Support (with `audio-support` feature)

```rust
#[cfg(feature = "audio-support")]
use torsh_data::audio::{AudioDataset, Spectrogram};

let dataset = AudioDataset::new(audio_files)?
    .with_sample_rate(16000)
    .with_transform(Spectrogram::new(n_fft, hop_length));
```

### Multi-Processing and Performance

```rust
// Parallel data loading
let dataloader = DataLoader::builder(dataset)
    .batch_size(64)
    .num_workers(8)  // Parallel loading threads
    .prefetch_factor(2)  // Prefetch batches
    .persistent_workers(true)  // Keep workers alive
    .pin_memory(true)  // Pin memory for faster GPU transfer
    .build()?;
```

## Integration with Polars

When the `dataframe` feature is enabled, torsh-data integrates with Polars for efficient tabular data handling:

```rust
use polars::prelude::*;
use torsh_data::tabular::DataFrameDataset;

let df = CsvReader::from_path("data.csv")?
    .has_header(true)
    .finish()?;

let dataset = DataFrameDataset::from_dataframe(df)
    .with_features(["col1", "col2"])
    .with_target("label");
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.