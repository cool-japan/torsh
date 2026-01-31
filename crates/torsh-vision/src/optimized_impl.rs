//! Optimized dataset implementations with lazy loading and memory management
//!
//! This module provides memory-efficient alternatives to the basic dataset implementations
//! with features like lazy loading, caching integration, and unified interfaces.

use crate::error_handling::{EnhancedVisionError, ErrorHandler};
use crate::io::VisionIO;
use crate::ImageCache;
use crate::{Result, VisionError};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use torsh_tensor::{creation, Tensor};

/// Configuration for optimized datasets
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Maximum number of items to keep in memory cache
    pub max_cache_items: usize,
    /// Maximum memory usage in MB for cached items
    pub max_cache_memory_mb: usize,
    /// Enable background prefetching
    pub enable_prefetch: bool,
    /// Batch size for prefetching
    pub prefetch_batch_size: usize,
    /// Enable data validation
    pub validate_data: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            max_cache_items: 1000,
            max_cache_memory_mb: 512,
            enable_prefetch: true,
            prefetch_batch_size: 16,
            validate_data: true,
        }
    }
}

/// Generic dataset trait for unified interface
pub trait OptimizedDataset: Send + Sync {
    type Item;

    /// Get the total number of items in the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an item by index with lazy loading
    fn get_item(&self, index: usize) -> Result<Self::Item>;

    /// Get multiple items efficiently with batching
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Item>> {
        indices.iter().map(|&i| self.get_item(i)).collect()
    }

    /// Get dataset metadata
    fn metadata(&self) -> DatasetMetadata;

    /// Prefetch items for improved performance
    fn prefetch(&self, indices: &[usize]) -> Result<()>;

    /// Clear cache to free memory
    fn clear_cache(&self);

    /// Get cache statistics
    fn cache_stats(&self) -> CacheStatistics;
}

/// Dataset metadata information
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: String,
    pub version: String,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub total_items: usize,
    pub item_shape: Vec<usize>,
    pub data_type: String,
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f64,
    pub memory_usage_mb: f64,
    pub cached_items: usize,
}

/// Optimized image classification dataset with lazy loading
pub struct OptimizedImageDataset {
    config: DatasetConfig,
    io: Arc<VisionIO>,
    cache: Arc<ImageCache>,
    image_paths: Vec<PathBuf>,
    labels: Vec<usize>,
    class_names: Vec<String>,
    class_to_idx: HashMap<String, usize>,
    metadata: DatasetMetadata,
}

impl OptimizedImageDataset {
    /// Create a new optimized image dataset
    pub fn new<P: AsRef<Path>>(root: P, config: DatasetConfig) -> Result<Self> {
        let root_path = root.as_ref();

        if !root_path.exists() {
            return Err(VisionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Dataset directory {:?} does not exist", root_path),
            )));
        }

        let io = Arc::new(VisionIO::new());
        let cache = Arc::new(ImageCache::new(config.max_cache_memory_mb));

        let mut image_paths = Vec::new();
        let mut labels = Vec::new();
        let mut class_names = Vec::new();
        let mut class_to_idx = HashMap::new();

        // Scan directory structure for classes and images
        for entry in std::fs::read_dir(root_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(class_name) = path.file_name() {
                    let class_str = class_name.to_string_lossy().to_string();
                    let class_idx = class_names.len();
                    class_names.push(class_str.clone());
                    class_to_idx.insert(class_str, class_idx);

                    // Scan images in this class directory
                    for img_entry in std::fs::read_dir(&path)? {
                        let img_entry = img_entry?;
                        let img_path = img_entry.path();

                        if img_path.is_file() && io.is_supported_image(&img_path) {
                            image_paths.push(img_path);
                            labels.push(class_idx);
                        }
                    }
                }
            }
        }

        if class_names.is_empty() {
            return Err(VisionError::TransformError(
                "No class directories found".to_string(),
            ));
        }

        let metadata = DatasetMetadata {
            name: "OptimizedImageDataset".to_string(),
            version: "1.0".to_string(),
            num_classes: class_names.len(),
            class_names: class_names.clone(),
            total_items: image_paths.len(),
            item_shape: vec![3, 224, 224], // Default shape, will be updated dynamically
            data_type: "f32".to_string(),
        };

        Ok(Self {
            config,
            io,
            cache,
            image_paths,
            labels,
            class_names,
            class_to_idx,
            metadata,
        })
    }

    /// Get class information
    pub fn classes(&self) -> &[String] {
        &self.class_names
    }

    /// Get class to index mapping
    pub fn class_to_idx(&self) -> &HashMap<String, usize> {
        &self.class_to_idx
    }
}

impl OptimizedDataset for OptimizedImageDataset {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.image_paths.len()
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index >= self.image_paths.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Index {} out of bounds for dataset of size {}",
                index,
                self.image_paths.len()
            )));
        }

        let image_path = &self.image_paths[index];
        let label = self.labels[index];

        // Load image using cached I/O
        let image = self.cache.get_or_load(image_path)?;

        // Convert to tensor
        let tensor = crate::utils::image_to_tensor(&image)?;

        Ok((tensor, label))
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Item>> {
        // Validate all indices first
        for &index in indices {
            if index >= self.image_paths.len() {
                return Err(VisionError::InvalidArgument(format!(
                    "Index {} out of bounds for dataset of size {}",
                    index,
                    self.image_paths.len()
                )));
            }
        }

        // Use parallel loading for large batches
        if indices.len() > 4 {
            // For now, fall back to sequential loading
            // In the future, could implement parallel loading with rayon
            indices.iter().map(|&i| self.get_item(i)).collect()
        } else {
            indices.iter().map(|&i| self.get_item(i)).collect()
        }
    }

    fn metadata(&self) -> DatasetMetadata {
        self.metadata.clone()
    }

    fn prefetch(&self, indices: &[usize]) -> Result<()> {
        if !self.config.enable_prefetch {
            return Ok(());
        }

        // Collect image paths for prefetching
        let paths: Vec<_> = indices
            .iter()
            .filter_map(|&i| self.image_paths.get(i))
            .collect();

        // Trigger background loading (simplified implementation)
        for path in paths {
            let _ = self.cache.get_or_load(path);
        }

        Ok(())
    }

    fn clear_cache(&self) {
        self.cache.clear();
    }

    fn cache_stats(&self) -> CacheStatistics {
        let stats = self.cache.stats();
        CacheStatistics {
            cache_hits: stats.hit_count,
            cache_misses: stats.miss_count,
            hit_rate: stats.hit_rate,
            memory_usage_mb: stats.current_size_bytes as f64 / (1024.0 * 1024.0),
            cached_items: stats.entry_count,
        }
    }
}

/// Optimized CIFAR dataset with lazy loading and validation
pub struct OptimizedCIFARDataset {
    config: DatasetConfig,
    data_path: PathBuf,
    is_cifar100: bool,
    is_train: bool,
    cached_data: Arc<std::sync::Mutex<HashMap<usize, (Tensor<f32>, usize)>>>,
    classes: Vec<String>,
    total_samples: usize,
    metadata: DatasetMetadata,
}

impl OptimizedCIFARDataset {
    /// Create a new optimized CIFAR dataset
    pub fn new<P: AsRef<Path>>(
        root: P,
        is_cifar100: bool,
        train: bool,
        config: DatasetConfig,
    ) -> Result<Self> {
        let root_path = root.as_ref();

        if !root_path.exists() {
            std::fs::create_dir_all(root_path)?;
        }

        let (data_path, total_samples, classes) = if is_cifar100 {
            let file_name = if train { "train.bin" } else { "test.bin" };
            let path = root_path.join(file_name);

            if !path.exists() {
                return Err(VisionError::TransformError(
                    format!("CIFAR-100 {} file not found in {:?}. Please download from https://www.cs.toronto.edu/~kriz/cifar.html", 
                           if train { "training" } else { "test" }, root_path)
                ));
            }

            let samples = if train { 50000 } else { 10000 };
            let classes = Self::get_cifar100_classes();
            (path, samples, classes)
        } else {
            // CIFAR-10
            let total_samples = if train { 50000 } else { 10000 };
            let path = if train {
                root_path.join("data_batch_1.bin") // We'll handle all batches in get_item
            } else {
                root_path.join("test_batch.bin")
            };

            let classes = vec![
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

            (path, total_samples, classes)
        };

        let metadata = DatasetMetadata {
            name: if is_cifar100 { "CIFAR-100" } else { "CIFAR-10" }.to_string(),
            version: "1.0".to_string(),
            num_classes: classes.len(),
            class_names: classes.clone(),
            total_items: total_samples,
            item_shape: vec![3, 32, 32],
            data_type: "f32".to_string(),
        };

        Ok(Self {
            config,
            data_path,
            is_cifar100,
            is_train: train,
            cached_data: Arc::new(std::sync::Mutex::new(HashMap::new())),
            classes,
            total_samples,
            metadata,
        })
    }

    fn get_cifar100_classes() -> Vec<String> {
        vec![
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn load_cifar_sample(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        // Check cache first
        {
            let cache = self
                .cached_data
                .lock()
                .expect("lock should not be poisoned");
            if let Some(cached_item) = cache.get(&index) {
                return Ok(cached_item.clone());
            }
        }

        // Load from disk
        let (tensor, label) = if self.is_cifar100 {
            self.load_cifar100_sample(index)?
        } else {
            self.load_cifar10_sample(index)?
        };

        // Cache the result
        {
            let mut cache = self
                .cached_data
                .lock()
                .expect("lock should not be poisoned");
            if cache.len() < self.config.max_cache_items {
                cache.insert(index, (tensor.clone(), label));
            }
        }

        Ok((tensor, label))
    }

    fn load_cifar10_sample(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        let (batch_idx, sample_idx) = if self.is_train {
            (index / 10000, index % 10000)
        } else {
            (0, index) // Test batch
        };

        let batch_file = if self.is_train {
            self.data_path
                .parent()
                .expect("data_path should have parent")
                .join(format!("data_batch_{}.bin", batch_idx + 1))
        } else {
            self.data_path.clone()
        };

        if !batch_file.exists() {
            return Err(VisionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("CIFAR-10 batch file {:?} not found", batch_file),
            )));
        }

        let data = std::fs::read(batch_file)?;
        let start_idx = sample_idx * 3073; // 1 label + 3072 pixels

        if start_idx + 3073 > data.len() {
            return Err(VisionError::TransformError(
                "Invalid CIFAR-10 file format".to_string(),
            ));
        }

        let label = data[start_idx] as usize;
        let tensor = torsh_tensor::creation::zeros(&[3, 32, 32])
            .map_err(|e| VisionError::TransformError(format!("Failed to create tensor: {}", e)))?;

        // Load RGB channels
        for channel in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    let pixel_idx = start_idx + 1 + channel * 1024 + y * 32 + x;
                    let pixel_val = data[pixel_idx] as f32 / 255.0;
                    tensor.set(&[channel, y, x], pixel_val)?;
                }
            }
        }

        Ok((tensor, label))
    }

    fn load_cifar100_sample(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        let data = std::fs::read(&self.data_path)?;
        let start_idx = index * 3074; // 2 labels + 3072 pixels

        if start_idx + 3074 > data.len() {
            return Err(VisionError::TransformError(
                "Invalid CIFAR-100 file format".to_string(),
            ));
        }

        let _coarse_label = data[start_idx] as usize;
        let fine_label = data[start_idx + 1] as usize;
        let tensor = torsh_tensor::creation::zeros(&[3, 32, 32])
            .map_err(|e| VisionError::TransformError(format!("Failed to create tensor: {}", e)))?;

        // Load RGB channels
        for channel in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    let pixel_idx = start_idx + 2 + channel * 1024 + y * 32 + x;
                    let pixel_val = data[pixel_idx] as f32 / 255.0;
                    tensor.set(&[channel, y, x], pixel_val)?;
                }
            }
        }

        Ok((tensor, fine_label))
    }

    /// Get class information
    pub fn classes(&self) -> &[String] {
        &self.classes
    }
}

impl OptimizedDataset for OptimizedCIFARDataset {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.total_samples
    }

    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index >= self.total_samples {
            return Err(VisionError::InvalidArgument(format!(
                "Index {} out of bounds for dataset of size {}",
                index, self.total_samples
            )));
        }

        self.load_cifar_sample(index)
    }

    fn metadata(&self) -> DatasetMetadata {
        self.metadata.clone()
    }

    fn prefetch(&self, indices: &[usize]) -> Result<()> {
        if !self.config.enable_prefetch {
            return Ok(());
        }

        for &index in indices {
            if index < self.total_samples {
                let _ = self.load_cifar_sample(index);
            }
        }

        Ok(())
    }

    fn clear_cache(&self) {
        let mut cache = self
            .cached_data
            .lock()
            .expect("lock should not be poisoned");
        cache.clear();
    }

    fn cache_stats(&self) -> CacheStatistics {
        let cache = self
            .cached_data
            .lock()
            .expect("lock should not be poisoned");
        CacheStatistics {
            cache_hits: 0, // Would need to track this
            cache_misses: 0,
            hit_rate: 0.0,
            memory_usage_mb: cache.len() as f64 * 3.0 * 32.0 * 32.0 * 4.0 / (1024.0 * 1024.0), // Rough estimate
            cached_items: cache.len(),
        }
    }
}

/// Optimized dataset builder for easy configuration
pub struct OptimizedDatasetBuilder {
    config: DatasetConfig,
}

impl OptimizedDatasetBuilder {
    /// Create a new dataset builder
    pub fn new() -> Self {
        Self {
            config: DatasetConfig::default(),
        }
    }

    /// Set cache configuration
    pub fn with_cache(mut self, max_items: usize, max_memory_mb: usize) -> Self {
        self.config.max_cache_items = max_items;
        self.config.max_cache_memory_mb = max_memory_mb;
        self
    }

    /// Enable or disable prefetching
    pub fn with_prefetch(mut self, enable: bool, batch_size: usize) -> Self {
        self.config.enable_prefetch = enable;
        self.config.prefetch_batch_size = batch_size;
        self
    }

    /// Enable or disable data validation
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.config.validate_data = enable;
        self
    }

    /// Build an optimized image dataset
    pub fn build_image_dataset<P: AsRef<Path>>(self, root: P) -> Result<OptimizedImageDataset> {
        OptimizedImageDataset::new(root, self.config)
    }

    /// Build an optimized CIFAR dataset
    pub fn build_cifar_dataset<P: AsRef<Path>>(
        self,
        root: P,
        is_cifar100: bool,
        train: bool,
    ) -> Result<OptimizedCIFARDataset> {
        OptimizedCIFARDataset::new(root, is_cifar100, train, self.config)
    }
}

impl Default for OptimizedDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_dataset_builder() {
        let builder = OptimizedDatasetBuilder::new()
            .with_cache(500, 256)
            .with_prefetch(true, 8)
            .with_validation(true);

        assert_eq!(builder.config.max_cache_items, 500);
        assert_eq!(builder.config.max_cache_memory_mb, 256);
        assert_eq!(builder.config.enable_prefetch, true);
        assert_eq!(builder.config.prefetch_batch_size, 8);
        assert_eq!(builder.config.validate_data, true);
    }

    #[test]
    fn test_dataset_metadata() {
        let metadata = DatasetMetadata {
            name: "Test".to_string(),
            version: "1.0".to_string(),
            num_classes: 10,
            class_names: vec!["class1".to_string(), "class2".to_string()],
            total_items: 1000,
            item_shape: vec![3, 32, 32],
            data_type: "f32".to_string(),
        };

        assert_eq!(metadata.name, "Test");
        assert_eq!(metadata.num_classes, 10);
        assert_eq!(metadata.total_items, 1000);
    }
}
