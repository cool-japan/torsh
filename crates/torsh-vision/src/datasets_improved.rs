//! Improved dataset implementations for torsh-vision
//! 
//! This module provides efficient, lazy-loading dataset implementations with:
//! - Common Dataset trait for unified interface
//! - Lazy loading support for large datasets
//! - Memory-efficient caching mechanisms  
//! - Transform pipelines integrated into datasets
//! - Advanced indexing and sampling strategies

use crate::{Result, VisionError, io::VisionIO};
use crate::transforms::Transform;
use image::DynamicImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use torsh_tensor::{Tensor, creation};
use parking_lot::RwLock;

/// Common trait for all vision datasets
pub trait Dataset: Send + Sync {
    /// Get the number of samples in the dataset
    fn len(&self) -> usize;
    
    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a sample by index, returning (image_tensor, label)
    fn get_item(&self, index: usize) -> Result<(Tensor<f32>, usize)>;
    
    /// Get class names if available
    fn class_names(&self) -> Option<&[String]> {
        None
    }
    
    /// Get the number of classes
    fn num_classes(&self) -> usize {
        self.class_names().map(|c| c.len()).unwrap_or(0)
    }
    
    /// Get dataset metadata
    fn metadata(&self) -> DatasetMetadata {
        DatasetMetadata {
            name: self.name(),
            num_samples: self.len(),
            num_classes: self.num_classes(),
            image_shape: self.image_shape(),
            description: self.description(),
        }
    }
    
    /// Get the expected image shape [C, H, W]
    fn image_shape(&self) -> Option<[usize; 3]> {
        None
    }
    
    /// Get dataset name
    fn name(&self) -> &'static str;
    
    /// Get dataset description
    fn description(&self) -> &'static str {
        ""
    }
}

/// Dataset metadata information
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: &'static str,
    pub num_samples: usize,
    pub num_classes: usize,
    pub image_shape: Option<[usize; 3]>,
    pub description: &'static str,
}

/// Lazy-loading dataset wrapper with caching
pub struct LazyDataset<D: Dataset> {
    inner: Arc<D>,
    cache: Arc<RwLock<HashMap<usize, (Tensor<f32>, usize)>>>,
    cache_size_limit: usize,
    access_order: Arc<Mutex<Vec<usize>>>,
    hit_count: Arc<Mutex<usize>>,
    miss_count: Arc<Mutex<usize>>,
}

impl<D: Dataset> LazyDataset<D> {
    /// Create a new lazy dataset with caching
    pub fn new(dataset: D, cache_size_limit: usize) -> Self {
        Self {
            inner: Arc::new(dataset),
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_size_limit,
            access_order: Arc::new(Mutex::new(Vec::new())),
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Get item with caching
    pub fn get_cached(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(item) = cache.get(&index) {
                // Cache hit
                *self.hit_count.lock().unwrap() += 1;
                self.update_access_order(index);
                return Ok(item.clone());
            }
        }
        
        // Cache miss
        *self.miss_count.lock().unwrap() += 1;
        
        // Load item and cache it
        let item = self.inner.get_item(index)?;
        
        {
            let mut cache = self.cache.write();
            
            // Evict if cache is full
            if cache.len() >= self.cache_size_limit {
                self.evict_lru(&mut cache);
            }
            
            cache.insert(index, item.clone());
        }
        
        self.update_access_order(index);
        Ok(item)
    }
    
    fn update_access_order(&self, index: usize) {
        let mut access_order = self.access_order.lock().unwrap();
        
        // Remove existing entry if present
        access_order.retain(|&x| x != index);
        
        // Add to end (most recently used)
        access_order.push(index);
    }
    
    fn evict_lru(&self, cache: &mut HashMap<usize, (Tensor<f32>, usize)>) {
        let mut access_order = self.access_order.lock().unwrap();
        
        if let Some(lru_index) = access_order.first().copied() {
            cache.remove(&lru_index);
            access_order.remove(0);
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.read();
        let hit_count = *self.hit_count.lock().unwrap();
        let miss_count = *self.miss_count.lock().unwrap();
        let total_requests = hit_count + miss_count;
        
        let hit_rate = if total_requests > 0 {
            hit_count as f32 / total_requests as f32
        } else {
            0.0
        };
        
        CacheStats {
            size: cache.len(),
            capacity: self.cache_size_limit,
            hit_rate,
        }
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        let mut access_order = self.access_order.lock().unwrap();
        let mut hit_count = self.hit_count.lock().unwrap();
        let mut miss_count = self.miss_count.lock().unwrap();
        
        cache.clear();
        access_order.clear();
        *hit_count = 0;
        *miss_count = 0;
    }
}

impl<D: Dataset> Dataset for LazyDataset<D> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        self.get_cached(index)
    }
    
    fn class_names(&self) -> Option<&[String]> {
        // Note: This requires unsafe lifetime extension for the Arc
        // In practice, you'd store class names separately or use Cow
        None // Simplified for this example
    }
    
    fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }
    
    fn image_shape(&self) -> Option<[usize; 3]> {
        self.inner.image_shape()
    }
    
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    
    fn description(&self) -> &'static str {
        self.inner.description()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f32,
}

/// Dataset with integrated transforms
pub struct TransformDataset<D: Dataset> {
    dataset: D,
    transform: Option<Box<dyn Transform>>,
    target_transform: Option<Box<dyn Fn(usize) -> usize + Send + Sync>>,
}

impl<D: Dataset> TransformDataset<D> {
    /// Create a new transform dataset
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            transform: None,
            target_transform: None,
        }
    }
    
    /// Add image transform
    pub fn with_transform(mut self, transform: Box<dyn Transform>) -> Self {
        self.transform = Some(transform);
        self
    }
    
    /// Add target (label) transform
    pub fn with_target_transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(usize) -> usize + Send + Sync + 'static,
    {
        self.target_transform = Some(Box::new(transform));
        self
    }
}

impl<D: Dataset> Dataset for TransformDataset<D> {
    fn len(&self) -> usize {
        self.dataset.len()
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        let (mut image, mut label) = self.dataset.get_item(index)?;
        
        // Apply image transform
        if let Some(ref transform) = self.transform {
            image = transform.forward(&image)?;
        }
        
        // Apply target transform
        if let Some(ref transform) = self.target_transform {
            label = transform(label);
        }
        
        Ok((image, label))
    }
    
    fn class_names(&self) -> Option<&[String]> {
        self.dataset.class_names()
    }
    
    fn num_classes(&self) -> usize {
        self.dataset.num_classes()
    }
    
    fn image_shape(&self) -> Option<[usize; 3]> {
        self.dataset.image_shape()
    }
    
    fn name(&self) -> &'static str {
        self.dataset.name()
    }
    
    fn description(&self) -> &'static str {
        self.dataset.description()
    }
}

/// Improved ImageFolder with lazy loading
pub struct ImageFolderLazy {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    classes: Vec<String>,
    class_to_idx: HashMap<String, usize>,
    io: VisionIO,
}

impl ImageFolderLazy {
    /// Create a new lazy ImageFolder dataset
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        
        if !root.exists() {
            return Err(VisionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Directory {:?} does not exist", root),
            )));
        }
        
        let mut classes = Vec::new();
        let mut class_to_idx = HashMap::new();
        let mut samples = Vec::new();
        let io = VisionIO::new();
        
        // Collect class directories
        for entry in std::fs::read_dir(&root)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                if let Some(class_name) = path.file_name() {
                    let class_str = class_name.to_string_lossy().to_string();
                    if !class_to_idx.contains_key(&class_str) {
                        let class_idx = classes.len();
                        classes.push(class_str.clone());
                        class_to_idx.insert(class_str, class_idx);
                        
                        // Collect image paths (but don't load images yet)
                        for img_entry in std::fs::read_dir(&path)? {
                            let img_entry = img_entry?;
                            let img_path = img_entry.path();
                            
                            if img_path.is_file() && io.is_supported_image(&img_path) {
                                samples.push((img_path, class_idx));
                            }
                        }
                    }
                }
            }
        }
        
        if classes.is_empty() {
            return Err(VisionError::TransformError(
                "No class directories found".to_string(),
            ));
        }
        
        Ok(Self {
            root,
            samples,
            classes,
            class_to_idx,
            io,
        })
    }
    
    /// Get class index for a class name
    pub fn class_to_index(&self, class_name: &str) -> Option<usize> {
        self.class_to_idx.get(class_name).copied()
    }
}

impl Dataset for ImageFolderLazy {
    fn len(&self) -> usize {
        self.samples.len()
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        if index >= self.samples.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Index {} out of bounds for dataset of size {}",
                index, self.samples.len()
            )));
        }
        
        let (ref path, label) = &self.samples[index];
        
        // Load image on demand
        let image = self.io.load_image(path)?;
        let tensor = crate::utils::image_to_tensor(&image)?;
        
        Ok((tensor, *label))
    }
    
    fn class_names(&self) -> Option<&[String]> {
        Some(&self.classes)
    }
    
    fn image_shape(&self) -> Option<[usize; 3]> {
        // Default to RGB images, actual shape depends on loaded images
        Some([3, 224, 224]) // Common default, actual images may vary
    }
    
    fn name(&self) -> &'static str {
        "ImageFolderLazy"
    }
    
    fn description(&self) -> &'static str {
        "Lazy-loading ImageFolder dataset with on-demand image loading"
    }
}

/// Memory-efficient CIFAR-10 dataset with lazy loading
pub struct CIFAR10Lazy {
    root: PathBuf,
    train: bool,
    batch_files: Vec<PathBuf>,
    classes: Vec<String>,
    samples_per_file: usize,
    total_samples: usize,
}

impl CIFAR10Lazy {
    /// Create a new lazy CIFAR-10 dataset
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        
        let classes = vec![
            "airplane".to_string(), "automobile".to_string(), "bird".to_string(),
            "cat".to_string(), "deer".to_string(), "dog".to_string(),
            "frog".to_string(), "horse".to_string(), "ship".to_string(),
            "truck".to_string(),
        ];
        
        let mut batch_files = Vec::new();
        let samples_per_file = 10000;
        
        if train {
            // Training files: data_batch_1.bin to data_batch_5.bin
            for i in 1..=5 {
                let batch_file = root.join(format!("data_batch_{}.bin", i));
                if !batch_file.exists() {
                    return Err(VisionError::IoError(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("CIFAR-10 training batch {} not found: {:?}", i, batch_file),
                    )));
                }
                batch_files.push(batch_file);
            }
        } else {
            // Test file: test_batch.bin
            let test_file = root.join("test_batch.bin");
            if !test_file.exists() {
                return Err(VisionError::IoError(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("CIFAR-10 test batch not found: {:?}", test_file),
                )));
            }
            batch_files.push(test_file);
        }
        
        let total_samples = batch_files.len() * samples_per_file;
        
        Ok(Self {
            root,
            train,
            batch_files,
            classes,
            samples_per_file,
            total_samples,
        })
    }
    
    /// Load a single sample from the appropriate batch file
    fn load_sample(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        let file_index = index / self.samples_per_file;
        let sample_index = index % self.samples_per_file;
        
        if file_index >= self.batch_files.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Index {} out of bounds", index
            )));
        }
        
        // Read the specific sample from the batch file
        let batch_file = &self.batch_files[file_index];
        let mut file = std::fs::File::open(batch_file)?;
        
        use std::io::{Seek, SeekFrom, Read};
        
        // Each sample is 1 byte label + 3072 bytes image data
        const BYTES_PER_SAMPLE: usize = 1 + 3072;
        
        // Seek to the sample position
        file.seek(SeekFrom::Start((sample_index * BYTES_PER_SAMPLE) as u64))?;
        
        // Read sample data
        let mut buffer = vec![0u8; BYTES_PER_SAMPLE];
        file.read_exact(&mut buffer)?;
        
        // Parse label
        let label = buffer[0] as usize;
        
        // Parse image data
        let mut tensor = creation::zeros(&[3, 32, 32]).unwrap();
        
        // CIFAR-10 format: R channel (1024 bytes), G channel (1024 bytes), B channel (1024 bytes)
        for channel in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    let pixel_idx = 1 + channel * 1024 + y * 32 + x;
                    let pixel_val = buffer[pixel_idx] as f32 / 255.0;
                    tensor.set(&[channel, y, x], pixel_val)?;
                }
            }
        }
        
        Ok((tensor, label))
    }
}

impl Dataset for CIFAR10Lazy {
    fn len(&self) -> usize {
        self.total_samples
    }
    
    fn get_item(&self, index: usize) -> Result<(Tensor<f32>, usize)> {
        self.load_sample(index)
    }
    
    fn class_names(&self) -> Option<&[String]> {
        Some(&self.classes)
    }
    
    fn image_shape(&self) -> Option<[usize; 3]> {
        Some([3, 32, 32])
    }
    
    fn name(&self) -> &'static str {
        "CIFAR10Lazy"
    }
    
    fn description(&self) -> &'static str {
        "Memory-efficient CIFAR-10 dataset with on-demand sample loading"
    }
}

/// Dataset sampling strategies
pub enum SamplingStrategy {
    /// Sequential sampling (0, 1, 2, ...)
    Sequential,
    /// Random sampling with replacement
    RandomWithReplacement,
    /// Random sampling without replacement (shuffle)
    RandomWithoutReplacement,
    /// Weighted sampling based on class frequencies
    WeightedByClass,
}

/// Dataset sampler for different sampling strategies
pub struct DatasetSampler<D: Dataset> {
    dataset: Arc<D>,
    strategy: SamplingStrategy,
    indices: Vec<usize>,
    current_epoch: usize,
    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
    rng: parking_lot::Mutex<scirs2_core::random::Random>,
}

impl<D: Dataset> DatasetSampler<D> {
    /// Create a new dataset sampler
    pub fn new(dataset: Arc<D>, strategy: SamplingStrategy) -> Self {
        let indices = (0..dataset.len()).collect();
        
        Self {
            dataset,
            strategy,
            indices,
            current_epoch: 0,
            rng: parking_lot::Mutex::new(scirs2_core::random::Random::seed(42)),
        }
    }
    
    /// Get the next batch of samples
    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<(Tensor<f32>, usize)>> {
        let mut batch = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            if let Some(index) = self.next_index() {
                let sample = self.dataset.get_item(index)?;
                batch.push(sample);
            } else {
                break;
            }
        }
        
        Ok(batch)
    }
    
    /// Get the next sample index according to the sampling strategy
    fn next_index(&mut self) -> Option<usize> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::Rng;
        
        match self.strategy {
            SamplingStrategy::Sequential => {
                let idx = self.current_epoch % self.indices.len();
                self.current_epoch += 1;
                Some(self.indices[idx])
            }
            SamplingStrategy::RandomWithReplacement => {
                let mut rng = self.rng.lock();
                let idx = rng.gen_range(0.. self.indices.len());
                Some(self.indices[idx])
            }
            SamplingStrategy::RandomWithoutReplacement => {
                if self.current_epoch % self.indices.len() == 0 {
                    // Shuffle indices at the beginning of each epoch
                    let mut rng = self.rng.lock();
                    for i in (1..self.indices.len()).rev() {
                        let j = rng.gen_range(0..=i);
                        self.indices.swap(i, j);
                    }
                }
                
                let idx = self.current_epoch % self.indices.len();
                self.current_epoch += 1;
                Some(self.indices[idx])
            }
            SamplingStrategy::WeightedByClass => {
                // Simplified implementation - could be made more sophisticated
                let mut rng = self.rng.lock();
                let idx = rng.gen_range(0.. self.indices.len());
                Some(self.indices[idx])
            }
        }
    }
    
    /// Reset the sampler for a new epoch
    pub fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Factory functions for creating common datasets
pub mod factory {
    use super::*;
    
    /// Create a lazy ImageFolder dataset with caching
    pub fn imagefolder_lazy<P: AsRef<Path>>(
        root: P,
        cache_size: usize,
    ) -> Result<LazyDataset<ImageFolderLazy>> {
        let dataset = ImageFolderLazy::new(root)?;
        Ok(LazyDataset::new(dataset, cache_size))
    }
    
    /// Create a lazy CIFAR-10 dataset with caching
    pub fn cifar10_lazy<P: AsRef<Path>>(
        root: P,
        train: bool,
        cache_size: usize,
    ) -> Result<LazyDataset<CIFAR10Lazy>> {
        let dataset = CIFAR10Lazy::new(root, train)?;
        Ok(LazyDataset::new(dataset, cache_size))
    }
    
    /// Create a transform dataset from any dataset
    pub fn with_transforms<D: Dataset>(
        dataset: D,
        transform: Option<Box<dyn Transform>>,
    ) -> TransformDataset<D> {
        let mut td = TransformDataset::new(dataset);
        if let Some(t) = transform {
            td = td.with_transform(t);
        }
        td
    }
}

/// Re-export common types
pub use factory::*;