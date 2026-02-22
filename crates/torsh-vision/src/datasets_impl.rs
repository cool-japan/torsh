//! Dataset loading and management for torsh-vision
//!
//! This module provides both legacy dataset implementations and optimized alternatives
//! with lazy loading, caching, and memory management features.

// Include the optimized implementations directly
pub use crate::optimized_impl::*;

// Legacy implementations (kept for backward compatibility)
use crate::utils::{image_to_tensor, load_images_from_dir};
use crate::{Result, VisionError};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use torsh_tensor::creation;
use torsh_tensor::Tensor;

/// Legacy ImageFolder dataset for loading images from a directory structure
/// where each subdirectory represents a class
///
/// **Note**: This implementation loads all images into memory at once.
/// For large datasets, consider using `OptimizedImageDataset` instead.
#[derive(Debug)]
pub struct ImageFolder {
    data: Vec<(Tensor<f32>, usize)>,
    class_to_idx: HashMap<String, usize>,
    classes: Vec<String>,
}

impl ImageFolder {
    /// Create a new ImageFolder dataset
    ///
    /// **Memory Warning**: This loads all images into memory immediately.
    /// For datasets larger than a few GB, use `OptimizedImageDataset`.
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        eprintln!("Warning: ImageFolder loads all data into memory. Consider using OptimizedImageDataset for large datasets.");

        let root_path = root.as_ref();

        if !root_path.exists() {
            return Err(VisionError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Directory {:?} does not exist", root_path),
            )));
        }

        let mut classes = Vec::new();
        let mut class_to_idx = HashMap::new();
        let mut data = Vec::new();

        // Collect all subdirectories as classes
        for entry in std::fs::read_dir(root_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(class_name) = path.file_name() {
                    let class_str = class_name.to_string_lossy().to_string();
                    if !class_to_idx.contains_key(&class_str) {
                        let class_idx = classes.len();
                        classes.push(class_str.clone());
                        class_to_idx.insert(class_str.clone(), class_idx);

                        // Load images from this class directory
                        let images = load_images_from_dir(&path)?;
                        for (image, _filename) in images {
                            let tensor = image_to_tensor(&image)?;
                            data.push((tensor, class_idx));
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
            data,
            class_to_idx,
            classes,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, usize)> {
        self.data.get(index).cloned()
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn class_to_idx(&self) -> &HashMap<String, usize> {
        &self.class_to_idx
    }
}

/// Legacy ImageNet dataset placeholder
#[derive(Debug)]
pub struct ImageNet {
    data: Vec<Tensor<f32>>,
    labels: Vec<usize>,
}

impl ImageNet {
    pub fn new(_root: &str, _train: bool) -> Result<Self> {
        eprintln!("Warning: ImageNet placeholder implementation. Use OptimizedImageDataset for real datasets.");
        Ok(Self {
            data: vec![creation::zeros(&[3, 224, 224]).expect("tensor creation should succeed")],
            labels: vec![0],
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, usize)> {
        if index < self.data.len() {
            Some((self.data[index].clone(), self.labels[index]))
        } else {
            None
        }
    }
}

/// Legacy CIFAR-10 dataset loader
///
/// **Note**: This implementation loads the entire dataset into memory at once.
/// For memory-efficient loading, use `OptimizedCIFARDataset` instead.
#[derive(Debug)]
pub struct CIFAR10 {
    data: Vec<Tensor<f32>>,
    labels: Vec<usize>,
    classes: Vec<String>,
}

impl CIFAR10 {
    /// Create a new CIFAR-10 dataset
    ///
    /// **Memory Warning**: This loads all data into memory immediately.
    pub fn new<P: AsRef<Path>>(root: P, train: bool, download: bool) -> Result<Self> {
        eprintln!("Warning: CIFAR10 loads all data into memory. Consider using OptimizedCIFARDataset for memory efficiency.");

        let root_path = root.as_ref();

        // Create directory if it doesn't exist
        if !root_path.exists() {
            std::fs::create_dir_all(root_path)?;
        }

        let classes = vec![
            "airplane".to_string(),
            "automobile".to_string(),
            "bird".to_string(),
            "cat".to_string(),
            "deer".to_string(),
            "dog".to_string(),
            "frog".to_string(),
            "horse".to_string(),
            "ship".to_string(),
            "truck".to_string(),
        ];

        let (all_data, all_labels) = if train {
            // Load training batches
            let mut data = Vec::new();
            let mut labels = Vec::new();

            for i in 1..=5 {
                let batch_file = root_path.join(format!("data_batch_{}.bin", i));
                if !batch_file.exists() {
                    if download {
                        return Err(VisionError::TransformError(
                            format!("CIFAR-10 files not found in {:?}. Please download them manually from https://www.cs.toronto.edu/~kriz/cifar.html", root_path)
                        ));
                    } else {
                        return Err(VisionError::IoError(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("CIFAR-10 training batch {} not found in {:?}", i, root_path),
                        )));
                    }
                }

                let (batch_data, batch_labels) = Self::load_batch(&batch_file)?;
                data.extend(batch_data);
                labels.extend(batch_labels);
            }

            (data, labels)
        } else {
            // Load test batch
            let test_file = root_path.join("test_batch.bin");
            if !test_file.exists() {
                if download {
                    return Err(VisionError::TransformError(
                        format!("CIFAR-10 files not found in {:?}. Please download them manually from https://www.cs.toronto.edu/~kriz/cifar.html", root_path)
                    ));
                } else {
                    return Err(VisionError::IoError(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("CIFAR-10 test batch not found in {:?}", root_path),
                    )));
                }
            }

            Self::load_batch(&test_file)?
        };

        Ok(Self {
            data: all_data,
            labels: all_labels,
            classes,
        })
    }

    fn load_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<Tensor<f32>>, Vec<usize>)> {
        let data = std::fs::read(path)?;

        // Each CIFAR-10 batch contains 10,000 samples
        // Each sample is 1 byte label + 3072 bytes image data (32x32x3)
        const SAMPLES_PER_BATCH: usize = 10000;
        const BYTES_PER_SAMPLE: usize = 1 + 3072; // 1 label + 32*32*3 pixels

        if data.len() != SAMPLES_PER_BATCH * BYTES_PER_SAMPLE {
            return Err(VisionError::TransformError(format!(
                "Invalid CIFAR-10 batch file size. Expected {}, got {}",
                SAMPLES_PER_BATCH * BYTES_PER_SAMPLE,
                data.len()
            )));
        }

        let mut images = Vec::with_capacity(SAMPLES_PER_BATCH);
        let mut labels = Vec::with_capacity(SAMPLES_PER_BATCH);

        for i in 0..SAMPLES_PER_BATCH {
            let start_idx = i * BYTES_PER_SAMPLE;

            // First byte is the label
            let label = data[start_idx] as usize;
            labels.push(label);

            // Next 3072 bytes are the image data (R, G, B channels in that order)
            let tensor = creation::zeros(&[3, 32, 32]).expect("tensor creation should succeed");

            // CIFAR-10 format: first 1024 bytes are red channel, next 1024 green, last 1024 blue
            for channel in 0..3 {
                for y in 0..32 {
                    for x in 0..32 {
                        let pixel_idx = start_idx + 1 + channel * 1024 + y * 32 + x;
                        let pixel_val = data[pixel_idx] as f32 / 255.0; // Normalize to [0, 1]
                        tensor.set(&[channel, y, x], pixel_val)?;
                    }
                }
            }

            images.push(tensor);
        }

        Ok((images, labels))
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, usize)> {
        if index < self.data.len() {
            Some((self.data[index].clone(), self.labels[index]))
        } else {
            None
        }
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }
}

/// Legacy MNIST dataset loader
///
/// **Note**: This implementation loads the entire dataset into memory at once.
/// For memory-efficient loading, consider using an optimized alternative.
#[derive(Debug)]
pub struct MNIST {
    data: Vec<Tensor<f32>>,
    labels: Vec<usize>,
}

impl MNIST {
    /// Create a new MNIST dataset
    ///
    /// **Memory Warning**: This loads all data into memory immediately.
    pub fn new<P: AsRef<Path>>(root: P, train: bool, download: bool) -> Result<Self> {
        eprintln!("Warning: MNIST loads all data into memory. Consider optimized alternatives for memory efficiency.");

        let root_path = root.as_ref();

        // Create directory if it doesn't exist
        if !root_path.exists() {
            std::fs::create_dir_all(root_path)?;
        }

        let (images_filename, labels_filename) = if train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        };

        let images_path = root_path.join(images_filename);
        let labels_path = root_path.join(labels_filename);

        // Check if files exist, if not and download is true, suggest downloading manually
        if !images_path.exists() || !labels_path.exists() {
            if download {
                return Err(VisionError::TransformError(
                    format!("MNIST files not found in {:?}. Please download them manually from http://yann.lecun.com/exdb/mnist/", root_path)
                ));
            } else {
                return Err(VisionError::IoError(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("MNIST files not found in {:?}", root_path),
                )));
            }
        }

        // Load images and labels
        let images = Self::load_images(&images_path)?;
        let labels = Self::load_labels(&labels_path)?;

        if images.len() != labels.len() {
            return Err(VisionError::TransformError(
                "Number of images and labels don't match".to_string(),
            ));
        }

        Ok(Self {
            data: images,
            labels,
        })
    }

    fn load_images<P: AsRef<Path>>(path: P) -> Result<Vec<Tensor<f32>>> {
        let data = std::fs::read(path)?;

        if data.len() < 16 {
            return Err(VisionError::TransformError(
                "Invalid MNIST images file format".to_string(),
            ));
        }

        // Read header
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let num_images = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;

        if magic != 0x00000803 {
            return Err(VisionError::TransformError(
                "Invalid MNIST images file magic number".to_string(),
            ));
        }

        let mut images = Vec::with_capacity(num_images);
        let image_size = rows * cols;

        for i in 0..num_images {
            let start_idx = 16 + i * image_size;
            let end_idx = start_idx + image_size;

            if end_idx > data.len() {
                break;
            }

            let tensor = creation::zeros(&[1, rows, cols]).expect("tensor creation should succeed");

            for (pixel_idx, &pixel_val) in data[start_idx..end_idx].iter().enumerate() {
                let y = pixel_idx / cols;
                let x = pixel_idx % cols;
                let normalized_val = pixel_val as f32 / 255.0;
                tensor.set(&[0, y, x], normalized_val)?;
            }

            images.push(tensor);
        }

        Ok(images)
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> Result<Vec<usize>> {
        let data = std::fs::read(path)?;

        if data.len() < 8 {
            return Err(VisionError::TransformError(
                "Invalid MNIST labels file format".to_string(),
            ));
        }

        // Read header
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let num_labels = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;

        if magic != 0x00000801 {
            return Err(VisionError::TransformError(
                "Invalid MNIST labels file magic number".to_string(),
            ));
        }

        if data.len() < 8 + num_labels {
            return Err(VisionError::TransformError(
                "MNIST labels file too short".to_string(),
            ));
        }

        let labels = data[8..8 + num_labels]
            .iter()
            .map(|&label| label as usize)
            .collect();

        Ok(labels)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, usize)> {
        if index < self.data.len() {
            Some((self.data[index].clone(), self.labels[index]))
        } else {
            None
        }
    }
}

/// Helper function to create optimized datasets with sensible defaults
pub fn create_optimized_image_dataset<P: AsRef<Path>>(root: P) -> Result<OptimizedImageDataset> {
    OptimizedDatasetBuilder::new()
        .with_cache(1000, 512) // 1000 items, 512MB cache
        .with_prefetch(true, 16) // Enable prefetching with batch size 16
        .build_image_dataset(root)
}

/// Helper function to create optimized CIFAR datasets
pub fn create_optimized_cifar_dataset<P: AsRef<Path>>(
    root: P,
    is_cifar100: bool,
    train: bool,
) -> Result<OptimizedCIFARDataset> {
    OptimizedDatasetBuilder::new()
        .with_cache(2000, 256) // 2000 items, 256MB cache
        .with_prefetch(true, 32) // Prefetch in larger batches for CIFAR
        .build_cifar_dataset(root, is_cifar100, train)
}

// Type aliases for backward compatibility
pub type CifarDataset = CIFAR10;
pub type MnistDataset = MNIST;

// Placeholder implementations for datasets that aren't fully implemented yet
#[derive(Debug)]
pub struct CocoDataset {
    data: Vec<Tensor<f32>>,
    labels: Vec<usize>,
}

impl CocoDataset {
    pub fn new<P: AsRef<Path>>(_root: P, _train: bool) -> Result<Self> {
        eprintln!("Warning: CocoDataset is a placeholder implementation");
        Ok(Self {
            data: vec![torsh_tensor::creation::zeros(&[3, 224, 224])
                .expect("tensor creation should succeed")],
            labels: vec![0],
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, usize)> {
        if index < self.data.len() {
            Some((self.data[index].clone(), self.labels[index]))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct VocDataset {
    data: Vec<Tensor<f32>>,
    labels: Vec<usize>,
}

impl VocDataset {
    pub fn new<P: AsRef<Path>>(_root: P, _train: bool) -> Result<Self> {
        eprintln!("Warning: VocDataset is a placeholder implementation");
        Ok(Self {
            data: vec![torsh_tensor::creation::zeros(&[3, 224, 224])
                .expect("tensor creation should succeed")],
            labels: vec![0],
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<(Tensor<f32>, usize)> {
        if index < self.data.len() {
            Some((self.data[index].clone(), self.labels[index]))
        } else {
            None
        }
    }
}
