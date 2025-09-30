//! Dataset loading and management for torsh-vision
//!
//! This module has been reorganized for better memory efficiency and performance.
//! The datasets are now split into:
//! - Legacy implementations (maintained for backward compatibility)
//! - Optimized implementations with lazy loading and caching
//!
//! For new projects, prefer the optimized versions:
//! - `OptimizedImageDataset` instead of `ImageFolder`
//! - `OptimizedCIFARDataset` instead of `CIFAR10`/`CIFAR100`

// Include datasets_impl content inline to resolve module loading issues

// Re-export from optimized implementations
pub use crate::optimized_impl::{
    CacheStatistics, DatasetConfig, DatasetMetadata, OptimizedCIFARDataset, OptimizedDataset,
    OptimizedImageDataset,
};

// Dataset error type alias
pub type DatasetError = crate::VisionError;

// Dataset statistics type alias
pub type DatasetStats = CacheStatistics;

// Basic dataset implementations for backward compatibility
use crate::utils::{image_to_tensor, load_images_from_dir};
use crate::{Result, VisionError};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use torsh_tensor::Tensor;

/// Legacy ImageFolder dataset
#[derive(Debug)]
pub struct ImageFolder {
    pub class_to_idx: HashMap<String, usize>,
    pub samples: Vec<(PathBuf, usize)>,
}

impl ImageFolder {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let mut class_to_idx = HashMap::new();
        let mut samples = Vec::new();

        // Simple implementation for compatibility
        let root_path = root.as_ref();
        if let Ok(entries) = std::fs::read_dir(root_path) {
            for (class_idx, entry) in entries.enumerate() {
                if let Ok(entry) = entry {
                    if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                        let class_name = entry.file_name().to_string_lossy().to_string();
                        class_to_idx.insert(class_name, class_idx);

                        // Add image files from this class directory
                        if let Ok(class_entries) = std::fs::read_dir(entry.path()) {
                            for class_entry in class_entries.flatten() {
                                if let Some(ext) = class_entry.path().extension() {
                                    if ["jpg", "jpeg", "png", "bmp"]
                                        .contains(&ext.to_string_lossy().to_lowercase().as_str())
                                    {
                                        samples.push((class_entry.path(), class_idx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            class_to_idx,
            samples,
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// Module aliases for better organization
pub mod optimized {
    pub use super::{
        DatasetConfig, DatasetMetadata, OptimizedCIFARDataset, OptimizedDataset,
        OptimizedImageDataset,
    };
}

pub mod legacy {
    pub use super::ImageFolder;
}
