use super::super::image::transforms::ImageToTensor;
use crate::{dataset::Dataset, transforms::Transform};
#[cfg(feature = "image-support")]
use image::DynamicImage;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use std::path::{Path, PathBuf};

/// ImageNet dataset
pub struct ImageNet {
    root: PathBuf,
    split: String,
    transform: Option<Box<dyn Transform<DynamicImage, Output = Tensor<f32>>>>,
    samples: Vec<(PathBuf, usize)>,
    classes: Vec<String>,
}

impl ImageNet {
    /// Create ImageNet dataset
    /// Expected directory structure:
    /// root/
    ///   train/
    ///     n01440764/
    ///       n01440764_10026.JPEG
    ///       ...
    ///     n01443537/
    ///       ...
    ///   val/
    ///     n01440764/
    ///       ILSVRC2012_val_00000293.JPEG
    ///       ...
    ///     ...
    pub fn new<P: AsRef<Path>>(root: P, split: &str) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let split_dir = root.join(split);

        if !split_dir.exists() {
            return Err(TorshError::IoError(format!(
                "ImageNet split directory does not exist: {split_dir:?}"
            )));
        }

        let mut classes = Vec::new();
        let mut samples = Vec::new();

        // Read class directories (WordNet IDs like n01440764)
        for entry in
            std::fs::read_dir(&split_dir).map_err(|e| TorshError::IoError(e.to_string()))?
        {
            let entry = entry.map_err(|e| TorshError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                let class_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| TorshError::IoError("Invalid class directory name".to_string()))?
                    .to_string();

                let class_idx = classes.len();
                classes.push(class_name);

                // Scan images in class directory
                for img_entry in
                    std::fs::read_dir(&path).map_err(|e| TorshError::IoError(e.to_string()))?
                {
                    let img_entry = img_entry.map_err(|e| TorshError::IoError(e.to_string()))?;
                    let img_path = img_entry.path();

                    if Self::is_image_file(&img_path) {
                        samples.push((img_path, class_idx));
                    }
                }
            }
        }

        // Sort samples for reproducibility
        samples.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(Self {
            root,
            split: split.to_string(),
            transform: None,
            samples,
            classes,
        })
    }

    /// Check if file is a supported image format
    fn is_image_file(path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(
                extension.to_uppercase().as_str(),
                "JPEG" | "JPG" | "PNG" | "BMP" | "GIF" | "TIFF" | "WEBP"
            )
        } else {
            false
        }
    }

    /// Load image from path
    #[cfg(feature = "image-support")]
    fn load_image(&self, path: &Path) -> Result<DynamicImage> {
        image::open(path).map_err(|e| {
            TorshError::IoError(format!("Failed to load ImageNet image {path:?}: {e}"))
        })
    }

    #[cfg(not(feature = "image-support"))]
    fn load_image(&self, _path: &Path) -> Result<DynamicImage> {
        Err(TorshError::UnsupportedOperation {
            op: "ImageNet image loading".to_string(),
            dtype: "DynamicImage".to_string(),
        })
    }

    /// Set transform
    pub fn with_transform<T>(mut self, transform: T) -> Self
    where
        T: Transform<DynamicImage, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Get class names
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Get number of classes (typically 1000 for ImageNet)
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get the root directory path
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the split name (e.g., "train", "val")
    pub fn split(&self) -> &str {
        &self.split
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }
}

impl Dataset for ImageNet {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.samples.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.samples.len(),
            });
        }

        let (ref path, class_idx) = self.samples[index];
        let image = self.load_image(path)?;

        let tensor = if let Some(ref transform) = self.transform {
            transform.transform(image)?
        } else {
            // Default: convert to tensor
            ImageToTensor.transform(image)?
        };

        Ok((tensor, class_idx))
    }
}
