use super::transforms::ImageToTensor;
use crate::{dataset::Dataset, transforms::Transform};
#[cfg(feature = "image-support")]
use image::DynamicImage;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use std::path::{Path, PathBuf};

/// Image dataset for loading images from a directory
pub struct ImageFolder {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    classes: Vec<String>,
    transform: Option<Box<dyn Transform<DynamicImage, Output = Tensor<f32>>>>,
}

impl ImageFolder {
    /// Create a new image folder dataset
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        if !root.exists() {
            return Err(TorshError::IoError(format!(
                "Directory does not exist: {root:?}"
            )));
        }

        let mut classes = Vec::new();
        let mut samples = Vec::new();

        // Scan subdirectories for classes
        for entry in std::fs::read_dir(&root).map_err(|e| TorshError::IoError(e.to_string()))? {
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

        Ok(Self {
            root,
            samples,
            classes,
            transform: None,
        })
    }

    /// Set transform to apply to images
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

    /// Get the root directory path
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Check if file is a supported image format
    fn is_image_file(path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            matches!(
                extension.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "gif" | "tiff" | "webp"
            )
        } else {
            false
        }
    }

    /// Load image from path
    #[cfg(feature = "image-support")]
    fn load_image(&self, path: &Path) -> Result<DynamicImage> {
        image::open(path)
            .map_err(|e| TorshError::IoError(format!("Failed to load image {path:?}: {e}")))
    }

    #[cfg(not(feature = "image-support"))]
    fn load_image(&self, _path: &Path) -> Result<DynamicImage> {
        Err(TorshError::UnsupportedOperation {
            op: "image loading".to_string(),
            dtype: "DynamicImage".to_string(),
        })
    }
}

impl Dataset for ImageFolder {
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
