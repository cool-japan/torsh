//! Vision-specific datasets and transformations

use crate::{dataset::Dataset, transforms::Transform};
#[cfg(feature = "image-support")]
use image::{DynamicImage, GenericImageView, ImageBuffer};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use std::path::{Path, PathBuf};

/// Image dataset for loading images from a directory
pub struct ImageFolder {
    #[allow(dead_code)]
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
                "Directory does not exist: {:?}",
                root
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
            .map_err(|e| TorshError::IoError(format!("Failed to load image {:?}: {}", path, e)))
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

/// Transform to convert image to tensor
pub struct ImageToTensor;

impl Transform<DynamicImage> for ImageToTensor {
    type Output = Tensor<f32>;

    fn transform(&self, input: DynamicImage) -> Result<Self::Output> {
        #[cfg(feature = "image-support")]
        {
            let rgb_image = input.to_rgb8();
            let (width, height) = rgb_image.dimensions();

            // Convert to CHW format (channels, height, width)
            let mut data = Vec::with_capacity((width * height * 3) as usize);

            // Extract channels separately
            for channel in 0..3 {
                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_image.get_pixel(x, y);
                        let value = pixel[channel] as f32 / 255.0;
                        data.push(value);
                    }
                }
            }

            Ok(Tensor::from_data(
                data,
                vec![3, height as usize, width as usize],
                torsh_core::device::DeviceType::Cpu,
            ))
        }

        #[cfg(not(feature = "image-support"))]
        {
            Err(TorshError::UnsupportedOperation {
                op: "image to tensor conversion".to_string(),
                dtype: "DynamicImage".to_string(),
            })
        }
    }
}

/// Transform to convert tensor to image
pub struct TensorToImage;

impl Transform<Tensor<f32>> for TensorToImage {
    type Output = DynamicImage;

    fn transform(&self, input: Tensor<f32>) -> Result<Self::Output> {
        #[cfg(feature = "image-support")]
        {
            let shape = input.shape();
            if shape.ndim() != 3 {
                return Err(TorshError::InvalidShape(
                    "Expected 3D tensor (C, H, W)".to_string(),
                ));
            }

            let dims = shape.dims();
            let (channels, height, width) = (dims[0], dims[1], dims[2]);

            if channels != 3 {
                return Err(TorshError::InvalidShape(
                    "Expected 3 channels for RGB image".to_string(),
                ));
            }

            let data = input.to_vec();
            let mut img_data = Vec::with_capacity(width * height * 3);

            // Convert from CHW to HWC format
            for y in 0..height {
                for x in 0..width {
                    for c in 0..3 {
                        let idx = c * height * width + y * width + x;
                        let value = (data[idx] * 255.0).clamp(0.0, 255.0) as u8;
                        img_data.push(value);
                    }
                }
            }

            let img_buffer = ImageBuffer::from_raw(width as u32, height as u32, img_data)
                .ok_or_else(|| {
                    TorshError::InvalidArgument("Failed to create image buffer".to_string())
                })?;

            Ok(DynamicImage::ImageRgb8(img_buffer))
        }

        #[cfg(not(feature = "image-support"))]
        {
            Err(TorshError::UnsupportedOperation {
                op: "tensor to image conversion".to_string(),
                dtype: "Tensor<f32>".to_string(),
            })
        }
    }
}

/// Common vision transforms
pub mod transforms {
    use super::*;
    use crate::transforms::Transform;

    /// Resize image
    pub struct Resize {
        size: (u32, u32),
    }

    impl Resize {
        pub fn new(size: (u32, u32)) -> Self {
            Self { size }
        }
    }

    impl Transform<DynamicImage> for Resize {
        type Output = DynamicImage;

        fn transform(&self, input: DynamicImage) -> Result<Self::Output> {
            #[cfg(feature = "image-support")]
            {
                Ok(input.resize_exact(
                    self.size.0,
                    self.size.1,
                    image::imageops::FilterType::Lanczos3,
                ))
            }

            #[cfg(not(feature = "image-support"))]
            {
                Err(TorshError::UnsupportedOperation {
                    op: "image resize".to_string(),
                    dtype: "DynamicImage".to_string(),
                })
            }
        }
    }

    /// Center crop image
    pub struct CenterCrop {
        size: (u32, u32),
    }

    impl CenterCrop {
        pub fn new(size: (u32, u32)) -> Self {
            Self { size }
        }
    }

    impl Transform<DynamicImage> for CenterCrop {
        type Output = DynamicImage;

        fn transform(&self, input: DynamicImage) -> Result<Self::Output> {
            #[cfg(feature = "image-support")]
            {
                let (width, height) = input.dimensions();
                let (crop_width, crop_height) = self.size;

                if crop_width > width || crop_height > height {
                    return Err(TorshError::InvalidArgument(
                        "Crop size cannot be larger than image size".to_string(),
                    ));
                }

                let x = (width - crop_width) / 2;
                let y = (height - crop_height) / 2;

                Ok(input.crop_imm(x, y, crop_width, crop_height))
            }

            #[cfg(not(feature = "image-support"))]
            {
                Err(TorshError::UnsupportedOperation {
                    op: "image center crop".to_string(),
                    dtype: "DynamicImage".to_string(),
                })
            }
        }
    }

    /// Normalize image values
    pub struct Normalize {
        #[allow(dead_code)]
        mean: [f32; 3],
        #[allow(dead_code)]
        std: [f32; 3],
    }

    impl Normalize {
        pub fn new(mean: [f32; 3], std: [f32; 3]) -> Self {
            Self { mean, std }
        }

        /// ImageNet normalization
        pub fn imagenet() -> Self {
            Self::new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }
    }

    impl Transform<Tensor<f32>> for Normalize {
        type Output = Tensor<f32>;

        fn transform(&self, input: Tensor<f32>) -> Result<Self::Output> {
            // TODO: Implement actual normalization when tensor operations are complete
            Ok(input)
        }
    }
}

/// CIFAR-10 dataset
pub struct CIFAR10 {
    #[allow(dead_code)]
    root: PathBuf,
    #[allow(dead_code)]
    train: bool,
    transform: Option<Box<dyn Transform<Tensor<f32>, Output = Tensor<f32>>>>,
    data: Vec<Tensor<f32>>,
    targets: Vec<usize>,
}

impl CIFAR10 {
    /// Create CIFAR-10 dataset
    pub fn new<P: AsRef<Path>>(root: P, train: bool) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        // TODO: Implement actual CIFAR-10 data loading
        // For now, create dummy data
        let data = vec![torsh_tensor::creation::rand::<f32>(&[3, 32, 32]); 100];
        let targets = (0..100).map(|i| i % 10).collect();

        Ok(Self {
            root,
            train,
            transform: None,
            data,
            targets,
        })
    }

    /// Set transform
    pub fn with_transform<T>(mut self, transform: T) -> Self
    where
        T: Transform<Tensor<f32>, Output = Tensor<f32>> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

impl Dataset for CIFAR10 {
    type Item = (Tensor<f32>, usize);

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.data.len() {
            return Err(TorshError::IndexError {
                index,
                size: self.data.len(),
            });
        }

        let mut data = self.data[index].clone();
        if let Some(ref transform) = self.transform {
            data = transform.transform(data)?;
        }

        Ok((data, self.targets[index]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_to_tensor() {
        // This test requires the image feature
        #[cfg(feature = "image-support")]
        {
            let transform = ImageToTensor;
            // Create a small test image
            let img = DynamicImage::new_rgb8(2, 2);
            let result = transform.transform(img);
            assert!(result.is_ok());

            let tensor = result.unwrap();
            assert_eq!(tensor.shape().dims(), &[3, 2, 2]);
        }
    }

    #[test]
    fn test_cifar10() {
        let dataset = CIFAR10::new("/tmp", true).unwrap();
        assert_eq!(dataset.len(), 100);

        let (data, label) = dataset.get(0).unwrap();
        assert_eq!(data.shape().dims(), &[3, 32, 32]);
        assert!(label < 10);
    }
}
