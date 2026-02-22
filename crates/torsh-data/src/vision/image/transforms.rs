use crate::transforms::Transform;
#[cfg(feature = "image-support")]
use image::{DynamicImage, GenericImageView, ImageBuffer};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

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

            Tensor::from_data(
                data,
                vec![3, height as usize, width as usize],
                torsh_core::device::DeviceType::Cpu,
            )
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

            let data = input.to_vec()?;
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

/// Compose multiple transforms
pub struct Compose<T> {
    transforms: Vec<Box<dyn Transform<T, Output = T>>>,
}

impl<T: 'static> Compose<T> {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn add_transform<F>(mut self, transform: F) -> Self
    where
        F: Transform<T, Output = T> + 'static,
    {
        self.transforms.push(Box::new(transform));
        self
    }

    pub fn add_boxed(mut self, transform: Box<dyn Transform<T, Output = T>>) -> Self {
        self.transforms.push(transform);
        self
    }
}

impl<T> Transform<T> for Compose<T> {
    type Output = T;

    fn transform(&self, mut input: T) -> Result<Self::Output> {
        for transform in &self.transforms {
            input = transform.transform(input)?;
        }
        Ok(input)
    }
}

impl<T: 'static> Default for Compose<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Random horizontal flip
pub struct RandomHorizontalFlip {
    prob: f32,
}

impl RandomHorizontalFlip {
    pub fn new(prob: f32) -> Self {
        Self { prob }
    }
}

impl Transform<DynamicImage> for RandomHorizontalFlip {
    type Output = DynamicImage;

    fn transform(&self, input: DynamicImage) -> Result<Self::Output> {
        #[cfg(feature = "image-support")]
        {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
            #[allow(unused_imports)] // Rng trait needed for random() method
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(0);
            if rng.random::<f32>() < self.prob {
                Ok(input.fliph())
            } else {
                Ok(input)
            }
        }

        #[cfg(not(feature = "image-support"))]
        {
            Err(TorshError::UnsupportedOperation {
                op: "random horizontal flip".to_string(),
                dtype: "DynamicImage".to_string(),
            })
        }
    }
}

/// Random vertical flip
pub struct RandomVerticalFlip {
    prob: f32,
}

impl RandomVerticalFlip {
    pub fn new(prob: f32) -> Self {
        Self { prob }
    }
}

impl Transform<DynamicImage> for RandomVerticalFlip {
    type Output = DynamicImage;

    fn transform(&self, input: DynamicImage) -> Result<Self::Output> {
        #[cfg(feature = "image-support")]
        {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
            #[allow(unused_imports)] // Rng trait needed for random() method
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(0);
            if rng.random::<f32>() < self.prob {
                Ok(input.flipv())
            } else {
                Ok(input)
            }
        }

        #[cfg(not(feature = "image-support"))]
        {
            Err(TorshError::UnsupportedOperation {
                op: "random vertical flip".to_string(),
                dtype: "DynamicImage".to_string(),
            })
        }
    }
}

/// Random rotation
pub struct RandomRotation {
    degrees: f32,
}

impl RandomRotation {
    pub fn new(degrees: f32) -> Self {
        Self { degrees }
    }
}

impl Transform<DynamicImage> for RandomRotation {
    type Output = DynamicImage;

    fn transform(&self, input: DynamicImage) -> Result<Self::Output> {
        #[cfg(all(feature = "image-support", feature = "imageproc"))]
        {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
            #[allow(unused_imports)] // Rng trait needed for random() method
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(0);
            let angle_deg = rng.gen_range(-self.degrees..=self.degrees);
            let angle_rad = angle_deg.to_radians();

            // Convert to RGB8 for processing
            let rgb_image = input.to_rgb8();

            // Perform rotation using imageproc
            // Use imageproc's rotation function
            let rotated = imageproc::geometric_transformations::rotate_about_center(
                &rgb_image,
                angle_rad,
                imageproc::geometric_transformations::Interpolation::Bilinear,
                image::Rgb([0u8, 0u8, 0u8]), // Black background
            );

            Ok(DynamicImage::ImageRgb8(rotated))
        }

        #[cfg(all(feature = "image-support", not(feature = "imageproc")))]
        {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
            #[allow(unused_imports)] // Rng trait needed for random() method
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(0);
            let _angle = rng.gen_range(-self.degrees..=self.degrees);
            // imageproc not available, return input unchanged
            // In production, you might want to log a warning here
            Ok(input)
        }

        #[cfg(not(feature = "image-support"))]
        {
            Err(TorshError::UnsupportedOperation {
                op: "random rotation".to_string(),
                dtype: "DynamicImage".to_string(),
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
        mean: [f32; 3],
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
            // Apply ImageNet-style normalization per channel
            // Assumes input tensor is in CHW format (channels, height, width)
            let shape_ref = input.shape();
            let shape = shape_ref.dims();

            if shape.len() != 3 {
                return Err(TorshError::InvalidShape(format!(
                    "Expected 3D tensor (C, H, W), got shape {shape:?}"
                )));
            }

            let channels = shape[0];
            if channels != 3 {
                return Err(TorshError::InvalidShape(format!(
                    "Expected 3 channels for RGB image, got {channels}"
                )));
            }

            // Get tensor data and apply normalization
            let mut data = input.to_vec()?;
            let height = shape[1];
            let width = shape[2];
            let channel_size = height * width;

            // Apply per-channel normalization: (pixel - mean) / std
            for c in 0..3 {
                let channel_start = c * channel_size;
                let channel_end = channel_start + channel_size;
                let mean = self.mean[c];
                let std = self.std[c];

                for pixel in &mut data[channel_start..channel_end] {
                    *pixel = (*pixel - mean) / std;
                }
            }

            // Create normalized tensor with same shape and device
            Tensor::from_data(data, shape.to_vec(), input.device())
        }
    }
}
