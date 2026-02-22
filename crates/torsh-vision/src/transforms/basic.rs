use super::core::Transform;
use crate::{Result, VisionError};
use torsh_tensor::Tensor;

/// Resize transform
///
/// Resizes input images to a specified size. This is one of the most commonly used
/// transforms for standardizing input dimensions.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Resize, Transform};
///
/// let resize = Resize::new((224, 224));
/// // Apply to tensor: result = resize.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct Resize {
    size: (usize, usize),
}

impl Resize {
    /// Create a new Resize transform
    ///
    /// # Arguments
    ///
    /// * `size` - Target size as (width, height)
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    /// Get the target size
    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}

impl Transform for Resize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::resize(input, self.size)
    }

    fn name(&self) -> &'static str {
        "Resize"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("size", format!("({}, {})", self.size.0, self.size.1))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Resize::new(self.size))
    }
}

/// Center crop transform
///
/// Crops the input image at the center to the specified size. Useful for creating
/// uniform image sizes while preserving the central content.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{CenterCrop, Transform};
///
/// let crop = CenterCrop::new((224, 224));
/// // Apply to tensor: result = crop.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct CenterCrop {
    size: (usize, usize),
}

impl CenterCrop {
    /// Create a new CenterCrop transform
    ///
    /// # Arguments
    ///
    /// * `size` - Target crop size as (width, height)
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }

    /// Get the crop size
    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}

impl Transform for CenterCrop {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::center_crop(input, self.size)
    }

    fn name(&self) -> &'static str {
        "CenterCrop"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("size", format!("({}, {})", self.size.0, self.size.1))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(CenterCrop::new(self.size))
    }
}

/// Convert PIL image to tensor
///
/// This transform typically converts PIL images or other formats to tensors.
/// For the current implementation, it acts as an identity transform.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{ToTensor, Transform};
///
/// let to_tensor = ToTensor::new();
/// // Apply to tensor: result = to_tensor.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct ToTensor;

impl ToTensor {
    /// Create a new ToTensor transform
    pub fn new() -> Self {
        Self
    }
}

impl Transform for ToTensor {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // This would typically convert from PIL image to tensor
        // For now, just return the input as is
        Ok(input.clone())
    }

    fn name(&self) -> &'static str {
        "ToTensor"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(ToTensor::new())
    }
}

/// Normalize transform
///
/// Normalizes tensor images with mean and standard deviation. This is typically
/// applied as the final preprocessing step before feeding data to neural networks.
///
/// The normalization formula is: `(input - mean) / std`
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Normalize, Transform};
///
/// // ImageNet normalization
/// let normalize = Normalize::new(
///     vec![0.485, 0.456, 0.406],  // RGB means
///     vec![0.229, 0.224, 0.225]   // RGB standard deviations
/// );
/// // Apply to tensor: result = normalize.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// Create a new Normalize transform
    ///
    /// # Arguments
    ///
    /// * `mean` - Per-channel means for normalization
    /// * `std` - Per-channel standard deviations for normalization
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `std` have different lengths
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(
            mean.len(),
            std.len(),
            "Mean and std must have the same length"
        );
        Self { mean, std }
    }

    /// Create ImageNet normalization (RGB)
    pub fn imagenet() -> Self {
        Self::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }

    /// Create CIFAR normalization (RGB)
    pub fn cifar() -> Self {
        Self::new(vec![0.4914, 0.4822, 0.4465], vec![0.2023, 0.1994, 0.2010])
    }

    /// Get the normalization mean
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Get the normalization standard deviation
    pub fn std(&self) -> &[f32] {
        &self.std
    }
}

impl Transform for Normalize {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::normalize(
            input,
            crate::ops::color::NormalizationConfig {
                method: crate::ops::color::NormalizationMethod::Custom,
                mean: Some(self.mean.clone()),
                std: Some(self.std.clone()),
                per_channel: true,
                eps: 1e-8,
            },
        )
    }

    fn name(&self) -> &'static str {
        "Normalize"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("mean", format!("{:?}", self.mean)),
            ("std", format!("{:?}", self.std)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Normalize::new(self.mean.clone(), self.std.clone()))
    }
}

/// Padding transform
///
/// Pads the input tensor with a specified value. Useful for increasing image
/// size before random cropping or for maintaining spatial dimensions.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Pad, Transform};
///
/// // Pad with 4 pixels on all sides, filled with black (0.0)
/// let pad = Pad::symmetric(4, 0.0);
///
/// // Asymmetric padding: (left, top, right, bottom)
/// let pad_custom = Pad::new((2, 4, 2, 4), 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct Pad {
    padding: (usize, usize, usize, usize), // (left, top, right, bottom)
    fill: f32,
}

impl Pad {
    /// Create a new Pad transform with asymmetric padding
    ///
    /// # Arguments
    ///
    /// * `padding` - Padding amounts as (left, top, right, bottom)
    /// * `fill` - Fill value for padded regions
    pub fn new(padding: (usize, usize, usize, usize), fill: f32) -> Self {
        Self { padding, fill }
    }

    /// Create symmetric padding (same amount on all sides)
    ///
    /// # Arguments
    ///
    /// * `pad` - Padding amount for all sides
    /// * `fill` - Fill value for padded regions
    pub fn symmetric(pad: usize, fill: f32) -> Self {
        Self {
            padding: (pad, pad, pad, pad),
            fill,
        }
    }

    /// Create padding for specific sides
    ///
    /// # Arguments
    ///
    /// * `horizontal` - Padding for left and right sides
    /// * `vertical` - Padding for top and bottom sides
    /// * `fill` - Fill value for padded regions
    pub fn sides(horizontal: usize, vertical: usize, fill: f32) -> Self {
        Self {
            padding: (horizontal, vertical, horizontal, vertical),
            fill,
        }
    }

    /// Get the padding configuration
    pub fn padding(&self) -> (usize, usize, usize, usize) {
        self.padding
    }

    /// Get the fill value
    pub fn fill(&self) -> f32 {
        self.fill
    }
}

impl Transform for Pad {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::pad(
            input,
            self.padding,
            crate::ops::PaddingMode::Zero,
            self.fill,
        )
    }

    fn name(&self) -> &'static str {
        "Pad"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "padding",
                format!(
                    "({}, {}, {}, {})",
                    self.padding.0, self.padding.1, self.padding.2, self.padding.3
                ),
            ),
            ("fill", format!("{:.2}", self.fill)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Pad::new(self.padding, self.fill))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_resize_creation() {
        let resize = Resize::new((224, 224));
        assert_eq!(resize.size(), (224, 224));
        assert_eq!(resize.name(), "Resize");

        let params = resize.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "size");
        assert_eq!(params[0].1, "(224, 224)");
    }

    #[test]
    fn test_center_crop_creation() {
        let crop = CenterCrop::new((128, 128));
        assert_eq!(crop.size(), (128, 128));
        assert_eq!(crop.name(), "CenterCrop");

        let params = crop.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "size");
        assert_eq!(params[0].1, "(128, 128)");
    }

    #[test]
    fn test_to_tensor_creation() {
        let to_tensor = ToTensor::new();
        assert_eq!(to_tensor.name(), "ToTensor");

        let params = to_tensor.parameters();
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_to_tensor_default() {
        let to_tensor = ToTensor::default();
        assert_eq!(to_tensor.name(), "ToTensor");
    }

    #[test]
    fn test_to_tensor_forward() {
        let to_tensor = ToTensor::new();
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let result = to_tensor.forward(&input).unwrap();
        assert_eq!(result.get(&[0, 0, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_normalize_creation() {
        let normalize = Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]);
        assert_eq!(normalize.mean(), &[0.5, 0.5, 0.5]);
        assert_eq!(normalize.std(), &[0.5, 0.5, 0.5]);
        assert_eq!(normalize.name(), "Normalize");
    }

    #[test]
    fn test_normalize_imagenet() {
        let normalize = Normalize::imagenet();
        assert_eq!(normalize.mean(), &[0.485, 0.456, 0.406]);
        assert_eq!(normalize.std(), &[0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_normalize_cifar() {
        let normalize = Normalize::cifar();
        assert_eq!(normalize.mean(), &[0.4914, 0.4822, 0.4465]);
        assert_eq!(normalize.std(), &[0.2023, 0.1994, 0.2010]);
    }

    #[test]
    #[should_panic(expected = "Mean and std must have the same length")]
    fn test_normalize_mismatched_lengths() {
        Normalize::new(vec![0.5, 0.5], vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_normalize_parameters() {
        let normalize = Normalize::new(vec![0.1, 0.2], vec![0.3, 0.4]);
        let params = normalize.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "mean");
        assert_eq!(params[1].0, "std");
    }

    #[test]
    fn test_pad_new() {
        let pad = Pad::new((1, 2, 3, 4), 0.5);
        assert_eq!(pad.padding(), (1, 2, 3, 4));
        assert_eq!(pad.fill(), 0.5);
        assert_eq!(pad.name(), "Pad");
    }

    #[test]
    fn test_pad_symmetric() {
        let pad = Pad::symmetric(5, 1.0);
        assert_eq!(pad.padding(), (5, 5, 5, 5));
        assert_eq!(pad.fill(), 1.0);
    }

    #[test]
    fn test_pad_sides() {
        let pad = Pad::sides(3, 7, 0.25);
        assert_eq!(pad.padding(), (3, 7, 3, 7));
        assert_eq!(pad.fill(), 0.25);
    }

    #[test]
    fn test_pad_parameters() {
        let pad = Pad::new((1, 2, 3, 4), 0.8);
        let params = pad.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "padding");
        assert_eq!(params[0].1, "(1, 2, 3, 4)");
        assert_eq!(params[1].0, "fill");
        assert_eq!(params[1].1, "0.80");
    }

    #[test]
    fn test_clone_transforms() {
        let resize = Resize::new((100, 100));
        let cloned = resize.clone_transform();
        assert_eq!(cloned.name(), "Resize");

        let normalize = Normalize::new(vec![0.1], vec![0.2]);
        let cloned = normalize.clone_transform();
        assert_eq!(cloned.name(), "Normalize");

        let pad = Pad::symmetric(2, 0.0);
        let cloned = pad.clone_transform();
        assert_eq!(cloned.name(), "Pad");
    }
}
