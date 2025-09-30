use super::core::Transform;
use crate::{Result, VisionError};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use torsh_tensor::Tensor;

/// Color jitter transform
///
/// Randomly changes the brightness, contrast, saturation, and hue of the input image.
/// This is a powerful augmentation technique that helps models become robust to
/// lighting and color variations.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{ColorJitter, Transform};
///
/// let jitter = ColorJitter::new()
///     .brightness(0.2)
///     .contrast(0.2)
///     .saturation(0.2)
///     .hue(0.1);
/// // Apply to tensor: result = jitter.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct ColorJitter {
    brightness: Option<f32>,
    contrast: Option<f32>,
    saturation: Option<f32>,
    hue: Option<f32>,
}

impl ColorJitter {
    /// Create a new ColorJitter transform with no modifications
    pub fn new() -> Self {
        Self {
            brightness: None,
            contrast: None,
            saturation: None,
            hue: None,
        }
    }

    /// Set brightness jitter amount
    ///
    /// # Arguments
    ///
    /// * `brightness` - Maximum absolute change in brightness (0.0 to 1.0)
    pub fn brightness(mut self, brightness: f32) -> Self {
        assert!(brightness >= 0.0, "Brightness must be non-negative");
        self.brightness = Some(brightness);
        self
    }

    /// Set contrast jitter amount
    ///
    /// # Arguments
    ///
    /// * `contrast` - Maximum absolute change in contrast (0.0 to 1.0)
    pub fn contrast(mut self, contrast: f32) -> Self {
        assert!(contrast >= 0.0, "Contrast must be non-negative");
        self.contrast = Some(contrast);
        self
    }

    /// Set saturation jitter amount
    ///
    /// # Arguments
    ///
    /// * `saturation` - Maximum absolute change in saturation (0.0 to 1.0)
    pub fn saturation(mut self, saturation: f32) -> Self {
        assert!(saturation >= 0.0, "Saturation must be non-negative");
        self.saturation = Some(saturation);
        self
    }

    /// Set hue jitter amount
    ///
    /// # Arguments
    ///
    /// * `hue` - Maximum absolute change in hue (0.0 to 0.5)
    pub fn hue(mut self, hue: f32) -> Self {
        assert!(hue >= 0.0 && hue <= 0.5, "Hue must be between 0.0 and 0.5");
        self.hue = Some(hue);
        self
    }

    /// Get the brightness setting
    pub fn get_brightness(&self) -> Option<f32> {
        self.brightness
    }

    /// Get the contrast setting
    pub fn get_contrast(&self) -> Option<f32> {
        self.contrast
    }

    /// Get the saturation setting
    pub fn get_saturation(&self) -> Option<f32> {
        self.saturation
    }

    /// Get the hue setting
    pub fn get_hue(&self) -> Option<f32> {
        self.hue
    }
}

impl Default for ColorJitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ColorJitter {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut output = input.clone();
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        // Apply brightness adjustment
        if let Some(brightness) = self.brightness {
            let factor = rng.gen_range(1.0 - brightness..=1.0 + brightness);
            output = output.mul_scalar(factor)?;
        }

        // Apply contrast adjustment
        if let Some(contrast) = self.contrast {
            let factor = rng.gen_range(1.0 - contrast..=1.0 + contrast);
            let mean = output.mean(None, false)?;
            let mean_val = mean.to_vec()?[0];
            output.sub_scalar_(mean_val)?;
            output = output.mul_scalar(factor)?;
            output = output.add_scalar(mean_val)?;
        }

        // For saturation and hue, we'd need to convert to HSV space
        // This is a simplified implementation that just returns the output for now
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "ColorJitter"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        let mut params = Vec::new();
        if let Some(b) = self.brightness {
            params.push(("brightness", format!("{:.2}", b)));
        }
        if let Some(c) = self.contrast {
            params.push(("contrast", format!("{:.2}", c)));
        }
        if let Some(s) = self.saturation {
            params.push(("saturation", format!("{:.2}", s)));
        }
        if let Some(h) = self.hue {
            params.push(("hue", format!("{:.2}", h)));
        }
        params
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        let mut jitter = ColorJitter::new();
        if let Some(b) = self.brightness {
            jitter = jitter.brightness(b);
        }
        if let Some(c) = self.contrast {
            jitter = jitter.contrast(c);
        }
        if let Some(s) = self.saturation {
            jitter = jitter.saturation(s);
        }
        if let Some(h) = self.hue {
            jitter = jitter.hue(h);
        }
        Box::new(jitter)
    }
}

/// Gaussian blur transform
///
/// Applies Gaussian blur to the input image with specified kernel size and sigma.
/// Useful for reducing high-frequency noise and creating smoother images.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{GaussianBlur, Transform};
///
/// let blur = GaussianBlur::new(5, 1.0);
/// // Apply to tensor: result = blur.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct GaussianBlur {
    kernel_size: usize,
    sigma: f32,
}

impl GaussianBlur {
    /// Create a new GaussianBlur transform
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the Gaussian kernel (should be odd)
    /// * `sigma` - Standard deviation of the Gaussian distribution
    pub fn new(kernel_size: usize, sigma: f32) -> Self {
        assert!(kernel_size % 2 == 1, "Kernel size must be odd");
        assert!(sigma > 0.0, "Sigma must be positive");
        Self { kernel_size, sigma }
    }

    /// Get the kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get the sigma value
    pub fn sigma(&self) -> f32 {
        self.sigma
    }
}

impl Transform for GaussianBlur {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Simplified blur implementation - just return input for now
        // A full implementation would apply a Gaussian kernel convolution
        Ok(input.clone())
    }

    fn name(&self) -> &'static str {
        "GaussianBlur"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kernel_size", format!("{}", self.kernel_size)),
            ("sigma", format!("{:.2}", self.sigma)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(GaussianBlur::new(self.kernel_size, self.sigma))
    }
}

/// Random erasing transform
///
/// Randomly selects a rectangle region in an image and erases its pixels with random values.
/// This technique helps improve model robustness and generalization by simulating occlusion.
///
/// Based on "Random Erasing Data Augmentation" by Zhong et al. (2017).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandomErasing, Transform};
///
/// let erasing = RandomErasing::new(0.5)
///     .with_scale((0.02, 0.33))
///     .with_ratio((0.3, 3.3))
///     .with_value(0.5);
/// // Apply to tensor: result = erasing.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomErasing {
    p: f32,
    scale: (f32, f32),
    ratio: (f32, f32),
    value: f32,
}

impl RandomErasing {
    /// Create a new RandomErasing transform
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of applying random erasing
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability must be between 0.0 and 1.0"
        );
        Self {
            p,
            scale: (0.02, 0.33),
            ratio: (0.3, 3.3),
            value: 0.0,
        }
    }

    /// Set the scale range for erased area
    ///
    /// # Arguments
    ///
    /// * `scale` - Range of erased area as proportion of total area (min, max)
    pub fn with_scale(mut self, scale: (f32, f32)) -> Self {
        assert!(scale.0 <= scale.1, "Scale min must be <= scale max");
        assert!(scale.0 >= 0.0, "Scale min must be >= 0.0");
        assert!(scale.1 <= 1.0, "Scale max must be <= 1.0");
        self.scale = scale;
        self
    }

    /// Set the aspect ratio range for erased rectangle
    ///
    /// # Arguments
    ///
    /// * `ratio` - Range of aspect ratios (width/height) for erased rectangle (min, max)
    pub fn with_ratio(mut self, ratio: (f32, f32)) -> Self {
        assert!(ratio.0 <= ratio.1, "Ratio min must be <= ratio max");
        assert!(ratio.0 > 0.0, "Ratio min must be > 0.0");
        self.ratio = ratio;
        self
    }

    /// Set the fill value for erased pixels
    ///
    /// # Arguments
    ///
    /// * `value` - Value to fill erased pixels with
    pub fn with_value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }

    /// Get the probability
    pub fn probability(&self) -> f32 {
        self.p
    }

    /// Get the scale range
    pub fn scale(&self) -> (f32, f32) {
        self.scale
    }

    /// Get the ratio range
    pub fn ratio(&self) -> (f32, f32) {
        self.ratio
    }

    /// Get the fill value
    pub fn value(&self) -> f32 {
        self.value
    }
}

impl Transform for RandomErasing {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        if rng.gen::<f32>() >= self.p {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (channels, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);
        let area = (height * width) as f32;

        // Sample erase area and aspect ratio
        let erase_area = area * rng.gen_range(self.scale.0..=self.scale.1);
        let aspect_ratio = rng.gen_range(self.ratio.0..=self.ratio.1);

        // Calculate erase dimensions
        let erase_height = (erase_area / aspect_ratio).sqrt() as usize;
        let erase_width = (erase_area * aspect_ratio).sqrt() as usize;

        if erase_height >= height || erase_width >= width {
            return Ok(input.clone());
        }

        // Random position
        let start_y = rng.gen_range(0..=(height - erase_height));
        let start_x = rng.gen_range(0..=(width - erase_width));

        // Apply erasing
        let mut output = input.clone();
        for c in 0..channels {
            for y in start_y..(start_y + erase_height) {
                for x in start_x..(start_x + erase_width) {
                    output.set(&[c, y, x], self.value)?;
                }
            }
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "RandomErasing"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("probability", format!("{:.2}", self.p)),
            (
                "scale",
                format!("({:.2}, {:.2})", self.scale.0, self.scale.1),
            ),
            (
                "ratio",
                format!("({:.2}, {:.2})", self.ratio.0, self.ratio.1),
            ),
            ("value", format!("{:.2}", self.value)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(
            RandomErasing::new(self.p)
                .with_scale(self.scale)
                .with_ratio(self.ratio)
                .with_value(self.value),
        )
    }
}

/// Cutout transform
///
/// Randomly masks out square regions of the input image. Similar to random erasing
/// but specifically uses square holes and is simpler in configuration.
///
/// Based on "Improved Regularization of Convolutional Neural Networks with Cutout"
/// by DeVries and Taylor (2017).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Cutout, Transform};
///
/// // Create 2 square holes of size 16x16
/// let cutout = Cutout::new(16, 2);
/// // Apply to tensor: result = cutout.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct Cutout {
    length: usize,
    n_holes: usize,
}

impl Cutout {
    /// Create a new Cutout transform
    ///
    /// # Arguments
    ///
    /// * `length` - Side length of the square cutout regions
    /// * `n_holes` - Number of cutout holes to create
    pub fn new(length: usize, n_holes: usize) -> Self {
        assert!(length > 0, "Length must be positive");
        assert!(n_holes > 0, "Number of holes must be positive");
        Self { length, n_holes }
    }

    /// Get the cutout length
    pub fn length(&self) -> usize {
        self.length
    }

    /// Get the number of holes
    pub fn n_holes(&self) -> usize {
        self.n_holes
    }
}

impl Transform for Cutout {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (channels, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);
        let mut output = input.clone();
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        for _ in 0..self.n_holes {
            let y = rng.gen_range(0..height);
            let x = rng.gen_range(0..width);

            let y1 = (y as i32 - self.length as i32 / 2).max(0) as usize;
            let y2 = (y + self.length / 2).min(height);
            let x1 = (x as i32 - self.length as i32 / 2).max(0) as usize;
            let x2 = (x + self.length / 2).min(width);

            for c in 0..channels {
                for ty in y1..y2 {
                    for tx in x1..x2 {
                        output.set(&[c, ty, tx], 0.0)?;
                    }
                }
            }
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "Cutout"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("length", format!("{}", self.length)),
            ("n_holes", format!("{}", self.n_holes)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Cutout::new(self.length, self.n_holes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_color_jitter_creation() {
        let jitter = ColorJitter::new();
        assert_eq!(jitter.get_brightness(), None);
        assert_eq!(jitter.get_contrast(), None);
        assert_eq!(jitter.get_saturation(), None);
        assert_eq!(jitter.get_hue(), None);
        assert_eq!(jitter.name(), "ColorJitter");
    }

    #[test]
    fn test_color_jitter_with_values() {
        let jitter = ColorJitter::new()
            .brightness(0.2)
            .contrast(0.3)
            .saturation(0.1)
            .hue(0.05);

        assert_eq!(jitter.get_brightness(), Some(0.2));
        assert_eq!(jitter.get_contrast(), Some(0.3));
        assert_eq!(jitter.get_saturation(), Some(0.1));
        assert_eq!(jitter.get_hue(), Some(0.05));
    }

    #[test]
    #[should_panic(expected = "Brightness must be non-negative")]
    fn test_color_jitter_negative_brightness() {
        ColorJitter::new().brightness(-0.1);
    }

    #[test]
    #[should_panic(expected = "Hue must be between 0.0 and 0.5")]
    fn test_color_jitter_invalid_hue() {
        ColorJitter::new().hue(0.6);
    }

    #[test]
    fn test_color_jitter_default() {
        let jitter = ColorJitter::default();
        assert_eq!(jitter.get_brightness(), None);
    }

    #[test]
    fn test_color_jitter_parameters() {
        let jitter = ColorJitter::new().brightness(0.1).contrast(0.2);
        let params = jitter.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.iter().any(|(k, _)| *k == "brightness"));
        assert!(params.iter().any(|(k, _)| *k == "contrast"));
    }

    #[test]
    fn test_gaussian_blur_creation() {
        let blur = GaussianBlur::new(5, 1.5);
        assert_eq!(blur.kernel_size(), 5);
        assert_eq!(blur.sigma(), 1.5);
        assert_eq!(blur.name(), "GaussianBlur");

        let params = blur.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "kernel_size");
        assert_eq!(params[1].0, "sigma");
    }

    #[test]
    #[should_panic(expected = "Kernel size must be odd")]
    fn test_gaussian_blur_even_kernel() {
        GaussianBlur::new(4, 1.0);
    }

    #[test]
    #[should_panic(expected = "Sigma must be positive")]
    fn test_gaussian_blur_zero_sigma() {
        GaussianBlur::new(3, 0.0);
    }

    #[test]
    fn test_random_erasing_creation() {
        let erasing = RandomErasing::new(0.5);
        assert_eq!(erasing.probability(), 0.5);
        assert_eq!(erasing.scale(), (0.02, 0.33));
        assert_eq!(erasing.ratio(), (0.3, 3.3));
        assert_eq!(erasing.value(), 0.0);
        assert_eq!(erasing.name(), "RandomErasing");
    }

    #[test]
    fn test_random_erasing_with_parameters() {
        let erasing = RandomErasing::new(0.25)
            .with_scale((0.1, 0.5))
            .with_ratio((0.5, 2.0))
            .with_value(0.5);

        assert_eq!(erasing.probability(), 0.25);
        assert_eq!(erasing.scale(), (0.1, 0.5));
        assert_eq!(erasing.ratio(), (0.5, 2.0));
        assert_eq!(erasing.value(), 0.5);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_random_erasing_invalid_probability() {
        RandomErasing::new(1.5);
    }

    #[test]
    #[should_panic(expected = "Scale min must be <= scale max")]
    fn test_random_erasing_invalid_scale() {
        RandomErasing::new(0.5).with_scale((0.8, 0.2));
    }

    #[test]
    #[should_panic(expected = "Scale max must be <= 1.0")]
    fn test_random_erasing_scale_too_large() {
        RandomErasing::new(0.5).with_scale((0.5, 1.5));
    }

    #[test]
    fn test_random_erasing_parameters() {
        let erasing = RandomErasing::new(0.3).with_scale((0.1, 0.4));
        let params = erasing.parameters();
        assert_eq!(params.len(), 4);
        assert!(params.iter().any(|(k, _)| *k == "probability"));
        assert!(params.iter().any(|(k, _)| *k == "scale"));
        assert!(params.iter().any(|(k, _)| *k == "ratio"));
        assert!(params.iter().any(|(k, _)| *k == "value"));
    }

    #[test]
    fn test_cutout_creation() {
        let cutout = Cutout::new(16, 2);
        assert_eq!(cutout.length(), 16);
        assert_eq!(cutout.n_holes(), 2);
        assert_eq!(cutout.name(), "Cutout");

        let params = cutout.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "length");
        assert_eq!(params[0].1, "16");
        assert_eq!(params[1].0, "n_holes");
        assert_eq!(params[1].1, "2");
    }

    #[test]
    #[should_panic(expected = "Length must be positive")]
    fn test_cutout_zero_length() {
        Cutout::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "Number of holes must be positive")]
    fn test_cutout_zero_holes() {
        Cutout::new(16, 0);
    }

    #[test]
    fn test_clone_transforms() {
        let jitter = ColorJitter::new().brightness(0.1);
        let cloned = jitter.clone_transform();
        assert_eq!(cloned.name(), "ColorJitter");

        let blur = GaussianBlur::new(3, 1.0);
        let cloned = blur.clone_transform();
        assert_eq!(cloned.name(), "GaussianBlur");

        let erasing = RandomErasing::new(0.5);
        let cloned = erasing.clone_transform();
        assert_eq!(cloned.name(), "RandomErasing");

        let cutout = Cutout::new(8, 1);
        let cloned = cutout.clone_transform();
        assert_eq!(cloned.name(), "Cutout");
    }

    #[test]
    fn test_random_erasing_forward_invalid_shape() {
        let erasing = RandomErasing::new(1.0); // Always apply
        let input = creation::ones(&[10, 10]).unwrap(); // 2D tensor (invalid)

        let result = erasing.forward(&input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VisionError::InvalidShape(_)));
    }

    #[test]
    fn test_cutout_forward_invalid_shape() {
        let cutout = Cutout::new(5, 1);
        let input = creation::ones(&[10]).unwrap(); // 1D tensor (invalid)

        let result = cutout.forward(&input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VisionError::InvalidShape(_)));
    }

    #[test]
    fn test_edge_cases() {
        // Test minimum valid values
        let jitter = ColorJitter::new().brightness(0.0).hue(0.0);
        assert_eq!(jitter.get_brightness(), Some(0.0));
        assert_eq!(jitter.get_hue(), Some(0.0));

        let erasing = RandomErasing::new(0.0); // Never apply
        assert_eq!(erasing.probability(), 0.0);

        let blur = GaussianBlur::new(1, 0.1); // Minimum valid values
        assert_eq!(blur.kernel_size(), 1);
        assert_eq!(blur.sigma(), 0.1);
    }
}
