//! Color Manipulation Transforms for ToRSh Vision
//!
//! This module contains transforms that manipulate the color properties of images:
//! - **ColorJitter**: Random variations in brightness, contrast, saturation, and hue
//! - **GaussianBlur**: Apply Gaussian blur for denoising or augmentation
//! - **Grayscale**: Convert color images to grayscale
//! - **ColorSpace**: Convert between different color spaces
//! - **Brightness**: Adjust image brightness
//! - **Contrast**: Adjust image contrast
//! - **Saturation**: Adjust color saturation
//! - **Hue**: Adjust color hue
//!
//! These transforms are essential for data augmentation and preprocessing,
//! helping to improve model robustness to lighting conditions and color variations.

use super::core::{Transform, TransformRequirements};
use crate::{Result, VisionError};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random};
use torsh_tensor::{creation, Tensor};

// ================================================================================================
// Color Jitter Transform
// ================================================================================================

/// Random color jittering for data augmentation
///
/// This transform randomly varies the brightness, contrast, saturation, and hue
/// of images. It's one of the most effective color augmentation techniques
/// for improving model robustness to lighting conditions.
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
/// let output = jitter.forward(&input_image)?;
/// ```
pub struct ColorJitter {
    brightness: Option<f32>,
    contrast: Option<f32>,
    saturation: Option<f32>,
    hue: Option<f32>,
    seed: Option<u64>,
    apply_order: JitterOrder,
}

/// Order in which color jitter operations are applied
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitterOrder {
    /// Fixed order: brightness, contrast, saturation, hue
    Fixed,
    /// Random order for each application
    Random,
}

impl Default for JitterOrder {
    fn default() -> Self {
        JitterOrder::Random
    }
}

impl ColorJitter {
    /// Create a new color jitter transform
    pub fn new() -> Self {
        Self {
            brightness: None,
            contrast: None,
            saturation: None,
            hue: None,
            seed: None,
            apply_order: JitterOrder::default(),
        }
    }

    /// Set brightness jitter amount (0.0 = no change, 1.0 = full range)
    pub fn brightness(mut self, brightness: f32) -> Self {
        self.brightness = Some(brightness.max(0.0));
        self
    }

    /// Set contrast jitter amount
    pub fn contrast(mut self, contrast: f32) -> Self {
        self.contrast = Some(contrast.max(0.0));
        self
    }

    /// Set saturation jitter amount
    pub fn saturation(mut self, saturation: f32) -> Self {
        self.saturation = Some(saturation.max(0.0));
        self
    }

    /// Set hue jitter amount (should be ≤ 0.5)
    pub fn hue(mut self, hue: f32) -> Self {
        self.hue = Some(hue.clamp(0.0, 0.5));
        self
    }

    /// Set all parameters at once for convenience
    pub fn all(brightness: f32, contrast: f32, saturation: f32, hue: f32) -> Self {
        Self::new()
            .brightness(brightness)
            .contrast(contrast)
            .saturation(saturation)
            .hue(hue)
    }

    /// Set seed for reproducible jittering
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the order in which operations are applied
    pub fn with_order(mut self, order: JitterOrder) -> Self {
        self.apply_order = order;
        self
    }

    /// Create a strong jitter preset for aggressive augmentation
    pub fn strong() -> Self {
        Self::all(0.4, 0.4, 0.4, 0.2)
    }

    /// Create a mild jitter preset for subtle augmentation
    pub fn mild() -> Self {
        Self::all(0.1, 0.1, 0.1, 0.05)
    }

    /// Get brightness parameter
    pub fn get_brightness(&self) -> Option<f32> {
        self.brightness
    }

    /// Get contrast parameter
    pub fn get_contrast(&self) -> Option<f32> {
        self.contrast
    }

    /// Get saturation parameter
    pub fn get_saturation(&self) -> Option<f32> {
        self.saturation
    }

    /// Get hue parameter
    pub fn get_hue(&self) -> Option<f32> {
        self.hue
    }

    /// Check if any jittering is enabled
    pub fn is_active(&self) -> bool {
        self.brightness.is_some()
            || self.contrast.is_some()
            || self.saturation.is_some()
            || self.hue.is_some()
    }

    /// Generate random factors for each parameter
    fn generate_factors(&self) -> (f32, f32, f32, f32) {
        let mut rng = if let Some(seed) = self.seed {
            Random::new(seed)
        } else {
            Random::thread_rng()
        };

        let brightness_factor = if let Some(brightness) = self.brightness {
            rng.gen_range((1.0 - brightness)..=(1.0 + brightness))
        } else {
            1.0
        };

        let contrast_factor = if let Some(contrast) = self.contrast {
            rng.gen_range((1.0 - contrast)..=(1.0 + contrast))
        } else {
            1.0
        };

        let saturation_factor = if let Some(saturation) = self.saturation {
            rng.gen_range((1.0 - saturation)..=(1.0 + saturation))
        } else {
            1.0
        };

        let hue_shift = if let Some(hue) = self.hue {
            rng.gen_range((-hue), =hue)
        } else {
            0.0
        };

        (brightness_factor, contrast_factor, saturation_factor, hue_shift)
    }

    /// Apply brightness adjustment
    fn apply_brightness(&self, input: &Tensor<f32>, factor: f32) -> Result<Tensor<f32>> {
        if (factor - 1.0).abs() < 1e-6 {
            return Ok(input.clone());
        }
        crate::ops::adjust_brightness(input, factor)
    }

    /// Apply contrast adjustment
    fn apply_contrast(&self, input: &Tensor<f32>, factor: f32) -> Result<Tensor<f32>> {
        if (factor - 1.0).abs() < 1e-6 {
            return Ok(input.clone());
        }
        crate::ops::adjust_contrast(input, factor)
    }

    /// Apply saturation adjustment
    fn apply_saturation(&self, input: &Tensor<f32>, factor: f32) -> Result<Tensor<f32>> {
        if (factor - 1.0).abs() < 1e-6 {
            return Ok(input.clone());
        }
        crate::ops::adjust_saturation(input, factor)
    }

    /// Apply hue adjustment
    fn apply_hue(&self, input: &Tensor<f32>, shift: f32) -> Result<Tensor<f32>> {
        if shift.abs() < 1e-6 {
            return Ok(input.clone());
        }
        crate::ops::adjust_hue(input, shift)
    }

    /// Determine the order of operations
    fn get_operation_order(&self) -> Vec<JitterOperation> {
        let mut operations = vec![
            JitterOperation::Brightness,
            JitterOperation::Contrast,
            JitterOperation::Saturation,
            JitterOperation::Hue,
        ];

        if self.apply_order == JitterOrder::Random {
            // Shuffle the operations randomly
            let mut rng = if let Some(seed) = self.seed {
                Random::new(seed.wrapping_add(1)) // Different seed for ordering
            } else {
                Random::thread_rng()
            };

            // Simple shuffle
            for i in (1..operations.len()).rev() {
                let j = rng.gen_range(0..=i);
                operations.swap(i, j);
            }
        }

        operations
    }
}

/// Individual jitter operations
#[derive(Debug, Clone, Copy)]
enum JitterOperation {
    Brightness,
    Contrast,
    Saturation,
    Hue,
}

impl Default for ColorJitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ColorJitter {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if !self.is_active() {
            return Ok(input.clone());
        }

        let (brightness_factor, contrast_factor, saturation_factor, hue_shift) =
            self.generate_factors();

        let mut output = input.clone();
        let operations = self.get_operation_order();

        for operation in operations {
            output = match operation {
                JitterOperation::Brightness if self.brightness.is_some() => {
                    self.apply_brightness(&output, brightness_factor)?
                }
                JitterOperation::Contrast if self.contrast.is_some() => {
                    self.apply_contrast(&output, contrast_factor)?
                }
                JitterOperation::Saturation if self.saturation.is_some() => {
                    self.apply_saturation(&output, saturation_factor)?
                }
                JitterOperation::Hue if self.hue.is_some() => {
                    self.apply_hue(&output, hue_shift)?
                }
                _ => output, // Skip if parameter is None
            };
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "ColorJitter"
    }

    fn is_deterministic(&self) -> bool {
        self.seed.is_some() || !self.is_active()
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        let mut params = Vec::new();
        if let Some(b) = self.brightness {
            params.push(("brightness", format!("{:.3}", b)));
        }
        if let Some(c) = self.contrast {
            params.push(("contrast", format!("{:.3}", c)));
        }
        if let Some(s) = self.saturation {
            params.push(("saturation", format!("{:.3}", s)));
        }
        if let Some(h) = self.hue {
            params.push(("hue", format!("{:.3}", h)));
        }
        if let Some(seed) = self.seed {
            params.push(("seed", seed.to_string()));
        }
        params.push(("order", format!("{:?}", self.apply_order)));
        params
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        let mut jitter = ColorJitter::new().with_order(self.apply_order);
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
        if let Some(seed) = self.seed {
            jitter = jitter.with_seed(seed);
        }
        Box::new(jitter)
    }

    fn computational_cost(&self) -> f32 {
        let mut cost = 0.0;
        if self.brightness.is_some() { cost += 1.0; }
        if self.contrast.is_some() { cost += 1.5; }
        if self.saturation.is_some() { cost += 2.0; }
        if self.hue.is_some() { cost += 3.0; }
        cost
    }

    fn input_requirements(&self) -> TransformRequirements {
        TransformRequirements {
            min_dimensions: 3,
            max_dimensions: 4,
            required_channels: Some(3), // RGB images for full color jittering
            requires_chw_format: true,
            requires_normalized: false, // Can work with any value range
            ..Default::default()
        }
    }
}

// ================================================================================================
// Gaussian Blur Transform
// ================================================================================================

/// Apply Gaussian blur to images
///
/// This transform applies a Gaussian blur filter to images, which can be used
/// for denoising, data augmentation, or preprocessing.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{GaussianBlur, Transform};
///
/// let blur = GaussianBlur::new(5, 1.0); // kernel_size=5, sigma=1.0
/// let output = blur.forward(&input_image)?;
/// ```
pub struct GaussianBlur {
    kernel_size: usize,
    sigma: f32,
    sigma_range: Option<(f32, f32)>,
    seed: Option<u64>,
}

impl GaussianBlur {
    /// Create a new Gaussian blur transform
    pub fn new(kernel_size: usize, sigma: f32) -> Self {
        let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
        Self {
            kernel_size,
            sigma: sigma.max(0.1),
            sigma_range: None,
            seed: None,
        }
    }

    /// Create with random sigma in given range
    pub fn with_random_sigma(kernel_size: usize, sigma_range: (f32, f32)) -> Self {
        let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
        Self {
            kernel_size,
            sigma: sigma_range.0, // Will be overridden during forward
            sigma_range: Some(sigma_range),
            seed: None,
        }
    }

    /// Set seed for reproducible random sigma
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get sigma value (if fixed)
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Get sigma range (if random)
    pub fn sigma_range(&self) -> Option<(f32, f32)> {
        self.sigma_range
    }

    /// Generate sigma value (either fixed or random)
    fn get_sigma(&self) -> f32 {
        if let Some((min_sigma, max_sigma)) = self.sigma_range {
            let mut rng = if let Some(seed) = self.seed {
                Random::new(seed)
            } else {
                Random::thread_rng()
            };
            rng.gen_range(min_sigma..max_sigma)
        } else {
            self.sigma
        }
    }

    /// Calculate kernel size from sigma (Gaussian rule of thumb)
    pub fn kernel_size_from_sigma(sigma: f32) -> usize {
        let size = ((sigma * 6.0).ceil() as usize).max(3);
        if size % 2 == 0 { size + 1 } else { size }
    }

    /// Create blur with automatically calculated kernel size
    pub fn auto_kernel(sigma: f32) -> Self {
        let kernel_size = Self::kernel_size_from_sigma(sigma);
        Self::new(kernel_size, sigma)
    }
}

impl Transform for GaussianBlur {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let sigma = self.get_sigma();
        crate::ops::gaussian_blur(input, self.kernel_size, sigma)
    }

    fn name(&self) -> &'static str {
        "GaussianBlur"
    }

    fn is_deterministic(&self) -> bool {
        self.sigma_range.is_none() || self.seed.is_some()
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        let mut params = vec![
            ("kernel_size", self.kernel_size.to_string()),
        ];
        if let Some((min_sigma, max_sigma)) = self.sigma_range {
            params.push(("sigma_range", format!("({:.2}, {:.2})", min_sigma, max_sigma)));
        } else {
            params.push(("sigma", format!("{:.2}", self.sigma)));
        }
        if let Some(seed) = self.seed {
            params.push(("seed", seed.to_string()));
        }
        params
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        let mut blur = if let Some(range) = self.sigma_range {
            GaussianBlur::with_random_sigma(self.kernel_size, range)
        } else {
            GaussianBlur::new(self.kernel_size, self.sigma)
        };
        if let Some(seed) = self.seed {
            blur = blur.with_seed(seed);
        }
        Box::new(blur)
    }

    fn computational_cost(&self) -> f32 {
        // Blur cost is proportional to kernel size squared
        (self.kernel_size * self.kernel_size) as f32 / 10.0
    }

    fn input_requirements(&self) -> TransformRequirements {
        TransformRequirements {
            min_dimensions: 3,
            max_dimensions: 4,
            requires_chw_format: true,
            min_spatial_size: Some((self.kernel_size, self.kernel_size)),
            ..Default::default()
        }
    }
}

// ================================================================================================
// Individual Color Adjustment Transforms
// ================================================================================================

/// Adjust image brightness
///
/// Applies a brightness adjustment to images by adding or multiplying a constant value.
pub struct BrightnessAdjust {
    factor: f32,
    additive: bool,
}

impl BrightnessAdjust {
    /// Create multiplicative brightness adjustment
    pub fn multiplicative(factor: f32) -> Self {
        Self {
            factor: factor.max(0.0),
            additive: false,
        }
    }

    /// Create additive brightness adjustment
    pub fn additive(value: f32) -> Self {
        Self {
            factor: value,
            additive: true,
        }
    }

    /// Get the adjustment factor/value
    pub fn factor(&self) -> f32 {
        self.factor
    }

    /// Check if adjustment is additive
    pub fn is_additive(&self) -> bool {
        self.additive
    }
}

impl Transform for BrightnessAdjust {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        if self.additive {
            crate::ops::add_brightness(input, self.factor)
        } else {
            crate::ops::adjust_brightness(input, self.factor)
        }
    }

    fn name(&self) -> &'static str {
        if self.additive {
            "BrightnessAdd"
        } else {
            "BrightnessMultiply"
        }
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("factor", self.factor.to_string()),
            ("mode", if self.additive { "additive" } else { "multiplicative" }.to_string()),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(BrightnessAdjust {
            factor: self.factor,
            additive: self.additive,
        })
    }

    fn computational_cost(&self) -> f32 {
        0.5 // Simple per-pixel operation
    }
}

/// Adjust image contrast
pub struct ContrastAdjust {
    factor: f32,
}

impl ContrastAdjust {
    /// Create contrast adjustment transform
    pub fn new(factor: f32) -> Self {
        Self {
            factor: factor.max(0.0),
        }
    }

    /// Get the contrast factor
    pub fn factor(&self) -> f32 {
        self.factor
    }
}

impl Transform for ContrastAdjust {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::adjust_contrast(input, self.factor)
    }

    fn name(&self) -> &'static str {
        "ContrastAdjust"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("factor", self.factor.to_string())]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(ContrastAdjust::new(self.factor))
    }

    fn computational_cost(&self) -> f32 {
        1.0 // Requires mean calculation
    }
}

/// Convert color images to grayscale
pub struct Grayscale {
    weights: [f32; 3], // RGB weights
    keep_channels: bool,
}

impl Grayscale {
    /// Create grayscale transform with luminance weights
    pub fn new() -> Self {
        Self {
            weights: [0.299, 0.587, 0.114], // Standard luminance weights
            keep_channels: false,
        }
    }

    /// Create grayscale with custom RGB weights
    pub fn with_weights(r: f32, g: f32, b: f32) -> Self {
        // Normalize weights
        let sum = r + g + b;
        Self {
            weights: [r / sum, g / sum, b / sum],
            keep_channels: false,
        }
    }

    /// Keep 3 channels (duplicate grayscale value) instead of reducing to 1
    pub fn keep_channels(mut self) -> Self {
        self.keep_channels = true;
        self
    }

    /// Get RGB weights
    pub fn weights(&self) -> [f32; 3] {
        self.weights
    }

    /// Check if channels are kept
    pub fn keeps_channels(&self) -> bool {
        self.keep_channels
    }
}

impl Default for Grayscale {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Grayscale {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::rgb_to_grayscale(input, self.weights, self.keep_channels)
    }

    fn name(&self) -> &'static str {
        "Grayscale"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("weights", format!("{:.3}, {:.3}, {:.3}", self.weights[0], self.weights[1], self.weights[2])),
            ("keep_channels", self.keep_channels.to_string()),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        let mut gray = Grayscale::with_weights(self.weights[0], self.weights[1], self.weights[2]);
        if self.keep_channels {
            gray = gray.keep_channels();
        }
        Box::new(gray)
    }

    fn computational_cost(&self) -> f32 {
        1.5 // Weighted sum across channels
    }

    fn input_requirements(&self) -> TransformRequirements {
        TransformRequirements {
            min_dimensions: 3,
            max_dimensions: 4,
            required_channels: Some(3), // RGB input required
            requires_chw_format: true,
            ..Default::default()
        }
    }
}

// ================================================================================================
// Comprehensive Test Suite
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::rand;

    #[test]
    fn test_color_jitter_creation() {
        let jitter = ColorJitter::new()
            .brightness(0.2)
            .contrast(0.1)
            .saturation(0.3)
            .hue(0.05);

        assert_eq!(jitter.get_brightness(), Some(0.2));
        assert_eq!(jitter.get_contrast(), Some(0.1));
        assert_eq!(jitter.get_saturation(), Some(0.3));
        assert_eq!(jitter.get_hue(), Some(0.05));
        assert!(jitter.is_active());
    }

    #[test]
    fn test_color_jitter_presets() {
        let strong = ColorJitter::strong();
        assert!(strong.is_active());
        assert!(strong.get_brightness().unwrap() > 0.3);

        let mild = ColorJitter::mild();
        assert!(mild.is_active());
        assert!(mild.get_brightness().unwrap() < 0.2);
    }

    #[test]
    fn test_color_jitter_determinism() {
        let jitter = ColorJitter::new().brightness(0.5);
        assert!(!jitter.is_deterministic());

        let jitter_seeded = ColorJitter::new().brightness(0.5).with_seed(42);
        assert!(jitter_seeded.is_deterministic());

        let inactive_jitter = ColorJitter::new();
        assert!(inactive_jitter.is_deterministic()); // No operations = deterministic
    }

    #[test]
    fn test_color_jitter_all_constructor() {
        let jitter = ColorJitter::all(0.1, 0.2, 0.3, 0.05);
        assert_eq!(jitter.get_brightness(), Some(0.1));
        assert_eq!(jitter.get_contrast(), Some(0.2));
        assert_eq!(jitter.get_saturation(), Some(0.3));
        assert_eq!(jitter.get_hue(), Some(0.05));
    }

    #[test]
    fn test_color_jitter_parameter_bounds() {
        let jitter = ColorJitter::new().hue(0.8); // Should be clamped to 0.5
        assert_eq!(jitter.get_hue(), Some(0.5));

        let jitter = ColorJitter::new().brightness(-0.1); // Should be clamped to 0.0
        assert_eq!(jitter.get_brightness(), Some(0.0));
    }

    #[test]
    fn test_gaussian_blur_creation() {
        let blur = GaussianBlur::new(5, 1.0);
        assert_eq!(blur.kernel_size(), 5);
        assert_eq!(blur.sigma(), 1.0);
        assert!(blur.is_deterministic());

        // Even kernel size should be adjusted to odd
        let blur_even = GaussianBlur::new(6, 1.5);
        assert_eq!(blur_even.kernel_size(), 7);
    }

    #[test]
    fn test_gaussian_blur_random_sigma() {
        let blur = GaussianBlur::with_random_sigma(7, (0.5, 2.0));
        assert_eq!(blur.sigma_range(), Some((0.5, 2.0)));
        assert!(!blur.is_deterministic());

        let blur_seeded = blur.with_seed(42);
        assert!(blur_seeded.is_deterministic());
    }

    #[test]
    fn test_gaussian_blur_auto_kernel() {
        let blur = GaussianBlur::auto_kernel(1.0);
        let expected_size = GaussianBlur::kernel_size_from_sigma(1.0);
        assert_eq!(blur.kernel_size(), expected_size);
        assert!(expected_size >= 3);
        assert_eq!(expected_size % 2, 1); // Should be odd
    }

    #[test]
    fn test_brightness_adjust() {
        let bright_mult = BrightnessAdjust::multiplicative(1.2);
        assert_eq!(bright_mult.factor(), 1.2);
        assert!(!bright_mult.is_additive());
        assert_eq!(bright_mult.name(), "BrightnessMultiply");

        let bright_add = BrightnessAdjust::additive(0.1);
        assert_eq!(bright_add.factor(), 0.1);
        assert!(bright_add.is_additive());
        assert_eq!(bright_add.name(), "BrightnessAdd");
    }

    #[test]
    fn test_contrast_adjust() {
        let contrast = ContrastAdjust::new(1.3);
        assert_eq!(contrast.factor(), 1.3);
        assert_eq!(contrast.name(), "ContrastAdjust");

        // Test negative factor clamping
        let contrast_clamped = ContrastAdjust::new(-0.5);
        assert_eq!(contrast_clamped.factor(), 0.0);
    }

    #[test]
    fn test_grayscale() {
        let gray = Grayscale::new();
        assert_eq!(gray.weights(), [0.299, 0.587, 0.114]);
        assert!(!gray.keeps_channels());

        let gray_keep = gray.keep_channels();
        assert!(gray_keep.keeps_channels());

        let gray_custom = Grayscale::with_weights(0.3, 0.6, 0.1);
        assert_eq!(gray_custom.weights(), [0.3, 0.6, 0.1]);
    }

    #[test]
    fn test_grayscale_weight_normalization() {
        let gray = Grayscale::with_weights(2.0, 4.0, 2.0); // Sum = 8.0
        let weights = gray.weights();
        assert!((weights[0] - 0.25).abs() < 1e-6);
        assert!((weights[1] - 0.5).abs() < 1e-6);
        assert!((weights[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_transform_requirements() {
        let jitter = ColorJitter::new().brightness(0.2);
        let req = jitter.input_requirements();
        assert_eq!(req.required_channels, Some(3));
        assert!(req.requires_chw_format);

        let blur = GaussianBlur::new(7, 1.0);
        let req = blur.input_requirements();
        assert_eq!(req.min_spatial_size, Some((7, 7)));

        let gray = Grayscale::new();
        let req = gray.input_requirements();
        assert_eq!(req.required_channels, Some(3));
    }

    #[test]
    fn test_computational_costs() {
        let jitter = ColorJitter::all(0.1, 0.1, 0.1, 0.05);
        assert!(jitter.computational_cost() > 5.0); // Sum of all operations

        let blur = GaussianBlur::new(5, 1.0);
        assert!(blur.computational_cost() > 2.0); // Proportional to kernel size

        let bright = BrightnessAdjust::multiplicative(1.2);
        assert_eq!(bright.computational_cost(), 0.5);

        let contrast = ContrastAdjust::new(1.1);
        assert_eq!(contrast.computational_cost(), 1.0);

        let gray = Grayscale::new();
        assert_eq!(gray.computational_cost(), 1.5);
    }

    #[test]
    fn test_transform_cloning() {
        let jitter = ColorJitter::all(0.1, 0.2, 0.3, 0.05).with_seed(42);
        let cloned = jitter.clone_transform();
        assert_eq!(cloned.name(), "ColorJitter");

        let blur = GaussianBlur::with_random_sigma(7, (0.5, 2.0)).with_seed(123);
        let cloned = blur.clone_transform();
        assert_eq!(cloned.name(), "GaussianBlur");

        let gray = Grayscale::with_weights(0.2, 0.7, 0.1).keep_channels();
        let cloned = gray.clone_transform();
        assert_eq!(cloned.name(), "Grayscale");
    }

    #[test]
    fn test_transform_parameters() {
        let jitter = ColorJitter::all(0.1, 0.2, 0.3, 0.05)
            .with_seed(42)
            .with_order(JitterOrder::Fixed);
        let params = jitter.parameters();
        assert!(params.iter().any(|(k, _)| *k == "brightness"));
        assert!(params.iter().any(|(k, _)| *k == "contrast"));
        assert!(params.iter().any(|(k, _)| *k == "saturation"));
        assert!(params.iter().any(|(k, _)| *k == "hue"));
        assert!(params.iter().any(|(k, _)| *k == "seed"));
        assert!(params.iter().any(|(k, _)| *k == "order"));

        let blur = GaussianBlur::with_random_sigma(5, (0.5, 1.5));
        let params = blur.parameters();
        assert!(params.iter().any(|(k, _)| *k == "kernel_size"));
        assert!(params.iter().any(|(k, _)| *k == "sigma_range"));
    }

    #[test]
    fn test_jitter_order() {
        assert_eq!(JitterOrder::default(), JitterOrder::Random);

        let jitter = ColorJitter::new().with_order(JitterOrder::Fixed);
        // Can't easily test the actual ordering without exposing internal methods,
        // but we can test that it doesn't crash
        let input = rand::<f32>(&[3, 32, 32]).unwrap();
        let _result = jitter.forward(&input);
        // Result depends on ops module implementation
    }

    #[test]
    fn test_forward_operations() {
        let input = rand::<f32>(&[3, 32, 32]).unwrap();

        // Test that transforms don't panic (actual results depend on ops module)
        let jitter = ColorJitter::new(); // Inactive jitter should return clone
        let result = jitter.forward(&input);
        // Should succeed as inactive jitter just returns input

        let bright = BrightnessAdjust::multiplicative(1.0); // No change
        let _result = bright.forward(&input);

        let gray = Grayscale::new();
        let _result = gray.forward(&input);
    }
}