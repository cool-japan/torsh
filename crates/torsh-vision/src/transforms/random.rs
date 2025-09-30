use super::core::Transform;
use crate::{Result, VisionError};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use torsh_tensor::Tensor;

/// Random horizontal flip transform
///
/// Horizontally flips the input image with a given probability. This is one of the
/// most commonly used data augmentation techniques.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandomHorizontalFlip, Transform};
///
/// // Flip with 50% probability
/// let flip = RandomHorizontalFlip::new(0.5);
/// // Apply to tensor: result = flip.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip {
    p: f32,
}

impl RandomHorizontalFlip {
    /// Create a new RandomHorizontalFlip transform
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of applying the flip (0.0 to 1.0)
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the range [0.0, 1.0]
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability must be between 0.0 and 1.0"
        );
        Self { p }
    }

    /// Get the flip probability
    pub fn probability(&self) -> f32 {
        self.p
    }
}

impl Transform for RandomHorizontalFlip {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        if rng.gen::<f32>() < self.p {
            crate::ops::horizontal_flip(input)
        } else {
            Ok(input.clone())
        }
    }

    fn name(&self) -> &'static str {
        "RandomHorizontalFlip"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("probability", format!("{:.2}", self.p))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(RandomHorizontalFlip::new(self.p))
    }
}

/// Random vertical flip transform
///
/// Vertically flips the input image with a given probability. Less commonly used
/// than horizontal flips, but useful for certain types of data.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandomVerticalFlip, Transform};
///
/// // Flip with 25% probability
/// let flip = RandomVerticalFlip::new(0.25);
/// // Apply to tensor: result = flip.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomVerticalFlip {
    p: f32,
}

impl RandomVerticalFlip {
    /// Create a new RandomVerticalFlip transform
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of applying the flip (0.0 to 1.0)
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the range [0.0, 1.0]
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Probability must be between 0.0 and 1.0"
        );
        Self { p }
    }

    /// Get the flip probability
    pub fn probability(&self) -> f32 {
        self.p
    }
}

impl Transform for RandomVerticalFlip {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        if rng.gen::<f32>() < self.p {
            crate::ops::vertical_flip(input)
        } else {
            Ok(input.clone())
        }
    }

    fn name(&self) -> &'static str {
        "RandomVerticalFlip"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("probability", format!("{:.2}", self.p))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(RandomVerticalFlip::new(self.p))
    }
}

/// Random crop transform
///
/// Randomly crops the input image to a specified size. The crop location is
/// chosen uniformly at random from valid positions.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandomCrop, Transform};
///
/// let crop = RandomCrop::new((224, 224));
/// // Apply to tensor: result = crop.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomCrop {
    size: (usize, usize),
}

impl RandomCrop {
    /// Create a new RandomCrop transform
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

impl Transform for RandomCrop {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::random_crop(input, self.size)
    }

    fn name(&self) -> &'static str {
        "RandomCrop"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("size", format!("({}, {})", self.size.0, self.size.1))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(RandomCrop::new(self.size))
    }
}

/// Random resized crop transform
///
/// This transform randomly selects a rectangular region from the input image,
/// then resizes it to the target size. Commonly used in ImageNet training.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandomResizedCrop, Transform};
///
/// let crop = RandomResizedCrop::new((224, 224))
///     .with_scale((0.08, 1.0))
///     .with_ratio((0.75, 1.33));
/// // Apply to tensor: result = crop.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomResizedCrop {
    size: (usize, usize),
    scale: (f32, f32),
    ratio: (f32, f32),
}

impl RandomResizedCrop {
    /// Create a new RandomResizedCrop transform with default parameters
    ///
    /// # Arguments
    ///
    /// * `size` - Target output size as (width, height)
    ///
    /// Default scale range: (0.08, 1.0)
    /// Default aspect ratio range: (3/4, 4/3)
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            size,
            scale: (0.08, 1.0),
            ratio: (3.0 / 4.0, 4.0 / 3.0),
        }
    }

    /// Set the scale range for random area sampling
    ///
    /// # Arguments
    ///
    /// * `scale` - Range of area to sample relative to input area (min, max)
    pub fn with_scale(mut self, scale: (f32, f32)) -> Self {
        assert!(scale.0 <= scale.1, "Scale min must be <= scale max");
        assert!(scale.0 > 0.0, "Scale min must be > 0");
        assert!(scale.1 <= 1.0, "Scale max must be <= 1.0");
        self.scale = scale;
        self
    }

    /// Set the aspect ratio range for random sampling
    ///
    /// # Arguments
    ///
    /// * `ratio` - Range of aspect ratios to sample (min, max)
    pub fn with_ratio(mut self, ratio: (f32, f32)) -> Self {
        assert!(ratio.0 <= ratio.1, "Ratio min must be <= ratio max");
        assert!(ratio.0 > 0.0, "Ratio min must be > 0");
        self.ratio = ratio;
        self
    }

    /// Get the target size
    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    /// Get the scale range
    pub fn scale(&self) -> (f32, f32) {
        self.scale
    }

    /// Get the ratio range
    pub fn ratio(&self) -> (f32, f32) {
        self.ratio
    }
}

impl Transform for RandomResizedCrop {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (_, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);
        let area = (height * width) as f32;

        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        // Sample target area and aspect ratio
        let target_area = area * rng.gen_range(self.scale.0..=self.scale.1);
        let aspect_ratio = rng.gen_range(self.ratio.0..=self.ratio.1);

        // Calculate crop dimensions
        let crop_height = (target_area / aspect_ratio).sqrt() as usize;
        let crop_width = (target_area * aspect_ratio).sqrt() as usize;

        // Ensure crop fits within image
        let crop_height = crop_height.min(height);
        let crop_width = crop_width.min(width);

        // Random crop then resize
        let cropped = crate::ops::random_crop(input, (crop_width, crop_height))?;
        crate::ops::resize(&cropped, self.size)
    }

    fn name(&self) -> &'static str {
        "RandomResizedCrop"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("size", format!("({}, {})", self.size.0, self.size.1)),
            (
                "scale",
                format!("({:.2}, {:.2})", self.scale.0, self.scale.1),
            ),
            (
                "ratio",
                format!("({:.2}, {:.2})", self.ratio.0, self.ratio.1),
            ),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(
            RandomResizedCrop::new(self.size)
                .with_scale(self.scale)
                .with_ratio(self.ratio),
        )
    }
}

/// Random rotation transform
///
/// Rotates the input image by a random angle sampled from the given range.
/// Useful for making models robust to orientation variations.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandomRotation, Transform};
///
/// // Rotate by random angle between -15 and +15 degrees
/// let rotation = RandomRotation::new((-15.0, 15.0));
/// // Apply to tensor: result = rotation.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomRotation {
    degrees: (f32, f32),
}

impl RandomRotation {
    /// Create a new RandomRotation transform
    ///
    /// # Arguments
    ///
    /// * `degrees` - Range of rotation angles in degrees (min, max)
    pub fn new(degrees: (f32, f32)) -> Self {
        assert!(
            degrees.0 <= degrees.1,
            "Minimum degree must be <= maximum degree"
        );
        Self { degrees }
    }

    /// Create a symmetric rotation range around 0
    ///
    /// # Arguments
    ///
    /// * `max_degrees` - Maximum rotation in either direction
    pub fn symmetric(max_degrees: f32) -> Self {
        Self::new((-max_degrees, max_degrees))
    }

    /// Get the rotation range
    pub fn degrees(&self) -> (f32, f32) {
        self.degrees
    }
}

impl Transform for RandomRotation {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        let angle = rng.gen_range(self.degrees.0..=self.degrees.1);
        crate::ops::rotate(input, angle)
    }

    fn name(&self) -> &'static str {
        "RandomRotation"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![(
            "degrees",
            format!("({:.1}°, {:.1}°)", self.degrees.0, self.degrees.1),
        )]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(RandomRotation::new(self.degrees))
    }
}

/// Fixed rotation transform (non-random)
///
/// Rotates the input image by a fixed angle. Useful for specific orientation
/// corrections or as part of test-time augmentation.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Rotation, Transform};
///
/// // Rotate by exactly 90 degrees
/// let rotation = Rotation::new(90.0);
/// // Apply to tensor: result = rotation.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct Rotation {
    angle: f32, // angle in degrees
}

impl Rotation {
    /// Create a new Rotation transform
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in degrees
    pub fn new(angle: f32) -> Self {
        Self { angle }
    }

    /// Get the rotation angle
    pub fn angle(&self) -> f32 {
        self.angle
    }
}

impl Transform for Rotation {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        crate::ops::rotate(input, self.angle)
    }

    fn name(&self) -> &'static str {
        "Rotation"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("angle", format!("{:.1}°", self.angle))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Rotation::new(self.angle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_random_horizontal_flip_creation() {
        let flip = RandomHorizontalFlip::new(0.7);
        assert_eq!(flip.probability(), 0.7);
        assert_eq!(flip.name(), "RandomHorizontalFlip");

        let params = flip.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "probability");
        assert_eq!(params[0].1, "0.70");
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_random_horizontal_flip_invalid_probability() {
        RandomHorizontalFlip::new(1.5);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_random_horizontal_flip_negative_probability() {
        RandomHorizontalFlip::new(-0.1);
    }

    #[test]
    fn test_random_vertical_flip_creation() {
        let flip = RandomVerticalFlip::new(0.3);
        assert_eq!(flip.probability(), 0.3);
        assert_eq!(flip.name(), "RandomVerticalFlip");

        let params = flip.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "probability");
        assert_eq!(params[0].1, "0.30");
    }

    #[test]
    fn test_random_crop_creation() {
        let crop = RandomCrop::new((128, 96));
        assert_eq!(crop.size(), (128, 96));
        assert_eq!(crop.name(), "RandomCrop");

        let params = crop.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "size");
        assert_eq!(params[0].1, "(128, 96)");
    }

    #[test]
    fn test_random_resized_crop_creation() {
        let crop = RandomResizedCrop::new((224, 224));
        assert_eq!(crop.size(), (224, 224));
        assert_eq!(crop.scale(), (0.08, 1.0));
        assert_eq!(crop.ratio(), (0.75, 1.3333333333333333));
        assert_eq!(crop.name(), "RandomResizedCrop");
    }

    #[test]
    fn test_random_resized_crop_with_scale() {
        let crop = RandomResizedCrop::new((224, 224)).with_scale((0.2, 0.8));
        assert_eq!(crop.scale(), (0.2, 0.8));
    }

    #[test]
    fn test_random_resized_crop_with_ratio() {
        let crop = RandomResizedCrop::new((224, 224)).with_ratio((0.5, 2.0));
        assert_eq!(crop.ratio(), (0.5, 2.0));
    }

    #[test]
    #[should_panic(expected = "Scale min must be <= scale max")]
    fn test_random_resized_crop_invalid_scale() {
        RandomResizedCrop::new((224, 224)).with_scale((0.8, 0.2));
    }

    #[test]
    #[should_panic(expected = "Scale min must be > 0")]
    fn test_random_resized_crop_zero_scale() {
        RandomResizedCrop::new((224, 224)).with_scale((0.0, 0.5));
    }

    #[test]
    #[should_panic(expected = "Scale max must be <= 1.0")]
    fn test_random_resized_crop_scale_too_large() {
        RandomResizedCrop::new((224, 224)).with_scale((0.5, 1.5));
    }

    #[test]
    fn test_random_resized_crop_parameters() {
        let crop = RandomResizedCrop::new((224, 224))
            .with_scale((0.1, 0.9))
            .with_ratio((0.8, 1.2));

        let params = crop.parameters();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].0, "size");
        assert_eq!(params[1].0, "scale");
        assert_eq!(params[2].0, "ratio");
    }

    #[test]
    fn test_random_rotation_creation() {
        let rotation = RandomRotation::new((-30.0, 30.0));
        assert_eq!(rotation.degrees(), (-30.0, 30.0));
        assert_eq!(rotation.name(), "RandomRotation");

        let params = rotation.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "degrees");
        assert_eq!(params[0].1, "(-30.0°, 30.0°)");
    }

    #[test]
    fn test_random_rotation_symmetric() {
        let rotation = RandomRotation::symmetric(45.0);
        assert_eq!(rotation.degrees(), (-45.0, 45.0));
    }

    #[test]
    #[should_panic(expected = "Minimum degree must be <= maximum degree")]
    fn test_random_rotation_invalid_range() {
        RandomRotation::new((30.0, -30.0));
    }

    #[test]
    fn test_rotation_creation() {
        let rotation = Rotation::new(90.0);
        assert_eq!(rotation.angle(), 90.0);
        assert_eq!(rotation.name(), "Rotation");

        let params = rotation.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "angle");
        assert_eq!(params[0].1, "90.0°");
    }

    #[test]
    fn test_clone_transforms() {
        let flip = RandomHorizontalFlip::new(0.5);
        let cloned = flip.clone_transform();
        assert_eq!(cloned.name(), "RandomHorizontalFlip");

        let crop = RandomCrop::new((100, 100));
        let cloned = crop.clone_transform();
        assert_eq!(cloned.name(), "RandomCrop");

        let rotation = Rotation::new(45.0);
        let cloned = rotation.clone_transform();
        assert_eq!(cloned.name(), "Rotation");
    }

    #[test]
    fn test_edge_case_probabilities() {
        // Test edge cases for probabilities
        let always_flip = RandomHorizontalFlip::new(1.0);
        assert_eq!(always_flip.probability(), 1.0);

        let never_flip = RandomVerticalFlip::new(0.0);
        assert_eq!(never_flip.probability(), 0.0);
    }

    #[test]
    fn test_zero_rotation() {
        let no_rotation = Rotation::new(0.0);
        assert_eq!(no_rotation.angle(), 0.0);

        let zero_range = RandomRotation::new((0.0, 0.0));
        assert_eq!(zero_range.degrees(), (0.0, 0.0));
    }
}
