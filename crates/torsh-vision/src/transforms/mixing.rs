use super::core::Transform;
use crate::{Result, VisionError};
//  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use torsh_tensor::{creation, Tensor};

/// MixUp data augmentation transform
///
/// MixUp creates virtual training examples by taking linear combinations of pairs
/// of examples and their labels. This technique improves generalization and reduces
/// memorization of corrupt labels.
///
/// Based on "mixup: Beyond Empirical Risk Minimization" by Zhang et al. (2017).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{MixUp, Transform};
/// use torsh_tensor::{creation, Tensor};
///
/// let mixup = MixUp::new(1.0);
/// let input1 = creation::ones(&[3, 32, 32]).unwrap();
/// let input2 = creation::zeros(&[3, 32, 32]).unwrap();
/// let (mixed_image, mixed_labels) = mixup.apply_pair(&input1, &input2, 0, 1, 10).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MixUp {
    alpha: f32,
}

impl MixUp {
    /// Create a new MixUp transform
    ///
    /// # Arguments
    ///
    /// * `alpha` - Parameter for Beta distribution. Higher values lead to stronger mixing.
    ///             Common values: 0.2, 1.0. Use 0.0 to disable mixing.
    pub fn new(alpha: f32) -> Self {
        assert!(alpha >= 0.0, "Alpha must be non-negative");
        Self { alpha }
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Apply MixUp to two samples
    ///
    /// # Arguments
    ///
    /// * `input1` - First input tensor
    /// * `input2` - Second input tensor
    /// * `label1` - Label for first input
    /// * `label2` - Label for second input
    /// * `num_classes` - Total number of classes for one-hot encoding
    ///
    /// # Returns
    ///
    /// A tuple containing the mixed image and mixed one-hot labels
    pub fn apply_pair(
        &self,
        input1: &Tensor<f32>,
        input2: &Tensor<f32>,
        label1: usize,
        label2: usize,
        num_classes: usize,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if label1 >= num_classes || label2 >= num_classes {
            return Err(VisionError::InvalidArgument(format!(
                "Labels ({}, {}) must be less than num_classes ({})",
                label1, label2, num_classes
            )));
        }

        //  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        // Sample lambda from uniform distribution (simplified approach)
        let lambda = if self.alpha > 0.0 {
            rng.gen_range(0.0..=1.0)
        } else {
            0.5
        };

        // Mix images: mixed_image = lambda * input1 + (1 - lambda) * input2
        let mixed_image = input1
            .mul_scalar(lambda)?
            .add(&input2.mul_scalar(1.0 - lambda)?)?;

        // Create one-hot labels and mix them
        let mut mixed_labels = creation::zeros(&[num_classes]).unwrap();
        if label1 == label2 {
            // If both labels are the same, the mixed label is 1.0
            mixed_labels.set(&[label1], 1.0)?;
        } else {
            mixed_labels.set(&[label1], lambda)?;
            mixed_labels.set(&[label2], 1.0 - lambda)?;
        }

        Ok((mixed_image, mixed_labels))
    }

    /// Apply MixUp with a specific lambda value (for reproducible results)
    ///
    /// # Arguments
    ///
    /// * `input1` - First input tensor
    /// * `input2` - Second input tensor
    /// * `label1` - Label for first input
    /// * `label2` - Label for second input
    /// * `num_classes` - Total number of classes
    /// * `lambda` - Mixing parameter (0.0 to 1.0)
    pub fn apply_pair_with_lambda(
        &self,
        input1: &Tensor<f32>,
        input2: &Tensor<f32>,
        label1: usize,
        label2: usize,
        num_classes: usize,
        lambda: f32,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "Lambda must be between 0.0 and 1.0"
        );

        if label1 >= num_classes || label2 >= num_classes {
            return Err(VisionError::InvalidArgument(format!(
                "Labels ({}, {}) must be less than num_classes ({})",
                label1, label2, num_classes
            )));
        }

        // Mix images
        let mixed_image = input1
            .mul_scalar(lambda)?
            .add(&input2.mul_scalar(1.0 - lambda)?)?;

        // Create mixed labels
        let mut mixed_labels = creation::zeros(&[num_classes]).unwrap();
        if label1 == label2 {
            // If both labels are the same, the mixed label is 1.0
            mixed_labels.set(&[label1], 1.0)?;
        } else {
            mixed_labels.set(&[label1], lambda)?;
            mixed_labels.set(&[label2], 1.0 - lambda)?;
        }

        Ok((mixed_image, mixed_labels))
    }
}

impl Transform for MixUp {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // MixUp requires two inputs, so this just returns the input unchanged
        // Use apply_pair for actual MixUp functionality
        Ok(input.clone())
    }

    fn name(&self) -> &'static str {
        "MixUp"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("alpha", format!("{:.2}", self.alpha))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(MixUp::new(self.alpha))
    }
}

/// CutMix data augmentation transform
///
/// CutMix combines two samples by cutting and pasting patches. It cuts a rectangular
/// region from one image and pastes it onto another, with labels mixed proportionally
/// to the area of the cut region.
///
/// Based on "CutMix: Regularization Strategy to Train Strong Classifiers with
/// Localizable Features" by Yun et al. (2019).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{CutMix, Transform};
/// use torsh_tensor::{creation, Tensor};
///
/// let cutmix = CutMix::new(1.0);
/// let input1 = creation::ones(&[3, 32, 32]).unwrap();
/// let input2 = creation::zeros(&[3, 32, 32]).unwrap();
/// let (mixed_image, mixed_labels) = cutmix.apply_pair(&input1, &input2, 0, 1, 10).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CutMix {
    alpha: f32,
}

impl CutMix {
    /// Create a new CutMix transform
    ///
    /// # Arguments
    ///
    /// * `alpha` - Parameter for Beta distribution. Higher values lead to larger cut regions.
    ///             Common values: 1.0. Use 0.0 to disable cutting.
    pub fn new(alpha: f32) -> Self {
        assert!(alpha >= 0.0, "Alpha must be non-negative");
        Self { alpha }
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Apply CutMix to two samples
    ///
    /// # Arguments
    ///
    /// * `input1` - First input tensor (base image)
    /// * `input2` - Second input tensor (source of cut patch)
    /// * `label1` - Label for first input
    /// * `label2` - Label for second input
    /// * `num_classes` - Total number of classes
    ///
    /// # Returns
    ///
    /// A tuple containing the mixed image and mixed one-hot labels
    pub fn apply_pair(
        &self,
        input1: &Tensor<f32>,
        input2: &Tensor<f32>,
        label1: usize,
        label2: usize,
        num_classes: usize,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if label1 >= num_classes || label2 >= num_classes {
            return Err(VisionError::InvalidArgument(format!(
                "Labels ({}, {}) must be less than num_classes ({})",
                label1, label2, num_classes
            )));
        }

        //  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        let shape = input1.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (channels, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);

        // Sample lambda (simplified approach)
        let lambda = if self.alpha > 0.0 {
            rng.gen_range(0.0..=1.0)
        } else {
            0.5
        };

        // Calculate cut ratio and dimensions
        let cut_ratio = (1.0f32 - lambda).sqrt();
        let cut_w = (width as f32 * cut_ratio) as usize;
        let cut_h = (height as f32 * cut_ratio) as usize;

        // Random position for the cut
        let cx = rng.gen_range(0..width);
        let cy = rng.gen_range(0..height);

        let x1 = (cx as i32 - cut_w as i32 / 2).max(0) as usize;
        let y1 = (cy as i32 - cut_h as i32 / 2).max(0) as usize;
        let x2 = (x1 + cut_w).min(width);
        let y2 = (y1 + cut_h).min(height);

        // Create mixed image
        let mut mixed_image = input1.clone();

        // Paste region from input2 to input1
        for c in 0..channels {
            for y in y1..y2 {
                for x in x1..x2 {
                    let pixel_val = input2.get(&[c, y, x])?;
                    mixed_image.set(&[c, y, x], pixel_val)?;
                }
            }
        }

        // Calculate actual lambda based on cut area
        let cut_area = (x2 - x1) * (y2 - y1);
        let total_area = width * height;
        let actual_lambda = 1.0 - (cut_area as f32 / total_area as f32);

        // Create mixed labels
        let mut mixed_labels = creation::zeros(&[num_classes]).unwrap();
        mixed_labels.set(&[label1], actual_lambda)?;
        mixed_labels.set(&[label2], 1.0 - actual_lambda)?;

        Ok((mixed_image, mixed_labels))
    }

    /// Apply CutMix with specific cut parameters (for reproducible results)
    ///
    /// # Arguments
    ///
    /// * `input1` - First input tensor
    /// * `input2` - Second input tensor
    /// * `label1` - Label for first input
    /// * `label2` - Label for second input
    /// * `num_classes` - Total number of classes
    /// * `x1`, `y1`, `x2`, `y2` - Cut region coordinates
    pub fn apply_pair_with_bbox(
        &self,
        input1: &Tensor<f32>,
        input2: &Tensor<f32>,
        label1: usize,
        label2: usize,
        num_classes: usize,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if label1 >= num_classes || label2 >= num_classes {
            return Err(VisionError::InvalidArgument(format!(
                "Labels ({}, {}) must be less than num_classes ({})",
                label1, label2, num_classes
            )));
        }

        let shape = input1.shape();
        if shape.dims().len() != 3 {
            return Err(VisionError::InvalidShape(format!(
                "Expected 3D tensor (C, H, W), got {}D",
                shape.dims().len()
            )));
        }

        let (channels, height, width) = (shape.dims()[0], shape.dims()[1], shape.dims()[2]);

        // Validate bounding box
        if x2 <= x1 || y2 <= y1 || x2 > width || y2 > height {
            return Err(VisionError::InvalidArgument(format!(
                "Invalid bounding box: ({}, {}, {}, {}) for image size {}x{}",
                x1, y1, x2, y2, width, height
            )));
        }

        // Create mixed image
        let mut mixed_image = input1.clone();

        // Paste region from input2 to input1
        for c in 0..channels {
            for y in y1..y2 {
                for x in x1..x2 {
                    let pixel_val = input2.get(&[c, y, x])?;
                    mixed_image.set(&[c, y, x], pixel_val)?;
                }
            }
        }

        // Calculate lambda based on cut area
        let cut_area = (x2 - x1) * (y2 - y1);
        let total_area = width * height;
        let lambda = 1.0 - (cut_area as f32 / total_area as f32);

        // Create mixed labels
        let mut mixed_labels = creation::zeros(&[num_classes]).unwrap();
        mixed_labels.set(&[label1], lambda)?;
        mixed_labels.set(&[label2], 1.0 - lambda)?;

        Ok((mixed_image, mixed_labels))
    }
}

impl Transform for CutMix {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // CutMix requires two inputs, so this just returns the input unchanged
        // Use apply_pair for actual CutMix functionality
        Ok(input.clone())
    }

    fn name(&self) -> &'static str {
        "CutMix"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("alpha", format!("{:.2}", self.alpha))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(CutMix::new(self.alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_mixup_creation() {
        let mixup = MixUp::new(1.0);
        assert_eq!(mixup.alpha(), 1.0);
        assert_eq!(mixup.name(), "MixUp");

        let params = mixup.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "alpha");
        assert_eq!(params[0].1, "1.00");
    }

    #[test]
    #[should_panic(expected = "Alpha must be non-negative")]
    fn test_mixup_negative_alpha() {
        MixUp::new(-0.1);
    }

    #[test]
    fn test_mixup_apply_pair() {
        let mixup = MixUp::new(0.0); // Disable randomness for testing
        let input1 = creation::ones(&[3, 4, 4]).unwrap();
        let input2 = creation::zeros(&[3, 4, 4]).unwrap();

        let result = mixup.apply_pair(&input1, &input2, 0, 1, 5);
        assert!(result.is_ok());

        let (mixed_image, mixed_labels) = result.unwrap();

        // Check image dimensions
        assert_eq!(mixed_image.shape().dims(), &[3, 4, 4]);

        // Check labels dimensions
        assert_eq!(mixed_labels.shape().dims(), &[5]);

        // With alpha=0, lambda should be 0.5
        assert!((mixed_labels.get(&[0]).unwrap() - 0.5).abs() < 1e-6);
        assert!((mixed_labels.get(&[1]).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mixup_apply_pair_with_lambda() {
        let mixup = MixUp::new(1.0);
        let input1 = creation::ones(&[3, 4, 4]).unwrap();
        let input2 = creation::zeros(&[3, 4, 4]).unwrap();

        let result = mixup.apply_pair_with_lambda(&input1, &input2, 0, 2, 5, 0.3);
        assert!(result.is_ok());

        let (mixed_image, mixed_labels) = result.unwrap();

        // Check exact lambda values
        assert!((mixed_labels.get(&[0]).unwrap() - 0.3).abs() < 1e-6);
        assert!((mixed_labels.get(&[2]).unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Lambda must be between 0.0 and 1.0")]
    fn test_mixup_invalid_lambda() {
        let mixup = MixUp::new(1.0);
        let input1 = creation::ones(&[3, 4, 4]).unwrap();
        let input2 = creation::zeros(&[3, 4, 4]).unwrap();

        mixup
            .apply_pair_with_lambda(&input1, &input2, 0, 1, 5, 1.5)
            .unwrap();
    }

    #[test]
    fn test_mixup_invalid_labels() {
        let mixup = MixUp::new(1.0);
        let input1 = creation::ones(&[3, 4, 4]).unwrap();
        let input2 = creation::zeros(&[3, 4, 4]).unwrap();

        let result = mixup.apply_pair(&input1, &input2, 5, 1, 5); // label1 >= num_classes
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_cutmix_creation() {
        let cutmix = CutMix::new(1.0);
        assert_eq!(cutmix.alpha(), 1.0);
        assert_eq!(cutmix.name(), "CutMix");

        let params = cutmix.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "alpha");
        assert_eq!(params[0].1, "1.00");
    }

    #[test]
    #[should_panic(expected = "Alpha must be non-negative")]
    fn test_cutmix_negative_alpha() {
        CutMix::new(-0.5);
    }

    #[test]
    fn test_cutmix_apply_pair() {
        let cutmix = CutMix::new(1.0);
        let input1 = creation::ones(&[3, 8, 8]).unwrap();
        let input2 = creation::zeros(&[3, 8, 8]).unwrap();

        let result = cutmix.apply_pair(&input1, &input2, 0, 1, 5);
        assert!(result.is_ok());

        let (mixed_image, mixed_labels) = result.unwrap();

        // Check image dimensions
        assert_eq!(mixed_image.shape().dims(), &[3, 8, 8]);

        // Check labels dimensions
        assert_eq!(mixed_labels.shape().dims(), &[5]);

        // Labels should sum to 1
        let label_sum = mixed_labels.get(&[0]).unwrap() + mixed_labels.get(&[1]).unwrap();
        assert!((label_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cutmix_apply_pair_with_bbox() {
        let cutmix = CutMix::new(1.0);
        let input1 = creation::ones(&[3, 8, 8]).unwrap();
        let input2 = creation::zeros(&[3, 8, 8]).unwrap();

        // Cut a 2x2 region (4 pixels out of 64 total)
        let result = cutmix.apply_pair_with_bbox(&input1, &input2, 0, 1, 5, 2, 2, 4, 4);
        assert!(result.is_ok());

        let (mixed_image, mixed_labels) = result.unwrap();

        // Check that cut region has been applied
        assert_eq!(mixed_image.get(&[0, 2, 2]).unwrap(), 0.0); // Should be from input2
        assert_eq!(mixed_image.get(&[0, 0, 0]).unwrap(), 1.0); // Should be from input1

        // Lambda should be 1 - (4/64) = 60/64 = 0.9375
        let expected_lambda = 1.0 - (4.0 / 64.0);
        assert!((mixed_labels.get(&[0]).unwrap() - expected_lambda).abs() < 1e-6);
        assert!((mixed_labels.get(&[1]).unwrap() - (1.0 - expected_lambda)).abs() < 1e-6);
    }

    #[test]
    fn test_cutmix_invalid_bbox() {
        let cutmix = CutMix::new(1.0);
        let input1 = creation::ones(&[3, 8, 8]).unwrap();
        let input2 = creation::zeros(&[3, 8, 8]).unwrap();

        // Invalid bbox: x2 <= x1
        let result = cutmix.apply_pair_with_bbox(&input1, &input2, 0, 1, 5, 4, 2, 4, 4);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));

        // Invalid bbox: x2 > width
        let result = cutmix.apply_pair_with_bbox(&input1, &input2, 0, 1, 5, 6, 2, 10, 4);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_cutmix_invalid_shape() {
        let cutmix = CutMix::new(1.0);
        let input1 = creation::ones(&[8, 8]).unwrap(); // 2D tensor (invalid)
        let input2 = creation::zeros(&[8, 8]).unwrap();

        let result = cutmix.apply_pair(&input1, &input2, 0, 1, 5);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VisionError::InvalidShape(_)));
    }

    #[test]
    fn test_cutmix_invalid_labels() {
        let cutmix = CutMix::new(1.0);
        let input1 = creation::ones(&[3, 4, 4]).unwrap();
        let input2 = creation::zeros(&[3, 4, 4]).unwrap();

        let result = cutmix.apply_pair(&input1, &input2, 0, 5, 5); // label2 >= num_classes
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_transforms_forward() {
        let mixup = MixUp::new(1.0);
        let cutmix = CutMix::new(1.0);
        let input = creation::ones(&[3, 8, 8]).unwrap();

        // Both transforms should return input unchanged in forward mode
        let mixup_result = mixup.forward(&input).unwrap();
        let cutmix_result = cutmix.forward(&input).unwrap();

        assert_eq!(mixup_result.get(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(cutmix_result.get(&[0, 0, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_clone_transforms() {
        let mixup = MixUp::new(0.8);
        let cloned = mixup.clone_transform();
        assert_eq!(cloned.name(), "MixUp");

        let cutmix = CutMix::new(1.2);
        let cloned = cutmix.clone_transform();
        assert_eq!(cloned.name(), "CutMix");
    }

    #[test]
    fn test_edge_cases() {
        // Test zero alpha values
        let mixup = MixUp::new(0.0);
        let cutmix = CutMix::new(0.0);
        assert_eq!(mixup.alpha(), 0.0);
        assert_eq!(cutmix.alpha(), 0.0);

        // Test minimal valid image
        let input1 = creation::ones(&[1, 1, 1]).unwrap();
        let input2 = creation::zeros(&[1, 1, 1]).unwrap();

        let mixup_result = mixup.apply_pair(&input1, &input2, 0, 1, 2);
        assert!(mixup_result.is_ok());

        let cutmix_result = cutmix.apply_pair(&input1, &input2, 0, 1, 2);
        assert!(cutmix_result.is_ok());
    }

    #[test]
    fn test_same_labels() {
        let mixup = MixUp::new(1.0);
        let input1 = creation::ones(&[3, 4, 4]).unwrap();
        let input2 = creation::zeros(&[3, 4, 4]).unwrap();

        // Test with same labels
        let result = mixup.apply_pair(&input1, &input2, 2, 2, 5);
        assert!(result.is_ok());

        let (_, mixed_labels) = result.unwrap();
        assert!((mixed_labels.get(&[2]).unwrap() - 1.0).abs() < 1e-6);

        // All other labels should be 0
        for i in 0..5 {
            if i != 2 {
                assert!((mixed_labels.get(&[i]).unwrap()).abs() < 1e-6);
            }
        }
    }
}
