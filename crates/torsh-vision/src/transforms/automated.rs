use super::augmentation::{ColorJitter, Cutout, RandomErasing};
use super::core::Transform;
use super::random::{RandomHorizontalFlip, RandomRotation};
use crate::{Result, VisionError};
//  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{Random, Rng};
use torsh_tensor::Tensor;

/// AutoAugment transform
///
/// AutoAugment implements a simplified version of the AutoAugment policy that
/// automatically selects and applies augmentation strategies. It uses pre-defined
/// policies that were found to work well on common datasets.
///
/// Based on "AutoAugment: Learning Augmentation Policies from Data" by Cubuk et al. (2019).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{AutoAugment, Transform};
///
/// let auto_aug = AutoAugment::new();
/// // Apply to tensor: result = auto_aug.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct AutoAugment {
    policies: Vec<Vec<(String, f32)>>, // (transform_name, probability)
}

impl AutoAugment {
    /// Create a new AutoAugment transform with default policies
    pub fn new() -> Self {
        // Simplified AutoAugment policies
        let policies = vec![
            // Policy 1: Rotate + Color transformations
            vec![
                ("rotate".to_string(), 0.7),
                ("color_jitter".to_string(), 0.8),
            ],
            // Policy 2: Flip + Cutout
            vec![
                ("flip_horizontal".to_string(), 0.8),
                ("cutout".to_string(), 0.6),
            ],
            // Policy 3: Erasing + Brightness
            vec![
                ("random_erasing".to_string(), 0.6),
                ("color_jitter".to_string(), 0.7),
            ],
        ];

        Self { policies }
    }

    /// Create AutoAugment with custom policies
    ///
    /// # Arguments
    ///
    /// * `policies` - Vector of policies, where each policy is a vector of (transform_name, probability) pairs
    pub fn with_policies(policies: Vec<Vec<(String, f32)>>) -> Self {
        assert!(!policies.is_empty(), "Policies cannot be empty");
        for policy in &policies {
            assert!(
                !policy.is_empty(),
                "Each policy must have at least one transform"
            );
            for (_, prob) in policy {
                assert!(
                    (0.0..=1.0).contains(prob),
                    "Probabilities must be between 0.0 and 1.0"
                );
            }
        }
        Self { policies }
    }

    /// Get the number of policies
    pub fn num_policies(&self) -> usize {
        self.policies.len()
    }

    /// Get a reference to the policies
    pub fn policies(&self) -> &[Vec<(String, f32)>] {
        &self.policies
    }

    /// Apply a specific policy by index
    pub fn apply_policy(&self, input: &Tensor<f32>, policy_idx: usize) -> Result<Tensor<f32>> {
        if policy_idx >= self.policies.len() {
            return Err(VisionError::InvalidArgument(format!(
                "Policy index {} out of range (max: {})",
                policy_idx,
                self.policies.len() - 1
            )));
        }

        let policy = &self.policies[policy_idx];
        let mut output = input.clone();
        //  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        // Apply transforms in the selected policy
        for (transform_name, probability) in policy {
            if rng.gen::<f32>() < *probability {
                output = match transform_name.as_str() {
                    "rotate" => {
                        let rotation = RandomRotation::new((-30.0, 30.0));
                        rotation.forward(&output)?
                    }
                    "color_jitter" => {
                        let jitter = ColorJitter::new().brightness(0.2).contrast(0.2);
                        jitter.forward(&output)?
                    }
                    "flip_horizontal" => {
                        let flip = RandomHorizontalFlip::new(1.0);
                        flip.forward(&output)?
                    }
                    "cutout" => {
                        let cutout = Cutout::new(16, 1);
                        cutout.forward(&output)?
                    }
                    "random_erasing" => {
                        let erasing = RandomErasing::new(0.5);
                        erasing.forward(&output)?
                    }
                    _ => output, // Unknown transform, skip
                };
            }
        }

        Ok(output)
    }
}

impl Default for AutoAugment {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for AutoAugment {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        //  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);

        // Randomly select a policy
        let policy_idx = rng.gen_range(0..self.policies.len());
        self.apply_policy(input, policy_idx)
    }

    fn name(&self) -> &'static str {
        "AutoAugment"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("num_policies", format!("{}", self.policies.len()))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(AutoAugment::with_policies(self.policies.clone()))
    }
}

/// RandAugment transform
///
/// RandAugment simplifies the search space by fixing the number of transformations
/// and only searching for the magnitude. It randomly selects N transformations
/// from a fixed set and applies them with a uniform magnitude.
///
/// Based on "RandAugment: Practical automated data augmentation with a reduced
/// search space" by Cubuk et al. (2020).
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{RandAugment, Transform};
///
/// // Apply 2 random transforms with magnitude 5
/// let rand_aug = RandAugment::new(2, 5.0);
/// // Apply to tensor: result = rand_aug.forward(&input_tensor)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandAugment {
    n: usize,       // Number of transformations to apply
    magnitude: f32, // Magnitude of transformations (0-10)
    available_transforms: Vec<String>,
}

impl RandAugment {
    /// Create a new RandAugment transform
    ///
    /// # Arguments
    ///
    /// * `n` - Number of transformations to apply (typically 1-3)
    /// * `magnitude` - Magnitude of transformations (0.0-10.0)
    pub fn new(n: usize, magnitude: f32) -> Self {
        assert!(n > 0, "Number of transformations must be positive");

        // Clamp magnitude to valid range instead of panicking
        let magnitude = magnitude.clamp(0.0, 10.0);

        let available_transforms = vec![
            "rotate".to_string(),
            "color_jitter".to_string(),
            "random_erasing".to_string(),
            "flip_horizontal".to_string(),
            "cutout".to_string(),
        ];

        Self {
            n,
            magnitude: magnitude.clamp(0.0, 10.0),
            available_transforms,
        }
    }

    /// Create RandAugment with custom transform set
    ///
    /// # Arguments
    ///
    /// * `n` - Number of transformations to apply
    /// * `magnitude` - Magnitude of transformations
    /// * `transforms` - Available transforms to choose from
    pub fn with_transforms(n: usize, magnitude: f32, transforms: Vec<String>) -> Self {
        assert!(n > 0, "Number of transformations must be positive");
        assert!(!transforms.is_empty(), "Transform list cannot be empty");

        // Clamp magnitude to valid range instead of panicking
        let magnitude = magnitude.clamp(0.0, 10.0);

        Self {
            n,
            magnitude,
            available_transforms: transforms,
        }
    }

    /// Get the number of transformations
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the magnitude
    pub fn magnitude(&self) -> f32 {
        self.magnitude
    }

    /// Get the available transforms
    pub fn available_transforms(&self) -> &[String] {
        &self.available_transforms
    }

    /// Apply specific transforms by names
    pub fn apply_transforms(
        &self,
        input: &Tensor<f32>,
        transform_names: &[String],
    ) -> Result<Tensor<f32>> {
        if transform_names.len() > self.available_transforms.len() {
            return Err(VisionError::InvalidArgument(
                "Too many transforms requested".to_string(),
            ));
        }

        let mut output = input.clone();

        for transform_name in transform_names {
            if !self.available_transforms.contains(transform_name) {
                return Err(VisionError::InvalidArgument(format!(
                    "Unknown transform: {}",
                    transform_name
                )));
            }

            output = match transform_name.as_str() {
                "rotate" => {
                    let rotation =
                        RandomRotation::new((-self.magnitude * 3.0, self.magnitude * 3.0));
                    rotation.forward(&output)?
                }
                "color_jitter" => {
                    let jitter = ColorJitter::new()
                        .brightness(self.magnitude * 0.05)
                        .contrast(self.magnitude * 0.05);
                    jitter.forward(&output)?
                }
                "random_erasing" => {
                    let erasing = RandomErasing::new(self.magnitude * 0.1);
                    erasing.forward(&output)?
                }
                "flip_horizontal" => {
                    let flip = RandomHorizontalFlip::new(0.5);
                    flip.forward(&output)?
                }
                "cutout" => {
                    let cutout = Cutout::new((self.magnitude * 2.0) as usize + 8, 1);
                    cutout.forward(&output)?
                }
                _ => output, // Unknown transform, skip
            };
        }

        Ok(output)
    }
}

impl Transform for RandAugment {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        //  SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        let mut rng = Random::seed(42);
        let mut output = input.clone();

        // Apply n random transformations
        for _ in 0..self.n {
            let transform_idx = rng.gen_range(0..self.available_transforms.len());
            let transform_name = &self.available_transforms[transform_idx];

            output = match transform_name.as_str() {
                "rotate" => {
                    let rotation =
                        RandomRotation::new((-self.magnitude * 3.0, self.magnitude * 3.0));
                    rotation.forward(&output)?
                }
                "color_jitter" => {
                    let jitter = ColorJitter::new()
                        .brightness(self.magnitude * 0.05)
                        .contrast(self.magnitude * 0.05);
                    jitter.forward(&output)?
                }
                "random_erasing" => {
                    let erasing = RandomErasing::new(self.magnitude * 0.1);
                    erasing.forward(&output)?
                }
                "flip_horizontal" => {
                    let flip = RandomHorizontalFlip::new(0.5);
                    flip.forward(&output)?
                }
                "cutout" => {
                    let cutout = Cutout::new((self.magnitude * 2.0) as usize + 8, 1);
                    cutout.forward(&output)?
                }
                _ => output, // Unknown transform, skip
            };
        }

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "RandAugment"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("n", format!("{}", self.n)),
            ("magnitude", format!("{:.1}", self.magnitude)),
        ]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(RandAugment::with_transforms(
            self.n,
            self.magnitude,
            self.available_transforms.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_auto_augment_creation() {
        let auto_aug = AutoAugment::new();
        assert_eq!(auto_aug.num_policies(), 3);
        assert_eq!(auto_aug.name(), "AutoAugment");

        let params = auto_aug.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "num_policies");
        assert_eq!(params[0].1, "3");
    }

    #[test]
    fn test_auto_augment_default() {
        let auto_aug = AutoAugment::default();
        assert_eq!(auto_aug.num_policies(), 3);
    }

    #[test]
    fn test_auto_augment_with_policies() {
        let custom_policies = vec![
            vec![("rotate".to_string(), 0.5)],
            vec![
                ("flip_horizontal".to_string(), 0.8),
                ("cutout".to_string(), 0.3),
            ],
        ];

        let auto_aug = AutoAugment::with_policies(custom_policies.clone());
        assert_eq!(auto_aug.num_policies(), 2);
        assert_eq!(auto_aug.policies(), &custom_policies);
    }

    #[test]
    #[should_panic(expected = "Policies cannot be empty")]
    fn test_auto_augment_empty_policies() {
        AutoAugment::with_policies(vec![]);
    }

    #[test]
    #[should_panic(expected = "Each policy must have at least one transform")]
    fn test_auto_augment_empty_policy() {
        AutoAugment::with_policies(vec![vec![]]);
    }

    #[test]
    #[should_panic(expected = "Probabilities must be between 0.0 and 1.0")]
    fn test_auto_augment_invalid_probability() {
        AutoAugment::with_policies(vec![vec![("rotate".to_string(), 1.5)]]);
    }

    #[test]
    fn test_auto_augment_apply_policy() {
        let auto_aug = AutoAugment::new();
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let result = auto_aug.apply_policy(&input, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_auto_augment_apply_policy_invalid_index() {
        let auto_aug = AutoAugment::new();
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let result = auto_aug.apply_policy(&input, 10);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_auto_augment_forward() {
        let auto_aug = AutoAugment::new();
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let result = auto_aug.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_rand_augment_creation() {
        let rand_aug = RandAugment::new(2, 5.0);
        assert_eq!(rand_aug.n(), 2);
        assert_eq!(rand_aug.magnitude(), 5.0);
        assert_eq!(rand_aug.name(), "RandAugment");
        assert_eq!(rand_aug.available_transforms().len(), 5);

        let params = rand_aug.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "n");
        assert_eq!(params[0].1, "2");
        assert_eq!(params[1].0, "magnitude");
        assert_eq!(params[1].1, "5.0");
    }

    #[test]
    #[should_panic(expected = "Number of transformations must be positive")]
    fn test_rand_augment_zero_n() {
        RandAugment::new(0, 5.0);
    }

    #[test]
    fn test_rand_augment_invalid_magnitude() {
        // Test that invalid magnitudes are clamped instead of panicking
        let rand_aug = RandAugment::new(2, 15.0);
        assert_eq!(rand_aug.magnitude(), 10.0); // Should be clamped to 10.0
    }

    #[test]
    fn test_rand_augment_magnitude_clamping() {
        let rand_aug = RandAugment::new(1, 12.0);
        assert_eq!(rand_aug.magnitude(), 10.0); // Should be clamped to 10.0
    }

    #[test]
    fn test_rand_augment_with_transforms() {
        let custom_transforms = vec!["rotate".to_string(), "flip_horizontal".to_string()];
        let rand_aug = RandAugment::with_transforms(1, 3.0, custom_transforms.clone());

        assert_eq!(rand_aug.n(), 1);
        assert_eq!(rand_aug.magnitude(), 3.0);
        assert_eq!(rand_aug.available_transforms(), &custom_transforms);
    }

    #[test]
    #[should_panic(expected = "Transform list cannot be empty")]
    fn test_rand_augment_empty_transforms() {
        RandAugment::with_transforms(1, 5.0, vec![]);
    }

    #[test]
    fn test_rand_augment_apply_transforms() {
        let rand_aug = RandAugment::new(2, 5.0);
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let transforms = vec!["rotate".to_string(), "flip_horizontal".to_string()];
        let result = rand_aug.apply_transforms(&input, &transforms);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_rand_augment_apply_transforms_unknown() {
        let rand_aug = RandAugment::new(1, 5.0);
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let transforms = vec!["unknown_transform".to_string()];
        let result = rand_aug.apply_transforms(&input, &transforms);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_rand_augment_apply_transforms_too_many() {
        let rand_aug = RandAugment::with_transforms(1, 5.0, vec!["rotate".to_string()]);
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let transforms = vec!["rotate".to_string(), "flip_horizontal".to_string()];
        let result = rand_aug.apply_transforms(&input, &transforms);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VisionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_rand_augment_forward() {
        let rand_aug = RandAugment::new(2, 3.0);
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let result = rand_aug.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[3, 32, 32]);
    }

    #[test]
    fn test_clone_transforms() {
        let auto_aug = AutoAugment::new();
        let cloned = auto_aug.clone_transform();
        assert_eq!(cloned.name(), "AutoAugment");

        let rand_aug = RandAugment::new(1, 2.0);
        let cloned = rand_aug.clone_transform();
        assert_eq!(cloned.name(), "RandAugment");
    }

    #[test]
    fn test_edge_cases() {
        // Test minimum valid values
        let rand_aug = RandAugment::new(1, 0.0);
        assert_eq!(rand_aug.n(), 1);
        assert_eq!(rand_aug.magnitude(), 0.0);

        let input = creation::ones(&[3, 8, 8]).unwrap();
        let result = rand_aug.forward(&input);
        assert!(result.is_ok());

        // Test single transform
        let single_transform = vec!["rotate".to_string()];
        let rand_aug_single = RandAugment::with_transforms(1, 5.0, single_transform);
        assert_eq!(rand_aug_single.available_transforms().len(), 1);
    }

    #[test]
    fn test_deterministic_application() {
        let rand_aug = RandAugment::new(1, 5.0);
        let input = creation::ones(&[3, 16, 16]).unwrap();

        // Apply specific transforms
        let transforms = vec!["rotate".to_string()];
        let result1 = rand_aug.apply_transforms(&input, &transforms);
        let result2 = rand_aug.apply_transforms(&input, &transforms);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // Both results should have the same shape
        assert_eq!(
            result1.unwrap().shape().dims(),
            result2.unwrap().shape().dims()
        );
    }
}
