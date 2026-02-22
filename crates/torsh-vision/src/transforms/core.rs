use crate::{Result, VisionError};
use torsh_tensor::Tensor;

/// Base trait for transforms with enhanced API
pub trait Transform: Send + Sync {
    /// Apply the transform to an input tensor
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>>;

    /// Get the name of this transform (for debugging and logging)
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Check if this transform modifies the input in-place (for optimization)
    fn is_inplace(&self) -> bool {
        false
    }

    /// Get transform parameters for introspection
    fn parameters(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// Clone the transform (using dynamic dispatch)
    fn clone_transform(&self) -> Box<dyn Transform>;
}

/// Compose multiple transforms into a single pipeline
///
/// Compose allows chaining multiple transforms together, applying them sequentially
/// to create complex transformation pipelines for data augmentation and preprocessing.
///
/// # Examples
///
/// ```rust
/// use torsh_vision::transforms::{Compose, Resize, Normalize};
///
/// let transforms = vec![
///     Box::new(Resize::new((224, 224))) as Box<dyn Transform>,
///     Box::new(Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])),
/// ];
/// let pipeline = Compose::new(transforms);
/// ```
pub struct Compose {
    pub(crate) transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    /// Create a new Compose transform with the given list of transforms
    ///
    /// # Arguments
    ///
    /// * `transforms` - Vector of boxed transforms to apply in sequence
    ///
    /// # Returns
    ///
    /// A new Compose instance that will apply all transforms in order
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    /// Get the number of transforms in this composition
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the composition is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Add a transform to the end of the pipeline
    pub fn add<T: Transform + 'static>(&mut self, transform: T) {
        self.transforms.push(Box::new(transform));
    }

    /// Get a reference to the transforms
    pub fn transforms(&self) -> &[Box<dyn Transform>] {
        &self.transforms
    }
}

impl Transform for Compose {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut output = input.clone();
        for transform in &self.transforms {
            output = transform.forward(&output)?;
        }
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "Compose"
    }

    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![("num_transforms", format!("{}", self.transforms.len()))]
    }

    fn clone_transform(&self) -> Box<dyn Transform> {
        Box::new(Compose::new(
            self.transforms
                .iter()
                .map(|t| t.clone_transform())
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    // Mock transform for testing
    struct MockTransform {
        name: &'static str,
        multiplier: f32,
    }

    impl MockTransform {
        fn new(name: &'static str, multiplier: f32) -> Self {
            Self { name, multiplier }
        }
    }

    impl Transform for MockTransform {
        fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
            input
                .mul_scalar(self.multiplier)
                .map_err(|e| VisionError::TensorError(e))
        }

        fn name(&self) -> &'static str {
            self.name
        }

        fn parameters(&self) -> Vec<(&'static str, String)> {
            vec![("multiplier", format!("{:.2}", self.multiplier))]
        }

        fn clone_transform(&self) -> Box<dyn Transform> {
            Box::new(MockTransform::new(self.name, self.multiplier))
        }
    }

    #[test]
    fn test_transform_trait() {
        let transform = MockTransform::new("TestTransform", 2.0);
        let input = creation::ones(&[3, 32, 32]).unwrap();

        let result = transform.forward(&input).unwrap();
        assert_eq!(result.get(&[0, 0, 0]).unwrap(), 2.0);

        assert_eq!(transform.name(), "TestTransform");
        assert_eq!(transform.is_inplace(), false);

        let params = transform.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "multiplier");
        assert_eq!(params[0].1, "2.00");
    }

    #[test]
    fn test_compose_empty() {
        let compose = Compose::new(vec![]);
        assert!(compose.is_empty());
        assert_eq!(compose.len(), 0);

        let input = creation::ones(&[3, 32, 32]).unwrap();
        let result = compose.forward(&input).unwrap();

        // Empty compose should return input unchanged
        assert_eq!(result.get(&[0, 0, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_compose_single_transform() {
        let transforms = vec![Box::new(MockTransform::new("Double", 2.0)) as Box<dyn Transform>];
        let compose = Compose::new(transforms);

        assert_eq!(compose.len(), 1);
        assert!(!compose.is_empty());

        let input = creation::ones(&[3, 32, 32]).unwrap();
        let result = compose.forward(&input).unwrap();

        assert_eq!(result.get(&[0, 0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_compose_multiple_transforms() {
        let transforms = vec![
            Box::new(MockTransform::new("Double", 2.0)) as Box<dyn Transform>,
            Box::new(MockTransform::new("Triple", 3.0)) as Box<dyn Transform>,
        ];
        let compose = Compose::new(transforms);

        assert_eq!(compose.len(), 2);

        let input = creation::ones(&[3, 32, 32]).unwrap();
        let result = compose.forward(&input).unwrap();

        // Should apply 2.0 * 3.0 = 6.0
        assert_eq!(result.get(&[0, 0, 0]).unwrap(), 6.0);
    }

    #[test]
    fn test_compose_add_transform() {
        let mut compose = Compose::new(vec![]);
        assert_eq!(compose.len(), 0);

        compose.add(MockTransform::new("Double", 2.0));
        assert_eq!(compose.len(), 1);

        compose.add(MockTransform::new("Triple", 3.0));
        assert_eq!(compose.len(), 2);
    }

    #[test]
    fn test_compose_clone() {
        let transforms = vec![
            Box::new(MockTransform::new("Double", 2.0)) as Box<dyn Transform>,
            Box::new(MockTransform::new("Triple", 3.0)) as Box<dyn Transform>,
        ];
        let compose = Compose::new(transforms);

        let cloned = compose.clone_transform();

        let input = creation::ones(&[3, 32, 32]).unwrap();
        let original_result = compose.forward(&input).unwrap();
        let cloned_result = cloned.forward(&input).unwrap();

        assert_eq!(
            original_result.get(&[0, 0, 0]).unwrap(),
            cloned_result.get(&[0, 0, 0]).unwrap()
        );
    }

    #[test]
    fn test_compose_parameters() {
        let transforms = vec![
            Box::new(MockTransform::new("Double", 2.0)) as Box<dyn Transform>,
            Box::new(MockTransform::new("Triple", 3.0)) as Box<dyn Transform>,
        ];
        let compose = Compose::new(transforms);

        let params = compose.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "num_transforms");
        assert_eq!(params[0].1, "2");
    }
}
