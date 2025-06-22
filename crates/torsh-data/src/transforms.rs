//! Data transformation utilities

use torsh_tensor::Tensor;
use torsh_core::{
    error::{Result, TorshError},
    dtype::TensorElement,
};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, boxed::Box};

/// Trait for data transformations
pub trait Transform<T>: Send + Sync {
    /// Output type after transformation
    type Output;
    
    /// Apply the transformation
    fn transform(&self, input: T) -> Result<Self::Output>;
}

/// Compose multiple transforms
pub struct Compose<T> {
    transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>,
}

impl<T> Compose<T> {
    /// Create a new compose transform
    pub fn new(transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>) -> Self {
        Self { transforms }
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

/// Normalize tensor values
pub struct Normalize<T: TensorElement> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T: TensorElement> Normalize<T> {
    /// Create a new normalize transform
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Result<Self> {
        if mean.len() != std.len() {
            return Err(TorshError::InvalidArgument(
                "Mean and std must have the same length".to_string()
            ));
        }
        
        Ok(Self { mean, std })
    }
}

impl<T: TensorElement> Transform<Tensor<T>> for Normalize<T> {
    type Output = Tensor<T>;
    
    fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
        // TODO: Implement actual normalization when tensor operations are complete
        // (input - mean) / std
        Ok(input)
    }
}

/// Convert tensor to a different dtype
pub struct ToType<From, To> {
    _phantom: std::marker::PhantomData<(From, To)>,
}

impl<From, To> Default for ToType<From, To> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<From, To> ToType<From, To> {
    /// Create a new type conversion transform
    pub fn new() -> Self {
        Self::default()
    }
}

impl<From: TensorElement, To: TensorElement> Transform<Tensor<From>> for ToType<From, To> {
    type Output = Tensor<To>;
    
    fn transform(&self, input: Tensor<From>) -> Result<Self::Output> {
        // TODO: Implement actual type conversion
        // For now, create a placeholder tensor
        Ok(torsh_tensor::creation::zeros::<To>(&[1]))
    }
}

/// Lambda transform for custom functions
pub struct Lambda<F> {
    func: F,
}

impl<F> Lambda<F> {
    /// Create a new lambda transform
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<T, O, F> Transform<T> for Lambda<F>
where
    F: Fn(T) -> Result<O> + Send + Sync,
{
    type Output = O;
    
    fn transform(&self, input: T) -> Result<Self::Output> {
        (self.func)(input)
    }
}

/// Common tensor transformations
pub mod tensor {
    use super::*;
    use torsh_core::dtype::FloatElement;
    
    /// Random horizontal flip
    pub struct RandomHorizontalFlip {
        prob: f32,
    }
    
    impl RandomHorizontalFlip {
        /// Create a new random horizontal flip transform
        pub fn new(prob: f32) -> Self {
            assert!((0.0..=1.0).contains(&prob), "Probability must be between 0 and 1");
            Self { prob }
        }
    }
    
    impl<T: FloatElement> Transform<Tensor<T>> for RandomHorizontalFlip {
        type Output = Tensor<T>;
        
        fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            
            if rng.gen::<f32>() < self.prob {
                // TODO: Implement actual horizontal flip
                Ok(input)
            } else {
                Ok(input)
            }
        }
    }
    
    /// Random crop
    pub struct RandomCrop {
        size: (usize, usize),
        padding: Option<usize>,
    }
    
    impl RandomCrop {
        /// Create a new random crop transform
        pub fn new(size: (usize, usize)) -> Self {
            Self {
                size,
                padding: None,
            }
        }
        
        /// Set padding
        pub fn with_padding(mut self, padding: usize) -> Self {
            self.padding = Some(padding);
            self
        }
    }
    
    impl<T: TensorElement> Transform<Tensor<T>> for RandomCrop {
        type Output = Tensor<T>;
        
        fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
            // TODO: Implement actual random crop
            Ok(input)
        }
    }
    
    /// Resize tensor
    pub struct Resize {
        size: (usize, usize),
        interpolation: InterpolationMode,
    }
    
    /// Interpolation modes for resizing
    #[derive(Clone, Copy, Debug)]
    pub enum InterpolationMode {
        Nearest,
        Linear,
        Bilinear,
        Bicubic,
    }
    
    impl Resize {
        /// Create a new resize transform
        pub fn new(size: (usize, usize)) -> Self {
            Self {
                size,
                interpolation: InterpolationMode::Bilinear,
            }
        }
        
        /// Set interpolation mode
        pub fn with_interpolation(mut self, mode: InterpolationMode) -> Self {
            self.interpolation = mode;
            self
        }
    }
    
    impl<T: FloatElement> Transform<Tensor<T>> for Resize {
        type Output = Tensor<T>;
        
        fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
            // TODO: Implement actual resize operation
            Ok(input)
        }
    }
    
    /// Center crop
    pub struct CenterCrop {
        #[allow(dead_code)]
        size: (usize, usize),
    }
    
    impl CenterCrop {
        /// Create a new center crop transform
        pub fn new(size: (usize, usize)) -> Self {
            Self { size }
        }
    }
    
    impl<T: TensorElement> Transform<Tensor<T>> for CenterCrop {
        type Output = Tensor<T>;
        
        fn transform(&self, input: Tensor<T>) -> Result<Self::Output> {
            // TODO: Implement actual center crop
            Ok(input)
        }
    }
}

/// String and text transformations
pub mod text {
    use super::*;
    
    /// Convert text to lowercase
    pub struct ToLowercase;
    
    impl Transform<String> for ToLowercase {
        type Output = String;
        
        fn transform(&self, input: String) -> Result<Self::Output> {
            Ok(input.to_lowercase())
        }
    }
    
    /// Remove punctuation
    pub struct RemovePunctuation;
    
    impl Transform<String> for RemovePunctuation {
        type Output = String;
        
        fn transform(&self, input: String) -> Result<Self::Output> {
            Ok(input.chars()
                .filter(|c| !c.is_ascii_punctuation())
                .collect())
        }
    }
    
    /// Tokenize text into words
    pub struct Tokenize {
        delimiter: String,
    }
    
    impl Tokenize {
        /// Create a new tokenize transform
        pub fn new(delimiter: String) -> Self {
            Self { delimiter }
        }
        
        /// Create with whitespace delimiter
        pub fn whitespace() -> Self {
            Self::new(" ".to_string())
        }
    }
    
    impl Transform<String> for Tokenize {
        type Output = Vec<String>;
        
        fn transform(&self, input: String) -> Result<Self::Output> {
            Ok(input.split(&self.delimiter)
                .map(|s| s.to_string())
                .collect())
        }
    }
}

/// Common transform builders
pub fn normalize<T: TensorElement>(mean: Vec<T>, std: Vec<T>) -> Result<Normalize<T>> {
    Normalize::new(mean, std)
}

pub fn to_type<From: TensorElement, To: TensorElement>() -> ToType<From, To> {
    ToType::new()
}

pub fn lambda<F>(func: F) -> Lambda<F> {
    Lambda::new(func)
}

pub fn compose<T>(transforms: Vec<Box<dyn Transform<T, Output = T> + Send + Sync>>) -> Compose<T> {
    Compose::new(transforms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;
    
    #[test]
    fn test_normalize() {
        let transform = Normalize::new(vec![0.5f32], vec![0.25f32]).unwrap();
        let input = ones::<f32>(&[3, 3]);
        let result = transform.transform(input);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_lambda() {
        let transform = Lambda::new(|x: i32| Ok(x * 2));
        let result = transform.transform(5).unwrap();
        assert_eq!(result, 10);
    }
    
    #[test]
    fn test_text_transforms() {
        let lowercase = text::ToLowercase;
        assert_eq!(
            lowercase.transform("HELLO WORLD".to_string()).unwrap(),
            "hello world"
        );
        
        let tokenize = text::Tokenize::whitespace();
        let tokens = tokenize.transform("hello world test".to_string()).unwrap();
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }
    
    #[test]
    fn test_tensor_transforms() {
        let flip = tensor::RandomHorizontalFlip::new(1.0);
        let input = ones::<f32>(&[3, 224, 224]);
        let result = flip.transform(input);
        assert!(result.is_ok());
        
        let resize = tensor::Resize::new((128, 128));
        let input = ones::<f32>(&[3, 224, 224]);
        let result = resize.transform(input);
        assert!(result.is_ok());
    }
}