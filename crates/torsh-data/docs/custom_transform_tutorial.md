# Custom Transform Tutorial

This tutorial demonstrates how to create custom data transformations in the ToRSh data loading framework.

## Table of Contents

1. [Transform Trait Overview](#transform-trait-overview)
2. [Simple Transforms](#simple-transforms)
3. [Stateful Transforms](#stateful-transforms)
4. [Transform Composition](#transform-composition)
5. [Builder Pattern Transforms](#builder-pattern-transforms)
6. [Advanced Transform Patterns](#advanced-transform-patterns)
7. [Testing Custom Transforms](#testing-custom-transforms)
8. [Performance Considerations](#performance-considerations)

## Transform Trait Overview

The core `Transform` trait defines the interface for all data transformations:

```rust
pub trait Transform<T>: Send + Sync {
    type Output;
    
    fn transform(&self, input: T) -> Result<Self::Output>;
    fn transform_batch(&self, inputs: Vec<T>) -> Result<Vec<Self::Output>>;
    fn is_deterministic(&self) -> bool;
}
```

Key characteristics:
- **Generic over input type `T`**: Works with any data type
- **Associated `Output` type**: Transformation can change the data type
- **Thread-safe**: Must implement `Send + Sync` for use in multi-threaded data loaders
- **Error handling**: Returns `Result` for proper error propagation
- **Batch processing**: Optional batch optimization
- **Determinism tracking**: Important for reproducible training

## Simple Transforms

### Using the `simple_transform!` Macro

For stateless transformations, use the `simple_transform!` macro:

```rust
use torsh_data::transforms::simple_transform;

// Basic text transformation
simple_transform!(
    UppercaseTransform,     // Transform name
    String,                 // Input type
    String,                 // Output type
    |s: String| s.to_uppercase()  // Transform function
);

// Numeric transformation
simple_transform!(
    SquareTransform,
    f32,
    f32,
    |x: f32| x * x
);

// Vector transformation
simple_transform!(
    SumTransform,
    Vec<f32>,
    f32,
    |v: Vec<f32>| v.iter().sum()
);

// Usage
let transform = UppercaseTransform;
let result = transform.transform("hello world".to_string())?;
assert_eq!(result, "HELLO WORLD");
```

### Non-Deterministic Transforms

For transforms involving randomness:

```rust
simple_transform!(
    AddNoiseTransform,
    f32,
    f32,
    |x: f32| {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        x + rng.gen_range(-0.1..0.1)
    },
    deterministic = false  // Mark as non-deterministic
);
```

### Manual Implementation

For more control, implement the trait manually:

```rust
use torsh_data::transforms::Transform;
use torsh_core::error::Result;

#[derive(Clone, Debug)]
pub struct MultiplyTransform {
    factor: f32,
}

impl MultiplyTransform {
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl Transform<f32> for MultiplyTransform {
    type Output = f32;
    
    fn transform(&self, input: f32) -> Result<Self::Output> {
        Ok(input * self.factor)
    }
    
    // Optimize batch processing
    fn transform_batch(&self, inputs: Vec<f32>) -> Result<Vec<Self::Output>> {
        Ok(inputs.into_iter().map(|x| x * self.factor).collect())
    }
    
    fn is_deterministic(&self) -> bool {
        true
    }
}
```

## Stateful Transforms

### Running Statistics Transform

```rust
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct RunningStatsTransform {
    stats: Arc<Mutex<RunningStats>>,
    normalize: bool,
}

#[derive(Debug)]
struct RunningStats {
    count: usize,
    mean: f32,
    variance: f32,
}

impl RunningStatsTransform {
    pub fn new(normalize: bool) -> Self {
        Self {
            stats: Arc::new(Mutex::new(RunningStats {
                count: 0,
                mean: 0.0,
                variance: 0.0,
            })),
            normalize,
        }
    }
    
    pub fn get_stats(&self) -> (f32, f32) {
        let stats = self.stats.lock().unwrap();
        (stats.mean, stats.variance.sqrt())
    }
}

impl Transform<f32> for RunningStatsTransform {
    type Output = f32;
    
    fn transform(&self, input: f32) -> Result<Self::Output> {
        let mut stats = self.stats.lock().unwrap();
        
        // Update running statistics
        stats.count += 1;
        let delta = input - stats.mean;
        stats.mean += delta / stats.count as f32;
        let delta2 = input - stats.mean;
        stats.variance += delta * delta2;
        
        if self.normalize && stats.count > 1 {
            let std = (stats.variance / (stats.count - 1) as f32).sqrt();
            if std > 1e-8 {
                Ok((input - stats.mean) / std)
            } else {
                Ok(input - stats.mean)
            }
        } else {
            Ok(input)
        }
    }
    
    fn is_deterministic(&self) -> bool {
        false // Output depends on previously seen data
    }
}
```

### Cache Transform

```rust
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone)]
pub struct CacheTransform<T, K> {
    inner: T,
    cache: Arc<Mutex<HashMap<K, T::Output>>>,
    key_fn: Arc<dyn Fn(&T::Input) -> K + Send + Sync>,
}

impl<T, K> CacheTransform<T, K>
where
    T: Transform<K::Input>,
    K: Hash + Eq + Clone,
{
    pub fn new<F>(inner: T, key_fn: F) -> Self
    where
        F: Fn(&T::Input) -> K + Send + Sync + 'static,
    {
        Self {
            inner,
            cache: Arc::new(Mutex::new(HashMap::new())),
            key_fn: Arc::new(key_fn),
        }
    }
}

impl<T, K> Transform<T::Input> for CacheTransform<T, K>
where
    T: Transform<T::Input>,
    T::Output: Clone,
    K: Hash + Eq + Clone,
{
    type Output = T::Output;
    
    fn transform(&self, input: T::Input) -> Result<Self::Output> {
        let key = (self.key_fn)(&input);
        
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.clone());
            }
        }
        
        // Not in cache, compute and store
        let result = self.inner.transform(input)?;
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, result.clone());
        }
        
        Ok(result)
    }
    
    fn is_deterministic(&self) -> bool {
        self.inner.is_deterministic()
    }
}
```

## Transform Composition

### Chaining Transforms

```rust
use torsh_data::transforms::{Transform, TransformExt};

// Chain transforms using the fluent API
let pipeline = MultiplyTransform::new(2.0)
    .then(AddNoiseTransform)
    .then(MultiplyTransform::new(0.5));

let result = pipeline.transform(10.0)?;
```

### Conditional Transforms

```rust
// Apply transform only to positive numbers
let conditional = MultiplyTransform::new(2.0)
    .when(|&x: &f32| x > 0.0);

assert_eq!(conditional.transform(5.0)?, 10.0);
assert_eq!(conditional.transform(-5.0)?, -5.0);
```

### Compose Multiple Transforms

```rust
use torsh_data::transforms::Compose;

let transforms: Vec<Box<dyn Transform<f32, Output = f32> + Send + Sync>> = vec![
    Box::new(MultiplyTransform::new(2.0)),
    Box::new(AddNoiseTransform),
    Box::new(MultiplyTransform::new(0.5)),
];

let composed = Compose::new(transforms);
let result = composed.transform(10.0)?;
```

## Builder Pattern Transforms

### Configurable Transform Builder

```rust
use torsh_data::transforms::TransformBuilder;

#[derive(Debug, Clone)]
pub struct ImageAugmentationBuilder {
    flip_horizontal: bool,
    flip_vertical: bool,
    rotation_range: Option<f32>,
    brightness_range: Option<(f32, f32)>,
    contrast_range: Option<(f32, f32)>,
    noise_level: Option<f32>,
}

impl Default for ImageAugmentationBuilder {
    fn default() -> Self {
        Self {
            flip_horizontal: false,
            flip_vertical: false,
            rotation_range: None,
            brightness_range: None,
            contrast_range: None,
            noise_level: None,
        }
    }
}

impl ImageAugmentationBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn flip_horizontal(mut self, enable: bool) -> Self {
        self.flip_horizontal = enable;
        self
    }
    
    pub fn flip_vertical(mut self, enable: bool) -> Self {
        self.flip_vertical = enable;
        self
    }
    
    pub fn rotation(mut self, max_degrees: f32) -> Self {
        self.rotation_range = Some(max_degrees);
        self
    }
    
    pub fn brightness(mut self, min: f32, max: f32) -> Self {
        self.brightness_range = Some((min, max));
        self
    }
    
    pub fn contrast(mut self, min: f32, max: f32) -> Self {
        self.contrast_range = Some((min, max));
        self
    }
    
    pub fn noise(mut self, level: f32) -> Self {
        self.noise_level = Some(level);
        self
    }
}

impl TransformBuilder for ImageAugmentationBuilder {
    type Transform = ImageAugmentation;
    
    fn build(self) -> Self::Transform {
        ImageAugmentation::new(self)
    }
}

#[derive(Clone)]
pub struct ImageAugmentation {
    config: ImageAugmentationBuilder,
}

impl ImageAugmentation {
    fn new(config: ImageAugmentationBuilder) -> Self {
        Self { config }
    }
}

impl Transform<Tensor<f32>> for ImageAugmentation {
    type Output = Tensor<f32>;
    
    fn transform(&self, input: Tensor<f32>) -> Result<Self::Output> {
        let mut result = input;
        
        if self.config.flip_horizontal {
            if rand::random::<f32>() < 0.5 {
                result = self.flip_horizontal(result)?;
            }
        }
        
        if self.config.flip_vertical {
            if rand::random::<f32>() < 0.5 {
                result = self.flip_vertical(result)?;
            }
        }
        
        if let Some(max_degrees) = self.config.rotation_range {
            let angle = rand::random::<f32>() * 2.0 * max_degrees - max_degrees;
            result = self.rotate(result, angle)?;
        }
        
        if let Some((min, max)) = self.config.brightness_range {
            let factor = rand::random::<f32>() * (max - min) + min;
            result = self.adjust_brightness(result, factor)?;
        }
        
        if let Some((min, max)) = self.config.contrast_range {
            let factor = rand::random::<f32>() * (max - min) + min;
            result = self.adjust_contrast(result, factor)?;
        }
        
        if let Some(level) = self.config.noise_level {
            result = self.add_noise(result, level)?;
        }
        
        Ok(result)
    }
    
    fn is_deterministic(&self) -> bool {
        false // Contains random augmentations
    }
}

impl ImageAugmentation {
    fn flip_horizontal(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // TODO: Implement horizontal flip
        Ok(input)
    }
    
    fn flip_vertical(&self, input: Tensor<f32>) -> Result<Tensor<f32>> {
        // TODO: Implement vertical flip
        Ok(input)
    }
    
    fn rotate(&self, input: Tensor<f32>, _angle: f32) -> Result<Tensor<f32>> {
        // TODO: Implement rotation
        Ok(input)
    }
    
    fn adjust_brightness(&self, input: Tensor<f32>, _factor: f32) -> Result<Tensor<f32>> {
        // TODO: Implement brightness adjustment
        Ok(input)
    }
    
    fn adjust_contrast(&self, input: Tensor<f32>, _factor: f32) -> Result<Tensor<f32>> {
        // TODO: Implement contrast adjustment
        Ok(input)
    }
    
    fn add_noise(&self, input: Tensor<f32>, _level: f32) -> Result<Tensor<f32>> {
        // TODO: Implement noise addition
        Ok(input)
    }
}

// Usage
let augmentation = ImageAugmentationBuilder::new()
    .flip_horizontal(true)
    .rotation(15.0)
    .brightness(0.8, 1.2)
    .noise(0.01)
    .build();
```

## Advanced Transform Patterns

### Multi-Input Transform

```rust
pub trait MultiInputTransform {
    type Input1;
    type Input2;
    type Output;
    
    fn transform(&self, input1: Self::Input1, input2: Self::Input2) -> Result<Self::Output>;
}

#[derive(Clone)]
pub struct CombineTransform;

impl MultiInputTransform for CombineTransform {
    type Input1 = Tensor<f32>;
    type Input2 = Tensor<f32>;
    type Output = Tensor<f32>;
    
    fn transform(&self, input1: Self::Input1, input2: Self::Input2) -> Result<Self::Output> {
        // Combine two tensors (e.g., concatenate, add, multiply)
        // For now, return the first input as placeholder
        Ok(input1)
    }
}

// Adapter to use with single-input Transform trait
pub struct MultiInputAdapter<T> {
    transform: T,
}

impl<T> Transform<(T::Input1, T::Input2)> for MultiInputAdapter<T>
where
    T: MultiInputTransform,
{
    type Output = T::Output;
    
    fn transform(&self, input: (T::Input1, T::Input2)) -> Result<Self::Output> {
        self.transform.transform(input.0, input.1)
    }
}
```

### Parametric Transform

```rust
use std::marker::PhantomData;

pub trait Parameters: Send + Sync + Clone {
    fn update(&mut self);
}

#[derive(Clone, Debug)]
pub struct GaussianNoiseParams {
    mean: f32,
    std: f32,
}

impl Parameters for GaussianNoiseParams {
    fn update(&mut self) {
        // Could implement parameter scheduling here
        // e.g., decay noise over time
    }
}

pub struct ParametricTransform<P: Parameters> {
    params: P,
    _phantom: PhantomData<P>,
}

impl<P: Parameters> ParametricTransform<P> {
    pub fn new(params: P) -> Self {
        Self {
            params,
            _phantom: PhantomData,
        }
    }
    
    pub fn update_params(&mut self) {
        self.params.update();
    }
    
    pub fn params(&self) -> &P {
        &self.params
    }
}

impl Transform<f32> for ParametricTransform<GaussianNoiseParams> {
    type Output = f32;
    
    fn transform(&self, input: f32) -> Result<Self::Output> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let noise = rng.gen::<f32>() * self.params.std + self.params.mean;
        Ok(input + noise)
    }
    
    fn is_deterministic(&self) -> bool {
        false
    }
}
```

### Transform with State Management

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct EpochAwareTransform {
    epoch: Arc<AtomicUsize>,
    base_transform: Box<dyn Transform<f32, Output = f32> + Send + Sync>,
}

impl EpochAwareTransform {
    pub fn new<T>(transform: T) -> Self
    where
        T: Transform<f32, Output = f32> + Send + Sync + 'static,
    {
        Self {
            epoch: Arc::new(AtomicUsize::new(0)),
            base_transform: Box::new(transform),
        }
    }
    
    pub fn next_epoch(&self) {
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn current_epoch(&self) -> usize {
        self.epoch.load(Ordering::Relaxed)
    }
}

impl Transform<f32> for EpochAwareTransform {
    type Output = f32;
    
    fn transform(&self, input: f32) -> Result<Self::Output> {
        let epoch = self.current_epoch();
        
        // Apply different transformations based on epoch
        let modified_input = if epoch < 10 {
            input * 1.1 // Slight augmentation early on
        } else if epoch < 50 {
            input // No augmentation in middle training
        } else {
            input * 0.9 // Reduce augmentation late in training
        };
        
        self.base_transform.transform(modified_input)
    }
    
    fn is_deterministic(&self) -> bool {
        false // Depends on epoch state
    }
}
```

## Testing Custom Transforms

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_multiply_transform() {
        let transform = MultiplyTransform::new(2.0);
        let result = transform.transform(5.0).unwrap();
        assert_relative_eq!(result, 10.0);
    }
    
    #[test]
    fn test_multiply_transform_batch() {
        let transform = MultiplyTransform::new(3.0);
        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let results = transform.transform_batch(inputs).unwrap();
        let expected = vec![3.0, 6.0, 9.0, 12.0];
        
        for (result, expected) in results.iter().zip(expected.iter()) {
            assert_relative_eq!(result, expected);
        }
    }
    
    #[test]
    fn test_transform_composition() {
        let transform = MultiplyTransform::new(2.0)
            .then(MultiplyTransform::new(3.0));
        
        let result = transform.transform(5.0).unwrap();
        assert_relative_eq!(result, 30.0);
    }
    
    #[test]
    fn test_conditional_transform() {
        let transform = MultiplyTransform::new(2.0)
            .when(|&x: &f32| x > 0.0);
        
        assert_relative_eq!(transform.transform(5.0).unwrap(), 10.0);
        assert_relative_eq!(transform.transform(-5.0).unwrap(), -5.0);
    }
    
    #[test]
    fn test_deterministic_flag() {
        let deterministic = MultiplyTransform::new(2.0);
        let non_deterministic = AddNoiseTransform;
        
        assert!(deterministic.is_deterministic());
        assert!(!non_deterministic.is_deterministic());
    }
}
```

### Property-Based Tests

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_multiply_transform_properties(
            factor in -1000.0f32..1000.0f32,
            input in -1000.0f32..1000.0f32
        ) {
            let transform = MultiplyTransform::new(factor);
            let result = transform.transform(input).unwrap();
            
            // Property: result should equal input * factor
            prop_assert!((result - input * factor).abs() < 1e-6);
            
            // Property: applying twice with factor and 1/factor should return original
            if factor.abs() > 1e-6 {
                let inverse_transform = MultiplyTransform::new(1.0 / factor);
                let roundtrip = inverse_transform.transform(result).unwrap();
                prop_assert!((roundtrip - input).abs() < 1e-5);
            }
        }
        
        #[test]
        fn test_composition_associativity(
            a in -100.0f32..100.0f32,
            b in -100.0f32..100.0f32,
            c in -100.0f32..100.0f32,
            input in -100.0f32..100.0f32
        ) {
            let t1 = MultiplyTransform::new(a);
            let t2 = MultiplyTransform::new(b);
            let t3 = MultiplyTransform::new(c);
            
            // (t1.then(t2)).then(t3) should equal t1.then(t2.then(t3))
            let left_assoc = t1.clone().then(t2.clone()).then(t3.clone());
            let right_assoc = t1.then(t2.then(t3));
            
            let result1 = left_assoc.transform(input).unwrap();
            let result2 = right_assoc.transform(input).unwrap();
            
            prop_assert!((result1 - result2).abs() < 1e-5);
        }
    }
}
```

### Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_data::{dataset::TensorDataset, dataloader::DataLoader};
    
    #[test]
    fn test_transform_with_dataloader() {
        // Create a simple dataset
        let data = torsh_tensor::creation::arange(0.0f32, 100.0, 1.0).unwrap();
        let dataset = TensorDataset::from_tensor(data);
        
        // Create a transform
        let transform = MultiplyTransform::new(2.0);
        
        // Apply transform in dataloader (conceptual - actual integration may differ)
        // This shows how transforms integrate with the broader system
        let mut results = Vec::new();
        for i in 0..dataset.len() {
            let item = dataset.get(i).unwrap();
            // In practice, transforms would be applied in the dataloader
            for tensor in item {
                let shape = tensor.shape().dims();
                if shape.len() == 1 && shape[0] == 1 {
                    // Apply transform to scalar tensors
                    let value = tensor.item::<f32>().unwrap();
                    let transformed = transform.transform(value).unwrap();
                    results.push(transformed);
                }
            }
        }
        
        // Verify first few results
        assert_relative_eq!(results[0], 0.0);
        assert_relative_eq!(results[1], 2.0);
        assert_relative_eq!(results[2], 4.0);
    }
}
```

## Performance Considerations

### Batch Processing Optimization

```rust
use rayon::prelude::*;

impl Transform<Vec<f32>> for MultiplyTransform {
    type Output = Vec<f32>;
    
    fn transform(&self, input: Vec<f32>) -> Result<Self::Output> {
        // Vectorized operation
        Ok(input.into_iter().map(|x| x * self.factor).collect())
    }
    
    fn transform_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Self::Output>> {
        // Parallel batch processing
        Ok(inputs
            .into_par_iter()
            .map(|input| input.into_iter().map(|x| x * self.factor).collect())
            .collect())
    }
}
```

### Memory-Efficient Transforms

```rust
pub struct InPlaceMultiplyTransform {
    factor: f32,
}

impl InPlaceMultiplyTransform {
    pub fn new(factor: f32) -> Self {
        Self { factor }
    }
}

impl Transform<&mut [f32]> for InPlaceMultiplyTransform {
    type Output = ();
    
    fn transform(&self, input: &mut [f32]) -> Result<Self::Output> {
        // Modify data in-place to avoid allocation
        for value in input.iter_mut() {
            *value *= self.factor;
        }
        Ok(())
    }
}
```

### SIMD Optimization

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct SIMDMultiplyTransform {
    factor: f32,
}

impl Transform<Vec<f32>> for SIMDMultiplyTransform {
    type Output = Vec<f32>;
    
    fn transform(&self, mut input: Vec<f32>) -> Result<Self::Output> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.transform_avx2(&mut input);
                }
            } else {
                // Fallback to scalar implementation
                for value in input.iter_mut() {
                    *value *= self.factor;
                }
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            for value in input.iter_mut() {
                *value *= self.factor;
            }
        }
        
        Ok(input)
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDMultiplyTransform {
    unsafe fn transform_avx2(&self, data: &mut [f32]) {
        let factor_vec = _mm256_set1_ps(self.factor);
        let chunks = data.chunks_exact_mut(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let values = _mm256_loadu_ps(chunk.as_ptr());
            let result = _mm256_mul_ps(values, factor_vec);
            _mm256_storeu_ps(chunk.as_mut_ptr(), result);
        }
        
        // Handle remainder
        for value in remainder {
            *value *= self.factor;
        }
    }
}
```

This comprehensive tutorial covers all aspects of creating custom transforms in the ToRSh framework, from simple stateless transforms to complex stateful and optimized implementations. Use these patterns as building blocks for your specific data transformation needs.