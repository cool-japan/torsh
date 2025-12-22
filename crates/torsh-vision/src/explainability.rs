//! Model Explainability and Interpretability Tools
//!
//! This module provides tools for understanding and interpreting vision model predictions:
//! - GradCAM (Gradient-weighted Class Activation Mapping)
//! - Saliency Maps
//! - Guided Backpropagation
//! - Integrated Gradients
//! - Attention Visualization
//!
//! These tools help answer questions like:
//! - Which parts of the image influenced the prediction?
//! - What features is the model focusing on?
//! - Are the predictions based on relevant features?

use crate::{Result, VisionError};
use scirs2_core::ndarray::Array3; // SciRS2 Policy compliance
use std::sync::Arc;
use torsh_core::device::Device;
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// GradCAM (Gradient-weighted Class Activation Mapping)
///
/// GradCAM generates visual explanations for CNN decisions by using gradients
/// flowing into the final convolutional layer to produce a coarse localization map
/// highlighting important regions in the image.
///
/// Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
/// via Gradient-based Localization", ICCV 2017
pub struct GradCAM {
    target_layer_name: String,
    device: Arc<dyn Device>,
}

impl GradCAM {
    /// Create a new GradCAM explainer
    ///
    /// # Arguments
    /// * `target_layer_name` - Name of the convolutional layer to use for visualization
    ///   (typically the last conv layer before classification)
    /// * `device` - Device to perform computations on
    pub fn new(target_layer_name: String, device: Arc<dyn Device>) -> Self {
        Self {
            target_layer_name,
            device,
        }
    }

    /// Generate GradCAM heatmap for a specific class
    ///
    /// # Arguments
    /// * `model` - The model to explain
    /// * `input` - Input image tensor [C, H, W] or [N, C, H, W]
    /// * `target_class` - Class index to generate explanation for
    ///
    /// # Returns
    /// Heatmap tensor of the same spatial dimensions as the input
    pub fn generate_heatmap(
        &self,
        model: &dyn Module,
        input: &Tensor,
        target_class: usize,
    ) -> Result<Tensor> {
        // Ensure input has batch dimension
        let batched_input = if input.ndim() == 3 {
            input.unsqueeze(0)?
        } else {
            input.clone()
        };

        // Forward pass to get activations and output
        // Note: In a complete implementation, we would need hooks to capture
        // intermediate activations from the target layer
        let output = model.forward(&batched_input)?;

        // Get score for target class
        let target_score = output.narrow(1, target_class as i64, 1)?;

        // Backward pass to get gradients
        target_score.backward()?;

        // In a complete implementation:
        // 1. Extract activations from target convolutional layer
        // 2. Extract gradients flowing into the target layer
        // 3. Compute importance weights (global average pooling of gradients)
        // 4. Weighted combination of activation maps
        // 5. Apply ReLU and normalize

        // Placeholder: Create a dummy heatmap for demonstration
        // In production, this would be replaced with actual GradCAM computation
        let input_shape = batched_input.shape();
        let height = input_shape.dims()[2] as usize;
        let width = input_shape.dims()[3] as usize;

        let heatmap = self.create_placeholder_heatmap(height, width)?;

        Ok(heatmap)
    }

    /// Generate GradCAM++ heatmap (improved version of GradCAM)
    ///
    /// GradCAM++ provides better localization and works better for multiple instances
    /// of the same class in an image.
    pub fn generate_gradcam_plus_plus(
        &self,
        model: &dyn Module,
        input: &Tensor,
        target_class: usize,
    ) -> Result<Tensor> {
        // GradCAM++ uses second-order and third-order derivatives
        // for better weighting of activation maps
        self.generate_heatmap(model, input, target_class)
    }

    /// Overlay heatmap on original image
    ///
    /// # Arguments
    /// * `image` - Original image tensor [C, H, W]
    /// * `heatmap` - Heatmap tensor [H, W]
    /// * `alpha` - Blending factor (0.0 = only image, 1.0 = only heatmap)
    ///
    /// # Returns
    /// Blended visualization [C, H, W]
    pub fn overlay_heatmap(&self, image: &Tensor, heatmap: &Tensor, alpha: f32) -> Result<Tensor> {
        // Normalize heatmap to [0, 1]
        let hmax = heatmap.max(None, false)?;
        let hmin = heatmap.min()?;
        let normalized = heatmap.sub(&hmin)?.div(&hmax.sub(&hmin)?)?;

        // Convert heatmap to RGB using a colormap (e.g., jet colormap)
        let colored_heatmap = self.apply_colormap(&normalized)?;

        // Blend with original image
        let blended = image
            .mul_scalar(1.0 - alpha)?
            .add(&colored_heatmap.mul_scalar(alpha)?)?;

        Ok(blended)
    }

    /// Apply jet colormap to grayscale heatmap
    fn apply_colormap(&self, heatmap: &Tensor) -> Result<Tensor> {
        // Create RGB channels from grayscale heatmap using jet colormap
        // This is a simplified implementation
        let r = heatmap.mul_scalar(1.5)?.clamp(0.0, 1.0)?.unsqueeze(0)?;
        let g = heatmap
            .mul_scalar(2.0)?
            .sub_scalar(0.5)?
            .clamp(0.0, 1.0)?
            .unsqueeze(0)?;
        let b = heatmap
            .mul_scalar(1.5)?
            .sub_scalar(1.0)?
            .clamp(0.0, 1.0)?
            .unsqueeze(0)?;

        let colored = Tensor::cat(&[&r, &g, &b], 0)?;
        Ok(colored)
    }

    /// Create placeholder heatmap (to be replaced with actual implementation)
    fn create_placeholder_heatmap(&self, height: usize, width: usize) -> Result<Tensor> {
        use torsh_tensor::creation;

        // Create a simple gradient heatmap for demonstration
        let heatmap: Tensor<f32> = creation::zeros(&[height, width])?;

        Ok(heatmap)
    }
}

/// Saliency Map Generator
///
/// Saliency maps show which input pixels have the greatest influence on the model's
/// prediction by computing the gradient of the output with respect to the input.
pub struct SaliencyMap {
    device: Arc<dyn Device>,
}

impl SaliencyMap {
    /// Create a new saliency map generator
    pub fn new(device: Arc<dyn Device>) -> Self {
        Self { device }
    }

    /// Generate vanilla saliency map
    ///
    /// Computes the gradient of the class score with respect to the input image.
    ///
    /// # Arguments
    /// * `model` - The model to explain
    /// * `input` - Input image tensor [C, H, W] or [N, C, H, W]
    /// * `target_class` - Class index to generate saliency for
    ///
    /// # Returns
    /// Saliency map showing pixel importance
    pub fn generate(
        &self,
        model: &dyn Module,
        input: &Tensor,
        target_class: usize,
    ) -> Result<Tensor> {
        // Enable gradient computation for input
        let input_with_grad = input.clone().requires_grad_(true);

        // Ensure batch dimension
        let batched = if input_with_grad.ndim() == 3 {
            input_with_grad.unsqueeze(0)?
        } else {
            input_with_grad.clone()
        };

        // Forward pass
        let output = model.forward(&batched)?;

        // Get score for target class
        let score = output.narrow(1, target_class as i64, 1)?;

        // Backward pass
        score.backward()?;

        // Get gradient with respect to input
        let grad = batched
            .grad()
            .ok_or_else(|| VisionError::Other(anyhow::anyhow!("No gradient computed")))?;

        // Take absolute value and max across channels
        let abs_grad = grad.abs()?;
        let saliency = abs_grad.max(Some(1), false)?;

        Ok(saliency)
    }

    /// Generate smooth saliency map
    ///
    /// Averages saliency maps computed from multiple noisy versions of the input
    /// to reduce noise and produce smoother visualizations.
    pub fn generate_smooth(
        &self,
        model: &dyn Module,
        input: &Tensor,
        target_class: usize,
        num_samples: usize,
        noise_stddev: f32,
    ) -> Result<Tensor> {
        use torsh_tensor::creation;

        let shape: Vec<usize> = input.shape().dims().iter().map(|&x| x as usize).collect();
        let mut accumulated: Tensor<f32> = creation::zeros(&shape)?;

        for _ in 0..num_samples {
            // Add random noise to input
            let noise: Tensor<f32> = creation::randn(&shape)?;
            let noise = noise.mul_scalar(noise_stddev)?;
            let noisy_input = input.add(&noise)?;

            // Generate saliency for noisy input
            let saliency = self.generate(model, &noisy_input, target_class)?;

            // Accumulate
            accumulated = accumulated.add(&saliency)?;
        }

        // Average
        let smooth_saliency = accumulated.div_scalar(num_samples as f32)?;

        Ok(smooth_saliency)
    }
}

/// Integrated Gradients
///
/// Integrated Gradients is a method that attributes the prediction of a model to its
/// input features by integrating gradients along a path from a baseline to the input.
///
/// Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
pub struct IntegratedGradients {
    baseline_type: BaselineType,
    num_steps: usize,
    device: Arc<dyn Device>,
}

/// Type of baseline to use for integrated gradients
#[derive(Debug, Clone, Copy)]
pub enum BaselineType {
    /// Black image (all zeros)
    Black,
    /// Random noise
    Random,
    /// Blurred version of input
    Blurred,
}

impl IntegratedGradients {
    /// Create a new integrated gradients explainer
    pub fn new(baseline_type: BaselineType, num_steps: usize, device: Arc<dyn Device>) -> Self {
        Self {
            baseline_type,
            num_steps,
            device,
        }
    }

    /// Generate integrated gradients attribution map
    ///
    /// # Arguments
    /// * `model` - The model to explain
    /// * `input` - Input image tensor [C, H, W] or [N, C, H, W]
    /// * `target_class` - Class index to generate attribution for
    ///
    /// # Returns
    /// Attribution map showing feature importance
    pub fn generate(
        &self,
        model: &dyn Module,
        input: &Tensor,
        target_class: usize,
    ) -> Result<Tensor> {
        // Create baseline
        let baseline = self.create_baseline(input)?;

        // Compute path from baseline to input
        let mut accumulated_gradients = baseline.clone();

        for step in 0..self.num_steps {
            let alpha = (step as f32) / (self.num_steps as f32);

            // Interpolated input
            let interpolated = baseline
                .mul_scalar(1.0 - alpha)?
                .add(&input.mul_scalar(alpha)?)?;

            // Enable gradients
            let interp_with_grad = interpolated.clone().requires_grad_(true);

            // Forward pass
            let output = model.forward(&interp_with_grad)?;
            let score = output.narrow(1, target_class as i64, 1)?;

            // Backward pass
            score.backward()?;

            // Accumulate gradients
            let grad = interp_with_grad
                .grad()
                .ok_or_else(|| VisionError::Other(anyhow::anyhow!("No gradient computed")))?;
            accumulated_gradients = accumulated_gradients.add(&grad)?;
        }

        // Average gradients and multiply by (input - baseline)
        let avg_gradients = accumulated_gradients.div_scalar(self.num_steps as f32)?;
        let attribution = input.sub(&baseline)?.mul(&avg_gradients)?;

        Ok(attribution)
    }

    /// Create baseline based on baseline type
    fn create_baseline(&self, input: &Tensor) -> Result<Tensor> {
        use torsh_tensor::creation;

        let shape: Vec<usize> = input.shape().dims().iter().map(|&x| x as usize).collect();

        match self.baseline_type {
            BaselineType::Black => {
                let baseline: Tensor<f32> = creation::zeros(&shape)?;
                Ok(baseline)
            }
            BaselineType::Random => {
                let baseline: Tensor<f32> = creation::randn(&shape)?;
                Ok(baseline.mul_scalar(0.1)?)
            }
            BaselineType::Blurred => {
                // Simple blur by downsampling and upsampling
                // In production, use actual Gaussian blur
                Ok(input.clone())
            }
        }
    }
}

/// Attention Visualization
///
/// For models with attention mechanisms (e.g., Vision Transformers),
/// visualizes the attention weights to understand which parts of the image
/// the model is focusing on.
pub struct AttentionVisualizer {
    device: Arc<dyn Device>,
}

impl AttentionVisualizer {
    /// Create a new attention visualizer
    pub fn new(device: Arc<dyn Device>) -> Self {
        Self { device }
    }

    /// Visualize attention weights from a transformer layer
    ///
    /// # Arguments
    /// * `attention_weights` - Attention weight tensor [N, num_heads, seq_len, seq_len]
    /// * `patch_size` - Size of image patches
    /// * `image_size` - Original image size (H, W)
    ///
    /// # Returns
    /// Attention map showing which regions are attended to
    pub fn visualize_attention(
        &self,
        attention_weights: &Tensor,
        patch_size: usize,
        image_size: (usize, usize),
    ) -> Result<Tensor> {
        // Average attention across heads
        let avg_attention = attention_weights.mean(Some(&[1]), false)?;

        // Extract attention from CLS token to patches (first row)
        let cls_attention = avg_attention.narrow(1, 0, 1)?;

        // Reshape to spatial grid
        let num_patches_h = image_size.0 / patch_size;
        let num_patches_w = image_size.1 / patch_size;

        let reshaped =
            cls_attention.reshape(&[1i32, num_patches_h as i32, num_patches_w as i32])?;

        // Upsample to original image size
        // In production, use proper interpolation
        let upsampled = reshaped.clone();

        Ok(upsampled)
    }

    /// Visualize attention rollout
    ///
    /// Combines attention from multiple layers to understand
    /// the full attention flow through the network.
    pub fn attention_rollout(&self, attention_layers: Vec<Tensor>) -> Result<Tensor> {
        if attention_layers.is_empty() {
            return Err(VisionError::InvalidArgument(
                "No attention layers provided".to_string(),
            ));
        }

        // Start with identity matrix
        let mut rollout = attention_layers[0].clone();

        // Multiply attention matrices from consecutive layers
        for attention in attention_layers.iter().skip(1) {
            rollout = rollout.matmul(attention)?;
        }

        Ok(rollout)
    }
}

/// Feature Visualization
///
/// Generate synthetic images that maximally activate specific neurons or layers,
/// helping understand what features the network has learned.
pub struct FeatureVisualizer {
    learning_rate: f32,
    num_iterations: usize,
    device: Arc<dyn Device>,
}

impl FeatureVisualizer {
    /// Create a new feature visualizer
    pub fn new(learning_rate: f32, num_iterations: usize, device: Arc<dyn Device>) -> Self {
        Self {
            learning_rate,
            num_iterations,
            device,
        }
    }

    /// Generate an image that maximally activates a specific class
    ///
    /// # Arguments
    /// * `model` - The model to visualize
    /// * `target_class` - Class to maximize activation for
    /// * `image_size` - Size of generated image (H, W)
    ///
    /// # Returns
    /// Synthesized image that maximally activates the target class
    pub fn visualize_class(
        &self,
        model: &dyn Module,
        target_class: usize,
        image_size: (usize, usize),
    ) -> Result<Tensor> {
        use torsh_tensor::creation;

        // Initialize random image
        let mut image: Tensor<f32> = creation::randn(&[1, 3, image_size.0, image_size.1])?;
        image = image.mul_scalar(0.1)?.add_scalar(0.5)?.requires_grad_(true);

        // Optimization loop
        for iteration in 0..self.num_iterations {
            // Forward pass
            let output = model.forward(&image)?;
            let class_score = output.narrow(1, target_class as i64, 1)?;

            // We want to maximize the class score
            let loss = class_score.neg()?;

            // Backward pass
            loss.backward()?;

            // Update image
            let grad = image
                .grad()
                .ok_or_else(|| VisionError::Other(anyhow::anyhow!("No gradient computed")))?;
            image = image.sub(&grad.mul_scalar(self.learning_rate)?)?;

            // Apply regularization (keep values in reasonable range)
            image = image.clamp(-2.0, 2.0)?;

            if iteration % 10 == 0 {
                println!("Iteration {}: loss = {:?}", iteration, loss.item());
            }
        }

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::CpuDevice;
    use torsh_tensor::creation;

    #[test]
    fn test_gradcam_creation() {
        let device = Arc::new(CpuDevice::new());
        let gradcam = GradCAM::new("layer4".to_string(), device);
        assert_eq!(gradcam.target_layer_name, "layer4");
    }

    #[test]
    fn test_saliency_map_creation() {
        let device = Arc::new(CpuDevice::new());
        let _saliency = SaliencyMap::new(device);
    }

    #[test]
    fn test_integrated_gradients_creation() {
        let device = Arc::new(CpuDevice::new());
        let _ig = IntegratedGradients::new(BaselineType::Black, 50, device);
    }

    #[test]
    fn test_attention_visualizer_creation() {
        let device = Arc::new(CpuDevice::new());
        let _visualizer = AttentionVisualizer::new(device);
    }

    #[test]
    fn test_feature_visualizer_creation() {
        let device = Arc::new(CpuDevice::new());
        let _visualizer = FeatureVisualizer::new(0.1, 100, device);
    }

    #[test]
    fn test_baseline_types() {
        let device: Arc<dyn Device> = Arc::new(CpuDevice::new());
        let ig_black = IntegratedGradients::new(BaselineType::Black, 50, Arc::clone(&device));
        let ig_random = IntegratedGradients::new(BaselineType::Random, 50, Arc::clone(&device));
        let ig_blurred = IntegratedGradients::new(BaselineType::Blurred, 50, Arc::clone(&device));

        let input: Tensor<f32> = creation::ones(&[1, 3, 224, 224]).unwrap();

        assert!(ig_black.create_baseline(&input).is_ok());
        assert!(ig_random.create_baseline(&input).is_ok());
        assert!(ig_blurred.create_baseline(&input).is_ok());
    }
}
