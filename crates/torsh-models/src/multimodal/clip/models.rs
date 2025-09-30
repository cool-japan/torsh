//! CLIP main model implementations

use super::super::common::utils::contrastive_loss;
use super::config::CLIPConfig;
use super::encoders::{CLIPTextTransformer, CLIPVisionTransformer};
use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Main CLIP Model for contrastive vision-language learning
pub struct CLIPModel {
    vision_model: CLIPVisionTransformer,
    text_model: CLIPTextTransformer,
    visual_projection: Linear,
    text_projection: Linear,
    logit_scale: Parameter,
    config: CLIPConfig,
}

impl CLIPModel {
    pub fn new(config: CLIPConfig) -> Result<Self> {
        let vision_model = CLIPVisionTransformer::new(config.vision_config.clone())?;
        let text_model = CLIPTextTransformer::new(config.text_config.clone())?;

        let visual_projection = Linear::new(
            config.vision_config.hidden_size,
            config.projection_dim,
            false,
        );
        let text_projection =
            Linear::new(config.text_config.hidden_size, config.projection_dim, false);

        // Initialize logit scale parameter
        let logit_scale = Parameter::new(creation::full(&[], config.logit_scale_init_value)?);

        Ok(Self {
            vision_model,
            text_model,
            visual_projection,
            text_projection,
            logit_scale,
            config,
        })
    }

    /// Create CLIP-ViT-B/32 model
    pub fn vit_b_32() -> Result<Self> {
        let config = CLIPConfig::vit_b_32();
        Self::new(config)
    }

    /// Create CLIP-ViT-B/16 model
    pub fn vit_b_16() -> Result<Self> {
        let config = CLIPConfig::vit_b_16();
        Self::new(config)
    }

    /// Create CLIP-ViT-L/14 model
    pub fn vit_l_14() -> Result<Self> {
        let config = CLIPConfig::vit_l_14();
        Self::new(config)
    }

    /// Encode images to feature representations
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let image_features = self.vision_model.forward(pixel_values)?;
        let image_features = self.visual_projection.forward(&image_features)?;

        // Normalize features
        let norm = image_features.norm()?;
        image_features.div(&norm)
    }

    /// Encode text to feature representations
    pub fn encode_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        let text_features = self.text_model.forward(input_ids)?;
        let text_features = self.text_projection.forward(&text_features)?;

        // Normalize features
        let norm = text_features.norm()?;
        text_features.div(&norm)
    }

    /// Forward pass computing both image and text features
    pub fn forward_features(
        &self,
        pixel_values: &Tensor,
        input_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let image_features = self.encode_image(pixel_values)?;
        let text_features = self.encode_text(input_ids)?;
        Ok((image_features, text_features))
    }

    /// Compute similarity matrix between images and texts
    pub fn compute_similarity(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let (image_features, text_features) = self.forward_features(pixel_values, input_ids)?;

        // Compute cosine similarity
        let logit_scale_tensor = self.logit_scale.tensor();
        let logit_scale_guard = logit_scale_tensor.read();
        let logit_scale = logit_scale_guard.exp()?;
        let logits_per_image = image_features.matmul(&text_features.transpose(-2, -1)?)?;
        let logits_per_image = logits_per_image.mul(&logit_scale)?;

        Ok(logits_per_image)
    }

    /// Compute contrastive loss
    pub fn compute_loss(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let (image_features, text_features) = self.forward_features(pixel_values, input_ids)?;
        let logit_scale_tensor = self.logit_scale.tensor();
        let logit_scale_guard = logit_scale_tensor.read();
        let temperature = logit_scale_guard.exp()?.item()?;
        contrastive_loss(&image_features, &text_features, temperature)
    }

    /// Get the current logit scale value
    pub fn get_logit_scale(&self) -> Result<f32> {
        let logit_scale_tensor = self.logit_scale.tensor();
        let logit_scale_guard = logit_scale_tensor.read();
        logit_scale_guard.exp()?.item()
    }
}

impl Module for CLIPModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For a single input, assume it's an image and return image features
        self.encode_image(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.vision_model.parameters() {
            params.insert(format!("vision_model.{}", name), param);
        }

        for (name, param) in self.text_model.parameters() {
            params.insert(format!("text_model.{}", name), param);
        }

        for (name, param) in self.visual_projection.parameters() {
            params.insert(format!("visual_projection.{}", name), param);
        }

        for (name, param) in self.text_projection.parameters() {
            params.insert(format!("text_projection.{}", name), param);
        }

        params.insert("logit_scale".to_string(), self.logit_scale.clone());

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.vision_model.training()
            && self.text_model.training()
            && self.visual_projection.training()
            && self.text_projection.training()
    }

    fn train(&mut self) {
        self.vision_model.train();
        self.text_model.train();
        self.visual_projection.train();
        self.text_projection.train();
    }

    fn eval(&mut self) {
        self.vision_model.eval();
        self.text_model.eval();
        self.visual_projection.eval();
        self.text_projection.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.vision_model.to_device(device)?;
        self.text_model.to_device(device)?;
        self.visual_projection.to_device(device)?;
        self.text_projection.to_device(device)?;
        Ok(())
    }
}

/// CLIP model output structure
#[derive(Debug)]
pub struct CLIPOutput {
    pub logits_per_image: Tensor,
    pub logits_per_text: Tensor,
    pub image_embeds: Tensor,
    pub text_embeds: Tensor,
}

impl CLIPOutput {
    pub fn new(
        logits_per_image: Tensor,
        logits_per_text: Tensor,
        image_embeds: Tensor,
        text_embeds: Tensor,
    ) -> Self {
        Self {
            logits_per_image,
            logits_per_text,
            image_embeds,
            text_embeds,
        }
    }
}

/// CLIP model with structured output
pub struct CLIPModelWithOutput {
    model: CLIPModel,
}

impl CLIPModelWithOutput {
    pub fn new(config: CLIPConfig) -> Result<Self> {
        let model = CLIPModel::new(config)?;
        Ok(Self { model })
    }

    /// Forward pass with structured output
    pub fn forward(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<CLIPOutput> {
        let (image_embeds, text_embeds) = self.model.forward_features(pixel_values, input_ids)?;

        // Compute logits
        let logit_scale_tensor = self.model.logit_scale.tensor();
        let logit_scale_guard = logit_scale_tensor.read();
        let logit_scale = logit_scale_guard.exp()?;
        let logits_per_image = image_embeds.matmul(&text_embeds.transpose(-2, -1)?)?;
        let logits_per_image = logits_per_image.mul(&logit_scale)?;
        let logits_per_text = logits_per_image.transpose(-2, -1)?;

        Ok(CLIPOutput::new(
            logits_per_image,
            logits_per_text,
            image_embeds,
            text_embeds,
        ))
    }

    /// Get the underlying CLIP model
    pub fn model(&self) -> &CLIPModel {
        &self.model
    }

    /// Get mutable reference to the underlying CLIP model
    pub fn model_mut(&mut self) -> &mut CLIPModel {
        &mut self.model
    }
}

impl Module for CLIPModelWithOutput {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.model.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.model.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.model.named_parameters()
    }

    fn training(&self) -> bool {
        self.model.training()
    }

    fn train(&mut self) {
        self.model.train();
    }

    fn eval(&mut self) {
        self.model.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.model.to_device(device)
    }
}
