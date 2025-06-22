//! Vision models and utilities

#[cfg(feature = "vision")]
use std::collections::HashMap;

use crate::{ModelError, ModelResult};
use torsh_nn::{Module, Parameter};
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;
use torsh_core::{TensorError, DeviceType, error::TorshError};
use std::sync::Arc;
use std::collections::HashMap;

/// Vision model architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionArchitecture {
    ResNet,
    EfficientNet,
    DenseNet,
    MobileNet,
    VisionTransformer,
    ConvNeXt,
    RegNet,
    SENet,
}

impl VisionArchitecture {
    /// Get architecture name
    pub fn name(&self) -> &'static str {
        match self {
            VisionArchitecture::ResNet => "ResNet",
            VisionArchitecture::EfficientNet => "EfficientNet",
            VisionArchitecture::DenseNet => "DenseNet",
            VisionArchitecture::MobileNet => "MobileNet",
            VisionArchitecture::VisionTransformer => "Vision Transformer",
            VisionArchitecture::ConvNeXt => "ConvNeXt",
            VisionArchitecture::RegNet => "RegNet",
            VisionArchitecture::SENet => "SENet",
        }
    }
    
    /// Get typical input size for architecture
    pub fn default_input_size(&self) -> (usize, usize, usize) {
        match self {
            VisionArchitecture::ResNet => (3, 224, 224),
            VisionArchitecture::EfficientNet => (3, 224, 224),
            VisionArchitecture::DenseNet => (3, 224, 224),
            VisionArchitecture::MobileNet => (3, 224, 224),
            VisionArchitecture::VisionTransformer => (3, 224, 224),
            VisionArchitecture::ConvNeXt => (3, 224, 224),
            VisionArchitecture::RegNet => (3, 224, 224),
            VisionArchitecture::SENet => (3, 224, 224),
        }
    }
}

/// Vision model variants
#[derive(Debug, Clone)]
pub struct VisionModelVariant {
    /// Architecture type
    pub architecture: VisionArchitecture,
    /// Model size/variant
    pub variant: String,
    /// Number of parameters
    pub parameters: u64,
    /// Input resolution
    pub input_size: (usize, usize, usize), // (C, H, W)
    /// Number of output classes
    pub num_classes: usize,
    /// ImageNet top-1 accuracy
    pub imagenet_top1_accuracy: Option<f32>,
    /// ImageNet top-5 accuracy
    pub imagenet_top5_accuracy: Option<f32>,
}

/// Common vision model variants
pub fn get_common_vision_models() -> Vec<VisionModelVariant> {
    vec![
        // ResNet variants
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet18".to_string(),
            parameters: 11_689_512,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(69.758),
            imagenet_top5_accuracy: Some(89.078),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet34".to_string(),
            parameters: 21_797_672,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(73.314),
            imagenet_top5_accuracy: Some(91.420),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet50".to_string(),
            parameters: 25_557_032,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(76.130),
            imagenet_top5_accuracy: Some(92.862),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet101".to_string(),
            parameters: 44_549_160,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(77.374),
            imagenet_top5_accuracy: Some(93.546),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::ResNet,
            variant: "resnet152".to_string(),
            parameters: 60_192_808,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(78.312),
            imagenet_top5_accuracy: Some(94.046),
        },
        
        // EfficientNet variants
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b0".to_string(),
            parameters: 5_288_548,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(77.692),
            imagenet_top5_accuracy: Some(93.532),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b1".to_string(),
            parameters: 7_794_184,
            input_size: (3, 240, 240),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(78.642),
            imagenet_top5_accuracy: Some(94.186),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::EfficientNet,
            variant: "efficientnet_b2".to_string(),
            parameters: 9_109_994,
            input_size: (3, 260, 260),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(79.688),
            imagenet_top5_accuracy: Some(94.876),
        },
        
        // MobileNet variants
        VisionModelVariant {
            architecture: VisionArchitecture::MobileNet,
            variant: "mobilenet_v2".to_string(),
            parameters: 3_504_872,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(71.878),
            imagenet_top5_accuracy: Some(90.286),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::MobileNet,
            variant: "mobilenet_v3_small".to_string(),
            parameters: 2_542_856,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(67.668),
            imagenet_top5_accuracy: Some(87.402),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::MobileNet,
            variant: "mobilenet_v3_large".to_string(),
            parameters: 5_483_032,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(74.042),
            imagenet_top5_accuracy: Some(91.340),
        },
        
        // Vision Transformer variants
        VisionModelVariant {
            architecture: VisionArchitecture::VisionTransformer,
            variant: "vit_base_patch16_224".to_string(),
            parameters: 86_567_656,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(81.072),
            imagenet_top5_accuracy: Some(95.318),
        },
        VisionModelVariant {
            architecture: VisionArchitecture::VisionTransformer,
            variant: "vit_large_patch16_224".to_string(),
            parameters: 304_326_632,
            input_size: (3, 224, 224),
            num_classes: 1000,
            imagenet_top1_accuracy: Some(82.632),
            imagenet_top5_accuracy: Some(96.176),
        },
    ]
}

/// Image preprocessing utilities
pub struct ImagePreprocessor {
    /// Target size for resizing
    pub target_size: (usize, usize),
    /// Normalization mean values (RGB)
    pub mean: [f32; 3],
    /// Normalization std values (RGB)
    pub std: [f32; 3],
    /// Whether to center crop
    pub center_crop: bool,
    /// Crop size if center cropping
    pub crop_size: Option<(usize, usize)>,
}

impl ImagePreprocessor {
    /// Create ImageNet preprocessor
    pub fn imagenet() -> Self {
        Self {
            target_size: (256, 256),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            center_crop: true,
            crop_size: Some((224, 224)),
        }
    }
    
    /// Create CIFAR-10 preprocessor
    pub fn cifar10() -> Self {
        Self {
            target_size: (32, 32),
            mean: [0.4914, 0.4822, 0.4465],
            std: [0.2023, 0.1994, 0.2010],
            center_crop: false,
            crop_size: None,
        }
    }
    
    /// Create custom preprocessor
    pub fn custom(
        target_size: (usize, usize),
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        Self {
            target_size,
            mean,
            std,
            center_crop: false,
            crop_size: None,
        }
    }
    
    /// Preprocess image tensor (placeholder implementation)
    pub fn preprocess(&self, _image_data: &[f32]) -> ModelResult<Vec<f32>> {
        // TODO: Implement actual image preprocessing
        // This would involve:
        // 1. Resize to target_size
        // 2. Center crop if enabled
        // 3. Normalize with mean and std
        // 4. Convert to tensor format
        
        Err(ModelError::LoadingError {
            reason: "Image preprocessing not yet implemented".to_string(),
        })
    }
}

/// Vision model utilities
pub struct VisionModelUtils;

impl VisionModelUtils {
    /// Get recommended preprocessor for a model
    pub fn get_preprocessor(model_name: &str) -> ImagePreprocessor {
        // Default to ImageNet preprocessing for most models
        match model_name {
            name if name.contains("cifar") => ImagePreprocessor::cifar10(),
            _ => ImagePreprocessor::imagenet(),
        }
    }
    
    /// Get model variant by name
    pub fn get_model_variant(name: &str) -> Option<VisionModelVariant> {
        let models = get_common_vision_models();
        models.into_iter().find(|m| m.variant == name)
    }
    
    /// List models by architecture
    pub fn list_models_by_architecture(arch: VisionArchitecture) -> Vec<VisionModelVariant> {
        let models = get_common_vision_models();
        models.into_iter().filter(|m| m.architecture == arch).collect()
    }
    
    /// Get top-k classification results
    pub fn get_top_k_predictions(
        logits: &[f32],
        k: usize,
        class_names: Option<&[String]>,
    ) -> Vec<(usize, f32, Option<String>)> {
        let mut predictions: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        // Sort by score descending
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        predictions
            .into_iter()
            .take(k)
            .map(|(idx, score)| {
                let class_name = class_names
                    .and_then(|names| names.get(idx))
                    .cloned();
                (idx, score, class_name)
            })
            .collect()
    }
    
    /// Apply softmax to logits
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum).collect()
    }
}

/// ImageNet class names (top 1000)
pub fn get_imagenet_class_names() -> Vec<String> {
    // This would normally be loaded from a file or embedded as a constant
    // For now, returning a subset as an example
    vec![
        "tench".to_string(),
        "goldfish".to_string(),
        "great white shark".to_string(),
        "tiger shark".to_string(),
        "hammerhead".to_string(),
        // ... (would contain all 1000 ImageNet classes)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vision_architecture_name() {
        assert_eq!(VisionArchitecture::ResNet.name(), "ResNet");
        assert_eq!(VisionArchitecture::EfficientNet.name(), "EfficientNet");
    }
    
    #[test]
    fn test_get_common_vision_models() {
        let models = get_common_vision_models();
        assert!(!models.is_empty());
        
        // Check if ResNet-18 is present
        let resnet18 = models.iter().find(|m| m.variant == "resnet18");
        assert!(resnet18.is_some());
        
        let resnet18 = resnet18.unwrap();
        assert_eq!(resnet18.architecture, VisionArchitecture::ResNet);
        assert_eq!(resnet18.parameters, 11_689_512);
    }
    
    #[test]
    fn test_image_preprocessor() {
        let preprocessor = ImagePreprocessor::imagenet();
        assert_eq!(preprocessor.target_size, (256, 256));
        assert_eq!(preprocessor.mean, [0.485, 0.456, 0.406]);
        assert_eq!(preprocessor.crop_size, Some((224, 224)));
    }
    
    #[test]
    fn test_vision_model_utils() {
        let variant = VisionModelUtils::get_model_variant("resnet18");
        assert!(variant.is_some());
        
        let variant = variant.unwrap();
        assert_eq!(variant.architecture, VisionArchitecture::ResNet);
    }
    
    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = VisionModelUtils::softmax(&logits);
        
        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that probabilities are in ascending order for ascending logits
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }
    
    #[test]
    fn test_top_k_predictions() {
        let logits = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let class_names = vec![
            "class0".to_string(),
            "class1".to_string(),
            "class2".to_string(),
            "class3".to_string(),
            "class4".to_string(),
        ];
        
        let top_3 = VisionModelUtils::get_top_k_predictions(&logits, 3, Some(&class_names));
        
        assert_eq!(top_3.len(), 3);
        assert_eq!(top_3[0].0, 3); // Index of highest score
        assert_eq!(top_3[0].2, Some("class3".to_string()));
        assert_eq!(top_3[1].0, 1); // Index of second highest
        assert_eq!(top_3[2].0, 2); // Index of third highest
    }
}

/// Basic residual block for ResNet
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    stride: usize,
}

impl BasicBlock {
    pub fn new(inplanes: usize, planes: usize, stride: usize, downsample: Option<Sequential>) -> Self {
        let conv1 = Conv2d::new(inplanes, planes, 3)
            .stride(stride)
            .padding(1)
            .bias(false);
        let bn1 = BatchNorm2d::new(planes);
        let relu = ReLU::new();
        let conv2 = Conv2d::new(planes, planes, 3)
            .stride(1)
            .padding(1)
            .bias(false);
        let bn2 = BatchNorm2d::new(planes);

        Self {
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            downsample,
            stride,
        }
    }
}

impl Module for BasicBlock {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, TensorError> {
        let identity = x.clone();

        let mut out = self.conv1.forward(x)?;
        out = self.bn1.forward(&out)?;
        out = self.relu.forward(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let identity = if let Some(ref mut downsample) = self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        out = out.add(&identity)?;
        out = self.relu.forward(&out)?;

        Ok(out)
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        self.conv2.train();
        self.bn2.train();
        if let Some(ref mut downsample) = self.downsample {
            downsample.train();
        }
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        self.conv2.eval();
        self.bn2.eval();
        if let Some(ref mut downsample) = self.downsample {
            downsample.eval();
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some(ref downsample) = self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<(), TorshError> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        self.conv2.to_device(device)?;
        self.bn2.to_device(device)?;
        if let Some(ref mut downsample) = self.downsample {
            downsample.to_device(device)?;
        }
        Ok(())
    }
}

/// ResNet model implementation
pub struct ResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: ModuleList<BasicBlock>,
    layer2: ModuleList<BasicBlock>,
    layer3: ModuleList<BasicBlock>,
    layer4: ModuleList<BasicBlock>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
    inplanes: usize,
}

impl ResNet {
    pub fn new(layers: &[usize], num_classes: usize) -> Self {
        let mut inplanes = 64;
        
        let conv1 = Conv2d::new(3, 64, 7)
            .stride(2)
            .padding(3)
            .bias(false);
        let bn1 = BatchNorm2d::new(64);
        let relu = ReLU::new();
        let maxpool = MaxPool2d::new(3)
            .stride(2)
            .padding(1);

        let mut model = Self {
            conv1,
            bn1,
            relu,
            maxpool,
            layer1: ModuleList::new(),
            layer2: ModuleList::new(),
            layer3: ModuleList::new(),
            layer4: ModuleList::new(),
            avgpool: AdaptiveAvgPool2d::new(1),
            fc: Linear::new(512, num_classes),
            inplanes,
        };

        model.layer1 = model._make_layer(64, layers[0], 1);
        model.inplanes = 64;
        model.layer2 = model._make_layer(128, layers[1], 2);
        model.inplanes = 128;
        model.layer3 = model._make_layer(256, layers[2], 2);
        model.inplanes = 256;
        model.layer4 = model._make_layer(512, layers[3], 2);

        model
    }

    fn _make_layer(&mut self, planes: usize, blocks: usize, stride: usize) -> ModuleList<BasicBlock> {
        let mut downsample = None;
        if stride != 1 || self.inplanes != planes {
            let mut seq = Sequential::new();
            seq.add_module("conv", Conv2d::new(self.inplanes, planes, 1)
                .stride(stride)
                .bias(false));
            seq.add_module("bn", BatchNorm2d::new(planes));
            downsample = Some(seq);
        }

        let mut layers = ModuleList::new();
        layers.add_module(BasicBlock::new(self.inplanes, planes, stride, downsample));
        self.inplanes = planes;

        for _ in 1..blocks {
            layers.add_module(BasicBlock::new(self.inplanes, planes, 1, None));
        }

        layers
    }

    /// Create ResNet-18
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(&[2, 2, 2, 2], num_classes)
    }

    /// Create ResNet-34
    pub fn resnet34(num_classes: usize) -> Self {
        Self::new(&[3, 4, 6, 3], num_classes)
    }

    /// Create ResNet-50 (would need Bottleneck blocks, simplified here)
    pub fn resnet50(num_classes: usize) -> Self {
        Self::new(&[3, 4, 6, 3], num_classes)
    }
}

impl Module for ResNet {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, TensorError> {
        let mut x = self.conv1.forward(x)?;
        x = self.bn1.forward(&x)?;
        x = self.relu.forward(&x)?;
        x = self.maxpool.forward(&x)?;

        for layer in self.layer1.modules_mut() {
            x = layer.forward(&x)?;
        }
        for layer in self.layer2.modules_mut() {
            x = layer.forward(&x)?;
        }
        for layer in self.layer3.modules_mut() {
            x = layer.forward(&x)?;
        }
        for layer in self.layer4.modules_mut() {
            x = layer.forward(&x)?;
        }

        x = self.avgpool.forward(&x)?;
        x = x.view(&[x.shape()[0], -1])?;
        x = self.fc.forward(&x)?;

        Ok(x)
    }

    fn train(&mut self) {
        self.conv1.train();
        self.bn1.train();
        for layer in self.layer1.modules_mut() {
            layer.train();
        }
        for layer in self.layer2.modules_mut() {
            layer.train();
        }
        for layer in self.layer3.modules_mut() {
            layer.train();
        }
        for layer in self.layer4.modules_mut() {
            layer.train();
        }
        self.fc.train();
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.bn1.eval();
        for layer in self.layer1.modules_mut() {
            layer.eval();
        }
        for layer in self.layer2.modules_mut() {
            layer.eval();
        }
        for layer in self.layer3.modules_mut() {
            layer.eval();
        }
        for layer in self.layer4.modules_mut() {
            layer.eval();
        }
        self.fc.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        for layer in self.layer1.modules() {
            params.extend(layer.parameters());
        }
        for layer in self.layer2.modules() {
            params.extend(layer.parameters());
        }
        for layer in self.layer3.modules() {
            params.extend(layer.parameters());
        }
        for layer in self.layer4.modules() {
            params.extend(layer.parameters());
        }
        params.extend(self.fc.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv1.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<(), TorshError> {
        self.conv1.to_device(device)?;
        self.bn1.to_device(device)?;
        for layer in self.layer1.modules_mut() {
            layer.to_device(device)?;
        }
        for layer in self.layer2.modules_mut() {
            layer.to_device(device)?;
        }
        for layer in self.layer3.modules_mut() {
            layer.to_device(device)?;
        }
        for layer in self.layer4.modules_mut() {
            layer.to_device(device)?;
        }
        self.fc.to_device(device)?;
        Ok(())
    }
}