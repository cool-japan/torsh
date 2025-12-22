//! Common types for vision models

/// Vision model architectures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionArchitecture {
    ResNet,
    EfficientNet,
    DenseNet,
    MobileNet,
    VisionTransformer,
    SwinTransformer,
    ConvNeXt,
    DETR,
    MaskRCNN,
    YOLO,
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
            VisionArchitecture::SwinTransformer => "Swin Transformer",
            VisionArchitecture::ConvNeXt => "ConvNeXt",
            VisionArchitecture::DETR => "DETR",
            VisionArchitecture::MaskRCNN => "Mask R-CNN",
            VisionArchitecture::YOLO => "YOLO",
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
            VisionArchitecture::SwinTransformer => (3, 224, 224),
            VisionArchitecture::ConvNeXt => (3, 224, 224),
            VisionArchitecture::DETR => (3, 800, 800), // DETR typically uses larger images
            VisionArchitecture::MaskRCNN => (3, 800, 800), // Mask R-CNN also uses larger images
            VisionArchitecture::YOLO => (3, 640, 640), // YOLO typically uses 640x640
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

/// Common vision tasks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionTask {
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// Instance segmentation
    InstanceSegmentation,
    /// Semantic segmentation
    SemanticSegmentation,
    /// Feature extraction
    FeatureExtraction,
    /// Image generation
    ImageGeneration,
}

impl VisionTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            VisionTask::ImageClassification => "image_classification",
            VisionTask::ObjectDetection => "object_detection",
            VisionTask::InstanceSegmentation => "instance_segmentation",
            VisionTask::SemanticSegmentation => "semantic_segmentation",
            VisionTask::FeatureExtraction => "feature_extraction",
            VisionTask::ImageGeneration => "image_generation",
        }
    }
}

/// Standard image normalization parameters
#[derive(Debug, Clone)]
pub struct ImageNormalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageNormalization {
    /// ImageNet normalization parameters
    pub fn imagenet() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// CIFAR-10 normalization parameters
    pub fn cifar10() -> Self {
        Self {
            mean: [0.4914, 0.4822, 0.4465],
            std: [0.2023, 0.1994, 0.2010],
        }
    }

    /// CIFAR-100 normalization parameters
    pub fn cifar100() -> Self {
        Self {
            mean: [0.5071, 0.4867, 0.4408],
            std: [0.2675, 0.2565, 0.2761],
        }
    }

    /// No normalization (identity)
    pub fn none() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }
}

/// Activation function types commonly used in vision models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionActivation {
    ReLU,
    ReLU6,
    Swish,
    HardSwish,
    GELU,
    Mish,
    Sigmoid,
    Tanh,
}

impl VisionActivation {
    pub fn as_str(&self) -> &'static str {
        match self {
            VisionActivation::ReLU => "relu",
            VisionActivation::ReLU6 => "relu6",
            VisionActivation::Swish => "swish",
            VisionActivation::HardSwish => "hard_swish",
            VisionActivation::GELU => "gelu",
            VisionActivation::Mish => "mish",
            VisionActivation::Sigmoid => "sigmoid",
            VisionActivation::Tanh => "tanh",
        }
    }
}

/// Configuration for model initialization
#[derive(Debug, Clone)]
pub struct ModelInitConfig {
    /// Initialize from pretrained weights
    pub pretrained: bool,
    /// Number of output classes (affects final layer)
    pub num_classes: usize,
    /// Whether to freeze backbone weights
    pub freeze_backbone: bool,
    /// Dropout rate for classification head
    pub dropout: f32,
}

impl Default for ModelInitConfig {
    fn default() -> Self {
        Self {
            pretrained: true,
            num_classes: 1000,
            freeze_backbone: false,
            dropout: 0.0,
        }
    }
}

impl ModelInitConfig {
    /// Configuration for fine-tuning with pretrained weights
    pub fn fine_tuning(num_classes: usize) -> Self {
        Self {
            pretrained: true,
            num_classes,
            freeze_backbone: false,
            dropout: 0.1,
        }
    }

    /// Configuration for feature extraction (frozen backbone)
    pub fn feature_extraction(num_classes: usize) -> Self {
        Self {
            pretrained: true,
            num_classes,
            freeze_backbone: true,
            dropout: 0.2,
        }
    }

    /// Configuration for training from scratch
    pub fn from_scratch(num_classes: usize) -> Self {
        Self {
            pretrained: false,
            num_classes,
            freeze_backbone: false,
            dropout: 0.1,
        }
    }
}
