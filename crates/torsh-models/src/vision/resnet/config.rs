//! ResNet model configuration

use super::super::common::types::ModelInitConfig;

/// ResNet architecture variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResNetVariant {
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
}

impl ResNetVariant {
    /// Get the layer configuration for each ResNet variant
    pub fn layer_config(&self) -> &'static [usize] {
        match self {
            ResNetVariant::ResNet18 => &[2, 2, 2, 2],
            ResNetVariant::ResNet34 => &[3, 4, 6, 3],
            ResNetVariant::ResNet50 => &[3, 4, 6, 3],
            ResNetVariant::ResNet101 => &[3, 4, 23, 3],
            ResNetVariant::ResNet152 => &[3, 8, 36, 3],
        }
    }

    /// Whether this variant uses bottleneck blocks
    pub fn uses_bottleneck(&self) -> bool {
        matches!(
            self,
            ResNetVariant::ResNet50 | ResNetVariant::ResNet101 | ResNetVariant::ResNet152
        )
    }

    /// Get the number of parameters (approximate)
    pub fn parameter_count(&self) -> u64 {
        match self {
            ResNetVariant::ResNet18 => 11_689_512,
            ResNetVariant::ResNet34 => 21_797_672,
            ResNetVariant::ResNet50 => 25_557_032,
            ResNetVariant::ResNet101 => 44_549_160,
            ResNetVariant::ResNet152 => 60_192_808,
        }
    }

    /// Get the typical ImageNet top-1 accuracy
    pub fn imagenet_accuracy(&self) -> f32 {
        match self {
            ResNetVariant::ResNet18 => 69.758,
            ResNetVariant::ResNet34 => 73.314,
            ResNetVariant::ResNet50 => 76.130,
            ResNetVariant::ResNet101 => 77.374,
            ResNetVariant::ResNet152 => 78.312,
        }
    }

    /// Get the variant name as string
    pub fn name(&self) -> &'static str {
        match self {
            ResNetVariant::ResNet18 => "resnet18",
            ResNetVariant::ResNet34 => "resnet34",
            ResNetVariant::ResNet50 => "resnet50",
            ResNetVariant::ResNet101 => "resnet101",
            ResNetVariant::ResNet152 => "resnet152",
        }
    }
}

/// ResNet configuration
#[derive(Debug, Clone)]
pub struct ResNetConfig {
    /// ResNet variant
    pub variant: ResNetVariant,
    /// Number of output classes
    pub num_classes: usize,
    /// Input channels (typically 3 for RGB)
    pub in_channels: usize,
    /// Initial conv layer channels
    pub stem_channels: usize,
    /// Whether to use zero padding for first conv
    pub zero_init_residual: bool,
    /// Groups for grouped convolution
    pub groups: usize,
    /// Base width for bottleneck blocks
    pub width_per_group: usize,
    /// Whether to replace stride with dilation
    pub replace_stride_with_dilation: Vec<bool>,
    /// Norm layer configuration
    pub norm_layer: String,
    /// Dropout probability for classifier
    pub dropout: f32,
    /// Squeeze-and-Excitation configuration
    pub use_se: bool,
    /// SE reduction ratio
    pub se_reduction_ratio: usize,
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            variant: ResNetVariant::ResNet50,
            num_classes: 1000,
            in_channels: 3,
            stem_channels: 64,
            zero_init_residual: false,
            groups: 1,
            width_per_group: 64,
            replace_stride_with_dilation: vec![false, false, false],
            norm_layer: "BatchNorm2d".to_string(),
            dropout: 0.0,
            use_se: false,
            se_reduction_ratio: 16,
        }
    }
}

impl ResNetConfig {
    /// Create ResNet-18 configuration
    pub fn resnet18(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet18,
            num_classes,
            ..Default::default()
        }
    }

    /// Create ResNet-34 configuration
    pub fn resnet34(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet34,
            num_classes,
            ..Default::default()
        }
    }

    /// Create ResNet-50 configuration
    pub fn resnet50(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet50,
            num_classes,
            ..Default::default()
        }
    }

    /// Create ResNet-101 configuration
    pub fn resnet101(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet101,
            num_classes,
            ..Default::default()
        }
    }

    /// Create ResNet-152 configuration
    pub fn resnet152(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet152,
            num_classes,
            ..Default::default()
        }
    }

    /// Create SE-ResNet configuration (with Squeeze-and-Excitation)
    pub fn se_resnet(variant: ResNetVariant, num_classes: usize) -> Self {
        Self {
            variant,
            num_classes,
            use_se: true,
            se_reduction_ratio: 16,
            ..Default::default()
        }
    }

    /// Create ResNeXt configuration (grouped convolution)
    pub fn resnext50_32x4d(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet50,
            num_classes,
            groups: 32,
            width_per_group: 4,
            ..Default::default()
        }
    }

    /// Create ResNeXt-101 configuration
    pub fn resnext101_32x8d(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet101,
            num_classes,
            groups: 32,
            width_per_group: 8,
            ..Default::default()
        }
    }

    /// Create Wide ResNet configuration
    pub fn wide_resnet50_2(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet50,
            num_classes,
            width_per_group: 128, // 2x wider
            ..Default::default()
        }
    }

    /// Create Wide ResNet-101 configuration
    pub fn wide_resnet101_2(num_classes: usize) -> Self {
        Self {
            variant: ResNetVariant::ResNet101,
            num_classes,
            width_per_group: 128, // 2x wider
            ..Default::default()
        }
    }

    /// Create configuration from ModelInitConfig
    pub fn from_init_config(variant: ResNetVariant, init_config: ModelInitConfig) -> Self {
        Self {
            variant,
            num_classes: init_config.num_classes,
            dropout: init_config.dropout,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.num_classes == 0 {
            return Err("Number of classes must be greater than 0".to_string());
        }

        if self.in_channels == 0 {
            return Err("Input channels must be greater than 0".to_string());
        }

        if self.stem_channels == 0 {
            return Err("Stem channels must be greater than 0".to_string());
        }

        if self.groups == 0 {
            return Err("Groups must be greater than 0".to_string());
        }

        if self.width_per_group == 0 {
            return Err("Width per group must be greater than 0".to_string());
        }

        if self.se_reduction_ratio == 0 {
            return Err("SE reduction ratio must be greater than 0".to_string());
        }

        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err("Dropout must be in range [0.0, 1.0)".to_string());
        }

        if self.replace_stride_with_dilation.len() != 3 {
            return Err("replace_stride_with_dilation must have exactly 3 elements".to_string());
        }

        Ok(())
    }

    /// Get the expected input size for this configuration
    pub fn input_size(&self) -> (usize, usize, usize) {
        (self.in_channels, 224, 224)
    }

    /// Get the output channels for each stage
    pub fn stage_channels(&self) -> Vec<usize> {
        if self.variant.uses_bottleneck() {
            vec![256, 512, 1024, 2048]
        } else {
            vec![64, 128, 256, 512]
        }
    }

    /// Get the expansion factor for residual blocks
    pub fn expansion(&self) -> usize {
        if self.variant.uses_bottleneck() {
            4
        } else {
            1
        }
    }
}
