//! Normalization layers for neural networks
//!
//! This module provides a comprehensive collection of normalization techniques used in
//! deep learning, organized into logical families for better maintainability and clarity.
//!
//! ## Module Structure
//!
//! The normalization module is organized into specialized sub-modules:
//!
//! - **`common`** - Shared utilities, configurations, and helper functions
//! - **`batch`** - Batch normalization variants (1D, 2D, 3D, synchronized, virtual, renormalization)
//! - **`instance`** - Instance normalization for all dimensions (1D, 2D, 3D)
//! - **`layer_group`** - Layer and group normalization techniques
//! - **`weight_based`** - Weight-based normalization (spectral, weight norm, weight standardization)
//! - **`advanced`** - Advanced adaptive normalization techniques (switchable norm)
//!
//! ## Usage Examples
//!
//! ```rust
//! use torsh_nn::layers::normalization::{
//!     BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm2d, SwitchableNorm2d
//! };
//!
//! // Create different normalization layers
//! let batch_norm = BatchNorm2d::new(64)?;
//! let layer_norm = LayerNorm::new(vec![128])?;
//! let group_norm = GroupNorm::new(8, 64)?;
//! let instance_norm = InstanceNorm2d::new(64)?;
//! let switchable_norm = SwitchableNorm2d::new(64)?;
//! ```
//!
//! ## Configuration and Customization
//!
//! Most normalization layers support custom configuration:
//!
//! ```rust
//! use torsh_nn::layers::normalization::{BatchNorm2d, NormalizationConfig};
//!
//! // Create with custom configuration
//! let config = NormalizationConfig::default()
//!     .with_eps(1e-6)
//!     .with_momentum(0.05);
//!
//! let batch_norm = BatchNorm2d::with_config(64, config)?;
//! ```

// Sub-modules containing different normalization families
pub mod advanced;
pub mod batch;
pub mod common;
pub mod instance;
pub mod layer_group;
pub mod weight_based;

// Re-export common utilities and configurations for convenience
pub use common::{utils, NormalizationConfig, NormalizationStats};

// Re-export all batch normalization variants
pub use batch::{BatchNorm1d, BatchNorm2d, BatchNorm3d};

// Re-export instance normalization variants
pub use instance::{InstanceNorm1d, InstanceNorm2d, InstanceNorm3d};

// Re-export layer and group normalization
pub use layer_group::{GroupNorm, LayerNorm};

// Re-export weight-based normalization techniques
pub use weight_based::{SpectralNorm, WeightNorm, WeightStandardization};

// Re-export advanced normalization techniques
pub use advanced::SwitchableNorm2d;

// Additional backward compatibility aliases for the most commonly used types
pub use BatchNorm2d as BatchNorm;
pub use GroupNorm as GN;
pub use InstanceNorm2d as InstanceNorm;
pub use LayerNorm as LN;

/// Normalization layer factory for common configurations
pub struct NormalizationFactory;

impl NormalizationFactory {
    /// Create a batch normalization layer for 2D inputs (most common)
    pub fn batch_norm(num_features: usize) -> torsh_core::error::Result<BatchNorm2d> {
        BatchNorm2d::new(num_features)
    }

    /// Create a layer normalization for transformer-like architectures
    pub fn layer_norm(normalized_shape: Vec<usize>) -> torsh_core::error::Result<LayerNorm> {
        LayerNorm::new(normalized_shape)
    }

    /// Create a group normalization layer
    pub fn group_norm(
        num_groups: usize,
        num_channels: usize,
    ) -> torsh_core::error::Result<GroupNorm> {
        GroupNorm::new(num_groups, num_channels)
    }

    /// Create an instance normalization layer for 2D inputs
    pub fn instance_norm(num_features: usize) -> torsh_core::error::Result<InstanceNorm2d> {
        InstanceNorm2d::new(num_features)
    }

    /// Create a switchable normalization layer that adapts between different norms
    pub fn switchable_norm(num_features: usize) -> torsh_core::error::Result<SwitchableNorm2d> {
        SwitchableNorm2d::new(num_features)
    }

    /// Create batch normalization optimized for training
    pub fn batch_norm_training(num_features: usize) -> torsh_core::error::Result<BatchNorm2d> {
        BatchNorm2d::with_config(num_features, NormalizationConfig::training())
    }

    /// Create batch normalization optimized for inference
    pub fn batch_norm_inference(num_features: usize) -> torsh_core::error::Result<BatchNorm2d> {
        BatchNorm2d::with_config(num_features, NormalizationConfig::inference())
    }

    /// Create layer normalization without learnable parameters
    pub fn layer_norm_non_affine(
        normalized_shape: Vec<usize>,
    ) -> torsh_core::error::Result<LayerNorm> {
        LayerNorm::with_config(normalized_shape, NormalizationConfig::non_affine())
    }
}

/// Common normalization presets for different architectures
pub struct NormalizationPresets;

impl NormalizationPresets {
    /// ResNet-style batch normalization
    pub fn resnet_batch_norm(num_features: usize) -> torsh_core::error::Result<BatchNorm2d> {
        BatchNorm2d::with_config(num_features, NormalizationConfig::with_momentum(0.1))
    }

    /// Transformer-style layer normalization
    pub fn transformer_layer_norm(hidden_size: usize) -> torsh_core::error::Result<LayerNorm> {
        LayerNorm::with_config(vec![hidden_size], NormalizationConfig::with_eps(1e-12))
    }

    /// Style transfer instance normalization (non-affine)
    pub fn style_transfer_instance_norm(
        num_features: usize,
    ) -> torsh_core::error::Result<InstanceNorm2d> {
        InstanceNorm2d::with_config(num_features, NormalizationConfig::non_affine())
    }

    /// Group normalization for small batch training
    pub fn small_batch_group_norm(num_channels: usize) -> torsh_core::error::Result<GroupNorm> {
        let num_groups = if num_channels >= 32 { 32 } else { num_channels };
        GroupNorm::new(num_groups, num_channels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_normalization_factory() {
        // Test factory methods
        let bn = NormalizationFactory::batch_norm(64).unwrap();
        assert_eq!(bn.num_features(), 64);

        let ln = NormalizationFactory::layer_norm(vec![128]).unwrap();
        assert_eq!(ln.normalized_shape(), &[128]);

        let gn = NormalizationFactory::group_norm(8, 64).unwrap();
        assert_eq!(gn.num_groups(), 8);
        assert_eq!(gn.num_channels(), 64);

        let inn = NormalizationFactory::instance_norm(32).unwrap();
        assert_eq!(inn.num_features(), 32);

        let sn = NormalizationFactory::switchable_norm(16).unwrap();
        assert_eq!(sn.num_features(), 16);
    }

    #[test]
    fn test_normalization_presets() {
        // Test preset configurations
        let resnet_bn = NormalizationPresets::resnet_batch_norm(64).unwrap();
        assert_eq!(resnet_bn.momentum(), 0.1);

        let transformer_ln = NormalizationPresets::transformer_layer_norm(768).unwrap();
        assert_eq!(transformer_ln.eps(), 1e-12);

        let style_in = NormalizationPresets::style_transfer_instance_norm(64).unwrap();
        // Non-affine should not have weight/bias parameters
        assert!(style_in.parameters().is_empty());

        let small_batch_gn = NormalizationPresets::small_batch_group_norm(64).unwrap();
        assert_eq!(small_batch_gn.num_groups(), 32);
    }

    #[test]
    fn test_module_integration() {
        // Test that different normalization layers work with sample inputs
        let input_2d = zeros(&[4, 64]).unwrap();
        let input_4d = zeros(&[4, 64, 32, 32]).unwrap();

        // Test BatchNorm2d
        let bn2d = BatchNorm2d::new(64).unwrap();
        assert!(bn2d.forward(&input_4d).is_ok());

        // Test BatchNorm1d
        let bn1d = BatchNorm1d::new(64).unwrap();
        assert!(bn1d.forward(&input_2d).is_ok());

        // Test LayerNorm
        let ln = LayerNorm::new(vec![64]).unwrap();
        assert!(ln.forward(&input_2d).is_ok());

        // Test GroupNorm
        let gn = GroupNorm::new(8, 64).unwrap();
        assert!(gn.forward(&input_4d).is_ok());

        // Test InstanceNorm2d
        let in2d = InstanceNorm2d::new(64).unwrap();
        assert!(in2d.forward(&input_4d).is_ok());
    }

    #[test]
    fn test_backward_compatibility_aliases() {
        // Test that aliases work correctly
        let bn = BatchNorm::new(32).unwrap();
        assert_eq!(bn.num_features(), 32);

        let ln = LN::new(vec![128]).unwrap();
        assert_eq!(ln.normalized_shape(), &[128]);

        let gn = GN::new(4, 32).unwrap();
        assert_eq!(gn.num_groups(), 4);

        let inn = InstanceNorm::new(16).unwrap();
        assert_eq!(inn.num_features(), 16);
    }

    #[test]
    fn test_configuration_variants() {
        // Test different configuration variants
        let training_config = NormalizationConfig::training();
        assert!(training_config.track_running_stats);
        assert!(training_config.affine);

        let inference_config = NormalizationConfig::inference();
        assert!(!inference_config.track_running_stats);

        let non_affine_config = NormalizationConfig::non_affine();
        assert!(!non_affine_config.affine);

        let custom_eps_config = NormalizationConfig::with_eps(1e-8);
        assert_eq!(custom_eps_config.eps, 1e-8);

        let custom_momentum_config = NormalizationConfig::with_momentum(0.05);
        assert_eq!(custom_momentum_config.momentum, 0.05);
    }
}
