//! 3D Vision model implementations for ToRSh
//!
//! This module provides various 3D vision architectures including
//! 3D Convolutional Neural Networks, PointNet, and PointNet++ for
//! processing 3D data such as point clouds and volumetric data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::DeviceType;
use torsh_nn::prelude::{BatchNorm1d, BatchNorm3d, Conv3d, Dropout, Linear, MaxPool3d};
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Configuration for 3D Convolutional Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNN3DConfig {
    pub input_channels: usize,
    pub num_classes: usize,
    pub conv_channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub paddings: Vec<usize>,
    pub pool_sizes: Vec<usize>,
    pub fc_hidden_dims: Vec<usize>,
    pub dropout_rate: f64,
    pub use_batch_norm: bool,
}

impl Default for CNN3DConfig {
    fn default() -> Self {
        Self {
            input_channels: 1,
            num_classes: 10,
            conv_channels: vec![64, 128, 256, 512],
            kernel_sizes: vec![3, 3, 3, 3],
            strides: vec![1, 1, 1, 1],
            paddings: vec![1, 1, 1, 1],
            pool_sizes: vec![2, 2, 2, 2],
            fc_hidden_dims: vec![1024, 512],
            dropout_rate: 0.5,
            use_batch_norm: true,
        }
    }
}

/// 3D Convolutional Neural Network
pub struct CNN3D {
    conv_layers: Vec<Conv3d>,
    batch_norms: Vec<Option<BatchNorm3d>>,
    pool_layers: Vec<MaxPool3d>,
    fc_layers: Vec<Linear>,
    dropout: Dropout,
    config: CNN3DConfig,
}

impl CNN3D {
    pub fn new(config: CNN3DConfig) -> torsh_core::error::Result<Self> {
        let mut conv_layers = Vec::new();
        let mut batch_norms = Vec::new();
        let mut pool_layers = Vec::new();

        let mut in_channels = config.input_channels;
        for (i, &out_channels) in config.conv_channels.iter().enumerate() {
            let kernel_size = config.kernel_sizes.get(i).copied().unwrap_or(3);
            let stride = config.strides.get(i).copied().unwrap_or(1);
            let padding = config.paddings.get(i).copied().unwrap_or(1);
            let pool_size = config.pool_sizes.get(i).copied().unwrap_or(2);

            conv_layers.push(Conv3d::new(
                in_channels,
                out_channels,
                (kernel_size, kernel_size, kernel_size),
                (stride, stride, stride),
                (padding, padding, padding),
                (1, 1, 1),
                true,
                1,
            ));

            if config.use_batch_norm {
                batch_norms.push(Some(BatchNorm3d::new(out_channels)?));
            } else {
                batch_norms.push(None);
            }

            pool_layers.push(MaxPool3d::new(
                (pool_size, pool_size, pool_size),
                Some((2, 2, 2)),
                (0, 0, 0),
                (1, 1, 1),
                false,
            ));
            in_channels = out_channels;
        }

        // Fully connected layers
        let mut fc_layers = Vec::new();
        let mut fc_input_dim = in_channels; // This would need to be calculated based on input size

        for &hidden_dim in &config.fc_hidden_dims {
            fc_layers.push(Linear::new(fc_input_dim, hidden_dim, true));
            fc_input_dim = hidden_dim;
        }

        // Output layer
        fc_layers.push(Linear::new(fc_input_dim, config.num_classes, true));

        Ok(Self {
            conv_layers,
            batch_norms,
            pool_layers,
            fc_layers,
            dropout: Dropout::new(config.dropout_rate as f32),
            config,
        })
    }
}

impl Module for CNN3D {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let mut x = input.clone();

        // Convolutional layers
        for (i, conv) in self.conv_layers.iter().enumerate() {
            x = conv.forward(&x)?;

            if let Some(bn) = &self.batch_norms[i] {
                x = bn.forward(&x)?;
            }

            x = x.relu()?;
            x = self.pool_layers[i].forward(&x)?;
        }

        // Flatten for fully connected layers
        let batch_size = x.size(0)? as i32;
        x = x.view(&[batch_size, -1])?;

        // Fully connected layers
        for (i, fc) in self.fc_layers.iter().enumerate() {
            x = fc.forward(&x)?;

            if i < self.fc_layers.len() - 1 {
                x = x.relu()?;
                x = self.dropout.forward(&x)?;
            }
        }

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, conv) in self.conv_layers.iter().enumerate() {
            for (name, param) in conv.parameters() {
                params.insert(format!("conv_{}.{}", i, name), param);
            }
        }

        for (i, bn_opt) in self.batch_norms.iter().enumerate() {
            if let Some(bn) = bn_opt {
                for (name, param) in bn.parameters() {
                    params.insert(format!("bn_{}.{}", i, name), param);
                }
            }
        }

        for (i, fc) in self.fc_layers.iter().enumerate() {
            for (name, param) in fc.parameters() {
                params.insert(format!("fc_{}.{}", i, name), param);
            }
        }

        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        for conv in &mut self.conv_layers {
            conv.train();
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.train();
            }
        }
        for fc in &mut self.fc_layers {
            fc.train();
        }
        self.dropout.train();
    }

    fn eval(&mut self) {
        for conv in &mut self.conv_layers {
            conv.eval();
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.eval();
            }
        }
        for fc in &mut self.fc_layers {
            fc.eval();
        }
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for conv in &mut self.conv_layers {
            conv.to_device(device)?;
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.to_device(device)?;
            }
        }
        for fc in &mut self.fc_layers {
            fc.to_device(device)?;
        }
        self.dropout.to_device(device)
    }
}

/// Configuration for PointNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointNetConfig {
    pub input_dim: usize,
    pub num_classes: usize,
    pub point_net_dims: Vec<usize>,
    pub global_feat_dim: usize,
    pub classifier_dims: Vec<usize>,
    pub dropout_rate: f64,
    pub use_batch_norm: bool,
    pub use_transform_net: bool,
}

impl Default for PointNetConfig {
    fn default() -> Self {
        Self {
            input_dim: 3, // x, y, z coordinates
            num_classes: 40,
            point_net_dims: vec![64, 128, 1024],
            global_feat_dim: 1024,
            classifier_dims: vec![512, 256],
            dropout_rate: 0.3,
            use_batch_norm: true,
            use_transform_net: true,
        }
    }
}

/// Transformation Network for PointNet
pub struct TransformNet {
    conv_layers: Vec<Linear>,
    batch_norms: Vec<Option<BatchNorm1d>>,
    fc_layers: Vec<Linear>,
    output_dim: usize,
}

impl TransformNet {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        use_batch_norm: bool,
    ) -> torsh_core::error::Result<Self> {
        let conv_dims = vec![64, 128, 1024];
        let mut conv_layers = Vec::new();
        let mut batch_norms = Vec::new();

        let mut in_dim = input_dim;
        for &out_dim in &conv_dims {
            conv_layers.push(Linear::new(in_dim, out_dim, true));
            if use_batch_norm {
                batch_norms.push(Some(BatchNorm1d::new(out_dim)?));
            } else {
                batch_norms.push(None);
            }
            in_dim = out_dim;
        }

        let fc_layers = vec![
            Linear::new(1024, 512, true),
            Linear::new(512, 256, true),
            Linear::new(256, output_dim * output_dim, true),
        ];

        Ok(Self {
            conv_layers,
            batch_norms,
            fc_layers,
            output_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> torsh_core::error::Result<Tensor> {
        let mut feat = x.clone();

        // Point-wise convolutions
        for (i, conv) in self.conv_layers.iter().enumerate() {
            feat = conv.forward(&feat)?;

            if let Some(bn) = &self.batch_norms[i] {
                feat = bn.forward(&feat)?;
            }

            feat = feat.relu()?;
        }

        // Global max pooling
        feat = feat.max(Some(1), true)?;

        // Fully connected layers
        for (i, fc) in self.fc_layers.iter().enumerate() {
            feat = fc.forward(&feat)?;

            if i < self.fc_layers.len() - 1 {
                feat = feat.relu()?;
            }
        }

        // Reshape to transformation matrix
        let batch_size = feat.size(0)? as i32;
        let transform = feat.view(&[batch_size, self.output_dim as i32, self.output_dim as i32])?;

        // Add identity matrix
        let identity = creation::eye(self.output_dim)?;
        let identity = identity
            .unsqueeze(0)?
            .repeat(&[batch_size as usize, 1, 1])?;

        transform.add(&identity)
    }
}

impl Module for TransformNet {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, conv) in self.conv_layers.iter().enumerate() {
            for (name, param) in conv.parameters() {
                params.insert(format!("conv_{}.{}", i, name), param);
            }
        }

        for (i, bn_opt) in self.batch_norms.iter().enumerate() {
            if let Some(bn) = bn_opt {
                for (name, param) in bn.parameters() {
                    params.insert(format!("bn_{}.{}", i, name), param);
                }
            }
        }

        for (i, fc) in self.fc_layers.iter().enumerate() {
            for (name, param) in fc.parameters() {
                params.insert(format!("fc_{}.{}", i, name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv_layers.iter().any(|layer| layer.training())
    }

    fn train(&mut self) {
        for conv in &mut self.conv_layers {
            conv.train();
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.train();
            }
        }
        for fc in &mut self.fc_layers {
            fc.train();
        }
    }

    fn eval(&mut self) {
        for conv in &mut self.conv_layers {
            conv.eval();
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.eval();
            }
        }
        for fc in &mut self.fc_layers {
            fc.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for conv in &mut self.conv_layers {
            conv.to_device(device)?;
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.to_device(device)?;
            }
        }
        for fc in &mut self.fc_layers {
            fc.to_device(device)?;
        }
        Ok(())
    }
}

/// PointNet for point cloud classification
pub struct PointNet {
    input_transform: Option<TransformNet>,
    feature_transform: Option<TransformNet>,
    point_conv_layers: Vec<Linear>,
    point_batch_norms: Vec<Option<BatchNorm1d>>,
    classifier: Vec<Linear>,
    dropout: Dropout,
    config: PointNetConfig,
}

impl PointNet {
    pub fn new(config: PointNetConfig) -> torsh_core::error::Result<Self> {
        let input_transform = if config.use_transform_net {
            Some(TransformNet::new(
                config.input_dim,
                config.input_dim,
                config.use_batch_norm,
            )?)
        } else {
            None
        };

        let mut point_conv_layers = Vec::new();
        let mut point_batch_norms = Vec::new();

        let mut in_dim = config.input_dim;
        for &out_dim in &config.point_net_dims {
            point_conv_layers.push(Linear::new(in_dim, out_dim, true));
            if config.use_batch_norm {
                point_batch_norms.push(Some(BatchNorm1d::new(out_dim)?));
            } else {
                point_batch_norms.push(None);
            }
            in_dim = out_dim;
        }

        let feature_transform = if config.use_transform_net && config.point_net_dims.len() > 0 {
            Some(TransformNet::new(
                config.point_net_dims[0],
                config.point_net_dims[0],
                config.use_batch_norm,
            )?)
        } else {
            None
        };

        // Classifier
        let mut classifier = Vec::new();
        let mut classifier_in_dim = config.global_feat_dim;

        for &hidden_dim in &config.classifier_dims {
            classifier.push(Linear::new(classifier_in_dim, hidden_dim, true));
            classifier_in_dim = hidden_dim;
        }

        classifier.push(Linear::new(classifier_in_dim, config.num_classes, true));

        Ok(Self {
            input_transform,
            feature_transform,
            point_conv_layers,
            point_batch_norms,
            classifier,
            dropout: Dropout::new(config.dropout_rate as f32),
            config,
        })
    }

    pub fn forward(&self, points: &Tensor) -> torsh_core::error::Result<Tensor> {
        let batch_size = points.size(0);
        let num_points = points.size(1);

        let mut x = points.clone();

        // Input transformation
        if let Some(input_transform) = &self.input_transform {
            let transform = input_transform.forward(&x)?;
            x = x.matmul(&transform)?;
        }

        // Point-wise feature extraction
        for (i, conv) in self.point_conv_layers.iter().enumerate() {
            x = conv.forward(&x)?;

            if let Some(bn) = &self.point_batch_norms[i] {
                x = bn.forward(&x)?;
            }

            x = x.relu()?;

            // Feature transformation after first layer
            if i == 0 {
                if let Some(feature_transform) = &self.feature_transform {
                    let transform = feature_transform.forward(&x)?;
                    x = x.matmul(&transform)?;
                }
            }
        }

        // Global max pooling
        let global_feat = x.max(Some(1), true)?; // Shape: [batch_size, feature_dim]

        // Classification
        let mut output = global_feat;
        for (i, fc) in self.classifier.iter().enumerate() {
            output = fc.forward(&output)?;

            if i < self.classifier.len() - 1 {
                output = output.relu()?;
                output = self.dropout.forward(&output)?;
            }
        }

        Ok(output)
    }
}

impl Module for PointNet {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        if let Some(input_transform) = &self.input_transform {
            for (name, param) in input_transform.parameters() {
                params.insert(format!("input_transform.{}", name), param);
            }
        }

        if let Some(feature_transform) = &self.feature_transform {
            for (name, param) in feature_transform.parameters() {
                params.insert(format!("feature_transform.{}", name), param);
            }
        }

        for (i, conv) in self.point_conv_layers.iter().enumerate() {
            for (name, param) in conv.parameters() {
                params.insert(format!("point_conv_{}.{}", i, name), param);
            }
        }

        for (i, bn_opt) in self.point_batch_norms.iter().enumerate() {
            if let Some(bn) = bn_opt {
                for (name, param) in bn.parameters() {
                    params.insert(format!("point_bn_{}.{}", i, name), param);
                }
            }
        }

        for (i, fc) in self.classifier.iter().enumerate() {
            for (name, param) in fc.parameters() {
                params.insert(format!("classifier_{}.{}", i, name), param);
            }
        }

        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        if let Some(input_transform) = &mut self.input_transform {
            input_transform.train();
        }
        if let Some(feature_transform) = &mut self.feature_transform {
            feature_transform.train();
        }
        for conv in &mut self.point_conv_layers {
            conv.train();
        }
        for bn_opt in &mut self.point_batch_norms {
            if let Some(bn) = bn_opt {
                bn.train();
            }
        }
        for fc in &mut self.classifier {
            fc.train();
        }
        self.dropout.train();
    }

    fn eval(&mut self) {
        if let Some(input_transform) = &mut self.input_transform {
            input_transform.eval();
        }
        if let Some(feature_transform) = &mut self.feature_transform {
            feature_transform.eval();
        }
        for conv in &mut self.point_conv_layers {
            conv.eval();
        }
        for bn_opt in &mut self.point_batch_norms {
            if let Some(bn) = bn_opt {
                bn.eval();
            }
        }
        for fc in &mut self.classifier {
            fc.eval();
        }
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        if let Some(input_transform) = &mut self.input_transform {
            input_transform.to_device(device)?;
        }
        if let Some(feature_transform) = &mut self.feature_transform {
            feature_transform.to_device(device)?;
        }
        for conv in &mut self.point_conv_layers {
            conv.to_device(device)?;
        }
        for bn_opt in &mut self.point_batch_norms {
            if let Some(bn) = bn_opt {
                bn.to_device(device)?;
            }
        }
        for fc in &mut self.classifier {
            fc.to_device(device)?;
        }
        self.dropout.to_device(device)
    }
}

/// Configuration for PointNet++
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointNetPlusPlusConfig {
    pub input_dim: usize,
    pub num_classes: usize,
    pub num_points: Vec<usize>,
    pub radius: Vec<f32>,
    pub num_samples: Vec<usize>,
    pub mlp_dims: Vec<Vec<usize>>,
    pub dropout_rate: f64,
    pub use_batch_norm: bool,
}

impl Default for PointNetPlusPlusConfig {
    fn default() -> Self {
        Self {
            input_dim: 3,
            num_classes: 40,
            num_points: vec![512, 128],
            radius: vec![0.2, 0.4],
            num_samples: vec![32, 64],
            mlp_dims: vec![vec![64, 64, 128], vec![128, 128, 256]],
            dropout_rate: 0.5,
            use_batch_norm: true,
        }
    }
}

/// Set Abstraction Layer for PointNet++
pub struct SetAbstractionLayer {
    mlp_layers: Vec<Linear>,
    batch_norms: Vec<Option<BatchNorm1d>>,
    num_points: usize,
    radius: f32,
    num_samples: usize,
}

impl SetAbstractionLayer {
    pub fn new(
        input_dim: usize,
        mlp_dims: Vec<usize>,
        num_points: usize,
        radius: f32,
        num_samples: usize,
        use_batch_norm: bool,
    ) -> torsh_core::error::Result<Self> {
        let mut mlp_layers = Vec::new();
        let mut batch_norms = Vec::new();

        let mut in_dim = input_dim;
        for &out_dim in &mlp_dims {
            mlp_layers.push(Linear::new(in_dim, out_dim, true));
            if use_batch_norm {
                batch_norms.push(Some(BatchNorm1d::new(out_dim)?));
            } else {
                batch_norms.push(None);
            }
            in_dim = out_dim;
        }

        Ok(Self {
            mlp_layers,
            batch_norms,
            num_points,
            radius,
            num_samples,
        })
    }

    pub fn forward(
        &self,
        xyz: &Tensor,
        features: Option<&Tensor>,
    ) -> torsh_core::error::Result<(Tensor, Tensor)> {
        // Simplified implementation - in practice, would need proper sampling and grouping
        let batch_size = xyz.size(0);
        let current_points = xyz.size(1);

        // Sample points (simplified as taking first num_points)
        let sample_indices: Vec<i64> = (0..self.num_points.min(current_points?))
            .map(|x| x as i64)
            .collect();
        let new_xyz = xyz.index_select(
            1,
            &Tensor::from_data(
                sample_indices.clone(),
                vec![sample_indices.len()],
                xyz.device(),
            )?,
        )?;

        // Group features (simplified)
        let mut grouped_features = if let Some(feat) = features {
            feat.index_select(
                1,
                &Tensor::from_data(
                    sample_indices.clone(),
                    vec![sample_indices.len()],
                    feat.device(),
                )?,
            )?
        } else {
            new_xyz.clone()
        };

        // Apply MLPs
        for (i, mlp) in self.mlp_layers.iter().enumerate() {
            grouped_features = mlp.forward(&grouped_features)?;

            if let Some(bn) = &self.batch_norms[i] {
                grouped_features = bn.forward(&grouped_features)?;
            }

            grouped_features = grouped_features.relu()?;
        }

        // Max pooling within groups
        let new_features = grouped_features.max(Some(2), true)?;

        Ok((new_xyz, new_features))
    }
}

impl Module for SetAbstractionLayer {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let (_new_xyz, new_features) = self.forward(input, None)?;
        Ok(new_features)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, mlp) in self.mlp_layers.iter().enumerate() {
            for (name, param) in mlp.parameters() {
                params.insert(format!("mlp_{}.{}", i, name), param);
            }
        }

        for (i, bn_opt) in self.batch_norms.iter().enumerate() {
            if let Some(bn) = bn_opt {
                for (name, param) in bn.parameters() {
                    params.insert(format!("bn_{}.{}", i, name), param);
                }
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.mlp_layers.iter().any(|layer| layer.training())
    }

    fn train(&mut self) {
        for mlp in &mut self.mlp_layers {
            mlp.train();
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.train();
            }
        }
    }

    fn eval(&mut self) {
        for mlp in &mut self.mlp_layers {
            mlp.eval();
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.eval();
            }
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for mlp in &mut self.mlp_layers {
            mlp.to_device(device)?;
        }
        for bn_opt in &mut self.batch_norms {
            if let Some(bn) = bn_opt {
                bn.to_device(device)?;
            }
        }
        Ok(())
    }
}

/// PointNet++ for hierarchical point cloud processing
pub struct PointNetPlusPlus {
    set_abstraction_layers: Vec<SetAbstractionLayer>,
    classifier: Vec<Linear>,
    dropout: Dropout,
    config: PointNetPlusPlusConfig,
}

impl PointNetPlusPlus {
    pub fn new(config: PointNetPlusPlusConfig) -> torsh_core::error::Result<Self> {
        let mut set_abstraction_layers = Vec::new();
        let mut input_dim = config.input_dim;

        for (i, &num_points) in config.num_points.iter().enumerate() {
            let radius = config.radius.get(i).copied().unwrap_or(0.2);
            let num_samples = config.num_samples.get(i).copied().unwrap_or(32);
            let mlp_dims = config
                .mlp_dims
                .get(i)
                .cloned()
                .unwrap_or_else(|| vec![64, 128]);

            set_abstraction_layers.push(SetAbstractionLayer::new(
                input_dim,
                mlp_dims.clone(),
                num_points,
                radius,
                num_samples,
                config.use_batch_norm,
            )?);

            input_dim = mlp_dims.last().copied().unwrap_or(128);
        }

        // Global feature extraction
        let global_mlp_dims = vec![256, 512, 1024];
        for &dim in &global_mlp_dims {
            set_abstraction_layers.push(SetAbstractionLayer::new(
                input_dim,
                vec![dim],
                1, // Single global feature
                f32::INFINITY,
                input_dim,
                config.use_batch_norm,
            )?);
            input_dim = dim;
        }

        // Classifier
        let classifier = vec![
            Linear::new(1024, 512, true),
            Linear::new(512, 256, true),
            Linear::new(256, config.num_classes, true),
        ];

        Ok(Self {
            set_abstraction_layers,
            classifier,
            dropout: Dropout::new(config.dropout_rate as f32),
            config,
        })
    }
}

impl Module for PointNetPlusPlus {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let mut xyz = input.clone();
        let mut features = None;

        // Set abstraction layers
        for sa_layer in &self.set_abstraction_layers {
            let (new_xyz, new_features) = sa_layer.forward(&xyz, features.as_ref())?;
            xyz = new_xyz;
            features = Some(new_features);
        }

        // Global feature should be [batch_size, feature_dim]
        let mut global_feat = features.unwrap();
        if global_feat.shape().dims().len() == 3 {
            global_feat = global_feat.squeeze(1)?;
        }

        // Classification
        let mut output = global_feat;
        for (i, fc) in self.classifier.iter().enumerate() {
            output = fc.forward(&output)?;

            if i < self.classifier.len() - 1 {
                output = output.relu()?;
                output = self.dropout.forward(&output)?;
            }
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, sa_layer) in self.set_abstraction_layers.iter().enumerate() {
            for (name, param) in sa_layer.parameters() {
                params.insert(format!("sa_{}.{}", i, name), param);
            }
        }

        for (i, fc) in self.classifier.iter().enumerate() {
            for (name, param) in fc.parameters() {
                params.insert(format!("classifier_{}.{}", i, name), param);
            }
        }

        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        for sa_layer in &mut self.set_abstraction_layers {
            sa_layer.train();
        }
        for fc in &mut self.classifier {
            fc.train();
        }
        self.dropout.train();
    }

    fn eval(&mut self) {
        for sa_layer in &mut self.set_abstraction_layers {
            sa_layer.eval();
        }
        for fc in &mut self.classifier {
            fc.eval();
        }
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for sa_layer in &mut self.set_abstraction_layers {
            sa_layer.to_device(device)?;
        }
        for fc in &mut self.classifier {
            fc.to_device(device)?;
        }
        self.dropout.to_device(device)
    }
}

/// 3D Vision Architecture enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Vision3DArchitecture {
    CNN3D,
    PointNet,
    PointNetPlusPlus,
}

/// Unified 3D Vision model type
pub enum Vision3DModel {
    CNN3D(CNN3D),
    PointNet(PointNet),
    PointNetPlusPlus(PointNetPlusPlus),
}

impl Module for Vision3DModel {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        match self {
            Vision3DModel::CNN3D(model) => model.forward(input),
            Vision3DModel::PointNet(model) => model.forward(input),
            Vision3DModel::PointNetPlusPlus(model) => model.forward(input),
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        match self {
            Vision3DModel::CNN3D(model) => model.parameters(),
            Vision3DModel::PointNet(model) => model.parameters(),
            Vision3DModel::PointNetPlusPlus(model) => model.parameters(),
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        match self {
            Vision3DModel::CNN3D(model) => model.named_parameters(),
            Vision3DModel::PointNet(model) => model.named_parameters(),
            Vision3DModel::PointNetPlusPlus(model) => model.named_parameters(),
        }
    }

    fn training(&self) -> bool {
        match self {
            Vision3DModel::CNN3D(model) => model.training(),
            Vision3DModel::PointNet(model) => model.training(),
            Vision3DModel::PointNetPlusPlus(model) => model.training(),
        }
    }

    fn train(&mut self) {
        match self {
            Vision3DModel::CNN3D(model) => model.train(),
            Vision3DModel::PointNet(model) => model.train(),
            Vision3DModel::PointNetPlusPlus(model) => model.train(),
        }
    }

    fn eval(&mut self) {
        match self {
            Vision3DModel::CNN3D(model) => model.eval(),
            Vision3DModel::PointNet(model) => model.eval(),
            Vision3DModel::PointNetPlusPlus(model) => model.eval(),
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        match self {
            Vision3DModel::CNN3D(model) => model.to_device(device),
            Vision3DModel::PointNet(model) => model.to_device(device),
            Vision3DModel::PointNetPlusPlus(model) => model.to_device(device),
        }
    }
}
