//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::layers::conv::*;
    use crate::Module;
    use torsh_core::device::DeviceType;
    use torsh_core::error::Result;
    use torsh_tensor::creation::zeros;
    #[test]
    fn test_conv1d_new() {
        let conv = Conv1d::new(3, 16, 3, 1, 0, 1, true, 1);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 0);
        assert_eq!(conv.groups, 1);
        assert!(conv.use_bias);
    }
    #[test]
    fn test_conv1d_with_defaults() {
        let conv = Conv1d::with_defaults(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 0);
    }
    #[test]
    fn test_conv1d_forward() -> Result<()> {
        let conv = Conv1d::with_defaults(3, 16, 3);
        let input = zeros(&[2, 3, 32])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 16, 30]);
        Ok(())
    }
    #[test]
    fn test_conv1d_forward_with_stride() -> Result<()> {
        let conv = Conv1d::new(1, 1, 3, 2, 0, 1, false, 1);
        let input = zeros(&[1, 1, 10])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4]);
        Ok(())
    }
    #[test]
    fn test_conv1d_forward_with_padding() -> Result<()> {
        let conv = Conv1d::new(1, 1, 3, 1, 1, 1, false, 1);
        let input = zeros(&[1, 1, 10])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 10]);
        Ok(())
    }
    #[test]
    fn test_conv1d_parameters() {
        let conv = Conv1d::with_defaults(3, 16, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
    #[test]
    fn test_conv1d_parameters_no_bias() {
        let conv = Conv1d::new(3, 16, 3, 1, 0, 1, false, 1);
        let params = conv.parameters();
        assert_eq!(params.len(), 1);
        assert!(params.contains_key("weight"));
        assert!(!params.contains_key("bias"));
    }
    #[test]
    fn test_conv1d_training_mode() {
        let mut conv = Conv1d::with_defaults(3, 16, 3);
        assert!(conv.training());
        conv.eval();
        assert!(!conv.training());
        conv.train();
        assert!(conv.training());
    }
    #[test]
    fn test_conv2d_new() {
        let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (0, 0), (1, 1), true, 1);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.dilation, (1, 1));
        assert_eq!(conv.groups, 1);
    }
    #[test]
    fn test_conv2d_with_defaults() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3));
    }
    #[test]
    fn test_conv2d_forward() -> Result<()> {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let input = zeros(&[2, 3, 32, 32])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 16, 30, 30]);
        Ok(())
    }
    #[test]
    fn test_conv2d_forward_with_stride() -> Result<()> {
        let conv = Conv2d::new(1, 1, (3, 3), (2, 2), (0, 0), (1, 1), false, 1);
        let input = zeros(&[1, 1, 8, 8])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 3, 3]);
        Ok(())
    }
    #[test]
    fn test_conv2d_forward_with_padding() -> Result<()> {
        let conv = Conv2d::new(1, 1, (3, 3), (1, 1), (1, 1), (1, 1), false, 1);
        let input = zeros(&[1, 1, 8, 8])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8, 8]);
        Ok(())
    }
    #[test]
    fn test_conv2d_forward_invalid_input() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let input = zeros(&[2, 3, 32]).unwrap();
        let result = conv.forward(&input);
        assert!(result.is_err());
    }
    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
    #[test]
    fn test_conv3d_new() {
        let conv = Conv3d::new(3, 16, (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1), true, 1);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3, 3));
        assert_eq!(conv.stride, (1, 1, 1));
        assert_eq!(conv.padding, (0, 0, 0));
    }
    #[test]
    fn test_conv3d_with_defaults() {
        let conv = Conv3d::with_defaults(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3, 3));
    }
    #[test]
    fn test_conv3d_forward() -> Result<()> {
        let conv = Conv3d::with_defaults(1, 8, 3);
        let input = zeros(&[1, 1, 16, 16, 16])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 8, 14, 14, 14]);
        Ok(())
    }
    #[test]
    fn test_conv3d_forward_with_stride() -> Result<()> {
        let conv = Conv3d::new(1, 1, (3, 3, 3), (2, 2, 2), (0, 0, 0), (1, 1, 1), false, 1);
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 3, 3, 3]);
        Ok(())
    }
    #[test]
    fn test_conv3d_parameters() {
        let conv = Conv3d::with_defaults(1, 8, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
    #[test]
    fn test_convtranspose1d_new() {
        let conv = ConvTranspose1d::new(16, 3, 3, 1, 0, 0, 1, true, 1);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 0);
        assert_eq!(conv.output_padding, 0);
    }
    #[test]
    fn test_convtranspose1d_with_defaults() {
        let conv = ConvTranspose1d::with_defaults(16, 3, 3);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, 3);
    }
    #[test]
    fn test_convtranspose1d_forward() -> Result<()> {
        let conv = ConvTranspose1d::with_defaults(16, 8, 4);
        let input = zeros(&[2, 16, 16])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 8, 19]);
        Ok(())
    }
    #[test]
    fn test_convtranspose1d_forward_with_stride() -> Result<()> {
        let conv = ConvTranspose1d::new(1, 1, 4, 2, 1, 0, 1, false, 1);
        let input = zeros(&[1, 1, 8])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 16]);
        Ok(())
    }
    #[test]
    fn test_convtranspose1d_parameters() {
        let conv = ConvTranspose1d::with_defaults(16, 3, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
    #[test]
    fn test_convtranspose2d_new() {
        let conv = ConvTranspose2d::new(16, 3, (3, 3), (1, 1), (0, 0), (0, 0), (1, 1), true, 1);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.output_padding, (0, 0));
    }
    #[test]
    fn test_convtranspose2d_with_defaults() {
        let conv = ConvTranspose2d::with_defaults(16, 3, 3);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3));
    }
    #[test]
    fn test_convtranspose2d_forward() -> Result<()> {
        let conv = ConvTranspose2d::with_defaults(16, 8, 4);
        let input = zeros(&[2, 16, 8, 8])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 8, 11, 11]);
        Ok(())
    }
    #[test]
    fn test_convtranspose2d_forward_with_stride() -> Result<()> {
        let conv = ConvTranspose2d::new(1, 1, (4, 4), (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
        let input = zeros(&[1, 1, 4, 4])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8, 8]);
        Ok(())
    }
    #[test]
    fn test_convtranspose2d_parameters() {
        let conv = ConvTranspose2d::with_defaults(16, 3, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
    #[test]
    fn test_convtranspose3d_new() {
        let conv = ConvTranspose3d::new(
            16,
            3,
            (3, 3, 3),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            true,
            1,
        );
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3, 3));
        assert_eq!(conv.stride, (1, 1, 1));
        assert_eq!(conv.padding, (0, 0, 0));
        assert_eq!(conv.output_padding, (0, 0, 0));
    }
    #[test]
    fn test_convtranspose3d_with_defaults() {
        let conv = ConvTranspose3d::with_defaults(16, 3, 3);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3, 3));
    }
    #[test]
    fn test_convtranspose3d_forward() -> Result<()> {
        let conv = ConvTranspose3d::with_defaults(8, 4, 4);
        let input = zeros(&[1, 8, 4, 4, 4])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 4, 7, 7, 7]);
        Ok(())
    }
    #[test]
    fn test_convtranspose3d_forward_with_stride() -> Result<()> {
        let conv = ConvTranspose3d::new(
            1,
            1,
            (4, 4, 4),
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            false,
            1,
        );
        let input = zeros(&[1, 1, 2, 2, 2])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_convtranspose3d_parameters() {
        let conv = ConvTranspose3d::with_defaults(16, 3, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
    #[test]
    fn test_module_training_modes() {
        let mut conv = Conv2d::with_defaults(3, 16, 3);
        assert!(conv.training());
        conv.set_training(false);
        assert!(!conv.training());
        conv.set_training(true);
        assert!(conv.training());
    }
    #[test]
    fn test_module_named_parameters() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let named_params = conv.named_parameters();
        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));
    }
    #[test]
    fn test_module_to_device() -> Result<()> {
        let mut conv = Conv2d::with_defaults(3, 16, 3);
        conv.to_device(DeviceType::Cpu)?;
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_new() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        assert_eq!(conv.in_channels(), 32);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.stride(), 1);
        assert_eq!(conv.padding(), 1);
    }
    #[test]
    fn test_depthwise_separable_conv_with_defaults() {
        let conv = DepthwiseSeparableConv::with_defaults(16, 32, 3);
        assert_eq!(conv.in_channels(), 16);
        assert_eq!(conv.out_channels(), 32);
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.stride(), 1);
        assert_eq!(conv.padding(), 1);
    }
    #[test]
    fn test_depthwise_separable_conv_forward() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let input = zeros(&[2, 32, 8, 8])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 64, 8, 8]);
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_kernel_size_3x3() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, false);
        let input = zeros(&[1, 16, 16, 16])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 32, 16, 16]);
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_kernel_size_5x5() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(24, 48, 5, 1, 2, true);
        let input = zeros(&[1, 24, 32, 32])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 48, 32, 32]);
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_stride_2() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 2, 1, true);
        let input = zeros(&[1, 32, 16, 16])?;
        let output = conv.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 64, 8, 8]);
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_with_bias() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, true);
        let params = conv.parameters();
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("depthwise.weight"));
        assert!(params.contains_key("depthwise.bias"));
        assert!(params.contains_key("pointwise.weight"));
        assert!(params.contains_key("pointwise.bias"));
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_without_bias() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, false);
        let params = conv.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("depthwise.weight"));
        assert!(params.contains_key("pointwise.weight"));
        assert!(!params.contains_key("depthwise.bias"));
        assert!(!params.contains_key("pointwise.bias"));
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_parameter_reduction() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let ratio = conv.parameter_reduction_ratio();
        assert!(
            ratio > 0.1 && ratio < 0.15,
            "Expected ratio around 0.127, got {}",
            ratio
        );
    }
    #[test]
    fn test_depthwise_separable_conv_multiple_channels() -> Result<()> {
        let configs = vec![(16, 32, 3), (32, 64, 3), (64, 128, 3), (128, 256, 3)];
        for (in_ch, out_ch, kernel) in configs {
            let conv = DepthwiseSeparableConv::new(in_ch, out_ch, kernel, 1, 1, true);
            let input = zeros(&[1, in_ch, 8, 8])?;
            let output = conv.forward(&input)?;
            let output_shape = output.shape();
            assert_eq!(output_shape.dims(), &[1, out_ch, 8, 8]);
        }
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_batch_processing() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(24, 48, 3, 1, 1, true);
        for batch_size in [1, 2, 4, 8] {
            let input = zeros(&[batch_size, 24, 16, 16])?;
            let output = conv.forward(&input)?;
            let output_shape = output.shape();
            assert_eq!(output_shape.dims(), &[batch_size, 48, 16, 16]);
        }
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_training_mode() {
        let mut conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        assert!(conv.training());
        conv.eval();
        assert!(!conv.training());
        conv.train();
        assert!(conv.training());
        conv.set_training(false);
        assert!(!conv.training());
    }
    #[test]
    fn test_depthwise_separable_conv_module_trait() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, true);
        let params = conv.parameters();
        assert_eq!(params.len(), 4);
        let named_params = conv.named_parameters();
        assert_eq!(named_params.len(), 4);
        assert!(named_params.contains_key("depthwise.weight"));
        assert!(named_params.contains_key("pointwise.weight"));
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_to_device() -> Result<()> {
        let mut conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        conv.to_device(DeviceType::Cpu)?;
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_shape_preservation() -> Result<()> {
        let test_cases = vec![(3, 1), (5, 2)];
        for (kernel, padding) in test_cases {
            let conv = DepthwiseSeparableConv::new(16, 32, kernel, 1, padding, true);
            let input = zeros(&[2, 16, 32, 32])?;
            let output = conv.forward(&input)?;
            let output_shape = output.shape();
            assert_eq!(output_shape.dims()[2], 32);
            assert_eq!(output_shape.dims()[3], 32);
        }
        Ok(())
    }
    #[test]
    fn test_depthwise_separable_conv_param_count() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let (dw_params, pw_params, total_params) = conv.param_count();
        assert_eq!(dw_params, 2);
        assert_eq!(pw_params, 2);
        assert_eq!(total_params, 4);
    }
    #[test]
    fn test_depthwise_separable_conv_debug_format() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let debug_str = format!("{:?}", conv);
        assert!(debug_str.contains("DepthwiseSeparableConv"));
        assert!(debug_str.contains("in_channels"));
        assert!(debug_str.contains("out_channels"));
        assert!(debug_str.contains("kernel_size"));
    }
}
