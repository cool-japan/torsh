//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::layers::pooling::*;
    use crate::Module;
    use torsh_core::device::DeviceType;
    use torsh_core::error::Result;
    use torsh_tensor::creation::zeros;
    #[test]
    fn test_maxpool2d_new() {
        let pool = MaxPool2d::new((2, 2), None, (0, 0), (1, 1), false);
        assert_eq!(pool.kernel_size, (2, 2));
        assert_eq!(pool.stride, None);
        assert_eq!(pool.padding, (0, 0));
        assert_eq!(pool.dilation, (1, 1));
        assert!(!pool.ceil_mode);
    }
    #[test]
    fn test_maxpool2d_with_kernel_size() {
        let pool = MaxPool2d::with_kernel_size(3);
        assert_eq!(pool.kernel_size, (3, 3));
        assert_eq!(pool.stride, None);
        assert_eq!(pool.padding, (0, 0));
    }
    #[test]
    fn test_maxpool2d_forward() -> Result<()> {
        let pool = MaxPool2d::with_kernel_size(2);
        let input = zeros(&[2, 3, 8, 8])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_maxpool2d_forward_with_stride() -> Result<()> {
        let pool = MaxPool2d::new((2, 2), Some((1, 1)), (0, 0), (1, 1), false);
        let input = zeros(&[1, 1, 4, 4])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 3, 3]);
        Ok(())
    }
    #[test]
    fn test_maxpool2d_forward_with_padding() -> Result<()> {
        let pool = MaxPool2d::new((2, 2), None, (1, 1), (1, 1), false);
        let input = zeros(&[1, 1, 4, 4])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 3, 3]);
        Ok(())
    }
    #[test]
    fn test_maxpool2d_training_mode() {
        let mut pool = MaxPool2d::with_kernel_size(2);
        assert!(pool.training());
        pool.eval();
        assert!(!pool.training());
        pool.train();
        assert!(pool.training());
    }
    #[test]
    fn test_maxpool2d_parameters() {
        let pool = MaxPool2d::with_kernel_size(2);
        let params = pool.parameters();
        assert_eq!(params.len(), 0);
    }
    #[test]
    fn test_avgpool2d_new() {
        let pool = AvgPool2d::new((2, 2), None, (0, 0), false, true);
        assert_eq!(pool.kernel_size, (2, 2));
        assert_eq!(pool.stride, None);
        assert_eq!(pool.padding, (0, 0));
        assert!(!pool.ceil_mode);
    }
    #[test]
    fn test_avgpool2d_with_kernel_size() {
        let pool = AvgPool2d::with_kernel_size(3);
        assert_eq!(pool.kernel_size, (3, 3));
    }
    #[test]
    fn test_avgpool2d_forward() -> Result<()> {
        let pool = AvgPool2d::with_kernel_size(2);
        let input = zeros(&[1, 1, 8, 8])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_avgpool2d_training_mode() {
        let mut pool = AvgPool2d::with_kernel_size(2);
        assert!(pool.training());
        pool.eval();
        assert!(!pool.training());
    }
    #[test]
    fn test_adaptive_avgpool2d_new() {
        let pool = AdaptiveAvgPool2d::new((Some(4), Some(4)));
        assert_eq!(pool.output_size, (Some(4), Some(4)));
    }
    #[test]
    fn test_adaptive_avgpool2d_with_output_size() {
        let pool = AdaptiveAvgPool2d::with_output_size(7);
        assert_eq!(pool.output_size, (Some(7), Some(7)));
    }
    #[test]
    fn test_adaptive_avgpool2d_forward() -> Result<()> {
        let pool = AdaptiveAvgPool2d::with_output_size(4);
        let input = zeros(&[2, 3, 16, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_adaptive_avgpool2d_forward_partial_none() -> Result<()> {
        let pool = AdaptiveAvgPool2d::new((Some(4), None));
        let input = zeros(&[1, 1, 8, 12])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 12]);
        Ok(())
    }
    #[test]
    fn test_adaptive_avgpool2d_forward_invalid_input() {
        let pool = AdaptiveAvgPool2d::with_output_size(4);
        let input = zeros(&[2, 3, 16]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
    }
    #[test]
    fn test_maxpool1d_new() {
        let pool = MaxPool1d::new(3, None, 0, 1, false);
        assert_eq!(pool.kernel_size, 3);
        assert_eq!(pool.stride, None);
        assert_eq!(pool.padding, 0);
    }
    #[test]
    fn test_maxpool1d_with_kernel_size() {
        let pool = MaxPool1d::with_kernel_size(4);
        assert_eq!(pool.kernel_size, 4);
    }
    #[test]
    fn test_maxpool1d_forward_3d() -> Result<()> {
        let pool = MaxPool1d::with_kernel_size(2);
        let input = zeros(&[2, 3, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 8]);
        Ok(())
    }
    #[test]
    fn test_maxpool1d_forward_2d() -> Result<()> {
        let pool = MaxPool1d::with_kernel_size(2);
        let input = zeros(&[2, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 8]);
        Ok(())
    }
    #[test]
    fn test_maxpool1d_forward_invalid_input() {
        let pool = MaxPool1d::with_kernel_size(2);
        let input = zeros(&[2]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
    }
    #[test]
    fn test_maxpool3d_new() {
        let pool = MaxPool3d::new((2, 2, 2), None, (0, 0, 0), (1, 1, 1), false);
        assert_eq!(pool.kernel_size, (2, 2, 2));
    }
    #[test]
    fn test_maxpool3d_with_kernel_size() {
        let pool = MaxPool3d::with_kernel_size(3);
        assert_eq!(pool.kernel_size, (3, 3, 3));
    }
    #[test]
    fn test_maxpool3d_forward() -> Result<()> {
        let pool = MaxPool3d::with_kernel_size(2);
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_maxpool3d_forward_invalid_input() {
        let pool = MaxPool3d::with_kernel_size(2);
        let input = zeros(&[1, 1, 8, 8]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
    }
    #[test]
    fn test_lppool1d_new() {
        let pool = LPPool1d::new(2.0, 3, None, false);
        assert_eq!(pool.norm_type, 2.0);
        assert_eq!(pool.kernel_size, 3);
    }
    #[test]
    fn test_lppool1d_forward() -> Result<()> {
        let pool = LPPool1d::new(2.0, 2, None, false);
        let input = zeros(&[1, 1, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8]);
        Ok(())
    }
    #[test]
    fn test_lppool1d_ceil_mode() -> Result<()> {
        let pool = LPPool1d::new(2.0, 3, Some(2), true);
        let input = zeros(&[1, 1, 10])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 5]);
        Ok(())
    }
    #[test]
    fn test_lppool2d_new() {
        let pool = LPPool2d::new(2.0, (3, 3), None, false);
        assert_eq!(pool.norm_type, 2.0);
        assert_eq!(pool.kernel_size, (3, 3));
    }
    #[test]
    fn test_lppool2d_forward() -> Result<()> {
        let pool = LPPool2d::new(2.0, (2, 2), None, false);
        let input = zeros(&[1, 1, 8, 8])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool1d_new() {
        let pool = FractionalMaxPool1d::new(2, Some(8), None, false);
        assert_eq!(pool.kernel_size, 2);
        assert_eq!(pool.output_size, Some(8));
        assert_eq!(pool.output_ratio, None);
    }
    #[test]
    fn test_fractional_maxpool1d_with_kernel_size() {
        let pool = FractionalMaxPool1d::with_kernel_size(3);
        assert_eq!(pool.kernel_size, 3);
        assert_eq!(pool.output_ratio, Some(0.5));
    }
    #[test]
    fn test_fractional_maxpool1d_with_output_ratio() {
        let pool = FractionalMaxPool1d::with_output_ratio(2, 0.75);
        assert_eq!(pool.output_ratio, Some(0.75));
    }
    #[test]
    fn test_fractional_maxpool1d_forward_with_output_size() -> Result<()> {
        let pool = FractionalMaxPool1d::new(2, Some(8), None, false);
        let input = zeros(&[1, 1, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool1d_forward_with_output_ratio() -> Result<()> {
        let pool = FractionalMaxPool1d::new(2, None, Some(0.5), false);
        let input = zeros(&[1, 1, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool1d_forward_with_indices() -> Result<()> {
        let pool = FractionalMaxPool1d::new(2, Some(8), None, true);
        let input = zeros(&[1, 1, 16])?;
        let (output, indices) = pool.forward_with_indices(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8]);
        assert!(indices.is_some());
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool1d_forward_with_indices_disabled() -> Result<()> {
        let pool = FractionalMaxPool1d::new(2, Some(8), None, false);
        let input = zeros(&[1, 1, 16])?;
        let (_, indices) = pool.forward_with_indices(&input)?;
        assert!(indices.is_none());
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool1d_no_output_size_or_ratio() {
        let pool = FractionalMaxPool1d::new(2, None, None, false);
        let input = zeros(&[1, 1, 16]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
    }
    #[test]
    fn test_fractional_maxpool2d_new() {
        let pool = FractionalMaxPool2d::new((2, 2), Some((4, 4)), None, false);
        assert_eq!(pool.kernel_size, (2, 2));
        assert_eq!(pool.output_size, Some((4, 4)));
    }
    #[test]
    fn test_fractional_maxpool2d_with_kernel_size() {
        let pool = FractionalMaxPool2d::with_kernel_size((3, 3));
        assert_eq!(pool.output_ratio, Some((0.5, 0.5)));
    }
    #[test]
    fn test_fractional_maxpool2d_forward_with_output_size() -> Result<()> {
        let pool = FractionalMaxPool2d::new((2, 2), Some((4, 4)), None, false);
        let input = zeros(&[1, 1, 8, 8])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool2d_forward_with_output_ratio() -> Result<()> {
        let pool = FractionalMaxPool2d::new((2, 2), None, Some((0.5, 0.5)), false);
        let input = zeros(&[1, 1, 16, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8, 8]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool2d_forward_with_indices() -> Result<()> {
        let pool = FractionalMaxPool2d::new((2, 2), Some((4, 4)), None, true);
        let input = zeros(&[1, 1, 8, 8])?;
        let (output, indices) = pool.forward_with_indices(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);
        assert!(indices.is_some());
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool2d_different_output_ratios() -> Result<()> {
        let input = zeros(&[2, 32, 100, 100])?;
        let pool_half = FractionalMaxPool2d::with_output_ratio((2, 2), (0.5, 0.5));
        let output_half = pool_half.forward(&input)?;
        assert_eq!(output_half.shape().dims(), &[2, 32, 50, 50]);
        let pool_70 = FractionalMaxPool2d::with_output_ratio((2, 2), (0.7, 0.7));
        let output_70 = pool_70.forward(&input)?;
        assert_eq!(output_70.shape().dims(), &[2, 32, 70, 70]);
        let pool_90 = FractionalMaxPool2d::with_output_ratio((2, 2), (0.9, 0.9));
        let output_90 = pool_90.forward(&input)?;
        assert_eq!(output_90.shape().dims(), &[2, 32, 90, 90]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool2d_training_vs_eval_mode() -> Result<()> {
        let mut pool = FractionalMaxPool2d::with_output_ratio((2, 2), (0.5, 0.5));
        let input = zeros(&[1, 3, 32, 32])?;
        pool.train();
        assert!(pool.training());
        let output_train = pool.forward(&input)?;
        assert_eq!(output_train.shape().dims(), &[1, 3, 16, 16]);
        pool.eval();
        assert!(!pool.training());
        let output_eval = pool.forward(&input)?;
        assert_eq!(output_eval.shape().dims(), &[1, 3, 16, 16]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool2d_batch_processing() -> Result<()> {
        let pool = FractionalMaxPool2d::with_output_ratio((3, 3), (0.5, 0.5));
        let input = zeros(&[16, 64, 64, 64])?;
        let output = pool.forward(&input)?;
        assert_eq!(output.shape().dims(), &[16, 64, 32, 32]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool2d_invalid_input_shape() {
        let pool = FractionalMaxPool2d::with_output_ratio((2, 2), (0.5, 0.5));
        let input = zeros(&[1, 32, 64]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("expects 4D input"));
        }
    }
    #[test]
    fn test_fractional_maxpool2d_invalid_ratio() {
        let pool = FractionalMaxPool2d::with_output_ratio((2, 2), (1.5, 0.5));
        let input = zeros(&[1, 1, 32, 32]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("output_ratio must be in range"));
        }
    }
    #[test]
    fn test_fractional_maxpool2d_no_output_spec() {
        let pool = FractionalMaxPool2d::new((2, 2), None, None, false);
        let input = zeros(&[1, 1, 32, 32]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Either output_size or output_ratio must be specified"));
        }
    }
    #[test]
    fn test_fractional_maxpool2d_module_traits() -> Result<()> {
        let mut pool = FractionalMaxPool2d::with_output_ratio((2, 2), (0.5, 0.5));
        assert!(pool.training());
        pool.eval();
        assert!(!pool.training());
        pool.train();
        assert!(pool.training());
        let params = pool.parameters();
        assert_eq!(params.len(), 0);
        pool.to_device(DeviceType::Cpu)?;
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool3d_new() {
        let pool = FractionalMaxPool3d::new((2, 2, 2), Some((4, 4, 4)), None, false);
        assert_eq!(pool.kernel_size, (2, 2, 2));
    }
    #[test]
    fn test_fractional_maxpool3d_with_kernel_size() {
        let pool = FractionalMaxPool3d::with_kernel_size((3, 3, 3));
        assert_eq!(pool.output_ratio, Some((0.5, 0.5, 0.5)));
    }
    #[test]
    fn test_fractional_maxpool3d_forward_with_output_size() -> Result<()> {
        let pool = FractionalMaxPool3d::new((2, 2, 2), Some((4, 4, 4)), None, false);
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool3d_forward_with_output_ratio() -> Result<()> {
        let pool = FractionalMaxPool3d::new((2, 2, 2), None, Some((0.5, 0.5, 0.5)), false);
        let input = zeros(&[1, 1, 16, 16, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8, 8, 8]);
        Ok(())
    }
    #[test]
    fn test_fractional_maxpool3d_forward_with_indices() -> Result<()> {
        let pool = FractionalMaxPool3d::new((2, 2, 2), Some((4, 4, 4)), None, true);
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let (output, indices) = pool.forward_with_indices(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4, 4]);
        assert!(indices.is_some());
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool2d_new() {
        let pool = AdaptiveMaxPool2d::new((Some(4), Some(4)), false);
        assert_eq!(pool.output_size, (Some(4), Some(4)));
        assert!(!pool.return_indices);
    }
    #[test]
    fn test_adaptive_maxpool2d_with_output_size() {
        let pool = AdaptiveMaxPool2d::with_output_size(5);
        assert_eq!(pool.output_size, (Some(5), Some(5)));
    }
    #[test]
    fn test_adaptive_maxpool2d_forward() -> Result<()> {
        let pool = AdaptiveMaxPool2d::with_output_size(4);
        let input = zeros(&[1, 1, 16, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool2d_forward_with_indices() -> Result<()> {
        let pool = AdaptiveMaxPool2d::new((Some(4), Some(4)), true);
        let input = zeros(&[1, 1, 8, 8])?;
        let (output, indices) = pool.forward_with_indices(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);
        assert!(indices.is_some());
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool2d_variable_input_sizes() -> Result<()> {
        let pool = AdaptiveMaxPool2d::with_output_size(7);
        let input1 = zeros(&[2, 64, 224, 224])?;
        let output1 = pool.forward(&input1)?;
        assert_eq!(output1.shape().dims(), &[2, 64, 7, 7]);
        let input2 = zeros(&[2, 64, 128, 128])?;
        let output2 = pool.forward(&input2)?;
        assert_eq!(output2.shape().dims(), &[2, 64, 7, 7]);
        let input3 = zeros(&[2, 64, 256, 256])?;
        let output3 = pool.forward(&input3)?;
        assert_eq!(output3.shape().dims(), &[2, 64, 7, 7]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool2d_non_square_output() -> Result<()> {
        let pool = AdaptiveMaxPool2d::new((Some(5), Some(3)), false);
        let input = zeros(&[1, 32, 100, 60])?;
        let output = pool.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 32, 5, 3]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool2d_batch_processing() -> Result<()> {
        let pool = AdaptiveMaxPool2d::with_output_size(4);
        let input = zeros(&[8, 128, 32, 32])?;
        let output = pool.forward(&input)?;
        assert_eq!(output.shape().dims(), &[8, 128, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool2d_invalid_input_shape() {
        let pool = AdaptiveMaxPool2d::with_output_size(4);
        let input = zeros(&[1, 32, 64]).unwrap();
        let result = pool.forward(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("expects 4D input"));
        }
    }
    #[test]
    fn test_adaptive_maxpool2d_module_traits() -> Result<()> {
        let mut pool = AdaptiveMaxPool2d::with_output_size(7);
        assert!(pool.training());
        pool.eval();
        assert!(!pool.training());
        pool.train();
        assert!(pool.training());
        let params = pool.parameters();
        assert_eq!(params.len(), 0);
        pool.to_device(DeviceType::Cpu)?;
        Ok(())
    }
    #[test]
    fn test_adaptive_avgpool1d_new() {
        let pool = AdaptiveAvgPool1d::new(8);
        assert_eq!(pool.output_size, 8);
    }
    #[test]
    fn test_adaptive_avgpool1d_forward() -> Result<()> {
        let pool = AdaptiveAvgPool1d::new(8);
        let input = zeros(&[2, 3, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 8]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool1d_new() {
        let pool = AdaptiveMaxPool1d::new(8, false);
        assert_eq!(pool.output_size, 8);
        assert!(!pool.return_indices);
    }
    #[test]
    fn test_adaptive_maxpool1d_forward() -> Result<()> {
        let pool = AdaptiveMaxPool1d::new(8, false);
        let input = zeros(&[1, 1, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 8]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool1d_forward_with_indices() -> Result<()> {
        let pool = AdaptiveMaxPool1d::new(8, true);
        let input = zeros(&[1, 1, 16])?;
        let (output, indices) = pool.forward_with_indices(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1, 8]);
        assert!(indices.is_some());
        Ok(())
    }
    #[test]
    fn test_adaptive_avgpool3d_new() {
        let pool = AdaptiveAvgPool3d::new((Some(4), Some(4), Some(4)));
        assert_eq!(pool.output_size, (Some(4), Some(4), Some(4)));
    }
    #[test]
    fn test_adaptive_avgpool3d_with_output_size() {
        let pool = AdaptiveAvgPool3d::with_output_size(7);
        assert_eq!(pool.output_size, (Some(7), Some(7), Some(7)));
    }
    #[test]
    fn test_adaptive_avgpool3d_forward() -> Result<()> {
        let pool = AdaptiveAvgPool3d::with_output_size(4);
        let input = zeros(&[1, 1, 16, 16, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool3d_new() {
        let pool = AdaptiveMaxPool3d::new((Some(4), Some(4), Some(4)), false);
        assert_eq!(pool.output_size, (Some(4), Some(4), Some(4)));
    }
    #[test]
    fn test_adaptive_maxpool3d_with_output_size() {
        let pool = AdaptiveMaxPool3d::with_output_size(5);
        assert_eq!(pool.output_size, (Some(5), Some(5), Some(5)));
    }
    #[test]
    fn test_adaptive_maxpool3d_forward() -> Result<()> {
        let pool = AdaptiveMaxPool3d::with_output_size(4);
        let input = zeros(&[1, 1, 16, 16, 16])?;
        let output = pool.forward(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_adaptive_maxpool3d_forward_with_indices() -> Result<()> {
        let pool = AdaptiveMaxPool3d::new((Some(4), Some(4), Some(4)), true);
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let (output, indices) = pool.forward_with_indices(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4, 4]);
        assert!(indices.is_some());
        Ok(())
    }
    #[test]
    fn test_global_avg_pool2d() -> Result<()> {
        let input = zeros(&[2, 3, 8, 8])?;
        let output = GlobalPool::global_avg_pool2d(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 1, 1]);
        Ok(())
    }
    #[test]
    fn test_global_max_pool2d() -> Result<()> {
        let input = zeros(&[2, 3, 8, 8])?;
        let output = GlobalPool::global_max_pool2d(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 1, 1]);
        Ok(())
    }
    #[test]
    fn test_global_avg_pool1d() -> Result<()> {
        let input = zeros(&[2, 3, 16])?;
        let output = GlobalPool::global_avg_pool1d(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 1]);
        Ok(())
    }
    #[test]
    fn test_global_max_pool1d() -> Result<()> {
        let input = zeros(&[2, 3, 16])?;
        let output = GlobalPool::global_max_pool1d(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[2, 3, 1]);
        Ok(())
    }
    #[test]
    fn test_global_avg_pool3d() -> Result<()> {
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let output = GlobalPool::global_avg_pool3d(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 1, 1, 1]);
        Ok(())
    }
    #[test]
    fn test_global_max_pool3d() -> Result<()> {
        let input = zeros(&[1, 1, 8, 8, 8])?;
        let output = GlobalPool::global_max_pool3d(&input)?;
        let output_shape = output.shape();
        assert_eq!(output_shape.dims(), &[1, 1, 1, 1, 1]);
        Ok(())
    }
    #[test]
    fn test_module_training_modes() {
        let mut pool = MaxPool2d::with_kernel_size(2);
        assert!(pool.training());
        pool.set_training(false);
        assert!(!pool.training());
        pool.set_training(true);
        assert!(pool.training());
    }
    #[test]
    fn test_module_named_parameters() {
        let pool = AdaptiveAvgPool2d::with_output_size(4);
        let named_params = pool.named_parameters();
        assert_eq!(named_params.len(), 0);
    }
    #[test]
    fn test_module_to_device() -> Result<()> {
        let mut pool = MaxPool2d::with_kernel_size(2);
        pool.to_device(DeviceType::Cpu)?;
        Ok(())
    }
}
