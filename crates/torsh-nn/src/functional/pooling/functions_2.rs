//! Tests for pooling functional operations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::functional::pooling::*;
    use torsh_core::error::Result;
    use torsh_tensor::Tensor;
    #[test]
    fn test_zero_pad2d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = zero_pad2d(&input, (1, 1, 1, 1))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 4, 4]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 0.0);
        assert_eq!(padded_data[5], 1.0);
        assert_eq!(padded_data[6], 2.0);
        assert_eq!(padded_data[9], 3.0);
        assert_eq!(padded_data[10], 4.0);
        Ok(())
    }
    #[test]
    fn test_replication_pad1d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3])?;
        let padded = replication_pad1d(&input, (2, 2))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 7]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 1.0);
        assert_eq!(padded_data[1], 1.0);
        assert_eq!(padded_data[2], 1.0);
        assert_eq!(padded_data[3], 2.0);
        assert_eq!(padded_data[4], 3.0);
        assert_eq!(padded_data[5], 3.0);
        assert_eq!(padded_data[6], 3.0);
        Ok(())
    }
    #[test]
    fn test_replication_pad2d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = replication_pad2d(&input, (1, 1, 1, 1))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 4, 4]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 1.0);
        assert_eq!(padded_data[5], 1.0);
        assert_eq!(padded_data[6], 2.0);
        assert_eq!(padded_data[15], 4.0);
        Ok(())
    }
    #[test]
    fn test_reflection_pad1d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4])?;
        let padded = reflection_pad1d(&input, (2, 2))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 8]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 3.0);
        assert_eq!(padded_data[1], 2.0);
        assert_eq!(padded_data[2], 1.0);
        assert_eq!(padded_data[3], 2.0);
        assert_eq!(padded_data[4], 3.0);
        assert_eq!(padded_data[5], 4.0);
        assert_eq!(padded_data[6], 3.0);
        assert_eq!(padded_data[7], 2.0);
        Ok(())
    }
    #[test]
    fn test_reflection_pad2d() -> Result<()> {
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )?;
        let padded = reflection_pad2d(&input, (1, 1, 1, 1))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 5, 5]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 5.0);
        let center_idx = 1 * 5 + 1;
        assert_eq!(padded_data[center_idx], 1.0);
        Ok(())
    }
    #[test]
    fn test_reflection_pad_validation() {
        let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 2]).expect("Tensor should succeed");
        let result = reflection_pad1d(&input, (2, 1));
        assert!(result.is_err());
        let input2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
            .expect("Tensor should succeed");
        let result2d = reflection_pad2d(&input2d, (2, 1, 1, 1));
        assert!(result2d.is_err());
    }
    #[test]
    fn test_general_pad_zero_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = pad(&input, &[(1, 1), (1, 1)], "zero", None)?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 0.0);
        Ok(())
    }
    #[test]
    fn test_general_pad_constant_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = pad(&input, &[(1, 1), (1, 1)], "constant", Some(5.0))?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 5.0);
        Ok(())
    }
    #[test]
    fn test_general_pad_reflect_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = pad(&input, &[(1, 1), (1, 1)], "reflect", None)?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);
        Ok(())
    }
    #[test]
    fn test_general_pad_replicate_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = pad(&input, &[(1, 1), (1, 1)], "replicate", None)?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 1.0);
        Ok(())
    }
    #[test]
    fn test_general_pad_invalid_mode() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
            .expect("Tensor should succeed");
        let result = pad(&input, &[(1, 1), (1, 1)], "invalid", None);
        assert!(result.is_err());
    }
    #[test]
    fn test_pad_shape_mismatch() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])
            .expect("Tensor should succeed");
        let result = pad(&input, &[(1, 1)], "zero", None);
        assert!(result.is_err());
    }
    #[test]
    fn test_zero_pad2d_asymmetric() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let padded = zero_pad2d(&input, (1, 2, 0, 1))?;
        assert_eq!(padded.shape().dims(), &[1, 1, 3, 5]);
        Ok(())
    }
    #[test]
    fn test_global_avg_pool1d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 4])?;
        let pooled = global_avg_pool1d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.5).abs() < 1e-5);
        assert!((pooled_data[1] - 6.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_avg_pool2d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let pooled = global_avg_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_avg_pool2d_multi_channel() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2])?;
        let pooled = global_avg_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.5).abs() < 1e-5);
        assert!((pooled_data[1] - 6.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_avg_pool3d() -> Result<()> {
        let input = Tensor::from_vec(vec![3.0; 8], &[1, 1, 2, 2, 2])?;
        let pooled = global_avg_pool3d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 3.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_max_pool1d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 6.0, 7.0, 4.0], &[1, 2, 4])?;
        let pooled = global_max_pool1d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 5.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_max_pool2d() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[1, 1, 2, 2])?;
        let pooled = global_max_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 5.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_max_pool2d_multi_channel() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2])?;
        let pooled = global_max_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 9.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_max_pool3d() -> Result<()> {
        let input = Tensor::from_vec(
            vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0, 4.0, 6.0],
            &[1, 1, 2, 2, 2],
        )?;
        let pooled = global_max_pool3d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 8.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_pool_with_negative_values() -> Result<()> {
        let input = Tensor::from_vec(vec![-5.0, -2.0, -8.0, -1.0], &[1, 1, 2, 2])?;
        let max_pooled = global_max_pool2d(&input)?;
        let max_data = max_pooled.to_vec()?;
        assert!((max_data[0] - (-1.0)).abs() < 1e-5);
        let avg_pooled = global_avg_pool2d(&input)?;
        let avg_data = avg_pooled.to_vec()?;
        assert!((avg_data[0] - (-4.0)).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_global_pool_shape_validation() {
        let input_2d = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("Tensor should succeed");
        assert!(global_avg_pool1d(&input_2d).is_err());
        assert!(global_max_pool1d(&input_2d).is_err());
        let input_3d = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 2]).expect("Tensor should succeed");
        assert!(global_avg_pool2d(&input_3d).is_err());
        assert!(global_max_pool2d(&input_3d).is_err());
        let input_4d =
            Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 1, 2]).expect("Tensor should succeed");
        assert!(global_avg_pool3d(&input_4d).is_err());
        assert!(global_max_pool3d(&input_4d).is_err());
    }
    #[test]
    fn test_adaptive_avg_pool1d_downsampling() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 8])?;
        let pooled = adaptive_avg_pool1d(&input, 4)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 1.5).abs() < 1e-5);
        assert!((pooled_data[1] - 3.5).abs() < 1e-5);
        assert!((pooled_data[2] - 5.5).abs() < 1e-5);
        assert!((pooled_data[3] - 7.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_avg_pool1d_same_size() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4])?;
        let pooled = adaptive_avg_pool1d(&input, 4)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 1.0).abs() < 1e-5);
        assert!((pooled_data[1] - 2.0).abs() < 1e-5);
        assert!((pooled_data[2] - 3.0).abs() < 1e-5);
        assert!((pooled_data[3] - 4.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_avg_pool2d_downsampling() -> Result<()> {
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        )?;
        let pooled = adaptive_avg_pool2d(&input, (Some(2), Some(2)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 3.5).abs() < 1e-5);
        assert!((pooled_data[1] - 5.5).abs() < 1e-5);
        assert!((pooled_data[2] - 11.5).abs() < 1e-5);
        assert!((pooled_data[3] - 13.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_avg_pool2d_mixed_dimensions() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 4])?;
        let pooled = adaptive_avg_pool2d(&input, (Some(2), Some(2)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);
        Ok(())
    }
    #[test]
    fn test_adaptive_avg_pool3d_downsampling() -> Result<()> {
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 1, 2, 2, 2],
        )?;
        let pooled = adaptive_avg_pool3d(&input, (Some(1), Some(1), Some(1)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 4.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_max_pool1d_downsampling() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 8.0, 3.0, 6.0, 2.0, 9.0, 4.0, 5.0], &[1, 1, 8])?;
        let pooled = adaptive_max_pool1d(&input, 4)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 8.0).abs() < 1e-5);
        assert!((pooled_data[1] - 6.0).abs() < 1e-5);
        assert!((pooled_data[2] - 9.0).abs() < 1e-5);
        assert!((pooled_data[3] - 5.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_max_pool2d_downsampling() -> Result<()> {
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        )?;
        let pooled = adaptive_max_pool2d(&input, (Some(2), Some(2)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 6.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        assert!((pooled_data[2] - 14.0).abs() < 1e-5);
        assert!((pooled_data[3] - 16.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_max_pool3d_downsampling() -> Result<()> {
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 15.0],
            &[1, 1, 2, 2, 2],
        )?;
        let pooled = adaptive_max_pool3d(&input, (Some(1), Some(1), Some(1)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 15.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_pool_with_negative_values() -> Result<()> {
        let input = Tensor::from_vec(vec![-5.0, -2.0, -8.0, -1.0], &[1, 1, 4])?;
        let avg_pooled = adaptive_avg_pool1d(&input, 2)?;
        let avg_data = avg_pooled.to_vec()?;
        assert!((avg_data[0] - (-3.5)).abs() < 1e-5);
        assert!((avg_data[1] - (-4.5)).abs() < 1e-5);
        let max_pooled = adaptive_max_pool1d(&input, 2)?;
        let max_data = max_pooled.to_vec()?;
        assert!((max_data[0] - (-2.0)).abs() < 1e-5);
        assert!((max_data[1] - (-1.0)).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_pool_identity() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4])?;
        let avg_pooled = adaptive_avg_pool1d(&input, 4)?;
        assert_eq!(avg_pooled.shape().dims(), input.shape().dims());
        let avg_data = avg_pooled.to_vec()?;
        assert_eq!(avg_data, vec![1.0, 2.0, 3.0, 4.0]);
        let max_pooled = adaptive_max_pool1d(&input, 4)?;
        assert_eq!(max_pooled.shape().dims(), input.shape().dims());
        let max_data = max_pooled.to_vec()?;
        assert_eq!(max_data, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
    #[test]
    fn test_adaptive_pool_multi_channel() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 4])?;
        let pooled = adaptive_avg_pool1d(&input, 2)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 1.5).abs() < 1e-5);
        assert!((pooled_data[1] - 3.5).abs() < 1e-5);
        assert!((pooled_data[2] - 5.5).abs() < 1e-5);
        assert!((pooled_data[3] - 7.5).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_adaptive_pool_batch() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 4])?;
        let pooled = adaptive_max_pool1d(&input, 2)?;
        assert_eq!(pooled.shape().dims(), &[2, 1, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.0).abs() < 1e-5);
        assert!((pooled_data[1] - 4.0).abs() < 1e-5);
        assert!((pooled_data[2] - 6.0).abs() < 1e-5);
        assert!((pooled_data[3] - 8.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool1d_basic() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 8.0, 4.0, 5.0, 3.0, 7.0, 6.0], &[1, 1, 8])?;
        let pooled = max_pool1d(&input, 2, None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        assert!((pooled_data[2] - 5.0).abs() < 1e-5);
        assert!((pooled_data[3] - 7.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool1d_with_stride() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0, 6.0], &[1, 1, 6])?;
        let pooled = max_pool1d(&input, 2, Some(1), None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 5]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 5.0).abs() < 1e-5);
        assert!((pooled_data[1] - 5.0).abs() < 1e-5);
        assert!((pooled_data[2] - 8.0).abs() < 1e-5);
        assert!((pooled_data[3] - 8.0).abs() < 1e-5);
        assert!((pooled_data[4] - 6.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool2d_basic() -> Result<()> {
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 13.0, 14.0, 11.0, 12.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        )?;
        let pooled = max_pool2d(&input, (2, 2), None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        assert!((pooled_data[2] - 12.0).abs() < 1e-5);
        assert!((pooled_data[3] - 16.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool2d_multi_channel() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2])?;
        let pooled = max_pool2d(&input, (2, 2), None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool3d_basic() -> Result<()> {
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 16.0],
            &[1, 1, 2, 2, 2],
        )?;
        let pooled = max_pool3d(&input, (2, 2, 2), None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 16.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool_with_negative_values() -> Result<()> {
        let input = Tensor::from_vec(vec![-5.0, -2.0, -8.0, -1.0], &[1, 1, 4])?;
        let pooled = max_pool1d(&input, 2, None, None, None)?;
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - (-2.0)).abs() < 1e-5);
        assert!((pooled_data[1] - (-1.0)).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool2d_with_padding() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let pooled = max_pool2d(&input, (2, 2), None, Some((1, 1)), None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);
        let pooled_data = pooled.to_vec()?;
        for &val in &pooled_data {
            assert!(val > f32::NEG_INFINITY);
        }
        Ok(())
    }
    #[test]
    fn test_max_pool1d_with_dilation() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0, 6.0], &[1, 1, 6])?;
        let pooled = max_pool1d(&input, 2, Some(2), None, Some(2))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.0).abs() < 1e-5);
        assert!((pooled_data[1] - 3.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool_batch() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 4.0, 2.0, 3.0, 5.0, 8.0, 6.0, 7.0], &[2, 1, 4])?;
        let pooled = max_pool1d(&input, 2, None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[2, 1, 2]);
        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 3.0).abs() < 1e-5);
        assert!((pooled_data[2] - 8.0).abs() < 1e-5);
        assert!((pooled_data[3] - 7.0).abs() < 1e-5);
        Ok(())
    }
    #[test]
    fn test_max_pool_shape_validation() {
        let input_2d = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("Tensor should succeed");
        assert!(max_pool1d(&input_2d, 2, None, None, None).is_err());
        let input_3d = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 2]).expect("Tensor should succeed");
        assert!(max_pool2d(&input_3d, (2, 2), None, None, None).is_err());
        let input_4d =
            Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 1, 2]).expect("Tensor should succeed");
        assert!(max_pool3d(&input_4d, (2, 2, 2), None, None, None).is_err());
    }
}
