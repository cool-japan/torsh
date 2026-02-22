//! Tests for convolution operations

use approx::assert_relative_eq;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::*;

#[test]
fn test_conv1d_simple() -> Result<()> {
    // Create a simple 1D convolution example
    // Input: batch=1, channels=1, length=5
    let input = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1, 1, 5],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=1, in_channels=1, kernel_size=3
    let weight = Tensor::from_data(vec![1.0, 0.0, -1.0], vec![1, 1, 3], DeviceType::Cpu)?;

    // Perform convolution with stride=1, padding=0
    let output = input.conv1d(&weight, None, 1, 0, 1, 1)?;

    // Expected output: [1*1 + 2*0 + 3*(-1), 2*1 + 3*0 + 4*(-1), 3*1 + 4*0 + 5*(-1)]
    // = [-2, -2, -2]
    assert_eq!(output.shape().dims(), &[1, 1, 3]);
    let output_data = output.data().unwrap();
    assert_relative_eq!(output_data[0], -2.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], -2.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], -2.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv1d_with_padding() -> Result<()> {
    // Input: batch=1, channels=1, length=3
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 1, 3], DeviceType::Cpu)?;

    // Kernel: out_channels=1, in_channels=1, kernel_size=3
    let weight = Tensor::from_data(vec![1.0, 1.0, 1.0], vec![1, 1, 3], DeviceType::Cpu)?;

    // Perform convolution with stride=1, padding=1
    let output = input.conv1d(&weight, None, 1, 1, 1, 1)?;

    // With padding=1, output length should be 3
    assert_eq!(output.shape().dims(), &[1, 1, 3]);
    let output_data = output.data().unwrap();

    // Expected: [0*1 + 1*1 + 2*1, 1*1 + 2*1 + 3*1, 2*1 + 3*1 + 0*1]
    // = [3, 6, 5]
    assert_relative_eq!(output_data[0], 3.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 6.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 5.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv1d_with_bias() -> Result<()> {
    // Input: batch=1, channels=1, length=4
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4], DeviceType::Cpu)?;

    // Kernel: out_channels=2, in_channels=1, kernel_size=2
    let weight = Tensor::from_data(vec![1.0, 1.0, -1.0, 1.0], vec![2, 1, 2], DeviceType::Cpu)?;

    // Bias for 2 output channels
    let bias = Tensor::from_data(vec![0.5, -0.5], vec![2], DeviceType::Cpu)?;

    // Perform convolution with bias
    let output = input.conv1d(&weight, Some(&bias), 1, 0, 1, 1)?;

    assert_eq!(output.shape().dims(), &[1, 2, 3]);
    let output_data = output.data().unwrap();

    // Channel 0: [1*1 + 2*1 + 0.5, 2*1 + 3*1 + 0.5, 3*1 + 4*1 + 0.5]
    // = [3.5, 5.5, 7.5]
    assert_relative_eq!(output_data[0], 3.5, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 5.5, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 7.5, epsilon = 1e-6);

    // Channel 1: [1*(-1) + 2*1 - 0.5, 2*(-1) + 3*1 - 0.5, 3*(-1) + 4*1 - 0.5]
    // = [0.5, 0.5, 0.5]
    assert_relative_eq!(output_data[3], 0.5, epsilon = 1e-6);
    assert_relative_eq!(output_data[4], 0.5, epsilon = 1e-6);
    assert_relative_eq!(output_data[5], 0.5, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv2d_simple() -> Result<()> {
    // Create a simple 2D convolution example
    // Input: batch=1, channels=1, height=3, width=3
    let input = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=1, in_channels=1, kernel_h=2, kernel_w=2
    let weight = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2], DeviceType::Cpu)?;

    // Perform convolution with stride=(1,1), padding=(0,0)
    let output = input.conv2d(&weight, None, (1, 1), (0, 0), (1, 1), 1)?;

    // Expected output: 2x2
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    let output_data = output.data().unwrap();

    // Expected values:
    // [1*1 + 2*0 + 4*0 + 5*1] = 6
    // [2*1 + 3*0 + 5*0 + 6*1] = 8
    // [4*1 + 5*0 + 7*0 + 8*1] = 12
    // [5*1 + 6*0 + 8*0 + 9*1] = 14
    assert_relative_eq!(output_data[0], 6.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 8.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 12.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 14.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv2d_stride() -> Result<()> {
    // Input: batch=1, channels=1, height=4, width=4
    let input = Tensor::from_data(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        vec![1, 1, 4, 4],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=1, in_channels=1, kernel_h=2, kernel_w=2
    let weight = Tensor::from_data(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2], DeviceType::Cpu)?;

    // Perform convolution with stride=(2,2)
    let output = input.conv2d(&weight, None, (2, 2), (0, 0), (1, 1), 1)?;

    // With stride=2, output should be 2x2
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    let output_data = output.data().unwrap();

    // Expected values (sum of 2x2 windows with stride 2):
    // Top-left: 1+2+5+6 = 14
    // Top-right: 3+4+7+8 = 22
    // Bottom-left: 9+10+13+14 = 46
    // Bottom-right: 11+12+15+16 = 54
    assert_relative_eq!(output_data[0], 14.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 22.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 46.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 54.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv2d_multi_channel() -> Result<()> {
    // Input: batch=1, channels=2, height=3, width=3
    let input = Tensor::from_data(
        vec![
            // Channel 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Channel 1
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        vec![1, 2, 3, 3],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=1, in_channels=2, kernel_h=2, kernel_w=2
    let weight = Tensor::from_data(
        vec![
            // Weights for channel 0
            1.0, 0.0, 0.0, 1.0, // Weights for channel 1
            0.0, 1.0, 1.0, 0.0,
        ],
        vec![1, 2, 2, 2],
        DeviceType::Cpu,
    )?;

    // Perform convolution
    let output = input.conv2d(&weight, None, (1, 1), (0, 0), (1, 1), 1)?;

    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    let output_data = output.data().unwrap();

    // Expected output combines both channels:
    // Position (0,0): Ch0:(1*1+2*0+4*0+5*1) + Ch1:(9*0+8*1+6*1+5*0) = 6 + 14 = 20
    assert_relative_eq!(output_data[0], 20.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv3d_simple() -> Result<()> {
    // Create a simple 3D convolution example
    // Input: batch=1, channels=1, depth=2, height=2, width=2
    let input = Tensor::from_data(
        vec![
            // Depth 0
            1.0, 2.0, 3.0, 4.0, // Depth 1
            5.0, 6.0, 7.0, 8.0,
        ],
        vec![1, 1, 2, 2, 2],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=1, in_channels=1, kernel_d=2, kernel_h=2, kernel_w=2
    let weight = Tensor::from_data(
        vec![
            // Depth 0
            1.0, 0.0, 0.0, 0.0, // Depth 1
            0.0, 0.0, 0.0, 1.0,
        ],
        vec![1, 1, 2, 2, 2],
        DeviceType::Cpu,
    )?;

    // Perform convolution
    let output = input.conv3d(&weight, None, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)?;

    // Output should be 1x1x1
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1, 1]);
    let output_data = output.data().unwrap();

    // Expected: 1*1 + 8*1 = 9
    assert_relative_eq!(output_data[0], 9.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv_groups() -> Result<()> {
    // Test grouped convolution
    // Input: batch=1, channels=4, length=4
    let input = Tensor::from_data(
        vec![
            // Channels 0-1 (group 0)
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Channels 2-3 (group 1)
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        vec![1, 4, 4],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=4, in_channels/groups=2, kernel_size=2
    // With groups=2, each group processes 2 input channels to produce 2 output channels
    let weight = Tensor::from_data(
        vec![
            // Group 0, output channel 0
            1.0, 0.0, 0.0, 0.0, // Group 0, output channel 1
            0.0, 0.0, 1.0, 0.0, // Group 1, output channel 2
            1.0, 0.0, 0.0, 0.0, // Group 1, output channel 3
            0.0, 0.0, 0.0, 1.0,
        ],
        vec![4, 2, 2],
        DeviceType::Cpu,
    )?;

    // Perform grouped convolution
    let output = input.conv1d(&weight, None, 1, 0, 1, 2)?;

    assert_eq!(output.shape().dims(), &[1, 4, 3]);

    Ok(())
}

#[test]
fn test_conv_dilation() -> Result<()> {
    // Test dilated convolution
    // Input: batch=1, channels=1, length=7
    let input = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![1, 1, 7],
        DeviceType::Cpu,
    )?;

    // Kernel: out_channels=1, in_channels=1, kernel_size=3
    let weight = Tensor::from_data(vec![1.0, 0.0, 1.0], vec![1, 1, 3], DeviceType::Cpu)?;

    // Perform convolution with dilation=2
    // Effective kernel spans positions: [0, 2, 4]
    let output = input.conv1d(&weight, None, 1, 0, 2, 1)?;

    // Output length: (7 - (3-1)*2 - 1) / 1 + 1 = 3
    assert_eq!(output.shape().dims(), &[1, 1, 3]);
    let output_data = output.data().unwrap();

    // Expected values:
    // Position 0: 1*1 + 3*0 + 5*1 = 6
    // Position 1: 2*1 + 4*0 + 6*1 = 8
    // Position 2: 3*1 + 5*0 + 7*1 = 10
    assert_relative_eq!(output_data[0], 6.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 8.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 10.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_depthwise_conv2d() -> Result<()> {
    // Test depthwise convolution where each input channel has its own kernel
    // Input: batch=1, channels=2, height=3, width=3
    let input = Tensor::from_data(
        vec![
            // Channel 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Channel 1
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        vec![1, 2, 3, 3],
        DeviceType::Cpu,
    )?;

    // Depthwise weight: (2, 1, 2, 2) - each channel has its own 2x2 kernel
    let weight = Tensor::from_data(
        vec![
            // Kernel for channel 0
            1.0, 0.0, 0.0, 1.0, // Kernel for channel 1
            -1.0, 0.0, 0.0, -1.0,
        ],
        vec![2, 1, 2, 2],
        DeviceType::Cpu,
    )?;

    // Perform depthwise convolution
    let output = input.depthwise_conv2d(&weight, None, (1, 1), (0, 0), (1, 1))?;

    // Output shape: (1, 2, 2, 2) - same number of channels as input
    assert_eq!(output.shape().dims(), &[1, 2, 2, 2]);
    let output_data = output.data().unwrap();

    // Channel 0: 1*1 + 5*1 = 6, 2*1 + 6*1 = 8, 4*1 + 8*1 = 12, 5*1 + 9*1 = 14
    assert_relative_eq!(output_data[0], 6.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 8.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 12.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 14.0, epsilon = 1e-6);

    // Channel 1: 9*(-1) + 5*(-1) = -14, 8*(-1) + 4*(-1) = -12, 6*(-1) + 2*(-1) = -8, 5*(-1) + 1*(-1) = -6
    assert_relative_eq!(output_data[4], -14.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[5], -12.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[6], -8.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[7], -6.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_depthwise_conv2d_with_bias() -> Result<()> {
    // Input: batch=1, channels=2, height=2, width=2
    let input = Tensor::from_data(
        vec![
            1.0, 2.0, 3.0, 4.0, // Channel 0
            5.0, 6.0, 7.0, 8.0, // Channel 1
        ],
        vec![1, 2, 2, 2],
        DeviceType::Cpu,
    )?;

    // Depthwise weight: (2, 1, 1, 1) - 1x1 kernels
    let weight = Tensor::from_data(
        vec![2.0, 3.0], // Channel 0: *2, Channel 1: *3
        vec![2, 1, 1, 1],
        DeviceType::Cpu,
    )?;

    // Bias for each channel
    let bias = Tensor::from_data(vec![1.0, -1.0], vec![2], DeviceType::Cpu)?;

    let output = input.depthwise_conv2d(&weight, Some(&bias), (1, 1), (0, 0), (1, 1))?;

    assert_eq!(output.shape().dims(), &[1, 2, 2, 2]);
    let output_data = output.data().unwrap();

    // Channel 0: [1*2+1, 2*2+1, 3*2+1, 4*2+1] = [3, 5, 7, 9]
    assert_relative_eq!(output_data[0], 3.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 5.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 7.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 9.0, epsilon = 1e-6);

    // Channel 1: [5*3-1, 6*3-1, 7*3-1, 8*3-1] = [14, 17, 20, 23]
    assert_relative_eq!(output_data[4], 14.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[5], 17.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[6], 20.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[7], 23.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_separable_conv2d() -> Result<()> {
    // Test separable convolution (depthwise followed by pointwise)
    // Input: batch=1, channels=2, height=3, width=3
    let input = Tensor::from_data(
        vec![
            // Channel 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Channel 1
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        vec![1, 2, 3, 3],
        DeviceType::Cpu,
    )?;

    // Depthwise weight: (2, 1, 2, 2)
    let depthwise_weight = Tensor::from_data(
        vec![
            // Kernel for channel 0
            1.0, 0.0, 0.0, 1.0, // Kernel for channel 1
            0.5, 0.5, 0.5, 0.5,
        ],
        vec![2, 1, 2, 2],
        DeviceType::Cpu,
    )?;

    // Pointwise weight: (3, 2, 1, 1) - 3 output channels from 2 input channels
    let pointwise_weight = Tensor::from_data(
        vec![
            1.0, 0.0, // Output channel 0: only from depthwise channel 0
            0.0, 1.0, // Output channel 1: only from depthwise channel 1
            0.5, 0.5, // Output channel 2: combination of both
        ],
        vec![3, 2, 1, 1],
        DeviceType::Cpu,
    )?;

    // Perform separable convolution
    let output = input.separable_conv2d(
        &depthwise_weight,
        &pointwise_weight,
        None,
        (1, 1),
        (0, 0),
        (1, 1),
    )?;

    // Output shape: (1, 3, 2, 2)
    assert_eq!(output.shape().dims(), &[1, 3, 2, 2]);

    Ok(())
}

#[test]
fn test_conv_transpose2d() -> Result<()> {
    // Test transposed convolution (deconvolution)
    // Input: batch=1, channels=1, height=2, width=2
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], DeviceType::Cpu)?;

    // Weight: (1, 1, 2, 2) - for transposed conv, input_channels x output_channels x kernel_h x kernel_w
    let weight = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], DeviceType::Cpu)?;

    // Perform transposed convolution with stride=1, padding=0
    let output = input.conv_transpose2d(&weight, None, (1, 1), (0, 0), (0, 0), (1, 1), 1)?;

    // Output should be larger than input: (1, 1, 3, 3)
    assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);
    let output_data = output.data().unwrap();

    // Each input pixel gets multiplied by the kernel and placed at its corresponding output position
    // This creates overlap between adjacent kernels

    // Position (0,0): 1*1 = 1
    assert_relative_eq!(output_data[0], 1.0, epsilon = 1e-6);
    // Position (0,1): 1*2 + 2*1 = 4
    assert_relative_eq!(output_data[1], 4.0, epsilon = 1e-6);
    // Position (0,2): 2*2 = 4
    assert_relative_eq!(output_data[2], 4.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv_transpose2d_with_stride() -> Result<()> {
    // Test transposed convolution with stride > 1
    // Input: batch=1, channels=1, height=2, width=2
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], DeviceType::Cpu)?;

    // Weight: (1, 1, 2, 2)
    let weight = Tensor::from_data(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2], DeviceType::Cpu)?;

    // Perform transposed convolution with stride=2
    let output = input.conv_transpose2d(&weight, None, (2, 2), (0, 0), (0, 0), (1, 1), 1)?;

    // Output should be much larger due to stride=2: (1, 1, 4, 4)
    assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);
    let output_data = output.data().unwrap();

    // With stride=2, kernels are placed at non-overlapping positions
    // Input pixel at (0,0) -> kernel at output (0:2, 0:2)
    // Input pixel at (0,1) -> kernel at output (0:2, 2:4)
    // Input pixel at (1,0) -> kernel at output (2:4, 0:2)
    // Input pixel at (1,1) -> kernel at output (2:4, 2:4)

    assert_relative_eq!(output_data[0], 1.0, epsilon = 1e-6); // (0,0)
    assert_relative_eq!(output_data[1], 1.0, epsilon = 1e-6); // (0,1)
    assert_relative_eq!(output_data[2], 2.0, epsilon = 1e-6); // (0,2)
    assert_relative_eq!(output_data[3], 2.0, epsilon = 1e-6); // (0,3)

    Ok(())
}

#[test]
fn test_conv_transpose2d_with_bias() -> Result<()> {
    // Input: batch=1, channels=1, height=1, width=1
    let input = Tensor::from_data(vec![5.0], vec![1, 1, 1, 1], DeviceType::Cpu)?;

    // Weight: (1, 2, 2, 2) - produces 2 output channels
    let weight = Tensor::from_data(
        vec![
            // Channel 0
            1.0, 2.0, 3.0, 4.0, // Channel 1
            0.1, 0.2, 0.3, 0.4,
        ],
        vec![1, 2, 2, 2],
        DeviceType::Cpu,
    )?;

    // Bias for 2 output channels
    let bias = Tensor::from_data(vec![10.0, 20.0], vec![2], DeviceType::Cpu)?;

    let output = input.conv_transpose2d(&weight, Some(&bias), (1, 1), (0, 0), (0, 0), (1, 1), 1)?;

    // Output: (1, 2, 2, 2)
    assert_eq!(output.shape().dims(), &[1, 2, 2, 2]);
    let output_data = output.data().unwrap();

    // Channel 0: 5*weight + 10
    assert_relative_eq!(output_data[0], 5.0 * 1.0 + 10.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 5.0 * 2.0 + 10.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 5.0 * 3.0 + 10.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 5.0 * 4.0 + 10.0, epsilon = 1e-6);

    // Channel 1: 5*weight + 20
    assert_relative_eq!(output_data[4], 5.0 * 0.1 + 20.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[5], 5.0 * 0.2 + 20.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[6], 5.0 * 0.3 + 20.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[7], 5.0 * 0.4 + 20.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_conv2d_dilation_advanced() -> Result<()> {
    // Test 2D dilated convolution more thoroughly
    // Input: batch=1, channels=1, height=5, width=5
    let input = Tensor::from_data(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
        ],
        vec![1, 1, 5, 5],
        DeviceType::Cpu,
    )?;

    // 3x3 kernel
    let weight = Tensor::from_data(
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        vec![1, 1, 3, 3],
        DeviceType::Cpu,
    )?;

    // Apply with dilation=(2, 2) - kernel spans 5x5 area but only samples at dilated positions
    let output = input.conv2d(&weight, None, (1, 1), (0, 0), (2, 2), 1)?;

    // With dilation=2, effective kernel size is 5x5, so output is 1x1
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);
    let output_data = output.data().unwrap();

    // Sampled positions with dilation=2 starting from (0,0):
    // (0,0), (0,4), (2,2), (4,0), (4,4) = 1, 5, 13, 21, 25
    // Result: 1*1 + 5*1 + 13*1 + 21*1 + 25*1 = 65
    assert_relative_eq!(output_data[0], 65.0, epsilon = 1e-6);

    Ok(())
}

// ==================== SIGNAL PROCESSING TESTS ====================

#[test]
fn test_xcorr1d_full() -> Result<()> {
    use torsh_tensor::conv::CorrelationMode;

    // Create simple 1D signals for cross-correlation
    let signal1 = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    let signal2 = Tensor::from_data(vec![0.5, 1.0], vec![2], DeviceType::Cpu)?;

    // Compute full cross-correlation
    let result = signal1.xcorr1d(&signal2, CorrelationMode::Full)?;

    // Full mode: output size = len(signal1) + len(signal2) - 1 = 3 + 2 - 1 = 4
    assert_eq!(result.shape().dims(), &[4]);

    let output_data = result.to_vec()?;

    // Expected cross-correlation values:
    // lag -1: 1*1 = 1
    // lag 0: 1*0.5 + 2*1 = 2.5
    // lag 1: 2*0.5 + 3*1 = 4
    // lag 2: 3*0.5 = 1.5
    assert_relative_eq!(output_data[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 2.5, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 4.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 1.5, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_xcorr1d_same() -> Result<()> {
    use torsh_tensor::conv::CorrelationMode;

    let signal1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)?;
    let signal2 = Tensor::from_data(vec![1.0, 1.0, 1.0], vec![3], DeviceType::Cpu)?;

    // Compute same-size cross-correlation
    let result = signal1.xcorr1d(&signal2, CorrelationMode::Same)?;

    // Same mode: output size = len(signal1) = 4
    assert_eq!(result.shape().dims(), &[4]);

    let output_data = result.to_vec()?;

    // Expected: moving average-like operation
    // Position 0: 0 + 1*1 + 2*1 = 3 (with edge padding)
    // Position 1: 1*1 + 2*1 + 3*1 = 6
    // Position 2: 2*1 + 3*1 + 4*1 = 9
    // Position 3: 3*1 + 4*1 + 0 = 7 (with edge padding)
    assert_relative_eq!(output_data[0], 3.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[1], 6.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[2], 9.0, epsilon = 1e-6);
    assert_relative_eq!(output_data[3], 7.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_autocorr1d() -> Result<()> {
    // Create a simple periodic signal
    let signal = Tensor::from_data(vec![1.0, 0.0, -1.0, 0.0], vec![4], DeviceType::Cpu)?;

    // Compute auto-correlation with max lag = 3
    let result = signal.autocorr1d(Some(3))?;

    // Output should have max_lag + 1 = 4 elements
    assert_eq!(result.shape().dims(), &[4]);

    let output_data = result.to_vec()?;

    // Auto-correlation at lag 0 should be highest (signal energy)
    // For the signal [1, 0, -1, 0]: energy = 1*1 + 0*0 + (-1)*(-1) + 0*0 = 2
    assert_relative_eq!(output_data[0], 2.0, epsilon = 1e-6);

    // At lag 2, we should see some correlation due to the pattern
    // lag 2: 1*(-1) + 0*0 = -1
    assert_relative_eq!(output_data[2], -1.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_xcorr2d_same() -> Result<()> {
    use torsh_tensor::conv::CorrelationMode;

    // Create 2D signals for cross-correlation
    let signal1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)?;
    let signal2 = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], DeviceType::Cpu)?;

    // Compute 2D cross-correlation with same mode
    let result = signal1.xcorr2d(&signal2, CorrelationMode::Same)?;

    // Same mode: output size = input size = (2, 2)
    assert_eq!(result.shape().dims(), &[2, 2]);

    let output_data = result.to_vec()?;

    // Expected correlation values depend on how the kernel aligns
    // This tests that the operation completes without error and produces expected shape
    assert_eq!(output_data.len(), 4);

    Ok(())
}

#[test]
fn test_median_filter1d() -> Result<()> {
    // Create a signal with some noise
    let signal = Tensor::from_data(
        vec![1.0, 5.0, 2.0, 3.0, 8.0, 4.0, 3.0],
        vec![7],
        DeviceType::Cpu,
    )?;

    // Apply median filter with window size 3
    let result = signal.median_filter1d(3)?;

    // Output should have same size as input
    assert_eq!(result.shape().dims(), &[7]);

    let output_data = result.to_vec()?;

    // Median filter should reduce the spike at position 1 (value 5) and position 4 (value 8)
    // For window [1,5,2] -> median = 2
    assert_relative_eq!(output_data[1], 2.0, epsilon = 1e-6);

    // For window [3,8,4] -> median = 4
    assert_relative_eq!(output_data[4], 4.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_median_filter2d() -> Result<()> {
    // Create a 3x3 signal with a spike in the center
    let signal = Tensor::from_data(
        vec![
            1.0, 1.0, 1.0, 1.0, 9.0, 1.0, // spike at center
            1.0, 1.0, 1.0,
        ],
        vec![3, 3],
        DeviceType::Cpu,
    )?;

    // Apply 3x3 median filter
    let result = signal.median_filter2d((3, 3))?;

    // Output should have same size as input
    assert_eq!(result.shape().dims(), &[3, 3]);

    let output_data = result.to_vec()?;

    // The spike at center (index 4) should be reduced to the median value
    // In a 3x3 window of mostly 1s with one 9, median = 1
    assert_relative_eq!(output_data[4], 1.0, epsilon = 1e-6);

    Ok(())
}

#[test]
fn test_gaussian_filter1d() -> Result<()> {
    // Create a simple step signal
    let signal = Tensor::from_data(
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        vec![7],
        DeviceType::Cpu,
    )?;

    // Apply Gaussian filter with sigma = 1.0
    let result = signal.gaussian_filter1d(1.0, Some(5))?;

    // Output should have same size as input
    assert_eq!(result.shape().dims(), &[7]);

    let output_data = result.to_vec()?;

    // Gaussian filter should smooth the step edges
    // The center value should be close to 1.0 but edges should be smoothed
    assert!(output_data[3] > 0.5); // Center should still be relatively high
    assert!(output_data[1] > 0.0); // Edges should have some non-zero value due to smoothing
    assert!(output_data[5] > 0.0);

    Ok(())
}

#[test]
fn test_gaussian_filter2d() -> Result<()> {
    // Create a 2D impulse signal (spike at center)
    let signal = Tensor::from_data(
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![3, 3],
        DeviceType::Cpu,
    )?;

    // Apply 2D Gaussian filter
    let result = signal.gaussian_filter2d((0.8, 0.8), Some((3, 3)))?;

    // Output should have same size as input
    assert_eq!(result.shape().dims(), &[3, 3]);

    let output_data = result.to_vec()?;

    // The center should still have the highest value
    let center_val = output_data[4];
    assert!(center_val > output_data[0]); // Center > corner
    assert!(center_val > output_data[1]); // Center > edge
    assert!(center_val > output_data[3]); // Center > adjacent

    Ok(())
}

#[test]
fn test_signal_processing_error_handling() -> Result<()> {
    use torsh_tensor::conv::CorrelationMode;

    let signal = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    let signal_2d = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)?;

    // Test dimension mismatch errors
    assert!(signal_2d.xcorr1d(&signal, CorrelationMode::Full).is_err());
    assert!(signal.xcorr2d(&signal_2d, CorrelationMode::Full).is_err());

    // Test median filter with even window size (should fail)
    assert!(signal.median_filter1d(4).is_err());
    assert!(signal_2d.median_filter2d((2, 3)).is_err());

    // Test Gaussian filter with invalid sigma
    assert!(signal.gaussian_filter1d(-1.0, None).is_err());
    assert!(signal_2d.gaussian_filter2d((-1.0, 1.0), None).is_err());

    Ok(())
}
