//! Simple CNN example using ToRSh
//!
//! This example demonstrates how to build a simple Convolutional Neural Network
//! using ToRSh's Conv2d layers.

use torsh::prelude::*;
use torsh_nn::modules::{Conv2d, Linear, ReLU};
use torsh_nn::Module;

/// A simple CNN model for image classification
struct SimpleCNN {
    conv1: Conv2d,
    relu1: ReLU,
    conv2: Conv2d,
    relu2: ReLU,
    fc1: Linear,
    relu3: ReLU,
    fc2: Linear,
}

impl SimpleCNN {
    fn new(num_classes: usize) -> Self {
        Self {
            // First convolutional layer: 3 input channels -> 32 output channels, 3x3 kernel
            conv1: Conv2d::new(3, 32, (3, 3), (1, 1), (1, 1), (1, 1), true, 1),
            relu1: ReLU::new(),

            // Second convolutional layer: 32 -> 64 channels, 3x3 kernel
            conv2: Conv2d::new(32, 64, (3, 3), (1, 1), (1, 1), (1, 1), true, 1),
            relu2: ReLU::new(),

            // Fully connected layers
            // Note: In a real CNN, we'd typically have pooling layers to reduce dimensions
            // For now, we'll use a smaller FC layer input size
            // With 32x32 input and two 3x3 convs with padding=1, output is still 32x32
            fc1: Linear::new(64 * 32 * 32, 128, true),
            relu3: ReLU::new(),
            fc2: Linear::new(128, num_classes, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply first conv block
        let x = self.conv1.forward(x)?;
        let x = self.relu1.forward(&x)?;

        // Apply second conv block
        let x = self.conv2.forward(&x)?;
        let x = self.relu2.forward(&x)?;

        // Flatten the tensor for the fully connected layers
        let shape = x.shape();
        let shape_dims = shape.dims();
        let batch_size = shape_dims[0] as i32;
        let flattened_size = (shape_dims[1] * shape_dims[2] * shape_dims[3]) as i32;
        println!(
            "Before flatten: {:?}, flattened size: {}",
            shape_dims, flattened_size
        );

        let x = x.view(&[batch_size, flattened_size])?;
        println!("After flatten: {:?}", x.shape().dims());

        // Apply fully connected layers
        let x = self.fc1.forward(&x)?;
        let x = self.relu3.forward(&x)?;
        let x = self.fc2.forward(&x)?;

        Ok(x)
    }
}

fn main() -> Result<()> {
    println!("ToRSh Simple CNN Example");
    println!("========================\n");

    // Create a simple CNN for 10-class classification
    let model = SimpleCNN::new(10);

    // Create a batch of random input images
    // Shape: [batch_size=4, channels=3, height=32, width=32]
    let input = randn(&[4, 3, 32, 32]);
    println!("Input shape: {:?}", input.shape().dims());

    // Forward pass through the model
    let output = model.forward(&input)?;
    println!("Output shape: {:?}", output.shape().dims());

    // The output should have shape [4, 10] - batch_size x num_classes
    assert_eq!(output.shape().dims(), &[4, 10]);

    println!("\nModel architecture:");
    println!("Conv2d(3, 32) -> ReLU -> Conv2d(32, 64) -> ReLU -> Flatten -> Linear(65536, 128) -> ReLU -> Linear(128, 10)");

    // Demonstrate parameter counting
    let conv1_params = model.conv1.parameters().len();
    let conv2_params = model.conv2.parameters().len();
    let fc1_params = model.fc1.parameters().len();
    let fc2_params = model.fc2.parameters().len();

    println!("\nParameter counts:");
    println!("Conv1: {} parameters", conv1_params);
    println!("Conv2: {} parameters", conv2_params);
    println!("FC1: {} parameters", fc1_params);
    println!("FC2: {} parameters", fc2_params);
    println!(
        "Total: {} parameters",
        conv1_params + conv2_params + fc1_params + fc2_params
    );

    // Show individual layer outputs
    println!("\nLayer-wise output shapes:");
    let x1 = model.conv1.forward(&input)?;
    println!("After Conv1: {:?}", x1.shape().dims());

    let x2 = model.conv2.forward(&x1)?;
    println!("After Conv2: {:?}", x2.shape().dims());

    let batch_size = x2.shape().dims()[0] as i32;
    let x_flat = x2.view(&[batch_size, -1])?;
    println!("After Flatten: {:?}", x_flat.shape().dims());

    let x3 = model.fc1.forward(&x_flat)?;
    println!("After FC1: {:?}", x3.shape().dims());

    let x4 = model.fc2.forward(&x3)?;
    println!("After FC2 (output): {:?}", x4.shape().dims());

    println!("\nCNN forward pass completed successfully!");

    Ok(())
}
