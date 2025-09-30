//! Comprehensive integration tests for torsh-nn
//!
//! This test suite validates end-to-end neural network functionality including
//! model construction, training loops, serialization, and real-world scenarios.

use torsh_core::TorshError;
use torsh_nn::container::Sequential;
use torsh_nn::functional::losses::l1_loss;
use torsh_nn::functional::mse_loss;
use torsh_nn::layers::*;
use torsh_nn::research::*;
use torsh_nn::Module;
use torsh_tensor::creation::*;

type Result<T> = std::result::Result<T, TorshError>;

/// Test basic MLP construction and forward pass
#[test]
fn test_mlp_end_to_end() -> Result<()> {
    // Create a simple MLP for MNIST-like classification
    let model = Sequential::new()
        .add(Linear::new(784, 256, true))
        .add(ReLU::new())
        .add(Dropout::new(0.2))
        .add(Linear::new(256, 128, true))
        .add(ReLU::new())
        .add(Dropout::new(0.2))
        .add(Linear::new(128, 10, true))
        .add(Softmax::new(Some(1)));

    // Test forward pass with different batch sizes
    let batch_sizes = vec![1, 8, 32];
    for batch_size in batch_sizes {
        let input = randn::<f32>(&[batch_size, 784])?;
        let output = model.forward(&input)?;

        assert_eq!(output.shape().dims(), &[batch_size, 10]);

        // Check that softmax outputs sum to approximately 1
        let sum = output.sum_dim(&[1], true)?;
        let sum_data = sum.data()?;
        for &val in sum_data.iter() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Softmax output doesn't sum to 1: {}",
                val
            );
        }
    }
    Ok(())
}

/// Test CNN construction and forward pass
#[test]
fn test_cnn_end_to_end() -> Result<()> {
    // Create a simple CNN for image classification
    let model = Sequential::new()
        // Convolutional layers
        .add(Conv2d::new(3, 32, (3, 3), (1, 1), (1, 1), (1, 1), false, 1))
        .add(BatchNorm2d::new(32)?)
        .add(ReLU::new())
        .add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
        .add(Conv2d::new(
            32,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ))
        .add(BatchNorm2d::new(64)?)
        .add(ReLU::new())
        .add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
        // Global average pooling and classifier
        .add(AdaptiveAvgPool2d::new((Some(1), Some(1))))
        .add(Flatten::new())  // Flatten from [batch, 64, 1, 1] to [batch, 64]
        .add(Linear::new(64, 10, true))
        .add(Softmax::new(Some(1)));

    // Test with CIFAR-10 like input
    let input = randn::<f32>(&[4, 3, 32, 32])?;
    let output = model.forward(&input)?;

    assert_eq!(output.shape().dims(), &[4, 10]);
    Ok(())
}

/// Test ResNet-like architecture with skip connections
#[test]
fn test_resnet_blocks() -> Result<()> {
    use torsh_nn::layers::blocks::*;

    // Test basic block
    let basic_block = BasicBlock::new(64, 64, 1, None)?;
    let input = randn::<f32>(&[2, 64, 32, 32])?;
    let output = basic_block.forward(&input)?;
    assert_eq!(output.shape().dims(), &[2, 64, 32, 32]);

    // Test basic block with downsampling
    let downsample_block = BasicBlock::with_downsample(64, 128, 2)?;
    let output2 = downsample_block.forward(&input)?;
    assert_eq!(output2.shape().dims(), &[2, 128, 16, 16]);

    // Test bottleneck block
    let bottleneck = BottleneckBlock::new(256, 256, 1, None)?;
    let bottleneck_input = randn::<f32>(&[2, 256, 16, 16])?;
    let bottleneck_output = bottleneck.forward(&bottleneck_input)?;
    assert_eq!(bottleneck_output.shape().dims(), &[2, 256, 16, 16]);
    Ok(())
}

/// Test attention mechanisms
#[test]
#[ignore = "Attention mechanisms need tensor shape handling fixes"]
fn test_attention_mechanisms() -> Result<()> {
    let batch_size = 2;
    let seq_len = 10;
    let embed_dim = 64;
    let num_heads = 8;

    // Test multi-head attention
    let mha = MultiheadAttention::with_config(
        embed_dim, num_heads, 0.1, true, false, false, None, None, false,
    );
    let input = randn::<f32>(&[batch_size, seq_len, embed_dim])?;
    let output = mha.forward(&input)?;

    assert_eq!(output.shape().dims(), &[batch_size, seq_len, embed_dim]);

    // Test self-attention vs cross-attention
    let query = randn::<f32>(&[batch_size, seq_len, embed_dim])?;
    let key = randn::<f32>(&[batch_size, seq_len / 2, embed_dim])?;
    let value = randn::<f32>(&[batch_size, seq_len / 2, embed_dim])?;

    let cross_attn_output = mha.forward_cross_attention(&query, &key, Some(&value))?;
    assert_eq!(
        cross_attn_output.shape().dims(),
        &[batch_size, seq_len, embed_dim]
    );
    Ok(())
}

/// Test transformer components
#[test]
#[ignore = "Transformer components need tensor shape handling fixes"]
fn test_transformer_components() -> Result<()> {
    let d_model = 128;
    let nhead = 8;
    let dim_feedforward = 512;
    let seq_len = 20;
    let batch_size = 2;

    // Test encoder layer
    let encoder_layer = TransformerEncoderLayer::new(
        d_model,
        nhead,
        Some(dim_feedforward),
        Some(0.1),
        Some("relu".to_string()),
        None,
        None,
        None,
    )?;

    let input = randn::<f32>(&[batch_size, seq_len, d_model])?;
    let encoder_output = encoder_layer.forward(&input)?;
    assert_eq!(
        encoder_output.shape().dims(),
        &[batch_size, seq_len, d_model]
    );

    // Test decoder layer
    // TODO: TransformerDecoderLayer is not yet implemented
    // let decoder_layer = TransformerDecoderLayer::new(d_model, nhead, dim_feedforward, 0.1, "relu")?;
    // let decoder_output = decoder_layer.forward(&input)?;
    // assert_eq!(
    //     decoder_output.shape().dims(),
    //     &[batch_size, seq_len, d_model]
    // );
    Ok(())
}

/// Test RNN/LSTM/GRU end-to-end
#[test]
fn test_recurrent_networks() -> Result<()> {
    let input_size = 50;
    let hidden_size = 100;
    let num_layers = 2;
    let seq_len = 15;
    let batch_size = 3;

    let input = randn::<f32>(&[batch_size, seq_len, input_size])?;

    // Test RNN
    let rnn = RNN::new(input_size, hidden_size, num_layers)?;
    let rnn_output = rnn.forward(&input)?;
    assert_eq!(
        rnn_output.shape().dims(),
        &[batch_size, seq_len, hidden_size]
    );

    // Test LSTM
    let lstm = LSTM::new(input_size, hidden_size, num_layers)?;
    let lstm_output = lstm.forward(&input)?;
    assert_eq!(
        lstm_output.shape().dims(),
        &[batch_size, seq_len, hidden_size]
    );

    // Test GRU
    let gru = GRU::new(input_size, hidden_size, num_layers)?;
    let gru_output = gru.forward(&input)?;
    assert_eq!(
        gru_output.shape().dims(),
        &[batch_size, seq_len, hidden_size]
    );

    // Test bidirectional GRU
    let bi_gru = GRU::with_config(input_size, hidden_size, num_layers, true, false, 0.0, true)?;
    let bi_output = bi_gru.forward(&input)?;
    assert_eq!(
        bi_output.shape().dims(),
        &[batch_size, seq_len, hidden_size * 2]
    );
    Ok(())
}

/// Test research components (Neural ODE, DARTS, etc.)
#[test]
#[ignore = "Research components need implementation fixes"]
fn test_research_components() -> Result<()> {
    let batch_size = 2;
    let hidden_dim = 32;
    let input = randn::<f32>(&[batch_size, hidden_dim])?;

    // Test Neural ODE
    let ode_func = Sequential::new()
        .add(Linear::new(hidden_dim, hidden_dim * 2, true))
        .add(ReLU::new())
        .add(Linear::new(hidden_dim * 2, hidden_dim, true));

    let neural_ode = NeuralODE::new(Box::new(ode_func), ODESolver::Euler, 1e-6, 0.1, 10);
    let ode_output = neural_ode.forward(&input)?;
    assert_eq!(ode_output.shape().dims(), &[batch_size, hidden_dim]);

    // Test DARTS cell
    let operations: Vec<Box<dyn Module>> = vec![
        Box::new(Linear::new(hidden_dim, hidden_dim, true)),
        Box::new(Linear::new(hidden_dim, hidden_dim, false)),
    ];
    let darts_cell = DARTSCell::new(operations, 4)?;
    let darts_output = darts_cell.forward(&input)?;
    assert_eq!(darts_output.shape().dims(), &[batch_size, hidden_dim]);

    // Test Capsule Layer
    let in_capsules = 4;
    let out_capsules = 2;
    let in_dim = 8;
    let out_dim = 16;
    let capsule_input = randn::<f32>(&[batch_size, in_capsules, in_dim])?;
    let capsule_layer = CapsuleLayer::new(in_capsules, out_capsules, in_dim, out_dim, 3)?;
    let capsule_output = capsule_layer.forward(&capsule_input)?;
    assert_eq!(
        capsule_output.shape().dims(),
        &[batch_size, out_capsules, out_dim]
    );
    Ok(())
}

/// Test graph neural networks
#[test]
fn test_graph_neural_networks() -> Result<()> {
    let num_nodes = 10;
    let in_features = 16;
    let out_features = 32;
    let batch_size = 1; // Graphs typically use batch_size=1

    let node_features = randn::<f32>(&[batch_size, num_nodes, in_features])?;
    let adjacency = randn::<f32>(&[num_nodes, num_nodes])?;

    // Test Graph Convolution
    let gcn = GraphConvLayer::new(in_features, out_features, true)?;
    // Reshape input to 2D for graph conv layer (flattening batch and nodes)
    let gcn_input = node_features.view(&[(batch_size * num_nodes) as i32, in_features as i32])?;
    let gcn_output = gcn.forward(&gcn_input)?;
    assert_eq!(
        gcn_output.shape().dims(),
        &[batch_size * num_nodes, out_features]
    );

    // Test Graph Attention
    let gat = GraphAttentionLayer::new(in_features, out_features, 4, 0.2, 0.2)?;
    let gat_output = gat.forward(&gcn_input)?;
    assert_eq!(
        gat_output.shape().dims(),
        &[batch_size * num_nodes, out_features]
    );
    Ok(())
}

/// Test model parameter management
#[test]
fn test_parameter_management() -> Result<()> {
    let model = Sequential::new()
        .add(Linear::new(100, 50, true))
        .add(ReLU::new())
        .add(Linear::new(50, 10, true));

    // Test parameter retrieval
    let params = model.parameters();
    assert!(!params.is_empty());

    // Test named parameters
    let named_params = model.named_parameters();
    assert!(!named_params.is_empty());

    // Check that parameter shapes are correct
    for (name, param) in named_params.iter() {
        let tensor = param.tensor();
        assert!(
            !tensor.read().shape().dims().is_empty(),
            "Parameter {} has empty shape",
            name
        );
    }
    Ok(())
}

/// Test training mode switching
#[test]
fn test_training_mode() -> Result<()> {
    let mut model = Sequential::new()
        .add(Linear::new(50, 25, true))
        .add(BatchNorm1d::new(25)?)
        .add(Dropout::new(0.5))
        .add(Linear::new(25, 10, true));

    // Test training mode
    model.train();
    assert!(model.training());

    // Test evaluation mode
    model.eval();
    assert!(!model.training());

    // Test forward pass in both modes
    let input = randn::<f32>(&[4, 50])?;

    model.train();
    let train_output = model.forward(&input)?;
    assert_eq!(train_output.shape().dims(), &[4, 10]);

    model.eval();
    let eval_output = model.forward(&input)?;
    assert_eq!(eval_output.shape().dims(), &[4, 10]);
    Ok(())
}

/// Test normalization layers with different inputs
#[test]
fn test_normalization_layers() -> Result<()> {
    let batch_size = 4;
    let channels = 32;
    let height = 16;
    let width = 16;

    let input = randn::<f32>(&[batch_size, channels, height, width])?;

    // Test BatchNorm2d
    let bn = BatchNorm2d::new(channels)?;
    let bn_output = bn.forward(&input)?;
    assert_eq!(
        bn_output.shape().dims(),
        &[batch_size, channels, height, width]
    );

    // Test LayerNorm
    let ln = LayerNorm::new(vec![channels, height, width])?;
    let ln_output = ln.forward(&input)?;
    assert_eq!(
        ln_output.shape().dims(),
        &[batch_size, channels, height, width]
    );

    // Test GroupNorm
    let gn = GroupNorm::new(8, channels)?; // 8 groups
    let gn_output = gn.forward(&input)?;
    assert_eq!(
        gn_output.shape().dims(),
        &[batch_size, channels, height, width]
    );

    // Test InstanceNorm2d
    let instance_norm = InstanceNorm2d::new(channels)?;
    let in_output = instance_norm.forward(&input)?;
    assert_eq!(
        in_output.shape().dims(),
        &[batch_size, channels, height, width]
    );
    Ok(())
}

/// Test activation functions with edge cases
#[test]
fn test_activation_functions() -> Result<()> {
    let input = randn::<f32>(&[8, 64])?;

    // Test ReLU
    let relu = ReLU::new();
    let relu_output = relu.forward(&input)?;
    assert_eq!(relu_output.shape().dims(), input.shape().dims());

    // Test LeakyReLU
    let leaky_relu = LeakyReLU::new(0.01);
    let leaky_output = leaky_relu.forward(&input)?;
    assert_eq!(leaky_output.shape().dims(), input.shape().dims());

    // Test GELU
    let gelu = GELU::new();
    let gelu_output = gelu.forward(&input)?;
    assert_eq!(gelu_output.shape().dims(), input.shape().dims());

    // Test Softmax with different dimensions
    let softmax_1d = Softmax::new(Some(1));
    let output_1d = softmax_1d.forward(&input)?;
    assert_eq!(output_1d.shape().dims(), &[8, 64]);

    // Verify softmax properties (sum to 1)
    let sum = output_1d.sum_dim(&[1], true)?;
    let sum_data = sum.data()?;
    for &val in sum_data.iter() {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "Softmax doesn't sum to 1: {}",
            val
        );
    }
    Ok(())
}

/// Test loss functions integration
#[test]
fn test_loss_functions() -> Result<()> {
    let batch_size = 8;
    let num_classes = 10;
    let predictions = randn::<f32>(&[batch_size, num_classes])?;
    let targets = randn::<f32>(&[batch_size, num_classes])?;

    // Test MSE using functional interface
    let mse_loss_result = mse_loss(&predictions, &targets, "mean")?;
    // Scalar losses have empty shape [] (like PyTorch)
    assert!(mse_loss_result.shape().dims().len() <= 1,
            "Loss should be scalar or 1D, got shape {:?}", mse_loss_result.shape().dims());

    // Test L1 using functional interface
    let l1_loss_result = l1_loss(&predictions, &targets, "mean")?;
    assert!(l1_loss_result.shape().dims().len() <= 1,
            "Loss should be scalar or 1D, got shape {:?}", l1_loss_result.shape().dims());

    // All losses should be non-negative
    let mse_data = mse_loss_result.data()?;
    let l1_data = l1_loss_result.data()?;
    assert!(mse_data[0] >= 0.0, "MSE loss should be non-negative");
    assert!(l1_data[0] >= 0.0, "L1 loss should be non-negative");
    Ok(())
}

/// Test mixed precision and quantization (if available)
#[test]
fn test_mixed_precision_quantization() -> Result<()> {
    // This test validates mixed precision and quantization components
    // when they become available in the implementation

    let input = randn::<f32>(&[4, 128])?;
    let linear = Linear::new(128, 64, true);

    // Standard forward pass (FP32)
    let output = linear.forward(&input)?;
    assert_eq!(output.shape().dims(), &[4, 64]);

    // TODO: Add mixed precision tests when AutocastModel is available
    // TODO: Add quantization tests when QAT layers are available
    Ok(())
}

/// Test memory efficiency and gradient computation
#[test]
fn test_gradient_computation() -> Result<()> {
    // This test validates gradient computation capabilities
    let input = randn::<f32>(&[2, 50])?;
    let target = randn::<f32>(&[2, 10])?;

    // Create a simple model
    let model = Sequential::new()
        .add(Linear::new(50, 25, true))
        .add(ReLU::new())
        .add(Linear::new(25, 10, true));

    // Forward pass
    let output = model.forward(&input)?;
    assert_eq!(output.shape().dims(), &[2, 10]);

    // Compute loss using functional interface
    let loss = mse_loss(&output, &target, "mean")?;

    // Verify loss is scalar or 1D (scalar losses have empty shape [] in PyTorch style)
    assert!(loss.shape().dims().len() <= 1,
            "Loss should be scalar or 1D, got shape {:?}", loss.shape().dims());

    // TODO: Add backward pass testing when autograd is fully integrated
    Ok(())
}
