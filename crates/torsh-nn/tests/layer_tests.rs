//! Tests for neural network layers

use torsh_nn::modules::*;
use torsh_nn::Module;
use torsh_tensor::creation::*;

#[test]
fn test_maxpool2d_basic() {
    let pool = MaxPool2d::new(
        (2, 2),
        Some((2, 2)),
        Some((0, 0)),
        Some((1, 1)),
        false,
        false,
    );
    
    let input = zeros(&[1, 3, 4, 4]);
    let output = pool.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[1, 3, 2, 2]);
}

#[test]
fn test_avgpool2d_basic() {
    let pool = AvgPool2d::new(
        (2, 2),
        Some((2, 2)),
        Some((0, 0)),
        false,
        true,
        None,
    );
    
    let input = zeros(&[1, 3, 4, 4]);
    let output = pool.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[1, 3, 2, 2]);
}

#[test]
fn test_adaptive_avgpool2d_basic() {
    let pool = AdaptiveAvgPool2d::new((Some(1), Some(1)));
    
    let input = zeros(&[1, 3, 4, 4]);
    let output = pool.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[1, 3, 1, 1]);
}

#[test]
fn test_batchnorm2d_basic() {
    let bn = BatchNorm2d::new(3, None, None, true, true);
    
    let input = zeros(&[2, 3, 4, 4]);
    let output = bn.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3, 4, 4]);
}

#[test]
fn test_layernorm_basic() {
    let ln = LayerNorm::new(vec![4, 4], None, true);
    
    let input = zeros(&[2, 3, 4, 4]);
    let output = ln.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3, 4, 4]);
}

#[test]
fn test_dropout_basic() {
    let dropout = Dropout::new(0.5, false);
    
    let input = ones(&[2, 3, 4, 4]);
    let output = dropout.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3, 4, 4]);
}

#[test]
fn test_gelu_activation() {
    let gelu = GELU::new(None);
    
    let input = ones(&[2, 3]);
    let output = gelu.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3]);
}

#[test]
fn test_leaky_relu_activation() {
    let leaky_relu = LeakyReLU::new(0.01, false);
    
    let input = ones(&[2, 3]);
    let output = leaky_relu.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3]);
}

#[test]
fn test_softmax_activation() {
    let softmax = Softmax::new(Some(-1));
    
    let input = ones(&[2, 3]);
    let output = softmax.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3]);
}

#[test]
fn test_logsoftmax_activation() {
    let logsoftmax = LogSoftmax::new(Some(-1));
    
    let input = ones(&[2, 3]);
    let output = logsoftmax.forward(&input).unwrap();
    
    assert_eq!(output.shape().dims(), &[2, 3]);
}

#[test]
fn test_layer_parameters() {
    let linear = Linear::new(10, 5, true);
    let params = linear.parameters();
    
    // Should have weight and bias
    assert_eq!(params.len(), 2);
    
    let named_params = linear.named_parameters();
    assert!(named_params.contains_key("weight"));
    assert!(named_params.contains_key("bias"));
}

#[test]
fn test_training_mode() {
    let mut dropout = Dropout::new(0.5, false);
    
    assert!(dropout.training());
    
    dropout.train(false);
    assert!(!dropout.training());
    
    dropout.train(true);
    assert!(dropout.training());
}