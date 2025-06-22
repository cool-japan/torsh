use torsh_nn::modules::{GRU, LSTM, RNN};
use torsh_nn::Module;
use torsh_tensor::creation::randn;

#[test]
fn test_rnn_basic() {
    let rnn = RNN::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    // Test basic properties
    assert_eq!(rnn.input_size(), 10);
    assert_eq!(rnn.hidden_size(), 20);
    assert_eq!(rnn.num_layers(), 1);
    assert_eq!(rnn.bias(), true);
    assert_eq!(rnn.bidirectional(), false);

    // Test parameter count
    let params = rnn.parameters();
    // Should have weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    assert_eq!(params.len(), 4);

    // Test forward pass with placeholder implementation
    let input = randn(&[2, 5, 10]); // [batch, seq_len, input_size]
    let output = rnn.forward(&input);
    assert!(output.is_ok());
}

#[test]
fn test_rnn_bidirectional() {
    let rnn = RNN::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        true,    // bidirectional
    );

    assert_eq!(rnn.bidirectional(), true);

    // Test parameter count with bidirectional
    let params = rnn.parameters();
    // Should have forward and reverse weights and biases
    assert_eq!(params.len(), 8); // 4 forward + 4 reverse
}

#[test]
fn test_rnn_multi_layer() {
    let rnn = RNN::new(
        10,      // input_size
        20,      // hidden_size
        Some(3), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    assert_eq!(rnn.num_layers(), 3);

    // Test parameter count with multiple layers
    let params = rnn.parameters();
    // Should have 4 parameters per layer * 3 layers
    assert_eq!(params.len(), 12);
}

#[test]
fn test_lstm_basic() {
    let lstm = LSTM::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    // Test basic properties
    assert_eq!(lstm.input_size(), 10);
    assert_eq!(lstm.hidden_size(), 20);
    assert_eq!(lstm.num_layers(), 1);
    assert_eq!(lstm.bias(), true);
    assert_eq!(lstm.bidirectional(), false);

    // Test parameter count
    let params = lstm.parameters();
    // Should have weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    assert_eq!(params.len(), 4);

    // Test forward pass with placeholder implementation
    let input = randn(&[2, 5, 10]); // [batch, seq_len, input_size]
    let output = lstm.forward(&input);
    assert!(output.is_ok());
}

#[test]
fn test_lstm_parameter_shapes() {
    let lstm = LSTM::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    let named_params = lstm.named_parameters();

    // Check weight shapes for LSTM (4 gates)
    let weight_ih = named_params.get("weight_ih_l0").unwrap();
    let weight_ih_shape = weight_ih.read().shape();
    assert_eq!(weight_ih_shape.dims(), &[4 * 20, 10]); // [4*hidden_size, input_size]

    let weight_hh = named_params.get("weight_hh_l0").unwrap();
    let weight_hh_shape = weight_hh.read().shape();
    assert_eq!(weight_hh_shape.dims(), &[4 * 20, 20]); // [4*hidden_size, hidden_size]

    // Check bias shapes
    let bias_ih = named_params.get("bias_ih_l0").unwrap();
    let bias_ih_shape = bias_ih.read().shape();
    assert_eq!(bias_ih_shape.dims(), &[4 * 20]); // [4*hidden_size]

    let bias_hh = named_params.get("bias_hh_l0").unwrap();
    let bias_hh_shape = bias_hh.read().shape();
    assert_eq!(bias_hh_shape.dims(), &[4 * 20]); // [4*hidden_size]
}

#[test]
fn test_gru_basic() {
    let gru = GRU::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    // Test basic properties
    assert_eq!(gru.input_size(), 10);
    assert_eq!(gru.hidden_size(), 20);
    assert_eq!(gru.num_layers(), 1);
    assert_eq!(gru.bias(), true);
    assert_eq!(gru.bidirectional(), false);

    // Test parameter count
    let params = gru.parameters();
    // Should have weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    assert_eq!(params.len(), 4);

    // Test forward pass with placeholder implementation
    let input = randn(&[2, 5, 10]); // [batch, seq_len, input_size]
    let output = gru.forward(&input);
    assert!(output.is_ok());
}

#[test]
fn test_gru_parameter_shapes() {
    let gru = GRU::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    let named_params = gru.named_parameters();

    // Check weight shapes for GRU (3 gates)
    let weight_ih = named_params.get("weight_ih_l0").unwrap();
    let weight_ih_shape = weight_ih.read().shape();
    assert_eq!(weight_ih_shape.dims(), &[3 * 20, 10]); // [3*hidden_size, input_size]

    let weight_hh = named_params.get("weight_hh_l0").unwrap();
    let weight_hh_shape = weight_hh.read().shape();
    assert_eq!(weight_hh_shape.dims(), &[3 * 20, 20]); // [3*hidden_size, hidden_size]

    // Check bias shapes
    let bias_ih = named_params.get("bias_ih_l0").unwrap();
    let bias_ih_shape = bias_ih.read().shape();
    assert_eq!(bias_ih_shape.dims(), &[3 * 20]); // [3*hidden_size]

    let bias_hh = named_params.get("bias_hh_l0").unwrap();
    let bias_hh_shape = bias_hh.read().shape();
    assert_eq!(bias_hh_shape.dims(), &[3 * 20]); // [3*hidden_size]
}

#[test]
fn test_rnn_training_mode() {
    let mut rnn = RNN::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        true,    // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    // Should be in training mode by default
    assert!(rnn.training());

    // Switch to evaluation mode
    rnn.train(false);
    assert!(!rnn.training());

    // Switch back to training mode
    rnn.train(true);
    assert!(rnn.training());
}

#[test]
fn test_lstm_no_bias() {
    let lstm = LSTM::new(
        10,      // input_size
        20,      // hidden_size
        Some(1), // num_layers
        false,   // bias
        true,    // batch_first
        None,    // dropout
        false,   // bidirectional
    );

    assert_eq!(lstm.bias(), false);

    // Test parameter count without bias
    let params = lstm.parameters();
    // Should only have weight_ih_l0, weight_hh_l0 (no bias parameters)
    assert_eq!(params.len(), 2);

    let named_params = lstm.named_parameters();
    assert!(named_params.get("bias_ih_l0").is_none());
    assert!(named_params.get("bias_hh_l0").is_none());
}

#[test]
fn test_gru_dropout() {
    let gru = GRU::new(
        10,        // input_size
        20,        // hidden_size
        Some(2),   // num_layers
        true,      // bias
        true,      // batch_first
        Some(0.5), // dropout
        false,     // bidirectional
    );

    assert_eq!(gru.dropout(), 0.5);
    assert_eq!(gru.num_layers(), 2);
}
