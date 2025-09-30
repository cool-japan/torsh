use torsh_nn::layers::{GRU, LSTM, RNN};
use torsh_nn::Module;
use torsh_tensor::creation::randn;

#[test]
fn test_rnn_basic() -> Result<(), Box<dyn std::error::Error>> {
    let rnn = RNN::new(
        10, // input_size
        20, // hidden_size
        1,  // num_layers
    )?;

    // Test parameter count
    let params = rnn.parameters();
    // Should have weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    assert_eq!(params.len(), 4);

    // Test forward pass with placeholder implementation
    let input = randn(&[2, 5, 10])?; // [batch, seq_len, input_size]
    let _output = rnn.forward(&input)?;

    Ok(())
}

#[test]
fn test_rnn_multi_layer() -> Result<(), Box<dyn std::error::Error>> {
    let rnn = RNN::new(
        10, // input_size
        20, // hidden_size
        3,  // num_layers
    )?;

    // Test parameter count with multiple layers
    let params = rnn.parameters();
    // Should have 4 parameters per layer * 3 layers
    assert_eq!(params.len(), 12);
    Ok(())
}

#[test]
fn test_lstm_basic() -> Result<(), Box<dyn std::error::Error>> {
    let lstm = LSTM::new(
        10, // input_size
        20, // hidden_size
        1,  // num_layers
    )?;

    // Test parameter count
    let params = lstm.parameters();
    // Should have weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    assert_eq!(params.len(), 4);

    // Test forward pass with placeholder implementation
    let input = randn(&[2, 5, 10])?; // [batch, seq_len, input_size]
    let _output = lstm.forward(&input)?;

    Ok(())
}

#[test]
fn test_lstm_parameter_shapes() -> Result<(), Box<dyn std::error::Error>> {
    let lstm = LSTM::new(
        10, // input_size
        20, // hidden_size
        1,  // num_layers
    )?;

    let named_params = lstm.named_parameters();

    // Check weight shapes for LSTM (4 gates)
    let weight_ih = named_params.get("weight_ih_l0").unwrap();
    let weight_ih_shape = weight_ih.tensor().read().shape();
    assert_eq!(weight_ih_shape.dims(), &[4 * 20, 10]); // [4*hidden_size, input_size]

    let weight_hh = named_params.get("weight_hh_l0").unwrap();
    let weight_hh_shape = weight_hh.tensor().read().shape();
    assert_eq!(weight_hh_shape.dims(), &[4 * 20, 20]); // [4*hidden_size, hidden_size]

    // Check bias shapes
    let bias_ih = named_params.get("bias_ih_l0").unwrap();
    let bias_ih_shape = bias_ih.tensor().read().shape();
    assert_eq!(bias_ih_shape.dims(), &[4 * 20]); // [4*hidden_size]

    let bias_hh = named_params.get("bias_hh_l0").unwrap();
    let bias_hh_shape = bias_hh.tensor().read().shape();
    assert_eq!(bias_hh_shape.dims(), &[4 * 20]); // [4*hidden_size]
    Ok(())
}

#[test]
fn test_gru_basic() -> Result<(), Box<dyn std::error::Error>> {
    let gru = GRU::new(
        10, // input_size
        20, // hidden_size
        1,  // num_layers
    )?;

    // Test parameter count
    let params = gru.parameters();
    // Should have weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    assert_eq!(params.len(), 4);

    // Test forward pass with placeholder implementation
    let input = randn(&[2, 5, 10])?; // [batch, seq_len, input_size]
    let _output = gru.forward(&input)?;

    Ok(())
}

#[test]
fn test_gru_parameter_shapes() -> Result<(), Box<dyn std::error::Error>> {
    let gru = GRU::new(
        10, // input_size
        20, // hidden_size
        1,  // num_layers
    )?;

    let named_params = gru.named_parameters();

    // Check weight shapes for GRU (3 gates)
    let weight_ih = named_params.get("weight_ih_l0").unwrap();
    let weight_ih_shape = weight_ih.tensor().read().shape();
    assert_eq!(weight_ih_shape.dims(), &[3 * 20, 10]); // [3*hidden_size, input_size]

    let weight_hh = named_params.get("weight_hh_l0").unwrap();
    let weight_hh_shape = weight_hh.tensor().read().shape();
    assert_eq!(weight_hh_shape.dims(), &[3 * 20, 20]); // [3*hidden_size, hidden_size]

    // Check bias shapes
    let bias_ih = named_params.get("bias_ih_l0").unwrap();
    let bias_ih_shape = bias_ih.tensor().read().shape();
    assert_eq!(bias_ih_shape.dims(), &[3 * 20]); // [3*hidden_size]

    let bias_hh = named_params.get("bias_hh_l0").unwrap();
    let bias_hh_shape = bias_hh.tensor().read().shape();
    assert_eq!(bias_hh_shape.dims(), &[3 * 20]); // [3*hidden_size]
    Ok(())
}

#[test]
fn test_rnn_training_mode() {
    let mut rnn = RNN::new(
        10, // input_size
        20, // hidden_size
        1,  // num_layers
    )
    .unwrap();

    // Should be in training mode by default
    assert!(rnn.training());

    // Switch to evaluation mode
    rnn.eval();
    assert!(!rnn.training());

    // Switch back to training mode
    rnn.train();
    assert!(rnn.training());
}
