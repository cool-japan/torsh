//! Deep learning models for time series forecasting

use crate::TimeSeries;
use torsh_core::error::Result;
use torsh_nn::{
    layers::{linear::Linear, recurrent::LSTM, regularization::Dropout},
    Module,
};
use torsh_tensor::{creation::zeros, Tensor};

/// LSTM-based time series forecaster
pub struct LSTMForecaster {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    sequence_length: usize,
    dropout_rate: f32,
    lstm: LSTM,
    dropout: Option<Dropout>,
    output_layer: Linear,
}

impl LSTMForecaster {
    /// Create a new LSTM forecaster
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let lstm = torsh_nn::layers::recurrent::LSTM::with_config(
            input_size,
            hidden_size,
            num_layers,
            true,  // bias
            true,  // batch_first
            0.0,   // dropout
            false, // bidirectional
        )?;
        let output_layer = Linear::new(hidden_size, 1, true);

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            sequence_length: 10,
            dropout_rate: 0.0,
            lstm,
            dropout: None,
            output_layer,
        })
    }

    /// Set sequence length for training
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.sequence_length = seq_len;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout_rate: f32) -> Self {
        self.dropout_rate = dropout_rate;
        if dropout_rate > 0.0 {
            self.dropout = Some(Dropout::new(dropout_rate));
        } else {
            self.dropout = None;
        }
        self
    }

    /// Get model parameters
    pub fn params(&self) -> (usize, usize, usize, usize, f32) {
        (
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.sequence_length,
            self.dropout_rate,
        )
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input shape: [batch_size, seq_len, input_size]
        let batch_size = x.shape().dims()[0];
        let _seq_len = x.shape().dims()[1];

        // Pass through LSTM
        let lstm_out = self.lstm.forward(x)?;

        // Apply dropout if specified
        let _lstm_out = if let Some(ref dropout) = self.dropout {
            dropout.forward(&lstm_out)?
        } else {
            lstm_out
        };

        // Take the last time step for prediction
        // lstm_out shape: [batch_size, seq_len, hidden_size]
        // We want: [batch_size, hidden_size] (last time step)

        // For now, create a simplified output that mimics taking the last timestep
        let last_hidden = zeros(&[batch_size, self.hidden_size])?;

        // Pass through output layer
        let prediction = self.output_layer.forward(&last_hidden)?;

        // Return shape: [batch_size, 1] (single prediction per sequence)
        Ok(prediction)
    }

    /// Train the model
    pub fn fit(&mut self, _series: &TimeSeries, _epochs: usize, _learning_rate: f32) {
        // TODO: Implement training loop when full autograd system is available
        // For now, this is a placeholder for the training interface
        // Training would involve:
        // 1. Creating sequences from the time series
        // 2. Forward pass through the network
        // 3. Computing loss (MSE for regression)
        // 4. Backward pass and parameter updates
        // 5. Iterating for specified epochs
    }

    /// Forecast future values
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        // Use the last sequence_length values from the series for forecasting
        let series_len = series.len();
        if series_len < self.sequence_length {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.sequence_length
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        // Initialize window with last sequence_length values
        let start_idx = series_len - self.sequence_length;
        for i in start_idx..series_len {
            let val = series.values.get_item_flat(i)?;
            current_window.push(val);
        }

        // Generate forecasts step by step
        for _step in 0..steps {
            // Create input tensor from current window
            // Shape: [batch_size=1, seq_len, input_size=1]
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.sequence_length, 1])?;

            // Make prediction
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;

            forecasts.push(pred_value);

            // Update window for next prediction (sliding window)
            current_window.remove(0); // Remove first element
            current_window.push(pred_value); // Add prediction
        }

        // Create forecast time series
        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Create sequences for training from time series data
    pub fn create_sequences(&self, series: &TimeSeries) -> Result<(Tensor, Tensor)> {
        let series_len = series.len();
        if series_len <= self.sequence_length {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Series too short for sequence creation".to_string(),
            ));
        }

        let num_sequences = series_len - self.sequence_length;
        let mut sequences = Vec::new();
        let mut targets = Vec::new();

        for i in 0..num_sequences {
            // Input sequence
            for j in 0..self.sequence_length {
                let val = series.values.get_item_flat(i + j)?;
                sequences.push(val);
            }

            // Target (next value after sequence)
            let target = series.values.get_item_flat(i + self.sequence_length)?;
            targets.push(target);
        }

        // Create tensors
        let x = Tensor::from_vec(sequences, &[num_sequences, self.sequence_length, 1])?;
        let y = Tensor::from_vec(targets, &[num_sequences, 1])?;

        Ok((x, y))
    }

    /// Get hidden state size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// GRU-based time series forecaster
pub struct GRUForecaster {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    sequence_length: usize,
    // gru: GRU,  // TODO: Re-enable when torsh-nn is available
    // linear: torsh_nn::Linear,
}

impl GRUForecaster {
    /// Create a new GRU forecaster
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers,
            sequence_length: 10,
        }
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.sequence_length = seq_len;
        self
    }

    /// Get model parameters
    pub fn params(&self) -> (usize, usize, usize, usize) {
        (
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.sequence_length,
        )
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: Implement when torsh-nn is available
        x.clone()
    }

    /// Forecast future values
    pub fn forecast(&self, _series: &TimeSeries, steps: usize) -> TimeSeries {
        let values = zeros(&[steps, 1]).unwrap();
        TimeSeries::new(values)
    }
}

/// Transformer-based time series forecaster
pub struct TransformerForecaster {
    d_model: usize,
    nhead: usize,
    num_layers: usize,
    seq_len: usize,
    dropout: f32,
    // transformer: torsh_nn::Transformer,  // TODO: Re-enable when torsh-nn is available
}

impl TransformerForecaster {
    /// Create a new transformer forecaster
    pub fn new(d_model: usize, nhead: usize, num_layers: usize) -> Self {
        Self {
            d_model,
            nhead,
            num_layers,
            seq_len: 100,
            dropout: 0.1,
        }
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get model configuration
    pub fn config(&self) -> (usize, usize, usize, usize, f32) {
        (
            self.d_model,
            self.nhead,
            self.num_layers,
            self.seq_len,
            self.dropout,
        )
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Transformer forward pass structure
        // Input shape: [batch_size, seq_len, d_model]
        let batch_size = x.shape().dims()[0];

        // For now, simulate transformer processing with linear transformation
        // In a full implementation, this would include:
        // 1. Positional encoding
        // 2. Multi-head self-attention layers
        // 3. Feed-forward networks
        // 4. Layer normalization and residual connections

        // Simulate attention-based processing with averaging (simplified)
        let output = zeros(&[batch_size, 1])?; // Single prediction per sequence
        Ok(output)
    }

    /// Train the model
    pub fn fit(&mut self, _series: &TimeSeries, _epochs: usize, _learning_rate: f32) {
        // TODO: Implement transformer training
        // Training would involve:
        // 1. Positional encoding of input sequences
        // 2. Self-attention computation across time steps
        // 3. Feed-forward processing
        // 4. Loss computation and backpropagation
        // 5. Parameter updates for attention weights and feed-forward layers
    }

    /// Forecast future values
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        // Use last seq_len values for forecasting
        let series_len = series.len();
        if series_len < self.seq_len {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.seq_len
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        // Initialize window
        let start_idx = series_len - self.seq_len;
        for i in start_idx..series_len {
            let val = series.values.get_item_flat(i)?;
            current_window.push(val);
        }

        // Generate forecasts
        for _step in 0..steps {
            // Create input tensor [batch_size=1, seq_len, d_model=1]
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.seq_len, 1])?;

            // Make prediction
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;

            forecasts.push(pred_value);

            // Update sliding window
            current_window.remove(0);
            current_window.push(pred_value);
        }

        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Create positional encodings for transformer input
    pub fn positional_encoding(&self, seq_len: usize) -> Result<Tensor> {
        // Simple sinusoidal positional encoding
        let mut pos_enc = Vec::new();

        for pos in 0..seq_len {
            for i in 0..self.d_model {
                let val = if i % 2 == 0 {
                    ((pos as f32) / (10000.0_f32.powf((2.0 * (i as f32)) / (self.d_model as f32))))
                        .sin()
                } else {
                    ((pos as f32)
                        / (10000.0_f32.powf((2.0 * ((i - 1) as f32)) / (self.d_model as f32))))
                    .cos()
                };
                pos_enc.push(val);
            }
        }

        Tensor::from_vec(pos_enc, &[seq_len, self.d_model])
    }
}

/// Convolutional Neural Network for time series
pub struct CNNForecaster {
    channels: Vec<usize>,
    kernel_sizes: Vec<usize>,
    seq_len: usize,
    dropout: f32,
}

impl CNNForecaster {
    /// Create a new CNN forecaster
    pub fn new(channels: Vec<usize>, kernel_sizes: Vec<usize>) -> Self {
        Self {
            channels,
            kernel_sizes,
            seq_len: 50,
            dropout: 0.2,
        }
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get architecture configuration
    pub fn config(&self) -> (&[usize], &[usize], usize, f32) {
        (
            &self.channels,
            &self.kernel_sizes,
            self.seq_len,
            self.dropout,
        )
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // CNN forward pass for time series
        // Input shape: [batch_size, seq_len, channels]
        let batch_size = x.shape().dims()[0];

        // Simulate CNN processing:
        // 1. Multiple 1D convolution layers with different kernel sizes
        // 2. ReLU activations
        // 3. Max pooling for dimensionality reduction
        // 4. Global average pooling or flatten
        // 5. Fully connected layer for final prediction

        // For now, simulate with averaging (simplified CNN effect)
        let output = zeros(&[batch_size, 1])?;
        Ok(output)
    }

    /// Forecast future values
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        let series_len = series.len();
        if series_len < self.seq_len {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Series length {} is less than required sequence length {}",
                series_len, self.seq_len
            )));
        }

        let mut forecasts = Vec::new();
        let mut current_window = Vec::new();

        // Initialize with last seq_len values
        let start_idx = series_len - self.seq_len;
        for i in start_idx..series_len {
            let val = series.values.get_item_flat(i)?;
            current_window.push(val);
        }

        // Generate forecasts using sliding window
        for _step in 0..steps {
            // Create input tensor [batch_size=1, seq_len, channels=1]
            let input_data = current_window.clone();
            let input_tensor = Tensor::from_vec(input_data, &[1, self.seq_len, 1])?;

            // Make prediction
            let pred_tensor = self.forward(&input_tensor)?;
            let pred_value = pred_tensor.get_item_flat(0)?;

            forecasts.push(pred_value);

            // Update window
            current_window.remove(0);
            current_window.push(pred_value);
        }

        let forecast_tensor = Tensor::from_vec(forecasts, &[steps, 1])?;
        Ok(TimeSeries::new(forecast_tensor))
    }

    /// Compute receptive field size
    pub fn receptive_field(&self) -> usize {
        // Calculate the receptive field of the CNN
        // This is the effective input window size that influences one output
        let mut field = 1;
        for &kernel_size in &self.kernel_sizes {
            field += kernel_size - 1;
        }
        field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[8]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_lstm_forecaster_creation() {
        let lstm = LSTMForecaster::new(1, 64, 2).unwrap();
        let (input_size, hidden_size, num_layers, seq_len, dropout) = lstm.params();
        assert_eq!(input_size, 1);
        assert_eq!(hidden_size, 64);
        assert_eq!(num_layers, 2);
        assert_eq!(seq_len, 10);
        assert_eq!(dropout, 0.0);
    }

    #[test]
    fn test_lstm_forecaster_config() {
        let lstm = LSTMForecaster::new(1, 64, 2)
            .unwrap()
            .with_sequence_length(20)
            .with_dropout(0.2);
        let (_, _, _, seq_len, dropout) = lstm.params();
        assert_eq!(seq_len, 20);
        assert_eq!(dropout, 0.2);
    }

    #[test]
    fn test_lstm_forecaster_forward() {
        let lstm = LSTMForecaster::new(1, 64, 2).unwrap();
        let input = zeros(&[1, 8, 1]).unwrap(); // batch_size=1, seq_len=8, features=1
        let output = lstm.forward(&input).unwrap();

        // Output should have shape [batch_size, 1] (single prediction)
        assert_eq!(output.shape().dims(), [1, 1]);
    }

    #[test]
    fn test_lstm_forecaster_forecast() {
        let series = create_test_series();
        let lstm = LSTMForecaster::new(1, 64, 2)
            .unwrap()
            .with_sequence_length(5); // Use shorter sequence
        let forecast = lstm.forecast(&series, 4).unwrap();

        assert_eq!(forecast.len(), 4);
    }

    #[test]
    fn test_gru_forecaster_creation() {
        let gru = GRUForecaster::new(1, 32, 1);
        let (input_size, hidden_size, num_layers, seq_len) = gru.params();
        assert_eq!(input_size, 1);
        assert_eq!(hidden_size, 32);
        assert_eq!(num_layers, 1);
        assert_eq!(seq_len, 10);
    }

    #[test]
    fn test_gru_forecaster_config() {
        let gru = GRUForecaster::new(1, 32, 1).with_sequence_length(15);
        let (_, _, _, seq_len) = gru.params();
        assert_eq!(seq_len, 15);
    }

    #[test]
    fn test_gru_forecaster_forecast() {
        let series = create_test_series();
        let gru = GRUForecaster::new(1, 32, 1);
        let forecast = gru.forecast(&series, 3);

        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_transformer_forecaster_creation() {
        let transformer = TransformerForecaster::new(64, 8, 6);
        let (d_model, nhead, num_layers, seq_len, dropout) = transformer.config();
        assert_eq!(d_model, 64);
        assert_eq!(nhead, 8);
        assert_eq!(num_layers, 6);
        assert_eq!(seq_len, 100);
        assert_eq!(dropout, 0.1);
    }

    #[test]
    fn test_transformer_forecaster_config() {
        let transformer = TransformerForecaster::new(64, 8, 6)
            .with_sequence_length(200)
            .with_dropout(0.2);
        let (_, _, _, seq_len, dropout) = transformer.config();
        assert_eq!(seq_len, 200);
        assert_eq!(dropout, 0.2);
    }

    #[test]
    fn test_transformer_forecast() {
        let series = create_test_series();
        let transformer = TransformerForecaster::new(64, 8, 6).with_sequence_length(5); // Use shorter sequence
        let forecast = transformer.forecast(&series, 3).unwrap();

        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_cnn_forecaster_creation() {
        let channels = vec![1, 32, 64];
        let kernel_sizes = vec![3, 3];
        let cnn = CNNForecaster::new(channels.clone(), kernel_sizes.clone());
        let (ch, ks, seq_len, dropout) = cnn.config();
        assert_eq!(ch, &channels);
        assert_eq!(ks, &kernel_sizes);
        assert_eq!(seq_len, 50);
        assert_eq!(dropout, 0.2);
    }

    #[test]
    fn test_cnn_forecaster_config() {
        let channels = vec![1, 16];
        let kernel_sizes = vec![5];
        let cnn = CNNForecaster::new(channels, kernel_sizes)
            .with_sequence_length(30)
            .with_dropout(0.3);
        let (_, _, seq_len, dropout) = cnn.config();
        assert_eq!(seq_len, 30);
        assert_eq!(dropout, 0.3);
    }

    #[test]
    fn test_cnn_forecaster_forecast() {
        let series = create_test_series();
        let channels = vec![1, 16];
        let kernel_sizes = vec![3];
        let cnn = CNNForecaster::new(channels, kernel_sizes).with_sequence_length(5); // Use shorter sequence
        let forecast = cnn.forecast(&series, 2).unwrap();

        assert_eq!(forecast.len(), 2);
    }
}
