//! Recurrent Neural Network operations for all backends
//!
//! This module provides optimized implementations of RNN, LSTM, and GRU cells
//! with support for different backends and optimization strategies.

use crate::{BackendResult, Buffer, Device};
use torsh_core::dtype::DType;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// RNN cell type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnCellType {
    /// Vanilla RNN cell
    Rnn,
    /// Long Short-Term Memory cell
    Lstm,
    /// Gated Recurrent Unit cell
    Gru,
    /// Bidirectional LSTM
    BiLstm,
    /// Bidirectional GRU
    BiGru,
}

/// RNN activation function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnActivation {
    /// Hyperbolic tangent
    Tanh,
    /// Rectified Linear Unit
    Relu,
    /// Sigmoid
    Sigmoid,
    /// Identity (no activation)
    Identity,
}

/// RNN direction for bidirectional networks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnDirection {
    /// Forward direction only
    Forward,
    /// Backward direction only
    Backward,
    /// Bidirectional (both forward and backward)
    Bidirectional,
}

/// RNN configuration
#[derive(Debug, Clone)]
pub struct RnnConfig {
    /// Cell type
    pub cell_type: RnnCellType,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub sequence_length: usize,
    /// Bias usage
    pub bias: bool,
    /// Dropout probability (between layers)
    pub dropout: f32,
    /// Direction
    pub direction: RnnDirection,
    /// Activation function for RNN cells
    pub activation: RnnActivation,
    /// Data type
    pub dtype: DType,
    /// Whether to return sequences or just the last output
    pub return_sequences: bool,
    /// Whether to return state
    pub return_state: bool,
}

impl RnnConfig {
    /// Create a new LSTM configuration
    pub fn lstm(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            cell_type: RnnCellType::Lstm,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            sequence_length,
            bias: true,
            dropout: 0.0,
            direction: RnnDirection::Forward,
            activation: RnnActivation::Tanh,
            dtype: DType::F32,
            return_sequences: true,
            return_state: false,
        }
    }

    /// Create a new GRU configuration
    pub fn gru(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            cell_type: RnnCellType::Gru,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            sequence_length,
            bias: true,
            dropout: 0.0,
            direction: RnnDirection::Forward,
            activation: RnnActivation::Tanh,
            dtype: DType::F32,
            return_sequences: true,
            return_state: false,
        }
    }

    /// Create a new vanilla RNN configuration
    pub fn rnn(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
        sequence_length: usize,
        activation: RnnActivation,
    ) -> Self {
        Self {
            cell_type: RnnCellType::Rnn,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
            sequence_length,
            bias: true,
            dropout: 0.0,
            direction: RnnDirection::Forward,
            activation,
            dtype: DType::F32,
            return_sequences: true,
            return_state: false,
        }
    }

    /// Set to bidirectional
    pub fn bidirectional(mut self) -> Self {
        self.direction = RnnDirection::Bidirectional;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set data type
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set whether to return sequences
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Set whether to return state
    pub fn with_return_state(mut self, return_state: bool) -> Self {
        self.return_state = return_state;
        self
    }

    /// Get the effective hidden size (considering bidirectional)
    pub fn effective_hidden_size(&self) -> usize {
        match self.direction {
            RnnDirection::Bidirectional => self.hidden_size * 2,
            _ => self.hidden_size,
        }
    }

    /// Get input buffer size in bytes
    pub fn input_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };
        self.batch_size * self.sequence_length * self.input_size * element_size
    }

    /// Get output buffer size in bytes
    pub fn output_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };

        let effective_hidden = self.effective_hidden_size();

        if self.return_sequences {
            self.batch_size * self.sequence_length * effective_hidden * element_size
        } else {
            self.batch_size * effective_hidden * element_size
        }
    }

    /// Get hidden state buffer size in bytes
    pub fn hidden_state_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };

        let num_directions = match self.direction {
            RnnDirection::Bidirectional => 2,
            _ => 1,
        };

        self.batch_size * self.num_layers * num_directions * self.hidden_size * element_size
    }

    /// Get cell state buffer size in bytes (for LSTM)
    pub fn cell_state_buffer_size(&self) -> usize {
        match self.cell_type {
            RnnCellType::Lstm | RnnCellType::BiLstm => self.hidden_state_buffer_size(),
            _ => 0,
        }
    }

    /// Get weight buffer size in bytes
    pub fn weight_buffer_size(&self) -> usize {
        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };

        let num_directions = match self.direction {
            RnnDirection::Bidirectional => 2,
            _ => 1,
        };

        let mut total_weights = 0;

        for layer in 0..self.num_layers {
            let layer_input_size = if layer == 0 {
                self.input_size
            } else {
                self.effective_hidden_size()
            };

            let weights_per_direction = match self.cell_type {
                RnnCellType::Rnn => {
                    // input-to-hidden + hidden-to-hidden
                    layer_input_size * self.hidden_size + self.hidden_size * self.hidden_size
                }
                RnnCellType::Lstm | RnnCellType::BiLstm => {
                    // 4 gates: input, forget, cell, output
                    // input-to-hidden (4 gates) + hidden-to-hidden (4 gates)
                    4 * (layer_input_size * self.hidden_size + self.hidden_size * self.hidden_size)
                }
                RnnCellType::Gru | RnnCellType::BiGru => {
                    // 3 gates: reset, update, new
                    // input-to-hidden (3 gates) + hidden-to-hidden (3 gates)
                    3 * (layer_input_size * self.hidden_size + self.hidden_size * self.hidden_size)
                }
            };

            total_weights += weights_per_direction * num_directions;
        }

        total_weights * element_size
    }

    /// Get bias buffer size in bytes
    pub fn bias_buffer_size(&self) -> usize {
        if !self.bias {
            return 0;
        }

        let element_size = match self.dtype {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            _ => 4,
        };

        let num_directions = match self.direction {
            RnnDirection::Bidirectional => 2,
            _ => 1,
        };

        let biases_per_layer_direction = match self.cell_type {
            RnnCellType::Rnn => self.hidden_size,
            RnnCellType::Lstm | RnnCellType::BiLstm => 4 * self.hidden_size, // 4 gates
            RnnCellType::Gru | RnnCellType::BiGru => 3 * self.hidden_size,   // 3 gates
        };

        self.num_layers * num_directions * biases_per_layer_direction * element_size
    }

    /// Check if the configuration is valid
    pub fn is_valid(&self) -> bool {
        self.input_size > 0
            && self.hidden_size > 0
            && self.num_layers > 0
            && self.batch_size > 0
            && self.sequence_length > 0
            && self.dropout >= 0.0
            && self.dropout <= 1.0
    }
}

/// RNN output containing sequences and optional states
#[derive(Debug, Clone)]
pub struct RnnOutput {
    /// Output sequences (batch_size, sequence_length, hidden_size)
    pub sequences: Option<Buffer>,
    /// Final hidden state (num_layers * num_directions, batch_size, hidden_size)
    pub hidden_state: Option<Buffer>,
    /// Final cell state for LSTM (num_layers * num_directions, batch_size, hidden_size)
    pub cell_state: Option<Buffer>,
}

/// RNN operations trait
#[async_trait::async_trait]
pub trait RnnOps: Send + Sync {
    /// Execute RNN forward pass
    async fn rnn_forward(
        &self,
        device: &Device,
        input: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        initial_hidden: Option<&Buffer>,
        initial_cell: Option<&Buffer>,
        config: &RnnConfig,
    ) -> BackendResult<RnnOutput>;

    /// Execute LSTM forward pass
    async fn lstm_forward(
        &self,
        device: &Device,
        input: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        initial_hidden: Option<&Buffer>,
        initial_cell: Option<&Buffer>,
        config: &RnnConfig,
    ) -> BackendResult<RnnOutput>;

    /// Execute GRU forward pass
    async fn gru_forward(
        &self,
        device: &Device,
        input: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        initial_hidden: Option<&Buffer>,
        config: &RnnConfig,
    ) -> BackendResult<RnnOutput>;

    /// Execute LSTM cell (single timestep)
    async fn lstm_cell(
        &self,
        device: &Device,
        input: &Buffer,
        hidden: &Buffer,
        cell: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        output_hidden: &Buffer,
        output_cell: &Buffer,
    ) -> BackendResult<()>;

    /// Execute GRU cell (single timestep)
    async fn gru_cell(
        &self,
        device: &Device,
        input: &Buffer,
        hidden: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        output_hidden: &Buffer,
    ) -> BackendResult<()>;

    /// Check if RNN operations are supported
    fn supports_rnn(&self) -> bool;

    /// Get supported RNN cell types
    fn supported_cell_types(&self) -> Vec<RnnCellType>;

    /// Get supported activations
    fn supported_activations(&self) -> Vec<RnnActivation>;
}

/// RNN performance hints for optimization
#[derive(Debug, Clone)]
pub struct RnnPerformanceHints {
    /// Preferred batch size for optimal performance
    pub optimal_batch_size: usize,
    /// Preferred sequence length for optimal performance
    pub optimal_sequence_length: usize,
    /// Whether to use fused operations
    pub use_fused_ops: bool,
    /// Whether to use cuDNN/optimized libraries when available
    pub use_optimized_libs: bool,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f32,
    /// Compute throughput in GOPS
    pub compute_throughput: f32,
}

impl Default for RnnPerformanceHints {
    fn default() -> Self {
        Self {
            optimal_batch_size: 32,
            optimal_sequence_length: 50,
            use_fused_ops: true,
            use_optimized_libs: true,
            memory_bandwidth: 100.0,
            compute_throughput: 500.0,
        }
    }
}

/// Default RNN operations implementation
pub struct DefaultRnnOps {
    performance_hints: RnnPerformanceHints,
}

impl DefaultRnnOps {
    pub fn new() -> Self {
        Self {
            performance_hints: RnnPerformanceHints::default(),
        }
    }

    pub fn with_performance_hints(mut self, hints: RnnPerformanceHints) -> Self {
        self.performance_hints = hints;
        self
    }
}

#[async_trait::async_trait]
impl RnnOps for DefaultRnnOps {
    async fn rnn_forward(
        &self,
        _device: &Device,
        _input: &Buffer,
        _weights: &Buffer,
        _bias: Option<&Buffer>,
        _initial_hidden: Option<&Buffer>,
        _initial_cell: Option<&Buffer>,
        _config: &RnnConfig,
    ) -> BackendResult<RnnOutput> {
        Err(torsh_core::error::TorshError::BackendError(
            "RNN operations not implemented for this backend".to_string(),
        ))
    }

    async fn lstm_forward(
        &self,
        _device: &Device,
        _input: &Buffer,
        _weights: &Buffer,
        _bias: Option<&Buffer>,
        _initial_hidden: Option<&Buffer>,
        _initial_cell: Option<&Buffer>,
        _config: &RnnConfig,
    ) -> BackendResult<RnnOutput> {
        Err(torsh_core::error::TorshError::BackendError(
            "LSTM operations not implemented for this backend".to_string(),
        ))
    }

    async fn gru_forward(
        &self,
        _device: &Device,
        _input: &Buffer,
        _weights: &Buffer,
        _bias: Option<&Buffer>,
        _initial_hidden: Option<&Buffer>,
        _config: &RnnConfig,
    ) -> BackendResult<RnnOutput> {
        Err(torsh_core::error::TorshError::BackendError(
            "GRU operations not implemented for this backend".to_string(),
        ))
    }

    async fn lstm_cell(
        &self,
        _device: &Device,
        _input: &Buffer,
        _hidden: &Buffer,
        _cell: &Buffer,
        _weights: &Buffer,
        _bias: Option<&Buffer>,
        _output_hidden: &Buffer,
        _output_cell: &Buffer,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "LSTM cell operations not implemented for this backend".to_string(),
        ))
    }

    async fn gru_cell(
        &self,
        _device: &Device,
        _input: &Buffer,
        _hidden: &Buffer,
        _weights: &Buffer,
        _bias: Option<&Buffer>,
        _output_hidden: &Buffer,
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "GRU cell operations not implemented for this backend".to_string(),
        ))
    }

    fn supports_rnn(&self) -> bool {
        false
    }

    fn supported_cell_types(&self) -> Vec<RnnCellType> {
        vec![]
    }

    fn supported_activations(&self) -> Vec<RnnActivation> {
        vec![]
    }
}

impl Default for DefaultRnnOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation function implementations
pub mod activations {
    /// Apply activation function to a single value
    pub fn apply_activation(value: f32, activation: super::RnnActivation) -> f32 {
        match activation {
            super::RnnActivation::Tanh => value.tanh(),
            super::RnnActivation::Relu => value.max(0.0),
            super::RnnActivation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            super::RnnActivation::Identity => value,
        }
    }

    /// Apply activation function to a slice of values
    pub fn apply_activation_slice(values: &mut [f32], activation: super::RnnActivation) {
        for value in values.iter_mut() {
            *value = apply_activation(*value, activation);
        }
    }
}

/// Cell implementations for CPU backend
pub mod cells {
    use super::*;

    /// LSTM cell implementation
    pub struct LstmCell;

    impl LstmCell {
        /// Execute LSTM cell forward pass
        pub fn forward(
            input: &[f32],
            hidden: &[f32],
            cell: &[f32],
            weights_ih: &[f32], // input-to-hidden weights
            weights_hh: &[f32], // hidden-to-hidden weights
            bias: Option<&[f32]>,
            output_hidden: &mut [f32],
            output_cell: &mut [f32],
        ) -> BackendResult<()> {
            let input_size = input.len();
            let hidden_size = hidden.len();

            if output_hidden.len() != hidden_size || output_cell.len() != hidden_size {
                return Err(torsh_core::error::TorshError::BackendError(
                    "Output buffer size mismatch".to_string(),
                ));
            }

            // LSTM gates: input, forget, cell, output
            let mut gates = vec![0.0; hidden_size * 4];

            // Compute input-to-hidden transformation
            for i in 0..hidden_size * 4 {
                let mut sum = 0.0;
                for j in 0..input_size {
                    sum += input[j] * weights_ih[i * input_size + j];
                }
                gates[i] = sum;
            }

            // Compute hidden-to-hidden transformation
            for i in 0..hidden_size * 4 {
                let mut sum = 0.0;
                for j in 0..hidden_size {
                    sum += hidden[j] * weights_hh[i * hidden_size + j];
                }
                gates[i] += sum;
            }

            // Add bias if provided
            if let Some(bias_data) = bias {
                for i in 0..hidden_size * 4 {
                    gates[i] += bias_data[i];
                }
            }

            // Apply activations and compute cell state
            for i in 0..hidden_size {
                let input_gate = activations::apply_activation(gates[i], RnnActivation::Sigmoid);
                let forget_gate =
                    activations::apply_activation(gates[i + hidden_size], RnnActivation::Sigmoid);
                let cell_gate =
                    activations::apply_activation(gates[i + 2 * hidden_size], RnnActivation::Tanh);
                let output_gate = activations::apply_activation(
                    gates[i + 3 * hidden_size],
                    RnnActivation::Sigmoid,
                );

                // Update cell state
                output_cell[i] = forget_gate * cell[i] + input_gate * cell_gate;

                // Update hidden state
                output_hidden[i] = output_gate
                    * activations::apply_activation(output_cell[i], RnnActivation::Tanh);
            }

            Ok(())
        }
    }

    /// GRU cell implementation
    pub struct GruCell;

    impl GruCell {
        /// Execute GRU cell forward pass
        pub fn forward(
            input: &[f32],
            hidden: &[f32],
            weights_ih: &[f32], // input-to-hidden weights
            weights_hh: &[f32], // hidden-to-hidden weights
            bias: Option<&[f32]>,
            output_hidden: &mut [f32],
        ) -> BackendResult<()> {
            let input_size = input.len();
            let hidden_size = hidden.len();

            if output_hidden.len() != hidden_size {
                return Err(torsh_core::error::TorshError::BackendError(
                    "Output buffer size mismatch".to_string(),
                ));
            }

            // GRU gates: reset, update, new
            let mut gates = vec![0.0; hidden_size * 3];

            // Compute input-to-hidden transformation
            for i in 0..hidden_size * 3 {
                let mut sum = 0.0;
                for j in 0..input_size {
                    sum += input[j] * weights_ih[i * input_size + j];
                }
                gates[i] = sum;
            }

            // Compute hidden-to-hidden transformation
            for i in 0..hidden_size * 3 {
                let mut sum = 0.0;
                for j in 0..hidden_size {
                    sum += hidden[j] * weights_hh[i * hidden_size + j];
                }
                gates[i] += sum;
            }

            // Add bias if provided
            if let Some(bias_data) = bias {
                for i in 0..hidden_size * 3 {
                    gates[i] += bias_data[i];
                }
            }

            // Apply activations and compute new hidden state
            for i in 0..hidden_size {
                let _reset_gate = activations::apply_activation(gates[i], RnnActivation::Sigmoid);
                let update_gate =
                    activations::apply_activation(gates[i + hidden_size], RnnActivation::Sigmoid);
                let new_gate =
                    activations::apply_activation(gates[i + 2 * hidden_size], RnnActivation::Tanh);

                // Compute new hidden state
                output_hidden[i] = (1.0 - update_gate) * hidden[i] + update_gate * new_gate;
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_config_creation() {
        let config = RnnConfig::lstm(128, 256, 2, 32, 50);

        assert_eq!(config.cell_type, RnnCellType::Lstm);
        assert_eq!(config.input_size, 128);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.sequence_length, 50);
        assert!(config.is_valid());
    }

    #[test]
    fn test_bidirectional_config() {
        let config = RnnConfig::lstm(128, 256, 1, 16, 20).bidirectional();

        assert_eq!(config.direction, RnnDirection::Bidirectional);
        assert_eq!(config.effective_hidden_size(), 512); // 256 * 2
    }

    #[test]
    fn test_buffer_size_calculations() {
        let config = RnnConfig::lstm(128, 256, 2, 4, 10);

        // Input: batch_size * sequence_length * input_size * element_size
        assert_eq!(config.input_buffer_size(), 4 * 10 * 128 * 4);

        // Output: batch_size * sequence_length * hidden_size * element_size
        assert_eq!(config.output_buffer_size(), 4 * 10 * 256 * 4);

        // Hidden state: batch_size * num_layers * hidden_size * element_size
        assert_eq!(config.hidden_state_buffer_size(), 4 * 2 * 256 * 4);

        // Cell state (for LSTM): same as hidden state
        assert_eq!(config.cell_state_buffer_size(), 4 * 2 * 256 * 4);
    }

    #[test]
    fn test_weight_buffer_size_lstm() {
        let config = RnnConfig::lstm(100, 200, 1, 1, 1);

        // LSTM has 4 gates
        // Layer 0: input-to-hidden (100 * 200 * 4) + hidden-to-hidden (200 * 200 * 4)
        let expected_weights = 4 * (100 * 200 + 200 * 200);
        assert_eq!(config.weight_buffer_size(), expected_weights * 4); // F32 = 4 bytes
    }

    #[test]
    fn test_gru_config() {
        let config = RnnConfig::gru(64, 128, 1, 8, 15);

        assert_eq!(config.cell_type, RnnCellType::Gru);
        assert_eq!(config.input_size, 64);
        assert_eq!(config.hidden_size, 128);
    }

    #[test]
    fn test_activation_functions() {
        use activations::*;

        assert!((apply_activation(0.0, RnnActivation::Tanh) - 0.0).abs() < 1e-6);
        assert!((apply_activation(0.0, RnnActivation::Sigmoid) - 0.5).abs() < 1e-6);
        assert_eq!(apply_activation(-1.0, RnnActivation::Relu), 0.0);
        assert_eq!(apply_activation(5.0, RnnActivation::Relu), 5.0);
        assert_eq!(apply_activation(42.0, RnnActivation::Identity), 42.0);
    }

    #[test]
    fn test_default_rnn_ops() {
        let ops = DefaultRnnOps::new();
        assert!(!ops.supports_rnn());
        assert!(ops.supported_cell_types().is_empty());
        assert!(ops.supported_activations().is_empty());
    }
}
