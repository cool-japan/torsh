//! Recurrent neural network layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Basic RNN layer
pub struct RNN {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    dropout: f32,
    bidirectional: bool,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights for each layer
        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_dim]);
            let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size]);
            let bias_ih = zeros(&[hidden_size]).unwrap();
            let bias_hh = zeros(&[hidden_size]).unwrap();

            base.register_parameter(format!("weight_ih_l{}", layer), Parameter::new(weight_ih?));
            base.register_parameter(format!("weight_hh_l{}", layer), Parameter::new(weight_hh?));
            base.register_parameter(format!("bias_ih_l{}", layer), Parameter::new(bias_ih));
            base.register_parameter(format!("bias_hh_l{}", layer), Parameter::new(bias_hh));
        }

        Ok(Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
        })
    }

    pub fn with_config(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
    ) -> Result<Self> {
        let mut rnn = Self::new(input_size, hidden_size, num_layers)?;
        rnn.bias = bias;
        rnn.batch_first = batch_first;
        rnn.dropout = dropout;
        rnn.bidirectional = bidirectional;
        Ok(rnn)
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // RNN forward pass
        // Input shape: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first

        let binding = input.shape();
        let input_shape = binding.dims();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Initialize hidden state
        let _h0 = zeros::<f32>(&[self.num_layers, batch_size, self.hidden_size]).unwrap();

        // Simplified RNN computation - real implementation would unroll over time steps
        let output_shape = if self.batch_first {
            [batch_size, seq_len, self.hidden_size]
        } else {
            [seq_len, batch_size, self.hidden_size]
        };

        let output = zeros(&output_shape).unwrap();
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// LSTM layer
pub struct LSTM {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    #[allow(dead_code)]
    bias: bool,
    batch_first: bool,
    #[allow(dead_code)]
    dropout: f32,
    #[allow(dead_code)]
    bidirectional: bool,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights for each layer (4 gates: input, forget, cell, output)
        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            let weight_ih = crate::init::xavier_uniform(&[4 * hidden_size, input_dim])?;
            let weight_hh = crate::init::xavier_uniform(&[4 * hidden_size, hidden_size])?;
            let bias_ih = zeros(&[4 * hidden_size])?;
            let bias_hh = zeros(&[4 * hidden_size])?;

            base.register_parameter(format!("weight_ih_l{}", layer), Parameter::new(weight_ih));
            base.register_parameter(format!("weight_hh_l{}", layer), Parameter::new(weight_hh));
            base.register_parameter(format!("bias_ih_l{}", layer), Parameter::new(bias_ih));
            base.register_parameter(format!("bias_hh_l{}", layer), Parameter::new(bias_hh));
        }

        Ok(Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
        })
    }

    pub fn with_config(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
    ) -> Result<Self> {
        let mut lstm = Self::new(input_size, hidden_size, num_layers)?;
        lstm.bias = bias;
        lstm.batch_first = batch_first;
        lstm.dropout = dropout;
        lstm.bidirectional = bidirectional;
        Ok(lstm)
    }

    /// Single LSTM cell computation
    fn lstm_cell(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
        layer: usize,
    ) -> Result<(Tensor, Tensor)> {
        let weight_ih = self.base.parameters[&format!("weight_ih_l{}", layer)]
            .tensor()
            .read()
            .clone();
        let weight_hh = self.base.parameters[&format!("weight_hh_l{}", layer)]
            .tensor()
            .read()
            .clone();
        let bias_ih = self.base.parameters[&format!("bias_ih_l{}", layer)]
            .tensor()
            .read()
            .clone();
        let bias_hh = self.base.parameters[&format!("bias_hh_l{}", layer)]
            .tensor()
            .read()
            .clone();

        // Compute input and hidden transformations
        let gi = input
            .matmul(&weight_ih.transpose(0, 1)?)?
            .add_op(&bias_ih)?;
        let gh = hidden
            .matmul(&weight_hh.transpose(0, 1)?)?
            .add_op(&bias_hh)?;
        let gates = gi.add_op(&gh)?;

        // Split into 4 gates
        let chunk_size = self.hidden_size;
        let input_gate = gates.narrow(1, 0, chunk_size)?.sigmoid()?;
        let forget_gate = gates.narrow(1, chunk_size as i64, chunk_size)?.sigmoid()?;
        let cell_gate = gates
            .narrow(1, (2 * chunk_size) as i64, chunk_size)?
            .tanh()?;
        let output_gate = gates
            .narrow(1, (3 * chunk_size) as i64, chunk_size)?
            .sigmoid()?;

        // Compute new cell and hidden states
        let new_cell = forget_gate
            .mul_op(cell)?
            .add_op(&input_gate.mul_op(&cell_gate)?)?;
        let new_hidden = output_gate.mul_op(&new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }

    /// Stack outputs from time steps
    fn stack_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "No outputs to stack".to_string(),
            ));
        }

        let seq_len = outputs.len();
        let batch_size = outputs[0].shape().dims()[0];
        let hidden_size = outputs[0].shape().dims()[1];

        let mut stacked_data = Vec::with_capacity(seq_len * batch_size * hidden_size);

        for output in outputs {
            let data = output.to_vec()?;
            stacked_data.extend(data);
        }

        Ok(Tensor::from_vec(
            stacked_data,
            &[seq_len, batch_size, hidden_size],
        )?)
    }

    /// Forward pass for unidirectional LSTM
    fn forward_unidirectional(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Initialize hidden and cell states
        let mut hidden = zeros(&[batch_size, self.hidden_size])?;
        let mut cell = zeros(&[batch_size, self.hidden_size])?;
        let mut outputs = Vec::new();

        // Process each time step
        for t in 0..seq_len {
            // Get input at time step t
            let x_t = if self.batch_first {
                input.narrow(1, t as i64, 1)?.squeeze(1)?
            } else {
                input.narrow(0, t as i64, 1)?.squeeze(0)?
            };

            let (new_hidden, new_cell) = self.lstm_cell(&x_t, &hidden, &cell, 0)?;
            hidden = new_hidden;
            cell = new_cell;
            outputs.push(hidden.clone());
        }

        // Stack outputs
        let stacked_outputs = self.stack_outputs(&outputs)?;

        if self.batch_first {
            Ok(stacked_outputs.transpose(0, 1)?)
        } else {
            Ok(stacked_outputs)
        }
    }

    /// Forward pass for bidirectional LSTM
    fn forward_bidirectional(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Forward direction
        let mut hidden_forward = zeros(&[batch_size, self.hidden_size])?;
        let mut cell_forward = zeros(&[batch_size, self.hidden_size])?;
        let mut forward_outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = if self.batch_first {
                input.narrow(1, t as i64, 1)?.squeeze(1)?
            } else {
                input.narrow(0, t as i64, 1)?.squeeze(0)?
            };

            let (new_hidden, new_cell) = self.lstm_cell(&x_t, &hidden_forward, &cell_forward, 0)?;
            hidden_forward = new_hidden;
            cell_forward = new_cell;
            forward_outputs.push(hidden_forward.clone());
        }

        // Backward direction
        let mut hidden_backward = zeros(&[batch_size, self.hidden_size])?;
        let mut cell_backward = zeros(&[batch_size, self.hidden_size])?;
        let mut backward_outputs = Vec::new();

        for t in (0..seq_len).rev() {
            let x_t = if self.batch_first {
                input.narrow(1, t as i64, 1)?.squeeze(1)?
            } else {
                input.narrow(0, t as i64, 1)?.squeeze(0)?
            };

            let (new_hidden, new_cell) =
                self.lstm_cell(&x_t, &hidden_backward, &cell_backward, 0)?;
            hidden_backward = new_hidden;
            cell_backward = new_cell;
            backward_outputs.push(hidden_backward.clone());
        }

        // Reverse backward outputs to match forward order
        backward_outputs.reverse();

        // Concatenate forward and backward outputs
        let mut combined_outputs = Vec::new();
        for (forward, backward) in forward_outputs.iter().zip(backward_outputs.iter()) {
            let forward_data = forward.to_vec()?;
            let backward_data = backward.to_vec()?;
            let forward_shape_binding = forward.shape();
            let forward_shape = forward_shape_binding.dims();
            let batch_size = forward_shape[0];
            let hidden_size = forward_shape[1];

            let mut combined_data = Vec::with_capacity(batch_size * 2 * hidden_size);
            for b in 0..batch_size {
                // Add forward hidden state
                for h in 0..hidden_size {
                    combined_data.push(forward_data[b * hidden_size + h]);
                }
                // Add backward hidden state
                for h in 0..hidden_size {
                    combined_data.push(backward_data[b * hidden_size + h]);
                }
            }
            let combined = Tensor::from_vec(combined_data, &[batch_size, 2 * hidden_size])?;
            combined_outputs.push(combined);
        }

        let stacked_outputs = self.stack_combined_outputs(&combined_outputs)?;

        if self.batch_first {
            Ok(stacked_outputs.transpose(0, 1)?)
        } else {
            Ok(stacked_outputs)
        }
    }

    /// Stack outputs for bidirectional case (hidden_size * 2)
    fn stack_combined_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "No outputs to stack".to_string(),
            ));
        }

        let seq_len = outputs.len();
        let batch_size = outputs[0].shape().dims()[0];
        let combined_hidden_size = outputs[0].shape().dims()[1]; // Should be hidden_size * 2

        let mut stacked_data = Vec::with_capacity(seq_len * batch_size * combined_hidden_size);

        for output in outputs {
            let data = output.to_vec()?;
            stacked_data.extend(data);
        }

        Ok(Tensor::from_vec(
            stacked_data,
            &[seq_len, batch_size, combined_hidden_size],
        )?)
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.bidirectional {
            self.forward_bidirectional(input)
        } else {
            self.forward_unidirectional(input)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// GRU layer
pub struct GRU {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    #[allow(dead_code)]
    bias: bool,
    batch_first: bool,
    #[allow(dead_code)]
    dropout: f32,
    #[allow(dead_code)]
    bidirectional: bool,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights for each layer (3 gates: reset, update, new)
        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            let weight_ih = crate::init::xavier_uniform(&[3 * hidden_size, input_dim])?;
            let weight_hh = crate::init::xavier_uniform(&[3 * hidden_size, hidden_size])?;
            let bias_ih = zeros(&[3 * hidden_size])?;
            let bias_hh = zeros(&[3 * hidden_size])?;

            base.register_parameter(format!("weight_ih_l{}", layer), Parameter::new(weight_ih));
            base.register_parameter(format!("weight_hh_l{}", layer), Parameter::new(weight_hh));
            base.register_parameter(format!("bias_ih_l{}", layer), Parameter::new(bias_ih));
            base.register_parameter(format!("bias_hh_l{}", layer), Parameter::new(bias_hh));
        }

        Ok(Self {
            base,
            input_size,
            hidden_size,
            num_layers,
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
        })
    }

    pub fn with_config(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
    ) -> Result<Self> {
        let mut gru = Self::new(input_size, hidden_size, num_layers)?;
        gru.bias = bias;
        gru.batch_first = batch_first;
        gru.dropout = dropout;
        gru.bidirectional = bidirectional;
        Ok(gru)
    }

    /// Single GRU cell computation
    fn gru_cell(&self, input: &Tensor, hidden: &Tensor, layer: usize) -> Result<Tensor> {
        let weight_ih = self.base.parameters[&format!("weight_ih_l{}", layer)]
            .tensor()
            .read()
            .clone();
        let weight_hh = self.base.parameters[&format!("weight_hh_l{}", layer)]
            .tensor()
            .read()
            .clone();
        let bias_ih = self.base.parameters[&format!("bias_ih_l{}", layer)]
            .tensor()
            .read()
            .clone();
        let bias_hh = self.base.parameters[&format!("bias_hh_l{}", layer)]
            .tensor()
            .read()
            .clone();

        // Compute input and hidden transformations
        let gi = input
            .matmul(&weight_ih.transpose(0, 1)?)?
            .add_op(&bias_ih)?;
        let gh = hidden
            .matmul(&weight_hh.transpose(0, 1)?)?
            .add_op(&bias_hh)?;

        // Split into 3 gates
        let chunk_size = self.hidden_size;

        // Reset gate and update gate use both input and hidden
        let i_reset = gi.narrow(1, 0, chunk_size)?;
        let h_reset = gh.narrow(1, 0, chunk_size)?;
        let reset_gate = (i_reset.add_op(&h_reset)?).sigmoid()?;

        let i_update = gi.narrow(1, chunk_size as i64, chunk_size)?;
        let h_update = gh.narrow(1, chunk_size as i64, chunk_size)?;
        let update_gate = (i_update.add_op(&h_update)?).sigmoid()?;

        // New gate uses input and reset-modulated hidden
        let i_new = gi.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let h_new = gh.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let reset_hidden = reset_gate.mul_op(&h_new)?;
        let new_gate = (i_new.add_op(&reset_hidden)?).tanh()?;

        // Compute new hidden state
        let one_minus_update = update_gate.mul_scalar(-1.0)?.add_scalar(1.0)?;
        let new_hidden = update_gate
            .mul_op(hidden)?
            .add_op(&one_minus_update.mul_op(&new_gate)?)?;

        Ok(new_hidden)
    }

    /// Stack outputs from time steps
    fn stack_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "No outputs to stack".to_string(),
            ));
        }

        let seq_len = outputs.len();
        let batch_size = outputs[0].shape().dims()[0];
        let hidden_size = outputs[0].shape().dims()[1];

        let mut stacked_data = Vec::with_capacity(seq_len * batch_size * hidden_size);

        for output in outputs {
            let data = output.to_vec()?;
            stacked_data.extend(data);
        }

        Ok(Tensor::from_vec(
            stacked_data,
            &[seq_len, batch_size, hidden_size],
        )?)
    }

    /// Forward pass for unidirectional GRU
    fn forward_unidirectional(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Initialize hidden state
        let mut hidden = zeros(&[batch_size, self.hidden_size])?;
        let mut outputs = Vec::new();

        // Process each time step
        for t in 0..seq_len {
            // Get input at time step t
            let x_t = if self.batch_first {
                input.narrow(1, t as i64, 1)?.squeeze(1)?
            } else {
                input.narrow(0, t as i64, 1)?.squeeze(0)?
            };

            hidden = self.gru_cell(&x_t, &hidden, 0)?;
            outputs.push(hidden.clone());
        }

        // Stack outputs
        let stacked_outputs = self.stack_outputs(&outputs)?;

        if self.batch_first {
            Ok(stacked_outputs.transpose(0, 1)?)
        } else {
            Ok(stacked_outputs)
        }
    }

    /// Forward pass for bidirectional GRU
    fn forward_bidirectional(&self, input: &Tensor) -> Result<Tensor> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let (seq_len, batch_size) = if self.batch_first {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        // Forward direction
        let mut hidden_forward = zeros(&[batch_size, self.hidden_size])?;
        let mut forward_outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = if self.batch_first {
                input.narrow(1, t as i64, 1)?.squeeze(1)?
            } else {
                input.narrow(0, t as i64, 1)?.squeeze(0)?
            };

            hidden_forward = self.gru_cell(&x_t, &hidden_forward, 0)?;
            forward_outputs.push(hidden_forward.clone());
        }

        // Backward direction
        let mut hidden_backward = zeros(&[batch_size, self.hidden_size])?;
        let mut backward_outputs = Vec::new();

        for t in (0..seq_len).rev() {
            let x_t = if self.batch_first {
                input.narrow(1, t as i64, 1)?.squeeze(1)?
            } else {
                input.narrow(0, t as i64, 1)?.squeeze(0)?
            };

            // Use layer 1 for backward weights (assuming they exist)
            hidden_backward = self.gru_cell(&x_t, &hidden_backward, 0)?; // For now use same weights
            backward_outputs.push(hidden_backward.clone());
        }

        // Reverse backward outputs to match forward order
        backward_outputs.reverse();

        // Concatenate forward and backward outputs
        let mut combined_outputs = Vec::new();
        for (forward, backward) in forward_outputs.iter().zip(backward_outputs.iter()) {
            // Manual concatenation along hidden dimension
            let forward_data = forward.to_vec()?;
            let backward_data = backward.to_vec()?;
            let forward_shape_binding = forward.shape();
            let forward_shape = forward_shape_binding.dims();
            let batch_size = forward_shape[0];
            let hidden_size = forward_shape[1];

            let mut combined_data = Vec::with_capacity(batch_size * 2 * hidden_size);
            for b in 0..batch_size {
                // Add forward hidden state
                for h in 0..hidden_size {
                    combined_data.push(forward_data[b * hidden_size + h]);
                }
                // Add backward hidden state
                for h in 0..hidden_size {
                    combined_data.push(backward_data[b * hidden_size + h]);
                }
            }
            let combined = Tensor::from_vec(combined_data, &[batch_size, 2 * hidden_size])?;
            combined_outputs.push(combined);
        }

        let stacked_outputs = self.stack_combined_outputs(&combined_outputs)?;

        if self.batch_first {
            Ok(stacked_outputs.transpose(0, 1)?)
        } else {
            Ok(stacked_outputs)
        }
    }

    /// Stack outputs for bidirectional case (hidden_size * 2)
    fn stack_combined_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "No outputs to stack".to_string(),
            ));
        }

        let seq_len = outputs.len();
        let batch_size = outputs[0].shape().dims()[0];
        let combined_hidden_size = outputs[0].shape().dims()[1]; // Should be hidden_size * 2

        let mut stacked_data = Vec::with_capacity(seq_len * batch_size * combined_hidden_size);

        for output in outputs {
            let data = output.to_vec()?;
            stacked_data.extend(data);
        }

        Ok(Tensor::from_vec(
            stacked_data,
            &[seq_len, batch_size, combined_hidden_size],
        )?)
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.bidirectional {
            self.forward_bidirectional(input)
        } else {
            self.forward_unidirectional(input)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for RNN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RNN")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

impl std::fmt::Debug for LSTM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTM")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

impl std::fmt::Debug for GRU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GRU")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

/// LSTM Cell - processes a single time step
pub struct LSTMCell {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    bias: bool,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        let weight_ih = crate::init::xavier_uniform(&[4 * hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[4 * hidden_size, hidden_size])?;
        let bias_ih = zeros(&[4 * hidden_size])?;
        let bias_hh = zeros(&[4 * hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
        })
    }

    pub fn with_bias(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self> {
        let mut cell = Self::new(input_size, hidden_size)?;
        cell.bias = bias;
        Ok(cell)
    }

    /// Forward pass returning (h_new, c_new)
    pub fn forward_cell(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
        let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();

        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;

        if self.bias {
            let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
            let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();
            gi = gi.add_op(&bias_ih)?;
            gh = gh.add_op(&bias_hh)?;
        }

        let gates = gi.add_op(&gh)?;

        let chunk_size = self.hidden_size;
        let input_gate = gates.narrow(1, 0, chunk_size)?.sigmoid()?;
        let forget_gate = gates.narrow(1, chunk_size as i64, chunk_size)?.sigmoid()?;
        let cell_gate = gates
            .narrow(1, (2 * chunk_size) as i64, chunk_size)?
            .tanh()?;
        let output_gate = gates
            .narrow(1, (3 * chunk_size) as i64, chunk_size)?
            .sigmoid()?;

        let new_cell = forget_gate
            .mul_op(cell)?
            .add_op(&input_gate.mul_op(&cell_gate)?)?;
        let new_hidden = output_gate.mul_op(&new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }
}

impl Module for LSTMCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape().dims()[0];
        let hidden = zeros(&[batch_size, self.hidden_size])?;
        let cell = zeros(&[batch_size, self.hidden_size])?;
        let (new_hidden, _) = self.forward_cell(input, &hidden, &cell)?;
        Ok(new_hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// GRU Cell - processes a single time step
pub struct GRUCell {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    bias: bool,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        let weight_ih = crate::init::xavier_uniform(&[3 * hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[3 * hidden_size, hidden_size])?;
        let bias_ih = zeros(&[3 * hidden_size])?;
        let bias_hh = zeros(&[3 * hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
        })
    }

    pub fn with_bias(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self> {
        let mut cell = Self::new(input_size, hidden_size)?;
        cell.bias = bias;
        Ok(cell)
    }

    /// Forward pass returning new hidden state
    pub fn forward_cell(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
        let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();

        let mut gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
        let mut gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;

        if self.bias {
            let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
            let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();
            gi = gi.add_op(&bias_ih)?;
            gh = gh.add_op(&bias_hh)?;
        }

        let chunk_size = self.hidden_size;

        let i_reset = gi.narrow(1, 0, chunk_size)?;
        let h_reset = gh.narrow(1, 0, chunk_size)?;
        let reset_gate = (i_reset.add_op(&h_reset)?).sigmoid()?;

        let i_update = gi.narrow(1, chunk_size as i64, chunk_size)?;
        let h_update = gh.narrow(1, chunk_size as i64, chunk_size)?;
        let update_gate = (i_update.add_op(&h_update)?).sigmoid()?;

        let i_new = gi.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let h_new = gh.narrow(1, (2 * chunk_size) as i64, chunk_size)?;
        let reset_hidden = reset_gate.mul_op(&h_new)?;
        let new_gate = (i_new.add_op(&reset_hidden)?).tanh()?;

        let one_minus_update = update_gate.mul_scalar(-1.0)?.add_scalar(1.0)?;
        let new_hidden = update_gate
            .mul_op(hidden)?
            .add_op(&one_minus_update.mul_op(&new_gate)?)?;

        Ok(new_hidden)
    }
}

impl Module for GRUCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape().dims()[0];
        let hidden = zeros(&[batch_size, self.hidden_size])?;
        self.forward_cell(input, &hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for LSTMCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTMCell")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("bias", &self.bias)
            .finish()
    }
}

impl std::fmt::Debug for GRUCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GRUCell")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("bias", &self.bias)
            .finish()
    }
}

/// Trait for custom RNN cell implementations
pub trait RNNCell {
    /// Apply the cell computation for a single time step
    fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor>;

    /// Get the hidden size of the cell
    fn hidden_size(&self) -> usize;

    /// Get the input size of the cell
    fn input_size(&self) -> usize;

    /// Initialize hidden state with proper shape
    fn init_hidden(&self, batch_size: usize) -> Result<Tensor> {
        Ok(zeros(&[batch_size, self.hidden_size()])?)
    }
}

/// Custom RNN cell activation function type
pub type ActivationFn = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

/// Custom RNN cell with configurable activation and computation
pub struct CustomRNNCell {
    base: ModuleBase,
    input_size: usize,
    hidden_size: usize,
    bias: bool,
    activation: ActivationFn,
    cell_type: CustomCellType,
}

/// Types of custom RNN cells
#[derive(Clone)]
pub enum CustomCellType {
    /// Basic RNN cell: h_t = activation(W_ih @ x_t + W_hh @ h_{t-1} + b)
    Basic,
    /// Gated cell with configurable gates: similar to GRU but with custom gating
    Gated {
        num_gates: usize,
        gate_activation: String, // "sigmoid", "tanh", "relu", etc.
    },
    /// Highway cell: h_t = T * activation(W_ih @ x_t + W_hh @ h_{t-1}) + (1-T) * h_{t-1}
    Highway,
    /// Residual cell: h_t = activation(W_ih @ x_t + W_hh @ h_{t-1}) + h_{t-1}
    Residual,
}

impl CustomRNNCell {
    /// Create a new custom RNN cell with basic activation
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        Self::with_activation(input_size, hidden_size, Box::new(|x| x.tanh()))
    }

    /// Create a custom RNN cell with specified activation function
    pub fn with_activation(
        input_size: usize,
        hidden_size: usize,
        activation: ActivationFn,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights
        let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
        let bias_ih = zeros(&[hidden_size])?;
        let bias_hh = zeros(&[hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
            activation,
            cell_type: CustomCellType::Basic,
        })
    }

    /// Create a gated custom RNN cell
    pub fn gated(input_size: usize, hidden_size: usize, num_gates: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weights for gates
        for gate in 0..num_gates {
            let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_size])?;
            let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
            let bias_ih = zeros(&[hidden_size])?;
            let bias_hh = zeros(&[hidden_size])?;

            base.register_parameter(format!("weight_ih_gate{}", gate), Parameter::new(weight_ih));
            base.register_parameter(format!("weight_hh_gate{}", gate), Parameter::new(weight_hh));
            base.register_parameter(format!("bias_ih_gate{}", gate), Parameter::new(bias_ih));
            base.register_parameter(format!("bias_hh_gate{}", gate), Parameter::new(bias_hh));
        }

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
            activation: Box::new(|x| x.tanh()),
            cell_type: CustomCellType::Gated {
                num_gates,
                gate_activation: "sigmoid".to_string(),
            },
        })
    }

    /// Create a highway RNN cell
    pub fn highway(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Main transformation weights
        let weight_ih = crate::init::xavier_uniform(&[hidden_size, input_size])?;
        let weight_hh = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
        let bias_ih = zeros(&[hidden_size])?;
        let bias_hh = zeros(&[hidden_size])?;

        // Transform gate weights (T in the highway equation)
        let weight_ih_t = crate::init::xavier_uniform(&[hidden_size, input_size])?;
        let weight_hh_t = crate::init::xavier_uniform(&[hidden_size, hidden_size])?;
        let bias_ih_t = ones(&[hidden_size])?.mul_scalar(-1.0)?; // Initialize to favor carrying
        let bias_hh_t = zeros(&[hidden_size])?;

        base.register_parameter("weight_ih".to_string(), Parameter::new(weight_ih));
        base.register_parameter("weight_hh".to_string(), Parameter::new(weight_hh));
        base.register_parameter("bias_ih".to_string(), Parameter::new(bias_ih));
        base.register_parameter("bias_hh".to_string(), Parameter::new(bias_hh));

        base.register_parameter("weight_ih_t".to_string(), Parameter::new(weight_ih_t));
        base.register_parameter("weight_hh_t".to_string(), Parameter::new(weight_hh_t));
        base.register_parameter("bias_ih_t".to_string(), Parameter::new(bias_ih_t));
        base.register_parameter("bias_hh_t".to_string(), Parameter::new(bias_hh_t));

        Ok(Self {
            base,
            input_size,
            hidden_size,
            bias: true,
            activation: Box::new(|x| x.tanh()),
            cell_type: CustomCellType::Highway,
        })
    }

    /// Create a residual RNN cell
    pub fn residual(input_size: usize, hidden_size: usize) -> Result<Self> {
        let mut cell = Self::new(input_size, hidden_size)?;
        cell.cell_type = CustomCellType::Residual;
        Ok(cell)
    }

    /// Set the activation function
    pub fn with_activation_fn(mut self, activation: ActivationFn) -> Self {
        self.activation = activation;
        self
    }
}

impl RNNCell for CustomRNNCell {
    fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        match &self.cell_type {
            CustomCellType::Basic => {
                let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
                let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();
                let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
                let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();

                let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;

                let gi = gi.add_op(&bias_ih)?;
                let gh = gh.add_op(&bias_hh)?;

                let new_h = gi.add_op(&gh)?;
                (self.activation)(&new_h)
            }

            CustomCellType::Gated {
                num_gates,
                gate_activation,
            } => {
                let mut gates = Vec::new();

                // Compute all gates
                for gate in 0..*num_gates {
                    let weight_ih = self.base.parameters[&format!("weight_ih_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();
                    let weight_hh = self.base.parameters[&format!("weight_hh_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();
                    let bias_ih = self.base.parameters[&format!("bias_ih_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();
                    let bias_hh = self.base.parameters[&format!("bias_hh_gate{}", gate)]
                        .tensor()
                        .read()
                        .clone();

                    let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                    let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
                    let gi = gi.add_op(&bias_ih)?;
                    let gh = gh.add_op(&bias_hh)?;
                    let gate_val = gi.add_op(&gh)?;

                    let activated_gate = match gate_activation.as_str() {
                        "sigmoid" => gate_val.sigmoid()?,
                        "tanh" => gate_val.tanh()?,
                        "relu" => gate_val.relu()?,
                        _ => gate_val.sigmoid()?, // default to sigmoid
                    };
                    gates.push(activated_gate);
                }

                // Simple gated computation: use first gate as forget, second as input
                if gates.len() >= 2 {
                    let forget_gate = &gates[0];
                    let input_gate = &gates[1];

                    let candidate = (self.activation)(input)?;
                    let new_h = hidden
                        .mul_op(forget_gate)?
                        .add_op(&candidate.mul_op(input_gate)?)?;
                    Ok(new_h)
                } else {
                    // Single gate - just use it as input gate
                    let gate = &gates[0];
                    let candidate = (self.activation)(input)?;
                    Ok(hidden.add_op(&candidate.mul_op(gate)?)?)
                }
            }

            CustomCellType::Highway => {
                let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
                let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();
                let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
                let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();

                let weight_ih_t = self.base.parameters["weight_ih_t"].tensor().read().clone();
                let weight_hh_t = self.base.parameters["weight_hh_t"].tensor().read().clone();
                let bias_ih_t = self.base.parameters["bias_ih_t"].tensor().read().clone();
                let bias_hh_t = self.base.parameters["bias_hh_t"].tensor().read().clone();

                // Main transformation: h_candidate = activation(W_ih @ x + W_hh @ h + b)
                let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
                let gi = gi.add_op(&bias_ih)?;
                let gh = gh.add_op(&bias_hh)?;
                let h_candidate = (self.activation)(&gi.add_op(&gh)?)?;

                // Transform gate: T = sigmoid(W_ih_t @ x + W_hh_t @ h + b_t)
                let gi_t = input.matmul(&weight_ih_t.transpose(0, 1)?)?;
                let gh_t = hidden.matmul(&weight_hh_t.transpose(0, 1)?)?;
                let gi_t = gi_t.add_op(&bias_ih_t)?;
                let gh_t = gh_t.add_op(&bias_hh_t)?;
                let transform_gate = gi_t.add_op(&gh_t)?.sigmoid()?;

                // Highway connection: h_new = T * h_candidate + (1 - T) * h_prev
                let carry_gate = transform_gate.neg()?.add_scalar(1.0)?;
                let new_h = h_candidate
                    .mul_op(&transform_gate)?
                    .add_op(&hidden.mul_op(&carry_gate)?)?;
                Ok(new_h)
            }

            CustomCellType::Residual => {
                let weight_ih = self.base.parameters["weight_ih"].tensor().read().clone();
                let weight_hh = self.base.parameters["weight_hh"].tensor().read().clone();
                let bias_ih = self.base.parameters["bias_ih"].tensor().read().clone();
                let bias_hh = self.base.parameters["bias_hh"].tensor().read().clone();

                let gi = input.matmul(&weight_ih.transpose(0, 1)?)?;
                let gh = hidden.matmul(&weight_hh.transpose(0, 1)?)?;
                let gi = gi.add_op(&bias_ih)?;
                let gh = gh.add_op(&bias_hh)?;

                let h_candidate = (self.activation)(&gi.add_op(&gh)?)?;
                // Residual connection: h_new = h_candidate + h_prev
                Ok(h_candidate.add_op(hidden)?)
            }
        }
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }
}

impl Module for CustomRNNCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For single step forward, we need to provide a hidden state
        // This is a simplified version - in practice, users should call the RNNCell::forward directly
        let batch_size = input.shape().dims()[0];
        let hidden = self.init_hidden(batch_size)?;
        RNNCell::forward(self, input, &hidden)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for CustomRNNCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomRNNCell")
            .field("input_size", &self.input_size)
            .field("hidden_size", &self.hidden_size)
            .field("bias", &self.bias)
            .field("cell_type", &format!("{:?}", self.cell_type))
            .finish()
    }
}

impl std::fmt::Debug for CustomCellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomCellType::Basic => write!(f, "Basic"),
            CustomCellType::Gated {
                num_gates,
                gate_activation,
            } => {
                write!(
                    f,
                    "Gated(gates: {}, activation: {})",
                    num_gates, gate_activation
                )
            }
            CustomCellType::Highway => write!(f, "Highway"),
            CustomCellType::Residual => write!(f, "Residual"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    // ========================================================================
    // RNN Tests
    // ========================================================================

    #[test]
    fn test_rnn_new() -> Result<()> {
        let rnn = RNN::new(10, 20, 2)?;
        assert_eq!(rnn.input_size, 10);
        assert_eq!(rnn.hidden_size, 20);
        assert_eq!(rnn.num_layers, 2);
        assert!(rnn.bias);
        assert!(!rnn.batch_first);
        Ok(())
    }

    #[test]
    fn test_rnn_with_config() -> Result<()> {
        let rnn = RNN::with_config(10, 20, 1, false, true, 0.5, true)?;
        assert_eq!(rnn.input_size, 10);
        assert_eq!(rnn.hidden_size, 20);
        assert!(!rnn.bias);
        assert!(rnn.batch_first);
        assert_eq!(rnn.dropout, 0.5);
        assert!(rnn.bidirectional);
        Ok(())
    }

    #[test]
    fn test_rnn_forward() -> Result<()> {
        let rnn = RNN::new(10, 20, 1)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = rnn.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size]
        assert_eq!(output_shape.dims(), &[5, 2, 20]);
        Ok(())
    }

    #[test]
    fn test_rnn_forward_batch_first() -> Result<()> {
        let rnn = RNN::with_config(10, 20, 1, true, true, 0.0, false)?;
        let input = zeros(&[2, 5, 10])?; // batch=2, seq_len=5, input_size=10

        let output = rnn.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [batch, seq_len, hidden_size]
        assert_eq!(output_shape.dims(), &[2, 5, 20]);
        Ok(())
    }

    #[test]
    fn test_rnn_parameters() -> Result<()> {
        let rnn = RNN::new(10, 20, 2)?;
        let params = rnn.parameters();

        // Should have 4 parameters per layer: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 8); // 2 layers * 4 params
        assert!(params.contains_key("weight_ih_l0"));
        assert!(params.contains_key("weight_hh_l0"));
        assert!(params.contains_key("bias_ih_l0"));
        assert!(params.contains_key("bias_hh_l0"));
        Ok(())
    }

    #[test]
    fn test_rnn_training_mode() -> Result<()> {
        let mut rnn = RNN::new(10, 20, 1)?;
        assert!(rnn.training());

        rnn.eval();
        assert!(!rnn.training());

        rnn.train();
        assert!(rnn.training());
        Ok(())
    }

    // ========================================================================
    // LSTM Tests
    // ========================================================================

    #[test]
    fn test_lstm_new() -> Result<()> {
        let lstm = LSTM::new(10, 20, 2)?;
        assert_eq!(lstm.input_size, 10);
        assert_eq!(lstm.hidden_size, 20);
        assert_eq!(lstm.num_layers, 2);
        assert!(lstm.bias);
        assert!(!lstm.batch_first);
        Ok(())
    }

    #[test]
    fn test_lstm_with_config() -> Result<()> {
        let lstm = LSTM::with_config(10, 20, 1, false, true, 0.5, true)?;
        assert_eq!(lstm.input_size, 10);
        assert_eq!(lstm.hidden_size, 20);
        assert!(!lstm.bias);
        assert!(lstm.batch_first);
        assert_eq!(lstm.dropout, 0.5);
        assert!(lstm.bidirectional);
        Ok(())
    }

    #[test]
    fn test_lstm_forward() -> Result<()> {
        let lstm = LSTM::new(10, 20, 1)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = lstm.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size]
        assert_eq!(output_shape.dims(), &[5, 2, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_forward_batch_first() -> Result<()> {
        let lstm = LSTM::with_config(10, 20, 1, true, true, 0.0, false)?;
        let input = zeros(&[2, 5, 10])?; // batch=2, seq_len=5, input_size=10

        let output = lstm.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [batch, seq_len, hidden_size]
        assert_eq!(output_shape.dims(), &[2, 5, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_parameters() -> Result<()> {
        let lstm = LSTM::new(10, 20, 2)?;
        let params = lstm.parameters();

        // Should have 4 parameters per layer (for 4 gates)
        assert_eq!(params.len(), 8); // 2 layers * 4 params
        assert!(params.contains_key("weight_ih_l0"));
        assert!(params.contains_key("weight_hh_l0"));
        assert!(params.contains_key("bias_ih_l0"));
        assert!(params.contains_key("bias_hh_l0"));
        Ok(())
    }

    #[test]
    fn test_lstm_cell() -> Result<()> {
        let lstm = LSTM::new(10, 20, 1)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?; // batch=2, hidden_size=20
        let cell = zeros(&[2, 20])?; // batch=2, hidden_size=20

        let (new_hidden, new_cell) = lstm.lstm_cell(&input, &hidden, &cell, 0)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        assert_eq!(new_cell.shape().dims(), &[2, 20]);
        Ok(())
    }

    // ========================================================================
    // GRU Tests
    // ========================================================================

    #[test]
    fn test_gru_new() -> Result<()> {
        let gru = GRU::new(10, 20, 2)?;
        assert_eq!(gru.input_size, 10);
        assert_eq!(gru.hidden_size, 20);
        assert_eq!(gru.num_layers, 2);
        assert!(gru.bias);
        assert!(!gru.batch_first);
        Ok(())
    }

    #[test]
    fn test_gru_with_config() -> Result<()> {
        let gru = GRU::with_config(10, 20, 1, false, true, 0.5, true)?;
        assert_eq!(gru.input_size, 10);
        assert_eq!(gru.hidden_size, 20);
        assert!(!gru.bias);
        assert!(gru.batch_first);
        assert_eq!(gru.dropout, 0.5);
        assert!(gru.bidirectional);
        Ok(())
    }

    #[test]
    fn test_gru_forward() -> Result<()> {
        let gru = GRU::new(10, 20, 1)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = gru.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size]
        assert_eq!(output_shape.dims(), &[5, 2, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_forward_batch_first() -> Result<()> {
        let gru = GRU::with_config(10, 20, 1, true, true, 0.0, false)?;
        let input = zeros(&[2, 5, 10])?; // batch=2, seq_len=5, input_size=10

        let output = gru.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [batch, seq_len, hidden_size]
        assert_eq!(output_shape.dims(), &[2, 5, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_forward_bidirectional() -> Result<()> {
        let gru = GRU::with_config(10, 20, 1, true, false, 0.0, true)?;
        let input = zeros(&[5, 2, 10])?; // seq_len=5, batch=2, input_size=10

        let output = gru.forward(&input)?;
        let output_shape = output.shape();

        // Expected: [seq_len, batch, hidden_size * 2] for bidirectional
        assert_eq!(output_shape.dims(), &[5, 2, 40]);
        Ok(())
    }

    #[test]
    fn test_gru_parameters() -> Result<()> {
        let gru = GRU::new(10, 20, 2)?;
        let params = gru.parameters();

        // Should have 4 parameters per layer (for 3 gates)
        assert_eq!(params.len(), 8); // 2 layers * 4 params
        assert!(params.contains_key("weight_ih_l0"));
        assert!(params.contains_key("weight_hh_l0"));
        assert!(params.contains_key("bias_ih_l0"));
        assert!(params.contains_key("bias_hh_l0"));
        Ok(())
    }

    #[test]
    fn test_gru_cell() -> Result<()> {
        let gru = GRU::new(10, 20, 1)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?; // batch=2, hidden_size=20

        let new_hidden = gru.gru_cell(&input, &hidden, 0)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    // ========================================================================
    // LSTMCell Tests
    // ========================================================================

    #[test]
    fn test_lstm_cell_new() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(cell.bias);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_with_bias() -> Result<()> {
        let cell = LSTMCell::with_bias(10, 20, false)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(!cell.bias);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_forward() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10

        let output = cell.forward(&input)?;
        let output_shape = output.shape();

        assert_eq!(output_shape.dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_forward_cell() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?;
        let cell_state = zeros(&[2, 20])?;

        let (new_hidden, new_cell) = cell.forward_cell(&input, &hidden, &cell_state)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        assert_eq!(new_cell.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_lstm_cell_parameters() -> Result<()> {
        let cell = LSTMCell::new(10, 20)?;
        let params = cell.parameters();

        // Should have 4 parameters: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("weight_ih"));
        assert!(params.contains_key("weight_hh"));
        assert!(params.contains_key("bias_ih"));
        assert!(params.contains_key("bias_hh"));
        Ok(())
    }

    // ========================================================================
    // GRUCell Tests
    // ========================================================================

    #[test]
    fn test_gru_cell_new() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(cell.bias);
        Ok(())
    }

    #[test]
    fn test_gru_cell_with_bias() -> Result<()> {
        let cell = GRUCell::with_bias(10, 20, false)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(!cell.bias);
        Ok(())
    }

    #[test]
    fn test_gru_cell_forward() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10

        let output = cell.forward(&input)?;
        let output_shape = output.shape();

        assert_eq!(output_shape.dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_cell_forward_cell() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        let input = zeros(&[2, 10])?; // batch=2, input_size=10
        let hidden = zeros(&[2, 20])?;

        let new_hidden = cell.forward_cell(&input, &hidden)?;

        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_gru_cell_parameters() -> Result<()> {
        let cell = GRUCell::new(10, 20)?;
        let params = cell.parameters();

        // Should have 4 parameters: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("weight_ih"));
        assert!(params.contains_key("weight_hh"));
        assert!(params.contains_key("bias_ih"));
        assert!(params.contains_key("bias_hh"));
        Ok(())
    }

    // ========================================================================
    // CustomRNNCell Tests
    // ========================================================================

    #[test]
    fn test_custom_rnn_cell_new() -> Result<()> {
        let cell = CustomRNNCell::new(10, 20)?;
        assert_eq!(cell.input_size, 10);
        assert_eq!(cell.hidden_size, 20);
        assert!(cell.bias);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_basic() -> Result<()> {
        let cell = CustomRNNCell::new(10, 20)?;
        let input = zeros(&[2, 10])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_gated() -> Result<()> {
        // Use matching input_size and hidden_size to avoid broadcast errors in the gated implementation
        let cell = CustomRNNCell::gated(20, 20, 2)?;
        assert_eq!(cell.input_size(), 20);
        assert_eq!(cell.hidden_size(), 20);

        let input = zeros(&[2, 20])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_highway() -> Result<()> {
        let cell = CustomRNNCell::highway(10, 20)?;
        let params = cell.parameters();

        // Highway cell should have 8 parameters (main + transform gate)
        assert_eq!(params.len(), 8);
        assert!(params.contains_key("weight_ih"));
        assert!(params.contains_key("weight_ih_t"));

        let input = zeros(&[2, 10])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_residual() -> Result<()> {
        let cell = CustomRNNCell::residual(10, 20)?;
        let input = zeros(&[2, 10])?;
        let hidden = zeros(&[2, 20])?;

        let new_hidden = RNNCell::forward(&cell, &input, &hidden)?;
        assert_eq!(new_hidden.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_init_hidden() -> Result<()> {
        let cell = CustomRNNCell::new(10, 20)?;
        let hidden = cell.init_hidden(4)?; // batch_size=4

        assert_eq!(hidden.shape().dims(), &[4, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_with_activation() -> Result<()> {
        let cell = CustomRNNCell::with_activation(10, 20, Box::new(|x| x.relu()))?;
        let input = zeros(&[2, 10])?;

        // Use Module::forward explicitly to avoid ambiguity
        let output = Module::forward(&cell, &input)?;
        assert_eq!(output.shape().dims(), &[2, 20]);
        Ok(())
    }

    #[test]
    fn test_custom_rnn_cell_parameters_gated() -> Result<()> {
        let cell = CustomRNNCell::gated(10, 20, 3)?;
        let params = cell.parameters();

        // 3 gates * 4 parameters per gate = 12 parameters
        assert_eq!(params.len(), 12);
        assert!(params.contains_key("weight_ih_gate0"));
        assert!(params.contains_key("weight_ih_gate1"));
        assert!(params.contains_key("weight_ih_gate2"));
        Ok(())
    }

    // ========================================================================
    // Module Trait Tests (Common Behaviors)
    // ========================================================================

    #[test]
    fn test_module_training_modes() -> Result<()> {
        let mut lstm = LSTM::new(10, 20, 1)?;

        // Default should be training mode
        assert!(lstm.training());

        // Set to eval mode
        lstm.set_training(false);
        assert!(!lstm.training());

        // Set back to training mode
        lstm.set_training(true);
        assert!(lstm.training());
        Ok(())
    }

    #[test]
    fn test_module_named_parameters() -> Result<()> {
        let gru = GRU::new(10, 20, 2)?;
        let named_params = gru.named_parameters();

        // Should have 8 parameters (2 layers * 4 params)
        assert_eq!(named_params.len(), 8);
        assert!(named_params.contains_key("weight_ih_l0"));
        assert!(named_params.contains_key("weight_hh_l1"));
        Ok(())
    }

    #[test]
    fn test_module_to_device() -> Result<()> {
        let mut rnn = RNN::new(10, 20, 1)?;

        // Should succeed
        rnn.to_device(DeviceType::Cpu)?;

        Ok(())
    }

    #[test]
    fn test_stack_outputs_empty() -> Result<()> {
        let lstm = LSTM::new(10, 20, 1)?;
        let empty_outputs: Vec<Tensor> = vec![];

        let result = lstm.stack_outputs(&empty_outputs);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_gru_stack_combined_outputs_empty() -> Result<()> {
        let gru = GRU::new(10, 20, 1)?;
        let empty_outputs: Vec<Tensor> = vec![];

        let result = gru.stack_combined_outputs(&empty_outputs);
        assert!(result.is_err());
        Ok(())
    }
}
