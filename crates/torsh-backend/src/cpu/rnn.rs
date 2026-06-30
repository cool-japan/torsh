//! CPU RNN implementation with optimized algorithms

use crate::cpu::buffer::BufferCpuExt;
use crate::rnn::{cells, RnnActivation, RnnOps, RnnOutput, RnnPerformanceHints};
use crate::{BackendResult, Buffer, Device};

// Re-export for benchmarks
pub use crate::rnn::{RnnCellType, RnnConfig};

/// Calculate the weight buffer size required for LSTM configuration
///
/// This is a standalone function for benchmarking that wraps the method
/// on CpuRnnOps.
pub fn calculate_weight_buffer_size_lstm(config: &RnnConfig) -> usize {
    CpuRnnOps::calculate_weight_buffer_size_lstm(config)
}

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// CPU RNN operations implementation
#[derive(Clone, Debug)]
pub struct CpuRnnOps {
    /// Performance hints for optimization
    #[allow(dead_code)]
    performance_hints: RnnPerformanceHints,
    /// Number of threads for parallel processing
    #[allow(dead_code)]
    num_threads: usize,
}

impl CpuRnnOps {
    /// Create a new CPU RNN operations instance
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| rayon::current_num_threads());

        Self {
            performance_hints: RnnPerformanceHints {
                optimal_batch_size: 16,
                optimal_sequence_length: 32,
                use_fused_ops: true,
                use_optimized_libs: false, // CPU doesn't use cuDNN
                memory_bandwidth: 100.0,   // CPU memory bandwidth
                compute_throughput: num_threads as f32 * 100.0, // Estimated GOPS
            },
            num_threads,
        }
    }

    /// Get data from buffer as f32 slice
    fn get_buffer_data_f32(&self, buffer: &Buffer) -> BackendResult<Vec<f32>> {
        if !buffer.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Buffer must be a CPU buffer".to_string(),
            ));
        }

        let ptr = buffer.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError("Failed to get buffer pointer".to_string())
        })?;

        unsafe {
            let data = std::slice::from_raw_parts(ptr as *const f32, buffer.size / 4);
            Ok(data.to_vec())
        }
    }

    /// Write data to buffer from f32 slice
    fn write_buffer_data_f32(&self, buffer: &Buffer, data: &[f32]) -> BackendResult<()> {
        if !buffer.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Buffer must be a CPU buffer".to_string(),
            ));
        }

        let ptr = buffer.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError("Failed to get buffer pointer".to_string())
        })?;

        let expected_elements = buffer.size / 4;
        if data.len() != expected_elements {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "Data size mismatch: expected {} elements, got {}",
                expected_elements,
                data.len()
            )));
        }

        unsafe {
            let buffer_data = std::slice::from_raw_parts_mut(ptr as *mut f32, expected_elements);
            buffer_data.copy_from_slice(data);
        }

        Ok(())
    }

    /// Allocate a new CPU [`Buffer`] and populate it with `data`.
    ///
    /// The buffer is sized to hold exactly `data.len()` `f32` values. This is
    /// used to return the genuinely computed RNN/LSTM/GRU outputs instead of
    /// discarding them.
    fn create_buffer_from_f32(&self, device: &Device, data: &[f32]) -> BackendResult<Buffer> {
        use crate::buffer::{BufferDescriptor, BufferUsage, MemoryLocation};
        use crate::cpu::buffer::CpuBuffer;

        let size_bytes = std::mem::size_of_val(data);
        let descriptor = BufferDescriptor {
            size: size_bytes,
            usage: BufferUsage::STORAGE_READ_WRITE,
            location: MemoryLocation::Host,
            dtype: Some(torsh_core::dtype::DType::F32),
            shape: None,
            initial_data: None,
            alignment: None,
            zero_init: false,
        };

        let buffer = CpuBuffer::new_buffer(device.clone(), &descriptor)?;
        self.write_buffer_data_f32(&buffer, data)?;
        Ok(buffer)
    }

    /// Execute LSTM sequence processing
    fn execute_lstm_sequence(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        initial_hidden: Option<&[f32]>,
        initial_cell: Option<&[f32]>,
        config: &RnnConfig,
    ) -> BackendResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let batch_size = config.batch_size;
        let sequence_length = config.sequence_length;
        let input_size = config.input_size;
        let hidden_size = config.hidden_size;

        // Initialize states
        let mut hidden_state = if let Some(h0) = initial_hidden {
            h0.to_vec()
        } else {
            vec![0.0; batch_size * hidden_size]
        };

        let mut cell_state = if let Some(c0) = initial_cell {
            c0.to_vec()
        } else {
            vec![0.0; batch_size * hidden_size]
        };

        // Output sequences
        let mut output_sequences = if config.return_sequences {
            vec![0.0; batch_size * sequence_length * hidden_size]
        } else {
            vec![0.0; batch_size * hidden_size]
        };

        // Extract weights (simplified - assuming single layer for now)
        let weights_per_gate = (input_size + hidden_size) * hidden_size;
        let total_weights_per_direction = weights_per_gate * 4; // 4 gates for LSTM

        if weights.len() < total_weights_per_direction {
            return Err(torsh_core::error::TorshError::BackendError(
                "Insufficient weight data".to_string(),
            ));
        }

        // Split weights into input-to-hidden and hidden-to-hidden
        let ih_size = input_size * hidden_size * 4;
        let hh_size = hidden_size * hidden_size * 4;
        let weights_ih = &weights[0..ih_size];
        let weights_hh = &weights[ih_size..ih_size + hh_size];

        // Process each timestep
        for t in 0..sequence_length {
            // Process each batch element
            for b in 0..batch_size {
                let input_start = b * sequence_length * input_size + t * input_size;
                let input_end = input_start + input_size;
                let batch_input = &input[input_start..input_end];

                let hidden_start = b * hidden_size;
                let hidden_end = hidden_start + hidden_size;
                let batch_hidden = &hidden_state[hidden_start..hidden_end];
                let batch_cell = &cell_state[hidden_start..hidden_end];

                let mut new_hidden = vec![0.0; hidden_size];
                let mut new_cell = vec![0.0; hidden_size];

                // Execute LSTM cell
                cells::LstmCell::forward(
                    batch_input,
                    batch_hidden,
                    batch_cell,
                    weights_ih,
                    weights_hh,
                    bias,
                    &mut new_hidden,
                    &mut new_cell,
                )?;

                // Update states
                hidden_state[hidden_start..hidden_end].copy_from_slice(&new_hidden);
                cell_state[hidden_start..hidden_end].copy_from_slice(&new_cell);

                // Store output
                if config.return_sequences {
                    let output_start = b * sequence_length * hidden_size + t * hidden_size;
                    let output_end = output_start + hidden_size;
                    output_sequences[output_start..output_end].copy_from_slice(&new_hidden);
                } else if t == sequence_length - 1 {
                    // Only store the last timestep
                    let output_start = b * hidden_size;
                    let output_end = output_start + hidden_size;
                    output_sequences[output_start..output_end].copy_from_slice(&new_hidden);
                }
            }
        }

        Ok((output_sequences, hidden_state, cell_state))
    }

    /// Execute GRU sequence processing
    fn execute_gru_sequence(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        initial_hidden: Option<&[f32]>,
        config: &RnnConfig,
    ) -> BackendResult<(Vec<f32>, Vec<f32>)> {
        let batch_size = config.batch_size;
        let sequence_length = config.sequence_length;
        let input_size = config.input_size;
        let hidden_size = config.hidden_size;

        // Initialize hidden state
        let mut hidden_state = if let Some(h0) = initial_hidden {
            h0.to_vec()
        } else {
            vec![0.0; batch_size * hidden_size]
        };

        // Output sequences
        let mut output_sequences = if config.return_sequences {
            vec![0.0; batch_size * sequence_length * hidden_size]
        } else {
            vec![0.0; batch_size * hidden_size]
        };

        // Extract weights (simplified - assuming single layer for now)
        let weights_per_gate = (input_size + hidden_size) * hidden_size;
        let total_weights_per_direction = weights_per_gate * 3; // 3 gates for GRU

        if weights.len() < total_weights_per_direction {
            return Err(torsh_core::error::TorshError::BackendError(
                "Insufficient weight data".to_string(),
            ));
        }

        // Split weights into input-to-hidden and hidden-to-hidden
        let ih_size = input_size * hidden_size * 3;
        let hh_size = hidden_size * hidden_size * 3;
        let weights_ih = &weights[0..ih_size];
        let weights_hh = &weights[ih_size..ih_size + hh_size];

        // Process each timestep
        for t in 0..sequence_length {
            // Process each batch element
            for b in 0..batch_size {
                let input_start = b * sequence_length * input_size + t * input_size;
                let input_end = input_start + input_size;
                let batch_input = &input[input_start..input_end];

                let hidden_start = b * hidden_size;
                let hidden_end = hidden_start + hidden_size;
                let batch_hidden = &hidden_state[hidden_start..hidden_end];

                let mut new_hidden = vec![0.0; hidden_size];

                // Execute GRU cell
                cells::GruCell::forward(
                    batch_input,
                    batch_hidden,
                    weights_ih,
                    weights_hh,
                    bias,
                    &mut new_hidden,
                )?;

                // Update state
                hidden_state[hidden_start..hidden_end].copy_from_slice(&new_hidden);

                // Store output
                if config.return_sequences {
                    let output_start = b * sequence_length * hidden_size + t * hidden_size;
                    let output_end = output_start + hidden_size;
                    output_sequences[output_start..output_end].copy_from_slice(&new_hidden);
                } else if t == sequence_length - 1 {
                    // Only store the last timestep
                    let output_start = b * hidden_size;
                    let output_end = output_start + hidden_size;
                    output_sequences[output_start..output_end].copy_from_slice(&new_hidden);
                }
            }
        }

        Ok((output_sequences, hidden_state))
    }

    /// Calculate the weight buffer size required for LSTM configuration
    ///
    /// This function computes the total size needed for all LSTM weight matrices
    /// including input-to-hidden, hidden-to-hidden, and bias vectors.
    ///
    /// # Arguments
    ///
    /// * `config` - LSTM configuration specifying dimensions
    ///
    /// # Returns
    ///
    /// Total buffer size in bytes required for all LSTM weights
    pub fn calculate_weight_buffer_size_lstm(config: &RnnConfig) -> usize {
        // LSTM has 4 gates (input, forget, output, cell), each requiring weights
        let input_size = config.input_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        // Weight matrices: input-to-hidden and hidden-to-hidden for each gate
        let weights_ih_size = input_size * hidden_size * 4; // 4 gates
        let weights_hh_size = hidden_size * hidden_size * 4; // 4 gates

        // Bias vectors: one for each gate
        let bias_size = hidden_size * 4; // 4 gates

        let total_params_per_layer = weights_ih_size + weights_hh_size + bias_size;
        let total_params = total_params_per_layer * num_layers;

        // Assuming f32 parameters (4 bytes each)
        total_params * 4
    }
}

#[async_trait::async_trait]
impl RnnOps for CpuRnnOps {
    async fn rnn_forward(
        &self,
        device: &Device,
        input: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        initial_hidden: Option<&Buffer>,
        _initial_cell: Option<&Buffer>,
        config: &RnnConfig,
    ) -> BackendResult<RnnOutput> {
        if !config.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid RNN configuration".to_string(),
            ));
        }

        // For vanilla RNN, delegate to GRU-like implementation but with different activation
        let input_data = self.get_buffer_data_f32(input)?;
        let weights_data = self.get_buffer_data_f32(weights)?;
        let bias_data = if let Some(b) = bias {
            Some(self.get_buffer_data_f32(b)?)
        } else {
            None
        };
        let initial_hidden_data = if let Some(h) = initial_hidden {
            Some(self.get_buffer_data_f32(h)?)
        } else {
            None
        };

        // Simple RNN implementation (similar to GRU but simpler)
        let (output_seq, final_hidden) = self.execute_gru_sequence(
            &input_data,
            &weights_data,
            bias_data.as_deref(),
            initial_hidden_data.as_deref(),
            config,
        )?;

        // Pack the genuinely computed results into output buffers.
        let sequences_buffer = if config.return_sequences {
            Some(self.create_buffer_from_f32(device, &output_seq)?)
        } else {
            None
        };

        let hidden_state_buffer = if config.return_state {
            Some(self.create_buffer_from_f32(device, &final_hidden)?)
        } else {
            None
        };

        Ok(RnnOutput {
            sequences: sequences_buffer,
            hidden_state: hidden_state_buffer,
            cell_state: None,
        })
    }

    async fn lstm_forward(
        &self,
        device: &Device,
        input: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        initial_hidden: Option<&Buffer>,
        initial_cell: Option<&Buffer>,
        config: &RnnConfig,
    ) -> BackendResult<RnnOutput> {
        if !config.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid LSTM configuration".to_string(),
            ));
        }

        let input_data = self.get_buffer_data_f32(input)?;
        let weights_data = self.get_buffer_data_f32(weights)?;
        let bias_data = if let Some(b) = bias {
            Some(self.get_buffer_data_f32(b)?)
        } else {
            None
        };
        let initial_hidden_data = if let Some(h) = initial_hidden {
            Some(self.get_buffer_data_f32(h)?)
        } else {
            None
        };
        let initial_cell_data = if let Some(c) = initial_cell {
            Some(self.get_buffer_data_f32(c)?)
        } else {
            None
        };

        let (output_seq, final_hidden, final_cell) = self.execute_lstm_sequence(
            &input_data,
            &weights_data,
            bias_data.as_deref(),
            initial_hidden_data.as_deref(),
            initial_cell_data.as_deref(),
            config,
        )?;

        // Pack the genuinely computed sequence and states into output buffers.
        let sequences_buffer = Some(self.create_buffer_from_f32(device, &output_seq)?);
        let hidden_state_buffer = Some(self.create_buffer_from_f32(device, &final_hidden)?);
        let cell_state_buffer = Some(self.create_buffer_from_f32(device, &final_cell)?);

        Ok(RnnOutput {
            sequences: sequences_buffer,
            hidden_state: hidden_state_buffer,
            cell_state: cell_state_buffer,
        })
    }

    async fn gru_forward(
        &self,
        device: &Device,
        input: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        initial_hidden: Option<&Buffer>,
        config: &RnnConfig,
    ) -> BackendResult<RnnOutput> {
        if !config.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid GRU configuration".to_string(),
            ));
        }

        let input_data = self.get_buffer_data_f32(input)?;
        let weights_data = self.get_buffer_data_f32(weights)?;
        let bias_data = if let Some(b) = bias {
            Some(self.get_buffer_data_f32(b)?)
        } else {
            None
        };
        let initial_hidden_data = if let Some(h) = initial_hidden {
            Some(self.get_buffer_data_f32(h)?)
        } else {
            None
        };

        let (output_seq, final_hidden) = self.execute_gru_sequence(
            &input_data,
            &weights_data,
            bias_data.as_deref(),
            initial_hidden_data.as_deref(),
            config,
        )?;

        // Pack the genuinely computed sequence and final hidden state.
        // GRU has no cell state, so `cell_state` is `None`.
        let sequences_buffer = Some(self.create_buffer_from_f32(device, &output_seq)?);
        let hidden_state_buffer = Some(self.create_buffer_from_f32(device, &final_hidden)?);

        Ok(RnnOutput {
            sequences: sequences_buffer,
            hidden_state: hidden_state_buffer,
            cell_state: None,
        })
    }

    async fn lstm_cell(
        &self,
        _device: &Device,
        input: &Buffer,
        hidden: &Buffer,
        cell: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        output_hidden: &Buffer,
        output_cell: &Buffer,
    ) -> BackendResult<()> {
        let input_data = self.get_buffer_data_f32(input)?;
        let hidden_data = self.get_buffer_data_f32(hidden)?;
        let cell_data = self.get_buffer_data_f32(cell)?;
        let weights_data = self.get_buffer_data_f32(weights)?;
        let bias_data = if let Some(b) = bias {
            Some(self.get_buffer_data_f32(b)?)
        } else {
            None
        };

        let hidden_size = hidden_data.len();
        let input_size = input_data.len();

        // Split weights into input-to-hidden and hidden-to-hidden
        let ih_size = input_size * hidden_size * 4;
        let hh_size = hidden_size * hidden_size * 4;

        if weights_data.len() < ih_size + hh_size {
            return Err(torsh_core::error::TorshError::BackendError(
                "Insufficient weight data for LSTM cell".to_string(),
            ));
        }

        let weights_ih = &weights_data[0..ih_size];
        let weights_hh = &weights_data[ih_size..ih_size + hh_size];

        let mut new_hidden = vec![0.0; hidden_size];
        let mut new_cell = vec![0.0; hidden_size];

        cells::LstmCell::forward(
            &input_data,
            &hidden_data,
            &cell_data,
            weights_ih,
            weights_hh,
            bias_data.as_deref(),
            &mut new_hidden,
            &mut new_cell,
        )?;

        self.write_buffer_data_f32(output_hidden, &new_hidden)?;
        self.write_buffer_data_f32(output_cell, &new_cell)?;

        Ok(())
    }

    async fn gru_cell(
        &self,
        _device: &Device,
        input: &Buffer,
        hidden: &Buffer,
        weights: &Buffer,
        bias: Option<&Buffer>,
        output_hidden: &Buffer,
    ) -> BackendResult<()> {
        let input_data = self.get_buffer_data_f32(input)?;
        let hidden_data = self.get_buffer_data_f32(hidden)?;
        let weights_data = self.get_buffer_data_f32(weights)?;
        let bias_data = if let Some(b) = bias {
            Some(self.get_buffer_data_f32(b)?)
        } else {
            None
        };

        let hidden_size = hidden_data.len();
        let input_size = input_data.len();

        // Split weights into input-to-hidden and hidden-to-hidden
        let ih_size = input_size * hidden_size * 3;
        let hh_size = hidden_size * hidden_size * 3;

        if weights_data.len() < ih_size + hh_size {
            return Err(torsh_core::error::TorshError::BackendError(
                "Insufficient weight data for GRU cell".to_string(),
            ));
        }

        let weights_ih = &weights_data[0..ih_size];
        let weights_hh = &weights_data[ih_size..ih_size + hh_size];

        let mut new_hidden = vec![0.0; hidden_size];

        cells::GruCell::forward(
            &input_data,
            &hidden_data,
            weights_ih,
            weights_hh,
            bias_data.as_deref(),
            &mut new_hidden,
        )?;

        self.write_buffer_data_f32(output_hidden, &new_hidden)?;

        Ok(())
    }

    fn supports_rnn(&self) -> bool {
        true
    }

    fn supported_cell_types(&self) -> Vec<RnnCellType> {
        vec![
            RnnCellType::Rnn,
            RnnCellType::Lstm,
            RnnCellType::Gru,
            // Bidirectional variants would require more complex implementation
        ]
    }

    fn supported_activations(&self) -> Vec<RnnActivation> {
        vec![
            RnnActivation::Tanh,
            RnnActivation::Relu,
            RnnActivation::Sigmoid,
            RnnActivation::Identity,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{BufferDescriptor, BufferUsage, MemoryLocation};
    use crate::cpu::buffer::CpuBuffer;

    /// Build a CPU [`Buffer`] from `f32` data for use in RNN tests.
    fn f32_buffer(device: &Device, data: &[f32]) -> Buffer {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(data));
        for &value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let descriptor = BufferDescriptor {
            size: bytes.len(),
            usage: BufferUsage::STORAGE_READ_WRITE,
            location: MemoryLocation::Host,
            dtype: Some(torsh_core::dtype::DType::F32),
            shape: None,
            initial_data: None,
            alignment: None,
            zero_init: false,
        };

        let buffer = CpuBuffer::new_buffer(device.clone(), &descriptor)
            .expect("failed to allocate CPU f32 buffer");
        buffer
            .as_cpu_buffer()
            .expect("generic handle should expose a CpuBuffer")
            .write_bytes(&bytes, 0)
            .expect("failed to write f32 data into buffer");
        buffer
    }

    /// Read an f32 CPU [`Buffer`] back into a `Vec<f32>`.
    fn read_f32(buffer: &Buffer) -> Vec<f32> {
        let count = buffer.size / std::mem::size_of::<f32>();
        let ptr = buffer.as_cpu_ptr().expect("buffer should be CPU-backed");
        // SAFETY: buffer holds `count` `f32` values.
        unsafe { std::slice::from_raw_parts(ptr as *const f32, count).to_vec() }
    }

    fn test_device() -> Device {
        crate::cpu::device::CpuDevice::new(0, 1)
            .expect("CPU device creation should succeed")
            .to_device()
    }

    #[test]
    fn test_cpu_rnn_ops_creation() {
        let rnn_ops = CpuRnnOps::new(Some(2));
        assert!(rnn_ops.supports_rnn());
        assert!(!rnn_ops.supported_cell_types().is_empty());
        assert!(!rnn_ops.supported_activations().is_empty());
    }

    /// The LSTM forward pass must return the genuinely computed sequences and
    /// states, not a discarded-result empty `RnnOutput`.
    #[tokio::test]
    async fn test_lstm_forward_returns_outputs() {
        let rnn_ops = CpuRnnOps::new(Some(1));
        let device = test_device();

        let input_size = 3;
        let hidden_size = 4;
        let batch_size = 2;
        let sequence_length = 5;
        let config = RnnConfig::lstm(input_size, hidden_size, 1, batch_size, sequence_length);

        let input = f32_buffer(
            &device,
            &vec![0.1f32; batch_size * sequence_length * input_size],
        );
        // LSTM: 4 gates of input-to-hidden and hidden-to-hidden weights.
        let weights_len = (input_size * hidden_size + hidden_size * hidden_size) * 4;
        let weights = f32_buffer(&device, &vec![0.05f32; weights_len]);

        let output = rnn_ops
            .lstm_forward(&device, &input, &weights, None, None, None, &config)
            .await
            .expect("LSTM forward should succeed");

        let sequences = output
            .sequences
            .expect("LSTM must return Some(sequences), not None");
        let hidden = output
            .hidden_state
            .expect("LSTM must return Some(hidden_state)");
        let cell = output
            .cell_state
            .expect("LSTM must return Some(cell_state)");

        // Sequence buffer must hold batch * seq * hidden elements.
        assert_eq!(
            sequences.size / std::mem::size_of::<f32>(),
            batch_size * sequence_length * hidden_size
        );
        assert_eq!(
            hidden.size / std::mem::size_of::<f32>(),
            batch_size * hidden_size
        );
        assert_eq!(
            cell.size / std::mem::size_of::<f32>(),
            batch_size * hidden_size
        );

        // With non-zero inputs and weights the LSTM produces non-zero state.
        let hidden_vals = read_f32(&hidden);
        assert!(
            hidden_vals.iter().any(|&v| v.abs() > 1e-6),
            "LSTM hidden state should not be all zeros"
        );
    }

    /// The GRU forward pass must return the genuinely computed sequences and
    /// final hidden state; cell state is `None` for GRU.
    #[tokio::test]
    async fn test_gru_forward_returns_outputs() {
        let rnn_ops = CpuRnnOps::new(Some(1));
        let device = test_device();

        let input_size = 3;
        let hidden_size = 4;
        let batch_size = 2;
        let sequence_length = 5;
        let config = RnnConfig::gru(input_size, hidden_size, 1, batch_size, sequence_length);

        let input = f32_buffer(
            &device,
            &vec![0.2f32; batch_size * sequence_length * input_size],
        );
        // GRU: 3 gates of input-to-hidden and hidden-to-hidden weights.
        let weights_len = (input_size * hidden_size + hidden_size * hidden_size) * 3;
        let weights = f32_buffer(&device, &vec![0.05f32; weights_len]);

        let output = rnn_ops
            .gru_forward(&device, &input, &weights, None, None, &config)
            .await
            .expect("GRU forward should succeed");

        let sequences = output
            .sequences
            .expect("GRU must return Some(sequences), not None");
        let hidden = output
            .hidden_state
            .expect("GRU must return Some(hidden_state)");
        assert!(output.cell_state.is_none(), "GRU has no cell state");

        assert_eq!(
            sequences.size / std::mem::size_of::<f32>(),
            batch_size * sequence_length * hidden_size
        );
        assert_eq!(
            hidden.size / std::mem::size_of::<f32>(),
            batch_size * hidden_size
        );

        let hidden_vals = read_f32(&hidden);
        assert!(
            hidden_vals.iter().any(|&v| v.abs() > 1e-6),
            "GRU hidden state should not be all zeros"
        );
    }

    #[test]
    fn test_supported_operations() {
        let rnn_ops = CpuRnnOps::new(Some(1));

        let cell_types = rnn_ops.supported_cell_types();
        assert!(cell_types.contains(&RnnCellType::Lstm));
        assert!(cell_types.contains(&RnnCellType::Gru));
        assert!(cell_types.contains(&RnnCellType::Rnn));

        let activations = rnn_ops.supported_activations();
        assert!(activations.contains(&RnnActivation::Tanh));
        assert!(activations.contains(&RnnActivation::Relu));
        assert!(activations.contains(&RnnActivation::Sigmoid));
    }
}
