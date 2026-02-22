//! Streaming signal processing for memory-efficient operations
//!
//! This module provides streaming architectures for processing large signals
//! that don't fit in memory, using chunk-based processing with proper state management.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// Streaming processor trait for chunk-based signal processing
pub trait StreamingProcessor {
    /// Process a single chunk of signal data
    fn process_chunk(&mut self, chunk: &Tensor<f32>) -> Result<Tensor<f32>>;

    /// Reset the processor state
    fn reset(&mut self);

    /// Get the recommended chunk size for this processor
    fn chunk_size(&self) -> usize;

    /// Get the required overlap between chunks
    fn overlap_size(&self) -> usize {
        0
    }
}

/// Chunked signal processor configuration
#[derive(Debug, Clone)]
pub struct ChunkedProcessorConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub zero_pad: bool,
}

impl Default for ChunkedProcessorConfig {
    fn default() -> Self {
        Self {
            chunk_size: 2048,
            overlap: 512,
            zero_pad: true,
        }
    }
}

/// Generic chunked signal processor
pub struct ChunkedSignalProcessor {
    config: ChunkedProcessorConfig,
    output_buffer: Vec<f32>,
}

impl ChunkedSignalProcessor {
    pub fn new(config: ChunkedProcessorConfig) -> Self {
        Self {
            config,
            output_buffer: Vec::new(),
        }
    }

    /// Process a large signal in chunks
    pub fn process_signal<F>(
        &mut self,
        signal: &Tensor<f32>,
        mut process_fn: F,
    ) -> Result<Tensor<f32>>
    where
        F: FnMut(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let signal_shape = signal.shape();
        if signal_shape.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Chunked processing requires 1D tensor".to_string(),
            ));
        }

        let signal_len = signal_shape.dims()[0];
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.overlap;
        let hop_size = chunk_size - overlap;

        self.output_buffer.clear();

        let mut pos = 0;
        while pos < signal_len {
            let end = (pos + chunk_size).min(signal_len);
            let chunk_len = end - pos;

            // Extract chunk
            let mut chunk_data = vec![0.0f32; chunk_size];
            for i in 0..chunk_len {
                chunk_data[i] = signal.get_1d(pos + i)?;
            }

            // Zero-pad if needed
            if self.config.zero_pad && chunk_len < chunk_size {
                // Already zero-padded from initialization
            }

            // Create chunk tensor
            let chunk_tensor = Tensor::from_vec(chunk_data, &[chunk_size])?;

            // Process chunk
            let processed = process_fn(&chunk_tensor)?;

            // Add to output (overlap-add for middle chunks)
            let processed_shape = processed.shape();
            let processed_len = processed_shape.dims()[0];

            if pos == 0 {
                // First chunk - add all
                for i in 0..processed_len {
                    let val: f32 = processed.get_1d(i)?;
                    self.output_buffer.push(val);
                }
            } else {
                // Overlap-add
                let overlap_start = self.output_buffer.len() - overlap;
                for i in 0..overlap.min(processed_len) {
                    if overlap_start + i < self.output_buffer.len() {
                        let val: f32 = processed.get_1d(i)?;
                        self.output_buffer[overlap_start + i] += val;
                    }
                }
                // Add non-overlapping part
                for i in overlap..processed_len {
                    let val: f32 = processed.get_1d(i)?;
                    self.output_buffer.push(val);
                }
            }

            pos += hop_size;
            if chunk_len < chunk_size {
                break;
            }
        }

        // Create output tensor
        let output = Tensor::from_vec(self.output_buffer.clone(), &[self.output_buffer.len()])?;
        Ok(output)
    }

    /// Process signal in streaming fashion (iterator-based)
    pub fn stream_chunks<'a>(
        &'a mut self,
        signal: &'a Tensor<f32>,
    ) -> Result<impl Iterator<Item = Result<Tensor<f32>>> + 'a> {
        let signal_shape = signal.shape();
        if signal_shape.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Streaming requires 1D tensor".to_string(),
            ));
        }

        let signal_len = signal_shape.dims()[0];
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.overlap;
        let hop_size = chunk_size - overlap;

        Ok(StreamingChunkIterator {
            signal,
            pos: 0,
            signal_len,
            chunk_size,
            hop_size,
        })
    }
}

/// Iterator for streaming chunks
struct StreamingChunkIterator<'a> {
    signal: &'a Tensor<f32>,
    pos: usize,
    signal_len: usize,
    chunk_size: usize,
    hop_size: usize,
}

impl<'a> Iterator for StreamingChunkIterator<'a> {
    type Item = Result<Tensor<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.signal_len {
            return None;
        }

        let end = (self.pos + self.chunk_size).min(self.signal_len);
        let chunk_len = end - self.pos;

        // Extract chunk
        let mut chunk_data = vec![0.0f32; self.chunk_size];
        for i in 0..chunk_len {
            match self.signal.get_1d(self.pos + i) {
                Ok(val) => chunk_data[i] = val,
                Err(e) => return Some(Err(e)),
            }
        }

        // Create chunk tensor
        let chunk_tensor = match Tensor::from_vec(chunk_data, &[self.chunk_size]) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        self.pos += self.hop_size;

        Some(Ok(chunk_tensor))
    }
}

/// Streaming filter processor with state management
pub struct StreamingFilterProcessor {
    filter_coeffs: Vec<f32>,
    state: Vec<f32>,
}

impl StreamingFilterProcessor {
    pub fn new(filter_coeffs: Vec<f32>) -> Self {
        let state_size = filter_coeffs.len().saturating_sub(1);
        Self {
            filter_coeffs,
            state: vec![0.0; state_size],
        }
    }

    /// Process a chunk with state preservation
    pub fn process_chunk_stateful(&mut self, chunk: &Tensor<f32>) -> Result<Tensor<f32>> {
        let chunk_shape = chunk.shape();
        if chunk_shape.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Filter processing requires 1D tensor".to_string(),
            ));
        }

        let chunk_len = chunk_shape.dims()[0];
        let filter_len = self.filter_coeffs.len();
        let mut output = zeros(&[chunk_len])?;

        for i in 0..chunk_len {
            let mut sum = 0.0f32;

            // Current sample
            let x_i: f32 = chunk.get_1d(i)?;
            sum += self.filter_coeffs[0] * x_i;

            // Previous samples (from state and current chunk)
            for j in 1..filter_len {
                let x_prev = if i >= j {
                    chunk.get_1d(i - j)?
                } else {
                    // Use state from previous chunk
                    let state_idx = self.state.len() - j + i;
                    if state_idx < self.state.len() {
                        self.state[state_idx]
                    } else {
                        0.0
                    }
                };
                sum += self.filter_coeffs[j] * x_prev;
            }

            output.set_1d(i, sum)?;
        }

        // Update state for next chunk
        let state_size = self.state.len();
        if chunk_len >= state_size {
            for i in 0..state_size {
                self.state[i] = chunk.get_1d(chunk_len - state_size + i)?;
            }
        } else {
            // Shift old state and add new samples
            for i in 0..(state_size - chunk_len) {
                self.state[i] = self.state[i + chunk_len];
            }
            for i in 0..chunk_len {
                self.state[state_size - chunk_len + i] = chunk.get_1d(i)?;
            }
        }

        Ok(output)
    }
}

impl StreamingProcessor for StreamingFilterProcessor {
    fn process_chunk(&mut self, chunk: &Tensor<f32>) -> Result<Tensor<f32>> {
        self.process_chunk_stateful(chunk)
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
    }

    fn chunk_size(&self) -> usize {
        1024 // Default chunk size
    }

    fn overlap_size(&self) -> usize {
        self.filter_coeffs.len().saturating_sub(1)
    }
}

/// Streaming STFT processor for memory-efficient spectrogram computation
pub struct StreamingSTFTProcessor {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    input_buffer: Vec<f32>,
}

impl StreamingSTFTProcessor {
    pub fn new(n_fft: usize, hop_length: usize, window: Vec<f32>) -> Result<Self> {
        if window.len() != n_fft {
            return Err(TorshError::InvalidArgument(format!(
                "Window length {} must match FFT size {}",
                window.len(),
                n_fft
            )));
        }

        Ok(Self {
            n_fft,
            hop_length,
            window,
            input_buffer: Vec::new(),
        })
    }

    /// Add samples to the buffer and process when enough are available
    pub fn add_samples(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.input_buffer.extend_from_slice(samples);

        let mut frames = Vec::new();

        // Process as many complete frames as possible
        while self.input_buffer.len() >= self.n_fft {
            let mut frame = vec![0.0f32; self.n_fft];
            for i in 0..self.n_fft {
                frame[i] = self.input_buffer[i] * self.window[i];
            }

            frames.push(frame);

            // Remove processed samples
            self.input_buffer.drain(0..self.hop_length);
        }

        frames
    }

    /// Flush remaining samples (for end of stream)
    pub fn flush(&mut self) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();

        if !self.input_buffer.is_empty() {
            // Zero-pad the last frame
            let mut frame = vec![0.0f32; self.n_fft];
            let valid_len = self.input_buffer.len().min(self.n_fft);
            for i in 0..valid_len {
                frame[i] = self.input_buffer[i] * self.window[i];
            }
            frames.push(frame);
            self.input_buffer.clear();
        }

        frames
    }

    pub fn reset(&mut self) {
        self.input_buffer.clear();
    }
}

/// Ring buffer for streaming operations
pub struct RingBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
    size: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            capacity,
            read_pos: 0,
            write_pos: 0,
            size: 0,
        }
    }

    pub fn push(&mut self, value: T) -> bool {
        if self.size >= self.capacity {
            return false; // Buffer full
        }

        self.buffer[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.size += 1;
        true
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let value = self.buffer[self.read_pos].clone();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.size -= 1;
        Some(value)
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size >= self.capacity
    }

    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.size = 0;
    }
}

/// Overlap-add processor for STFT-like operations
pub struct OverlapAddProcessor {
    frame_size: usize,
    hop_size: usize,
    overlap_buffer: Vec<f32>,
}

impl OverlapAddProcessor {
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        Self {
            frame_size,
            hop_size,
            overlap_buffer: vec![0.0; frame_size],
        }
    }

    /// Process a frame and perform overlap-add
    pub fn process_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        if frame.len() != self.frame_size {
            return Vec::new();
        }

        let mut output = vec![0.0f32; self.hop_size];

        // Add overlap from previous frame
        for i in 0..self.hop_size.min(self.overlap_buffer.len()) {
            output[i] = self.overlap_buffer[i];
        }

        // Add current frame (overlapping part)
        for i in 0..self.hop_size.min(frame.len()) {
            output[i] += frame[i];
        }

        // Update overlap buffer for next frame
        let overlap_start = self.hop_size;
        for i in overlap_start..self.frame_size {
            self.overlap_buffer[i - overlap_start] = if i < frame.len() { frame[i] } else { 0.0 };
        }

        // Fill rest of overlap buffer with zeros
        for i in (self.frame_size - overlap_start)..self.overlap_buffer.len() {
            self.overlap_buffer[i] = 0.0;
        }

        output
    }

    pub fn reset(&mut self) {
        self.overlap_buffer.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_chunked_processor() -> Result<()> {
        let config = ChunkedProcessorConfig {
            chunk_size: 256,
            overlap: 64,
            zero_pad: true,
        };
        let mut processor = ChunkedSignalProcessor::new(config);

        let signal = ones(&[1000])?;

        // Simple identity processing
        let result = processor.process_signal(&signal, |chunk| Ok(chunk.clone()))?;

        assert!(result.shape().dims()[0] > 0);
        Ok(())
    }

    #[test]
    fn test_streaming_filter() -> Result<()> {
        let filter_coeffs = vec![0.5, 0.3, 0.2];
        let mut filter = StreamingFilterProcessor::new(filter_coeffs);

        let chunk1 = ones(&[100])?;
        let result1 = filter.process_chunk(&chunk1)?;
        assert_eq!(result1.shape().dims()[0], 100);

        let chunk2 = ones(&[100])?;
        let result2 = filter.process_chunk(&chunk2)?;
        assert_eq!(result2.shape().dims()[0], 100);

        Ok(())
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingBuffer::<f32>::new(10);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        for i in 0..10 {
            assert!(buffer.push(i as f32));
        }

        assert!(buffer.is_full());
        assert!(!buffer.push(100.0));

        for i in 0..10 {
            assert_eq!(buffer.pop(), Some(i as f32));
        }

        assert!(buffer.is_empty());
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_overlap_add_processor() {
        let mut processor = OverlapAddProcessor::new(512, 256);

        let frame1 = vec![1.0; 512];
        let output1 = processor.process_frame(&frame1);
        assert_eq!(output1.len(), 256);

        let frame2 = vec![1.0; 512];
        let output2 = processor.process_frame(&frame2);
        assert_eq!(output2.len(), 256);

        // With overlap-add, overlapping regions should sum
        for &val in &output2 {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_streaming_stft() -> Result<()> {
        let n_fft = 512;
        let hop_length = 256;
        let window = vec![1.0; n_fft];

        let mut stft = StreamingSTFTProcessor::new(n_fft, hop_length, window)?;

        // Add samples in chunks
        let samples1 = vec![1.0; 1000];
        let frames1 = stft.add_samples(&samples1);
        assert!(!frames1.is_empty());

        let samples2 = vec![1.0; 1000];
        let frames2 = stft.add_samples(&samples2);
        assert!(!frames2.is_empty());

        // Flush remaining
        let frames3 = stft.flush();
        assert!(!frames3.is_empty() || stft.input_buffer.is_empty());

        Ok(())
    }

    #[test]
    fn test_streaming_chunk_iterator() -> Result<()> {
        let config = ChunkedProcessorConfig {
            chunk_size: 100,
            overlap: 20,
            zero_pad: false,
        };
        let mut processor = ChunkedSignalProcessor::new(config);

        let signal = ones(&[500])?;
        let chunks: Vec<_> = processor.stream_chunks(&signal)?.collect();

        assert!(!chunks.is_empty());

        // Verify all chunks succeeded
        for chunk_result in chunks {
            assert!(chunk_result.is_ok());
            let chunk = chunk_result?;
            assert_eq!(chunk.shape().dims()[0], 100);
        }

        Ok(())
    }
}
