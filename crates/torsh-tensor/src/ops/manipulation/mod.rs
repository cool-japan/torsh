//! Tensor manipulation operations for combining, splitting, and transforming tensors
//!
//! This module provides PyTorch-compatible tensor manipulation operations including:
//! - Concatenation: cat, stack
//! - Splitting: split, chunk
//! - Flipping: flip, fliplr, flipud, rot90
//! - Repeating: tile, repeat, repeat_interleave
//! - Rolling: roll
//! - Dimension manipulation: movedim, moveaxis, swapaxes, swapdims
//! - Shape transformation: unflatten
//! - Advanced indexing: take_along_dim

mod core_ops;
mod dim_ops;

#[cfg(test)]
mod tests;
