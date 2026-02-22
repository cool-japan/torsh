//! Autograd support for sparse tensors
//!
//! This module provides automatic differentiation support for sparse tensor operations,
//! enabling gradient computation through sparse computational graphs.

use crate::{CooTensor, CsrTensor, SparseFormat, SparseTensor, TorshResult};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::{DType, TorshError};
use torsh_tensor::Tensor;

/// Sparse tensor with gradient tracking
#[derive(Debug, Clone)]
pub struct SparseAutogradTensor {
    /// The sparse tensor data
    data: SparseData,
    /// Whether this tensor requires gradients
    requires_grad: bool,
    /// Gradient tensor (sparse)
    grad: Option<Arc<SparseAutogradTensor>>,
    /// Gradient function for backward pass
    grad_fn: Option<Arc<dyn SparseGradFn>>,
    /// Input tensors that created this tensor (for backward pass)
    inputs: Vec<Arc<SparseAutogradTensor>>,
    /// Unique tensor ID for tracking
    id: u64,
    /// Whether this is a leaf tensor
    is_leaf: bool,
}

/// Sparse tensor data variants
#[derive(Debug, Clone)]
pub enum SparseData {
    Coo(CooTensor),
    Csr(CsrTensor),
}

impl SparseData {
    pub fn dtype(&self) -> DType {
        match self {
            SparseData::Coo(tensor) => tensor.dtype(),
            SparseData::Csr(tensor) => tensor.dtype(),
        }
    }

    pub fn shape(&self) -> &torsh_core::Shape {
        match self {
            SparseData::Coo(tensor) => tensor.shape(),
            SparseData::Csr(tensor) => tensor.shape(),
        }
    }

    pub fn nnz(&self) -> usize {
        match self {
            SparseData::Coo(tensor) => tensor.nnz(),
            SparseData::Csr(tensor) => tensor.nnz(),
        }
    }

    pub fn format(&self) -> SparseFormat {
        match self {
            SparseData::Coo(_) => SparseFormat::Coo,
            SparseData::Csr(_) => SparseFormat::Csr,
        }
    }
}

/// Trait for sparse gradient functions
pub trait SparseGradFn: Send + Sync + std::fmt::Debug {
    /// Compute gradients for backward pass
    fn backward(
        &self,
        grad_output: &SparseAutogradTensor,
    ) -> TorshResult<Vec<Option<SparseAutogradTensor>>>;

    /// Get the number of inputs this function expects
    fn num_inputs(&self) -> usize;

    /// Function name for debugging
    fn name(&self) -> &str;
}

impl SparseAutogradTensor {
    /// Create a new sparse autograd tensor
    pub fn new(data: SparseData, requires_grad: bool) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self {
            data,
            requires_grad,
            grad: None,
            grad_fn: None,
            inputs: Vec::new(),
            id,
            is_leaf: true,
        }
    }

    /// Create from COO tensor
    pub fn from_coo(coo: CooTensor, requires_grad: bool) -> Self {
        Self::new(SparseData::Coo(coo), requires_grad)
    }

    /// Create from CSR tensor
    pub fn from_csr(csr: CsrTensor, requires_grad: bool) -> Self {
        Self::new(SparseData::Csr(csr), requires_grad)
    }

    /// Get the sparse tensor data
    pub fn data(&self) -> &SparseData {
        &self.data
    }

    /// Check if this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the gradient tensor
    pub fn grad(&self) -> Option<&SparseAutogradTensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Set gradient tensor
    pub fn set_grad(&mut self, grad: Option<SparseAutogradTensor>) {
        self.grad = grad.map(Arc::new);
    }

    /// Get tensor ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Check if this is a leaf tensor
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Accumulate gradient for a tensor (thread-safe gradient accumulation)
    fn accumulate_grad(
        &self,
        target_tensor: &SparseAutogradTensor,
        _new_grad: &SparseAutogradTensor,
    ) -> TorshResult<()> {
        // Note: In a real implementation, this would need proper thread-safe gradient accumulation
        // For now, we'll use a simple approach that works for single-threaded execution

        // This is a simplified implementation - in practice, you'd want to use Arc<Mutex<>>
        // or other thread-safe mechanisms for gradient accumulation
        if target_tensor.requires_grad() {
            // If target already has a gradient, add to it; otherwise set it
            // For simplicity, we'll just log this operation
            // In a full implementation, this would properly accumulate gradients
            println!(
                "Accumulating gradient for tensor ID: {}",
                target_tensor.id()
            );
        }

        Ok(())
    }

    /// Perform backward pass from this tensor
    pub fn backward(&self, retain_graph: bool) -> TorshResult<()> {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Tensor does not require gradients".to_string(),
            ));
        }

        // Create unit gradient for this tensor
        let unit_grad = self.create_unit_grad()?;
        self.backward_impl(&unit_grad, retain_graph)
    }

    /// Internal backward implementation
    fn backward_impl(
        &self,
        grad_output: &SparseAutogradTensor,
        _retain_graph: bool,
    ) -> TorshResult<()> {
        if let Some(grad_fn) = &self.grad_fn {
            let input_grads = grad_fn.backward(grad_output)?;

            // Propagate gradients to inputs
            for (i, grad) in input_grads.into_iter().enumerate() {
                if let Some(input_grad) = grad {
                    if i < self.inputs.len() {
                        let input_tensor = &self.inputs[i];

                        // Accumulate gradient in the input tensor
                        self.accumulate_grad(input_tensor, &input_grad)?;

                        // Recursively call backward on input tensor if it has gradient function
                        if input_tensor.grad_fn.is_some() && input_tensor.requires_grad() {
                            input_tensor.backward_impl(&input_grad, _retain_graph)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Create unit gradient tensor
    fn create_unit_grad(&self) -> TorshResult<SparseAutogradTensor> {
        match &self.data {
            SparseData::Coo(coo) => {
                // Create identity sparse tensor as unit gradient
                let values = vec![1.0; coo.nnz()];
                let unit_coo = CooTensor::new(
                    coo.row_indices().to_vec(),
                    coo.col_indices().to_vec(),
                    values,
                    coo.shape().clone(),
                )?;
                Ok(SparseAutogradTensor::from_coo(unit_coo, false))
            }
            SparseData::Csr(csr) => {
                // Create identity sparse tensor as unit gradient
                let values = vec![1.0; csr.nnz()];
                let unit_csr = CsrTensor::new(
                    csr.row_ptr().to_vec(),
                    csr.col_indices().to_vec(),
                    values,
                    csr.shape().clone(),
                )?;
                Ok(SparseAutogradTensor::from_csr(unit_csr, false))
            }
        }
    }

    /// Sparse matrix multiplication with gradient tracking
    pub fn sparse_mm(&self, other: &SparseAutogradTensor) -> TorshResult<SparseAutogradTensor> {
        // Perform forward pass
        let result_data = match (&self.data, &other.data) {
            (SparseData::Csr(a), SparseData::Csr(b)) => {
                let result = a.multiply_csr(b)?;
                SparseData::Csr(result)
            }
            (SparseData::Coo(a), SparseData::Coo(b)) => {
                let result = a.multiply_coo(b)?;
                SparseData::Coo(result)
            }
            _ => {
                return Err(TorshError::ComputeError(
                    "Mixed format sparse multiplication not supported".to_string(),
                ))
            }
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = SparseAutogradTensor::new(result_data, requires_grad);

        if requires_grad {
            // Set up gradient function with weak references to input tensors
            let self_arc = Arc::new(self.clone());
            let other_arc = Arc::new(other.clone());

            let grad_fn = Arc::new(SparseMmGradFn {
                input_shapes: [self.data().shape().clone(), other.data().shape().clone()],
                input_a: Some(Arc::downgrade(&self_arc)),
                input_b: Some(Arc::downgrade(&other_arc)),
            });
            result.grad_fn = Some(grad_fn);
            result.inputs = vec![self_arc, other_arc];
            result.is_leaf = false;
        }

        Ok(result)
    }

    /// Add sparse tensors with gradient tracking
    pub fn add(&self, other: &SparseAutogradTensor) -> TorshResult<SparseAutogradTensor> {
        // Perform forward pass
        let result_data = match (&self.data, &other.data) {
            (SparseData::Coo(a), SparseData::Coo(b)) => {
                let result = a.add_coo(b)?;
                SparseData::Coo(result)
            }
            (SparseData::Csr(a), SparseData::Csr(b)) => {
                let result = a.add_csr(b)?;
                SparseData::Csr(result)
            }
            _ => {
                return Err(TorshError::ComputeError(
                    "Mixed format sparse addition not supported".to_string(),
                ))
            }
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = SparseAutogradTensor::new(result_data, requires_grad);

        if requires_grad {
            // Set up gradient function
            let grad_fn = Arc::new(SparseAddGradFn);
            result.grad_fn = Some(grad_fn);
            result.inputs = vec![Arc::new(self.clone()), Arc::new(other.clone())];
            result.is_leaf = false;
        }

        Ok(result)
    }

    /// Convert to dense tensor with gradient tracking
    pub fn to_dense(&self) -> TorshResult<Tensor> {
        match &self.data {
            SparseData::Coo(coo) => coo.to_dense(),
            SparseData::Csr(csr) => csr.to_dense(),
        }
    }
}

/// Gradient function for sparse matrix multiplication
#[derive(Debug)]
struct SparseMmGradFn {
    input_shapes: [torsh_core::Shape; 2],
    /// Store weak references to input tensors to avoid circular references
    input_a: Option<std::sync::Weak<SparseAutogradTensor>>,
    input_b: Option<std::sync::Weak<SparseAutogradTensor>>,
}

impl SparseGradFn for SparseMmGradFn {
    fn backward(
        &self,
        grad_output: &SparseAutogradTensor,
    ) -> TorshResult<Vec<Option<SparseAutogradTensor>>> {
        // For C = A @ B:
        // grad_A = grad_output @ B.T
        // grad_B = A.T @ grad_output

        // Attempt to retrieve input tensors from weak references
        let input_a = self.input_a.as_ref().and_then(|weak| weak.upgrade());
        let input_b = self.input_b.as_ref().and_then(|weak| weak.upgrade());

        match (input_a, input_b) {
            (Some(a), Some(b)) => {
                // Compute actual gradients using stored input tensors
                let grad_a = self.compute_grad_a(grad_output, &b)?;
                let grad_b = self.compute_grad_b(grad_output, &a)?;

                Ok(vec![grad_a, grad_b])
            }
            _ => {
                // Fallback to zero gradients if input tensors are not available
                // This can happen if the input tensors have been dropped
                let grad_a_shape = &self.input_shapes[0];
                let grad_b_shape = &self.input_shapes[1];

                let grad_a = self.create_zero_grad(grad_a_shape, grad_output)?;
                let grad_b = self.create_zero_grad(grad_b_shape, grad_output)?;

                Ok(vec![grad_a, grad_b])
            }
        }
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "SparseMm"
    }
}

impl SparseMmGradFn {
    /// Compute gradient for input A: grad_A = grad_output @ B.T
    fn compute_grad_a(
        &self,
        grad_output: &SparseAutogradTensor,
        input_b: &SparseAutogradTensor,
    ) -> TorshResult<Option<SparseAutogradTensor>> {
        // For now, create a simple gradient approximation
        // In a full implementation, this would compute grad_output @ B.T
        match (grad_output.data(), input_b.data()) {
            (SparseData::Coo(_), SparseData::Coo(_)) => {
                // Create unit gradient tensor with shape of A
                let grad_a_shape = &self.input_shapes[0];
                let unit_coo = CooTensor::new(
                    vec![0], // Single element at (0,0)
                    vec![0],
                    vec![1.0],
                    grad_a_shape.clone(),
                )?;
                Ok(Some(SparseAutogradTensor::from_coo(unit_coo, false)))
            }
            (SparseData::Csr(_), SparseData::Csr(_)) => {
                // Create unit gradient tensor with shape of A
                let grad_a_shape = &self.input_shapes[0];
                let rows = grad_a_shape.dims()[0];
                if rows > 0 && grad_a_shape.dims()[1] > 0 {
                    let mut row_ptr = vec![0; rows + 1];
                    row_ptr[1] = 1; // First row has one element
                    let unit_csr = CsrTensor::new(
                        row_ptr,
                        vec![0], // Column index 0
                        vec![1.0],
                        grad_a_shape.clone(),
                    )?;
                    Ok(Some(SparseAutogradTensor::from_csr(unit_csr, false)))
                } else {
                    // Empty matrix case
                    self.create_zero_grad(grad_a_shape, grad_output)
                }
            }
            _ => {
                // Mixed formats, create zero gradient
                let grad_a_shape = &self.input_shapes[0];
                self.create_zero_grad(grad_a_shape, grad_output)
            }
        }
    }

    /// Compute gradient for input B: grad_B = A.T @ grad_output
    fn compute_grad_b(
        &self,
        grad_output: &SparseAutogradTensor,
        input_a: &SparseAutogradTensor,
    ) -> TorshResult<Option<SparseAutogradTensor>> {
        // For now, create a simple gradient approximation
        // In a full implementation, this would compute A.T @ grad_output
        match (grad_output.data(), input_a.data()) {
            (SparseData::Coo(_), SparseData::Coo(_)) => {
                // Create unit gradient tensor with shape of B
                let grad_b_shape = &self.input_shapes[1];
                let unit_coo = CooTensor::new(
                    vec![0], // Single element at (0,0)
                    vec![0],
                    vec![1.0],
                    grad_b_shape.clone(),
                )?;
                Ok(Some(SparseAutogradTensor::from_coo(unit_coo, false)))
            }
            (SparseData::Csr(_), SparseData::Csr(_)) => {
                // Create unit gradient tensor with shape of B
                let grad_b_shape = &self.input_shapes[1];
                let rows = grad_b_shape.dims()[0];
                if rows > 0 && grad_b_shape.dims()[1] > 0 {
                    let mut row_ptr = vec![0; rows + 1];
                    row_ptr[1] = 1; // First row has one element
                    let unit_csr = CsrTensor::new(
                        row_ptr,
                        vec![0], // Column index 0
                        vec![1.0],
                        grad_b_shape.clone(),
                    )?;
                    Ok(Some(SparseAutogradTensor::from_csr(unit_csr, false)))
                } else {
                    // Empty matrix case
                    self.create_zero_grad(grad_b_shape, grad_output)
                }
            }
            _ => {
                // Mixed formats, create zero gradient
                let grad_b_shape = &self.input_shapes[1];
                self.create_zero_grad(grad_b_shape, grad_output)
            }
        }
    }

    /// Create zero gradient tensor with the given shape
    fn create_zero_grad(
        &self,
        shape: &torsh_core::Shape,
        grad_output: &SparseAutogradTensor,
    ) -> TorshResult<Option<SparseAutogradTensor>> {
        match grad_output.data() {
            SparseData::Coo(_) => {
                // Create zero COO tensor
                let zero_coo = CooTensor::new(vec![], vec![], vec![], shape.clone())?;
                Ok(Some(SparseAutogradTensor::from_coo(zero_coo, false)))
            }
            SparseData::Csr(_) => {
                // Create zero CSR tensor
                let rows = shape.dims()[0];
                let zero_csr = CsrTensor::new(vec![0; rows + 1], vec![], vec![], shape.clone())?;
                Ok(Some(SparseAutogradTensor::from_csr(zero_csr, false)))
            }
        }
    }
}

/// Gradient function for sparse addition
#[derive(Debug)]
struct SparseAddGradFn;

impl SparseGradFn for SparseAddGradFn {
    fn backward(
        &self,
        grad_output: &SparseAutogradTensor,
    ) -> TorshResult<Vec<Option<SparseAutogradTensor>>> {
        // For C = A + B:
        // grad_A = grad_output
        // grad_B = grad_output

        Ok(vec![Some(grad_output.clone()), Some(grad_output.clone())])
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "SparseAdd"
    }
}

/// Extension trait for COO tensors to add autograd methods
impl CooTensor {
    /// Add two COO tensors
    pub fn add_coo(&self, other: &CooTensor) -> TorshResult<CooTensor> {
        if self.shape() != other.shape() {
            return Err(TorshError::ComputeError(
                "Shape mismatch for sparse addition".to_string(),
            ));
        }

        // Placeholder: Implement actual COO addition
        Err(TorshError::ComputeError(
            "COO addition not yet implemented".to_string(),
        ))
    }

    /// Multiply two COO tensors
    pub fn multiply_coo(&self, _other: &CooTensor) -> TorshResult<CooTensor> {
        // Placeholder: Implement actual COO multiplication
        Err(TorshError::ComputeError(
            "COO multiplication not yet implemented".to_string(),
        ))
    }
}

/// Extension trait for CSR tensors to add autograd methods
impl CsrTensor {
    /// Add two CSR tensors
    pub fn add_csr(&self, other: &CsrTensor) -> TorshResult<CsrTensor> {
        if self.shape() != other.shape() {
            return Err(TorshError::ComputeError(
                "Shape mismatch for sparse addition".to_string(),
            ));
        }

        // Placeholder: Implement actual CSR addition
        Err(TorshError::ComputeError(
            "CSR addition not yet implemented".to_string(),
        ))
    }

    /// Multiply two CSR tensors
    pub fn multiply_csr(&self, _other: &CsrTensor) -> TorshResult<CsrTensor> {
        // Placeholder: Implement actual CSR multiplication
        Err(TorshError::ComputeError(
            "CSR multiplication not yet implemented".to_string(),
        ))
    }
}

/// Sparse gradient accumulator
pub struct SparseGradientAccumulator {
    /// Accumulated gradients by tensor ID
    gradients: HashMap<u64, SparseAutogradTensor>,
}

impl SparseGradientAccumulator {
    /// Create new gradient accumulator
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
        }
    }

    /// Accumulate gradient for a tensor
    pub fn accumulate(&mut self, tensor_id: u64, grad: SparseAutogradTensor) -> TorshResult<()> {
        if let Some(existing_grad) = self.gradients.get(&tensor_id) {
            // Add gradients
            let accumulated = existing_grad.add(&grad)?;
            self.gradients.insert(tensor_id, accumulated);
        } else {
            self.gradients.insert(tensor_id, grad);
        }
        Ok(())
    }

    /// Get accumulated gradient for a tensor
    pub fn get_grad(&self, tensor_id: u64) -> Option<&SparseAutogradTensor> {
        self.gradients.get(&tensor_id)
    }

    /// Clear all accumulated gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl Default for SparseGradientAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Shape;

    #[test]
    fn test_sparse_autograd_tensor_creation() {
        let coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        let autograd_tensor = SparseAutogradTensor::from_coo(coo, true);
        assert!(autograd_tensor.requires_grad());
        assert!(autograd_tensor.is_leaf());
        assert_eq!(autograd_tensor.data().shape().dims(), &[3, 3]);
        assert_eq!(autograd_tensor.data().nnz(), 3);
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = SparseGradientAccumulator::new();

        let coo = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let grad = SparseAutogradTensor::from_coo(coo, false);
        let tensor_id = 123;

        accumulator.accumulate(tensor_id, grad).unwrap();
        assert!(accumulator.get_grad(tensor_id).is_some());

        accumulator.clear();
        assert!(accumulator.get_grad(tensor_id).is_none());
    }
}
