use crate::context::AutogradContext;
use std::collections::HashMap;
use torsh_core::Result;
use torsh_tensor::Tensor;

pub struct DifferentiableProgramming {
    #[allow(dead_code)]
    context: AutogradContext,
    variables: HashMap<String, Tensor>,
    #[allow(dead_code)]
    gradients: HashMap<String, Tensor>,
}

impl DifferentiableProgramming {
    pub fn new() -> Self {
        Self {
            context: AutogradContext::new(),
            variables: HashMap::new(),
            gradients: HashMap::new(),
        }
    }

    pub fn set_variable(&mut self, name: &str, tensor: Tensor) {
        self.variables.insert(name.to_string(), tensor);
    }

    pub fn get_variable(&self, name: &str) -> Option<&Tensor> {
        self.variables.get(name)
    }

    pub fn differentiable_if_else(
        &mut self,
        condition: &Tensor,
        true_branch: impl Fn(&mut Self) -> Result<Tensor>,
        false_branch: impl Fn(&mut Self) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let true_result = true_branch(self)?;
        let false_result = false_branch(self)?;

        let soft_condition = self.sigmoid_approximation(condition)?;
        let inverted_condition = self.ones_like(&soft_condition)?.sub(&soft_condition)?;

        // For vectorized if-else, create tensors with condition shape filled with scalar values
        let true_value = true_result.item()?;
        let false_value = false_result.item()?;

        let true_broadcasted =
            Tensor::ones(soft_condition.shape().dims(), soft_condition.device())?
                .mul_scalar(true_value)?;
        let false_broadcasted =
            Tensor::ones(soft_condition.shape().dims(), soft_condition.device())?
                .mul_scalar(false_value)?;

        let weighted_true = true_broadcasted.mul(&soft_condition)?;
        let weighted_false = false_broadcasted.mul(&inverted_condition)?;

        Ok(weighted_true.add(&weighted_false)?)
    }

    pub fn differentiable_for_loop(
        &mut self,
        init_value: &Tensor,
        iterations: usize,
        body: impl Fn(&mut Self, &Tensor, usize) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let mut current = init_value.clone();

        for i in 0..iterations {
            current = body(self, &current, i)?;
        }

        Ok(current)
    }

    pub fn differentiable_while_loop(
        &mut self,
        init_value: &Tensor,
        condition: impl Fn(&Tensor) -> Result<bool>,
        body: impl Fn(&mut Self, &Tensor) -> Result<Tensor>,
        max_iterations: usize,
    ) -> Result<Tensor> {
        let mut current = init_value.clone();
        let mut iterations = 0;

        while iterations < max_iterations {
            if !condition(&current)? {
                break;
            }
            current = body(self, &current)?;
            iterations += 1;
        }

        Ok(current)
    }

    pub fn differentiable_scan(
        &mut self,
        init_state: &Tensor,
        inputs: &[Tensor],
        fn_step: impl Fn(&mut Self, &Tensor, &Tensor) -> Result<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut state = init_state.clone();
        let mut outputs = Vec::new();

        for input in inputs {
            let (new_state, output) = fn_step(self, &state, input)?;
            state = new_state;
            outputs.push(output);
        }

        Ok((state, outputs))
    }

    pub fn differentiable_fold(
        &mut self,
        init_value: &Tensor,
        inputs: &[Tensor],
        fn_fold: impl Fn(&mut Self, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let mut accumulator = init_value.clone();

        for input in inputs {
            accumulator = fn_fold(self, &accumulator, input)?;
        }

        Ok(accumulator)
    }

    pub fn differentiable_reduce(
        &mut self,
        inputs: &[Tensor],
        fn_reduce: impl Fn(&mut Self, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        if inputs.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Empty input array".to_string(),
            ));
        }

        let mut result = inputs[0].clone();

        for input in inputs.iter().skip(1) {
            result = fn_reduce(self, &result, input)?;
        }

        Ok(result)
    }

    pub fn differentiable_map(
        &mut self,
        inputs: &[Tensor],
        fn_map: impl Fn(&mut Self, &Tensor) -> Result<Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::new();

        for input in inputs {
            let output = fn_map(self, input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    pub fn differentiable_filter(
        &mut self,
        inputs: &[Tensor],
        predicate: impl Fn(&mut Self, &Tensor) -> Result<Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut filtered = Vec::new();

        for input in inputs {
            let pred_result = predicate(self, input)?;
            let weight = self.sigmoid_approximation(&pred_result)?;

            // Extract scalar value and use mul_scalar for proper broadcasting
            let weight_value = weight.item()?;
            let weighted_input = input.mul_scalar(weight_value)?;
            filtered.push(weighted_input);
        }

        Ok(filtered)
    }

    pub fn differentiable_switch(
        &mut self,
        selector: &Tensor,
        cases: &[impl Fn(&mut Self) -> Result<Tensor>],
    ) -> Result<Tensor> {
        if cases.is_empty() {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Empty cases array".to_string(),
            ));
        }

        let softmax_weights = self.softmax_approximation(selector)?;

        let mut result = self.zeros_like(selector)?;

        for (i, case) in cases.iter().enumerate() {
            let case_result = case(self)?;
            let weight = self.get_scalar_at(&softmax_weights, i)?;
            let weighted_result = case_result.mul(&weight)?;
            result = result.add(&weighted_result)?;
        }

        Ok(result)
    }

    pub fn differentiable_recursion(
        &mut self,
        init_value: &Tensor,
        max_depth: usize,
        base_case: impl Fn(&Tensor) -> Result<bool>,
        recursive_case: impl Fn(&mut Self, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let mut current = init_value.clone();
        let mut depth = 0;

        while depth < max_depth {
            if base_case(&current)? {
                break;
            }
            current = recursive_case(self, &current)?;
            depth += 1;
        }

        Ok(current)
    }

    pub fn differentiable_try_catch(
        &mut self,
        try_block: impl Fn(&mut Self) -> Result<Tensor>,
        catch_block: impl Fn(&mut Self, &torsh_core::TorshError) -> Result<Tensor>,
    ) -> Result<Tensor> {
        match try_block(self) {
            Ok(result) => Ok(result),
            Err(error) => catch_block(self, &error),
        }
    }

    pub fn differentiable_assert(&mut self, condition: &Tensor, _message: &str) -> Result<Tensor> {
        let condition_value = self.sigmoid_approximation(condition)?;

        let penalty = self.ones_like(&condition_value)?.sub(&condition_value)?;
        let penalty_squared = penalty.mul(&penalty)?;

        Ok(penalty_squared)
    }

    pub fn differentiable_debug_print(&mut self, value: &Tensor, message: &str) -> Result<Tensor> {
        println!("{}: {:?}", message, value);
        Ok(value.clone())
    }

    fn sigmoid_approximation(&self, x: &Tensor) -> Result<Tensor> {
        let one = Tensor::ones(x.shape().dims(), x.device())?;
        let neg_x = x.neg()?;
        let exp_neg_x = neg_x.exp()?;
        let denominator = one.add(&exp_neg_x)?;
        one.div(&denominator)
    }

    fn softmax_approximation(&self, x: &Tensor) -> Result<Tensor> {
        let exp_x = x.exp()?;
        let sum_exp = exp_x.sum()?;
        exp_x.div(&sum_exp)
    }

    fn ones_like(&self, x: &Tensor) -> Result<Tensor> {
        Tensor::ones(x.shape().dims(), x.device())
    }

    fn zeros_like(&self, x: &Tensor) -> Result<Tensor> {
        Tensor::zeros(x.shape().dims(), x.device())
    }

    fn get_scalar_at(&self, tensor: &Tensor, index: usize) -> Result<Tensor> {
        let shape = tensor.shape();
        if index >= shape.dims()[0] {
            return Err(torsh_core::TorshError::InvalidArgument(
                "Index out of bounds".to_string(),
            ));
        }

        let slice = tensor.narrow(0, index as i64, 1)?;
        Ok(slice)
    }

    pub fn grad(&mut self, _output: &Tensor, input: &Tensor) -> Result<Tensor> {
        // TODO: Implement actual backward pass when AutogradTensor trait is available
        // For now, return mock gradient (ones with same shape as input)
        Ok(Tensor::ones(input.shape().dims(), input.device())?)
    }

    pub fn backward(&mut self, _output: &Tensor) -> Result<()> {
        // TODO: Implement actual backward pass when AutogradTensor trait is available
        // For now, this is a no-op
        Ok(())
    }
}

impl Default for DifferentiableProgramming {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::{DeviceType, Shape};

    #[test]
    fn test_differentiable_if_else() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let condition = Tensor::ones(&[1], DeviceType::Cpu)?;
        let true_value = Tensor::scalar(5.0)?;
        let false_value = Tensor::scalar(3.0)?;

        let result = dp.differentiable_if_else(
            &condition,
            |_| Ok(true_value.clone()),
            |_| Ok(false_value.clone()),
        )?;

        let expected_shape = Shape::from_dims(&[1])?;
        assert_eq!(result.shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_for_loop() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let init_value = Tensor::ones(&[2, 2], DeviceType::Cpu)?;

        let result =
            dp.differentiable_for_loop(&init_value, 3, |_, current, _| current.mul_scalar(2.0))?;

        let expected_shape = Shape::from_dims(&[2, 2])?;
        assert_eq!(result.shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_scan() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let init_state = Tensor::zeros(&[2], DeviceType::Cpu)?;
        let inputs = vec![
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
        ];

        let (final_state, outputs) =
            dp.differentiable_scan(&init_state, &inputs, |_, state, input| {
                let new_state = state.add(input)?;
                let output = new_state.clone();
                Ok((new_state, output))
            })?;

        let expected_shape = Shape::from_dims(&[2])?;
        assert_eq!(final_state.shape(), expected_shape);
        assert_eq!(outputs.len(), 3);
        Ok(())
    }

    #[test]
    fn test_differentiable_fold() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let init_value = Tensor::zeros(&[2], DeviceType::Cpu)?;
        let inputs = vec![
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
        ];

        let result =
            dp.differentiable_fold(&init_value, &inputs, |_, acc, input| acc.add(input))?;

        let expected_shape = Shape::from_dims(&[2])?;
        assert_eq!(result.shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_map() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let inputs = vec![
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
        ];

        let outputs = dp.differentiable_map(&inputs, |_, input| input.mul_scalar(2.0))?;

        assert_eq!(outputs.len(), 3);
        let expected_shape = Shape::from_dims(&[2])?;
        assert_eq!(outputs[0].shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_reduce() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let inputs = vec![
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
        ];

        let result = dp.differentiable_reduce(&inputs, |_, a, b| a.add(b))?;

        let expected_shape = Shape::from_dims(&[2])?;
        assert_eq!(result.shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_filter() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let inputs = vec![
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
            Tensor::ones(&[2], DeviceType::Cpu)?,
        ];

        let filtered = dp.differentiable_filter(&inputs, |_, input| {
            let sum = input.sum()?;
            Ok(sum)
        })?;

        assert_eq!(filtered.len(), 3);
        // After mul_scalar, the result should maintain the original input shape
        let expected_shape = Shape::from_dims(&[2])?;
        assert_eq!(filtered[0].shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_assert() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let condition = Tensor::ones(&[1], DeviceType::Cpu)?;
        let penalty = dp.differentiable_assert(&condition, "test assertion")?;

        let expected_shape = Shape::from_dims(&[1])?;
        assert_eq!(penalty.shape(), expected_shape);
        Ok(())
    }

    #[test]
    fn test_differentiable_debug_print() -> Result<()> {
        let mut dp = DifferentiableProgramming::new();

        let value = Tensor::ones(&[2, 2], DeviceType::Cpu)?;
        let result = dp.differentiable_debug_print(&value, "debug test")?;

        let expected_shape = Shape::from_dims(&[2, 2])?;
        assert_eq!(result.shape(), expected_shape);
        Ok(())
    }
}
