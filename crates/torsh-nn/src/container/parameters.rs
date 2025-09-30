//! Parameter container modules for organizing trainable parameters

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// ParameterList container for parameters
pub struct ParameterList {
    base: ModuleBase,
    parameters: Vec<Parameter>,
}

impl ParameterList {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            parameters: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    pub fn push(&mut self, parameter: Parameter) {
        self.parameters.push(parameter);
    }

    pub fn extend<I>(&mut self, parameters: I)
    where
        I: IntoIterator<Item = Parameter>,
    {
        self.parameters.extend(parameters);
    }

    pub fn get(&self, index: usize) -> Option<&Parameter> {
        self.parameters.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Parameter> {
        self.parameters.get_mut(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Parameter> {
        self.parameters.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Parameter> {
        self.parameters.iter_mut()
    }
}

impl Default for ParameterList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ParameterList {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        // ParameterList doesn't define forward pass - parameters should be used individually
        Err(TorshError::InvalidArgument(
            "ParameterList doesn't define forward pass".to_string(),
        ))
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, param) in self.parameters.iter().enumerate() {
            params.insert(i.to_string(), param.clone());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, param) in self.parameters.iter().enumerate() {
            params.insert(i.to_string(), param.clone());
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        // Move all parameters to the specified device
        for param in &mut self.parameters {
            let tensor_guard = param.tensor();
            let tensor = tensor_guard.read().clone();
            let moved_tensor = tensor.to(device)?;
            *tensor_guard.write() = moved_tensor;
        }
        Ok(())
    }
}

/// ParameterDict container for named parameters
pub struct ParameterDict {
    base: ModuleBase,
    parameters: HashMap<String, Parameter>,
}

impl ParameterDict {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            parameters: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    pub fn insert(&mut self, key: String, parameter: Parameter) {
        self.parameters.insert(key, parameter);
    }

    pub fn get(&self, key: &str) -> Option<&Parameter> {
        self.parameters.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Parameter> {
        self.parameters.get_mut(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.parameters.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &Parameter> {
        self.parameters.values()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Parameter> {
        self.parameters.values_mut()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Parameter)> {
        self.parameters.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut Parameter)> {
        self.parameters.iter_mut()
    }
}

impl Default for ParameterDict {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ParameterDict {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        // ParameterDict doesn't define forward pass - parameters should be used individually
        Err(TorshError::InvalidArgument(
            "ParameterDict doesn't define forward pass".to_string(),
        ))
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.parameters.clone()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters.clone()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        // Move all parameters to the specified device
        for param in self.parameters.values_mut() {
            let tensor_guard = param.tensor();
            let tensor = tensor_guard.read().clone();
            let moved_tensor = tensor.to(device)?;
            *tensor_guard.write() = moved_tensor;
        }
        Ok(())
    }
}
