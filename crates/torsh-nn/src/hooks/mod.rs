//! Hook system infrastructure for neural network modules
//!
//! This module provides a comprehensive hook system that allows users to register
//! callbacks that execute at different stages of module execution.

use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Hook types for different stages of module execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookType {
    /// Called before forward pass
    PreForward,
    /// Called after forward pass
    PostForward,
    /// Called before backward pass (when autograd is available)
    PreBackward,
    /// Called after backward pass (when autograd is available)
    PostBackward,
}

/// Hook callback function signature
pub type HookCallback =
    Box<dyn Fn(&dyn crate::Module, &Tensor, Option<&Tensor>) -> Result<()> + Send + Sync>;

/// Hook handle for removing hooks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookHandle(usize);

/// Hook registry for managing module hooks
#[derive(Default)]
pub struct HookRegistry {
    hooks: HashMap<HookType, Vec<(HookHandle, HookCallback)>>,
    next_handle: usize,
}

impl core::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("hooks_count", &self.hooks.len())
            .field("next_handle", &self.next_handle)
            .finish()
    }
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            next_handle: 0,
        }
    }

    /// Register a hook for a specific type
    pub fn register_hook(&mut self, hook_type: HookType, callback: HookCallback) -> HookHandle {
        let handle = HookHandle(self.next_handle);
        self.next_handle += 1;

        self.hooks
            .entry(hook_type)
            .or_default()
            .push((handle, callback));

        handle
    }

    /// Remove a hook by handle
    pub fn remove_hook(&mut self, hook_type: HookType, handle: HookHandle) -> bool {
        if let Some(hooks) = self.hooks.get_mut(&hook_type) {
            if let Some(pos) = hooks.iter().position(|(h, _)| *h == handle) {
                let _ = hooks.remove(pos);
                return true;
            }
        }
        false
    }

    /// Execute all hooks of a specific type
    pub fn execute_hooks(
        &self,
        hook_type: HookType,
        module: &dyn crate::Module,
        input: &Tensor,
        output: Option<&Tensor>,
    ) -> Result<()> {
        if let Some(hooks) = self.hooks.get(&hook_type) {
            for (_, callback) in hooks {
                callback(module, input, output)?;
            }
        }
        Ok(())
    }

    /// Check if any hooks are registered for a type
    pub fn has_hooks(&self, hook_type: HookType) -> bool {
        self.hooks.get(&hook_type).is_some_and(|h| !h.is_empty())
    }

    /// Get the number of hooks for a specific type
    pub fn hook_count(&self, hook_type: HookType) -> usize {
        self.hooks.get(&hook_type).map_or(0, |h| h.len())
    }

    /// Clear all hooks of a specific type
    pub fn clear_hooks(&mut self, hook_type: HookType) {
        self.hooks.remove(&hook_type);
    }

    /// Clear all hooks
    pub fn clear_all_hooks(&mut self) {
        self.hooks.clear();
    }
}
