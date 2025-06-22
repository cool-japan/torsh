//! Autograd context management

use crate::GRAD_MODE;
// Temporarily disable scirs2 integration
// use scirs2::autograd::{Context as SciContext, VariableEnvironment};
use torsh_core::error::Result;

/// Autograd context that wraps scirs2's context
pub struct AutogradContext {
    /// Placeholder for context data (scirs2 integration disabled)
    _placeholder: (),
    /// Whether this context owns gradient computation
    owns_grad: bool,
}

impl Default for AutogradContext {
    fn default() -> Self {
        Self {
            _placeholder: (),
            owns_grad: true,
        }
    }
}

impl AutogradContext {
    /// Create a new autograd context
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Check if gradients are enabled in this context
    pub fn is_grad_enabled(&self) -> bool {
        GRAD_MODE.read().enabled && self.owns_grad
    }
    
    /// Execute a function within this context (placeholder)
    pub fn run<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        f()
    }
    
    /// Clear the computation graph (placeholder)
    pub fn clear_graph(&mut self) {
        // Placeholder - would clear scirs2 graph when integrated
    }
    
    /// Enable gradient checkpointing for memory efficiency (placeholder)
    pub fn enable_checkpointing(&mut self) {
        // Placeholder - would enable scirs2 checkpointing when integrated
    }
    
    /// Disable gradient checkpointing (placeholder)
    pub fn disable_checkpointing(&mut self) {
        // Placeholder - would disable scirs2 checkpointing when integrated
    }
    
    /// Get the number of operations in the graph (placeholder)
    pub fn graph_size(&self) -> usize {
        // Placeholder - would return scirs2 graph size when integrated
        0
    }
}

// Thread-local autograd context
thread_local! {
    static THREAD_CONTEXT: std::cell::RefCell<Option<AutogradContext>> = 
        const { std::cell::RefCell::new(None) };
}

/// Get or create the thread-local autograd context
pub fn get_or_create_context() -> Result<AutogradContext> {
    THREAD_CONTEXT.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        if ctx_ref.is_none() {
            *ctx_ref = Some(AutogradContext::new());
        }
        Ok(ctx_ref.take().unwrap())
    })
}

/// Execute a function with an autograd context
pub fn with_context<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut AutogradContext) -> Result<R>,
{
    let mut ctx = get_or_create_context()?;
    let result = f(&mut ctx);
    
    // Store context back
    THREAD_CONTEXT.with(|thread_ctx| {
        *thread_ctx.borrow_mut() = Some(ctx);
    });
    
    result
}