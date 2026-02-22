//! Platform-specific profiling implementations

pub mod cpu;
pub mod gpu;
pub mod system;

// Re-export platform types
pub use cpu::*;
pub use gpu::*;
pub use system::*;
