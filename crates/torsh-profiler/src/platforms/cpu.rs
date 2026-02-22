//! CPU profiling platform integrations

// Re-export from existing modules for backward compatibility
pub use crate::cpu::*;
pub use crate::instruments::*;
pub use crate::vtune::*;

/// Unified CPU profiling interface
pub struct CpuProfilerPlatform {
    pub cpu_profiler: Option<crate::cpu::CpuProfiler>,
    #[cfg(target_os = "macos")]
    pub instruments_profiler: Option<crate::instruments::InstrumentsProfiler>,
    #[cfg(target_os = "linux")]
    pub vtune_profiler: Option<crate::vtune::VTuneProfiler>,
}

impl CpuProfilerPlatform {
    pub fn new() -> Self {
        Self {
            cpu_profiler: None,
            #[cfg(target_os = "macos")]
            instruments_profiler: None,
            #[cfg(target_os = "linux")]
            vtune_profiler: None,
        }
    }

    pub fn with_cpu_profiler(mut self) -> Self {
        self.cpu_profiler = Some(crate::cpu::CpuProfiler::new());
        self
    }

    #[cfg(target_os = "macos")]
    pub fn with_instruments(mut self) -> Self {
        self.instruments_profiler = Some(crate::instruments::create_instruments_profiler());
        self
    }

    #[cfg(target_os = "linux")]
    pub fn with_vtune(mut self) -> Self {
        self.vtune_profiler = Some(crate::vtune::create_vtune_profiler());
        self
    }

    pub fn start_profiling(&mut self) -> crate::TorshResult<()> {
        if let Some(ref mut cpu) = self.cpu_profiler {
            // CPU profiler start logic would go here
        }

        #[cfg(target_os = "macos")]
        if let Some(ref mut instruments) = self.instruments_profiler {
            // Instruments profiler start logic would go here
        }

        #[cfg(target_os = "linux")]
        if let Some(ref mut vtune) = self.vtune_profiler {
            // VTune profiler start logic would go here
        }

        Ok(())
    }

    pub fn stop_profiling(&mut self) -> crate::TorshResult<()> {
        // Stop all active profilers
        Ok(())
    }
}

impl Default for CpuProfilerPlatform {
    fn default() -> Self {
        Self::new()
    }
}
