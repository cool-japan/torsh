//! Type definitions for adaptive controller
//!
//! This module contains all type definitions used by the adaptive optimization controller.
//! The types are split across multiple files for maintainability using include!().

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// Core type definitions (structs, enums, and basic impls)
include!("types_defs.rs");

// Controller implementation and advanced types
include!("types_impl.rs");
