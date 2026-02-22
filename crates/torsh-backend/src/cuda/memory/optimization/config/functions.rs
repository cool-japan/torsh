//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ConfigError, OptimizationConfig, ValidationResult};

pub trait ConfigValidator: std::fmt::Debug + Send + Sync {
    fn validate(
        &self,
        config: &OptimizationConfig,
    ) -> Result<ValidationResult, ConfigError>;
    fn get_name(&self) -> &str;
    fn get_description(&self) -> &str;
}
