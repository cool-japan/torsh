# torsh-convert

Model converter CLI for the ToRSh deep learning framework. Convert between different model formats with ease.

## Overview

`torsh-convert` is a command-line tool for converting trained model files between various serialization formats supported by the ToRSh ecosystem. It provides a streamlined workflow for model interoperability, format migration, and deployment preparation.

## Features

- **Format Conversion**: Convert models between ToRSh-native and external formats
- **Batch Processing**: Process multiple model files using glob patterns and directory walking
- **Progress Tracking**: Visual progress indicators for long-running conversions
- **JSON/TOML Support**: Read and write model metadata in JSON and TOML formats
- **Async Runtime**: Built on Tokio for efficient I/O during large model conversions
- **Structured Logging**: Configurable logging via `env_logger` for debugging conversion pipelines

## Installation

```bash
cargo install torsh-convert
```

Or build from the workspace root:

```bash
cargo build --release -p torsh-convert
```

## Usage

```bash
# Basic conversion
torsh-convert input_model.torsh --output output_model.json

# Convert all models in a directory
torsh-convert ./models/*.torsh --output-dir ./converted/
```

## Dependencies

This tool relies on the following ToRSh crates:

- `torsh` - Main framework entry point
- `torsh-core` - Core types and device abstraction
- `torsh-tensor` - Tensor data structures
- `torsh-nn` - Neural network module definitions

## License

Licensed under the same terms as the ToRSh workspace. See the repository root for details.

## Links

- [Repository](https://github.com/cool-japan/torsh)
- [Documentation](https://docs.rs/torsh-convert)
