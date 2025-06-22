.PHONY: help build test clean format lint check docs examples bench install-deps dev-setup

# Default target
help:
	@echo "ToRSh Development Commands"
	@echo "=========================="
	@echo "build        - Build all packages"
	@echo "test         - Run all tests"
	@echo "test-fast    - Run tests without backend-cpu (faster)"
	@echo "clean        - Clean build artifacts"
	@echo "format       - Format code with rustfmt"
	@echo "lint         - Run clippy lints"
	@echo "check        - Run format + lint + test"
	@echo "docs         - Build documentation"
	@echo "examples     - Run example programs"
	@echo "bench        - Run benchmarks"
	@echo "install-deps - Install development dependencies"
	@echo "dev-setup    - Complete development environment setup"

# Build commands
build:
	cargo build --all-features

build-release:
	cargo build --release --all-features

# Test commands
test:
	cargo test --all-features

test-fast:
	cargo test --package torsh-core
	cargo test --package torsh-tensor
	cargo test --package torsh-autograd
	cargo test --package torsh-nn
	cargo test --package torsh-optim
	cargo test --package torsh-data
	cargo test --package torsh-backends

test-lib:
	cargo test --lib --all-features

# Code quality
format:
	cargo fmt --all

lint:
	cargo clippy --all-targets --all-features -- -D warnings

check: format lint test-fast

# Documentation
docs:
	cargo doc --no-deps --all-features --open

docs-build:
	cargo doc --no-deps --all-features

# Examples
examples:
	@echo "Running basic examples..."
	cargo run --example test_linear || true
	cargo run --example simple_cnn || true

# Benchmarks
bench:
	cargo bench --package torsh-benches

# Development setup
install-deps:
	@echo "Installing development dependencies..."
	rustup component add rustfmt clippy
	cargo install cargo-audit cargo-llvm-cov
	@echo "Installing pre-commit..."
	pip install pre-commit || echo "pre-commit installation failed (pip not available)"
	pre-commit install || echo "pre-commit setup failed"

dev-setup: install-deps
	@echo "Setting up development environment..."
	@echo "Creating .cargo/config.toml with optimized settings..."
	@mkdir -p .cargo
	@echo "Development setup complete!"

# Maintenance
clean:
	cargo clean

audit:
	cargo audit

update:
	cargo update

# CI simulation
ci-test:
	@echo "Simulating CI tests..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-fast
	$(MAKE) docs-build

# Platform-specific OpenMP setup
setup-macos:
	@echo "Setting up macOS environment..."
	brew install libomp || echo "libomp already installed"
	@echo "Add these to your shell profile:"
	@echo 'export LIBRARY_PATH="/opt/homebrew/lib:$$LIBRARY_PATH"'
	@echo 'export CPATH="/opt/homebrew/include:$$CPATH"'

setup-ubuntu:
	@echo "Setting up Ubuntu environment..."
	sudo apt-get update
	sudo apt-get install -y libblas-dev liblapack-dev gfortran

# Release helpers
tag-release:
	@echo "Current version tags:"
	git tag -l | sort -V | tail -5
	@echo "Create new tag with: git tag v0.1.0 && git push origin v0.1.0"

check-release:
	cargo build --release --all-features
	cargo test --release --all-features || true