#!/bin/bash

# ToRSh WASM Build Script

echo "🚀 Building ToRSh WASM module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed!"
    echo "Please install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Check if wasm32-unknown-unknown target is installed
if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
    echo "📦 Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# Clean previous build
if [ -d "pkg" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf pkg
fi

# Build based on argument
if [ "$1" = "release" ]; then
    echo "🏗️  Building in release mode..."
    wasm-pack build --target web --out-dir pkg --release
else
    echo "🏗️  Building in debug mode..."
    echo "   (Use './build.sh release' for optimized build)"
    wasm-pack build --target web --out-dir pkg --dev
fi

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "📊 Build artifacts:"
    ls -la pkg/
    echo ""
    echo "🌐 To run the example:"
    echo "   1. Start a web server: python3 -m http.server 8000"
    echo "   2. Open browser at: http://localhost:8000"
else
    echo "❌ Build failed!"
    exit 1
fi