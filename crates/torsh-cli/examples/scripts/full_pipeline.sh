#!/bin/bash
# Complete ML Pipeline: Dataset → Train → Benchmark → Quantize → Deploy
# This script demonstrates a complete end-to-end workflow

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET="cifar10"
MODEL="resnet18"
DATA_DIR="./data/${DATASET}"
RUNS_DIR="./runs"
MODELS_DIR="./models"
BENCHMARKS_DIR="./benchmarks"
DEVICE="cpu"  # Change to cuda or metal if available

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ToRSh CLI Complete ML Pipeline                    ║${NC}"
echo -e "${BLUE}║  Dataset → Train → Benchmark → Quantize → Deploy  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Create directories
mkdir -p "$DATA_DIR" "$RUNS_DIR" "$MODELS_DIR" "$BENCHMARKS_DIR"

# ========================================
# Phase 1: Dataset Preparation
# ========================================
echo -e "${GREEN}[1/6] Downloading and preparing dataset...${NC}"
if [ ! -d "$DATA_DIR/train" ]; then
    torsh dataset download "$DATASET" --output "$DATA_DIR"
    echo -e "${GREEN}✓ Dataset downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Dataset already exists, skipping download${NC}"
fi

# Validate dataset
echo -e "${GREEN}[1/6] Validating dataset...${NC}"
torsh dataset validate "$DATA_DIR/train"
echo -e "${GREEN}✓ Dataset validated${NC}"

# Calculate statistics
echo -e "${GREEN}[1/6] Calculating dataset statistics...${NC}"
torsh dataset statistics "$DATA_DIR/train" --output "$DATA_DIR/stats.json"
echo -e "${GREEN}✓ Statistics calculated${NC}"
echo ""

# ========================================
# Phase 2: Model Training
# ========================================
echo -e "${GREEN}[2/6] Training ${MODEL} model...${NC}"
echo -e "${YELLOW}This may take several minutes depending on hardware${NC}"

# Note: In real usage, use a proper config file
# For this example, we'll use command-line args
torsh train start \
    --config ../configs/train_${MODEL}_${DATASET}.yaml \
    --data "$DATA_DIR" \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --device "$DEVICE" \
    --optimizer adam \
    --scheduler cosine \
    --save-every 5 \
    --output-dir "$RUNS_DIR"

# Get the latest run directory
LATEST_RUN=$(ls -t "$RUNS_DIR" | head -1)
TRAINED_MODEL="$RUNS_DIR/$LATEST_RUN/best_model.ckpt"

echo -e "${GREEN}✓ Training completed${NC}"
echo -e "${BLUE}   Model saved to: $TRAINED_MODEL${NC}"
echo ""

# ========================================
# Phase 3: Model Inspection
# ========================================
echo -e "${GREEN}[3/6] Inspecting trained model...${NC}"
torsh model inspect "$TRAINED_MODEL" --detailed --stats --memory

echo -e "${GREEN}✓ Model inspection completed${NC}"
echo ""

# ========================================
# Phase 4: Benchmarking Original Model
# ========================================
echo -e "${GREEN}[4/6] Benchmarking original model...${NC}"

BENCHMARK_ORIGINAL="$BENCHMARKS_DIR/original_${DATASET}_${MODEL}.json"

torsh benchmark \
    --model "$TRAINED_MODEL" \
    --devices "$DEVICE" \
    --batch-sizes 1,8,16,32 \
    --input-shape 3,32,32 \
    --warmup-iterations 5 \
    --benchmark-iterations 50 \
    --output-format json \
    --output "$BENCHMARK_ORIGINAL"

echo -e "${GREEN}✓ Original model benchmarked${NC}"
echo -e "${BLUE}   Results saved to: $BENCHMARK_ORIGINAL${NC}"

# Extract key metrics
ORIGINAL_THROUGHPUT=$(jq -r '.summary.best_throughput.metric_value' "$BENCHMARK_ORIGINAL")
ORIGINAL_LATENCY=$(jq -r '.summary.best_latency.metric_value' "$BENCHMARK_ORIGINAL")
echo -e "${BLUE}   Throughput: ${ORIGINAL_THROUGHPUT} samples/sec${NC}"
echo -e "${BLUE}   Latency: ${ORIGINAL_LATENCY} ms${NC}"
echo ""

# ========================================
# Phase 5: Model Quantization
# ========================================
echo -e "${GREEN}[5/6] Quantizing model to INT8...${NC}"

QUANTIZED_MODEL="$MODELS_DIR/${MODEL}_${DATASET}_int8.torsh"

torsh quantize \
    --input "$TRAINED_MODEL" \
    --output "$QUANTIZED_MODEL" \
    --mode dynamic \
    --precision int8 \
    --per-channel \
    --symmetric

echo -e "${GREEN}✓ Model quantized${NC}"
echo -e "${BLUE}   Quantized model: $QUANTIZED_MODEL${NC}"

# Get file sizes
ORIGINAL_SIZE=$(du -h "$TRAINED_MODEL" | cut -f1)
QUANTIZED_SIZE=$(du -h "$QUANTIZED_MODEL" | cut -f1)
echo -e "${BLUE}   Original size: $ORIGINAL_SIZE${NC}"
echo -e "${BLUE}   Quantized size: $QUANTIZED_SIZE${NC}"
echo ""

# ========================================
# Phase 6: Benchmarking Quantized Model
# ========================================
echo -e "${GREEN}[6/6] Benchmarking quantized model...${NC}"

BENCHMARK_QUANTIZED="$BENCHMARKS_DIR/quantized_${DATASET}_${MODEL}.json"

torsh benchmark \
    --model "$QUANTIZED_MODEL" \
    --devices "$DEVICE" \
    --batch-sizes 1,8,16,32 \
    --input-shape 3,32,32 \
    --warmup-iterations 5 \
    --benchmark-iterations 50 \
    --output-format json \
    --output "$BENCHMARK_QUANTIZED"

echo -e "${GREEN}✓ Quantized model benchmarked${NC}"
echo -e "${BLUE}   Results saved to: $BENCHMARK_QUANTIZED${NC}"

# Extract key metrics
QUANTIZED_THROUGHPUT=$(jq -r '.summary.best_throughput.metric_value' "$BENCHMARK_QUANTIZED")
QUANTIZED_LATENCY=$(jq -r '.summary.best_latency.metric_value' "$BENCHMARK_QUANTIZED")
echo -e "${BLUE}   Throughput: ${QUANTIZED_THROUGHPUT} samples/sec${NC}"
echo -e "${BLUE}   Latency: ${QUANTIZED_LATENCY} ms${NC}"
echo ""

# ========================================
# Summary
# ========================================
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Pipeline Completed Successfully! 🎉               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo -e "  Dataset: ${DATASET}"
echo -e "  Model: ${MODEL}"
echo -e "  Device: ${DEVICE}"
echo ""
echo -e "${BLUE}Performance Comparison:${NC}"
echo -e "  Original Model:"
echo -e "    - Throughput: ${ORIGINAL_THROUGHPUT} samples/sec"
echo -e "    - Latency: ${ORIGINAL_LATENCY} ms"
echo -e "    - Size: $ORIGINAL_SIZE"
echo ""
echo -e "  Quantized Model (INT8):"
echo -e "    - Throughput: ${QUANTIZED_THROUGHPUT} samples/sec"
echo -e "    - Latency: ${QUANTIZED_LATENCY} ms"
echo -e "    - Size: $QUANTIZED_SIZE"
echo ""

# Calculate speedup
SPEEDUP=$(echo "scale=2; $QUANTIZED_THROUGHPUT / $ORIGINAL_THROUGHPUT" | bc)
echo -e "${GREEN}  Speedup: ${SPEEDUP}x${NC}"
echo ""

echo -e "${BLUE}Generated Artifacts:${NC}"
echo -e "  - Trained model: $TRAINED_MODEL"
echo -e "  - Quantized model: $QUANTIZED_MODEL"
echo -e "  - Original benchmark: $BENCHMARK_ORIGINAL"
echo -e "  - Quantized benchmark: $BENCHMARK_QUANTIZED"
echo -e "  - Dataset statistics: $DATA_DIR/stats.json"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Deploy the quantized model for inference"
echo -e "  2. Fine-tune hyperparameters for better performance"
echo -e "  3. Try different quantization modes (static, QAT)"
echo -e "  4. Benchmark on different hardware (CUDA, Metal)"
echo ""
echo -e "${GREEN}Done! 🚀${NC}"
