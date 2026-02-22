#!/bin/bash

# ToRSh Benchmarks Validation and Cleanup Script
# Run this script once build system issues are resolved

set -e  # Exit on any error

echo "🚀 ToRSh Benchmarks - Comprehensive Validation and Cleanup"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 Step $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Clean build artifacts
print_step 1 "Cleaning build artifacts"
cargo clean
print_success "Build artifacts cleaned"

# Step 2: Check compilation
print_step 2 "Checking compilation status"
if cargo check --all-features; then
    print_success "Compilation successful"
else
    print_error "Compilation failed - fix errors before proceeding"
    exit 1
fi

# Step 3: Run clippy for warnings
print_step 3 "Running clippy analysis"
echo "Current warnings:"
cargo clippy --all-features 2>&1 | grep -E "(warning|error)" | wc -l || echo "0"

# Step 4: Clean up warnings (if cleanup script available)
print_step 4 "Cleaning up warnings"
if [[ -f "cleanup_warnings.rs" ]]; then
    print_warning "Automated cleanup script available but not yet compiled"
    print_warning "Manual cleanup required - see cleanup_warnings.rs for guidance"
else
    print_warning "Manual cleanup of warnings recommended"
fi

# Step 5: Run tests
print_step 5 "Running test suite"
if command -v cargo-nextest &> /dev/null; then
    echo "Using cargo nextest (as specified in user preferences)"
    cargo nextest run
else
    echo "Using standard cargo test"
    cargo test
fi

# Step 6: Run benchmarks
print_step 6 "Running benchmark validation"
if cargo bench --no-run 2>/dev/null; then
    echo "Benchmark compilation successful"
    print_success "Benchmarks ready for execution"
else
    print_warning "Benchmark compilation failed - check benchmark implementations"
fi

# Step 7: Test cross-framework features
print_step 7 "Testing cross-framework functionality"

echo "Testing PyTorch integration:"
if cargo test --features pytorch test_pytorch_comparison 2>/dev/null; then
    print_success "PyTorch integration working"
else
    print_warning "PyTorch integration not available or failing"
fi

echo "Testing NumPy integration:"
if cargo test --features numpy_baseline test_numpy_comparison 2>/dev/null; then
    print_success "NumPy integration working"
else
    print_warning "NumPy integration not available or failing"
fi

# Step 8: Generate comprehensive report
print_step 8 "Generating validation report"

REPORT_FILE="validation_report_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT_FILE" << EOF
# ToRSh Benchmarks Validation Report

Generated: $(date)

## Summary

### Compilation
- Status: $(if cargo check --all-features &>/dev/null; then echo "✅ SUCCESS"; else echo "❌ FAILED"; fi)
- Warnings: $(cargo clippy --all-features 2>&1 | grep -c "warning" || echo "0")
- Errors: $(cargo clippy --all-features 2>&1 | grep -c "error" || echo "0")

### Tests
- Unit Tests: $(if cargo test --lib &>/dev/null; then echo "✅ PASS"; else echo "❌ FAIL"; fi)
- Integration Tests: $(if cargo test --test '*' &>/dev/null; then echo "✅ PASS"; else echo "❌ FAIL"; fi)

### Benchmarks
- Compilation: $(if cargo bench --no-run &>/dev/null; then echo "✅ SUCCESS"; else echo "❌ FAILED"; fi)

### Cross-Framework
- PyTorch: $(if cargo test --features pytorch &>/dev/null; then echo "✅ AVAILABLE"; else echo "❌ NOT AVAILABLE"; fi)
- NumPy: $(if cargo test --features numpy_baseline &>/dev/null; then echo "✅ AVAILABLE"; else echo "❌ NOT AVAILABLE"; fi)

## Next Steps

1. Fix any failing tests or compilation errors
2. Clean up remaining warnings
3. Run full benchmark suite: \`cargo bench\`
4. Set up CI integration for continuous benchmarking

## Files Available

- \`validate_benchmarks.rs\` - Comprehensive validation script
- \`cleanup_warnings.rs\` - Automated cleanup script  
- \`IMPLEMENTATION_STATUS.md\` - Complete status documentation

EOF

print_success "Validation report saved to: $REPORT_FILE"

# Step 9: Final recommendations
print_step 9 "Final recommendations"

echo ""
echo "🎯 Validation Complete!"
echo ""
echo "📋 Next Actions:"
echo "  1. Review validation report: $REPORT_FILE"
echo "  2. Fix any identified issues"
echo "  3. Run full benchmark suite: cargo bench"
echo "  4. Consider setting up CI integration"
echo ""

# Check overall status
WARNINGS=$(cargo clippy --all-features 2>&1 | grep -c "warning" || echo "0")
ERRORS=$(cargo clippy --all-features 2>&1 | grep -c "error" || echo "0")

if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    print_success "All checks passed! ToRSh benchmarks are ready for production."
elif [[ $ERRORS -eq 0 ]]; then
    print_warning "Minor warnings detected ($WARNINGS warnings). Consider cleanup."
else
    print_error "Issues detected ($ERRORS errors, $WARNINGS warnings). Fix before production use."
fi

echo ""
echo "📖 For detailed implementation status, see: IMPLEMENTATION_STATUS.md"
echo "🛠️  For cleanup guidance, see: cleanup_warnings.rs"
echo "🧪 For validation tools, see: validate_benchmarks.rs"