#!/usr/bin/env bash
# check_perf_regression.sh — Compare cargo bench output against a saved baseline.
#
# Usage:
#   cargo bench --bench regression_baselines -- --save-baseline <name>   # save
#   cargo bench --bench regression_baselines -- --baseline <name>        # compare
#   bash scripts/check_perf_regression.sh [threshold_percent]            # check JSON
#
# The script reads criterion's change/estimates.json files and fails if any
# benchmark regressed by more than THRESHOLD_PCT percent.
#
# Requires: jq

set -euo pipefail

THRESHOLD_PCT="${1:-10}"
CRITERION_DIR="target/criterion"
FAILED=0
CHECKED=0

if [[ ! -d "$CRITERION_DIR" ]]; then
    echo "WARN: $CRITERION_DIR not found. Run 'cargo bench' first." >&2
    exit 0
fi

while IFS= read -r estimates_file; do
    bench_name=$(echo "$estimates_file" | sed "s|$CRITERION_DIR/||;s|/change/estimates.json||")

    if ! jq -e '.mean' "$estimates_file" > /dev/null 2>&1; then
        continue
    fi

    mean_pct=$(jq -r '.mean.point_estimate * 100' "$estimates_file" 2>/dev/null || echo "0")

    # A positive mean_pct means the benchmark got slower (regression)
    if awk -v pct="$mean_pct" -v thr="$THRESHOLD_PCT" 'BEGIN { exit !(pct > thr) }'; then
        echo "REGRESSION: $bench_name slowed by ${mean_pct}% (threshold: ${THRESHOLD_PCT}%)" >&2
        FAILED=1
    elif awk -v pct="$mean_pct" 'BEGIN { exit !(pct < -5) }'; then
        echo "IMPROVEMENT: $bench_name improved by $(echo "$mean_pct" | awk '{printf "%.1f", -$1}')%"
    else
        echo "OK: $bench_name (${mean_pct}% change)"
    fi
    CHECKED=$((CHECKED + 1))
done < <(find "$CRITERION_DIR" -name "estimates.json" -path "*/change/*" 2>/dev/null | sort)

if [[ $CHECKED -eq 0 ]]; then
    echo "WARN: No baseline comparison data found. Run with --baseline <name> to compare." >&2
    exit 0
fi

echo ""
echo "Checked $CHECKED benchmarks with threshold ${THRESHOLD_PCT}%"

if [[ $FAILED -ne 0 ]]; then
    echo "FAIL: One or more benchmarks regressed." >&2
    exit 1
fi

echo "PASS: No regressions detected."
exit 0
