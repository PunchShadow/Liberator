#!/bin/bash
# Verification script: runs Liberator algorithms with CPU verification on KONECT datasets.
# Usage: ./verify_all.sh [model]
#   model: 0 (Ascetic baseline) or 7 (Liberator). Default: both.
#
# Large graphs (friendster, twitter, uk-2007, sk-2005) may take significant time
# for CPU verification. Use SKIP_LARGE=1 to skip them.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/../datasets/KONECT"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/ptGraph"

# Parse args
MODEL_FILTER="${1:-all}"  # 0, 7, or all
SKIP_LARGE="${SKIP_LARGE:-0}"

# Large graphs to optionally skip
LARGE_GRAPHS="friendster_konect.bcsr twitter_mpi.bcsr uk-2007.bcsr sk-2005.bcsr"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found at $BINARY. Building..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake ..
    make -j$(nproc)
    cd "$SCRIPT_DIR"
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

PASS=0
FAIL=0
SKIP=0

run_test() {
    local file="$1"
    local type="$2"
    local model="$3"
    local extra_args="$4"
    local basename=$(basename "$file")

    # Check skip
    if [ "$SKIP_LARGE" = "1" ]; then
        for lg in $LARGE_GRAPHS; do
            if [ "$basename" = "$lg" ]; then
                echo "[SKIP] $type model=$model $basename (large graph)"
                SKIP=$((SKIP + 1))
                return
            fi
        done
    fi

    echo "----------------------------------------------"
    echo "[TEST] $type model=$model $basename"
    echo "----------------------------------------------"

    if "$BINARY" --input "$file" --type "$type" --model "$model" --testTime 1 $extra_args --verify 2>&1 | tee /dev/stderr | grep -q "PASSED"; then
        echo "[PASS] $type model=$model $basename"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $type model=$model $basename"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

should_run_model() {
    local m="$1"
    [ "$MODEL_FILTER" = "all" ] || [ "$MODEL_FILTER" = "$m" ]
}

echo "=============================================="
echo "  Liberator Verification Suite"
echo "  Datasets: $DATASET_DIR"
echo "  Model filter: $MODEL_FILTER"
echo "  Skip large: $SKIP_LARGE"
echo "=============================================="
echo ""

# BFS on .bcsr files
echo "=== BFS Tests ==="
for f in "$DATASET_DIR"/*.bcsr; do
    [ -f "$f" ] || continue
    if should_run_model 0; then run_test "$f" bfs 0 "--source 0"; fi
    if should_run_model 7; then run_test "$f" bfs 7 "--source 0"; fi
done

# CC on .bcsr files
echo "=== CC Tests ==="
for f in "$DATASET_DIR"/*.bcsr; do
    [ -f "$f" ] || continue
    if should_run_model 0; then run_test "$f" cc 0 ""; fi
    if should_run_model 7; then run_test "$f" cc 7 ""; fi
done

# SSSP on .bwcsr files
echo "=== SSSP Tests ==="
for f in "$DATASET_DIR"/*.bwcsr; do
    [ -f "$f" ] || continue
    if should_run_model 0; then run_test "$f" sssp 0 "--source 0"; fi
    if should_run_model 7; then run_test "$f" sssp 7 "--source 0"; fi
done

# PR: requires .bcsc files (none available in KONECT, skip)
echo "=== PR Tests ==="
echo "[SKIP] No .bcsc files available in KONECT for PageRank verification."
echo ""

echo "=============================================="
echo "  RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "=============================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
