#!/usr/bin/env bash
#
# Run every benchmark in this directory, in order: throughput.jl (CPU, every
# generator) first, then gpu_throughput.jl (CPU vs GPU, Philox4x32-10 only;
# skips itself with a message if CUDA.functional() is false). Each still
# prints its usual report to stdout as it runs -- this script's own job is
# the sequencing, a results directory, and making sure a number saved today
# can still be read in six months:
#
#   results/<timestamp>/sysinfo.txt        -- host, OS, Julia, GPU/CUDA
#   results/<timestamp>/throughput.csv     -- every number from throughput.jl
#   results/<timestamp>/gpu_throughput.csv -- every number from gpu_throughput.jl
#
# Works from anywhere, not just this directory.
#
#     ./scripts/benchmarks/run-all.sh

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly OUT_DIR="$SCRIPT_DIR/results/$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR"
echo "Results directory: $OUT_DIR"

julia "$SCRIPT_DIR/sysinfo.jl" > "$OUT_DIR/sysinfo.txt"
cat "$OUT_DIR/sysinfo.txt"

julia -O3 "$SCRIPT_DIR/throughput.jl" "$OUT_DIR/throughput.csv"
julia -O3 "$SCRIPT_DIR/gpu_throughput.jl" "$OUT_DIR/gpu_throughput.csv"

echo "Done. Results saved under $OUT_DIR"
