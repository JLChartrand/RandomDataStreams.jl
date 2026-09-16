#!/usr/bin/env bash
#
# Run every benchmark in this directory, in order: throughput.jl (CPU, every
# generator) first, then gpu_throughput.jl (CPU vs GPU, Philox4x32-10 only;
# skips itself with a message if CUDA.functional() is false). Each writes its
# own report to stdout -- this script adds nothing but the sequencing, so it
# works from anywhere, not just this directory.
#
#     ./scripts/benchmarks/run-all.sh

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

julia -O3 "$SCRIPT_DIR/throughput.jl"
julia -O3 "$SCRIPT_DIR/gpu_throughput.jl"
