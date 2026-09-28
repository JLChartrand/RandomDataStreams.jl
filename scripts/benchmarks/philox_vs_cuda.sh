#!/usr/bin/env bash
#
# Philox4x32-10 (this package) vs CUDA.jl's randn, on the GPU, for a big
# sum(exp(n)), n ~ N(0,1), in Float32 and Float64. See philox_vs_cuda.jl for
# the contenders, the workload and how to read the numbers.
#
#     ./scripts/benchmarks/philox_vs_cuda.sh
#
# Every (contender, eltype) case runs in its own Julia process, so a CUDA fault
# in one case cannot poison the next, and a failed case does not stop the run.
#
# A run is written to results/<timestamp>-philox-vs-cuda/ (gitignored scratch,
# so the working tree stays clean while it runs and every result is stamped
# with a clean commit). When it finishes, the directory is copied to
# results/philox-vs-cuda/<timestamp>/ (tracked), committed, and pushed. Its
# contents:
#
#   philox_vs_cuda.csv   one row per case that completed
#   errors.txt           ONLY EXISTS IF SOMETHING WENT WRONG: every exception
#                        (with backtrace, GPU state, CUDA.versioninfo()), every
#                        case that crashed or timed out without a Julia
#                        exception, and any generator whose output failed the
#                        E[exp(Z)] sanity check
#   logs/<case>.log      full stdout+stderr of each case
#   sysinfo.txt          host, OS, Julia, GPU/CUDA versions
#   nvidia-smi.txt       GPU clocks/temperature/other processes, before the run
#   ../<same name>.tar.gz  all of the above in one file (scratch only, never
#                        committed)
#
# Tunables (environment variables):
#
#   BENCH_TOTAL_LOG2   draws per repetition, log2      (default 30, ~1.07e9)
#   BENCH_CHUNK_LOG2   draws per buffer fill, log2     (default 26; keep it a
#                      power of two, see philox_vs_cuda.jl)
#   BENCH_REPS         timed repetitions per case      (default 5)
#   BENCH_CONTENDERS   space-separated subset to run   (default: all five)
#   BENCH_TIMEOUT      seconds before a case is killed (default 1800)
#   BENCH_NO_PUSH      set to 1 to keep the results local: nothing is copied
#                      into the tracked folder, committed or pushed (use for
#                      smoke tests, so they do not land in git)
#
# e.g. a quick smoke test:  BENCH_TOTAL_LOG2=24 BENCH_REPS=2 ./philox_vs_cuda.sh
#
# Publishing needs a git identity and push access on the workstation. If it
# fails, the results are still complete in the scratch directory, the reason is
# printed and written to git-publish-error.txt beside them, and the exit status
# is non-zero. Only the published run's directory is committed, whatever else
# is staged.
#
# Exit status: 0 if every case completed and the results were published, 1
# otherwise. (Failing cases are recorded, not fatal: the script always runs all
# of them.)

# No `set -e`: a failing case must not stop the others. `-u` and pipefail stay.
set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RUN_NAME="$(date +%Y%m%d-%H%M%S)-philox-vs-cuda"
readonly OUT_DIR="$SCRIPT_DIR/results/$RUN_NAME"
readonly CSV="$OUT_DIR/philox_vs_cuda.csv"
readonly ERRORS="$OUT_DIR/errors.txt"
readonly TIMEOUT="${BENCH_TIMEOUT:-1800}"
readonly CONTENDERS="${BENCH_CONTENDERS:-philox_boxmuller philox_inversion curand gpuarrays native}"
readonly ELTYPES="Float32 Float64"

mkdir -p "$OUT_DIR/logs"
echo "Results directory: $OUT_DIR"

# Append a titled block to errors.txt (creating it), reading the body on stdin.
note_error() {
    {
        printf '%s\n' "==============================================================================="
        printf 'ERROR    %s   %s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S')"
        cat
        printf '\n'
    } >> "$ERRORS"
}

if ! command -v julia >/dev/null 2>&1; then
    echo "julia not found on PATH" | note_error "setup"
    echo "julia not found on PATH -- see $ERRORS" >&2
    exit 1
fi

# The machine, so the numbers can be interpreted later. Best-effort: a failure
# here is recorded but does not stop the benchmark.
if ! julia "$SCRIPT_DIR/sysinfo.jl" > "$OUT_DIR/sysinfo.txt" 2> "$OUT_DIR/logs/sysinfo.log"; then
    { echo "sysinfo.jl exited non-zero; its stderr:"; tail -n 60 "$OUT_DIR/logs/sysinfo.log"; } \
        | note_error "sysinfo.jl"
fi
cat "$OUT_DIR/sysinfo.txt"

# GPU clocks, temperature and any other process using the card: a benchmark on a
# GPU that is also driving a desktop or running someone's job is not comparable.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$OUT_DIR/nvidia-smi.txt" 2>&1 || true
else
    echo "nvidia-smi not found" > "$OUT_DIR/nvidia-smi.txt"
fi
echo

failed=0
total_cases=0

for T in $ELTYPES; do
    for C in $CONTENDERS; do
        total_cases=$((total_cases + 1))
        log="$OUT_DIR/logs/${C}_${T}.log"
        echo "---- $C / $T ----"

        # -O3 as in the other benchmarks. `timeout` so a hung GPU cannot hang
        # the run; --foreground keeps it killable from the terminal.
        timeout --foreground "$TIMEOUT" \
            julia -O3 "$SCRIPT_DIR/philox_vs_cuda.jl" "$C" "$T" "$CSV" "$ERRORS" \
            2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}

        case "$rc" in
            0)  ;;
            3)  # philox_vs_cuda.jl caught the exception and already wrote errors.txt
                failed=$((failed + 1)) ;;
            *)  # Julia never got to record anything: a crash, a signal, a failed
                # package load, or the timeout. The log is all there is.
                failed=$((failed + 1))
                case "$rc" in
                    124) why="timed out after ${TIMEOUT}s" ;;
                    137) why="killed (SIGKILL; out of host memory?)" ;;
                    139) why="segmentation fault" ;;
                    *)   why="julia exited with status $rc" ;;
                esac
                { echo "$why"; echo "last 80 lines of $log:"; echo; tail -n 80 "$log"; } \
                    | note_error "$C $T" ;;
        esac
        echo
    done
done

echo "=============================================================================="
if [ -f "$CSV" ]; then
    echo "Throughput, Gdraws/s (higher is better):"
    # contender, eltype, sum(exp) Gdraws/s, fill-only Gdraws/s
    cut -d, -f1-4 "$CSV" | { column -s, -t 2>/dev/null || cat; }
else
    echo "No case completed; there is no CSV."
fi
echo

if [ -f "$ERRORS" ]; then
    echo "$failed of $total_cases case(s) failed, and/or warnings were raised."
    echo "Details: $ERRORS"
else
    echo "All $total_cases cases completed; no errors."
fi

# One file with everything, for when git is not an option.
tar -C "$SCRIPT_DIR/results" -czf "$SCRIPT_DIR/results/$RUN_NAME.tar.gz" "$RUN_NAME" \
    && echo "Archive: $SCRIPT_DIR/results/$RUN_NAME.tar.gz"

# Publish: copy into the tracked folder, commit just that, push.
# Reads the details on stdin. The flag is the error file's existence, not a
# variable: this runs on the right of a pipe, i.e. in a subshell.
readonly PUBLISH_ERR="$OUT_DIR/git-publish-error.txt"
readonly PUBLISH_LOG="$SCRIPT_DIR/results/$RUN_NAME.publish.log"   # outside OUT_DIR: not copied
publish_error() {
    { echo "$1"; cat; } > "$PUBLISH_ERR"
    echo "PUBLISH FAILED: $1" >&2
    echo "  details: $PUBLISH_ERR" >&2
    echo "  the results are intact in $OUT_DIR" >&2
}

if [ "${BENCH_NO_PUSH:-0}" = "1" ]; then
    echo "BENCH_NO_PUSH=1: results kept local, nothing committed or pushed."
else
    readonly PUB_REL="scripts/benchmarks/results/philox-vs-cuda/${RUN_NAME%-philox-vs-cuda}"
    repo_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>&1)" \
        || { echo "$repo_root" | publish_error "not inside a git checkout"; repo_root=""; }

    if [ -n "$repo_root" ]; then
        pub_dir="$repo_root/$PUB_REL"
        {
            mkdir -p "$pub_dir" \
                && cp -r "$OUT_DIR"/. "$pub_dir"/ \
                && git -C "$repo_root" add -f -- "$PUB_REL" \
                && git -C "$repo_root" commit -q \
                    -m "bench: philox-vs-cuda results, ${RUN_NAME%-philox-vs-cuda} ($failed of $total_cases cases failed)" \
                    -- "$PUB_REL" \
                && git -C "$repo_root" push origin HEAD
        } > "$PUBLISH_LOG" 2>&1 \
            && echo "Published $PUB_REL (committed and pushed)." \
            || tail -n 40 "$PUBLISH_LOG" | publish_error "committing or pushing the results failed"
    fi
fi

[ "$failed" -eq 0 ] && [ ! -f "$PUBLISH_ERR" ]
