#!/usr/bin/env bash
#
# One-command setup for a validation campaign on a dedicated machine.
#
# The pieces this wraps -- throughput.jl, campaign.jl, the systemd unit -- each
# do one thing and are documented where they live. What was still manual is the
# order and the arithmetic: benchmark before the machine gets busy, calibrate
# before committing to a plan, size --jobs from the measured peak RSS rather
# than from nproc, and install a supervisor that survives a logout and a reboot.
#
#     ./scripts/run-campaign.sh prepare --battery=crush
#     ./scripts/run-campaign.sh start   --battery=crush
#     ./scripts/run-campaign.sh run     --battery=crush   # both of the above
#     ./scripts/run-campaign.sh status  --battery=crush
#     ./scripts/run-campaign.sh stop    --battery=crush
#     ./scripts/run-campaign.sh ensure  --battery=crush   # start only if stopped
#     ./scripts/run-campaign.sh collect --battery=crush
#     ./scripts/run-campaign.sh replay  --battery=crush   # resolve the suspects
#
# There is no default battery, here or in campaign.jl. A stray argument must not
# be able to start a two-week BigCrush sweep.
#
# The battery names are smallcrush, crush, bigcrush, alphabit and rabbit. The
# last two consume a fixed number of bits rather than running to a fixed size:
# --bits=<n> sets it, and campaign.jl's default (2^30) applies otherwise. Note
# that a campaign's `bits` suite runs alphabit and rabbit whatever --battery
# says, since Crush and BigCrush were never built for an integer stream; pass
# --bits-battery to campaign.jl directly to change that.
#
# A campaign need not run the whole matrix. --suite=<suite>[,<suite>] keeps
# some of single, interleaved and bits, or asks for substreams, which the whole
# matrix leaves out; --shapes=<s>x<v>[,<s>x<v>] runs the interleaved and
# substreams suites in each of those shapes -- s streams, v values from each in
# turn -- instead of the default 64x1; --generator=<name>[,<name>] keeps some of
# the generators. All three are recorded next to the results,
# so that `status`, `stop` and the watchdog run against the same matrix without
# being told again.
#
#   --skip-calibration    with --jobs=N, start without first running one job
#                         alone to measure it. For BigCrush that one job takes
#                         hours, during which every other core waits.
#
# Supervisors. A systemd user unit is the best of them -- it restarts a driver
# that died and comes back after a reboot -- but it only outlives your login if
# `loginctl enable-linger` has been run for your account, which on a shared
# machine is the administrator's call. Where that is refused,
# --supervisor=detached runs the driver in a session of its own (setsid, nohup,
# a pidfile), and --watchdog recovers what systemd was giving you:
#
#     ./scripts/run-campaign.sh run --battery=crush --supervisor=detached --watchdog
#
#   --supervisor=auto     systemd if linger is on or can be turned on, else
#                         detached (the default)
#   --supervisor=tmux     a detached tmux session you can attach to -- but only
#                         where tmux leaves its panes in your login session.
#                         A tmux built with systemd support moves them into the
#                         user manager, which without linger stops, and kills
#                         them, when your last session ends; `start` checks,
#                         and refuses.
#   --watchdog            a user crontab entry that restarts the campaign if it
#                         is not running, and at reboot. Needs no privileges,
#                         and `stop` removes it again.
#
# Suspects. A summary report singles out every p-value outside [0.001, 0.999],
# which is a printing threshold and not a verdict: across a campaign a handful
# of those are expected from chance alone. `replay` re-applies just those tests,
# on streams the original run never drew from, which is what tells a fluke from
# a failure. See docs/src/validation.md for the rule it serves.
#
# Kerberos. Where the checkout or the results sit on a krb5-secured filesystem
# -- an NFS home, at most sites -- those files answer to a ticket rather than to
# your uid, and a ticket is measured in hours while a campaign is measured in
# days. `prepare` says so, and gives the campaign a copy of your ticket that
# outlives your login, renewed by a krenew of its own; `stop` destroys it.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly TEMPLATE="$SCRIPT_DIR/testu01/randomdatastreams-campaign.service"

BATTERY=""
BITS=""                     # size for the bit-level batteries; campaign.jl's default applies when empty
BITS_OPT=""                 # "--bits=N" when BITS is set, empty otherwise
SUITE=""                    # suites to run; campaign.jl's default (all) applies when empty
SHAPES=""                   # interleaved shapes; campaign.jl's default (64x1) applies when empty
GENERATOR=""                # generators to run; campaign.jl's default (all) applies when empty
MATRIX_OPT=""               # the three above as campaign.jl options, recorded in $OUT/matrix
SKIP_CALIBRATION=0
OUT=""
JOBS=""
SKIP_BENCHMARK=0
ALLOW_DIRTY=0
SUPERVISOR=""
WATCHDOG=0
REPS=3
DRY_RUN=0

# tmux keeps its socket under /run/user/$UID by default, and that directory is
# created for a login session and removed when the last one ends -- precisely
# the case this script exists to handle. Put the socket somewhere that survives
# instead, so a session started from cron, from a login shell or from a later
# reconnection is always the same session.
export TMUX_TMPDIR="${TMUX_TMPDIR:-$HOME/.cache/randomdatastreams-tmux}"

# ------------------------------------------------------------------ utilities

say()  { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# `producer | grep -q pattern` is a race under pipefail: grep -q exits at the
# first match, the producer may still be writing, and the SIGPIPE it gets fails
# the whole pipeline -- so a line that is there reads as absent, now and then.
# That is how GNU time went "not found" while it sat in ~/.local/bin. This
# reads its input to the end.
matches() { grep "$@" >/dev/null; }

usage() {
    # the header block, however long it grows
    sed -n '3,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

unit_name() { echo "randomdatastreams-$BATTERY"; }
unit_file() { echo "$HOME/.config/systemd/user/$(unit_name).service"; }
session()   { echo "randomdatastreams-$BATTERY"; }
sup_file()  { echo "$OUT/supervisor"; }
pid_file()  { echo "$OUT/driver.pid"; }      # the detached supervisor's driver
matrix_file() { echo "$OUT/matrix"; }
launcher()  { echo "$OUT/campaign-command.sh"; }
# Not campaign.log: campaign.jl writes that one itself, through note(), while
# the tmux and detached supervisors send the same stdout here. Aimed at a single
# file the two of them wrote every line twice, which `status` then showed back
# doubled. This file earns its place rather than being dropped -- it catches
# what never reaches note(): a Julia stack trace, a bash error out of the
# launcher, anything the driver said on its way down.
runlog()    { echo "$OUT/supervisor.log"; }
cron_tag()  { echo "# randomdatastreams-campaign $BATTERY"; }

parse_args() {
    for a in "$@"; do
        case "$a" in
            --battery=*)      BATTERY="${a#*=}" ;;
            --out=*)          OUT="${a#*=}" ;;
            --jobs=*)         JOBS="${a#*=}" ;;
            --skip-benchmark) SKIP_BENCHMARK=1 ;;
            --skip-calibration) SKIP_CALIBRATION=1 ;;
            --suite=*)        SUITE="${a#*=}" ;;
            --shapes=*)       SHAPES="${a#*=}" ;;
            --generator=*)    GENERATOR="${a#*=}" ;;
            --allow-dirty)    ALLOW_DIRTY=1 ;;
            --supervisor=*)   SUPERVISOR="${a#*=}" ;;
            --watchdog)       WATCHDOG=1 ;;
            --reps=*)         REPS="${a#*=}" ;;
            --bits=*)         BITS="${a#*=}" ;;
            --dry-run)        DRY_RUN=1 ;;
            -h|--help)        usage 0 ;;
            *)                die "unknown option: $a" ;;
        esac
    done
    case "$BATTERY" in
        smallcrush|crush|bigcrush|alphabit|rabbit) ;;
        "") die "--battery is required: smallcrush, crush, bigcrush, alphabit or rabbit" ;;
        *)  die "unknown battery: $BATTERY" ;;
    esac
    OUT="${OUT:-$HOME/$BATTERY-results}"
    if [ -n "$BITS" ]; then BITS_OPT="--bits=$BITS"; fi
    if [ -n "$SUITE" ]; then MATRIX_OPT="--suite=$SUITE"; fi
    if [ -n "$SHAPES" ]; then MATRIX_OPT="${MATRIX_OPT:+$MATRIX_OPT }--shapes=$SHAPES"; fi
    if [ -n "$GENERATOR" ]; then MATRIX_OPT="${MATRIX_OPT:+$MATRIX_OPT }--generator=$GENERATOR"; fi
    resolve_matrix

    case "$SUPERVISOR" in
        ""|auto|systemd|detached|tmux) ;;
        *) die "unknown supervisor: $SUPERVISOR (auto, systemd, detached, tmux)" ;;
    esac
}

# ------------------------------------------------------------------- checks

check_environment() {
    say "Checking the environment"

    command -v julia >/dev/null || die "julia is not on PATH"
    info "julia:  $(julia --version)"

    [ -f "$TEMPLATE" ] || die "missing unit template: $TEMPLATE"

    git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 \
        || die "$REPO_ROOT is not a git checkout -- the scripts need one, and
       every result file records the commit that produced it. Clone the
       repository rather than using an installed copy of the package."

    local commit dirty
    commit=$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)
    dirty=$(git -C "$REPO_ROOT" status --porcelain)
    info "commit: $commit"

    if [ -n "$dirty" ]; then
        if [ "$ALLOW_DIRTY" -eq 1 ]; then
            info "tree:   DIRTY (results will be marked so; you passed --allow-dirty)"
        else
            printf '\nerror: the working tree has uncommitted changes, so the commit\n' >&2
            printf '       recorded in every result file would not describe the code\n' >&2
            printf '       that ran. What is dirty:\n\n' >&2
            printf '%s\n' "$dirty" | sed 's/^/         /' >&2
            printf '\n       ` M` is a tracked file you or a tool modified; `??` is an\n' >&2
            printf '       untracked file that is not in .gitignore. Running Pkg inside\n' >&2
            printf '       scripts/*/ or building the docs is the usual cause.\n\n' >&2
            printf '       Fix it one of these ways:\n' >&2
            printf '         git diff                       # see what changed\n' >&2
            printf '         git checkout -- <file>         # discard a tool edit\n' >&2
            printf '         git stash                      # park real work\n' >&2
            printf '         %s run --battery=%s --allow-dirty   # exploratory only\n' \
                   "$0" "$BATTERY" >&2
            exit 1
        fi
    else
        info "tree:   clean"
    fi

    check_kerberos
    resolve_supervisor
}

# ------------------------------------------------------ Kerberos-backed files

have() { command -v "$1" >/dev/null 2>&1; }

# A krb5-secured mount answers to a Kerberos ticket, not to a uid. When the
# ticket lapses every process you own loses the directory at once -- the driver
# in the middle of writing a result, the watchdog trying to chdir into it before
# it can even start -- and none of that depends on whether anyone is logged in.
# Tickets last hours and campaigns last days, so this is not a corner case; it
# is what happens by default to a sweep left to run.
#
# The path need not exist yet -- `--out` usually does not on a first `prepare`
# -- so the question is asked of the nearest directory that does.
krb5_path() {
    have findmnt || return 1
    local p=$1 opts
    while [ ! -e "$p" ] && [ "$p" != / ]; do p=$(dirname "$p"); done
    opts=$(findmnt -T "$p" -no OPTIONS 2>/dev/null) || return 1
    case ",$opts," in *,sec=krb5*) return 0 ;; *) return 1 ;; esac
}

# Prefix for the driver command in the launcher: empty, or krenew and its
# arguments. Set by check_kerberos, consumed by write_launcher.
KRENEW=""

# The cache the campaign runs on, which is not the one you logged in with.
#
# Where every login gets a cache of its own -- /tmp/krb5cc_<uid>_XXXXXX, made by
# sshd or pam_sss and destroyed at logout -- a krenew renewing yours dies when
# you disconnect, and so does a driver whose krenew inherited it: "No
# credentials cache found", and the sweep stops. cron, for its part, never sees
# a KRB5CCNAME and looks for /tmp/krb5cc_<uid>, which nobody made. So the
# campaign gets a cache of its own, copied from yours and renewed by a krenew
# that no login owns. The NFS client does not need to be told: rpc.gssd takes
# any cache in /tmp that you own and whose name starts with krb5cc_, which is
# also why one login can keep the mount readable after another has gone.
campaign_cc()    { echo "/tmp/krb5cc_$(id -u)_randomdatastreams-$BATTERY"; }
krenew_pidfile() { echo "$HOME/.cache/randomdatastreams/krenew-$BATTERY.pid"; }

# Set by check_kerberos when the campaign has a cache of its own; consumed by
# write_launcher, which names it there.
CAMPAIGN_CC=""
KERBEROS_CHECKED=0

krenew_running() {
    local pid
    pid=$(cat "$(krenew_pidfile)" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# The file behind the cache klist reports, or failure when it is not a file:
# a KEYRING: or KCM: cache cannot be copied, and usually outlives a login.
current_cc_file() {
    local listing cc
    listing=$(LC_ALL=C klist 2>/dev/null) || return 1
    cc=${listing#*Ticket cache: }
    cc=${cc%%$'\n'*}
    case "$cc" in
        FILE:/*) cc=${cc#FILE:} ;;
        /*)      ;;
        *)       return 1 ;;
    esac
    [ -f "$cc" ] && printf '%s\n' "$cc"
}

# Quiet, and a no-op when one is already running, so that the watchdog can
# call it from cron to bring back a krenew that died. -K is in minutes, and the
# interval only has to be shorter than the ticket's lifetime, which is hours.
# -t is AFS's, not ours, and krenew treats it as an error when AKLOG is unset.
start_krenew() {
    krenew_running && return 0
    mkdir -p "$(dirname "$(krenew_pidfile)")"
    krenew -k "FILE:$(campaign_cc)" -K 30 -b -p "$(krenew_pidfile)"
}

# Refreshes the campaign's cache from yours, and makes sure something renews
# it. Fails when yours is not a file, and check_kerberos falls back to renewing
# the cache you have.
setup_campaign_cc() {
    local src cc tmp
    cc=$(campaign_cc)
    src=$(current_cc_file) || return 1
    if [ "$src" != "$cc" ]; then
        tmp=$(mktemp "$cc.XXXXXX")          # mode 600, like any ticket cache
        cp "$src" "$tmp"
        mv -f "$tmp" "$cc"
    fi
    start_krenew || die "krenew would not start on $cc"
    CAMPAIGN_CC=$cc
    export KRB5CCNAME="FILE:$cc"
}

# For `stop`: the cache existed only for the campaign, so it goes with it.
release_campaign_cc() {
    local cc
    cc=$(campaign_cc)
    if krenew_running; then
        kill "$(cat "$(krenew_pidfile)")" 2>/dev/null || true
    fi
    rm -f "$(krenew_pidfile)"
    if [ -f "$cc" ]; then
        kdestroy -c "FILE:$cc" 2>/dev/null || rm -f "$cc"
        info "destroyed the campaign's ticket cache, $cc"
    fi
}

# Nothing here parses a ticket's expiry. The format is a locale's business and
# krenew already knows the arithmetic; all that is asked is whether a ticket
# exists now and whether it may be renewed later.
check_kerberos() {
    # `run` is `prepare` then `start`, and both come through here
    [ "$KERBEROS_CHECKED" -eq 1 ] && return 0
    KERBEROS_CHECKED=1

    local guarded=()
    krb5_path "$REPO_ROOT" && guarded+=("the checkout, $REPO_ROOT")
    krb5_path "$OUT"       && guarded+=("the results, $OUT")
    [ "${#guarded[@]}" -gt 0 ] || return 0

    say "Kerberos"
    local p
    for p in "${guarded[@]}"; do info "krb5-secured: $p"; done

    have klist || die "these files need a Kerberos ticket, and klist is not
       installed, so this script cannot tell whether you have one. Install the
       Kerberos client tools, or put the checkout and --out on a filesystem
       that does not need a ticket."

    if ! klist -s 2>/dev/null; then
        die "no valid Kerberos ticket, so the campaign would lose these files
       as soon as it started. Run 'kinit' and try again."
    fi

    if ! LC_ALL=C klist 2>/dev/null | matches 'renew until'; then
        info "WARNING: your ticket cannot be renewed, so the campaign will lose"
        info "         these files when it expires -- with no error you will see"
        info "         until you come back to a half-finished sweep. Either run"
        info "         the campaign from a filesystem that needs no ticket:"
        info "           $0 prepare --battery=$BATTERY --out=/var/tmp/\$USER-$BATTERY"
        info "         (the checkout too), or ask for a renewable ticket."
        return 0
    fi

    if ! have krenew; then
        info "WARNING: your ticket is renewable but krenew is not installed, so"
        info "         nothing will renew it. Install it (kstart), or renew by"
        info "         hand for as long as the campaign runs."
        return 0
    fi

    if setup_campaign_cc; then
        info "the campaign has a ticket cache of its own, $CAMPAIGN_CC,"
        info "copied from yours and renewed every 30 minutes by a krenew that no"
        info "login owns, so logging out does not take it away. 'stop' destroys it."
    else
        KRENEW="$(command -v krenew) -K 30 -- "
        info "your ticket cache is not a file, so the campaign cannot have a copy"
        info "of its own: the driver will run under krenew on the cache you have,"
        info "renewing every 30 minutes. Check that it outlives a logout before"
        info "leaving a sweep to run."
    fi
    info "note: renewal has its own deadline -- 'klist -f' shows 'renew until'."
    info "      A fresh kinit before a long sweep restarts that clock."
}

# ------------------------------------------------------------- the supervisor

systemd_usable() { systemctl --user show-environment >/dev/null 2>&1; }
linger_on()      { [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; }

# Whether logging out kills what you leave behind. With KillUserProcesses=yes a
# detached driver dies at logout exactly as an unlingered service does, so
# neither detached nor tmux is a workaround there and the campaign needs linger
# after all -- worth knowing before a two-week sweep, not after.
kill_user_processes() {
    local v
    v=$(busctl get-property org.freedesktop.login1 /org/freedesktop/login1 \
            org.freedesktop.login1.Manager KillUserProcesses 2>/dev/null) || return 2
    case "$v" in *true) return 0 ;; *false) return 1 ;; *) return 2 ;; esac
}

# Sets SUPERVISOR when it is empty or `auto`, and checks the chosen one exists.
#
# The part of the matrix a campaign runs is recorded next to its results, as its
# supervisor is, and for the same reason: `status` has to count what is left of
# the matrix that was started, and the watchdog's `ensure` passes only --battery
# and --out. Asking for another matrix in the same directory is refused rather
# than mixed in with what is there.
resolve_matrix() {
    local recorded=""
    [ -f "$(matrix_file)" ] && recorded=$(cat "$(matrix_file)")
    if [ -z "$MATRIX_OPT" ]; then
        MATRIX_OPT="$recorded"
    elif [ -f "$(matrix_file)" ] && [ "$MATRIX_OPT" != "$recorded" ]; then
        die "$OUT holds a campaign of ${recorded:-the whole matrix}, not $MATRIX_OPT -- give this one another --out"
    fi
}

# A campaign that was started detached must still be visible to `status` and
# stoppable by `stop` when those are called without the flag, so the choice is
# recorded next to the results and read back here. Re-deriving it instead would
# quietly look at the wrong supervisor on a machine where both are available.
resolve_supervisor() {
    local why=""
    if { [ -z "$SUPERVISOR" ] || [ "$SUPERVISOR" = auto ]; } && [ -f "$(sup_file)" ]; then
        SUPERVISOR=$(cat "$(sup_file)")
        why="recorded in $(sup_file)"
    fi

    if [ -z "$SUPERVISOR" ] || [ "$SUPERVISOR" = auto ]; then
        if systemd_usable && linger_on; then
            SUPERVISOR=systemd
            info "supervisor: systemd (linger is already enabled)"
        elif systemd_usable && loginctl enable-linger "$USER" 2>/dev/null; then
            SUPERVISOR=systemd
            info "supervisor: systemd (linger enabled for $USER just now)"
        elif have setsid; then
            # Not tmux, which was the fallback until a tmux built with systemd
            # support took a sweep down with the last login (see in_user_manager).
            SUPERVISOR=detached
            info "supervisor: detached -- systemd was not usable without linger,"
            info "            which only an administrator can grant here"
        else
            die "no usable supervisor: systemd needs linger (ask an administrator
       for 'loginctl enable-linger $USER'), and setsid is not installed
       (util-linux). Install it, or run campaign.jl in the foreground."
        fi
    else
        info "supervisor: $SUPERVISOR (${why:-you chose it})"
    fi

    case "$SUPERVISOR" in
        systemd)
            systemd_usable || die "systemctl --user is not available in this session"
            linger_on || info "WARNING: linger is off, so this service stops at logout.
       Ask for 'loginctl enable-linger $USER', or use --supervisor=detached."
            ;;
        detached) have setsid || die "setsid is not installed (util-linux)" ;;
        tmux)     have tmux || die "tmux is not installed" ;;
    esac

    if [ "$SUPERVISOR" != systemd ]; then
        local kup=0
        kill_user_processes || kup=$?
        if [ "$kup" -eq 0 ]; then
            info "WARNING: logind has KillUserProcesses=yes on this machine, so a"
            info "         $SUPERVISOR driver is killed at logout too."
            info "         Only linger fixes that -- ask an administrator."
        elif [ "$kup" -eq 2 ]; then
            info "note: could not read logind's KillUserProcesses setting. If it is"
            info "      'yes' on this machine, a $SUPERVISOR driver dies at logout."
        fi
        info "note: $SUPERVISOR does not survive a reboot and does not restart a"
        info "      driver that died. --watchdog covers both, through cron."
    fi
}

# --------------------------------------------------------------- benchmarking

run_benchmark() {
    if [ "$SKIP_BENCHMARK" -eq 1 ]; then
        say "Skipping the throughput benchmark (--skip-benchmark)"
        return
    fi

    say "Throughput benchmark, before the machine gets busy"
    local out="$HOME/throughput-$(hostname)-$(date +%Y%m%d).txt"

    # Pin to one core: on a hybrid CPU the scheduler will otherwise move the
    # process between performance and efficiency cores mid-run, and the minimum
    # reported comes from whichever core happened to be fastest.
    local pin=()
    if command -v taskset >/dev/null; then
        pin=(taskset -c 0)
        info "pinned to CPU 0 -- note in the write-up which core type that is"
    else
        info "taskset not found: not pinned. On a hybrid CPU, treat the numbers"
        info "as indicative only."
    fi

    "${pin[@]}" julia -O3 "$SCRIPT_DIR/benchmarks/throughput.jl" | tee "$out"
    info "written to $out"
}

# --------------------------------------------------------------- calibration

physical_cores() {
    if command -v lscpu >/dev/null; then
        local n
        n=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
        [ "${n:-0}" -gt 0 ] && { echo "$n"; return; }
    fi
    getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

# GNU time, wherever it happens to live. Only one line of its -v output is
# wanted -- the peak RSS that sizes --jobs -- and only GNU prints it: BSD time,
# which is what /usr/bin/time is on macOS, spells that -l, and the shell's own
# `time` is a keyword that measures no memory at all. So candidates are probed
# rather than trusted by name, and a copy someone installed without root counts
# for as much as the system one. Prints the path, or fails if there is none.
gnu_time() {
    local c
    for c in /usr/bin/time "$HOME/.local/bin/time" \
             "$(command -v gtime 2>/dev/null || true)" \
             "$(command -v time 2>/dev/null || true)"; do
        case "$c" in /*) ;; *) continue ;; esac   # empty, or the shell keyword
        [ -x "$c" ] || continue
        if "$c" -v true 2>&1 | matches 'Maximum resident set size'; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

# Sets JOBS, unless the caller already did.
calibrate() {
    if [ "$SKIP_CALIBRATION" -eq 1 ]; then
        [ -n "$JOBS" ] || die "--skip-calibration needs --jobs=N: without a measured job there is nothing to size it from"
        say "Skipping the calibration (--skip-calibration)"
        info "jobs: $JOBS (you set it explicitly)"
        return
    fi

    say "Calibrating: one job of $BATTERY, measured"

    local log="$HOME/calibration-$BATTERY.txt"
    local cores mem_kb peak_kb
    cores=$(physical_cores)
    mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)

    local timer=""
    timer=$(gnu_time) || true

    if [ -n "$timer" ]; then
        [ "$timer" = /usr/bin/time ] || info "GNU time: $timer"
        "$timer" -v julia "$SCRIPT_DIR/testu01/campaign.jl" \
            --battery="$BATTERY" --out="$OUT" $BITS_OPT $MATRIX_OPT --calibrate 2>&1 | tee "$log"
        peak_kb=$(awk '/Maximum resident set size/ {print $NF}' "$log" | tail -1)
    else
        info "GNU time not found, so the peak RSS cannot be measured and --jobs"
        info "falls back to cores/4 -- on a large machine, a quarter of the"
        info "campaign it could be running. Either pass --jobs=N yourself, or"
        info "build GNU time into your home directory (no root needed):"
        info "  curl -LO https://ftp.gnu.org/gnu/time/time-1.9.tar.gz"
        info "  tar xzf time-1.9.tar.gz && cd time-1.9"
        info "  ./configure --prefix=\$HOME/.local && make && make install"
        julia "$SCRIPT_DIR/testu01/campaign.jl" \
            --battery="$BATTERY" --out="$OUT" $BITS_OPT $MATRIX_OPT --calibrate 2>&1 | tee "$log"
        peak_kb=""
    fi

    info "calibration log: $log"

    if [ -n "$JOBS" ]; then
        info "jobs: $JOBS (you set it explicitly)"
        return
    fi

    if [ -n "${peak_kb:-}" ] && [ "${peak_kb:-0}" -gt 0 ] && [ "$mem_kb" -gt 0 ]; then
        # Leave a fifth of memory to the rest of the machine, and never exceed
        # the physical core count: these jobs are CPU-bound, so hyperthreads buy
        # little and cost cache.
        local by_mem
        by_mem=$(( mem_kb * 8 / 10 / peak_kb ))
        JOBS=$(( by_mem < cores ? by_mem : cores ))
        [ "$JOBS" -lt 1 ] && JOBS=1
        info "peak RSS per job: $(( peak_kb / 1024 )) MiB"
        info "physical cores:   $cores"
        info "memory:           $(( mem_kb / 1024 / 1024 )) GiB"
        info "jobs chosen:      $JOBS   (min of cores and 80% of memory / peak)"
    else
        JOBS=$(( cores / 4 ))
        [ "$JOBS" -lt 1 ] && JOBS=1
        info "no memory measurement: falling back to cores/4 = $JOBS"
    fi
}

# ------------------------------------------------------------------ launching

# The exact command, written down once. systemd gets it through ExecStart; the
# detached and tmux supervisors run this file. Having it on disk also means
# `status` can show what is running, and a watchdog can restart it without
# re-deriving --jobs.
#
# cron gives a near-empty PATH, so the interpreter is resolved here and now.
write_launcher() {
    local julia
    julia=$(command -v julia)
    mkdir -p "$OUT"

    local krb_note="" krb_env=""
    if [ -n "$CAMPAIGN_CC" ]; then
        krb_note="# These files are on a krb5 filesystem. The campaign's own ticket cache,
# renewed by its own krenew, outlives the login that started it."
        krb_env="export KRB5CCNAME=\"FILE:$CAMPAIGN_CC\""
    elif [ -n "$KRENEW" ]; then
        krb_note="# krenew keeps the Kerberos ticket alive: these files are on a krb5
# filesystem, and without it the driver loses them mid-sweep."
    fi

    cat > "$(launcher)" <<EOF
#!/usr/bin/env bash
# Generated by run-campaign.sh on $(date -Is). Re-run it rather than editing.
set -euo pipefail
cd "$REPO_ROOT"
export CAMPAIGN_OUT="$OUT"
# Nice=10 and IOSchedulingClass=idle in the systemd unit; the same politeness
# here, so a campaign never competes with interactive work or a benchmark.
$krb_note
$krb_env
exec nice -n 10 ionice -c 3 ${KRENEW}"$julia" --startup-file=no \\
     scripts/testu01/campaign.jl --battery=$BATTERY --jobs=$JOBS --out="$OUT" $BITS_OPT $MATRIX_OPT
EOF
    chmod +x "$(launcher)"
    printf '%s\n' "$SUPERVISOR" > "$(sup_file)"
    printf '%s\n' "$MATRIX_OPT" > "$(matrix_file)"
    info "launcher: $(launcher)"
}

# --- is it running? -----------------------------------------------------------

is_running() {
    case "$SUPERVISOR" in
        systemd)  systemctl --user is-active --quiet "$(unit_name)" ;;
        detached) driver_alive ;;
        tmux)     tmux has-session -t "=$(session)" 2>/dev/null ;;
        *)        return 1 ;;      # no supervisor resolved yet: nothing is running
    esac
}

# The pidfile names the driver, but a pid is reused once its process is gone,
# so the process found there must still be a campaign driver to count.
driver_alive() {
    local pid
    pid=$(cat "$(pid_file)" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    ps -p "$pid" -o args= 2>/dev/null | matches -F "campaign.jl"
}

# Whether a process lives under the systemd user manager, user@<uid>.service,
# rather than in the scope of the login or cron job that started it.
#
# tmux built with systemd support moves each pane it spawns into a scope of its
# own under that manager, so that the pane outlives the terminal it came from.
# Without linger the manager is stopped when your last session ends, and every
# scope under it with it. KillUserProcesses=no does not help: it spares what
# stays in a session's scope, and the pane has left it. On 2026-10-16 that took
# a BigCrush sweep's driver and its 24 jobs, five hours in, with the last
# logout -- exactly what tmux was there to survive.
in_user_manager() {
    [ -r "/proc/$1/cgroup" ] && matches -E '/user@[0-9]+\.service/' "/proc/$1/cgroup"
}

# --- start --------------------------------------------------------------------

start_supervised() {
    case "$SUPERVISOR" in
        systemd)
            [ -f "$(unit_file)" ] || die "no unit for $BATTERY yet -- run 'prepare' first"
            systemctl --user start "$(unit_name)"
            ;;
        detached)
            [ -x "$(launcher)" ] || die "no launcher at $(launcher) -- run 'prepare' first"
            # A session of its own, so no terminal's hangup reaches it, left in
            # the scope of the login or cron job that started it, which
            # KillUserProcesses=no spares when that session ends. The pid is
            # written by the process itself before it becomes the launcher, and
            # the launcher execs the driver, so the file names the driver --
            # and the leader of its process group -- whether or not setsid had
            # to fork.
            rm -f "$(pid_file)"
            setsid nohup bash -c 'echo $$ > "$1"; exec bash "$2"' _ "$(pid_file)" "$(launcher)" \
                >> "$(runlog)" 2>&1 < /dev/null &
            local waited=0
            while [ ! -s "$(pid_file)" ] && [ "$waited" -lt 100 ]; do
                sleep 0.1; waited=$((waited + 1))
            done
            [ -s "$(pid_file)" ] || die "the driver did not start -- see $(runlog)"
            ;;
        tmux)
            [ -x "$(launcher)" ] || die "no launcher at $(launcher) -- run 'prepare' first"
            # A unix socket path is limited to about 100 bytes, and tmux appends
            # roughly 20 to this directory. Over the limit it fails with
            # "File name too long", which says nothing about the cause.
            [ "${#TMUX_TMPDIR}" -lt 80 ] || die "TMUX_TMPDIR is too long for a unix
       socket ($TMUX_TMPDIR). Set it to something short, e.g.
       TMUX_TMPDIR=\$HOME/.cache/rds-tmux $0 start --battery=$BATTERY"
            mkdir -p "$TMUX_TMPDIR"; chmod 700 "$TMUX_TMPDIR"
            tmux new-session -d -s "$(session)" \
                 "bash '$(launcher)' 2>&1 | tee -a '$(runlog)'"
            # Asked of the pane itself, once tmux has placed it: whether this
            # tmux was built with systemd support is not something to guess.
            local pane=""
            sleep 1
            pane=$(tmux display-message -p -t "$(session):" '#{pane_pid}' 2>/dev/null) || true
            if [ -n "$pane" ] && in_user_manager "$pane" && ! linger_on; then
                tmux kill-session -t "=$(session)" 2>/dev/null || true
                die "this tmux moved the campaign into your systemd user manager,
       which without linger is stopped -- and the campaign with it -- when
       your last session ends. Stopped it again before it started a job.
       Use the detached supervisor instead:
         $0 start --battery=$BATTERY --supervisor=detached"
            fi
            ;;
    esac
}

# --- stop ---------------------------------------------------------------------

# SIGINT, not SIGKILL: the driver stops scheduling and waits for the jobs in
# flight, which is what makes an interrupted campaign cheap. The systemd unit
# says the same thing with KillSignal=SIGINT and TimeoutStopSec=6h.
stop_supervised() {
    case "$SUPERVISOR" in
        systemd)
            systemctl --user stop "$(unit_name)"
            ;;
        detached)
            if ! driver_alive; then
                info "no driver running (no live process in $(pid_file))"
                rm -f "$(pid_file)"
                return
            fi
            local pid waited=0
            pid=$(cat "$(pid_file)")
            # To the whole process group, which the driver leads and its jobs
            # belong to: what Ctrl-C does in a terminal, and what the tmux
            # supervisor's send-keys amounts to.
            kill -INT -- "-$pid"
            info "SIGINT sent; waiting for the jobs in flight (up to 6h)"
            while driver_alive && [ "$waited" -lt 21600 ]; do
                sleep 5; waited=$((waited + 5))
            done
            if driver_alive; then
                info "still running after 6h -- killing it"
                kill -KILL -- "-$pid" 2>/dev/null || true
            fi
            rm -f "$(pid_file)"
            ;;
        tmux)
            if ! is_running; then
                info "no tmux session named $(session)"
                return
            fi
            # the session name, not "=name": that form matches a session for
            # has-session but does not resolve to a pane for send-keys
            tmux send-keys -t "$(session):" C-c
            info "SIGINT sent; waiting for the jobs in flight (up to 6h)"
            local waited=0
            while is_running && [ "$waited" -lt 21600 ]; do
                sleep 5; waited=$((waited + 5))
            done
            if is_running; then
                info "still running after 6h -- killing the session"
                tmux kill-session -t "=$(session)"
            fi
            ;;
    esac
}

# ------------------------------------------------------------------ the unit

install_unit() {
    say "Installing the systemd user unit"

    mkdir -p "$(dirname "$(unit_file)")"

    sed -e "s|^WorkingDirectory=.*|WorkingDirectory=$REPO_ROOT|" \
        -e "s|^Environment=CAMPAIGN_OUT=.*|Environment=CAMPAIGN_OUT=$OUT|" \
        -e "s|^ExecStart=.*|ExecStart=/usr/bin/env julia --startup-file=no scripts/testu01/campaign.jl --battery=$BATTERY --jobs=$JOBS --out=$OUT $BITS_OPT $MATRIX_OPT|" \
        -e "/^ *--battery=bigcrush/d" \
        -e "s|^Description=.*|Description=RandomDataStreams.jl $BATTERY validation campaign|" \
        -e "s|^# --- edit these three.*|# --- generated by scripts/run-campaign.sh; re-run it rather than editing ---|" \
        "$TEMPLATE" > "$(unit_file)"

    # `linger` is the line that matters: without it systemd kills user services
    # at logout, which is exactly what this campaign has to survive.
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
        info "enabling linger so the campaign survives logout"
        loginctl enable-linger "$USER" \
            || info "could not enable linger -- ask an administrator, or the"
        info "campaign will stop when you disconnect."
    else
        info "linger already enabled"
    fi

    systemctl --user daemon-reload
    info "unit: $(unit_file)"
    grep '^ExecStart=' "$(unit_file)" | sed 's/^/    /'
}

# -------------------------------------------------------------------- watchdog

# What systemd gave for free and tmux does not: come back after a reboot, and
# after a driver that died. cron needs no privileges and runs outside any login
# session, so it works exactly where linger is refused. The campaign resumes
# from summary.tsv, so restarting it costs only the jobs that were in flight.
watchdog_lines() {
    local self="$REPO_ROOT/scripts/run-campaign.sh"
    local args="--battery=$BATTERY --out=$OUT --supervisor=$SUPERVISOR"
    printf '%s\n' \
        "@reboot $self ensure $args >/dev/null 2>&1 $(cron_tag)" \
        "*/15 * * * * $self ensure $args >/dev/null 2>&1 $(cron_tag)"
}

crontab_without_ours() {
    crontab -l 2>/dev/null | grep -vF "$(cron_tag)" || true
}

# Never `crontab -l | ... | crontab -`: the two ends of that pipeline run at the
# same time, on the same table, and whether the reader finishes before the
# writer replaces it is not something to bet somebody's other cron jobs on.
# Build the whole new table first, then install it in one go. Callers pass it
# on stdin, and this reads to EOF before writing anything, so whoever produced
# the table has finished with it by the time the table is replaced.
#
# A table that came out empty usually means something went wrong upstream, and
# installing it would silently delete every other job the user has -- so it is
# refused. The one case where an empty table is exactly right is `stop` on a
# machine whose crontab holds nothing but our own two lines, which is the usual
# state of a machine dedicated to a campaign. Only remove_watchdog knows it is
# in that case, so only it passes allow_empty; without that, stop could never
# take back the watchdog start had installed.
replace_crontab() {
    local allow_empty="${1:-0}"
    local new
    new=$(mktemp)
    cat > "$new"
    if [ ! -s "$new" ] && [ "$allow_empty" != 1 ] && [ -n "$(crontab -l 2>/dev/null)" ]; then
        rm -f "$new"
        die "refusing to install an empty crontab over a non-empty one"
    fi
    crontab "$new"
    rm -f "$new"
}

backup_crontab() {
    mkdir -p "$OUT"
    crontab -l > "$OUT/crontab.backup" 2>/dev/null || true
    [ -s "$OUT/crontab.backup" ] && info "crontab backed up to $OUT/crontab.backup"
    return 0
}

install_watchdog() {
    have crontab || die "--watchdog needs crontab, which is not installed"
    say "Installing the watchdog in your user crontab"
    backup_crontab
    # The table is staged in replace_crontab's own temporary file rather than
    # under $OUT: a die between writing a staging file there and removing it
    # leaves it behind, where the next `collect` bundles it with the results.
    replace_crontab 0 < <(crontab_without_ours; watchdog_lines)
    watchdog_lines | sed 's/^/    /'
    info "your other cron jobs are untouched; removed again by:"
    info "  $0 stop --battery=$BATTERY"

    # cron chdirs into $HOME before running anything, so an expired ticket stops
    # it before it can read this script, and the only trace is a line in the
    # system's cron log that says "chdir failed". The campaign's own krenew
    # covers that, since it runs whether or not the driver does; only without
    # one -- a ticket cache that is not a file -- does the watchdog need a
    # ticket kept some other way.
    if [ -z "$CAMPAIGN_CC" ] && { krb5_path "$OUT" || krb5_path "$REPO_ROOT"; }; then
        info ""
        info "NOTE: cron cannot enter a krb5 filesystem without a live ticket,"
        info "      so the watchdog stops working the moment yours lapses. Keep"
        info "      one renewed independently of the campaign:"
        info "        krenew -K 30 -b"
    fi
}

remove_watchdog() {
    have crontab || return 0
    crontab -l 2>/dev/null | matches -F "$(cron_tag)" || return 0
    info "removing the watchdog from your crontab"
    backup_crontab
    # Empty is allowed here, and is the common case: on a machine set aside for
    # a campaign, our two lines are the whole crontab.
    replace_crontab 1 < <(crontab_without_ours)
}

# ------------------------------------------------------------------ commands

cmd_prepare() {
    check_environment
    run_benchmark
    calibrate
    write_launcher
    if [ "$SUPERVISOR" = systemd ]; then
        install_unit
    fi

    say "Ready"
    info "output directory: $OUT"
    info "supervisor:       $SUPERVISOR"
    cat <<EOF

    Start it with:

        $0 start --battery=$BATTERY --supervisor=$SUPERVISOR

EOF
}

cmd_start() {
    resolve_supervisor
    is_running && die "already running -- '$0 status --battery=$BATTERY' to see it"
    # a `stop` destroyed the campaign's ticket cache, and the launcher names it
    check_kerberos
    start_supervised
    # `start --supervisor=...` may name another supervisor than `prepare` did,
    # as the tmux refusal tells you to; status, stop and the watchdog must then
    # look for this one.
    printf '%s\n' "$SUPERVISOR" > "$(sup_file)"
    if [ "$WATCHDOG" -eq 1 ]; then
        install_watchdog
    fi

    say "Started"
    case "$SUPERVISOR" in
        systemd)  systemctl --user --no-pager status "$(unit_name)" | head -12 || true ;;
        detached) info "driver: pid $(cat "$(pid_file)"), in a session of its own"
                  info "        follow with: tail -f $(runlog)" ;;
        tmux)     info "session: $(session)   attach with: tmux attach -t $(session)" ;;
    esac
    if [ "$SUPERVISOR" != systemd ]; then
        info "log:     $(runlog)"
    fi

    cat <<EOF

    It now runs without you. Useful later:

        $0 status  --battery=$BATTERY     # how far along
        $0 collect --battery=$BATTERY     # bundle the results
        touch $OUT/STOP                   # stop cleanly, keeping what is running
        $0 stop    --battery=$BATTERY     # stop now, losing only jobs in flight

    Restarting resumes: the driver reads summary.tsv and does what is missing.

EOF
}

# Idempotent start, for the watchdog: do nothing if it is already running, if
# there is nothing prepared, if the campaign was stopped on purpose, or if it
# is finished. `touch <out>/STOP` is the documented clean stop, and a watchdog
# that undid it fifteen minutes later would make that instruction a lie. The
# driver writes <out>/DONE once every job of the matrix is in summary.tsv; the
# krenew is still revived until `stop`, since a replay may yet write there.
cmd_ensure() {
    resolve_supervisor >/dev/null 2>&1 || exit 0
    # Bring back the campaign's krenew if it died, while its ticket still holds:
    # the driver and the watchdog both depend on it, and nothing else renews it.
    if [ -f "$(campaign_cc)" ] && klist -s -c "FILE:$(campaign_cc)" 2>/dev/null; then
        start_krenew >/dev/null 2>&1 || true
    fi
    is_running && exit 0
    [ -x "$(launcher)" ] || exit 0
    [ -f "$OUT/STOP" ] && exit 0
    [ -f "$OUT/DONE" ] && exit 0
    start_supervised
}

cmd_status() {
    resolve_supervisor

    say "Supervisor ($SUPERVISOR)"
    case "$SUPERVISOR" in
        systemd)
            systemctl --user --no-pager status "$(unit_name)" 2>/dev/null | head -12 \
                || info "not running"
            ;;
        detached|tmux)
            local what="session $(session)"
            [ "$SUPERVISOR" = detached ] && what="driver (pidfile $(pid_file))"
            if is_running; then
                info "$what: running"
            else
                info "$what: not running"
            fi
            if [ -f "$(runlog)" ]; then
                info "last lines of $(runlog):"
                tail -5 "$(runlog)" | sed 's/^/      /'
            fi
            ;;
    esac

    if have crontab && crontab -l 2>/dev/null | matches -F "$(cron_tag)"; then
        info "watchdog: installed (restarts it if it stops, and at reboot)"
    fi

    say "Progress"
    julia "$SCRIPT_DIR/testu01/campaign.jl" \
        --battery="$BATTERY" --out="$OUT" $MATRIX_OPT --status || true

    if [ -f "$OUT/summary.tsv" ]; then
        say "Finished jobs: $(( $(wc -l < "$OUT/summary.tsv") - 1 ))"
        info "in $OUT/summary.tsv"
    fi
}

cmd_stop() {
    resolve_supervisor
    remove_watchdog          # or it would start the campaign again in 15 minutes
    stop_supervised
    release_campaign_cc
    say "Stopped. Everything already finished is recorded in $OUT/summary.tsv"
}

# Every run the campaign finished, asked again about the tests it singled out.
# replay.jl does the thinking -- which log in a run directory is the complete
# one, which tests were flagged, which streams to skip so the output is disjoint
# -- so this only has to walk the directories and keep going when one of them
# has nothing to say.
cmd_replay() {
    [ -d "$OUT/runs" ] || die "no runs directory at $OUT/runs -- has this campaign run?"

    say "Replaying what $BATTERY singled out"
    info "each flagged test re-applied $REPS time(s), on streams the original never used"
    [ "$DRY_RUN" -eq 1 ] && info "--dry-run: listing only, nothing will be run"

    local d name rc
    local ran=0 quiet=0 skipped=0
    for d in "$OUT"/runs/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        rc=0
        julia --project="$SCRIPT_DIR/testu01" "$SCRIPT_DIR/testu01/replay.jl" \
              --log="$d" --reps="$REPS" \
              $([ "$DRY_RUN" -eq 1 ] && echo --list) || rc=$?
        if [ "$rc" -ne 0 ]; then
            skipped=$((skipped + 1))
            info "skipped $name (replay.jl said why, above)"
        else
            ran=$((ran + 1))
        fi
    done

    say "Done"
    info "run directories read: $ran   skipped: $skipped"
    info "replays are indexed in each run directory's replays.tsv"
    cat <<EOF

    A p-value that stays extreme on independent output is the generator
    failing that test. One that scatters was the fluke the replay ruled out.
    Report the p-values themselves, not a verdict.

EOF
}

cmd_collect() {
    [ -d "$OUT" ] || die "no output directory at $OUT"

    local stamp bundle staging
    stamp=$(date +%Y%m%d-%H%M)
    bundle="$HOME/randomdatastreams-$BATTERY-$stamp.tar.gz"
    staging=$(mktemp -d)
    trap 'rm -rf "$staging"' RETURN

    # Everything a later session needs to write up the results: the batteries,
    # the timings, the calibration, and the commit they were all measured at.
    mkdir -p "$staging/results"
    cp -r "$OUT/." "$staging/results/"
    cp "$HOME"/throughput-*.txt "$staging/" 2>/dev/null || true
    cp "$HOME"/calibration-*.txt "$staging/" 2>/dev/null || true

    {
        echo "host:    $(hostname)"
        echo "date:    $(date -Is)"
        echo "battery: $BATTERY"
        echo "commit:  $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "julia:   $(julia --version)"
        command -v lscpu >/dev/null && lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'
    } > "$staging/environment.txt"

    tar -czf "$bundle" -C "$staging" .

    say "Bundled"
    info "$bundle  ($(du -h "$bundle" | cut -f1))"
}

# ---------------------------------------------------------------------- main

[ $# -ge 1 ] || usage 1
command="$1"; shift

case "$command" in -h|--help|help) usage 0 ;; esac

parse_args "$@"

case "$command" in
    prepare) cmd_prepare ;;
    start)   cmd_start ;;
    ensure)  cmd_ensure ;;
    run)     cmd_prepare; cmd_start ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    collect) cmd_collect ;;
    replay)  cmd_replay ;;
    -h|--help|help) usage 0 ;;
    *)       die "unknown command: $command" ;;
esac
