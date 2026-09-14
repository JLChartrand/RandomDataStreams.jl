# Which code produced a measurement.
#
# A BigCrush campaign runs for weeks. When its p-values are quoted in a paper,
# "which version of the package produced this?" is the first question a reviewer
# asks, and by then the runs cannot be repeated to find out. So every result file
# records the commit it came from, and whether the working tree was clean.
#
# Included by validate.jl, campaign.jl and throughput.jl, which live in
# different script environments and would otherwise each grow their own copy.

const _REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

"""
    provenance() -> (; commit, dirty, version)

`commit` is the short hash of `HEAD`, or `"unknown"` outside a git checkout (an
installed copy of the package, for instance). `dirty` says whether the working
tree had uncommitted changes, which makes `commit` insufficient to identify the
code. `version` comes from `Project.toml`.
"""
function provenance(repo::AbstractString = _REPO_ROOT)
    commit, dirty = "unknown", false
    try
        commit = strip(read(`git -C $repo rev-parse --short=12 HEAD`, String))
        dirty  = !isempty(strip(read(`git -C $repo status --porcelain`, String)))
    catch
        # not a checkout, or no git: leave the defaults, and say so rather than
        # inventing a hash
    end
    version = "unknown"
    try
        for line in eachline(joinpath(repo, "Project.toml"))
            m = match(r"^version\s*=\s*\"([^\"]+)\"", line)
            m === nothing || (version = m.captures[1]; break)
        end
    catch
    end
    return (commit = commit, dirty = dirty, version = version)
end

"One line naming the code, for a log or a report header."
function provenance_line(p = provenance())
    return string("RandomDataStreams v", p.version, " @ ", p.commit,
                  p.dirty ? " (WORKING TREE DIRTY -- results are not attributable" *
                            " to a commit)" : "")
end

"Loud warning when a long run is started from an unrecorded state."
function warn_if_dirty(p = provenance())
    p.dirty || return p
    println(stderr)
    println(stderr, "!! The working tree has uncommitted changes.")
    println(stderr, "!! Results produced now cannot be traced to a commit, which for a")
    println(stderr, "!! multi-week campaign means they cannot be defended in a paper.")
    println(stderr, "!! Commit first, or accept that the run is exploratory.")
    println(stderr)
    return p
end
