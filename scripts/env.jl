# Making a script run against the checkout it lives in, and proving that it did.
#
# The scripts each have their own environment, with the package listed as a
# dependency. That listing is not enough: `Pkg.instantiate()` will happily
# satisfy it from the General registry, so on a fresh clone -- where no
# Manifest exists yet -- a script measures the *released* version while
# `provenance.jl` stamps the current commit on every result it writes. The
# failure is silent, and it invalidates whatever campaign produced it.
#
# `Pkg.develop` fixes the resolution. Calling it unconditionally on every job is
# not an option either: a campaign runs its jobs as concurrent processes, and
# they would race writing the same Manifest. So the work is done only when the
# Manifest does not already point at this checkout, which makes the second and
# subsequent jobs free -- and `assert_checkout` verifies the outcome regardless
# of which path was taken.

using Pkg, TOML

"Repository root, from a script directory two levels below it."
repo_root(scriptdir::AbstractString) = abspath(joinpath(scriptdir, "..", ".."))

"""
`true` when the environment's Manifest already resolves the package to `root`
*and* covers everything the Project asks for.

The second half matters when a dependency is added to `Project.toml` later: the
Manifest still points at the checkout, so the resolution would be skipped, and
the script would then fail at `using` on a package that was never installed.
"""
function _manifest_current(scriptdir::AbstractString, root::AbstractString)
    manifest = joinpath(scriptdir, "Manifest.toml")
    isfile(manifest) || return false
    try
        parsed = TOML.parsefile(manifest)
        # Julia 1.6 manifests are flat; 1.7 and later nest everything under
        # "deps". Accept both, since the package supports 1.6.
        entries = get(parsed, "deps", parsed)
        entry = get(entries, "RandomDataStreams", nothing)
        entry === nothing && return false
        entry isa AbstractVector && (entry = first(entry))
        path = get(entry, "path", nothing)
        path === nothing && return false
        realpath(abspath(joinpath(scriptdir, path))) == realpath(root) || return false
        project = TOML.parsefile(joinpath(scriptdir, "Project.toml"))
        return all(haskey(entries, name) for name in keys(get(project, "deps", Dict())))
    catch
        return false          # unreadable or unexpected: redo the resolution
    end
end

"""
    ensure_checkout_env(scriptdir) -> root

Activate the script's environment and make `RandomDataStreams` resolve to the
checkout `scriptdir` belongs to. Cheap and side-effect free when a previous run
already arranged it, so concurrent jobs do not fight over the Manifest.
"""
function ensure_checkout_env(scriptdir::AbstractString)
    root = repo_root(scriptdir)
    Pkg.activate(scriptdir)
    if !_manifest_current(scriptdir, root)
        Pkg.develop(path = root)
        Pkg.instantiate()
    end
    return root
end

"""
    assert_checkout(mod, scriptdir)

Fail loudly if `mod` was loaded from anywhere but the checkout. `Pkg.develop`
should make this impossible; it is checked rather than trusted because the
failure it guards against produces plausible numbers under the wrong version.
"""
function assert_checkout(mod::Module, scriptdir::AbstractString)
    want, got = repo_root(scriptdir), pkgdir(mod)
    got !== nothing && realpath(want) == realpath(got) && return nothing
    error("""
        $(nameof(mod)) was loaded from
            $got
        instead of this checkout
            $want
        Any result produced now would describe a different version of the code
        than the commit recorded beside it. Delete
        $(joinpath(scriptdir, "Manifest.toml")) and run again.""")
end
