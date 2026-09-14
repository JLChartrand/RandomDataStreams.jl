using RandomDataStreams
using Documenter

DocMeta.setdocmeta!(RandomDataStreams, :DocTestSetup, :(using RandomDataStreams); recursive = true)

# `julia --project=docs docs/make.jl` builds the HTML site;
# `julia --project=docs docs/make.jl pdf` builds docs/build/RandomDataStreams.pdf
# instead, which is the file committed as docs/RandomDataStreams.pdf. Keeping
# both behind one `pages` list is what stops the PDF from drifting away from
# the site, as it did between the 0.2 and the counter-based releases.
const PDF = "pdf" in ARGS

const PAGES = [
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Streams & Substreams" => "streams.md",
    "API Reference" => "api.md",
    "Docstrings" => "docstrings.md",
    "Implementation Notes" => "implementation.md",
    "Validation" => "validation.md",
    "Generator Comparison" => "comparison.md",
    "FAQ" => "faq.md",
]

makedocs(;
    sitename = "RandomDataStreams.jl",
    modules = [RandomDataStreams],
    checkdocs = :exports,
    format = PDF ? Documenter.LaTeX(platform = "native") :
                   Documenter.HTML(;
                       prettyurls = get(ENV, "CI", nothing) == "true",
                       canonical = "https://jlchartrand.github.io/RandomDataStreams.jl",
                       edit_link = "master",
                   ),
    pages = PAGES,
)

# Documenter names the PDF after the site; the repository keeps it under the
# package name, so copy it into place rather than committing whatever the
# writer happened to call it.
if PDF
    built = only(filter(endswith(".pdf"), readdir(joinpath(@__DIR__, "build"); join = true)))
    cp(built, joinpath(@__DIR__, "RandomDataStreams.pdf"); force = true)
    @info "wrote docs/RandomDataStreams.pdf" from = basename(built)
end

if !PDF && get(ENV, "CI", nothing) == "true"
    deploydocs(;
        repo = "github.com/JLChartrand/RandomDataStreams.jl.git",
        devbranch = "master",
    )
end
