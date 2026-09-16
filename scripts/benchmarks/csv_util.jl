# Minimal CSV writer, shared by throughput.jl and gpu_throughput.jl. A real
# CSV.jl dependency is more machinery than a handful of numeric columns needs,
# and none of the values written here (generator names, metric names, numbers)
# can contain a comma or a quote, so no escaping is needed either.

"""
    write_csv(path, header, rows)

Write `header` (a `Vector{String}`) as the first line, then one line per
element of `rows` (each an iterable of the same length as `header`), comma
-joined. Overwrites `path` if it already exists.
"""
function write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end
