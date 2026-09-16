#!/usr/bin/env julia

# Plain-text description of the machine a benchmark ran on. A throughput
# number means nothing on its own -- "83 M draws/s" needs the CPU, and a GPU
# speedup needs the card and driver -- so this is meant to sit next to a
# benchmark's CSV output, not to be read alone.
#
#     julia scripts/benchmarks/sysinfo.jl > sysinfo.txt

include(joinpath(@__DIR__, "..", "env.jl"))
ensure_checkout_env(@__DIR__)

using RandomDataStreams, CUDA, Dates

assert_checkout(RandomDataStreams, @__DIR__)

include(joinpath(@__DIR__, "..", "provenance.jl"))

function main()
    println("RandomDataStreams benchmark host")
    println(provenance_line())
    println("generated ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    println()

    println("Host:   ", gethostname())
    println("OS:     ", Sys.KERNEL, " ", Sys.MACHINE)
    println("CPU:    ", Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " threads")
    println("RAM:    ", round(Sys.total_memory() / 2^30, digits = 1), " GiB")
    println("Julia:  ", VERSION)
    println()

    if CUDA.functional()
        dev = CUDA.device()
        println("GPU:            ", CUDA.name(dev))
        println("Compute cap.:   ", CUDA.capability(dev))
        println("CUDA driver:    ", CUDA.driver_version())
        println("CUDA runtime:   ", CUDA.runtime_version())
        println("GPU memory:     ", round(CUDA.totalmem(dev) / 2^30, digits = 1), " GiB")
    else
        println("GPU: none functional (CUDA.functional() == false)")
    end
end

main()
