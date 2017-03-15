using LinOps
using ProgressMeter

const cuda = if Pkg.installed("CUDArt") != nothing
    using CUDArt
    using CUBLAS
    using CUSPARSE
    true
else
    false
end

function SI(v, base = 1000)
    if v > 1 && v < 1e24
        prefixes = ["k", "M", "G", "T", "P", "E"]
        p = min(max(floor(Int, log(v)/log(base)), 0), length(prefixes))
        v /= base^p
        @sprintf("%.2f %s", v, p != 0 ? prefixes[p] : "")
    else
        @sprintf("%.3g ", v)
    end
end

function integrate(observe::Function,
                   v₀::KindOfVector,
                   tmax::Number, steps::Integer,
                   propagator::MagnusPropagator;
                   save_intermediate::Bool = false,
                   verbose::Bool = false)
    τ = tmax/steps
    V = similar(v₀, eltype(v₀), length(v₀), save_intermediate ? steps + 1 : 1)
    copy!(view(V, :, 1), v₀)
    cur = 1
    next = save_intermediate ? 2 : 1
    verbose && println("$(typeof(propagator))")
    prog = Progress(steps, 0.1, "Integrating ")
    tic()
    for i = 1:steps
        propagator((i-1)*τ, τ, view(V, :, cur), view(V, :, next))

        cur = next
        observe(view(V, :, next), i, τ)
        save_intermediate && (next += 1)
        verbose && ProgressMeter.update!(prog, i)
    end
    ms = toq()*1000
    verbose && println("Grid points/ms: ", SI(length(v₀)*steps/ms))
    Dict(:V => V,
         :milliseconds => ms,
         :performance => length(v₀)*steps/ms)
end

integrate(v₀::KindOfVector,
          tmax::Number, steps::Integer,
          propagator::MagnusPropagator;
          save_intermediate::Bool = false,
          verbose::Bool = false) =
              integrate((V,i,τ) -> (),
                        v₀,
                        tmax, steps, propagator;
                        save_intermediate = save_intermediate,
                        verbose = verbose)

function integrate(observe::Function,
                   v₀::KindOfVector,
                   tmax::Number, steps::Integer,
                   B::KindOfMatrix,
                   f::Function,
                   C::KindOfMatrix,
                   a::Number = 1;
                   propagator = :cfet,
                   exponentiator = :lanczos,
                   lanczos_m = 30,
                   mode = :cpu,
                   save_intermediate::Bool = false,
                   verbose::Bool = false)
    if mode == :gpu
        !cuda && error("Cuda not available, gpu integration impossible")
        verbose && println("Active device: $(CUDArt.name(CUDArt.device_properties(device())))")
        v₀ = CudaArray(v₀)
        B = upload(B)
        C = upload(C)
    end

    B = LinOp(B)
    C = LinOp(C)

    Exp = if exponentiator == :lanczos
        LanczosExponentiator(lanczos_m, v₀)
    elseif exponentiator == :scalar
        ScalarExponentiator(v₀)
    else
        error("Unknown exponentiator, $(string(exponentiator))")
    end

    propagator = if propagator == :cfet
        CFET4BfCPropagator(B, f, C, a, Exp)
    elseif propagator == :midpoint
        MidpointPropagator(t -> B + f(t)*C, a, Exp)
    else
        error("Unknown propagator, $(string(propagator))")
    end

    results = integrate(observe,
                        v₀,
                        tmax, steps, propagator;
                        save_intermediate = save_intermediate,
                        verbose = verbose)
    if mode == :gpu
        Dict(results..., :V => to_host(results[:V]))
    else
        results
    end
end

integrate(v₀::KindOfVector,
          tmax::Number, steps::Integer,
          B::KindOfMatrix,
          f::Function,
          C::KindOfMatrix,
          a::Number = 1;
          propagator = :cfet,
          exponentiator = :lanczos,
          lanczos_m = 30,
          mode = :cpu,
          save_intermediate::Bool = false,
          verbose::Bool = false) =
              integrate((V,i,τ) -> (),
                        v₀,
                        tmax, steps,
                        B, f, C, a;
                        propagator = propagator,
                        exponentiator = exponentiator,
                        lanczos_m = lanczos_m,
                        mode = mode,
                        save_intermediate = save_intermediate,
                        verbose = verbose)

export integrate
