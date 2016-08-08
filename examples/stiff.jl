using LinOps
using Magnus

using PyCall
pygui(:qt)
using PyPlot

include("plot_convergence.jl")

type ExpEulerExponentiator <: Magnus.Exponentiator
end

import Base: call
function call{T<:Number}(SE::ExpEulerExponentiator,
                         Ω::LinOp{T}, τ::Number,
                         v, w)
    a = Ω([one(T)])[1]
    w[:] = (1 + τ*a)*v
end

function test_stiff(A,y0,y_exact,tmax,N,name)
    t = linspace(0,tmax,N+1)
    y_exact = y_exact(t)

    figure(name)
    clf()
    subplot(311)
    plot(t, y_exact, "r", linewidth = 2.0, label="Exact")

    for (Exp,label) in [(ScalarExponentiator(y0),"Exact exp")
                        # (LanczosExponentiator(2,y0),"Lanczos")
                        (ExpEulerExponentiator(),"Explicit Euler")]
        prop = MidpointPropagator(t -> LinOp(A), Exp)

        y = vec(integrate(y0, tmax, N, prop, save_intermediate = true))

        subplot(311)
        plot(t,y,label=label)

        plot_convergence(y_exact[end], y0, tmax, 0, 3, prop, [312,313], label)
    end
    subplot(311)
    margins(0,0.1)
    legend()

    subplot(312)
    xscale("log")
    yscale("log")
    xlabel(L"N")
    ylabel("Error")

    subplot(313)
    xscale("log")
    yscale("log")
    xlabel("Error")
    ylabel("Execution time [s]")

    tight_layout()
end

test_stiff(-15., [1.0], t -> exp(-15.0t), 1.0, 10, "stiff")
