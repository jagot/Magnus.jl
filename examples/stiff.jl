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
    a = Ω(v)
    w[:] = v + τ*a
end

function test_stiff(A,y0,y_exact,tmax,N,name,
                    Exps = [:scalar, :exp_euler],
                    Na = 0, Nb = 3)
    t = linspace(0,tmax,N+1)
    y_exact = y_exact(t)

    figure(name)
    clf()
    subplot(311)
    plot(t, y_exact, "r", linewidth = 2.0, label="Exact")

    Exp_objs = Dict(:scalar => (ScalarExponentiator(y0),"Exact exp"),
                    :lanczos => (LanczosExponentiator(2,y0),"Lanczos"),
                    :exp_euler => (ExpEulerExponentiator(),"Explicit Euler"))
    Exps = [Exp_objs[e] for e in Exps]

    for (Exp,label) in Exps
        prop = MidpointPropagator(t -> LinOp(A), Exp)

        y = vec(integrate(y0, tmax, N, prop, save_intermediate = true))

        subplot(311)
        plot(t,y,label=label)

        plot_convergence(y_exact[end], y0, tmax, Na, Nb,
                         prop, [312,313], label)
    end
    subplot(311)
    margins(0,0.1)
    legend(framealpha=0.75)

    subplot(312)
    xscale("log")
    yscale("log")
    xlabel(L"N")
    ylabel("Error")
    margins(0,1)

    subplot(313)
    xscale("log")
    yscale("log")
    xlabel("Error")
    ylabel("Execution time [s]")
    margins(0,1)

    tight_layout()
end

test_stiff(-15., [1.0], t -> exp(-15.0t), 1.0, 10, "stiff")

if Pkg.installed("LambertW") != nothing
    using LambertW
    δ = 0.01
    a = 1/δ - 1
    y = t -> 1./(lambertw(a*exp(a-t)) + 1)
    # Technically not a linear operator, but, hey.
    # See http://mathworks.com/company/newsletters/articles/stiff-differential-equations.html
    fireball = LinOp{Float64}((y,x,α,β) -> y[:] = α*(x.^2 - x.^3))
    test_stiff(fireball, [δ], y, 1./δ, 200, "Fireball, half interval",
               [:exp_euler], 0, 4)
    test_stiff(fireball, [δ], y, 2./δ, 200, "Fireball, full interval",
               [:exp_euler], 0, 4)
end
