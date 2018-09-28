using LinearMaps
using Magnus
using PyPlot

include("plot_convergence.jl")

b = -1.
ω = 2π
f = t -> sin(ω*t)
c = 10.
y0 = [1.0]

tmax = π
N = 200

t = linspace(0,tmax,N+1)
y = exp.(-(b*t - (cos.(ω*t)-1)/ω * c))*y0[1]

function plot_approx(t_approx,y_approx,subs,label)
    st = length(t_approx) > 100 ? "-" : ".-"
    subplot(subs[1])
    plot(t_approx, y_approx, st, label=label)
    subplot(subs[2])
    semilogy(t_approx, abs(y_approx-y), st, label=label)
end

Exp = ScalarExponentiator(y0)
midpoint = MidpointPropagator(t -> LinearMap(b + f(t)*c), -1, Exp)
cfet = CFET4BfCPropagator(LinearMap(b), f, LinearMap(c), -1, Exp)

figure("scalar",figsize=(10,14))
clf()
subplot(411)
for (prop,name) in [(midpoint,"Midpoint"),
                    (cfet, "CFET (25)")]
    result = integrate(y0, tmax, N, prop,
                       save_intermediate = true,
                       verbose = true)
    plot_approx(t, vec(result[:V]),
                [411,412], name)
end
legend(framealpha=0.75)
margins(0,1)
xlabel(L"t")
ylabel("Error")
subplot(411)
plot(t, y, "k--", label="Exact")
ylabel(L"y(t)")
margins(0,0.1)
gca()[:set_xticklabels]([])

Nl = 0,5
plot_convergence(y[end], y0, tmax, Nl..., midpoint, [413,414], "Midpoint")
plot_convergence(y[end], y0, tmax, Nl..., cfet, [413,414], "CFET (25)")
subplot(413)
Nl = logspace(Nl...,10)
plot(Nl,1./Nl, ":", label="Linear")
plot(Nl,1./Nl.^2, ":", label="Quadratic")
plot(Nl,1./Nl.^3, ":", label="Cubic")
plot(Nl,1./Nl.^4, ":", label="Quartic")
plot(Nl,1./Nl.^5, ":", label="Quintic")
plot(Nl,1./Nl.^6, ":", label="Sextic")

xscale("log")
yscale("log")
xlabel("Number of time steps")
ylabel("Error")
legend(framealpha=0.75,loc=4,ncol=2)

subplot(414)
xscale("log")
yscale("log")
xlabel("Error")
ylabel("Execution time [s]")
legend()
tight_layout()
