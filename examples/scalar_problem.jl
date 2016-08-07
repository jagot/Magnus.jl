using LinOps
using Magnus

using PyCall
pygui(:qt)
using PyPlot

using ProgressMeter

function propagate(propagator::MagnusPropagator, y0,
                   tmax::Number, N::Integer, save::Bool = false)
    dt = tmax/N
    y = Vector{Float64}(1 + (save ? N : 0))
    y[1] = y0[1]
    cur = 1
    next = save ? 2 : 1
    for i = 1:N
        propagator((i-1)*dt, dt, sub(y, cur), sub(y, next))
        cur = next
        save && (next += 1)
    end
    linspace(0,tmax,N+1),y
end

function test_convergence(propagator::MagnusPropagator, exact, Na, Nb, subs, label, args...)
    Ns = round(Int, logspace(Na, Nb, 100))
    errors = Vector{Float64}(length(Ns))
    times = Vector{Float64}(length(Ns))
    @showprogress for j = eachindex(Ns)
        tic()
        r = propagate(propagator, args..., Ns[j])[2][1]
        times[j] = toq()
        errors[j] = abs(exact-r)
    end
    subplot(subs[1])
    plot(Ns, errors, ".-", label = label)
    subplot(subs[2])
    plot(errors, times, ".-", label = label)
end

b = -1.
ω = 2π
f = t -> sin(ω*t)
c = 10.
y0 = [1.0]

tmax = π
N = 200

t = linspace(0,tmax,N+1)
y = exp(-(b*t - (cos(ω*t)-1)/ω * c))*y0[1]

function plot_approx(t_approx,y_approx,subs,label)
    st = length(t_approx) > 100 ? "-" : ".-"
    subplot(subs[1])
    plot(t_approx, y_approx, st, label=label)
    subplot(subs[2])
    semilogy(t_approx, abs(y_approx-y), st, label=label)
end

Exp = ScalarExponentiator(y0)
midpoint = MidpointPropagator(t -> LinOp(b + f(t)*c), -1, Exp)
cfet = CFET4BfCPropagator(LinOp(b), f, LinOp(c), -1, Exp)

figure("scalar",figsize=(10,14))
clf()
subplot(411)
plot_approx(propagate(midpoint, y0, tmax, N, true)..., [411,412], "Midpoint")
plot_approx(propagate(cfet, y0, tmax, N, true)..., [411,412], "CFET (25)")
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
test_convergence(midpoint, y[end], Nl..., [413,414], "Midpoint", y0, tmax)
test_convergence(cfet, y[end], Nl..., [413,414], "CFET (25)", y0, tmax)
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
ylabel("Error")
legend(framealpha=0.75,loc=4,ncol=2)
gca()[:set_xticklabels]([])

subplot(414)
xscale("log")
yscale("log")
xlabel("Error")
ylabel("Execution time [s]")
legend()
tight_layout()
