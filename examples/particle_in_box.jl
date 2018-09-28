using LinearMaps
using Magnus

using PyPlot

function particle_in_box(M = 20,L = 1)
    dx = L/(M+1)
    x = linspace(dx,L-dx,M)
    T = SymTridiagonal(2ones(M), -ones(M-1))/dx^2
    ei = eigfact(T)
    T,ei,x
end

function state(ei, c::Vector, t::Real = 0)
    c /= sqrt(sum(abs2,c))
    sel = 1:length(c)
    c = c.*exp.(-im*ei[:values][sel]*t)
    psi = ei[:vectors][:,sel]*c
    E = dot(ei[:values][sel], abs2.(c))
    psi,E
end


function test_propagation(name, psi0, tmax, period, N, propagator)
    @time psi = integrate(psi0,
                          tmax*period, N,
                          propagator;
                          save_intermediate = true,
                          verbose = true)[:V]
    t = linspace(0,tmax,N+1)
    n = Vector{Float64}(length(t))
    p = Vector{Complex128}(length(t))
    for i = eachindex(t)
        n[i] = norm(psi[:,i])
        p[i] = dot(psi0,psi[:,i])
        psi[:,i] -= p[i]*psi0
    end

    figure(name,figsize=(8,10))
    clf()
    subplot(311)
    pcolormesh(t, x, abs2.(psi), rasterized=true, cmap = plt[:cm][:get_cmap]("viridis"))
    margins(0,0)
    gca()[:set_xticklabels]([])
    title(L"|\Psi-\Psi_0\langle\Psi_0|\Psi\rangle|^2")
    ylabel(L"x")
    subplot(312)
    plot(t,field(t))
    margins(0,0.1)
    gca()[:set_xticklabels]([])
    ylabel(L"F(t)")
    subplot(313)
    plot(t,n, "-")
    margins(0,0.1)
    xlabel(L"t/T")
    ylabel(L"|\Psi|^2")
    a2 = gca()[:twinx]()
    a2[:plot](t,abs.(p), "--")
    margins(0,0.1)
    ylabel(L"|\langle\Psi_0|\Psi\rangle|")
    tight_layout()
end

M = 50
T,ei,x = particle_in_box(M)
psi0,E0 = state(ei, [1])

H₀ = LinearMap(sparse(T))
F₀ = 1e4
period = 1.0/abs(ei[:values][1]-ei[:values][2])
ω = 2π/4
mx = Diagonal(collect(x-mean(x)))
D = LinearMap(sparse(mx))
tmax = 50
field = t -> F₀*exp.(-(t-tmax/2).^2/2dt^2).*sin.(ω*t)
F = t -> field(t*period)
Hᵢ = t -> F(t)*D
H = t -> H₀ + Hᵢ(t)

midpoint = MidpointPropagator(H, -im, LanczosExponentiator(30,psi0))
cfet = CFET4BfCPropagator(H₀, F, D, -im, LanczosExponentiator(30,psi0))

ndt = 300
dt = 6

test_propagation("midpoint", psi0, tmax, period, tmax*300, midpoint)
test_propagation("CFET", psi0, tmax, period, tmax*30, cfet)
