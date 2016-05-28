using LinearOperators
using Magnus

using PyCall
pygui(:qt)
using PyPlot

function particle_in_box(M = 20,L = 1)
    dx = L/(M+1)
    x = linspace(dx,L-dx,M)
    T = SymTridiagonal(2ones(M), -ones(M-1))/dx^2
    ei = eigfact(T)
    T,ei,x
end

function state(ei, c::Vector, t::Real = 0)
    c /= sqrt(sumabs2(c))
    sel = 1:length(c)
    c = c.*exp(-im*ei[:values][sel]*t)
    psi = ei[:vectors][:,sel]*c
    E = dot(ei[:values][sel], abs2(c))
    psi,E
end

function propagate{T<:AbstractFloat}(ψ₀::AbstractVector{Complex{T}},
                                     tmax::Number, N::Integer,
                                     propagator::MagnusPropagator,
                                     save::Bool = false)
    τ = tmax/(N+1)
    Ψ = save ? Matrix{Complex{T}}(length(ψ₀),N+1) : Vector{Complex{T}}(length(ψ₀))
    Ψ[:,1] = ψ₀
    cur = 1
    next = save ? 2 : 1
    for i = 1:N
        propagator((i-1)*τ, τ, sub(Ψ, :, cur), sub(Ψ, :, next))
        cur = next
        save && (next += 1)
    end
    Ψ
end

M = 50
T,ei,x = particle_in_box(M)
psi0,E0 = state(ei, [1])

H₀ = LinearOperator(T)
F₀ = 1e4
period = 1.0/abs(ei[:values][1]-ei[:values][2])
ω = 2π/4
mx = Diagonal(collect(x-mean(x)))
tmax = 50
ndt = 300
dt = 6
F = t -> F₀*exp(-(t-tmax/2).^2/2dt^2).*sin(ω*t)
Hᵢ = t -> F(t*period)*LinearOperator(mx, symmetric=true, hermitian=true)
H = t -> H₀ + Hᵢ(t)


propagator = MidpointPropagator(H, -im, LanczosExponentiator(30,psi0))
@time psi = propagate(psi0,
                      tmax*period, tmax*ndt,
                      propagator,
                      true)

t = linspace(0,tmax,tmax*ndt+1)
n = Vector{Float64}(length(t))
p = Vector{Complex128}(length(t))
for i = eachindex(t)
    n[i] = norm(psi[:,i])
    p[i] = dot(psi0,psi[:,i])
    psi[:,i] -= p[i]*psi0
end

figure(1,figsize=(8,10))
clf()
subplot(311)
pcolormesh(t, x, abs2(psi), rasterized=true, cmap = plt[:cm][:get_cmap]("viridis"))
margins(0,0)
gca()[:set_xticklabels]([])
title(L"|\Psi-\Psi_0\langle\Psi_0|\Psi\rangle|^2")
ylabel(L"x")
subplot(312)
plot(t,F(t))
margins(0,0.1)
gca()[:set_xticklabels]([])
ylabel(L"F(t)")
subplot(313)
plot(t,n, "-")
margins(0,0.1)
xlabel(L"t/T")
ylabel(L"|\Psi|^2")
a2 = gca()[:twinx]()
a2[:plot](t,abs(p), "--")
ylabel(L"|\langle\Psi_0|\Psi\rangle|")
tight_layout()
show()
