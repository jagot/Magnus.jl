using LinearAlgebra
using Printf

function exp_lanczos!(A::L,
                      v::AbstractVector,
                      τ::T, m::Integer,
                      vp::AbstractVector,
                      V::AbstractMatrix,
                      α::Vector{R},
                      β::Vector{R},
                      sub_v::Vector{T},
                      d_sub_v::AbstractVector,
                      sw::stegr_work;
                      atol::R = 1.0e-8,
                      rtol::R = 1.0e-4,
                      verbose::Bool = false) where {L<:LinearMap, T<:Number, R<:Real}
    β₀ = norm(v)
    copyto!(view(V,:,1), v)
    lmul!(one(T)/β₀, view(V,:,1))

    ε = atol + rtol * β₀
    verbose && @printf("Initial norm: β₀ %e, stopping threshold: %e\n", β₀, ε)

    j = 1
    jj = 1 # Which Krylov subspace to use, in the end

    for j = 1:m
        x,y = view(V,:,j),view(V,:,j+1)
        mul!(y,A,x)
        α[j] = real(dot(x,y))
        j > 1 && (y .-= β[j-1]*view(V,:,j-1))
        y .-= α[j]*x
        β[j] = norm(y)
        lmul!(one(T)/β[j], y)

        expT(view(α, 1:jj), view(β, 1:jj-1), τ, view(sub_v, 1:j), sw)
        σ = β[j]*abs(sub_v[j])
        verbose && @printf("iter %d, α[%d] %e, β[%d] %e, σ %e\n",j, j, α[j], j, β[j], σ)

        if σ < ε
            break
        else
            j != m && (jj += 1)
        end
    end
    verbose && println("Krylov subspace size: ", jj)
    copyto!(d_sub_v, sub_v)
    vp[:] = β₀*view(V,:,1:jj)*view(d_sub_v, 1:jj)
end

mutable struct LanczosExponentiator{T<:AbstractFloat,U<:Number} <: KrylovExponentiator
    m::Integer
    V::AbstractMatrix
    α::Vector{T}
    β::Vector{T}
    sub_v::Vector{U}
    d_sub_v::AbstractVector
    sw::stegr_work{T}
    atol::T
    rtol::T
    verbose::Bool
end
function LanczosExponentiator(m::Integer, v::AbstractVector;
                              atol::Float64 = 1.0e-8,
                              rtol::Float64 = 1.0e-4,
                              verbose::Bool = false)
    U = eltype(v)
    T = real(U)
    N = length(v)
    V = similar(v, U, N, m + 1)
    α = Array{real(U)}(undef,m)
    β = Array{real(U)}(undef,m)
    sub_v = Array{U}(undef,m)
    d_sub_v = similar(v, U, m)
    LanczosExponentiator(m, V, α, β, sub_v, d_sub_v,
                         stegr_work(real(U), LinearAlgebra.BlasInt(m)),
                         T(atol), T(rtol), verbose)
end

(LE::LanczosExponentiator{T,U})(Ω::L, τ::U,
                                v::AbstractVector, w::AbstractVector) where {L<:LinearMap, T<:AbstractFloat,U<:Number} =
                                    exp_lanczos!(Ω, v, τ, LE.m, w,
                                                 LE.V, LE.α, LE.β,
                                                 LE.sub_v, LE.d_sub_v,
                                                 LE.sw;
                                                 atol = LE.atol,
                                                 rtol = LE.rtol,
                                                 verbose = LE.verbose)

export LanczosExponentiator
