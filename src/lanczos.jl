KindOfVector = LinOps.KindOfVector
KindOfMatrix = LinOps.KindOfMatrix

import LinOps: axpy!, A_mul_B!, norm

function exp_lanczos!{T<:Number, R<:Real}(A::LinOp,
                                          v::KindOfVector,
                                          τ::T, m::Integer,
                                          vp::KindOfVector,
                                          V::KindOfMatrix,
                                          α::Vector{R},
                                          β::Vector{R},
                                          sub_v::Vector{T},
                                          d_sub_v::KindOfVector,
                                          sw::stegr_work;
                                          atol::R = 1.0e-8,
                                          rtol::R = 1.0e-4,
                                          verbose::Bool = false)
    β₀ = norm(v)
    copy!(sub(V,:,1), v)
    scale!(sub(V,:,1), 1.0/β₀)

    ε = atol + rtol * β₀
    verbose && @printf("Initial norm: β₀ %e, stopping threshold: %e\n", β₀, ε)

    j = 1
    jj = 1 # Which Krylov subspace to use, in the end

    for j = 1:m
        x,y = sub(V,:,j),sub(V,:,j+1)
        A(y,x)
        α[j] = real(vecdot(x,y))
        j > 1 && axpy!(T(-β[j-1]), sub(V,:,j-1), y)
        axpy!(T(-α[j]), x, y)
        β[j] = norm(y)
        scale!(y, 1.0/β[j])

        expT(sub(α, 1:jj), sub(β, 1:jj-1), τ, sub(sub_v, 1:j), sw)
        σ = β[j]*abs(sub_v[j])
        verbose && @printf("iter %d, α[%d] %e, β[%d] %e, σ %e\n",j, j, α[j], j, β[j], σ)

        if σ < ε
            break
        else
            j != m && (jj += 1)
        end
    end
    verbose && println("Krylov subspace size: ", jj)
    copy!(d_sub_v, sub_v)
    A_mul_B!(T(β₀), sub(V,:,1:jj),
             sub(d_sub_v, 1:jj),
             zero(T), vp)
end

type LanczosExponentiator{T<:AbstractFloat,U<:Number} <: KrylovExponentiator
    m::Integer
    V::KindOfMatrix
    α::Vector{T}
    β::Vector{T}
    sub_v::Vector{U}
    d_sub_v::KindOfVector
    sw::stegr_work{T}
    atol::T
    rtol::T
    verbose::Bool
end
function LanczosExponentiator(m::Integer, v::KindOfVector;
                              atol::Float64 = 1.0e-8,
                              rtol::Float64 = 1.0e-4,
                              verbose::Bool = false)
    U = eltype(v)
    N = length(v)
    V = similar(v, U, N, m + 1)
    α = Array(real(U), m)
    β = Array(real(U), m)
    sub_v = Array(U, m)
    d_sub_v = similar(v, U, m)
    LanczosExponentiator(m, V, α, β, sub_v, d_sub_v,
                         stegr_work(real(U), BlasInt(m)),
                         atol, rtol, verbose)
end

import Base: call
call{T<:AbstractFloat,U<:Number}(LE::LanczosExponentiator{T,U},
     Ω::LinOp, τ::U,
     v::KindOfVector, w::KindOfVector) = exp_lanczos!(Ω, v, τ, LE.m, w,
                                                      LE.V, LE.α, LE.β,
                                                      LE.sub_v, LE.d_sub_v,
                                                      LE.sw;
                                                      atol = LE.atol,
                                                      rtol = LE.rtol,
                                                      verbose = LE.verbose)

export LanczosExponentiator, call
