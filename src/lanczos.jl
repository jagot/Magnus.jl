using Krylov
using LinearOperators

type LanczosExponentiator{T<:AbstractFloat,U<:Number} <: Exponentiator
    m::Integer
    V::Matrix{U}
    α::Vector{T}
    β::Vector{T}
    tmp::Vector{U}
end
function LanczosExponentiator{U<:Number}(m::Integer, v::StridedVector{U})
    M = length(v)
    V = zeros(U, M, m + 1)
    α = zeros(real(U), m)
    β = zeros(real(U), m)
    LanczosExponentiator(m, V, α, β, similar(v))
end

import Base: call
call(LE::LanczosExponentiator,
     Ω::AbstractLinearOperator, τ::Number,
     v::StridedVector, w::StridedVector) =
         exp_lanczos!(Ω, v, τ, LE.m, w, LE.V, LE.α, LE.β)

export LanczosExponentiator, call
