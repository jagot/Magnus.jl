using Polynomials
using LinearOperators
using PolynomialOperators

type PadeExponentiator{T<:Number} <: Exponentiator
    p::Poly
    q::Poly
    tmp::Matrix{T}
end
function PadeExponentiator{T<:Number}(m::Integer, n::Integer,
                                      v::StridedArray{T})
    pade = Pade(Poly(1.//convert(Vector{Int},gamma(1:17)),"x"), m, n)
    PadeExponentiator(pade.p, pade.q, Matrix{eltype(v)}(length(v), 2))
end

function (PE::PadeExponentiator)(Ω::AbstractLinearOperator, τ::Number,
                                 v::StridedArray, w::StridedArray)
    PE.tmp[:,2] = polyval(PE.p, τ*Ω, view(PE.tmp, :, 1))*v
    solve!(w, polyval(PE.q, τ*Ω, view(PE.tmp, :, 1)), PE.tmp[:,2])[1]
end

export PadeExponentiator
