using LinearOperators

type ScalarExponentiator <: Exponentiator
end
function ScalarExponentiator{T<:Number}(v::StridedArray{T})
    size(v,1) == 1 || error("Scalar exponentiator only applicable for 1×1 operators")
    ScalarExponentiator()
end

import Base: call
function call(SE::ScalarExponentiator,
              Ω::AbstractLinearOperator, τ::Number,
              v::StridedArray, w::StridedArray)
    a = (Ω*[1])[1]
    w[:] = exp(τ*a)*v
end

export ScalarExponentiator, call
