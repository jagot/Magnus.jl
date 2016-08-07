KindOfVector = LinOps.KindOfVector
KindOfMatrix = LinOps.KindOfMatrix

type ScalarExponentiator <: Exponentiator
end
function ScalarExponentiator(v::KindOfVector)
    size(v,1) == 1 || error("Scalar exponentiator only applicable for 1×1 operators")
    ScalarExponentiator()
end

import Base: call
function call{T<:Number}(SE::ScalarExponentiator,
                         Ω::LinOp{T}, τ::Number,
                         v, w)
    a = Ω([one(T)])[1]
    w[:] = exp(τ*a)*v
end

export ScalarExponentiator, call
