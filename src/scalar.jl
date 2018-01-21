type ScalarExponentiator <: Exponentiator
end
function ScalarExponentiator(v::AbstractVector)
    size(v,1) == 1 || error("Scalar exponentiator only applicable for 1×1 operators")
    ScalarExponentiator()
end

function (SE::ScalarExponentiator){T<:Number}(Ω::LinearMap{T}, τ::Number,
                                              v, w)
    a = Ω([one(T)])[1]
    w[:] = exp(τ*a)*v
end

export ScalarExponentiator
