using LinearOperators

import Base: call

abstract MagnusPropagator{E<:Exponentiator}

type MidpointPropagator{E<:Exponentiator} <: MagnusPropagator{E}
    A::Function
    a::Number
    Exp::E
end
MidpointPropagator{E<:Exponentiator}(A::Function, Exp::E) =
    MidpointPropagator(A, 1, Exp)

call{E<:Exponentiator}(p::MidpointPropagator{E},
                       t::Real, τ::Real,
                       v::StridedArray, w::StridedArray) =
                           p.Exp(p.A(t+τ/2), p.a*τ, v, w)


export MagnusPropagator, MidpointPropagator
