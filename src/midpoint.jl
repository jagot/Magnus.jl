struct MidpointPropagator{E<:Exponentiator} <: MagnusPropagator
    A::Function
    a::Number
    Exp::E
end
MidpointPropagator(A::Function, Exp::E) where E<:Exponentiator =
    MidpointPropagator(A, 1, Exp)
MidpointPropagator(B::LinearMap,
                   f::Function,
                   C::LinearMap,
                   a::Number,
                   Exp::E) where E<:Exponentiator =
                       MidpointPropagator(t -> B + f(t)*C, a, Exp)

(p::MidpointPropagator{E})(t::Real, τ::Real,
                           v::AbstractVector, w::AbstractVector) where E<:Exponentiator =
                               p.Exp(p.A(t+τ/2), p.a*τ, v, w)

export MidpointPropagator
