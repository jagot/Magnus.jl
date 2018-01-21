struct MidpointPropagator{E<:Exponentiator} <: MagnusPropagator
    A::Function
    a::Number
    Exp::E
end
MidpointPropagator{E<:Exponentiator}(A::Function, Exp::E) =
    MidpointPropagator(A, 1, Exp)
MidpointPropagator{E<:Exponentiator}(B::LinearMap,
                                     f::Function,
                                     C::LinearMap,
                                     a::Number,
                                     Exp::E) =
                                         MidpointPropagator(t -> B + f(t)*C, a, Exp)

(p::MidpointPropagator{E}){E<:Exponentiator}(t::Real, τ::Real,
                                             v::AbstractVector, w::AbstractVector) =
                                                 p.Exp(p.A(t+τ/2), p.a*τ, v, w)

export MidpointPropagator
