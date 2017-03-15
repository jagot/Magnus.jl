import Base: call

type MidpointPropagator{E<:Exponentiator} <: MagnusPropagator
    A::Function
    a::Number
    Exp::E
end
MidpointPropagator{E<:Exponentiator}(A::Function, Exp::E) =
    MidpointPropagator(A, 1, Exp)
MidpointPropagator{E<:Exponentiator}(B::LinOp,
                                     f::Function,
                                     C::LinOp,
                                     a::Number,
                                     Exp::E) =
                                         MidpointPropagator(t -> B + f(t)*C, a, Exp)

(p::MidpointPropagator{E}){E<:Exponentiator}(t::Real, τ::Real,
                                             v::KindOfVector, w::KindOfVector) =
                                                 p.Exp(p.A(t+τ/2), p.a*τ, v, w)

export MidpointPropagator
