import Base: call

#=

Implementation of commutator-free exponential time propagators, as
described in

  Alvermann, A., Fehske, H., & Littlewood, P. B. (2012). Numerical
  time propagation of quantum systems in radiation fields. New
  J. Phys., 14(10),
  105008. http://dx.doi.org/10.1088/1367-2630/14/10/105008

=#
type CFET4BfCPropagator{E<:Exponentiator} <: MagnusPropagator{E}
    B::LinOp
    f::Function
    C::LinOp
    a::Number
    Exp::E
end
CFET4BfCPropagator{E<:Exponentiator}(B::LinOp,
                                     f::Function,
                                     C::LinOp,
                                     Exp::E) =
                                         CFET4BfCPropagator(B, f, C, 1, Exp)

function call{E<:Exponentiator}(p::CFET4BfCPropagator{E},
                                t::Real, τ::Real,
                                v::KindOfVector, w::KindOfVector)
    h1 = 37/66 - 400/957*sqrt(5/3)
    h2 = -4/33
    h3 = 37/66 + 400/957*sqrt(5/3)
    h4 = -11/162
    h5 = 92/81

    ff1 = p.f(t + (1/2-sqrt(3/20))*τ)
    ff2 = p.f(t + 1/2*τ)
    ff3 = p.f(t + (1/2+sqrt(3/20))*τ)

    f1 = h1*ff1 + h2*ff2 + h3*ff3
    f2 = h4*ff1 + h5*ff2 + h4*ff3
    f3 = h3*ff1 + h2*ff2 + h1*ff3

    τ1 = 11/40*τ
    τ2 = 9/20*τ

    p.Exp(p.B + f3*p.C, p.a*τ1, v, w)
    p.Exp(p.B + f2*p.C, p.a*τ2, w, w)
    p.Exp(p.B + f1*p.C, p.a*τ1, w, w)
end

export CFET4BfCPropagator
