module Magnus

using LinOps

abstract Exponentiator
abstract KrylovExponentiator <: Exponentiator

include("scalar.jl")
# include("pade.jl")
include("sub_exp.jl")
include("lanczos.jl")
include("propagators.jl")

end # module
