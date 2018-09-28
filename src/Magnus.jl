__precompile__(true)
module Magnus

using LinearMaps

abstract type Exponentiator end
abstract type KrylovExponentiator <: Exponentiator end

include("scalar.jl")
# include("pade.jl")
include("sub_exp.jl")
include("lanczos.jl")
include("propagators.jl")
include("midpoint.jl")
include("cfet.jl")
include("integrate.jl")
include("convergence.jl")

end # module
