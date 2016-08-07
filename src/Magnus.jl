module Magnus

using LinOps

abstract Exponentiator

include("scalar.jl")
# include("pade.jl")
include("lanczos.jl")
include("propagators.jl")

end # module
