module D92PVInversion

using OffsetArrays
using IterativeSolvers
using LinearMaps
using PyPlot; const plt = PyPlot

# domain.jl
export Params, Domain, new_field, dx, dy, dz, halo
export field_from_rhs, field_from_rhs!, rhs_from_field, rhs_from_field!
# initialization.jl
export allocate_fields, allocate_rhs, set_background_ψ!, set_background_ϕ!, set_q!
# operators.jl
export generate_Lψ, generate_Lϕ
# halos.jl
export fill_ψ_halos!, fill_ϕ_halos!
# plots.jl
export plot_state

include("domain.jl")
include("profiles.jl")
include("initialization.jl")
include("operators.jl")
include("halos.jl")
include("plots.jl")

end # module