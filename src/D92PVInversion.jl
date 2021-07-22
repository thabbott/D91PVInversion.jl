module D92PVInversion

using OffsetArrays
using LinearMaps
using LinearAlgebra
using PyPlot; const plt = PyPlot

# domain.jl
export Params, Domain, new_field, dx, dy, dz, halo, float_type
export field_from_rhs, field_from_rhs!, rhs_from_field, rhs_from_field!
# initialization.jl
export allocate_fields, allocate_rhs, set_background_ψ!, set_background_ϕ!, set_q!
# operators.jl
export generate_qg_Lψ, generate_Lψ, generate_Lϕ
# halos.jl
export fill_ψ_halos!, fill_ϕ_halos!
# boundaries.jl
export set_qg_∂ψ!
# rhsides.jl
export set_qg_bψ!
# diagnostics.jl
export diagnose_u, diagnose_v, diagnose_umag, diagnose_ζ, diagnose_θ
# residuals.jl
export compute_qg_rψ
# plots.jl
export plot_state

include("domain.jl")
include("profiles.jl")
include("initialization.jl")
include("operators.jl")
include("halos.jl")
include("boundaries.jl")
include("rhsides.jl")
include("diagnostics.jl")
include("residuals.jl")
include("plots.jl")

end # module