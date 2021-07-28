module D91PVInversion

using OffsetArrays
using LinearMaps
using LinearAlgebra
using IterativeSolvers
using JLD2

# structs.jl
export Params, float_type
export Domain, new_field, new_rhs, dx, dy, dz, halo,
       field_from_rhs, field_from_rhs!, rhs_from_field, rhs_from_field!
export Solver, is_converged
# initialization.jl
export allocate_fields, allocate_linear_fields, allocate_rhs, allocate_linear_rhs, 
       set_background_ψ!, set_background_ϕ!, set_q!, set_q′!
# operators.jl
export generate_linear_Lψ, generate_linear_Lϕ, generate_Lψ, generate_Lϕ
# halos.jl
export fill_ψ′_halos!, fill_ϕ′_halos!, fill_ψ_halos!, fill_ϕ_halos!
# boundaries.jl
export set_linear_∂ψ!, set_linear_∂ϕ!, set_∂ψ!, set_∂ϕ!
# rhsides.jl
export set_linear_bψ!, set_linear_bϕ!, set_bψ!, set_bϕ!
# diagnostics.jl
export diagnose_u, diagnose_v, diagnose_umag, diagnose_ζ, diagnose_θ
# residuals.jl
export compute_residual
# relaxation.jl
export relax!
# inversions.jl
export LinearInversion, NLInversion, initialize!, iterate!, is_converged, save_inversion_results

include("structs.jl")
include("profiles.jl")
include("initialization.jl")
include("operators.jl")
include("halos.jl")
include("boundaries.jl")
include("rhsides.jl")
include("diagnostics.jl")
include("residuals.jl")
include("relaxation.jl")
include("inversions.jl")

end # module