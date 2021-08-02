module D91PVInversion

using OffsetArrays
using LinearMaps
using LinearAlgebra
using IterativeSolvers
using JLD2

# structs.jl
export Params, float_type
export Domain, new_field, new_rhs, dx, dy, dz, halo,
       field_from_rhs, field_from_rhs!, 
       fields_from_linearized_rhs, fields_from_linearized_rhs!, 
       rhs_from_field, rhs_from_field!
export Solver, is_converged
# profiles.jl
export background_ϕ, background_ψ, background_θ
# initialization.jl
export allocate_fields, allocate_rhs, allocate_linearized_rhs, 
       set_background_ψ!, set_background_ϕ!, set_q!, set_q′!
# operators.jl
export generate_linearized_L, generate_Lψ, generate_Lϕ,
       ∂x, ∂y, ∂z, ∂x², ∂y², ∂z², ∂xy, ∂xz, ∂yz
# halos.jl
export fill_ψ′_halos!, fill_ϕ′_halos!, fill_ψ_halos!, fill_ϕ_halos!
# boundaries.jl
export set_∂ψ!, set_∂ϕ!
# rhsides.jl
export set_linearized_b!, set_linear_bϕ!, set_bψ!, set_bϕ!
# diagnostics.jl
export diagnose_u, diagnose_v, diagnose_umag, diagnose_ζ, diagnose_θ, diagnose_penetration_depth
# residuals.jl
export compute_residual
# relaxation.jl
export relax!
# inversions.jl
export ColumnarVortex, LinearizedInversion, NLInversion, 
       initialize!, solve!, iterate!, is_converged, save_inversion_results
# redimensionalize.jl
export dimensional_π, dimensional_p, dimensional_pseudoz, dimensional_r, dimensional_q, dimensional_u,
       redimensionalize_q!, redimensionalize_u!

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
include("redimensionalize.jl")

end # module