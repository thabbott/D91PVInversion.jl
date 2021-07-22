using D92PVInversion
using IterativeSolvers
using Random
using Printf
import PyPlot; const plt = PyPlot

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (48, 48, 8), x = (-3, 3), y = (-3, 3))
println(params)
println(domain)

# Create solvers
sψ = Solver(params; ω = 0.7)
sϕ = Solver(params; ω = 0.7)

# Allocate storage for fields and vectors
ψ, ϕ, q = allocate_fields(domain)
xψ, ∂ψ, bψ, xϕ, ∂ϕ, bϕ = allocate_rhs(domain)

# Set initial values
set_background_ψ!(ψ, domain, params)
set_background_ϕ!(ϕ, domain, params)

# Set PV field
set_q!(q, domain, params; A = 30)

# Construct linear operators
Lψ = generate_Lψ(ϕ, domain)
Lϕ = generate_Lϕ(ψ, domain)

# Fill halos
fill_ψ_halos!(ψ, domain, params)
fill_ϕ_halos!(ϕ, domain, params)

# Solver iteratively
while true

    # Set boundary contributions to ψ RHS
    set_∂ψ!(∂ψ, ϕ, domain, params)

    # Set ψ RHS
    set_bψ!(bψ, ψ, ϕ, q, domain)

    # Compute and plot residual
    # rhs_from_field!(xψ, ψ, domain)
    # rψ = field_from_rhs(compute_residual(Lψ, xψ, ∂ψ, bψ, domain), domain)
    # nx, ny, nz = size(domain)
    # rψmid = 0.5*(rψ[1:nx,24,1:nz] + rψ[1:nx,25,1:nz])
    # plt.figure()
    # plt.imshow(rψmid')
    # plt.gca().invert_yaxis()
    # plt.title("ψ residual")
    # plt.show()

    # Solve for ψ
    Random.seed!(1234)
    @. bψ = bψ - ∂ψ
    bicgstabl!(xψ, Lψ, bψ; log = false, verbose = false)
    relax!(ψ, xψ, sψ, domain; verbose = true)
    fill_ψ_halos!(ψ, domain, params)

    # Plot solution
    # nx, ny, nz = size(domain)
    # ψmid = 0.5*(ψ[1:nx,24,1:nz] + ψ[1:nx,25,1:nz])
    # plt.figure()
    # plt.imshow(ψmid')
    # plt.gca().invert_yaxis()
    # plt.title("ψ")
    # plt.show()

    # Set boundary contributions to ϕ RHS
    set_∂ϕ!(∂ϕ, ψ, domain, params)

    # Set ϕ RHS
    set_bϕ!(bϕ, ψ, ϕ, q, domain)

    # Compute and plot residual
    # rhs_from_field!(xϕ, ϕ, domain)
    # rϕ = field_from_rhs(compute_residual(Lϕ, xϕ, ∂ϕ, bϕ, domain), domain)
    # nx, ny, nz = size(domain)
    # rϕmid = 0.5*(rϕ[1:nx,24,1:nz] + rϕ[1:nx,25,1:nz])
    # plt.figure()
    # plt.imshow(rϕmid')
    # plt.gca().invert_yaxis()
    # plt.title("ϕ residual")
    # plt.show()

    # Solve for ϕ
    @. bϕ = bϕ - ∂ϕ
    bicgstabl!(xϕ, Lϕ, bϕ; log = false, verbose = false)
    relax!(ϕ, xϕ, sϕ, domain; verbose = true)
    fill_ϕ_halos!(ϕ, domain, params)

    # Plot solution
    # nx, ny, nz = size(domain)
    # ϕmid = 0.5*(ϕ[1:nx,24,1:nz] + ϕ[1:nx,25,1:nz])
    # plt.figure()
    # plt.imshow(ϕmid')
    # plt.gca().invert_yaxis()
    # plt.title("ϕ")
    # plt.show()

    # Exit loop if converged
    is_converged(sψ) && is_converged(sϕ) && break

end

# Compute diagnostics
u = diagnose_umag(ψ, domain)
v = diagnose_v(ψ, domain)
ζ = diagnose_ζ(ψ, domain)
θ = diagnose_θ(ϕ, domain)
nx, ny, nz = size(domain)
x = domain.xc[1:nx]
z = domain.zc[1:nz]
vmid = 0.5*(v[1:nx,ny÷2,1:nz] + v[1:nx,ny÷2+1,1:nz])
θmid = 0.5*(θ[1:nx,ny÷2,1:nz] + θ[1:nx,ny÷2+1,1:nz])
ζmid = 0.5*(ζ[1:nx,ny÷2,1:nz] + ζ[1:nx,ny÷2+1,1:nz])
qmid = 0.5*(q[1:nx,ny÷2,1:nz] + q[1:nx,ny÷2+1,1:nz])

# Calculate vertical penetration
ubot = u[1:nx,1:ny,1]
umax = 0.5*(u[1:nx,1:ny,nz÷2] + u[1:nx,1:ny,nz÷2+1])
vpen = maximum(ubot)/maximum(umax)
print("Vertical penetration: ", vpen)

# Plot solution
fig, axes = plt.subplots(
    figsize = (9.5, 3), nrows = 1, ncols = 4, 
    sharey = true, sharex = true, constrained_layout = true
)
c = axes[1].contour(x, z, qmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[1].set_title(@sprintf("q\nmax %.1e\ncontour interval %.1e", maximum(q), Δc))
c = axes[2].contour(x, z, vmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[2].set_title(@sprintf("v\nmax %.1e\ncontour interval %.1e", maximum(v), Δc))
c = axes[3].contour(x, z, ζmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[3].set_title(@sprintf("ζ\nmax %.1e\ncontour interval %.1e", maximum(ζ), Δc))
c = axes[4].contour(x, z, θmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[4].set_title(@sprintf("θ\nmax %.1e\ncontour interval %.1e", maximum(θ), Δc))
axes[1].set_xlim([-1, 1])
axes[1].invert_yaxis()
plt.show()