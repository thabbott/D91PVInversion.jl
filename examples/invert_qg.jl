using D92PVInversion
using IterativeSolvers
using Random
import PyPlot; const plt = PyPlot

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (48, 48, 8), x = (-3, 3), y = (-3, 3))
println(params)
println(domain)

# Allocate storage for fields and vectors
ψ, ϕ, q = allocate_fields(domain)
xψ, ∂ψ, bψ, xϕ, ∂ϕ, bϕ = allocate_rhs(domain)

# Set initial values
set_background_ψ!(ψ, domain, params)

# Set PV field
set_q!(q, domain, params; A = 10)

# Construct linear operator for QG-like inversion
Lψ = generate_qg_Lψ(domain; T = float_type(params))

# Fill halos
fill_ψ_halos!(ψ, domain, params)

# Set boundary contributions to RHS 
set_qg_∂ψ!(∂ψ, domain, params)

# Set RHS
set_qg_bψ!(bψ, q, domain)

# Compute residual
rhs_from_field!(xψ, ψ, domain)
rψ = field_from_rhs(compute_qg_rψ(Lψ, xψ, ∂ψ, bψ, domain), domain)
nx, ny, nz = size(domain)
rψmid = 0.5*(rψ[1:nx,24,1:nz] + rψ[1:nx,25,1:nz])

# Plot residual
plt.figure()
plt.imshow(rψmid')
plt.gca().invert_yaxis()
plt.title("ψ residual")
plt.show()

# Solve
Random.seed!(1234)
@. bψ = bψ - ∂ψ
bicgstabl!(xψ, Lψ, bψ; log = false, verbose = true)
field_from_rhs!(ψ, xψ, domain)
fill_ψ_halos!(ψ, domain, params)

# Compute diagnostics
u = diagnose_umag(ψ, domain)
v = diagnose_v(ψ, domain)
ζ = diagnose_ζ(ψ, domain)
θ = diagnose_θ(ψ, domain)
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
    figsize = (9.5, 2), nrows = 1, ncols = 4, 
    sharey = true, sharex = true, constrained_layout = true
)
c = axes[1].contour(x, z, qmid', colors = "black")
c = axes[2].contour(x, z, vmid', colors = "black")
c = axes[3].contour(x, z, ζmid', colors = "black")
c = axes[4].contour(x, z, θmid', colors = "black")
axes[1].set_xlim([-1, 1])
axes[1].invert_yaxis()
plt.show()