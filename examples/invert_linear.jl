using D91PVInversion
using IterativeSolvers
using Random
using Printf
import PyPlot; const plt = PyPlot

# Define functional form of PV anomaly
function q′fun(x, y, z, A, L, H, p)
    πc = p.π0 - 0.5
    return A*exp(-(x^2 + y^2)/L^2)*exp(-(z-πc)^2/H^2)
end

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (48, 48, 8), x = (-3, 3), y = (-3, 3))

# Set up background fields
ψ0, ϕ0, q0 = allocate_fields(domain)
set_background_ψ!(ψ0, domain, params)
set_background_ϕ!(ϕ0, domain, params)
fill_ψ_halos!(ψ0, domain, params)
fill_ϕ_halos!(ϕ0, domain, params)

# Create inversion problem
inv = LinearInversion(
    ψ0 = ψ0, ϕ0 = ϕ0,
    domain = domain, params = params,
    sψ′ = Solver(params; ω = 0.7), sϕ′ = Solver(params; ω = 0.7)
)

# Initialize problem
A = 100
L = 0.1
H = 0.1
initialize!(inv, q′fun, A, L, H, params)

# Solve
Random.seed!(1234)
while true
    iterate!(inv; verbose = true)
    is_converged(inv) && break
end

# Compute diagnostics
ψ′ = inv.ψ′
ϕ′ = inv.ϕ′
q′ = inv.q′
u′ = diagnose_umag(ψ′, domain)
v′ = diagnose_v(ψ′, domain)
ζ′ = diagnose_ζ(ψ′, domain)
θ′ = diagnose_θ(ϕ′, domain)
nx, ny, nz = size(domain)
x = domain.xc[1:nx]
z = domain.zc[1:nz]
v′mid = 0.5*(v′[1:nx,ny÷2,1:nz] + v′[1:nx,ny÷2+1,1:nz])
θ′mid = 0.5*(θ′[1:nx,ny÷2,1:nz] + θ′[1:nx,ny÷2+1,1:nz])
ζ′mid = 0.5*(ζ′[1:nx,ny÷2,1:nz] + ζ′[1:nx,ny÷2+1,1:nz])
q′mid = 0.5*(q′[1:nx,ny÷2,1:nz] + q′[1:nx,ny÷2+1,1:nz])

# Calculate vertical penetration
u′bot = u′[1:nx,1:ny,1]
u′max = 0.5*(u′[1:nx,1:ny,nz÷2] + u′[1:nx,1:ny,nz÷2+1])
vpen = maximum(u′bot)/maximum(u′max)
print("Vertical penetration: ", vpen, "\n")

# Plot solution
fig, axes = plt.subplots(
    figsize = (9.5, 3), nrows = 1, ncols = 4, 
    sharey = true, sharex = true, constrained_layout = true
)
c = axes[1].contour(x, z, q′mid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[1].set_title(@sprintf("q′\nmax %.1e\ncontour interval %.1e", maximum(q′mid), Δc))
c = axes[2].contour(x, z, v′mid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[2].set_title(@sprintf("v′\nmax %.1e\ncontour interval %.1e", maximum(v′mid), Δc))
c = axes[3].contour(x, z, ζ′mid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[3].set_title(@sprintf("ζ′\nmax %.1e\ncontour interval %.1e", maximum(ζ′mid), Δc))
c = axes[4].contour(x, z, θ′mid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[4].set_title(@sprintf("θ′\nmax %.1e\ncontour interval %.1e", maximum(θ′mid), Δc))
axes[1].set_xlim([-1, 1])
axes[1].invert_yaxis()
plt.show()