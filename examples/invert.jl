using D91PVInversion
using IterativeSolvers
using Random
using Printf
import PyPlot; const plt = PyPlot

# Define functional form of PV anomaly
function q′(x, y, z, A, L, H, p)
    πc = p.π0 - 0.5
    return A*exp(-(x^2 + y^2)/L^2)*exp(-(z-πc)^2/H^2)
end

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (48, 48, 8), x = (-3, 3), y = (-3, 3))

# Create inversion problem
inv = NLInversion(
    domain = domain, params = params; 
    sψ = Solver(params; ω = 0.7), sϕ = Solver(params; ω = 0.7)
)

# Initialize problem
A = 0.1
L = 0.1
H = 0.1
initialize!(inv, q′, A, L, H, params)

# Solve
Random.seed!(1234)
while true
    iterate!(inv; verbose = true)
    is_converged(inv) && break
end

# Compute diagnostics
ψ = inv.ψ
ϕ = inv.ϕ
q = inv.q
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
axes[1].set_title(@sprintf("q\nmax %.1e\ncontour interval %.1e", maximum(qmid), Δc))
c = axes[2].contour(x, z, vmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[2].set_title(@sprintf("v\nmax %.1e\ncontour interval %.1e", maximum(vmid), Δc))
c = axes[3].contour(x, z, ζmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[3].set_title(@sprintf("ζ\nmax %.1e\ncontour interval %.1e", maximum(ζmid), Δc))
c = axes[4].contour(x, z, θmid', colors = "black")
Δc = length(c.levels) > 1 ? c.levels[2] - c.levels[1] : NaN
axes[4].set_title(@sprintf("θ\nmax %.1e\ncontour interval %.1e", maximum(θmid), Δc))
axes[1].set_xlim([-1, 1])
axes[1].invert_yaxis()
plt.show()

# Save output
save_inversion_results(@sprintf("output/invert_A%.1f.jld2", A), inv)