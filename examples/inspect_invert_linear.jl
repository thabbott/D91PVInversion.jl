using D91PVInversion
using IterativeSolvers
using Random
import PyPlot; const plt = PyPlot

# Define functional form of PV anomaly
function q′fun(x, y, z, A, L, H, p::Params)
    πc = p.π0 - 0.5
    Π = p.Π
    return A*exp(-(x^2 + y^2)/L^2)*exp(-(z-πc)^2/H^2)
end

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (3, 3, 3), x = (-1, 1)./2, y = (-1, 1)./2)
println(params)
println(domain)

# Set up background fields
ψ0, ϕ0, q0 = allocate_fields(domain)
set_background_ψ!(ψ0, domain, params)
set_background_ϕ!(ϕ0, domain, params)
fill_ψ_halos!(ψ0, domain, params)
fill_ϕ_halos!(ϕ0, domain, params)

# Create solvers
sψ′ = Solver(params)
sϕ′ = Solver(params)

# Allocate storage for fields and vectors
ψ′, ϕ′, q′ = allocate_linear_fields(domain)
xψ′, ∂ψ′, bψ′, xϕ′, ∂ϕ′, bϕ′ = allocate_linear_rhs(domain)

# Set initial values
@. ψ′ = 0
@. ϕ′ = 0

# Set PV field
A = 0.1
L = 0.1
H = 0.1
set_q′!(q′, domain, params, q′fun, A, L, H, params)

# Plot fields
plt.rc("font", size = 8)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ψ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Construct linear operators
Lψ′ = generate_linear_Lψ(ϕ0, domain)
Lϕ′ = generate_linear_Lϕ(ψ0, domain)

# Plot operators
plt.figure()
plt.imshow(Matrix(Lψ′))
plt.show()
plt.figure()
plt.imshow(Matrix(Lϕ′))
plt.show()

# Fill halos
fill_ψ′_halos!(ψ′, domain)
fill_ϕ′_halos!(ϕ′, domain)

# Re-plot fields
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ψ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Re-plot operators
plt.figure()
plt.imshow(Matrix(Lψ′))
plt.show()
plt.figure()
plt.imshow(Matrix(Lϕ′))
plt.show()

# BEGIN ITERATION HERE
# Set boundary contributions to ψ RHS
set_linear_∂ψ!(∂ψ′, ϕ0, domain, params)

# Plot boundary contributions to ψ RHS
f∂ψ′ = field_from_rhs(∂ψ′, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(f∂ψ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("∂ψ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Set ψ RHS
set_linear_bψ!(bψ′, ψ0, ϕ0, ψ′, ϕ′, q′, domain)

# Plot q and RHS
fbψ′ = field_from_rhs(bψ′, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(q′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("q′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(fbψ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("bψ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Compute residual
rhs_from_field!(xψ′, ψ′, domain)
rψ′ = field_from_rhs(compute_residual(Lψ′, xψ′, ∂ψ′, bψ′, domain), domain)

# Plot residual
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(rψ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ′ residual, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Solve for ψ
Random.seed!(1234)
@. bψ′ = bψ′ - ∂ψ′
bicgstabl!(xψ′, Lψ′, bψ′; log = false, verbose = true)
relax!(ψ′, xψ′, sψ′, domain; verbose = true)
fill_ψ′_halos!(ψ′, domain)
println("is_converged(sψ′) = ", is_converged(sψ′))

# Plot solution
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ψ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Set boundary contributions to ϕ RHS
set_linear_∂ϕ!(∂ϕ′, ψ0, domain, params)

# Plot boundary contributions to ϕ RHS
f∂ϕ′ = field_from_rhs(∂ϕ′, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(f∂ϕ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("∂ϕ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Set ϕ RHS
set_linear_bϕ!(bϕ′, ψ0, ϕ0, ψ′, ϕ′, q′, domain)

# Plot q and RHS
fbϕ′ = field_from_rhs(bϕ′, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(q′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("q′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(fbϕ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("bϕ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Compute residual
rhs_from_field!(xϕ′, ϕ′, domain)
rϕ′ = field_from_rhs(compute_residual(Lϕ′, xϕ′, ∂ϕ′, bϕ′, domain), domain)

# Plot residual
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(rϕ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ϕ′ residual, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Solve for ϕ
@. bϕ′ = bϕ′ - ∂ϕ′
bicgstabl!(xϕ′, Lϕ′, bϕ′; log = false, verbose = true)
relax!(ϕ′, xϕ′, sϕ′, domain; verbose = true)
fill_ϕ′_halos!(ϕ′, domain)
println("is_converged(sϕ′) = ", is_converged(sϕ′))

# Plot solution
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ϕ′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ϕ′, j = ", i))
end
axes[1].invert_yaxis()
plt.show()