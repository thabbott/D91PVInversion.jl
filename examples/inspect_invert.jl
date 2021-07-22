using D92PVInversion
using IterativeSolvers
using Random
import PyPlot; const plt = PyPlot

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (3, 3, 3), x = (-1, 1)./2, y = (-1, 1)./2)
println(params)
println(domain)

# Create solvers
sψ = Solver(params)
sϕ = Solver(params)

# Allocate storage for fields and vectors
ψ, ϕ, q = allocate_fields(domain)
xψ, ∂ψ, bψ, xϕ, ∂ϕ, bϕ = allocate_rhs(domain)

# Set initial values
set_background_ψ!(ψ, domain, params)
set_background_ϕ!(ϕ, domain, params)

# Set PV field
set_q!(q, domain, params; A = 0)

# Plot fields
plt.rc("font", size = 8)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ψ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Construct linear operators
Lψ = generate_Lψ(ϕ, domain)
Lϕ = generate_Lϕ(ψ, domain)

# Plot operators
plt.figure()
plt.imshow(Matrix(Lψ))
plt.show()
plt.figure()
plt.imshow(Matrix(Lϕ))
plt.show()

# Fill halos
fill_ψ_halos!(ψ, domain, params)
fill_ϕ_halos!(ϕ, domain, params)

# Re-plot fields
plt.rc("font", size = 8)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ψ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Re-plot operators
plt.figure()
plt.imshow(Matrix(Lψ))
plt.show()
plt.figure()
plt.imshow(Matrix(Lϕ))
plt.show()

# BEGIN ITERATION HERE
# Set boundary contributions to ψ RHS
set_∂ψ!(∂ψ, ϕ, domain, params)

# Plot boundary contributions to ψ RHS
f∂ψ = field_from_rhs(∂ψ, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(f∂ψ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("∂ψ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Set ψ RHS
set_bψ!(bψ, ψ, ϕ, q, domain)

# Plot q and RHS
fbψ = field_from_rhs(bψ, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(q[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("q, j = ", i))
end
axes[1].invert_yaxis()
plt.show()
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(fbψ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("bψ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Compute residual
rhs_from_field!(xψ, ψ, domain)
rψ = field_from_rhs(compute_residual(Lψ, xψ, ∂ψ, bψ, domain), domain)

# Plot residual
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(rψ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ residual, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Solve for ψ
Random.seed!(1234)
@. bψ = bψ - ∂ψ
bicgstabl!(xψ, Lψ, bψ; log = false, verbose = true)
relax!(ψ, xψ, sψ, domain; verbose = true)
fill_ψ_halos!(ψ, domain, params)
println("is_converged(sψ) = ", is_converged(sψ))

# Plot solution
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ψ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ψ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Set boundary contributions to ϕ RHS
set_∂ϕ!(∂ϕ, ψ, domain, params)

# Plot boundary contributions to ϕ RHS
f∂ϕ = field_from_rhs(∂ϕ, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(f∂ϕ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("∂ϕ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Set ψ RHS
set_bϕ!(bϕ, ψ, ϕ, q, domain)

# Plot q and RHS
fbϕ = field_from_rhs(bϕ, domain)
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(q[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("q, j = ", i))
end
axes[1].invert_yaxis()
plt.show()
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(fbϕ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("bϕ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Compute residual
rhs_from_field!(xϕ, ϕ, domain)
rϕ = field_from_rhs(compute_residual(Lϕ, xϕ, ∂ϕ, bϕ, domain), domain)

# Plot residual
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(rϕ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ϕ residual, j = ", i))
end
axes[1].invert_yaxis()
plt.show()

# Solve for ϕ
@. bϕ = bϕ - ∂ϕ
bicgstabl!(xϕ, Lϕ, bϕ; log = false, verbose = true)
relax!(ϕ, xϕ, sϕ, domain; verbose = true)
fill_ϕ_halos!(ϕ, domain, params)
println("is_converged(sϕ) = ", is_converged(sϕ))

# Plot solution
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(ϕ[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("ϕ, j = ", i))
end
axes[1].invert_yaxis()
plt.show()