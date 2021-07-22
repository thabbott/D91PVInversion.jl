using D92PVInversion
using IterativeSolvers
using Random
import PyPlot; const plt = PyPlot

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (3, 3, 3), x = (-1, 1)./2, y = (-1, 1)./2)
println(params)
println(domain)

# Allocate storage for fields and vectors
ψ, ϕ, q = allocate_fields(domain)
xψ, ∂ψ, bψ, xϕ, ∂ϕ, bϕ = allocate_rhs(domain)

# Set initial values
set_background_ψ!(ψ, domain, params)
set_background_ϕ!(ϕ, domain, params)

# Set PV field
set_q!(q, domain, params)

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

# Construct linear operator for QG-like inversion
Lψ = generate_qg_Lψ(domain; T = float_type(params))

# Plot operators
plt.figure()
plt.imshow(Matrix(Lψ))
plt.show()

# Fill halos
fill_ψ_halos!(ψ, domain, params)
fill_ϕ_halos!(ϕ, domain, params)

# Re-plot fields
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

# Set boundary contributions to RHS 
set_qg_∂ψ!(∂ψ, domain, params)

# Plot boundary contributions
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

# Set RHS
set_qg_bψ!(bψ, q, domain)

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

# Solve
Random.seed!(1234)
@. bψ = bψ - ∂ψ
bicgstabl!(xψ, Lψ, bψ; log = false, verbose = true)
field_from_rhs!(ψ, xψ, domain)
fill_ψ_halos!(ψ, domain, params)

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