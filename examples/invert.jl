using D92PVInversion
import PyPlot; const plt = PyPlot

# Set up inversion domain
params = Params(Float64)
domain = Domain(params, size = (3, 3, 3), x = (-1, 1)./2, y = (-1, 1)./2)
println(params)
println(domain)

# Allocate storage for fields and vectors
ψ, ϕ, q = allocate_fields(domain)
xψ, ∂ψ, bψ, xϕ, ∂ϕ, bϕ, bq = allocate_rhs(domain)

# Set initial values
set_background_ψ!(ψ, domain, params)
set_background_ϕ!(ϕ, domain, params)

# Set PV field
set_q!(q, domain, params)
bq = rhs_from_field!(bq, q, domain)

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