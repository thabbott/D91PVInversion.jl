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

# Allocate storage for fields and vectors
ψ′, ϕ′, q′ = allocate_fields(domain)
x, b = allocate_linearized_rhs(domain)

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
L = generate_linearized_L(ψ0, ϕ0, domain)

# Plot operators
plt.figure()
plt.imshow(Matrix(L))
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
M = Matrix(L)
Minv = inv(M)
plt.figure()
plt.imshow(M)
plt.show()
plt.figure()
plt.imshow(Minv)
plt.show()
plt.figure()
plt.imshow(Minv*M)
plt.show()

# Set RHS
set_linearized_b!(b, q′, domain)

# Plot q and RHS
fig, axes = plt.subplots(
    figsize = (9.5, 2), nrows = 1, ncols = 5, 
    sharey = true, sharex = true, constrained_layout = true
)
for i = 1:5
    im = axes[i].imshow(q′[:,i-1,:]')
    plt.colorbar(im, ax = axes[i], location = "bottom", aspect = 5, shrink = 0.7)
    axes[i].set_title(string("q′, j = ", i))
end

# Solve
Random.seed!(1234)
set_linearized_b!(b, q′, domain)
bicgstabl!(x, L, b; log = false, verbose = true)
fields_from_linearized_rhs!(ϕ′, ψ′, x, domain)

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