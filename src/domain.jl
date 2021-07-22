struct Params{FT}
    θ0 :: FT
    π0 :: FT
    Π :: FT
    S :: FT
    f :: FT
end

struct Domain{R,I,FT,A}
    x :: R
    y :: R
    z :: R
    nx :: I
    ny :: I
    nz :: I
    hx :: I
    hy :: I
    hz :: I
    Δx :: FT
    Δy :: FT
    Δz :: FT
    xc :: A
    yc :: A
    zc :: A
end

function Params(T = Float64; 
    θ0 = 285, θT = 340, π0 = 1000, πT = 500, f = 2/86400*sind(30)
)
    Π = π0 - πT
    S = (θT - θ0)/Π
    θ0 = θ0/(S*Π)
    π0 = π0/Π
    return Params(T(θ0), T(π0), T(Π), T(S), T(f))
end

function get_Rd(p::Params)
    return p.S*p.Π^2/p.f^2
end

function Domain(p::Params{T}; 
    x = (-3, 3), y = (-3, 3),
    size = (32, 32, 32), halo = (1,1,1)
) where {T}

    π0 = p.π0
    z = (π0, π0 - 1)
    nx, ny, nz = size
    hx, hy, hz = halo
    Δx = (x[2] - x[1])/nx
    Δy = (y[2] - y[1])/ny
    Δz = (z[2] - z[1])/nz
    xc = range(x[1] + (0.5 - hx)*Δx, x[2] - (0.5 - hx)*Δx, length = nx + 2*hx)
    xc = OffsetArray(xc, -hx)
    yc = range(y[1] + (0.5 - hy)*Δy, y[2] - (0.5 - hy)*Δy, length = ny + 2*hy)
    yc = OffsetArray(yc, -hy)
    zc = range(z[1] + (0.5 - hz)*Δz, z[2] - (0.5 - hz)*Δz, length = nz + 2*hz)
    zc = OffsetArray(zc, -hz)
    return Domain(x, y, z, nx, ny, nz, hx, hy, hz, T(Δx), T(Δy), T(Δz), xc, yc, zc)

end

halo(d::Domain) = (d.hx, d.hy, d.hz)
Base.size(d::Domain) = (d.nx, d.ny, d.nz)
Base.IteratorsMD.CartesianIndices(d::Domain) = CartesianIndices((d.nx, d.ny, d.nz))
Base.LinearIndices(d::Domain) = LinearIndices((d.nx, d.ny, d.nz))
Base.getindex(d::Domain, I::CartesianIndex{3}) = (d.xc[I[1]], d.yc[I[2]], d.zc[I[3]])
Base.getindex(d::Domain, i, j, k) = (d.xc[i], d.yc[j], d.zc[k])
const dx = CartesianIndex(1,0,0)
const dy = CartesianIndex(0,1,0)
const dz = CartesianIndex(0,0,1)
function new_field(d::Domain)
    underlying_data = zeros(d.nx + 2d.hx, d.ny + 2d.hy, d.nz + 2d.hz)
    return OffsetArray(underlying_data, -d.hx, -d.hy, -d.hz)
end

function new_rhs(d::Domain)
    return zeros(prod(size(d)))
end
function field_from_rhs!(f, rhs, d::Domain)
    l = LinearIndices(d)
    c = CartesianIndices(d)
    for I = c
        f[I] = rhs[l[I]]
    end
    return f
end
function field_from_rhs(rhs, d::Domain)
    f = new_field(d)
    return field_from_rhs!(f, rhs, d)
end
function rhs_from_field!(rhs, f, d::Domain)
    l = LinearIndices(d)
    c = CartesianIndices(d)
    for I = c
        rhs[l[I]] = f[I]
    end
    return rhs
end
function rhs_from_field(f, d::Domain)
    rhs = new_rhs(d)
    return rhs_from_field!(rhs, f, d)
end