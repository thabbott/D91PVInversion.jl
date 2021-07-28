const default_p0 = 1e5
const default_κ = 2/7
const default_g = 9.81

function dimensional_π(z, p::Params)
    return z*p.Π
end

function dimensional_p(z, p::Params; p0 = default_p0, κ = default_κ)
    return p0*(z/p.π0)^(1/κ)
end

function dimensional_r(r, p::Params)
    return get_Rd(p)*r
end

function dimensional_pseudoz(z, p::Params; g = default_g)
    θ0 = p.θ0*p.S*p.Π
    cp = p.π0*p.Π
    H = cp*θ0/g
    return H*(1 - z/p.π0)
end

function dimensional_q(q, z, p::Params; p0 = default_p0, κ = default_κ, g = default_g)
    πp = dimensional_π(z,p)/dimensional_p(z, p; p0 = p0, κ = κ)
    return p.S*p.f*g*κ*πp*q
end

function dimensional_u(u, p::Params)
    return get_Rd(p)*p.f*u
end

function redimensionalize_q!(q, d::Domain, p::Params; p0 = default_p0, κ = default_κ, g = default_g)
    for I = CartesianIndices(d)
        x, y, z = d[I]
        q[I] = dimensional_q(q[I], z, p; p0 = p0, κ = κ, g = g)
    end
    return q
end

function redimensionalize_u!(u, d::Domain, p::Params)
    for I = CartesianIndices(d)
        u[I] = dimensional_u(u[I], p)
    end
    return u
end