function allocate_fields(d::Domain)
    return (ψ = new_field(d), ϕ = new_field(d), q = new_field(d))
end

function allocate_rhs(d::Domain)
    return (
        ψ = new_rhs(d), ∂ψ = new_rhs(d), bψ = new_rhs(d),
        ϕ = new_rhs(d), ∂ϕ = new_rhs(d), bϕ = new_rhs(d)
    )
end

function set_background_ϕ!(ϕ, d::Domain, p::Params)
    for I = CartesianIndices(d)
        x, y, z = d[I]
        ϕ[I] = background_ϕ(z, p)
    end
    return ϕ
end


function set_background_ψ!(ψ, d::Domain, p::Params)
    for I = CartesianIndices(d)
        x, y, z = d[I]
        ψ[I] = background_ψ(z, p)
    end
    return ψ
end

function set_q!(q, d::Domain, p::Params; A = 0.1, L = 0.1, H = 0.1)
    πc = p.π0 - 0.5
    Π = p.Π
    for I = CartesianIndices(d)
        x, y, z = d[I]
        q[I] = 1 + A*exp(-(x^2 + y^2)/L^2)*exp(-(z-πc)^2/H^2)
    end
end