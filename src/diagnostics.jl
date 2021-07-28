function diagnose_u(ψ, d::Domain)
    u = new_field(d)
    for I = CartesianIndices(d)
        u[I] = -∂y(I,ψ,d)
    end
    return u
end

function diagnose_v(ψ, d::Domain)
    v = new_field(d)
    for I = CartesianIndices(d)
        v[I] = ∂x(I,ψ,d)
    end
    return v
end

function diagnose_umag(ψ, d::Domain)
    umag = new_field(d)
    for I = CartesianIndices(d)
        umag[I] = sqrt(∂y(I,ψ,d)^2 + ∂x(I,ψ,d)^2)
    end
    return umag
end

function diagnose_ζ(ψ, d::Domain)
    ζ = new_field(d)
    for I = CartesianIndices(d)
        ζ[I] = ∂x²(I,ψ,d) + ∂y²(I,ψ,d)
    end
    return ζ
end

function diagnose_θ(ϕ, d::Domain)
    θ = new_field(d)
    for I = CartesianIndices(d)
        θ[I] = -∂z(I,ϕ,d)
    end
    return θ
end

function diagnose_penetration_depth(ψ, q, d::Domain)
    qmax, Imax = findmax(q)
    kmax = Imax[3]
    umag = diagnose_umag(ψ, d)
    ubot = umag[:,:,1]
    umax = umag[:,:,kmax]
    return maximum(ubot)/maximum(umax)
end