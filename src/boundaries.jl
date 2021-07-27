function set_∂ψ!(∂ψ, ϕ, d::Domain, p::Params, ψ0::Function)
    Δx² = d.Δx^2
    Δy² = d.Δy^2
    c = CartesianIndices(d)
    l = LinearIndices(d)
    @. ∂ψ = 0
    # ∂x²
    for I = c[1,:,:]
        x, y, z = d[I-dx]
        ∂ψ[l[I]] += ψ0(z)/Δx²
    end
    for I = c[end,:,:]
        x, y, z = d[I+dx]
        ∂ψ[l[I]] += ψ0(z)/Δx²
    end
    # + ∂y²
    for I = c[:,1,:]
        x, y, z = d[I-dy]
        ∂ψ[l[I]] += ψ0(z)/Δy²
    end
    for I = c[:,end,:]
        x, y, z = d[I+dy]
        ∂ψ[l[I]] += ψ0(z)/Δy²
    end
    # * (1 + ∂z²ϕ)
    for I = c
        ∂ψ[l[I]] *= (1 + ∂z²(I,ϕ,d))
    end
    return ∂ψ
end

function set_linear_∂ψ!(∂ψ′, ϕ0, d::Domain, p::Params)
    return set_∂ψ!(∂ψ′, ϕ0, d, p, (z) -> 0.0)
end

function set_∂ψ!(∂ψ, ϕ, d::Domain, p::Params)
    return set_∂ψ!(∂ψ, ϕ, d, p, (z) -> background_ψ(z,p))
end

function set_∂ϕ!(∂ϕ, ψ, d::Domain, p::Params, ϕ0::Function)
    Δx² = d.Δx^2
    Δy² = d.Δy^2
    Δz² = d.Δz^2
    c = CartesianIndices(d)
    l = LinearIndices(d)
    @. ∂ϕ = 0
    # ∂z²
    for I = c[:,:,1]
        x′, y′, z′ = d[I]
        x, y, z = d[I-dz]
        ∂ϕ[l[I]] += (ϕ0(z) - ϕ0(z′))/Δz²
    end
    for I = c[:,:,end]
        x′, y′, z′ = d[I]
        x, y, z = d[I+dz]
        ∂ϕ[l[I]] += (ϕ0(z) - ϕ0(z′))/Δz²
    end
    # * (1 + ∇²ψ)
    for I = c
        ∂ϕ[l[I]] *= (1 + ∂x²(I,ψ,d) + ∂y²(I,ψ,d))
    end
    # + ∂x²
    for I = c[1,:,:]
        x, y, z = d[I-dx]
        ∂ϕ[l[I]] += ϕ0(z)/Δx²
    end
    for I = c[end,:,:]
        x, y, z = d[I+dx]
        ∂ϕ[l[I]] += ϕ0(z)/Δx²
    end
    # + ∂y²
    for I = c[:,1,:]
        x, y, z = d[I-dy]
        ∂ϕ[l[I]] += ϕ0(z)/Δy²
    end
    for I = c[:,end,:]
        x, y, z = d[I+dy]
        ∂ϕ[l[I]] += ϕ0(z)/Δy²
    end
    return ∂ϕ
end

function set_linear_∂ϕ!(∂ϕ′, ψ0, d::Domain, p::Params)
    return set_∂ϕ!(∂ϕ′, ψ0, d, p, (z) -> 0.0)
end

function set_∂ϕ!(∂ϕ, ψ, d::Domain, p::Params)
    return set_∂ϕ!(∂ϕ, ψ, d, p, (z) -> background_ϕ(z,p))
end
