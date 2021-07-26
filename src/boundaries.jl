function set_linear_∂ψ!(∂ψ, d::Domain, p::Params)
    Δx² = d.Δx^2
    Δy² = d.Δy^2
    Δz² = d.Δz^2
    ψ0(z) = background_ψ(z, p)
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
    # ∂y²
    for I = c[:,1,:]
        x, y, z = d[I-dy]
        ∂ψ[l[I]] += ψ0(z)/Δy²
    end
    for I = c[:,end,:]
        x, y, z = d[I+dy]
        ∂ψ[l[I]] += ψ0(z)/Δy²
    end
    # ∂z²
    for I = c[:,:,1]
        x′, y′, z′ = d[I]
        x, y, z = d[I-dz]
        ∂ψ[l[I]] += (ψ0(z) - ψ0(z′))/Δz²
    end
    for I = c[:,:,end]
        x′, y′, z′ = d[I]
        x, y, z = d[I+dz]
        ∂ψ[l[I]] += (ψ0(z) - ψ0(z′))/Δz²
    end
    return ∂ψ
end

function set_∂ψ!(∂ψ, ϕ, d::Domain, p::Params)
    Δx² = d.Δx^2
    Δy² = d.Δy^2
    ψ0(z) = background_ψ(z, p)
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

function set_∂ϕ!(∂ϕ, ψ, d::Domain, p::Params)
    Δx² = d.Δx^2
    Δy² = d.Δy^2
    Δz² = d.Δz^2
    ϕ0(z) = background_ϕ(z, p)
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
