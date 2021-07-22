function set_qg_∂ψ!(∂ψ, d::Domain, p::Params)
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
