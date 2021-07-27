function set_linear_bψ!(bψ′, ψ0, ϕ0, ψ′, ϕ′, q′, d::Domain)
    c = CartesianIndices(d)
    l = LinearIndices(d)
    for I = c
        bψ′[l[I]] = (
            q′[I] + ∂x²(I,ϕ′,d) + ∂y²(I,ϕ′,d) - (1 + ∂x²(I,ψ0,d) + ∂y²(I,ψ0,d))*∂z²(I,ϕ′,d)
          - 2*(∂x²(I,ψ′,d)*∂y²(I,ψ0,d) + ∂x²(I,ψ0,d)*∂y²(I,ψ′,d) - 2*∂xy(I,ψ′,d)*∂xy(I,ψ0,d))
          + ∂xz(I,ψ′,d)*∂xz(I,ϕ0,d) + ∂xz(I,ψ0,d)*∂xz(I,ϕ′,d) + ∂yz(I,ψ′,d)*∂yz(I,ϕ0,d) + ∂yz(I,ψ0,d)*∂yz(I,ϕ′,d)
        )
    end
    return bψ′
end

function set_bψ!(bψ, ψ, ϕ, q, d::Domain)
    c = CartesianIndices(d)
    l = LinearIndices(d)
    for I = c
        bψ[l[I]] = (
            q[I] + ∂x²(I,ϕ,d) + ∂y²(I,ϕ,d) - ∂z²(I,ϕ,d)
          - 2*(∂x²(I,ψ,d)*∂y²(I,ψ,d) - ∂xy(I,ψ,d)^2)
          + ∂xz(I,ψ,d)*∂xz(I,ϕ,d) + ∂yz(I,ψ,d)*∂yz(I,ϕ,d)
        )
    end
    return bψ
end

function set_linear_bϕ!(bϕ′, ψ0, ϕ0, ψ′, ϕ′, q′, d::Domain)
    c = CartesianIndices(d)
    l = LinearIndices(d)
    for I = c
        bϕ′[l[I]] = (
            q′[I] + (1 - ∂z²(I,ϕ0,d))*(∂x²(I,ψ′,d) + ∂y²(I,ψ′,d))
          + 2*(∂x²(I,ψ′,d)*∂y²(I,ψ0,d) + ∂x²(I,ψ0,d)*∂y²(I,ψ′,d) - 2*∂xy(I,ψ′,d)*∂xy(I,ψ0,d))
          + ∂xz(I,ψ′,d)*∂xz(I,ϕ0,d) + ∂xz(I,ψ0,d)*∂xz(I,ϕ′,d) + ∂yz(I,ψ′,d)*∂yz(I,ϕ0,d) + ∂yz(I,ψ0,d)*∂yz(I,ϕ′,d)
        )
    end
    return bϕ′
end


function set_bϕ!(bϕ, ψ, ϕ, q, d::Domain)
    c = CartesianIndices(d)
    l = LinearIndices(d)
    for I = c
        bϕ[l[I]] = (
            q[I] + ∂x²(I,ψ,d) + ∂y²(I,ψ,d)
          + 2*(∂x²(I,ψ,d)*∂y²(I,ψ,d) - ∂xy(I,ψ,d)^2)
          + ∂xz(I,ψ,d)*∂xz(I,ϕ,d) + ∂yz(I,ψ,d)*∂yz(I,ϕ,d)
        )
    end
    return bϕ
end