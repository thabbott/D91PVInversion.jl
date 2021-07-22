function set_qg_bψ!(bψ, q, d::Domain)
    c = CartesianIndices(d)
    l = LinearIndices(d)
    for I = c
        bψ[l[I]] = q[I]
    end
    return bψ
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