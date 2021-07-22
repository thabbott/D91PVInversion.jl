function generate_qg_Lψ(d::Domain; T = Float64)
    function L!(y::AbstractVector, x::AbstractVector)
        Δx² = d.Δx^2
        Δy² = d.Δy^2
        Δz² = d.Δz^2
        c = CartesianIndices(d)
        l = LinearIndices(d)
        # ∂z² (Neumann BC)
        for I = c[:,:,1]
            y[l[I]] = (-x[l[I]] + x[l[I+dz]])/Δz²
        end
        for I = c[:,:,2:end-1]
            y[l[I]] = (x[l[I-dz]] - 2x[l[I]] + x[l[I+dz]])/Δz²
        end
        for I = c[:,:,end]
            y[l[I]] = (x[l[I-dz]] - x[l[I]])/Δz²
        end
        # + ∂x² (Dirichlet BC)
        for I = c[1,:,:]
            y[l[I]] += (-2x[l[I]] + x[l[I+dx]])/Δx²
        end
        for I = c[2:end-1,:,:]
            y[l[I]] += (x[l[I-dx]] - 2x[l[I]] + x[l[I+dx]])/Δx²
        end
        for I = c[end,:,:]
            y[l[I]] += (x[l[I-dx]] - 2x[l[I]])/Δx²
        end
        # + ∂y² (Dirichlet BC)
        for I = c[:,1,:]
            y[l[I]] += (-2x[l[I]] + x[l[I+dy]])/Δy²
        end
        for I = c[:,2:end-1,:]
            y[l[I]] += (x[l[I-dy]] - 2x[l[I]] + x[l[I+dy]])/Δy²
        end
        for I = c[:,end,:]
            y[l[I]] += (x[l[I-dy]] - 2x[l[I]])/Δy²
        end
    end

    return LinearMap{T}(L!, prod(size(d)); ismutating = true)
end


function generate_Lψ(ϕ, d::Domain; T = Float64)

    function L!(y::AbstractVector, x::AbstractVector)
        Δx² = d.Δx^2
        Δy² = d.Δy^2
        c = CartesianIndices(d)
        l = LinearIndices(d)
        # ∂x² (Dirichlet BC)
        for I = c[1,:,:]
            y[l[I]] = (-2x[l[I]] + x[l[I+dx]])/Δx²
        end
        for I = c[2:end-1,:,:]
            y[l[I]] = (x[l[I-dx]] - 2x[l[I]] + x[l[I+dx]])/Δx²
        end
        for I = c[end,:,:]
            y[l[I]] = (x[l[I-dx]] - 2x[l[I]])/Δx²
        end
        # + ∂y² (Dirichlet BC)
        for I = c[:,1,:]
            y[l[I]] += (-2x[l[I]] + x[l[I+dy]])/Δy²
        end
        for I = c[:,2:end-1,:]
            y[l[I]] += (x[l[I-dy]] - 2x[l[I]] + x[l[I+dy]])/Δy²
        end
        for I = c[:,end,:]
            y[l[I]] += (x[l[I-dy]] - 2x[l[I]])/Δy²
        end
        # * (1 + ∂z²ϕ)
        for I = c
            y[l[I]] *= (1 + ∂z²(I,ϕ,d))
        end
    end

    return LinearMap{T}(L!, prod(size(d)); ismutating = true)
end

function generate_Lϕ(ψ, d::Domain; T = Float64)

    function L!(y::AbstractVector, x::AbstractVector)
        Δx² = d.Δx^2
        Δy² = d.Δy^2
        Δz² = d.Δz^2
        c = CartesianIndices(d)
        l = LinearIndices(d)
        # ∂z² (Neumann BC)
        for I = c[:,:,1]
            y[l[I]] = (-x[l[I]] + x[l[I+dz]])/Δz²
        end
        for I = c[:,:,2:end-1]
            y[l[I]] = (x[l[I-dz]] - 2x[l[I]] + x[l[I+dz]])/Δz²
        end
        for I = c[:,:,end]
            y[l[I]] = (x[l[I-dz]] - x[l[I]])/Δz²
        end
        # * (1 + ∇²ψ)
        for I = c
            y[l[I]] *= (1 + ∂x²(I,ψ,d) + ∂y²(I,ψ,d))
        end
        # + ∂x² (Dirichlet BC)
        for I = c[1,:,:]
            y[l[I]] += (-2x[l[I]] + x[l[I+dx]])/Δx²
        end
        for I = c[2:end-1,:,:]
            y[l[I]] += (x[l[I-dx]] - 2x[l[I]] + x[l[I+dx]])/Δx²
        end
        for I = c[end,:,:]
            y[l[I]] += (x[l[I-dx]] - 2x[l[I]])/Δx²
        end
        # + ∂y² (Dirichlet BC)
        for I = c[:,1,:]
            y[l[I]] += (-2x[l[I]] + x[l[I+dy]])/Δy²
        end
        for I = c[:,2:end-1,:]
            y[l[I]] += (x[l[I-dy]] - 2x[l[I]] + x[l[I+dy]])/Δy²
        end
        for I = c[:,end,:]
            y[l[I]] += (x[l[I-dy]] - 2x[l[I]])/Δy²
        end
    end

    return LinearMap{T}(L!, prod(size(d)); ismutating = true)
end

∂x(I::CartesianIndex{3},f,d::Domain) = (f[I+dx] - f[I-dx])/(2d.Δx)
∂y(I::CartesianIndex{3},f,d::Domain) = (f[I+dy] - f[I-dy])/(2d.Δy)
∂z(I::CartesianIndex{3},f,d::Domain) = (f[I+dz] - f[I-dz])/(2d.Δz)
∂x²(I::CartesianIndex{3},f,d::Domain) = (f[I-dx] - 2f[I] + f[I+dx])/d.Δx^2
∂y²(I::CartesianIndex{3},f,d::Domain) = (f[I-dy] - 2f[I] + f[I+dy])/d.Δy^2
∂z²(I::CartesianIndex{3},f,d::Domain) = (f[I-dz] - 2f[I] + f[I+dz])/d.Δz^2
∂xy(I::CartesianIndex{3},f,d::Domain) = (
    f[I+dx+dy] + f[I-dx-dy] - f[I+dx-dy] - f[I-dx+dy]
)/(4*d.Δx*d.Δy)
∂xz(I::CartesianIndex{3},f,d::Domain) = (
    f[I+dx+dz] + f[I-dx-dz] - f[I+dx-dz] - f[I-dx+dz]
)/(4*d.Δx*d.Δz)
∂yz(I::CartesianIndex{3},f,d::Domain) = (
    f[I+dy+dz] + f[I-dy-dz] - f[I+dy-dz] - f[I-dy+dz]
)/(4*d.Δy*d.Δz)