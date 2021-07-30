function generate_linearized_L(ψ0, ϕ0, d::Domain; T = Float64)

    function L!(y::AbstractVector, x::AbstractVector)

        # Set up indices and views
        Δx = d.Δx
        Δy = d.Δy
        Δz = d.Δz
        c = CartesianIndices(d)
        l = LinearIndices(d)
        oc = last(c)
        ol = last(l)
        ϕ′ = @view x[1:ol]
        ψ′ = @view x[ol+1:end]
        q′ = @view y[1:ol]
        nil = @view y[ol+1:end]

        # Compute q′
        @. q′ = 0
        # (1 + ∇²ψ0)*∂z²ϕ′ (Neumann BC)
        for I = c[:,:,1]
            q′[l[I]] += (1 + ∂x²(I,ψ0,d) + ∂y²(I,ψ0,d))*(-ϕ′[l[I]] + ϕ′[l[I+dz]])/Δz^2
        end
        for I = c[:,:,2:end-1]
            q′[l[I]] += (1 + ∂x²(I,ψ0,d) + ∂y²(I,ψ0,d))*(ϕ′[l[I-dz]] - 2ϕ′[l[I]] + ϕ′[l[I+dz]])/Δz^2
        end
        for I = c[:,:,end]
            q′[l[I]] += (1 + ∂x²(I,ψ0,d) + ∂y²(I,ψ0,d))*(ϕ′[l[I-dz]] - ϕ′[l[I]])/Δz^2
        end
        # ∂z²ϕ0*∂x²ψ′ (Dirichlet BC)
        for I = c[1,:,:]
            q′[l[I]] += ∂z²(I,ϕ0,d)*(-2ψ′[l[I]] + ψ′[l[I+dx]])/Δx^2
        end
        for I = c[2:end-1,:,:]
            q′[l[I]] += ∂z²(I,ϕ0,d)*(ψ′[l[I-dx]] - 2ψ′[l[I]] + ψ′[l[I+dx]])/Δx^2
        end
        for I = c[end,:,:]
            q′[l[I]] += ∂z²(I,ϕ0,d)*(ψ′[l[I-dx]] - 2ψ′[l[I]])/Δx^2
        end
        # ∂z²ϕ0*∂y²ψ′ (Dirichlet BC)
        for I = c[:,1,:]
            q′[l[I]] += ∂z²(I,ϕ0,d)*(-2ψ′[l[I]] + ψ′[l[I+dy]])/Δy^2
        end
        for I = c[:,2:end-1,:]
            q′[l[I]] += ∂z²(I,ϕ0,d)*(ψ′[l[I-dy]] - 2ψ′[l[I]] + ψ′[l[I+dy]])/Δy^2
        end
        for I = c[:,end,:]
            q′[l[I]] += ∂z²(I,ϕ0,d)*(ψ′[l[I-dy]] - 2ψ′[l[I]])/Δy^2
        end
        # -∂xzψ0*∂xzϕ′ (Neumann on z, Dirichlet on x)
        for I = c[1,:,1]         # x bottom, z bottom
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I+dx+dz]] - ϕ′[l[I+dx]])/(4*Δx*Δz)
        end
        for I = c[2:end-1,:,1]   # x interior, z bottom
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I+dx+dz]] + ϕ′[l[I-dx]] - ϕ′[l[I-dx+dz]] - ϕ′[l[I+dx]])/(4*Δx*Δz)
        end
        for I = c[end,:,1]       # x top, z bottom
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I-dx]] - ϕ′[l[I-dx+dz]])/(4*Δx*Δz)
        end
        for I = c[1,:,2:end-1]     # x bottom, z interior
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I+dx+dz]] - ϕ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[2:end-1,:,2:end-1] # x interior, z interior
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I+dx+dz]] + ϕ′[l[I-dx-dz]] - ϕ′[l[I-dx+dz]] - ϕ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[end,:,2:end-1]   # x top, z interior
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I-dx-dz]] - ϕ′[l[I-dx+dz]])/(4*Δx*Δz)
        end
        for I = c[1,:,end]       # x bottom, z top
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I+dx]] - ϕ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[2:end-1,:,end]   # x interior, z top
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I+dx]] + ϕ′[l[I-dx-dz]] - ϕ′[l[I-dx]] - ϕ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[end,:,end]     # x top, z top
            q′[l[I]] += -∂xz(I,ψ0,d)*(ϕ′[l[I-dx-dz]] - ϕ′[l[I-dx]])/(4*Δx*Δz)
        end
        # -∂xzϕ0*∂xzψ′ (Neumann on z, Dirichlet on x)
        for I = c[1,:,1]         # x bottom, z bottom
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I+dx+dz]] - ψ′[l[I+dx]])/(4*Δx*Δz)
        end
        for I = c[2:end-1,:,1]   # x interior, z bottom
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I+dx+dz]] + ψ′[l[I-dx]] - ψ′[l[I-dx+dz]] - ψ′[l[I+dx]])/(4*Δx*Δz)
        end
        for I = c[end,:,1]       # x top, z bottom
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I-dx]] - ψ′[l[I-dx+dz]])/(4*Δx*Δz)
        end
        for I = c[1,:,2:end-1]     # x bottom, z interior
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I+dx+dz]] - ψ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[2:end-1,:,2:end-1] # x interior, z interior
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I+dx+dz]] + ψ′[l[I-dx-dz]] - ψ′[l[I-dx+dz]] - ψ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[end,:,2:end-1]   # x top, z interior
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I-dx-dz]] - ψ′[l[I-dx+dz]])/(4*Δx*Δz)
        end
        for I = c[1,:,end]       # x bottom, z top
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I+dx]] - ψ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[2:end-1,:,end]   # x interior, z top
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I+dx]] + ψ′[l[I-dx-dz]] - ψ′[l[I-dx]] - ψ′[l[I+dx-dz]])/(4*Δx*Δz)
        end
        for I = c[end,:,end]     # x top, z top
            q′[l[I]] += -∂xz(I,ϕ0,d)*(ψ′[l[I-dx-dz]] - ψ′[l[I-dx]])/(4*Δx*Δz)
        end
        # -∂yzψ0*∂yzϕ′ (Neumann on z, Dirichlet on y)
        for I = c[:,1,1]         # y bottom, z bottom
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I+dy+dz]] - ϕ′[l[I+dy]])/(4*Δy*Δz)
        end
        for I = c[:,2:end-1,1]   # y interior, z bottom
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I+dy+dz]] + ϕ′[l[I-dy]] - ϕ′[l[I-dy+dz]] - ϕ′[l[I+dy]])/(4*Δy*Δz)
        end
        for I = c[:,end,1]       # y top, z bottom
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I-dy]] - ϕ′[l[I-dy+dz]])/(4*Δy*Δz)
        end
        for I = c[:,1,2:end-1]     # y bottom, z interior
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I+dy+dz]] - ϕ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,2:end-1,2:end-1] # y interior, z interior
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I+dy+dz]] + ϕ′[l[I-dy-dz]] - ϕ′[l[I-dy+dz]] - ϕ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,end,2:end-1]   # y top, z interior
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I-dy-dz]] - ϕ′[l[I-dy+dz]])/(4*Δy*Δz)
        end
        for I = c[:,1,end]       # y bottom, z top
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I+dy]] - ϕ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,2:end-1,end]   # y interior, z top
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I+dy]] + ϕ′[l[I-dy-dz]] - ϕ′[l[I-dy]] - ϕ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,end,end]     # y top, z top
            q′[l[I]] += -∂yz(I,ψ0,d)*(ϕ′[l[I-dy-dz]] - ϕ′[l[I-dy]])/(4*Δy*Δz)
        end
        # -∂yzϕ0*∂yzψ′ (Neumann on z, Dirichlet on y)
        for I = c[:,1,1]         # y bottom, z bottom
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I+dy+dz]] - ψ′[l[I+dy]])/(4*Δy*Δz)
        end
        for I = c[:,2:end-1,1]   # y interior, z bottom
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I+dy+dz]] + ψ′[l[I-dy]] - ψ′[l[I-dy+dz]] - ψ′[l[I+dy]])/(4*Δy*Δz)
        end
        for I = c[:,end,1]       # y top, z bottom
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I-dy]] - ψ′[l[I-dy+dz]])/(4*Δy*Δz)
        end
        for I = c[:,1,2:end-1]     # y bottom, z interior
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I+dy+dz]] - ψ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,2:end-1,2:end-1] # y interior, z interior
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I+dy+dz]] + ψ′[l[I-dy-dz]] - ψ′[l[I-dy+dz]] - ψ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,end,2:end-1]   # y top, z interior
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I-dy-dz]] - ψ′[l[I-dy+dz]])/(4*Δy*Δz)
        end
        for I = c[:,1,end]       # y bottom, z top
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I+dy]] - ψ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,2:end-1,end]   # y interior, z top
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I+dy]] + ψ′[l[I-dy-dz]] - ψ′[l[I-dy]] - ψ′[l[I+dy-dz]])/(4*Δy*Δz)
        end
        for I = c[:,end,end]     # y top, z top
            q′[l[I]] += -∂yz(I,ϕ0,d)*(ψ′[l[I-dy-dz]] - ψ′[l[I-dy]])/(4*Δy*Δz)
        end
        
        # Compute residual of nonlinear balance equation
        @. nil = 0
        # ∂x²ϕ′ - (1 + 2∂y²ψ0)∂x²ψ′ (Dirichlet BC)
        for I = c[1,:,:]
            nil[l[I]] += (
                (-2ϕ′[l[I]] + ϕ′[l[I+dx]])
              - (1 + 2*∂y²(I,ψ0,d))*(-2ψ′[l[I]] + ψ′[l[I+dx]])
            )/Δx^2
        end
        for I = c[2:end-1,:,:]
            nil[l[I]] += (
                (ϕ′[l[I-dx]] - 2ϕ′[l[I]] + ϕ′[l[I+dx]])
              - (1 + 2*∂y²(I,ψ0,d))*(ψ′[l[I-dx]] - 2ψ′[l[I]] + ψ′[l[I+dx]])
            )/Δx^2
        end
        for I = c[end,:,:]
            nil[l[I]] += (
                (ϕ′[l[I-dx]] - 2ϕ′[l[I]])
              - (1 + 2*∂y²(I,ψ0,d))*(ψ′[l[I-dx]] - 2ψ′[l[I]])
            )/Δx^2
        end
        # ∂y²ϕ′ - (1 + 2∂x²ψ0)∂y²ψ′ (Dirichlet BC)
        for I = c[:,1,:]
            nil[l[I]] += (
                (-2ϕ′[l[I]] + ϕ′[l[I+dy]])
              - (1 + 2*∂x²(I,ψ0,d))*(-2ψ′[l[I]] + ψ′[l[I+dy]])
            )/Δy^2
        end
        for I = c[:,2:end-1,:]
            nil[l[I]] += (
                (ϕ′[l[I-dy]] - 2ϕ′[l[I]] + ϕ′[l[I+dy]])
              - (1 + 2*∂x²(I,ψ0,d))*(ψ′[l[I-dy]] - 2ψ′[l[I]] + ψ′[l[I+dy]])
            )/Δy^2
        end
        for I = c[:,end,:]
            nil[l[I]] += (
                (ϕ′[l[I-dy]] - 2ϕ′[l[I]])
              - (1 + 2*∂x²(I,ψ0,d))*(ψ′[l[I-dy]] - 2ψ′[l[I]])
            )/Δy^2
        end
        # 4∂xyψ0*∂xyψ′ (Dirichlet BC)
        for I = c[1,1,:]             # x bottom, y bottom     
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I+dx+dy]])/(4*Δx*Δy)
        end
        for I = c[2:end-1,1,:]       # x interior, y bottom
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I+dx+dy]] - ψ′[l[I-dx+dy]])/(4*Δx*Δy)
        end
        for I = c[end,1,:]           # x top, y bottom
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(-ψ′[l[I-dx+dy]])/(4*Δx*Δy)
        end
        for I = c[1,2:end-1,:]       # x bottom, y interior
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I+dx+dy]] - ψ′[l[I+dx-dy]])/(4*Δx*Δy)
        end
        for I = c[2:end-1,2:end-1,:] # x interior, y interior
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I+dx+dy]] + ψ′[l[I-dx-dy]] - ψ′[l[I-dx+dy]] - ψ′[l[I+dx-dy]])/(4*Δx*Δy)
        end
        for I = c[end,2:end-1,:]     # x top, y interior
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I-dx-dy]] - ψ′[l[I-dx+dy]])/(4*Δx*Δy)
        end
        for I = c[1,end,:]           # x bottom, y top
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(-ψ′[l[I+dx-dy]])/(4*Δx*Δy)     
        end
        for I = c[2:end-1,end,:]     # x interior, y top
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I-dx-dy]] - ψ′[l[I+dx-dy]])/(4*Δx*Δy)
        end
        for I = c[end,end,:]         # x top, y top
            nil[l[I]] += 4*∂xy(I,ψ0,d)*(ψ′[l[I-dx-dy]])/(4*Δx*Δy)
        end

    end

    return LinearMap{T}(L!, 2*prod(size(d)); ismutating = true)
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