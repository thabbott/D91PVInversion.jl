function fill_halos!(f, domain, f0)
    nx, ny, nz = size(domain)
    hx, hy, hz = halo(domain)
    for k = 1-hx:nz+hx
        for j = 1-hy:ny+hy
            for i = 1-hx:0
                x, y, z = domain[i,j,k]
                f[i,j,k] = f0(z)
            end
            for i = nx+1:nx+hx
                x, y, z = domain[i,j,k]
                f[i,j,k] = f0(z)
            end
        end
        for j = 1-hy:0
            for i = 1:nx
                x, y, z = domain[i,j,k]
                f[i,j,k] = f0(z)
            end
        end
        for j = ny+1:ny+hy
            for i = 1:nx
                x, y, z = domain[i,j,k]
                f[i,j,k] = f0(z)
            end
        end
    end
    for k = 1-hz:0
        for j = 1:ny
            for i = 1:nx
                x′, y′, z′ = domain[i,j,1]
                x, y, z = domain[i,j,k]
                f[i,j,k] = f[i,j,1] + f0(z) - f0(z′)
            end
        end
    end
    for k = nz+1:nz+hz
        for j = 1:ny
            for i = 1:nx
                x′, y′, z′ = domain[i,j,nz]
                x, y, z = domain[i,j,k]
                f[i,j,k] = f[i,j,nz] + f0(z) - f0(z′)
            end
        end
    end
    return f
end

function fill_ψ_halos!(ψ, d::Domain, p::Params)
    return fill_halos!(ψ, d, (z) -> background_ψ(z, p))
end

function fill_ϕ_halos!(ϕ, d::Domain, p::Params)
    return fill_halos!(ϕ, d, (z) -> background_ϕ(z, p))
end