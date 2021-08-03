abstract type AbstractBC end
struct NeumannBC <: AbstractBC end
struct DirichletBC <: AbstractBC end

struct One end
struct MinusOne end
Base.getindex(::One, args...) = 1
Base.getindex(::MinusOne, args...) = -1
Base.:-(::One) = MinusOne()
Base.:-(::MinusOne) = One()

struct Derivative{O,F,D,S}
    ∂::O
    f::F
    domain::D
    scale::S
end
Base.getindex(d::Derivative, I::CartesianIndex{3}) = d.scale*d.∂(I,d.f,d.domain)
Base.:-(d::Derivative) = Derivative(d.∂, d.f, d.domain, -d.scale)
Base.:*(a, d::Derivative) = Derivative(d.∂, d.f, d.domain, a*d.scale)
Base.:*(d::Derivative, a) = *(a, d)

∂x²f(f,d::Domain) = Derivative(∂x²,f,d,1)
∂y²f(f,d::Domain) = Derivative(∂y²,f,d,1)
∂z²f(f,d::Domain) = Derivative(∂z²,f,d,1)
∂xyf(f,d::Domain) = Derivative(∂xy,f,d,1)
∂xzf(f,d::Domain) = Derivative(∂xz,f,d,1)
∂yzf(f,d::Domain) = Derivative(∂yz,f,d,1)

function new_sparse_matrix(d::Domain)
    N = prod(size(d))
    S = spzeros(N,N)
    l = LinearIndices(d)
    ldx = l[2,1,1] - l[1,1,1]
    ldy = l[1,2,1] - l[1,1,1]
    ldz = l[1,1,2] - l[1,1,1]
    for K = -1:1
        for J = -1:1
            for I = -1:1
                d = I*ldx + J*ldy + K*ldz
                n = N - abs(d)
                S += spdiagm(d => ones(n))
            end
        end
    end
    @. S = 0
    return S
end

function add_f∂a²!(S, f, da, Δa, Na, domain::Domain, Aa)
    Na = Na-1
    Δa² = Δa^2
    c = CartesianIndices(domain)
    l = LinearIndices(domain)
    for I = first(c):last(c)-Na*da
        S[l[I],l[I     ]] += -Aa/Δa² * f[I]
        S[l[I],l[I + da]] +=   1/Δa² * f[I]
    end
    for I = first(c)+da:last(c)-da
        S[l[I],l[I - da]] +=  1/Δa² * f[I]
        S[l[I],l[I     ]] += -2/Δa² * f[I]
        S[l[I],l[I + da]] +=  1/Δa² * f[I]
    end
    for I = first(c)+Na*da:last(c)
        S[l[I],l[I - da]] +=   1/Δa² * f[I]
        S[l[I],l[I     ]] += -Aa/Δa² * f[I]
    end
    return S
end
add_f∂x²!(S, f, domain::Domain) = add_f∂a²!(S, f, dx, domain.Δx, domain.nx, domain, 2)
add_f∂y²!(S, f, domain::Domain) = add_f∂a²!(S, f, dy, domain.Δy, domain.ny, domain, 2)
add_f∂z²!(S, f, domain::Domain) = add_f∂a²!(S, f, dz, domain.Δz, domain.nz, domain, 1)
add_∂x²!(S, domain::Domain) = add_f∂x²!(S, One(), domain)
add_∂y²!(S, domain::Domain) = add_f∂y²!(S, One(), domain)
add_∂z²!(S, domain::Domain) = add_f∂z²!(S, One(), domain)

function add_f∂ab!(S, f, da, db, Δa, Δb, Na, Nb, domain::Domain, Aa, Ab)
    Na = Na-1
    Nb = Nb-1
    c = CartesianIndices(domain)
    l = LinearIndices(domain)
    for I = first(c):last(c)-Na*da-Nb*db # a bottom, b bottom
        S[l[I],l[I + da + db]] +=      1/(4*Δa*Δb) * f[I]
        S[l[I],l[I          ]] +=  Aa*Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I + da     ]] += -   Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I      + db]] += -Aa   /(4*Δa*Δb) * f[I]
    end
    for I = first(c)+da:last(c)-da-Nb*db # a interior, b bottom
        S[l[I],l[I + da + db]] +=   1/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da     ]] +=  Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I + da     ]] += -Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da + db]] += - 1/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+Na*da:last(c)-Nb*db # a top, b bottom
        S[l[I],l[I      + db]] +=  Aa   /(4*Δa*Δb) * f[I]
        S[l[I],l[I - da     ]] +=     Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I          ]] += -Aa*Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da + db]] += -    1/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+db:last(c)-Na*da-db # a bottom, b interior
        S[l[I],l[I + da + db]] +=   1/(4*Δa*Δb) * f[I]
        S[l[I],l[I      - db]] +=  Aa/(4*Δa*Δb) * f[I]
        S[l[I],l[I + da - db]] += - 1/(4*Δa*Δb) * f[I]
        S[l[I],l[I      + db]] += -Aa/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+da+db:last(c)-da-db # a interior, b interior
        S[l[I],l[I + da + db]] +=  1/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da - db]] +=  1/(4*Δa*Δb) * f[I]
        S[l[I],l[I + da - db]] += -1/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da + db]] += -1/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+Na*da+db:last(c)-db # a top, b interior
        S[l[I],l[I      + db]] +=  Aa/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da - db]] +=   1/(4*Δa*Δb) * f[I]
        S[l[I],l[I      - db]] += -Aa/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da + db]] += - 1/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+Nb*db:last(c)-Na*da # a bottom, b top
        S[l[I],l[I + da     ]] +=     Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I      - db]] +=  Aa   /(4*Δa*Δb) * f[I]
        S[l[I],l[I + da - db]] += -    1/(4*Δa*Δb) * f[I]
        S[l[I],l[I          ]] += -Aa*Ab/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+da+Nb*db:last(c)-da # a interior, b top
        S[l[I],l[I + da     ]] +=  Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da - db]] +=   1/(4*Δa*Δb) * f[I]
        S[l[I],l[I + da - db]] += - 1/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da     ]] += -Ab/(4*Δa*Δb) * f[I]
    end
    for I = first(c)+Na*da+Nb*db:last(c) # a top, b top
        S[l[I],l[I          ]] +=  Aa*Ab/(4*Δa*Δb) * f[I]
        S[l[I],l[I - da - db]] +=      1/(4*Δa*Δb) * f[I]
        S[l[I],l[I      - db]] += -Aa   /(4*Δa*Δb) * f[I]
        S[l[I],l[I - da     ]] += -   Ab/(4*Δa*Δb) * f[I]
    end
    return S
end
add_f∂xy!(S, f, domain::Domain) = add_f∂ab!(S, f, dx, dy, domain.Δx, domain.Δy, domain.nx, domain.ny, domain, 0, 0)
add_f∂xz!(S, f, domain::Domain) = add_f∂ab!(S, f, dx, dz, domain.Δx, domain.Δz, domain.nx, domain.nz, domain, 0, 1)
add_f∂yz!(S, f, domain::Domain) = add_f∂ab!(S, f, dy, dz, domain.Δy, domain.Δz, domain.ny, domain.nz, domain, 0, 1)
add_∂xy!(S, domain::Domain) = add_f∂xy!(S, One(), domain)
add_∂xz!(S, domain::Domain) = add_f∂xz!(S, One(), domain)
add_∂yz!(S, domain::Domain) = add_f∂yz!(S, One(), domain)

function generate_sparse_linearized_L(ψ0, ϕ0, d::Domain)

    UL = new_sparse_matrix(d)
    add_∂z²!(UL, d)
    add_f∂z²!(UL, ∂x²f(ψ0,d), d)
    add_f∂z²!(UL, ∂y²f(ψ0,d), d)
    add_f∂xz!(UL, -∂xzf(ψ0,d), d)
    add_f∂yz!(UL, -∂yzf(ψ0,d), d)

    UR = new_sparse_matrix(d)
    add_f∂x²!(UR, ∂z²f(ϕ0,d), d)
    add_f∂y²!(UR, ∂z²f(ϕ0,d), d)
    add_f∂xz!(UR, -∂xzf(ϕ0,d), d) 
    add_f∂yz!(UR, -∂yzf(ϕ0,d), d)

    BL = new_sparse_matrix(d)
    add_∂x²!(BL, d)
    add_∂y²!(BL, d)

    BR = new_sparse_matrix(d)
    add_f∂x²!(BR, -One(), d)
    add_f∂y²!(BR, -One(), d)
    add_f∂x²!(BR, -2*∂y²f(ψ0,d), d)
    add_f∂y²!(BR, -2*∂x²f(ψ0,d), d)
    add_f∂xy!(BR, 4*∂xyf(ψ0,d), d)

    S = hvcat((2,2), UL, UR, BL, BR)
    return S
    
end

