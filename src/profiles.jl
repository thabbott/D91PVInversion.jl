function background_θ(z, p::Params)
    return p.θ0 + (p.π0 - z)
end

function background_ϕ(z, p::Params)
    θ = background_θ(z, p)
    return 0.5*(p.θ0 + θ)*(p.π0 - z)
end

function background_ψ(z, p::Params)
    return background_ϕ(z, p)
end
