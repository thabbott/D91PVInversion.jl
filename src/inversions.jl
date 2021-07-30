abstract type AbstractIterativeInversion end

struct LinearizedInversion{F,R,L,D,P,T}
    ψ0   :: F
    ϕ0   :: F
    ψ′   :: F
    ϕ′   :: F
    q′   :: F
    x  :: R
    b  :: R
    L :: L
    domain :: D 
    params :: P
    atolϕ :: T
end

function LinearizedInversion(;
    ψ0, ϕ0, domain, params, atolϕ = 1.0
)
    ψ′, ϕ′, q′ = allocate_fields(domain)
    x, b = allocate_linearized_rhs(domain)
    L = generate_linearized_L(ψ0, ϕ0, domain; T = float_type(params))
    fill_ψ_halos!(ψ0, domain, params)
    fill_ϕ_halos!(ϕ0, domain, params)
    atolϕ = atolϕ/(params.S*params.Π^2)
    return LinearizedInversion(
        ψ0, ϕ0, ψ′, ϕ′, q′, 
        x, b, L,
        domain, params,
        atolϕ
    )
end

function initialize!(inv::LinearizedInversion, q′fun::Function, args...)
    q′ = inv.q′
    domain = inv.domain
    params = inv.params
    set_q′!(q′, domain, params, q′fun, args...)
    return inv
end

function initialize!(inv::LinearizedInversion, q′arr::AbstractArray)
    @. inv.q′ = q′arr
    return inv
end

function solve!(inv::LinearizedInversion; verbose = false, use_atol = true)
    ϕ′ = inv.ϕ′
    ψ′ = inv.ψ′
    q′ = inv.q′
    x = inv.x
    b = inv.b
    L = inv.L
    domain = inv.domain
    set_linearized_b!(b, q′, domain)
    atol = use_atol ? inv.atolϕ : zero(eltype(b))
    idrs!(x, L, b; log = false, verbose = verbose, abstol = atol)
    fields_from_linearized_rhs!(ϕ′, ψ′, x, domain)
    fill_ψ′_halos!(ψ′, domain)
    fill_ϕ′_halos!(ϕ′, domain)
    return inv
end

function save_inversion_results(fname, inv::LinearizedInversion)
    jldopen(fname, "w") do file
        file["ψ0"] =  inv.ψ0
        file["ϕ0"] =  inv.ϕ0
        file["ψ′"] =  inv.ψ′
        file["ϕ′"] =  inv.ϕ′
        file["q′"] =  inv.q′
        file["domain"] = inv.domain
        file["params"] = inv.params
    end
end

struct NLInversion{F,R,LS,LG,D,P,S} <: AbstractIterativeInversion
    ψ :: F
    ϕ :: F 
    q :: F 
    xψ :: R 
    ∂ψ :: R 
    bψ :: R 
    xϕ :: R 
    ∂ϕ :: R 
    bϕ :: R 
    Lψ :: LS 
    Lϕ :: LG
    domain :: D 
    params :: P
    sψ :: S
    sϕ :: S
end

function NLInversion(; 
    domain, params, 
    sψ = Solver(params; ω = 0.7),
    sϕ = Solver(params; ω = 0.7)
)

    ψ, ϕ, q = allocate_fields(domain)
    xψ, ∂ψ, bψ, xϕ, ∂ϕ, bϕ = allocate_rhs(domain)
    Lψ = generate_Lψ(ϕ, domain; T = float_type(params))
    Lϕ = generate_Lϕ(ψ, domain; T = float_type(params))
    return NLInversion(
        ψ, ϕ, q,
        xψ, ∂ψ, bψ,
        xϕ, ∂ϕ, bϕ,
        Lψ, Lϕ,
        domain, params,
        sψ, sϕ
    )

end

function initialize!(inv::NLInversion, q′::Function, args...)
    ψ = inv.ψ
    ϕ = inv.ϕ
    q = inv.q
    domain = inv.domain
    params = inv.params
    set_background_ψ!(ψ, domain, params)
    set_background_ϕ!(ϕ, domain, params)
    set_q!(q, domain, params, q′, args...)
    fill_ψ_halos!(ψ, domain, params)
    fill_ϕ_halos!(ϕ, domain, params)
    return inv
end

function initialize!(inv::NLInversion, ψi::AbstractArray, ϕi::AbstractArray, q′::Function, args...)
    ψ = inv.ψ
    ϕ = inv.ϕ
    q = inv.q
    domain = inv.domain
    params = inv.params
    @. ψ = ψi
    @. ϕ = ϕi
    set_q!(q, domain, params, q′, args...)
    fill_ψ_halos!(ψ, domain, params)
    fill_ϕ_halos!(ϕ, domain, params)
    return inv
end

function iterate!(inv::NLInversion; verbose = false)
    
    ψ = inv.ψ
    ϕ = inv.ϕ
    q = inv.q
    xψ = inv.xψ
    ∂ψ = inv.∂ψ
    bψ = inv.bψ
    xϕ = inv.xϕ
    ∂ϕ = inv.∂ϕ
    bϕ = inv.bϕ
    Lψ = inv.Lψ
    Lϕ = inv.Lϕ
    domain = inv.domain
    params = inv.params
    sψ = inv.sψ
    sϕ = inv.sϕ

    set_∂ψ!(∂ψ, ϕ, domain, params)
    set_bψ!(bψ, ψ, ϕ, q, domain)
    @. bψ = bψ - ∂ψ
    bicgstabl!(xψ, Lψ, bψ; log = false, verbose = false)
    relax!(ψ, xψ, sψ, domain; verbose = verbose)
    fill_ψ_halos!(ψ, domain, params)

    set_∂ϕ!(∂ϕ, ψ, domain, params)
    set_bϕ!(bϕ, ψ, ϕ, q, domain)
    @. bϕ = bϕ - ∂ϕ
    bicgstabl!(xϕ, Lϕ, bϕ; log = false, verbose = false)
    relax!(ϕ, xϕ, sϕ, domain; verbose = verbose)
    fill_ϕ_halos!(ϕ, domain, params)

    return inv

end

function is_converged(inv::NLInversion)
    return is_converged(inv.sψ) && is_converged(inv.sϕ)
end

function save_inversion_results(fname, inv::NLInversion)
    is_converged(inv) || throw(ArgumentError("inv has not converged"))
    jldopen(fname, "w") do file
        file["ψ"] =  inv.ψ
        file["ϕ"] =  inv.ϕ
        file["q"] =  inv.q
        file["domain"] = inv.domain
        file["params"] = inv.params
    end
end