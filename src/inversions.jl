abstract type AbstractIterativeInversion end

struct LinearInversion{F,R,LS,LP,D,P,S} <: AbstractIterativeInversion
    ψ0   :: F
    ϕ0   :: F
    ψ′   :: F
    ϕ′   :: F
    q′   :: F
    xψ′  :: R
    ∂ψ′  :: R
    bψ′  :: R
    xϕ′  :: R
    ∂ϕ′  :: R
    bϕ′  :: R
    Lψ′ :: LS
    Lϕ′ :: LP
    domain :: D 
    params :: P
    sψ′ :: S
    sϕ′ :: S
end

function LinearInversion(;
    ψ0, ϕ0, domain, params,
    sψ′ = Solver(params; ω = 0.7),
    sϕ′ = Solver(params; ω = 0.7)
)
    ψ′, ϕ′, q′ = allocate_linear_fields(domain)
    xψ′, ∂ψ′, bψ′, xϕ′, ∂ϕ′, bϕ′ = allocate_linear_rhs(domain)
    Lψ′ = generate_linear_Lψ(ϕ0, domain; T = float_type(params))
    Lϕ′ = generate_linear_Lϕ(ψ0, domain; T = float_type(params))
    return LinearInversion(
        ψ0, ϕ0, ψ′, ϕ′, q′, 
        xψ′, ∂ψ′, bψ′, 
        xϕ′, ∂ϕ′, bϕ′,
        Lψ′, Lϕ′, 
        domain, params, sψ′, sϕ′
    )
end

function initialize!(inv::LinearInversion, q′fun::Function, args...)
    ψ′ = inv.ψ′
    ϕ′ = inv.ϕ′
    q′ = inv.q′
    domain = inv.domain
    params = inv.params
    @. ψ′ = 0
    @. ϕ′ = 0
    set_q′!(q′, domain, params, q′fun, args...)
    fill_ψ′_halos!(ψ′, domain)
    fill_ϕ′_halos!(ϕ′, domain)
    return inv
end

function iterate!(inv::LinearInversion; verbose = false)
    ψ0 = inv.ψ0
    ϕ0 = inv.ϕ0
    ψ′ = inv.ψ′
    ϕ′ = inv.ϕ′
    q′ = inv.q′
    xψ′ = inv.xψ′
    ∂ψ′ = inv.∂ψ′
    bψ′ = inv.bψ′
    xϕ′ = inv.xϕ′
    ∂ϕ′ = inv.∂ϕ′
    bϕ′ = inv.bϕ′
    Lψ′ = inv.Lψ′
    Lϕ′ = inv.Lϕ′
    domain = inv.domain
    params = inv.params
    sψ′ = inv.sψ′
    sϕ′ = inv.sϕ′

    set_linear_∂ψ!(∂ψ′, ϕ0, domain, params)
    set_linear_bψ!(bψ′, ψ0, ϕ0, ψ′, ϕ′, q′, domain)
    @. bψ′ = bψ′ - ∂ψ′
    bicgstabl!(xψ′, Lψ′, bψ′; log = false, verbose = false)
    relax!(ψ′, xψ′, sψ′, domain; verbose = verbose)
    fill_ψ′_halos!(ψ′, domain)

    set_linear_∂ϕ!(∂ϕ′, ψ0, domain, params)
    set_linear_bϕ!(bϕ′, ψ0, ϕ0, ψ′, ϕ′, q′, domain)
    @. bϕ′ = bϕ′ - ∂ϕ′
    bicgstabl!(xϕ′, Lϕ′, bϕ′; log = false, verbose = false)
    relax!(ϕ′, xϕ′, sϕ′, domain; verbose = verbose)
    fill_ϕ′_halos!(ϕ′, domain)

    return inv
end

function is_converged(inv::LinearInversion)
    return is_converged(inv.sψ′) && is_converged(inv.sϕ′)
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