function compute_qg_rψ(Lψ, xψ, ∂ψ, bψ, d::Domain)
    rψ = new_rhs(d)
    mul!(rψ, Lψ, xψ)
    @. rψ = rψ + ∂ψ - bψ
    return rψ
end
