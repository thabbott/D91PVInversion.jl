function compute_residual(L, x, ∂, b, d::Domain)
    ϵ = new_rhs(d)
    mul!(ϵ, L, x)
    @. ϵ = ϵ + ∂ - b
    return ϵ
end
