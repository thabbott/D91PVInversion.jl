function relax!(f, xf, s::Solver, d::Domain; verbose = false)
    ω = s.ω
    c = CartesianIndices(d)
    l = LinearIndices(d)
    Δ = 0.0
    for I = c
        δ = xf[l[I]] - f[I]
        Δ = max(abs(δ), abs(Δ))
        f[I] += ω*δ
    end
    set_Δ!(s, Δ)
    increment!(s)
    verbose && print_convergence_info(s)
    return f
end