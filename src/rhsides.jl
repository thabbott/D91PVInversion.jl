function set_qg_bψ!(bψ, q, d::Domain)
    c = CartesianIndices(d)
    l = LinearIndices(d)
    for I = c
        bψ[l[I]] = q[I]
    end
    return bψ
end