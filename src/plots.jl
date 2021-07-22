function plot_state(ψ, ϕ, q, d::Domain; i = 1, j = 1)

    plt.rc("font", size = 8)
    fig, axes = plt.subplots(
        figsize = (6.5, 8), nrows = 3, ncols = 2,
        sharex = "col", sharey = true, constrained_layout = true
    )
    nx, ny, nz = size(d)
    x = d.xc[1:nx]
    y = d.yc[1:ny]
    z = d.zc[1:nz]
    c = axes[1,1].contour(x, z, ψ[1:nx,j,1:nz]', colors = "black")
    plt.clabel(c)
    c = axes[1,2].contour(y, z, ψ[i,1:ny,1:nz]', colors = "black")
    plt.clabel(c)
    c = axes[2,1].contour(x, z, ϕ[1:nx,j,1:nz]', colors = "black")
    plt.clabel(c)
    c = axes[2,2].contour(y, z, ϕ[i,1:ny,1:nz]', colors = "black")
    plt.clabel(c)
    c = axes[3,1].contour(x, z, q[1:nx,j,1:nz]', colors = "black")
    plt.clabel(c)
    c = axes[3,2].contour(y, z, q[i,1:ny,1:nz]', colors = "black")
    plt.clabel(c)
    axes[1,1].set_title("ψ")
    axes[1,2].set_title("ψ")
    axes[2,1].set_title("ϕ")
    axes[2,2].set_title("ϕ")
    axes[3,1].set_title("q")
    axes[3,2].set_title("q")
    axes[1,1].set_ylabel("π")
    axes[2,1].set_ylabel("π")
    axes[3,1].set_ylabel("π")
    axes[3,1].set_xlabel("x")
    axes[3,2].set_xlabel("x")
    axes[1,1].invert_yaxis()
    return fig, axes

end