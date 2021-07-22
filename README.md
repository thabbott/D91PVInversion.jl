# D92PVInversion.jl

Julia code (I hesitate to call it a package yet...) for non-linear potential vorticity inversion following [Davis and Emanuel (1991)](https://doi.org/10.1175/1520-0493(1991)119<1929:PVDOC>2.0.CO;2). It's still under development, *hasn't been tested thoroughly*, isn't well-documented, hasn't been used for any publications, and hasn't been benchmarked against any other codes that perform similar calculations. So feel free to use it, but don't trust it and don't assume it's more performant than other codes!

Most of the code is for setting up the linear inversion problems (A.3) and (A.4) from Davis and Emanuel (1991), which are solved using matrix-free iterative algorithms. All of the heavy lifting is done by [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) and [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl).
