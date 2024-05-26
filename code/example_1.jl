using Pkg 
Pkg.activate(".")

using Plots 
using DifferentialEquations
using ModelingToolkit
using LinearAlgebra 
using Symbolics


function f_bound(x)
    return zeros(size(x))
end

function f_init(x)
    return sin(2*π*x) + 0.5 * sin(π*x)
end

function get_rho(v_xs::AbstractVector, v_us::AbstractVector)
    v_uxx = get_uxx(v_xs, v_us)
    α = get_alpha(v_xs, v_uxx)
    #v_rho = (1. .+ alpha^-1 .* v_uxx.^2).^(1/3.)
    v_rho = (1. .+ (v_uxx.^2) ./ α).^(1/3)
    v_rho = _averaging(v_rho)
    v_rho = _averaging(v_rho)
    v_rho = _averaging(v_rho)
    v_rho = _averaging(v_rho)
    v_rho = _averaging(v_rho)
    return v_rho 
end


function get_alpha(v_xs::AbstractVector, v_uxx::AbstractVector)
    x_diff = _diff(v_xs, -1, 0)
    u_xx_j = v_uxx[2:end]
    u_xx_jm1 = v_uxx[1:end-1]

    _alpha = sum(0.5 .* x_diff .* (abs.(u_xx_j).^(2/3) .+ abs.(u_xx_jm1).^(2/3)))^3.
    alpha = max(1., _alpha)
    return alpha 
end

function get_init(ngrid::Integer)
    v_xs0 = range(0, 1, length=ngrid)
    v_us0 = f_init.(v_xs0)
    return [v_xs0; v_us0]
end


function get_dxdt(v_xs, v_us, tau)
    N = length(v_xs)
    dxi = 1. / (N-1)
    v_ux = get_ux(v_xs, v_us)
    v_rho = get_rho(v_xs, v_us)
    v_dx_forward = _diff(v_xs, 0, +1)
    v_dx_backward = _diff(v_xs,-1,0)

    _v_dx = 0.5 .* (v_rho[3:end] .+ v_rho[2:end-1]) .* v_dx_forward[2:end] - 0.5 .* (v_rho[2:end-1] .+ v_rho[1:end-2]) .*v_dx_backward[1:end-1]
    coeff = 1. ./ (v_rho[2:end-1] .* tau .* dxi^2)
    _v_dx = coeff .* _v_dx
    return [0.; _v_dx; 0.]
end


function get_dudt(v_xs, v_us, v_dxdt, ϵ::AbstractFloat)
    dx_center = _diff(v_xs, -1, +1)
    dx_forward = _diff(v_xs, 0, +1)[2:end]
    dx_backward = _diff(v_xs, -1, 0)[1:end-1]

    du_forward = _diff(v_us, 0, +1)[2:end]
    du_backward = _diff(v_us, -1, 0)[1:end-1]
    du_center = _diff(v_us, -1, +1)
    du_center_sq = _diff(v_us .^2, -1, +1)

    dux_center = du_center ./ dx_center
    dux_forward = du_forward ./ dx_forward
    dux_backward = du_backward ./ dx_backward

    _v_dudt = dux_center .* v_dxdt[2:end-1] .+ 2 * ϵ .* (dux_forward .- dux_backward) ./ dx_center .- 0.5 .* du_center_sq ./ dx_center

    v_duxt = [0; _v_dudt; 0]
    return v_duxt
end

function get_ux(v_xs, v_us)
    dv_us = _diff(v_us, -1, +1)
    dv_xs = _diff(v_xs, -1, +1)
    v_ux = dv_us ./ dv_xs
    du0 = (v_us[2] - v_us[1]) / (v_xs[2] - v_xs[1])
    duN = (v_us[end] - v_us[end-1]) / (v_xs[end] - v_xs[end-1])
    v_ux = [du0; v_ux; duN]
    return v_ux
end

function get_uxx(v_xs, v_us)
    dv_us_right = _diff(v_us, 0, +1)[2:end]
    dv_xs_right = _diff(v_xs, 0, +1)[2:end]
    v_ux_right = dv_us_right ./ dv_xs_right

    dv_us_left = _diff(v_us, -1, 0)[1:end-1]
    dv_xs_left = _diff(v_xs, -1, 0)[1:end-1]
    v_ux_left = dv_us_left ./ dv_xs_left

    v_dx = _diff(v_xs, -1, +1)
    _v_uxx = (v_ux_right .- v_ux_left) ./ (v_dx ./ 2.)

    x = v_xs 
    u = v_us 
    N = length(v_xs)

    uxx_0 = 2. * ( (x[2] - x[1])*(u[3] - u[1]) - (x[3] - x[1])*(u[2] - u[1]) ) / ((x[3]-x[1])*(x[2]-x[1])*(x[3]-x[2]))
    uxx_N = 2. * ((x[end-1]-x[end])*(u[end-2] - u[end]) - (x[end-2]-x[end])*(u[end-1]-u[end])) / ((x[end-2]-x[end])*(x[end-1]-x[end])*(x[end-2]-x[end-1]))
    v_uxx = [uxx_0; _v_uxx; uxx_N]
    return v_uxx 
end

function _diff(arr, left, right)
    arr_p = arr[-left+right+1:end]
    arr_m = arr[1:end-(-left+right)]
    darr = arr_p .- arr_m 
    return darr
end

function _averaging(arr)
    v_m1 = @view arr[1:end-2]
    v_m = @view arr[2:end-1]
    v_p1 = @view arr[3:end]

    v_arr_smooth = 0.25 .* v_m1 .+ 0.5 .* v_m .+ 0.25 .* v_p1 
    v1 = 0.5 * arr[1] + 0.5 * arr[2]
    vN = 0.5 * arr[end] + 0.5 * arr[end-1]
    return [v1; v_arr_smooth; vN]
end


function f!(dy, y, p, t)
    τ, ϵ, ngrid = p
    v_xs = y[1:ngrid]
    v_us = y[ngrid+1:end]
    dxdt = dy[1:ngrid]
    dudt = dy[ngrid+1:end]

    dxdt = get_dxdt(v_xs, v_us, τ)
    dudt = get_dudt(v_xs, v_us, dxdt, ϵ)
    dy[1:end] = [dxdt; dudt]
end

function main(ngrid)
    τ = 1e-2
    ϵ = 1e-4
    tspan = (0., 1.0)
    p = (τ, ϵ, ngrid)
    y0 = get_init(ngrid)

    dy0 = copy(y0)
    #jac_sparsity = Symbolics.jacobian_sparsity((dy, y) -> f!(dy, y, p, 0.0), dy0, y0)
    #@show jac_sparsity

    f = ODEFunction(f!)#; jac_prototype = float.(jac_sparsity))
    prob = ODEProblem(f, y0, tspan, p)
    sol = solve(prob, Kvaerno3(), reltol=1e-8, abstol=1e-8)
    return sol
end

function plot_sol(sol, ngrid)
    m = plot()
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        _sol = sol(t)
        x = _sol[1:ngrid]
        m = plot!(m, x, _sol[ngrid+1:end], label="t=$t", marker=(:circle,5))
    end
    xlabel!(m, "x")
    ylabel!(m, "u")
    #savefig(m, "img/MovFD_Burgers.png")
    display(m)
end

function plot_grids(sol, ngrid)
    t = 0.:0.01:1.0 
    sols = sol(t)
    m = plot(legend=false)
    for i in 1:ngrid 
        plot!(m, getindex.(sols.u, i),t, color=:black)
    end
    xlabel!(m, "x")
    ylabel!(m, "t")
    #savefig(m, "img/MovFD_Burgers_grids.png")
    display(m)
end




ngrid=41
@time sol = main(ngrid);
plot_sol(sol, ngrid);
plot_grids(sol, ngrid);