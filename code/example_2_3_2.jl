"""
Example 2.3.2
"""

using Pkg;
Pkg.activate(".");
using LinearAlgebra
using Plots
using DifferentialEquations
using DiffEqCallbacks
using Interpolations 

function plot_result(xs, ρ) where {F<:Function}
    p = plot()
    plot!(p, ρ, 0, 1, label="ρ(x)", legend=:topright, xlabel="x", ylabel="ρ(x)")
    scatter!(p, xs, ρ.(xs), label="equi mesh")
    return p
end

function ρ(x)
    1.0 + 20.0 * (1.0 - tanh(20 * (x - 0.25))^2) + 30.0 * (1.0 - tanh(30 * (x - 0.5))^2) + 10.0 * (1.0 - tanh(10 * (x - 0.75))^2)
end

function mmpde5xi!(dξ, ξ, p, t)
    τ, interf = p 
    xs = interf.(ξ)
    # Boundary mesh in computational domain
    dξ[1] = 0.0
    dξ[end] = 0.
    # Interior mesh
    for i in 2:length(ξ)-1
        coeff = 2 * (τ * (xs[i+1] - xs[i-1]))^-1
        ρ_l_inv = 2 * (ρ(xs[i+1]) + ρ(xs[i]))^-1
        ρ_r_inv = 2 * (ρ(xs[i]) + ρ(xs[i-1]))^-1
        ρξ_l = ρ_l_inv * (ξ[i+1] - ξ[i]) / (xs[i+1] - xs[i])
        ρξ_r = ρ_r_inv * (ξ[i] - ξ[i-1]) / (xs[i] - xs[i-1])
        dξ[i] = coeff * (ρξ_l - ρξ_r)
    end
end

function interp!(int)
    ξ = int.u
    xs = int.p[2].(ξ)
    int.p[2] = linear_interpolation(ξ, xs, extrapolation_bc=Line())
    p[2] = int.p[2]
end
condition = function (u, t, integrator)
    true
end

event = ContinuousCallback(condition, interp!)
ngrid = 100
ξ0 = collect(range(0, stop=1, length=100))
x0 = collect(range(0, stop=1, length=100))
interf = linear_interpolation(ξ0, x0, extrapolation_bc=Line()) # ξ→x
p = [1.0, interf]
tspan = (0., 1.)
prob = ODEProblem{true}(mmpde5xi!, ξ0, tspan, p)
sol = solve(prob, ImplicitEuler(), callback = event)


# plot 
xs = p[2].(sol(tspan[end]))
plt1 = plot_result(xs, ρ)