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

function Quality(xs, ρ::F) where {F<:Function}
    ξs = range(0,1,length(xs))
    x_diff = @. xs[2:end] - xs[1:end-1]
    ξ_diff = @. ξs[2:end] - ξs[1:end-1]
    σₕ = sum(@. (x_diff) * (ρ(xs[2:end]) + ρ(xs[1:end-1])) / 2)
    Qs = @. (x_diff / ξ_diff) * (ρ(xs[2:end]) + ρ(xs[1:end-1])) / (2 * σₕ)
    Qmax = maximum(abs.(Qs))
    return Qmax
end

function mmpde5xi!(dξ, ξ, p, t)
    τ, xs = p 
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


function affect!(int)
    ξ = int.u
    xs = int.p[2]
    interf = linear_interpolation(ξ, xs)
    xs_new = interf.(range(0, stop=1, length=length(ξ))[2:end-1])
    xs_new = [0.; xs_new; 1.] # set boundary points at physical domain
    int.p[2] = xs_new
end

condition = function (u, t, integrator)
    true
end

event = DiscreteCallback(condition, affect!)
ngrid = 81
ξ0 = collect(range(0, stop=1, length=100))
x0 = collect(range(0, stop=1, length=100))
p = [1.0, x0]
tspan = (0., 1.)
prob = ODEProblem{true}(mmpde5xi!, ξ0, tspan, p)
sol = solve(prob, Tsit5(), callback = event)


# plot 
xs = p[2]
plt1 = plot_result(xs, ρ)
display(plt1)
@show Quality(xs, ρ)