"""
Example 2.3.1
"""

using Pkg;
Pkg.activate(".");
using LinearAlgebra
using Plots
using DifferentialEquations

function ρ(x)
    1.0 + 20.0 * (1.0 - tanh(20 * (x - 0.25))^2) + 30.0 * (1.0 - tanh(30 * (x - 0.5))^2) + 10.0 * (1.0 - tanh(10 * (x - 0.75))^2)
end

function mmpde5_modified!(dx, x, p, t)
    τ, Δξ = p 

    # Bounday mesh in physical domain
    dx[1] = 0.0 
    dx[end] = 0.0

    # Interior mesh
    for i in 2:length(x)-1
        coeff = (ρ(i)*τ*Δξ^2)^-1
        x_diff_r = x[i+1] - x[i]
        x_diff_l = x[i] - x[i-1]
        ρ_sum_r = ρ(x[i+1]) + ρ(x[i])
        ρ_sum_l = ρ(x[i]) + ρ(x[i-1])
        dx[i] = coeff * (0.5* ρ_sum_r * x_diff_r - 0.5 * ρ_sum_l * x_diff_l)
    end
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

function plot_result(xs, ρ) where {F<:Function}
    p = plot()
    plot!(p, ρ, 0, 1, label="ρ(x)", legend=:topright, xlabel="x", ylabel="ρ(x)")
    scatter!(p, xs, ρ.(xs), label="equi mesh")
    return p
end


# Derive equidistant points
ngrid = 81
p = (τ=1., Δξ=1. / (ngrid -1))
tspan = (0., 1.)
x_init = collect(range(0, 1, length=ngrid))
prob = ODEProblem(mmpde5_modified!, collect(x_init), tspan, p)
sol = solve(prob, ImplicitEuler())

# Plot
plt1 = plot_result(sol(tspan[end]), ρ)
plt2 = plot(Quality.(sol.(0:0.01:1), Ref(ρ)), xlabel="Time", ylabel="Quality", label="Quality", legend=:topright)
plts = plot(plt1, plt2, layout=(1, 2))
display(plts)
