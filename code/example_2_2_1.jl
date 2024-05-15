"""
Example 2.2.1
"""

using Pkg; 
Pkg.activate("."); 
using LinearAlgebra
using Plots

function ρ(x)
    1. + 20. * (1. - tanh(20*(x-0.25))^2) + 30. * (1. - tanh(30*(x-0.5))^2) + 10. * (1. - tanh(10 * (x - 0.75))^2)
end

function P(ys, ρ::F) where F<:Function 
    len = length(ys)
    y_right = view(ys, 2:len)
    y_left = view(ys, 1:len-1)
    y_diff = y_right .- y_left
    ρ_sum = ρ.(y_right) .+ ρ.(y_left)
    ps = cumsum(ρ_sum .* y_diff ./ 2)
    return [0; ps] # ∫ᵃₐ = 0
end

function x_init(a, b, ngrid)
    return range(a, b, length=ngrid)
end

function get_ξs(ngrid)
    return range(0, 1, length=ngrid)
end

function step(ys, ρ::F) where F<:Function
    Ps = P(ys, ρ)
    ξs = get_ξs(length(ys))

    ξPb = ξs[2:end-1] .* Ps[end]
    ks_l = searchsortedlast.(Ref(Ps), ξPb)
    ks_r = ks_l .+ 1


    xs_inter = @. ys[ks_l] + 2 * (ξPb - Ps[ks_l]) / (ρ(ys[ks_l]) + ρ(ys[ks_r]))
    xs = [ys[1]; xs_inter; ys[end]]
    return xs
end


function find_equidist(xs_init, ρ::F, maxIter=10000, tol=1e-8) where F<:Function
    xs = copy(xs_init)
    qs = []
    status = :fail
    for _ in 1:maxIter 
        xs_new = step(xs, ρ)
        if norm(xs_new - xs) < 1e-8
            status = :success
            break
        end
        xs = xs_new
        qs = [qs; Quality(xs, ρ)]
    end
    if status == :fail
        @warn "Failed to find equidistribution points"
    end
    return (grid=xs, quality=qs, status=status)
end

function Quality(xs, ρ::F) where F<:Function
    ξs = get_ξs(length(xs))
    x_diff = @. xs[2:end] - xs[1:end-1]
    ξ_diff = @. ξs[2:end] - ξs[1:end-1]
    σₕ = sum(@. (x_diff) * (ρ(xs[2:end]) + ρ(xs[1:end-1])) /2) 
    Qs = @. (x_diff / ξ_diff) * (ρ(xs[2:end]) + ρ(xs[1:end-1])) / (2*σₕ)
    Qmax = maximum(abs.(Qs))
    return Qmax 
end

function plot_result(xs, ρ) where F<:Function
    p = plot()
    plot!(p, ρ, 0, 1, label="ρ(x)")
    xlabel!(p, "x")
    ylabel!(p, "ρ(x)")
    scatter!(p, xs, ρ.(xs), label="equidistant points")
    return p
end

# Derivation
ys = x_init(0, 1, 81)
res = find_equidist(ys, ρ)
Qmax = Quality(res.grid, ρ)

# Display result
p1 = plot_result(res.grid, ρ)
p2 = plot(res.quality, label="Quality", xlabel="Iteration", ylabel="Quality", legend=:bottomright)
ps = plot(p1, p2, layout=(1,2))
display(ps)
