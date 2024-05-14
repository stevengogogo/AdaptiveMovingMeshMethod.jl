"""
Example 2.2.1
"""

using Pkg; 
Pkg.activate("."); 

function ρ(x)
    return 1. + 20. * (1. - tanh(20*(x-0.25)^2)) 
              + 30. * (1. - tanh(30*(x-0.5)^2)) 
              + 10. * (1. - tanh(10 * (x - 0.75)^2))
end

function P(ys, ρ::F) where F<:Function 
    len = length(ys)
    y_right = view(ys, 2:len)
    y_left = view(ys, 1:len-1)
    y_diff = y_right .- y_left
    ρ_sum = ρ(y_right) .+ ρ(y_left)
    ps = cumsum(ρ_sum .* y_diff ./ 2)
    return ps
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

    ξPb = ξs .* Ps[end]
    ks = searchsortedfirst.(Ref(Ps), ξPb)
end