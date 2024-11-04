#=

Robust measures of skewness and kurtosis, based on Pinsky and Klawansky 2023 and Pinsky 2024. 

Study for inclusion in categoricals (after checking length(unique(y)))

Reference to Pinsky 2024

@article{article,
author = {Pinsky, Eugene},
year = {2024},
month = {09},
pages = {1-3},
title = {Mean Absolute Deviation (About Mean) Metric for Kurtosis},
volume = {2},
journal = {Advance Research in Sciences (ARS)},
doi = {10.54026/ARS/1021}
}

=#

using Random,Statistics 

n = 1_000_000
p = 1

x = randn(n).^1
x = rand(n)

sk = mad_skewness(x) 
k  = mad_kurtosis(x)
println("skewness: $sk, kurtosis")

@time mad_skewness(x)
@time mad_kurtosis(x)

# measure of skewenss based on MAD. Equivalent to Groeneveld and Meeden
function mad_skewness(x::Vector{T}) where T<:Real 
    m = mean(x)
    med = median(x)
    mad = mean(abs.(x .- m))  # 
    if mad>0
        return (m - med)/mad
    else
        return T(0)
    end          
end

# measure of kurtosis based on MAD, proposed by Pinsky 2024 (eq. 8)
# if the measure is not defined (x has 1 unique value), returns 0.6, the kurtosis of a Gaussian with this measure
function mad_kurtosis(x::Vector{T}) where T<:Real

    if length(x)==1
        return T(0.6)    
    end

    μ  = mean(x)
    omega_L = x .< μ
    omega_R = x .>= μ
    μ_L = mean(x[omega_L])
    μ_R = mean(x[omega_R])
    H_L = mean(abs.(x[omega_L] .- μ_L))
    H_R = mean(abs.(x[omega_R] .- μ_R))
    H   = mean(abs.(x .- μ))
    if H>0
        xs  = sort(x)
        F   = sum(xs .<= μ)/length(x) 
        k  = (F*H_L + (1-F)*H_R)/H
        return k
    else 
        return T(0.6)
    end     

end

