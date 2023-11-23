#
# Function used in initializing gamma0
#
# Huber loss
# - stdw_robust()
# - huber_mean()
#
# Student loss
# - student_mean()
#
#


# Calibrate parameter of Huber function.
function stdw_robust(r,w)

    T = eltype(r)
    m = mean(r)
    sw = T(1.25)*mean(abs.(r .- m))

    return T(sw)

end



# Huber estimator of the mean by iterative weighted least squares, with fixed standard deviation.
# stde is a robust estimate of the std, e.g. stde=stdw_robust(y,weights), t = 1.34 (for 95% efficiency with Gaussian densities).
# w = ones(n) typically.
function huber_mean(y::AbstractVector{T},w::AbstractVector{T},stde;t=1.34,niter=2) where T<:AbstractFloat

    n  = length(y)
    c  = T(stde*t)
    sumw = sum(w)
    μ  = sum(y.*w)/sumw

    r  = Vector{T}(undef,n)  # residuals
    ψ  = Vector{T}(undef,n)   # Huber residuals
    ω  = Vector{T}(undef,n)   # Huber weights

    for iter in 1:niter
        @. r = (y - μ)
        @. ψ = abs(r)*(abs(r)<c) + c*(abs(r)≥c)  
        @. ω = ψ/abs(r)

        μ  = sum(@. y*ω*w)/(sum(@. ω*w)) # initialize mean
    end

    return μ

end



function student_mean(y::AbstractVector{T},w::AbstractVector{T}) where T<:AbstractFloat

    res = Newton_MAP(y,gH_student_with_mean,start_value_student_with_mean;w=w)
    μ   = res.minimizer[1]

    return μ

end
