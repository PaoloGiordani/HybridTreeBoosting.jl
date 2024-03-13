#
# Collects strucures (other than those in param.jl) 
#
# HTBtree           struct
# HTBoostTrees     struct
# HTBoostTrees     function
# Info_xi             struct
#   
# PG note: SVectors would make no difference speed-wise, since not building and storing this information barely changes timing (except possibly at VERY low n): not repeated enough times to make a difference


"""
    struct HTBtree{T<:AbstractFloat,I<:Int}
Collects information about a single HTB tree of depth d

# Fields
- `i`:             vector, d indices of splitting features.
- `μ`:             vector, d values of μ at each split
- `τ`:             vector, d values of τ at each split
- `m`:             vector, d values of m (NaN => m) at each split. NaN if no missing in the feature.
- `β`:             vector of dimension p (number of features), leaf values
- `fi2`:           vector, d values of feature importance squared: increase in R2 at each split.
- `σᵧ`:            std of gammafit for projection pursuit
"""
struct HTBtree{T<:AbstractFloat,I<:Int}
    i::AbstractVector{I}                           # selected feature
    μ::AbstractVector{T}
    τ::AbstractVector{T}
    m::AbstractVector{T}
    β::Vector{AbstractVector{T}}  # β from first and second phase
    fi2::AbstractVector{T}    # feature importance squared: increase in R2 at each split, a (depth,1) vector.
    σᵧ::T 
end



"""
    struct Info_xi{I<:Int}
Collects information about a feature that is used only in estimation (not needed for forecasting)

# Fields
- `i`:             I, position of feature in data.x
- `exclude`:       Bool, feature not to be considered for splitting
- `dichotomous`:   Bool, feature with two values
- `pp`:            Bool, true if it is a projection pursuit feature
- `n`:             I, full sample size (may be required for priors when subsampling) 
- `n_cat`:         I, number of categories. 1 if non-categorical or if dichotomous. 
- `force_sharp`:   Bool, true if the split is forced to be sharp
- `force_smooth`:  Bool, true if the split is forced to be smooth 
- `n_unique`:      Number of unique values
- `mixed_dc`:      Bool, true for a mixed discrete-continuous feature
- `kd`:            Float, approximate Kantorovic distance from a standard Gaussian (:logistic) or from y (:L2, ... ) 
- `m`:             Float, mean of feature prior to standardization 
- `s`:             Float, std of feature prior to standardization 
- `min_xi          Float, minimum(xi)   
- `max_xi          Float, mximum(xi)
"""
struct Info_xi{T<:AbstractFloat,I<:Int}
    i::I                                        
    exclude::Bool 
    dichotomous::Bool
    pp::Bool
    n::I            
    n_cat::I
    force_sharp::Bool           
    force_smooth::Bool           
    n_unique::I
    mixed_dc::Bool
    kd::T
    m::T
    s::T
    min_xi::T
    max_xi::T
end




"""
    struct HTBoostTrees{T<:AbstractFloat,I<:Int}
Collects information about the ensemble of HTB trees.

# Fields
- `param::HTBparam`:        values of param
- `gamma0`:                   initialization (prior to fitting any tree) of natural parameter, e.g. mean(data.y) for regression
- `trees::Vector{HTBtree}`: element i collects the info about the i-th tree
- `infeatures::Vector{Bool}`: element i is true if feature i is included in at least one tree
- `fi2::Vector`:              feature importance square
- `f::Vector`:                feature frequency
- `meanx`:                    vector of values m used in standardizing (x .- m')./s'
- `stdx`:                     vector of values m used in standardizing (x .- m')./s'
- `gammafit`:                 fitted values of natural parameter (fitted values of y for regression)
- `R2simul`
- `Info_x`                     p vector with information about each feature 
"""
mutable struct HTBoostTrees{T<:AbstractFloat,I<:Int}

    param::HTBparam
    gamma0::T
    trees::Vector{HTBtree{T,I}}
    infeatures::Vector{Bool}
    fi2::Vector{T}          # feature importance squared: Breiman et al. 1984, equation 10.42 in Hastie et al, "The Elements of Statistical Learning", second edition
    fr::Vector{I}           # feature inclusion count  
    fi::Vector{T}           # always in [0 1]    see updateHTBtrees!
    meanx::Array{T}
    stdx::Array{T}
    gammafit::Vector{T}
    R2simul::Vector{T}
    Info_x::Vector{Info_xi}

end



function HTBoostTrees(param,gamma0,offset,n,p,meanx,stdx,Info_x)

    T,I = param.T,param.I
    trees, infeatures,R2simul,fi,fr = HTBtree{typeof(gamma0),typeof(p)}[], fill(false,p), T[],zeros(T,p),zeros(I,p)
    gammafit   = offset + fill(gamma0,n)
    HTBtrees = HTBoostTrees(param,gamma0,trees,infeatures,fi,fr,fi,meanx,stdx,gammafit,R2simul,Info_x)

    return HTBtrees
end
