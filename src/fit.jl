
#
#  Auxiliary functions called in the boosting algorithm
#
#  The following 4 functions are the most computationally expensive:
#
#  updateG!
#  update_GGh_Gr!
#  computeGβ
#  sigmoidf
#
#  logpdft
#  lnpτ
#  lnpμ
#  pb_compute
#  robustify_GGh!
#  fitβ              Newton optimization of log posterior
#  update_GGh_Gr       called in    "
#   diagonal_product_01
#  logpdfpriors           "
#  lnpM               sparsity prior
#  lnpMTE             prior (minus) penalization for mean target encoding
#  Δβ                 called in Newton optimization
#  Gfitβ
#  Gfitβ2
#  Gfitβm
#  updateinfeatures
#  add_depth          add one layer to the tree, for given i, using a rough grid search (optional: full optimization)
#  loopfeatures       add_depth looped over features, to select best-fitting feature
#  best_μτ_excluding_nan   in preliminary variable selection
#  best_m_given_μτ         in prelminary vs; estimate for missing values at a given μ,τ
#  refineOptim        having chosen a feature (typically via a rough grid), refines the optimization
#  refineOptim_μτ_excluding_nan
#  refineOptim_m_given_μτ
#  optimize_μτ     called in refineOptim
#  fit_one_tree
#  updateSMARTtrees!
#  SMARTtreebuild
#  median_weighted_tau  a weighted median value of τ for each feature
#  (tau_info)        provides info on posterior variance of estimated tau.
# speedup_preliminaryvs() 
# calibrate_n_preliminaryvs!()





function updateG!(G::AbstractMatrix{T},G0::AbstractArray{T},g::AbstractVector{T}) where T<:AbstractFloat

    p = size(G0,2)

    @views for i in 1:p
        @. G[:,i]   =  G0[:,i]*g
        @. G[:,i+p] =  G0[:,i]*(1 - g)
    end

end



# returns diag(G'G) assuming that G is a matrix of 0 or 1 and G'G is diagonal.
function diagonal_product_01(G::AbstractMatrix{T}) where T<:AbstractFloat

    p = size(G,2)
    GGh = Vector{T}(undef,p)

    @views for i in 1:p
       GGh[i] = sum(G[:,i])
    end

    return GGh

end


# returns diag((G*h)'G) assuming that G is a matrix of 0 or 1 and G'G is diagonal.
function diagonal_product_01(G::AbstractMatrix{T},h::AbstractVector{T}) where T<:AbstractFloat

    p = size(G,2)
    GGh = Vector{T}(undef,p)

    @views for i in 1:p
       GGh[i] = sum( h[G[:,i] .> 0])
    end

    return GGh

end


# Efficient computations of G'(G.*h). (and of G'G)
# If h is constant, GGh does not change and only need computing once.
# If h is not constant, computing GGh = G'(G.*h) can be speeded up a lot by i) exploiting symmetry/transpose: GGh=Gh'Gh,
# where Gh = G.*sqrt(t), ii) pre-allocating Gh. GGh is an input in case it does not get modified.
# Note to PG: For low tree depth, it would be significantly faster to input sqrt.(h) rather than doing the computation internally,
# but I will stick to the safer version for now.
function update_GGh_Gr(GGh,G,Gh,r,nh,h,iter,T,priortype)

    if nh==1

        if iter==1
            if priortype == :sharp
                GGh = diagonal_product_01(G)      # returns a vector
            else
                GGh = G'G
            end
        end

    else

        if priortype == :sharp
            GGh = diagonal_product_01(G,h)           # returns a vector
        else
            sqrth = sqrt.(h)
            @views for i in 1:size(G)[2]
                @. Gh[:,i] = G[:,i]*sqrth
            end
            GGh = Gh'Gh
        end

    end

    Gr = G'r

    return GGh,Gr

end



function computeGβ(G,β)
    return G*β
end



# If τ==Inf, becomes a sharp threshold.
function sigmoidf(x::AbstractVector{T},μ::T,τ::T,sigmoid::Symbol;dichotomous::Bool = false) where T<:AbstractFloat

    if dichotomous   # return 0 if x<=0 and x>1 otherwise. x is assumed de-meaned
        g = @. T(0) + T(1)*(x>0)
        return g
    end

    if τ==T(Inf)
        g = @. T(0)*(x<=μ) + T(1)*(x>μ)
        return g
    end

    if sigmoid==:sigmoidsqrt
         g = @. T(0.5) + T(0.5)*( T(0.5)*τ*(x-μ)/sqrt(( T(1.0) + ( T(0.5)*τ*(x-μ) )^2  )) )
    elseif sigmoid==:sigmoidlogistic
        g = @. T(1.0) - T(1.0)/(T(1.0) + (exp(τ * (x - μ))))
    end

    return g
end



# Paolo: currently not used (minimal speed gains and code needs updating in several places).
#=
function sigmoidf!(g::AbstractVector{T},x::AbstractVector{T}, μ::T, τ::T,sigmoid::Symbol;dichotomous::Bool = false) where T<:AbstractFloat

    if dichotomous   # return 0 if x<=0 and x>1 otherwise. x is assumed de-meaned
        @. g = T(0) + T(1)*(x>0)
    else
        if τ==T(Inf)
            @. g = T(0)*(x<=μ) + T(1)*(x>μ)
        else

            if sigmoid==:sigmoidsqrt
                @. g = T(0.5) + T(0.5)*( T(0.5)*τ*(x-μ)/sqrt(( T(1.0) + ( T(0.5)*τ*(x-μ) )^2  )) )
            elseif sigmoid==:sigmoidlogistic
                @. g =  T(1.0) - T(1.0)/(T(1.0) + (exp(τ * (x - μ))))
            end

        end
    end
end
=#



function logpdft(x::T,m::T,s::T,v::T) where T<:AbstractFloat

    z       = ( x - m)/s
    logpdfz = T(-0.5723649429247001)+SpecialFunctions.loggamma((v+T(1))/T(2))-SpecialFunctions.loggamma(v/T(2))-T(0.5)*log(v)-T(0.5)*(T(1)+v)*log(T(1)+(z^2)/v)

    return logpdfz - log(s)

end



# NOTE: for the purpose of evaluating the density, μ is truncated at μmax (e.g. 5). (Some extremely non-Gaussian features may have very large values even when standardized)
function lnpμ(μ0::Union{T,Vector{T}},varmu::T,dofmu::T;μmax =T(5)) where T<:AbstractFloat

    μ = @. T( (abs(μ0)<=μmax)*μ0 +  (abs(μ0)>μmax)*μmax )
    #s  = sqrt(varmu*(dofmu-T(2))/dofmu)  # to intrepret varmu as an actual variance.
    s  = sqrt(varmu)                 # to intrepret varmu as a dispersion
    lnp = sum(logpdft.(μ,T(0),s,dofmu))

    return T(lnp)

end



# Pb = pb*I(p), where Pb is the precision matrix of β. Pb is the prediction of the prior on β
function pb_compute(r::AbstractVector{T},param::SMARTparam,GGh,n) where T<:AbstractFloat

    ndims(GGh)==1 ? diagGGh=sum(GGh) : diagGGh=sum(diag(GGh))
    pb = param.theta*diagGGh/(n*param.varGb)

    return T(max(pb,1e-10))

end



# Computes log t-density of the prior on log(τ). The mean of this density is a function of the Kantorovic distance of feature i.
# τ is trucated at τmax=100 to allow for Inf (sharp threshold) to have finite density. d is tree dimension as in size(G,2)= 2^d (e.g. 1 for d=depth1+1)
# function lnpτ(τ0::Union{T,Vector{T}},param::SMARTparam,info_i,d;τmax=T(100) )::T where T<:AbstractFloat     # NOTE: modifications required for τ0 a vector.
function lnpτ(τ0::T,param::SMARTparam,info_i,d;τmax=T(100) )::T where T<:AbstractFloat

    if param.priortype==:sharp || info_i.force_sharp==true
        return T(0)
    end

    τ = @. T( (abs(τ0)<=τmax)*τ0 +  (τ0>τmax)*τmax )
    stdlntau   = T(sqrt(param.varlntau))                 # to intrepret varlntau as dispersion
    #depth = T( param.depth1+maximum([0.0, 0.5*(param.depth-param.depth1) ]) )  # matters only if prior is modified for d
    #stdlntau  = sqrt( (param.varlntau)*(param.doflntau-T(2))/param.doflntau )  # to intrepret varlntau as an actual variance.

    # Adjust prior for Kantorovic distance. This adjustment is smaller when Kantorovic distance between y and xi is not as informative, as for :logistic
    if param.loss==:L2 || param.loss==:Huber || param.loss==:quantile || param.loss==:t
        α=T(1)
    elseif param.loss==:logistic
        α=T(0.5)
    else
        @error "loss function misspelled or not implemented"
    end

    β        = T(0.3)
    m        =  param.meanlntau + α*param.multiplier_stdtau*( -β + info_i.kd  )

    lnp = sum(logpdft.(log.(τ),T(m),T(stdlntau),T(param.doflntau) ))

    # If mixture of two student-t
    # k2,prob2 = T(3),T(0.2)
    # lnp2 = sum(logpdft.(log.(τ),T(m*k2),T(stdlntau),param.doflntau))
    # lnpmax  = maximum([lnp,lnp2])        # numerically more robust alternative to  lnp = log( (1-prob2)*exp.(lnp) + prob2*exp.(lnp) ), should work well even with Float32
    # lnp     = lnpmax + log( (1-prob2)*exp(lnp-lnpmax) + prob2*exp(lnp2-lnpmax) )

    return T(lnp)

end


# Increase the probability that GGh is invertible (alternative I did not exlore: use a pseudo-inverse, see Δβ)
function robustify_GGh!(GGh::Matrix,p,T)
    maxdiagGGh = maximum(diag(GGh))
    [GGh[i,i]  = maximum([GGh[i,i],maxdiagGGh*T(0.00001)])  for i in 1:p]  # nearly ensures invertible G'G, effectively tightening the prior for empty and near-empty leafs
end



function robustify_GGh!(GGh::Vector,p,T)
    maxdiagGGh = maximum(GGh)
    [GGh[i]  = maximum([GGh[i],maxdiagGGh*T(0.00001)])  for i in 1:p]  # nearly ensures invertible G'G, effectively tightening the prior for empty and near-empty leafs
end



# fitβ: Newton optimization of log posterior for SMARTboost with smooth threshold.
function fitβ(y,w,gammafit_ensemble,r0::AbstractVector{T},h0::AbstractVector{T},G::AbstractArray{T},Gh,param::SMARTparam,infeatures,fi,info_i::Info_xi,
    μ::Union{T,Vector{T}},τ::Union{T,Vector{T}},m::T,llik0::T;finalβ="false")::Tuple{T,Vector{T},Vector{T}}  where T<:AbstractFloat

    r,h = copy(r0),copy(h0)
    n,p = size(G)
    nh  = length(h)
    d   = Int(round(log(p)/log(2)))  # NB: not the actual depth if depth>depth1, but what is relevant for priors
    GGh  = Matrix{T}(undef,p,p)

    if finalβ=="true"
        tol,maxsteps = param.newton_tol_final,param.newton_max_steps_final
    elseif finalβ=="refineOptim"   # maxiter as final, tol as vs, at all depth
        tol,maxsteps = param.newton_tol,param.newton_max_steps_refineOptim
    else
        tol,maxsteps = param.newton_tol,param.newton_max_steps
    end

    maxsteps>=10 ? α=T(0.5) : α=1  # half-steps in Newton-Raphson increase the chance of convergence

    #loss0 = -(llik0 +  logpdfpriors(β0,μ,τ,m,d,p,T(1),param,info_i,infeatures,fi,T)) # facenda: requires β0 from previous level
    β0    = zeros(T,p)
    loss0 = T(Inf)

    loss,β,Gβ = loss0,β0,Vector{T}(undef,n)

    for iter in 1:maxsteps

        GGh,Gr = update_GGh_Gr(GGh,G,Gh,r,nh,h,iter,T,param.priortype)
        robustify_GGh!(GGh,p,T)
        pb  = pb_compute(r,param,GGh,n)

        β   = β0 + α*Δβ(GGh,Gr,d,pb*I(p),param,n,p,T)
        Gβ  = computeGβ(G,β)

        if param.loss==:L2 || maxsteps>1 || param.newton_gaussian_approx==false
            llik  = loglik(param.loss,param,y,gammafit_ensemble+Gβ,w)
        else    # Gaussian approximation  to the log-likelihood, often much faster to evaluate, e.g. for :logistic
            ll  = w.*( - T(0.5)*(((r .- Gβ.*h).^2)./h)/(param.multiply_pb))
            llik = sum(ll)/(param.loglikdivide*mean(w))
        end

        logpriors = logpdfpriors(β,μ,τ,m,d,p,pb,param,info_i,infeatures,fi,T)

        loss      = -( llik + logpriors)

        if iter==maxsteps || loss0-loss<tol

            if loss>loss0
                loss,Gβ,β= loss0,G*β0,β0
            end

            break
        else
            β0,loss0 = copy(β),loss
            rh,param = gradient_hessian(y,w,gammafit_ensemble+Gβ,param,100_000)
            r,h = rh.r,rh.h
        end

    end

    return loss,Gβ,β

end



function logpdfpriors(β,μ,τ,m,d,p,pb,param,info_i,infeatures,fi,T)

    logpdfβ = -T(0.5)*( p*T(log(2π)) - p*log(pb) + pb*(β'β) )      # NB: assumes Pb is pb*I.

    if info_i.dichotomous
        logpdfμ,logpdfτ,logpdfm = 0,0,0
    elseif param.priortype==:sharp
        logpdfτ = 0
        logpdfμ = lnpμ(μ,param.varmu,param.dofmu)
        #logpdfm   = lnpμ(m,param.varmu,param.dofmu)     # prior for missing value same as for μ
        logpdfm = T(0)
    else
        logpdfμ = lnpμ(μ,param.varmu,param.dofmu)
        logpdfτ = T(lnpτ(τ,param,info_i,d))
        #logpdfm   = lnpμ(m,param.varmu,param.dofmu)     # prior for missing value same as for μ
        logpdfm = T(0)
    end

    logpdfM   = lnpM(param,info_i,infeatures,fi,T)   # sparsity prior
    logpdfMTE = lnpMTE(param,info_i,T)               # penalization for mean target encoding features

    return logpdfβ+logpdfμ+logpdfτ+logpdfm+logpdfM+logpdfMTE

end



# prior (minus) penalization for mean target encoding
function lnpMTE(param,info_i,T)

    n_cat = info_i.n_cat

    if n_cat <= 2
        return T(0)
    else
        ncat = n_cat - 2
        ncatn = ncat/info_i.n
        pe = 0.5*ncat + 1500*(ncatn.^3).*((ncatn .- 0.025).>0)
        return -T( param.mean_encoding_penalization*pe )
    end

end


# prior for feature selection
function lnpM(param,info_i,infeatures,fi,T)
    lnp_AIC = lnpAIC(param,info_i,infeatures,fi,T)            # output is a NEGATIVE number (a log density)
    return lnp_AIC
end



# Computes -penalization on selecting feature i on a new split. Can be used to encourage sparsity.
# The penalization is on one additional feature. 
# param.sparsity_penalization* ( -0.2*dummy + 1.5*(1-dummy) + 0.9*log(p)       )
# This penalization assumes that: i) dummies are uncorrelated, ii) continuous have corr=0.5, iii) polynomial of order 3 is a good approximation to a tree split.  
# In trees, the slope coefficient may be different from 0.9. It depends on the features cross-correlations, the 
# number of candidate splits, what type of tree (smooth or sharp), at what level it comes in (first, second ...).
# However, trying the values {0,1} seems a good choice param.sparsity_penalization, and the range [0-1.5] sensible for full cv.
function lnpAIC(param::SMARTparam,info_i::Info_xi,infeatures,fi,T)

    isnew=(infeatures[info_i.i]==false)  # feature not currently in model? (to be penalized)

    if isnew==false || param.sparsity_penalization==T(0)
        return T(0)
    end

    # don't penalize the first param.depth features ? 
    #(sum(infeatures)+isnew)<=param.depth ? return T(0) : nothing

    isdummy  = info_i.dichotomous
    penaltyM = param.sparsity_penalization*( -0.2*isdummy + 1.5*(1-isdummy) + 0.9*log(param.p0) ) 

    return T(-penaltyM)

end



# There are three steps taken to reduce the risk of SingularException errors: 1) In fitβ, if a leaf is near-empty, the diagonal of GGh is increased very slightly,
# as     [GGh[i,i] = maximum([GGh[i,i],maxdiagGGh*T(0.00001)])  for i in 1:p], effectively tightening the prior for empty and near-empty leafs
# try ... catch, and 2) increase the diagonal of Pb and 3) switch to Float64.
# To reduce the danger of near-singularity associated with empty leaves, bound elements on the diagonal of GGh, here at 0.001% of the largest element.
function Δβ(GGh,Gr,d,Pb,param,n,p,T)

    Δβ = zeros(T,p)

    # GGh is a vector (sharp splits)
    if size(GGh,2)==1
        return (Gr)./( GGh .+ param.multiply_pb*param.loglikdivide*Pb[1,1])
    end

    # GGh is a matrix (smooth or hybrid splits)
    try
        Δβ = (GGh + param.multiply_pb*param.loglikdivide*Pb )\(Gr)
    catch err

        if isa(err,SingularException) || isa(err,PosDefException)
            #@info "Near-singularity detected. ";

            # Option 1: pseudo-inverse (β smallest norm among all solutions), 15 times slower
            #A = pinv( GGh + param.multiply_pb*param.loglikdivide*Pb )
            #Δβ = T.(A*(Gr))

           # Option 2: stronger prior
           while isa(err,SingularException) || isa(err,PosDefException)
            try
                  err         = "no error"
                  Pb          = Pb*Float64(3)  # switch to Float64 if there are invertibility problems.
                  Δβ = T.((GGh + param.multiply_pb*param.loglikdivide*Pb )\Gr)
                catch err
              end
           end

       end
    end

    return Δβ
end



function Gfitβ(y,w,gammafit_ensemble,r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},param::SMARTparam,infeatures,fi,info_i::Info_xi,μlogτm,G::AbstractMatrix{T},Gh,llik0)::T where T<:AbstractFloat

    μ = μlogτm[1]
    τ = exp(μlogτm[2])
    m = μlogτm[3]

    gL  = sigmoidf(xi,μ,τ,param.sigmoid,dichotomous=info_i.dichotomous)
    updateG!(G,G0,gL)
    loss,gammafit,β = fitβ(y,w,gammafit_ensemble,r,h,G,Gh,param,infeatures,fi,info_i,μ,τ,m,llik0)

    return loss

end



# used in refineOptim
function Gfitβ2(y,w,gammafit_ensemble,r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},param::SMARTparam,infeatures,fi,info_i::Info_xi,μv,τ::T,m::T,G::AbstractMatrix{T},Gh,llik0)::T where T<:AbstractFloat

    μ = T(μv[1])
    τ = maximum((τ,T(0.2)))  # Anything lower than 0.2 is still essentially linear, with very flat log-likelihood

    gL  = sigmoidf(xi,μ,τ,param.sigmoid,dichotomous=info_i.dichotomous)
    updateG!(G,G0,gL)
    loss,gammafit,β = fitβ(y,w,gammafit_ensemble,r,h,G,Gh,param,infeatures,fi,info_i,μ,τ,m,llik0,finalβ="refineOptim")

    return loss
end



# keeps track of which feature have been selected at any node in any tree (0-1)
function updateinfeatures(infeatures,ifit)

    x = deepcopy(infeatures)

    for i in ifit
        x[i] = true
    end

    return x

end



# Selects best (μ,τ) for a given feature using a rough grid in τ and, as default, a rough grid on μ.
# If loss increases, break loop over tau (reduces computation costs by some 25%), which assumes monotonicity in τ. (Remove? Make it an option?)
function add_depth(t)

    loss,τ,μ,nan_present = best_μτ_excluding_nan(t)

    if nan_present==true
        loss,m = best_m_given_μτ(t,μ,τ)
    else
        m = t.param.T(0)  # the value of 0 (hence meanx[i] since x here is standardized) should be relevant only when subsampling, so the training set has no missing but the full set does    end
    end

    return [loss,τ,μ,m]

end



# Best value of xi at which to set the missing values of xi, computed on the same grid as μ.
function best_m_given_μτ(t,μ,τ)

    y,w,gammafit_ensemble,r,h,G0,xi0 = t.y,t.w,t.gammafit_ensemble,t.r,t.h,t.G0,t.xi
    param,info_i,fi,μgridi,infeatures = t.param,t.info_i,t.fi,t.μgridi,t.infeatures
    T = param.T

    if (param.priortype==:sharp) || (τ==Inf)  # with sharp splits, only need to evaluate left and right of μ
        μgridi = [μ-1,μ+1]
    end

    lossmatrix = fill(T(Inf),length(μgridi))

    n,p = size(G0)
    G   = Matrix{T}(undef,n,2*p)
    Gh  = similar(G)                # pre-allocate for fast computation of G'(G.*h)

    llik0  = T(Inf)
    xi     = copy(xi0)

    if info_i.n_unique<2
        loss,m = T(Inf),T(0)

    elseif info_i.dichotomous==true  # PG: I don't think this can happen (dichotomous with missing will be categorical....)

        xi[isnan.(xi0)] .= info_i.min_xi
        loss_min = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[T(0),T(0),T(info_i.min_xi)],G,Gh,llik0)
        xi[isnan.(xi0)] .= info_i.max_xi
        loss_max = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[T(0),T(0),T(info_i.max_xi)],G,Gh,llik0)

        if loss_min<loss_max
            m,loss = info_i.min_xi,loss_min
        else
            m,loss = info_i.max_xi,loss_max
        end

    else

        for (j,m) in enumerate(μgridi)
            xi[isnan.(xi0)] .= m
            lossmatrix[j] = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[μ,log(τ),m],G,Gh,llik0)
        end

        minindex = argmin(lossmatrix)
        loss     = lossmatrix[minindex]
        m        = μgridi[minindex[1]]

    end

    return  loss,m

end




# Computes the loss for a given (μ,τ) and returns the best (μ,τ) for a given feature, having excluded all missing values.
function best_μτ_excluding_nan(t)

    miss_a = isnan.(t.xi)          # exclude missing values
    sum(miss_a)>0 ? nan_present=true : nan_present=false

    if nan_present
        keep_a = miss_a .== false
        y,w,gammafit_ensemble,r,G0,xi = t.y[keep_a],t.w[keep_a],t.gammafit_ensemble[keep_a],t.r[keep_a],t.G0[keep_a,:],t.xi[keep_a]
        length(t.h)==1 ? h = t.h : h = t.h[keep_a]
    else
        y,w,gammafit_ensemble,r,h,G0,xi = t.y,t.w,t.gammafit_ensemble,t.r,t.h,t.G0,t.xi
    end

    param,info_i,fi,τgrid,μgridi,infeatures = t.param,t.info_i,t.fi,t.τgrid,t.μgridi,t.infeatures

    T = param.T
    lossmatrix = fill(T(Inf64),length(τgrid),length(μgridi))

    n,p = size(G0)
    G   = Matrix{T}(undef,n,2*p)
    Gh  = similar(G)                # pre-allocate for fast computation of G'(G.*h)

    llik0  = loglik(param.loss,param,y,gammafit_ensemble,w)    # actual log-lik, not 2nd order approx

    if info_i.n_unique<2
        loss,μ,τ = T(Inf),T(0),T(1)
    elseif info_i.dichotomous==true   # no optimization needed
        loss = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[T(0),T(0),T(0)],G,Gh,llik0)
        τ,μ  = T(Inf),T(0)
    else

        if param.priortype==:sharp || info_i.force_sharp==true
            τgrid=[T(Inf)]
        else
            τgrid=τgrid
        end

        for (indexμ,μ) in enumerate(μgridi)

            for (indexτ,τ) in enumerate(τgrid)
                lossmatrix[indexτ,indexμ] = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[μ,log(τ),T(0)],G,Gh,llik0)
                if indexτ>1 && (lossmatrix[indexτ,indexμ])>(lossmatrix[indexτ-1,indexμ]); break; end  #  if loss increases, break loop over tau (reduces computation costs by some 25%)
            end

        end

        minindex = argmin(lossmatrix)  # returns a Cartesian index
        loss     = lossmatrix[minindex]
        τ        = τgrid[minindex[1]]
        μ        = μgridi[minindex[2]]

    end

    return loss,τ,μ,nan_present

end


# looping using @distributed or (since @distributed requires SharedArray, which can crash on Windows) Distributed.@spawn.
# pmap is as slow as map in this settings, which I don't understand, since it works as expected in refineOptim.
function loopfeatures(y,w,gammafit_ensemble,r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},x::AbstractMatrix{T},ifit,infeatures,fi,μgrid,Info_x,τgrid::AbstractVector{T},param::SMARTparam,ntree)::AbstractArray{T} where T<:AbstractFloat

    n,p   = size(x)
    ps    = Vector(1:p)                    # default: included all features

    # If sparsevs, loop only through the features in best_features, unless it's a scheduled update (then loop through all) 
#    if param.sparsevs==:On && (ntree in fibonacci(20,param.lambda,param.frequency_update) || isempty(param.best_features) )

    if param.sparsevs==:On && ( ntree in fibonacci(20,param.lambda,param.frequency_update) || isempty(param.best_features) )
        update_best_features = true
    else 
        update_best_features = false
    end     

    if update_best_features==false && param.sparsevs==:On && param.subsampleshare_columns==1
  
        ps = Vector{Int64}(undef,0)
  
        for i in 1:p
            if Info_x[i].exclude==false && Info_x[i].n_unique>1
                i in param.best_features ? push!(ps,i) : nothing
            end
        end
  
    end

    # @distributed for requires SharedArrays, which can crash on Windows. Distributed@spawn maybe less efficient
    # than @distributed in allocating jobs (I am not sure), but does not require SharedArrays. 
    outputarray = Matrix{T}(undef,p,4)  # [loss, τ, μ, m ]  p, not p_new (if p>p_new, some will be Inf)

    try
        outputarray = SharedArray{T}(p,4)
    catch 
        outputarray = Matrix{T}(undef,p,4)
    end     

    if update_best_features || ntree==0
        allow_preliminaryvs = false
    else 
        allow_preliminaryvs = true
    end

    if typeof(outputarray)<:SharedArray
        outputarray = loopfeatures_distributed(outputarray,n,p,ps,y,w,gammafit_ensemble,r,h,G0,x,ifit,infeatures,fi,μgrid,Info_x,τgrid,param,ntree,allow_preliminaryvs)
    else     
        outputarray = loopfeatures_spawn(outputarray,n,p,ps,y,w,gammafit_ensemble,r,h,G0,x,ifit,infeatures,fi,μgrid,Info_x,τgrid,param,ntree,allow_preliminaryvs)
    end

    return outputarray

end



# Using (in place of @sync @distributed for) the structure
# @sync for
#    @ async begin
#       future = Distributed.@sapwn myfunction()
#       outputarray[i,:] = fetch(future)
#
# outputarray is Matrix{T}(undef,p,4) #  [loss, τ, μ, m ]  p, not p_new
function loopfeatures_spawn(outputarray,n,p,ps,y,w,gammafit_ensemble,r,h,G0,x,ifit,infeatures,fi,μgrid,Info_x,τgrid,param,ntree,allow_preliminaryvs)

    T = param.T
    p_new = length(ps)
    outputarray[:,1] = fill(T(Inf),p)   #                   p, not p_new (if p>p_new, some will be Inf)
    psecondvs = max(2,Int(ceil(p.*param.fsecondvs)))

    speed_up_pvs = speedup_preliminarvys(param,n,ps,psecondvs)
    
    # optional first-stage, preliminary variable selection on a row sub-sample. Requires n>=50k and speed-gains>2 and p>3*ncores
    if allow_preliminaryvs && param.preliminaryvs !== :Off && speed_up_pvs>2 && n>=50_000 && p_new>=3*nprocs()

        ssi         = randperm(Random.MersenneTwister(param.seed_subsampling+2*ntree),n)[1:param.n_preliminaryvs]  # subs-sample, no reimmission
        ys=y[ssi];ws=w[ssi];gammafit_ensembles=gammafit_ensemble[ssi];rs=r[ssi];G0s=G0[ssi,:];
        xs = SharedMatrixErrorRobust(x[ssi,:],param)      # NB: could be smarter, taking only columns I need to avoid large copies.
        length(h)==1 ? hs=h : hs=h[ssi]

        @sync for i in ps
            @async begin # use @async to create a task that will be scheduled to run on any available worker process
                if Info_x[i].exclude==false  && Info_x[i].n_unique>1
                    t   = (y=ys,w=ws,gammafit_ensemble=gammafit_ensembles,r=rs,h=hs,G0=G0s,xi=xs[:,i],infeatures=infeatures,fi=fi,info_i=Info_x[i],μgridi=μgrid[i],τgrid=τgrid,param=param)
                    future = Distributed.@spawn add_depth(t)
                    outputarray[i,:] = fetch(future)
                end
            end
        end

        ps = sortperm(outputarray[:,1])[1:psecondvs]    # redefine ps
        outputarray[:,1] = fill(T(Inf),p)

    elseif param.subsampleshare_columns < 1
        psmall = convert(Int64,round(p*param.subsampleshare_columns))
        ps     = ps[randperm(Random.MersenneTwister(param.seed_subsampling+2*ntree),p)[1:psmall]]                  # subs-sample, no reimmission
    end

    # (second stage) variable selection.
    @sync for i in ps        
        @async begin # use @async to create a task that will be scheduled to run on any available worker process
            if Info_x[i].exclude==false  && Info_x[i].n_unique>1
                t   = (y=y,w=w,gammafit_ensemble=gammafit_ensemble,r=r,h=h,G0=G0,xi=x[:,i],infeatures=infeatures,fi=fi,info_i=Info_x[i],μgridi=μgrid[i],τgrid=τgrid,param=param)
                future = Distributed.@spawn add_depth(t)
                outputarray[i,:] = fetch(future)
            end
        end
    end

    calibrate_n_preliminaryvs!(param,outputarray,psecondvs,n,y,gammafit_ensemble,w)   # done only in first iteration, if param.preliminaryvs=:On

    return outputarray
end     



# outputarray is SharedArray{T}(p,4)
function loopfeatures_distributed(outputarray,n,p,ps,y,w,gammafit_ensemble,r,h,G0,x,ifit,infeatures,fi,μgrid,Info_x,τgrid,param,ntree,allow_preliminaryvs)

    T = param.T
    p_new = length(ps)
    outputarray[:,1] = fill(T(Inf),p)   #                   p, not p_new (if p>p_new, some will be Inf)
    psecondvs = max(2,Int(ceil(p.*param.fsecondvs)))

    speed_up_pvs = speedup_preliminarvys(param,n,ps,psecondvs)

    # optional first-stage, preliminary variable selection on a row sub-sample. Requires n>=50k and speed-gains>2 and p>3*ncores
    if allow_preliminaryvs && param.preliminaryvs !== :Off && speed_up_pvs>2 && n>=50_000 && p_new>=3*nprocs()

        ssi = randperm(Random.MersenneTwister(param.seed_subsampling+2*ntree),n)[1:param.n_preliminaryvs]  # subs-sample, no reimmission
        ys=y[ssi];ws=w[ssi];gammafit_ensembles=gammafit_ensemble[ssi];rs=r[ssi];G0s=G0[ssi,:];
        xs = SharedMatrixErrorRobust(x[ssi,:],param)      # NB: could be smarter, taking only columns I need to avoid large copies.
        length(h)==1 ? hs=h : hs=h[ssi]

        @sync @distributed for i in ps
            if Info_x[i].exclude==false  && Info_x[i].n_unique>1
                t   = (y=ys,w=ws,gammafit_ensemble=gammafit_ensembles,r=rs,h=hs,G0=G0s,xi=xs[:,i],infeatures=infeatures,fi=fi,info_i=Info_x[i],μgridi=μgrid[i],τgrid=τgrid,param=param)
                outputarray[i,:] = add_depth(t)     # [loss, τ, μ, m ]
            end
        end

        ps = sortperm(outputarray[:,1])[1:psecondvs]    # redefine ps
        outputarray[:,1] = fill(T(Inf),p)

    elseif param.subsampleshare_columns < 1
        psmall = convert(Int64,round(p*param.subsampleshare_columns))
        ps     = ps[randperm(Random.MersenneTwister(param.seed_subsampling+2*ntree),p)[1:psmall]]                  # subs-sample, no reimmission
    end

    # (second stage) variable selection.
    @sync @distributed for i in ps

        if Info_x[i].exclude==false  && Info_x[i].n_unique>1
            t   = (y=y,w=w,gammafit_ensemble=gammafit_ensemble,r=r,h=h,G0=G0,xi=x[:,i],infeatures=infeatures,fi=fi,info_i=Info_x[i],μgridi=μgrid[i],τgrid=τgrid,param=param)
            outputarray[i,:] = add_depth(t)     # [loss, τ, μ, m ]
        end
  
    end

    calibrate_n_preliminaryvs!(param,outputarray,psecondvs,n,y,gammafit_ensemble,w)   # done only in first iteration, if param.preliminaryvs=:On

    return Array(outputarray)   # convert SharedArray to Array

end



# Rough estimate of the speed-up from preliminary variable selection, assuming n and n_preliminaryvs are both fairly large, say > 10k.
# At lower values, the speed-up will be smaller. Preliminaryvs should probably require n>50k to be worth it.
function speedup_preliminarvys(param,n,ps,psecondvs)

    if typeof(param.n_preliminaryvs) <: Int
        p = length(ps)
        n = n*param.sharevs      
        cost_full_sample   = n*p
        cost_preliminaryvs = param.n_preliminaryvs*p + n*psecondvs 

        return cost_full_sample/cost_preliminaryvs
    
    else 

        return 0

    end

end      

# rough calibration of n_preliminaryvs on the first split of the first tree. On the full sample, computes the difference in loss between the best feature and 
# the psecondvs-th feature, and a rough estimate of the variance of this difference. Then computes the sample size that, in expectation, makes the difference
# (computed on the subsample) K times larger than the stde.  
function calibrate_n_preliminaryvs!(param,outputarray,psecondvs,n,y,gammafit_ensemble,w)

    if param.preliminaryvs==:Off || typeof(param.n_preliminaryvs) <: Int   # n_preliminaryvs already provided by user of computed in previous iterations
        return
    end

    K     = param.target_ratio_preliminaryvs 
    min_α = param.min_fraction_preliminaryvs

    ps = sortperm(outputarray[:,1]) 
    loss_best = outputarray[ps[1],1]
    loss_last = outputarray[ps[psecondvs],1]
 
    ll  = loglik_vector(param.loss,param,y,gammafit_ensemble)   
    
    Δ  = (loss_best - loss_last)/sum(w)                    # loss = sum(ll.*w)/loglikdivide*mean(w)  + priors
    wm = w/mean(w) 
    𝚺  = 2*(var(ll.*wm)/n)/(param.loglikdivide^2)      # 2 because I want var(Δ)
    
    α = (K*sqrt(𝚺)/Δ)^2

    α = min(α,10)   # In case of errors causing Inf 
    α = max(min_α,α) 

    param.n_preliminaryvs = Int(round(α*n))

end




function refineOptim(y,w,gammafit_ensemble,r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},infeatures,fi,info_i::Info_xi,μ0::T,τ0::T,m0::T,
    param::SMARTparam,gridvectorτ::AbstractVector{T}) where T<:AbstractFloat

    loss,τ,μ,nan_present = refineOptim_μτ_excluding_nan(y,w,gammafit_ensemble,r,h,G0,xi,infeatures,fi,info_i,μ0,τ0,param,gridvectorτ)

    if nan_present==true && loss<Inf      # loss=Inf for exluded features
        loss,m = refineOptim_m_given_μτ(y,w,gammafit_ensemble,r,h,G0,xi,infeatures,fi,info_i,μ,τ,m0,param)
    else
        m = m0
    end

    return loss,τ,μ,m
end



function refineOptim_m_given_μτ(y,w,gammafit_ensemble,r,h,G0,xi0,infeatures,fi,info_i,μ,τ,m0,param)

    llik0  = param.T(Inf)
    xi     = copy(xi0)

    if info_i.dichotomous
        xi[isnan.(xi0)] .= m0        # the estimate in best_m_given_μτ() won't change
        loss = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[T(0),T(0),m0],G,Gh,llik0)
        return loss,m0
    end

    m,loss = optimize_m(y,w,gammafit_ensemble,r,h,G0,xi0,param,infeatures,fi,info_i,τ,μ,m0,param.T,llik0)

    return loss,m

end



function optimize_m(y,w,gammafit_ensemble,r,h,G0,xi0,param,infeatures,fi,info_i,τ,μ,m0,T,llik0)

    n,p = size(G0)
    G   = Matrix{T}(undef,n,p*2)
    Gh  = similar(G)

    if (param.priortype==:sharp) || (τ==Inf)  # only two values of m to check
        μgridi = [μ-1,μ+1]
        lossmatrix = fill(T(Inf),2)
        xi = copy(xi0)

        for (j,m) in enumerate(μgridi)
            xi[isnan.(xi0)] .= m
            lossmatrix[j] = Gfitβ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,[μ,log(τ),m],G,Gh,llik0)
        end

        if argmin(lossmatrix)==1
            m,loss = μgridi[1],lossmatrix[1]
        else
            m,loss = μgridi[2],lossmatrix[2]
        end

    else
        res  = Optim.optimize( m -> Gfitβm(y,w,gammafit_ensemble,r,h,G0,xi0,param,infeatures,fi,info_i,μ,τ,m,G,Gh,llik0),[m0],Optim.BFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations = 100,x_tol = T(0.01) ))
        m,loss =  res.minimizer[1],res.minimum
    end

    return m,loss
end



function Gfitβm(y,w,gammafit_ensemble,r,h,G0,xi0,param,infeatures,fi,info_i,μ,τ,m,G,Gh,llik0)

    xi = copy(xi0)
    xi[isnan.(xi0)] .= m

    gL  = sigmoidf(xi,μ,τ,param.sigmoid,dichotomous=info_i.dichotomous)
    updateG!(G,G0,gL)
    loss,gammafit,β = fitβ(y,w,gammafit_ensemble,r,h,G,Gh,param,infeatures,fi,info_i,μ,τ,m[1],llik0,finalβ="refineOptim")

    return loss
end



# After completing the first step (selecting a feature), use μ0 and τ0 as starting points for a more refined optimization. Uses Optim
# Tolerance is set on μ, with smaller tolerance (1/2) if τ is  higher than 10
function refineOptim_μτ_excluding_nan(y,w,gammafit_ensemble,r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},infeatures,fi,info_i::Info_xi,μ0::T,τ0::T,
    param::SMARTparam,gridvectorτ::AbstractVector{T}) where T<:AbstractFloat

    miss_a = isnan.(xi)          # exclude missing values
    sum(miss_a)>0 ? nan_present=true : nan_present=false

    if nan_present
        keep_a = miss_a .== false
        y,w,gammafit_ensemble,r,G0,xi = y[keep_a],w[keep_a],gammafit_ensemble[keep_a],r[keep_a],G0[keep_a,:],xi[keep_a]
        length(h)==1 ? h = h : h = h[keep_a]
    end

    if info_i.dichotomous
       return T(Inf),τ0,μ0,nan_present
    end

    if param.priortype==:sharp || info_i.force_sharp==true
        τgrid = T[Inf]
    elseif τ0==T(Inf)  # best fitting value in preliminary feature selection was a sharp threshold; still allow for some lower values of τ as finer grid over μ gives sharp trees an advantage in the preliminary phase
        τgrid = convert(Vector{T},[5,10,20,40,80,Inf])
    else
        # optimize tau on a grid, and mu by NR with line search. If param.mugridpoints is smaller, the grid is wider
        if param.taugridpoints == 1
            τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -2.7:0.3:2.7 ]))
        elseif param.taugridpoints == 2
            τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -1.8:0.3:1.8 ]))
        else
            τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -0.9:0.3:0.9 ]))   # largest τ around 22 if largest tau in grid is 9
        end

        if τ0==gridvectorτ[1] && length(gridvectorτ)==3; τgrid=vcat( T(0.2),τgrid[1:end-1] ); end; # allow linear behavior

        if τ0==gridvectorτ[end] && param.priortype !==:smooth
            τgrid = vcat(τgrid,T.([40,80,Inf])) # Inf appealing for mixed discrete continuous, but differentiability is lost...
        end

    end

    if (param.priortype == :smooth) || (info_i.force_smooth==true)
        τgrid = τgrid[τgrid .<= param.max_tau_smooth]
        isempty(τgrid) ? τgrid = [param.max_tau_smooth] : nothing
    end

    lossmatrix = SharedArray{T}(length(τgrid),2)
    lossmatrix = fill!(lossmatrix,T(Inf))
    llik0  = loglik(param.loss,param,y,gammafit_ensemble,w)    # actual log-lik, not 2nd order approx

    if param.method_refineOptim == :pmap

        t = (y=y,w=w,gammafit_ensemble=gammafit_ensemble,r=r,h=h,G0=G0,xi=xi,param=param,infeatures=infeatures,
            fi=fi,info_i=info_i,τgrid=τgrid,μ0=μ0,T=T,llik0=llik0)
        curry(f,t) = i->f(i,t)
        optimize_mutau_map(i,t) = optimize_μτ(t.y,t.w,t.gammafit_ensemble,t.r,t.h,t.G0,t.xi,t.param,t.infeatures,t.fi,
            t.info_i,t.τgrid[i],t.μ0,t.T,t.llik0)
        res_map = pmap(curry(optimize_mutau_map,t),1:length(τgrid))  # vector of Optim.MultivariateOptimizationResults

        for indexτ in 1:length(τgrid)
            lossmatrix[indexτ,1] = res_map[indexτ].minimum
            lossmatrix[indexτ,2] = res_map[indexτ].minimizer[1]
        end

    elseif param.method_refineOptim == :distributed

        @sync @distributed for indexτ in 1:eachindex(τgrid)
            res = optimize_μτ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,τgrid[indexτ],μ0,T,llik0)
            lossmatrix[indexτ,1] = res.minimum
            lossmatrix[indexτ,2] = res.minimizer[1]
        end

    else
        @error " param.method_refineOptim incorrectly specified "
    end

    lossmatrix = Array{T}(lossmatrix)

    minindex = argmin(lossmatrix[:,1])
    loss  = lossmatrix[minindex,1]
    τ     = τgrid[minindex]
    μ     = lossmatrix[minindex,2]


    return loss,τ,μ,nan_present

end



# G  created here
function optimize_μτ(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,τ,μ0,T,llik0)

    n,p = size(G0)
    G   = Matrix{T}(undef,n,p*2)
    Gh  = similar(G)
    res  = Optim.optimize( μ -> Gfitβ2(y,w,gammafit_ensemble,r,h,G0,xi,param,infeatures,fi,info_i,μ,τ,T(0),G,Gh,llik0),[μ0],Optim.BFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations = 100,x_tol = param.xtolOptim/T(1+(τ>=10)+(τ>=0)+(τ>=40)+6*(τ>100))  ))

    return res

end



# returns a vector with the first approximately k Fibonacci numbers, as in [1,2,3,5...], multiplied by
# frequency_update*0.2/lambda, ensuring v[1] = 1
function fibonacci(k,lambda,frequency_update)
 
    v = Vector{Int}(undef,k+1)
    v[1] = 1
    v[2] = 1

    for i in 3:k+1
        v[i] = v[i-1] + v[i-2]
    end

    v = Int.(ceil.(v.*(0.2/lambda))*frequency_update )
    v[2] = 1  # in case lambda<0.2 

    return v[2:end]
end 






# param.best_features updated here
function fit_one_tree_inner(y::AbstractVector{T},w,SMARTtrees::SMARTboostTrees,r::AbstractVector{T},h::AbstractVector{T},x::AbstractArray{T},μgrid,Info_x,τgrid,param::SMARTparam;
    depth2=0,G0::AbstractMatrix{T}=Matrix{T}(undef,0,0) ) where T<:AbstractFloat

    gammafit_ensemble,infeatures,fi = SMARTtrees.gammafit,SMARTtrees.infeatures,SMARTtrees.fi
    best_features_current = Vector{param.I}(undef,0)
    ntree = length(SMARTtrees.trees)+1

    n,p   = size(x)
    I     = param.I

    if isempty(G0)
        G0 = ones(T,n,1)
    end

    loss0,gammafit0, ifit,μfit,τfit,mfit,infeatures,fi2,βfit = T(Inf),zeros(T,n),Int64[],T[],T[],T[],copy(infeatures),T[],T[]
    n_vs  = I(round(n*param.sharevs))        # number of observations to sub-sample

    if n_vs ≥ n
        ssi         = collect(1:n)
    else
        ssi         = randperm(Random.MersenneTwister(param.seed_subsampling+ntree),n)[1:n_vs]  # subs-sample, no reimmission
    end

    depth2==0 ? maxdepth=minimum([param.depth1,param.depth]) : maxdepth=depth2

    if length(ssi)<n
        xs = SharedMatrixErrorRobust(x[ssi,:],param)    # randomize once for the entire tree (SharedMatrix has a one-time cost)
        ys=y[ssi]; ws=w[ssi]; gammafit_ensembles=gammafit_ensemble[ssi]; rs=r[ssi]
        length(h)==1 ? hs=h : hs=h[ssi]
    end

    for depth in 1:maxdepth

        # variable selection, optionally including preliminaryvs, and optionally using a random sub-sample of the sample
        if length(ssi) == n
            outputarray = loopfeatures(y,w,gammafit_ensemble,r,h,G0,x,ifit,infeatures,fi,μgrid,Info_x,τgrid,param,ntree)  # loops over all variables
        else            # Variable selection using a random sub-set of the sample.
            outputarray = loopfeatures(ys,ws,gammafit_ensembles,rs,hs,G0[ssi,:],xs,ifit,infeatures,fi,μgrid,Info_x,τgrid,param,ntree)  # loops over all variables
        end

        i               = argmin(outputarray[:,1])  # outputarray[:,1] is loss (minus log marginal likelihood) vector
        loss0,τ0,μ0,m0  = outputarray[i,1],outputarray[i,2],outputarray[i,3],outputarray[i,4]
        infeatures      = updateinfeatures(infeatures,i)
        best_features_current = union(best_features_current,sortperm(outputarray[:,1])[1:min(p,param.number_best_features)] )

        if length(ssi)<n && param.refine_obs_from_vs
            loss,τ,μ,m = refineOptim(ys,ws,gammafit_ensembles,rs,hs,G0[ssi,:],x[ssi,i],infeatures,fi,Info_x[i],μ0,τ0,m0,param,τgrid)
        elseif param.n_refineOptim>=n
            loss,τ,μ,m = refineOptim(y,w,gammafit_ensemble,r,h,G0,x[:,i],infeatures,fi,Info_x[i],μ0,τ0,m0,param,τgrid)
        else           
            ssi2 = randperm(Random.MersenneTwister(param.seed_subsampling+ntree),n)[1:param.n_refineOptim]  # subs-sample, no reimmission
            length(h)==1 ? hs2=h : hs2=h[ssi2]
            loss,τ,μ,m = refineOptim(y[ssi2],w[ssi2],gammafit_ensemble[ssi2],r[ssi2],hs2,G0[ssi2,:],x[ssi2,i],infeatures,fi,Info_x[i],μ0,τ0,m0,param,τgrid)
        end

        # compute β on the full sample, at selected (ι,τ,μ,m) and update G0
        G   = Matrix{T}(undef,n,2^depth)
        Gh  = similar(G)
        depth==maxdepth ? finalβ="true" : finalβ="false"

        xi = copy(x[:,i])
        xi[isnan.(xi)] .= m      # replace missing with previously estimated m, and fit β on the entire sample

        gL  = sigmoidf(xi,μ,τ,param.sigmoid,dichotomous=Info_x[i].dichotomous)
        updateG!(G,G0,gL)

        if param.finalβ_obs_from_vs && length(ssi)<n && depth==maxdepth
            length(h)==1 ? hs=h : hs=h[ssi]
            loss,gammafit,β = fitβ(y[ssi],w[ssi],gammafit_ensemble[ssi],r[ssi],hs,G[ssi,:],Gh[ssi,:],param,infeatures,fi,Info_x[i],μ,τ,m,T(-Inf),finalβ=finalβ)
            gammafit = G*β
        else
            loss,gammafit,β = fitβ(y,w,gammafit_ensemble,r,h,G,Gh,param,infeatures,fi,Info_x[i],μ,τ,m,T(-Inf),finalβ=finalβ)
        end

        # store values and update matrices and param.best_features
        ifit, μfit, τfit, mfit, βfit  = vcat(ifit,i),vcat(μfit,μ),vcat(τfit,τ),vcat(mfit,m), β
        fi2 =vcat(fi2,( sum(gammafit.^2) - sum(gammafit0.^2) )/n)  # compute feature importance: decrease in mse
        G0, loss0, gammafit0 = G, loss, gammafit

    end

    if param.sparsevs==:On && ntree in fibonacci(20,param.lambda,param.frequency_update)
        param.best_features = union(param.best_features,best_features_current)
    end     

    # linesearch on the full sample (not needed if param.newton_max_steps_final>1)
    if param.newton_max_steps_final==1 && param.linesearch==true
        gammafit0,βfit,α = linesearch_α!(gammafit0,βfit,y,w,gammafit_ensemble,param)
    end

    return gammafit0,ifit,μfit,τfit,mfit,βfit,fi2

end



# modified gammafit,β
function linesearch_α!(gammafit,β,y,w,gammafit_ensemble,param)  # operates on weights

    T = typeof(param.lambda)

    if param.loss==:L2     # NB: if optimizing for any regression, weights must be imported as well if used.
        return gammafit,β,T(1)
    end

    α_grid = T.([0.5,0.75,1])   # numbers larger than 1 not recommended.

    loss_grid = SharedVector(fill(T(Inf),length(α_grid)))

    @sync @distributed for i in eachindex(α_grid)  # perhaps faster not to parallelize and break?
        loss_grid[i] = -loglik(param.loss,param,y,gammafit_ensemble .+ α_grid[i]*gammafit,w)
    end

    α_grid = Array(α_grid)
    α  = α_grid[argmin(loss_grid)]

    @. gammafit = α*gammafit
    @. β        = α*β

    return gammafit,β,α

end



# if depth>depth1, cycles of depth1, at the end of each cycle re-start the tree with fitted values. (only for smooth trees)
function fit_one_tree(y::AbstractVector{T},w,SMARTtrees::SMARTboostTrees,r::AbstractVector{T},
    h::AbstractVector{T},x::AbstractArray{T},μgrid,Info_x,τgrid,param::SMARTparam) where T<:AbstractFloat

    I = param.I
    nrounds = I(ceil( 1+(param.depth-param.depth1)/param.depth1 ))

    βfit = Vector{AbstractVector{T}}(undef,nrounds)

    gammafit0,ifit,μfit,τfit,mfit,β1,fi2=fit_one_tree_inner(y,w,SMARTtrees,r,h,x,μgrid,Info_x,τgrid,param;depth2=0)   # standard
    βfit[1]=β1

    for round in 2:nrounds
        d2 = minimum([param.depth1,param.depth-(round-1)*param.depth1 ])
        gammafit0,ifit_2,μfit_2,τfit_2,mfit_2,β2,fi2_2=fit_one_tree_inner(y,w,SMARTtrees,r,h,x,μgrid,Info_x,τgrid,param;depth2=d2,G0=gammafit0[:,:])
        βfit[round]=β2;ifit=vcat(ifit,ifit_2); μfit=vcat(μfit,μfit_2); τfit=vcat(τfit,τfit_2); mfit=vcat(mfit,mfit_2); fi2=vcat(fi2,fi2_2)
    end

    return gammafit0,ifit,μfit,τfit,mfit,βfit,fi2

end



function updateSMARTtrees!(SMARTtrees,Gβ,tree,ntree,param)

  T   = typeof(Gβ[1])
  n, depth = length(Gβ), length(tree.i)

  SMARTtrees.gammafit   = SMARTtrees.gammafit + SMARTtrees.param.lambda*Gβ
  push!(SMARTtrees.trees,tree)

  SMARTtrees.param = param

  for d in 1:depth
    SMARTtrees.fi2[tree.i[d]]  += tree.fi2[d]
    SMARTtrees.fr[tree.i[d]]  += 1
  end

  #fi = sqrt.(abs.(SMARTtrees.fi2.*(SMARTtrees.fi2 .>=0.0) ))  # fi is feature importance
  fi = SMARTtrees.fr                                           # fi is frequecy of inclusion
  SMARTtrees.fi = fi/sum(fi)

end



function SMARTtreebuild(x::AbstractMatrix{T},ij,μj::AbstractVector{T},τj::AbstractVector{T},mj::AbstractVector{T},βj,param::SMARTparam)::AbstractVector{T} where T<:AbstractFloat

    sigmoid = param.sigmoid
    depth1   = param.depth1
    missing_features = param.missing_features

    n,p   = size(x)
    depth = length(ij)
    gammafit = ones(T,n)

    I = typeof(depth1)
    nrounds = I(ceil( 1+(depth-depth1)/depth1 ))
    G   = Matrix{T}(undef,n,2^depth)

    for round in 1:nrounds
        G0 = copy(gammafit)

        for d in depth1*(round-1)+1:minimum([depth1*round,depth])
            G   = Matrix{T}(undef,n,2^d)
            i,μ,τ,m = ij[d], μj[d], τj[d], mj[d]

            if i in missing_features
                xi = copy(x[:,i])                 #  Redundant? Julia would make a copy with xi = x[:,i] anyway
                xi[isnan.(xi)] .= m
            else
                xi = @view(x[:,i])
            end

            gL      = sigmoidf(xi,μ,τ,sigmoid)
            updateG!(G,G0,gL)
            G0    = copy(G)
        end

        gammafit = G*βj[round]
    end

    return gammafit

end



# Computes mean weighted value of tau as exp(mean(logtau*w)), where w = sqrt(fi2). Dichotomous features are not counted.
# Can then be used to force sharp splits on those features where SMARTboost selects high τ, if these features contribute non-trivially to the fit.
# Argument: sharpness may be difficultfor SMARTboost to fit due to the greedy, iterative nature of the algorithm (the first values will tend to be smooth)
#
# Use:
# weighted_meean_tau = meean_weighted_tau(output.SMARTtrees)   # output is (p,1), vector of median weighted values of tau
function mean_weighted_tau(SMARTtrees)

    i,μ,τ,fi2 = SMARToutput(SMARTtrees)

    T = eltype(τ)
    p = max(length(SMARTtrees.infeatures),length(SMARTtrees.meanx))  # they should be the same ...
    avgtau = fill(T(0),p)
    Info_x = SMARTtrees.Info_x
    depth  = length(SMARTtrees.trees[1].i)

    @. fi2 = fi2*(fi2>0) + T(0)*(fi2<0)  # there can be some tiny negative values

    for j in 1:p

      τj = τ[i.==j]
      wj = sqrt.(fi2[i.==j])

      if length(τj)>0 && Info_x[j].dichotomous==false
        @. τj = τj*(τj<=100) + T(100)*(τj>100)    # bound Inf at 100
        avgtau[j] = sum(τj.*wj)/sum(wj)
      end

    end

    return avgtau

end




#=
    tau_info(SMARTtrees::SMARTboostTrees,warnings)

Provides info on posterior distribution of parameters, particularly mean and variance of log(tau) (with individual values weighted by their variance contribution).
variance is computed from posterior mean, mse from prior mean

# Example of use
output = SMARTfit(data,param)
avglntau,varlntau,mselntau,postprob2 = tau_info(output.SMARTtrees,warnings=:On)

Note: this computes a variance, while varlntau is a precision (the distribution is tau).
=#
# mean and variance of log(τ), weighted by feature importance for each tree
function tau_info(SMARTtrees::SMARTboostTrees)

    i,μ,τ,fi2 = SMARToutput(SMARTtrees)

    lnτ   = log.(τ)
    mw    = sum(lnτ.*fi2)/sum(fi2)
    varw  = sum(fi2.*(lnτ .- mw).^2)/sum(fi2)
    mse   = sum(fi2.*(lnτ .- SMARTtrees.param.meanlntau).^2)/sum(fi2)

    # ex-post probability of second (sharp) component

    param=SMARTtrees.param
    s   = sqrt(param.varlntau/param.depth)                 # to intrepret varlntau as dispersion

    T  = typeof(param.varlntau)
    dm2 = T(2)
    k2  = T(4)

    p1  = exp.( logpdft.(lnτ,param.meanlntau,s,param.doflntau) )
    p2  = exp.( logpdft.(lnτ,T(param.meanlntau+dm2/sqrt(param.depth)),s*k2,param.doflntau) )
    postprob2 = mean(p2./(p1+p2))

    return mw,varw,mse,postprob2
end
