
#
# SMARTparam    struct. Includes several options for cross-validation.
# SMARTdata     struct
# SMARTparam    function
# SMARTdata     function
# SMARTdata_sharedarray  function 
#
# param_given_data!               Sets those parameters that require having the complete data set (e.g. the total number of features...)
#                                 Typically, those that have a default :Auto in SMARTparam()
# param_constraints!
# SharedMatrixErrorRobust
# convert_df_matrix   # converts a dataframe to a Matrix{T} 
#





# NOTE: several fields are redundant, reflecting early experimentation, and will disappear in later versions
mutable struct SMARTparam{T<:AbstractFloat, I<:Int}

    T::Type 
    I::Type
    loss::Symbol             # loss function (log-likelihood)
    losscv::Symbol
    modality::Symbol
    coeff::Union{T,Vector{T}}        # coefficients (if any) which can be set by user and not updated
    coeff_updated::Vector{Vector{T}}  # coefficients (if any) computed internally, hidden to user, and typically updated
    verbose ::Symbol         # :On, :Off
    warnings::Symbol         # :On, :Off
    num_warnings::I
    # options for cross-validation (apply to ntrees and the outer loop over depth or other paramter)
    randomizecv::Bool        # true to scramble data for cv, false for contiguous blocks (PG: safer when there is a time-series aspect)
    nfold::I                 # n in n-fold cv. 1 for a single validation set (early stopping in the case of ntrees)
    nofullsample::Bool
    sharevalidation::Union{T,I}   # e.g. 0.2 or 1230. Relevant if nfoldcv = 1. The LAST block of observations is used if randomizecv = false
    indtrain_a::Vector{Vector{I}} # for user's provided Vector{Vector} of indices of train data ..
    indtest_a::Vector{Vector{I}}  # .. and test data. This over-writes nfold. 
    stderulestop::T         # e.g. 0.05. Applies to stopping while adding trees to the ensemble

    # learning rate
    lambda::T

    # Tree structure, priors, categorical data, missing
    depth::I
    depth1::I
    sigmoid::Symbol  # which simoid function. :sigmoidsqrt or :sigmoidlogistic. sqrt x/sqrt(1+x^2) 10 times faster than exp.
    meanlntau::T              # Assume a mixture of two student-t for log(tau).
    varlntau::T               #
    doflntau::T
    multiplier_stdtau::T
    varmu::T                # Helps in preventing very large mu, which otherwise can happen in final trees. 1.0 or even 2.0
    dofmu::T
    priortype::Symbol
    max_tau_smooth::T         # maximum value of tau allowed when :smooth. 10 can still capture very steep functions in boosting
    min_unique::I         # sharp thresholds are imposed on features with less than min_unique unique values
    mixed_dc_sharp::Bool   # true to force sharp splits on discrete and mixed discrete-continuous features (defined as having over 20% obs on a single value)
    force_sharp_splits::Vector{Bool}
    force_smooth_splits::Vector{Bool}
    exclude_features::Vector{Bool}
    augment_mugrid::Vector{Vector{T}}
    cat_features::Union{Vector{I},Vector{Symbol},Vector{String}}   # indices of categorical features, or names in DataFrame 
    cat_features_extended::Vector{I}                               # indices of categorical features requiring an extended representation  
    cat_dictionary::Vector{Dict}  # each element in the vector stores the dictionary mapping each category to a number
    cat_values::Vector{NamedTuple}
    cat_globalstats::NamedTuple
    cat_representation_dimension::I
    n0_cat::T                         # prior on number of observations for categorical data
    mean_encoding_penalization::T     # penalization on mean encoding  
    delete_missing::Bool
    mask_missing::Bool 
    missing_features::Vector{I}
    info_date::NamedTuple              # information on date feature
    sparsity_penalization::T  # AIC penalization on (number of features - depth)
    p0::Union{Symbol,I}       # :Auto (then set to p) or p0 

    # sub-sampling and pre-selection of features
    sharevs::Union{Symbol,T}  # if <1, adds noise to vs, in vs phase takes a random subset of observations
    refine_obs_from_vs::Bool  # true to add randomization to (μ,τ), assuming sharevs<1
    finalβ_obs_from_vs::Bool
    n_refineOptim::Union{Symbol,I}      # Maximum number of observations for refineOptim, to compute (μ,τ). β is then computed on all obs.
    subsampleshare_columns::T  # if <1.0, only a random share of features is used at each split (re-drawn at each split)
    sparsevs::Symbol           # :On, :Off, :Auto 
    frequency_update::T 
    number_best_features::I 
    best_features::Vector{I}
    pvs::Symbol              # :On, :Off, :Auto  
    p_pvs::I    # number of features taken to second stage (i.e. when actual G0 is used).
    min_d_pvs::I    # minimum depth at which to start preliminaryvs. 2 as a minimum. 
    # grid and optimization parameters
    mugridpoints::I  # points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data can benefit from more (due to with highly non-Gaussian (and non-uniform))
    taugridpoints::I  # points at which to evaluate τ during variable selection. 1-5 are supported. If less than 3, refinement is then on a grid with more points
    xtolOptim::T
    method_refineOptim::Symbol  # which method for refineOptim, e.g. pmap, @distributed, .... 
    points_refineOptim::I      # number of values of tau for refineOptim. Default 12. Other values allowed are 4,7.
    # others
    ntrees::I   # number of trees
    theta::T  # penalization of β: multiplies the default precision. Default 1. Higher values give tighter priors.
    loglikdivide::Union{Symbol,T}  # the log-likelhood is divided by this scalar. Used to improve inference when observations are correlated.
    overlap::I       # used in purged-CV and SMARTloglikdivide
    multiply_pb::T
    varGb::T
    ncores::I        # nprocs()-1, number of cores available
    seed_datacv::I     # sets random seed if randomizecv=true
    seed_subsampling::I   # sets random seed used on subsampling iterations (will be seed=seed_subsampling + iter)
    # Newton optimization
    newton_gaussian_approx::Bool
    newton_max_steps::I
    newton_max_steps_final::I
    newton_tol::T
    newton_tol_final::T
    newton_max_steps_refineOptim::I 
    linesearch::Bool

end



# TO DO: extend cat_ind so it can be a vector of strings or names from df 
struct SMARTdata{T<:AbstractFloat,D<:Any,I<:Int}
    y::AbstractVector{T}
    x::AbstractMatrix{T}
    weights::AbstractVector{T}
    dates::Vector{D}
    fnames::Vector{String}
    cat_ind::Vector{I}    # vector of indices of categorical features in matrix x, e.g. [1,3]
end




"""
    SMARTparam(;)
Parameters for SMARTboost

# Inputs that are more likely to be modified by user (all inputs are keywords with default values)

- `loss:Symbol`             [:L2] :L2,:logistic,:Huber,:t,:lognormal,:logt are supported.
                            :lognormal and :logt require y > 0 ( y~logt(m,sigma,v) if log(y)~t(m,sigma,v) );
                            in SMARTpredict(), they give predictions for E(y) if predict=:Ey, and for E(log(y)) if predict=:Egamma

- `modality:Symbol`         [:compromise] Options are: :accurate, :compromise, :fast, :fastest.
                            :fast runs only one model (only cv number of trees) at values defined in param = SMARTparam(). 
                            :fastest runs only one model, setting nfold=1 and nofullsample=true (does not re-estimate on the full sample after cv).
                            Recommended for faster preliminary analysis only.
                            For loss=:logistic, :fast and :fastest also use the quadratic approximation to the loss for large samples.
                            :accurate cross-validates several models (see SMARTfit() for more info).
                            :compromise cross-validates fewer models than :accurate. It is roughly 5 times slower than :fast, and 10 time slower than :fastest,
                                        
- `priortype`               [:hybrid] :hybrid encourages smoothness, but allows both smooth and sharp splits, :smooth forces smooth splits, :sharp forces sharp splits, :disperse has no prior on smoothness.
                            Set to :smooth if you want to force derivatives to be defined everywhere. 

- `randomizecv::Bool`       [false] default is purged-cv (see paper); a time series or panel structure is automatically detected (see SMARTdata) if
                            if a date column is provided. Set to true for standard cv.
- `nfold::Int`              [4] n in n-fold cv. Set nfold = 1 for a single validation set (by default the last param.sharevalidation share of the sample).
                            nfold, sharevalidation, and randomizecv are disregarded if train and test observations are provided by the user.
- `sharevalidation:`        [0.30] Can be: a) Size of the validation set, or b) Float, share of validation set.
                            Relevant only if nfold = 1. 
- `indtrain_a:Vector{Vector{I}} ` [] for user's provided array of indices of train sets. e.g. vector of 5 vectors, each with indices of train set observations
- `indtest_a:Vector{Vector{I}} `  [] for user's provided array of indices of test sets. e.g. vector of 5 vectors, each with indices of train set observations
- `nofullsample::Bool`      [false] if true and nfold=1, SMARTboost is not re-estimated on the full sample after cross-validation.
                            Reduces computing time by roughly 60%, at the cost of a modest loss of efficiency.
                            Useful for very large datasets, in preliminary analysis, in simulations, and when instructions specify a train/validation split with no re-estimation on full sample.
- `overlap::Int`            [0] number of overlaps in time series and panels. Typically overlap = h-1, where y(t) = Y(t+h)-Y(t). Used for purged-CV.
- `verbose::Symbol`         [:Off] verbosity :On or :Off
- `warnings::Symbol`        [:On] or :Off

- 'weights`                 NOTE: weights for weighted likelihood are set in SMARTdata, not in SMARTparam.
- 'cat_features`            [] vector of indices of categorical features, e.g. [2,5], or vector of names in DataFrame,
                            e.g. [:wage,:age] or ["wage","age"]. If empty, categoricals are automatically detected.
                            Set cat_features=[0] to override the automatic detection and force no categorical feature.  

- `lambda::Float`           [0.1 or 0.2] Learning rate. 0.10 for (nearly) best performance. 0.2 is a good compromise. Default is 0.1 of modality=:accruate, and 0.2 otherwise.
- `depth::Int`              [5] tree depth. Unless modality = :fast or :fastest, this is over-written as depth is cross-validated. See SMARTfit() for more options.
- `sparsity_penalization`   [0.3] positive numbers encourage sparsity. The range [0.0-1.5] should cover most scenarios. 
                            Automatically cv in modality=:compromise and :accurate. Increase to obtain a more parsimonious model.

# Inputs that may sometimes be modified by user (all inputs are keyword with default values)

- `ntrees::Int`             [2000] Maximum number of trees. SMARTfit will automatically stop when cv loss stops decreasing.
- `sharevs`                 [1.0] row subsampling in variable selection phase (only to choose feature on which to split.)
                            :Auto sets sharevs so that the subsample size is proportional to 50k*sqrt(n/50k).
                            At high n, sharevs<1 speeds up computations, but can reduce accuracy, particularly in sparse setting with low SNR.         
- `subsampleshare_columns`  [1.0] column subsampling (not recommended).
- `min_unique`              [:default] sharp splits are imposed on features with less than min_unique values (default is 5 for modality=:compromise or :accurate, else 10)
- `mixed_dc_sharp`          [false] true to force sharp splits on discrete and mixed discrete-continuous features (defined as having over 20% obs on a single value)
- `stderulestop::Float`     [0.01] A positive number stops iterations while the loss is still decreasing. This results in faster computations at minimal loss of fit.
- `delete_missing`          [false] true to delete rows with missing values in any feature, false to handle missing internally (recommended).
- `theta`                   [1]  numbers larger than 1 imply tighter penalization on β (final leaf values) compared to default.
- `meanlntau::Float`        [1.0] prior mean of log(τ). Set to higher numbers to suggest less smooth functions.        
- `mugridpoints::Int`       [10] number of points at which to evaluate μ during variable selection. 5 is sufficient on simulated data with normal or uniform distributions, but actual data may benefit from more (due to with highly non-Gaussian features).
                            For extremely complex and nonlinear features, more than 10 may be needed.        
- `force_sharp_splits`      [] optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a sharp split.
- `force_smooth_splits`     [] optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a smooth split (high values of τ not allowed).
- `cat_representation_dimension`  [3] 1 for mean encoding, 2 adds frequency, 3 adds variance.
- `losscv`                  [:default] loss function for cross-validation (:mse,:mae,:logistic,:sign). 
- `n_refineOptim::Int`      [10^6] MAXIMUM number of observations to use fit μ and τ (split point and smoothness).
                            Lower numbers can provide speed-ups with very large n at some cost in terms of fit.
- `loglikdivide::Float`     [1.0] with time series and longitudinal (or panel) data, higher numbers increase the strength or all priors. The defaults sets it internally using SMARTloglikdivide()

# Additional inputs can be set in SMARTfit(), but keeping the defaults is generally encouraged.

# Example
    param = SMARTparam()
# Example
    param = SMARTparam(nfold=1,sharevalidation=0.2)
"""
function SMARTparam(;

    T = Float32,   # Float32 or Float64. Float32 is up to twice as fast for sufficiently large n (due not to faster computations, but faster copying and storing)
    I = typeof(1),   
    loss = :L2,               # loss function
    losscv = :default,        # loss used for early stopping and stacking. :default, or :mse, :mae, :Huber, :logistic, :sign
    modality = :compromise,     # :accurate, :fast, :compromise 
    coeff = coeff_user(loss,T),      # coefficients (if any) used in loss
    coeff_updated = Vector{T}[],
    verbose = :Off,      # level of verbosity, :On, :Off
    warnings=:On,
    num_warnings=0,
    randomizecv = false,        # true to scramble data for cv, false for contiguous blocks ('Block CV')
    nfold       = 4,                 # n in n-fold cv. 1 for a single validation set (early stopping in the case of ntrees)
    nofullsample = false,        # true to skip the full sample fit after validation (only relevant if nfold=1)
    sharevalidation = 0.30,      #[0.30] Size of the validation set (integer), last sharevalidation rows of data. Or float, share of validation set. Relevant only if nfold = 1.
    indtrain_a = Vector{Vector{I}}(undef,0), # for user's provided Vector{Vector} of indices of train data ..
    indtest_a = Vector{Vector{I}}(undef,0),  # .. and test data. This over-writes nfold. 
    stderulestop = 0.01,         # e.g. 0.01. Applies to stopping while adding trees to the ensemble. larger numbers give smaller ensembles.
    lambda = 0.20,
    # Tree structure and priors
    depth  = 5,        # 3 allows 2nd degree interaction and is fast. 4 takes almost twice as much per tree on average. 5 can be 8-10 times slower per tree. However, fewer deeper trees are required, so the actual increase in computing costs is smaller.
    depth1 = 10,
    sigmoid = :sigmoidsqrt,  # :sigmoidsqrt or :sigmoidlogistic
    meanlntau= 1.0,    # Assume a Gaussian for log(tau).
    varlntau = 0.5^2,  # NB see loss.jl/multiplier_stdlogtau_y(). Centers toward quasi-linearity. This is the dispersion of the student-t distribution (not the variance unless dof is high).
    doflntau = 5.0,
    multiplier_stdtau = 5.0,
    varmu   = 2.0^2,    # smaller number make it increasingly unlikely to have nonlinear behavior in the tails. DISPERSION, not variance
    dofmu   = 10.0,
    priortype = :hybrid,
    max_tau_smooth = 20,         # maximum value of tau allowed when :smooth. 10 can still capture very steep functions in boosting
    min_unique  = :Auto,         # Note: over-writes force_sharp_splits unless set to a large number. minimum number of unique values to consider a feature as continuous
    mixed_dc_sharp = false,
    force_sharp_splits = Vector{Bool}(undef,0),  # typically hidden to user: optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a sharp split.
    force_smooth_splits = Vector{Bool}(undef,0),  # typically hidden to user: optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a smooth split (high values of λ not allowed)
    exclude_features = Vector{Bool}(undef,0),    # typically hidden to user optionally, a p vector of Bool, with j-th value set to true if the j-th feature should not be considered as a candidate for a split
    augment_mugrid  = Vector{Vector{T}}(undef,0),
    cat_features = Vector{I}(undef,0),        # p vector of Bool, j-th value set to true for a  categorical feature. If empty, strings are interpreted as categorical   
    cat_features_extended = Vector{I}(undef,0),
    cat_dictionary=Vector{Dict}(undef,0),       # global variable to store the dictionary mapping each category to a number
    cat_values = Vector{NamedTuple}(undef,0),
    cat_globalstats = (mean_y=T(0),var_y=T(0),q_y=T(0),n_0=T(0)),
    cat_representation_dimension = 3,           # dimension of the representation of categorical features
    n0_cat = 1,                                  # leave at 1! See preliminary_cv for why (multiplier). automatically cv prior on number of observations for categorical data
                                                 # cv tries higher values but not lower than n0_cat.
    mean_encoding_penalization = 0.3,            # automatically cv. The upper bound: 0.5 applies half the penalization, since this is done for all trees
    delete_missing = false,  
    mask_missing = false,                       # If true, adds a mask (dummy) for missing values. Does not seem required in any instance, and may worsen performance, although it may occasionally improve performance in small samples.
    missing_features = Vector{I}(undef,0), 
    info_date = (date_column=0,date_first=0,date_last=0),
    sparsity_penalization = 0.3,
    p0       = :Auto,
    sharevs  = 1.0,             # if <1, adds noise to vs, in vs phase takes a random subset. :Auto is 0.5 if n>=250k     
    refine_obs_from_vs = false,  # true to add randomization to (μ,τ), assuming sharevs<1
    finalβ_obs_from_vs  = false,  # true to add randomization to final β
    n_refineOptim = 10_000_000,   # Subsample size fore refineOptim. beta is always computed on the full sample.
    subsampleshare_columns = 1.0,  # if <1.0, only a random share of features is used at each split (re-drawn at each split)
    sparsevs = :Auto,           # :Auto switches it :On if sparsity_penalization>0, else :Off 
    frequency_update = 1.0,       # when sparsevs, 1 for Fibonacci, 2 to update at 2*Fibonacci etc...               
    number_best_features = 10,    # number of best feature in each node to store into best_features    
    best_features   = Vector{I}(undef,0),
    pvs = :Off,              # :On, :Off, :Auto. Experiments suggests modest speed gains when sparsevs=:On, but could speed up with very large p and depth > 5, at some loss of fit.   
    p_pvs = 100,           # number of features taken to second stage (i.e. when actual G0 is used). 100 chosen by experimentation. No gains to set it lower. 
    min_d_pvs = 4,        # minimum depth at which to start preliminary vs. >=2. 
    # grid and optimization parameters
    mugridpoints = 10,  # points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data can benefit from more (due to with highly non-Gaussian features
    taugridpoints = 3,  # points at which to evaluate τ during variable selection. 1-5 are supported. If less than 3, refinement is then on a grid with more points
    xtolOptim = 0.02,  # tolerance in the optimization e.g. 0.02 (measured in dμ). It is automatically reduced if tau is large 
    method_refineOptim = :pmap, #  :pmap, :distributed 
    points_refineOptim = 12,    # number of values of tau for refineOptim. Default 12. Other values allowed are 4,7.
    # miscel
    ntrees = 2000, # number of trees. 1000 is CatBoost default, but in SMARTboost trees are shallow.  
    theta = 1.0,   # numbers larger than 1 imply tighter penalization on β compared to default. 
    loglikdivide = :Auto,   # the log-likelhood is divided by this scalar. Used to improve inference when observations are correlated.
    overlap = 0,
    multiply_pb = 1.0,
    varGb   = NaN,      # Relevant only for first tree
    seed_datacv = 1,       # sets random seed if randomizecv=true
    seed_subsampling = 2,  # sets random seed used on subsampling iterations (will be seed=seed_subsampling + iter)
    # Newton optimization: good default is one step in preliminary phase, and evaluate actual log-lik, and iterate to convergence for final β
    newton_gaussian_approx = false, # true has large speed gains for logistic for large n (loss=...exp.()). If true, and if newton_max_steps=1 (hence not in final) evaluates the Gaussian approximation to the log-likelihood rather than the likelihood itself except in final phase (given i,mu,tau)
    newton_max_steps = 1,         # vs phase. 1 seems sufficient, in combination with gaussian_approx=false, to get most of the gains  
    newton_max_steps_final = 20,  # small impact on cost and large gains.
    newton_tol = 1,        # Keep large (e.g. 1.0) to avoid unnecessay iterations in preliminary phase
    newton_tol_final= 0.01,
    newton_max_steps_refineOptim=1, # number of steps in refineOptim
    linesearch=true                # true to perform line search after final estimates (α*β)
    )

    if typeof(T)!==DataType; if T=="Float32"; T=Float32; else; T = Float64; end; end;  # for R users
           
    if loss==:L2    
        newton_max_steps,newton_max_steps_final,newton_gaussian_approx = 1,1,false 
    end     

    if loss==:Huber && warnings==:On 
        @warn "loss = :t is recommended instead of loss = :Huber. The t loss automatically estimates the degres of freedom and typically converges faster. "
    end     

    @assert(doflntau>T(2), " doflntau must be greater than 2.0 (for variance to be defined) ")
    @assert(T(1e-20) < xtolOptim, "xtolOptim must be positive ")

    if depth>=7 && warnings==:On
        @warn "setting param.depth higher than 6 typically results in very high computing costs"
    end

    ncores = nprocs()-1
    

    # The following works even if sharevalidation is a Float which is meant as an integer (e.g. in R wrapper)
    if T(sharevalidation)==T(round(sharevalidation))  # integer
        sharevalidation=I(sharevalidation)
    else
        sharevalidation = T(sharevalidation)
    end

    if eltype(cat_features) <: Real 
        cat_features = I.(cat_features)
    end

    sharevs==:Auto ? nothing : sharevs=T(sharevs)
    loglikdivide==:Auto ? nothing : loglikdivide=T(loglikdivide)
    p0==:Auto ? nothing : p0=I(p0)   # if :Auto, set in param_given_data!()
 
    if min_unique==:Auto
        modality in [:fast,:fastest] ? min_unique = 10 : min_unique = 5
    end     

    # Functions on indtrain_a and indtest_a, if user provides them 
    if length(indtrain_a)>length(indtest_a)
        @error "There are $(length(indtrain_a)) vectors in indtrain_a, but $(length(indtest_a)) in indtest_a"
    end 

    nfold_user = length(indtrain_a)

    if nfold_user>0           # over-write randomizecv and nfold_user
        nfold = nfold_user
        randomizecv = false
        indtrain_a,indtest_a = Vector{Vector{I}}(undef,nfold),Vector{Vector{I}}(undef,nfold)    
        for i in 1:nfold_user  # ensure indices are integers (for R, Python etc..)
            indtrain_a[i] = I.(round.(indtrain_a[i]))
            indtest_a[i]  = I.(round.(indtest_a[i]))
        end
    end 
     
    param = SMARTparam(T,I,loss,losscv,Symbol(modality),T.(coeff),coeff_updated,Symbol(verbose),Symbol(warnings),I(num_warnings),randomizecv,I(nfold),nofullsample,T(sharevalidation),indtrain_a,indtest_a,T(stderulestop),T(lambda),I(depth),I(depth1),Symbol(sigmoid),
        T(meanlntau),T(varlntau),T(doflntau),T(multiplier_stdtau),T(varmu),T(dofmu),Symbol(priortype),T(max_tau_smooth),I(min_unique),mixed_dc_sharp,force_sharp_splits,force_smooth_splits,exclude_features,augment_mugrid,cat_features,cat_features_extended,cat_dictionary,cat_values,cat_globalstats,I(cat_representation_dimension),T(n0_cat),T(mean_encoding_penalization),Bool(delete_missing),mask_missing,missing_features,info_date,T(sparsity_penalization),p0,sharevs,refine_obs_from_vs,finalβ_obs_from_vs,
        I(n_refineOptim),T(subsampleshare_columns),Symbol(sparsevs),T(frequency_update),
        I(number_best_features),best_features,Symbol(pvs),I(p_pvs),I(min_d_pvs),I(mugridpoints),I(taugridpoints),T(xtolOptim),Symbol(method_refineOptim),
        I(points_refineOptim),I(ntrees),T(theta),loglikdivide,I(overlap),T(multiply_pb),T(varGb),I(ncores),I(seed_datacv),I(seed_subsampling),newton_gaussian_approx,
        I(newton_max_steps),I(newton_max_steps_final),T(newton_tol),T(newton_tol_final),I(newton_max_steps_refineOptim),linesearch)

    param_constraints!(param) # enforces constraints across options. Must be repeated in SMARTfit.

    return param
end



# Sets some parameters that require knowledge of data. Notice that data.x (but no data.y) may contain NaN. 
function param_given_data!(param::SMARTparam,data::SMARTdata)

    n,p = size(data.x)
    I = typeof(param.nfold)
    T = typeof(param.lambda)

    # compute loglikdivide unless user provided it 
    if param.loglikdivide == :Auto
        lld,ess = SMARTloglikdivide(data.y,data.dates,overlap=param.overlap)
        param.loglikdivide = T(lld)
    end

    if param.p0==:Auto 
        param.p0=p 
    end

    if param.sparsevs==:Auto
        param.sparsity_penalization>0 ? param.sparsevs=:On : param.sparsevs=:Off
    end 

    if param.sharevs==:Auto
        n = 0.75*n        # assuming n i train+sample
        α = T(min(1,sqrt(50_000/n)))
        α>0.75 ? α=T(1) : nothing
        param.sharevs = α
    end 

    if param.pvs==:Auto
        p>10*param.p_pvs && param.depth>5 ? param.pvs=:On : param.pvs=:Off
    end 
    
end         



# separate function enforces constraints across options.
function param_constraints!(param::SMARTparam)

    if param.meanlntau==Inf; param.priortype==:sharp; end;

    if param.priortype==:smooth
        param.doflntau=100
    end

    if param.loss == :logistic 
        param.cat_representation_dimension = min(2,param.cat_representation_dimension)
    end     

end



# converts a dataframe to a Matrix{T} of x for SharedArray
function convert_df_matrix(df,T)

  n,p = size(df)
  x   = Matrix{T}(undef,n,p)

  for i in 1:p
    x[:,i] = T.(df[:,i])
  end

  return x

end



"""
        SMARTdata(y,x,param,[dates];T=Float32,fnames=[],enforce_dates=true)
Collects and pre-processes data in preparation for fitting SMARTboost

# Inputs

- `y::Vector`:              Vector of responses. Can be a vector of lables, or a dataframe with one column. 
- `x`:                      Matrix of features. Can be a vector or matrix of floats, or a dataframe. Converted internally to a Matrix{T}, T as defined in SMARTparam
- `param::SMARTparam`:


# Optional Inputs

- `dates::AbstractVector`:  [1:n] Typically Vector{Date} or Vector{Int}. Used in cross-validation to determine splits.
                            If not supplied, the default 1:n assumes a cross-section of independent realizations (conditional on x) or a single time series.
- 'weights':                [ones(T,n)] vector of floats or Floats, weights for weighted likelhood
- `fnames::Vector{String}`: [x1, x2, ... ] feature names


# Examples of use
    data = SMARTdata(y,x,param)
    data = SMARTdata(y,x,param,dates=dates,fnames=names)
    data = SMARTdata(y,df[:,[:CAPE, :momentum ]],param,dates=df.dates,fnames=df.names)
    data = SMARTdata(y,df[:,3:end],param)

# Notes
- When dates are provided, the data will be ordered chronologically (for cross-validation functions) unless the user has provided explicit training and validation sets.
"""
function SMARTdata(y0::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},x::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},
    param::SMARTparam,dates::AbstractVector=[];weights::AbstractVector=[],fnames = Vector{String}[])  

    T    = param.T

    if typeof(y0)<:AbstractDataFrame || eltype(y0) <: Union{Bool,Number} || typeof(y0)<:AbstractMatrix
        y = y0[:,1]
    else 
        @error "in SMARTdata, y must be of type Number or Bool (true,false)"
    end      

    check_admissible_data(y,param)  # check if data is admissible given param (typically param.loss)

    if param.loss in [:lognormal,:logt]
        @. y = log(y)
    end             

    if isempty(weights)
        weights=ones(T,length(y))
    else    
        @assert(minimum(weights)>0.0, " weights in SMARTdata() must be strictly positive ")
    end    

    if isempty(fnames) && typeof(x)<:AbstractDataFrame
        fnames = names(x)
    elseif !isempty(fnames) && typeof(x)<:AbstractDataFrame
        rename!(x, fnames)
    elseif isempty(fnames)
        fnames = ["x$i" for i in 1:size(x,2)]    
    end 

    if isempty(dates)
        dates = collect(1:length(y))
        ordered_dates = true
    else
        datesu  = unique(dates)
        ordered_dates = datesu==sort(datesu)   # Are data in chronological order?

        # if dates are not in chronological order, sort y,x,weights by date (done below for xp) UNLESS user has provided indtrain_a 
        # (cross-validation obs), in which case issue a warning 
        if !ordered_dates && !isempty(param.indtrain_a)
            @warn " Dates are not in chronological order, but user has provided explicity training and validation sets, so data will be left unordered."
        end
  
    end

    # pre-process data

    # if x is Float, transform to dataframe for convenient data manipulation
    if typeof(x) <: AbstractDataFrame
        xp = deepcopy(x)    
    elseif typeof(x) <: AbstractVector    
        xp = DataFrame(hcat(x),Symbol.(fnames))
    else 
        xp = DataFrame(x,Symbol.(fnames))    
    end

    if !ordered_dates && isempty(param.indtrain_a)
        sortindex = sortperm(dates)
        y       = y[sortindex]
        xp      = xp[sortindex,:]
        weights = weights[sortindex]
        dates   = dates[sortindex]
    end 

    xp=replace_nan_with_missing(xp)    # facilitates functions using skipmissing, and identification of missing into a single case

    if param.delete_missing == true
        keep_obs = .!vec(any(ismissing.(x),dims=2))
        xp = xp[keep_obs,:]
        y  = y[keep_obs]
        weights = weights[keep_obs]
        dates   = dates[keep_obs]
    end     

    # delete any row where y is missing 
    y = replace_nan_with_missing(y)    # facilitates functions using skipmissing, and identification of missing into a single case
    keep_obs = .!(ismissing.(y))
    xp  = xp[keep_obs,:]
    y   = y[keep_obs]
    weights = weights[keep_obs]
    dates   = dates[keep_obs]

    convert_dates_to_real!(xp,param)   # modifies both param and x: updates info_date in param
    categorical_features!(param,xp)    # updates param.cat_features_bool and param.cat_features
    missing_features!(param,xp)        # updates param.missing_features
    if param.mask_missing == true 
        xp,fnames = missing_features_extend_x(param,xp)       #  adds columns to xp 
    end
    map_cat_convert_to_float!(xp,param,create_dictionary=true)  # map cat to Dictionary, converts to float 0,1,....
    extended_categorical_features!(param,fnames)   # finds categorical features needing an extensive (more than one column) representation
    xp = replace_missing_with_nan(xp)   # SharedArray do not accept missing.

    data = SMARTdata(T.(y),convert_df_matrix(xp,T),T.(weights),dates,fnames,param.cat_features)

    return data

end


# SMARTdata() should be invoked only once, by the user. SMARTdata_sharedarray to be used everywhere else.
# It creates an instance of SMARTdata where x is a SharedArray.
# I don't think there gains from making y and weights SharedVector. Only x.
# SharedArray can cause a "mmap: Access is denied" error, presumably on systems shared by several users
function SMARTdata_sharedarray(y::Union{AbstractVector,AbstractDataFrame},x::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},
   param::SMARTparam,dates::AbstractVector,weights::AbstractVector,fnames::Vector{String} )
    
   data = SMARTdata(y,SharedMatrixErrorRobust(x,param),weights,dates,fnames,param.cat_features)

   return data

end     



# Attempts to transform x into a SharedMatrix. If it fails, returns x.
# SharedArrays are 10-30% faster (depending on the number of features and the number of cores).  
function SharedMatrixErrorRobust(x,param)
    try
        return SharedMatrix(x)
    catch err
        if param.warnings==:On && param.num_warnings<1 # warn only once
            #@info "System error (mmap: Access denied) in allocating SharedArray. This problem seems exclusive to Windows, and to high number of features.
            #Switching to standard array, which is 10-30% slower when the number of features and/or the number of cores is high."
            #param.num_warnings = param.num_warnings + 1
        end
        return x
    end
end
