
#
# HTBparam    struct. Includes several options for cross-validation.
# HTBdata     struct
# HTBparam    function
# HTBdata     function
# HTBdata_sharedarray  function 
# HTBdata_subset  function 
#
# param_given_data!               Sets those parameters that require having the complete data set (e.g. the total number of features...)
#                                 Typically, those that have a default :Auto in HTBparam()
# param_constraints!
# SharedMatrixErrorRobust
# convert_df_matrix   # converts a dataframe to a Matrix{T} 
# effective_lambda                Potentially allows a schedule of lambda(iter)
#




# NOTE: several fields are redundant, reflecting early experimentation, and will disappear in later versions
mutable struct HTBparam{T<:AbstractFloat, I<:Int,R<:Real}

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
    depthppr::I        # projection pursuit depth. 0 to disactivate.
    ppr_in_vs::Symbol  # :On for projection pursuit included in feature selection stage. Can occasionally go wrong. :Off is a safer default, and :On could be used for stacking. 
    sigmoid::Symbol  # which simoid function. From sharper to smoother separation: :smoothstep (a softer version of truncated linear), :sigmoidlogistic, :sigmoidsqrt, :smoothsigmoid                 
                     # running times are: smoothstep 0.3, :sigmoidsqrt 0.4, :sigmoidlogistic 1.6, :smoothsigmoid 3.0 (slowest).
    meanlntau::T              # Assume a mixture of two student-t for log(tau).
    varlntau::T               #
    doflntau::T
    multiplier_stdtau::T
    d_meanlntau_cat::T       # difference in intercept meanlntau for categorical features (prior that they are less smooth)
    varmu::T                # Helps in preventing very large mu, which otherwise can happen in final trees. e.g. 2.0. No gain in accuracy though.
    dofmu::T
    meanlntau_ppr::T         # for projection pursuit 
    varlntau_ppr::T 
    doflntau_ppr::T 
    priortype::Symbol
    max_tau_smooth::T         # maximum value of tau allowed when :smooth. 10 can still capture very steep functions in boosting
    min_unique::I         # sharp thresholds are imposed on features with less than min_unique unique values
    mixed_dc_sharp::Bool   # true to force sharp splits on discrete and mixed discrete-continuous features (defined as having over 20% obs on a single value)
    tau_threshold::T       # threshold for imposing sharp splits
    half_life_depth::I     # half-life of max_depth expressed in number of trees.  
    min_depth_ratio::T     # minimum depth is ceil(min_depth_ratio*depth)
    force_sharp_splits::Vector{Bool}
    force_smooth_splits::Vector{Bool}
    exclude_features::Vector{Bool}
    augment_mugrid::Vector{Vector{T}}
    cat_features::Union{Vector{I},Vector{Symbol},Vector{String}}   # indices of categorical features, or names in DataFrame 
    cat_features_extended::Vector{I}                               # indices of categorical features requiring an extended representation  
    cat_dictionary::Vector{Dict}  # each element in the vector stores the dictionary mapping each category to a number
    cat_values::Vector{NamedTuple}
    cat_globalstats::NamedTuple
    cat_representation_dimension::I   # dimension of the representation of categorical features: mean,frequency,std,skew,kurt (robust measures)
    n0_cat::T                         # prior on number of observations for categorical data
    mean_encoding_penalization::T     # penalization on mean encoding
    cv_categoricals::Symbol           # whether to run a preliminary cross-validation for categorical features
    class_values::Union{Vector{String},Vector{R}} # for loss=:multiclass, stores the values of unique(y)
    delete_missing::Bool
    mask_missing::Bool 
    missing_features::Vector{I}
    info_date::NamedTuple              # information on date feature
    sparsity_penalization::T  # AIC penalization on (number of features - depth)
    same_feature_penalization::T  # penalization multiplies the number of times the feature appears in different splits of the same tree. Think of it as log(prob_split_different feature/prob_split_same_feature), e.g. [0 2]
    same_feature_penalization_start::I # the first same_feature_penalization_start appearances are not penalized. 
    p0::Union{Symbol,I}       # :Auto (then set to p) or p0 

    # row and column sub-sampling and two schemes to deal with large p: sparsevs and pvs
    sharevs::Union{Symbol,T}  # if <1, adds noise to vs, in vs phase takes a random subset of observations
    refine_obs_from_vs::Bool  # true to add randomization to (μ,τ), assuming sharevs<1
    finalβ_obs_from_vs::Bool
    n_refineOptim::Union{Symbol,I}      # Maximum number of observations for refineOptim, to compute (μ,τ). β is then computed on all obs.
    subsampleshare_columns::T  # if <1.0, only a random share of features is used at each split (re-drawn at each split)
    sparsevs::Symbol           # :On, :Off
    frequency_update::T 
    number_best_features::I 
    best_features::Vector{I}
    # grid and optimization parameters
    mugridpoints::I  # points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data can benefit from more (due to with highly non-Gaussian (and non-uniform))
    taugridpoints::I  # points at which to evaluate τ during variable selection. 1-5 are supported. If less than 3, refinement is then on a grid with more points
    depth_coarse_grid::I # at this depth and higher, vs uses every second point in mugrid. 
    depth_coarse_grid2::I # at this depth and higher, vs uses only τ=5 and every third point in mugrid. 
    depth_coarse_grid3::I # at this depth and higher, vs uses only τ=5 and only one point (central) in mugrid. 
    xtolOptim::T
    method_refineOptim::Symbol  # which method for refineOptim, e.g. pmap, @distributed, .... 
    points_refineOptim::I      # number of values of tau for refineOptim. Default 12. Other values allowed are 4,7.
    # number of trees
    ntrees::I   # number of trees
    # penalization of β
    theta::T  # penalization of β: multiplies the default precision. Default 1. Higher values give tighter priors.
    # others
    loglikdivide::Union{Symbol,T}  # the log-likelhood is divided by this scalar. Used to improve inference when observations are correlated.
    overlap::I       # used in purged-CV and HTBloglikdivide
    multiply_pb::T
    varGb::T
    ncores::I        # nprocs()-1, number of cores available
    seed_datacv::I     # sets random seed if randomizecv=true
    seed_subsampling::I   # sets random seed used on subsampling iterations (will be seed=seed_subsampling + iter)
    iter::I               # used internally, which tree is currently being run 
    # Newton optimization
    newton_gauss_approx::Union{Symbol,Bool}
    newton_max_steps::I
    newton_max_steps_final::I
    newton_tol::T
    newton_tol_final::T
    newton_max_steps_refineOptim::I 
    linesearch::Bool

end



# TO DO: extend cat_ind so it can be a vector of strings or names from df 
struct HTBdata{T<:AbstractFloat,D<:Any,I<:Int}
    y::AbstractVector{T}
    x::AbstractMatrix{T}
    weights::AbstractVector{T}
    dates::Vector{D}
    fnames::Vector{String}
    cat_ind::Vector{I}    # vector of indices of categorical features in matrix x, e.g. [1,3]
    offset::Vector{T} 
end



# Creates a HTBdata dataset by taking a subsample (indexed by ind::Vector{Bool}) of another. 
# Example of use: data_not0 = data_take_subset(data,data.y .!= 0)
function HTBdata_subset(data::HTBdata,param::HTBparam,ind)

    y   = data.y[ind]
    x   = data.x[ind,:]
    weights = data.weights[ind]
    dates   = data.dates[ind]
    offset  = data.offset[ind]

    fnames  = data.fnames 
    cat_features = param.cat_features 

    data_ind = HTBdata(y,SharedMatrixErrorRobust(x,param),weights,dates,fnames,cat_features,offset)

    return data_ind 

end 



"""
    HTBparam(;)
Parameters for HTBoost

Note: all Julia symbols can be replaced by strings. e.g. :L2 can be replaced by "L2".

# Parameters that are most likely to be modified by user (all inputs are keywords with default values)

- `loss`             [:L2] Supported distributions:
    - :L2 (Gaussian), aliases :l2,:mse,:Gaussian,:normal
    - :logistic, aliases :binary (binary classification)
    - :multiclass (multiclass classification)
    - :t, aliases :student (student-t, robust alternative to :L2)
    - :Huber, aliases :huber 
    - :gamma
    - :lognormal, aliases :logL2, :logl2 (positive continuous data) 
    - :Poisson (count data)
    - :gammaPoisson, aliases :gamma_Poisson,:gamma_poisson,:negbin,:negative_binomial (aka negative binomial, count data)
    - :L2loglink, aliases :l2loglink (alternative to :L2 if y≥0)
    - :hurdleGamma (zero-inflated y)
    - :hurdleL2loglink (zero-inflated y)
    - :hurdleL2 (zero-inflated y)

- See the examples for uses of each loss function. Fixed coefficients (such as shape for :gamma, dispersion and dof for :t, and overdispersion for :gammaPoisson) are computed internally by maximum likelihood. Inspect them using *HTBcoeff()*. In *HTBpredict()*, predictions are for E(y) if predict=:Ey (default), while predict=:Egamma forecasts the fitted parameter ( E(logit(prob) for :logistic, log(E(y)) for :gamma etc ... )

- `modality`         [:compromise] Options are: :accurate, :compromise (default), :fast, :fastest.
                     These options are meant to replace the need for the user to cross-validate parameters. Advanced users with a big computational budget can still do so.
                     :fast and :fastest run only one model, while :compromise and :accurate cross-validate the most important parameters.
                     :fast runs only one model (only cv number of trees) at values defined in param = HTBparam(). 
                     :fastest runs only one model, setting lambda=0.2, nfold=1 and nofullsample=true (does not re-estimate on the full sample after cv).
                      Recommended for faster preliminary analysis only.
                      In most cases, :fast and :fastest also use the quadratic approximation to the loss for large samples.
                      :compromise and :accurate cross-validates several models at the most important parameters (see HTBfit() for details),
                      then stack all the cv models.
                                        
- `priortype`               [:hybrid] :hybrid encourages smoothness, but allows both smooth and sharp splits, :smooth forces smooth splits,
                            :disperse is :hybrid but with no penalization encouraging smooth functions (not recommended).
                            Set to :smooth if you want to force derivatives to be defined everywhere. 

- `randomizecv`       [false] default is block-cv (aka purged cv); a time series or panel structure is automatically detected (see HTBdata)
                            if a date column is provided. Set to true for standard cv.

- `nfold`              [4] n in n-fold cv. Set nfold = 1 for a single validation set (by default the last param.sharevalidation share of the sample).
                            nfold, sharevalidation, and randomizecv are disregarded if train and test observations are provided by the user.

- `sharevalidation:`        [0.30] Can be: a) Integer, size of the validation set, or b) Float, share of validation set.
                            Relevant only if nfold = 1.

- `indtrain_a:Vector{Vector{I}} ` [] for user's provided array of indices of train sets. e.g. vector of 5 vectors, each with indices of train set observations

- `indtest_a:Vector{Vector{I}} `  [] for user's provided array of indices of test sets. e.g. vector of 5 vectors, each with indices of train set observations

- `nofullsample`      [false] if true and nfold=1, HTBoost is not re-estimated on the full sample after validation.
                            Reduces computing time by roughly 60%, at the cost of a modest loss of accuracy.
                            Useful for very large datasets, in preliminary analysis, in simulations, and when instructions specify a train/validation
                            split with no re-estimation on full sample. Activated by default when modality=:fastest.     

- `cat_features`            [] vector of indices of categorical features, e.g. [2] or [2,5], or vector of names in DataFrame,
                            e.g. [:wage,:age] or ["wage","age"]. If empty, categoricals are automatically detected as non-numerical features.

- `cv_categoricals`     [:default] whether to run preliminary cross-validation on parameters related to categorical features.
                        :none uses default parameters 
                        :penalty runs a rough cv the penalty associated to the number of categories; recommended if n/n_cat if low for any feature, particularly if SNR is low                             
                        :n0 runs a rough of cv the strength of the prior shrinking categorical values to the overall mean; recommended with highly unequal number of observations in different categories
                        :both runs a rough cv or penalty and n0 
                        :default uses :none for modality in [:fastest,:fast], :penalty for :compromise, and :both for :accurate        

- `overlap:`            [0] number of overlaps in time series and panels. Typically overlap = h-1, where y(t) = Y(t+h)-Y(t). Used for purged-CV.

- `verbose`         [:Off] verbosity :On or :Off

- `warnings`        [:On] or :Off


# Parameters that may sometimes be be modified by user

- `lambda`           [0.1 or 0.2] Learning rate. 0.1 for (nearly) best performance. 0.2 can be almost as accurate, particularly if the function is smooth and p is small.
                     The default is 0.1, except in modality = :fastest, where it's 0.2. Modality = :compromise carries out the cv at lambda=0.2 and then fits the best model at 0.1.
                     Consider 0.05 if tiny improvements in accuracy are important and computing time is not a concern.

- `depth`              [5] tree depth. Unless modality = :fast or :fastest, this is over-written as depth is cross-validated. See HTBfit() for more options.

- `weights`                 NOTE: weights for weighted likelihood are set in HTBdata, not in HTBparam.
- `offset`                  NOTE: offsets (aka exposures) are set in HTBdata, not in HTBparam. See examples/Offset or exposure.jl     

- `sparsity_penalization`   [0.3] positive numbers encourage sparsity. The range [0.0-1.5] should cover most scenarios. 
                            Automatically cv in modality=:compromise and :accurate. Increase to obtain a more parsimonious model, set to 0 for standard boosting.

- `ntrees`             [3000] Maximum number of trees. HTBfit will automatically stop when cv loss stops decreasing.

- `sharevs`                 [1.0] row subsampling in variable selection phase (only to choose feature on which to split.) Default is no subsampling.
                            sharevs = :Auto sets the subsample size to min(n,50k*sqrt(n/50k)).
                            At high n, sharevs<1 speeds up computations, but can reduce accuracy, particularly in sparse setting with low SNR.         

- `subsampleshare_columns`  [1.0] column subsampling (aka feature subsampling) by level.

- `min_unique`              [:default] sharp splits are imposed on features with less than min_unique values (default is 5 for modality=:compromise or :accurate, else 10)

- `mixed_dc_sharp`          [false] true to force sharp splits on discrete and mixed discrete-continuous features (defined as having over 20% obs on a single value)

- `delete_missing`          [false] true to delete rows with missing values in any feature, false to handle missing internally (recommended).

- `theta`                   [1]  numbers larger than 1 imply tighter penalization on β (final leaf values) compared to default.

- `meanlntau`               [1.0] prior mean of log(τ). Set to higher numbers to suggest less smooth functions.        

- `mugridpoints`       [11] number of points at which to evaluate μ during variable selection. 5 is sufficient on simulated data with normal or uniform distributions, but actual data may benefit from more (due to with highly non-Gaussian features).
                            For extremely complex and nonlinear features, more than 10 may be needed.        

- `force_sharp_splits`      [] optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a sharp split.

- `force_smooth_splits`     [] optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a smooth split (high values of τ not allowed).

- `cat_representation_dimension`  [4 (2 for classification)] 1 for mean encoding, 2 adds frequency, 3 adds variance, 4 adds robust skewness, 5 adds robust kurtosis

- `losscv`                  [:default] loss function for cross-validation (:mse,:mae,:logistic,:sign). 

- `n_refineOptim`      [10^6] MAXIMUM number of observations to use fit μ and τ (split point and smoothness).
                            Lower numbers can provide speed-ups with very large n at some cost in terms of fit.

- `loglikdivide`         [1.0] Higher numbers increase the strength or all priors. The defaults sets it internally using HTBloglikdivide(),
                            when it detects a dates series in HTBdata().

- `tau_threshold`         [10.0] lowest threshold for imposing sharp splits. Lower numbers give more sharp splits.

- `multiplier_stdtau`    [5.0] The default priors suggest smoother splits on features whose unconditional distribution (appropriately transformed according to the link function) is closer to the unconditional distribution of *y* or, when not applicable, to a Gaussian. To disengage this feature, set *multiplier_stdtau* = 0

# Additional parameters to control the cross-validation process can be set in HTBfit(), but keeping the defaults is generally encouraged.

# Example
    param = HTBparam()
# Example
    param = HTBparam(nfold=1,nofullsample=true)
"""
function HTBparam(;

    T = Float32,   # Float32 or Float64. Float32 is up to twice as fast for sufficiently large n (due not to faster computations, but faster copying and storing)
    I = typeof(1),   
    loss = :L2,               # loss function
    losscv = :default,        # loss used for early stopping and stacking. :default, or :mse, :mae, :Huber, :logistic, :sign
    modality = :compromise,     # :accurate, :fast, :compromise 
    coeff = coeff_user(Symbol(loss),T),      # coefficients (if any) used in loss
    coeff_updated = Vector{Vector{T}}(undef,2), # first vector for coeff used in loglik, second for coeff in losscv (where coefficients should be constant)
    verbose = :Off,      # level of verbosity, :On, :Off
    warnings=:On,
    num_warnings=0,
    randomizecv = false,        # true to scramble data for cv, false for contiguous blocks ('Block CV')
    nfold       = 4,            # n in n-fold cv. 1 for a single validation set (early stopping in the case of ntrees)
    nofullsample = false,        # true to skip the full sample fit after validation (only relevant if nfold=1)
    sharevalidation = 0.30,      #[0.30] Size of the validation set (integer), last sharevalidation rows of data. Or float, share of validation set. Relevant only if nfold = 1.
    indtrain_a = Vector{Vector{I}}(undef,0), # for user's provided Vector{Vector} of indices of train data ..
    indtest_a = Vector{Vector{I}}(undef,0),  # .. and test data. This over-writes nfold. 
    stderulestop = 0.01,         # e.g. 0.01. Applies to stopping while adding trees to the ensemble. larger numbers give smaller ensembles.
    lambda = 0.10,
    # Tree structure and priors
    depth  = 5,        # 3 allows 2nd degree interaction and is fast. 4 takes almost twice as much per tree on average. 5 can be 8-10 times slower per tree. However, fewer deeper trees are required, so the actual increase in computing costs is smaller.
    depth1 = 10,
    depthppr = 2,      # projection pursuit depth. 0 to disactive. 
    ppr_in_vs = :Off,    # :On for projection pursuit included in variable selection phase. Can occasionally go wrong. :Off is a safer default, and :On could be used for stacking. 
    sigmoid = :sigmoidsqrt, # In order from more to less smooth: :smoothsigmoid, :sigmoidsqrt, :sigmoidlogistic, :smoothstep      
    meanlntau= 1.0,    # Assume a Gaussian for log(tau).
    varlntau = 0.5^2,  # [0.5^2]. Set to Inf to disactivate (log(p(τ)=0)).  See loss.jl/multiplier_stdlogtau_y().  This is the dispersion of the student-t distribution (not the variance unless dof is high).
    doflntau = 5.0,
    multiplier_stdtau = 5.0,
    d_meanlntau_cat = 1.0,  # difference in intercept of meanlntau for categorical features (prior that they are less smooth)
    varmu   = Inf,    # default Inf to disactivate (then lnpμ sets p(μ)=1.) Otherwise 1-3. Smaller number make it increasingly unlikely to have nonlinear behavior in the tails. DISPERSION, not variance
    dofmu   = 10.0,
    meanlntau_ppr = log(0.2),  # for projection pursuit regression. Center on quasi-linearity. log(0.2) ≈ -1.6.  
    varlntau_ppr = 1^2,        # one-sided 
    doflntau_ppr = 5,        
    priortype = :hybrid,
    max_tau_smooth = 20,         # maximum value of tau allowed when :smooth. 10 can still capture very steep functions in boosting
    min_unique  = :Auto,         # Note: over-writes force_sharp_splits unless set to a large number. minimum number of unique values to consider a feature as continuous
    mixed_dc_sharp = false,
    tau_threshold = 10,         # threshold for imposing sharp splits
    half_life_depth = 100_000,   # half-life of max_depth expressed in number of trees. 100_000 to disactivate.  
    min_depth_ratio = 0.5,       # minimum depth is ceil(min_depth_ratio*depth)
    force_sharp_splits = Vector{Bool}(undef,0),  # typically hidden to user: optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a sharp split.
    force_smooth_splits = Vector{Bool}(undef,0),  # typically hidden to user: optionally, a p vector of Bool, with j-th value set to true if the j-th feature is forced to enter with a smooth split (high values of λ not allowed)
    exclude_features = Vector{Bool}(undef,0),    # typically hidden to user optionally, a p vector of Bool, with j-th value set to true if the j-th feature should not be considered as a candidate for a split
    augment_mugrid  = Vector{Vector{T}}(undef,0),
    cat_features = Vector{I}(undef,0),        # p vector of Bool, j-th value set to true for a  categorical feature. If empty, strings are interpreted as categorical   
    cat_features_extended = Vector{I}(undef,0),
    cat_dictionary=Vector{Dict}(undef,0),       # global variable to store the dictionary mapping each category to a number
    cat_values = Vector{NamedTuple}(undef,0),
    cat_globalstats = (mean_y=T(0),var_y=T(0),q_y=T(0),n_0=T(0),mad_y=T(0),skew_y=T(0),kurt_y=T(0)),  # global statistics for categorical features
    cat_representation_dimension = 4,           # dimension of the representation of categorical features: mean,frequency,std,skew,kurt (robust measures)
    n0_cat = 1,                                  # leave at 1! See preliminary_cv for why (multiplier). automatically cv prior on number of observations for categorical data
                                                 # cv tries higher values but not lower than n0_cat.
    mean_encoding_penalization = 1.0,
    cv_categoricals = :default,            
    class_values = Vector{T}(undef,0),
    delete_missing = false,  
    mask_missing = false,                       # If true, adds a mask (dummy) for missing values. Does not seem required in any instance, and may worsen performance, although it may occasionally improve performance in small samples.
    missing_features = Vector{I}(undef,0), 
    info_date = (date_column=0,date_first=0,date_last=0),
    sparsity_penalization = 0.3,
    same_feature_penalization = 0.0,  # penalization multiplies the number of times the feature appears in different splits of the same tree. Think of it as log(prob_split_different feature/prob_split_same_feature), e.g. [0 2]
    same_feature_penalization_start = 0, # the first same_feature_penalization_start appearances are not penalized.
    p0       = :Auto,
    sharevs  = 1.0,              # if <1, adds noise to vs, in vs phase takes a random subset. :Auto is 0.5 if n>=250k. ? Speed gains are surprisingly small, maybe because of the need to slices large matrices ?      
    refine_obs_from_vs = false,  # true to add randomization to (μ,τ), assuming sharevs<1
    finalβ_obs_from_vs  = false,  # true to add randomization to final β
    n_refineOptim = 10_000_000,   # Subsample size for refineOptim. beta is always computed on the full sample.
    subsampleshare_columns = 1.0,  # if <1.0, only a random share of features is used at each split (re-drawn at each split)
    sparsevs = :On,           # 
    frequency_update = 1.0,       # when sparsevs, 1 for Fibonacci, 2 to update at 2*Fibonacci etc...               
    number_best_features = 10,    # number of best feature in each node (level) to store into best_features    
    best_features = Vector{I}(undef,0),
    # grid and optimization parameters
    mugridpoints = 11,  # points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data can benefit from more (due to with highly non-Gaussian features. 11 so that at d>=depth_coarse_grid, it takes [2,4,6,8,10] so symmetric.
    taugridpoints = 2,  # points at which to evaluate τ during variable selection. 1-5 are supported. If less than 3, refinement is then on a grid with more points
    depth_coarse_grid = 5, # at this depth and higher, vs uses only τ=5 and every other second point in mugrid. 
    depth_coarse_grid2 = 8, # at this depth and higher, vs uses only τ=5 and every third second point in mugrid. 
    depth_coarse_grid3 = 10, # at this depth and higher, vs uses only τ=5 and only one point in mugrid. 
    xtolOptim = 0.01,  # tolerance in the optimization e.g. 0.01 (measured in dμ). It is automatically reduced if tau is large 
    method_refineOptim = :distributed, #  :pmap, :distributed 
    points_refineOptim = 12,    # number of values of tau for refineOptim. Default 12. Other values allowed are 4,7.
    # miscel                    # becomes 7 once coarse_grid kicks in, unless ncores>12.
    ntrees = 3000,    # number of trees. 1000 is CatBoost default (with maxdepth = 6). May need more if trees are shallower or can become shallower due to penalizations.
    theta = 1.0,   # numbers larger than 1 imply tighter penalization on β compared to default.                     
    loglikdivide = :Auto,   # the log-likelhood is divided by this scalar. Used to improve inference when observations are correlated.
    overlap = 0,
    multiply_pb = 1.0,
    varGb   = NaN,      # Relevant only for first tree
    seed_datacv = 1,       # sets random seed if randomizecv=true
    seed_subsampling = 1,  # sets random seed used on subsampling iterations (will be seed=seed_subsampling + iter)
    iter = 0,
    # Newton optimization: good default is one step in preliminary phase, and evaluate actual log-lik, and iterate to convergence for final β
    newton_gauss_approx = :Auto, # true has large speed gains for logistic for large n (loss=...exp.()). If true, and if newton_max_steps=1 (hence not in final) evaluates the Gaussian approximation to the log-likelihood rather than the likelihood itself except in final phase (given i,mu,tau)
    newton_max_steps = 1,         # vs phase. 1 seems sufficient, in combination with gaussian_approx=false, to get most of the gains  
    newton_max_steps_final = 20,  # small impact on cost and large gains.
    newton_tol = 1,        # Keep large (e.g. 1.0) to avoid unnecessay iterations in preliminary phase
    newton_tol_final= 0.01,
    newton_max_steps_refineOptim=1, # number of steps in refineOptim
    linesearch=true                # true to perform line search after final estimates (α*β)
    )

    if typeof(T)!==DataType; if T=="Float32"; T=Float32; else; T = Float64; end; end;  # for R users
           
    if loss==:L2    
        newton_max_steps,newton_max_steps_final,newton_gauss_approx = 1,1,false 
    end          

    if loss==:Huber && warnings==:On 
        @warn "loss = :t is recommended instead of loss = :Huber. The t loss automatically estimates the degres of freedom and typically converges faster. "
    end     

    @assert(doflntau>T(2), " doflntau must be greater than 2.0 (for variance to be defined) ")
    @assert(T(1e-20) < xtolOptim, "xtolOptim must be positive ")

    if depth>8 && warnings==:On
        @warn "Setting param.depth higher than 7-8 may result in very high computing costs and potential numerical instability (ill-defined matrix inverse), which the algorithm should be able to handle but with a further increase in computing time. "
    end

    ncores = nprocs()-1
    

    # The following works even if sharevalidation is a Float which is meant as an integer (e.g. in R wrapper)
    if T(sharevalidation)==T(round(sharevalidation))  # integer
        sharevalidation=I(sharevalidation)
    else
        sharevalidation = T(sharevalidation)
    end

    if !isa(cat_features,AbstractVector)  # ensure cat_features is a vector 
        cat_features = [cat_features]
    end 
    
    if eltype(cat_features) <: Real 
        cat_features = I.(cat_features)
    end

    sharevs==:Auto ? nothing : sharevs=T(sharevs)
    loglikdivide==:Auto ? nothing : loglikdivide=T(loglikdivide)
    p0==:Auto ? nothing : p0=I(p0)   # if :Auto, set in param_given_data!()

    if cv_categoricals==:default
        if modality in [:fastest,:fast]
            cv_categoricals=:none
        elseif modality == :compromise
            cv_categoricals=:penalty
        else     
            cv_categoricals=:both
        end 
    end     

    if min_unique==:Auto
        modality in [:fast,:fastest] ? min_unique = 10 : min_unique = 5
    end     

    # Functions on indtrain_a and indtest_a, if user provides them 
    if length(indtrain_a)>length(indtest_a)
        @error "There are $(length(indtrain_a)) vectors in indtrain_a, but $(length(indtest_a)) in indtest_a"
    end 

    nfold_user = length(indtrain_a)

    if nfold_user>0           # over-write randomizecv and nfold
        nfold = nfold_user
        randomizecv = false
        for i in 1:nfold_user  # ensure indices are integers (for R, Python etc..)
            indtrain_a[i] = I.(round.(indtrain_a[i]))
            indtest_a[i]  = I.(round.(indtest_a[i]))
        end
    end 
    
    param = HTBparam(T,I,Symbol(loss),Symbol(losscv),Symbol(modality),T.(coeff),coeff_updated,Symbol(verbose),Symbol(warnings),I(num_warnings),randomizecv,I(nfold),nofullsample,sharevalidation,indtrain_a,indtest_a,T(stderulestop),T(lambda),I(depth),I(depth1),I(depthppr),
        Symbol(ppr_in_vs),Symbol(sigmoid),
        T(meanlntau),T(varlntau),T(doflntau),T(multiplier_stdtau),T(d_meanlntau_cat),T(varmu),T(dofmu),
        T(meanlntau_ppr),T(varlntau_ppr),T(doflntau_ppr),Symbol(priortype),T(max_tau_smooth),I(min_unique),mixed_dc_sharp,T(tau_threshold),I(half_life_depth),
        T(min_depth_ratio),force_sharp_splits,force_smooth_splits,exclude_features,augment_mugrid,cat_features,cat_features_extended,cat_dictionary,cat_values,cat_globalstats,I(cat_representation_dimension),T(n0_cat),T(mean_encoding_penalization),
        Symbol(cv_categoricals),
        class_values,Bool(delete_missing),mask_missing,missing_features,info_date,T(sparsity_penalization),T(same_feature_penalization),
        I(same_feature_penalization_start),p0,sharevs,refine_obs_from_vs,finalβ_obs_from_vs,
        I(n_refineOptim),T(subsampleshare_columns),Symbol(sparsevs),T(frequency_update),
        I(number_best_features),best_features,I(mugridpoints),I(taugridpoints),
        I(depth_coarse_grid),I(depth_coarse_grid2),I(depth_coarse_grid3),T(xtolOptim),Symbol(method_refineOptim),
        I(points_refineOptim),I(ntrees),T(theta),loglikdivide,I(overlap),T(multiply_pb),T(varGb),I(ncores),I(seed_datacv),I(seed_subsampling),I(iter),newton_gauss_approx,
        I(newton_max_steps),I(newton_max_steps_final),T(newton_tol),T(newton_tol_final),I(newton_max_steps_refineOptim),Bool(linesearch) )

    param_constraints!(param) # enforces constraints across options. Must be repeated in HTBfit.

    return param
end


# Sets some parameters that require knowledge of data. Notice that data.x (but no data.y) may contain NaN. 
function param_given_data!(param::HTBparam,data::HTBdata)

    n,p = size(data.x)
    I = typeof(param.nfold)
    T = typeof(param.lambda)

    # compute loglikdivide unless user provided it 
    if param.loglikdivide == :Auto
        lld,ess = HTBloglikdivide(data.y,data.dates,overlap=param.overlap)
        param.loglikdivide = T(lld)
    end

    if param.p0==:Auto 
        param.p0=p 
    end

    #:Auto switches it :On if sparsity_penalization >=0, else :Off. Not active as computing times with large p can be very large without sparsevs.  
    #if param.sparsevs==:Auto
    #    param.sparsity_penalization>0 ? param.sparsevs=:On : param.sparsevs=:Off
    #end 

    if param.sharevs==:Auto
        n = 0.75*n        # assuming n i train+sample
        α = T(min(1,sqrt(50_000/n)))
        α>0.75 ? α=T(1) : nothing
        param.sharevs = α
    end 
    
    if param.newton_gauss_approx == :Auto
        
        param.newton_gauss_approx = false
        loss = param.loss

        if loss in [:gamma,:gammaPoisson,:Poisson,:L2loglink]    # large speed gains  
            param.newton_gauss_approx = true
        end 

        if loss == :logistic && size(data.x,2)>20_000 && param.modality in [:fast,:fastest]
            param.newton_gauss_approx = true
        end     

    end  

    # change aliases for loss functions to default denomination
    loss = param.loss 
    if loss in [:l2,:mse,:Gaussian,:normal] 
        param.loss = :L2
    elseif loss in [:logL2, :logl2]
        param.loss = :lognormal
    elseif loss in [:binary]
        param.loss = :logistic
    elseif loss in [:student]
        param.loss = :t
    elseif loss in [:huber]
        param.loss = :Huber
    elseif loss in [:gamma_Poisson,:gamma_poisson,:negbin,:negative_binomial]
        param.loss = :gammaPoisson
    elseif loss in [:l2loglink]
        param.loss = :L2loglink                         
    end     

end         



# separate function enforces constraints across options. (Some warnings and constrains may be duplicated from HTBparam() because HTBfit() modified param.)
function param_constraints!(param::HTBparam)

    if param.depth>8 && param.warnings==:On  
        @warn "Setting param.depth higher than 7-8 may result in very high computing costs and potential numerical instability (ill-defined matrix inverse), which the algorithm should be able to handle but with a further increase in computing time. "
    end

    # no ppr if depth=1? Probably worse performance without ppr, but half the computing cost. 
    #param.depth==1 ? param.depthppr=param.I(0) : nothing 
    
    if param.meanlntau==Inf; param.priortype==:sharp; end;

    if param.priortype==:smooth
        param.doflntau=100
    end

    if param.priortype==:disperse
        param.varlntau=Inf
    end

    if param.depth_coarse_grid2 < param.depth_coarse_grid
        @error "param.depth_coarse_grid2 must be greater than or equal to param.depth_coarse_grid. Setting it equal"
        param.depth_coarse_grid2 = param.depth_coarse_grid
    end 

    if param.depth_coarse_grid3 < param.depth_coarse_grid2
        @error "param.depth_coarse_grid3 must be greater than or equal to param.depth_coarse_grid2. Setting it equal."
        param.depth_coarse_grid3 = param.depth_coarse_grid2
    end 

    # no sense taking variance, skew and kurtosis of y when y is binary: the mean has the same info
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
        HTBdata(y,x,param,[dates];weights=[],fnames=[],offset=[])
Collects and pre-processes data in preparation for fitting HTBoost

# Inputs

- `y::Vector`              Vector of responses. Can be a vector of lables, or a dataframe with one column. 
- `x`                      Matrix of features. Can be a vector or matrix of floats, or a dataframe. Converted internally to a Matrix{T}, T as defined in HTBparam
- `param::HTBparam`


# Optional Inputs

- `dates::AbstractVector`  [1:n] Typically Vector{Date} or Vector{Int}. Used in cross-validation to determine splits.
                            If not supplied, the default 1:n assumes a cross-section of independent realizations (conditional on x) or a single time series.
- 'weights'                [ones(T,n)] vector of floats or Floats, weights for weighted likelhood
- `fnames::Vector{String}` [x1, x2, ... ] feature names
- 'offset'                 vector of offsets (exposure), in logs if the loss adopts a loss link (:gamma,:gammaPoisson,:L2loglink,....) 

# Examples of use
    data = HTBdata(y,x,param)
    data = HTBdata(y,x,param,dates,fnames=names)
    data = HTBdata(y,df[:,[:CAPE, :momentum ]],param,df.dates,fnames=df.names)
    data = HTBdata(y,df[:,3:end],param)

# Notes
- When dates are provided, the data will be ordered chronologically (for cross-validation functions) unless the user has provided explicit training and validation sets.
"""
function HTBdata(y0::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},x::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},
    param::HTBparam,dates::AbstractVector=[];weights::AbstractVector=[],fnames = Vector{String}[],offset=[])  

    T    = param.T
    n    = size(y0,1)

    if typeof(y0)<:AbstractDataFrame || eltype(y0) <: Union{Bool,Number} || typeof(y0)<:AbstractMatrix
    elseif eltype(y0)==String && param.loss==:multiclass  # accept strings for multiclass 
    else 
        @error "in HTBdata, y must be of type Number or Bool (true,false) or DataFrame"
    end      

    y = y0[:,1]    
    miny = minimum(y)

    if param.warnings == :On 
        if param.loss in [:L2,:Huber,:t] && miny>0
            @info "Since minimum(y)>0, consider using a loss function such as L2loglink,gamma,or lognormal." 
            if param.loss in [:L2,:Huber,:t] && miny>0
                @info "Since minimum(y)>0, consider using a loss function such as :L2loglink,:gamma,or :lognormal." 
            elseif param.loss in [:L2,:Huber,:t] && miny==0
                @info "Since minimum(y)=0, consider using a loss function such as :L2loglink,:hurdleGamma,:hurdleL2loglink,:hurdleL2,:hurldelognormal" 
            end 
        end 
    end

    y = store_class_values!(param,y)   # stores class values in param, and transforms y in T, leaving missing as missing 

    check_admissible_data(y,param)  # check if data is admissible given param (typically param.loss)

    if param.loss in [:lognormal]
        @. y = log(y)
    end             

    if isempty(weights)
        weights=ones(T,length(y))
    else    
        @assert(minimum(weights)>0.0, " weights in HTBdata() must be strictly positive ")
    end    

    fnames1 = copy(fnames)

    if isempty(fnames1) && typeof(x)<:AbstractDataFrame
        fnames1 = names(x)
    elseif !isempty(fnames1) && typeof(x)<:AbstractDataFrame
        rename!(x, fnames1)
    elseif isempty(fnames1)
        fnames1 = ["x$i" for i in 1:size(x,2)]    
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
            @warn " Dates are not in chronological order, but the user has provided training and validation sets, so data will not be sorted by date."
        end
  
    end

    # pre-process data

    # if x is Float, transform to dataframe for convenient data manipulation
    if typeof(x) <: AbstractDataFrame
        xp = deepcopy(x)    
    elseif typeof(x) <: AbstractVector    
        xp = DataFrame(hcat(x),Symbol.(fnames1))
    else 
        xp = DataFrame(x,Symbol.(fnames1))    
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
        xp,fnames1 = missing_features_extend_x(param,xp)       #  adds columns to xp 
    end

    map_cat_convert_to_float!(xp,param,create_dictionary=true)  # map cat to Dictionary, converts to float 0,1,....
    extended_categorical_features!(param,fnames1)   # finds categorical features needing an extensive (more than one column) representation
    xp = replace_missing_with_nan(xp)   # SharedArray do not accept missing.

    if !isempty(param.cat_features) && !isempty(offset) && param.warnings==:On
        @warn "Categorical features with more than two categories are not currently handled correctly (by the mean targeting transformation)
        with offsets. The program will run but categorical information will be used sub-optimally, particularly if the
        average offset differs across categories. If categorical features are important, it may be better
        to omit the offset from HTBdata(), and instead model y/offset with a :L2loglink loss instead of a :gamma, :Poisson or :gammaPoisson."
    end 

    isempty(offset) ? offset = zeros(T,n) : offset = T.(offset)

    data = HTBdata(T.(y),convert_df_matrix(xp,T),T.(weights),dates,fnames1,param.cat_features,offset)

    return data

end


# HTBdata() should be invoked only once, by the user. HTBdata_sharedarray to be used everywhere else.
# It creates an instance of HTBdata where x is a SharedArray.
# I don't think there gains from making y and weights SharedVector. Only x.
# SharedArray can cause a "mmap: Access is denied" error, presumably on systems shared by several users
function HTBdata_sharedarray(y::Union{AbstractVector,AbstractDataFrame},x::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},
   param::HTBparam,dates::AbstractVector,weights::AbstractVector,fnames::Vector{String},offset::AbstractVector)
    
   data = HTBdata(y,SharedMatrixErrorRobust(x,param),weights,dates,fnames,param.cat_features,offset)

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


# Possibly varying lambda.
# λᵢ = effective_lambda(param,iter)
function effective_lambda(param::HTBparam,iter::Int)

    T = param.T 
    lambda = param.lambda

    return T(min(1,lambda))
end     
