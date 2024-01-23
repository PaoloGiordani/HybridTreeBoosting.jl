#
#  Collects most functions that are exported
#
#  INFORMATION about available functions   
#  SMARTinfo               basic information about the main functions in SMARTboost
# 
#  SETTING UP THE MODEL, and CALIBRATING PRIOR for panel data and/or overlapping data
#  SMARTloglikdivide       calibrates param.loglikdivide for panel data and/or overlapping data
#  SMARTindexes_from_dates train and test sets indexes for expanding window cross-validation
#  SMARTparam              parameters, defaults or user-provided
#
#  FITTING and FORECASTING
#  SMARTfit                fits SMARTboost with cv (or validation/early stopping) of number of trees and, optionally, depth or other parameter
#  SMARTbst                fits SMART when nothing needs to be cross-validated (not even number of trees)
#  SMARTpredict            prediction from SMARTtrees::SMARTboostTrees
#    preparedata_predict
#
#  POST-ESTIMATION ANALYSIS
#  SMARTcoeff              provides information on constant coefficients
#  SMARTrelevance          computes feature importance (Breiman et al 1984 relevance)
#  SMARTpartialplot        partial dependence plots (keeping all other features fixed, not integrating out)
#  SMARTmarginaleffect     provisional! Numerical computation of marginal effects.
#  SMARToutput             collects fitted parameters in matrices
#  tight_sparsevs          warns if sparsevs seems to compromise fit 
#  SMARTweightedtau        computes weighted smoothing parameter

"
    SMARTinfo()
Basic information about the main functions in SMARTboost (see help on each function for more details)


# Setting up the model 
- `SMARTindexes_from_dates` builds train and test sets indexes for expanding window cross-validation
- `SMARTparam`           parameters, defaults or user-provided.
- `SMARTdata`            y,x and, optionally, dates, weights, and names of features

# Fitting and forecasting 
- `SMARTfit`             fits SMARTboost with cv (or validation/early stopping) of number of trees and, optionally, depth or other parameter
- `SMARTbst`             (rerely needed by user) fits SMARTboost when nothing needs to be cross-validated (not even number of trees)                         
- `SMARTpredict`         predictions for y or natural parameter

#  POST-ESTIMATION ANALYSIS
- `SMARTcoeff`           provides information on constant coefficients, e.g. dispersion and dof for loss=:t
- `SMARTrelevance`       computes feature importance (Breiman et al 1984 relevance)
- `SMARTpartialplot`     partial dependence plots (keeping all other features fixed, not integrating out)
- `SMARTmarginaleffect`  provisional! Numerical computation of marginal effects.
- `SMARToutput`          collects fitted parameters in matrices
- `SMARTweightedtau`     computes weighted smoothing parameter to help assess function smoothness

Example of use of info function 

    help?> SMARTinfo
    or ...
    julia> SMARTinfo()

To find more information about a specific function, e.g. SMARTfit

    help?> SMARTfit

Example of basic use of SMARTboost functions with iid data and default settings

    param  = SMARTparam()     # or param = SMARTparam(loss=:logistic)                            
    data   = SMARTdata(y,x,param)
    output = SMARTfit(data,param)
    yf     = SMARTpredict(x_test,output)  

See the files in the folder examples for more examples of use. 
"
function SMARTinfo()

    println("Documentation: $(Base.doc(SMARTinfo))")

end



#=
    SMARTloglikdivide(df,y_symbol,date_symbol;overlap=0)

loglikdivide is now computed internally, so the user does not need to call this function for lld,
only if interested in effective_sample_size.

Old documentation:
Suggests a value for param.loglikdivide., where nominal sample size/loglikedivide = effective sample size.
Relevant panel data (longitudinal data) and some time series. 
The only effect of loglikdivide in SMARTboost is to calibrate the strength of the prior in relation to the likelihood evidence.
Accounts (roughly) for cross-sectional correlation using a clustered standard errors approach, and for serial correlation induced
by overlapping observation when y(t) = Y(t+horizon) - Y(t).

# Inputs
- `df::DataFrame`        dataframe including y and dates
- `y_symbol::Symbol`     symbol or string of the dependent variable in the dataframe
- `date_symbol::Symbol`  symbol or string of date column in the dataframe
- `overlap::Int`         (keyword) [0] e.g. if y(t) = Y(t+horizon) - Y(t), then overlap = horizon - 1. 

# Output
- `loglikdivide::Float`   
- `effective_sample_size::Float`

# Example of use
    lld,ess =  SMARTloglikdivide(df,:excessret,:date,overlap=h-1)

=#
function SMARTloglikdivide(df::DataFrame,y_symbol,date_symbol;overlap = 0)

    overlap = Int(overlap); y_symbol = Symbol(y_symbol); date_symbol = Symbol(date_symbol)  # required by R wrapper

    dates     = unique(df.date)
    y         = df[:,y_symbol] .- mean(df[:,y_symbol])
    ssc       = 0.0

    for date in dates
        ssc   = ssc + (sum(y[df.date.==date]))^2
    end

    loglikdivide  = ssc/sum(y.^2)   # roughly accounts for cross-correlation as in clustered standard errors.

    if loglikdivide<0.9
      @warn "loglikdivide is calculated to be $loglikdivide (excluding any overlap). Numbers smaller than one imply negative cross-correlation, perhaps induced by output transformation (e.g. from y to rank(y)).
      loglikvidide WILL BE SET TO 1.0 by default. If the negative cross-correlation is genuine, the original value of $loglikdivide can be used, which would imply weaker priors."
      loglikdivide = 1.0
    end

    loglikdivide  = loglikdivide*( 1 + overlap/2 ) # roughly accounts for auto-correlation induced by overlapping, e.g. y(t) = p(t+h) - p(t)
    effective_sample_size = length(y)/loglikdivide

   return loglikdivide,effective_sample_size

end



function SMARTloglikdivide(y::AbstractVector{T},dates_all;overlap=0) where T<:Real

    overlap = Int(overlap)
    dates   = unique(dates_all)
    y      = y .- mean(y)
    ssc    = 0.0

    for date in dates
        ssc   = ssc + (sum(y[dates_all.==date]))^2
    end

    loglikdivide  = ssc/sum(y.^2)   # roughly accounts for cross-correlation as in clustered standard errors.

    if loglikdivide<0.9
      @warn "loglikdivide is calculated to be $loglikdivide (excluding any overlap). Numbers smaller than one imply negative cross-correlation, perhaps induced by output transformation (e.g. from y to rank(y)).
      loglikvidide WILL BE SET TO 1.0 by default. If the negative cross-correlation is genuine, the original value of $loglikdivide can be used, which would imply weaker priors."
      loglikdivide = 1.0
    end

    loglikdivide  = loglikdivide*( 1 + overlap/2 ) # roughly accounts for auto-correlation induced by overlapping, e.g. y(t) = p(t+h) - p(t)
    effective_sample_size = length(y)/loglikdivide

   return loglikdivide,effective_sample_size

end



"""
SMARTindexes_from_dates(df::DataFrame,datesymbol::Symbol,first_date::Date,n_reestimate::Int)

Computes indexes of training set and test set for cumulative CV and pseudo-real-time forecasting exercises

* INPUTS

- datesymbol            symbol name of the date
- first_date            when the first training set ENDS (end date of the first training set)
- n_reestimate          every how many periods to re-estimate (update the training set)

* OUTPUT
  indtrain_a,indtest_a are arrays of arrays of indexes of train and test samples

* Example of use

- first_date = Date("2017-12-31", Dates.DateFormat("y-m-d"))
- indtrain_a,indtest_a = SMARTindexes_from_dates(df,:date,first_date,12)

* NOTES

- Inefficient for large datasets

"""
function SMARTindexes_from_dates(df::DataFrame,datesymbol::Symbol,first_date,n_reestimate)

    n_reestimate = Int(n_reestimate)

    indtrain_a = Vector{Int}[]
    indtest_a  = Vector{Int}[]

    dates   = df[:,datesymbol]
    datesu  = unique(dates)
    date1   = first_date    # end date of training set

    N       = length(datesu)
    i       = length(datesu[datesu.<=first_date])
    date2   = datesu[i+n_reestimate]   # end date of test set
    finish  = false

    while finish == false

        df_train   = df[df[:,datesymbol].<= date1,:]
        indtrain   = Vector{Int}[collect(1:size(df_train,1))]
        df_test    = df[(df[:,datesymbol].> date1).*(df[:,datesymbol].<= date2),:]
        indtest    = Vector{Int}[indtrain[end][end] .+ collect(1:size(df_test,1))]

        i      = i+n_reestimate

        if i >= N
            finish = true
        else
            date1  = datesu[i]
            date2  = datesu[minimum(hcat(N,i+n_reestimate))]
            indtrain_a = append!(indtrain_a,indtrain)
            indtest_a  = append!(indtest_a,indtest)
        end

    end

    return indtrain_a,indtest_a

end





"""
    SMARTbst(data::SMARTdata, param::SMARTparam)
SMARTboost fit, number of trees defined by param.ntrees, not cross-validated.

# Output
- `SMARTtrees::SMARTboostTrees`

# Example of use
    SMARTtrees =  SMARTbst(data,param)
"""
function SMARTbst(data0::SMARTdata, param::SMARTparam )

    # initialize SMARTtrees
    param,data,meanx,stdx          = preparedataSMART(data0,param)
    τgrid,μgrid,Info_x,n,p         = preparegridsSMART(data,param,meanx,stdx)

    τgrid,μgrid,Info_x,n,p         = preparegridsSMART(data,param,meanx,stdx)

    gamma0                         = initialize_gamma0(data,param)
    gammafit                       = fill(gamma0,n)

    param          = updatecoeff(param,data.y,gammafit,data.weights,0)
    SMARTtrees     = SMARTboostTrees(param,gamma0,n,p,meanx,stdx,Info_x)
    rh,param       = gradient_hessian( data.y,data.weights,gammafit,param,0)

    # prelimiminary run to calibrate coefficients and priors
    Gβ,trash  = fit_one_tree(data.y,data.weights,SMARTtrees,rh.r,rh.h,data.x,μgrid,Info_x,τgrid,param)
    param = updatecoeff(param,data.y,SMARTtrees.gammafit+Gβ,data.weights,0) # +Gβ, NOT +λGβ
    trash,param = gradient_hessian( data.y,data.weights,SMARTtrees.gammafit+Gβ,param,1)

    for iter in 1:param.ntrees
        displayinfo(param.verbose,iter)
        Gβ,i,μ,τ,m,β,fi2  = fit_one_tree(data.y,data.weights,SMARTtrees,rh.r,rh.h,data.x,μgrid,Info_x,τgrid,param)
        param = updatecoeff(param,data.y,SMARTtrees.gammafit+Gβ,data.weights,iter) # +Gβ, NOT +λGβ
        updateSMARTtrees!(SMARTtrees,Gβ,SMARTtree(i,μ,τ,m,β,fi2),iter,param)          # updates gammafit=gammafit_old+λGβ
        rh,param = gradient_hessian( data.y,data.weights,SMARTtrees.gammafit,param,2)
    end

    # bias adjustment
    bias,SMARTtrees.gammafit = bias_correct(SMARTtrees.gammafit,data.y,SMARTtrees.gammafit,param)
    SMARTtrees.gamma0 +=  bias

    return SMARTtrees

end



function displayinfo(verbose::Symbol,iter::Int,meanloss_iter,stdeloss_iter)

    if verbose == :On
        println("Tree number ", iter, "  mean and standard error of validation loss ", [meanloss_iter, stdeloss_iter])
    end

end


function displayinfo(verbose::Symbol,iter::Int)
    if verbose == :On
        println("Tree number ", iter )
    end
end


#=
    SMARTpredict_internal(x,SMARTtrees::SMARTboostTrees)
Forecasts from SMARTboost. Expects x to be standardized, with categorical already transformed

# Inputs
- `x`                           (n,p) matrix of forecast origins (type<:real) or p vector of forecast origin
                                 Assumes x is standardized, and all categorical have been transformed. 

- `SMARTtrees::SMARTboostTrees` from previously fitted SMARTbst or SMARTfit

# Optional inputs
- `cutoff_paralellel`          [20_000] if x has more than these rows, a parallellized algorithm is called (which is slower for few forecasts)

# Output
- `yfit`                        (n) vector of forecasts of y (or, outside regression, of the natural parameter), or scalar forecast if n = 1

# Example of use
    output = SMARTfit(data,param)
    yf     = SMARTpredict(x_oos,output.SMARTtrees)
=#
function SMARTpredict_internal(x::AbstractMatrix,SMARTtrees::SMARTboostTrees,predict;cutoff_parallel=20_000)

    n,p = size(x)

    if p != length(SMARTtrees.meanx)
        @error "In SMARTpredict, the input matrix x has column dimension $p while SMARTfit has been estimated with $(length(SMARTtrees.meanx)) features in data.x"
    end

    if n>cutoff_parallel
        gammafit = SMARTpredict_distributed(x,SMARTtrees)
    else

        T       = typeof(SMARTtrees.gamma0)
        gammafit = SMARTtrees.gamma0*ones(T,size(x,1))
    
        for j in 1:length(SMARTtrees.trees)
            tree     =  SMARTtrees.trees[j]          
            gammafit += SMARTtrees.param.lambda*SMARTtreebuild(x,tree.i,tree.μ,tree.τ,tree.m,tree.β,SMARTtrees.param)    
        end
    end

    pred = from_gamma_to_Ey(gammafit,SMARTtrees.param,predict) # # from natural parameter to E(y), depending on predict

    return pred

end


# Expects x to be standardized, with categories and missing already transformed
function SMARTpredict_distributed(x::AbstractMatrix,SMARTtrees::SMARTboostTrees)

    T       = typeof(SMARTtrees.gamma0)
    x       = SharedMatrixErrorRobust(x,SMARTtrees.param)

    gammafit = @distributed (+) for j = 1:length(SMARTtrees.trees)
        SMARTtrees.param.lambda*SMARTtreebuild(x,SMARTtrees.trees[j].i,SMARTtrees.trees[j].μ,SMARTtrees.trees[j].τ,SMARTtrees.trees[j].m,SMARTtrees.trees[j].β,SMARTtrees.param)
    end

    return gammafit + SMARTtrees.gamma0*ones(T,size(x,1))

end



"""
    SMARTpredict(x,output)
Forecasts from SMARTboost, for y or the natural parameter.

# Inputs
- `x`                           (n,p) DataFrame or Float matrix of forecast origins (type<:real) or p vector of forecast origin
                                In the same format as the x given as input is SMARTdata(y,x,...). May contain missing or NaN.
- `output`                      output from SMARTfit

# Optional inputs
- `predict`                    [:Ey], :Ey or :Egamma. :Ey returns the forecast of y, :Egamma returns the forecast of the natural parameter.
                               The natural parameter is logit(prob) for :logistic, and mean(log(y)) if :lognormal and :logt.
- `best_model`                 [false] true to use only the single best model, false to use stacked weighted average
- `cutoff_paralellel`          [20_000] if x has more than these rows, a parallellized algorithm is called (which is slower for few forecasts)

# Output
- `yf`                         (n) vector of forecasts of y (or, outside regression, of the natural parameter), or scalar forecast if n = 1

# Note:
- For loss=:logt, E(y) is not available in closed form. An approximation is provided, which holds for small variance or large degrees of freedom. 

# Example of use
    output = SMARTfit(data,param)
    yf     = SMARTpredict(x_oos,output)
    yf     = SMARTpredict(x_oos,output,best_model=true)
"""
function SMARTpredict(x0::Union{AbstractDataFrame,AbstractArray},output::NamedTuple;best_model=false,cutoff_parallel=20_000,predict=:Ey)

    x = preparedata_predict(x0,output.SMARTtrees)

    if best_model==true || length(output.w)==1
        gammafit = SMARTpredict_internal(x,output.SMARTtrees,predict,cutoff_parallel=cutoff_parallel)  # SMARTtrees is for best model, SMARTtrees_a collects all
    else
        gammafit = zeros(output.T,size(x,1))

        for i in 1:length(output.w)

            if output.w[i]>0
                gammafit += output.w[i]*SMARTpredict_internal(x,output.SMARTtrees_a[i],predict,cutoff_parallel=cutoff_parallel)
            end

        end
    end
    
    return gammafit    # gammafit is actually Ey if predict = :Ey in SMARTpredict_internal

end 


"""
    SMARTcoeff(output;verbose=true)

Provides some information on constant coefficients for best model (in the form of a tuple.)
For example, error variance for :L2, dispersion and dof for :t.

# Inputs
- `output`                      output from SMARTfit

# Output
- `coeff`                      named tuple with information on fixed coefficients (e.g. variance for :L2, dispersion and dof for :t)

# Example of use
    output = SMARTfit(data,param)
    coeff  = SMARTcoeff(output,verbose=false)
"""
function SMARTcoeff(output;verbose=true)

    loss = output.bestparam.loss 
    coeff = output.bestparam.coeff_updated[1]

    if loss == :logistic
        θ    = (loss=loss,coeff="none")
    elseif loss in [:L2,:lognormal,:L2loglink] 
        θ    = (loss=loss,variance=coeff[1]^2)
    elseif loss == :t || loss == :logt
        s2,v    = exp(coeff[1]),exp(coeff[2])
        θ    = (loss=loss,scale=s2,dof=v,variance="scale*dof/(dof-2)" )
    elseif loss == :Huber
        σ2,ψ = coeff[1]^2,coeff[2] 
        θ    = (loss=loss,variance=σ2,psi=output.bestparam.coeff_user[1])
    elseif loss == :gamma 
        k    = coeff[1][1]
        θ    = (loss=loss,shape=k)    
    else 
        @error "loss not supported or misspelled. loss must be in [:logistic,:gamma,:L2,:Huber,:t,:quantile,:lognormal,:logt,:L2loglink]. "
    end

    if verbose==true
        display(θ)  
    end

    return  θ

end     



# Prepare the data, which may come as a DataFrame and have missing and categorical, with the
# Same transformations as in SMARTdata() and preparedataSMARTfor convenient data manipulation
function preparedata_predict(x0::Union{AbstractDataFrame,AbstractArray},SMARTtrees::SMARTboostTrees)

    param = SMARTtrees.param

    if typeof(x0) <: AbstractDataFrame
        x = deepcopy(x0)
    else
        typeof(x0) <: AbstractVector ? x0 = reshape(x0, (length(x0), 1)) : nothing    # transform vector to a matrix since DataFrames does not accept vectors  
        x = DataFrame(x0,:auto)
    end

    x = nan_and_missing_predict(x,param)  # for categorical, replaces nan with missing. For non-categorical, does the contrary 
    convert_dates_to_real!(x,param,predict=true)   
    if param.mask_missing == true
        x,fnames = missing_features_extend_x(param,x)      # extend x with dummy features if there were missing in the original dataset
    end 
    replace_nan_meanx!(x,param,SMARTtrees.meanx)  # only for features NOT in categorical or missing_features
    
    x = prepares_categorical_predict(x,param)  # categoricals are mapped to target encoding values; new categories allowed 
                                               # columns are added if param.cat_representation_dimension>1
    x = (x .- SMARTtrees.meanx)./SMARTtrees.stdx
    x = convert_df_matrix(x,param.T)

    return x
end 



# This function is used only in SMARTfit to compute residuals for add_different_loss
function SMARTpredict(x0::Union{AbstractDataFrame,AbstractArray},SMARTtrees::SMARTboostTrees;cutoff_parallel=20_000,predict=:Ey)

    x        = preparedata_predict(x0,SMARTtrees)
    gammafit = SMARTpredict_internal(x,SMARTtrees,predict,cutoff_parallel=cutoff_parallel)  # SMARTtrees is for best model, SMARTtrees_a collects all

    return gammafit    # gammafit is actually Ey if predict = :Ey in SMARTpredict_internal

end 


# This version takes in SMARTdata type and SMARTboosTrees, assumes one model.
# Used only in one place, to produce forecasts within SMARTfit. Get rid of it?   
function SMARTpredict_internal(data::SMARTdata,SMARTtrees::SMARTboostTrees,predict;cutoff_parallel=20_000)

    # Prepare the data, which may come as a DataFrame and have missing and categorical
    param = deepcopy(SMARTtrees.param)
    x = copy(data.x)
    x = replace_nan_with_missing(x)
    convert_dates_to_real!(x,param,predict=true)   

    map_cat_convert_to_float!(x,param)      # categorical are now in the form 0,1,2...
    x = replace_missing_with_nan(x)         # SharedArray do not accept missing.
    # replace categoricals with target encoding values, and standardize all features
    x = preparedataSMART_test(x,param,SMARTtrees.meanx,SMARTtrees.stdx)
 
    if typeof(x)<:AbstractDataFrame
        x = convert_df_matrix(x,param.T)
    end

    if size(x,2)==1  
        x=convert(Matrix,reshape(x0,length(x0),1))
    end

    gammafit = SMARTpredict_internal(x,SMARTtrees,predict,cutoff_parallel=cutoff_parallel)  # SMARTtrees is for best model, SMARTtrees_a collects all

    return gammafit

end



# from natural parameter to E(y)
# NB: for logt, E(y) is an approximation that only holds for small s2 and/or large v. 
function from_gamma_to_Ey(gammafit,param,predict)

    loss  = param.loss
    isempty(param.coeff_updated) ? nothing : coeff = param.coeff_updated[1]
    T     = param.T

    if predict == :Egamma
        return gammafit 
    end 

    if loss == :logistic
        pred = @. exp(gammafit)/(1+exp(gammafit))
    elseif loss in [:L1,:L2,:Huber,:t,:quantile]
        pred  = gammafit
    elseif loss in [:gamma,:L2loglink] 
        pred  = exp.(gammafit)    
    elseif loss == :lognormal
        σ    = coeff[1]
        pred = @. exp(gammafit + 0.5*σ^2)     
    elseif loss == :logt
        s2,v    = exp(coeff[1]),exp(coeff[2])
        vart    = s2*v/(v-2)
        pred = @. exp(gammafit + 0.5*vart)   # could be Inf
    else 
        @error "loss not supported or misspelled. loss must be in [:logistic,:L2,:Huber,:t,:quantile,:lognormal,:logt]. "
    end

    return pred

end




"""
    SMARTfit(data,param;cv_grid=[],add_hybrid=true,add_sparse=true,add_different_loss=false,add_sharp=false)

Fits SMARTboost with with k-fold cross-validation of number of trees and depth, and possibly a few more models.

If param.modality is :fast or :fastest, fits one model, at param, and if needed a second where sharp splits are 
forced on features with high average values of τ. For param.modality=:accurate or :compromise,  
may then fit, where appropriate, a few more models, which incrementally modify the original specification.

The additional models considered in modality=:accurate and :compromise are:
- Rough cross-validation of parameters for categorical features, if any.
- A penalization to encourage sparsity (fewer relevant features), unless user sets add_sparse=false.

If param.modality=:accurate, lambda for all models is set set min(lambda,0.1). If modality=:compromise, the default learning rate
lambda=0.2 is used, and the best model is then refitted with lambda = 0.1. 

Finally, all the estimated models considered are stacked, with weights chosen to minimize the cross-validated (original) loss.   



# Inputs
- `data::SMARTdata`
- `param::SMARTparam`

# Optional inputs

- `cv_grid::Vector`         The code performs a search in the space depth in [2,3,4,5,6], trying to fit few models if possible. Provide a
                            vector to over-ride (e.g. [2,4])   
- `add_sharp::Bool`         [false] models with sharp splits everywhere. This is then added to the stack. There is almost nothing to be gained from this 
                            with symmetric (obvlivious) trees; it is preferable to stack with standard trees (not implemented internally yet.)
- `min_p_sparsity`          [10] minimum number of features for sparsity or density penalizations to be considered

# Output (named tuple)

- `indtest::Vector{Vector{Int}}`  indexes of validation samples
- `bestvalue::Float`              best value of depth in cv_grid
- `bestparam::SAMRTparam`         param for best model  
- `ntrees::Int`                   number of trees (best value of param.ntrees) for best model
- `loss::Float`                   best cv loss
- `lossw::Float`                  loss of stacked models
- `meanloss:Vector{Float}`        mean cv loss at bestvalue of param for param.ntrees = 1,2,....
- `stdeloss:Vector{Float}`        standard errror of cv loss at bestvalue of param for param.ntrees = 1,2,....
- `lossgrid::Vector{Float}`       cv loss for best tree size for each grid value 
- `SMARTtrees::SMARTboostTrees`   for the best cv value of param and ntrees
- `SMARTtrees_a`                  length(cv_grid) vector of SMARTtrees
- `i`                             (ntrees,depth) matrix of threshold features for best model
- `mu`                            (ntrees,depth) matrix of threshold points  for best model
- `tau`                           (ntrees,depth) matrix of sigmoid parameters for best model
- `fi2`                           (ntrees,depth) matrix of feature importance, increase in R2 at each split, for best model
- `w`                             length(cv_grid) vector of stacked weights
- `ratio_actual_max`              ratio of actual number of candidate features over potential maximum. Relevant if sparsevs=:On: indicates sparsevs should be switched off if too high (e.g. higher than 0.5).
- `T`                             type for floating numbers
- `problems`                      true if there were computational problems in any of the models: NaN loss or loss jumping up

# Notes
- The following options for cross-validation are specified in param: randomizecv, nfold, sharevalidation, stderulestop

# Examples of use:
    param = SMARTparam()
    data   = SMARTdata(y,x,param)
    output = SMARTfit(data,param)

"""
function SMARTfit( data::SMARTdata, param::SMARTparam; cv_grid=[],add_different_loss::Bool=false,add_sharp::Bool=false,
    add_sparse=true,add_hybrid=true,min_p_sparsity=10,skip_full_sample=false)   # skip_full_sample enforces nofullsample even if nfold=1 (used in other functions, not by user)

    T,I = param.T,param.I

    additional_models     = 6    # After going through cv_grid, selects best value and fits: hybrid, sparse (up to 3), sharp, different distribution
    modality              = param.modality
    param0                = deepcopy(param)

    # if isempty(cv_grid), fits depth=2,3,4. If 2 is best, fits 1. If 3 is best, stops. If 4 is best, fits 5 and (if :accurate) 6.
    if isempty(cv_grid)
        user_provided_grid = false
        cv_grid=[1,2,3,4,5,6]     # NB: later code assumes this grid
    else     
        user_provided_grid = true
    end     

    if modality in [:fast,:fastest] 

        if user_provided_grid==false
            cv_grid = [param0.depth]
            user_provided_grid = true 
       end      

       add_sparse = false
    end  

    if modality==:fastest
        param0.nofullsample = true
        isempty(param0.indtrain_a) ? param0.nfold = 1 : nothing 

        if param.warnings==:On
            if isempty(param0.indtrain_a)
                @info "modality=:fastest is typically for preliminary explorations only. Setting param.nfold=1 and param.nofullsample=true.
                       Switch off this warning with param.warnings=:Off"
            else
                @info "modality=:fastest is typically for preliminary explorations only. Setting param.nofullsample=true.
                Switch off this warning with param.warnings=:Off"
            end          
        end  
    end

    if modality==:accurate
        param0.lambda=min(T(0.1),param0.lambda)
    end

    preliminary_cv!(param0,data,param0.nofullsample)       # preliminary cv of categorical parameters, if modality is not :fast.

    cvgrid0 = deepcopy(cv_grid)
    cvgrid  = vcat(cvgrid0,fill(cvgrid0[1],additional_models))

    treesize, lossgrid    = Array{I}(undef,length(cvgrid)), fill(T(Inf),length(cvgrid))  
    meanloss_a,stdeloss_a = Array{Array{T}}(undef,length(cvgrid)), Array{Array{T}}(undef,length(cvgrid))
    SMARTtrees_a          = Array{SMARTboostTrees}(undef,length(cvgrid))
    gammafit_test_a       = Vector{Vector{T}}(undef,length(cvgrid))
    y_test_a              = T[]
    indtest_a             = I[]
    problems_somewhere    = 0     

    param.randomizecv==true ? indices = shuffle(Random.MersenneTwister(param.seed_datacv),Vector(1:length(data.y))) : indices = Vector(1:length(data.y)) # done here to guarantee same allocation if randomizecv=true

    # Cross-validate depths in the model described by param0, on user-defined grid  
    if user_provided_grid==true 
        for (i,depth) in enumerate(cvgrid0)

            param = deepcopy(param0)
            param.depth = depth
            param_given_data!(param,data)
            param_constraints!(param)

            ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)

            treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i],SMARTtrees_a[i],gammafit_test_a[i],indtest_a,y_test_a = ntrees,loss,meanloss,stdeloss,SMARTtrees1st,gammafit_test,indtest,y_test 
            problems_somewhere = problems_somewhere + problems

        end    
    end 

    # Cross-validate depths in the model described by param0. 
    # option 1: if isempty(cv_grid), fits depth=3,4,5. If 3 is best, fits 2. If 4 is best, stops. If 5 is best, fits 6.
    # option 2 (faster): if isempty(cv_grid), fits depth=3,5. If 3 is best, fits 2. If 5 is best, fits 6.
    # NB: assumes cv_grid = [1,2,3,4,5,6]
    if user_provided_grid==false

        # option 1
        # i_a = [3,4,5]
        # option 2 
        i_a = [3,5]

        for _ in 1:2

            for i in i_a 

                param = deepcopy(param0)
                param.depth = cvgrid[i]
                param_given_data!(param,data)
                param_constraints!(param)

                ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)

                treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i],SMARTtrees_a[i],gammafit_test_a[i],indtest_a,y_test_a = ntrees,loss,meanloss,stdeloss,SMARTtrees1st,gammafit_test,indtest,y_test 
                problems_somewhere = problems_somewhere + problems

               # i==4 && lossgrid[4]>lossgrid[3] ? break : nothing   # would save time, but sensitive to noise

            end     

            if argmin(lossgrid)==3
                i_a = [2]
            elseif argmin(lossgrid)==4
                break
            else
                i_a = [6]
                #param.modality==:accurate ? i_a = [6] : break      
            end     
 
        end 

    end 

    # Additional models: Fit model with sparsity-inducing penalization, on best model fitted so far (including hybrid)

    best_i      = argmin(lossgrid)
    param       = deepcopy(SMARTtrees_a[best_i].param)
    p           = size(data.x,2)

    sparsity_grid = T.([0.7,1.1,1.5])           

    if add_sparse && p>=min_p_sparsity

        fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(SMARTtrees_a[best_i],data)
        s2 = sqrt(sum(fi.^2)/p)   # L2 measure of std
        s1 = 1.25*mean(fi)        # L1 measure of std, here just 125/p
        # s2/s1= 1 for Gaussian, <1 is playtokurtic, and >1 if leptokurtic, suggesting sparsity
        
        for (j,sparsity_penalization) in enumerate(sparsity_grid)

            param.sparsity_penalization = sparsity_penalization
            param.exclude_features = fi .< (0.01/p)       # increase speed by excluding features that are irrelevant even without penalization  
 
            param_given_data!(param,data)
            param_constraints!(param)

            ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)

            if j<length(sparsity_grid)
                fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(SMARTtrees1st,data)
            end     

            i = length(cvgrid0)+j
            cvgrid[i]   = cvgrid[best_i]
            treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]     = ntrees, loss, meanloss,stdeloss
            SMARTtrees_a[i],gammafit_test_a[i]                      = SMARTtrees1st, gammafit_test
            problems_somewhere = problems_somewhere + problems

            #If either 0.7 or 1.1 is better than 0.3, try 1.5. If neither is better, try 0.0.  
            if j>1 && min(lossgrid[i],lossgrid[i-1])>lossgrid[best_i]   # break (no need for more sparsity)

                param.sparsity_penalization = T(0)
                param.exclude_features = fill(false,p)

                param_given_data!(param,data)
                param_constraints!(param)

                ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)

                i = length(cvgrid0)+2 
                cvgrid[i]   = cvgrid[best_i]
                treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]     = ntrees, loss, meanloss,stdeloss
                SMARTtrees_a[i],gammafit_test_a[i]                      = SMARTtrees1st, gammafit_test
                problems_somewhere = problems_somewhere + problems

                break
            end
        
        end

    end 

    # Additional model: Fit hybrid model, with sharp splits forced on features with high τ. Threshold set at tau=10.
    # Only if the high τ are for features with non-trivial importance (fi).
    if param.priortype==:hybrid && add_hybrid
  
        best_i      = argmin(lossgrid)
        bestvalue   = cvgrid[best_i]
        force_sharp_splits = 10 .< mean_weighted_tau(SMARTtrees_a[best_i]) .<100 # 100 indicates that the split is always sharp anyway
        fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(SMARTtrees_a[best_i],data)
        fi_sharp = sort(fi[force_sharp_splits.==true],rev=true)
        p_sharp = length(fi_sharp)

        force_smooth_splits = mean_weighted_tau(SMARTtrees_a[best_i]).<3       
        fi_smooth = sort(fi[force_smooth_splits.==true],rev=true)
        p_smooth  = length(fi_smooth )

        if p_sharp>0    
            condition1 = fi_sharp[1]>10
            condition2 = sum(fi_sharp[1:minimum([3,p_sharp])])>15
            condition3 = sum(fi_sharp[1:minimum([10,p_sharp])])>20
            condition4 = sum(fi_sharp)>25
            condition_sharp = condition1 || condition2 || condition3 || condition4
        else 
            condition_sharp = false     
        end

        if p_smooth>0    
            condition5 = fi_smooth[1]>10
            condition6 = sum(fi_smooth[1:minimum([2,p_smooth])])>15
            condition7 = sum(fi_smooth[1:minimum([3,p_smooth])])>20
            condition8 = sum(fi_smooth[1:minimum([10,p_smooth])])>33
            condition9 = sum(fi_sharp)>40
            condition_smooth = condition5 || condition6 || condition7 || condition8 || condition9
        else
            condition_smooth = false
        end

#        if (p_sharp+p_smooth)>0 && (condition_sharp || condition_smooth)      # fit hybrid model.  
         if (p_sharp)>0 && condition_sharp                  # fit hybrid model 
            i = length(cvgrid0)+3+1                         # 3 is length(sparsity_grid)
            cvgrid[i]  = bestvalue        
            param      = deepcopy(SMARTtrees_a[best_i].param)
            param.force_sharp_splits = force_sharp_splits
            #param.force_smooth_splits = force_smooth_splits

            param_given_data!(param,data)
            param_constraints!(param)

            ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)
            treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]   = ntrees,loss,meanloss,stdeloss
            SMARTtrees_a[i],gammafit_test_a[i]                    = SMARTtrees1st,gammafit_test
            problems_somewhere = problems_somewhere + problems
        end 

    end        


    # Additional model: :sharptree, at previous best values of sparsity and depth.
    # Notes: i) INEFFICIENT IMPLEMENTATION     ii) different from a standard symmetric tree if depth>depth1

    if param.priortype==:hybrid && add_sharp 

        best_i      = argmin(lossgrid)
        param = deepcopy(SMARTtrees_a[best_i].param)
        param.priortype = :sharp
        i = length(cvgrid0)+5
        cvgrid[i]  = cvgrid[best_i]        

        ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)
        treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]   = ntrees,loss,meanloss,stdeloss
        SMARTtrees_a[i],gammafit_test_a[i]                    = SMARTtrees1st,gammafit_test
        problems_somewhere = problems_somewhere + problems

    end 

    # Before trying a different distribution, store param from the best solution. This will be used in 
    # SMARTmodelweights, which for some loss functions involving additional coefficients (:t,:gamma,...)
    # needs values of these coefficients, which param0 does not have.
    bestparam_original_loss = SMARTtrees_a[argmin(lossgrid)].param

    # Additional model 6: try a different distribution (loss), at previous best values of depth and sparsity
    best_i = argmin(lossgrid)
    param  = deepcopy(SMARTtrees_a[best_i].param)

    if add_different_loss 
        if param.loss == :L2   # fit a t distribution to residuals, and leave :L2 unless dof<10. 
 
            yfit = SMARTpredict(data.x,SMARTtrees_a[best_i],predict=:Ey) 
            res = Newton_MAP(data.y - yfit,gH_student,start_value_student,w=data.weights)
            dof = exp(res.minimizer[2])

            dof<10 ? param.loss = :t : nothing 
            #param.losscv = :mse     # Not needed if this is the last model fitted. 
        end 
    end         

    # Experimental. Not well tested yet. 
    # NB: cv loss may NOT comparable across different distributions.
    if add_different_loss==true && (param.loss != SMARTtrees_a[best_i].param.loss) 
 
        param_given_data!(param,data)
        param_constraints!(param)

        ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)

        i = length(cvgrid0)+6
        cvgrid[i]   = cvgrid[best_i]
        lossgrid[i],meanloss_a[i], stdeloss_a[i] = loss,meanloss,stdeloss  # The loss is NOT comparable. 
        treesize[i]     = ntrees
        SMARTtrees_a[i],gammafit_test_a[i] = SMARTtrees1st, gammafit_test
        problems_somewhere = problems_somewhere + problems
    
    end    

    # if modality==:compromise and lambda>0.1, fits the best model with param0.lambda = 0.1, and replaces it.
    # If model with lowest loss does not have the user-specified distribution, takes the model with the highest stacking weight 
    best_i = argmin(lossgrid)

    if SMARTtrees_a[best_i].param.loss != param0.loss
        w,lossw = SMARTmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
        best_i = argmax(w)
    end     
    
    param       = deepcopy(SMARTtrees_a[best_i].param)
               
    if modality==:compromise && param.lambda>0.1
        
        param.lambda = T(0.1)

        param_given_data!(param,data)
        param_constraints!(param)

        ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest,gammafit_test,y_test,problems = SMARTsequentialcv(data,param,indices=indices)
        gammafit_test = from_gamma_to_Ey(gammafit_test,param,:Ey)    #  gammafit_test now comparable.

        i = best_i   # replaces the best model

        if loss<lossgrid[i]   # exceptions may happen when interacting with sparsity_penalization, which is ideally calibrated at the final lambda
            lossgrid[i],meanloss_a[i],stdeloss_a[i] = loss,meanloss,stdeloss 
            treesize[i]     = ntrees
            SMARTtrees_a[i],gammafit_test_a[i] = SMARTtrees1st, gammafit_test
            problems_somewhere = problems_somewhere + problems
        end     

    end 

    # If there is a NaN in lossgrid, Julia takes it as the minimum, hence...
    if isnan(minimum(lossgrid))
        @warn "In SMARTfit, some output is NaN. Switching to Float64 may solve the problem (param=SMARTparam(T=Float64)). "
        problems_somewhere = problems_somewhere + 1 
    end

    lossgrid = replace( lossgrid,T(NaN)=>T(Inf) )

    # select model with lowest loss. If this does not have the user-specified distribution, takes the model with the highest stacking weight 
    best_i = argmin(lossgrid)

    if SMARTtrees_a[best_i].param.loss != param0.loss
        w,lossw = SMARTmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
        best_i = argmax(w)
    end     

    bestvalue,ntrees,loss,meanloss,stdeloss = cvgrid[best_i],treesize[best_i],lossgrid[best_i],meanloss_a[best_i],stdeloss_a[best_i]

    # Fit again on the full sample (unless nofullsample=true and nfold==1). This is done for all the values in cvgrid
    # with weight>=0.1 (to reduce computing time). 
    if (param0.nofullsample==false || param0.nfold>1) && skip_full_sample==false

        w,lossw = SMARTmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
        w[best_i]=max(w[best_i],0.1)   # the best model should be refitted
        w = w.*(w .>= 0.1)
        w = w/sum(w)
    
        m = max(1,log(length(data.y))/log( length(data.y)-length(y_test_a)/param.nfold))  # m = avg(n_train+n_test)/avg(n_train), avg over nfold
        ntrees = Int(floor(ntrees*m))                                             # small adjustment for n>n_train. Tiny effect.

        for i in 1:length(cvgrid)    
        
            if w[i]>0

                param = deepcopy(SMARTtrees_a[i].param)    
                param.ntrees   = Int(floor(treesize[i]*m))

                param_given_data!(param,data)
                param_constraints!(param)

                SMARTtrees_a[i]  = SMARTbst(data,param)
            end

        end
        SMARTtrees = SMARTtrees_a[best_i]
    else
        SMARTtrees = SMARTtrees_a[best_i]
    end

    param = deepcopy(param0)

    # Ensembles of stacked trees.
    if sum(lossgrid.<Inf)==1
        lossw = loss
        w = zeros(T,length(lossgrid))
        w[argmin(lossgrid)] = T(1)
    else
        w,lossw = SMARTmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
    end

    # provide some additional output
    i,μ,τ,fi2=SMARToutput(SMARTtrees)  # on the best value
    avglntau,varlntau,mselntau,postprob2 = tau_info(SMARTtrees)
    ratio_actual_max = tight_sparsevs(ntrees,SMARTtrees.param) # ratio of actual vs max number of candidate features

    for i in eachindex(lossgrid)   # done to trigger warning if sparsevs seems too tight in ANY of the model
        if lossgrid[i]<Inf
            aux = tight_sparsevs(treesize[i],SMARTtrees_a[i].param)
        end 
    end

    return ( indtest=indtest_a,bestvalue=bestvalue,bestparam=SMARTtrees.param,ntrees=ntrees,loss=loss,meanloss=meanloss,stdeloss=stdeloss,lossgrid=lossgrid,SMARTtrees=SMARTtrees,
    i=i,mu=μ,tau=τ,fi2=fi2,avglntau=avglntau,SMARTtrees_a=SMARTtrees_a,w=w,lossw=lossw,T=T,problems=(problems_somewhere>0),ratio_actual_max=ratio_actual_max)

end



function SMARTrelevance(SMARTtrees::SMARTboostTrees,data::SMARTdata )

    fi2 = deepcopy(SMARTtrees.fi2) 
    @. fi2 = abs( fi2*(fi2>=0) )   # Ocassional (tiny) negative numbers set to zero

    fi         = sqrt.(fi2)
    fi         = 100*fi./sum(fi)
    sortedindx = sortperm(fi,rev = true)
    fnames     = data.fnames

    return fnames,fi,fnames[sortedindx],fi[sortedindx],sortedindx
end


"""
    SMARTrelevance(output,data::SMARTdata;verbose=true,best_model=false)

Computes feature importance (summing to 100), defined by the relevance measure of Breiman et al. (1984), equation 10.42 in
Hastie et al., "The Elements of Statistical Learning", second edition, except that the normalization is for sum = 100, not for largest = 100.
Relevance is defined on the fit of the trees on pseudo-residuals.
best_model=true for single model with lowest CV loss, best_model= false for weighted average (weights optimized by stacking)

# Output
- `fnames::Vector{String}`         feature names, same order as in data
- `fi::Vector{Float}`              feature importance, same order as in data
- `fnames_sorted::Vector{String}`  feature names, sorted from highest to lowest importance
- `fi_sorted::Vector{Float}`       feature importance, sorted from highest to lowest
- `sortedindx::Vector{Int}`        feature indices, sorted from highest to lowest importance


# Example of use
    output = SMARTfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data,verbose = false)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data,best_model=true)

"""
function SMARTrelevance(output,data::SMARTdata;verbose=true,best_model=false )

    T   = output.T 
    w   = output.w

    if best_model==true   # SMARTtrees for best model, SMARTtrees_a for all
        fi2 = deepcopy(output.SMARTtrees.fi2) 
        @. fi2 = abs( fi2*(fi2>=0) )   # Ocassional (tiny) negative numbers set to zero
    else
        fi2 = zeros(T,size(data.x,2))

        for i in eachindex(w)

            if w[i]>0
                fi2 = deepcopy(output.SMARTtrees_a[i].fi2)
                fi2 += w[i]*(abs.(fi2.*(fi2 .>=0) ))    
            end 
        end

    end 

    fi         = sqrt.(max.(0,fi2))   # tiny negative numbers are possible in fi2 
    fi         = 100*fi./sum(fi)
    sortedindx = sortperm(fi,rev = true)
    fnames     = data.fnames

#    verbose == true ? printmat(hcat(fnames[sortedindx],fi[sortedindx])) :
    if verbose == true
        m   = Matrix(undef,length(fi),2)
        m[:,1] = fnames[sortedindx]
        m[:,2] = fi[sortedindx]
        printlnPs("\nFeature relevance, sorted from highest to lowest, adding up to 100 \n")
        printmat(m)                          # Paul Soderlind's printmat()
    end

    return fnames,fi,fnames[sortedindx],fi[sortedindx],sortedindx
end



"""
    SMARTpartialplot(data::SMARTdata,output,features::Vector{Int64};,predict=:Egamma,best_model=false,other_xs::Vector=[],q1st=0.01,npoints=1000))
Partial dependence plot for selected features. Notice: Default is for natural parameter (gamma) rather than y.
For feature i, computes gamma(x_i) - gamma(x_i=mean(x_i)) for x_i between q1st and 1-q1st quantile, with all other features at their mean.

# Inputs

- `data::SMARTdata`
- `output`
- `features::Vector{Int}`        position index (in data.x) of features to compute partial dependence plot for
- `other_xs::Vector{Float}`      (keyword), a size(data.x)[1] vector of values at which to evaluate the responses. []
                                 Note: other_xs should be expressed in standardized units, i.e. for (x_i-mean(x_i))/std(x_i)   
- `q1st::Float`                  (keyword) first quantile to compute, e.g. 0.001. Last quantile is 1-q1st. [0.01]
- `npoints::Int'                 (keyword) number of points at which to evalute f(x). [1000]

# Optional inputs
- `predict`                    [:Egamma], :Ey or :Egamma. :Ey returns the impact on the forecast of y, :Egamma on the natural parameter.  
- `best_model`                 [false]  true for single model with lowest CV loss, =false for weighted average (by stacking)

# Output
- `q::Matrix`                   (npoints,length(features)), values of x_i at which f(x_i) is evaluated
- `pdp::Matrix`                 (npoints,length(features)), values of f(x_i)

# Example of use 
    output = SMARTfit(data,param)
    q,pdp  = SMARTpartialplot(data,output.SMARTtrees,sortedindx[1,2],q1st=0.001)
"""
function SMARTpartialplot(data::SMARTdata,output,features;best_model=false,other_xs::Vector =[],q1st=0.01,npoints = 1000,predict=:Egamma)

    # data.x is SharedMatrix, not standardized, categoricals are 0,1,2 .... dates are [0,1]
    # replace categoricals with target encoding values, and standardize all features
    x = preparedataSMART_test(data.x,output.SMARTtrees.param,output.SMARTtrees.meanx,output.SMARTtrees.stdx)

    if best_model==true || length(output.SMARTtrees_a)==1
        q,pdp  = SMARTpartialplot(x,output.SMARTtrees,features,predict,other_xs=other_xs,q1st=q1st,npoints=npoints)
    else
        T = output.T
        pdp = zeros(T,npoints,length(features) )

        for i in 1:length(output.SMARTtrees_a)
            if output.w[i]>0
                q,pdp_i=SMARTpartialplot(x,output.SMARTtrees_a[i],features,predict,other_xs=other_xs,q1st=q1st,npoints=npoints)
                pdp  += output.w[i]*pdp_i
            end
        end
    end

    meanx,stdx = output.SMARTtrees.meanx,output.SMARTtrees.stdx
    q = q.*stdx[features]' .+ meanx[features]'      # convert back to original scale

    return q,pdp
end



# x is standardized (and previously categoricals replaced by target encoding values)
function SMARTpartialplot(x::AbstractArray,SMARTtrees::SMARTboostTrees,features,predict;other_xs::Vector =[],q1st=0.01,npoints = 1000)

    T = SMARTtrees.param.T
    features = Int.(features)
    npoints = Int(npoints)
 
    if isempty(other_xs)
        other_xs =  T.(mean(x,dims=1))
    else
        other_xs  = T.(other_xs')
    end

    step = (1-2*q1st)/(npoints-1)
    p    = [i for i in q1st:step:1-q1st]

    pdp = Matrix{T}(undef,length(p),length(features))
    q  = Matrix{T}(undef,length(p),length(features))

    for (i,f) in enumerate(features)
        q[:,i] = T.(quantile_allow_nan(x[:,f],p))
        h = ones(T,length(p)).*other_xs
        h0 = copy(h)
        h[:,f] = q[:,i]
        pdp[:,i] = SMARTpredict_internal(h,SMARTtrees,predict) - SMARTpredict_internal(h0,SMARTtrees,predict)
    end

    return q,pdp
end



function quantile_allow_nan(x::AbstractVector,p)

    miss_a = isnan.(x) 

    if sum(miss_a)==0
        return quantile(x,p)
    else
        keep_a = miss_a .== false 
        return quantile(x[keep_a],p)
    end 

end 



"""
    SMARTmarginaleffect(data::SMARTdata,output,features::Vector{Int64};predict=:Egamma,best_model=false,other_xs::Vector =[],q1st=0.01,npoints=50,epsilon=0.02)
APPROXIMATE Computation of marginal effects using NUMERICAL derivatives (default ϵ=0.01)

# Inputs

- `data::SMARTdata`
- `SMARTtrees::SMARTboostTrees`
- `features::Vector{Int}`        position index (in data.x) of features to compute partial dependence plot for
- `other_xs::Vector{Float}`      (keyword), a size(data.x)[1] vector of values at which to evaluate the marginal effect. []
                                 Note: other_xs should be expressed in standardized units, i.e. for (x_i-mean(x_i))/std(x_i)   
- `q1st::Float`                  (keyword) first quantile to compute, e.g. 0.001. Last quantile is 1-q1st. [0.01]
- `npoints::Int'                 (keyword) number of points at which to evalute df(x_i)/dx_i. [50]
- `epsilon::Float'               (keyword) epsilon for numerical derivative, [0.01]

# Optional inputs
- `predict`                     [:Egamma], :Ey or :Egamma. :Ey returns the impact on the forecast of y, :Egamma on the natural parameter.  
- `best_model`                  [false]  true for single model with lowest CV loss, =false for weighted average (by stacking)


# Output
- `q::Matrix`                   (npoints,length(features)), values of x_i at which f(x_i) is evaluated, or vector if npoints = 1
- `d::Matrix`                   (npoints,length(features)), values of marginal effects, or vector if npoints = 1

# NOTE: Provisional! APPROXIMATE Computation of marginal effects using NUMERICAL derivatives. (Analytical derivatives are available)

# NOTE: To compute marginal effect at one point x0 rather than over a grid, set npoints = 1 and other_xs = x0 (a p vector, p the number of features)

# Example
    output = SMARTfit(data,param)
    q,m    = SMARTmarginaleffect(data,output.SMARTtrees,[1,3])

# Example
    q,m  = SMARTmarginaleffect(data,output.SMARTtrees,[1,2,3,4],other_xs = zeros(p),npoints = 1)

# Example
    output = SMARTfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data,verbose=false)
    q,m  = SMARTmarginaleffect(data,output,sortedindx[1,2],q1st=0.001)

"""
function SMARTmarginaleffect(data::SMARTdata,output,features;best_model=false,other_xs::Vector =[],q1st=0.01,npoints = 50,epsilon=0.02,predict=:Egamma)

    if output.SMARTtrees.param.priortype !== :smooth
        @warn "Derivatives computed in SMARTmarginaleffects may not be defined unless param.priortype=:smooth"
    end    

    x = preparedataSMART_test(data.x,output.SMARTtrees.param,output.SMARTtrees.meanx,output.SMARTtrees.stdx)

    if best_model==true || length(output.SMARTtrees_a)==1
        q,m  = SMARTmarginaleffect(x,output.SMARTtrees,features,predict,q1st=q1st,npoints=npoints,epsilon=epsilon)
    else
        T = typeof(data.y[1])
        m = zeros(T,npoints,length(features) )

        for i in 1:length(output.SMARTtrees_a)
            if output.w[i]>0
                q,m_i  = SMARTmarginaleffect(x,output.SMARTtrees_a[i],features,predict,q1st=q1st,npoints=npoints,epsilon=epsilon)
                m    = m + output.w[i]*m_i
            end
        end

    end

    meanx,stdx = output.SMARTtrees.meanx,output.SMARTtrees.stdx
    q = q.*stdx[features]' .+ meanx[features]'      # convert back to original scale

    return q,m
end


# expects x to be standardized, with categoricals replaced by target encoding values
function SMARTmarginaleffect(x::AbstractArray,SMARTtrees::SMARTboostTrees,features,predict;other_xs::Vector =[],q1st=0.01,npoints = 50,epsilon=0.01)

    param = SMARTtrees.param
    I = param.I
    T = SMARTtrees.param.T
    features = I.(features)
    npoints = I(npoints)

    # compute a numerical derivative
    if npoints==1

        if length(other_xs)==0
            other_xs =  T.(mean(x,dims=1))
        else
            other_xs  = T.(other_xs')
        end

        d   = Vector{T}(undef,length(features))
        q   = Vector{T}(undef,length(features))

        for (i,f) in enumerate(features)
            q[i] = other_xs[f]
            h1 = copy(other_xs)
            h2 = copy(other_xs)
            h1[f] = h1[f] + T(epsilon)
            h2[f] = h2[f] - T(epsilon)
            d[i] = (( SMARTpredict_internal(h1,SMARTtrees,predict) - SMARTpredict_internal(h2,SMARTtrees,predict) )/T(2*epsilon))[1]
        end

        return q,d

    else
        q,pdp   = SMARTpartialplot(x,SMARTtrees,features,predict,other_xs = other_xs,q1st = q1st,npoints = npoints+2)
        n       = size(q,1)
        d       = (pdp[1:n-2,:] - pdp[3:n,:])./(q[1:n-2,:] - q[3:n,:] )  # numerical derivatives at q[i]: f(q[i+1]-f(q[i-1])/(q[i+1]-q[i-1]) )
        return q[2:n-1,:],d
    end

end



"""

    SMARToutput(SMARTtrees::SMARTboostTrees)

Output fitted parameters estimated from each tree, collected in matrices.

# Output
- `i`         (ntrees,depth) matrix of threshold features
- `μ`         (ntrees,depth) matrix of threshold points
- `τ`         (ntrees,depth) matrix of sigmoid parameters
- `fi2`       (ntrees,depth) matrix of feature importance, increase in R2 at each split

# Example of use
output = SMARTfit(data,param)
i,μ,τ,β,fi2 = SMARToutput(output.SMARTtrees)

"""
function SMARToutput(SMARTtrees::SMARTboostTrees)

    I = typeof(SMARTtrees.param.depth)
    T = typeof(SMARTtrees.param.lambda)
    ntrees = length(SMARTtrees.trees)
    d = length(SMARTtrees.trees[1].i)

    i   = Matrix{I}(undef,ntrees,d)
    μ   = Matrix{T}(undef,ntrees,d)
    τ   = Matrix{T}(undef,ntrees,d)
    fi2 = Matrix{T}(undef,ntrees,d)

    for j in 1:ntrees
        tree = SMARTtrees.trees[j]
        i[j,:],μ[j,:],τ[j,:],fi2[j,:] = tree.i,tree.μ,tree.τ,tree.fi2
    end

    return i,μ,τ,fi2
end


# compute actual number of elements in param.best_features and compares it with the theoretical max,
# issuing a warning if the ratio is too high.
function tight_sparsevs(ntrees,param)       #ntrees is the number of trees in the final model

    fib_sequence = fibonacci(20,param.lambda,param.frequency_update)
    fib_sequence = fib_sequence[fib_sequence .<= ntrees]

    max_number       = length(fib_sequence)*param.depth*param.number_best_features
    actual_number    = length(param.best_features)
    ratio_actual_max = actual_number/max_number

    param.depth==1 ? warning_threshold = 0.7 : warning_threshold = 0.5 

    if param.warnings==:On && param.sparsevs==:On && ratio_actual_max > warning_threshold
        @warn "WARNING: with sparsevs=:On, the number of candidate features is $(round(100*ratio_actual_max,digits=1))% of the theoretical max, which suggests a dense setting in which sparsevs may be placing a constraint on feature selection,
        potentially leading to a loss of accuracy. Consider setting sparsevs=:Off, or increasing number_best_features (default 10) to increase the number of candidate features. "
    end

    return ratio_actual_max

end

""" 

    SMARTweightedtau(output,data;verbose=true,plot_tau=true,best_model=false)

Computes weighted (by variance importance gain at each split) smoothing parameter τ for each
feature, and for the entire model (features are averaged by variance importance)
statistics for each feature, averaged over all trees. Sharp thresholds (τ=Inf) are bounded at 40.
best_model=true for single model with lowest CV loss, best_model= false for weighted average (weights optimized by stacking)

# Input 
- `output`   output from SMARTfit
- `data`     data input to SMARTfit

# Optional inputs 
- `verbose`   [true]  prints out the results to screen as DataFrame
- `plot_tau`  [true]  plots a sigmoid with avgtau (see below) to get a sense of function smoothness.

# Output
- `avgtau`         scalar, average importance weighted τ over all features (also weighted by variance importance) 
- `avg_explogtau`  scalar, exponential of average importance weighted log τ over all features (also weighted by variance importance) 
- `avgtau_a`       p-vector of avg importance weighted τ for each feature 
- `df`             dataframe collecting avgtau_a information (only if verbose=true)


# Example of use
output = SMARTfit(data,param)
avgtau,avg_explogtau,avgtau_a,dftau = SMARTweightedtau(output,data)
avgtau,avg_explogtau,avgtau_a,dftau = SMARTweightedtau(output,data,verbose=false,plot_tau=false,best_model=true)

"""
function SMARTweightedtau(output,data;verbose::Bool=true,plot_tau::Bool=true,best_model::Bool=false)

    T = Float64
    SMARTtrees = output.SMARTtrees
    p = max(length(SMARTtrees.infeatures),length(SMARTtrees.meanx))  # they should be the same ...
    
    if best_model==true
        avgtau_a = mean_weighted_tau(SMARTtrees)
    else     
        avgtau_a = zeros(p)
        w        = output.w
        
        for i in eachindex(w)
            if w[i]>0 
                avgtau_a += w[i]*mean_weighted_tau(output.SMARTtrees_a[i])
            end 
        end 
    end     

    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data,verbose=false,best_model=best_model)
    
    avgtau  = sum(avgtau_a.*fi)/sum(fi)
    exp_avglogtau = exp( sum(log.(avgtau_a).*fi)/sum(fi) )

    df = DataFrame(feature = fnames, importance = fi, avgtau = avgtau_a,
           sorted_feature = fnames_sorted, sorted_importance = fi_sorted, sorted_avgtau = avgtau_a[sortedindx])

    if verbose==true
        df = DataFrame(feature = fnames, importance = fi, avgtau = avgtau_a,
        sorted_feature = fnames_sorted, sorted_importance = fi_sorted, sorted_avgtau = avgtau_a[sortedindx])
        display(df)
        println("\n Average smoothing parameter τ is $(round(avgtau,digits=1)).")
        println("\n In sufficiently large samples, and if modality=:compromise or :accurate:")
        println("\n - Values above 20-25 suggest little smoothness in important features. SMARTboost may slightly outperform or slightly underperform other gradient boosting machines.")
        println(" - At 10-15 or lower, SMARTboost should outperform other gradient boosting machines, or at least be worth including in an ensemble.")
        println(" - At 5-7 or lower, SMARTboost should strongly outperform other gradient boosting machines.")

    else 
        df = nothing     
    end 

    if plot_tau==true
        x = range(-2,stop=2,length=100)
        τ,μ = avgtau,0.0
        g = @. 0.5 + 0.5*( 0.5*τ*(x-μ)/sqrt(( 1.0 + ( 0.5*τ*(x-μ) )^2  )) )

        display(plot(x,g,
                title="measures of average smoothing",
                xlabel = "standardized x",
                label = ["avg(tau)"],
                legend = :bottomright,
                linecolor = :blue,
                linestyle = :solid,
                linewidth = 5,
                ))
    end

    return T(avgtau),T(exp_avglogtau),T.(avgtau_a),df

end 
