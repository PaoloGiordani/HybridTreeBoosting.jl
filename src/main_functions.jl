#
#  Collects most functions that are exported, as well as some that are not exported
#
#  INFORMATION about available functions   
#  HTBinfo               basic information about the main functions in HTBoost
# 
#  SETTING UP THE MODEL, and CALIBRATING PRIOR for panel data and/or overlapping data
#  HTBloglikdivide       calibrates param.loglikdivide for panel data and/or overlapping data
#  HTBindexes_from_dates train and test sets indexes for expanding window cross-validation
#  HTBparam              parameters, defaults or user-provided
#
#  FITTING and FORECASTING
#  HTBfit                fits HTBoost with cv (or validation/early stopping) of number of trees and, optionally, depth or other parameter
#    HTBfit_single       main function, building block for composite and multi-parameter models 
#       find_force_sharp_splits 
#    HTBfit_hurdle       hurdle models: y continuous, split into y=0 and y /=0   
#    HTBfit_multiclass   multiclass classicaition 
#  HTBbst                fits HTB when nothing needs to be cross-validated (not even number of trees)
#  HTBpredict            prediction from HTBtrees::HTBoostTrees
#    HTBpredict          output is vector of tuples (composite models)
#    preparedata_predict
#    HTBpredict_hurdle
#    HTBpredict_multiclass
#
#  POST-ESTIMATION ANALYSIS
#  HTBcoeff              provides information on constant coefficients
#  HTBrelevance          computes feature importance (Breiman et al 1984 relevance)
#  HTBpartialplot        partial dependence plots (keeping all other features fixed, not integrating out)
#  HTBmarginaleffect     provisional! Numerical computation of marginal effects.
#  HTBoutput             collects fitted parameters in matrices
#  tight_sparsevs          warns if sparsevs seems to compromise fit 
#  HTBweightedtau        computes weighted smoothing parameter
#  HTBplotppr            plots projection pursuit regression 
#
# AUXILIARY FUNCTIONS (not for export)
# from_gamma_to_Ey
# displayinfo
# impose_sharp_splits  


"
    HTBinfo()
Basic information about the main functions in HTBoost (see help on each function for more details)


# Setting up the model 
- `HTBindexes_from_dates` builds train and test sets indexes for expanding window cross-validation (if the user wants to over-ride the default block-cv)
- `HTBparam`           parameters, defaults or user-provided.
- `HTBdata`            y,x and, optionally, dates, weights, and names of features

# Fitting and forecasting 
- `HTBfit`             fits HTBoost with cv (or validation/early stopping) of number of trees and, optionally, depth or other parameter
- `HTBpredict`         predictions for y or natural parameter

#  POST-ESTIMATION ANALYSIS
- `HTBcoeff`           provides information on constant coefficients, e.g. dispersion and dof for loss=:t
- `HTBrelevance`       computes feature importance (Breiman et al 1984 relevance)
- `HTBpartialplot`     partial dependence plots (keeping all other features fixed, not integrating out)
- `HTBmarginaleffect`  Numerical computation of marginal effects.
- `HTBoutput`          collects fitted parameters in matrices
- `HTBweightedtau`     computes weighted smoothing parameter to help assess function smoothness

Example of use of info function 

    help?> HTBinfo
    or ...
    julia> HTBinfo()

To find more information about a specific function, e.g. HTBfit

    help?> HTBfit

Example of basic use of HTBoost functions with iid data and default settings

    param  = HTBparam()                                 
    data   = HTBdata(y,x,param)
    output = HTBfit(data,param)
    yf     = HTBpredict(x_test,output)  

See the examples and tutorials for illustrations of use. 
"
function HTBinfo()

    println("Documentation: $(Base.doc(HTBinfo))")
  
end



#=
    HTBloglikdivide(df,y_symbol,date_symbol;overlap=0)

loglikdivide is computed internally, so the user does not need to call this function for HTBoost.
loglikdivide can be found in output = SMARTfit()

The only effect of loglikdivide in HTBoost is to calibrate the strength of the prior in relation to the likelihood evidence.
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
    lld,ess =  HTBloglikdivide(df,:excessret,:date,overlap=h-1)

=# 
function HTBloglikdivide(df::DataFrame,y_symbol,date_symbol;overlap = 0)

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



function HTBloglikdivide(y::AbstractVector{T},dates_all;overlap=0) where T<:Real

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
HTBindexes_from_dates(df::DataFrame,datesymbol::Symbol,first_date::Date,n_reestimate::Int)

Computes indexes of training set and test set for cumulative CV and pseudo-real-time forecasting exercises

* INPUTS

- datesymbol            symbol name of the date
- first_date            when the first training set ENDS (end date of the first training set)
- n_reestimate          every how many periods to re-estimate (update the training set)

* OUTPUT
  indtrain_a,indtest_a are arrays of arrays of indexes of train and test samples

* Example of use

- first_date = Date("2017-12-31", Dates.DateFormat("y-m-d"))
- indtrain_a,indtest_a = HTBindexes_from_dates(df,:date,first_date,12)

* NOTES

- Inefficient for large datasets

"""
function HTBindexes_from_dates(df::DataFrame,datesymbol::Symbol,first_date,n_reestimate)

    n_reestimate = Int(n_reestimate)

    indtrain_a = Vector{Int}[]
    indtest_a  = Vector{Int}[]

    dates   = df[:,datesymbol]
    datesu  = sort(unique(dates))   # must sort
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





#=
    HTBbst(data::HTBdata, param::HTBparam)
HTBoost fit, number of trees defined by param.ntrees, not cross-validated.

# Output
- `HTBtrees::HTBoostTrees`

# Example of use
    HTBtrees =  HTBbst(data,param)
=#
function HTBbst(data0::HTBdata, param::HTBparam )

    # initialize HTBtrees
    param,data,meanx,stdx          = preparedataHTB(data0,param)
    τgrid,μgrid,Info_x,n,p         = preparegridsHTB(data,param,meanx,stdx)

    τgrid,μgrid,Info_x,n,p         = preparegridsHTB(data,param,meanx,stdx)

    gamma0                         = initialize_gamma0(data,param)
    gammafit                       = data0.offset + fill(gamma0,n) 

    param          = updatecoeff(param,data.y,gammafit,data.weights,0)
    HTBtrees       = HTBoostTrees(param,gamma0,data0.offset,n,p,meanx,stdx,Info_x)
    rh,param       = gradient_hessian( data.y,data.weights,gammafit,param,0)

    # prelimiminary run to calibrate coefficients and priors
    Gβ,trash  = fit_one_tree(data.y,data.weights,HTBtrees,rh.r,rh.h,data.x,μgrid,Info_x,τgrid,param)
    param = updatecoeff(param,data.y,HTBtrees.gammafit+Gβ,data.weights,0) # +Gβ, NOT +λGβ
    trash,param = gradient_hessian( data.y,data.weights,HTBtrees.gammafit+Gβ,param,1)

    for iter in 1:param.ntrees
        displayinfo(param.verbose,iter)
        Gβ,i,μ,τ,m,β,fi2,σᵧ  = fit_one_tree(data.y,data.weights,HTBtrees,rh.r,rh.h,data.x,μgrid,Info_x,τgrid,param)
        param = updatecoeff(param,data.y,HTBtrees.gammafit+Gβ,data.weights,iter) # +Gβ, NOT +λGβ
        updateHTBtrees!(HTBtrees,Gβ,HTBtree(i,μ,τ,m,β,fi2,σᵧ),iter,param)          # updates gammafit=gammafit_old+λGβ
        rh,param = gradient_hessian( data.y,data.weights,HTBtrees.gammafit,param,2)
    end

    # bias adjustment
    bias,HTBtrees.gammafit = bias_correct(HTBtrees.gammafit,data.y,HTBtrees.gammafit,param)
    HTBtrees.gamma0 +=  bias

    return HTBtrees

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


#Forecasts from HTBoost. Expects x to be standardized, with categorical already transformed
function HTBpredict_internal(x::AbstractMatrix,HTBtrees::HTBoostTrees,predict;cutoff_parallel=20_000,offset=[])

    n,p = size(x)
    T   = HTBtrees.param.T
 
    if p != length(HTBtrees.meanx)
        @error "In HTBpredict, the input matrix x has column dimension $p while HTBfit has been estimated with $(length(HTBtrees.meanx)) features in data.x"
    end

    if n>cutoff_parallel
        gammafit = HTBpredict_distributed(x,HTBtrees)
    else

        gammafit = HTBtrees.gamma0*ones(T,size(x,1))
    
        for j in 1:length(HTBtrees.trees)
            tree     =  HTBtrees.trees[j]          
            gammafit += HTBtrees.param.lambda*HTBtreebuild(x,tree.i,tree.μ,tree.τ,tree.m,tree.β,tree.σᵧ,HTBtrees.param)    
        end
    end

    if !isempty(offset)
        gammafit = T.(offset) + gammafit
    end     

    pred = from_gamma_to_Ey(gammafit,HTBtrees.param,predict) # from natural parameter to E(y), depending on predict

    return pred

end


# Expects x to be standardized, with categories and missing already transformed
function HTBpredict_distributed(x::AbstractMatrix,HTBtrees::HTBoostTrees)

    T       = typeof(HTBtrees.gamma0)
    x       = SharedMatrixErrorRobust(x,HTBtrees.param)

    gammafit = @distributed (+) for j = 1:length(HTBtrees.trees)
        HTBtrees.param.lambda*HTBtreebuild(x,HTBtrees.trees[j].i,HTBtrees.trees[j].μ,HTBtrees.trees[j].τ,HTBtrees.trees[j].m,HTBtrees.trees[j].β,HTBtrees.trees[j].σᵧ,HTBtrees.param)
    end

    return gammafit + HTBtrees.gamma0*ones(T,size(x,1))

end



"""
    HTBpredict(x,output)
Forecasts from HTBoost, for y or the natural parameter.

# Inputs
- `x`                           (n,p) DataFrame or Float matrix of forecast origins (type<:real) or p vector of forecast origin
                                In the same format as the x given as input is HTBdata(y,x,...). May contain missing or NaN.
- `output`                      output from HTBfit

# Optional inputs
- `predict`                    [:Ey], :Ey or :Egamma. :Ey returns the forecast of y, :Egamma returns the forecast of the natural parameter.
                               The natural parameter is logit(prob) for :logistic, and mean(log(y)) if :lognormal.
- `best_model`                 [false] true to use only the single best model, false to use stacked weighted average
- `offset`                     (n) vector, offset (or exposure), in terms of gamma (log exposure if the loss has a log-link)     
- `cutoff_paralellel`          [20_000] if x has more than these rows, a parallellized algorithm is called (which is slower for few forecasts)

# Output for standard models
- `yf`                         (n) vector of forecasts of y (or, outside regression, of the natural parameter), or scalar forecast if n = 1

# Output for hurdle models 
- `yf`                         (n) vector of forecasts of E(y|x) for the combined model
- `prob0`                      (n) vector of forecasts of prob(y=0|x)
- `prob0`                      (n) vector of forecasts of E(y|x,y /=0)

# Output for loss = :multiclass
- `yf`                        (n,num_class) matrix, yf[i,j] if the probability that observation i belongs to class j
- `class_values`              (num_clas) vector, class_value[j] is the value of y associated with class j
- `ymax`                      (n) vector, ymax[i] is the class value with highest probability at observation i. 

# Example of use
    output = HTBfit(data,param)
    yf     = HTBpredict(x_oos,output)
    yf     = HTBpredict(x_oos,output,best_model=true)
    yf     = HTBpredict(x_oos,output,offset = log.(exposure) )

    yf,prob0,yf_not0 = HTBpredict(x_oos,output)  # for hurdle models 
    yf,class_value,ymax = HTBpredict(x_oos,output)  # for multiclass 

"""
function HTBpredict(x0::Union{AbstractDataFrame,AbstractArray},output::NamedTuple;best_model=false,cutoff_parallel=20_000,predict=:Ey,offset=[])

    x = preparedata_predict(x0,output.HTBtrees)

    if best_model==true || length(output.w)==1
        gammafit = HTBpredict_internal(x,output.HTBtrees,predict,cutoff_parallel=cutoff_parallel,offset=offset)  # HTBtrees is for best model, HTBtrees_a collects all
    else
        gammafit = zeros(output.bestparam.T,size(x,1))

        for i in 1:length(output.w)

            if output.w[i]>0
                gammafit += output.w[i]*HTBpredict_internal(x,output.HTBtrees_a[i],predict,cutoff_parallel=cutoff_parallel,offset=offset)
            end

        end
    end
    
    return gammafit    # gammafit is actually Ey if predict = :Ey in HTBpredict_internal

end 


# HTBpredict when output is not NamedTuple (will typically be a vector of NamedTuple)
function HTBpredict(x::Union{AbstractDataFrame,AbstractArray},output;best_model=false,cutoff_parallel=20_000,predict=:Ey,offset=[])

    # output saves the loss functions of the component models: find the original loss
    loss1,loss2 = output[1].bestparam.loss,output[2].bestparam.loss

    if loss1==loss2==:logistic 
        loss = :multiclass
    elseif loss1==:logistic
        loss = :hurdlefamily  # could be :hurdleL2,:hurdleGamma,:hurdleL2loglink   
    end 

    if loss == :hurdlefamily 

        if predict==:Egamma; @error " prediction with hurdle models require predict=:Ey"; end 
        yf,prob0,yf_not0 = HTBpredict_hurdle(x,output,best_model,cutoff_parallel,predict,offset)
        return (yf,prob0,yf_not0)
    
    elseif loss == :multiclass 
        yf,class_values,ymax = HTBpredict_multiclass(x,output,best_model,cutoff_parallel,predict,offset)
        return yf,class_values,ymax 
    end 
    
end 


# prediction for hurdle model, where  param.loss in [:hurdleGamma,:hurdleL2loglink,:hurdleL2]
# The first model is logistic regression for prob, the second is :gamma of :L2 or :L2loglink.
# yf,prob0,yf_not0 = HTBpredict_hurdle  
function HTBpredict_hurdle(x,output,best_model,cutoff_parallel,predict,offset)

    yf_0     = HTBpredict(x,output[1],offset=[],best_model=best_model,cutoff_parallel=cutoff_parallel,predict=predict)   # offset is meant for 
    yf_not0  = HTBpredict(x,output[2],offset=offset,best_model=best_model,cutoff_parallel=cutoff_parallel,predict=predict)   # offset is meant for 

    yf    = @. yf_0 * yf_not0    # E(y|x)
    prob0  = @. 1 - yf_0          # prob(y=0)

    return yf,prob0,yf_not0 

end 


function HTBpredict_multiclass(x,output,best_model,cutoff_parallel,predict,offset)

    if predict == :Egamma 
        @error "predict for loss=:multiclass is only available as predict=:Ey (i.e. for probabilities) "
    end 

    class_values = output[1].bestparam.class_values 
    num_class    = length(class_values)
    prob = Matrix{output[1].bestparam.T}(undef,size(x,1),num_class)

    for i in 1:num_class
        prob[:,i] = HTBpredict(x,output[i],offset=[],best_model=best_model,cutoff_parallel=cutoff_parallel,predict=predict)
    end 

    prob = prob./sum(prob,dims=2)
    
    ymax = Vector{eltype(class_values)}(undef,size(x,1))

    for i in eachindex(ymax)
        ymax[i] = class_values[argmax(prob[i,:])]
    end     

    return prob,class_values,ymax 

end 



"""
    HTBcoeff(output;verbose=true)

Provides some information on constant coefficients for best model (in the form of a tuple.)
For example, error variance for :L2, dispersion and dof for :t.

# Inputs
- `output`                      output from HTBfit

# Output
- `coeff`                      named tuple with information on fixed coefficients (e.g. variance for :L2, dispersion and dof for :t)

# Example of use
    output = HTBfit(data,param)
    coeff  = HTBcoeff(output,verbose=false)
"""
function HTBcoeff(output;verbose=true)

    if output.bestparam.loss in [:hurdleGamma, :hurdleL2loglink, :hurdleL2]
        @error "HTBcoeff not yet supported for hurdle models"
    end     

    loss = output.bestparam.loss 
    coeff = output.bestparam.coeff_updated[1]

    if loss in [:logistic,:Poisson]
        θ    = (loss=loss,coeff="none")
    elseif loss in [:L2,:lognormal,:L2loglink] 
        θ    = (loss=loss,variance=coeff[1]^2)
    elseif loss == :t
        s2,v    = exp(coeff[1]),exp(coeff[2])
        θ    = (loss=loss,scale=s2,dof=v,variance="scale*dof/(dof-2)" )
    elseif loss == :Huber
        σ2,ψ = coeff[1]^2,coeff[2] 
        θ    = (loss=loss,variance=σ2,psi=output.bestparam.coeff_user[1])
    elseif loss == :gamma 
        k    = exp(coeff[1][1])
        θ    = (loss=loss,shape=k)
    elseif loss == :gammaPoisson
        α    = exp(coeff[1][1])
        θ    = (loss=loss,overdispersion=α,info="α=1/r in the negative binomial parameterization (r=number of successes). var(y)=μ(1+αμ) ")
    else 
        @error "loss not supported or misspelled. loss must be in [:logistic,:gamma,:L2,:Huber,:t,:quantile,:lognormal,:L2loglink,:Poisson,:gammaPoisson]. "
    end

    if verbose==true
        display(θ)  
    end

    return  θ

end     



# Prepare the data, which may come as a DataFrame and have missing and categorical, with the
# Same transformations as in HTBdata() and preparedataHTBfor convenient data manipulation
function preparedata_predict(x0::Union{AbstractDataFrame,AbstractArray},HTBtrees::HTBoostTrees)

    param = HTBtrees.param

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

    replace_nan_meanx!(x,param,HTBtrees.meanx)  # only for features NOT in categorical or missing_features
    
    x = prepares_categorical_predict(x,param)  # categoricals are mapped to target encoding values; new categories allowed 
                                               # columns are added if param.cat_representation_dimension>1
    x = (x .- HTBtrees.meanx)./HTBtrees.stdx
    x = convert_df_matrix(x,param.T)

    return x
end 



# This function is used only in HTBfit to compute residuals for cv_different_loss
function HTBpredict(x0::Union{AbstractDataFrame,AbstractArray},HTBtrees::HTBoostTrees;cutoff_parallel=20_000,predict=:Ey)

    x        = preparedata_predict(x0,HTBtrees)
    gammafit = HTBpredict_internal(x,HTBtrees,predict,cutoff_parallel=cutoff_parallel)  # HTBtrees is for best model, HTBtrees_a collects all

    return gammafit    # gammafit is actually Ey if predict = :Ey in HTBpredict_internal

end 


# This version takes in HTBdata type and HTBoostrees, assumes one model.
# Used only in one place, to produce forecasts within HTBfit. Get rid of it?   
function HTBpredict_internal(data::HTBdata,HTBtrees::HTBoostTrees,predict;cutoff_parallel=20_000)

    # Prepare the data, which may come as a DataFrame and have missing and categorical
    param = deepcopy(HTBtrees.param)
    x = copy(data.x)
    x = replace_nan_with_missing(x)
    convert_dates_to_real!(x,param,predict=true)   

    map_cat_convert_to_float!(x,param)      # categorical are now in the form 0,1,2...
    x = replace_missing_with_nan(x)         # SharedArray do not accept missing.
    # replace categoricals with target encoding values, and standardize all features
    x = preparedataHTB_test(x,param,HTBtrees.meanx,HTBtrees.stdx)
 
    if typeof(x)<:AbstractDataFrame
        x = convert_df_matrix(x,param.T)
    end

    if size(x,2)==1  
        x=convert(Matrix,reshape(x0,length(x0),1))
    end

    gammafit = HTBpredict_internal(x,HTBtrees,predict,cutoff_parallel=cutoff_parallel)  # HTBtrees is for best model, HTBtrees_a collects all

    return gammafit

end



# from natural parameter to E(y)
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
    elseif loss in [:gamma,:L2loglink,:Poisson,:gammaPoisson] 
        pred  = exp.(gammafit)    
    elseif loss == :lognormal
        σ    = coeff[1]
        pred = @. exp(gammafit + 0.5*σ^2)     
    else 
        @error "loss not supported or misspelled. loss must be in [:logistic,:L2,:Huber,:t,:quantile,:lognormal,:Poisson,:gammaPoisson]. "
    end

    return pred

end



"""
    HTBfit(data,param;cv_grid=[],cv_sparsity=:Auto,cv_depthppr=:Auto)

Fits HTBoost with with k-fold cross-validation of number of trees and depth, and possibly a few more models.

If param.modality is :fast or :fastest, fits one model, at param, and if needed a second where sharp splits are 
forced on features with high average values of τ. For param.modality=:accurate or :compromise,  
may then cross-validate the following hyperparamters:

1) Parameters for categorical features, if any.
2) depth in the range 1-6.
3) A penalization to encourage sparsity (fewer relevant features), unless n/p is large.
4) A model without projection pursuit regression (only if modality=:accurate)

The default range can be replaced by providing a vector cv_grid, e.g. 
  
    HTBfit(data,param,cv_grid = [2,4,6])

The sparsity penalization can be de-activated by setting cv_sparsity=false, e.g. 
    
        HTBfit(data,param,cv_sparsity=false)
        
If param.modality=:accurate, the learning rate lambda for all models is left at param.lambda (0.1 in default).
If modality=:compromise, lambda=0.2 is used in cv, and the best model is then refitted with lambda = param.lambda. 

Finally, all the estimated models considered are stacked, with weights chosen to minimize the cross-validated (original) loss.   
Unless modality=:accurate, this stacking will typically be equivalent to the best model.


# Inputs
- `data::HTBdata`
- `param::HTBparam`

# Optional inputs

- `cv_grid::Vector`         Defaul [2,3,5,6]. The code performs a search in the space depth in [2,3,5,6], trying to fit few models if possible. Provide a
                            vector to over-ride (e.g. [2,4])
- `cv_sparsity`             Default :Auto. Set to true to guarantee search over sparsity penalization or false to disactivate (and save computing time.)
                            In :Auto, whether the cv is performed or not depends on :modality, on the n/p ratio and on the signal-to-noise ratio. 
- `cv_depthppr`             true to cv whether to add projection pursuit regression. Default is true for modality=:accurate, else false.


# Output (named tuple, or vector of named tuple for hurdle models)

- `indtest::Vector{Vector{Int}}`  indexes of validation samples
- `bestvalue::Float`              best value of depth in cv_grid
- `bestparam::SAMRTparam`         param for best model  
- `ntrees::Int`                   number of trees (best value of param.ntrees) for best model
- `loss::Float`                   best cv loss
- `lossw::Float`                  loss of stacked models
- `meanloss:Vector{Float}`        mean cv loss at bestvalue of param for param.ntrees = 1,2,....
- `stdeloss:Vector{Float}`        standard errror of cv loss at bestvalue of param for param.ntrees = 1,2,....
- `lossgrid::Vector{Float}`       cv loss for best tree size for each grid value 
- `loglikdivide:Float`            loglikdivide. effective sample size = n/loglikvide. Roughly accounts for cross-correlation and serial correlation 
- `HTBtrees::HTBoostTrees`        for the best cv value of param and ntrees
- `HTBtrees_a`                    length(cv_grid) vector of HTBtrees
- `i`                             (ntrees,depth) matrix of threshold features for best model
- `mu`                            (ntrees,depth) matrix of threshold points  for best model
- `tau`                           (ntrees,depth) matrix of sigmoid parameters for best model
- `fi2`                           (ntrees,depth) matrix of feature importance, increase in R2 at each split, for best model
- `w`                             length(cv_grid) vector of stacked weights
- `ratio_actual_max`              ratio of actual number of candidate features over potential maximum. Relevant if sparsevs=:On: indicates sparsevs should be switched off if too high (e.g. higher than 0.5).
- `problems`                      true if there were computational problems in any of the models: NaN loss or loss jumping up

# Notes
- The following options for cross-validation are specified in param: randomizecv, nfold, sharevalidation, stderulestop

# Examples of use:
    param = HTBparam()
    data   = HTBdata(y,x,param)
    output = HTBfit(data,param)
    ntrees = output.ntrees 
    best_depth = output.bestvalue 

    Example for hudle models (loss in [:hurdleGamma,:hurdleL2loglink,:hurdleL2])
    ntrees_0    = output[1].ntrees   # number of trees for logistic regression, 0-not0
    ntrees_not0 = output[2].ntrees   # number of trees for gamma or L2 loss

"""
function HTBfit(data::HTBdata, param::HTBparam; cv_grid=[],cv_different_loss::Bool=false,cv_sharp::Bool=false,
    cv_sparsity=:Auto,cv_hybrid=true,cv_depthppr=:Auto,skip_full_sample=false)   # skip_full_sample enforces nofullsample even if nfold=1 (used in other functions, not by user)

    if param.loss in [:hurdleGamma,:hurdleL2loglink,:hurdleL2]
        output = HTBfit_hurdle(data,param,cv_grid=cv_grid,cv_different_loss=cv_different_loss,cv_sharp=cv_sharp,cv_sparsity=cv_sparsity,
                                cv_hybrid=cv_hybrid,cv_depthppr=cv_depthppr,skip_full_sample=skip_full_sample)
    elseif param.loss == :multiclass
        output = HTBfit_multiclass(data,param,cv_grid=cv_grid,cv_different_loss=cv_different_loss,cv_sharp=cv_sharp,cv_sparsity=cv_sparsity,
                                cv_hybrid=cv_hybrid,cv_depthppr=cv_depthppr,skip_full_sample=skip_full_sample)
    else 
        output = HTBfit_single(data,param,cv_grid=cv_grid,cv_different_loss=cv_different_loss,cv_sharp=cv_sharp,cv_sparsity=cv_sparsity,
                       cv_hybrid=cv_hybrid,cv_depthppr=cv_depthppr,skip_full_sample=skip_full_sample)
    end 

    return output 
end 

# Conditions to hybrid model, with sharp splits forced on features with high τ.
# Only if the high τ are for features with non-trivial importance (fi), and only if user did not specify sharp_splits
# Use: condition_sharp,force_sharp_splits = find_force_sharp_splits(HTBtrees,data,param,cv_hybrid)
# HTBtrees can be HTBtrees_a[i] or HTBtrees_a[argmin(lossgrid)]        
function find_force_sharp_splits(HTBtrees,data,param,cv_hybrid)

    if param.priortype==:smooth || cv_hybrid==false || !isempty(param.force_sharp_splits)
        return false,fill(false,1)
    end 

    force_sharp_splits = impose_sharp_splits(HTBtrees,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(HTBtrees,data)
    fi_sharp = sort(fi[force_sharp_splits.==true],rev=true)
    p_sharp = length(fi_sharp)

    if p_sharp>0    
        condition1 = fi_sharp[1]>10
        condition2 = sum(fi_sharp[1:minimum([3,p_sharp])])>15
        condition3 = sum(fi_sharp[1:minimum([10,p_sharp])])>20
        condition4 = sum(fi_sharp)>25
        condition_sharp = condition1 || condition2 || condition3 || condition4
    else 
        condition_sharp = false
        force_sharp_splits = fill(false,1)     
    end

    return condition_sharp,force_sharp_splits
end 



# HTBfit for a single model.  
function HTBfit_single(data::HTBdata, param::HTBparam; cv_grid=[],cv_different_loss::Bool=false,cv_sharp::Bool=false,
        cv_sparsity=:Auto,cv_hybrid=true,cv_depthppr=false,skip_full_sample=false)   # skip_full_sample enforces nofullsample even if nfold=1 (used in other functions, not by user)
    
    T,I = param.T,param.I

    n_additional_models  = 6   # Excluding hybrid with force_sharp_splits. After going through cv_grid, possibly twice(cv_hybrid), selects best value and potentially fits: depthppr=0, sparse (up to 3), sharp, different distribution

    modality              = param.modality
    param0                = deepcopy(param)

    if isempty(cv_grid)
        user_provided_grid = false
        cv_grid = [1,2,3,4,5,6,7]     # NB: later code assumes this grid when cv depth
    else     
        user_provided_grid = true
        if maximum(cv_grid)>7 && param.warnings==:On
            @warn "setting param.depth higher than 6, perhaps 7, typically results in very high computing costs."
        end
    end     

    lambda0 = param.lambda

    if modality == :compromise
        param0.lambda = max(param.lambda,param.T(0.2))
    end 

    if modality in [:fast,:fastest] 

        if user_provided_grid==false
            cv_grid = [param0.depth]
            user_provided_grid = true 
       end      

    end  

    if modality==:fastest
        param0.lambda = max(param.lambda,param.T(0.2))
        param0.nofullsample = true
        isempty(param0.indtrain_a) ? param0.nfold = 1 : nothing 

        if param.warnings==:On
            if isempty(param0.indtrain_a)
                @info "modality=:fastest is typically for preliminary explorations only. Setting param.nfold=1, param.nofullsample=true, lambda=$(param0.lambda).
                       Switch off this warning with param.warnings=:Off"
            else
                @info "modality=:fastest is typically for preliminary explorations only. Setting lambda = 0.2.
                Switch off this warning with param.warnings=:Off"
            end          
        end  
    end

    preliminary_cv!(param0,data)       # preliminary cv of categorical parameters, if modality is not :fast.

    cvgrid0 = deepcopy(cv_grid)            
    cvgrid  = vcat(cvgrid0,cvgrid0,fill(cvgrid0[1],n_additional_models)) # default,force_sharp_splits,n_additional_models

    treesize, lossgrid    = Array{I}(undef,length(cvgrid)), fill(T(Inf),length(cvgrid))  
    meanloss_a,stdeloss_a = Array{Array{T}}(undef,length(cvgrid)), Array{Array{T}}(undef,length(cvgrid))
    HTBtrees_a            = Array{HTBoostTrees}(undef,length(cvgrid))
    gammafit_test_a       = Vector{Vector{T}}(undef,length(cvgrid))
    y_test_a              = T[]
    indtest_a             = I[]
    problems_somewhere    = 0     

    param.randomizecv==true ? indices = shuffle(Random.MersenneTwister(param.seed_datacv),Vector(1:length(data.y))) : indices = Vector(1:length(data.y)) # done here to guarantee same allocation if randomizecv=true

    # Cross-validate depths, on user-defined grid.
    if user_provided_grid==true
        for (d,depth) in enumerate(cvgrid0)

            param = deepcopy(param0)
            param.depth = depth

            param_given_data!(param,data)
            param_constraints!(param)

            ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)

            i = d
            treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i],HTBtrees_a[i],gammafit_test_a[i],indtest_a,y_test_a = ntrees,loss,meanloss,stdeloss,HTBtrees1st,gammafit_test,indtest,y_test 
            problems_somewhere = problems_somewhere + problems

            # If needed, fit again with force_sharp_splits. Store in 2*i
            condition_sharp,force_sharp_splits = find_force_sharp_splits(HTBtrees_a[i],data,param,cv_hybrid)

            if condition_sharp                 

                i = length(cvgrid0) + d
                param = deepcopy(param0)
                param.depth = depth 
                param.force_sharp_splits = force_sharp_splits
        
                param_given_data!(param,data)
                param_constraints!(param)
        
                ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)
                treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]   = ntrees,loss,meanloss,stdeloss
                HTBtrees_a[i],gammafit_test_a[i]                    = HTBtrees1st,gammafit_test
                problems_somewhere = problems_somewhere + problems
                 
            end     

        end    
    end 

    # Cross-validate depths with no user-defined grid. NB: assumes cv_grid = [1,2,3,4,5,6,7]
    # If isempty(cv_grid), fits depth=3,4,5. If 3 is best, fits 1,2. If 4 is best, stops. If 5 is best, fits 6.
    # For each depth, also fit hybrid version with force_sharp_splits if needed

    if user_provided_grid==false
 
        modality==:compromise ? i_a = [3,5] : i_a = [3,4,5]   

        for _ in 1:2
            for d in i_a 

                param = deepcopy(param0)
                param.depth = cvgrid0[d]
                param_given_data!(param,data)
                param_constraints!(param)

                ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)

                i = d
                treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i],HTBtrees_a[i],gammafit_test_a[i],indtest_a,y_test_a = ntrees,loss,meanloss,stdeloss,HTBtrees1st,gammafit_test,indtest,y_test 
                problems_somewhere = problems_somewhere + problems

                # If needed, fit again with force_sharp_splits. Store in 2*i
                condition_sharp,force_sharp_splits = find_force_sharp_splits(HTBtrees_a[i],data,param,cv_hybrid)

                if condition_sharp                  # fit hybrid model 

                    i = length(cvgrid0) + d
                    param = deepcopy(param0)
                    param.depth = cvgrid0[d] 
                    param.force_sharp_splits = force_sharp_splits
        
                    param_given_data!(param,data)
                    param_constraints!(param)
        
                    ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)
                    treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]   = ntrees,loss,meanloss,stdeloss
                    HTBtrees_a[i],gammafit_test_a[i]                    = HTBtrees1st,gammafit_test
                    problems_somewhere = problems_somewhere + problems
                 
                end     

            end     

            if argmin(lossgrid) in [3,length(cvgrid0)+3]   #  depth = 3, either original tree or force_sharp_splits
                i_a = [1,2]                         
            elseif argmin(lossgrid) in [4,length(cvgrid0)+4]
                break
            else
               i_a = [6]
               # modality==:accurate ? i_a = [6,7] : i_a = [6]  # 7 can be very slow              
            end     
 
        end 
    end 

    # Additional models: Fit model with sparsity-inducing penalization, on best model fitted so far 

    best_i      = argmin(lossgrid)
    param       = deepcopy(HTBtrees_a[best_i].param)
    n,p         = size(data.x)   # not quite the correct #var if, as for categoricals, more are created by HTBdata, but good enough 

    sparsity_grid = T.([0.7,1.1,1.5])           

    if cv_sparsity == :Auto      

        yfit = HTBpredict(data.x,HTBtrees_a[best_i],predict=:Ey)    
        R2   = var(yfit)/var(data.y)
        ess50 = (n/param.loglikdivide)*(R2/(1-R2))       # approximate effective sample size corresponding to R2 at 50%.
        np_ratio = ess50/p 

        if modality in [:fastest,:fast]
            cv_sparsity = false 
        elseif modality == :accurate
            np_ratio>5_000 ? cv_sparsity = false : cv_sparsity = true  
        elseif modality == :compromise   # rule of thumb to decide whether it is worth searching over sparsity penalizations. Could be improved.
            np_ratio>1_000 ? cv_sparsity = false : cv_sparsity = true  
        end           

    end  

    if cv_sparsity==true 

        fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(HTBtrees_a[best_i],data)
        #s2 = sqrt(sum(fi.^2)/p)   # L2 measure of std
        #s1 = 1.25*mean(fi)        # L1 measure of std, here just 125/p
        # s2/s1= 1 for Gaussian, <1 is playtokurtic, and >1 if leptokurtic, suggesting sparsity. 
        
        for (j,sparsity_penalization) in enumerate(sparsity_grid)

            param.sparsity_penalization = sparsity_penalization
            param.exclude_features = fi .< (0.01/p)       # increase speed by excluding features that are irrelevant (even at default)  

            param_given_data!(param,data)
            param_constraints!(param)

            ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)

            if j<length(sparsity_grid)
                fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(HTBtrees1st,data)
            end     

            i = 2*length(cvgrid0)+j                                  
            cvgrid[i]   = cvgrid[best_i]
            treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]     = ntrees, loss, meanloss,stdeloss
            HTBtrees_a[i],gammafit_test_a[i]                      = HTBtrees1st, gammafit_test
            problems_somewhere = problems_somewhere + problems

            #If either 0.7 or 1.1 is better than 0.3, continue to 1.5. If neither is better, try 0.0.  
            if j>1 && min(lossgrid[i],lossgrid[i-1]) > lossgrid[best_i] # break (no need for more sparsity)
 
                param.sparsity_penalization = T(0)
                param.exclude_features = Vector{Bool}(undef,0)

                param_given_data!(param,data)
                param_constraints!(param)

                ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)

                i = 2*length(cvgrid0)+length(sparsity_grid) 
                cvgrid[i]   = cvgrid[best_i]
                treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]     = ntrees, loss, meanloss,stdeloss
                HTBtrees_a[i],gammafit_test_a[i]                      = HTBtrees1st, gammafit_test
                problems_somewhere = problems_somewhere + problems

                break
            end
        
        end

    end 


    # Additional model: no projection pursuit regression
    if cv_depthppr==:Auto
        modality==:accurate ? cv_depthppr = true : cv_depthppr = false 
    end     
 
    if cv_depthppr 
        i = 2*length(cvgrid0)+length(sparsity_grid) + 1
        cvgrid[i]  = HTBtrees_a[best_i].param.depth        
        param      = deepcopy(HTBtrees_a[best_i].param)
        param.depthppr = param.I(0)

        param_given_data!(param,data)
        param_constraints!(param)

        ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)
        treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]   = ntrees,loss,meanloss,stdeloss
        HTBtrees_a[i],gammafit_test_a[i]                    = HTBtrees1st,gammafit_test
        problems_somewhere = problems_somewhere + problems
    end 


    # Additional model: :sharptree, at previous best values of sparsity and depth.
    # VERY INEFFICIENT IMPLEMENTATION! 
    
    if param.priortype==:hybrid && cv_sharp 

        best_i      = argmin(lossgrid)
        param = deepcopy(HTBtrees_a[best_i].param)
        param.priortype = :sharp
        i = 2*length(cvgrid0)+length(sparsity_grid) + 2
        cvgrid[i]  = cvgrid[best_i]        

        ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)
        treesize[i],lossgrid[i],meanloss_a[i],stdeloss_a[i]   = ntrees,loss,meanloss,stdeloss
        HTBtrees_a[i],gammafit_test_a[i]                    = HTBtrees1st,gammafit_test
        problems_somewhere = problems_somewhere + problems

    end 

    # Before trying a different distribution, store param from the best solution. This will be used in 
    # HTBmodelweights, which for some loss functions involving additional coefficients (:t,:gamma,...)
    # needs values of these coefficients, which param0 does not have.
    bestparam_original_loss = HTBtrees_a[argmin(lossgrid)].param

    # Additional model: try a different distribution (loss), at previous best values of depth and sparsity
    best_i = argmin(lossgrid)
    param  = deepcopy(HTBtrees_a[best_i].param)

    if cv_different_loss 
        if param.loss == :L2   # fit a t distribution to residuals, and leave :L2 unless dof<10. 
 
            yfit = HTBpredict(data.x,HTBtrees_a[best_i],predict=:Ey) 
            res = Newton_MAP(data.y - yfit,gH_student,start_value_student,w=data.weights)
            dof = exp(res.minimizer[2])

            dof<10 ? param.loss = :t : nothing 
            #param.losscv = :mse     # Not needed if this is the last model fitted. 
        end 
    end         

    # Experimental. Not well tested yet. 
    # NB: cv loss may NOT comparable across different distributions.
    if cv_different_loss==true && (param.loss != HTBtrees_a[best_i].param.loss) 
 
        param_given_data!(param,data)
        param_constraints!(param)

        ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)

        i = 2*length(cvgrid0)+length(sparsity_grid) + 3
        cvgrid[i]   = cvgrid[best_i]
        lossgrid[i],meanloss_a[i], stdeloss_a[i] = loss,meanloss,stdeloss  # The loss is NOT comparable. 
        treesize[i]     = ntrees
        HTBtrees_a[i],gammafit_test_a[i] = HTBtrees1st, gammafit_test
        problems_somewhere = problems_somewhere + problems
    
    end    

    # if modality==:compromise, fits the best model with param0.lambda at original value and replaces it.
    # If model with lowest loss does not have the user-specified distribution, takes the model with the highest stacking weight 
    best_i = argmin(lossgrid)

    if HTBtrees_a[best_i].param.loss != param0.loss
        w,lossw = HTBmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
        best_i = argmax(w)
    end     
    
    param       = deepcopy(HTBtrees_a[best_i].param)
               
    if modality==:compromise
        
        param.lambda = lambda0

        param_given_data!(param,data)
        param_constraints!(param)

        ntrees,loss,meanloss,stdeloss,HTBtrees1st,indtest,gammafit_test,y_test,problems = HTBsequentialcv(data,param,indices=indices)
        gammafit_test = from_gamma_to_Ey(gammafit_test,param,:Ey)    #  gammafit_test now comparable.

        i = best_i   # replaces the best model

        if loss<lossgrid[i]   # exceptions may happen when interacting with sparsity_penalization, which is ideally calibrated at the final lambda
            lossgrid[i],meanloss_a[i],stdeloss_a[i] = loss,meanloss,stdeloss 
            treesize[i]     = ntrees
            HTBtrees_a[i],gammafit_test_a[i] = HTBtrees1st, gammafit_test
            problems_somewhere = problems_somewhere + problems
        end     

    end 

    # If there is a NaN in lossgrid, Julia takes it as the minimum, hence...
    if isnan(minimum(lossgrid))
        @warn "In HTBfit, some output is NaN. "
        problems_somewhere = problems_somewhere + 1 
    end

    lossgrid = replace( lossgrid,T(NaN)=>T(Inf) )

    # select model with lowest loss. If this does not have the user-specified distribution, takes the model with the highest stacking weight 
    best_i = argmin(lossgrid)

    if HTBtrees_a[best_i].param.loss != param0.loss
        w,lossw = HTBmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
        best_i = argmax(w)
    end     

    bestvalue,ntrees,loss,meanloss,stdeloss = cvgrid[best_i],treesize[best_i],lossgrid[best_i],meanloss_a[best_i],stdeloss_a[best_i]

    # Fit again on the full sample (unless nofullsample=true and nfold==1). This is done for all the values in cvgrid
    # with weight>=0.1 (to reduce computing time). 
    if (param0.nofullsample==false || param0.nfold>1) && skip_full_sample==false

        w,lossw = HTBmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
        w[best_i]=max(w[best_i],0.1)   # the best model should be refitted
        w = w.*(w .>= 0.1)
        w = w/sum(w)
    
        m = max(1,log(length(data.y))/log( length(data.y)-length(y_test_a)/param.nfold))  # m = avg(n_train+n_test)/avg(n_train), avg over nfold
        ntrees = Int(floor(ntrees*m))                                             # small adjustment for n>n_train. Tiny effect.

        for i in 1:length(cvgrid)    
        
            if w[i]>0

                param = deepcopy(HTBtrees_a[i].param)    
                param.ntrees   = Int(floor(treesize[i]*m))

                param_given_data!(param,data)
                param_constraints!(param)

                HTBtrees_a[i]  = HTBbst(data,param)
            end

        end
        HTBtrees = HTBtrees_a[best_i]
    else
        HTBtrees = HTBtrees_a[best_i]
    end

    # Ensembles of stacked trees.
    if sum(lossgrid.<Inf)==1
        lossw = loss
        w = zeros(T,length(lossgrid))
        w[argmin(lossgrid)] = T(1)
    else
        w,lossw = HTBmodelweights(lossgrid,y_test_a,indtest_a,gammafit_test_a,data,bestparam_original_loss)
    end

    # provide some additional output
    i,μ,τ,fi2=HTBoutput(HTBtrees)  # on the best value
    avglntau,varlntau,mselntau,postprob2 = tau_info(HTBtrees)
    ratio_actual_max = tight_sparsevs(ntrees,HTBtrees.param) # ratio of actual vs max number of candidate features

    for i in eachindex(lossgrid)   # done to trigger warning if sparsevs seems too tight in ANY of the model
        if lossgrid[i]<Inf
            aux = tight_sparsevs(treesize[i],HTBtrees_a[i].param)
        end 
    end

    additional_info = [[T(NaN)]]

    return ( indtest=indtest_a,bestvalue=bestvalue,bestparam=HTBtrees.param,ntrees=ntrees,loss=loss,meanloss=meanloss,stdeloss=stdeloss,lossgrid=lossgrid,loglikdivide=HTBtrees.param.loglikdivide,HTBtrees=HTBtrees,
    i=i,mu=μ,tau=τ,fi2=fi2,avglntau=avglntau,HTBtrees_a=HTBtrees_a,w=w,lossw=lossw,problems=(problems_somewhere>0),ratio_actual_max=ratio_actual_max,additional_info=additional_info)

end


# Hurdle models for zero-inflated continuous data:
# a logistic regression for 0-not0, coupled with :gamma, :L2loglink or :L2 loss for y /=0 data (a subset).
# The two models can be fit separately with no loss since the continuous distribution has zero mass at 0. 
# The cv is completely independent for the two models. 
function HTBfit_hurdle(data::HTBdata, param::HTBparam; cv_grid=[],cv_different_loss::Bool=false,cv_sharp::Bool=false,
    cv_sparsity=true,cv_hybrid=true,cv_depthppr=false,skip_full_sample=false)   # skip_full_sample enforces nofullsample even if nfold=1 (used in other functions, not by user)

    if param.loss == :hurdleGamma
        loss_not0 = :gamma
    elseif param.loss == :hurdleL2loglink 
        loss_not0 = :L2loglink
    elseif param.loss == :hurdleL2 
        loss_not0 = :L2
    end 

    # 0-not0 
    T           = param.T 
    data_0      = deepcopy(data)
    @. data_0.y = data.y != 0      
    @. data_0.offset = T(0)          # offset is intended for the y>0 part. 

    param_0      = deepcopy(param)
    param_0.loss = :logistic
    output_0 = HTBfit_single(data_0,param_0,cv_grid=cv_grid,cv_different_loss=cv_different_loss,cv_sharp=cv_sharp,cv_sparsity=cv_sparsity,
                            cv_hybrid=cv_hybrid,skip_full_sample=skip_full_sample) 
    data_0 = 0  # free memory 

    # y /=0  
    data_not0 = HTBdata_subset(data,param,data.y .!= 0)
    param_not0 = deepcopy(param)
    param_not0.loss = loss_not0 
    output_not0 = HTBfit_single(data_not0,param_not0,cv_grid=cv_grid,cv_different_loss=cv_different_loss,cv_sharp=cv_sharp,cv_sparsity=cv_sparsity,
                       cv_hybrid=cv_hybrid,cv_depthppr=cv_depthppr,skip_full_sample=skip_full_sample) 

                        
    output = [output_0,output_not0]

    return output 
end 


function HTBfit_multiclass(data::HTBdata, param::HTBparam; cv_grid=[],cv_different_loss::Bool=false,cv_sharp::Bool=false,
    cv_sparsity=true,cv_hybrid=true,cv_depthppr=false,skip_full_sample=false)   # skip_full_sample enforces nofullsample even if nfold=1 (used in other functions, not by user)

    num_classes  = length(param.class_values)

    output = Vector(undef,num_classes)
    y0     = copy(data.y) 
    param_i = deepcopy(param)
    param_i.loss = :logistic
    T       = param.T

    for i in eachindex(param.class_values)

        new_class_value = T(i-1)          # original class values converted to 0,1,2...
        @. data.y = y0 == new_class_value    
        output[i] = HTBfit_single(data,param_i,cv_grid=cv_grid,cv_different_loss=cv_different_loss,cv_sharp=cv_sharp,cv_sparsity=cv_sparsity,
                         cv_hybrid=cv_hybrid,cv_depthppr=cv_depthppr,skip_full_sample=skip_full_sample)      
    end 
 
    @. data.y = y0  

 return output 
end 



function HTBrelevance(HTBtrees::HTBoostTrees,data::HTBdata )

    fi2 = deepcopy(HTBtrees.fi2) 
    @. fi2 = abs( fi2*(fi2>=0) )   # Ocassional (tiny) negative numbers set to zero

    fi         = sqrt.(fi2)
    fi         = 100*fi./sum(fi)
    sortedindx = sortperm(fi,rev = true)
    fnames     = data.fnames

    return fnames,fi,fnames[sortedindx],fi[sortedindx],sortedindx
end


"""
    HTBrelevance(output,data::HTBdata;verbose=true,best_model=false)

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
    output = HTBfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose = false)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,best_model=true)

"""
function HTBrelevance(output,data::HTBdata;verbose=true,best_model=false )

    if output.bestparam.loss in [:hurdleGamma, :hurdleL2loglink, :hurdleL2]
        @error "HTBrelevance not yet supported for hurdle models"
    end     

    T   = output.bestparam.T 
    w   = output.w

    if best_model==true   # HTBtrees for best model, HTBtrees_a for all
        fi2 = deepcopy(output.HTBtrees.fi2) 
        @. fi2 = abs( fi2*(fi2>=0) )   # Ocassional (tiny) negative numbers set to zero
    else
        fi2 = zeros(T,size(data.x,2))

        for i in eachindex(w)

            if w[i]>0
                fi2 = deepcopy(output.HTBtrees_a[i].fi2)
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
    HTBpartialplot(data::HTBdata,output,features::Vector{Int64};,predict=:Egamma,best_model=false,other_xs::Vector=[],q1st=0.01,npoints=1000))
Partial dependence plot for selected features. Notice: Default is for natural parameter (gamma) rather than y.
For feature i, computes gamma(x_i) - gamma(x_i=mean(x_i)) for x_i between q1st and 1-q1st quantile, with all other features at their mean.

# Inputs

- `data::HTBdata`
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
    output = HTBfit(data,param)
    q,pdp  = HTBpartialplot(data,output.HTBtrees,sortedindx[1,2],q1st=0.001)
"""
function HTBpartialplot(data::HTBdata,output,features;best_model=false,other_xs::Vector =[],q1st=0.01,npoints = 1000,predict=:Egamma)

    if output.bestparam.loss in [:hurdleGamma, :hurdleL2loglink, :hurdleL2]
        @error "HTBpartialplot not yet supported for hurdle models"
    end     

    # data.x is SharedMatrix, not standardized, categoricals are 0,1,2 .... dates are [0,1]
    # replace categoricals with target encoding values, and standardize all features
    x = preparedataHTB_test(data.x,output.HTBtrees.param,output.HTBtrees.meanx,output.HTBtrees.stdx)

    if best_model==true || length(output.HTBtrees_a)==1
        q,pdp  = HTBpartialplot(x,output.HTBtrees,features,predict,other_xs=other_xs,q1st=q1st,npoints=npoints)
    else
        T = output.bestparam.T
        pdp = zeros(T,npoints,length(features) )

        for i in 1:length(output.HTBtrees_a)
            if output.w[i]>0
                q,pdp_i=HTBpartialplot(x,output.HTBtrees_a[i],features,predict,other_xs=other_xs,q1st=q1st,npoints=npoints)
                pdp  += output.w[i]*pdp_i
            end
        end
    end

    meanx,stdx = output.HTBtrees.meanx,output.HTBtrees.stdx
    q = q.*stdx[features]' .+ meanx[features]'      # convert back to original scale

    return q,pdp
end



# x is standardized (and previously categoricals replaced by target encoding values)
function HTBpartialplot(x::AbstractArray,HTBtrees::HTBoostTrees,features,predict;other_xs::Vector =[],q1st=0.01,npoints = 1000)

    T = HTBtrees.param.T
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
        pdp[:,i] = HTBpredict_internal(h,HTBtrees,predict) - HTBpredict_internal(h0,HTBtrees,predict)
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
    HTBmarginaleffect(data::HTBdata,output,features::Vector{Int64};predict=:Egamma,best_model=false,other_xs::Vector =[],q1st=0.01,npoints=50,epsilon=0.02)
APPROXIMATE Computation of marginal effects using NUMERICAL derivatives (default ϵ=0.01)

# Inputs

- `data::HTBdata`
- `HTBtrees::HTBoostTrees`
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
    output = HTBfit(data,param)
    q,m    = HTBmarginaleffect(data,output.HTBtrees,[1,3])

# Example
    q,m  = HTBmarginaleffect(data,output.HTBtrees,[1,2,3,4],other_xs = zeros(p),npoints = 1)

# Example
    output = HTBfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output.HTBtrees,data,verbose=false)
    q,m  = HTBmarginaleffect(data,output,sortedindx[1,2],q1st=0.001)

"""
function HTBmarginaleffect(data::HTBdata,output,features;best_model=false,other_xs::Vector =[],q1st=0.01,npoints = 50,epsilon=0.02,predict=:Egamma)

    if output.bestparam.loss in [:hurdleGamma, :hurdleL2loglink, :hurdleL2]
        @error "marginal effects not yet supported for hurdle models"
    end     

    if output.HTBtrees.param.priortype !== :smooth
        @warn "Derivatives computed in HTBmarginaleffects may not be defined unless param.priortype=:smooth"
    end    

    x = preparedataHTB_test(data.x,output.HTBtrees.param,output.HTBtrees.meanx,output.HTBtrees.stdx)

    if best_model==true || length(output.HTBtrees_a)==1
        q,m  = HTBmarginaleffect(x,output.HTBtrees,features,predict,q1st=q1st,npoints=npoints,epsilon=epsilon)
    else
        T = typeof(data.y[1])
        m = zeros(T,npoints,length(features) )

        for i in 1:length(output.HTBtrees_a)
            if output.w[i]>0
                q,m_i  = HTBmarginaleffect(x,output.HTBtrees_a[i],features,predict,q1st=q1st,npoints=npoints,epsilon=epsilon)
                m    = m + output.w[i]*m_i
            end
        end

    end

    meanx,stdx = output.HTBtrees.meanx,output.HTBtrees.stdx
    q = q.*stdx[features]' .+ meanx[features]'      # convert back to original scale

    return q,m
end


# expects x to be standardized, with categoricals replaced by target encoding values
function HTBmarginaleffect(x::AbstractArray,HTBtrees::HTBoostTrees,features,predict;other_xs::Vector =[],q1st=0.01,npoints = 50,epsilon=0.01)

    param = HTBtrees.param
    I = param.I
    T = HTBtrees.param.T
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
            d[i] = (( HTBpredict_internal(h1,HTBtrees,predict) - HTBpredict_internal(h2,HTBtrees,predict) )/T(2*epsilon))[1]
        end

        return q,d

    else
        q,pdp   = HTBpartialplot(x,HTBtrees,features,predict,other_xs = other_xs,q1st = q1st,npoints = npoints+2)
        n       = size(q,1)
        d       = (pdp[1:n-2,:] - pdp[3:n,:])./(q[1:n-2,:] - q[3:n,:] )  # numerical derivatives at q[i]: f(q[i+1]-f(q[i-1])/(q[i+1]-q[i-1]) )
        return q[2:n-1,:],d
    end

end



"""

    HTBoutput(HTBtrees::HTBoostTrees;exclude_pp = true)

Output fitted parameters estimated from each tree, collected in matrices. Excluded projection pursuit regression parameters.

# Inputs 
- The default exclude_pp does not give μ and τ for projection pursuit regression. 

# Output
- `i`         (ntrees,depth) matrix of threshold features
- `μ`         (ntrees,depth) matrix of threshold points
- `τ`         (ntrees,depth) matrix of sigmoid parameters
- `fi2`       (ntrees,depth) matrix of feature importance, increase in R2 at each split

# Example of use
output = HTBfit(data,param)
i,μ,τ,fi2 = HTBoutput(output.HTBtrees)

"""
function HTBoutput(HTBtrees::HTBoostTrees;exclude_pp = true)

    I = typeof(HTBtrees.param.depth)
    T = typeof(HTBtrees.param.lambda)
    ntrees = length(HTBtrees.trees)
    d = length(HTBtrees.trees[1].τ)

    i   = Matrix{I}(undef,ntrees,d)
    μ   = Matrix{T}(undef,ntrees,d)
    τ   = Matrix{T}(undef,ntrees,d)
    fi2 = Matrix{T}(undef,ntrees,d)

    for j in 1:ntrees
        tree = HTBtrees.trees[j]
        i[j,:],μ[j,:],τ[j,:],fi2[j,:] = tree.i,tree.μ,tree.τ,tree.fi2
    end

    # delete the columns that refer to projection pursuit regression
    depthppr = HTBtrees.param.depthppr 
    depth   = HTBtrees.param.depth

    if depthppr>0 && exclude_pp
        i = i[:,1:depth]
        μ = μ[:,1:depth]
        τ = τ[:,1:depth]
        fi2 = fi2[:,1:depth]
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

    HTBweightedtau(output,data;verbose=true,best_model=false)

Computes weighted (by variance importance gain at each split) smoothing parameter τ for each
feature, and for the entire model (features are averaged by variance importance)
statistics for each feature, averaged over all trees. Sharp thresholds (τ=Inf) are bounded at 40.
best_model=true for single model with lowest CV loss, best_model= false for weighted average (weights optimized by stacking)

# Input 
- `output`   output from HTBfit
- `data`     data input to HTBfit

# Optional inputs 
- `verbose`   [true]  prints out the results to screen as DataFrame

# Output
- `avgtau`         scalar, average importance weighted τ over all features (also weighted by variance importance) 
- `avg_explogtau`  scalar, exponential of average importance weighted log τ over all features (also weighted by variance importance) 
- `avgtau_a`       p-vector of avg importance weighted τ for each feature 
- `df`             dataframe collecting avgtau_a information (only if verbose=true)
- `x_plot`         x-axis to plot sigmoid for avgtau, in range [-2 2] for standardized feature 
- `g_plot`         y-axis to plot sigmoid for avgtau 

# Example of use

output = HTBfit(data,param)
avgtau,avg_explogtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data)
avgtau,avg_explogtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data,verbose=false,plot_tau=false,best_model=true)

using Plots
plot(x_plot,g_plot,title="avg smoothness of splits",xlabel="standardized x",label=:none,legend=:bottomright)
"""
function HTBweightedtau(output,data;verbose::Bool=true,best_model::Bool=false)

    T = Float64

    HTBtrees = output.HTBtrees

    p = max(length(HTBtrees.infeatures),length(HTBtrees.meanx))  # they should be the same ...
    
    if best_model==true
        avgtau_a = mean_weighted_tau(HTBtrees)
    else     
        avgtau_a = zeros(p)
        w        = output.w
        
        for i in eachindex(w)
            if w[i]>0 
                avgtau_a += w[i]*mean_weighted_tau(output.HTBtrees_a[i])
            end 
        end 
    end     

    fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false,best_model=best_model)
    
    avgtau  = sum(avgtau_a.*fi)/sum(fi)
    exp_avglogtau = exp( sum(log.(avgtau_a).*fi)/sum(fi) )

    df = DataFrame(feature = fnames, importance = fi, avgtau = avgtau_a,
           sorted_feature = fnames_sorted, sorted_importance = fi_sorted, sorted_avgtau = avgtau_a[sortedindx])

    if verbose==true
        df = DataFrame(feature = fnames, importance = fi, avgtau = avgtau_a,
        sorted_feature = fnames_sorted, sorted_importance = fi_sorted, sorted_avgtau = avgtau_a[sortedindx])
        display(df)
        println("\n Average smoothing parameter τ is $(round(avgtau,digits=1)).")
        println("\n In sufficiently large samples, and if modality=:compromise or :accurate")
        println("\n - Values above 20-25 suggest little smoothness in important features. HTBoost's performance may slightly outperform or slightly underperform other gradient boosting machines.")
        println(" - At 10-15 or lower, HTBoost should outperform other gradient boosting machines, or at least be worth including in an ensemble.")
        println(" - At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.")

    else 
        df = nothing     
    end 

    x_plot = collect(-2.0:0.01:2)
    g_plot = sigmoidf(x_plot,0.0,avgtau,output.bestparam.sigmoid)

    return T(avgtau),T(exp_avglogtau),T.(avgtau_a),df,x_plot,g_plot

end 


# Visualize impact of projection pursuit.
# Use: 
# if depthppr>0
#    yf1,yf0 = HTBplot_ppr(output,which_tree=1)
#    plot(yf0,yf1,title="depthppr=$(param.depthppr)")
# end
# 
# where yf0 is the standardized prediction from the tree, and yf1 is the (non-standardized) prediction after ppr      
function HTBplot_ppr(output;which_tree=1)

    t = output.HTBtrees.trees[which_tree]
    param = output.bestparam
    depthppr = param.depthppr

    T  = Float32
    xi = T.([0.01*i for i in -300:300])

    n  = length(xi)
    G0 = ones(T,n)
    G   = Matrix{T}(undef,n,2*param.depthppr)

    for d in 1:depthppr 
        μ  = t.μ[param.depth+d]
        τ =  t.τ[param.depth+d]
        G   = Matrix{T}(undef,n,2*size(G0,2))
        gL  = sigmoidf(xi,μ,τ,param.sigmoid)
        updateG!(G,G0,gL)
        G0 = copy(G) 
    end

    β =  t.β[end]

    return G*β,xi
end 


# Use: force_sharp_splits = impose_sharp_splits(HTBtrees_a[best_i],param)
function impose_sharp_splits(HTBtrees::HTBoostTrees,param) 

    avgtau = mean_weighted_tau(HTBtrees)
    force_sharp_splits = avgtau .> param.tau_threshold
    
    return force_sharp_splits 
end