#
# Data preprocessing functions
#
#
# The following functions are applied once, on the first HTBdata() 
#  
# store_class_values!()
# replace_nan_with_missing()
# replace_missing_with_nan()
# convert_dates_to_real!()                          
# categorical_features!()
# missing_features!
# missing_features_extend_x()                          
# map_cat_convert_to_float!()              
# check_admissible_data()                check if data is admissable given param
#
# The following functions are applied to each train set: 
#
# preparedataHTB()                     preliminary operations on data before starting boosting loop: standardize x using robust measures of dispersion.
# preparegridsHTB()
# gridvectorτ()
# gridmatrixμ()
# kantorovic_distance()                   rough approximate distance between the empirical distribution of xi and y, or between xi and a Gaussian, using deciles
#

# in HTBdata(), if loss=:multiclass, stores unique values of y as param.class_values, and then replaces y by 0,1,2... Leaves missing as missing
function store_class_values!(param,y)

    if param.loss !== :multiclass 
        return y
    end 

    class_values       = sort(unique(skipmissing(y)))
    eltype(class_values) <: Number ? class_values = param.T.(class_values) : nothing 
    param.class_values = class_values
    new_values         = param.T.(collect(0:1:length(param.class_values)-1))

    n     = length(y)
    y_new = Vector{param.T}(undef,n)

    for (j,class_value) in enumerate(param.class_values)
        new_value = new_values[j]

        for i in 1:n
            y[i] == class_value ? y_new[i] = new_value : nothing 
        end 
    end  

    return y_new         

end 


# replace NaN by missing  
function replace_nan_with_missing(x0)

    x = copy(x0)

    if typeof(x) <: AbstractDataFrame
        for col in names(x)                       # Iterate through the DataFrame and replace NaN with missing while preserving the data type
            if eltype(x[!, col]) <: Number
                x[!, col] .= ifelse.(isnan.(x[!, col]), missing, x[!, col])
            end
        end
    else
        x = replace(x0, NaN => missing)  
    end

    return x

end 



function replace_missing_with_nan(x)

    if typeof(x) <: AbstractDataFrame
        x .= coalesce.(x, NaN)
    else
        x = replace(x, missing => NaN)
    end

    return x  
end 



function convert_dates_to_real!(x,param::HTBparam;predict=false)
end    



# convert dates in x to [0 1], and save a named tuple in param.infodate 
function convert_dates_to_real!(x::AbstractDataFrame,param::HTBparam;predict=false)

    if predict==false

        for i in 1:size(x,2)

            if eltype(x[:,i])==Date
                t_first = minimum(skipmissing(x[!,i]))
                t_last  = maximum(skipmissing(x[!,i]))
                x[!,i] = (x[!,i] - t_first)/(t_last - t_first)  # now a vector of floats between 0 and 1
                t_column = i
                param.info_date = (date_column=t_column,date_first=t_first,date_last=t_last) 
                break                  # at most one date allowed
            end
    
        end

    else    # param.info_date already populated 

        i = param.info_date.date_column

        if i>0 
            if eltype(x[:,i])==Date 
                x[!,i] = (x[!,i] .- param.info_date.date_first)/(param.info_date.date_last - param.info_date.date_first)  # now a vector of floats between 0 and 1
            elseif minimum(x[!,i]) != 0 || maximum(x[!,i]) != 1
                @warn "date column is a Float but not in [0,1]"
            end
        end    

    end

end


# add columns to x, for those columns with missing or NaN values
function missing_features_extend_x(param::HTBparam,x::AbstractDataFrame)

    T   = param.T

    for j in param.missing_features

        column_name = names(x)[j]*"_missing"
        dummy_vec = [ismissing(x[i,j]) || isnan(x[i,j]) ? T(1) : T(0) for i in 1:size(x)[1]]

       x[!, column_name] = dummy_vec

    end    

    fnames = names(x)

    return x,fnames
end         


# fills param.cat_features_extended, vector of categorical features requiring an extended representation
# also modifies fnames 
function extended_categorical_features!(param,fnames) 

    for (i,j) in enumerate(param.cat_features)
        if length(param.cat_dictionary[i]) > 2   # dummies (two categories) are left at 0,1 
            push!(param.cat_features_extended,j)    # keep track of extended cat features

            for c in 1:(param.cat_representation_dimension-1)
                if c==1; name_ext = "_cat_freq"; elseif c==2; name_ext = "_cat_var"; else; name_ext = "cat_ext"*"$c"; end
                fnames = push!(fnames,fnames[j]*name_ext)
            end
    
        end     
    end     

end    




# fills param.missing_features (to a vector of I). Only non-categorical features can be missing.
function missing_features!(param::HTBparam,x::AbstractDataFrame)

    for j in 1:size(x,2)
        if any(ismissing.(x[!,j]))  # missing in non-cat features treated as a category
            push!(param.missing_features,j)  
        end
    end

end         




# Modifies param.cat_features (to a vector of I) .
# If user provides param.cat_features, it can be a vector of integers or String or Symbols.  
# Otherwise, it assumes:
# - any type not in Real is a categorical features.
# - floating numbers are not categorical features.
function categorical_features!(param::HTBparam,x::AbstractDataFrame)

    n,p = size(x)
    I = typeof(param.ntrees)

    if param.cat_features == [0]   # user forces no cat features
        param.cat_features = Vector{I}(undef,0)
        return
    end     

    if isempty(param.cat_features)    # user did not provide info => detect categorical features automatically as strings and CategoricalArrays
        param.cat_features = Vector{I}(undef,0)

        for i in 1:p

            if eltype(x[!,i]) <: Union{Number,Missing}  # floating numbers cannot be categorical
            else
                push!(param.cat_features,i)
            end

        end 

    else
        
        if eltype(param.cat_features) == I
            return
        end

        pos = Vector{I}(undef,0)

        for i in 1:p    # cat_features is a vector of String or Symbol
 
            if (String(names(x)[i]) in String.(param.cat_features))  # allows cat_features to be a vector of I or String or Symbols
                push!(pos,i)
            end

        end 

        param.cat_features = pos

    end    
end



# converts categorical features (as from param.cat_features) to 0,1,... by mapping each unique value to a number,
# and saves the mapping in param.cat_dictionary
# missing are converted to "missing" for strings, and to NaN for numbers
function map_cat_convert_to_float!(x::AbstractDataFrame,param::HTBparam;create_dictionary::Bool=false)

    T = param.T

    for (i,j) in enumerate(param.cat_features)

        z = x[!,j]
        Tz = eltype(z)

        if Tz <: Union{AbstractString,Missing}
            z = replace(z,missing => "missing")
        elseif Tz <: Union{Bool,Missing}
            z = replace(z,missing => "missing",true => "true", false => "false")
        elseif Tz <: Union{Real,Missing}
            z = replace(z,missing => T(NaN) )
        elseif Tz <: Union{Missing,CategoricalValue}
            z = replace(z,missing => "missing")
        else 
            @error "map_cat_convert_to_float!(): categorical features must be of type String,Real,Bool, or Categorical"    
        end

        if create_dictionary
            z = CategoricalArray(z)
            L = levels(z)         #                   # retrieve all the possible values
            D = Dict(L[i] => T(i-1) for i in 1:length(L))  # create a dictionary mapping each value to a number
            push!(param.cat_dictionary,D)
        else
            D = param.cat_dictionary[i]
        end         

        x[!,j] = map(x -> D[x],z)           # convert each categorical value to a number using the dictionary

    end     

end 


# check if data is admissable given param
function check_admissible_data(y,param)

    if param.loss==:logistic
        if minimum(y) < 0 || maximum(y) > 1
            @warn "data.y (label) must be in [0,1] for loss = :logistic"
        end
        lu = length(unique(y))
        if lu < 2
            @error "data.y takes only one value"
        elseif lu > 2
            @warn "data.y takes $lu unique values, while loss=:logistic requires two. Switch to loss=:multiclass"
        end         
    end

    if param.loss in [:gamma,:lognormal]
        if minimum(y) <= 0
            @error "data.y must be strictly positive for loss = $(param.loss) "
        end 
    elseif param.loss in [:L2loglink,:Poisson,:gammaPoisson]
        if minimum(y) < 0 
            @error "data.y must be non-negative for loss = $(param.loss)"
        end     
    end

end



# Unlike the first version of the code, I use 1.25*mean|z| instead of 1.42*median|z|, because the second measure breaks down (goes to 0) with very sparse data.
# In the first version of the paper and code, I computed the robust std only on non-zero values, now I computed on all values,
# which implies a prior of much sharper, less smooth functions on sparse features. Mixed discrete-continuous features therefore are given a much weaker prior.
function robust_mean_std(x::AbstractMatrix{T}) where T <: AbstractFloat # where x = data.x

    p = size(x,2)
    meanx,stdx = Matrix{T}(undef,1,p),Matrix{T}(undef,1,p)

    for i in 1:p
        meanx[i],stdx[i] = robust_mean_std(x[:,i])
    end

    return meanx,stdx

end


# compute robust measures of centrality and dispersion.
function robust_mean_std(x0::AbstractVector{T}) where T <: AbstractFloat 

    x   = filter(!isnan,x0)

    meanx =  mean(x)  
    stdxL2 = std(x)      # this alone is very poor with sparse data in combinations with default priors on mu and tau (x/stdx becomes very large)
    stdxL1 = copy(stdxL2)

    unique_values = unique(x)

    if length(unique_values)>2     
        meanx  = median(x)  # median rather than mean
        stdxL1 = (T(1.25)*mean(abs.(x .- meanx)))
        stdx   = minimum(vcat(stdxL2,stdxL1))  # conservative, take smallest measure of dispersion
    elseif length(unique_values)==2
        stdx = maximum(abs.(unique_values ))
    else   # only one unique value: set stdx>0 to avoid any trouble when normalizing
        stdx = max(unique_values[1],T(1))     
    end

    return meanx,stdx

end



# replace nan with mean ONLY if not categorical and not in missing_features
function replace_nan_meanx!(x,param::HTBparam,meanx)

    for i in 1:size(x)[2]
        if !(i in union(param.missing_features,param.cat_features))
            x[:,i] = replace(x[:,i],NaN => meanx[i])
        end
    end

end



# Preliminary operations on training data before starting the boosting loop:
# Example: param_nf,data_nf,meanx,stdx   = preparedataHTB(data_nf,param), where data_nf is standardized and missing are replaced by NaN
# data.x may contain NaN. 
function preparedataHTB(data::HTBdata,param0::HTBparam)

    param = deepcopy(param0)

    target_encoding_values!(param,data) # computes target encoding values and stores them in param.cat_values
 
    x = copy(data.x)                       
    x = target_encoding(x,param)        # Replace 0,1,2,... with values of target mean. Affects data and param
                                        # NB: Expands x if some categorical features require an extensive (>1 column) representation
    meanx,stdx = robust_mean_std(x)     # meanx,stdx computed using target encoding values      

    data_standardized = HTBdata_sharedarray( data.y,(x .- meanx)./stdx,param,data.dates,data.weights,data.fnames,data.offset) # standardize

    param_given_data!(param,data_standardized)    # sets additional parameters that require data. 

    if param0.loss in [:L2,:Huber,:t,:lognormal,:gamma,:L2loglink,:logistic,:quantile,:logvar,:Poisson,:gammaPoisson]
    else 
        @error "add the new loss to categorical.jl"
    end    

    return param,data_standardized,meanx,stdx  
    
end


# Leaves NaN. 
# Transform any categorical feature using the stored values for target encoding.
# Assumes categorical features are already transformed to 0,1,.... using the dictionary in param.cat_dictionary
# Finally standardizes. 
# x_test = preparedataHTB_test(x_test,param,meanx,stdx) 
function preparedataHTB_test(x,param,meanx,stdx)   # x = data.x[indtest,:]

    x_test = copy(x)
    n      = size(x_test,1)

    for (i,j) in enumerate(param.cat_features)
 
        if isassigned(param.cat_values,i)       # dichotomous variables are not assigned
            xj         = copy(x_test[:,j])
            D          = param.cat_dictionary[i]
            values_cat = sort(collect(values(D)))    # values of categories, now in the order 0,1,...,

            for (c,value) in enumerate(values_cat)
                ind = findall(xj .== value)
                x_test[ind,j] .= param.cat_values[i].m[c,1]  
            end

            if j in param.cat_features_extended && param.cat_representation_dimension>1 

                x_add = Matrix{param.T}(undef,n,param.cat_representation_dimension-1)
 
                for col in 1:(param.cat_representation_dimension - 1)

                    for (c,value) in enumerate(values_cat)
                        ind = findall(xj .== value)
                        x_add[ind,col] .= param.cat_values[i].m[c,col+1]  
                    end
                    
                end
                
                x_test = hcat(x_test,x_add)

            end    

        end
    end
    
    x_test = (x_test .- meanx)./stdx

    return x_test


end 


# for HTBpredict: for categorical, replaces nan with missing. For non-categorical, replaces missing with nan
function nan_and_missing_predict(x0,param) 

    x = copy(x0)

    for j in 1:size(x)[2]
        if j in param.cat_features # replace nan with missing
            if eltype(x[!,j]) <: Number 
                x[!,j] .= ifelse.(isnan.(x[!,j]), missing, x[!,j])
            end     
        else                      # replace missing with nan 
            x[!,j] .= coalesce.(x[!,j], NaN)
        end 
    end

    return x
end 


# data.x is now assumed already standardized
function preparegridsHTB(data,param,meanx,stdx)

    τgrid             = gridvectorτ(param.meanlntau,param.varlntau,param.taugridpoints,priortype=param.priortype)
    μgrid,Info_x      = gridmatrixμ(data,param,meanx,stdx)
    n,p               = size(data.x)

    return τgrid,μgrid,Info_x,n,p

end



#Returns the grid of points at which to evaluate τ in the variable selection phase.
function gridvectorτ(meanlntau::T,varlntau::T,taugridpoints::Int;priortype::Symbol = :hybrid)::AbstractVector{T} where T<:AbstractFloat

    @assert(1 ≤ taugridpoints ≤ 5, "taugridpoints must be between 1 and 5")

    if priortype==:sharp
        τgrid = [Inf]  
    else
        if taugridpoints==1
            τgrid = [5]
        elseif taugridpoints==2
            τgrid = [2,8]
        elseif taugridpoints==3
            τgrid = [2,4,8]
        elseif taugridpoints==4
            τgrid = [2,4,8,16]
        elseif taugridpoints==5
            τgrid = [2,4,8,16,Inf]
        end
    end

    return T.(τgrid)

end



# Rough approximation to the Kantorovic (aslo known as Wasserstein) distance, in terms of quantiles, between p(xi) and a standard normal
# or between xi and another vector y (not necessarily of the same length)
# The p-distance is [∫( q1(u) - q(u) )^p du]^1/p, https://en.wikipedia.org/wiki/Wasserstein_metric.
# Quantiles are computed in the interval [0.025 0.975] rather than, say, [0.01 0.99] to avoid excessively penalizing fat tails, which HTBoost
# can handle quite well at default parametrs (τ=1 flattens out in the tail, and features are standardized with a robust std)
# NB: assumes xi and y are standardized, and, if y is provided, takes the smallest between d(xi,y) and d(xi,-y) (allowing for a minus sign)
# NOTE: distance(xi,y) rather than (xi,N) is appropriate for :L2, :t, :Huber, :quantile, not for :logistic
# sqrtw=sqrt.(data.weights). y and w are weighted, so the quantiles are computed on ys*sqrt(w), xs*sqrt(w) (multiplied by n/sum(sqrtw) to make the distance a pure number.),
# where ys and xs are (robust) standardized.
function kantorovic_distance(xi::AbstractVector{T},sqrtw::AbstractVector{T};p=1,y::AbstractVector{T}=T[])::T  where T<:AbstractFloat

    if length(xi)>=40
        qu = T.([i for i in 0.025:0.025:0.975])
        q_N = T.([0.06,0.13,0.19,0.25,0.32,0.39,0.45,0.52,0.60,0.67,0.76,0.84,0.93,1.04,1.15,1.28,1.44,1.64,1.96])
        q_N = vcat( vcat(-reverse(q_N),T(0),q_N) )
    elseif length(xi)>=20
        qu = T.([i for i in 0.05:0.05:0.95])
        q_N = T.([-1.64,-1.28,-1.04,-0.84,-0.67,-0.52,-0.39,-0.25,-0.13,0.0,0.13,0.25,0.39,0.52,0.67,0.84,1.04,1.28,1.64])    # deciles of the standardized Gaussian
    else
        qu = T.([i for i in 0.1:0.1:0.9])
        q_N = T.([-1.28,-0.84,-0.52,-0.25,0.0,0.25,0.52,0.84,1.28])    # deciles of the standardized Gaussian
    end

    q_xi = quantile(xi.*sqrtw*(length(xi)/sum(sqrtw)),qu)

    if isempty(y)  # distance from a Gaussian
        d = (mean( (abs.(q_N - q_xi)).^p ) )^(1/p)
    else             # distance between xi and y
        ys  = sqrtw.*y*(length(xi)/sum(sqrtw))
        q_y = quantile(ys,qu)
        d1 = (mean( (abs.(q_y - q_xi)).^p ) )^(1/p)
        q_y = quantile(-ys,qu)
        d2 = (mean( (abs.(q_y - q_xi)).^p ) )^(1/p)
        d = minimum([d1,d2])
    end

    return T(d)
end



# This long function computes some statistics on each feature: dichotomous, Kantorovic distance, mixed discrete-continuous ....
# These statistics are then a) passed on as Info_x, which is used to form priors on log(τ), b) used to form mugrid, grid of candidate split points for preliminary vs. 
# (Vector{Vector}, element i is a vector of of values of μ at which to evaluate the sigmoidal function for feature i in the variable selection stage.)
# I considered three ways of selecting the candidate split points: i) quantiles(), ii) clustering, like k-means or fuzzy k-means on each column of x. (Note: If the matrix is sparse, e.g. lots of 0.0, but not dichotomous, in general k-means will NOT place a value at exactly zero (but close))
# iii) an equally spaced grid between minimum(xi) and maximum(xi). Option iii) is fastest. Here I use i), but at most maxn (default 100_000) observations are sampled,
# and @distributed for. Deciles are interpolated.
# features on which sharp splits are imposed are given three times as many points in mugrid (so computing times comparable with other features, which evaluate on three values of τ)    
# If distrib_threads = false, uses @distributed and SharedArray. SharedArray can occasionally produce an error in Windows (not in Linux).
# If this happens, the code switches to distrib_threads = true 
# If distrib_threads = true, uses Threads.@threads. 
function gridmatrixμ(data::HTBdata,param::HTBparam,meanx,stdx;maxn::Int = 100_000,tol = 0.005, maxiter::Int = 100, fuzzy::Bool = false,
         distrib_threads::Bool=false)

    x        = data.x
    T        = param.T 
    npoints0 = param.mugridpoints
    loss     = param.loss
    npoints  = npoints0 + 1
    n,p      = size(data.x)

    # if forcing sharp splits, triple the number of grid points for μ
    if isempty(param.force_sharp_splits)
        npoints_mugrid0 = npoints
        sharp_splits = fill(false,p)
    else 
        npoints_mugrid0 = 1*param.mugridpoints + 1  # features that force sharp splits may have more points in mugrid (as an alternative to refineOptim)
        sharp_splits = param.force_sharp_splits
    end     

    @assert(npoints_mugrid0<n,"npoints cannot be larger than n")

    # grid for mu and information on feature x[:,i]    

    if distrib_threads==false
        try 
            mugrid0     = SharedArray{T}(npoints_mugrid0,p)
            dichotomous = SharedVector{Bool}(p)
            kd          = SharedVector{T}(p)     # Kantorovic 2-distance from feature i to a normal
            n_unique_a  = SharedVector{Int}(p)   # number of unique values in xi
            mixed_dc    = SharedVector{Bool}(p)  # if a feature is discrete or mixed discrete-continuous: some deciles are repeated
        catch ex
            distrib_threads = true
        end           
    end     

    if distrib_threads == true 
        mugrid0     = Matrix{T}(undef,npoints_mugrid0,p)
        dichotomous = Vector{Bool}(undef,p)
        kd          = Vector{T}(undef,p)     # Kantorovic 2-distance from feature i to a normal
        n_unique_a  = Vector{Int}(undef,p)   # number of unique values in xi
        mixed_dc    = Vector{Bool}(undef,p)  # if a feature is discrete or mixed discrete-continuous: some deciles are repeated
    end 

    n>maxn ? ssi=randperm(Random.MersenneTwister(param.seed_datacv),n)[1:maxn] : ssi=collect(1:n)
    w         = data.weights[ssi]     

    # Compute standardized y, used later to compute Kantorovic distance from standardized y if y is continuous, else distance from a Gaussian (ys=[]) ).  
    if loss in [:L2,:Huber,:t,:quantile,:lognormal]    
        m = median(data.y[ssi])
        ys  = (data.y[ssi] .- m)/(T(1.25)*mean(abs.(data.y .- m))) # standardize similarly to how x has been de-meaned
    elseif loss in [:gamma,:L2loglink,:Poisson,:gammaPoisson]   # lognormal sets data.y=log, but for not for gamma. Take logs as it is log(mean) which is modeled. 
        ys  = log.(data.y[ssi].+T(0.01))
        m   = median(ys)
        ys  = (ys .- m)/(T(1.25)*mean(abs.(ys .- m)))    # standardize similarly to how x has been de-meaned       
    elseif loss==:logistic
        ys = T[]
    else
        @error "loss function misspeled or not coded"
    end

    if distrib_threads
        @threads for i = 1:p
            fill_vectors_mugridmatrix!(x,ssi,i,ys,w,mugrid0,sharp_splits[i],npoints,n_unique_a,mixed_dc,dichotomous,kd)
        end    
    else     
        @sync @distributed for i = 1:p
            fill_vectors_mugridmatrix!(x,ssi,i,ys,w,mugrid0,sharp_splits[i],npoints,n_unique_a,mixed_dc,dichotomous,kd)
        end     
    end 

    # Interpolate npoints+1 deciles, and reshape mugrid0 into a vector of vectors (categorical features may have fewer than npoints unique values, and then using npoints would be wasteful)
    mugrid = Vector{Vector{T}}(undef,p)

    for i in 1:p
        sharp_splits[i]==true ? npoints_i=npoints_mugrid0 : npoints_i=npoints  # if not sharp splits, keep only the first npoints values
        m = sort(unique(mugrid0[1:npoints_i,i]))

        if length(m) < npoints_i  # can only happens if number(unique)<npoints; then we don't want to interpolate and lose one point
            mugrid[i]=m
        else
            m_int = Vector{T}(undef,length(m)-1)
            for j in 1:length(m_int)
                m_int[j]=(m[j]+m[j+1])/2
            end
            mugrid[i] = m_int
        end

    end

    # If augment_mugrid is not empty, add these points to mugrid[i], unless i is dichotomous 
    if !isempty(param.augment_mugrid)
        for i in 1:p
            if dichotomous[i]==false
                mugrid[i] = vcat(mugrid[i],param.augment_mugrid[i])
            end     
        end
    end      

    # Info_x collects information on the feature: dichotomous, number of unique values, mixed discrete-continuous etc...
    # Information that is not needed for forecasting.
    Info_x = Vector{Info_xi}(undef,p+1)   # p features + projection pursuit
    total_sharp = 0

    for i in 1:p
        isempty(param.force_sharp_splits) ? force_sharp=false : force_sharp=param.force_sharp_splits[i] 
        n_unique_a[i]<param.min_unique ? force_sharp=true : nothing
        mixed_dc[i]==true && param.mixed_dc_sharp==true ? force_sharp=true : nothing
        total_sharp += force_sharp

        isempty(param.force_smooth_splits) ? force_smooth=false : force_smooth=param.force_smooth_splits[i] 
        isempty(param.exclude_features) ? exclude_feature=false : exclude_feature=param.exclude_features[i]

        if i in param.cat_features         # number of categories         
            j = findfirst(param.cat_features .== i)
            if isassigned(param.cat_values,j)
                n_cat = size(param.cat_values[j].m,1)
            else   # dummy 
                n_cat = 1
            end
        else 
            n_cat = 1    
        end     

        Info_x[i] = Info_xi( i,exclude_feature,dichotomous[i],false,n,n_cat,force_sharp,force_smooth,
        n_unique_a[i],mixed_dc[i],T(kd[i]),meanx[i],stdx[i],minimum(x[:,i]),maximum(x[:,i]) )

    end
    
    # projection pursuit Info_xi 
    Info_x[p+1] = Info_xi(p+1,false,false,true,1,1,false,false,1,false,T(0),T(0),T(1),T(0),T(0) )

    return mugrid,Info_x
end



function fill_vectors_mugridmatrix!(x,ssi,i,ys,w,mugrid0,sharp_split_i,npoints,n_unique_a,mixed_dc,dichotomous,kd)

    npoints_sharp = size(mugrid0,1) 
    xi        = x[ssi,i]            # already standardized

    miss_a   = isnan.(xi)          # exclude missing values     
    xi       = xi[miss_a .== false]
    wi       = w[miss_a .== false]
    isempty(ys) ? yi = ys : yi = ys[miss_a .== false]

    # I need the corresponding y, subsample too 
    unique_xi = unique(xi)
    n_unique  = length(unique_xi)
    n_unique_a[i] = n_unique
    dichotomous[i] = (n_unique==2)
    kd[i] = kantorovic_distance(xi,sqrt.(wi);y=yi)  
    #if n_unique==1; @warn "feature $i has only one unique value in either the full sample or in a cross-validated sample. Setting randomizecv=true may solve the problem.";

    if dichotomous[i] == false
    
        sharp_split_i==true ? npoints_i=npoints_sharp : npoints_i=npoints  

        if n_unique<=npoints_i  
            mugrid0[1:n_unique,i]=unique_xi
            u = unique(mugrid0[1:npoints_i,i])
            length_u = length(u)

            if npoints_i>n_unique
                mugrid0[n_unique+1:npoints_i,i]=fill(unique_xi[1],npoints_i-n_unique)
            end # replicated values will be eliminated later
        
        else   # Method i), quantiles, merging repeated values.
            mugrid0[1:npoints_i,i] = quantile(xi,[i for i = 0.5/npoints_i:1/npoints_i:1-0.5/npoints_i])  # length = npoints = npoints0+1
            u = unique(mugrid0[1:npoints_i,i])
            length_u = length(u)

            if length_u<=Int(ceil(npoints_i*2/3))   # repeated deciles
                mugrid0[1:length_u,i] = u[1:length_u]
                mugrid0[length_u+1:npoints_i,i] = quantile(unique_xi,[i for i = 1/(npoints_i+1-length_u):1/(npoints_i+1-length_u):1-1/(npoints_i+1-length_u)])
            end

            #=
            # Method ii), clustering                 
            if fuzzy==true
                r   = Clustering.fuzzy_cmeans(x[ssi,i]', npoints_i, 2.0; tol = tol, maxiter=maxiter, display = :none)
            else
                r   = Clustering.kmeans(x[ssi,i]', npoints_i; tol = tol, maxiter=maxiter, display = :none)  # input matrix is the adjoint (p,n). Output is Matrix
            end

            mugrid0[1:npoints_i,i] = sort(vec(r.centers))
            =#
        end

        # Is the feature mixed discrete-continuous? (Enough deciles are repeated.)
        if npoints_i>=10
            length_u<10 ? mixed_dc[i]=true : mixed_dc[i]=false  
        elseif npoints_i>=6
            length_u<6 ? mixed_dc[i]=true : mixed_dc[i]=false  
        else
            mixed_dc[i]=false
        end
        
    end

end 
 