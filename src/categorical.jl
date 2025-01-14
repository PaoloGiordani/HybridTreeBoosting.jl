# 
# Functions to work with categorical data.
# Target encoding for categorical features, where the encoding depends on the loss function:
# 
# A categorical feature is replaced by param.cat_representation_dimension features. The first new feature, which replacs the original, 
# is a standard target encoding, adapted to the loss functions. 
# 
# - :L2, :Huber, :t  -> mean 
# - :quantile        -> quantile
# - :logistic        -> logit of mean
# - :sigma           -> log of std dev
# 
# If param.cat_representation_dimension>1, additional columns are added to the data frame, each with a different target encoding:
# 2) frequency, 3) robust std, 4) robust skewness, 5) robust kurtosis.
#
#
# The following functions in data_preparation.jl are also used to detect categorical data:
# they are applied once, when HTBdata() is first called, so that all possible categories can be listed.
#
# - categorical_features!()         modifies param.cat_features (to a vector of I) 
# - map_cat_convert_to_float!()     converts categorical features (as from param.cat_features) to 0,1,... by mapping each unique value to a number,
#                                   and saves the mapping in param.cat_dictionary
#
# The following components of param store information with categoricals
#
#  - param.cat_features        can be provided by the user or automatically detected by categorical_features!() 
#  - param.cat_dictionary      mapping between original value of the category and 0,1,...
#                             needed to convert test data to the same format as training data
#  - param.cat_values          Vector{NamedTuple} of dimension length(param.cat_features), each element is for a categorical feature
#                             NamedTuple is a collection of matrices
#
# The functions below are applied ex-novo to each training set, and to each test set.
#   Note for PG: these functions could be simplied... they were written for a Bayesian approach, drawing encoding values from their posterior
#
#  - global_stats()                  computes mean,variance,skew,....,quantile (if :quantile) etc... for all data in a training set
#    auxiliary functions: mad_skewness(),mad_kurtosis()
#  -f_posteriors_list()               Creates a vector of functions, each computing the posterior mean of a representation dimension for target encoding (e.g. mean,frequency)
#  -f_priors_list()                   Creates a vector of functions, each computing the prior mean (for forecasting) of a representation dimension for target encoding (e.g. mean,frequency)
#                                    Not all representation are always used: how many depends on param.cat_representation_dimension
#                                    Priors are used in forecasting. 
#  - target_encoding_values!()         Computes target encoding values for categorical features, stores them in param.cat_values Done once for each training set. 
#   - draw_categoricals               Currently a misnomer, and I dropped randomness in favor of deterministic target encoding.
#     - f_posterior_m                 function to compute the appropriate mean target encoding (first moment) depending on loss 
#       cat_posterior_L2,cat_posterior_logistic,cat_posterior_quantile,cat_posterior_logvar,....
#       cat_prior_L2,....
#     - f_posterior_ni                function to compute the number of observations in each category
#     - f_posterior_s                function to compute the appropriate log variance (or log dispersion) target encoding (second moment) depending on loss
#   - f_prior_m
#   - f_prior_ni
#   - f_prior_s
#  - target_encoding()                Replace categoricals column with target encoding values stored in param.cat_values
#
# The functions below are used for forecasting
#
#  - prepares_categorical_predict()   Maps categorical features to their target encoding values, allowing for new categories.
#                                     Inefficient! Could be speeded up.

# measure of skewenss based on MAD. Equivalent to Groeneveld and Meeden. If mad is not defined (x has 1 unique value), returns 0.
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

# measure of kurtosis based on MAD, proposed by Pinsky 2024 (eq. 8) "Mean Absolute Deviation (About Mean) Metric for Kurtosis
# if the measure is not defined (x has 1 unique value), returns 0.6, the kurtosis of a Gaussian (with this metric)
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


# compute mean,variance,quantile (if :quantile) and other statistics for all data
# NB: updates param.globalstats = (mean_y=mean_y,var_y=var_y,q_y=q_y,n_0=param.n0_cat,mad_y=mad_y,skew_y=skew_y,kurt_y=kurt_y)
#    Modify param.cat_globalstats in param.jl if you want to add more statistics or change their names
function global_stats!(param,data)

    mean_y  = mean(data.y)
    var_y   = var(data.y)
    mad_y   = mean(abs.(data.y .- mean_y))
    skew_y  = mad_skewness(data.y)
    kurt_y  = mad_kurtosis(data.y)

    param.loss==:quantile ? q_y=quantile(data.y,param.coeff[1]) : q_y=quantile(data.y,0.5)

    globalstats = (mean_y=mean_y,var_y=var_y,q_y=q_y,n_0=param.n0_cat,mad_y=mad_y,skew_y=skew_y,kurt_y=kurt_y)
    param.cat_globalstats = globalstats

end 


# f_posteriors = f_posteriors_list(loss) 
# Target encoding of values of y in each category:
# 1) mean 2) frequency 3) logvar 4) robust skew  5) robust kurtosis. The measures may depend on the loss function. 
function f_posteriors_list(loss)
    f_posteriors = [f_posterior_m(loss),f_posterior_ni(loss),f_posterior_s(loss),f_posterior_sk(loss),f_posterior_k(loss)]
    return f_posteriors
end 


# f_priors = f_priors_list(loss)
# Values assigned in forecasting to new categories 
function f_priors_list(loss)
    f_priors = [f_prior_m(loss),f_prior_ni(loss),f_prior_s(loss),f_prior_sk(loss),f_prior_k(loss)]
    return f_priors
end 



# Replaces categoricals column with target encoding values stored in param.cat_values, unless the variable is dichotomous. 
# If some categorial features requires an extensive (>1 column) representation, it adds columns to x 
function target_encoding(x0::AbstractMatrix,param::HTBparam)  # modifies only x 

    if isempty(param.cat_features)
        return x0 
    end

    T,n = param.T,size(x0,1)
    x   = copy(x0)  

    for (i,j) in enumerate(param.cat_features)
        
        if length(param.cat_dictionary[i]) > 2   # dummies (two categories) are left at 0,1

            xj         = copy(x[:,j])               #
            D          = param.cat_dictionary[i]
            values_cat = sort(collect(values(D)))    # values of categories, now in the order 0,1,...,
            
            for (c,value) in enumerate(values_cat)
                ind = findall(xj .== value)
                x[ind,j] .= param.cat_values[i].m[c,1]  
            end
            
            if j in param.cat_features_extended && param.cat_representation_dimension>1
                x_add = Matrix{T}(undef,n,param.cat_representation_dimension-1)

                for col in 1:(param.cat_representation_dimension - 1)
                    for (c,value) in enumerate(values_cat)
                        ind = findall(xj .== value)
                        x_add[ind,col] .= param.cat_values[i].m[c,col+1]  
                    end
                end
                
                x = hcat(x,x_add)

            end    
        end     
    end 

    return x

end 



# Computes/draws target encoding values for categorical features.   
# Updates param.cat_values with draws of the conditional mean (or other transformation) of y given xj for each category
# Features with two categories are left at 0,1 and treated as dummies. 
function target_encoding_values!(param::HTBparam,data::HTBdata)    # modifies param, not data 

    global_stats!(param,data)     # updates param.cat_globalstats    
    param.cat_values = Vector{NamedTuple}(undef,length(param.cat_features))

    for (i,j) in enumerate(param.cat_features)
        
        if length(param.cat_dictionary[i]) > 2
            param.cat_values[i] = draw_categoricals(param.cat_globalstats,data.y,data.x[:,j],i,param)  # draw_categoricals returns a matrix, draws of different transformations             
        end

    end 

end 


# Computes some statistics on the full sample, and calls the appropriate function given param.loss
function draw_categoricals(globalstats::NamedTuple,y::Vector{T},xj::Vector{T},i::Int,param::HTBparam) where T<:AbstractFloat   
 
    I = param.I
    n = length(y)

    D          = param.cat_dictionary[i]
    num_cat    = length(D)
    values_cat = sort(collect(values(D)))    # values of categories, now in the order 0,1,...,
 
    indexes = Vector{Vector{I}}(undef,num_cat)   

    for (i,value) in enumerate(values_cat) # sorted() so the first category corresponds to 0, the second to 1 etc...
        indexes[i] = findall(xj .== value) 
    end
    
    cat_values_m = Matrix{param.T}(undef,length(indexes),param.cat_representation_dimension)  

    for j in 1:param.cat_representation_dimension
        f_posterior = f_posteriors_list(param.loss)[j]
        for (i,ind) in enumerate(indexes)
            cat_values_m[i,j] = f_posterior(y,ind,globalstats) 
        end
    end    

    cat_values_i = (m=cat_values_m,other_info=[])   

    return cat_values_i 

end 



# In forecasting, maps categorical features to their target encoding values, allowing for new categories.
# x = prepares_categorical_predict(x,param).
# Returns a df with features not standardized.
# PG NOTE: inefficient! Could be speeded up.  
function prepares_categorical_predict(x,param)

    globalstats,T = param.cat_globalstats,param.T

    for (i,j) in enumerate(param.cat_features)

        z = x[!,j]
        Tj = eltype(z)

        if Tj <: Union{AbstractString,Missing}
            z = replace(z,missing => "missing")
        elseif Tj <: Union{Bool,Missing}
            z = replace(z,missing => "missing",true => "true", false => "false")
        elseif Tj <: Union{Real,Missing}
            z = replace(z,missing => T(NaN) )
        elseif Tj <: Union{Missing,CategoricalValue}
            z = replace(z,missing => "missing")
        else 
            @error "map_cat_convert_to_float!(): categorical features must be of type AbstractString,Real,Bool, or Categorical"    
        end

        if j in param.cat_features_extended && param.cat_representation_dimension>1
            xj = deepcopy(x[!,j])
        end       

        f_priors = f_priors_list(param.loss)
    
        D = param.cat_dictionary[i]
        keys_new = unique(z)
        z_new = Vector{T}(undef,length(z))   # numerical, with categorical encoded values

        f_prior  = f_priors[1]
        prior_value = f_prior(globalstats)

        for key in keys_new

            ind = z .== key

            if haskey(D,key)             # map to dictionary, then to target encoding value
                c = Int(D[key]+1)        # 0,1,2... mapped to row 1,2,3... of cat_values[i].m
                if isassigned(param.cat_values,i)
                    z_new[ind] .= param.cat_values[i].m[c,1]
                else   #  send to dummy: 0,1...  
                    z_new[ind] .= D[key]        
                end     
            else
                z_new[ind] .= prior_value   
            end
        end

        x[!,j] = z_new

        # Add columns to the df if param.cat_representation_dimension>1

        if j in param.cat_features_extended && param.cat_representation_dimension>1 

            for col in 1:(param.cat_representation_dimension - 1)

                f_prior  = f_priors[col+1]
                prior_value = f_prior(globalstats)
                column_name = "extend"*"$j"*"$col"
                z_new = Vector{T}(undef,length(z))   # numerical, with categorical encoded values

                for key in keys_new

                    ind = z .== key
        
                    if haskey(D,key)             # map to dictionary, then to target encoding value
                        c = Int(D[key]+1)        # 0,1,2... mapped to row 1,2,3... of cat_values[i].m
                        if isassigned(param.cat_values,i)
                            z_new[ind] .= param.cat_values[i].m[c,col+1]
                        else   #  send to dummy: 0,1...  
                            z_new[ind] .= D[key]        
                        end     
                    else
                        z_new[ind] .= prior_value   
                    end
                end

                x[!, column_name] = z_new
            end
        end    

    end 
    
    return x

end 


# For each category, computes posterior mean (moment 1). How this is done depends on the loss function.
function f_posterior_m(loss)

    if loss in [:L2,:Huber,:t,:lognormal,:gamma,:L2loglink,:Poisson,:gammaPoisson]
        f_posterior = cat_posterior_L2
    elseif loss == :logistic
        f_posterior = cat_posterior_logistic   
    elseif loss == :quantile               
        f_posterior = cat_posterior_quantile
    elseif loss == :logvar              
        f_posterior = cat_posterior_logvar    
    else 
        @error "f_posterior not coded for loss = $(loss) in categorical.jl"    
    end     

    return f_posterior
end 


# For each category, computes a robust measure of posterior std (moment 2).
function f_posterior_s(loss)
    f_posterior = cat_posterior_mad
    return f_posterior
end 


# For each category, computes number of instances n_0 + n_i in that category
function f_posterior_ni(loss)
    f_posterior = cat_posterior_ni
    return f_posterior
end  


function f_posterior_sk(loss)
    f_posterior = cat_posterior_skew
    return f_posterior
end 

function f_posterior_k(loss)
    f_posterior = cat_posterior_kurtosis
    return f_posterior
end 


# prior for first moment 
function f_prior_m(loss)

    if loss in [:L2,:Huber,:t,:lognormal,:gamma,:L2loglink,:Poisson,:gammaPoisson]
        f_prior = cat_prior_L2
    elseif loss == :logistic
        f_prior = cat_prior_logistic    
    elseif loss == :quantile               
        f_prior = cat_prior_quantile
    elseif loss == :logvar              
        f_prior = cat_prior_logvar    
    else 
        @error "f_prior not coded for loss = $(loss) in categorical.jl"    
    end     

    return f_prior
end 


function cat_prior_L2(globalstats)  
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return mean_y
end

function cat_prior_logistic(globalstats)  
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return log(mean_y/(1-mean_y))
end

function cat_prior_logvar(globalstats)  
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return log(var_y)
end

function cat_prior_quantile(globalstats)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return q_y
end

function cat_prior_ni(globalstats)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return n_0
end

function cat_prior_mad(globalstats)
    mean_y,var_y,q_y,n_0,mad_y,sk_y,k_y = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0,
                                            globalstats.mad_y,globalstats.skew_y,globalstats.kurt_y
    return mad_y
end


function cat_prior_skew(globalstats)
    mean_y,var_y,q_y,n_0,mad_y,sk_y,k_y = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0,
                                            globalstats.mad_y,globalstats.skew_y,globalstats.kurt_y
    return sk_y
end

function cat_prior_kurtosis(globalstats)
    mean_y,var_y,q_y,n_0,mad_y,sk_y,k_y = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0,
                                            globalstats.mad_y,globalstats.skew_y,globalstats.kurt_y
    return k_y
end



function f_prior_ni(loss)
    f_prior = cat_prior_ni
    return f_prior
end 

function f_prior_s(loss)   
    f_prior = cat_prior_mad
    return f_prior
end 

function f_prior_sk(loss)  
    f_prior = cat_prior_skew
    return f_prior
end 

function f_prior_k(loss)  
    f_prior = cat_prior_kurtosis
    return f_prior
end 

# For each category, computes number of instances n_0 + n_i in that category
function cat_posterior_ni(y,ind,globalstats)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return length(ind) + n_0
end  


# mad measure of dispersion   
function cat_posterior_mad(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0,mad_y,skew_y,kurt_y = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0,globalstats.mad_y,globalstats.skew_y,globalstats.kurt_y

    if isempty(ind)
        return mad_y
    else
        mad_i =  mean( abs.(y[ind] .- mean(y[ind])) )
        return sqrt( (n_0*mad_y^2 + n_i*mad_i^2)/(n_0+n_i ) )  # in a quasi-Gaussian setting, this seems the right formula
        #return (n_0*mad_y + n_i*mad_i)/(n_0+n_i )  # not obvious to me which is the best way to combine the two MADs in a non-Gaussian setting
    end     

end


function cat_posterior_skew(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0,mad_y,skew_y,kurt_y = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0,globalstats.mad_y,globalstats.skew_y,globalstats.kurt_y

    if isempty(ind)
        return skew_y
    else
        skew_i =  mad_skewness(y[ind])
        return (n_0*skew_y + n_i*skew_i)/(n_0+n_i )
    end     

end


function cat_posterior_kurtosis(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0,mad_y,skew_y,kurt_y = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0,globalstats.mad_y,globalstats.skew_y,globalstats.kurt_y

    if isempty(ind)
        return kurt_y
    else
        kurt_i =  mad_kurtosis(y[ind])
        return (n_0*kurt_y + n_i*kurt_i)/(n_0+n_i )
    end     

end




# log of approximate posterior mean of var   
function cat_posterior_logvar(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0

    if isempty(ind)
        var_i  = var_y
    else
        var_i =  mean((y[ind] .- mean(y[ind])).^2)
    end     

    posterior_mean  = log((n_0*var_y + n_i*var_i )/(n_0+n_i)) 

    return posterior_mean

end


# approximate posterior mean of mean
function cat_posterior_L2(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0

    if isempty(ind)
        return mean_y
    else
        mean_i = mean(y[ind])
        return (n_0*mean_y + n_i*mean_i)/(n_0+n_i)
    end     

    return posterior_mean

end


# approximate posterior distribution of mean for logistic loss  
function cat_posterior_logistic(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0

    if isempty(ind)
        mean_i = mean_y
    else
        mean_i = mean(y[ind])
    end     

    m              = (n_0*mean_y + n_i*mean_i)/(n_0+n_i)
    posterior_mean = log(m/(1-m))

    return posterior_mean

end



# prior is stronger for smaller quantile: In HTBparam, param.n0_cat = param.n0_cat/minimum([τ,1-τ])
# if there are too few obs, quantile() will take the min or max, but here I use the global value instead
function cat_posterior_quantile(y,ind,globalstats)    

    n_i = length(ind)

    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    yb = y[ind]    

    if isempty(ind) || n_i<1/τmin
        q_i = q_y 
    else
        q_i = quantile(yb,τ)            # yb not y 
    end
    
    posterior_mean = (n_0*q_y + n_i*q_i)/(n_0+n_i)

    return posterior_mean 

end
