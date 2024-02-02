# 
# Functions to work with categorical data.
# Target encoding for categorical features, where the encoding depends on the loss function:
#  
# - :L2, :Huber, :t  -> mean 
# - :quantile        -> quantile
# - :logistic        -> logit of mean
# - :sigma           -> log of std dev
# 
#
# The following functions in data_preparation.jl are also used to detect categorical data:
# they are applied once, when SMARTdata() is first called, so that all possible categories can be listed.
#
# - categorical_features!()         modifies param.cat_features (to a vector of I) 
# - map_cat_convert_to_float!()     converts categorical features (as from param.cat_features) to 0,1,... by mapping each unique value to a number,
#                                   and saves the mapping in param.cat_dictionary
#
# The following components of param store information on with categoricals
#
# - param.cat_features        can be provided by the user or automatically detected by categorical_features!() 
# - param.cat_dictionary      mapping between original value of the category and 0,1,...
#                             needed to convert test data to the same format as training data
# - param.cat_values          Vector{NamedTuple} of dimension length(param.cat_features), each element is for a categorical feature
#                             NamedTuple is a collection of matrices
#
# The functions below are applied ex-novo to each training set, and to each test set.
# NB: categorical encoding is provisional, currently using posterior mean only. 
#
# -f_posteriors_list()               Creates a vector of functions, each computing the posterior mean of a representation dimension for target encoding (e.g. mean,frequency)
# -f_priors_list()                   Creates a vector of functions, each computing the prior mean (for forecasting) of a representation dimension for target encoding (e.g. mean,frequency)
#                                    Not all representation are always used: how many depends on param.cat_representation_dimension
# - target_encoding_values!()         Computes/draws target encoding values for categorical features. Done once for each training set. 
#   - global_stats()                  computes mean,variance,quantile (if :quantile) etc... for all  data
#   - draw_categoricals
#     - f_posterior_1                 function to compute the appropriate mean target encoding (first moment) depending on loss 
#       cat_posterior_L2,cat_posterior_logistic,cat_posterior_quantile,cat_posterior_logvar
#       cat_prior_L2,....
#     - f_posterior_ni                function to compute the number of observations in each category
#     - f_posterior_2                function to compute the appropriate log variance (or log dispersion) target encoding (second moment) depending on loss
#   - f_prior_1
#   - f_prior_ni
#   - f_prior_2
# - target_encoding                 Replace categoricals column with target encoding values stored in param.cat_values


# f_posteriors = f_posteriors_list(loss)  
function f_posteriors_list(loss)
    f_posteriors = [f_posterior_1(loss), f_posterior_ni(loss), f_posterior_2(loss)]
    return f_posteriors
end 


# f_priors = f_priors_list(loss)
function f_priors_list(loss)
    f_priors = [f_prior_1(loss), f_prior_ni(loss), f_prior_2(loss)]
    return f_priors
end 


# compute mean,variance,quantile (if :quantile) for all data
# updates param.globalstats = (mean_y=mean_y,var_y=var_y,q_y=q_y)
function global_stats!(param,data)

        mean_y  = mean(data.y)
        var_y   = var(data.y)
        param.loss==:quantile ? q_y=quantile(data.y,param.coeff[1]) : q_y=quantile(data.y,0.5)
    
        globalstats = (mean_y=mean_y,var_y=var_y,q_y=q_y,n_0=param.n0_cat)
        param.cat_globalstats = globalstats

end 



# Replaces categoricals column with target encoding values stored in param.cat_values, unless the variable is dichotomous. 
# NOTE: assumes categorical information is encoed in one column.
# If some categorial features requires an extensive (>1 column) representation, it adds columns to x 
function target_encoding(x0::AbstractMatrix,param::SMARTparam)  # modifies only x 

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
function target_encoding_values!(param::SMARTparam,data::SMARTdata)    # modifies param, not data 

    global_stats!(param,data)     # updates param.cat_globalstats    
    param.cat_values = Vector{NamedTuple}(undef,length(param.cat_features))

    for (i,j) in enumerate(param.cat_features)
        
        if length(param.cat_dictionary[i]) > 2
            param.cat_values[i] = draw_categoricals(param.cat_globalstats,data.y,data.x[:,j],i,param)  # draw_categoricals returns a matrix, draws of different transformations             
        end

    end 

end 




# Computes some statistics on the full sample, and calls the appropriate function given param.loss
function draw_categoricals(globalstats::NamedTuple,y::Vector{T},xj::Vector{T},i::Int,param::SMARTparam) where T<:AbstractFloat   
 
    I = param.I
    n = length(y)

    D          = param.cat_dictionary[i]
    num_cat    = length(D)
    values_cat = sort(collect(values(D)))    # values of categories, now in the order 0,1,...,
 
    indexes = Vector{Vector{I}}(undef,num_cat)   

    for (i,value) in enumerate(values_cat) # sorted() so the first category corresponds to 0, the second to 1 etc...
        indexes[i] = findall(xj .== value) 
    end
    
    f_posteriors = f_posteriors_list(param.loss)
    cat_values_m = Matrix{param.T}(undef,length(indexes),param.cat_representation_dimension)  

    for j in 1:param.cat_representation_dimension
        f_posterior = f_posteriors[j]
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
function f_posterior_1(loss)

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


# For each category, computes posterior log variance (moment 2). How this is done depends on the loss function.
# For some functions (e.g. :logistic) it returns nothing (e.g. f = f_posterior_2(loss) ... check with isnothing(f))
function f_posterior_2(loss)
    if loss==:logistic
        return nothing 
    else f_posterior = cat_posterior_logvar
    end 

    return f_posterior
end 


# For each category, computes number of instances n_0 + n_i in that category
function f_posterior_ni(loss)
    f_posterior = cat_posterior_ni
    return f_posterior
end  


# prior for first moment 
function f_prior_1(loss)

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

function f_prior_ni(loss)
    f_prior = cat_prior_ni
    return f_prior
end 


function f_prior_2(loss)   # check isnothing(f), where f = f_prior_2(loss)
    if loss==:logistic
        return nothing 
    else
        f_prior = cat_prior_logvar
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

function cat_prior_quantile(globalstats)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return q_y
end

function cat_prior_ni(globalstats)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return n_0
end

function cat_prior_logvar(globalstats)  
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return log(var_y)
end

# For each category, computes number of instances n_0 + n_i in that category
function cat_posterior_ni(y,ind,globalstats)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0
    return length(ind) + n_0
end  

# log of approximate posterior mean of var   
function cat_posterior_logvar(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0

    if isempty(ind)
        mean_i = mean_y
        var_i  = var_y
    else
        mean_i = mean(y[ind])
        var_i =  mean((y[ind] .- mean_i).^2)
    end     

    posterior_mean  = log((n_0*var_y + n_i*var_i )/(n_0+n_i)) 

    return posterior_mean

end


# posterior distribution.  
function cat_posterior_L2(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0

    if isempty(ind)
        mean_i = mean_y
        var_i  = var_y
    else
        mean_i = mean(y[ind])
        var_i =  mean((y[ind] .- mean_i).^2)
    end     

    posterior_mean = (n_0*mean_y + n_i*mean_i)/(n_0+n_i)

    return posterior_mean

end


# posterior distribution.  
function cat_posterior_logistic(y,ind,globalstats)  

    n_i = length(ind)
    mean_y,var_y,q_y,n_0 = globalstats.mean_y, globalstats.var_y,globalstats.q_y,globalstats.n_0

    if isempty(ind)
        mean_i = mean_y
        var_i  = var_y
    else
        mean_i = mean(y[ind])
        var_i =  mean((y[ind] .- mean_i).^2)
    end     

    m              = (n_0*mean_y + n_i*mean_i)/(n_0+n_i)
    posterior_mean = log(m/(1-m))

    return posterior_mean

end



# prior is stronger for smaller quantile: In SMARTparam, param.n0_cat = param.n0_cat/minimum([τ,1-τ])
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
