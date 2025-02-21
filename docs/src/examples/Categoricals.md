# Categoricals

## How to inform HTBoost about categorical features: 

- If cat_features is not specified, non-numerical features (e.g. Strings) are treated as categorical
- If cat_features is specified, it can be a vector of Integers (positions), a vector of Strings (corresponding to 
  data.fnames, which must be provided) or a vector of Symbols (the features' names in the dataframe).

**Example of use: all categorical features are non-numerical.**
```julia 
    param = HTBparam()  
```
**Example of use: specify positions in data.x**
```julia 
    param = HTBparam(cat_features=[1])
    param = HTBparam(cat_features=[1,9])
```
**Example of use: specify names from data.fnames** 
```julia 
    data = HTBdata(y,x,param,fnames=["country","industry","earnings","sales"]) 
    param = HTBparam(cat_features=["country","industry"])
```

**Example of use: specify names in dataframe** 
```julia 
    data = HTBdata(y,x) #  where x is DataFrame                           
    param = HTBparam(cat_features=[:country,:industry])         
```

## How HTBoost handles categorical features:

- Missing values are assigned to a new category.
- If there are only 2 categories, a 0-1 dummy is created. For anything more than two categories, it uses a variation of target encoding.
- The categories are encoded by their mean, frequency and variance. (For financial variables, the variance may be more informative than the mean.)
- Smooth splits are particularly promising with target encoding, particularly in high dimensions (lots of categories). 
- One-hot-encoding with more than 2 categories is not supported, but is easily implemented as data preprocessing.
- If modality is :accurate or :compromise, a quick and rough cross-validation is performed over two parameters:
  - The prior sample size n0 for target encoding.
  - The penalization attached to categorical features, which mitigates the over-representation of categorical features
    in the final model due to data leakage. (This seems to be a new approach to the problem: it has the advantage of
    being fast and using the entire sample. Data leakage remains, and implies that train set loss is lower than validation loss). 


**An example with simulated data**

The code below simulate data from y = f(x1,x2) + u, where x1 is continuous and x2 is categorical of possibly very high dimensionality.
Each category of x2 is assigned its own coefficient drawn from a distribution ("uniform", "normal", "chi", "lognormal").
The user can specify the form of f(x1,x2).

LightGBM does not use target encoding, and can completely break down (very poor in-sample and oos fit) when the number of categories
is high in relation to n (e.g. n=10k,n_cat=1k). (The LightGBM manual https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html suggests
treating high dimensional categorical features as numerical or embedding them in a lower-dimensional space.)

CatBoost (note: code in separate Python file), in contrast, adopts target encoding as default, can handle very high dimensionality and
has a sophisticated approach to avoiding data leakage which HTBoost is missing.
In spite of the absence of data leakage (which would presumably benefit HTBoost as well), in this simple simulation set-up
HTBoost substantially outperforms CatBoost if n_cat is high and the categorical feature interacts with the continuous feature,
presumably because target encoding generates smooth functions in this set-up.   
It seems reasonable to assume that target encoding, by its very nature, will generate smooth functions in most settings, making 
HTBoost a promising tool for high dimensional categorical features. The current treatment of categorical features is however quite
crude compared to CatBoost, so some of these gains are not yet realized. 

```julia
number_workers  = 8  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HybridTreeBoosting

using Random
using Statistics
using LightGBM

# USER'S OPTIONS 

# Options to generate data 

Random.seed!(1)

n          =     10_000   # sample size   
ncat       =     100     # number of categories (actual number may be lower as they are drawn with reimmission)

bcat       =     1.0      # coeff of categorical feature (if 0, categories are not predictive)
b1         =     1.0      # coeff of continuous feature 
stde       =     1.0      # error std

cat_distrib =   "chi"  # distribution for categorical effects: "uniform", "normal", "chi", "lognormal" for U(0,1), N(0,1), chi-square(1), lognormal(0,1)
interaction_type = "step" # "none", "multiplicative", "step", "linear"

# specify the function f(x1,x2), with the type of interaction (if any) between x1 (continuous) and x2 (categorical)
function yhat_x1xcat(b1,b2,x1,interaction_type)

    if interaction_type=="none"
        yhat = b2 + b1*x1
    elseif interaction_type=="multiplicative"     
        yhat = b2 + b1*b2*x1
    elseif interaction_type=="step"     
        yhat = b2 + b1*x1*(x1>0)    
    elseif interaction_type=="linear"  
        yhat = b2 + (b1-b2)*x1
    end 

    return yhat
end 

# HTBoost parameters 
loss         = :L2
modality     = :compromise # :accurate, :compromise, :fast, :fastest
depth        = 3           # fix depth to speed up estimation  
nfold        = 1           # number of folds in cross-validation. 1 for fair comparison with LightGBM 
nofullsample = true        # true to speed up execution when nfold=1. true for fair comparison with LightGBM 
verbose      = :Off

cat_features = [2]       # The second feature is categorical. Needs to be an input in param (see below) since it is numerical.  

# LightGBM parameters 

# Accoding to the LightGBM manual (https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html):
# "For a categorical feature with high cardinality (#category is large), it often works best to treat the feature as numeric, either by simply ignoring the categorical interpretation of the integers or by embedding the categories in a low-dimensional numeric space."

ignore_cat_lightgbm = false  # true to ignore the categorical nature and treat as numerical in lightGBM 

# END USER'S OPTIONS   

# create data 
n_test  = 100_000 
cate            = collect(1:ncat)[randperm(ncat)]   # create numerical categories (integers) 
xcat,xcat_test  = rand(cate,n),rand(cate,n_test)    # draw element of categorical features from the list of categories
x,x_test        = hcat(randn(n),xcat),hcat(randn(n_test),xcat_test)
yhat,yhat_test  = zeros(n),zeros(n_test)

if cat_distrib=="uniform"
    b = bcat*rand(ncat)   # uniform fixed-effects
elseif cat_distrib=="normal"
    b = bcat*randn(ncat)  # Gaussian fixed effects
elseif cat_distrib=="chi"                        
    b = bcat*randn(ncat).^2 # chi(1) fixed effects (long tail)
elseif cat_distrib=="lognormal"                        
    b = bcat*exp.(randn(ncat)) # chi(1) fixed effects (very long tail)
else
    @error "cat_distribution is misspelled"
end 

for r in 1:n
    b2 = b[findfirst(cate .== xcat[r])]
    yhat[r] = yhat_x1xcat(b1,b2,x[r,1],interaction_type) 
end

for r in 1:n_test
    b2 = b[findfirst(cate .== xcat_test[r])] 
    yhat_test[r] = yhat_x1xcat(b1,b2,x_test[r,1],interaction_type) 
end

y = yhat + stde*randn(n)
y_test = yhat_test + stde*randn(n_test)

# Fit HTBoost 
param  = HTBparam(loss=loss,modality=modality,depth=depth,nfold=nfold,nofullsample=nofullsample,verbose=verbose,cat_features=cat_features)
data   = HTBdata(y,x,param)
output = HTBfit(data,param,cv_grid=[depth])

ntrain = Int(round(n*(1-param.sharevalidation)))
yhat   = HTBpredict(x[1:ntrain,:],output) 
yf     = HTBpredict(x_test,output) 

println("\n n and number of unique features ", [n length(unique(xcat))])
println("\n HTBoost " )
println("\n in-sample R2           ", 1 - mean((yhat - y[1:ntrain]).^2)/var(y[1:ntrain])  )
println(" validation  R2         ", 1 - output.loss/var(y[ntrain+1:end]) )
println(" out-of-samplesample R2 ", 1 - mean((yf - y_test).^2)/var(y_test) )

println(" Average τ on categorical feature is low, suggesting gains from smoothness. ")
avgtau,gavgtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data,verbose=true,best_model=false)

# LightGBM
if ignore_cat_lightgbm == true
    estimator = LGBMRegression(objective = "regression",num_iterations = 1000,early_stopping_round = 100)
else 
    estimator = LGBMRegression(objective = "regression",num_iterations = 1000,early_stopping_round = 100,
            categorical_feature = cat_features)
end 

n_train         = Int(round((1-param.sharevalidation)*length(y)))
x_train,y_train = x[1:n_train,:], Float64.(y[1:n_train])
x_val,y_val     = x[n_train+1:end,:], Float64.(y[n_train+1:end])

println(" \n Running LightGBM, which is usually lightning fast, but can be quite slow with high-dimensional categorical features.")
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
yhat_gbm_default = LightGBM.predict(estimator,x)[:,1]
yf_gbm_default = LightGBM.predict(estimator,x_test)[:,1]

println("\n LightGBM default, ignore_cat_lightgbm = $ignore_cat_lightgbm  " )
println("\n in-sample R2      ", 1 - mean((yhat_gbm_default - y).^2)/var(y)  )
println(" out-of-sample R2  ", 1 - mean((yf_gbm_default - y_test).^2)/var(y_test) )

```
