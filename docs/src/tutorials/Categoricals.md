## Categoricals features

HTBoost is promising for categorical features, particularly if high dimensionals.  
This tutorials shows:
- How to inform HTBoost about categorical features
- Parameters related to categorical features, and their default values.
- Comparison with LightGBM and CatBoost, with discussion.  

### How to inform HTBoost about categorical features

- If cat_features is not specified, non-numerical features (e.g. String or CategoricalValue) are treated as categorical
- If cat_features is specified, it can be a vector of Integers (positions), a vector of Strings (corresponding to 
  data.fnames, which must be provided) or a vector of Symbols (the features' names in the dataframe). Notice that if cat_features is specified, it must include all features to be treated as categorical (no automatic detection).

**Example of use: all categorical features are non-numerical**
``` 
param = HTBparam()  
```
***Examples of use: specify positions in data.x***
```
param = HTBparam(cat_features=[1])    # cat_features must be a vector
param = HTBparam(cat_features=[1,9])
```
**Example of use: specify names from data.fnames** 
```
data = HTBdata(y,x,param,fnames=["country","industry","earnings","sales"]) 
param = HTBparam(cat_features=["country","industry"])
```
**Example of use: specify names in dataframe** 
```
data = HTBdata(y,x) #  where x is DataFrame                           
param = HTBparam(cat_features=[:country,:industry])         
```

See [examples/Categoricals](../examples/Categoricals.md) for a discussion of how HTBoost treats categoricals under the hood. Key points:
- Missing values are assigned to a new category.
- If there are only 2 categories, a 0-1 dummy is created. For anything more than two categories, it uses a variation of target encoding.
- The categories are encoded by 4 values in default mode: mean, frequency, variance (robust) and skew(robust). (For financial variables, the variance and skew may be more informative than the mean.) Set cat_representation_dimension = 1 to encode by mean only.
- One-hot-encoding with more than 2 categories is not supported, but can of course be implemented as data preprocessing.

## Cross-validation of categorical parameters 

 `param.cv_categoricals` can be used to perform a rough cross-validation of `n0_cat` and/or `mean_encoding_penalization`, as follows:
 - `cv_categoricals = :none` uses default parameters 
 - `cv_categoricals = :penalty` runs a rough cv the penalty associated to the number of categories; recommended if n/n_cat if high for any feature, particularly if SNR is low                             
 - `cv_categoricals = :n0` runs a rough of cv the strength of the prior shrinking categorical values to the overall mean; recommended with highly unequal number of observations in different categories.
- `cv_categoricals = :both` runs a rough cv of penalty and n0 

The default is :none if :modality in [:fastest,:fast], :penalty if :compromise, and :both if :accurate. 


### Comparison to LightGBM and CatBoost

Different packages differ substantially in their treatment of categorical features.  
LightGBM does not use target encoding, and can completely break down (very poor in-sample and oos fit) when the number of categories is high in relation to n (e.g. n=10k, #cat=1k). The LightGBM manual suggests
treating high dimensional categorical features as numerical or embedding them in a lower-dimensional space. LightGBM can, however, perform very well in lower-dimensional cases.

CatBoost, in contrast, adopts mean target encoding as default, can handle very high dimensionality and
has a sophisticated approach to avoiding data leakage which HTBoost is missing. (HTBoost resorts to a penalization on categorical features instead.) CatBoost also interacts categorical features, while HTBoost does not.
In spite of the less sophisticated treatment of categoricals, in this simple simulation set-up HTBoost substantially outperforms CatBoost if n_cat is high and the categorical feature interacts with the continuous feature,
presumably because target encoding generates smooth functions  by construction in this setting.

It seems reasonable to assume that high dimensional target encoding, by its very nature, will generate smooth functions in many settings, making 
HTBoost a promising tool for high dimensional categorical features. The current treatment of categorical features is however quite
crude compared to CatBoost, so some of these gains are not yet realized. 


### An example with simulated data

The code below simulate data from *y = f(x1,x2) + u*, where *x1* is continuous and *x2* is categorical of possibly very high dimensionality.
Each category of x2 is assigned its own coefficient drawn from a distribution ("uniform", "normal", "chi", "lognormal").
The user can specify the form of *f(x1,x2).*

---
---

```julia
number_workers  = 8  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random
using Statistics
using LightGBM

```

In this example there is only one categorical features. We can control:
- the number of categories
- the distribution of their average effect: "uniform" or "normal" for symmetric, thin-tailed effects, and "chi" or "lognormal" for right-skewed distributions.
- the type of interaction with the continuous feature *x1*

```julia

Random.seed!(1)

n          =     10_000   # sample size   
ncat       =     100     # number of categories (actual number may be lower as they are drawn with reimmission)

bcat       =     1.0      # coeff of categorical feature (if 0, categories are not predictive)
b1         =     1.0      # coeff of continuous feature 
stde       =     1.0      # error std

cat_distrib =   "normal"  # distribution for categorical effects: "uniform", "normal", "chi", "lognormal" for U(0,1), N(0,1), chi-square(1), lognormal(0,1)
interaction_type = "multiplicative" # "none", "multiplicative", "step", "linear"

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

```
cv_categoricals = :none uses default parameters for categorical features.
To speed up estimation, we set modality=:fast, depth=3, nfold=1 and nofullsample= true.

```julia

# HTBoost parameters 
loss         = :L2
modality     = :fast       # :accurate, :compromise, :fast, :fastest
cv_categoricals = :none    # :none (default), :penalty, :n0, :both

depth        = 3           # fix depth to further speed up estimation  
nfold        = 1           # number of folds in cross-validation. 1 for fair comparison with LightGBM 
nofullsample = true        # true to speed up execution when nfold=1. true for fair comparison with LightGBM 
verbose      = :Off

```

The second feature is categorical. Needs to be an input in param (see below) since it is numerical, and therefore will not be automatically detected as categorical by HTBoost.  
```julia

cat_features = [2]      

```
**LightGBM parameters** 

Accoding to the LightGBM manual (https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html):
"For a categorical feature with high cardinality (#category is large), it often works best to treat the feature as numeric, either by simply ignoring the categorical interpretation of the integers or by embedding the categories in a low-dimensional numeric space."

```julia

ignore_cat_lightgbm = false  # true to ignore the categorical nature and treat as numerical in lightGBM 

```

**Having specified all our options, we simulate the data**

```julia

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

```

and now fit HTBoost and LightGBM. (We don't show CatBoost here.)

```julia

# Fit HTBoost 
param  = HTBparam(loss=loss,modality=modality,depth=depth,nfold=nfold,
                 nofullsample=nofullsample,verbose=verbose,cat_features=cat_features,cv_categoricals=cv_categoricals)
data   = HTBdata(y,x,param)
output = HTBfit(data,param,cv_grid=[depth])  # cv_grid=[depth] needed to combine modality=:compromise with depth not cv    

ntrain = Int(round(n*(1-param.sharevalidation)))
yhat   = HTBpredict(x[1:ntrain,:],output)    # in-sample fitted value
yf     = HTBpredict(x_test,output)           # out-of-sample forecasts

println("\n n and number of categories ", [n length(unique(xcat))])
println("\n HTBoost " )
println("in-sample R2           ", 1 - mean((yhat - y[1:ntrain]).^2)/var(y[1:ntrain])  )
println(" validation  R2         ", 1 - output.loss/var(y[ntrain+1:end]) )
println(" out-of-samples R2      ", 1 - mean((yf - y_test).^2)/var(y_test) )

```
Test-set fit is higher than training-sample, but in line with validation-set. 

```
 n and number of categories [10000 100]

 HTBoost
 in-sample R2           0.657
 validation  R2         0.645
 out-of-samples R2      0.643
```

Let's inspect function smoothness to gauge whether accuracy gains vs LightGBM can be expected, with the caveat that the different treatment of categoricals makes this comparison less reliable here.

```julia
avgtau,gavgtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data,verbose=true,best_model=false)
```
The categorical feature has a very low average τ, indicating near-linearity. Large gains compared to LightGBM and CatBoost would not be surprising in this case.  


```markdown

Row │ feature      importance  avgtau    sorted_feature  sorted_importance  sorted_avgtau 
     │ String       Float32     Float64   String          Float32            Float64       
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │ x1             45.4137   0.682357  x2                       46.9929        0.558092
   2 │ x2             46.9929   0.558092  x1                       45.4137        0.682357
   3 │ x2_cat_freq     1.78321  1.92352   x2_cat_var                5.81018       1.26407
   4 │ x2_cat_var      5.81018  1.26407   x2_cat_freq               1.78321       1.92352

 Average smoothing parameter τ is 0.7.
 
 ...

 At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.

```
Fit LightGBM, at default values. 

```julia

# LightGBM
if ignore_cat_lightgbm == true
    estimator = LGBMRegression(objective = "regression",num_iterations = 1000,early_stopping_round = 100,max_depth=4)
else 
    estimator = LGBMRegression(objective = "regression",num_iterations = 1000,early_stopping_round = 100,max_depth=4,
            categorical_feature = cat_features)
end 

n_train         = Int(round((1-param.sharevalidation)*length(y)))
x_train,y_train = x[1:n_train,:], Float64.(y[1:n_train])
x_val,y_val     = x[n_train+1:end,:], Float64.(y[n_train+1:end])

LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
yhat_gbm_default = LightGBM.predict(estimator,x)[:,1]
yf_gbm_default = LightGBM.predict(estimator,x_test)[:,1]

println("\n LightGBM default, ignore_cat_lightgbm = $ignore_cat_lightgbm  " )
println("\n in-sample R2      ", 1 - mean((yhat_gbm_default - y).^2)/var(y)  )
println(" out-of-sample R2  ", 1 - mean((yf_gbm_default - y_test).^2)/var(y_test) )

```

```markdown
LightGBM default, ignore_cat_lightgbm = false

 in-sample R2      0.660
 out-of-sample R2  0.635
 ```

**High dimensional categoricals**

If we re-run the script withn n=10k, n_cat = 1k, LightGBM breaks down, as expected.  
Note: while LightGBM is not suited to high-dimensional categorical, it may outperform HTBoost in lower dimensional settings in which target encoding is not appropriate or effective.  

```markdown
 n and number of categories [10000 1000]

HTBoost
 out-of-samples R2      0.500

LightGBM default, ignore_cat_lightgbm = false
 out-of-sample R2  0.095

LightGBM default, ignore_cat_lightgbm = true
 out-of-sample R2  0.052

```

In LightGBM, the bottlenecks seems to be n/n_cat rather than n_cat per se:
here we set n=100k, n_cat=1k and its performance is again satisfactory.
The difference in R2 may seem small, but in this example LightGBM would require a sample of n > 1_000_000 to match HTBoost with n = 100_000. 
(Because of the extremely smooth function used to simulate the data.)

```markdown

 n and number of unique features [100000 1000]

HTBoost
 out-of-samples R2      0.648

LightGBM default, ignore_cat_lightgbm = false
 out-of-sample R2       0.634

```

