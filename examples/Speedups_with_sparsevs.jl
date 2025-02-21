
#=

**Short description:**

- Explore how sparsevs affects speed and accuracy.

Extensive description: 

Sparsevs is used by HTBoost to speed up HTBoost with large number of features (p>>100).
The idea is to to store, at predetermined intervals (a Fibonacci sequence in default), the 
ten (or other number: param.number_best_features) features that had the lowest loss in each
split of the tree. For example, for a tree of depth 4, between 10 and 40 features will be stored
in this group of best_features at the first update (10 if the same features have the lowest loss
at each split). In the next tree, only the features in this group will be
considered as candidates for splitting, saving time for large p. At the next predetermined
update, the best features are added to this group.
Dichotomous features (i.e. dummies, taking only two values) are always included, since much faster.
Since features never leave the group of best_features, this group can get large if the environment
is dense, and will stay small if the environment is sparse. Large speed-ups gains are therefore
not guaranteed, but the forecasting accuracy should not be strongly affected except in extreme
cases where several hundred features are needed to accurately fit the data. To prevent loss of fit,
an automatic warning is issued if the size of the group of best_features is over 50% the largest
theoretical size (output.ratio_actual_max>0.5).

The speed-ups gains are typically smaller than may be expected, due to the fact that i) parallelization
becomes more efficient for larger p*/number_workers (where p* is the number of candidate features at
a given split, and p*<p if sparsevs = :On ), and ii) there is a fixed cost for refineOptim 
(refine otimization of μ,τ,m given i). Speed-ups are larger for very large p (e.g. p=2000)

=#
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HybridTreeBoosting

using Random,Statistics
using LightGBM

# USER'S OPTIONS 
Random.seed!(123)

# Options for data generation 
n         = 1_000
p         = 1_000        # number of features 
stde      = 1            

# Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

sparsevs         = :On
modality         = :compromise   # :accurate, :compromise (default), :fast, :fastest

number_best_features = 10  # Default 10. <10 to consider fewer features in each split (larger speed gains, less precision)
frequency_update     = 1       # Default 1. >1 to update less frequently (larger speed gains, less precision)

nfold            = 1     # 1 for fair comparison with LightGBM
nofullsample     = true  # true for fair comparison with LightGBM

verbose          = :Off
warnings         = :On

ntrees           = 1000   # maximum number of trees 

# function: Friedman of linear, with pstar relevant features.
# Increasing the number of relevant features increases the difficulty of the problem and can be
# used to evaluate speed gains and accuracy losses of sparsevs.

Friedman_function(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

p_star    = 10       # number of relevant features 
β = randn(p_star)
dgp(x)  = x[:,1:length(β)]*β

# END USER'S INPUTS 

if dgp==Friedman_function
    x,x_test = rand(n,p), rand(200_000,p)    # Friedman function on U(0,1)
else 
    x,x_test = randn(n,p), randn(200_000,p)    
end     

f       = dgp(x)
y      = f + stde*randn(n)
f_true = dgp(x_test)

# LightGBM

# Create an estimator with the desired parameters—leave other parameters at the default values.
estimator = LGBMRegression(   # LGBMRegression(...)
    objective = "regression",
    categorical_feature = [],  # or [1,2,3,5,6,7,8], treating date as a category, as probably Lightgbm would.
    num_iterations = ntrees,   # default 100
    learning_rate = 0.1,    # default 0.1
    early_stopping_round = 50,  # default 0, i.e. Inf
    bagging_fraction = 1.0,
    feature_fraction = 1.0,
    metric = ["l2"],
    num_threads = number_workers,
    max_depth = -1,         # -1 for default
    #device_type="cpu",
    #max_depth = -1   # no limit
)

sharevalidation = 0.3
n_train = Int(round((1-sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = y[1:n_train]
x_val   = x[n_train+1:end,:]; y_val = y[n_train+1:end]

LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
yf_gbm = LightGBM.predict(estimator,x_test)


# HTBoost

param   = HTBparam(modality=modality,ntrees=ntrees,sparsevs=sparsevs,
                    frequency_update=frequency_update,number_best_features=number_best_features,
                    nfold=nfold,verbose=:Off,warnings=:On)

data  = HTBdata(y,x,param)

println("\n n = $n, p = $p, modality = $modality, sparsevs = $sparsevs, frequency_update = $frequency_update")
println(" time to fit ")

@time output = HTBfit(data,param);
yf = HTBpredict(x_test,output,predict=:Ey)  # predict

println("\n RMSE of HTBoost from true E(y|x)                   ", sqrt(mean((yf-f_true).^2)) )
println(" RMSE of LightGBM (default param) from true E(y|x)     ", sqrt(mean((yf_gbm-f_true).^2)) )

fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false);
println(" HTBoost number of included features in final model $(sum(fi.>0))")

if param.sparsevs == :On
    @info " sparsevs is :On. Repeat with sparsevs=:Off to track speed gains and any accuracy loss. Speed gains will be smaller if many features are relevant."
else 
    @info " sparsevs is :Off. Repeat with sparsevs=:On to track speed loss and any accuracy gains. Speed gains will be smaller if many features are relevant."
end     
