## Logistic regression (binary classification with logloss)

**Summary**

- Comparison with lightGBM on a logistic regression problem with simulated data. 
- param.modality as the most important user's choice.
- In default modality, HTBoost performs automatic hyperparameter tuning.

**Main points** 

Sketch of a comparison of HTBoost and lightGBM on a logistic regression problem.
The comparison with LightGBM will favor HTBoost if the function generating the data is 
smooth in some features (this is easily changed by the user). lightGBM is cross-validated over max_depth and num_leaves, with the number of trees set to 1000 and found by early stopping.

---
---
**Import HTBoost for distributed parallelization on the desired number of workers.**

This step is not required by other GMBs, which rely on shared parallelization.  
The time to first plot increases with the number of cores.
HTBoost parallelizes well up to 8 cores, and quite well up to 16 if p/#cores is sufficiently high. 

```julia

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

# import required packages for this script
using Random,Statistics
using Plots
using LightGBM

```

**Options for HTBparam( ).**  

Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
:fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
automatic hyperparameter tuning. In HTBoost, it is not recommended that the user performs 
hyperparameter tuning by cross-validation, because this process is done automatically if modality is
:compromise or :accurate. The recommended process is to first run in modality=:fast or :fastest,
for exploratory analysis and to gauge computing time, and then switch to :compromise (default)
or :accurate.

```julia

Random.seed!(1)

# Options for HTBparam()
loss      = :logistic
modality  = :compromise   # :accurate, :compromise (default), :fast, :fastest
verbose   = :Off
warnings  = :On

```

**Options for cross-validation:**

While the default in other GBM is to randomize the allocation to train and validation sets,
the default in HTBoost is block cv, which is suitable for time series and panels.
Set randomizecv=true to bypass this default. 
See examples/Global equity Panel.jl for further options on cross-validation (e.g. sequential cv).

```julia
randomizecv = false       # false (default) to use block-cv. 

```

**Options for data generation** 

Define the function f(x), where x are indendent N~(0,1), and f(x) is for the natural parameter,
so f(x) = log(prob/(1-prob))

```julia

n         = 10_000
p         = 10           # mumber of features. p>=4.  
n_test    = 100_000   

f_1(x,b)    = b*x .+ 1 
f_2(x,b)    = sin.(b*x)  
f_3(x,b)    = b*x.^2
f_4(x,b)    = b./(1.0 .+ (exp.(10*(x .- 0.5) )))    

b1,b2,b3,b4 = 1.5,2.0,0.5,2.0
dgp(x)       = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)

```

End of user's options.

Build model for HTBoost and LightGBM.
In *HTBparam( )*, nfold=1 and nofullsample=true for a fair comparison with LightGBM (both models have the sample training and validation sets.)
The validation set is the last 30% of the data. (Change param.sharevalidation to change this percentage, and randomizecv=true for a random subsample.)

```julia

param  = HTBparam(loss=loss,nfold=1,nofullsample=true,modality=modality,warnings=warnings)

 estimator = LGBMClassification(   # LGBMRegression(...)
    objective = "binary",
    num_class = 1,
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    metric = ["binary_logloss"],
    num_threads = number_workers,
    max_depth = -1,          # -1 default
    min_data_in_leaf = 100,  # 100 default 
    num_leaves = 127         # 127 default  
 )

```

**Generate data and fit both models**

In *HTBpredict*, predict=:Ey (default) predicts E(y|x), i.e. prob(y=1|x), while predict=:Egamma predicts the natural parameter. Here we choose the latter for a less noisy comparison, since we are working with simulated data.  

```julia

x,x_test    = randn(n,p), randn(n_test,p)
ftrue       = dgp(x)
ftrue_test  = dgp(x_test)

y = (exp.(ftrue)./(1.0 .+ exp.(ftrue))).>rand(n) 

# HTBoost
data   = HTBdata(y,x,param)
output = HTBfit(data,param)
yf     = HTBpredict(x_test,output,predict=:Egamma)  
MSE_HTB = sum((yf - ftrue_test).^2)/n_test

```
For LightGBM, we specify a parameter search over num_leaves and max_depth.

```julia

# lightGBM. Specify 
y       = Float64.(y)                 
n_train = Int(round((1-param.sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])
    
splits = (collect(1:n_train),collect(1:min(n_train,100)))  # goes around the problem that at least two training sets are required by search_cv (we want the first)

params = [Dict(:num_leaves => num_leaves,
               :max_depth => max_depth) for
          num_leaves in (4,16,32,64,127,256),
          max_depth in (2,3,5,6,8)]

lightcv = LightGBM.search_cv(estimator,x,y,splits,params,verbosity=-1)

loss_cv = [lightcv[i][2]["validation"][estimator.metric[1]][1] for i in eachindex(lightcv)]
minind = argmin(loss_cv)

estimator.num_leaves = lightcv[minind][1][:num_leaves]
estimator.max_depth  = lightcv[minind][1][:max_depth]

# fit at best parameters
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)

yf_gbm = LightGBM.predict(estimator,x_test)[:,1]
yf_gbm = log.(yf_gbm./(1.0 .- yf_gbm))   #  prediction for the natural parameter

MSE_Light    = sum((yf_gbm - ftrue_test).^2)/n_test

```

Print the results for a comparison.

```julia
println("\n n = $n, p = $p")
println("RMSE from true natural parameter, HTBoost, modality = $modality ", sqrt(MSE_HTB) )
println("RMSE from true natural parameter, lightGBM                      ", sqrt(MSE_Light) )

```

**Comparing accuracy**  

- modality=:fast uses depth=5, while :compromise selects depth=1 (dgp(x) is additive), resulting in better performance. 
- HTBboost outperforms LightGBM in terms of accuracy (but with much longer training time).
- The extent of this outperformance will vary depending on the smoothness of dgp(x)

```markdown

n = 10000, p = 10
RMSE from true natural parameter, HTBoost, modality = fast        0.5615
RMSE from true natural parameter, HTBoost, modality = compromise  0.5125
RMSE from true natural parameter, lightGBM                        0.6435

```

Feature importance and average smoothing parameter for each feature.  

tau is the smoothness parameter; lower values give smoother functions, while tau=Inf is a sharp split (tau is trancated at 40 for this function).  
avgtau is a summary of the smoothness of f(x), with features weighted by their importance.
avgtau_a is a vector array with the importance weighted tau for each feature.  

```julia
avgtau,avg_explogtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data,verbose=true);

plot(x_plot,g_plot,title="smoothness of splits",xlabel="standardized x",label=:none)
```

The plot gives an idea of the average (importance weighted) smoothness across all splits. In this case, the average across features is 3.6, which is substantial smoothness. 

<img src="../assets/avgtau_logistic.png" width="400" height="250">

The function *HTBweightedtau( )* with verbose=true produces the following table, from which we can see that f(x) is quite smooth with respect to all features.
This explains why HTBoost outperforms LightGBM so strongly.  

```markdown

Row │ feature  importance  avgtau    sorted_feature  sorted_importance  sorted_avgtau 
     │ String   Float32     Float64   String          Float32            Float64       
─────┼─────────────────────────────────────────────────────────────────────────────────
   1 │ x1        40.4326     1.04795  x1                      40.4326          1.04795
   2 │ x2        17.9978    11.9914   x4                      19.9693          2.63904
   3 │ x3        14.4546     2.57154  x2                      17.9978         11.9914
   4 │ x4        19.9693     2.63904  x3                      14.4546          2.57154
   5 │ x5         1.09162    1.5      x9                       2.63876         1.54776
   6 │ x6         1.33539    1.5      x6                       1.33539         1.5
   7 │ x7         0.304846   1.5      x5                       1.09162         1.5
   8 │ x8         0.907542   1.5      x8                       0.907542        1.5
   9 │ x9         2.63876    1.54776  x10                      0.867542        1.5
  10 │ x10        0.867542   1.5      x7                       0.304846        1.5

 Average smoothing parameter τ is 3.6.

 In sufficiently large samples, and if modality=:compromise or :accurate

 - Values above 20-25 suggest little smoothness in important features. HTBoost's performance may slightly outperform or slightly underperform other gradient boosting machines.
 - At 10-15 or lower, HTBoost should outperform other gradient boosting machines, or at least be worth including in an ensemble.
 - At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.

```

**Larger sample size**

As the sample size gets larger, HTBoost can match the accuracy of LightGBM with only a fraction of the data. For example, HTBoost is approximately as accurate with n = 100k
as LightGBM with n = 500k.  
(The key to this result lies in the smoothness of f(x) with respect to at least a subset
of important features, not in f(x) being additive).     

```markdown

n = 100000, p = 10
RMSE from true natural parameter, HTBoost, modality = compromise  0.2201
RMSE from true natural parameter, lightGBM                        0.3346

n = 500000, p = 10
RMSE from true natural parameter, HTBoost, modality = compromise  0.???
RMSE from true natural parameter, lightGBM                        0.2115

```


