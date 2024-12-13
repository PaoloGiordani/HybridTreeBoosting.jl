
## Speeding up HTBoost with large n

HTBoost is very slow in comparison with other GBMs. Here we discuss some options to speed up training when n is large. 

**If HTBoost predominantly chooses hard splits, consider switching to CatBoost**

If preliminary analysis (e.g. on a subsample and/or with modality=:fastest) suggests that the average value of tau is high (higher than 15-20, see [Basic use](Basic_use.md)), HTBoost is effectively fitting symmetric trees with hard rather than smooth splits; CatBoost is then a much more efficient option to fit symmetric trees, if the other features of HTBoost (see [index](../index.md)) are not required. For Julia and R users, EvoTrees can also build symmetric trees (tree_type = "oblivious"). 

**Some options to speed up training for HTBoost with large n**

HTBoost runs much faster (particularly with large n) with multiple cores than with one, after the initial one-off cost.
The improvements in speed are roughly linear in the number of cores, up to 8 cores, and still good up to 16 cores,
particularly when p/#cores is large. Gains after 16 cores are modest at best.  

**Option 1. modality = :fast, nfold = 1, nofullsample = true.** 

The easiest way to speed up training is by setting nfold=1 (a single validation set), nofullsample=true, and modality=:fast or :fastest. These modalities do not perform cv. 
:fast will typically still produces a competitive model in terms of accuracy, particularly if n/p is large.
If nfold=1, setting nofullsample=true further reduces computing time by 60% at the cost of fitting the model
on a smaller sample.
modality = :fastest automatically sets nfold=1, nofullsample=true, and also lambda = 0.2 instead of 0.1.
lambda = 0.2 can perform almost as well as 0.1 if the function is smooth and n/p is large.

See Option 4 for an alternative.

**Option 2. Use a coarser grid for feature selection at deeper levels of the tree. (Can be combined with any value of modality)** 

Examples of use: 
```julia
    param = HTBparam(depth_coarse_grid =4,depth_coarse_grid2=5,modality=:fast)
    param = HTBparam(depth_coarse_grid =4,depth_coarse_grid2=5,modality=:compromise)
```
Replacing the defaults (5,7) with (4,5) may speed up computations by 25-33%, with no or little loss of fit in most cases. 

**Option 3. Don't allow forcing sharp splits (in combination with modality=:fast). Warning: potential for decreased performance!**

In situations where some features may require imposing sharp splits, the model is estimated twice.
To avoid this, run 
```julia
output = SMARTfit(data,param,cv_hybrid=false)
```
(in combination with modality=:fast or :fastest) then cuts computing times in half. The loss of fit is modest in some cases, but can be substantial in others.

**Option 4. Cross-validate on a sub-sample, then one run best model on full sample.**

Setting modality = :fast fits the model at default parameters. If some cross-validation is desired, 
the following strategy can be used to speed up cross-validation, typically with only small deterioration in performance if n is large. 

When n is very large, and it takes too long to fit HTBoost in modality = :compromise or :accurate,
one way to proceed is to cv on a subsample of the data (say 20%) and then fit only one model on the full sample, using the best parameters found in the subsample, except for the number of trees.
If the subsample is large enough, the best parameters found in the subsample will be close to the best parameters in the full sample. (Again, the number of trees is optimized on the full sample.)
(Of course the subset is more noisy and will prefer simpler models, but the difference should be modest if n is large.)

This can be accomplished as follows:

- Set modality=:compromise or :accurate, take a subsample of the data (20%), and run *output=HTBfit()* on that.
- Set param=output.bestparam, and then param.modality=:fast, and run HTBfit() on the full data.
 
An example is given below: 

```julia
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Statistics

# USER'S OPTIONS 
Random.seed!(123)

# Options for data generation 
n         = 500_000
p         = 100         # number of features 
dummies   = true        # if true if x, x_test are 0-1 (much faster training).
stde      = 1            

# Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast only fits one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

randomsubset      = 0.2          # e.g. 0.2. Share of observations in the first sub-set 
modality_subs     = :compromise  # :accurate or :compromise (default)
modality_full     = :fast        # :fast

nfold_subs       = 1             # number of cv folds. 1 sufficient if the sub-sample is sufficiently large 
nfold_full       = 1         
nofullsample_full = true         # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation          
randomizecv       = false       # false (default) to use block-cv.

verbose          = :Off
warnings         = :On

# simple f(x), with pstar relevant features.

p_star    = 10       # number of relevant features 
β         = randn(p_star)    # draw linear coefficients from a Gaussian distribution
dgp(x)    = x[:,1:length(β)]*β

# END USER'S INPUTS 

if dummies
    x,x_test = randn(n,p),randn(200_000,p) 
    x,x_test = Float64.(x .> 0), Float64.(x_test .> 0)
else
    x,x_test = randn(n,p), randn(200_000,p)    
end     

y       = dgp(x) + stde*randn(n)
f_test  = dgp(x_test)

# HTBoost on a sub-sample 
param_subs   = HTBparam(modality=modality_subs,nfold=nfold_subs,nofullsample=true,randomizecv=randomizecv,
                verbose=verbose,warnings=warnings)
data         = HTBdata(y,x,param_subs)
n            = length(data.y)

ind       = randperm(n)[1:convert(Int,round(randomsubset*n))]
data_subs = HTBdata(y[ind],x[ind,:],param_subs)

output_subs = HTBfit(data_subs,param_subs) # performs cv on subset

# HTBoost on full sample 
param          = output_subs.bestparam        # sets param at best configuration in subset, then modify where appropriate

param.ntrees   = 2_000                        # number of trees should not be from subsample! Early stopping must be on full sample.
param.modality = modality_full      
param.nfold    = nfold_full
param.nofullsample = nofullsample_full

data  = HTBdata(y,x,param)

println("\n n = $n, p = $p, dummies=$dummies, modality = $(param.modality)")

println("\n Time to train the full model.")
@time output = HTBfit(data,param);

yf = HTBpredict(x_test,output,predict=:Ey)  # predict
println("\n RMSE of HTBoost from true E(y|x) ", sqrt(mean((yf-f_test).^2)) )

```