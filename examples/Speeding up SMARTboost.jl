
"""

# Some options to speed up training for SMARTboost, particularly with large n and large p.

## 1) Modality = :fast 

The safest way to speed up training is by setting modality=:fast. If nfold=1, setting nofullsample=true
(or, equivalently, modality=:fastest) further reduces computing time by 60% at the cost of fitting the model
on a smaller sample. 

## 2) Use a coarser grid for feature selection at deeper levels of the tree. (Can be combined with any value of modality) 

Examples of use: 
'''
    param = SMARTparam(depth_coarse_grid =4,depth_coarse_grid2=5,modality=:fast)
    param = SMARTparam(depth_coarse_grid =4,depth_coarse_grid2=5,modality=:compromise)
'''
Replacing the defaults (5,7) with (4,5) may speed up computations by 25-33%, with no or little loss of
fit in most cases. 

## 3) Force smooth splits (in combination with modality=:fast). Warning: decreased performance! 

In situations where some features require sharp splits, the model is estimated twice. Setting priortype=:smooth
(in combination with modality=:fast or :fastest) then cuts computing times in half. The loss of fit
can be substantial, so this is only recommended for preliminary investigation. The resulting model may still
perform well when stacked with a standard GBM light LightGBM.  

## 4) Cross-validate on a sub-sample, then one run on full sample.

Setting modality = :fast fits the model at default parameters either. If some cross-validation is desired, 
the following strategy can be used to speed up cross-validation, typically with only small deterioration in performance
if n is large and n/p is large.

When n is very large, and it takes too long to fit SMARTboost in modality = :compromise or :accurate,
one way to proceed is to cv on a subsample of the data (say 20%) and then fit only one model on the full sample.
This can be accomplished as follows:

- Set modality=:compromise or :accurate, take a subsample of the data (say 10% or 20%), and run output=SMARTfit() on that.
- Set param=output.bestparam, and then param.modality=:fast, and run SMARTfit() on the full data.
 
An example is given below: 
"""
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboostPrivate

using Random,Statistics

# USER'S OPTIONS 
Random.seed!(123)

# Options for data generation 
n         = 400_000
p         = 100          # number of features 
dummies   = true        # if true if x, x_test are 0-1 (much faster training).
stde      = 1            

# Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast only fits one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

randomsubset      = 0.1          # e.g. 0.1 or 0.2. Share of observations in the first sub-set 
modality_subs     = :compromise  # :accurate or :compromise (default)
modality_full     = :fast        # :fast

nfold_subs       = 1             # number of cv folds. 1 sufficient if the sub-sample is sufficiently large 
nfold_full       = 1         

verbose          = :Off
warnings         = :On

# simple f(x), with pstar relevant features.

Friedman_function(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

p_star    = 10       # number of relevant features 
β = randn(p_star)    # draw linear coefficients from a Gaussian distribution
Linear_function_Gaussian(x)   = x[:,1:length(β)]*β
f_dgp    = Linear_function_Gaussian

# END USER'S INPUTS 

if dummies
    x,x_test  = randn(n,p),randn(200_000,p) 
    x,x_test = Float64.(x .> 0), Float64.(x_test .> 0)
else
    x,x_test = randn(n,p), randn(200_000,p)    
end     

y       = f_dgp(x) + stde*randn(n)
f_test  = f_dgp(x_test)

# SMARtboost

param_subs   = SMARTparam(modality=modality_subs,nfold=nfold_subs,
                verbose=verbose,warnings=warnings)
data         = SMARTdata(y,x,param_subs)
n            = length(data.y)

ind       = randperm(n)[1:convert(Int,round(randomsubset*n))]
data_subs = SMARTdata(y[ind],x[ind,:],param)

output_subs = SMARTfit(data_subs,param_subs) # performs cv on subset
param       = output_subs.bestparam        # sets param at best configuration in subset
param.modality = modality_full 
param.nfold    = nfold_full


# ************
param = SMARTparam(modality=:fast,nfold=1,nofullsample=true,priortype=:smooth,
                   verbose=:Off)
data  = SMARTdata(y,x,param)

println("\n n = $n, p = $p, dummies=$dummies, modality = $(param.modality)")

println("\n Time to train the full model.")
@time output = SMARTfit(data,param);

yf = SMARTpredict(x_test,output,predict=:Ey)  # predict
println("\n RMSE of SMARTboost from true E(y|x) ", sqrt(mean((yf-f_test).^2)) )
