
"""

Short description:

- Suggestions to speed up fitting with large sample sizes. 

Extensive description: 

When n is very large, and it takes too long to fit SMARTboost in modality = :compromise or :accurate,
one way to proceed is to cv on a subsample of the data (say 20%) and then fit only one model on the full sample.
This can be accomplished as follow

- set modality=:compromise or :accurate, take a subsample of the data (say 10% or 20%), and run output=SMARTfit() on that.
- set param=output.bestparam, and then param.modality=:fast or :fastest, and run SMARTfit() on the full data.
  
This process works because 

paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboost

using Random,Statistics

# USER'S OPTIONS 
Random.seed!(123)

# Options for data generation 
n         = 400_000
p         = 10          # number of features 
dummies   = true        # true if x, x_test are 0-1 (much faster).
stde      = 1            

# Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

randomsubset      = 0.1          # e.g. 0.1 or 0.2. Share of observations in the first sub-set 
modality_subs     = :compromise  # :accurate or :compromise (default)
modality_full     = :fast        # :fast or :fastest

nfold_subs       = 1             # number of cv folds. 1 sufficient if the sub-sample is sufficiently large 
nfold_full       = 1         

verbose          = :Off
warnings         = :On


# function: Friedman of linear, with pstar relevant features.
# Increasing the number of relevant features increases the difficulty of the problem and can be
# used to evaluate speed gains and accuracy losses of sparsevs.

Friedman_function(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

p_star    = 10       # number of relevant features 
β = randn(p_star)
Linear_function_Gaussian(x)   = x[:,1:length(β)]*β

#f_dgp     = Friedman_function     
f_dgp    = Linear_function_Gaussian

# END USER'S INPUTS 

if f_dgp==Friedman_function
    x,x_test = rand(n,p), rand(200_000,p)    # Friedman function on U(0,1)
else 
    x,x_test = randn(n,p), randn(200_000,p)    

    if dummies 
        x,x_test = Float64.(x .> 0), Float64.(x_test .> 0)
    end 
end     


f       = f_dgp(x)
y      = f + stde*randn(n)
f_true = f_dgp(x_test)

# SMARtboost

param_subs   = SMARTparam(modality=modality_subs,nfold=nfold_subs,verbose=verbose,warnings=warnings)
data         = SMARTdata(y,x,param)
n            = length(data.y)

ind       = randperm(n)[1:convert(Int,round(randomsubset*n))]
data_subs = SMARTdata(y[ind],x[ind,:],param)

output_subs = SMARTfit(data_subs,param_subs)
param_full = output_subs.bestparam 
param_full.modality = modality_full 
param_full.nfold    = nfold_full

output     = SMARTfit(data,param_full)

println("\n n = $n, p = $p, dummies=$dummies, modality = $modality")
println(" time to fit ")

@time output = SMARTfit(data,param);

yf = SMARTpredict(x_test,output,predict=:Ey)  # predict
println("\n RMSE of SMARTboost from true E(y|x) ", sqrt(mean((yf-f_true).^2)) )

