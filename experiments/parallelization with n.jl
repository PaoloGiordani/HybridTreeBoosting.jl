
"""

Goal: more efficient parallelization of SMARTboost.

Background:
- Current parallelization is most efficient when n is large
- If n is small, we'll typically perform cross-validation with 4-10 folds, and should be 
  more efficient to parallelize the folds, or, depending on the number of workers, some
  combination of folds and features.
- For example, if param.nfold = 4 (default), we could assign 8
  cores to each fold. This should lead to overall speed-ups close to theoretical max all the way to 32 cores.   
  If nfold = 1, we could assign 8 or 16 cores to each model in SMARTfit, essentially cross-validating several
  models at once. These are embarassingly parallel tasks, and may lead to excellent speed-ups for large
  number of workers. I don't know how to code this, since @parallel loops cannot be nested, but I am pretty
  sure that it can be done in Julia with some work and skill.  
- Here we associate some initial numbers to these statements.  

Notes:
- p = 100, which may be effectively less since sparsevs is on. We know from previous experiments
  that parallelization is more efficient when p/workers is largish. 

Findings:
- Computing time increases less than linearly with n up to roughly 10k. 
- Speed-ups at low n are good only for small number of workers. More than 8 cores are wasted
  unless n >> 10k.
- Speeds-ups from 8 to 16 cores are modest here (20-25%), even at large n, because p is small and sparsevs is on.
  With p = 500 and sparsevs off, 8 to 16 approaches 40%, but this would happen rarely. 
- ? Why is 1 worker so slow on the remote server? Not so on my laptop, where 1 worker is fast. 

Preliminary conclusions:
- In what I consider to be the most common use case, 8 cores parallelize very well in the current settings
  (across features). If more cores are available, it should be more efficient to build a nested parallelization,
  and parallelize across folds and/or models. 
- If param.nfold = 4 (default) it may be a decent first approach (although not optimized) to assign 8
  cores to each fold. This should lead to overall speed-ups close to theoretical max all the way to 32 cores.   


TIMED RUNS

 p = 100, 100 trees 
 
 cores   time  

        n= 1_000       n= 10_000      n = 100_000     n = 1_000_000 

 1        30.5            202            841      
 
 2        9.9             43.9           402
 4        5.2             21.5           171
 8        4.9             15.3           101            1294
 16       6.3             15.1           83             981 
 


paolo.giordani@bi.no
"""


number_workers  = 16  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboostPrivate

using Random,Statistics

# USER'S OPTIONS 
#for n in [1_000,10_000,100_000] 
for n in [1_000_000]

Random.seed!(123)

# Options for data generation 
#n         = 1_000
p         = 1_00        # number of features 
dummies   = false        # true if x, x_test are 0-1 (faster).
stde      = 1            

# Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

sparsevs         = :On
modality         = :fastest   # :accurate, :compromise (default), :fast, :fastest

number_best_features = 10  # Default 10. <10 to consider fewer features in each split (larger speed gains, less precision)
frequency_update     = 1       # Default 1. >1 to update less frequently (larger speed gains, less precision)

nfold            = 1     # 1 for fair comparison with LightGBM
nofullsample     = true  # true for fair comparison with LightGBM

verbose          = :Off
warnings         = :On

ntrees           = 100   # maximum number of trees 

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


param   = SMARTparam(modality=modality,ntrees=ntrees,sparsevs=sparsevs,
                    frequency_update=frequency_update,number_best_features=number_best_features,
                    nfold=nfold,verbose=:Off,warnings=:Off)

data  = SMARTdata(y,x,param)

# run with one tree so @time is reliable 
param_compile = deepcopy(param); param_compile.ntrees = 1
output = SMARTfit(data,param_compile)

println("\n n = $n, p = $p, dummies=$dummies, modality = $modality, sparsevs = $sparsevs, frequency_update = $frequency_update")
println(" time to fit ")

@time output = SMARTfit(data,param);
yf = SMARTpredict(x_test,output,predict=:Ey)  # predict

println("\n RMSE of SMARTboost from true E(y|x)                   ", sqrt(mean((yf-f_true).^2)) )

end 