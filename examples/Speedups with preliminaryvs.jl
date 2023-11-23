
"""

Short description:

- Explore how preliminaryvs affects speed and accuracy when both n and p are large.
  

Extensive description: 

In large n and large p situations, the following approximation can be used to speed up computations:
Instead of using the full sample to select the best feature , we proceed in two steps:
in the first step, features are ordered by their loss function computed on a random subset of the data
of size ns (n_prelimiminaryvs in code). The most promising (lowest loss) ps features are then taken to
a second stage, where the full sample is used to select the best feature.
Asymptotically, the computational speed-up from the variable selection phase is n*p/(ns*p + n*ps).
For example if ns=0.1 and ps=0.1p, the theoretical speed-up is 500%.

The default for preliminaryvs is :Off, because there is a possibility of reduced accuracy.
When n and p are large, swithing preliminaryvs=:On is an interesting option for preliminary exploration,
or if the sample size is simply too large to fit SMARTboost in a reasonable time.

The subsample size can be set by the user, or automatically calibrated by letting n_preliminaryvs=:Auto.
The :Auto default takes into account the sample size and signal-to-noise ratio, so that the speedup is 
larger when n is very large and/or the signal-to-noise ratio is high.

preliminaryvs and sparsevs can be used separately or together.

paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboostPrivate

using Random,Statistics

# USER'S OPTIONS 

Random.seed!(123)

# Options for data generation 
n         = 100_000    # sample size 
p         = 100        # number of features 
stde      = 1            

# Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

modality         = :fast   # :accurate, :compromise (default), :fast, :fastest

preliminaryvs    = :Off     # :On or :Off
n_preliminaryvs  = :Auto   # :Auto or integer, e.g. 20_000

sparsevs         = :Off    # preliminaryvs and sparsevs can be used separately or together

verbose          = :Off
warnings         = :On

ntrees           = 50     # maximum number of trees. 

# Increasing the number of relevant features increases the difficulty of the problem and can be
# used to evaluate speed gains and accuracy losses of sparsevs.

p_star    = 10         # number of relevant features 
β = randn(p_star)
Linear_function_Gaussian(x)   = x[:,1:length(β)]*β

f_dgp    = Linear_function_Gaussian

# END USER'S INPUTS 

x,x_test = randn(n,p), randn(200_000,p)    

f       = f_dgp(x)
y      = f + stde*randn(n)
f_true = f_dgp(x_test)

# SMARtboost

param   = SMARTparam(modality=modality,ntrees=ntrees,n_vs=n_vs,n_preliminaryvs=n_preliminaryvs,nofullsample=true,
                    preliminaryvs=preliminaryvs,sparsevs=sparsevs,verbose=:Off,warnings=:On)

data  = SMARTdata(y,x,param)

println("\n n = $n, p = $p, modality = $modality, sparsevs = $sparsevs, preliminaryvs = $preliminaryvs, n_preliminaryvs = $n_preliminaryvs")
println(" time to fit ")

@time output = SMARTfit(data,param);
yf = SMARTpredict(x_test,output,predict=:Ey) 

println("\n RMSE of SMARTboost from true E(y|x)                   ", sqrt(mean((yf-f_true).^2)) )

println(" The subsample size for the preliminary feature selection phase is $(output.bestparam.n_preliminaryvs)")

if preliminaryvs==:On
    ns = output.bestparam.n_preliminaryvs
    n1 = n*0.7
    theoretical_speedup = n1*p/(ns*p + n1*p*0.1)
end 

if param.preliminaryvs == :On
    @info " preliminaryvs is :On. Repeat with preliminaryvs=:Off to track speed gains and any accuracy loss."
else 
    @info " preliminaryvs is :Off. Repeat with sparsevs=:On to track speed loss and any accuracy gains."
end     

