#= 

Explore how preliminary variables (feature) selection affects speed and accuracy.

The preliminary phase picks 100 features as follows: for d>=min_d_pvs, instead of adding a level,
it adds a stomp. The idea is that features that interact well should also leave an additive trace. 
Preliminary experiments show no loss of fit in the simple dgp simulated below, and modest loss
of fit on real data. Noticeable speed gains seen only for depth >= 6.
The speed gains are 50-60% without sparsevs, but more like 25-30% with sparsevs, and smaller if depth < 6. 
Due to the potential loss of fit, the default is :Off, but it may be useful in exploratory phases with very large p and depth>5.

Reasons for the modest computational gains, particularly since sparsevs is on:
- parallelization becomes more efficient for larger p*/ncores, where p* is the number of candidate features at a given split.
- sparsevs sets p*<p. 
- fixed cost of refineOptim

paolo.giordani@bi.no
=#

number_workers  = 16  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Statistics
using LightGBM

# USER'S OPTIONS 
Random.seed!(123)

# Options for data generation 
n         = 10_000
p         = 2000        # number of features 
dummies   = false        # true if x, x_test are 0-1 (faster).
stde      = 1            

# Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

depth           = 6 

pvs              = :On
p_pvs            = 100
min_d_pvs        = 4    # If I set too low, more trees may be needed .... 

sparsevs         = :On      # when :On, pvs will have no impact except in very dense settings

for p_pvs in [100,100_000]

@show p_pvs 
Random.seed!(123)

modality         = :fast   # :accurate, :compromise (default), :fast, :fastest


nfold            = 1     # 1 for fair comparison with LightGBM
nofullsample     = true  # true for fair comparison with LightGBM

verbose          = :Off
warnings         = :On

ntrees           = 20   # maximum number of trees 

# function: Friedman of linear, with pstar relevant features.
# Increasing the number of relevant features increases the difficulty of the problem and can be
# used to evaluate speed gains and accuracy losses of sparsevs.

Friedman_function(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

p_star    = min(10,p)       # number of relevant features 
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

param   = HTBparam(modality=modality,ntrees=ntrees,sparsevs=sparsevs,pvs=pvs,p_pvs=p_pvs,min_d_pvs=min_d_pvs,depth=depth,
                    nfold=nfold,verbose=:Off,warnings=:On,nofullsample=nofullsample)

data  = HTBdata(y,x,param)

println("\n n = $n, p = $p, dummies=$dummies, modality = $modality, depth=$depth, sparsevs = $sparsevs, pvs = $pvs")
println(" time to fit ")

@time output = HTBfit(data,param);
println(" ntrees $(output.ntrees)")
yf = HTBpredict(x_test,output,predict=:Ey)  # predict

println("\n RMSE of HTBoost from true E(y|x)                   ", sqrt(mean((yf-f_true).^2)) )
println(" RMSE of LightGBM (default param) from true E(y|x)     ", sqrt(mean((yf_gbm-f_true).^2)) )

fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false);
println(" HTBoost number of included features in final model $(sum(fi.>0))")

if param.pvs == :On
    @info " pvs is :On. Repeat with pvs=:Off to track speed gains and any accuracy loss."
else 
    @info " pvs is :Off. Repeat with sparsevs=:On to track speed loss and any accuracy gains."
end     

end 
