
#=

**Short description:**

- Comparison with lightGBM on a logistic regression problem with simulated data.
- param.modality as the most important user's choice.
- In default modality, HTBoost performs automatic hyperparameter tuning.


**Extensive description:** 

Sketch of a comparison of HTBoost and lightGBM on a logistic regression problem.
The comparison with LightGBM is biased toward HTBoost if the function generating the data is 
smooth in some features (this is easily changed by the user). lightGBM is cross-validated over max_depth and num_leaves,
with the number of trees set to 1000 and found by early stopping.

Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
:fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
automatic hyperparameter tuning. In HTBoost, it is not recommended that the user performs 
hyperparameter tuning by cross-validation, because this process is done automatically if modality is
:compromise or :accurate. The recommended process is to first run in modality=:fast or :fastest,
for exploratory analysis and to gauge computing time, and then switch to :compromise (default)
or :accurate.

=#
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HybridTreeBoosting

using Random,Statistics
using LightGBM

# USER'S OPTIONS 

Random.seed!(1)

# Options for data generation 
n         = 10_000
p         = 10      # mumber of features. p>=4. Only the first 4 variables are used in the function f(x) below 
nsimul    = 1       # number of simulated datasets. 

# Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

modality  = :fast   # :accurate, :compromise (default), :fast, :fastest

# define the function f(x), where x are indendent N~(0,1), and f(x) is for the natural parameter,
# so f(x) = log(prob/(1-prob))

f_1(x,b)    = b*x .+ 1 
f_2(x,b)    = sin.(b*x)  
f_3(x,b)    = b*x.^2
f_4(x,b)    = b./(1.0 .+ (exp.(5*(x .- 0.5) )))   

b1,b2,b3,b4 = 1.5,2.0,0.5,2.0
dgp(x)        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
 
# END USER'S OPTIONS  

function simul_logistic(n,p,nsimul,modality,dgp)

 n_test = 100_000
 loss = :logistic

 # initialize containers and parameters for HTBoost and lightGBM
 MSE1 = zeros(nsimul)
 MSE2 = zeros(nsimul)

 param  = HTBparam(loss=loss,nfold=1,nofullsample=true,modality=modality,warnings=:Off,newton_gauss_approx =true)

 # Create an estimator with the desired parameters—leave other parameters at the default values.
 estimator = LGBMClassification(   # LGBMRegression(...)
    objective = "binary",
    num_class = 1,              # relevant only for multiclass
    categorical_feature = Int[],
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    metric = ["binary_logloss"],
    num_threads = number_workers,
    device_type="cpu",
    max_depth = -1,      # -1 default
    min_data_in_leaf = 100,  # 100 default 
    num_leaves = 127         # 127 default  
 )

 for simul in 1:nsimul

    # generate data
    x,x_test = randn(n,p), randn(n_test,p)
    ftrue       = dgp(x)
    ftrue_test  = dgp(x_test)

    y = (exp.(ftrue)./(1.0 .+ exp.(ftrue))).>rand(n) 
    data   = HTBdata(y,x,param)

    output = HTBfit(data,param)
    yf     = HTBpredict(x_test,output,predict=:Egamma)  # predict the natural parameter (only with simulated data: typically we'll want to predict=:Ey (default))
    MSE1[simul]    = sum((yf - ftrue_test).^2)/n_test

    # lightGBM
    y       = Float64.(y)                 
    n_train = Int(round((1-param.sharevalidation)*length(y)))
    x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
    x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])
    
   # parameter search over num_leaves and max_depth
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

   # fit at cv parameters
   LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)

    yf_gbm = LightGBM.predict(estimator,x_test)[:,1]
    yf_gbm = log.(yf_gbm./(1.0 .- yf_gbm))

    MSE2[simul]    = sum((yf_gbm - ftrue_test).^2)/n_test


 end     

 return MSE1,MSE2

end 


MSE1,MSE2 = simul_logistic(n,p,nsimul,modality,dgp)

println("\n n = $n, p = $p, number of simulations = $nsimul, modality = $modality")
println(" avg out-of-sample RMSE from true natural parameter, HTBoost    ", sqrt(mean(MSE1)) )
println(" avg out-of-sample RMSE from true natural parameter, lightGBM      ", sqrt(mean(MSE2)) )

