"""

Short description:

- Generates data from a gamma distribution, which is then transformed to produce excess zeros. 
- The :L2 loss, while not optimal, may perform quite well in modality = :compromise or :accurate, which
  cross-validate whether to use an identity link or a log-link.
- A comparison with LightGBM using the Tweedie loss is promising. (See note below.)  
- Specialized distributions like Tweedie or Hurdle models are planned for SMARTboost, but meanwhile the defaults
  should provide competitive results in many cases even if y is right-skewed, zero-inflated, count, rank ...
  as long as the interest is on the conditional mean rather than in the entire distribution. 
       
Note: The comparison with LightGBM is biased toward SMARTboost because the function generating the data is 
smooth in some features (this is easily changed by the user), and because lightGBM is run at
default parameters (except that number of trees is set to 1000 and found by early stopping.)
 
paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboostPrivate

using Random,Plots,Distributions 

# USER'S OPTIONS 

Random.seed!(1234)

# Some options for SMARTboost
loss      = :L2            # :L2 or (faster) :L2loglink   
modality  = :compromise     # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees

randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :On

# options to generate data. 
true_k      = 10     # dispersion parameter of gamma distribution
α           = 0.6    # Generates y=0 data. Set to 0 for all y strictly positive, 0.2 (0.3,0.6) has around 40% (60%,80%) y=0 

n,p,n_test  = 100_000,4,100_000

f_1(x,b)    = b*x  
f_2(x,b)    = -b*(x.<0.5) + b*(x.>=0.5)   
f_3(x,b)    = b*x
f_4(x,b)    = -b*(x.<0.5) + b*(x.>=0.5)

b1,b2,b3,b4 = 0.2,0.2,0.2,0.2

# generate data
x,x_test = randn(n,p), randn(n_test,p)

c        = -2  
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

μ        = exp.(f)        # conditional mean 
μ_test   = exp.(f_test)   # conditional mean 

# k can depend on features for a /= 1
a        = -0   
logk     = log(true_k)
logk     = logk .+ a*(b1*x[:,1] + b2*x[:,2] + b4*x[:,4])
k        = exp.(logk)
logk_test = log(true_k) .+ a*(b1*x_test[:,1] + b2*x_test[:,2] + b4*x_test[:,4] )
k_test    = exp.(logk_test) 
# end k dependent on features
scale    = μ./k
scale_test = μ_test./k_test
y       = zeros(n)
y_test  = zeros(n_test)

for i in eachindex(y)
    y[i]  = rand(Gamma.(k[i],scale[i]))
    μ[i] < α*rand(1)[1] ? y[i] = 0.0 : nothing    # zero-inflated data
end 

for i in eachindex(y_test)
    y_test[i]  = rand(Gamma.(k_test[i],scale_test[i]))
    μ_test[i] < α*rand(1)[1] ? y_test[i] = 0.0 : nothing    # zero-inflated data
end 

println("\n share of y=0 is $(mean(y.==0)) \n")
histogram(y,title="y, unconditional distribution") 
# set up SMARTparam and SMARTdata, then fit and predit

# coefficient estimated internally. 
param  = SMARTparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = SMARTdata(y,x,param)

output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output,predict=:Ey)

println(" \n loss = $loss, modality = $(param.modality), nfold = $nfold ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" out-of-sample RMSE (y-yf)               ", sqrt(sum((yf - y_test).^2)/n_test) )


# lightGBM at default values 

# ligthGBM parameters 
estimator = LGBMRegression(
    objective = "tweedie",
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    num_threads = number_workers,
    max_depth = 6,      # -1 default
    min_data_in_leaf = 100,  # 100 default 
    num_leaves = 127         # 127 default  
)

# Fit lightGBM 

n_train = Int(round((1-param.sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])
    
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm = LightGBM.predict(estimator,x_test)
yf_gbm = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 

println("\n oss RMSE from truth, μ, LightGBM default ", sqrt(sum((yf_gbm - y_test).^2)/n_test) )

