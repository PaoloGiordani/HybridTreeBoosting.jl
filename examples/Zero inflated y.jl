"""

#Strategies for zero-inflated data.

## Discussion of options available in HTBoost.

HTBoost has three loss functions for zero inflated data:
    ```:hurdleGamma, :hurdleL2, :hurdleL2loglink```
The :hurdleGamma is closest to the Tweedie distribution in LightGBM, XGB, and CatBoost.
Hurdle models in HTBoost build to separate models, one with logistic loss to predict
the occurence of a zero, and a second model with loss gamma or L2 or L2loglink to predict
y|y≠0. Compared to a Tweedie regression, hurdle models have richer parametrization but
far weaker constraints on the process, implying higher variance and smaller bias.
My reading of the literature is that hurdle model typically outperform Tweedy in terms of
forecasting. 

While :hurdleGamma and :hurdleL2loglink require y≥0, a :hurdleL2 loss can be used if
some y are negative. A hurdleL2 loss could therefore also be used if an otherwise continuous
y has positive mass at some value v other than zero, by working with y-v. 

A hurdleL2loglink loss can be a strong alternative to a hurdleGamma loss, if the gamma
assumption is incorrect.

A hurdleL2loglink is also an option for zero-inflated count data, as an alternative to
a Poisson or GammaPoisson. 

## What this script does. 

- Generates data from a gamma distribution, which is then transformed to produce excess zeros. 
- Fit HTBoost, with loss = :hurdleGamma or :hurdleL2loglink
- A comparison with LightGBM using the Tweedie loss is promising.
- HTBpredict takes the form:
```    yf,prob0,yf_not0     = HTBpredict(x_test,output) ``
where yf = E(y|x) = (1-prob0)*yf_not0

"""
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Plots,Distributions 
using LightGBM

# USER'S OPTIONS 

Random.seed!(1)

# Some options for HTBoost
loss      = :hurdleGamma    # options for y>=0 data are :L2loglink, :L2, :gamma, :hurdleGamma, :hurdleL2loglink, :hurdleL2     
modality  = :compromise     # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees

randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :On

# options to generate data. 
true_k      = 10     # dispersion parameter of gamma distribution
α           = 0.5    # Generates y=0 data. Set to 0 for all y strictly positive, 0.2 (0.3,0.6) has around 40% (60%,80%) y=0 

n,p,n_test  = 10_000,4,100_000

f_1(x,b)    = b./(1.0 .+ (exp.(1.0*(x .- 1.0) ))) .- 0.1*b 
f_2(x,b)    = b./(1.0 .+ (exp.(4.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(8.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(16.0*(x .+ 0.5) ))) .- 0.1*b

b1,b2,b3,b4 = 0.2,0.2,0.2,0.2

# generate data
x,x_test = randn(n,p), randn(n_test,p)

c        = -2  
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

μ        = exp.(f)        # conditional mean 
μ_test   = exp.(f_test)   # conditional mean 

# k can depend on features for a ≠ 1
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
    μ_test[i] < α*rand(1)[1] ? y_test[i] = 0.0 : nothing    
end 

println("\n share of y=0 is $(mean(y.==0)) \n")
histogram(y,title="y, unconditional distribution") 

# set up HTBparam and HTBdata, then fit and predit

param  = HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = HTBdata(y,x,param)
output = HTBfit(data,param)

println(" \n loss = $loss, modality = $(param.modality), nfold = $nfold ")

if loss in [:hurdleL2,:hurdleL2loglink,:hurdleGamma]
    yf,prob0,yf_not0     = HTBpredict(x_test,output)
    println(" depth logistic = $(output[1].bestvalue), number of trees logistic = $(output[1].ntrees) ")
    println(" depth = $(output[2].bestvalue), number of trees = $(output[2].ntrees) ")
else     
    yf     = HTBpredict(x_test,output,predict=:Ey)
    println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
end 

println(" out-of-sample RMSE (y-yf), HTBoost       ", sqrt(sum((yf - y_test).^2)/n_test) )

# ligthGBM 
estimator = LGBMRegression(
    objective = "tweedie",
    metric = ["tweedie"],
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    num_threads = number_workers,
    max_depth = 6,      # -1 default
    min_data_in_leaf = 100,  # 100 default 
    num_leaves = 127         # 127 default  
)

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

# re-fit at cv parameters
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm = LightGBM.predict(estimator,x_test)
yf_gbm = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 

println("\n out-of-sample RMSE (y-yf), LightGBM cv      ", sqrt(sum((yf_gbm - y_test).^2)/n_test) )




    





