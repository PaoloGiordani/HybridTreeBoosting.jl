"""

Short description:

- Generates data from a gamma distribution, which is then transformed to produce excess zeros. 
- The :L2 loss and :L2loglink, while not optimal, may perform quite well.
- A comparison with LightGBM using the Tweedie loss is promising. (See note below.)  

I NOW HAVE :hurdleGamma, :hurdleL2, :hurdleL2loglink !!!!!!!

L2loglink can be superior to gamma (say this also on gamma distribution.jl)
for y>0 as it is less restrictive. 

L2loglink and L2 can still be competitive ....


TRY ON AN ACTUAL DATASET ..... 

VARIABLE IMPORTANCE ETC.... WILL PRODUCE AN ERROR: 
- ! Code a message ! 
- can be done separately each model 
- for the joint model? 

Note: The comparison with LightGBM is biased toward SMARTboost if the function generating the data is 
smooth in some features (this is easily changed by the user). lightGBM is cross-validated over max_depth and num_leaves,
with the number of trees set to 1000 and found by early stopping.
 
paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboostPrivate
include("E:\\Users\\A1810185\\Documents\\A_Julia-scripts\\Modules\\SMARTboostPrivateLOCAL.jl") # no package

using Random,Plots,Distributions 
using LightGBM

# USER'S OPTIONS 

Random.seed!(1)

# Some options for SMARTboost
loss      = :hurdleL2loglink    # options for y>=0 data are :L2loglink, :L2, :gamma, :hurdleGamma, :hurdleL2loglink, :hurdleL2     
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

println(" \n loss = $loss, modality = $(param.modality), nfold = $nfold ")

if loss in [:hurdleL2,:hurdleL2loglink,:hurdleGamma]
    yf,prob0,yf_not0     = SMARTpredict(x_test,output)
    println(" depth logistic = $(output[1].bestvalue), number of trees logistic = $(output[1].ntrees) ")
    println(" depth = $(output[2].bestvalue), number of trees = $(output[2].ntrees) ")
else     
    yf     = SMARTpredict(x_test,output,predict=:Ey)
    println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
end 

println(" out-of-sample RMSE (y-yf)               ", sqrt(sum((yf - y_test).^2)/n_test) )

# lightGBM at default values 

# ligthGBM parameters 
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

# Fit lightGBM 

n_train = Int(round((1-param.sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])
    
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm_default = LightGBM.predict(estimator,x_test)[:,1]

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

println("\n oss RMSE from truth, μ, LightGBM cv      ", sqrt(sum((yf_gbm_default - y_test).^2)/n_test) )
println(" oss RMSE from truth, μ, LightGBM default ", sqrt(sum((yf_gbm - y_test).^2)/n_test) )




    





