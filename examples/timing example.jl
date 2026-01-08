
#=

This file is a modification of "Basic_use.jl" from the HybridTreeBoosting.jl package

My times to run this are: 

number_workers = 1   50''
number_workers = 2   50''
number_workers = 4   39''
number_workers = 8   37''

out-of-sample RMSE from truth 0.35

(In this example the number of features is only 6, so the speedup is limited).

=#


number_workers = 1

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HybridTreeBoosting

using Random

# USER'S OPTIONS 

Random.seed!(1)

# Some options for HTBoost
loss      = :L2            # :L2 is default. Other options for regression are :L2loglink (if y≥0), :t, :Huber
modality  = :fastest       # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees
randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :Off
 
# options to generate data. y = sum of six additive nonlinear functions + Gaussian noise.
n,p,n_test  = 10_000,6,100_000
stde        = 1.0

f_1(x,b)    = @. b*x + 1 
f_2(x,b)    = @. 2*sin(2.5*b*x)  
f_3(x,b)    = @. b*x^3
f_4(x,b)    = @. b*(x < 0.5) 
f_5(x,b)    = @. b/(1.0 + (exp(4.0*x )))
f_6(x,b)    = @. b*(-0.25 < x < 0.25) 

b1,b2,b3,b4,b5,b6 = 1.5,2.0,0.5,4.0,5.0,5.0

# END USER'S OPTIONS

# generate data
x,x_test = randn(n,p), randn(n_test,p)

f        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4) + f_5(x[:,5],b5) +  f_6(x[:,6],b6)
f_test   = f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4) + f_5(x_test[:,5],b5) + f_6(x_test[:,6],b6)

y = f + stde*randn(n)

# set up HTBparam and HTBdata, then fit and predit
param  = HTBparam(loss=loss,priortype=priortype,nfold=nfold,verbose=verbose,
                warnings=warnings,modality=modality,nofullsample=nofullsample)

data   = HTBdata(y,x,param)

# run once with one tree to eliminate compilation time from timing
param_compile = deepcopy(param)
param_compile.ntrees = 1 
output = HTBfit(data,param_compile)


# now run for real
@time output = HTBfit(data,param)
yf     = HTBpredict(x_test,output)  
println(" out-of-sample RMSE from truth ", round(sqrt(sum((yf - f_test).^2)/n_test) ,digits=4) )

