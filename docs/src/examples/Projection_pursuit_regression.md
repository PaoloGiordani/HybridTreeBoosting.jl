# Hybrid trees incorporate elements of projection pursuit regression

**A simulated example in which adding a single index model to each tree improves accuracy**

- The boosted sequence of single index models builds a projection pursuit regressio of sort. 
- Data is simulated from f(z), where z is a linear combination of the features and f() is ReLu.
- Adding a projection pursuit regression to each tree improves accuracy considerably in this admittedly 
  artificial example.
- LightGBM and XGB struggle with this type of data, typically requiring many trees and delivering poor accuracy.

```julia 

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using HTBoost

using Random,Plots 
using LightGBM

# USER'S OPTIONS 

# Some options for HTBoost
loss      = :L2            # :L2 or :sigmoid (or :Huber or :t). 
modality  = :compromise     # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees

randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :Off
 
# options to generate data. y = sum of six additive nonlinear functions + Gaussian noise.
n,n_test  = 10_000,100_000
stde      = 1.0
b         = 5.0
rndseed   = 1234

# END USER'S OPTIONS

function simulatedata(n,b,stde;rndseed=1)
    
    Random.seed!(rndseed)

    x     = randn(n,5)
    z     = 2.5*x[:,1] + 2.0*x[:,2] + 1.5*x[:,3] + 1.0*x[:,4] + 0.5*x[:,5]

    f = b*z.*(z.>0)
    y = f + stde*randn(n)

    return y,x,f 

end 

y,x,f             = simulatedata(n,b,stde,rndseed=rndseed)
y_test,x_test,f_test = simulatedata(100_000,b,stde,rndseed=rndseed)

# set up HTBparam and HTBdata, then fit and predit
param  = HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,warnings=warnings,
           modality=modality,nofullsample=nofullsample,lambda=lambda)

data   = HTBdata(y,x,param)

for depthppr in [0,2]

    param.depthppr = depthppr

    @time local output = HTBfit(data,param)
    local yf  = HTBpredict(x_test,output) 

    println(" \n modality = $(param.modality), nfold = $nfold, depthppr=$(param.depthppr) ")
    println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
    println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

   if param.depthppr>0    # visualize impact of projection pursuit transformation on first tree
      yf1,yf0,tau = HTBppr_plot(output,which_tree=1)
      plot(yf0,yf1,title="depthppr=$(param.depthppr)")
   end


end 

# LightGBM 
estimator = LGBMRegression(objective = "regression",num_iterations = 1000,early_stopping_round = 100)

y       = Float64.(data.y)                 
n_train = Int(round((1-param.sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])

# parameter search over num_leaves and max_depth
splits = (collect(1:n_train),collect(1:min(n_train,100)))  # goes around the problem that at least two training sets are required by search_cv (we want the first)

params = [Dict(:num_leaves => num_leaves,
           :max_depth => max_depth) for
          num_leaves in (4,16,64,256),
         max_depth in (2,4,6,8)]

println("\n running lightGBM")
@time lightcv = LightGBM.search_cv(estimator,x,y,splits,params,verbosity=-1)

loss_cv = [lightcv[i][2]["validation"][estimator.metric[1]][1] for i in eachindex(lightcv)]
minind = argmin(loss_cv)

estimator.num_leaves = lightcv[minind][1][:num_leaves]
estimator.max_depth  = lightcv[minind][1][:max_depth]

# fit at cv parameters
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
yf_gbm = LightGBM.predict(estimator,x_test)[:,1]

println(" lightgbm depth = $(estimator.max_depth), nleaves = $(estimator.num_leaves) ")
println(" out-of-sample RMSE lightgbm  ", sqrt(sum((yf_gbm - f_test).^2)/n_test) )

```