
"""

Purpose and main results:

- Show how Huber loss functions leads to biased fitted and predicted values when the errors have a skewed distribution,
  and the resulting MSE can be much higher than for L2 loss.
- In contrast, if the errors are fat-tailed but symmetric, the lightGBM Huber loss tends to outperform L2 loss.
- SMARTboost with loss = :t and loss=:Huber corrects for biases due to skewed errors. (t recommended over Huber)
- In SMARTboost, the t loss (plus de-biasing) improves on the L2 loss in this settings (due to IID errors) 
- Correcting the bias improves the MSE of lightGBM predictions compared to the original version, but
  the RMSE is often inferior to L2 loss. 
- The impact of the bias is stronger if the signal-to-noise is low. 
- With high signal-to-noise, the lightGBM Huber loss gives, after de-biasing, nearly identical results to the L2 loss, because the 
  robustness parameter is calibrated once on the unconditional data. In contrast, SMARTboost re-estimates all parameters
  after each tree, so that the t loss outperforms the L2 loss for any signal-to-noise (with IID errors). 

paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboostPrivate
#include("E:\\Users\\A1810185\\Documents\\A_Julia-scripts\\Modules\\SMARTboostPrivateLOCAL.jl") # no package

using Random,Statistics,Plots
using LightGBM

# USER'S OPTIONS 

Random.seed!(123)

# Options for data generation 
n         = 10_000
p         = 5      # p>=5. Only the first 4 variables are used in the function f(x) below 
stde      = 10     # e.g. 1 for high SNR, 5 for lowish, 10 for low (R2 around 4%) 

m2        = 3*stde  # mean of second component of mixture of normal. 0 for symmetric fat tails, 3*stde for skewed

# Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

loss      = :t      # :t (recommended) or :Huber for SMARTboost. :t automatically estimates degrees of freedom and can recover a Gaussian 
modality  = :fast   # :accurate, :compromise (default), :fast, :fastest
ntrees    = 200     # maximum number of trees for SMARTboost. 

# define the function dgp(x), here the Friedman's function for x~U  
dgp(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

# End user's options 

# generate data. x is standard uniform, and errors are a mixture of two normals, with right skew
x,x_test   = rand(n,p), rand(n_test,p)
ftrue      = dgp(x)
ftrue_test = dgp(x_test)

stde2 = 3*stde
u1    = randn(n)*stde
u2    = m2 .+ randn(n)*stde2
prob  = 0.3
S1    = rand(n).>prob 
u     = @. u1*S1 + u2*(1 - S1) - prob*m2     # skewed distribution with zero mean  
y      = ftrue + u

histogram(u,title="errors",label="")

# SMARTboost parameters
param  = SMARTparam(loss=loss,nfold=1,ntrees=ntrees,nofullsample=true,modality=modality,verbose=:Off)
data   = SMARTdata(y,x,param)

# ligthGBM parameters 
estimator = LGBMRegression(
    objective = "regression",
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    num_threads = number_workers,
    max_depth = 6,      # -1 default
    min_data_in_leaf = 100,  # 100 default 
    num_leaves = 127         # 127 default  
)

estimator_huber = LGBMRegression(
    objective = "huber",
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    num_threads = number_workers,
    max_depth = -1,      # -1 default
    min_data_in_leaf = 100,  # 100 default 
    num_leaves = 127         # 127 default  
)


# Fit lightGBM 

n_train = Int(round((1-param.sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])
    
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm = LightGBM.predict(estimator,x_test)
yf_gbm2 = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 
MSE2    = sum((yf_gbm2 - ftrue_test).^2)/n_test

println("\n bias = E(prediction) - E(y) ")
println("\n bias of lightGBM with L2 loss    ", mean(yf_gbm-ftrue_test))

LightGBM.fit!(estimator_huber,x_train,y_train,(x_val,y_val),verbosity=-1)
yf_gbm = LightGBM.predict(estimator_huber,x_test)
yf_gbm = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 
MSE3    = sum((yf_gbm - ftrue_test).^2)/n_test

println(" bias of lightGBM with Huber loss ", mean(yf_gbm-ftrue_test))

println("\n correlation of fitted values, lightGBM L2 and Huber ", cor(yf_gbm,yf_gbm2))
# correct the bias of lightGBM 
yhat = LightGBM.predict(estimator_huber,x_train)
bias = mean(yhat) - mean(y_train)
yf_unbiased = yf_gbm .- bias 
MSE4    = sum((yf_unbiased - ftrue_test).^2)/n_test

println("\n oos RMSE from true f(x), lightGBM, Huber loss                    ", sqrt(MSE3) )
println(" oos RMSE from true f(x), lightGBM, Huber loss, de-biased         ", sqrt(MSE4) )
println(" oos RMSE from true f(x), lightGBM, L2 loss                       ", sqrt(MSE2) )

# Fit SMARTboost, :t (or :Huber) 
output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output)  
MSE1    = sum((yf - ftrue_test).^2)/n_test

println(" oos RMSE from true f(x) parameter, SMARTboost, loss=$loss            ", sqrt(MSE1) )

# Fit SMARTboost, :L2 
param.loss = :L2 
output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output)  
MSE0    = sum((yf - ftrue_test).^2)/n_test

println(" oos RMSE from true f(x) parameter, SMARTboost, loss=L2           ", sqrt(MSE0) )

