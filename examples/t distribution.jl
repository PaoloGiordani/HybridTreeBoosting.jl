
"""

**Purpose and main results:**

- Generates synthetic data with t-distributed errors.
- Show how to use the option loss = :t for leptokurtic (fat tailed) data, which is recommended over 
  the Huber loss since the degrees of freedom of the t distribution are estimated internally, thus
  providing the "right" amount of robustness (in the sense of maximizing the likelihood.)
- IF the errors are IID, the :t loss will generally outperform the :L2 loss. However, errors
  are often not IID, and then the :L2 loss can outperform the :t or :Huber loss even with
  leptokurtic errors. 

"""
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Statistics,Plots,Distributions
using LightGBM

# USER'S OPTIONS 

Random.seed!(123)

# Options for data generation 
n         = 5_000
p         = 5      # p>=5. Only the first 4 variables are used in the function f(x) below 
stde      = 1     # e.g. 1 for high SNR, 5 for lowish, 10 for low (R2 around 4%). Not really the stde unless dof is high. 
dof       = 3     # degrees of freedom for the t distribution to generate data. e.g. 3 or 5 or 10.

# Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

loss      = :t      # :t  
modality  = :fast   # :accurate, :compromise (default), :fast, :fastest

# define the function dgp(x), here the Friedman's function for x~U  
dgp(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

# End user's options 

# generate data. x is standard uniform, and errors are a mixture of two normals, with right skew
x,x_test   = rand(n,p),rand(n_test,p)
ftrue      = dgp(x)
ftrue_test = dgp(x_test)

t_object = TDist(dof)

u = rand(t_object,n)

y      = ftrue + u*stde

histogram(u,title="errors",label="")

# HTBoost parameters
param  = HTBparam(loss=loss,nfold=1,nofullsample=true,modality=modality,verbose=:Off)
data   = HTBdata(y,x,param)

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

LightGBM.fit!(estimator_huber,x_train,y_train,(x_val,y_val),verbosity=-1)
yf_gbm = LightGBM.predict(estimator_huber,x_test)
yf_gbm = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 
MSE3    = sum((yf_gbm - ftrue_test).^2)/n_test

println("\n oos RMSE from true f(x), lightGBM, Huber loss                    ", sqrt(MSE3) )
println(" oos RMSE from true f(x), lightGBM, L2 loss                       ", sqrt(MSE2) )

# Fit HTBoost, :t (or :Huber) 
output = HTBfit(data,param)
yf     = HTBpredict(x_test,output)  
MSE1   = sum((yf - ftrue_test).^2)/n_test
θ      = HTBcoeff(output,verbose=false)         # info on estimated coeff

println(" oos RMSE from true f(x) parameter, HTBoost, loss = $loss          ", sqrt(MSE1) )

# Fit HTBoost, :L2 
param.loss = :L2 
output_L2 = HTBfit(data,param)
yf     = HTBpredict(x_test,output_L2)  
MSE0    = sum((yf - ftrue_test).^2)/n_test

println(" oos RMSE from true f(x) parameter, HTBoost, loss = L2         ", sqrt(MSE0) )

println("\n true dof = $dof and estimated dof = $(θ.dof) ")
println("\n For more information about coefficients, use HTBcoeff(output) ")
HTBcoeff(output)