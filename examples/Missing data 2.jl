""" 

Question: when can data imputation outperform the in-built process for handling missing values in SMARTboost? 

Experiment set-up: 

- True data-generating-process is a function of x1,...,xp which are independent N(0,1)
- There are p more features. e.g. for p=4, with x5=x1+noise, x6=x2+noise etc..., where noise is N(0,σ²)
- Data is missing only in the first p features and at random, which is the best case for data imputation. 
- An upper bound for the efficiency of data imputation is to set missing values to
  E(x[i,j]|x[i,≠j]) = (x[i,j+p]/σ²)/(1+1/σ²) by Bayes theorem. 

Results: 

- Imputation with a single value is not consistent in general: does not deliver the true E(y|x) as n --> ∞,
  unless f(x) is linear, while GBM are Bayes consistent (Josse et al 2020, "On the consistency of supervised
  learning with missing values".) 
- Imputation will work best when:
  - f() is closer to linear and/or 
  - stdn is small: missing features are highly predictable
- For sufficiently large n, imputation has small gains (compared to SMARTboost internal process for missing)
  in the best-case scenario (linear+high correlations), and very large losses in worst-case scenario.    

In this example, for 
- stdn = 0.5, correlation(x[:,j],x[:j+p]) = 0.89   
- stdn = 1, correlation(x[:,j],x[:j+p]) = 0.7   
- stdn = 2, correlation(x[:,j],x[:j+p]) = 0.45
- stdn = 4, correlation(x[:,j],x[:j+p]) = 0.24

""" 

using DataFrames, Random, Statistics, Plots, Distributions

number_workers  = 8  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboost
#include("E:\\Users\\A1810185\\Documents\\A_Julia-scripts\\Modules\\SMARTboostPrivateLOCAL.jl") # no package

Random.seed!(1)

# USER'S OPTIONS 

Random.seed!(1)

# Options for data generation 
n         = 100_000
stdn      = 1.0     # std of the noise in features. As stdn --> 0, missing values should cause no loss of oss performance
prob_miss = 0.33    # percentage of missing values in first p features (missing at random)
stde      = 1.0     # y = f(x) + stde*u, where n~N(0,1)

b         = [1.0,1.0,1.0,1.0]   # vector. E(y|x) = b1*x1^a1 + ...      

function power_coeff(f_dgp)          #  E(y|x) = b1*x1^a1 + ...
  f_dgp == "linear" ? a = [1,1,1,1] : (f_dgp == "quadratic" ? a = [2,2,2,2] : a=[1,1,2,2])
  return a 
end 
  
# Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

loss      = :L2 
modality  = :fastest       # :accurate, :compromise (default), :fast, :fastest

priortype = :hybrid
nfold     = 1

verbose    = :Off 
warnings   = :Off

# END USER'S OPTIONS 

p      = length(b)
n_test = 100_000 

for stdn in [0.5,1.0,2.0,4.0]
for f_dgp in ["linear","quadratic","mixed"]

  a = power_coeff(f_dgp)
  
  dgp(x,b,a) = sum(b[i]*x[:,i].^a[i] for i in eachindex(b))   # E(y|x)

  # draw features, without missing values, and compute f_true = E(y|x)
  x1 = randn(n,p)
  x2 = x1 + randn(n,p)*stdn
  x  = hcat(x1,x2)

  x1 = randn(n_test,p)
  x2 = x1 + randn(n_test,p)*stdn
  x_test = hcat(x1,x2)
  x_test_no_missing = copy(x_test)

  f_true = dgp(x,b,a)
  f_test = dgp(x_test,b,a)
  y      = f_true + randn(n)*stde

  # Add missing values (at random)
  ind      = rand(n,p) .< prob_miss
  ind_test = rand(n_test,p) .< prob_miss

  for j in 1:p
    x[ind[:,j],j] .= NaN
    x_test[ind_test[:,j],j] .= NaN 
  end 

  # Compute optimal imputation values using Bayes theorem 
  x_impute,x_test_impute = copy(x),copy(x_test)
  varn = stdn^2

  for j in 1:p
    for i in 1:n
      isnan(x[i,j]) ? x_impute[i,j] = (x[i,j+p]/varn)/(1+1/varn) : nothing  
    end 

    for i in 1:n_test
      isnan(x_test[i,j]) ? x_test_impute[i,j] = (x_test[i,j+p]/varn)/(1+1/varn) : nothing  
    end 

  end   

  # Compare SMARTboost with missing data vs SMARTboost with imputed data 
  param        = SMARTparam(priortype=priortype,randomizecv=true,nfold=nfold,modality=modality,verbose=verbose,warnings=warnings )
  data         = SMARTdata(y,x,param)
  data_impute  = SMARTdata(y,x_impute,param)

  output        = SMARTfit(data,param)
  output_impute = SMARTfit(data_impute,param)

  yf        = SMARTpredict(x_test,output)  
  yf_impute = SMARTpredict(x_test_impute,output_impute)

  println("\n ** DGP is $f_dgp ** ")
  println("\n Share of missing = $prob_miss, stdn = $stdn, n = $n ")
  println("\n out-of-sample RMSE from truth, x has missing, SMARTboost with missing ", sqrt(sum((yf - f_test).^2)/n_test) )
  println(" out-of-sample RMSE from truth, x has missing, SMARTboost imputed x    ", sqrt(sum((yf_impute - f_test).^2)/n_test) )

  yf        = SMARTpredict(x_test_no_missing,output)  
  yf_impute = SMARTpredict(x_test_no_missing,output_impute)

  println("\n out-of-sample RMSE from truth, x has NO missing, SMARTboost with missing ", sqrt(sum((yf - f_test).^2)/n_test) )
  println(" out-of-sample RMSE from truth, x has NO missing, SMARTboost imputed x    ", sqrt(sum((yf_impute - f_test).^2)/n_test) )

  fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data,verbose=false)

end
end  