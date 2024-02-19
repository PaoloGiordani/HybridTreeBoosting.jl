""" 

Small variation on *Missing data 2.jl*

The set-up now mimicks a situation in which missing values can be interpolated from two features (as opposed to just one),
which may be common in time series settings, as well as more options.  

The first change is that there are 2*p (as opposed to just p) more features. e.g. for p=4, with x5=x1+noise, x6=x2+noise ....,
x9=x1+noise,x10=x2+noise, where noise is N(0,σ²). Hence a missing value can be interpolated from two other features.
By Bayes' theorem, E(x[i,j]|x[i,≠j]) = ((x[i,j+p]+x[i,j+2*p])/σ²)/(1+2/σ²) 
The results are in line with those of *Missing data 2.jl*.

The second change is that more options for treating missing is added: besides 1) SMARTboost default and 2) optimally imputed x,
3) replace missing by the unconditional mean and add a mask (a dummy feature with value 1 for missing), see Josse et al. 2020, "On the consistency of supervised learning with missing values"
4) replace missing by the optimally imputed value AND add a mask. 

The results are in line with Josse et al.: 3) is also consistent (unlike imputing the true conditional mean!),
but it is slightly inferior to Block Propagation 1), particularly when the missing value is highly predictable (low σ²).  
The best option in finite sample is 4), followed by 1). 

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
n         = 10_000
stdn      = 1.0     # std of the noise in features. As stdn --> 0, missing values should cause no loss of oss performance
prob_miss = 0.33    # percentage of missing values in first p features (missing at random)
stde      = 0.5     # y = f(x) + stde*u, where n~N(0,1)

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

for stdn in [0.5,1.0,2.0]
#for f_dgp in ["linear","quadratic","mixed"]
for f_dgp in ["linear","quadratic"]

  a = power_coeff(f_dgp)
  
  dgp(x,b,a) = sum(b[i]*x[:,i].^a[i] for i in eachindex(b))   # E(y|x)

  # draw features, without missing values, and compute f_true = E(y|x)
  x1 = randn(n,p)
  x2 = x1 + randn(n,p)*stdn
  x3 = x1 + randn(n,p)*stdn
  x  = hcat(x1,x2,x3)

  x1 = randn(n_test,p)
  x2 = x1 + randn(n_test,p)*stdn
  x3 = x1 + randn(n_test,p)*stdn
  x_test = hcat(x1,x2,x3)
  x_test_no_missing = copy(x_test)

  f_true = dgp(x,b,a)
  f_test = dgp(x_test,b,a)
  y      = f_true + randn(n)

  # Add missing values (at random)
  ind      = rand(n,p) .< prob_miss
  ind_test = rand(n_test,p) .< prob_miss

  for j in 1:p
    x[ind[:,j],j] .= NaN
    x_test[ind_test[:,j],j] .= NaN 
  end 

  # Compute optimal imputation values using Bayes theorem 
  x_impute,x_test_impute = copy(x),copy(x_test)
  x_impute_mm,x_test_impute_mm = hcat(copy(x),zeros(n,p)),hcat(copy(x_test),zeros(n_test,p))   # impute mean and mask 
  x_impute_em,x_test_impute_em = hcat(copy(x),zeros(n,p)),hcat(copy(x_test),zeros(n_test,p))   # impute expectation and mask 
     
  varn = stdn^2

  for j in 1:p
    m = mean(x[.!isnan.(x[:,j]),j])
    for i in 1:n
        if isnan(x[i,j])
            x_impute[i,j] = ((x[i,j+p]+x[i,j+2*p])/varn)/(1+2/varn)
            x_impute_mm[i,j] = m
            x_impute_mm[i,3*p+j] = 1.0
            x_impute_em[i,j] = x_impute[i,j]
            x_impute_em[i,3*p+j] = 1.0
        end      
    end 

    for i in 1:n_test
        if isnan(x_test[i,j])
            x_test_impute[i,j] = ((x_test[i,j+p]+x_test[i,j+2*p])/varn)/(1+2/varn)
            x_test_impute_mm[i,j] =  m
            x_test_impute_mm[i,3*p+j] = 1.0
            x_test_impute_em[i,j] = x_test_impute[i,j]
            x_test_impute_em[i,3*p+j] = 1.0
        end      
    end 

  end   



  # Compare SMARTboost with missing data vs SMARTboost with imputed data 
  param        = SMARTparam(priortype=priortype,randomizecv=true,nfold=nfold,modality=modality,verbose=verbose,warnings=warnings )
  data         = SMARTdata(y,x,param)
  data_impute  = SMARTdata(y,x_impute,param)
  data_impute_mm  = SMARTdata(y,x_impute_mm,param)
  data_impute_em  = SMARTdata(y,x_impute_em,param)

  output        = SMARTfit(data,param)
  output_impute = SMARTfit(data_impute,param)
  output_impute_mm = SMARTfit(data_impute_mm,param)
  output_impute_em = SMARTfit(data_impute_em,param)

  yf        = SMARTpredict(x_test,output)  
  yf_impute = SMARTpredict(x_test_impute,output_impute)
  yf_impute_mm = SMARTpredict(x_test_impute_mm,output_impute_mm)
  yf_impute_em = SMARTpredict(x_test_impute_em,output_impute_em)

  println("\n ** DGP is $f_dgp ** ")
  println("\n Share of missing = $prob_miss, stdn = $stdn, n = $n ")
  println("\n oos RMSE from truth, x has missing, SMARTboost with missing             ", sqrt(sum((yf - f_test).^2)/n_test) )
  println(" oos RMSE from truth, x has missing, SMARTboost optimally imputed x      ", sqrt(sum((yf_impute - f_test).^2)/n_test) )
  println(" oos RMSE from truth, x has missing, SMARTboost imputed mean + mask      ", sqrt(sum((yf_impute_mm - f_test).^2)/n_test) )
  println(" oos RMSE from truth, x has missing, SMARTboost optimally imputed + mask ", sqrt(sum((yf_impute_em - f_test).^2)/n_test) )

  yf        = SMARTpredict(x_test_no_missing,output)  
  yf_impute = SMARTpredict(x_test_no_missing,output_impute)
  yf_impute_mm = SMARTpredict(hcat(x_test_no_missing,zeros(n_test,p)),output_impute_mm)
  yf_impute_em = SMARTpredict(hcat(x_test_no_missing,zeros(n_test,p)),output_impute_em)

  println("\n oos RMSE from truth, x has NO missing, SMARTboost with missing             ", sqrt(sum((yf - f_test).^2)/n_test) )
  println(" oos RMSE from truth, x has NO missing, SMARTboost optimally imputed x      ", sqrt(sum((yf_impute - f_test).^2)/n_test) )
  println(" oos RMSE from truth, x has NO missing, SMARTboost imputed mean + mask      ", sqrt(sum((yf_impute_mm - f_test).^2)/n_test) )
  println(" oos RMSE from truth, x has NO missing, SMARTboost optimally imputed + mask ", sqrt(sum((yf_impute_em - f_test).^2)/n_test) )

 # fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data,verbose=false);

end
end  