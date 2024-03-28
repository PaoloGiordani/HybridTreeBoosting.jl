#=

**HTBoost handles missing values automatically. Imputation is optional.** 

This example reproduces the set-up in the simulations i) Experiment 1 and ii) Model 2 and Model 3 in Experiment 2 in the paper:
"On the consistency of supervised learning with missing values" by Josse et al., 2020. 
Data can be missing at random, or missing not at random as a function of x only, or missing not at random as a function of E(y).

- The approach to missing values in HTBoost is Block Propagation (see Josse et al.).
- There is a however a a key difference compared to lightGBM and other GBM when the split is soft (τ < Inf): the value
  m at which to set all missing is estimated/optimized at each node. With standard trees (sharp splits), it only matters whether
  missing are sent to the left or right branch, but in HTBoost the allocation of missing values is also smooth (as long as the split is smooth),
  and the proportion to which missings are sent left AND right is decided by a new split parameter m, distinct from μ. 
  The result is more efficient inference with missing values. When the split is sharp or the feature takes only two values, HTBoost
  assigns missing in the same way as LigthGBM (by Block Propagation).
- The native procedure to handle missing values is Bayes consistent (see Josse et al.), i.e. efficient in large samples,
  and is convenient in that it can handle data of mixed types (continuous, discrete, categorical).
  When feasible, a good imputation of missing values + mask (see Josse et al. ) can perform better, particularly in small samples,
  high predictability of missing values from non-missing values, linear or quasi-linear f(x), and missing at random (in line
  with the results of Josse et al.)  
 
The comparison with LightGBM is biased toward HTBoost if the function generating the data is smooth in some features.
LightGBM is cross-validated over max_depth and num_leaves, with the number of trees set to 1000 and found by early stopping.
    
=# 
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using DataFrames, Random, Statistics
using LinearAlgebra,Plots, Distributions
using LightGBM

Random.seed!(1)

# Options to generate data. y is the of four additive nonlinear functions + Gaussian noise(0,stde^2)

Experiment      = "1"     # "1" or "2 Friedman" or "2 Linear"  (see Josse et al., 2020, experiment 1, and experiment 2, Friedman or Linear dgp)
missing_pattern = 1       # Only relevant for Experiment = "1". 1 for MCAR (missing at random), 2 for Censoring MNAR (at 1), 3 for Predictive missingness 

n,p,n_test  = 10_000,10,100_000  # n=1000, p= 9 or 10 in paper. Since this is one run, consider larger n. 
stde        = 0.1                # 0.1 in paper
ρ           = 0.5                # 0.5 in paper, cross-correlation of features 

# Some options for HTBoost
priortype = :hybrid        # :hybrid (default) or :smooth or :sharp
modality  = :compromise    # :accurate, :compromise, :fast, :fastest 

nfold           = 1        # nfold cv. 1 faster (single validation sets), default 4 is slower, but more accurate. Here nfold = 1 for fair comparison with LightGBM.

plot_results = false
mask_missing = false     # default = false. True to introduce an additional feature, a dummy with value 'true' if x is missing. Not necessary.

# END USER'S OPTIONS

# generate data

# Missing at random (MCAR)
function model1_missingpattern1(x) 

    prob = 0.2  # probability of miss. 0.2 in their experiments 
    α,β = 1,2  # α,β = 1,2 in their experiments
    i   = 1    # i=1 in their experiments. Which feature is x^2 (miss is for 1st)
    ind = rand(size(x,1)) .< prob
    f   = α*x[:,i].^β  
    x[ind,1] .= NaN

    return f,x
end 


function model1_missingpattern2(x) 

    prob = 0.2                    # probability of miss. 0.2 in their experiments 
    q  = quantile(x[:,1],1-prob)
    α,β = 1,2  # α,β = 1,2 in their experiments
    i   = 1    # i=1 in their experiments. Which feature is x^2 (miss is for 1st)
    ind = x[:,1] .< q
    f   = α*x[:,i].^β  
    x[ind,1] .= NaN

    return f,x
end 



function model1_missingpattern3(x) 

    prob = 0.2   # probability of miss. 0.2 in their experiments 
    α,β = 1,2  # α,β = 1,2 in their experiments
    i   = 1    # i=1 in their experiments. Which feature is x^2 (miss is for 1st)
    ind = rand(size(x,1)) .< prob
    f   = α*x[:,i].^β  + 3*ind 
    x[ind,1] .= NaN

    return f,x
end     


function Friedman(x) 

    prob = 0.2   # probability of miss. 0.2 in their experiments 
    n,p  = size(x)
    f    = @. 10*sin(π*x[:,1]*x[:,2]) + 20*(x[:,3] - 0.5)^2 + 10*x[:,4] + 5*x[:,5]
    MissData = rand(n,p) .< prob
    x[MissData] .= NaN

    return f,x
end     



function Linear(x) 

    prob = 0.2   # probability of miss. 0.2 in their experiments 
    n,p  = size(x)
    f    =  x[:,1] + 2*x[:,2] - x[:,3] + 3*x[:,4] - 0.5*x[:,5] - x[:,6] + 0.3*x[:,7] + 1.7*x[:,8] 
            + 0.4*x[:,9] - 0.3*x[:,10] 
    MissData = rand(n,p) .< prob
    x[MissData] .= NaN

    return f,x
end     



function missing_function(missing_pattern)

    if missing_pattern==1
        return model1_missingpattern1
    end 
    
    if missing_pattern==2
        return model1_missingpattern2
    end 

    if missing_pattern==3
        return model1_missingpattern3
    end 

end 


if Experiment == "1"
    f_pattern = missing_function(missing_pattern)
elseif Experiment == "2 Friedman" 
    f_pattern = Friedman
elseif Experiment == "2 Linear" 
    f_pattern = Linear
else 
    @error "Experiment misspelled."    
end 


#cross-correlated data
u = ones(p)
μ = ones(p)
V = ρ*(u*u') + (1-ρ)*I
d = Distributions.MvNormal(μ,V)
x = copy(rand(d,n)')            # copy() because LightGBM does not copy with Adjoint type  
x_test = copy(rand(d,n_test)')

f,x            = f_pattern(x)
f_test,x_test  = f_pattern(x_test) 

y      = f + stde*randn(n)
y_test = f_test + stde*randn(n_test)

# set up HTBparam and HTBdata, then fit and predit
param  = HTBparam(priortype=priortype,randomizecv=true,nfold=nfold,modality=modality )
data   = HTBdata(y,x,param)

output = HTBfit(data,param)
yf     = HTBpredict(x_test,output)  # predict

yf  = HTBpredict(x_test,output)  # predict

# Evaluate predictions at a few points
println("\n E(y|x) with x1 = 1 or missing, and x2 = 0 or 1") 

for ov in [0.0,1.0]
    x_t    = fill(ov,size(x,2),p)
    x_t[1,1] = 1.0
    yf1     = HTBpredict(x_t,output)  # predict
    x_t[1,1] = NaN
    yf2     = HTBpredict(x_t,output)  # predict

    println(" prediction at x1 = 1 and at x = miss, other variables at $ov ", [yf1[1],yf2[1]])
end 

# LigthGBM 

estimator = LGBMRegression(
    objective = "regression",
    metric = ["l2"],         
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

# fit at cv parameters
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
yf_gbm = LightGBM.predict(estimator,x_test)   # (n_test,num_class) 

println("\n Experiment = $Experiment, missing_pattern = $missing_pattern, n = $n")
println("\n out-of-sample RMSE from truth, HTBoost, modality=:modality  ", sqrt(sum((yf - f_test).^2)/n_test) )
println(" out-of-sample RMSE from truth, LigthGBM cv                     ", sqrt(sum((yf_gbm - f_test).^2)/n_test) )

