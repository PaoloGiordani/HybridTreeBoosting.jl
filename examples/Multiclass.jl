
#=

**Multiclass classification in HTBoost. Key points:**

- y is a vector: only one outcome possible. (For multilabel classification, fit a sequence of :logistic models.)
- Elements of y can be numerical or string. Numericals do not need to be in the format (0,1,2,...). e.g. (1,2,3,...), (22,7,48,...)
    ("a","b","c",...), ("Milan","Rome","Naples",...) are all allowed.
- num_class is detected automatically, not a user input.  
- The output from HTBpredict() is: 
  yf,class_values,ymax = HTBpredict(x_test,output). yf is a (n_test,num_class) matrix of probabilities, with yf[i,j] the probability
  that observation i takes value class_value[j]. class_values is a (num_class) vector of the unique values of y, sorted from smallest to largest.
  ymax is a (n_test) vector the most likely outcome.
- The training time is proportional to num_classes, and it can therefore be high. There is room for future improvements though,
  since the one-vs-rest approach is embarassingly parallel.  
  
=#
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HybridTreeBoosting

using Random,Statistics
using LightGBM

# USER'S OPTIONS 

Random.seed!(12)

# Options for data generation 
n         = 10_000
p         = 5      # p>=4. Only the first 4 variables are used in the function f(x) below 

# Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
# :fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
# automatic hyperparameter tuning. 

loss      = :multiclass 
modality  = :compromise   # :accurate, :compromise (default), :fast, :fastest
priortype = :hybrid

nfold     = 1             # nfold=1 and nofullsample=true for fair comparison to LightGBM
nofullsample = true 

verbose    = :Off 
warnings   = :On

# Multinomial logistic to draw data.

f_1(x,b)    = b./(1.0 .+ (exp.(1.0*(x .- 1.0) ))) .- 0.1*b 
f_2(x,b)    = b./(1.0 .+ (exp.(2.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(4.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(8.0*(x .+ 0.5) ))) .- 0.1*b

b1v = [0.0,1.0,1.0,0.3,0.3]   # coefficients first class
b2v = [0.0,0.3,0.3,1.0,1.0]   # coefficients second class 

# END USER'S OPTIONS  

# generates data from 3 classes 
function rnd_3classes(x,b1v,b2v)

  class_values = [0.0,1.0,2.0]
#  class_values = [1.0,2.0,3.0]     # HTBoost works. LightGBM needs converting to 0,1,2 ...
#  class_values = ["orange","apple","mango"]     # HTBoost works. LightGBM needs converting to 0,1,2 ...

  b0,b1,b2,b3,b4 = b1v[1],b1v[2],b1v[3],b1v[4],b1v[5]
  d1 = b0 .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)

  b0,b1,b2,b3,b4 = b2v[1],b2v[2],b2v[3],b2v[4],b2v[5]
  d2 = b0 .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)

  sumexpd = @. exp(d1) + exp(d2) + 1
  prob1 = @. exp(d1)/sumexpd 
  prob2 = @. exp(d2)/sumexpd 
  
  n  = size(x,1)
  y  = Vector{eltype(class_values)}(undef,n)
  u  = rand(n)

  for i in eachindex(y)
    u[i] < prob1[i] ? y[i] = class_values[1] : (u[i] < prob1[i]+prob2[i] ? y[i] = class_values[2] : y[i] = class_values[3]) 
  end   

  return y 
end 

# generate data
n_test = 100_000
x,x_test = randn(n,p), randn(n_test,p)
y        = rnd_3classes(x,b1v,b2v)
y_test   = rnd_3classes(x_test,b1v,b2v)

# fit HTBoost
param  = HTBparam(loss=loss,priortype=priortype,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
    
data   = HTBdata(y,x,param)

output = HTBfit(data,param)
yf,class_values,ymax = HTBpredict(x_test,output)
hit_rate = mean(ymax .== y_test)

loss     = HTBmulticlass_loss(y_test,yf,param.class_values)

println("\n HTBoost hit rate $hit_rate and loss $loss ")

# lightGBM 
estimator = LGBMClassification(   # LGBMRegression(...)
    objective = "multiclass",
    num_class = 3,
    categorical_feature = Int[],
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    metric = ["multi_logloss"],
    num_threads = number_workers,
    max_depth = -1,      # -1 default
    min_data_in_leaf = 20,  # 20 default 
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

ymax_gbm = zeros(eltype(class_values),n_test)

for i in eachindex(ymax)
  ymax_gbm[i] = class_values[argmax(yf_gbm[i,:])]
end     

hit_rate = mean(ymax_gbm .== y_test)
loss     = HTBmulticlass_loss(y_test,yf_gbm,param.class_values)

println("\n LightGBM hit rate $hit_rate and loss $loss ")


