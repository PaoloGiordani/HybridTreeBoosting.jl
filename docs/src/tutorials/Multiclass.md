
## Multiclass classification in HTBoost 

**Key points:**

- y is a vector: only one outcome is possible. (For multilabel classification, fit a sequence of :logistic models.)
- Elements of y can be numerical or string. Numericals do not need to be in the format (0,1,2,...). e.g. (1,2,3,...), (22,7,48,...)
    ("a","b","c",...), ("Milan","Rome","Naples",...) are all allowed.
- num_class is detected automatically, not a user input.  
- The output from *HTBpredict( )* is: 
  yf,class_values,ymax = *HTBpredict*(x_test,output). yf is a (n_test,num_class) matrix of probabilities, with yf[i,j] the probability
  that observation i takes value class_value[j]. class_values is a (num_class) vector of the unique values of y, sorted from smallest to largest.
  ymax is a (n_test) vector the most likely outcome.
- HTBoost employs a one-vs-rest approach (not a multinomial logistic).   
- The training time is proportional to num_classes, and it can therefore be high. There is room for future improvements though,
  since the one-vs-rest approach can be made embarassingly parallel.  

**Comparison with LightGBM**

In this example we draw simulated data from a multinomial with 3 classes, and compare HTBoost and LightGBM. The smoother the function, the more HTBoost will outperform. 


```julia 
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Statistics
using LightGBM

```

Options to generate data 

```julia 

Random.seed!(12)

# Options for data generation 
n         = 10_000
p         = 5      # p>=4. Only the first 4 variables are relevant

# Multinomial logistic to draw data. Functions vary from very smooth (exp(1*...))
# to only moderately smooth (exp(8*...)). 

f_1(x,b)    = b./(1.0 .+ (exp.(1.0*(x .- 1.0) ))) .- 0.1*b 
f_2(x,b)    = b./(1.0 .+ (exp.(2.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(4.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(8.0*(x .+ 0.5) ))) .- 0.1*b

b1v = [0.0,1.0,1.0,0.3,0.3]   # coefficients first class
b2v = [0.0,0.3,0.3,1.0,1.0]   # coefficients second class 

``` 
Options for HTBoost.
loss = :multiclass.
The number of classes is not a user's input. 
We set nfold=1 and nofullsample=true for fair comparison to LightGBM. 

```julia

loss      = :multiclass 
modality  = :compromise   # :accurate, :compromise (default), :fast, :fastest

nfold     = 1             
nofullsample = true 
verbose     = :Off 

```

Simulate data.

```julia 

# generates data from a multinomials with 3 classes 
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


n_test = 100_000
x,x_test = randn(n,p), randn(n_test,p)
y        = rnd_3classes(x,b1v,b2v)
y_test   = rnd_3classes(x_test,b1v,b2v)

```
Fit HTBoost.
Notice the additional output of *HTBpredict( )*
Compare predictions using the (minus) multinomial log-likelihood and the hit rate.

```julia

# fit HTBoost
param  = HTBparam(loss=loss,nfold=nfold,nofullsample=nofullsample,verbose=verbose,modality=modality)
data   = HTBdata(y,x,param)
output = HTBfit(data,param)

yf,class_values,ymax = HTBpredict(x_test,output) 
hit_rate = mean(ymax .== y_test)

loss     = HTBmulticlass_loss(y_test,yf,param.class_values)

println("\n HTBoost hit rate $hit_rate and loss $loss ")

```
Fit LightGBM.
Notice the additional output of *HTBpredict( )*

```julia

# lightGBM 
estimator = LGBMClassification(   # LGBMRegression(...)
    objective = "multiclass",
    num_class = 3,
    categorical_feature = Int[],
    num_iterations = 1000,
    learning_rate = 0.1,
    early_stopping_round = 100,
    metric = ["multi_logloss"],
    num_threads = number_workers
 )

n_train = Int(round((1-param.sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = Float64.(y[1:n_train])
x_val   = x[n_train+1:end,:]; y_val = Float64.(y[n_train+1:end])

# parameter search over num_leaves and max_depth
splits = (collect(1:n_train),collect(1:min(n_train,100)))  

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

# from probabilities, compute the most likely outcome. 
ymax_gbm = zeros(eltype(class_values),n_test)

for i in eachindex(ymax)
  ymax_gbm[i] = class_values[argmax(yf_gbm[i,:])]
end     

hit_rate = mean(ymax_gbm .== y_test)
loss     = HTBmulticlass_loss(y_test,yf_gbm,param.class_values)

println("\n LightGBM hit rate $hit_rate and loss $loss ")

```
HTBoost in this case has a substantially lower loss, which is not surprising given that the simulated data has substantial smoothness. 
```markdown 
HTBoost hit rate 0.5263 and loss 96175.13
LightGBM hit rate 0.52407 and loss 96276.94
```