
# Finding sparsity

**Short description.**

HTBoost has an in-built penalization encouraging a sparse representation.
This penalization is very mild in modality = :fast, which is close to standard boosting. 
When modality=:compromise or modality=:accurate, the amount of penalization is automatically cross-validated, ranging
from none (standard boosting) to quite strong (to capture very sparse representations.)
Note: this cv is not performed if the effective sample size (which depends on n, loglikvide, and var(yhat)/var(y))
is large compared to the number of features p.

**Further notes and comments.**

HTB's approach to sparsity builds on Xu et al. 2019, "Gradient Boosted Feature Selection", designed for the n>>p case, 
implemented by penalizing the introduction of any feature not previously selected, with some important innovations. 
Since HTBoost is much slower than other GBMs, it is essential for cross-validation to require only a handful
of evaluations. This is not possible in the representation of Xu et al., where the penalization has no obvious
range, and a grid from 0.125 to 500 is used. HTBoost normalizes the penalization by taking into
account the number of features as well as their nature (continuous or binary), which allows for a more focused search
of a few values in a narrow range. 

In the example below, only a small subset p_star of features are relevant.
All features are dichotomous, and they don't interact, so there is no smoothness for 
HTBoost to take advantage of, but HTBoost outperforms a version without
sparsity penalization (especially in modality=:compromise or :accurate).

Sparsity penalization is most useful with small n, very large p, or low SNR, as
GBM are already quite good at selecting away redundant features. 

```julia 

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Statistics
using LightGBM

# USER'S OPTIONS 
Random.seed!(1)

# Options for data generation 
n         = 1_000
p         = 500       # number of features 
p_star    = 20        # number of relevant features (p_star < p )
stde      = 1            

# Options for HTBoost
modality = :compromise  

depth     = 2     # depth fixed to speed up computations
nfold     = 4     # nfold=1 is not adequate in a small n situation. Leave at default (4)

verbose          = :Off
warnings         = :On

β = randn(p_star)
f_dgp(x,β) = x[:,1:length(β)]*β

# END USER'S INPUTS 

x,x_test = randn(n,p),randn(200_000,p)    

y      = f_dgp(x,β) + stde*randn(n)
f_true = f_dgp(x_test,β)

# LightGBM

estimator = LGBMRegression(   # LGBMRegression(...)
    objective = "regression",
    num_iterations = 1000,   # default 100
    learning_rate = 0.1,    # default 0.1
    early_stopping_round = 100,  
    metric = ["l2"],
    num_threads = number_workers,
    max_depth = depth         # setting it to 1 in this experiment only 
)

sharevalidation = 0.3
n_train = Int(round((1-sharevalidation)*length(y)))
x_train = x[1:n_train,:]; y_train = y[1:n_train]
x_val   = x[n_train+1:end,:]; y_val = y[n_train+1:end]

LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)

yf_gbm = LightGBM.predict(estimator,x_test)[:,1]   # (n_test,num_class) 

# HTBoost

param   = HTBparam(modality=modality,verbose=:Off,warnings=:On,depth=depth,nfold=nfold)
data  = HTBdata(y,x,param)

output = HTBfit(data,param,cv_grid=[depth]);
yf = HTBpredict(x_test,output,predict=:Ey)  # predict

println("\n RMSE of HTBoost from true E(y|x)                   ", sqrt(mean((yf-f_true).^2)) )
println(" RMSE of LightGBM (default param) from true E(y|x)     ", sqrt(mean((yf_gbm-f_true).^2)) )

fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false);
println(" HTBoost number of included features in final model $(sum(fi.>0))")

# HTBoost without scarcity penalization

param   = HTBparam(modality=modality,verbose=:Off,warnings=:On,depth=depth,nfold=nfold,
                    sparsity_penalization = 0)

output = HTBfit(data,param,cv_grid=[depth],cv_sparsity=false);
yf = HTBpredict(x_test,output,predict=:Ey)  # predict

println("\n RMSE of HTBoost without sparsity penalization      ", sqrt(mean((yf-f_true).^2)) )

fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false);
println(" HTBoost without sparsity penalization: #included features  $(sum(fi.>0))")

```