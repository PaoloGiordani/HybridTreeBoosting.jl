""" 

Question: What gains from :gammaPoisson rather than :Poisson if the data do show over-dispersion but the
interest is in E(y|x) only? 

Results:
- The gains are small in going from Poisson to gammaPoisson, but they are also small in going
  from L2 to Poisson, presumably because we are concerned about the conditional mean (the MLE
  of the unconditional mean is indeed the same for Poisson, gammaPoisson and L2).
- The gains in going from L2 to Poisson are very small for lightgbm as well. 

paolo.giordani@bi.no
""" 

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboostPrivate

using Random,Plots,Distributions 
using LightGBM

# USER'S OPTIONS 

# Some options for SMARTboost
modality  = :compromise    # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees
 
randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :On

f_1(x,b)    = b./(1.0 .+ (exp.(2.0*(x .- 1.0) ))) .- 0.1*b 
f_2(x,b)    = b./(1.0 .+ (exp.(3.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(5.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(8.0*(x .+ 0.5) ))) .- 0.1*b

#b1,b2,b3,b4 = 0.2,0.2,0.2,0.2  # low predictability 
b1,b2,b3,b4 = 0.5,0.5,0.5,0.5   # std(mu) = 0.5std(y)
#b1,b2,b3,b4 = 1.0,0.0,1.0,1.0    # high predictability 

# dgp=:gammaPoisson or :Poisson. α is overdispersion parameter 
function simul_gammaPoisson(n,p,α,nsimul,modality,dgp)   

RMSE = zeros(nsimul,5)      # Poisson, gammaPoisson, L2, lightgbm Poisson, lightGBM L2
RMSE_gammafit = zeros(nsimul,5)  
deviance = zeros(nsimul,5)  # Here I used the deviance of a Poisson for ALL  
ntrees = zeros(nsimul,5)
depth  = zeros(nsimul,5)

for simul in 1:nsimul

# generate data
Random.seed!(simul)

# generate data
n_test = 50_000 
x,x_test = randn(n,p), randn(n_test,p)

c        = -1   # smaller numbers (e.g. -1) for smaller E(y) and more skewed distributions 
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

μ        = exp.(f)        # conditional mean 
μ_test   = exp.(f_test)   # conditional mean 

y        = zeros(n)
y_test   = zeros(n_test)

if dgp==:gammaPoisson
    for i in eachindex(y)
        r = 1/α                    # gammaPoisson is negative binomial with r=1\α, and p = r./(μ .+ r)
        pg = r./(μ .+ r)
        y[i] = rand(NegativeBinomial(r,pg[i]))     
    end 
else 
    for i in eachindex(y)
        y[i]  = rand(Poisson.(μ[i]))
    end 
end 


if dgp==:gammaPoisson
    for i in eachindex(y_test)
        r = 1/α                    # gammaPoisson is negative binomial with r=1\α, and p = r./(μ .+ r)
        pg = r./(μ_test .+ r)
        y_test[i] = rand(NegativeBinomial(r,pg[i]))     
    end 
else 
    for i in eachindex(y)
        y_test[i]  = rand(Poisson.(μ_test[i]))
    end 
end 



param  = SMARTparam(loss=:Poisson,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = SMARTdata(y,x,param)

output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output,predict=:Ey)

deviance[simul,1] = -2(mean(loglik_vector(:Poisson,output.bestparam,y_test,log.(yf))))
RMSE[simul,1] = sqrt(sum((yf - μ_test).^2)/n_test) 
RMSE_gammafit[simul,1] = sqrt(sum((log.(yf) - f_test).^2)/n_test) 

ntrees[simul,1] = output.ntrees 
depth[simul,1] = output.bestvalue

output.ntrees==1 ? display(plot(output.meanloss,title="Poisson")) : nothing 


# gammaPoisson 
param  = SMARTparam(loss=:gammaPoisson,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)

output = SMARTfit(data,param)
yf2    = SMARTpredict(x_test,output,predict=:Ey)  

deviance[simul,2] = -2(mean(loglik_vector(:Poisson,output.bestparam,y_test,log.(yf2))))
RMSE[simul,2] = sqrt(sum((yf2 - μ_test).^2)/n_test)
ntrees[simul,2] = output.ntrees 
depth[simul,2] = output.bestvalue
RMSE_gammafit[simul,2] = sqrt(sum((log.(yf2) - f_test).^2)/n_test) 

output.ntrees==1 ? display(plot(output.meanloss,title="gammaPoisson")) : nothing 

# L2

param  = SMARTparam(loss=:L2,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)

output = SMARTfit(data,param)
yf2    = SMARTpredict(x_test,output,predict=:Ey)  

yf2 = @. yf2*(yf2 > 0) + 0.001*(yf2 <= 0) 
deviance[simul,3] = -2(mean(loglik_vector(:Poisson,output.bestparam,y_test,log.(yf2))))
RMSE[simul,3] = sqrt(sum((yf2 - μ_test).^2)/n_test)
ntrees[simul,3] = output.ntrees 
depth[simul,3] = output.bestvalue
RMSE_gammafit[simul,3] = sqrt(sum((log.(yf2) - f_test).^2)/n_test) 

# lightGBM with Poisson loss 
estimator = LGBMRegression(
    objective = "poisson",
    metric = ["poisson"],    # default seems (strangely) "l2" regardless of objective: LightGBM.jl bug?  
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

# fit at parameters given by estimator, no cv
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm_default = LightGBM.predict(estimator,x_test)

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

# re-fit at cv parameters
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm = LightGBM.predict(estimator,x_test)
yf_gbm = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 

deviance[simul,4] = -2(mean(loglik_vector(:Poisson,output.bestparam,y_test,log.(yf_gbm))))
RMSE[simul,4] = sqrt(sum((yf_gbm - μ_test).^2)/n_test)
RMSE_gammafit[simul,4] = sqrt(sum((log.(yf_gbm) - f_test).^2)/n_test) 


# lightGBM with L2 loss 

# lightGBM with Poisson loss 
estimator = LGBMRegression(
    objective = "regression",
    metric = ["l2"],    # default seems (strangely) "l2" regardless of objective: LightGBM.jl bug?  
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

# fit at parameters given by estimator, no cv
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm_default = LightGBM.predict(estimator,x_test)

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

# re-fit at cv parameters
LightGBM.fit!(estimator,x_train,y_train,(x_val,y_val),verbosity=-1)
    
yf_gbm = LightGBM.predict(estimator,x_test)
yf_gbm = yf_gbm[:,1]    # drop the second dimension or a (n_test,1) matrix 

yf_gbm = @. yf_gbm*(yf_gbm > 0) + 0.001*(yf_gbm <= 0) 
deviance[simul,5] = -2(mean(loglik_vector(:Poisson,output.bestparam,y_test,log.(abs.(yf_gbm)))))
RMSE[simul,5] = sqrt(sum((yf_gbm - μ_test).^2)/n_test)
RMSE_gammafit[simul,5] = sqrt(sum((log.(yf_gbm) - f_test).^2)/n_test) 

@show simul 
@show mean(RMSE[1:simul,:],dims=1)
@show mean(RMSE_gammafit[1:simul,:],dims=1)
@show simul,mean(deviance[1:simul,:],dims=1)


end 

return (deviance=deviance,RMSE=RMSE,ntrees=ntrees,depth=depth)
end 


n,p = 5_000,9 
nsimul = 50
modality = :compromise 
dgp      = :gammaPoisson
#dgp      = :Poisson
α        = 1.0 

@time deviance,RMSE,ntrees,depth= simul_gammaPoisson(n,p,α,nsimul,modality,dgp)
println("\n dgp = $dgp, n = $n")
println("\n average RMSE   Poisson, gammaPoisson, L2, lighPoisson, lightL2  $(mean(RMSE,dims=1))")
println(" average deviance Poisson, gammaPoisson, L2, , lighPoisson, lightL2 $(mean(deviance,dims=1))")
println(" ntrees $(mean(ntrees,dims=1)) and depth  $(mean(depth,dims=1))")
println(" stde of RMSE $(std(RMSE[:,1])/sqrt(nsimul) )  ")


n,p = 50_000,9 
nsimul = 50
modality = :compromise 
#dgp      = :gammaPoisson
dgp      = :Poisson
α        = 1.0 

@time deviance,RMSE,ntrees,depth= simul_gammaPoisson(n,p,α,nsimul,modality,dgp)
println("\n dgp = $dgp, n = $n")
println("\n average RMSE Poisson, gammaPoisson, L2 $(mean(RMSE,dims=1))")
println(" ntrees $(mean(ntrees,dims=1)) and depth  $(mean(depth,dims=1))")
println(" stde of RMSE $(std(RMSE[:,1])/sqrt(nsimul) )  ")


#=
5044.303078 seconds (1.77 G allocations: 232.356 GiB, 3.15% gc time, 0.01% compilation time)

 dgp = gammaPoisson, n = 1000

 average RMSE Poisson, gammaPoisson, L2, gammaPoisson with mse cv
  [0.2835632797156633 0.281681887542709 0.2603426454347667 0.2786748512169952]
 ntrees [11.92 11.51 14.0 13.33] and depth  [3.14 3.28 3.46 3.22]

 14090.865310 seconds (1.13 G allocations: 668.091 GiB, 3.93% gc time)

 dgp = gammaPoisson, n = 10000

 average RMSE Poisson, gammaPoisson, 
 
 L2 [0.12491948880697508 0.12320784534984783 0.11337215463673549 0.1270235686229118]
 ntrees [27.94 31.98 26.8 28.58] and depth  [2.62 2.44 2.7 2.8]

Now with c = -1 instead of 1, so that the distribution is extremely asymmetric. 
And more predictability: 0.5 instead of 0.2 in 
b1,b2,b3,b4 = 0.5,0.5,0.5,0.5   # std(mu) = 0.5std(y)

n = 5_000, 50 simulations. 
NB: in some cases, cv selects ONE tree, probably for L2, as in simul = 18. Look into it !
- gammaPoisson does worse of rmse, but well in terms of deviance.
- L2 is only a tiny bit worse than Poisson 
- for lightgbm, the gain in switching from L2 to Poisson is larger. 

 average RMSE   Poisson, gammaPoisson, L2, lighPoisson, lightL2
 [0.0998 0.103 0.1020 0.1231 0.1319]
 average deviance Poisson, gammaPoisson, L2, , lighPoisson, lightL2
 ??
 ntrees [36.72 34.06 30.22 ... ] and depth  [2.52 3.26 2.9 ....]
 stde of RMSE 0.002

 n = 50_000, 7 simulations 

- gammaPoisson almost as good as Poisson
- some gains vs L2, but again not as large as for lightgbm 

RMSE [0.03847 0.0402 0.04326 0.05875 0.06546]
deviance [2.187 2.187 2.187 2.189 2.190]

=#
