# gammaPoisson 

**gammaPoisson for (potentially) overdisperesed count data.**

- HTBoost with gammaPoisson (aka negative binomial) distribution on simulated data: E(y)=μ(x), var(y)=μ(1+αμ)
- The overdispersion parameter α is estimated internally. 
- loss=:Poisson is also available (α=0)

Note: LightGMB does not have a gammaPoisson option. loss = poisson is used. 

```julia 

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Plots,Distributions 
using LightGBM

# USER'S OPTIONS 

Random.seed!(1)

# Some options for HTBoost
loss      = :gammaPoisson      # :gammaPoisson or Poisson              
modality  = :fastest         # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees
 
randomizecv = false       # false (default) to use block-cv. 
verbose     = :On
warnings    = :On

# options to generate data.
α           = 0.5   # overdispersion parameter. Poisson for α -> 0 
n,p,n_test  = 50_000,5,100_000

# no interaction terms  
f_1(x,b)    = b./(1.0 .+ (exp.(2.0*(x .- 1.0) ))) .- 0.1*b 
f_2(x,b)    = b./(1.0 .+ (exp.(4.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(7.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(10.0*(x .+ 0.5) ))) .- 0.1*b

#b1,b2,b3,b4 = 0.2,0.2,0.2,0.2
b1,b2,b3,b4 = 0.6,0.6,0.6,0.6

# generate data
α>0 ? ddgp = :gammaPoisson : ddgp = :Poisson 
x,x_test = randn(n,p), randn(n_test,p)

c        = 0    #  
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

μ        = exp.(f)        # conditional mean 
μ_test   = exp.(f_test)   # conditional mean 

y        = zeros(n)
y_test   = zeros(n_test)

if ddgp==:gammaPoisson
    for i in eachindex(y)
        r = 1/α                    # gammaPoisson is negative binomial with r=1\α, and p = r./(μ .+ r)
        pg = r./(μ .+ r)
        y[i] = rand(NegativeBinomial(r,pg[i]))     
    end 
else 
    α = 0.0
    for i in eachindex(y)
        y[i]  = rand(Poisson.(μ[i]))
    end 
end 


histogram(y)
@show [mean(y), std(y), std(μ), maximum(y)]

# set up HTBparam and HTBdata, then fit and predit

# coefficient estimated internally. 
param  = HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = HTBdata(y,x,param)

output = HTBfit(data,param)
yf     = HTBpredict(x_test,output,predict=:Ey)

println(" \n loss = $(param.loss), modality = $(param.modality), nfold = $nfold ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" out-of-sample RMSE from true μ     ", sqrt(sum((yf - μ_test).^2)/n_test) )
println(" out-of-sample MAD from true μ      ", mean(abs.(yf - μ_test)) )


println("\n true overdispersion = $α, estimated = $(exp(output.bestparam.coeff_updated[1][1])) ")

println("\n For more information about coefficients, use HTBcoeff(output) ")
HTBcoeff(output)

# lightGBM

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

println("\n oss RMSE from truth, μ, LightGBM default ", sqrt(sum((yf_gbm_default - μ_test).^2)/n_test) )
println(" oss RMSE from true μ, LightGBM cv      ", sqrt(sum((yf_gbm - μ_test).^2)/n_test) )
println(" oos MAD from true μ, LightGBM cv       ", mean(abs.(yf_gbm - μ_test)) )

# HTBoost partial plots

q,pdp  = HTBpartialplot(data,output,[1,2,3,4],predict=:Egamma)

# plot partial dependence in terms of the natural parameter 
pl   = Vector(undef,4)
f,b  = [f_1,f_2,f_3,f_4],[b1,b2,b3,b4]

for i in 1:length(pl)
        pl[i]   = plot( [q[:,i]],[pdp[:,i] f[i](q[:,i],b[i]) - f[i](q[:,i]*0,b[i])],
           label = ["HTB" "dgp"],
           legend = :bottomright,
           linecolor = [:blue :red],
           linestyle = [:solid :dot],

           linewidth = [5 5],
           titlefont = font(15),
           legendfont = font(12),
           xlabel = "x",
           ylabel = "f(x)",
           )
end

display(plot(pl[1], pl[2], pl[3], pl[4], layout=(2,2), size=(1300,800)))  # display() will show it in Plots window.

```