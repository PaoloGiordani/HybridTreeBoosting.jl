#=
Generate data from gamma, compare L2 and L2logLink 
  
paolo.giordani@bi.no
=#

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using HTBoost

using Random,Plots,Distributions 
using LightGBM

# USER'S OPTIONS 

# Some options for HTBoost
modality  = :compromise    # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees
 
randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :On

# options to generate data.
n,p,n_test  = 5_000,4,100_000

f_1(x,b)    = b./(1.0 .+ (exp.(2.0*(x .- 1.0) ))) .- 0.1*b 
f_1(x,b)    = b./(1.0 .+ (exp.(3.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(5.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(8.0*(x .+ 0.5) ))) .- 0.1*b

b1,b2,b3,b4 = 1.0,1.0,1.0,1.0

function simul_gamma(n,p,k,nsimul,modality,dgp)  # dgp is :N or :logN 

RMSE = zeros(nsimul,2)  # L2 and Loglink 
ntrees = zeros(nsimul,2)
depth  = zeros(nsimul,2)

for simul in 1:nsimul

# generate data
Random.seed!(simul)

# generate data
x,x_test = randn(n,p), randn(n_test,p)

c        = -2  
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

μ        = exp.(f)        # conditional mean 
μ_test   = exp.(f_test)   # conditional mean 

scale    = μ/k
scale_test = μ_test/k
y       = zeros(n)

for i in eachindex(y)
    y[i]  = rand(Gamma.(k,scale[i]))
end 
# set up HTBparam and HTBdata, then fit and predit

# coefficient estimated internally. 
param  = HTBparam(loss=:L2,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = HTBdata(y,x,param)

output = HTBfit(data,param,cv_link=false)
yf     = HTBpredict(x_test,output,predict=:Ey)

RMSE[simul,1] = sqrt(sum((yf - μ_test).^2)/n_test) 
ntrees[simul,1] = output.ntrees 
depth[simul,1] = output.bestvalue

# Repeat for log link   
param = deepcopy(param) 
param.loss = :L2loglink 
output = HTBfit(data,param)
yf    = HTBpredict(x_test,output,predict=:Ey)  

RMSE[simul,2] = sqrt(sum((yf - μ_test).^2)/n_test)
ntrees[simul,2] = output.ntrees 
depth[simul,2] = output.bestvalue

end 

return (RMSE=RMSE,ntrees=ntrees,depth=depth)
end 


n,p = 4_000,9 
k   = 10
nsimul = 10
modality = :compromise 
dgp      = :logN 

@time RMSE,ntrees,depth= simul_gamma(n,p,k,nsimul,modality,dgp)
println("\n dgp = $dgp, n = $n")
println("\n average RMSE L2 and L2loglink $(mean(RMSE,dims=1))")
println(" ntrees L2 and L2loglink $(mean(ntrees,dims=1))")
println(" depth L2 and L2loglink $(mean(depth,dims=1))")


dgp = :N 
@time RMSE,ntrees,depth= simul_gamma(n,p,k,nsimul,modality,dgp)
println("n dgp = $dgp, n = $n")
println("\n average RMSE L2 and L2loglink $(mean(RMSE,dims=1))")
println(" ntrees L2 and L2loglink $(mean(ntrees,dims=1))")
println(" depth L2 and L2loglink $(mean(depth,dims=1))")
