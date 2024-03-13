
"""

**Short description:**

- Illustrates basic use on a regression problem.
- param.modality as the most important user's choice, depending on time budget. 
- In default modality, HTBoost performs automatic hyperparameter tuning.

**Extensive description:** 

Simulated iid data, additively nonlinear dgp.

- default loss is :L2. Other options for continuous y are :Huber, :t (recommended in place of :Huber), :gamma,
  :L2loglink. For zero-inflated continuous y, options are :hurdleGamma, :hurdleL2loglink, :hurdleL2 (see examples/Zero inflated y.jl)   
- default is block cross-validation: use randomizecv = true to scramble the data.
- fit, with automatic hyperparameter tuning if modality is :compromise or :accurate
- save fitted model (upload fitted model)
- feature importance
- average τ (smoothness parameter), which is also plotted. (Smoother functions ==> larger gains compared to other GBM)
- partial effects plots

Options for HTBoost: modality is the key parameter guiding hyperparameter tuning and learning rate.
:fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
automatic hyperparameter tuning. In HTBoost, it is not recommended that the user performs 
hyperparameter tuning by cross-validation, because this process is done automatically if modality is
:compromise or :accurate. The recommended process is to first run in modality=:fast or :fastest,
for exploratory analysis and to gauge computing time, and then switch to :compromise (default)
or :accurate.

Block cross-validation:
While the default in other GBM is to randomize the allocation to train and validation sets,
the default in HTBoost is block cv, which is suitable for time series and panels.
Set randomizecv=true to bypass this default. 
See examples/Global equity Panel.jl for further options on cross-validation (e.g. sequential cv).

"""
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Plots 

# USER'S OPTIONS 

Random.seed!(1)

# Some options for HTBoost
loss      = :L2            # :L2 is default. Other options for regression are :L2loglink (if y≥0), :t, :Huber
modality  = :fast          # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees

randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :Off
 
# options to generate data. y = sum of six additive nonlinear functions + Gaussian noise.
n,p,n_test  = 10_000,6,100_000
stde        = 1.0

f_1(x,b)    = b*x .+ 1 
f_2(x,b)    = 2*sin.(2.5*b*x)  # for higher nonlinearities, try #f_2(x,b) = 2*sin.(2.5*b*x)
f_3(x,b)    = b*x.^3
f_4(x,b)    = b./(1.0 .+ (exp.(40.0*(x .- 0.5) ))) .- 0.1*b
f_5(x,b)    = b./(1.0 .+ (exp.(4.0*(x .- 0.5) ))) .- 0.1*b
f_6(x,b)    = b*(-0.25 .< x .< 0.25) 

b1,b2,b3,b4,b5,b6 = 1.5,2.0,0.5,4.0,5.0,5.0

# END USER'S OPTIONS

# generate data
x,x_test = randn(n,p), randn(n_test,p)

f        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4) + f_5(x[:,5],b5) +  f_6(x[:,6],b6)
f_test   = f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4) + f_5(x_test[:,5],b5) + f_6(x_test[:,6],b6)

y = f + stde*randn(n)

# set up HTBparam and HTBdata, then fit and predit
param  = HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,
                warnings=warnings,modality=modality,nofullsample=nofullsample)

data   = HTBdata(y,x,param)

@time output = HTBfit(data,param)
yf     = HTBpredict(x_test,output,predict=:Egamma)  # predict the natural parameter

avgtau,avg_explogtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data,verbose=true,best_model=false);
plot(x_plot,g_plot,title="smoothness of splits",xlabel="standardized x",label=:none)

println(" \n modality = $(param.modality), nfold = $nfold ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees), avgtau $avgtau ")
println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

# save (load) fitted model
# using JLD2
#@save "output.jld2" output
#@load "output.jld2" output    # Note: key must be the same, e.g. @load "output.jld2" output2 is a KeyError

# feature importance, partial dependence plots and marginal effects
fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false);
q,pdp  = HTBpartialplot(data,output,[1,2,3,4,5,6],predict=:Egamma)

# plot partial dependence in terms of the natural parameter 
pl   = Vector(undef,6)
f,b  = [f_1,f_2,f_3,f_4,f_5,f_6],[b1,b2,b3,b4,b5,b6]

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

display(plot(pl[1],pl[2],pl[3],pl[4],pl[5],pl[6],layout=(3,2), size=(1300,800)))  

