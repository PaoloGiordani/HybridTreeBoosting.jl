
"""

Short description:

- Illustrates basic use on a regression problem.
- param.modality as the most important user's choice.
- In default modality, SMARTboost performs automatic hyperparameter tuning.


Extensive description: 

Simulated iid data, additively nonlinear dgp.

- loss can be :L2 or :logistic or :Huber or :t (recommended in place of :Huber).
   If :logistic, the fitted and forecast values are for the log odds ratio
- fit, with automatic hyperparameter tuning if modality is :compromise or :accurate
- save fitted model (upload fitted model)
- feature importance
- average τ (smoothness parameter), which is also plotted
- partial effects plots


Options for SMARTboost: modality is the key parameter guiding hyperparameter tuning and learning rate.
:fast and :fastest only fit one model at default parameters, while :compromise and :accurate perform
automatic hyperparameter tuning. In SMARTboost, it is not recommended that the user performs 
hyperparameter tuning by cross-validation, because this process is done automatically if modality is
:compromise or :accurate. The recommended process is to first run in modality=:fast or :fastest,
for exploratory analysis and to gauge computing time, and then switch to :compromise (default)
or :accurate.

paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboost

using Random,Plots 

# USER'S OPTIONS 

Random.seed!(123)

# Some options for SMARTboost
loss      = :L2           # :L2 or :logistic (or :Huber or :t). 
modality  = :compromise    # ::accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees

randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :On
 
# options to generate data. y = sum of four additive nonlinear functions + Gaussian noise.
n,p,n_test  = 10_000,5,100_000
stde        = 1.0
  
f_1(x,b)    = b*x .+ 1 
f_2(x,b)    = sin.(b*x)  # for higher nonlinearities, try #f_2(x,b) = 2*sin.(2.5*b*x)
f_3(x,b)    = b*x.^3
f_4(x,b)    = b./(1.0 .+ (exp.(40.0*(x .- 0.5) ))) .- 0.1*b

b1,b2,b3,b4 = 1.5,2.0,0.5,2.0

# END USER'S OPTIONS

# generate data
x,x_test = randn(n,p), randn(n_test,p)

f        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

loss==:logistic ? y = (exp.(f)./(1.0 .+ exp.(f))).>rand(n) : y=stde*randn(n)+f

# set up SMARTparam and SMARTdata, then fit and predit
param  = SMARTparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,warnings=warnings,
           modality=modality,nofullsample=nofullsample)
data   = SMARTdata(y,x,param)

@time output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output,predict=:Egamma)  # predict the natural parameter

avgtau,avg_explogtau,avgtau_a,dftau,x_plot,g_plot = SMARTweightedtau(output,data,verbose=true,best_model=false)
plot(x_plot,g_plot,title="smoothness of splits",xlabel="standardized x",label=:none)

println(" \n modality = $(param.modality), nfold = $nfold ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees), avgtau $avgtau ")
println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

# save (load) fitted model
# using JLD2
#@save "output.jld2" output
#@load "output.jld2" output    # Note: key must be the same, e.g. @load "output.jld2" output2 is a KeyError

# feature importance, partial dependence plots and marginal effects
fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output,data,verbose=false);
q,pdp  = SMARTpartialplot(data,output,[1,2,3,4],predict=:Egamma)

# plot partial dependence in terms of the natural parameter 
pl   = Vector(undef,4)
f,b  = [f_1,f_2,f_3,f_4],[b1,b2,b3,b4]

for i in 1:length(pl)
    pl[i]   = plot( [q[:,i]],[pdp[:,i] f[i](q[:,i],b[i]) - f[i](q[:,i]*0,b[i])],
           label = ["smart" "dgp"],
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


