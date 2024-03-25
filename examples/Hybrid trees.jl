
"""

**Hybrid trees: why smooth trees are not always good enough.**

There are two characteristics of HTB trees that make them hybrid:
1. Splits are smooth by default, but can be forced to be sharp for some features.
2. The fitted value from each tree is the input for a univariate nonlinear transformation.

Here we explore the first characteristic, while examples/Projection Pursuit Regreession.jl 
explores the second.

When the smoothness parameter τ is estimated for each split and allowed to take high values,
including τ=Inf (which corresponds to a sharp split), it's perhaps intuitive to think that a
smooth tree can capture both smooth and sharp functions. However, this is not necessarily true
in a boosting context, as illustrated in this script.

Consider the following example:
If we are trying to approximate a step function with a single step (second column in the plot),
a smooth tree which can take high values of τ performs asymptotically as well as a sharp split.
However, if the step functions has multiple steps (third column in the plot), the fitting function
is initially smooth, and it is then impossible for the subsequent trees to fully recover a sharp function.
In this example, the greedy nature of boosting gets the algorithm stuck in a local minimu. 
A hybrid tree attempts to recognize such cases from a preliminary run, and, if necessary, imposes
sharp splits on some features to avoid the local minima. The cross-validated loss is then 
used to decide in which combination to use the hybrid and the smooth tree.

"""
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Plots 

# USER'S OPTIONS 

Random.seed!(1)

# Some options for HTBoost
loss      = :L2            # :L2 or :logistic (or :Huber or :t). 
modality  = :fastest       # :accurate, :compromise (default), :fast, :fastest 

ntrees    = 1000          # maximum number of trees  
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.

verbose     = :Off
warnings    = :Off
 
# options to generate data. y = sum of two additive nonlinear functions + Gaussian noise.
n,p,n_test  = 10_000,6,100_000
stde        = 0.2

f_1(x,b)    = 1.0./(1.0 .+ (exp.(4.0*(x .- 0.5) ))) .- 0.1*b
f_2(x,b)    = b*(x.>0.5) .- 0.1*b                      # step function with one step 
f_3(x,b)    = b*( (x .> -0.5) +  (x .> 0.5) )        # step function with several steps
f_4(x,b)    =  (-0.25 .< x .< 0.25)                     # "tower" function 
#f_interact(x1,x2,b)  = b*(-0.25 .< x1 .< 0.25).*(0.25 .< x2 .< 0.5)  # sharp interaction
f_interact(x1,x2,b)  = b*(-0.25 .< x1 .< 0.25).*(0.25 .< x2 .< 0.5) + (-1.0 .< x1 .< -0.75).*(1.25 .< x2 .< 1.5)  # sharp interaction

b1,b2,b3,b4 = 1.0,1.0,1.0,1.0
b_interact  = 3.0          # sharp interactions can be difficult to approximate well for a smooth tree

# END USER'S OPTIONS

# generate data
x,x_test = randn(n,p), randn(n_test,p)

f        = f_interact(x[:,5],x[:,6],b_interact) + f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4) 
f_test   = f_interact(x_test[:,5],x_test[:,6],b_interact) + f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4) 

y = f + stde*randn(n)

# set up HTBparam and HTBdata, then fit and predit
ntrees == 1 ? lambda = 1 : lambda = 0.2

param  = HTBparam(loss=loss,nfold=nfold,verbose=verbose,warnings=warnings,
           modality=modality,nofullsample=true,lambda=lambda,ntrees=ntrees)

data   = HTBdata(y,x,param)

output = HTBfit(data,param)
yf     = HTBpredict(x_test,output)  # predict the natural parameter

println(" \n modality = $(param.modality), nfold = $nfold ")
println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

println(" \n with smooth rather than hybrid trees ")
output_s = HTBfit(data,param,cv_hybrid=false)
yf     = HTBpredict(x_test,output_s)  # predict the natural parameter

println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

# After 1 tree, with lambda = 1
param.ntrees=1 
param.lambda=param.T(1)

output1 = HTBfit(data,param)

# partial plots 
q_s,pdp_s  = HTBpartialplot(data,output_s,[1,2,3,4])
q,pdp  = HTBpartialplot(data,output,[1,2,3,4])
q,pdp_1  = HTBpartialplot(data,output1,[1,2,3,4])

pl   = Vector(undef,12)
f,b  = [f_1,f_2,f_3,f_4],[b1,b2,b3,b4]

for i in 1:4
    pl[i]   = plot( [q_s[:,i]],[pdp_1[:,i] f[i](q_s[:,i],b[i]) - f[i](q_s[:,i]*0,b[i])],
           label = ["1st tree λ=1" "dgp"],
           legend = :bottomright,
           linecolor = [:blue :red],
           linestyle = [:solid :dot],
           linewidth = [5 2],
           titlefont = font(15),
           legendfont = font(12),
           xlabel = "x",
           ylabel = "f(x)",
           )
end

for i in 1:4
    pl[i+4]   = plot( [q_s[:,i]],[pdp_s[:,i] f[i](q_s[:,i],b[i]) - f[i](q_s[:,i]*0,b[i])],
           label = ["smooth" "dgp"],
           legend = :bottomright,
           linecolor = [:blue :red],
           linestyle = [:solid :dot],
           linewidth = [5 2],
           titlefont = font(15),
           legendfont = font(12),
           xlabel = "x",
           ylabel = "f(x)",
           )
end

for i in 1:4
    pl[i+8]  = plot( [q[:,i]],[pdp[:,i] f[i](q[:,i],b[i]) - f[i](q[:,i]*0,b[i])],
           label = ["hybrid" "dgp"],
           legend = :bottomright,
           linecolor = [:blue :red],
           linestyle = [:solid :dot],
           linewidth = [5 2],
           titlefont = font(15),
           legendfont = font(12),
           xlabel = "x",
           ylabel = "f(x)",
           )
end

display(plot(pl[1],pl[2],pl[3],pl[4],pl[5],pl[6],pl[7],pl[8],pl[9],pl[10],pl[11],pl[12],layout=(3,4), size=(1300,800)))  # display() will show it in Plots window.
