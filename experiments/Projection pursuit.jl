
"""

dgp is hard for a tree to capture ... 


"""
number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboostPrivate
include("E:\\Users\\A1810185\\Documents\\A_Julia-scripts\\Modules\\SMARTboostLOCAL.jl") # no package

using Random,Plots 

# USER'S OPTIONS 

Random.seed!(1)

# Some options for SMARTboost
loss      = :L2            # :L2 or :logistic (or :Huber or :t). 
modality  = :fast    # :accurate, :compromise (default), :fast, :fastest 

depthpp   = 2       # depth projection pursuit  
ntrees    = 1000 

priortype = :hybrid       # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees

randomizecv = false       # false (default) to use block-cv. 
verbose     = :Off
warnings    = :Off
 
# options to generate data. y = sum of six additive nonlinear functions + Gaussian noise.
n,p,n_test  = 10_000,5,100_000
stde        = 1.0

# END USER'S OPTIONS

# generate data
x,x_test = randn(n,p), randn(n_test,p)

f        = 0.5*[0*sum(x[i,:]) + (sum(x[i,:]))^2 for i in 1:n]
f_test   = 0.5*[0*sum(x_test[i,:]) + (sum(x_test[i,:]))^2 for i in 1:n_test]

# Try with another function? 

y = f + stde*randn(n)

# set up SMARTparam and SMARTdata, then fit and predit
param  = SMARTparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,warnings=warnings,
           modality=modality,nofullsample=nofullsample,depthpp=depthpp,ntrees=ntrees)

data   = SMARTdata(y,x,param)

#for depthpp in [0,1,2]
param.depthpp = depthpp

output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output,predict=:Egamma)  # predict the natural parameter

avgtau,avg_explogtau,avgtau_a,dftau,x_plot,g_plot = SMARTweightedtau(output,data,verbose=false,best_model=false)

println(" \n modality = $(param.modality), nfold = $nfold, depthpp=$(param.depthpp) ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees), avgtau $avgtau ")
println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

#end 

# I need to plot f(x)
i = 1

function plot_pp(output,which_tree)

    t = output.SMARTtrees.trees[which_tree]
    param = output.bestparam

    T  = Float64
    xi = [-3.0:0.01:3]

    n  = length(xi)
    G0 = ones(T,n)
    G   = Matrix{T}(undef,n,2*param)

    for d in depthpp 
        μ  = t.μ[param.depth+1:param.depth+d]
        τ =  t.τ[param.depth+1:param.depth+d]
        β=
        G   = Matrix{T}(undef,n,2*size(G0,2))
        gL  = sigmoidf(xi,μ,τ,sigmoid)
        updateG!(G,G0,gL)
        G0 = copy(G) 
    end

    return G*β[end]
end 