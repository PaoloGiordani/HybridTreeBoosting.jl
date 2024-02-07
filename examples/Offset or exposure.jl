"""

Short description:

offset is added to gammafit, so user should take logs if needed ... give 
instructions 

    if loss in [:L2,:lognormal,:t,:Huber,:quantile,]
        γ = offset
    elseif loss==:logistic
        γ = log.(offset./(ones(eltype(offset)) .- offset))
    elseif loss in [:gamma,L2loglink,:Poisson,:gammaPoisson]
        γ = log.(offset)
    else
        @error "loss not implemented or misspelled"    
    end



Notice that the offset vector is in terms of E(y|x), not in terms of the natural parameters (e.g. it is not logged.)

CATEGORICALS WITH MORE THAN 2 CATEGORIES WILL NOT WORK PROPERLY WITH OFFSET.

offset for ytest 

Set it in SMARTpredict()

PG: unless we are absolutely sure that an offset enteres exactly with coeff 1, couldn't we 
    simply add it to the list of coefficients? (in logs if log-link).... We could, but then it
    would have to be added to each tree ... 
    Neat solution? Model it as a standard offset, but then ALSO (could recommend to user...)
    add it to the list of features, so it can be picked if not exactly 1....

paolo.giordani@bi.no
"""

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
#@everywhere using SMARTboostPrivate
#include("E:\\Users\\A1810185\\Documents\\A_Julia-scripts\\Modules\\SMARTboostPrivateLOCAL.jl") # no package

using Random,Plots
import Distributions 

# USER'S OPTIONS 

Random.seed!(12)

# Some options for SMARTboost
loss      = :gamma          
modality  = :fast       # :accurate, :compromise (default), :fast, :fastest 

priortype = :hybrid      # :hybrid (default) or :smooth to force smoothness 
nfold     = 1             # number of cv folds. 1 faster (single validation sets), default 5 is slower, but more accurate.
nofullsample = true       # if nfold=1 and nofullsample=true, the model is not re-fitted on the full sample after validation of the number of trees
 
randomizecv = false       # false (default) to use block-cv. 
verbose     = :On
warnings    = :On

# options to generate data.
k           = 10         # shape parameter
n,p,n_test  = 10_000,4,100_000

f_1(x,b)    = b*x  
f_2(x,b)    = -b*(x.<0.5) + b*(x.>=0.5)   
f_3(x,b)    = b*x
f_4(x,b)    = -b*(x.<0.5) + b*(x.>=0.5)

b1,b2,b3,b4 = 0.2,0.2,0.2,0.2

# generate data
x,x_test = randn(n,p), randn(n_test,p)

c        = -2  
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

offset  = 2*(randn(n)*std(f) .+ 0.2)

μ        = exp.(f + offset)        # conditional mean 
μ_test   = exp.(f_test)   # conditional mean 

scale    = μ/k
scale_test = μ_test/k
y       = zeros(n)

for i in eachindex(y)
    y[i]  = rand(Distributions.Gamma.(k,scale[i]))
end 

histogram(y)
@show [mean(y), std(y), std(μ), maximum(y)]

# set up SMARTparam and SMARTdata, then fit and predit

# coefficient estimated internally. 
param  = SMARTparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = SMARTdata(y,x,param,offset=offset)

output = SMARTfit(data,param)
yf     = SMARTpredict(x_test,output,predict=:Ey)
#yhat  = SMARTpredict(x,output,predict=:Ey,offset=offset)

println(" \n loss = $loss, modality = $(param.modality), nfold = $nfold ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" out-of-sample RMSE from truth, μ     ", sqrt(sum((yf - μ_test).^2)/n_test) )

println("\n true shape = $k, estimated = $(exp(output.bestparam.coeff_updated[1][1])) ")
println("\n For information about coefficients, use SMARTcoeff(output) ")
SMARTcoeff(output)


# Repeat for L2 loss  
#=
param_L2 = deepcopy(param) 
param_L2.loss = :L2
output_L2 = SMARTfit(data,param_L2)
yf    = SMARTpredict(x_test,output_L2,predict=:Ey)  

println(" \n loss = $(param_L2.loss), modality = $(param.modality), nfold = $nfold, cv_link=true ")
println(" depth = $(output_L2.bestvalue), number of trees = $(output_L2.ntrees) ")
println(" out-of-sample RMSE from truth, μ      ", sqrt(sum((yf - μ_test).^2)/n_test) )
=#


# Plot 

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


