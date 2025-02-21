# Incorporating an offset 

**Short description:**

offset is added to γ (NOT multiplied by E(y|x)), where γ = link(E(y|x)). It should therefore be in logs for
loss in [:L2loglink,:gamma,:Poisson,:gammaPoisson,:hurdleGamma, :hurdleL2loglink],
and logit for loss in [:logistic].

For example, if loss = :gamma and E(y|x) = offset*f, then the offset that is fed into HTBdata() should be log(offset),
data   = HTBdata(y,x,param,offset=log(offset))    
    
Instructions 

1) Make sure that the offset vector has the same length as y and that it is appropriately transformed (so it can be
   *added* to the link-transformed E(y|x)).

```julia 

    if loss in [:gamma,:L2loglink,:Poisson,:gammaPoisson,:hurdleGamma, :hurdleL2loglink]
        γ = log.(offset)
    else
        γ = offset
    end
``` 

2) Add the offset as an argument in *HTBdata( )*
```julia
data   = HTBdata(y,x,param,offset=γ)
```
3)  If predicting, add any offset as an argument in *HTBpredict( )*
```julia
yhat  = HTBpredict(x_test,output,predict=:Ey,offset=γ_test)
```

**Warning!:**

>Categorical features with more than two categories are not currently handled correctly (by the mean targeting transformation)
with offsets. The program will run but categorical information will be used sub-optimally, particularly if the
average offset differs across categories. If categorical features are important, it may be better
to omit the offset from HTBdata(), and instead model y/offset with a :L2loglink loss instead of a :gamma, :Poisson or :gammaPoisson. 

Some advice: 

Unless you are absolutely sure that an offset enters exactly with coeff 1, I recommend also adding it 
as a feature. HTBoost will then be able to capture possibly subtle nonlinearities and interaction effects
in E(y|offset).  

```julia 


number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HybridTreeBoosting

using Random,Plots,Statistics
import Distributions 

# USER'S OPTIONS 

Random.seed!(12)

# Some options for HTBoost
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
f_2(x,b)    = -b*(x.<-0.5) + b*(x.>=0.5)   
f_3(x,b)    = b*x.*(x.<0)
f_4(x,b)    = b*(x.>=0.5)

b1,b2,b3,b4 = 0.2,0.2,0.2,0.2

# generate data
x,x_test = randn(n,p), randn(n_test,p)

c        = -2  
f        = c .+ f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = c .+ f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)

offset  = exp.(2*(randn(n)*std(f) .+ 0.2))  # offset here in terms of E(y|x), to be transformed later.

if loss in [:gamma,:L2loglink,:Poisson,:gammaPoisson,:hurdleGamma, :hurdleL2loglink]
    γ = log.(offset)
else
    γ = offset
end

μ        = @. exp(f)*offset     # conditional mean 
μ_test   = exp.(f_test)          

scale    = μ/k
scale_test = μ_test/k
y       = zeros(n)

for i in eachindex(y)
    y[i]  = rand(Distributions.Gamma.(k,scale[i]))
end 

histogram(y)
@show [mean(y), std(y), std(μ), maximum(y)]

# set up HTBparam and HTBdata, then fit and predit

# coefficient estimated internally. 
param  = HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,
                   verbose=verbose,warnings=warnings,modality=modality,nofullsample=nofullsample)
data   = HTBdata(y,x,param,offset=γ)

output = HTBfit(data,param)
yf     = HTBpredict(x_test,output,predict=:Ey)
#yhat  = HTBpredict(x_test,output,predict=:Ey,offset=γ_test)

println(" \n loss = $loss, modality = $(param.modality), nfold = $nfold ")
println(" depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" out-of-sample RMSE from truth, μ     ", sqrt(sum((yf - μ_test).^2)/n_test) )

println("\n true shape = $k, estimated = $(exp(output.bestparam.coeff_updated[1][1])) ")
println("\n For information about coefficients, use HTBcoeff(output) ")
HTBcoeff(output)


# Plot 

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
