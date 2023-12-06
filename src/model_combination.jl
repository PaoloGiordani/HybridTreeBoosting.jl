#=

Functions related to stacking.

All models are fitted, then optimization to find non-negative weights by maximizing the loss in CV test set (stacking rather
than blending, which would require a genuine hold-out period.)

For example, with models having different depths, the model becomes: 
gammafit = w[i]*Trees of depth[i] + .... + w[I]*Trees of depth[I].

Stacking is as described in Friedman et al, or at https://medium.com/@stevenyu530_73989/stacking-and-blending-intuitive-explanation-of-advanced-ensemble-methods-46b295da413c

Opimization is by NelderMead and simulated annealing (then taking best solution) or a grid if univariate.
PG: consider some alternatives, like https://github.com/robertfeldt/BlackBoxOptim.jl, or code my own random search.

PG:  have a look at sklearn tools for stacking ... sklearn.ensemble import StackingClassifier, e.g. https://www.analyticsvidhya.com/blog/2021/08/ensemble-stacking-for-machine-learning-and-deep-learning/
=#

# Stacking
#   SMARTmodelweights
#   lossgiven beta
#   findw_grid
#
# Priors for stacking (currently not used: uniform prior )
#   SMARTlnpoisson
#   SMARTpoisson
#



# Compute weights by stacking.
# Accepts elements in lossgrid with Inf loss, giving them zero weights.
# w,lossw = SMARTmodelweights(output,data)
function SMARTmodelweights(output,data::SMARTdata,param::SMARTparam)
    w,lossw =  SMARTmodelweights(output.lossgrid,output.y_test,output.indtest,output.gammafit_test_a,data,param)
    return w,lossw
end



function SMARTmodelweights(lossgrid,y_test,indtest,gammafit_test_a,data::SMARTdata,param::SMARTparam;min_w=0.02)

    T = typeof(y_test[1])
    ind = Vector(1:length(lossgrid))[lossgrid.<Inf]  # indices with finite loss
    J = length(ind)                            # number of models with finite loss 
    y = y_test
    ind_test=[]

    for f in indtest
        ind_test=vcat(ind_test,f)
    end

    weights=data.weights[ind_test]
    fM = Matrix{T}(undef,length(y_test),J)
 
    for j in 1:J  # create (n,J) matrix of predictions from J models 
        fM[:,j] = gammafit_test_a[ind[j]]
    end

    if J<=1
        w0,lossw = [T(1)],lossgrid[1]
    elseif J==2  # optimize on a grid
        w0,lossw = findw_grid(fM,y,weights,param)
    else  
        # initialize with weights proportional to the position (in a ranking) of their loss
        # this requires all losses to be loglik or to come from the same model, which may not be the case...
        #β0 = invperm(sortperm(lossgrid[ind],rev=true))      # equivalent to StatsBase.tiedrank or R rank()  
        #β0 = (β0 .- β0[end])/J

        # initialize from equal weights
        β0 = fill(T(1),length(ind))

        # last element is always fixed at zero
        β0 = β0[1:end-1]    

        # derivative-based methods like Optim.BFGS and Optim.GradientDescent don't move. Why is the gradient zero? 
        # Try Nelder and SimulatedAnnealing (usually much slower), pick the best 
        res = Optim.optimize(β -> lossgivenbeta(β,fM,y,weights,param),β0, Optim.NelderMead(),Optim.Options(iterations = 5000))
        lossw,β = res.minimum,copy(res.minimizer)

        res = Optim.optimize(β -> lossgivenbeta(β,fM,y,weights,param),β0, Optim.SimulatedAnnealing(),Optim.Options(iterations = 5000))
 
        if res.minimum<lossw
            lossw,β = res.minimum,copy(res.minimizer)
        end 

        # set tiny weights to zero to speed-up forecasting
        β = vcat(β,0)  # re-instate last element
        @. β = @. β*(abs(β)<20) + sign(β)*20*(abs(β)>=20)  

        w0 = exp.(β)/sum(exp.(β))
        w0 = @. w0*(w0.>min_w)
        w0 = w0/sum(w0)     

    end

    w  = fill(T(0),length(lossgrid))
    w[ind] = w0

    # ensure that solution is superior to giving full weight to the best value
    if minimum(lossgrid)<lossw
        w  = fill(T(0),length(lossgrid))
        w[argmin(lossgrid)] = T(1)
    end

    return T.(w),T(lossw)

end



function lossgivenbeta(β,fM,y_test,weights,param)
    #T = typeof(f[1])
    T = Float64   # more robust in this setting

    maxvalue = 10
    β1 = vcat(β,T(0))
    @. β1 = β1*(abs(β1)<maxvalue) + sign(β1)*maxvalue*(abs(β1)>=maxvalue)
    w  = exp.(β1)/sum(exp.(β1))
    f  = fM*w
    loss,lossV = losscv(param,y_test,f,weights)

    return loss
end


# only univariate: grid
# w,lossw = findw_grid(fM,y_test,weights,param)
function findw_grid(fM,y_test,weights,param)  # operates on weights

    T = typeof(y_test[1])

    if size(fM,2) != 2
        @error "findw_grid only works for J==2 (finding one weight)"
    end

    w_grid = T.(Vector(0.0:0.05:1.0))
    loss_grid = fill(T(Inf),length(w_grid))

    for (i,w1) in enumerate(w_grid)
        w = vcat(w1,T(1)-w1)
        f  = fM*w
        loss_grid[i],lossV = losscv(param,y_test,f,weights)
    end

    minindex = argmin(loss_grid)
    w = vcat(w_grid[minindex],T(1)-w_grid[minindex])
    loss = loss_grid[minindex]
    
    return w,loss

end



#=
Poisson distribution as prior for depth d with parameter λ.  mean λ, var λ, mode λ-1 and λ.

p(d) = (λ^d)*exp(-λ)/d!
log(p(d)) = d*log(λ)-λ-log(d!)
=#
SMARTlnpoisson(d,λ) = d*log(λ)-λ-log(factorial(d))
SMARTpoisson(d,λ)   = (λ^d)*exp(-λ)/factorial(d)

