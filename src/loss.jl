#
# Functions defining loss, gradient and hessian for various models. 
#
# General functions defining loss, gradient and hessian: 
# gradient_hessian()              computes gradient r and hessian h (typically not hessian but Fisher scoring or outer product, as more robust)
# losscv()                        loss used in CV to determine the number of trees and stacking weights.
# loglik()                        should be a proper log-likelihood (because of priors) associated with each param.loss
#
# General functions to update coefficients specific to each loss:
# initialize_gamma0()             estimates unconditional mean of gammafit
# updatecoeff()
# coeff_user()
# bias_correct()
# 
# Loss-specific functions
# prob_logistic!()
# bound_gammafit!()
# loglik_student()
# g_student() 
#
#
# Available models:
#
# :L2             (default)
# :logistic
# :Huber          (gradient only) default calibration of psi is 1.34*stdr, a robust measure of std. Could be CV. Asymptotically, psi should go to infinity to recover the conditional mean in asymmetric distributions
# :t              student t with dispersion and dof estimated by MLE for each iteration.
#
#
# Note: the notation gammafit is used instead of yfit because for general loss functions/likelihoods, gamma is the natural parameter on
# gamma(i) = sum of trees. e.g. log-odds for logit.
#


# NOTE: returns sum loglik, i.e. MINUS loss up to constant which does not depend on any coefficient
function loglik(loss,param::SMARTparam,y,gammafit,weights)
    
    ll = loglik_vector(loss,param,y,gammafit)
    @. ll = ll*weights

    return sum(ll)/(mean(weights)*param.loglikdivide) 

end


# returns vector of log-likelihods.
function loglik_vector(loss,param,y,gammafit)

    T = eltype(y)

    if loss==:L2 || loss == :lognormal
        σ2 = (param.coeff_updated[1][1])^2
        logσ2 = log(σ2)
        ll  = @. -logσ2/2 - ((y - gammafit)^2)/(2*σ2)         # ll is a vector, i.e. loglik for individual observations
    elseif loss==:Huber   # in quasi log-likelihood form (the density will not integrate to exactly one)
        r  = @. y - gammafit
        σ2,ψ = param.coeff_updated[1][1]^2, param.coeff_updated[1][2]
        logσ2 = log(σ2)
        ll = @. -T(0.5)*logσ2 - ( (T(0.5)*r^2)*(abs(r)<ψ)  + ( ψ*abs(r) - T(0.5)*ψ^2  )*(abs(r)≥ψ) )/σ2
    elseif loss==:t
        ll = loglik_student(y - gammafit,param.coeff_updated[1])
    elseif loss==:logistic
        bound_gammafit!(gammafit)
        ll  = @. y*gammafit - log(1 + exp(gammafit))   # exp() is slow. This loss evaluation is very slow.
    elseif loss==:gamma
        k  = exp(param.coeff_updated[1][1]) 
        μ  = exp.(gammafit) 
        ll = (k-1)*log.(y) - k*(y./μ) - k*gammafit .+log(k)  .- SpecialFunctions.loggamma(k)
    elseif loss==:L2loglink
        σ2 = (param.coeff_updated[1][1])^2
        logσ2 = log(σ2)
        μ  = exp.(gammafit)
        ll  = @. -logσ2/2 - ((y - μ)^2)/(2*σ2)
    elseif loss==:Poisson 
        ll = @. -SpecialFunctions.loggamma(y+1) + y*gammafit - exp(gammafit)  # factorial(y)=gamma(y+1)
    elseif loss==:gammaPoisson
        μ  = exp.(gammafit)
        α  = exp(param.coeff_updated[1][1])
        ll = @. SpecialFunctions.loggamma(y + 1/α) - SpecialFunctions.loggamma(1/α) - SpecialFunctions.loggamma(y + 1) + y*log((μ*α)/(1 + μ*α)) + (1/α)*log(1/(1 + μ*α))                 
    else 
        @error "loss not implemented"     
    end

end 
    


# losscv is used in cv only (and it does not need to be a proper log-likelihood).
# if losscv == :default, then same loss as in gradient_hessian is used.
# Other options for losscv are :mse :rmse :mae :logistic :sign
function losscv(param0::SMARTparam,y,gammafit,weights) # iter = 1 when no tree built yet, iter = j for j-1 trees already in the ensemble

    T = param0.T
    n = length(y)
    param = deepcopy(param0)
    losscv = param.losscv

    if losscv == :default

        if isempty(param.coeff_updated)  # when stacking models coeff_updated is not initialized 
            param = updatecoeff(param,y,gammafit,weights,1) # some log-lik require coefficients (e.g. var,dof)
        end    

        if param.loss==:L2 || param.loss == :lognormal
            loss  = @. (y - gammafit)^2
        elseif param.loss==:Huber
            ψ,r = T.(param.coeff_updated[2][2]),y .- gammafit   # coeff_updated[2], which do not change after the first iter
            loss  = @. T(2)*( (T(0.5)*r^2)*(abs(r)<ψ)  + ( ψ*abs(r) - T(0.5)*ψ^2  )*(abs(r)≥ψ) )  # multiply by 2 so it is a MSE,
        elseif param.loss==:t
            loss = -2*loglik_student(y - gammafit,param.coeff_updated[2])  # deviance=-2loglik. Notice it's coeff_updated[2], which do not change after the first tree
        elseif param.loss == :logistic           # gammafit  = logodds
            bound_gammafit!(gammafit)
            loss  = -y.*gammafit + log.(ones(T,n) + exp.(gammafit))
        elseif param.loss == :gamma 
            k = exp(param.coeff_updated[2][1])   # notice it's coeff_udpated[2], which do not change after the first tree
            μ  = exp.(gammafit)
            loss = - ( (k-1)*log.(y) - k*(y./μ) - k*gammafit .+log(k)  .- SpecialFunctions.loggamma(k) ) 
        elseif param.loss == :L2loglink
            μ  = exp.(gammafit)
            loss  = @. (y - μ)^2
        elseif param.loss==:Poisson 
            loss = @. SpecialFunctions.loggamma(y+1) - y*gammafit + exp(gammafit)
        elseif param.loss==:gammaPoisson
            μ  = exp.(gammafit)
            α  = exp(param.coeff_updated[2][1])
            loss = @. -(SpecialFunctions.loggamma(y + 1/α) - SpecialFunctions.loggamma(1/α) - SpecialFunctions.loggamma(y + 1) + y*log((μ*α)/(1 + μ*α)) + (1/α)*log(1/(1 + μ*α)) )                
        else 
            @error "loss not implemented"    
        end

    else

        if losscv == :mse

            if param.loss in [:L2,:t,:Huber,:lognormal]
                μ     = gammafit
            elseif param.loss in [:gamma,:Poisson,:gammaPoisson]
                μ     = exp.(gammafit)
            else 
                @error "losscv=:mse not coded for loss=$(param.loss)"
            end

            loss  = (y - μ).^2

        elseif losscv == :Huber
            ψ = T.(param.coeff_updated[1][2])
            r = @. y - gammafit
            loss  = @. T(2)*( (T(0.5)*r^2)*(abs(r)<ψ)  + ( ψ*abs(r) - T(0.5)*ψ^2  )*(abs(r)≥ψ) )  # multiply by 2 so it is a MSE,
        elseif losscv == :mae
            loss  = abs.(y - gammafit)
        elseif losscv == :logistic
            bound_gammafit!(gammafit)
            loss  = -y.*gammafit + log.(ones(T,n) + exp.(gammafit))
        elseif losscv == :sign
            loss = sign.(y) .== sign.(gammafit)
        elseif losscv == :logloss
            if param.loss in [:L2,:t,:Huber]
                loss = @. ( log(y+T(0.0001)) - log(gammafit+T(0.0001)) )^2
            elseif param.loss in [:gamma,:L2loglink]
                loss = @. ( log(y+T(0.0001)) - log(exp(gammafit+T(0.0001))) )^2
            else
                @error "losscv = :logloss not coded for loss=$loss"
            end         

        end

    end

    @. loss = loss*weights
    meanloss = mean(loss)/mean(weights)  

    return T(meanloss), T.(loss)

end



# Initializes roughly at corresponding parameter value for the unconditional distribution
function initialize_gamma0(data::SMARTdata,param::SMARTparam)

  y,weights,offset = data.y,data.weights,data.offset
  T = eltype(y)
  #meany  = sum(data.y.*data.weights)/sum(data.weights)  # not consistent if E(y|x) correlates with weights

  if param.loss in [:L2,:lognormal]
      gamma0 = mean(y) - mean(offset)
  elseif param.loss == :logistic
      gamma0 = log( mean(y)/(one(T) - mean(y)) ) - mean(offset)
  elseif param.loss == :Huber
      gamma0 = huber_mean(y-offset,data.weights,stdw_robust(y,weights);t=param.coeff[1])
  elseif param.loss == :t
    gamma0  = student_mean(y-offset,weights)
  elseif param.loss in [:gamma,:L2loglink,:Poisson,:gammaPoisson] 
    gamma0 = log(mean(y./exp.(offset)))     
 else
     @error "param.loss is misspelled or not supported."
  end

  return T(gamma0)

end



# param.coeff at iter = 0 or continuously to set default parameters (e.g. for Huber loss, so that user can input a pure number)
function updatecoeff(param0,y,gammafit,weights,iter)

    param=deepcopy(param0)
    T    = param.T

    if iter>=0  # >=0 continuously, or ==0 for standard

        if param.loss==:L2 || param.loss == :lognormal
            param.coeff_updated[1] = [std(y - gammafit)]
        elseif param.loss==:Huber  # 1) σ   2) σ*psi
            σ = stdw_robust(y-gammafit,weights)
            param.coeff_updated[1] = [σ, param.coeff[1]*σ ]
        elseif param.loss==:t
            res = Newton_MAP(y-gammafit,gH_student,start_value_student,w=weights)
            param.coeff_updated[1] = [res.minimizer[1],res.minimizer[2]]
        elseif param.loss in [:logistic,:Poisson]   # no parameter to update
            param.coeff_updated[1] = [T(NaN)]
        elseif param.loss==:gamma
            if isnan(param0.coeff[1])   # user has not provided a coeff: estimate it  
                res = Newton_MAP(y,gH_gamma,start_value_gamma,w=weights,x=gammafit)        
                param.coeff_updated[1] = [res.minimizer[1]]
            elseif iter<=1 
                param.coeff_updated[1] = [log(param.coeff[1])]
            end
        elseif param.loss==:gammaPoisson 
            if isnan(param0.coeff[1])   # user has not provided a coeff: estimate it  
                μ   = @. exp(gammafit)    # Newton was brittle: grid search instead. 
                α0  = max( T(log(0.01)),(var(y) - mean(y))/mean(y)^2 )   # gammaPoisson not defined for α = 0. α-->0 can give numerical problems.
                res  = Optim.optimize( logα -> loss_gammaPoisson(logα,y,μ),[log(α0)],Optim.BFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations = 100,x_tol = 0.01 ))
                param.coeff_updated[1] = [max(log(T(0.01)),res.minimizer[1])]     # gammaPoisson not defined for α = 0. α-->0 can give numerical problems.
            elseif iter<=1 
                param.coeff_updated[1] = [log(param.coeff[1])]
            end
        elseif param.loss==:L2loglink        
            param.coeff_updated[1] = [std(y - exp.(gammafit))]                
        else 
            @error "loss not implemented"    
        end

        # After the preliminary and first tree, coefficients for cv are not updated to reduce noise. 
        if iter<2 
            param.coeff_updated[2] = param.coeff_updated[1]           # update parameter used in cv (do only for iter<=1 to keep a constant parameter in cv)
        end     

    end

    return param
end



# gradient and MINUS hessian (actually Fisher scoring or outer product as more robust) of log-lik (NOT loss) wrt gammafit
# (Minus hessian to carry around h=1 instead of -1. Then Newton step is then +inv(h)g rather than -).
# h = -hessian of log-lik, so it must be positive (g is the gradient of the log-likelihod)
# multiply_pb multiplies the precision matrix of the prior for β. If g and h are (for convenience), the
# gradient and -hessian multiplied by a scalar α, then multiply_pb = α. e.g. for Gaussian α=σ^2 
# action_varGb=0 to use var(g) prior to preliminar run, 1 to set the prior in preliminary run, 2 not to update. 
function gradient_hessian(y::AbstractVector{T},weights::AbstractVector{T},gammafit::AbstractVector{T},param0::SMARTparam,action_varGb) where T<:AbstractFloat

    param=deepcopy(param0)

    if param.loss == :L2 || param.loss == :lognormal       # loss = ( y - gamma)^2
        g  = @. y - gammafit
        multiply_pb = (param.coeff_updated[1][1])^2  #
        h  = ones(T,1)           # Vector{T} of length 1. This is picked up by various functions to avoid un-necessary computations
    # Huber loss: 2nd order approximation not available -> 1st order used.
    elseif param.loss == :Huber   # loss = 0.5*r^2 if abs(r)<psi, and = psi*abs(r)-0.5*psi^2 if abs(r)>=psi. psi = 1.345 of std(r) is a reasonable setting. Or CV.
        psi  = T.(param.coeff_updated[1][2])
        r     = @. y - gammafit
        g     = @. r*(abs(r)<psi) + psi*sign(r)*(abs(r)>=psi)
        multiply_pb = (param.coeff_updated[1][1])^2
        h     = [mean(g.^2)]          
    elseif param.loss == :t
        g     = g_student(y - gammafit,param.coeff_updated[1])
        multiply_pb = T(1)
        h    = [mean(g.^2)]          
    elseif param.loss == :logistic  # loss = -loglik = sum(-y*gamma + log(1+exp(gamma)) ), where gamma = logodds = log( prob/(1-prob)), where prob is prob(y=1)
        prob,gammafit  = prob_logistic!(gammafit)   # # prob=exp(g)/(1+exp(g)) will be NaN if abs(g) is too large, which can happen with near-perfect classification.
        g     = @. y - prob
        h     = @.  prob*( one(T) - prob)
        h     = maximum(hcat(h,fill(T(0.0001),length(h))),dims=2)[:,1] # some offset necessary for robustness for high n and sharply separated classes.
        multiply_pb = T(1)
    elseif param.loss == :gamma
        k  = exp(param.coeff_updated[1][1])  
        μ  = exp.(gammafit) 
        g  = @. y/μ - 1
        g    = minimum(hcat(g,fill(T(1000),length(g))),dims=2)[:,1] 
        multiply_pb = 1/k  
        h  = ones(T,1)       # Fisher scoring 
    elseif param.loss == :Poisson 
        μ  = exp.(gammafit)
        g  = @. y - μ
        h  = μ
        h  = maximum(hcat(h,fill(T(0.001),length(h))),dims=2)[:,1]  
        multiply_pb = T(1)
    elseif param.loss == :gammaPoisson 
        μ  = exp.(gammafit)
        α  = exp(param.coeff_updated[1][1])
        g  = @. (y - μ)./(1 + α*μ)
        h  = @. μ/(1 + α*μ)   # Fisher scoring         
        h    = maximum(hcat(h,fill(T(0.0001),length(h))),dims=2)[:,1]  
        multiply_pb = T(1)              
    elseif param.loss == :L2loglink
        μ  = exp.(gammafit)
        g  = @. (y - μ)*μ
        multiply_pb = (param.coeff_updated[1][1])^2  #
        h  = μ.^2          # Fisher scoring, E(y)=μ
        h  = maximum(hcat(h,fill(T(0.0001*mean(μ.^2)),length(h))),dims=2)[:,1] # some offset necessary for robustness if h -> 0
    else      
        @error "param.loss is misspelled or not supported."    
    end

    # multiply g,h by weights
    if minimum(weights)<T(1)  # mean(weights)=1 in SMARTdata.
        @. g = g*weights
        h = @. h*weights     
    end     

    # divide g,h and multiply_pb by mean(h) so sum(h) = n, as assumed by pb_compute (so /n rather than divided by sum(h))
    hm  = mean(h)
    @. g = g/hm
    @. h = h/hm
    param.multiply_pb = multiply_pb/hm

    # unlike coefficients, priors are set only in preliminary runs
    if action_varGb==0
    	param.varGb=T(0.9)*var(g)            
    elseif action_varGb==1
    	param.varGb=var(gammafit)   
    end

    gh   = (r=g,h=h)

    return gh, param

end

# Vector{Floats}, with any coefficient needed for a given loss function. NOTE: assumes a pure number.
function coeff_user(loss::Symbol,T)

    coeff = T.([NaN])

    if loss==:Huber
        coeff = T.([1.34])       
    end

    return coeff

end



# Avoids the following problem in logistic regression:
# prob=exp(γ)/(1+exp(γ)) will be NaN if abs(γ) is too large, which can happen with near-perfect classification.
function bound_gammafit!(gammafit::AbstractVector{T}) where T<:AbstractFloat

    T==Float64 ? maxvalue = T(100) : maxvalue = 50
    @. gammafit = gammafit*(abs(gammafit)<maxvalue) + sign(gammafit)*maxvalue*(abs(gammafit)>=maxvalue)
 
end



# prob=exp(γ)/(1+exp(γ)) will be NaN if abs(γ) is too large, which can happen with near-perfect classification.
function prob_logistic!(gammafit::AbstractVector{T}) where T<:AbstractFloat

    T==Float64 ? maxvalue = T(100) : maxvalue = 50
    @. gammafit = gammafit*(abs(gammafit)<maxvalue) + sign(gammafit)*maxvalue*(abs(gammafit)>=maxvalue)
    expg = @. exp(gammafit)
    prob = @. expg/(T(1)+expg)

    return prob,gammafit

end



# Ensures mean(forecast_train)=mean(y_train). Only imposed for some loss functions. 
function bias_correct(gammafit,y_train,gammafit_train,param)  # gammafit can be on test or train set

    T = eltype(y_train)

    if param.loss in [:Huber,:t]
        bias = mean(y_train) - mean(gammafit_train)   # actually -bias 
        gammafit_ba = gammafit .+ bias
    else
        gammafit_ba = gammafit
        bias = T(0)      
    end 

    return bias,gammafit_ba

end     


# Functions used for MLE of fixed parameters (e.g. (log(s2),log(dof)) for :t, log(k) for :gamma )

# loglik for student t with m=0. coeff = [log(s2),log(dof)]. returns a vector
function loglik_student(y,coeff)  #  y=y-m 

    s2,v    = exp(coeff[1]),exp(coeff[2])
    T       = eltype(y)   
    z       = @. ( y )/sqrt(s2)
    logpdfz = T(-0.5723649429247001) .+ SpecialFunctions.loggamma((v+1)/2) .- SpecialFunctions.loggamma(v/2) .- T(0.5)*log(v) .- T(0.5)*(1+v)*log.(1 .+ (z.^2)/v) #     # @. here takes twice the time as Specialfunction is computed for every obs  

    return  logpdfz .- T(0.5)*log(s2)

end     


function g_student(y,coeff)   # gradient of log-lik for the mean. y=resid, coeff = log(s2),log(dof).

    vary,v = exp(coeff[1]),exp(coeff[2])
    y2   =  @. (y^2)/vary
    aux1 =  1 .+ y2/v
    g_mean  = ((v+1)/v )*((y./aux1)./vary)  

    return g_mean

end


function loss_gammaPoisson(logα,y,μ)
    α  = exp.(logα)[1]
    loss = @. -(SpecialFunctions.loggamma(y + 1/α) - SpecialFunctions.loggamma(1/α) - SpecialFunctions.loggamma(y + 1) + y*log((μ*α)/(1 + μ*α)) + (1/α)*log(1/(1 + μ*α)) )                
    return sum(loss)
end     
