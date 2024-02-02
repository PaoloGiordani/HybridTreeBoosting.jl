#
# Functions to estimate constant coefficients (excluding the one parameter modeled as a sum
# of trees), if any, by MLE or MAP (e.g. logvar and logdof for a student-t) 
#    
# Newton_MAP    general function for MLE and MAP by outer score
#   res = Newton_MAP(y,gH_student,start_value_student;λ=0.25;w=...,x=...)
#   res = Newton_MAP(y,gH_student_with_mean,start_value_student_with_mean)
# 
# Functions specific to student-T distribution (here with zero mean), in terms of log(s2) and log(v), where s=std for Gaussian
#   start_value_student()
#   gH_student( )            
#   start_value_student_with_mean()
#   gH_student_with_mean()      , used in initialization of the mean 
#
# Functions specific to gamma distribution, in terms of log(k), where k is the shape parameter.
#
#   start_value_gamma()   
#   gH_gamma 
#   


# Newton minimizer of minus log-likelihood or log posterior, with outer score approximation of the Hessian
# gH() should return  g,H = gH(y,x,θ;w=..), with H (p,p) and g (p,1). θ₀ = startvalue(y,x)   
# returns a named tuple (minimizer,niter,converge)
function Newton_MAP(y,gH::Function,startvalue::Function; x=missing,w=1,λ=0.5,maxiter=100,tol=0.0001,max_attempts=5)

    θ₀ = startvalue(y,x)
    θ  = copy(θ₀)
    i  = 0
    T  = eltype(y)
    λ = T(λ)
    attempt = 0

    while attempt<max_attempts

        i,θ = Newton_inner_loop(y,x,λ,maxiter,gH,θ₀,tol,w)

        if sum(isnan.(θ))>0
            λ = T(λ/2)
            attempt = attempt + 1
        else
            break
        end     

    end     
    
    converge = i<maxiter && attempt<max_attempts

    sum(isnan.(θ))>0 ? θ=θ₀ : nothing     # PG facenda: switch to a more robust optimizer 

    return (minimizer=θ,niter=i,converge=converge)
end


function Newton_inner_loop(y,x,λ,maxiter,gH,θ_init,tol,w)

    θ₀ = copy(θ_init)
    θ  = copy(θ_init)
    i  = 0

    while i<maxiter
        g,H = gH(y,x,θ₀,w=w)
        θ  = θ₀ - λ*inv(H)*g
        maximum(abs.(θ-θ₀))<tol || sum(isnan.(θ))>0 ? break : i=i+1
        θ₀ = θ  
    end

    return i,θ
end


function start_value_student(y,x)   # assumes m = 0

    T  = eltype(y)
    v  = T(8) 
    s2  = var(y)*(v-2)/v 

    return [log(s2),log(v)] 

end     



# Gradient and Hessian for minus logpdf_student (mean=0). Hessian by outer score.
# A prior on log(dof) is essential if using Newton methods. Here it's N(2.3,0.5^2) (i.e.g centered on 10 dof), added to g,H 
function gH_student(y,x,coeff;w=1)  # w and x can be missing

    #g,H computed for logpdf, then output g,H is for logloss 
    T = eltype(y)
    mean_prior,var_prior = T(2.3),T(0.5^2)  # prior mean and variance for log(dof)
    vary,v = exp(coeff[1]),exp(coeff[2])
    y2   =  @. (y^2)/vary
    aux1 =  1 .+ y2/v

    #g_mean  = ((v+1)/v )*((y./aux1)./vary)       
    e2aux1  =  y2./aux1
    g_lnvar  =  T(0.5)*( ((v+1)/v )* e2aux1 .- 1 )

    a1,a2 = (v+1)/2,v/2
    lnaux1 =  @. log(aux1)
    g_lnv  =  T(0.5)*( SpecialFunctions.digamma(a1) .- SpecialFunctions.digamma(a2) .- 1/v .- lnaux1 .+ (v+1)/(v^2)*e2aux1 ).*v 

    gₘ = hcat(g_lnvar,g_lnv)

    if w != 1
       @. gₘ = gₘ*w
    end       

    H  = -gₘ'gₘ 
    g = sum(gₘ,dims=1)'

    # Add priors for log(v)
    g[2]   += -T(0.5)*(log(v) - mean_prior)/var_prior
    H[2,2] += -T(0.5)/var_prior 

    return -g,-H
end


function start_value_student_with_mean(y,x)   # assumes m = 0, so x is irrelevant 

    T  = eltype(y)
    m  = mean(y)
    v  = T(8) 
    s2  = var(y)*(v-2)/v 

    return [m,log(s2),log(v)] 

end     



# Gradient and Hessian for minus logpdf_student (mean estimated). Hessian by outer score.
# A prior on log(dof) is essential if using Newton methods. Here it's N(2.3,0.5^2) (centered on 10 dof), added to g,H 
function gH_student_with_mean(y0,x,coeff;w=1)  # w and x can be missing

    #g,H computed for logpdf, then output g,H is for logloss 
    T = eltype(y0)
    mean_prior,var_prior = T(2.3),T(0.5^2)  # prior mean and variance for log(dof)

    m,vary,v = coeff[1],exp(coeff[2]),exp(coeff[3])
    y = @. y0 - m 

    y2   =  @. (y^2)/vary
    aux1 =  1 .+ y2/v
    g_mean  = ((v+1)/v )*((y./aux1)./vary)   

    e2aux1  =  y2./aux1
    g_lnvar  =  T(0.5)*( ((v+1)/v )* e2aux1 .- 1 )

    a1,a2 = (v+1)/2,v/2
    lnaux1 =  @. log(aux1)

    g_lnv  =  T(0.5)*( SpecialFunctions.digamma(a1) .- SpecialFunctions.digamma(a2) .- 1/v .- lnaux1 .+ (v+1)/(v^2)*e2aux1 ).*v 

    gₘ = hcat(g_mean,g_lnvar,g_lnv)

    if w != 1
        @. gₘ = gₘ*w
     end       
 
    H  = -gₘ'gₘ 
    g = sum(gₘ,dims=1)'

    # Add priors
    g[3]   += -T(0.5)*(log(v) - mean_prior)/var_prior
    H[3,3] += -T(0.5)/var_prior 

    return -g,-H
end



# start value for log(k), for the unconditional distribution. by method of moments.
# μ is estimated as mean(y), and std(y)=μ/sqrt(k) 
function start_value_gamma(y,x)   

    μ  = mean(y)
    k  = (μ^2)/var(y)

    return log(k) 

end     


# Gradient and Hessian for minus logpdf_gamma (x=gammafit). Hessian by outer score.
# coeff = log(k), k the dispersion parameter.
# Could be made more efficient by having both gammafit and μ as inputs.
function gH_gamma(y,gammafit,coeff;w=1)  # w and x can be missing

    T = eltype(y)
    μ = exp.(gammafit)
    k = exp(coeff[1])
    # center a weak prior at start value. Wasteful to re-compute it at every it, as here ... 
    k_unconditional  = (mean(y)^2)/var(y)
    mean_prior,var_prior = log(k_unconditional),T(0.5^2)
    
    g = k*( log.(y) - y./μ - gammafit .+ log(k) .+ 1 .- SpecialFunctions.digamma(k) )
    
    if w != 1
        @. g = g*w
    end 

    H = -g'g  
    g = sum(g)

    # Add priors 
    g += -T(0.5)*(coeff[1] - mean_prior)/var_prior
    H += -T(0.5)/var_prior 

    return -g,-H
end



