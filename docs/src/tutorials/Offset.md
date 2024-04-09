## Offset (exposure)

HTBoost provides the option to include an offset (exposure), a practice common in many fields, including biology and insurance. 

The offset is added to γ (NOT multiplied by E(y|x)), where γ = link(E(y|x)). It should therefore be in logs for
loss ∈ [:L2loglink, :gamma, :Poisson, :gammaPoisson, :hurdleGamma, :hurdleL2loglink],
and logit for loss =:logistic.

For example, if loss = :gamma and E(y|x) = offset*f(x), then the offset that is fed into *HTBdata( )* should be logged. 
```julia
loss     = :gamma 

exposure = df[1:n_train,:duration]
exposure_test = df[n_train+1:end,:duration]

if loss in [:gamma,:L2loglink,:Poisson,:gammaPoisson,:hurdleGamma,:hurdleL2loglink]
   offset = log.(exposure)
   offset_test = log.(exposure_test)
else
    offset = exposure 
    offset_test = exposure_test
end

param    =  HTBparam(loss=loss)              
data     =  HTBdata(y,x,param,offset=offset)    
yf       =  HTBpredict(x_test,output,predict=:Ey,offset=offset_test)
```

### Some advice 

Unless you are absolutely sure that an offset enters exactly with coeff 1, I recommend also adding it 
as a feature. HTBoost will then be able to capture possibly subtle nonlinearities and interaction effects
in E(y|offset).  


### Warning! Offset does not work well with categorical features

Categorical features with more than two categories are not currently handled correctly (by the mean targeting transformation)
with offsets. The program will run but categorical information will be used sub-optimally, particularly if the
average offset differs across categories. If categorical features are important, it may be better
to omit the offset from *HTBdata( )*, and instead model *y/offset* with a :L2loglink loss instead of a :gamma, :Poisson or :gammaPoisson. 

See [Offset.jl](../examples/Offset.jl) for a worked-out example.
