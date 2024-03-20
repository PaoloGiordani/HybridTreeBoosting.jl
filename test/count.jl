
# generate data from a gammaPoisson 
α    = 0.5   # overdispersion parameter. Poisson for α -> 0 
n,p  = 10_000,4

f_1(x,b)    = b./(1.0 .+ (exp.(2.0*(x .- 1.0) ))) .- 0.1*b 
f_2(x,b)    = b./(1.0 .+ (exp.(4.0*(x .- 0.5) ))) .- 0.1*b 
f_3(x,b)    = b./(1.0 .+ (exp.(7.0*(x .+ 0.0) ))) .- 0.1*b
f_4(x,b)    = b./(1.0 .+ (exp.(10.0*(x .+ 0.5) ))) .- 0.1*b

b1,b2,b3,b4 = 0.6,0.6,0.6,0.6

# generate data
x  = randn(n,p)

f        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
μ        = exp.(f)        # conditional mean 
y        = zeros(n)

for i in eachindex(y)
    r = 1/α                    # gammaPoisson is negative binomial with r=1\α, and p = r./(μ .+ r)
    pg = r./(μ .+ r)
    y[i] = rand(NegativeBinomial(r,pg[i]))     
end 

@testset "HTBoost loss functions for count data" begin
    
    @testset "gammaPoisson loss" begin
        loss   = :gammaPoisson 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Ey)
        rmse   = sqrt(sum((yf - μ).^2)/n)
        θ      = HTBcoeff(output,verbose=false)
        @test abs(θ.overdispersion/α -1) < 0.1 
        @test rmse < 0.25
    end 

    @testset "Poisson loss" begin
        loss   = :Poisson 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Ey)
        rmse   = sqrt(sum((yf - μ).^2)/n)
        @test rmse < 0.25
    end 
        
end
