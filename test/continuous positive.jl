using Distributions 

# create data from a gamma distribution 

Random.seed!(1)

n,p = 10_000,4
k   = 5        # shape parameter of the gamma distribution

x  = randn(n,p)
b  = [0.2,0.15,0.1,0.05]
f  = x*b
μ  = exp.(f)        # conditional mean 
s  = μ/k            # scale parameter of the gamma distribution

y = zeros(n)

for i in eachindex(y)
    y[i]  = rand(Gamma.(k,s[i]))
end 

@testset "HTBoost loss function for y>0 continuous" begin 

    @testset "L2loglink loss" begin
        loss   = :L2loglink
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.05
    end     

    @testset "Gamma loss" begin
        loss   = :gamma 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        θ      = HTBcoeff(output,verbose=false)
        @test  abs(θ.shape-k) < 0.5  
        @test sqrt(sum((yf - f).^2)/n) < 0.05
    end     


end
