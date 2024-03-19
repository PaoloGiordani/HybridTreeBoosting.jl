using Distributions 

# create data from a gamma distribution plus Gaussian noise, then truncate at zero 

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

y  = y + randn(n)    # add Gaussian noise
y  = max.(y,0)       # truncate at zero

@testset "HTBoost loss functions for zero-inflated continuous y" begin

    @testset "hurdleL2" begin
        loss   = :hurdleL2 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf,prob0,yf_not0 = HTBpredict(x,output,predict=:Ey)
        rmse   = sqrt(sum((yf - μ).^2)/n)
        @test rmse < 0.15
    end

    @testset "hurdleL2loglink" begin
        loss   = :hurdleL2loglink 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf,prob0,yf_not0 = HTBpredict(x,output,predict=:Ey)
        rmse   = sqrt(sum((yf - μ).^2)/n)
        @test rmse < 0.15
    end

    @testset "hurdleGamma" begin
        loss   = :hurdleGamma 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf,prob0,yf_not0 = HTBpredict(x,output,predict=:Ey)
        rmse   = sqrt(sum((yf - μ).^2)/n)
        @test rmse < 0.15
    end


end     

