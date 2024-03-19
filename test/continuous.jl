
# create data 

Random.seed!(1)

n,p = 10_000,5 
x   = randn(n,p)
b   = [2.0,1.5,1.0,0.5,0.0]
f   = x*b
y   = f + randn(n)

@testset "HTBoost loss functions for continuous y" begin
    
    @testset "L2 loss" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end

    @testset "t loss" begin
        loss = :t
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        θ      = HTBcoeff(output,verbose=false)         # info on estimated coeff
        @test θ.dof > 20 
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end

    @testset "Huber loss" begin
        loss = :Huber
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true,warnings=:Off)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.2
    end

end