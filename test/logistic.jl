
Random.seed!(1)

n,p = 10_000,4

x  = randn(n,p)
b  = [2.0,1.5,1.0,0.5]
f  = x*b
y = (exp.(f)./(1.0 .+ exp.(f))).>rand(n)

@testset "HTBoost logistic regression" begin 

    @testset "Logistic loss with y BitVector" begin
        loss   = :logistic 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        rmse    = sqrt(sum((yf - f).^2)/n)
        @test rmse < 0.5
    end     

    @testset "Logistic loss with y Floating " begin
        loss   = :logistic 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(Float64.(y),x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        rmse   = sqrt(sum((yf - f).^2)/n)
        @test rmse < 0.5
    end     

    @testset "Logistic loss with y Integer " begin
        loss   = :logistic 
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(Int64.(y),x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        rmse   = sqrt(sum((yf - f).^2)/n)
        @test rmse < 0.5
    end     

end

