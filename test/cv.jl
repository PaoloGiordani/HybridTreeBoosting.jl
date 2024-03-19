
Random.seed!(1)

n,p = 4_000,5 
x   = randn(n,p)
b   = [2.0,1.5,1.0,0.5,0.0]
f   = x*b
y   = f + randn(n)

@testset "different types of cross-validation" begin
    
    @testset "block-cv (default)" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=4)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end
    
    @testset "randomized cv" begin 
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=4,randomizecv=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end 

    @testset "expanding cv" begin 
        indtrain_a = [collect(1:2000),collect(2001:4000)]
        indtest_a  = [collect(2001:4000),collect(1:2000)]    

        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,indtrain_a=indtrain_a,indtest_a=indtest_a)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end 

end