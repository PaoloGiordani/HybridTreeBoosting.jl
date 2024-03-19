
Random.seed!(1)

n,p = 5_000,5 
x   = randn(n,p)
b   = [2.0,1.5,1.0,0.5,0.0]
f   = x*b
y   = f + randn(n)

@testset "modalities other than fast" begin
    
    @testset "compromise" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:compromise,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x0,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end
    
    @testset "accurate" begin 
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:accurate,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x0,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end 

end