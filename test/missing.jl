# create data 

Random.seed!(1)

n,p = 5_000,5 
prob = 0.2    # prob of missing 
x0  = randn(n,p)
b   = [2.0,1.5,1.0,0.5,0.0]
f   = x0*b
y   = f + randn(n)

ind = rand(n,p) .< prob
x   = @. x0*(1-ind) + ind*NaN

@testset "missing data" begin
    
    @testset "NaN" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x0,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.20
    end

    @testset "missing in matrix" begin 
        x   = @. x0*(1-ind) + ind*NaN
        x  = map(x->isnan(x) ? missing : x, x)
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x0,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.20
    end 

    @testset "missing in dataframe" begin 
        loss = :L2
        df = DataFrame(x,:auto)
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data = HTBdata(y,df,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x0,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.20
   
    end 

end