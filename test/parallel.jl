
Random.seed!(1)

n,p = 10_000,5 
x   = randn(n,p)
b   = [2.0,1.5,1.0,0.5,0.0]
f   = x*b
y   = f + randn(n)

@testset "parallelization" begin

    @testset "compare single core and parallel" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        rmse1  =  sqrt(sum((yf - f).^2)/n) < 0.10

        using Distributed
        number_workers  = 4  # desired number of workers
        nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
        @everywhere using HTBoost
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        rmse2  =  sqrt(sum((yf - f).^2)/n) < 0.10
        @test rmse1 == rmse2
        
    end
    
end 