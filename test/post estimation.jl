
Random.seed!(1)

n,p = 5_000,5 
x   = randn(n,p)
b   = [2.0,1.5,1.0,0.5,0.0]
f   = x*b
y   = f + randn(n)

@testset "post estimation" begin

    loss = :L2
    param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
    data   = HTBdata(y,x,param)
    output = HTBfit(data,param)

    @testset "avgtau" begin
        avgtau,gavgtau,avgtau_a,dftau,x_plot,g_plot = HTBweightedtau(output,data,verbose=false)
        @test avgtau < 1
    end

    @testset "variable importance" begin
        fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false)
        @test argmax(fi) == 1
    end
    
    @testset "partial plots" begin
        q,pdp  = HTBpartialplot(data,output,[1,2,3,4],predict=:Egamma)
    end

end