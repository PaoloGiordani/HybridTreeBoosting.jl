# create a DataFrame
Random.seed!(1)

n,p = 5_000,4 
x   = randn(n,p)
b   = [2.0,1.5,1.0,0.5]
f   = x*b
y   = f + randn(n)

df = DataFrame(hcat(y,x),:auto)
rename!(df,[:y,:x1,:x2,:x3,:x4])

@testset "HTBoost with dataframe inputs" begin
    
    @testset "y,x belong to a common dataframe" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(df.y,df[:,2:end],param,fnames=names(df)[2:end])
        output = HTBfit(data,param)
        yf1     = HTBpredict(x,output,predict=:Egamma)
        yf2     = HTBpredict(df[:,2:end],output,predict=:Egamma)
        @test sqrt(sum((yf1 - f).^2)/n) < 0.150
        @test yf1 == yf2
    end 

end 

