using CategoricalArrays

# create data 

Random.seed!(1)

n,p  = 10_000,2
ncat = 10  

bcat       =     1.0      # coeff of categorical feature (if 0, categories are not predictive)
b1         =     1.0      # coeff of continuous feature 
stde       =     1.0      # error std

# create a categorical feature and a continuous feature
cate = collect(1:ncat)[randperm(ncat)]    
xcat  = rand(cate,n)
x     = hcat(randn(n),xcat)
b     = bcat*randn(ncat)  # Gaussian fixed effects

f     = zeros(n)

for r in 1:n
    bc   = b[findfirst(cate .== xcat[r])]
    f[r] = bc + b1*x[r,1]
end

y = f + stde*randn(n)

@testset "categoricals and their specification" begin

    @testset "categorical specified in cat_features" begin
        loss = :L2
        param  = HTBparam(cat_features = [2],loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        data   = HTBdata(y,x,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(x,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end 

    @testset "categorical not specified, but a string in a dataframe" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        x_cat  = map(string,x[:,2])
        df     = DataFrame(x1=x[:,1],x2=x_cat)
        data   = HTBdata(y,df,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(df,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end 

    @testset "categorical not specified, but x2 is Categorical" begin
        loss = :L2
        param  = HTBparam(loss=loss,depth=3,modality=:fast,nfold=1,nofullsample=true)
        df     = DataFrame(x1=x[:,1],x2=categorical(x[:,2]))
        data   = HTBdata(y,df,param)
        output = HTBfit(data,param)
        yf     = HTBpredict(df,output,predict=:Egamma)
        @test sqrt(sum((yf - f).^2)/n) < 0.10
    end 


end 
