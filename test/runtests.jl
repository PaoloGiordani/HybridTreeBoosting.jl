
using HTBoost
using Test 

using Random 
using DataFrames 

@testset "HTBoost.jl" begin
 
    @testset "loss functions (distributions)" begin
        include("continuous.jl")             
        include("continuous positive.jl")    
        include("logistic.jl")
        include("count.jl")
        include("zero inflated.jl")     
    end

    @testset "various formats for data" begin
        include("dataframe.jl")              
        include("categorical.jl")
        include("missing.jl")                           
    end

    @testset "modalities and cv" begin
        include("modalities.jl")              
        include("cv.jl")
    end

    @testset "post-estimation analysis" begin
        include("post estimation.jl")
    end     

    @testset "parallelization" begin
        include("parallel.jl")
    end

end
