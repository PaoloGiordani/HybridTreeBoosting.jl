#= 
Short description:

Speed benchmark: i) across platforms, ii) for different number_workers
Note: with sparse_vs, gains for more cores may be limited ! 



PC home 


8 cores 

16 cores, ntrees=20, sparse_vs = true 

             d=5     d=6      d=7     
10k,10        4.0    5.2      7.5
10k,100       5.9    7.4      11.0
10k,500       6.5    7.5      10.8

d=5        4 cores  8 cores     16 cores     24 cores. bit SLOWER than 16.   
100k,10      36.7    32.6       24.2        26.9
100k,100     62.5    49.8       42.5        49.1        
100k,500     177     127.2      119.0       128 




NB: This was done without limiting ntrees, and shows that depth = 6 and even 7 need not be discarded
n                  depth=5    depth=6       depth=7     depth=8
10k,10              19.3       20.3       24.1 
10k,100             23.5       30.8       40.5      75  now it really hits. Still feasible with enogh resources.



paolo.giordani@bi.no

=#

number_workers  = 8  # desired number of workers

using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using HTBoost

using Random,Statistics

# USER'S OPTIONS 
Random.seed!(123)

# Options for data generation 
n         = 100_000
p         = 10       # number of features 
dummies   = false    # true if x, x_test are 0-1 (faster).
stde      = 0.1      # small number to ensure all trees needed           

# Options for HTBoost

ntrees          = 20   
depth           = 5

for p in [10,100,500]

# function: Friedman of linear, with pstar relevant features.
# Increasing the number of relevant features increases the difficulty of the problem and can be
# used to evaluate speed gains and accuracy losses of sparsevs.

Friedman_function(x) = 10.0*sin.(π*x[:,1].*x[:,2]) + 20.0*(x[:,3].-0.5).^2 + 10.0*x[:,4] + 5.0*x[:,5]

p_star    = min(10,p)       # number of relevant features 
β = randn(p_star)
Linear_function_Gaussian(x)   = x[:,1:length(β)]*β

#f_dgp     = Friedman_function     
f_dgp    = Linear_function_Gaussian

# END USER'S INPUTS 


if f_dgp==Friedman_function
    x,x_test = rand(n,p), rand(200_000,p)    # Friedman function on U(0,1)
else 
    x,x_test = randn(n,p), randn(200_000,p)    

    if dummies 
        x,x_test = Float64.(x .> 0), Float64.(x_test .> 0)
    end 
end     


f       = f_dgp(x)
y      = f + stde*randn(n)
f_true = f_dgp(x_test)

# HTBoost

param   = HTBparam(modality=:fast,ntrees=ntrees,depth=depth,nfold=1,nofullsample=true)
data    = HTBdata(y,x,param)

println("\n n = $n, p = $p, dummies=$dummies, depth=$depth, ntrees=$ntrees")
println(" time to fit ")

@time output = HTBfit(data,param);
println(" ntrees $(output.ntrees)")
yf = HTBpredict(x_test,output,predict=:Ey)  # predict
println("\n RMSE of HTBoost from true E(y|x)                   ", sqrt(mean((yf-f_true).^2)) )

end 