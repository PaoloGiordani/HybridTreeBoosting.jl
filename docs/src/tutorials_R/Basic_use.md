## Basic use

**Summary**

- Illustrates use of the main functions on a regression problem with simulated data. 
- param.modality as the most important user's choice, depending on time budget. 
- In default modality, HTBoost performs automatic hyperparameter tuning.
- The R bindings use JuliaConnectoR, which requires Julia to be installed. Please read [Installation in R](Installation_and_use_in_R.md).
- A few tutorials are provided for R. For more tutorials and examples, see [Julia tutorials](../Tutorials.md) and [Julia examples](../Examples.md), using the guidelines in [Installation in R](Installation_and_use_in_R.md) to adapt the code to R. 


**Main points** 

- default loss is "L2". Other options for continuous y are "Huber", "t" (recommended in place of "Huber"), "gamma", "gammaPoisson",
  "L2loglink". For zero-inflated continuous y, options are "hurdleGamma", "hurdleL2loglink", "hurdleL2"   
- default is block cross-validation with nfolds=4: use randomizecv = TRUE to scramble the data. See [Global Equity Panel](../examples/Global_Equity_Panel.md) for further options on cross-validation (e.g. sequential cv, or generally controlling the training and validation sets).
 
- fit, with automatic hyperparameter tuning if modality is :compromise or :accurate
- save fitted model (upload fitted model)
- average τ (smoothness parameter), which is also plotted. (Smoother functions ==> larger gains compared to other GBM)
- feature importance
- partial effects plots


---
---
**Install and load required packages. More detailed explanations in steps 1-4 in [Installation in R](Installation_and_use_in_R.md). (Note: Julia must be installed )**

```r

# ensure that R can find the path to julia.exe
julia_path = "C:\\Users\\.julia\\juliaup\\julia-1.11.2+0.x64.w64.mingw32\\bin"  # replace with your path to Julia
Sys.setenv(JULIA_BINDIR = julia_path)

if (!require(JuliaConnectoR)) {
  install.packages("JuliaConnectoR")
}

library(JuliaConnectoR)

# install packages in Julia (if needed) without leaving R
juliaEval('using Pkg; Pkg.add("Distributed")')
juliaEval('using Pkg; Pkg.add("DataFrames")')
#To install HTBoost from the registry, use the following command.
#juliaEval('using Pkg; Pkg.add("HTBoost")')
# Alternatively, this will work even if the package is not in the registry.
juliaEval('using Pkg; Pkg.add("https://github.com/PaoloGiordani/HTBoost.jl")')

# load packages 
HTBoost = juliaImport("HTBoost")
DataFrames = juliaImport("DataFrames")   


```

#### Set the desired number of workers (cores) to be used in parallel.

This step is not required by other GMBs, which rely on shared parallelization.  
Note: the first run of HTBoost (or any julia script) includes compile time, which increases in the number of workers and can be quite high. It is 80'' for 8 workers on my machine.
HTBoost parallelizes well up to 8 cores, and quite well up to 16 if p/#cores is sufficiently high. 

```r
juliaEval('
           number_workers  = 8  # desired number of workers, e.g. 8
           using Distributed
           nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
           @everywhere using HTBoost
           ')
```

### End of preliminary steps (required in all scripts). Now we generate data. 

y is the sum of six additive nonlinear functions, plus Gaussian noise.

```r

n      = 10000
p      = 6
stde   = 1.0
n_test = 100000

f_1 = function(x,b)  b*x + 1 
f_2 = function(x,b)  2*sin(2.5*b*x)  
f_3 = function(x,b)  b*x^3
f_4 = function(x,b)  b*(x < 0.5) 
f_5 = function(x,b)  b/(1.0 + (exp(4.0*x )))
f_6 = function(x, b) {b * (x > -0.25 & x < 0.25)}

b1  = 1.5
b2  = 2.0
b3  = 0.5
b4  = 4.0
b5  = 5.0
b6  = 5.0

x      = matrix(rnorm(n*p),nrow = n,ncol = p)
x_test = matrix(rnorm(n_test*p),nrow = n_test,ncol = p)
f      = f_1(x[,1],b1) + f_2(x[,2],b2) + f_3(x[,3],b3) + f_4(x[,4],b4) + f_5(x[,5],b5) + f_6(x[,6],b6)
f_test = f_1(x_test[,1],b1) + f_2(x_test[,2],b2) + f_3(x_test[,3],b3) + f_4(x_test[,4],b4) + f_5(x_test[,5],b5) + f_6(x_test[,6],b6)
y      = f + rnorm(n)*stde

fnames = c("x1", "x2", "x3", "x4", "x5", "x6")

```

When y and x are numerical matrices, we could feed them to HTBoost directly.
However, a more general procedue (which allows strings), is to transform the data into a dataframe and then into a Julia DataFrame. This is done as follows (here only for x, but the same applies to y if it is not numerical).

```r

df     =  data.frame(x)   # create a R dataframe
df_test = data.frame(x_test)
fnames = colnames(df)

x = DataFrames$DataFrame(df)
x_test = DataFrames$DataFrame(df_test)
DataFrames$describe(x)
```

**Options for HTBparam( ).**  

I prefer to specify parameter settings separately (here at the top of the script) rather than directly in HTBparam( ), which is of course also possible.  
modality is the key parameter: automatic hyperparameter tuning if modality is "compromise" or "accurate", no tuning (except of #trees) if "fast" or "fastest".    
In HTBoost, it is not recommended that the user performs 
hyperparameter tuning by cross-validation, because this process is done automatically if modality is
"compromise" or "accurate". The recommended process is to first run in modality="fast" or "fastest",
for exploratory analysis and to gauge computing time, and then switch to "compromise" (default)
or "accurate". For a tutorial on user-controlled cross-validation, see [User's controlled cross-validation.](../tutorials/User_controlled_cv.md)

```r

loss      = "L2"        # :L2 is default. Other options for regression are :L2loglink (if y≥0), :t, :Huber
modality  = "fast"      # "accurate", "compromise" (default), "fast", "fastest". The first two perform parameter tuning internally by cv.
priortype = "hybrid"    # "hybrid" (default) or "smooth" to force smoothness 
nfold     = 1           # number of cv folds. 1 faster (single validation sets), default 4 is slower, but more accurate.
nofullsample = TRUE     # if nfold=1 and nofullsample=TRUE, the model is not re-fitted on the full sample after validation of the number of trees
randomizecv = FALSE     # FALSE (default) to use block-cv. 
verbose     = "Off"
```

**Options for cross-validation:**

While the default in other GBM is to randomize the allocation to train and validation sets,
the default in HTBoost is block cv, which is suitable for time series and panels.
Set randomizecv=true to bypass this default. 
See [Time_series_and_panels](Time_series_and_panels.md) for further options on cross-validation (e.g. sequential cv, or generally controlling the training and validation sets).

```r
randomizecv = FALSE       # false (default) to use block-cv. 

```


### Set up HTBparam and HTBdata, then fit. Optionally, save the model (or load it). Predict. Print some information. 

```r
param = HTBoost$HTBparam(loss=loss,priortype=priortype,randomizecv=randomizecv,nfold=nfold,verbose=verbose,modality=modality,nofullsample=nofullsample)

data  = HTBoost$HTBdata(y,x,param,fnames=fnames)   # fnames is optional
output = HTBoost$HTBfit(data,param)

# save (load) fitted model. NOTE: use the same filename as the name of the object that is being saved.
#save(output, file = "output")
#load("output")

yhat = HTBoost$HTBpredict(x, output)      # Fitted values
yf = HTBoost$HTBpredict(x_test, output)   # Predicted values

cat("modality =", param$modality, ", nfold =", nfold, "\n")
cat("depth =", output$bestvalue, ", number of trees =", output$ntrees, "\n")
cat("out-of-sample RMSE from truth", sqrt(sum((yf - f_test)^2)/n_test), "\n")

```

which prints

```r
modality = fast , nfold = 1 
depth = 5 , number of trees = 293 
out-of-sample RMSE from truth 0.3563147

```

### Feature importance and average smoothing parameter for each feature.  

tau is the smoothness parameter; lower values give smoother functions, while tau=Inf is a sharp split (tau is truncated at 40 for this function).  
avgtau is a summary of the smoothness of f(x), with features weighted by their importance.
avgtau_a is a vector array with the importance weighted tau for each feature.  

The Julia functions do not always print nearly in R. To control printing, create a R dataframe  

```r
tuple = HTBoost$HTBweightedtau(output,data,verbose=FALSE)  # output is a Julia tuple, which can be converted to a R list
list = juliaGet(tuple)

# Create a data frame: features from 1 to p 
df <- data.frame(
  feature = list$fnames,
  importance = list$fi,
  avgtau = list$avgtau_a
)

# Create a data frame: features sorted by importance
df_sorted <- data.frame(
  sortedindx = list$sortedindx,
  sorted_feature = list$fnames_sorted,
  sorted_importance = list$fi_sorted,
  sorted_avgtau = list$avgtau_a[list$sortedindx]
)

#  print("\n Variable importance and smoothness, from first to last feature (not sorted)")
#  print(df)
print("\n Variable importance and smoothness, from most to least important feature")
print(df_sorted)

cat("\n Average smoothing parameter τ is", round(list$gavgtau, digits = 1), ".\n")
cat("\n In sufficiently large samples, and if modality is 'compromise' or 'accurate':\n")
cat(" - Values above 20-25 suggest very little smoothness in important features. HTBoost's performance may slightly outperform or slightly underperform other gradient boosting machines.\n")
cat(" - At 10-15 or lower, HTBoost should outperform other gradient boosting machines, or at least be worth including in an ensemble.\n")
cat(" - At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.\n")
```




### Create a plot to visualize the average smoothness of the splits 

The plot gives an idea of the average (importance weighted) smoothness across all splits.... 

```r

# Create a plot to visualize the average smoothness of the splits
library(ggplot2)
library(gridExtra)

df <- data.frame(x = list$x_plot, g = list$g_plot)

p <- ggplot(df, aes(x = list$x_plot, y = list$g_plot)) +
  geom_line() +
  labs(title = "avg smoothness of splits", x = "standardized x", y = "") +
  theme_minimal() + 
  theme(
    plot.title = element_text(size = 30, face = "bold"),
    axis.title.x = element_text(size = 20),
    axis.title.y = element_text(size = 20),
    axis.text = element_text(size = 20)
  )

print(p)


```


The plot gives an idea of the average (importance weighted) smoothness across all splits.... 

<img src="../assets/avgtau.png" width="400" height="250">

... which in this case is a mix of very different values across features: approximate linearity for x1, smooth functions for x3 and x5, and essentially sharp splits for x2, x4, and x6.
Note: Variable (feature) importance is computed as in Hastie et al., "The Elements of Statistical Learning", second edition, except that the normalization is for sum=100.  


```markdown
 Row │ feature  importance  avgtau     sorted_feature  sorted_importance  sorted_avgtau 
     │ String   Float32     Float64    String          Float32            Float64
─────┼──────────────────────────────────────────────────────────────────────────────────
   1 │ x1          14.5666   0.458996  x3                        19.0633       3.30638
   2 │ x2          12.9643  19.6719    x5                        18.6942       3.72146
   3 │ x3          19.0633   3.30638   x6                        17.9862      35.1852
   4 │ x4          16.7254  36.0846    x4                        16.7254      36.0846
   5 │ x5          18.6942   3.72146   x1                        14.5666       0.458996
   6 │ x6          17.9862  35.1852    x2                        12.9643      19.6719

 Average smoothing parameter τ is 7.3.

 In sufficiently large samples, and if modality=:compromise or :accurate

 - Values above 20-25 suggest very little smoothness in important features. HTBoost's performance may slightly outperform or slightly underperform other gradient boosting machines.
 - At 10-15 or lower, HTBoost should outperform other gradient boosting machines, or at least be worth including in an ensemble.
 - At 5-7 or lower, HTBoost should strongly outperform other gradient boosting machines.
```

Some examples of smoothness corresponding to a few values of tau (for a single split) help to interpret values of avgtau

<img src="../assets/Sigmoids.png" width="600" height="400">

On simulated data, we can evaluate the RMSE from the true f(x), exluding noise:


**Hybrid trees outperform both smooth and standard trees**

Here is the output for n=10k (nfold=1, nofullsample=true). 
Hybrid trees strongly outperform both smooth trees and standard symmetric (aka oblivious) trees. (Note: modality = :sharp is a very inefficient way to run a symmetric tree; use CatBoost or EvoTrees instead!)

```markdown
  modality = fastest, nfold = 1, priortype = hybrid
 depth = 5, number of trees = 141, gavgtau 7.3
 out-of-sample RMSE from truth 0.3136

modality = fastest, nfold = 1, priortype = smooth 
 depth = 5, number of trees = 121, gavgtau 4.5
 out-of-sample RMSE from truth 0.5751

 modality = fastest, priortype = sharp
depth = 5, number of trees = 183, avgtau 40.0
 out-of-sample RMSE from truth 0.5320
```

**Partial dependence plots**

Partial dependence assumes (in default) that other features are kept at their mean.

```r

fnames,fi,fnames_sorted,fi_sorted,sortedindx = HTBrelevance(output,data,verbose=false);
q,pdp  = HTBpartialplot(data,output,[1,2,3,4,5,6]) # partial effects for the first 6 variables 

# plot partial dependence
tuple = HTBoost$HTBpartialplot(data,output,c(1,2,3,4,5,6))
list = juliaGet(tuple)   #  this is not necessary in most cases (e.g. tuple$q works fine)

q  = list$q     
pdp = list$pdp

library(ggplot2)
library(gridExtra)

# Assuming q and pdp are matrices and f_1, f_2, ..., f_6 are functions, b1, b2, ..., b6 are parameters
f_list <- list(f_1, f_2, f_3, f_4, f_5, f_6)
b_list <- list(b1, b2, b3, b4, b5, b6)

pl <- vector("list", 6)

for (i in 1:length(pl)) {
  df <- data.frame(
    x = q[, i],
    HTB = pdp[, i],
    true = f_list[[i]](q[, i], b_list[[i]]) - f_list[[i]](q[, i] * 0, b_list[[i]])
  )
  
  pl[[i]] <- ggplot(df, aes(x = x)) +
    geom_line(aes(y = HTB, color = "HTB"), size = 1.5) +
    geom_line(aes(y = true, color = "true"), linetype = "dotted", size = 1) +
    labs(title = paste("PDP feature", i), x = "x", y = "f(x)") +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 15),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 12)
    ) +
    scale_color_manual(values = c("HTB" = "blue", "true" = "red"))
}

# Arrange the plots in a 3x2 grid layout
grid.arrange(grobs = pl, ncol = 3, nrow = 2)```

```

Partial plots for n = 1k,10k,100k, with modality = :fastest and nfold = 1.   
Notice how plots are smooth only for some features. 

### n = 1_000
<img src="../assets/Minimal1k.png" width="600" height="400">

### n = 10_000
<img src="../assets/Minimal10k.png" width="600" height="400">

### n = 100_000
<img src="../assets/Minimal100k.png" width="600" height="400">

